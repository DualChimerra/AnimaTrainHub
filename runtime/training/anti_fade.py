"""Anti-fade: auxiliary contrast and edge terms against washed-out, blurry LoRAs.

With plain MSE on a small dataset the model hedges when it is unsure: the
mean of dark and light is a pale mid-tone, the mean of several line positions
is a soft smear. All terms work on the clean-image estimate recovered from
the velocity prediction (x0 = x_t - t * v) and only penalise the prediction
for being *flatter* than the real latent:

- contrast: per-channel spatial std of x0_pred below that of x0;
- edges: spatial-gradient magnitude of x0_pred below that of x0;
- range: the ends of each channel's value distribution (5th / 95th percentile) pulled
  inward compared with x0 — shallow shadows and dull highlights. The std above is
  dominated by mid-tones; this term looks only at the tails.

Samples with t > max_t are skipped: near pure noise the x0 estimate is a
legitimate blurry guess, and forcing contrast there only adds noise.
Computed in latent space (no VAE decode), so it is an approximation.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn import functional as F

_EPS = 1e-6
#: Percentile of each tail used by the range term
_RANGE_TAIL = 0.05


def _weighted_std(x: torch.Tensor, w: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    den = w.sum(dims).clamp_min(_EPS)
    mean = (x * w).sum(dims) / den
    var = ((x - mean.view(*mean.shape, *([1] * len(dims)))) ** 2 * w).sum(dims) / den
    return (var + 1e-8).sqrt()


def _edge_deficit(pred: torch.Tensor, real: torch.Tensor, w: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """Relative shortfall of edge magnitude along one spatial axis pair (H and W)."""
    num = pred.new_zeros(pred.shape[:2])
    den = pred.new_zeros(pred.shape[:2])
    for axis in (-2, -1):
        n = pred.shape[axis] - 1
        if n < 1:
            continue
        gp = pred.narrow(axis, 1, n) - pred.narrow(axis, 0, n)
        gr = real.narrow(axis, 1, n) - real.narrow(axis, 0, n)
        wa = w.narrow(axis, 1, n) * w.narrow(axis, 0, n)
        num = num + (F.relu(gr.abs() - gp.abs()) * wa).sum(dims)
        den = den + (gr.abs() * wa).sum(dims)
    return num / den.clamp_min(_EPS)


def _range_deficit(pred: torch.Tensor, real: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Per-sample shrinkage of the low / high percentiles of each channel, relative to the real range."""
    out = []
    for b in range(pred.shape[0]):
        keep = (w[b, 0] > 0.5).flatten()
        p = pred[b].flatten(1)[:, keep]
        r = real[b].flatten(1)[:, keep]
        if p.shape[1] < 2:
            out.append(pred.new_zeros(()))
            continue
        q = torch.tensor([_RANGE_TAIL, 1.0 - _RANGE_TAIL], device=p.device, dtype=p.dtype)
        p_lo, p_hi = torch.quantile(p, q, dim=1)
        r_lo, r_hi = torch.quantile(r, q, dim=1)
        span = (r_hi - r_lo).clamp_min(1e-3)
        out.append(((F.relu(p_lo - r_lo) + F.relu(r_hi - p_hi)) / span).mean())
    return torch.stack(out)


def anti_fade_loss(
    pred: torch.Tensor,
    noisy: torch.Tensor,
    latents: torch.Tensor,
    t: torch.Tensor,
    *,
    contrast_weight: float,
    edge_weight: float,
    max_t: float,
    range_weight: float = 0.0,
    spatial_mask: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batch-mean anti-fade penalty. pred/noisy/latents: (B, C, T, H, W); t: (B,)."""
    t = t.float().view(-1)
    active = (t <= max_t).float()
    if not bool(active.any()):
        return pred.new_zeros((), dtype=torch.float32)
    t_exp = t.view(-1, *([1] * (pred.dim() - 1)))
    x0_pred = noisy.float() - t_exp * pred.float()
    x0_real = latents.float()
    dims = tuple(range(2, pred.dim()))
    w = spatial_mask.float() if spatial_mask is not None else x0_real.new_ones(
        (pred.shape[0], 1, *pred.shape[2:]))
    w = w.expand(pred.shape[0], 1, *pred.shape[2:])

    per_sample = pred.new_zeros(pred.shape[0], dtype=torch.float32)
    if contrast_weight > 0:
        s_pred = _weighted_std(x0_pred, w, dims)
        s_real = _weighted_std(x0_real, w, dims)
        contrast = (F.relu(s_real - s_pred) / (s_real + 1e-3)).mean(1)
        per_sample = per_sample + contrast_weight * contrast
    if edge_weight > 0:
        per_sample = per_sample + edge_weight * _edge_deficit(x0_pred, x0_real, w, dims).mean(1)
    if range_weight > 0:
        per_sample = per_sample + range_weight * _range_deficit(x0_pred, x0_real, w)

    per_sample = per_sample * active
    if sample_weight is not None:
        per_sample = per_sample * sample_weight.to(per_sample).view(-1)
    return per_sample.mean()
