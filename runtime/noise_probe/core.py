"""Gap profile and noise ladder math.

Convention (same as the trainer): ``x_t = (1-t)*x0 + t*noise``, ``t=0`` is
clean, ``t=1`` is noise, and the model predicts ``v = noise - x0``.

Gap profile: the base model's denoising error per t on the artist's images,
divided by its error on images it generated itself from the same captions.
Where the ratio exceeds 1, the artist departs from the model's own prior --
that is the noise band a style LoRA has to learn. Bands where the ratio is ~1
the base model already reproduces, and training there mostly teaches poses
and content.

Noise ladder: noise one image to t0 and let the base model finish it. The
smallest t0 where the artist's palette / shapes / pose get replaced by the
model's defaults shows which band carries which trait.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor]


def bin_edges(n_bins: int = 20) -> torch.Tensor:
    """Equal-width t bins on [0, 1], float64."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    return torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)


def seeded_noise(like: torch.Tensor, seed: int) -> torch.Tensor:
    """Gaussian noise drawn on CPU, so a seed means the same noise on any device."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(like.shape, generator=generator, dtype=torch.float32)
    return noise.to(device=like.device, dtype=like.dtype)


def shifted_sigmas(steps: int, shift: float) -> torch.Tensor:
    """``steps+1`` sigmas from 1 to 0 with the SD3/Comfy shift ``s*u/(1+(s-1)*u)``."""
    if steps < 1:
        raise ValueError("steps must be positive")
    u = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    s = float(shift)
    return (s * u) / (1 + (s - 1) * u)


def truncate_schedule(sigmas: torch.Tensor, t0: float) -> torch.Tensor:
    """Start a full schedule at ``t0``: ``[t0, sigmas below t0..., 0]``.

    Keeps the remaining steps of the full schedule, so a ladder rung spends
    the same per-step budget as a normal generation over that range.
    """
    if not 0.0 < t0 <= 1.0:
        raise ValueError("t0 must be in (0, 1]")
    rest = sigmas[sigmas < t0 - 1e-9]
    out = torch.cat([torch.tensor([t0], dtype=sigmas.dtype), rest])
    if float(out[-1]) != 0.0:
        out = torch.cat([out, torch.zeros(1, dtype=sigmas.dtype)])
    return out


@torch.no_grad()
def euler_denoise(
    velocity_fn: VelocityFn,
    x_t: torch.Tensor,
    sigmas: torch.Tensor,
    cond: Any,
    *,
    uncond: Any = None,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """Deterministic Euler ODE from ``sigmas[0]`` to ``sigmas[-1]``.

    Deterministic on purpose: with the noise fixed, every ladder rung differs
    only by where the base model takes over, not by sampler randomness.
    """
    x = x_t
    use_cfg = uncond is not None and float(cfg_scale) != 1.0
    for i in range(len(sigmas) - 1):
        t = torch.full((x.shape[0],), float(sigmas[i]), device=x.device, dtype=torch.float32)
        v = velocity_fn(x, t, cond).float()
        if use_cfg:
            v_uncond = velocity_fn(x, t, uncond).float()
            v = v_uncond + float(cfg_scale) * (v - v_uncond)
        x = (x.float() + float(sigmas[i + 1] - sigmas[i]) * v).to(x_t.dtype)
    return x


@torch.no_grad()
def denoise_mse(
    velocity_fn: VelocityFn,
    x0: torch.Tensor,
    cond: Any,
    t_values: Sequence[float],
    seeds: Sequence[int],
) -> torch.Tensor:
    """Mean velocity MSE of one latent at each t, averaged over the given noise seeds."""
    out = torch.zeros(len(t_values), dtype=torch.float64)
    for seed in seeds:
        noise = seeded_noise(x0, seed)
        target = (noise.float() - x0.float())
        for i, t in enumerate(t_values):
            t_vec = torch.full((x0.shape[0],), float(t), device=x0.device, dtype=torch.float32)
            x_t = ((1 - float(t)) * x0.float() + float(t) * noise.float()).to(x0.dtype)
            pred = velocity_fn(x_t, t_vec, cond).float()
            out[i] += float(((pred - target) ** 2).mean())
    return out / max(1, len(seeds))


@dataclass(frozen=True)
class GapProfile:
    edges: torch.Tensor
    mse_artist: torch.Tensor
    mse_reference: torch.Tensor

    @property
    def centers(self) -> torch.Tensor:
        return (self.edges[:-1] + self.edges[1:]) / 2

    @property
    def relative_gap(self) -> torch.Tensor:
        """``mse_artist / mse_reference - 1``: 0.25 = the base model is 25% worse on the artist there."""
        return self.mse_artist / self.mse_reference.clamp_min(1e-12) - 1

    def target_masses(self, *, floor: float = 0.02, smooth: bool = True) -> torch.Tensor:
        """Suggested probability per bin: proportional to the positive gap, plus a uniform floor.

        Heuristic, not a derivation: train more where the base model departs
        from the artist more. The floor keeps every band minimally covered.
        """
        gap = self.relative_gap.clamp_min(0)
        if smooth and gap.numel() >= 3:
            padded = torch.cat([gap[:1], gap, gap[-1:]])
            gap = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        n = gap.numel()
        if float(gap.sum()) <= 0:
            return torch.full((n,), 1.0 / n, dtype=torch.float64)
        masses = gap / gap.sum()
        return (1 - floor) * masses + floor / n


def gap_profile(
    velocity_fn: VelocityFn,
    artist: Sequence[tuple[torch.Tensor, Any]],
    reference: Sequence[tuple[torch.Tensor, Any]],
    *,
    edges: torch.Tensor | None = None,
    seeds: Sequence[int] = (0, 1),
    progress: Callable[[str, int, int], None] | None = None,
) -> GapProfile:
    """Average :func:`denoise_mse` over both sets at the bin centers.

    ``artist`` / ``reference`` are ``(latent [1,C,T,H,W], cond)`` pairs. Each
    image uses its own seeds offset by its index, identically in both sets.
    """
    if not artist or not reference:
        raise ValueError("both artist and reference sets must be non-empty")
    edges = bin_edges() if edges is None else edges
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    def _mean(items, label):
        total = torch.zeros(len(centers), dtype=torch.float64)
        for k, (latent, cond) in enumerate(items):
            image_seeds = [int(s) + 1000 * k for s in seeds]
            total += denoise_mse(velocity_fn, latent, cond, centers, image_seeds)
            if progress is not None:
                progress(label, k + 1, len(items))
        return total / len(items)

    return GapProfile(edges=edges, mse_artist=_mean(artist, "artist"),
                      mse_reference=_mean(reference, "reference"))


def scheduled_velocity(
    velocity_fn: VelocityFn,
    set_scale: Callable[[float], None],
    scale_at: Callable[[float], float],
) -> VelocityFn:
    """Wrap a velocity so an adapter's strength follows the current t.

    Used to switch a LoRA on only inside a noise window during sampling; the
    whole batch shares one t, as in :func:`euler_denoise`.
    """
    def velocity(x, t, cond):
        set_scale(float(scale_at(float(t.reshape(-1)[0]))))
        return velocity_fn(x, t, cond)
    return velocity
