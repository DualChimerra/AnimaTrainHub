"""Flow Matching timestep sampling: logit_normal / uniform / mode and other modes.

Extracted from the original runtime/anima_train.py L1817-1846 (ADR 0003 PR-A).

ADR 0003 keeps the different modes in one file (multiple fns, one file) --
each mode is a single line of sigmoid / shift / clamp math, doesn't
introduce a schema field (uses schema.timestep_sampling: str), and doesn't
need a plugin subfolder.
"""

from __future__ import annotations

import torch

from dual_peak_sampling import DualPeakConfig, sample_timesteps

from training.families.anima import ANIMA_SPEC as _ANIMA_SPEC


def sample_t(
    bs,
    device,
    mode: str = "logit_normal",
    shift: float = 3.0,
    mix_low_prob: float = 0.0,
    timestep_schedule_shift: float = 1.0,
    style_snr_mean: float = -6.0,
    style_snr_sigma: float = 2.0,
    dual_peak_config: DualPeakConfig | None = None,
) -> torch.Tensor:
    """Sample the Flow Matching timestep t in (0, 1).

    mode:
      dual_peak          — Portable custom style mixture; independent peak positions,
                           widths and weights. Ignores shift; see dual_peak_config.
      logit_normal       — SD3/Anima default, biased toward the middle t; shift>1 pushes toward the high-noise end
      uniform            — uniform sampling, more balanced coverage of both the detail and structure ends
      logit_normal_low   — logit-normal with the shift reversed, biased toward the low-noise/detail end
      mode               — SD3 mode-distribution, concentrated near a particular sigma
      mixed_uniform_low  — each sample independently takes logit_normal_low with probability mix_low_prob,
                           the rest take uniform; strengthens the detail end while keeping uniform coverage
      mixed_uniform_logit— same as above but the biased end uses logit_normal (standard), for when you don't want a low-t bias
      style_friendly     — Style-Friendly SNR Sampler (arXiv 2411.14793): samples directly on the
                           log-SNR axis from a normal distribution, concentrating training quality on the
                           high-noise window where style forms. **Does not use shift** (the offset is given directly by mean/sigma)

    mix_low_prob — the fraction of samples that take the biased end under mixed_* modes (default 0 = all uniform)
    style_snr_mean / style_snr_sigma — only apply to style_friendly, see sample_t_style_friendly
    timestep_schedule_shift — an extra sigma-schedule shift applied to t after sampling (default 1.0 = no shift);
                     formula t' = (t * s) / (1 + (s - 1) * t) (the SD3/FLUX shifted schedule,
                     a Mobius transform); different from timestep_shift: the latter acts on the post-sigmoid
                     u value inside logit-normal, the former acts on the final t
    """
    mode = (mode or "logit_normal").lower()

    if mode == "dual_peak":
        # Like style_friendly, positions are explicit: ignore timestep_shift.
        t = sample_timesteps(bs, device, dual_peak_config or DualPeakConfig())
        return _apply_timestep_schedule_shift(t, timestep_schedule_shift)

    if mode == "style_friendly":
        t = sample_t_style_friendly(bs, device, mean=style_snr_mean, sigma=style_snr_sigma)
        return _apply_timestep_schedule_shift(t, timestep_schedule_shift)

    if mode == "uniform":
        t = torch.rand(bs, device=device).clamp(1e-4, 1 - 1e-4)
        return _apply_timestep_schedule_shift(t, timestep_schedule_shift)

    if mode in ("mixed_uniform_low", "mixed_uniform_logit"):
        p = max(0.0, min(1.0, float(mix_low_prob)))
        use_biased = torch.rand(bs, device=device) < p
        t_uniform = torch.rand(bs, device=device).clamp(1e-4, 1 - 1e-4)
        biased_mode = "logit_normal_low" if mode == "mixed_uniform_low" else "logit_normal"
        # Recursively call the base mode to sample the biased end (don't
        # pass mix_low_prob again, to avoid a loop; don't apply the schedule
        # shift internally here either, leave it to be applied once at the end)
        t_biased = sample_t(bs, device, mode=biased_mode, shift=shift, mix_low_prob=0.0, timestep_schedule_shift=1.0)
        t = torch.where(use_biased, t_biased, t_uniform)
        return _apply_timestep_schedule_shift(t, timestep_schedule_shift)

    u = torch.sigmoid(torch.randn(bs, device=device))

    if mode == "logit_normal_low":
        s = max(float(shift), 1e-4)
        u = (u * (1.0 / s)) / (1 + (1.0 / s - 1) * u)
        return _apply_timestep_schedule_shift(u.clamp(1e-4, 1 - 1e-4), timestep_schedule_shift)

    if mode == "mode":
        s = float(shift)
        u = 1 - u - s * (torch.cos(torch.pi * 0.5 * u) ** 2 - 1 + u)
        return _apply_timestep_schedule_shift(u.clamp(1e-4, 1 - 1e-4), timestep_schedule_shift)

    # logit_normal (default) + shift
    s = float(shift)
    u = (u * s) / (1 + (s - 1) * u)
    return _apply_timestep_schedule_shift(u.clamp(1e-4, 1 - 1e-4), timestep_schedule_shift)


def _apply_timestep_schedule_shift(t: torch.Tensor, timestep_schedule_shift: float) -> torch.Tensor:
    """Apply an extra sigma-schedule shift to the sampled t: t' = (t * s) / (1 + (s - 1) * t).

    The Mobius transform of the SD3/FLUX shifted schedule; identity map when
    s == 1.0 (default behavior unchanged). s > 1 pushes toward the
    high-noise end; s < 1 biases toward the low-noise end.
    """
    s = float(timestep_schedule_shift)
    if s == 1.0:
        return t
    return ((t * s) / (1 + (s - 1) * t)).clamp(1e-4, 1 - 1e-4)


def sample_t_style_friendly(
    bs,
    device,
    mean: float = -6.0,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Style-Friendly SNR Sampler (arXiv 2411.14793): samples normally on the log-SNR axis.

    Paper's observation: **style (the overall feel of color use, lighting,
    composition, brushwork) forms at high-noise levels**, while detail
    texture forms at low-noise levels. A standard SD3/FLUX sampler spreads
    quality across the middle range, undersampling the style range. Pushing
    the log-SNR distribution overall toward low values (= high noise)
    noticeably improves style fidelity -- in the paper, rank 32 with this
    sampler beats rank 128 with the SD3 sampler, i.e. "sampling the right
    noise level" matters more than "stacking trainable parameters".

    Math (this repo's rectified flow convention: t=0 is the data end, t=1 is
    the noise end, x_t=(1-t)x0+t*x1):

        signal-to-noise ratio SNR = ((1-t)/t)^2  ->  log-SNR lambda = 2*ln((1-t)/t)
        inverting                                 ->  t = sigmoid(-lambda/2)

    So lambda ~ N(mean, sigma^2) maps directly to t. At mean=-6, the median t
    ~= sigmoid(3) ~= 0.953; sigma controls the window width (the paper uses
    2.0; 2 to 3 are both in the recommended range).

    Relationship to ``timestep_shift``: the latter is a Mobius offset acting
    on the post-sigmoid u; this mode doesn't go through that path at all --
    the offset is given entirely by mean, and combining the two would
    double-offset it, so the schema layer disables timestep_shift when this
    mode is selected. ``timestep_schedule_shift`` can still be layered on
    top (the caller's responsibility), and the resolution correction
    ``apply_resolution_shift`` continues to apply independently as usual.
    """
    sigma_v = max(float(sigma), 1e-4)
    lam = float(mean) + sigma_v * torch.randn(bs, device=device)
    t = torch.sigmoid(-lam * 0.5)
    return t.clamp(1e-4, 1 - 1e-4)


def latent_token_counts(
    latents, patch_spatial: int = _ANIMA_SPEC.latent.patch_spatial
) -> list[int]:
    """Per-sample patch-token count (used by ``timestep_shift_resolution_aware``).

    - list/tuple (NaViT per-image latents, each ``[.,C,T,h_i,w_i]``, shapes
      can be heterogeneous) -> counted per image;
    - a batched grid tensor ``[B,C,T,H,W]`` (ARB / multi-resolution, same
      size within the batch) -> B equal counts.

    ``patch_spatial=2`` matches the token convention used by
    ``CachedLatentDataset._fill_bucket_for_index`` (1 token = 16x16px).
    """
    if isinstance(latents, (list, tuple)):
        return [
            int(l.shape[-2] // patch_spatial) * int(l.shape[-1] // patch_spatial)
            for l in latents
        ]
    n = int(latents.shape[-2] // patch_spatial) * int(latents.shape[-1] // patch_spatial)
    return [n] * int(latents.shape[0])


def apply_resolution_shift(t: torch.Tensor, token_counts, base_tokens: int) -> torch.Tensor:
    """SD3 (arXiv 2403.03206 SS5.3.2) resolution-dependent timestep shift correction.

    Applies a per-sample Mobius offset t' = (t*s)/(1+(s-1)*t), with
    s_i = sqrt(n_i / n_base). At the same t, a higher-resolution image has
    stronger correlation between adjacent pixels and a higher effective SNR,
    so it needs a higher t to reach a degree of corruption equivalent to the
    base resolution -- hence larger images (n_i > n_base) are pushed toward
    the high-noise end and smaller ones the opposite way; n_i == n_base is
    the identity.

    Mobius offsets compose multiplicatively over s (shift(s1) . shift(s2) ==
    shift(s1*s2)), so this correction is orthogonal to the global
    timestep_shift / timestep_schedule_shift: the global value is still "the
    calibrated value for the base resolution", and this function only adds
    the difference relative to the base resolution.
    """
    s = (
        torch.as_tensor(token_counts, dtype=t.dtype, device=t.device)
        .clamp_min(1)
        .div(float(max(1, int(base_tokens))))
        .sqrt()
    )
    return ((t * s) / (1 + (s - 1) * t)).clamp(1e-4, 1 - 1e-4)
