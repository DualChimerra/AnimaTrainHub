"""Training noise generation: base Gaussian + noise_offset + pyramid multi-scale low frequency.

Extracted from the original runtime/anima_train.py L1848-1896 (ADR 0003 PR-A).

Parameterized pure math functions; a new noise scheme (e.g. fp_offset_v2) can
just add an if branch in this file, no plugin subfolder needed.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def noise_params_from_args(args) -> tuple[float, int, float]:
    """Dispatch on noise_enhancement_type to the active noise-enhancement params -> (offset, iters, discount).

    The schema is "one type choice + two parameter groups"; the historical
    implementation read the raw parameter values directly, with `type` having
    zero involvement on the runtime side -- when both groups were non-zero,
    offset and pyramid would silently stack (design doc Sec 10.1 audit #3).
    The yaml/CLI main path's mutual exclusion is enforced by
    migrate_noise_enhancement_type at the config layer (post-R1 the trainer
    goes through that same path too); this function is runtime defense in
    depth, guarding against args sources that bypass config construction
    (old-format pause snapshots / hand-built in-process namespaces), and
    making the "type is the only switch" contract hold at the consumer end.
    Both the standard loop path and the navit packing path share this
    function -- don't read args.noise_offset / pyramid_* directly elsewhere.
    """
    ne_type = str(getattr(args, "noise_enhancement_type", "none") or "none")
    offset = (
        float(getattr(args, "noise_offset", 0.0) or 0.0) if ne_type == "offset" else 0.0
    )
    iters = (
        int(getattr(args, "pyramid_noise_iters", 0) or 0) if ne_type == "pyramid" else 0
    )
    discount = float(getattr(args, "pyramid_noise_discount", 0.35) or 0.35)
    return offset, iters, discount


def make_noise(
    latents: torch.Tensor,
    noise_offset: float = 0.0,
    pyramid_iters: int = 0,
    pyramid_discount: float = 0.35,
) -> torch.Tensor:
    """Generate training noise, optionally stacking low-frequency perturbations.

    noise_offset   -- adds a per-sample/channel low-frequency offset, mitigating
                     brightness-mean bias (the SDXL trick)
    pyramid_iters  -- stacks multi-scale low-frequency noise, helping the model
                     learn global lighting/composition faster; bilinear
                     interpolation avoids the blocky artifacts of nearest
    """
    noise = torch.randn_like(latents)

    if noise_offset > 0:
        shape = list(latents.shape)
        for ax in range(2, latents.ndim):
            shape[ax] = 1
        offset = torch.randn(*shape, device=latents.device, dtype=latents.dtype)
        noise = noise + noise_offset * offset

    if pyramid_iters > 0:
        try:
            spatial = list(latents.shape[-2:])
            cur = noise.clone()
            for i in range(pyramid_iters):
                r = 2 ** (i + 1)
                sh, sw = max(spatial[0] // r, 1), max(spatial[1] // r, 1)
                if latents.ndim == 5:
                    extra = torch.randn(
                        latents.shape[0], latents.shape[1], latents.shape[2], sh, sw,
                        device=latents.device, dtype=latents.dtype,
                    )
                    extra = F.interpolate(
                        extra.flatten(0, 1), size=spatial, mode="bilinear", align_corners=False,
                    ).view(latents.shape[0], latents.shape[1], latents.shape[2], *spatial)
                else:
                    extra = torch.randn(latents.shape[0], latents.shape[1], sh, sw,
                                        device=latents.device, dtype=latents.dtype)
                    extra = F.interpolate(extra, size=spatial, mode="bilinear", align_corners=False)
                cur = cur + extra * (pyramid_discount ** (i + 1))
                if min(sh, sw) <= 1:
                    break
            noise = cur / cur.std().clamp(min=1e-6)
        except Exception as exc:
            logger.warning(f"pyramid_noise failed, falling back to standard noise: {exc}")

    return noise
