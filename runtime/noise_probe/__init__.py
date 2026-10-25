"""Training-free probes for choosing a style timestep distribution.

Model-agnostic: everything takes a ``velocity_fn(x_t, t, cond) -> v`` callable
(rectified flow, ``v = noise - x0``), so the math is testable on CPU with an
analytic model and the GPU tool only supplies the real DiT.
"""

from .core import (
    GapProfile,
    bin_edges,
    denoise_mse,
    euler_denoise,
    gap_profile,
    scheduled_velocity,
    seeded_noise,
    shifted_sigmas,
    truncate_schedule,
)
from .fit import TRAIT_IDENTITY, config_bin_masses, fit_dual_peak, trait_target
from .metrics import compare_images, effect_share

__all__ = [
    "GapProfile",
    "TRAIT_IDENTITY",
    "bin_edges",
    "compare_images",
    "config_bin_masses",
    "denoise_mse",
    "effect_share",
    "euler_denoise",
    "fit_dual_peak",
    "gap_profile",
    "scheduled_velocity",
    "seeded_noise",
    "shifted_sigmas",
    "trait_target",
    "truncate_schedule",
]
