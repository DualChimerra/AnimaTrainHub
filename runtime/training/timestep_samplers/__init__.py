"""Timestep sampler plugin registry (follows the ADR 0003 PR-C adapter registry pattern).

`build_timestep_sampler(args, total_steps)` dispatches on args to a concrete sampler:
- Defaults to baseline (wraps sample_t's 4 modes)
- args.infonoise_enabled == True routes to InfoNoise adaptive sampling
- args.timestep_sampling == "krea2_shift" routes to Krea2 resolution-aware sampling

To add a new sampler:
1. Write timestep_samplers/{name}.py with `build(args, total_steps) -> TimestepSamplerProtocol`.
2. Add a line to the BUILDERS dict / build_timestep_sampler dispatch logic in this file.
3. Add the corresponding enable field to studio/schema.py.
4. Done -- zero changes needed in phases/optimizer.py / loop.py / context.py.
"""

from __future__ import annotations

from training.timestep_samplers import baseline, infonoise, krea2_shift
from training.timestep_samplers.protocol import TimestepSamplerProtocol

__all__ = [
    "TimestepSamplerProtocol",
    "BUILDERS",
    "build_timestep_sampler",
]


# Single source of truth: registry of all sampler build factories.
# baseline is the fallback (buildable from any args); the rest are adaptive
# (enabled based on args fields).
BUILDERS: dict[str, callable] = {
    "baseline": baseline.build,
    "infonoise": infonoise.build,
    "krea2_shift": krea2_shift.build,
}


def build_timestep_sampler(args, total_steps) -> TimestepSamplerProtocol:
    """Dispatch on args to the matching sampler; always returns a non-None
    instance (baseline is the fallback).

    Krea2 dynamic shift and InfoNoise both define the timestep distribution,
    so they cannot both be enabled at once.
    """
    mode = str(getattr(args, "timestep_sampling", "logit_normal") or "logit_normal").lower()
    if mode == "krea2_shift":
        if getattr(args, "infonoise_enabled", False):
            raise ValueError("krea2_shift and infonoise_enabled cannot both be enabled")
        return BUILDERS["krea2_shift"](args, total_steps)
    if getattr(args, "infonoise_enabled", False):
        return BUILDERS["infonoise"](args, total_steps)
    return BUILDERS["baseline"](args, total_steps)
