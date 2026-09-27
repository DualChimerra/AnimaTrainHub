"""Inference sampler plugin registry (ADR 0003 PR-C).

Steps to add a new sampler (Euler / Heun / DPM++2M / DDIM / unipc):
1. Write training/inference_samplers/{variant}.py containing `sample(denoise_fn, x, sigmas, **kw)`
2. Add one line to the BUILDERS dict in this file
3. (Optional) Update sample_sampler_name in studio/schema.py to add Literal validation

See ADR 0003 "Case 8: Euler / DPM++2M" for details.

Note: sample_sampler_name is a str in the schema, not a Literal — an unregistered
name falls back to the simplified inline Euler ODE path in sampling.py (preserving
the original main() behavior).
"""

from __future__ import annotations

from typing import Callable

from training.inference_samplers import dpmpp_3m_sde, er_sde

__all__ = ["BUILDERS", "build_inference_sampler"]


BUILDERS: dict[str, Callable] = {
    "er_sde": er_sde.sample,
    "dpmpp_3m_sde": dpmpp_3m_sde.sample,
}


def build_inference_sampler(name: str) -> Callable:
    """Look up the sampler fn by name; returns None if unregistered (caller should fall back)."""
    return BUILDERS.get(str(name).lower().strip())
