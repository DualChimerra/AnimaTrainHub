"""Unit tests for the timestep shift resolution correction (timestep_shift_resolution_aware).

Covers:
  1) apply_resolution_shift: identity at the base resolution / t goes up for
     larger images, down for smaller ones / numeric points of the formula.
  2) Möbius multiplicative composition law: applying a global shift then the
     resolution correction == a single combined-product shift (basis for
     orthogonality).
  3) per-sample vector: within one batch, samples with different token counts
     each get their own correction.
  4) endpoint clamp: extreme token ratios never leave (1e-4, 1-1e-4).
  5) latent_token_counts: both batched grid-tensor input and NaViT's
     heterogeneous latent-list input.
  6) config: defaults to off, can be enabled.

CPU-only, no GPU required: runs fine in CI (Linux, no GPU).
"""
from __future__ import annotations

import math

import pytest
import torch

from training.timestep_sampling import (
    apply_resolution_shift,
    latent_token_counts,
)


# --------------------------------------------------------- formula properties
def test_identity_at_base_tokens():
    t = torch.linspace(0.05, 0.95, 10)
    out = apply_resolution_shift(t, [4096] * 10, 4096)
    assert torch.allclose(out, t, atol=1e-6)


def test_direction_and_known_values():
    # 4x tokens -> s=2; 1/4 tokens -> s=0.5. Analytic value at t=0.5 is 2/3 and 1/3.
    t = torch.full((4,), 0.5)
    up = apply_resolution_shift(t, [4096 * 4] * 4, 4096)
    down = apply_resolution_shift(t, [4096 // 4] * 4, 4096)
    assert torch.all(up > t) and torch.all(down < t)
    assert torch.allclose(up, torch.full((4,), 2.0 / 3.0), atol=1e-5)
    assert torch.allclose(down, torch.full((4,), 1.0 / 3.0), atol=1e-5)


def test_composes_multiplicatively_with_global_shift():
    """Möbius shifts compose multiplicatively in s: global shift(s1) followed by resolution correction(s2) == shift(s1*s2).

    This is the mathematical basis for "the global timestep_shift remains the
    base-resolution calibration value, and this correction only makes up the
    resolution difference."
    """
    t = torch.linspace(0.05, 0.95, 17)
    s1, n, base = 3.0, 16384, 4096  # s2 = sqrt(16384/4096) = 2
    a = (t * s1) / (1 + (s1 - 1) * t)
    a = apply_resolution_shift(a, [n] * len(t), base)
    s12 = s1 * math.sqrt(n / base)
    b = ((t * s12) / (1 + (s12 - 1) * t)).clamp(1e-4, 1 - 1e-4)
    assert torch.allclose(a, b, atol=1e-5)


def test_per_sample_vector():
    t = torch.tensor([0.5, 0.5, 0.5])
    out = apply_resolution_shift(t, [4096, 16384, 1024], 4096)
    assert out[0] == pytest.approx(0.5, abs=1e-6)
    assert out[1] > 0.5 > out[2]


def test_bounds_clamped():
    t = torch.tensor([1e-4, 1 - 1e-4])
    out = apply_resolution_shift(t, [1_000_000, 1], 4096)
    assert torch.all(out >= 1e-4) and torch.all(out <= 1 - 1e-4)


# --------------------------------------------------------- token counting convention
def test_latent_token_counts_batched_tensor():
    # 1024px -> latent 128x128 -> 64x64 tokens = 4096 (matches CachedLatentDataset's convention)
    lat = torch.zeros(3, 16, 1, 128, 128)
    assert latent_token_counts(lat) == [4096, 4096, 4096]


def test_latent_token_counts_navit_list():
    # NaViT heterogeneous list: both 5D [1,C,T,h,w] and 4D [C,T,h,w] are computed from the last two dims
    lst = [torch.zeros(1, 16, 1, 62, 93), torch.zeros(16, 1, 48, 48)]
    assert latent_token_counts(lst) == [31 * 46, 24 * 24]


# --------------------------------------------------------- config switch
def test_config_default_off():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.timestep_shift_resolution_aware is False


def test_config_can_enable():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig(timestep_shift_resolution_aware=True)
    assert cfg.timestep_shift_resolution_aware is True
