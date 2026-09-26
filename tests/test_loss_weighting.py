from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from studio.schema import TrainingConfig
from training.loss_weighting import compute_loss_weight


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_detail_inv_t_default_matches_legacy_clamp():
    t = torch.tensor([0.5, 0.1, 0.9, 0.999])
    w = compute_loss_weight(t, scheme="detail_inv_t")
    expected = torch.tensor([2.0, 5.0, 1.0 / 0.9, 1.0 / 0.999])
    torch.testing.assert_close(w, expected, rtol=1e-4, atol=1e-4)


def test_detail_inv_t_extreme_low_t_clamps_to_max():
    t = torch.tensor([1e-5, 1e-3, 0.0])
    w = compute_loss_weight(t, scheme="detail_inv_t")
    assert torch.all(w <= 5.0 + 1e-6)
    assert torch.all(w == 5.0)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_detail_inv_t_custom_max_hazy_profile():
    t = torch.tensor([0.5, 0.1, 0.9])
    w = compute_loss_weight(t, scheme="detail_inv_t", detail_inv_t_min=1.0, detail_inv_t_max=3.0)
    expected = torch.tensor([2.0, 3.0, 1.0 / 0.9])
    torch.testing.assert_close(w, expected, rtol=1e-4, atol=1e-4)


def test_detail_inv_t_custom_max_aggressive_profile():
    t = torch.tensor([0.5, 0.1, 0.05])
    w = compute_loss_weight(t, scheme="detail_inv_t", detail_inv_t_min=1.0, detail_inv_t_max=8.0)
    expected = torch.tensor([2.0, 8.0, 8.0])
    torch.testing.assert_close(w, expected, rtol=1e-4, atol=1e-4)


def test_detail_inv_t_custom_min_lifts_high_t():
    t = torch.tensor([0.9, 0.999])
    w = compute_loss_weight(t, scheme="detail_inv_t", detail_inv_t_min=1.5, detail_inv_t_max=5.0)
    assert torch.all(w >= 1.5 - 1e-6)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_detail_inv_t_min_equals_max_collapses_to_constant():
    t = torch.tensor([0.05, 0.5, 0.95])
    w = compute_loss_weight(t, scheme="detail_inv_t", detail_inv_t_min=2.5, detail_inv_t_max=2.5)
    expected = torch.full_like(t, 2.5)
    torch.testing.assert_close(w, expected)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_detail_inv_t_bounds_do_not_affect_min_snr():
    t = torch.tensor([0.1, 0.5, 0.9])
    w_a = compute_loss_weight(t, scheme="min_snr", detail_inv_t_min=99.0, detail_inv_t_max=0.5)
    w_b = compute_loss_weight(t, scheme="min_snr")
    torch.testing.assert_close(w_a, w_b)


def test_detail_inv_t_bounds_do_not_affect_cosmap():
    t = torch.tensor([0.1, 0.5, 0.9])
    w_a = compute_loss_weight(t, scheme="cosmap", detail_inv_t_min=99.0, detail_inv_t_max=0.5)
    w_b = compute_loss_weight(t, scheme="cosmap")
    torch.testing.assert_close(w_a, w_b)


def test_detail_inv_t_bounds_do_not_affect_none():
    t = torch.tensor([0.1, 0.5, 0.9])
    w = compute_loss_weight(t, scheme="none", detail_inv_t_min=99.0, detail_inv_t_max=0.5)
    torch.testing.assert_close(w, torch.ones_like(t))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _minimal_cfg(**overrides) -> dict:
    base = {
        "data_dir": "./dataset",
        "transformer_path": "x.safetensors",
        "vae_path": "x.safetensors",
        "text_encoder_path": "x",
        "t5_tokenizer_path": "x",
    }
    base.update(overrides)
    return base


def test_schema_rejects_detail_inv_t_min_above_max():
    with pytest.raises(ValidationError, match="detail_inv_t_min"):
        TrainingConfig(**_minimal_cfg(detail_inv_t_min=5.0, detail_inv_t_max=1.0))


def test_schema_rejects_detail_inv_t_min_below_one():
    with pytest.raises(ValidationError):
        TrainingConfig(**_minimal_cfg(detail_inv_t_min=0.5))


def test_schema_rejects_detail_inv_t_min_zero():
    with pytest.raises(ValidationError):
        TrainingConfig(**_minimal_cfg(detail_inv_t_min=0.0))


def test_schema_rejects_detail_inv_t_min_above_upper_bound():
    with pytest.raises(ValidationError):
        TrainingConfig(**_minimal_cfg(detail_inv_t_min=21.0))


def test_schema_rejects_detail_inv_t_max_above_upper_bound():
    with pytest.raises(ValidationError):
        TrainingConfig(**_minimal_cfg(detail_inv_t_max=51.0))


def test_schema_rejects_weight_cap_ratio_above_upper_bound():
    with pytest.raises(ValidationError):
        TrainingConfig(**_minimal_cfg(weight_cap_ratio=51.0))


def test_schema_accepts_min_equals_max():
    cfg = TrainingConfig(**_minimal_cfg(detail_inv_t_min=3.0, detail_inv_t_max=3.0))
    assert cfg.detail_inv_t_min == 3.0 and cfg.detail_inv_t_max == 3.0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_weight_cap_constrains_detail_inv_t_spread():
    t = torch.tensor([0.1, 0.5, 0.9])
    w = compute_loss_weight(t, scheme="detail_inv_t", weight_cap_ratio=2.0)
    w_min = float(w.min())
    w_max = float(w.max())
    assert w_max <= w_min * 2.0 + 1e-5, f"max/min={w_max/w_min:.3f} > cap 2.0"


def test_weight_cap_disabled_preserves_detail_inv_t_spread():
    t = torch.tensor([0.1, 0.5, 0.9])
    w_no_cap = compute_loss_weight(t, scheme="detail_inv_t", weight_cap_ratio=0.0)
    assert abs(float(w_no_cap.max()) - 5.0) < 1e-4
