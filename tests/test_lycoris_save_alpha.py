from __future__ import annotations

import pytest
import torch

from utils.lycoris_adapter import _rewrite_per_layer_alpha_


class _FakeLora:
    def __init__(self, name: str, scale: float, dim: int) -> None:
        self.lora_name = name
        self.scale = scale
        self.lora_dim = dim


class _FakeNet:
    def __init__(self, loras: list) -> None:
        self.loras = loras


def test_rewrite_alpha_uses_scale_times_dim() -> None:
    net = _FakeNet([
        _FakeLora("layer_a", scale=5.6569, dim=1),
        _FakeLora("layer_b", scale=5.6569, dim=8),
        _FakeLora("layer_c", scale=5.6569, dim=32),
        _FakeLora("layer_d", scale=5.6569, dim=64),
    ])
    sd = {
        "layer_a.alpha": torch.tensor(181.0, dtype=torch.float32),
        "layer_b.alpha": torch.tensor(181.0, dtype=torch.float32),
        "layer_c.alpha": torch.tensor(181.0, dtype=torch.float32),
        "layer_d.alpha": torch.tensor(181.0, dtype=torch.float32),
        "layer_a.lokr_w2_a": torch.zeros(2, 1),
    }
    _rewrite_per_layer_alpha_(net, sd)
    assert sd["layer_a.alpha"].item() == pytest.approx(5.6569, abs=1e-4)
    assert sd["layer_b.alpha"].item() == pytest.approx(45.2552, abs=1e-4)
    assert sd["layer_c.alpha"].item() == pytest.approx(181.0208, abs=1e-3)
    assert sd["layer_d.alpha"].item() == pytest.approx(362.0416, abs=1e-3)
    assert sd["layer_a.lokr_w2_a"].shape == (2, 1)


def test_rewrite_alpha_noop_when_dim_equals_base() -> None:
    net = _FakeNet([_FakeLora("x", scale=1.0, dim=32)])
    sd = {"x.alpha": torch.tensor(32.0)}
    _rewrite_per_layer_alpha_(net, sd)
    assert sd["x.alpha"].item() == pytest.approx(32.0)


def test_rewrite_alpha_handles_none_network() -> None:
    sd = {"x.alpha": torch.tensor(99.0)}
    _rewrite_per_layer_alpha_(None, sd)
    assert sd["x.alpha"].item() == 99.0


def test_rewrite_alpha_skips_layer_without_alpha_key() -> None:
    net = _FakeNet([_FakeLora("missing", scale=1.0, dim=4)])
    sd = {"other.alpha": torch.tensor(7.0)}
    _rewrite_per_layer_alpha_(net, sd)
    assert sd["other.alpha"].item() == 7.0
    assert "missing.alpha" not in sd


def test_rewrite_alpha_preserves_dtype() -> None:
    net = _FakeNet([_FakeLora("x", scale=5.6569, dim=1)])
    sd = {"x.alpha": torch.tensor(181.0, dtype=torch.bfloat16)}
    _rewrite_per_layer_alpha_(net, sd)
    assert sd["x.alpha"].dtype == torch.bfloat16


def test_rewrite_alpha_skips_lora_missing_attrs() -> None:
    class _NoName:
        scale = 1.0
        lora_dim = 4

    class _NoScale:
        lora_name = "x"
        lora_dim = 4

    net = _FakeNet([_NoName(), _NoScale()])
    sd = {"x.alpha": torch.tensor(7.0)}
    _rewrite_per_layer_alpha_(net, sd)
    assert sd["x.alpha"].item() == 7.0
