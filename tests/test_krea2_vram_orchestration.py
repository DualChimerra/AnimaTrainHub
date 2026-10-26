
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import training.families.krea2 as krea2_module
from training.families.krea2 import (
    Krea2Family,
    _sampling_headroom_bytes,
    _should_offload_te,
    _should_yield_dit,
)

_GIB = 1024 ** 3


def test_sampling_headroom_scales_with_area():
    assert _sampling_headroom_bytes(1024, 1024) == int(5.0 * _GIB)
    assert _sampling_headroom_bytes(1536, 1536) == int((2.0 + 3.0 * 2.25) * _GIB)


def test_should_yield_dit_policies(monkeypatch):
    assert _should_yield_dit("performance", "cuda") is False
    assert _should_yield_dit("save_vram", "cuda") is True
    monkeypatch.setattr(krea2_module, "_cuda_free_bytes", lambda d: 17 * _GIB)
    assert _should_yield_dit("auto", "cuda") is False
    monkeypatch.setattr(krea2_module, "_cuda_free_bytes", lambda d: 2 * _GIB)
    assert _should_yield_dit("auto", "cuda") is True
    monkeypatch.setattr(krea2_module, "_cuda_free_bytes", lambda d: None)
    assert _should_yield_dit("auto", "cuda") is False


def test_should_offload_te_policies(monkeypatch):
    assert _should_offload_te("performance", "cuda", 1024, 1024, False) is False
    assert _should_offload_te("save_vram", "cuda", 1024, 1024, False) is True
    assert _should_offload_te("auto", "cuda", 1024, 1024, True) is True
    monkeypatch.setattr(krea2_module, "_cuda_free_bytes", lambda d: 9 * _GIB)
    assert _should_offload_te("auto", "cuda", 1024, 1024, False) is False
    monkeypatch.setattr(krea2_module, "_cuda_free_bytes", lambda d: 3 * _GIB)
    assert _should_offload_te("auto", "cuda", 1024, 1024, False) is True
    monkeypatch.setattr(krea2_module, "_cuda_free_bytes", lambda d: None)
    assert _should_offload_te("auto", "cuda", 1024, 1024, False) is True


class _OrchestraModel:
    def __init__(self):
        self.moves: list[str] = []

    def to(self, device):
        self.moves.append(str(device))
        return self


class _OrchestraText:

    def __init__(self, *, cached: bool = False, resident: bool = False):
        self._cached = cached
        self.is_model_on_device = resident
        self.offload_calls = 0

    def online_conditions_cached(self, captions):
        return self._cached

    def offload_model(self):
        self.offload_calls += 1


def _run_sample(monkeypatch, *, policy, text, free_gib=None):
    family = Krea2Family()
    model = _OrchestraModel()
    calls = {}

    monkeypatch.setattr(
        krea2_module, "_cuda_free_bytes",
        lambda d: None if free_gib is None else int(free_gib * _GIB),
    )
    monkeypatch.setattr(
        krea2_module, "prepare_sampling_condition",
        lambda *a, **k: calls.setdefault("prepared", True) or "cond",
    )
    monkeypatch.setattr(
        krea2_module, "sample_image",
        lambda *a, **k: "image",
    )
    result = family.sample_image(
        model, object(), text, "a prompt",
        distilled=True, device="cpu", vram_policy=policy,
    )
    assert result == "image"
    assert calls.get("prepared")
    return model, text


def test_policy_none_keeps_legacy_behavior(monkeypatch):
    model, text = _run_sample(monkeypatch, policy=None, text=_OrchestraText())
    assert model.moves == []
    assert text.offload_calls == 1


def test_auto_with_ample_vram_keeps_all_resident(monkeypatch):
    model, text = _run_sample(
        monkeypatch, policy="auto", text=_OrchestraText(), free_gib=15,
    )
    assert model.moves == []
    assert text.offload_calls == 0


def test_auto_tight_vram_yields_dit_and_offloads_te(monkeypatch):
    model, text = _run_sample(
        monkeypatch, policy="auto", text=_OrchestraText(), free_gib=2,
    )
    assert model.moves == ["cpu", "cpu"]
    assert text.offload_calls == 1


def test_auto_lru_hit_skips_yield_entirely(monkeypatch):
    model, text = _run_sample(
        monkeypatch, policy="auto",
        text=_OrchestraText(cached=True), free_gib=2,
    )
    assert model.moves == []


def test_auto_te_already_resident_skips_yield(monkeypatch):
    model, text = _run_sample(
        monkeypatch, policy="auto",
        text=_OrchestraText(resident=True), free_gib=9,
    )
    assert model.moves == []
    assert text.offload_calls == 0


# -- per-sample yield under block swap: only the resident part moves, swapped layers never follow .to() --


def _swap_model():
    """Register a swap handle on a four-layer ModuleList (a stand-in is enough: this only checks finding and passing the handle)."""
    from torch import nn

    from training.block_swap import _OWNER_ATTR

    blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
    swap = object()
    setattr(blocks, _OWNER_ATTR, swap)
    return nn.Sequential(blocks), swap


def test_find_block_swap_resolves_registered_handle():
    from torch import nn

    from training.block_swap import find_block_swap

    model, swap = _swap_model()
    assert find_block_swap(model) is swap
    # plain model without swap / duck-typed stand-in (no modules()) -> None, falls back to model.to()
    assert find_block_swap(nn.Sequential(nn.Linear(2, 2))) is None
    assert find_block_swap(_OrchestraModel()) is None


def test_yield_threads_swap_handle_through_both_moves(monkeypatch):
    """Both the yield and the move back must carry the swap handle into
    move_module_excluding -- a bare model.to() in either place pushes the
    swapped layers' master copy onto the card.

    (The real skipping of managed tensors is checked on CUDA in
    tests/test_block_swap_krea2_yield.py.)
    """
    model, swap = _swap_model()
    moves = []
    monkeypatch.setattr(
        krea2_module, "move_module_excluding",
        lambda m, device, s: moves.append((m, str(device), s)),
    )
    monkeypatch.setattr(
        krea2_module, "prepare_sampling_condition", lambda *a, **k: "cond",
    )
    monkeypatch.setattr(krea2_module, "sample_image", lambda *a, **k: "image")
    result = Krea2Family().sample_image(
        model, object(), _OrchestraText(), "a prompt",
        distilled=True, device="cpu", vram_policy="save_vram",
    )
    assert result == "image"
    assert moves == [(model, "cpu", swap), (model, "cpu", swap)]


def test_yield_without_swap_passes_none(monkeypatch):
    """No block swap: swap=None -> move_module_excluding degrades to model.to()."""
    from torch import nn

    model = nn.Sequential(nn.Linear(4, 4))
    moves = []
    monkeypatch.setattr(
        krea2_module, "move_module_excluding",
        lambda m, device, s: moves.append(s),
    )
    monkeypatch.setattr(
        krea2_module, "prepare_sampling_condition", lambda *a, **k: "cond",
    )
    monkeypatch.setattr(krea2_module, "sample_image", lambda *a, **k: "image")
    Krea2Family().sample_image(
        model, object(), _OrchestraText(), "a prompt",
        distilled=True, device="cpu", vram_policy="save_vram",
    )
    assert moves == [None, None]
