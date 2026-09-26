from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "_anima_train_for_test", REPO_ROOT / "runtime" / "anima_train.py"
)
try:
    _at = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_at)
    save_training_state = _at.save_training_state
    load_training_state = _at.load_training_state
except Exception as e:
    pytest.skip(f"anima_train.py failed to load: {e}", allow_module_level=True)

from utils.lycoris_adapter import AnimaLycorisAdapter
from training.families.anima.preset import ANIMA_PRESET


class MockDiT(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.output_proj = nn.Linear(d, d, bias=False)


def _make_trained_adapter(seed: int = 42) -> tuple[AnimaLycorisAdapter, MockDiT, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lokr", rank=4, alpha=4, factor=8)
    adapter.inject(model)

    optimizer = torch.optim.AdamW(adapter.get_params(), lr=1e-3)
    for _ in range(3):
        x = torch.randn(2, 128)
        y = model.q_proj(x).sum()
        y.backward()
        optimizer.step()
        optimizer.zero_grad()
    return adapter, model, optimizer


def test_adapter_state_dict_roundtrip_bit_exact(tmp_path):
    adapter, _, _ = _make_trained_adapter()
    sd = adapter.state_dict()

    model2 = MockDiT()
    adapter2 = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lokr", rank=4, alpha=4, factor=8)
    adapter2.inject(model2)
    adapter2.load_state_dict(sd, strict=True)

    sd2 = adapter2.state_dict()
    assert set(sd.keys()) == set(sd2.keys()), "key set mismatch"
    for k in sd:
        if "alpha" in k:
            continue
        assert torch.equal(sd[k], sd2[k]), f"tensor mismatch: {k}"


def test_save_training_state_roundtrip(tmp_path):
    adapter, _, optimizer = _make_trained_adapter()

    state_path = tmp_path / "state.pt"
    save_training_state(
        state_path, adapter, optimizer,
        epoch=2, global_step=42,
        loss_history=[0.5, 0.3, 0.2],
        monitor_state={"foo": "bar"},
    )
    assert state_path.exists()

    model2 = MockDiT()
    adapter2 = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lokr", rank=4, alpha=4, factor=8)
    adapter2.inject(model2)
    optimizer2 = torch.optim.AdamW(adapter2.get_params(), lr=1e-3)

    epoch, step, history, monitor = load_training_state(state_path, adapter2, optimizer2)
    assert epoch == 2
    assert step == 42
    assert history == [0.5, 0.3, 0.2]
    assert monitor == {"foo": "bar"}

    sd1 = adapter.state_dict()
    sd2 = adapter2.state_dict()
    for k in sd1:
        if "alpha" in k:
            continue
        assert torch.equal(sd1[k], sd2[k]), f"weight mismatch after resume: {k}"

    assert len(optimizer2.state) == len(optimizer.state)


def test_save_load_preserves_w1_no_decay_grouping(tmp_path):
    adapter, _, optimizer = _make_trained_adapter()
    state_path = tmp_path / "state.pt"
    save_training_state(state_path, adapter, optimizer, epoch=0, global_step=0)

    model2 = MockDiT()
    adapter2 = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lokr", rank=4, alpha=4, factor=8)
    adapter2.inject(model2)
    optimizer2 = torch.optim.AdamW(adapter2.get_params(), lr=1e-3)
    load_training_state(state_path, adapter2, optimizer2)

    groups = adapter2.get_param_groups(weight_decay=0.01)
    assert len(groups) == 2
    assert any(g["weight_decay"] == 0.0 for g in groups), "missing wd=0 group (w1 should be excluded)"
    assert any(g["weight_decay"] == 0.01 for g in groups), "missing wd=0.01 group (w2 family)"


def test_legacy_state_dict_strict_false_does_not_crash(tmp_path):
    adapter, _, optimizer = _make_trained_adapter()

    state_path = tmp_path / "legacy.pt"
    fake_legacy_sd = {
        "lora_unet_q_proj.lokr_w1": torch.zeros(8, 8),
        "lora_unet_q_proj.lokr_w2_a": torch.zeros(16, 4),
        "lora_unet_q_proj.lokr_w2_b": torch.zeros(4, 16),
    }
    torch.save({
        "lora_state_dict": fake_legacy_sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 0,
        "global_step": 0,
    }, state_path)

    model2 = MockDiT()
    adapter2 = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lokr", rank=4, alpha=4, factor=8)
    adapter2.inject(model2)
    optimizer2 = torch.optim.AdamW(adapter2.get_params(), lr=1e-3)

    load_training_state(state_path, adapter2, optimizer2)


def test_model_eval_cascades_to_lycoris_network():
    model = MockDiT()
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, 
        algo="lokr", rank=4, alpha=4, factor=8,
        rank_dropout=0.1,
    )
    adapter.inject(model)

    assert adapter.network.training is True

    model.eval()
    assert adapter.network.training is False, "lycoris network did not follow model.eval()"
    for lora in adapter.network.loras:
        assert lora.training is False, f"{lora.lora_name} not in eval mode"

    model.train()
    assert adapter.network.training is True


def test_rng_state_restored(tmp_path):
    adapter, _, optimizer = _make_trained_adapter()

    torch.manual_seed(123)
    random.seed(123)
    sample_a = (torch.randn(3).tolist(), random.random())

    state_path = tmp_path / "state.pt"
    save_training_state(state_path, adapter, optimizer, epoch=0, global_step=0)

    expected_next = (torch.randn(3).tolist(), random.random())

    torch.manual_seed(999)
    random.seed(999)
    model2 = MockDiT()
    adapter2 = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lokr", rank=4, alpha=4, factor=8)
    adapter2.inject(model2)
    optimizer2 = torch.optim.AdamW(adapter2.get_params(), lr=1e-3)
    load_training_state(state_path, adapter2, optimizer2)

    actual_next = (torch.randn(3).tolist(), random.random())
    assert actual_next == expected_next, "RNG was not restored correctly"
