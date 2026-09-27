"""Resume compatibility tests for Automagic v1/v2 + Lion + CAME.

Exercises the real save_training_state -> load_training_state chain (torch.save/load
serialization), covering each optimizer's resume risk points:

- Automagic v1: lr_mask (Auto8bitTensor) serializes to a plain dict, int8/bool state is moved
  back to the param device after load, bf16 Kahan shift dtype stays stable (PyTorch load casts
  float state to the param dtype -- shift shares the param dtype so the roundtrip doesn't drift).
- Automagic v2: scalar lr / second moment are restored to fp32 after the load fixup (PyTorch
  would otherwise downcast them to bf16 under a bf16 param); the fused backward hook keeps
  working after resume.
- Lion: standard PyTorch state (exp_avg), no custom hooks, verifies values are preserved and
  training can continue.
- CAME: factored state (row/col second moment + instability) preserves values; under a bf16
  param, the load fixup restores fp32 state (PyTorch load would otherwise cast to the param
  dtype).

Note: pause snapshot freeze (bootstrap) guarantees the UI resume path never changes the
optimizer type / variant; a manual cross-variant resume (editing yaml via the CLI) is
unsupported -- v2 loading v1 state fails fast with a KeyError on the first update rather than
failing silently.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from utils.optimizer_utils import CAME, Automagic, Automagic2, Lion


class _StubInjector:
    """state.py only requires state_dict() / load_state_dict(sd, strict=False)."""

    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"stub.weight": torch.zeros(1)}

    def load_state_dict(self, sd, strict=True):
        self.loaded = sd
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def _roundtrip(tmp_path: Path, optimizer, model_factory, optimizer_factory):
    """save -> new model/optimizer -> load, returns (new optimizer, new model)."""
    from training.state import load_training_state, save_training_state

    ckpt = tmp_path / "state.pt"
    save_training_state(
        ckpt, _StubInjector(), optimizer, epoch=3, global_step=42,
        loss_history=[1.0, 0.5],
    )

    model2 = model_factory()
    optim2 = optimizer_factory(model2)
    epoch, global_step, loss_history, _monitor = load_training_state(
        ckpt, _StubInjector(), optim2,
    )
    assert (epoch, global_step) == (3, 42)
    assert loss_history == [1.0, 0.5]
    return optim2, model2


# ---------------------------------------------------------------------------
# Automagic v1
# ---------------------------------------------------------------------------


def test_automagic_v1_resume_roundtrip_fp32(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False)
    optim = Automagic(model.parameters(), lr=1e-5, lr_bump=5e-5)
    for _ in range(3):
        model(torch.randn(2, 8)).sum().backward()
        optim.step()
        optim.zero_grad()
    lr_before = optim.get_avg_learning_rate()
    p1 = next(iter(model.parameters()))
    mask_before = optim.state[p1]["lr_mask"].dequantize()

    optim2, model2 = _roundtrip(
        tmp_path, optim,
        lambda: nn.Linear(8, 8, bias=False),
        lambda m: Automagic(m.parameters(), lr=1e-5, lr_bump=5e-5),
    )
    p2 = next(iter(model2.parameters()))
    st = optim2.state[p2]

    # lr_mask values are preserved exactly (int8 quantized + scale serialize to a plain dict)
    assert torch.allclose(st["lr_mask"].dequantize(), mask_before, atol=1e-9)
    assert optim2.get_avg_learning_rate() == pytest.approx(lr_before)
    # can continue training: the lr trajectory continues from the restored value instead of resetting
    model2(torch.randn(2, 8)).sum().backward()
    optim2.step()
    assert torch.isfinite(p2).all()


def test_automagic_v1_resume_bf16_kahan_shift_stable(tmp_path: Path) -> None:
    """bf16 training resume: shift shares dtype with param, doesn't drift after the roundtrip;
    bool last_polarity / int8 lr_mask are moved back to the param device (a no-op on CPU, but
    the dtype assertions still hold)."""
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    optim = Automagic(model.parameters(), lr=1e-5)
    model(torch.randn(2, 8, dtype=torch.bfloat16)).sum().backward()
    optim.step()
    optim.zero_grad()

    optim2, model2 = _roundtrip(
        tmp_path, optim,
        lambda: nn.Linear(8, 8, bias=False).to(torch.bfloat16),
        lambda m: Automagic(m.parameters(), lr=1e-5),
    )
    p2 = next(iter(model2.parameters()))
    st = optim2.state[p2]
    assert st["shift"].dtype == p2.dtype, "Kahan shift dtype must stay stable after resume"
    assert st["lr_mask"].quantized.dtype == torch.int8
    # the Kahan path continues to work
    model2(torch.randn(2, 8, dtype=torch.bfloat16)).sum().backward()
    optim2.step()
    assert torch.isfinite(p2).all()


# ---------------------------------------------------------------------------
# Automagic v2
# ---------------------------------------------------------------------------


def test_automagic_v2_resume_scalar_lr_fp32_and_hook_alive(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False)
    optim = Automagic2(model.parameters(), lr=1e-6, lr_bump=1e-6)
    for _ in range(3):
        model(torch.randn(2, 8)).sum().backward()  # the update happens inside the hook
        optim.zero_grad()
    p1 = next(iter(model.parameters()))
    lr_before = float(optim.state[p1]["lr"])
    step_before = optim.state[p1]["step"]

    optim2, model2 = _roundtrip(
        tmp_path, optim,
        lambda: nn.Linear(8, 8, bias=False),
        lambda m: Automagic2(m.parameters(), lr=1e-6, lr_bump=1e-6),
    )
    p2 = next(iter(model2.parameters()))
    st = optim2.state[p2]
    assert st["lr"].dtype == torch.float32
    assert float(st["lr"]) == pytest.approx(lr_before)
    assert st["step"] == step_before

    # the fused hook keeps working after resume: backward updates the param and clears grad
    before = p2.detach().clone()
    model2(torch.randn(2, 8)).sum().backward()
    assert p2.grad is None, "the fused hook should consume grad during backward"
    assert not torch.equal(p2.detach(), before), "the hook should keep updating params after resume"
    assert st["step"] == step_before + 1, "step should keep incrementing from the restored value"


def test_automagic_v2_resume_bf16_second_moment_fp32(tmp_path: Path) -> None:
    """Under a bf16 param, PyTorch load downcasts fp32 state to bf16 --
    Automagic2.load_state_dict's fixup must restore fp32."""
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    optim = Automagic2(model.parameters(), lr=1e-6)
    model(torch.randn(2, 8, dtype=torch.bfloat16)).sum().backward()
    optim.zero_grad()

    optim2, model2 = _roundtrip(
        tmp_path, optim,
        lambda: nn.Linear(8, 8, bias=False).to(torch.bfloat16),
        lambda m: Automagic2(m.parameters(), lr=1e-6),
    )
    p2 = next(iter(model2.parameters()))
    st = optim2.state[p2]
    assert st["lr"].dtype == torch.float32
    assert st["exp_avg_sq_row"].dtype == torch.float32
    assert st["exp_avg_sq_col"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Lion
# ---------------------------------------------------------------------------


def test_lion_resume_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False)
    optim = Lion(model.parameters(), lr=1e-5)
    for _ in range(3):
        model(torch.randn(2, 8)).sum().backward()
        optim.step()
        optim.zero_grad()
    p1 = next(iter(model.parameters()))
    exp_avg_before = optim.state[p1]["exp_avg"].clone()

    optim2, model2 = _roundtrip(
        tmp_path, optim,
        lambda: nn.Linear(8, 8, bias=False),
        lambda m: Lion(m.parameters(), lr=1e-5),
    )
    p2 = next(iter(model2.parameters()))
    assert torch.allclose(optim2.state[p2]["exp_avg"], exp_avg_before)
    assert optim2.param_groups[0]["lr"] == 1e-5
    # training can continue
    model2(torch.randn(2, 8)).sum().backward()
    optim2.step()
    assert torch.isfinite(p2).all()


# ---------------------------------------------------------------------------
# CAME
# ---------------------------------------------------------------------------


def test_came_resume_roundtrip_fp32(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False)
    optim = CAME(model.parameters(), lr=1e-3)
    for _ in range(3):
        model(torch.randn(2, 8)).sum().backward()
        optim.step()
        optim.zero_grad()
    p1 = next(iter(model.parameters()))
    st1 = optim.state[p1]
    row_before = st1["exp_avg_sq_row"].clone()
    res_row_before = st1["exp_avg_res_row"].clone()
    exp_avg_before = st1["exp_avg"].clone()

    optim2, model2 = _roundtrip(
        tmp_path, optim,
        lambda: nn.Linear(8, 8, bias=False),
        lambda m: CAME(m.parameters(), lr=1e-3),
    )
    p2 = next(iter(model2.parameters()))
    st2 = optim2.state[p2]
    assert torch.allclose(st2["exp_avg"], exp_avg_before)
    assert torch.allclose(st2["exp_avg_sq_row"], row_before)
    assert torch.allclose(st2["exp_avg_res_row"], res_row_before)
    assert st2["step"] == 3
    # training can continue
    model2(torch.randn(2, 8)).sum().backward()
    optim2.step()
    assert torch.isfinite(p2).all()


def test_came_resume_bf16_state_restored_fp32(tmp_path: Path) -> None:
    """Under a bf16 param, PyTorch load downcasts fp32 state to bf16 --
    CAME.load_state_dict's fixup must restore fp32."""
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    optim = CAME(model.parameters(), lr=1e-3)
    model(torch.randn(2, 8, dtype=torch.bfloat16)).sum().backward()
    optim.step()
    optim.zero_grad()

    optim2, model2 = _roundtrip(
        tmp_path, optim,
        lambda: nn.Linear(8, 8, bias=False).to(torch.bfloat16),
        lambda m: CAME(m.parameters(), lr=1e-3),
    )
    p2 = next(iter(model2.parameters()))
    st = optim2.state[p2]
    for key in ("exp_avg", "exp_avg_sq_row", "exp_avg_sq_col",
                "exp_avg_res_row", "exp_avg_res_col"):
        assert st[key].dtype == torch.float32, f"{key} should be restored to fp32 after resume"
    # the stochastic rounding writeback continues to work
    model2(torch.randn(2, 8, dtype=torch.bfloat16)).sum().backward()
    optim2.step()
    assert torch.isfinite(p2).all()
