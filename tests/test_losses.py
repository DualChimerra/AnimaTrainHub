from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from training.losses import BUILDERS, build_loss, validate_schema_consistency
from training.losses.huber import HuberLoss
from training.losses.huber import build as build_huber
from training.losses.mse import MseLoss
from training.losses.mse import build as build_mse
from training.losses.protocol import LossProtocol


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_mse_matches_f_mse_loss_bitwise():
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 4)
    target = torch.randn(2, 3, 4, 4)
    t = torch.rand(2)

    expected = F.mse_loss(pred, target, reduction="none")
    actual = MseLoss().compute(pred, target, t)
    assert torch.equal(actual, expected)


def test_mse_ignores_t():
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 4)
    target = torch.randn(2, 3, 4, 4)
    mse = MseLoss()
    out1 = mse.compute(pred, target, torch.tensor([0.1, 0.5]))
    out2 = mse.compute(pred, target, torch.tensor([0.9, 0.99]))
    torch.testing.assert_close(out1, out2)


def test_mse_build_returns_instance():
    class Args:
        pass
    loss = build_mse(Args())
    assert isinstance(loss, MseLoss)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_huber_inside_quadratic_region():
    pred = torch.tensor([[0.05]])
    target = torch.tensor([[0.0]])
    t = torch.tensor([0.5])
    huber = HuberLoss(c=0.15)
    expected = torch.tensor([[0.5 * 0.05 * 0.05]])
    actual = huber.compute(pred, target, t)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_huber_outside_linear_region():
    pred = torch.tensor([[1.0]])
    target = torch.tensor([[0.0]])
    t = torch.tensor([0.5])
    huber = HuberLoss(c=0.15)
    expected = torch.tensor([[0.15 * (1.0 - 0.5 * 0.15)]])  # 0.13875
    actual = huber.compute(pred, target, t)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_huber_continuous_at_boundary():
    delta = 0.15
    pred = torch.tensor([[delta]])
    target = torch.tensor([[0.0]])
    t = torch.tensor([0.5])
    huber = HuberLoss(c=delta)
    expected = torch.tensor([[delta * (delta - 0.5 * delta)]])  # 0.5 * delta^2
    actual = huber.compute(pred, target, t)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_huber_smaller_than_mse_for_large_diff():
    pred = torch.tensor([[5.0]])
    target = torch.tensor([[0.0]])
    t = torch.tensor([0.5])
    mse_val = MseLoss().compute(pred, target, t)
    huber_val = HuberLoss(c=0.15).compute(pred, target, t)
    assert huber_val.item() < mse_val.item()  # 12.5 vs 0.74


def test_huber_t_independent():
    huber = HuberLoss(c=0.15)
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 4)
    target = torch.randn(2, 3, 4, 4)
    out1 = huber.compute(pred, target, torch.tensor([0.1, 0.5]))
    out2 = huber.compute(pred, target, torch.tensor([0.9, 0.99]))
    torch.testing.assert_close(out1, out2)


def test_huber_build_reads_args():
    class Args:
        huber_c = 0.3
    h = build_huber(Args())
    assert h.c == 0.3


def test_huber_build_defaults_when_args_missing():
    class OldArgs:
        pass
    h = build_huber(OldArgs())
    assert h.c == 0.15


# ---------------------------------------------------------------------------
# plugin registry
# ---------------------------------------------------------------------------


def test_builders_dict_keys():
    assert set(BUILDERS) == {"mse", "huber"}


def test_build_loss_dispatches_mse():
    class Args:
        loss_type = "mse"
    loss = build_loss(Args())
    assert isinstance(loss, MseLoss)


def test_build_loss_dispatches_huber():
    class Args:
        loss_type = "huber"
        huber_c = 0.2
    loss = build_loss(Args())
    assert isinstance(loss, HuberLoss)
    assert loss.c == 0.2


def test_build_loss_unknown_type_raises():
    class Args:
        loss_type = "ghost"
    with pytest.raises(ValueError, match="loss_type"):
        build_loss(Args())


def test_build_loss_defaults_to_mse_when_arg_missing():
    class OldArgs:
        pass
    loss = build_loss(OldArgs())
    assert isinstance(loss, MseLoss)


def test_validate_schema_consistency_passes_on_clean_dev():
    validate_schema_consistency()


# ---------------------------------------------------------------------------
# runtime_checkable Protocol
# ---------------------------------------------------------------------------


def test_mse_satisfies_loss_protocol():
    assert isinstance(MseLoss(), LossProtocol)


def test_huber_satisfies_loss_protocol():
    assert isinstance(HuberLoss(), LossProtocol)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss", [
    MseLoss(),
    HuberLoss(c=0.15),
])
def test_compute_shape_and_finite_on_latent_like_input(loss):
    torch.manual_seed(42)
    pred = torch.randn(4, 3, 8, 8)
    target = torch.randn(4, 3, 8, 8)
    t = torch.rand(4)
    out = loss.compute(pred, target, t)
    assert out.shape == pred.shape
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# HuberLoss dtype / shape edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_huber_handles_low_precision_dtypes(dtype):
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 4, dtype=dtype)
    target = torch.randn(2, 3, 4, 4, dtype=dtype)
    t = torch.rand(2)
    huber = HuberLoss(c=0.15)
    out = huber.compute(pred, target, t)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


def test_huber_handles_5d_input_shape():
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 3, 8, 8)  # (B, C, T, H, W) 5D
    target = torch.randn(2, 4, 3, 8, 8)
    t = torch.rand(2)
    huber = HuberLoss(c=0.15)
    out = huber.compute(pred, target, t)
    assert out.shape == pred.shape
    assert torch.isfinite(out).all()


def test_huber_handles_3d_input_shape():
    torch.manual_seed(0)
    pred = torch.randn(4, 16, 32)
    target = torch.randn(4, 16, 32)
    t = torch.rand(4)
    huber = HuberLoss(c=0.15)
    out = huber.compute(pred, target, t)
    assert out.shape == pred.shape


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_validate_schema_consistency_detects_extra_builder(monkeypatch):
    from training.losses import BUILDERS as ORIG_BUILDERS

    fake_builders = dict(ORIG_BUILDERS)
    fake_builders["ghost"] = lambda args: None
    monkeypatch.setattr("training.losses.BUILDERS", fake_builders)
    with pytest.raises(RuntimeError, match="loss registry out of sync with schema"):
        validate_schema_consistency()


def test_validate_schema_consistency_detects_missing_builder(monkeypatch):
    from training.losses import BUILDERS as ORIG_BUILDERS

    fake_builders = {k: v for k, v in ORIG_BUILDERS.items() if k != "huber"}
    monkeypatch.setattr("training.losses.BUILDERS", fake_builders)
    with pytest.raises(RuntimeError, match="loss registry out of sync with schema"):
        validate_schema_consistency()


def test_infonoise_raw_mse_decoupled_from_loss_type():
    from training.timestep_samplers.infonoise import InfoNoiseScheduler

    torch.manual_seed(0)
    pred = torch.randn(4, 4, 8, 8)
    target = torch.randn(4, 4, 8, 8)
    t = torch.rand(4).clamp(1e-3, 1 - 1e-3)

    huber = HuberLoss(c=0.15)
    loss_per_sample = huber.compute(pred.float(), target.float(), t)
    huber_per_sample = loss_per_sample.mean(dim=list(range(1, loss_per_sample.dim())))

    raw_mse_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").detach()
    raw_mse = raw_mse_per_sample.mean(dim=list(range(1, raw_mse_per_sample.dim())))

    assert not torch.allclose(raw_mse, huber_per_sample, rtol=1e-2), (
        "test invalid: huber output too close to MSE, rewrite the test or use a more extreme c."
    )

    mse_reference = MseLoss().compute(pred.float(), target.float(), t)
    mse_per_sample = mse_reference.mean(dim=list(range(1, mse_reference.dim())))
    torch.testing.assert_close(raw_mse, mse_per_sample)

    scheduler = InfoNoiseScheduler(K=16, N_warm=10, M=5, B=10, N_min=1)
    scheduler.record(t.detach(), raw_mse)
    assert scheduler._internal_step == 1
    total_recorded = sum(len(buf) for buf in scheduler._fifo)
    assert total_recorded == 4
