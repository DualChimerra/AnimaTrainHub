from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.families.anima.leap import (  # noqa: E402
    _finalize_loss,
    bridge_training_step,
    lagrange_training_step,
    leap_training_step,
    sample_activation_timesteps,
    sample_two_timesteps,
    sparse_training_step,
)


class _IdentityVelocityModel:

    def __init__(self, x0: torch.Tensor, noise: torch.Tensor):
        self._v = noise - x0
        self.n_calls = 0

    def __call__(self, latents, timesteps, cross, padding_mask=None):  # noqa: ARG002
        self.n_calls += 1
        return self._v.clone()


def _make_batch(bs: int = 4, c: int = 3, h: int = 8, w: int = 8):
    torch.manual_seed(0)
    x0 = torch.randn(bs, c, h, w)
    noise = torch.randn(bs, c, h, w)
    cross = torch.zeros(bs, 1, 16)
    pad_mask = torch.zeros(bs, 1, h, w)
    return x0, noise, cross, pad_mask


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ngc", [0.0, 0.3, 1.0])
def test_original_exact_reconstruction(ngc: float) -> None:
    x0, noise, cross, pad_mask = _make_batch()
    model = _IdentityVelocityModel(x0, noise)
    t_k, t_j = sample_two_timesteps(x0.shape[0], x0.device, min_gap=0.1)
    loss = leap_training_step(
        model, x0, noise, cross, pad_mask, t_k, t_j, nested_grad_coe=ngc,
    )
    assert torch.all(loss < 1e-10), f"original loss not ~0: {loss}"


@pytest.mark.parametrize("ngc", [0.0, 0.3, 1.0])
def test_bridge_exact_reconstruction(ngc: float) -> None:
    x0, noise, cross, pad_mask = _make_batch()
    model = _IdentityVelocityModel(x0, noise)
    t_k, t_j = sample_two_timesteps(x0.shape[0], x0.device, min_gap=0.1)
    loss = bridge_training_step(
        model, x0, noise, cross, pad_mask, t_k, t_j, nested_grad_coe=ngc,
    )
    assert torch.all(loss < 1e-10), f"bridge loss not ~0: {loss}"


@pytest.mark.parametrize("ngc", [0.0, 0.3, 1.0])
def test_lagrange_exact_reconstruction(ngc: float) -> None:
    x0, noise, cross, pad_mask = _make_batch()
    model = _IdentityVelocityModel(x0, noise)
    t_k, t_j = sample_two_timesteps(x0.shape[0], x0.device, min_gap=0.1)
    loss = lagrange_training_step(
        model, x0, noise, cross, pad_mask, t_k, t_j, nested_grad_coe=ngc,
    )
    assert torch.all(loss < 1e-10), f"lagrange loss not ~0: {loss}"


@pytest.mark.parametrize("k", [2, 3, 5, 8])
def test_sparse_exact_reconstruction(k: int) -> None:
    x0, noise, cross, pad_mask = _make_batch()
    model = _IdentityVelocityModel(x0, noise)
    t_steps = sample_activation_timesteps(x0.shape[0], x0.device, k=k)
    loss = sparse_training_step(
        model, x0, noise, cross, pad_mask, t_steps,
    )
    assert torch.all(loss < 1e-10), f"sparse(k={k}) loss not ~0: {loss}"


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------


class _PositionVelocityModel:

    def __init__(self) -> None:
        self.n_calls = 0

    def __call__(self, latents, timesteps, cross, padding_mask=None):  # noqa: ARG002
        self.n_calls += 1
        return latents.clone()


def test_lagrange_simpson_weights_and_endpoint() -> None:
    x0, noise, cross, pad_mask = _make_batch()
    bs = x0.shape[0]
    t_k = torch.full((bs,), 0.8)
    t_j = torch.full((bs,), 0.4)

    loss = lagrange_training_step(
        _PositionVelocityModel(), x0, noise, cross, pad_mask, t_k, t_j,
        nested_grad_coe=1.0,
    )

    view = (-1, *([1] * (x0.ndim - 1)))
    k = t_k.view(*view)
    j = t_j.view(*view)
    x_k = (1.0 - k) * x0 + k * noise
    x_j_real = (1.0 - j) * x0 + j * noise
    m1 = (k + j) * 0.5
    x_m1 = (1.0 - m1) * x0 + m1 * noise
    v_seg1 = (x_k + 4.0 * x_m1 + x_j_real) / 6.0
    x_hat_j = x_k - (k - j) * v_seg1
    x_j_in = x_j_real
    m2 = j * 0.5
    x_m2 = (1.0 - m2) * x0 + m2 * noise
    v_seg2 = (x_j_in + 4.0 * x_m2 + x0) / 6.0
    x_hat_0_expected = x_j_in - j * v_seg2
    expected_loss = (x_hat_0_expected.float() - x0.float()).pow(2).mean(dim=(1, 2, 3))

    assert torch.allclose(loss, expected_loss, atol=1e-6), (
        f"lagrange x̂0 deviates from the Simpson(1:4:1)+x0 endpoint analytic value: {loss} vs {expected_loss}"
    )

    v_j_start = x_j_in  # v=x
    x_hat_0_euler = x_j_in - j * v_j_start
    v_seg2_euler = (x_j_in + 4.0 * x_m2 + x_hat_0_euler) / 6.0
    x_hat_0_euler_full = x_j_in - j * v_seg2_euler
    euler_loss = (x_hat_0_euler_full.float() - x0.float()).pow(2).mean(dim=(1, 2, 3))
    assert not torch.allclose(loss, euler_loss, atol=1e-6), (
        "segment endpoint ground truth vs Euler prediction should be distinguishable; equal means the position-field test degenerated"
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_forward_counts() -> None:
    x0, noise, cross, pad_mask = _make_batch()
    t_k, t_j = sample_two_timesteps(x0.shape[0], x0.device, min_gap=0.1)

    m = _IdentityVelocityModel(x0, noise)
    leap_training_step(m, x0, noise, cross, pad_mask, t_k, t_j)
    assert m.n_calls == 2, "original should be 2x forward"

    m = _IdentityVelocityModel(x0, noise)
    bridge_training_step(m, x0, noise, cross, pad_mask, t_k, t_j)
    assert m.n_calls == 2, "bridge should be 2x forward"

    m = _IdentityVelocityModel(x0, noise)
    lagrange_training_step(m, x0, noise, cross, pad_mask, t_k, t_j)
    assert m.n_calls == 6, "lagrange should be 6x forward (three points on each of two segments)"

    for k in (2, 3, 5):
        m = _IdentityVelocityModel(x0, noise)
        t_steps = sample_activation_timesteps(x0.shape[0], x0.device, k=k)
        sparse_training_step(m, x0, noise, cross, pad_mask, t_steps)
        assert m.n_calls == k, f"sparse(k={k}) should be {k}x forward"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5, 8])
def test_activation_timesteps_descending_and_open_interval(k: int) -> None:
    t = sample_activation_timesteps(64, torch.device("cpu"), k=k)
    assert t.shape == (64, k)
    diffs = t[:, :-1] - t[:, 1:]
    assert torch.all(diffs > 0), "each row should be strictly descending"
    assert torch.all(t > 0.0) and torch.all(t < 1.0), "should lie within the open interval (0,1)"


def test_activation_timesteps_spans_full_range() -> None:
    t = sample_activation_timesteps(256, torch.device("cpu"), k=4)
    assert t[:, 0].mean() > 0.6, "noise-end support points did not spread to high t"
    assert t[:, -1].mean() < 0.4, "data-end support points did not spread to low t"


def test_activation_timesteps_rejects_small_k() -> None:
    with pytest.raises(ValueError, match="k>=2"):
        sample_activation_timesteps(4, torch.device("cpu"), k=1)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_finalize_loss_no_weighting() -> None:
    x0 = torch.randn(4, 3, 8, 8)
    x_hat_0 = x0 + 0.1
    loss = _finalize_loss(x_hat_0, x0, traj_sim_weighting=False)
    expected = (x_hat_0.float() - x0.float()).pow(2).mean(dim=(1, 2, 3))
    assert torch.allclose(loss, expected)


def test_finalize_loss_traj_sim_two_endpoints() -> None:
    x0 = torch.randn(4, 3, 8, 8)
    x_hat_0 = x0 + 0.1
    x_inter_real = torch.randn(4, 3, 8, 8)
    x_hat_inter = x_inter_real + 0.2
    loss = _finalize_loss(
        x_hat_0, x0, x_hat_inter=x_hat_inter, x_inter_real=x_inter_real,
        traj_sim_weighting=True, traj_sim_min=0.01,
    )
    base = (x_hat_0.float() - x0.float()).pow(2).mean(dim=(1, 2, 3))
    d0 = (x0 - x_hat_0).abs().mean(dim=(1, 2, 3)).clamp(min=0.01)
    di = (x_inter_real - x_hat_inter).abs().mean(dim=(1, 2, 3)).clamp(min=0.01)
    assert torch.allclose(loss, base / (di + d0))


def test_finalize_loss_traj_sim_sparse_endpoint_only() -> None:
    x0 = torch.randn(4, 3, 8, 8)
    x_hat_0 = x0 + 0.1
    loss = _finalize_loss(
        x_hat_0, x0, traj_sim_weighting=True, traj_sim_min=0.01,
    )
    base = (x_hat_0.float() - x0.float()).pow(2).mean(dim=(1, 2, 3))
    d0 = (x0 - x_hat_0).abs().mean(dim=(1, 2, 3)).clamp(min=0.01)
    assert torch.allclose(loss, base / d0)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["original", "sparse", "bridge", "lagrange"])
def test_config_leap_exclusive_infonoise(variant: str) -> None:
    from pydantic import ValidationError

    from studio.schema import TrainingConfig

    with pytest.raises(ValidationError, match="infonoise"):
        TrainingConfig(
            transformer_path="x", data_dir="x", output_dir="x",
            leap_enabled=True, leap_variant=variant, infonoise_enabled=True,
        )


@pytest.mark.parametrize("variant", ["original", "sparse", "bridge", "lagrange"])
def test_config_leap_exclusive_huber(variant: str) -> None:
    from pydantic import ValidationError

    from studio.schema import TrainingConfig

    with pytest.raises(ValidationError, match="huber"):
        TrainingConfig(
            transformer_path="x", data_dir="x", output_dir="x",
            leap_enabled=True, leap_variant=variant, loss_type="huber",
        )


@pytest.mark.parametrize("variant", ["original", "sparse", "bridge", "lagrange"])
def test_config_leap_variant_accepts_all(variant: str) -> None:
    from studio.schema import TrainingConfig

    cfg = TrainingConfig(
        transformer_path="x", data_dir="x", output_dir="x",
        leap_enabled=True, leap_variant=variant,
    )
    assert cfg.leap_variant == variant
