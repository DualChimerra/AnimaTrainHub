"""Riemannion + Riemann LoRA tests against dense references.

Adapted from WalkingMeatAxolotl/AnimaLoraStudio (GPL-3.0); local additions cover
narrow layers, layers whose first gradient arrives late, and the Muon +
Use Riemannian selection. Pure CPU, float64 references, fixed seeds.
"""
from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from utils.riemann_lora_adapter import RiemannLoRALinear
from utils.riemannion_optimizer import RiemannionWithAuxAdam, project_lr

M, N, R = 12, 10, 3


def _layer(seed: int = 0, m: int = M, n: int = N, r: int = R) -> RiemannLoRALinear:
    base = nn.Linear(n, m, bias=False)
    nn.init.zeros_(base.weight)
    return RiemannLoRALinear("lora_unet_x", base, rank=r, generator=torch.Generator().manual_seed(seed))


def _set_grads_from_dense(layer: RiemannLoRALinear, g: torch.Tensor) -> None:
    """Same result as autograd (see test_gradient_trick_matches_dense_gradient)."""
    layer.lora_z1.grad = (g @ layer.b_r).float()
    layer.lora_b.grad = (g.mT @ layer.a_l).float()


def _dense_proj(a_l, b_r, z):
    pa = a_l @ a_l.mT
    return pa @ z + (torch.eye(a_l.shape[0], dtype=z.dtype) - pa) @ z @ b_r @ b_r.mT


def _dense_ortho(z):
    """Dense Newton-Schulz with the same coefficients: the same odd polynomial on singular values."""
    from utils.riemannion_optimizer import NS_COEFFICIENTS, NS_STEPS

    a, b, c = NS_COEFFICIENTS
    x = z / z.norm()
    for _ in range(NS_STEPS):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x


def _trunc(y, r):
    u, s, vh = torch.linalg.svd(y, full_matrices=False)
    return (u[:, :r] * s[:r]) @ vh[:r]


def test_gradient_trick_matches_dense_gradient() -> None:
    torch.manual_seed(0)
    layer = _layer()
    with torch.no_grad():
        layer.lora_b.copy_(torch.randn(N, R))
    x = torch.randn(7, N)
    c = torch.randn(7, M)
    y = layer(x)
    torch.testing.assert_close(y, x @ layer.delta_weight().mT)  # the Z1 branch does not change the output
    (c * y).sum().backward()
    g = c.mT @ x  # ∂L/∂ΔW
    torch.testing.assert_close(layer.lora_z1.grad, g @ layer.b_r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(layer.lora_b.grad, g.mT @ layer.a_l, rtol=1e-5, atol=1e-5)


def test_project_lr_matches_dense_projection() -> None:
    torch.manual_seed(0)
    a_l, _ = torch.linalg.qr(torch.randn(M, R, dtype=torch.float64))
    b_r, _ = torch.linalg.qr(torch.randn(N, R, dtype=torch.float64))
    a, b = torch.randn(M, 5, dtype=torch.float64), torch.randn(N, 5, dtype=torch.float64)
    a_dot, b_dot = project_lr(a, b, a_l, b_r)
    torch.testing.assert_close(a_dot @ b_r.mT + a_l @ b_dot.mT, _dense_proj(a_l, b_r, a @ b.mT))


@pytest.mark.parametrize("start_from_zero", [True, False])
def test_two_steps_match_dense_reference(start_from_zero) -> None:
    torch.manual_seed(1)
    layer = _layer()
    if not start_from_zero:
        with torch.no_grad():
            layer.lora_b.copy_(torch.randn(N, R))
            q, _ = torch.linalg.qr(layer.lora_b)
            layer.b_r.copy_(q)
    lr, wd, beta = 0.05, 0.1, 0.9
    opt = RiemannionWithAuxAdam([layer.lora_b, layer.lora_z1], lr=lr, weight_decay=wd, momentum=beta)

    x_ref = layer.delta_weight().double()
    a_l, b_r = layer.a_l.double(), layer.b_r.double()
    mom = None
    for _ in range(2):
        g = torch.randn(M, N, dtype=torch.float64)
        # dense reference
        xi = _dense_proj(a_l, b_r, g)
        if mom is not None:
            xi = beta * _dense_proj(a_l, b_r, mom) + xi
        mom = xi
        step_dir = _dense_proj(a_l, b_r, _dense_ortho(xi))
        x_ref = _trunc(x_ref * (1 - lr * wd) - lr * step_dir, R)
        # implementation
        _set_grads_from_dense(layer, g.float())
        opt.step()
        a_l, b_r = layer.a_l.double(), layer.b_r.double()
        torch.testing.assert_close(layer.delta_weight().double(), x_ref, rtol=1e-4, atol=1e-5)


def test_fits_low_rank_target_and_keeps_invariants() -> None:
    torch.manual_seed(0)
    m, n, r = 24, 32, 4
    layer = _layer(m=m, n=n, r=r)
    target = torch.randn(m, r) @ torch.randn(r, n) * 0.3
    x = torch.randn(256, n)
    y = x @ target.T
    opt = RiemannionWithAuxAdam([layer.lora_b, layer.lora_z1], lr=0.05)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = nn.functional.mse_loss(layer(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        assert torch.all(layer.lora_z1 == 0)
    assert losses[-1] < losses[0] * 0.05, f"{losses[0]} -> {losses[-1]}"
    eye = torch.eye(r)
    torch.testing.assert_close(layer.a_l.mT @ layer.a_l, eye, atol=1e-4, rtol=0)
    torch.testing.assert_close(layer.b_r.mT @ layer.b_r, eye, atol=1e-4, rtol=0)
    assert torch.linalg.matrix_rank(layer.delta_weight()) <= r


def test_export_plain_lora_is_exact_and_loads_back() -> None:
    torch.manual_seed(0)
    layer = _layer()
    with torch.no_grad():
        layer.lora_b.copy_(torch.randn(N, R))
    sd = layer.distilled_lora_state(dtype=torch.float32)
    up, down = sd["lora_unet_x.lora_up.weight"], sd["lora_unet_x.lora_down.weight"]
    scale = float(sd["lora_unet_x.alpha"]) / R
    torch.testing.assert_close(scale * up @ down, layer.delta_weight(), rtol=1e-5, atol=1e-5)

    other = _layer(seed=5)
    other.load_plain_lora_(down, up, scale)
    torch.testing.assert_close(other.delta_weight(), layer.delta_weight(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(other.a_l.mT @ other.a_l, torch.eye(R), atol=1e-5, rtol=0)


def test_state_dict_roundtrip_resumes_identically() -> None:
    from utils.riemann_lora_adapter import RiemannLoRAAdapter

    torch.manual_seed(0)
    x, c = torch.randn(16, N), torch.randn(16, M)

    def make():
        model = nn.Sequential(nn.Linear(N, M, bias=False))
        model[0].weight.data.zero_()
        adapter = RiemannLoRAAdapter(rank=R, preset={"target_name": ["0"]}, seed=0)
        adapter.inject(model)
        opt = RiemannionWithAuxAdam(adapter.get_param_groups(0.0), lr=0.05)
        return model, adapter, opt

    def run(model, opt, n):
        for _ in range(n):
            opt.zero_grad()
            (c * model(x)).sum().backward()
            opt.step()

    model_a, adapter_a, opt_a = make()
    run(model_a, opt_a, 3)
    model_b, adapter_b, opt_b = make()
    adapter_b.load_state_dict(adapter_a.state_dict())
    opt_b.load_state_dict(opt_a.state_dict())
    run(model_a, opt_a, 3)
    run(model_b, opt_b, 3)
    assert torch.equal(adapter_a.loras[0].delta_weight(), adapter_b.loras[0].delta_weight())


def test_unmarked_params_go_to_aux_adam() -> None:
    layer = _layer()
    bias = nn.Parameter(torch.zeros(4))
    opt = RiemannionWithAuxAdam([layer.lora_b, layer.lora_z1, bias], lr=1e-2)
    assert [(g["use_riemann"], len(g["params"])) for g in opt.param_groups] == [(True, 2), (False, 1)]


def test_cholesky_qr2_matches_householder_and_falls_back() -> None:
    from utils.riemannion_optimizer import _qr

    torch.manual_seed(0)
    x = torch.randn(3, 200, 16)
    q, r = _qr(x)
    torch.testing.assert_close(q @ r, x, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(q.mT @ q, torch.eye(16).expand(3, 16, 16), atol=1e-5, rtol=0)
    x[:, :, 3] = 0  # rank deficient -> Cholesky fails -> Householder fallback, Q stays orthonormal
    q, r = _qr(x)
    torch.testing.assert_close(q @ r, x, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(q.mT @ q, torch.eye(16).expand(3, 16, 16), atol=1e-5, rtol=0)


def test_zero_gradient_layer_is_skipped() -> None:
    layer = _layer()
    opt = RiemannionWithAuxAdam([layer.lora_b, layer.lora_z1], lr=0.1)
    before = (layer.a_l.clone(), layer.lora_b.clone())
    _set_grads_from_dense(layer, torch.zeros(M, N))
    opt.step()
    assert torch.equal(layer.a_l, before[0]) and torch.equal(layer.lora_b, before[1])


def test_wraps_int8_base_without_reading_weights() -> None:
    """INT8 base (MOSS): the wrapper only reads weight shapes and calls the quantized forward."""
    base = nn.Linear(N, M, bias=False)
    base.weight = nn.Parameter(torch.randint(-127, 127, (M, N), dtype=torch.int8), requires_grad=False)
    calls = []

    def fake_int8_forward(x):
        calls.append(x.shape)
        return torch.zeros(*x.shape[:-1], M, dtype=x.dtype)

    base.forward = fake_int8_forward
    layer = RiemannLoRALinear("lora_unet_x", base, rank=R, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        layer.lora_b.copy_(torch.randn(N, R))
    x = torch.randn(5, N)
    torch.testing.assert_close(layer(x), x @ layer.delta_weight().mT)
    assert calls == [torch.Size([5, N])]




def test_resume_lora_from_plain_lora_and_rejects_incompatible_file(tmp_path) -> None:
    from safetensors.torch import save_file

    from utils.riemann_lora_adapter import RiemannLoRAAdapter

    def make():
        model = nn.Sequential(nn.Linear(N, M, bias=False))
        adapter = RiemannLoRAAdapter(rank=R, preset={"target_name": ["0"]}, seed=0)
        adapter.inject(model)
        return adapter

    src = make()
    with torch.no_grad():
        src.loras[0].lora_b.copy_(torch.randn(N, R))
    plain = tmp_path / "plain.safetensors"
    src.save(plain)  # the export is a standard LoRA
    dst = make()
    dst.load(plain)
    # exported as bf16, compare at bf16 precision
    torch.testing.assert_close(dst.loras[0].delta_weight(), src.loras[0].delta_weight(), rtol=2e-2, atol=5e-3)

    lokr = tmp_path / "lokr.safetensors"
    save_file({"lora_unet_0.lokr_w1": torch.zeros(2, 2)}, str(lokr))
    with pytest.raises(RuntimeError, match="resume_lora"):
        make().load(lokr)


def test_adapter_tensors_follow_training_device_not_offloaded_weight() -> None:
    """Block swap: offloaded base weights live elsewhere; adapter tensors stay on the training device."""
    base = nn.Linear(N, M, bias=False, device="meta")
    layer = RiemannLoRALinear(
        "lora_unet_x", base, rank=R, generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    for t in (layer.a_l, layer.b_r, layer.lora_b, layer.lora_z1):
        assert t.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gesvda is CUDA-only")
def test_svd_falls_back_when_gesvda_fails_to_converge() -> None:
    """Many repeated zero singular values (warm start from a low effective-rank LoRA) make gesvda fail."""
    from utils.riemannion_optimizer import _svd

    torch.manual_seed(0)
    u, _ = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))
    s = torch.cat([torch.rand(10, dtype=torch.float64) * 0.1, torch.zeros(54, dtype=torch.float64)])
    core = ((u * s) @ v.T).float().cuda().expand(4, 64, 64).contiguous()
    with pytest.raises(torch.linalg.LinAlgError):
        torch.linalg.svd(core, driver="gesvda")
    uu, ss, vh = _svd(core)
    assert uu.is_cuda
    torch.testing.assert_close((uu * ss.unsqueeze(-2)) @ vh, core, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(("m", "n", "r"), [(5, 12, 3), (12, 4, 3), (4, 4, 4)])
def test_narrow_layer_steps_match_dense_reference(m, n, r) -> None:
    # 2r > min(m, n): the orthogonal complement cannot hold r more columns.
    torch.manual_seed(2)
    layer = _layer(m=m, n=n, r=r)
    lr, wd, beta = 0.05, 0.1, 0.9
    opt = RiemannionWithAuxAdam([layer.lora_b, layer.lora_z1], lr=lr, weight_decay=wd, momentum=beta)
    x_ref = layer.delta_weight().double()
    a_l, b_r = layer.a_l.double(), layer.b_r.double()
    mom = None
    for _ in range(3):
        g = torch.randn(m, n, dtype=torch.float64)
        xi = _dense_proj(a_l, b_r, g)
        if mom is not None:
            xi = beta * _dense_proj(a_l, b_r, mom) + xi
        mom = xi
        step_dir = _dense_proj(a_l, b_r, _dense_ortho(xi))
        x_ref = _trunc(x_ref * (1 - lr * wd) - lr * step_dir, r)
        _set_grads_from_dense(layer, g.float())
        opt.step()
        a_l, b_r = layer.a_l.double(), layer.b_r.double()
        torch.testing.assert_close(layer.delta_weight().double(), x_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(layer.a_l.mT @ layer.a_l, torch.eye(r), atol=1e-4, rtol=0)


def test_layer_with_late_first_gradient_matches_standalone_run() -> None:
    torch.manual_seed(3)
    early, late = _layer(seed=0), _layer(seed=1)
    solo_early, solo_late = _layer(seed=0), _layer(seed=1)
    shared = RiemannionWithAuxAdam([early.lora_b, early.lora_z1, late.lora_b, late.lora_z1], lr=0.05)
    opt_early = RiemannionWithAuxAdam([solo_early.lora_b, solo_early.lora_z1], lr=0.05)
    opt_late = RiemannionWithAuxAdam([solo_late.lora_b, solo_late.lora_z1], lr=0.05)
    for step in range(3):
        g_early, g_late = torch.randn(M, N), torch.randn(M, N)
        for layer, solo, g, opt in ((early, solo_early, g_early, opt_early), (late, solo_late, g_late, opt_late)):
            if layer is late and step == 0:
                layer.lora_b.grad = layer.lora_z1.grad = None
                continue
            _set_grads_from_dense(layer, g)
            _set_grads_from_dense(solo, g)
            opt.step()
        shared.step()
        torch.testing.assert_close(early.delta_weight(), solo_early.delta_weight(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(late.delta_weight(), solo_late.delta_weight(), rtol=1e-5, atol=1e-6)


def test_use_riemannian_selects_fixed_rank_adapter_and_rejects_conflicts() -> None:
    from training.adapters import build_adapter
    from training.optimizers import build_optimizer, validate_optimizer
    from utils.riemann_lora_adapter import RiemannLoRAAdapter

    args = types.SimpleNamespace(optimizer_type="muon", use_riemannian=True, lora_type="lora", lora_rank=R)
    assert isinstance(build_adapter(args, preset={"target_name": ["0"]}), RiemannLoRAAdapter)
    validate_optimizer(args)
    layer = _layer()
    assert isinstance(build_optimizer(args, [layer.lora_b, layer.lora_z1], 0.01, 0.0), RiemannionWithAuxAdam)
    for field in ("ema_enabled", "train_norm", "train_text_encoder", "lora_dora"):
        with pytest.raises(ValueError, match=field):
            validate_optimizer(types.SimpleNamespace(**vars(args), **{field: True}))
    with pytest.raises(ValueError, match="LoRA"):
        validate_optimizer(types.SimpleNamespace(**{**vars(args), "lora_type": "lokr"}))


def test_config_migrates_old_riemann_selection_and_hides_ignored_fields() -> None:
    from studio.domain.config_prune import prune_inactive_fields
    from studio.schema import TrainingConfig

    for old in ({"optimizer_type": "riemannion"}, {"lora_type": "riemann"}):
        cfg = TrainingConfig(**old)
        assert (cfg.optimizer_type, cfg.lora_type, cfg.use_riemannian) == ("muon", "lora", True)
    ignored = ("lora_alpha", "lora_dora", "lora_rs", "lora_dropout",
               "lora_rank_dropout", "lora_module_dropout", "lora_reg_dims", "muon_momentum")
    cfg = TrainingConfig(optimizer_type="muon", use_riemannian=True)
    pruned = prune_inactive_fields(cfg.model_dump(mode="python"))
    assert not set(ignored) & set(pruned)
    assert "riemannion_momentum" in pruned
