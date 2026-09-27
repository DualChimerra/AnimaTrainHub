"""Optimizer utils tests -- covers the PPSF integration + Automagic v1/v2.

- optimizer_eval_mode context manager behavior for PPSF / non-PPSF optimizers
- create_prodigy_plus_schedulefree factory gives a friendly error when the dependency is missing
- factory forces lr=1.0 + betas default-override logic
- Automagic v1: step, bf16 Kahan (shift shares param dtype, aligned with diffusion-pipe), state_dict roundtrip
- Automagic v2: fused backward hook, scalar lr
"""
from __future__ import annotations

import builtins
import importlib
import sys
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from utils.optimizer_utils import (
    CAME,
    Automagic,
    Automagic2,
    Lion,
    create_automagic,
    create_automagic_v2,
    create_came,
    create_optimizer,
    create_prodigy_plus_schedulefree,
    get_optimizer_monitor_metrics,
    optimizer_eval_mode,
)


# ---------------------------------------------------------------------------
# optimizer_eval_mode
# ---------------------------------------------------------------------------


def test_eval_mode_noop_for_plain_adamw() -> None:
    """AdamW has no .eval/.train methods, the ctx should silently no-op without raising."""
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # AdamW has no .train / .eval methods -- the ctx should silently pass through
    with optimizer_eval_mode(optim):
        # calls normally -- no side effect
        assert optim.param_groups[0]["lr"] == 1e-3


def test_eval_mode_calls_eval_and_train_on_schedulefree_like() -> None:
    """A PPSF-like optimizer calls .eval() on ctx enter, .train() on exit."""
    fake_opt = MagicMock(spec=["eval", "train"])
    with optimizer_eval_mode(fake_opt):
        fake_opt.eval.assert_called_once_with()
        fake_opt.train.assert_not_called()
    fake_opt.train.assert_called_once_with()


def test_eval_mode_restores_train_on_exception() -> None:
    """Must still switch back to .train() when an exception is raised inside the ctx -- otherwise the training weights stay stuck in averaged state forever."""
    fake_opt = MagicMock(spec=["eval", "train"])

    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with optimizer_eval_mode(fake_opt):
            raise Boom()

    fake_opt.eval.assert_called_once_with()
    fake_opt.train.assert_called_once_with()


def test_eval_mode_skips_if_only_partial_methods() -> None:
    """An optimizer with only .eval and no .train (or vice versa) -- treated as non-PPSF, no-op.
    Prevents accidentally calling a one-sided method and corrupting internal state."""
    fake_opt = MagicMock(spec=["eval"])  # only eval, no train
    with optimizer_eval_mode(fake_opt):
        pass
    fake_opt.eval.assert_not_called()


# ---------------------------------------------------------------------------
# get_optimizer_monitor_metrics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lion
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Automagic
# ---------------------------------------------------------------------------


def test_create_automagic_optimizer_updates_parameters() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    # start lr set to max_lr so a single step's change clears torch.allclose's default tolerance;
    # the default lr=1e-6 gives a ~1e-7 change per step, which would be judged "close" under atol=1e-8+rtol=1e-5.
    optim = create_optimizer(
        "automagic",
        model.parameters(),
        learning_rate=1e-3,
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=1e-5,
        weight_decay=0.0,
    )

    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optim.step()

    assert isinstance(optim, Automagic)
    assert not torch.allclose(model.weight, torch.ones_like(model.weight))
    assert "lr_mask" in optim.state[model.weight]
    assert optim.get_avg_learning_rate() >= optim.min_lr


def test_automagic_monitor_metrics_use_dynamic_lr() -> None:
    model = nn.Linear(2, 1)
    optim = create_optimizer("automagic", model.parameters(), learning_rate=1e-6)
    for p in model.parameters():
        optim.initialize_state(p)
    metrics = get_optimizer_monitor_metrics(optim)
    assert metrics["lr"] == pytest.approx(1e-6)
    assert metrics["actual_lr"] == pytest.approx(1e-6)


def test_automagic_sign_agreement_increases_lr() -> None:
    """Same-sign gradients across two steps → sign_agreement>0 → lr_mask bumps up."""
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optim = create_optimizer(
        "automagic",
        model.parameters(),
        learning_rate=1e-5,
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=5e-5,  # larger than the 8-bit quantization granularity, guarantees observability
        weight_decay=0.0,
    )

    # step 1: establish last_polarity
    model(torch.ones(1, 2)).sum().backward()
    optim.step()
    lr_after_step1 = optim.get_avg_learning_rate()
    optim.zero_grad()

    # step 2: same-sign gradient -> sign_agreement>0 -> lr should go up
    model(torch.ones(1, 2)).sum().backward()
    optim.step()
    lr_after_step2 = optim.get_avg_learning_rate()

    assert lr_after_step2 > lr_after_step1, (
        f"lr_mask should bump up when sign-agreement is positive, but {lr_after_step1} -> {lr_after_step2}"
    )


def test_automagic_sign_flip_decreases_lr() -> None:
    """Sign-flipped gradients between steps → sign_agreement<0 → lr_mask bumps down."""
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optim = create_optimizer(
        "automagic",
        model.parameters(),
        learning_rate=5e-4,  # start a bit higher, leaving room to decrease
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=5e-5,
        weight_decay=0.0,
    )

    # step 1: positive gradient
    model(torch.ones(1, 2)).sum().backward()
    optim.step()
    lr_after_step1 = optim.get_avg_learning_rate()
    optim.zero_grad()

    # step 2: negative gradient (sign flipped)
    (-model(torch.ones(1, 2)).sum()).backward()
    optim.step()
    lr_after_step2 = optim.get_avg_learning_rate()

    assert lr_after_step2 < lr_after_step1, (
        f"lr_mask should bump down on a sign flip, but {lr_after_step1} -> {lr_after_step2}"
    )


def test_automagic_does_not_register_grad_accum_hook() -> None:
    """Aligned with upstream diffusion-pipe: does not register a stochastic-rounding grad
    accum hook, avoiding a silent conflict with AMP GradScaler.unscale_ / clip_grad_norm_."""
    model = nn.Linear(2, 2).to(torch.bfloat16)
    optim = create_optimizer("automagic", model.parameters(), learning_rate=1e-6)

    # p.grad should remain after backward (if the hook were running, it would delete p.grad and move it to _accum_grad)
    out = model(torch.ones(1, 2, dtype=torch.bfloat16)).sum()
    out.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None, (
                "p.grad was unexpectedly deleted; no accum hook should be active"
            )
            assert not hasattr(p, "_accum_grad"), (
                "_accum_grad buffer present; stochastic-rounding hook must not run"
            )


def test_automagic_bf16_step_runs_finite() -> None:
    """A single bf16 training step: the Kahan summation path + lr_mask update normally, loss/p stay finite."""
    model = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
    optim = create_optimizer(
        "automagic",
        model.parameters(),
        learning_rate=1e-5,
        min_lr=1e-7,
        max_lr=1e-3,
        weight_decay=0.0,
    )
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    loss = model(x).sum()
    loss.backward()
    optim.step()

    for p in model.parameters():
        assert torch.isfinite(p).all(), "bf16 Kahan path produced non-finite p"
        # the Kahan path must create the shift state
        assert "shift" in optim.state[p]
        assert optim.state[p]["shift"].dtype == torch.bfloat16


def test_automagic_state_dict_roundtrip() -> None:
    """state_dict / load_state_dict preserves lr_mask (Auto8bitTensor) + values stay consistent."""
    model1 = nn.Linear(4, 4, bias=False)
    optim1 = create_optimizer(
        "automagic", model1.parameters(),
        learning_rate=1e-5, min_lr=1e-7, max_lr=1e-3, lr_bump=5e-5,
    )
    # run two steps to establish state
    for _ in range(2):
        model1(torch.randn(2, 4)).sum().backward()
        optim1.step()
        optim1.zero_grad()
    sd = optim1.state_dict()

    # load with a second optim
    model2 = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.copy_(p1)
    optim2 = create_optimizer(
        "automagic", model2.parameters(),
        learning_rate=1e-5, min_lr=1e-7, max_lr=1e-3, lr_bump=5e-5,
    )
    optim2.load_state_dict(sd)

    # lr_mask's quantized + scale both must match
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        s1 = optim1.state[p1]
        s2 = optim2.state[p2]
        assert "lr_mask" in s2
        assert torch.equal(s1["lr_mask"].quantized, s2["lr_mask"].quantized)
        assert s1["lr_mask"].scale == s2["lr_mask"].scale


# ---------------------------------------------------------------------------
# CAME
# ---------------------------------------------------------------------------


def _came_reference_step(
    p: torch.Tensor,
    grad: torch.Tensor,
    lr: float,
    betas: tuple[float, float, float],
    eps: tuple[float, float],
    clip_threshold: float,
) -> torch.Tensor:
    """Independently recompute the post-first-step parameter value per the official
    implementation (yangluo7/CAME, Algorithm 1).

    The factored path for the first step (all EMA state starting at 0), used to codify the
    formula against regressions (AGENTS.md §0.4: verify an external paper's implementation
    first, then pin it down with a unit test).
    """
    beta1, beta2, beta3 = betas
    eps1, eps2 = eps

    def rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    def approx(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        r = (row / row.mean(dim=-1, keepdim=True)).rsqrt().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        return r * c

    sq = grad**2 + eps1
    row = (1.0 - beta2) * sq.mean(dim=-1)
    col = (1.0 - beta2) * sq.mean(dim=-2)
    update = approx(row, col) * grad
    update = update / torch.clamp(rms(update) / clip_threshold, min=1.0)
    exp_avg = (1.0 - beta1) * update
    res = (update - exp_avg) ** 2 + eps2
    res_row = (1.0 - beta3) * res.mean(dim=-1)
    res_col = (1.0 - beta3) * res.mean(dim=-2)
    final = approx(res_row, res_col) * exp_avg
    return p - lr * final


def test_came_first_step_matches_reference_formula() -> None:
    """CAME's first step (factored 2D param) matches the result independently recomputed per the paper/official implementation."""
    torch.manual_seed(3)
    lr, betas, eps, clip = 1e-2, (0.9, 0.999, 0.9999), (1e-30, 1e-16), 1.0

    p = nn.Parameter(torch.randn(6, 4))
    grad = torch.randn(6, 4)
    expected = _came_reference_step(p.detach(), grad, lr, betas, eps, clip)

    optim = CAME([p], lr=lr, betas=betas, eps=eps, clip_threshold=clip)
    p.grad = grad.clone()
    optim.step()

    assert torch.allclose(p.detach(), expected, atol=1e-7), (
        f"CAME first step disagrees with the reference formula: max diff = "
        f"{(p.detach() - expected).abs().max().item():.3e}"
    )


def test_came_factored_state_keys() -> None:
    """A 2D param takes the factored path (row/col second moment + row/col instability), a 1D param uses full second moment."""
    mat = nn.Parameter(torch.randn(4, 3))
    vec = nn.Parameter(torch.randn(5))
    optim = CAME([mat, vec], lr=1e-3)
    mat.grad = torch.randn_like(mat)
    vec.grad = torch.randn_like(vec)
    optim.step()

    st_mat = optim.state[mat]
    assert {"exp_avg", "exp_avg_sq_row", "exp_avg_sq_col",
            "exp_avg_res_row", "exp_avg_res_col"} <= set(st_mat)
    assert "exp_avg_sq" not in st_mat
    assert st_mat["exp_avg_sq_row"].shape == (4,)
    assert st_mat["exp_avg_sq_col"].shape == (3,)

    st_vec = optim.state[vec]
    assert "exp_avg_sq" in st_vec
    assert "exp_avg_sq_row" not in st_vec


def test_came_bf16_param_state_fp32_and_finite() -> None:
    """Under a bf16 param, state is pinned to fp32 (so the EMA delta isn't swallowed by bf16 rounding), and the writeback stays finite."""
    p = nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    optim = CAME([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    optim.step()

    st = optim.state[p]
    for key in ("exp_avg", "exp_avg_sq_row", "exp_avg_sq_col",
                "exp_avg_res_row", "exp_avg_res_col"):
        assert st[key].dtype == torch.float32, f"{key} should be fp32"
    assert torch.isfinite(p).all()


def test_create_came_backfills_beta3_and_eps_pair() -> None:
    """When the upstream create_optimizer passes its Adam defaults (betas=(0.9,0.999) /
    eps=1e-8), the factory maps the default sentinel to the paper's defaults of
    beta3=0.9999 / eps=(1e-30, 1e-16)."""
    model = nn.Linear(4, 4)
    optim = create_optimizer("came", model.parameters(), learning_rate=1e-4)
    assert isinstance(optim, CAME)
    assert optim.param_groups[0]["betas"] == (0.9, 0.999, 0.9999)
    assert optim.param_groups[0]["eps"] == (1e-30, 1e-16)

    # an explicitly passed 3-tuple / eps pair is respected
    model2 = nn.Linear(4, 4)
    optim2 = create_came(
        model2.parameters(), lr=1e-4, betas=(0.8, 0.99, 0.999), eps=(1e-20, 1e-12),
    )
    assert optim2.param_groups[0]["betas"] == (0.8, 0.99, 0.999)
    assert optim2.param_groups[0]["eps"] == (1e-20, 1e-12)


def test_create_came_rejects_explicit_scalar_eps_and_short_betas() -> None:
    """A wrong-shaped config that isn't the default sentinel must fail loudly, not be silently
    substituted: an explicit scalar eps (!= Adam's default 1e-8) errors; an explicit 2-tuple
    betas (!= Adam's default) is rejected by CAME.__init__."""
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="eps1, eps2"):
        create_came(model.parameters(), lr=1e-4, eps=1e-6)
    with pytest.raises(ValueError, match="Invalid betas"):
        create_came(model.parameters(), lr=1e-4, betas=(0.8, 0.99))


def test_came_rejects_invalid_hyperparams() -> None:
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="Invalid betas"):
        CAME(model.parameters(), lr=1e-4, betas=(0.9, 1.5, 0.9999))
    with pytest.raises(ValueError, match="Invalid learning rate"):
        CAME(model.parameters(), lr=0.0)
    # beta=1.0 would freeze the EMA at zero -> _approx_sq_grad's first step is 0/0=NaN, must be rejected
    with pytest.raises(ValueError, match="Invalid betas"):
        CAME(model.parameters(), lr=1e-4, betas=(0.9, 1.0, 0.9999))
    with pytest.raises(ValueError, match="Invalid betas"):
        CAME(model.parameters(), lr=1e-4, betas=(0.9, 0.999, 1.0))
    # eps=0 gives 0/0=NaN when the whole gradient is zero (a LoRA zero-init first step), must be rejected
    with pytest.raises(ValueError, match="Invalid eps"):
        CAME(model.parameters(), lr=1e-4, eps=(0.0, 1e-16))
    with pytest.raises(ValueError, match="Invalid eps"):
        CAME(model.parameters(), lr=1e-4, eps=(1e-30, 0.0))


def test_came_weight_decay_shrinks_params() -> None:
    """weight_decay is decoupled L2: under a zero gradient, params shrink by (1 - wd*lr)."""
    p = nn.Parameter(torch.ones(4, 4))
    optim = CAME([p], lr=1e-2, weight_decay=0.1)
    p.grad = torch.zeros_like(p)
    optim.step()
    # grad=0 -> update is all zero, leaving only the weight decay term: p <- p - wd*lr*p
    assert torch.allclose(p.detach(), torch.full((4, 4), 1.0 - 0.1 * 1e-2))


def test_create_lion_optimizer_updates_parameters() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optim = create_optimizer(
        "lion",
        model.parameters(),
        learning_rate=0.1,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )

    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optim.step()

    assert isinstance(optim, Lion)
    assert torch.allclose(model.weight, torch.full_like(model.weight, 0.9))
    assert "exp_avg" in optim.state[model.weight]


def test_create_lion_rejects_invalid_betas() -> None:
    model = nn.Linear(2, 1)
    with pytest.raises(ValueError, match="Invalid beta1"):
        create_optimizer("lion", model.parameters(), learning_rate=1e-4, betas=(1.0, 0.99))


def test_monitor_metrics_uses_plain_lr_for_adamw() -> None:
    """AdamW-style optimizers keep the historical monitor lr unchanged."""
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    assert get_optimizer_monitor_metrics(optim) == {"lr": 1e-4}


def test_monitor_metrics_reports_prodigy_effective_lr_from_d() -> None:
    """Prodigy/PPSF expose base lr=1; monitor should show d-adjusted LR."""
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=1.0)
    optim.param_groups[0]["d"] = 2e-4

    metrics = get_optimizer_monitor_metrics(optim)

    assert metrics["lr"] == 2e-4
    assert metrics["actual_lr"] == 2e-4
    assert metrics["base_lr"] == 1.0
    assert metrics["d"] == 2e-4


def test_monitor_metrics_uses_ppsf_effective_lr_multiplier() -> None:
    """PPSF v2 recommends logging d * effective_lr."""
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=1.0)
    optim.param_groups[0]["d"] = 2e-4
    optim.param_groups[0]["effective_lr"] = 0.25

    metrics = get_optimizer_monitor_metrics(optim)

    assert metrics["lr"] == 5e-5
    assert metrics["actual_lr"] == 5e-5
    assert metrics["base_lr"] == 1.0
    assert metrics["effective_lr"] == 0.25


def test_monitor_metrics_uses_ppsf_shared_d_when_split_groups_mean() -> None:
    """PPSF split_groups_mean uses shared_d for the dynamic learning rate."""
    model = nn.Linear(4, 4)
    optim = torch.optim.AdamW(model.parameters(), lr=1.0)
    optim.param_groups[0]["d"] = 2e-4
    optim.param_groups[0]["shared_d"] = 5e-5
    optim.param_groups[0]["split_groups"] = True
    optim.param_groups[0]["split_groups_mean"] = True

    metrics = get_optimizer_monitor_metrics(optim)

    assert metrics["lr"] == 5e-5
    assert metrics["actual_lr"] == 5e-5
    assert metrics["d"] == 5e-5


# ---------------------------------------------------------------------------
# create_prodigy_plus_schedulefree
# ---------------------------------------------------------------------------


def _has_ppsf() -> bool:
    try:
        importlib.import_module("prodigyplus")
        return True
    except ImportError:
        return False


def test_create_ppsf_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """When PPSF isn't installed, the error message must include an install hint, not a bare
    ImportError.

    Uses builtins.__import__ to force `from prodigyplus import ...` to raise ImportError,
    independent of whether PPSF is actually installed in the running environment (works in
    CI / dev / a local venv alike).
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "prodigyplus" or name.startswith("prodigyplus."):
            raise ImportError("simulated: no prodigyplus")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "prodigyplus", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    model = nn.Linear(4, 4)
    with pytest.raises(ImportError, match="prodigy-plus-schedule-free"):
        create_prodigy_plus_schedulefree(model.parameters(), lr=1.0)


@pytest.mark.skipif(not _has_ppsf(), reason="PPSF not installed")
def test_create_ppsf_forces_lr_to_one(caplog: pytest.LogCaptureFixture) -> None:
    """A non-1.0 lr is forcibly overridden with a WARN (PPSF requires lr=1.0)."""
    import logging
    model = nn.Linear(4, 4)
    with caplog.at_level(logging.WARNING, logger="utils.optimizer_utils"):
        optim = create_prodigy_plus_schedulefree(model.parameters(), lr=1e-4)
    assert any(
        "lr=1.0" in r.getMessage() and r.levelno >= logging.WARNING
        for r in caplog.records
    ), f"expected WARNING with 'lr=1.0', got: {[r.getMessage() for r in caplog.records]}"
    assert optim.param_groups[0]["lr"] == 1.0


@pytest.mark.skipif(not _has_ppsf(), reason="PPSF not installed")
def test_create_ppsf_overrides_pytorch_default_betas() -> None:
    """When the upstream create_optimizer's default betas=(0.9, 0.999) is passed, the factory
    overrides it internally with PPSF's recommended (0.9, 0.99).
    An explicitly-passed other value is respected."""
    model = nn.Linear(4, 4)
    # betas not explicitly passed -- should be overridden by the factory
    optim = create_prodigy_plus_schedulefree(model.parameters(), lr=1.0, betas=(0.9, 0.999))
    assert optim.param_groups[0]["betas"] == (0.9, 0.99)

    # explicitly passed is respected
    model2 = nn.Linear(4, 4)
    optim2 = create_prodigy_plus_schedulefree(model2.parameters(), lr=1.0, betas=(0.95, 0.98))
    assert optim2.param_groups[0]["betas"] == (0.95, 0.98)


@pytest.mark.skipif(not _has_ppsf(), reason="PPSF not installed")
def test_create_ppsf_exposes_train_eval_methods() -> None:
    """After instantiation, .train / .eval methods must exist -- otherwise optimizer_eval_mode is forever a no-op."""
    model = nn.Linear(4, 4)
    optim = create_prodigy_plus_schedulefree(model.parameters(), lr=1.0)
    assert hasattr(optim, "train") and callable(optim.train)
    assert hasattr(optim, "eval") and callable(optim.eval)


# ---------------------------------------------------------------------------
# Automagic v1
# ---------------------------------------------------------------------------


def test_automagic_bf16_shift_dtype_stable_across_resume() -> None:
    """A bf16 param's Kahan shift buffer aligns with upstream diffusion-pipe: shares the
    param dtype (bf16). The key property is resume stability -- PyTorch load_state_dict casts
    float state to the param dtype, so a bf16 shift keeps its dtype after the roundtrip (an
    fp32 shift would instead be silently downcast to bf16, causing behavior drift)."""
    p = nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    optim = Automagic([p], lr=1e-4)

    p.grad = torch.randn_like(p)
    optim.step()

    state = optim.state[p]
    assert "shift" in state, "a bf16 param should create a shift buffer"
    assert state["shift"].dtype == p.dtype

    # dtype must not drift after the state_dict roundtrip
    sd = optim.state_dict()
    p2 = nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    optim2 = Automagic([p2], lr=1e-4)
    optim2.load_state_dict(sd)
    assert optim2.state[p2]["shift"].dtype == p2.dtype


# ---------------------------------------------------------------------------
# Automagic v2
# ---------------------------------------------------------------------------


def test_automagic_v2_scalar_lr_updates() -> None:
    """Automagic v2 uses a fused backward hook; verifies params actually get updated."""
    torch.manual_seed(7)
    model = nn.Linear(8, 4, bias=False)
    p_before = model.weight.data.clone()

    optim = create_automagic_v2(model.parameters(), lr=1e-3)

    # v2's hook automatically updates params during backward
    for _ in range(5):
        out = model(torch.randn(2, 8))
        loss = out.sum()
        loss.backward()
        # v2 doesn't need a manual step (done inside the hook), but calling it is harmless
        optim.zero_grad()

    assert not torch.allclose(model.weight.data, p_before), "the v2 backward hook should change the params"


def test_automagic_v2_get_avg_learning_rate() -> None:
    """A v2 instance should expose the get_avg_learning_rate method."""
    model = nn.Linear(4, 4)
    optim = create_automagic_v2(model.parameters(), lr=1e-3)
    assert hasattr(optim, "get_avg_learning_rate")
    avg_lr = optim.get_avg_learning_rate()
    assert isinstance(avg_lr, float) or isinstance(avg_lr, torch.Tensor)


# ---------------------------------------------------------------------------
# get_optimizer_monitor_metrics — Automagic duck typing
# ---------------------------------------------------------------------------


def test_automagic_monitor_metrics_uses_get_avg_learning_rate() -> None:
    """get_optimizer_monitor_metrics prefers the get_avg_learning_rate duck-typed path."""
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optim = create_automagic(model.parameters(), lr=1e-4)

    # run a few steps so lr has a value
    for _ in range(3):
        out = model(torch.randn(2, 4))
        out.sum().backward()
        optim.step()
        optim.zero_grad()

    metrics = get_optimizer_monitor_metrics(optim)
    assert "lr" in metrics
    assert "actual_lr" in metrics
    assert metrics["lr"] > 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_create_optimizer_dispatches_automagic() -> None:
    """create_optimizer(optimizer_type='automagic') returns an Automagic instance."""
    model = nn.Linear(4, 4)
    optim = create_optimizer("automagic", model.parameters(), learning_rate=1e-4)
    assert isinstance(optim, Automagic)


def test_create_optimizer_dispatches_automagic_v2() -> None:
    """create_optimizer(optimizer_type='automagic_v2') returns an Automagic2 instance."""
    model = nn.Linear(4, 4)
    optim = create_optimizer("automagic_v2", model.parameters(), learning_rate=1e-3)
    assert isinstance(optim, Automagic2)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_automagic_v2_skips_nonfinite_grad() -> None:
    """The fused path bypasses the training loop's step-boundary NaN check, so it must defend itself inside the hook."""
    p = nn.Parameter(torch.ones(4, 4))
    optim = Automagic2([p], lr=1e-4)
    before = p.detach().clone()
    p.grad = torch.full_like(p, float("nan"))
    optim._update_param(p, optim.param_groups[0])
    assert torch.equal(p.detach(), before), "a NaN gradient should not touch the params"
    assert p.grad is None, "a bad gradient should be discarded"


def test_automagic_v2_second_moment_fp32_under_bf16() -> None:
    """Second moment is pinned to fp32: bf16 storage would let a (1-beta2)=1e-3 EMA delta be swallowed by rounding."""
    p = nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
    optim = Automagic2([p], lr=1e-6)
    p.grad = torch.randn_like(p)
    optim._update_param(p, optim.param_groups[0])
    st = optim.state[p]
    assert st["exp_avg_sq_row"].dtype == torch.float32
    assert st["exp_avg_sq_col"].dtype == torch.float32
    assert st["lr"].dtype == torch.float32


def test_automagic_v2_validate_rejects_grad_accum() -> None:
    """v2's fused backward conflicts with gradient accumulation semantics, must be intercepted at startup."""
    from types import SimpleNamespace
    from training.optimizers import automagic as automagic_builder

    args = SimpleNamespace(
        lr_scheduler="none", automagic_variant="v2",
        mixed_precision="bf16", grad_clip_max_norm=0, grad_accum=4,
    )
    with pytest.raises(ValueError, match="grad_accum"):
        automagic_builder.validate(args)

    args.grad_accum = 1
    automagic_builder.validate(args)  # should not raise


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _fake_bnb(captured: dict):
    """Minimal bitsandbytes stand-in: records the params it receives, returns a dummy optimizer."""
    from types import SimpleNamespace

    def _adamw8bit(params, **kwargs):
        captured["params"] = params
        captured["kwargs"] = kwargs
        return SimpleNamespace(param_groups=[])

    return SimpleNamespace(optim=SimpleNamespace(AdamW8bit=_adamw8bit))


def test_create_8bit_adamw_accepts_param_groups(monkeypatch) -> None:
    """Regression: the trainer goes through injector.get_param_groups() -> [{"params": [...]}, ...].

    Directly calling p.numel() when tallying param count would hit 'dict' object has no
    attribute 'numel'; on a real machine this manifested as training crashing during the
    optimizer phase (nobody exercised this path before adamw8bit was wired into the registry,
    so it went unnoticed).
    """
    from utils import optimizer_utils as ou

    captured: dict = {}
    monkeypatch.setattr(ou, "BITSANDBYTES_AVAILABLE", True)
    monkeypatch.setattr(ou, "bnb", _fake_bnb(captured), raising=False)

    a = nn.Parameter(torch.zeros(4, 8))
    b = nn.Parameter(torch.zeros(2, 2))
    groups = [
        {"params": [a], "weight_decay": 0.01},
        {"params": [b], "weight_decay": 0.0},
    ]

    ou.create_8bit_adamw(params=groups, lr=1e-4)

    # the group shape must be passed to bnb as-is -- flattening would lose the per-group weight_decay
    assert captured["params"] == groups


def test_create_8bit_adamw_still_accepts_flat_params(monkeypatch) -> None:
    from utils import optimizer_utils as ou

    captured: dict = {}
    monkeypatch.setattr(ou, "BITSANDBYTES_AVAILABLE", True)
    monkeypatch.setattr(ou, "bnb", _fake_bnb(captured), raising=False)

    params = [nn.Parameter(torch.zeros(4, 8))]
    ou.create_8bit_adamw(params=params, lr=1e-4)

    assert captured["params"] == params


def test_iter_optimizer_params_flattens_both_shapes() -> None:
    from utils.optimizer_utils import iter_optimizer_params

    a = nn.Parameter(torch.zeros(4, 8))
    b = nn.Parameter(torch.zeros(2, 2))

    assert list(iter_optimizer_params([a, b])) == [a, b]
    assert list(iter_optimizer_params([{"params": [a]}, {"params": [b]}])) == [a, b]
    assert list(iter_optimizer_params([{"weight_decay": 0.0}])) == []
