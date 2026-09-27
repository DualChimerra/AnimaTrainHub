from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("lycoris")

from utils.lycoris_adapter import AnimaLycorisAdapter  # noqa: E402
from training.families.anima.preset import ANIMA_PRESET


def _bare_tlora_adapter(rank: int, min_rank: int, alpha: float) -> AnimaLycorisAdapter:
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, 
        algo="tlora",
        rank=rank,
        alpha=float(rank),
        tlora_min_rank=min_rank,
        tlora_alpha_rank_scale=alpha,
    )
    placeholder = nn.Module()
    adapter._tlora_modules = [placeholder]
    return adapter




def test_tlora_mask_clean_timestep_full_rank() -> None:
    rank, min_rank = 32, 4
    adapter = _bare_tlora_adapter(rank=rank, min_rank=min_rank, alpha=1.0)
    adapter._set_tlora_mask(torch.tensor([0.0]))
    mask = adapter._tlora_mask
    assert mask is not None
    assert int(mask.sum().item()) == rank, (
        f"t=0 should be full rank ({rank}), actual active={int(mask.sum().item())}"
    )


def test_tlora_mask_max_noise_timestep_min_rank() -> None:
    rank, min_rank = 32, 4
    adapter = _bare_tlora_adapter(rank=rank, min_rank=min_rank, alpha=1.0)
    adapter._set_tlora_mask(torch.tensor([1.0]))
    mask = adapter._tlora_mask
    assert mask is not None
    assert int(mask.sum().item()) == min_rank, (
        f"t=1 should be min_rank ({min_rank}), actual active={int(mask.sum().item())}"
    )
    expected = torch.cat(
        [torch.ones(min_rank), torch.zeros(rank - min_rank)]
    ).to(mask.device)
    assert torch.equal(mask, expected)


def test_tlora_mask_midpoint_linear() -> None:
    rank, min_rank = 32, 4
    adapter = _bare_tlora_adapter(rank=rank, min_rank=min_rank, alpha=1.0)
    adapter._set_tlora_mask(torch.tensor([0.5]))
    # frac = (1 - 0.5)^1 = 0.5; active = 4 + 0.5*28 = 18.0; floor(18) = 18
    expected = min_rank + 0.5 * (rank - min_rank)
    mask = adapter._tlora_mask
    assert int(mask.sum().item()) == int(expected), (
        f"t=0.5, alpha=1 should be active={int(expected)}, actual={int(mask.sum().item())}"
    )


def test_tlora_mask_alpha_power_steepens_schedule() -> None:
    rank, min_rank = 32, 4
    adapter = _bare_tlora_adapter(rank=rank, min_rank=min_rank, alpha=2.0)
    adapter._set_tlora_mask(torch.tensor([0.5]))
    expected = min_rank + (1 - 0.5) ** 2 * (rank - min_rank)
    mask = adapter._tlora_mask
    assert int(mask.sum().item()) == int(expected), (
        f"t=0.5, alpha=2 should be active={int(expected)}, actual={int(mask.sum().item())}"
    )


def test_tlora_mask_monotone_t_to_active_rank() -> None:
    adapter = _bare_tlora_adapter(rank=32, min_rank=4, alpha=1.0)
    prev = 33  # > rank
    for t in torch.linspace(0.0, 1.0, 11):
        adapter._set_tlora_mask(t.reshape(1))
        cur = int(adapter._tlora_mask.sum().item())
        assert cur <= prev, f"not monotonic: t={float(t):.2f} active={cur} > prev={prev}"
        prev = cur




def test_tlora_clear_mask_restores_full_rank() -> None:
    rank, min_rank = 32, 4
    adapter = _bare_tlora_adapter(rank=rank, min_rank=min_rank, alpha=1.0)
    adapter._set_tlora_mask(torch.tensor([0.7]))
    assert int(adapter._tlora_mask.sum().item()) < rank, "precondition: mask should be partial"
    adapter.clear_timestep_mask()
    assert int(adapter._tlora_mask.sum().item()) == rank, (
        f"should be full rank ({rank}) after clear, "
        f"actual active={int(adapter._tlora_mask.sum().item())}"
    )
    adapter._set_tlora_mask(torch.tensor([1.0]))
    assert int(adapter._tlora_mask.sum().item()) == min_rank


def test_tlora_weight_rebuild_uses_lycoris4_functional_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain T-LoRA keeps its mask but delegates ΔW to LyCORIS 4 kernels."""
    from lycoris.functional import locon as functional_locon

    class _TinyDiT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(16, 16, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.q_proj(x)

    model = _TinyDiT()
    adapter = AnimaLycorisAdapter(
        preset=ANIMA_PRESET,
        algo="tlora",
        rank=4,
        alpha=4.0,
        tlora_min_rank=1,
    )
    adapter.inject(model)
    adapter._set_tlora_mask(torch.tensor([0.5]))

    calls: list[tuple[torch.Tensor, ...]] = []
    real_diff_weight = functional_locon.diff_weight

    def _spy_diff_weight(*weights, **kwargs):
        calls.append(weights)
        return real_diff_weight(*weights, **kwargs)

    monkeypatch.setattr(functional_locon, "diff_weight", _spy_diff_weight)
    layer = adapter._tlora_modules[0]
    weight = layer.make_weight(device=layer.lora_up.weight.device)

    assert calls, "T-LoRA make_weight должен использовать functional dispatch LyCORIS 4"
    assert weight.shape == layer.shape

    bypass_calls: list[tuple[torch.Tensor, ...]] = []
    real_bypass = functional_locon.bypass_forward_diff

    def _spy_bypass(*weights, **kwargs):
        bypass_calls.append(weights)
        return real_bypass(*weights, **kwargs)

    monkeypatch.setattr(functional_locon, "bypass_forward_diff", _spy_bypass)
    model(torch.randn(2, 16))

    assert layer.bypass_mode is True
    assert bypass_calls, "T-LoRA forward должен использовать fused bypass API LyCORIS 4"




def test_run_sample_clears_tlora_mask() -> None:
    from unittest.mock import MagicMock
    from runtime.training import sample_runner as sr
    from runtime.training.families.anima import ANIMA_SPEC
    from runtime.training.sample_runner import run_sample

    clear_called = {"n": 0}

    class _StubInjector:
        def clear_timestep_mask(self) -> None:
            clear_called["n"] += 1

    ctx = MagicMock()
    ctx.injector = _StubInjector()
    ctx.args = MagicMock(
        resolution=64,
        sample_width=0,
        sample_height=0,
        sample_cfg_scale=1.0,
        sample_negative_prompt="",
        sample_seed=0,
        sample_infer_steps=1,
        sample_sampler_name="er_sde",
        sample_scheduler="simple",
    )
    ctx.optimizer = MagicMock()
    ctx.model = MagicMock()
    ctx.wandb_monitor = MagicMock(log_samples=False)
    ctx.monitor_server = None
    ctx.dtype = torch.float32
    ctx.device = torch.device("cpu")
    ctx.family.spec = ANIMA_SPEC

    from contextlib import contextmanager

    @contextmanager
    def _noop_ctx(*a, **kw):
        yield

    orig_eval = sr.optimizer_eval_mode
    sr.optimizer_eval_mode = _noop_ctx  # type: ignore[assignment]
    ctx.emit = MagicMock()

    try:
        run_sample(ctx, prompt="x", sample_path=MagicMock())
    finally:
        sr.optimizer_eval_mode = orig_eval  # type: ignore[assignment]

    assert clear_called["n"] == 1, (
        "run_sample must call injector.clear_timestep_mask() so sample runs at full rank"
    )


def test_run_sample_no_clear_method_does_not_crash() -> None:
    from unittest.mock import MagicMock
    from runtime.training import sample_runner as sr
    from runtime.training.families.anima import ANIMA_SPEC
    from runtime.training.sample_runner import run_sample

    class _NonTloraInjector:
        pass

    ctx = MagicMock()
    ctx.injector = _NonTloraInjector()
    ctx.args = MagicMock(
        resolution=64,
        sample_width=0,
        sample_height=0,
        sample_cfg_scale=1.0,
        sample_negative_prompt="",
        sample_seed=0,
        sample_infer_steps=1,
        sample_sampler_name="er_sde",
        sample_scheduler="simple",
    )
    ctx.optimizer = MagicMock()
    ctx.model = MagicMock()
    ctx.wandb_monitor = MagicMock(log_samples=False)
    ctx.monitor_server = None
    ctx.dtype = torch.float32
    ctx.device = torch.device("cpu")
    ctx.family.spec = ANIMA_SPEC

    from contextlib import contextmanager

    @contextmanager
    def _noop_ctx(*a, **kw):
        yield

    orig_eval = sr.optimizer_eval_mode
    sr.optimizer_eval_mode = _noop_ctx  # type: ignore[assignment]
    ctx.emit = MagicMock()

    try:
        run_sample(ctx, prompt="x", sample_path=MagicMock())
    finally:
        sr.optimizer_eval_mode = orig_eval  # type: ignore[assignment]


# ── test 4: DoRA bypass invariant ──────────────────────────────────────────


def test_tlora_dora_keeps_rebuild_path() -> None:

    class _TinyDiT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(16, 16, bias=False)

    adapter = AnimaLycorisAdapter(
        preset=ANIMA_PRESET,
        algo="tlora",
        rank=4,
        alpha=4.0,
        weight_decompose=True,
    )
    adapter.inject(_TinyDiT())

    assert adapter._tlora_modules
    assert all(not layer.bypass_mode for layer in adapter._tlora_modules)
