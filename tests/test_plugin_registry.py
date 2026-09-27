
from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path

import pytest
from training.families.anima.preset import ANIMA_PRESET


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = REPO_ROOT / "runtime"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_DIR))


@pytest.fixture(scope="module")
def AnimaLycorisAdapter(preset=ANIMA_PRESET, ):
    pytest.importorskip("lycoris")
    from utils.lycoris_adapter import AnimaLycorisAdapter as cls
    return cls


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_adapter_builders_dict_has_lokr_loha_lora() -> None:
    from training.adapters import BUILDERS
    assert set(BUILDERS) == {"lokr", "loha", "lora", "ortho", "tlora"}


def test_optimizer_builders_dict_has_9_variants() -> None:
    from training.optimizers import BUILDERS, VALIDATORS
    assert set(BUILDERS) == {
        "adamw", "adamw8bit", "automagic", "came", "lion", "prodigy",
        "prodigy_plus_schedulefree", "soap", "soap_sf",
    }
    assert set(VALIDATORS) == {
        "adamw8bit", "automagic", "prodigy_plus_schedulefree", "soap_sf",
    }


def test_adamw8bit_build_dispatches_without_rewriting_hyperparams(monkeypatch) -> None:
    from training.optimizers import adamw8bit

    captured = {}

    def fake_create_optimizer(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "utils.optimizer_utils",
        types.SimpleNamespace(create_optimizer=fake_create_optimizer),
    )

    adamw8bit.build(argparse.Namespace(), params=[], lr=1e-4, weight_decay=0.01)

    assert captured["optimizer_type"] == "adamw8bit"
    assert captured["learning_rate"] == 1e-4
    assert captured["weight_decay"] == 0.01


def test_adamw8bit_validate_rejects_missing_bitsandbytes(monkeypatch) -> None:
    from training.optimizers import adamw8bit

    monkeypatch.setitem(
        sys.modules,
        "utils.optimizer_utils",
        types.SimpleNamespace(BITSANDBYTES_AVAILABLE=False),
    )
    with pytest.raises(SystemExit, match="bitsandbytes"):
        adamw8bit.validate(argparse.Namespace())

    monkeypatch.setitem(
        sys.modules,
        "utils.optimizer_utils",
        types.SimpleNamespace(BITSANDBYTES_AVAILABLE=True),
    )
    adamw8bit.validate(argparse.Namespace())


def test_scheduler_builders_dict_excludes_none() -> None:
    from training.schedulers import BUILDERS, SCHEMA_ONLY_OPTIONS
    assert set(BUILDERS) == {
        "constant_then_cosine",
        "cosine",
        "cosine_cycles",
        "cosine_with_restart",
        "cosine_with_warmup",
    }
    assert SCHEMA_ONLY_OPTIONS == {"none"}


def test_inference_sampler_builders_has_er_sde() -> None:
    from training.inference_samplers import BUILDERS
    assert "er_sde" in BUILDERS


def test_loss_builders_dict_has_mse_huber() -> None:
    from training.losses import BUILDERS
    assert set(BUILDERS) == {"mse", "huber"}


def test_build_adapter_raises_on_unknown_lora_type() -> None:
    from training.adapters import build_adapter
    args = argparse.Namespace(lora_type="bogus_xyz")
    with pytest.raises(ValueError, match="Unknown lora_type"):
        build_adapter(args, preset=ANIMA_PRESET)


def test_build_adapter_forwards_explicit_family_preset(monkeypatch) -> None:
    from training.adapters import BUILDERS, build_adapter

    family_preset = {"target_name": ["family-only"]}
    sentinel = object()

    def _build(args, *, preset):
        assert args.lora_type == "lora"
        assert preset is family_preset
        return sentinel

    monkeypatch.setitem(BUILDERS, "lora", _build)
    assert build_adapter(
        argparse.Namespace(lora_type="lora"),
        preset=family_preset,
    ) is sentinel


def test_build_scheduler_returns_none_when_lr_scheduler_is_none() -> None:
    from training.schedulers import build_scheduler
    args = argparse.Namespace(lr_scheduler="none")
    assert build_scheduler(args, optimizer=None, total_steps=None) is None


def test_cosine_with_warmup_scheduler_warms_then_decays() -> None:
    torch = pytest.importorskip("torch")
    from training.schedulers import build_scheduler

    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0)
    args = argparse.Namespace(
        lr_scheduler="cosine_with_warmup",
        lr_scheduler_warmup_steps=2,
        lr_scheduler_eta_min=0.1,
    )
    scheduler = build_scheduler(args, optimizer, total_steps=6)

    lrs = []
    for _ in range(6):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[0] == pytest.approx(0.0)
    assert lrs[1] == pytest.approx(0.5)
    assert lrs[2] == pytest.approx(1.0)
    assert lrs[-1] < lrs[2]
    assert lrs[-1] >= 0.1


def test_cosine_with_warmup_zero_warmup_starts_at_base_lr() -> None:
    torch = pytest.importorskip("torch")
    from training.schedulers import build_scheduler

    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0)
    args = argparse.Namespace(
        lr_scheduler="cosine_with_warmup",
        lr_scheduler_warmup_steps=0,
        lr_scheduler_eta_min=0.0,
    )
    scheduler = build_scheduler(args, optimizer, total_steps=4)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5 * (1.0 + math.cos(math.pi * 0.25)))


def test_cosine_with_warmup_multi_param_group_respects_eta_min() -> None:
    torch = pytest.importorskip("torch")
    from training.schedulers import build_scheduler

    p1 = torch.nn.Parameter(torch.tensor([1.0]))
    p2 = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD(
        [{"params": [p1], "lr": 1e-3}, {"params": [p2], "lr": 1e-4}],
    )
    args = argparse.Namespace(
        lr_scheduler="cosine_with_warmup",
        lr_scheduler_warmup_steps=0,
        lr_scheduler_eta_min=1e-6,
    )
    scheduler = build_scheduler(args, optimizer, total_steps=10)

    for _ in range(10):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6, rel=1e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-6, rel=1e-5)


def test_cosine_with_warmup_no_total_steps_returns_none() -> None:
    from training.schedulers import build_scheduler

    args = argparse.Namespace(
        lr_scheduler="cosine_with_warmup",
        lr_scheduler_warmup_steps=10,
        lr_scheduler_eta_min=0.0,
    )
    assert build_scheduler(args, optimizer=None, total_steps=None) is None
    assert build_scheduler(args, optimizer=None, total_steps=0) is None
    assert build_scheduler(args, optimizer=None, total_steps=-5) is None


def test_cosine_with_warmup_clamps_negative_eta_min() -> None:
    torch = pytest.importorskip("torch")
    from training.schedulers import build_scheduler

    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0)
    args = argparse.Namespace(
        lr_scheduler="cosine_with_warmup",
        lr_scheduler_warmup_steps=0,
        lr_scheduler_eta_min=-0.5,
    )
    scheduler = build_scheduler(args, optimizer, total_steps=4)

    for _ in range(4):
        assert optimizer.param_groups[0]["lr"] >= 0.0
        optimizer.step()
        scheduler.step()


def test_cosine_cycles_supports_per_peak_min_and_max_lr() -> None:
    torch = pytest.importorskip("torch")
    from training.schedulers import build_scheduler

    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    args = argparse.Namespace(
        lr_scheduler="cosine_cycles",
        lr_scheduler_cycle_count=3,
        lr_scheduler_cycle_max_lr=None,
        lr_scheduler_cycle_min_lr=0.0,
        lr_scheduler_cycle_max_lrs=[1e-3, 8e-4, 6e-4],
        lr_scheduler_cycle_min_lrs=[5e-4, 4e-4, 1e-5],
    )
    scheduler = build_scheduler(args, optimizer, total_steps=12)

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(12):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] == pytest.approx(1e-3)
    assert lrs[4] == pytest.approx(8e-4)
    assert lrs[8] == pytest.approx(6e-4)
    assert lrs[12] == pytest.approx(1e-5)
    assert min(lrs[0:4]) >= 5e-4
    assert min(lrs[4:8]) >= 4e-4


def test_constant_then_cosine_stays_flat_then_decays() -> None:
    torch = pytest.importorskip("torch")
    from training.schedulers import build_scheduler

    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    args = argparse.Namespace(
        lr_scheduler="constant_then_cosine",
        lr_scheduler_decay_start_ratio=2.0 / 3.0,
        lr_scheduler_decay_min_lr=1e-5,
    )
    scheduler = build_scheduler(args, optimizer, total_steps=12)

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(12):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[:9] == pytest.approx([1e-3] * 9)
    assert lrs[9] < 1e-3
    assert lrs[9] > lrs[10] > lrs[11] > lrs[12]
    assert lrs[12] == pytest.approx(1e-5)


def test_ppsf_zero_prodigy_steps_disables_freeze(monkeypatch) -> None:
    """ppsf_prodigy_steps=0 keeps the upstream PPSF meaning: never freeze d."""
    from training.optimizers import prodigy_plus_schedulefree as ppsf

    captured = {}

    def fake_create_optimizer(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "utils.optimizer_utils",
        types.SimpleNamespace(create_optimizer=fake_create_optimizer),
    )
    args = argparse.Namespace(
        ppsf_beta1=0.9,
        ppsf_beta2=0.99,
        ppsf_d_coef=1.0,
        ppsf_prodigy_steps=0,
        ppsf_split_groups=True,
        ppsf_split_groups_mean=False,
        ppsf_use_speed=False,
        ppsf_fused_back_pass=False,
        ppsf_use_stableadamw=True,
    )

    ppsf.build(args, params=[], lr=1.0, weight_decay=0.0)

    assert captured["prodigy_steps"] == 0


def test_ppsf_explicit_prodigy_steps_is_preserved(monkeypatch) -> None:
    from training.optimizers import prodigy_plus_schedulefree as ppsf

    captured = {}

    def fake_create_optimizer(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "utils.optimizer_utils",
        types.SimpleNamespace(create_optimizer=fake_create_optimizer),
    )
    args = argparse.Namespace(
        ppsf_beta1=0.9,
        ppsf_beta2=0.99,
        ppsf_d_coef=1.0,
        ppsf_prodigy_steps=750,
        ppsf_split_groups=True,
        ppsf_split_groups_mean=False,
        ppsf_use_speed=False,
        ppsf_fused_back_pass=False,
        ppsf_use_stableadamw=True,
    )

    ppsf.build(args, params=[], lr=1.0, weight_decay=0.0)

    assert captured["prodigy_steps"] == 750


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_adapter_schema_consistency_passes_on_clean_dev() -> None:
    from training.adapters import validate_schema_consistency
    validate_schema_consistency()


def test_optimizer_schema_consistency_passes_on_clean_dev() -> None:
    from training.optimizers import validate_schema_consistency
    validate_schema_consistency()


def test_scheduler_schema_consistency_passes_on_clean_dev() -> None:
    from training.schedulers import validate_schema_consistency
    validate_schema_consistency()


def test_loss_schema_consistency_passes_on_clean_dev() -> None:
    from training.losses import validate_schema_consistency
    validate_schema_consistency()


def test_schema_consistency_raises_when_builder_missing(monkeypatch) -> None:
    from training import adapters
    monkeypatch.delitem(adapters.BUILDERS, "tlora")
    from studio.schema import TrainingConfig
    field = TrainingConfig.model_fields["lora_type"]
    original = field.annotation
    try:
        from typing import Literal
        field.annotation = Literal["lora", "lokr", "loha", "ortho", "tlora"]  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="out of sync"):
            adapters.validate_schema_consistency()
    finally:
        field.annotation = original


# ---------------------------------------------------------------------------
# AdapterProtocol runtime_checkable
# ---------------------------------------------------------------------------


def test_animalycoris_satisfies_adapter_protocol(AnimaLycorisAdapter) -> None:
    from training.adapters.protocol import AdapterProtocol
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lokr")
    assert isinstance(adapter, AdapterProtocol)


def test_animalycoris_hooks_are_noop(AnimaLycorisAdapter) -> None:
    from training.adapters.protocol import StepContext
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lokr")

    import torch
    step_ctx = StepContext(
        global_step=0,
        total_steps=100,
        epoch=0,
        sigma_t=torch.zeros(1),
        args=argparse.Namespace(),
    )

    assert adapter.on_step_begin(step_ctx) is None
    assert adapter.regularization_loss(step_ctx) is None


def test_animalycoris_lokr_excludes_weight_decay_for_w1(AnimaLycorisAdapter) -> None:
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lokr")
    assert adapter.excludes_weight_decay("lora_unet_xxx.lokr_w1") is True
    assert adapter.excludes_weight_decay("lora_unet_xxx.lokr_w2_a") is False


def test_animalycoris_non_lokr_does_not_exclude_weight_decay(AnimaLycorisAdapter) -> None:
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="lora")
    assert adapter.excludes_weight_decay("lora_unet_xxx.lokr_w1") is False
    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="loha")
    assert adapter.excludes_weight_decay("lora_unet_xxx.lokr_w1") is False


def test_tlora_mask_changes_with_sigma_and_is_not_saved(AnimaLycorisAdapter) -> None:
    import torch
    from training.adapters.protocol import StepContext

    adapter = AnimaLycorisAdapter(preset=ANIMA_PRESET, algo="tlora", rank=8, tlora_min_rank=2, tlora_alpha_rank_scale=1.0)
    adapter._tlora_modules = [types.SimpleNamespace()]
    adapter.on_step_begin(StepContext(0, 10, 0, torch.tensor([0.0]), argparse.Namespace()))
    assert adapter._tlora_mask.tolist() == [1.0] * 8
    adapter.on_step_begin(StepContext(1, 10, 0, torch.tensor([1.0]), argparse.Namespace()))
    assert adapter._tlora_mask.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert adapter.state_dict() == {}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class _MockTLoRAAdapter:

    def __init__(self) -> None:
        self.step_begin_calls = 0
        self.last_sigma_t = None

    def inject(self, model) -> None: pass
    def get_param_groups(self, weight_decay): return []
    def save(self, path) -> None: pass
    def load(self, path) -> None: pass

    def on_step_begin(self, ctx) -> None:
        self.step_begin_calls += 1
        self.last_sigma_t = ctx.sigma_t

    def regularization_loss(self, ctx): return None
    def excludes_weight_decay(self, name): return False


class _MockOFTAdapter:

    def __init__(self) -> None:
        self.reg_calls = 0

    def inject(self, model) -> None: pass
    def get_param_groups(self, weight_decay): return []
    def save(self, path) -> None: pass
    def load(self, path) -> None: pass
    def on_step_begin(self, ctx) -> None: pass

    def regularization_loss(self, ctx):
        import torch
        self.reg_calls += 1
        return torch.tensor(0.42)

    def excludes_weight_decay(self, name): return False


def test_mock_tlora_implements_protocol() -> None:
    from training.adapters.protocol import AdapterProtocol
    assert isinstance(_MockTLoRAAdapter(), AdapterProtocol)


def test_mock_oft_regularization_returns_tensor() -> None:
    import torch
    from training.adapters.protocol import StepContext
    adapter = _MockOFTAdapter()
    ctx = StepContext(global_step=5, total_steps=100, epoch=0,
                      sigma_t=torch.zeros(1), args=argparse.Namespace())
    loss = adapter.regularization_loss(ctx)
    assert isinstance(loss, torch.Tensor)
    assert float(loss) == pytest.approx(0.42)
    assert adapter.reg_calls == 1


def test_mock_tlora_on_step_begin_receives_sigma() -> None:
    import torch
    from training.adapters.protocol import StepContext
    adapter = _MockTLoRAAdapter()
    sigma = torch.tensor([0.3, 0.7])
    ctx = StepContext(global_step=10, total_steps=100, epoch=0,
                      sigma_t=sigma, args=argparse.Namespace())
    adapter.on_step_begin(ctx)
    assert adapter.step_begin_calls == 1
    assert torch.equal(adapter.last_sigma_t, sigma)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_no_optimizer_type_dispatch_in_phases_optimizer() -> None:
    text = (RUNTIME_DIR / "training" / "phases" / "optimizer.py").read_text(encoding="utf-8")
    assert 'if ctx.optimizer_type == "prodigy"' not in text
    assert 'if optimizer_type ==' not in text
    assert 'optimizer_type == "prodigy_plus_schedulefree"' not in text


def test_no_lora_type_dispatch_in_phases_models() -> None:
    text = (RUNTIME_DIR / "training" / "phases" / "models.py").read_text(encoding="utf-8")
    assert "AnimaLycorisAdapter(preset=ANIMA_PRESET, " not in text
    assert "build_adapter(args, preset=ctx.family.lora_preset())" in text


def test_adapter_builders_do_not_import_anima_preset() -> None:
    adapters_dir = RUNTIME_DIR / "training" / "adapters"
    for filename in ("lycoris.py", "ortho.py", "tlora.py"):
        text = (adapters_dir / filename).read_text(encoding="utf-8")
        assert "families.anima.preset" not in text


def test_no_lr_scheduler_dispatch_in_phases_optimizer() -> None:
    text = (RUNTIME_DIR / "training" / "phases" / "optimizer.py").read_text(encoding="utf-8")
    assert 'if lr_sched == "cosine"' not in text
    assert 'CosineAnnealingLR' not in text
    assert 'CosineAnnealingWarmRestarts' not in text


def test_no_er_sde_inline_dispatch_in_sampling() -> None:
    text = (RUNTIME_DIR / "training" / "families" / "anima" / "sampling.py").read_text(encoding="utf-8")
    assert 'if sampler_name_l == "er_sde"' not in text
    assert "build_inference_sampler" in text
