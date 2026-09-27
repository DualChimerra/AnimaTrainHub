from __future__ import annotations

import pytest
from pydantic import ValidationError

from studio.domain.config_rules import (
    ADVISORY_DISABLE_FIELDS,
    apply_disable_rule_fixes,
    disable_rule_violations,
    iter_forbid_rules,
    iter_pin_rules,
)
from studio.schema import TrainingConfig


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_pin_setdefault_fills_missing_targets() -> None:
    cfg = TrainingConfig(navit_packing=True)
    assert cfg.attention_backend == "xformers"
    assert cfg.cache_latents is True
    assert cfg.batch_size == 1


def test_navit_batch_size_explicit_violation_rejected() -> None:
    with pytest.raises(ValidationError, match="batch_size"):
        TrainingConfig(navit_packing=True, batch_size=2)


def test_navit_batch_size_tolerant_fix_pins_not_gates() -> None:
    data = {"navit_packing": True, "batch_size": 2}
    fixed, fields = apply_disable_rule_fixes(data, TrainingConfig)
    assert fixed["navit_packing"] is True
    assert fixed["batch_size"] == 1
    assert "batch_size" in fields


def test_navit_native_orphan_tolerant_fix() -> None:
    data = {"navit_packing": False, "navit_native_resolution": True}
    fixed, fields = apply_disable_rule_fixes(data, TrainingConfig)
    assert fixed["navit_native_resolution"] is False
    assert fixed["navit_packing"] is False
    assert "navit_native_resolution" in fields


def test_navit_native_with_packing_still_allowed() -> None:
    cfg = TrainingConfig(navit_packing=True, navit_native_resolution=True)
    assert cfg.navit_native_resolution is True


def test_pin_setdefault_does_not_touch_when_gate_off() -> None:
    cfg = TrainingConfig()
    assert cfg.attention_backend == "flash_attn"


def test_explicit_violation_fails_fast_not_coerced() -> None:
    with pytest.raises(ValidationError, match="attention_backend"):
        TrainingConfig(navit_packing=True, attention_backend="flash_attn")
    with pytest.raises(ValidationError, match="cache_latents"):
        TrainingConfig(navit_packing=True, cache_latents=False)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("overrides", [
    {"optimizer_type": "prodigy", "lr_scheduler": "cosine"},
    {"optimizer_type": "soap_sf", "lr_scheduler": "cosine"},
    {"infonoise_enabled": True, "loss_weighting": "min_snr"},
    {"infonoise_enabled": True, "timestep_schedule_shift": 2.0},
    {"infonoise_enabled": True, "loss_type": "huber"},
    {"infonoise_enabled": True, "noise_enhancement_type": "offset"},
    {"leap_enabled": True, "infonoise_enabled": True},
    {"leap_enabled": True, "loss_weighting": "min_snr"},
    {"leap_enabled": True, "loss_type": "huber"},
    {"navit_packing": True, "leap_enabled": True},
    {"navit_packing": True, "infonoise_enabled": True},
    {"navit_packing": True, "sra_enabled": True},
    {"navit_packing": True, "masked_loss": True},
    {"leap_enabled": True, "masked_loss": True},
    {"navit_native_resolution": True},
])
def test_mutex_pairs_still_rejected(overrides) -> None:
    with pytest.raises(ValidationError, match="cross-field constraint"):
        TrainingConfig(**overrides)


def test_forbid_navit_tlora_rejected() -> None:
    with pytest.raises(ValidationError, match="lora_type"):
        TrainingConfig(navit_packing=True, lora_type="tlora")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_leap_sra_now_rejected() -> None:
    with pytest.raises(ValidationError, match="sra_enabled"):
        TrainingConfig(leap_enabled=True, sra_enabled=True)


def test_automagic_v2_grad_clip_pinned_zero() -> None:
    cfg = TrainingConfig(optimizer_type="automagic", automagic_variant="v2")
    assert cfg.grad_clip_max_norm == 0.0
    with pytest.raises(ValidationError, match="grad_clip_max_norm"):
        TrainingConfig(
            optimizer_type="automagic", automagic_variant="v2",
            grad_clip_max_norm=1.0,
        )
    assert TrainingConfig(optimizer_type="automagic").grad_clip_max_norm == 1.0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_advisory_learning_rate_not_enforced() -> None:
    assert "learning_rate" in ADVISORY_DISABLE_FIELDS
    cfg = TrainingConfig(optimizer_type="prodigy", learning_rate=0.5)
    assert cfg.learning_rate == 0.5
    assert all(name != "learning_rate" for name, *_ in iter_pin_rules(TrainingConfig))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_fix_gate_first_preserves_user_investment() -> None:
    data = {"infonoise_enabled": True, "loss_type": "huber", "loss_weighting": "min_snr"}
    fixed, fields = apply_disable_rule_fixes(data, TrainingConfig)
    assert fixed["infonoise_enabled"] is False
    assert fixed["loss_type"] == "huber"
    assert fixed["loss_weighting"] == "min_snr"
    assert fields == ["infonoise_enabled"]


def test_fix_pins_target_when_no_gate() -> None:
    data = {"optimizer_type": "prodigy", "lr_scheduler": "cosine"}
    fixed, fields = apply_disable_rule_fixes(data, TrainingConfig)
    assert fixed["optimizer_type"] == "prodigy"
    assert fixed["lr_scheduler"] == "none"
    assert fields == ["lr_scheduler"]


def test_fix_noop_on_valid_data() -> None:
    fixed, fields = apply_disable_rule_fixes({"epochs": 5}, TrainingConfig)
    assert fixed == {"epochs": 5}
    assert fields == []


def test_tolerant_validate_uses_rule_fixes() -> None:
    from studio.services.presets.io import _tolerant_validate

    cfg, dropped, defaulted = _tolerant_validate({
        "infonoise_enabled": True, "loss_weighting": "detail_inv_t",
    })
    assert cfg.infonoise_enabled is False
    assert cfg.loss_weighting == "detail_inv_t"
    assert "infonoise_enabled" in defaulted
    assert dropped == []


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_all_rules_have_hints() -> None:
    for name, _expr, _pin, hint in iter_pin_rules(TrainingConfig):
        assert hint, f"{name} disable_when missing disable_hint"
    for name, value, _expr, hint in iter_forbid_rules(TrainingConfig):
        assert hint, f"{name} option_disable_when[{value}] missing disable_hint"


def test_default_config_has_no_violations() -> None:
    dumped = TrainingConfig().model_dump(mode="python")
    assert disable_rule_violations(dumped, TrainingConfig) == []
