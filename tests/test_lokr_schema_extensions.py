from __future__ import annotations

from studio.schema import TrainingConfig


def test_lora_type_accepts_loha():
    cfg = TrainingConfig(lora_type="loha")
    assert cfg.lora_type == "loha"


def test_lora_type_accepts_tlora():
    cfg = TrainingConfig(lora_type="tlora")
    assert cfg.lora_type == "tlora"
    assert cfg.tlora_min_rank == 1
    assert cfg.tlora_alpha_rank_scale == 1.0
    assert cfg.tlora_use_ortho is True


def test_lora_type_accepts_ortho():
    cfg = TrainingConfig(lora_type="ortho")
    assert cfg.lora_type == "ortho"


def test_lora_type_still_accepts_legacy_values():
    assert TrainingConfig(lora_type="lora").lora_type == "lora"
    assert TrainingConfig(lora_type="lokr").lora_type == "lokr"


def test_lora_type_rejects_unknown():
    import pytest
    with pytest.raises(Exception):
        TrainingConfig(lora_type="random_string")


def test_dora_field_default_off():
    assert TrainingConfig().lora_dora is False


def test_rs_lora_field_default_off():
    assert TrainingConfig().lora_rs is False


def test_dropout_fields_default_zero():
    cfg = TrainingConfig()
    assert cfg.lora_dropout == 0.0
    assert cfg.lora_rank_dropout == 0.0
    assert cfg.lora_module_dropout == 0.0


def test_dropout_fields_clamped_to_unit_range():
    import pytest
    with pytest.raises(Exception):
        TrainingConfig(lora_dropout=1.5)
    with pytest.raises(Exception):
        TrainingConfig(lora_rank_dropout=-0.1)


def test_legacy_yaml_config_loads_with_only_old_fields():
    legacy = {
        "lora_type": "lokr",
        "lora_rank": 32,
        "lora_alpha": 32.0,
        "lokr_factor": 8,
    }
    cfg = TrainingConfig.model_validate(legacy)
    assert cfg.lora_dora is False
    assert cfg.lora_rs is False
    assert cfg.lora_dropout == 0.0


def test_new_fields_in_lora_group():
    schema = TrainingConfig.model_json_schema()
    props = schema["properties"]
    for f in (
        "lora_dora", "lora_rs", "lora_dropout", "lora_rank_dropout", "lora_module_dropout",
        "tlora_min_rank", "tlora_alpha_rank_scale", "tlora_use_ortho",
    ):
        assert f in props, f"missing field: {f}"
        assert props[f].get("group") == "lora", f"{f} is not in the lora group"
