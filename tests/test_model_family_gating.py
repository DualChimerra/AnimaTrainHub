
from __future__ import annotations

from typing import get_args

import pytest

from studio.domain.common import (
    FAMILY_CAPABILITIES,
    FAMILY_CONFIG_DEFAULTS,
    FAMILY_SAMPLING,
    FIELD_CAPABILITY_REQUIREMENTS,
    MODEL_FAMILIES,
    TIMESTEP_SAMPLING_OPTION_FAMILIES,
    cap_gate,
    capability_violations,
    option_gates,
    sampling_option_gates,
)
from studio.domain.config_prune import eval_show_when
from studio.schema import TrainingConfig


def test_capability_matrix_single_source():
    from training.families import SPECS

    assert set(FAMILY_CAPABILITIES) == set(SPECS)
    for fid, spec in SPECS.items():
        assert spec.capabilities is FAMILY_CAPABILITIES[fid], (
            f"SPECS['{fid}'].capabilities no longer references the studio single source -- it degenerated into a mirror"
        )
        assert spec.config_defaults is FAMILY_CONFIG_DEFAULTS[fid], fid


def test_sampling_whitelist_single_source():
    from training.families import SPECS

    assert set(FAMILY_SAMPLING) == set(SPECS)
    for fid, spec in SPECS.items():
        source = FAMILY_SAMPLING[fid]
        assert spec.sampling.samplers is source["samplers"], fid
        assert spec.sampling.schedulers is source["schedulers"], fid
        assert source["samplers"][0] == spec.sampling.default_sampler, fid
        assert source["schedulers"][0] == spec.sampling.default_scheduler, fid
        cd = spec.config_defaults
        if "sample_sampler_name" in cd:
            assert cd["sample_sampler_name"] == spec.sampling.default_sampler, fid
        if "sample_scheduler" in cd:
            assert cd["sample_scheduler"] == spec.sampling.default_scheduler, fid
        if "sample_infer_steps" in cd:
            assert cd["sample_infer_steps"] == spec.sampling.default_steps, fid
        if "sample_cfg_scale" in cd:
            assert cd["sample_cfg_scale"] == spec.sampling.default_cfg, fid


def test_timestep_option_gate_targets_registered_strategy():
    from training.timestep_samplers import BUILDERS

    literal = set(get_args(TrainingConfig.model_fields["timestep_sampling"].annotation))
    for option, fams in TIMESTEP_SAMPLING_OPTION_FAMILIES.items():
        assert option in literal, option
        assert option in BUILDERS, option
        assert set(fams) <= set(MODEL_FAMILIES), option


def test_schema_literal_matches_families():
    literal_values = get_args(TrainingConfig.model_fields["model_family"].annotation)
    assert tuple(literal_values) == MODEL_FAMILIES
    assert TrainingConfig.model_fields["model_family"].default == "anima"


def test_cap_gate_renders_field_comparison():
    assert cap_gate("navit") == "model_family==anima"
    with pytest.raises(ValueError):
        cap_gate("warp_drive")


def test_gated_fields_follow_family_capabilities():
    values = {"model_family": "anima"}
    for field in ("t5_tokenizer_path", "navit_packing", "masked_loss",
                  "leap_enabled", "sra_enabled", "shuffle_caption"):
        extra = TrainingConfig.model_fields[field].json_schema_extra
        assert eval_show_when(extra.get("show_when"), values) is True, field
    krea2 = {"model_family": "krea2"}
    assert eval_show_when(
        TrainingConfig.model_fields["masked_loss"].json_schema_extra["show_when"],
        krea2,
    ) is True
    assert eval_show_when(
        TrainingConfig.model_fields["text_encoder_cache"].json_schema_extra["show_when"],
        krea2,
    ) is True
    for field in ("t5_tokenizer_path", "navit_packing", "leap_enabled",
                  "sra_enabled", "shuffle_caption"):
        extra = TrainingConfig.model_fields[field].json_schema_extra
        assert eval_show_when(extra.get("show_when"), krea2) is False, field


def test_default_config_valid_and_yaml_roundtrip():
    cfg = TrainingConfig()
    assert cfg.model_family == "anima"
    TrainingConfig(navit_packing=True)
    TrainingConfig(masked_loss=True)
    krea2 = TrainingConfig(model_family="krea2")
    assert krea2.shuffle_caption is False
    assert krea2.text_encoder_cache is True
    assert krea2.attention_backend == "none"
    assert krea2.timestep_sampling == "krea2_shift"
    assert (krea2.sample_sampler_name, krea2.sample_scheduler) == (
        "euler", "simple",
    )
    assert (krea2.sample_infer_steps, krea2.sample_cfg_scale) == (28, 4.5)




def test_option_gates_render_expressions():
    assert sampling_option_gates("samplers") == {
        "er_sde": "model_family==anima",
        "dpmpp_3m_sde": "model_family==anima",
        "euler": "model_family==krea2",
    }
    assert option_gates(TIMESTEP_SAMPLING_OPTION_FAMILIES) == {
        "krea2_shift": "model_family==krea2",
    }


def test_option_show_when_wired_into_schema_fields():
    for field, kind in (
        ("sample_sampler_name", "samplers"),
        ("sample_scheduler", "schedulers"),
    ):
        extra = TrainingConfig.model_fields[field].json_schema_extra
        gates = extra.get("option_show_when")
        assert gates == sampling_option_gates(kind), field
        literal = set(get_args(TrainingConfig.model_fields[field].annotation))
        assert set(gates) <= literal, field
    ts_extra = TrainingConfig.model_fields["timestep_sampling"].json_schema_extra
    assert ts_extra.get("option_show_when") == option_gates(
        TIMESTEP_SAMPLING_OPTION_FAMILIES
    )


def test_cross_family_sampler_asymmetric_grandfather():
    legacy = TrainingConfig(
        model_family="anima",
        sample_sampler_name="euler", sample_scheduler="krea2_shift",
    )
    assert (legacy.sample_sampler_name, legacy.sample_scheduler) == (
        "er_sde", "simple",
    )
    with pytest.raises(ValueError, match="er_sde"):
        TrainingConfig(model_family="krea2", sample_sampler_name="er_sde")
    with pytest.raises(ValueError, match="sgm_uniform"):
        TrainingConfig(model_family="krea2", sample_scheduler="sgm_uniform")


def test_krea2_shift_scheduler_value_renamed_to_simple():
    cfg = TrainingConfig(model_family="krea2", sample_scheduler="krea2_shift")
    assert cfg.sample_scheduler == "simple"


def test_legacy_garbage_sampler_still_migrates_to_family_default():
    cfg = TrainingConfig(
        sample_sampler_name="ancient_free_text", sample_scheduler="karras",
    )
    assert (cfg.sample_sampler_name, cfg.sample_scheduler) == ("er_sde", "simple")
    krea2 = TrainingConfig(
        model_family="krea2",
        sample_sampler_name="ancient_free_text", sample_scheduler="karras",
    )
    assert (krea2.sample_sampler_name, krea2.sample_scheduler) == (
        "euler", "simple",
    )


def test_krea2_shift_timestep_allowed_for_any_family():
    cfg = TrainingConfig(model_family="anima", timestep_sampling="krea2_shift")
    assert cfg.timestep_sampling == "krea2_shift"


def test_capability_violations_flags_unsupported(monkeypatch):
    monkeypatch.setitem(FAMILY_CAPABILITIES, "fakefam", frozenset({"masked_loss", "text_cache"}))
    bad = capability_violations("fakefam", {
        "navit_packing": True, "masked_loss": True,
        "shuffle_caption": True, "keep_tokens": 0, "tag_dropout": 0.0,
    })
    assert bad == ["navit_packing", "shuffle_caption"]
    assert capability_violations("fakefam", {k: False for k in FIELD_CAPABILITY_REQUIREMENTS}) == []
    assert capability_violations("no-such", {"navit_packing": True}) == []


def test_requirements_fields_exist_in_schema():
    for field in FIELD_CAPABILITY_REQUIREMENTS:
        assert field in TrainingConfig.model_fields, field


def test_prune_keeps_gated_fields_for_default_dump():
    from studio.domain.config_prune import prune_inactive_fields

    dumped = prune_inactive_fields(TrainingConfig().model_dump(mode="python"))
    assert dumped.get("model_family") == "anima"
    for field in ("t5_tokenizer_path", "shuffle_caption", "masked_loss",
                  "navit_packing", "leap_enabled", "sra_enabled"):
        assert field in dumped, field

    krea2 = prune_inactive_fields(
        TrainingConfig(model_family="krea2").model_dump(mode="python")
    )
    assert krea2["text_encoder_cache"] is True
    assert krea2["masked_loss"] is False
    for field in ("t5_tokenizer_path", "shuffle_caption", "keep_tokens",
                  "tag_dropout", "navit_packing", "leap_enabled", "sra_enabled"):
        assert field not in krea2, field
