"""argparse_bridge -- tests for the pydantic model -> argparse parser generator."""
from __future__ import annotations

from typing import Literal, Optional

import pytest
from pydantic import BaseModel, Field

from studio.infrastructure import argparse_bridge as bridge
from studio.schema import TrainingConfig


# ---------------------------------------------------------------------------
# Type mapping -- checked one by one with a minimal fixture model
# ---------------------------------------------------------------------------


class _Sample(BaseModel):
    """Covers every type branch supported by the bridge."""
    name: str = Field("alpha")
    count: int = Field(3, ge=0)
    rate: float = Field(0.5)
    enabled: bool = Field(True)
    mode: Literal["a", "b", "c"] = Field("a")
    tags: list[str] = Field(default_factory=list)
    optional_path: Optional[str] = Field(None)


def test_int_field_parsed_as_int() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    ns = parser.parse_args(["--count", "42"])
    assert ns.count == 42 and isinstance(ns.count, int)


def test_float_field_parsed_as_float() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    ns = parser.parse_args(["--rate", "0.125"])
    assert ns.rate == 0.125


def test_bool_uses_paired_flag() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    # default True
    assert parser.parse_args([]).enabled is True
    # --no-enabled flips it
    assert parser.parse_args(["--no-enabled"]).enabled is False
    # --enabled turns it on explicitly
    assert parser.parse_args(["--enabled"]).enabled is True


def test_bool_field_named_no_x_uses_paired_store_actions() -> None:
    """Field names starting with no_ fall back to a pair of mutually exclusive
    store_true/store_false flags.

    py3.13+ argparse rejects passing --no-X to BooleanOptionalAction (issue #170).
    This test codifies the fallback path: default preserved / --no-X sets True /
    --X sets False.
    """
    from studio.schema import TrainingConfig

    parser = bridge.build_parser(TrainingConfig, add_config_arg=False)
    # default True (no_progress defaults to True in the schema)
    assert parser.parse_args([]).no_progress is True
    # --no-progress explicitly turns it on (backward compatible with old CLI habits)
    assert parser.parse_args(["--no-progress"]).no_progress is True
    # --progress turns it off -> no_progress=False
    assert parser.parse_args(["--progress"]).no_progress is False


def test_help_tolerates_percent_in_description() -> None:
    """format_help should not crash when description contains a bare `%`, and
    the output should still show a single `%`.

    argparse expands description as a printf template via `% params`; an
    unescaped `%` triggers a ValueError. Schema descriptions are also used by
    the Web UI / i18n, so they should not be polluted by argparse semantics --
    the bridge escapes them as a fallback.
    """
    class _PctSample(BaseModel):
        ratio: float = Field(0.5, description="new value is 90%; higher responds faster")

    parser = bridge.build_parser(_PctSample, add_config_arg=False)
    text = parser.format_help()  # used to raise ValueError outright on py3.10+
    # the user still sees a single % (not %%)
    assert "is 90%;" in text
    assert "%%" not in text


def test_literal_emits_choices() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    assert parser.parse_args(["--mode", "b"]).mode == "b"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "x"])


def test_list_uses_nargs_star() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    assert parser.parse_args([]).tags == []
    assert parser.parse_args(["--tags", "a", "b", "c"]).tags == ["a", "b", "c"]


def test_optional_str_default_is_none() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    assert parser.parse_args([]).optional_path is None
    assert parser.parse_args(["--optional-path", "/x"]).optional_path == "/x"


def test_dest_matches_field_name() -> None:
    """dest must use underscores (the YAML field name), not the CLI's hyphenated form."""
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    ns = parser.parse_args(["--optional-path", "x"])
    assert hasattr(ns, "optional_path") and not hasattr(ns, "optional-path")


# ---------------------------------------------------------------------------
# CLI alias -- declared explicitly via json_schema_extra
# ---------------------------------------------------------------------------


class _Aliased(BaseModel):
    learning_rate: float = Field(1e-4, json_schema_extra={"cli_alias": "--lr"})


def test_cli_alias_overrides_default_flag() -> None:
    parser = bridge.build_parser(_Aliased, add_config_arg=False)
    ns = parser.parse_args(["--lr", "5e-5"])
    assert ns.learning_rate == 5e-5
    # the default flag --learning-rate should no longer exist (unsupported)
    with pytest.raises(SystemExit):
        parser.parse_args(["--learning-rate", "5e-5"])


# ---------------------------------------------------------------------------
# YAML merge semantics (suppress parser + namespace_from_config)
# ---------------------------------------------------------------------------


def _sparse(argv: list[str], model_cls=_Sample) -> "bridge.argparse.Namespace":
    parser = bridge.build_parser(model_cls, add_config_arg=False, suppress_defaults=True)
    return parser.parse_args(argv)


def test_suppress_parser_yields_only_explicit_keys() -> None:
    """With suppress_defaults=True the namespace only contains keys the user passed explicitly."""
    assert vars(_sparse([])) == {}
    ns = _sparse(["--count", "7", "--no-enabled"])
    assert vars(ns) == {"count": 7, "enabled": False}


def test_yaml_fills_fields_cli_did_not_set() -> None:
    args = _sparse([])
    merged = bridge.namespace_from_config(args, {"count": 99}, _Sample)
    assert merged.count == 99
    # fields absent from both CLI and YAML fall back to the schema default
    assert merged.name == "alpha" and merged.enabled is True


def test_cli_wins_over_yaml() -> None:
    args = _sparse(["--count", "7"])
    merged = bridge.namespace_from_config(args, {"count": 99}, _Sample)
    assert merged.count == 7  # CLI wins


def test_cli_explicit_default_value_still_wins() -> None:
    """Explicitly passing a value that happens to equal the default still counts as
    explicit -- the old "value == default" heuristic had a blind spot here; with
    SUPPRESS detection the CLI wins exactly."""
    args = _sparse(["--count", "3"])  # 3 == schema default
    merged = bridge.namespace_from_config(args, {"count": 99}, _Sample)
    assert merged.count == 3


def test_yaml_unknown_keys_ignored() -> None:
    merged = bridge.namespace_from_config(
        _sparse([]), {"this_does_not_exist": 123}, _Sample
    )
    assert not hasattr(merged, "this_does_not_exist")


def test_non_schema_namespace_keys_pass_through() -> None:
    """CLI-only switches (--interactive and other non-schema keys) pass through
    to the output namespace unchanged."""
    args = _sparse([])
    args.interactive = True
    args.monitor_state_file = None
    merged = bridge.namespace_from_config(args, {"count": 5}, _Sample)
    assert merged.interactive is True
    assert merged.monitor_state_file is None
    assert merged.count == 5


def test_namespace_from_config_runs_validators() -> None:
    """The merged result is validated by pydantic as a whole -- invalid combinations
    (like mutually exclusive options) fail fast instead of being silently allowed."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="loss_weighting"):
        bridge.namespace_from_config(
            _sparse([], TrainingConfig),
            {"infonoise_enabled": True, "loss_weighting": "min_snr"},
            TrainingConfig,
        )


def test_krea2_pruned_yaml_gets_family_defaults() -> None:
    """Regression test for the shuffle_caption training-rejection bug
    (docs/design/config-pipeline-refactor.md #1):

    krea2's on-disk config has shuffle_caption pruned by show_when, so it never
    gets saved; when the trainer loads it, the missing key must fall back to
    False via the FAMILY_CONFIG_DEFAULTS overlay (krea2 semantics), not
    argparse's bare default of True (anima semantics) -- otherwise capability
    validation rejects the run.
    """
    from studio.domain.common import capability_violations
    from studio.domain.config_prune import prune_inactive_fields

    pruned = prune_inactive_fields(
        TrainingConfig(model_family="krea2").model_dump(mode="python")
    )
    assert "shuffle_caption" not in pruned  # precondition: pruning actually removed it
    merged = bridge.namespace_from_config(_sparse([], TrainingConfig), pruned, TrainingConfig)
    assert merged.shuffle_caption is False
    assert capability_violations("krea2", vars(merged)) == []


# ---------------------------------------------------------------------------
# Full self-check against the real TrainingConfig
# ---------------------------------------------------------------------------


def test_training_config_builds_without_collisions() -> None:
    """Every field registers on a single parser with no dest/flag collisions."""
    parser = bridge.build_parser(TrainingConfig)
    # --config is added automatically by add_config_arg=True
    ns = parser.parse_args([])
    assert hasattr(ns, "config")
    # sampled fields all parse to the correct type
    assert ns.lora_rank == 32
    assert ns.lora_type == "lora"
    assert ns.cache_latents is True
    assert ns.vae_cache_batch_size == 0
    assert ns.sample_prompts == []
    assert ns.optimizer_type == "adamw"


def test_training_config_cli_smoke() -> None:
    """Simulates a user changing a few fields from the CLI."""
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--lora-rank", "64",
        "--optimizer-type", "prodigy",
        "--no-shuffle-caption",
        "--sample-prompts", "p1", "p2",
    ])
    assert ns.lora_rank == 64
    assert ns.optimizer_type == "prodigy"
    assert ns.shuffle_caption is False
    assert ns.sample_prompts == ["p1", "p2"]


def test_training_config_cli_tlora() -> None:
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--lora-type", "tlora",
        "--tlora-min-rank", "12",
        "--tlora-alpha-rank-scale", "1.5",
        "--tlora-use-ortho",
    ])
    assert ns.lora_type == "tlora"
    assert ns.tlora_min_rank == 12
    assert ns.tlora_alpha_rank_scale == 1.5
    assert ns.tlora_use_ortho is True


def test_training_config_cli_ortho() -> None:
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args(["--lora-type", "ortho"])
    assert ns.lora_type == "ortho"


def test_training_config_cli_lion() -> None:
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--optimizer-type", "lion",
        "--lion-beta1", "0.95",
        "--lion-beta2", "0.98",
    ])
    assert ns.optimizer_type == "lion"
    assert ns.lion_beta1 == 0.95
    assert ns.lion_beta2 == 0.98


def test_training_config_cli_came() -> None:
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--optimizer-type", "came",
        "--came-beta1", "0.85",
        "--came-beta3", "0.9995",
        "--came-eps2", "1e-15",
        "--came-clip-threshold", "0.8",
    ])
    assert ns.optimizer_type == "came"
    assert ns.came_beta1 == 0.85
    assert ns.came_beta2 == 0.999  # default value
    assert ns.came_beta3 == 0.9995
    assert ns.came_eps2 == 1e-15
    assert ns.came_clip_threshold == 0.8


def test_training_config_cli_automagic() -> None:
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--optimizer-type", "automagic",
        "--automagic-min-lr", "1e-8",
        "--automagic-max-lr", "0.001",
        "--automagic-lr-bump", "2e-6",
        "--automagic-beta2", "0.998",
        "--automagic-clip-threshold", "0.8",
    ])
    assert ns.optimizer_type == "automagic"
    assert ns.automagic_min_lr == 1e-8
    assert ns.automagic_max_lr == 0.001
    assert ns.automagic_lr_bump == 2e-6
    assert ns.automagic_beta2 == 0.998
    assert ns.automagic_clip_threshold == 0.8


def test_training_config_cli_cosine_with_warmup() -> None:
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--lr-scheduler", "cosine_with_warmup",
        "--lr-scheduler-warmup-steps", "25",
        "--lr-scheduler-eta-min", "1e-7",
    ])
    assert ns.lr_scheduler == "cosine_with_warmup"
    assert ns.lr_scheduler_warmup_steps == 25
    assert ns.lr_scheduler_eta_min == 1e-7


def test_training_config_cli_cosine_cycles_float_lists() -> None:
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--lr-scheduler", "cosine_cycles",
        "--lr-scheduler-cycle-count", "3",
        "--lr-scheduler-cycle-max-lrs", "1e-4", "8e-5", "6e-5",
        "--lr-scheduler-cycle-min-lrs", "5e-5", "4e-5", "0",
    ])
    assert ns.lr_scheduler == "cosine_cycles"
    assert ns.lr_scheduler_cycle_max_lrs == [1e-4, 8e-5, 6e-5]
    assert ns.lr_scheduler_cycle_min_lrs == [5e-5, 4e-5, 0.0]


def test_training_config_yaml_round_trip() -> None:
    """Runs the full CLI -> YAML merge path, confirming yaml_dict fields all get read into args."""
    args = bridge.namespace_from_config(
        _sparse([], TrainingConfig),
        {
            "lora_rank": 16,
            "epochs": 3,
            "optimizer_type": "prodigy",
            "prodigy_d_coef": 0.5,
        },
        TrainingConfig,
    )
    assert args.lora_rank == 16
    assert args.epochs == 3
    assert args.optimizer_type == "prodigy"
    assert args.prodigy_d_coef == 0.5


def test_training_config_help_does_not_crash() -> None:
    """Generating the help text should not trigger any NoneType / formatting errors."""
    parser = bridge.build_parser(TrainingConfig)
    text = parser.format_help()
    assert "--lora-rank" in text
    assert "--no-cache-latents" in text  # reverse flag for a bool field


def test_training_config_cli_ppsf() -> None:
    """All ProdigyPlusScheduleFree CLI fields parse correctly."""
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--optimizer-type", "prodigy_plus_schedulefree",
        "--ppsf-d-coef", "0.5",
        "--ppsf-prodigy-steps", "500",
        "--ppsf-beta2", "0.95",
        "--ppsf-use-speed",
        "--no-ppsf-split-groups-mean",
    ])
    assert ns.optimizer_type == "prodigy_plus_schedulefree"
    assert ns.ppsf_d_coef == 0.5
    assert ns.ppsf_prodigy_steps == 500
    assert ns.ppsf_beta2 == 0.95
    assert ns.ppsf_use_speed is True
    assert ns.ppsf_split_groups_mean is False


def test_training_config_yaml_ppsf() -> None:
    """PPSF fields can be merged into the namespace via YAML."""
    args = bridge.namespace_from_config(
        _sparse([], TrainingConfig),
        {
            "optimizer_type": "prodigy_plus_schedulefree",
            "ppsf_d_coef": 0.3,
            "ppsf_beta1": 0.9,
            "ppsf_beta2": 0.99,
            "ppsf_prodigy_steps": 1000,
            "ppsf_use_stableadamw": True,
        },
        TrainingConfig,
    )
    assert args.optimizer_type == "prodigy_plus_schedulefree"
    assert args.ppsf_d_coef == 0.3
    assert args.ppsf_beta1 == 0.9
    assert args.ppsf_beta2 == 0.99
    assert args.ppsf_prodigy_steps == 1000
    assert args.ppsf_use_stableadamw is True


# ---------------------------------------------------------------------------
# Fields added across three PRs: verify bridge auto-generates the CLI flags
# ---------------------------------------------------------------------------


def test_training_config_emits_detail_inv_t_flags() -> None:
    """PR #72 introduced detail_inv_t_min/max; bridge should auto-generate --detail-inv-t-min/--max."""
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--detail-inv-t-min", "1.5",
        "--detail-inv-t-max", "8.0",
    ])
    assert ns.detail_inv_t_min == 1.5
    assert ns.detail_inv_t_max == 8.0


def test_training_config_emits_timestep_mix_low_prob_flag() -> None:
    """PR #73 introduced timestep_mix_low_prob; bridge should auto-generate --timestep-mix-low-prob."""
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args(["--timestep-mix-low-prob", "0.25"])
    assert ns.timestep_mix_low_prob == 0.25


def test_training_config_emits_timestep_schedule_shift_flag() -> None:
    """PR #73 introduced timestep_schedule_shift (the name after PR-A's rename);
    bridge should auto-generate --timestep-schedule-shift."""
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args(["--timestep-schedule-shift", "1.5"])
    assert ns.timestep_schedule_shift == 1.5


def test_training_config_emits_mixed_uniform_modes() -> None:
    """PR #73 introduced two new modes; the Literal choices should be accepted by bridge."""
    parser = bridge.build_parser(TrainingConfig)
    ns_low = parser.parse_args(["--timestep-sampling", "mixed_uniform_low"])
    assert ns_low.timestep_sampling == "mixed_uniform_low"
    ns_logit = parser.parse_args(["--timestep-sampling", "mixed_uniform_logit"])
    assert ns_logit.timestep_sampling == "mixed_uniform_logit"


def test_training_config_emits_loss_type_and_huber_flags() -> None:
    """PR #75 introduced loss_type / huber_c; bridge should auto-generate --loss-type / --huber-c."""
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args(["--loss-type", "huber", "--huber-c", "0.2"])
    assert ns.loss_type == "huber"
    assert ns.huber_c == 0.2
