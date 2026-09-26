"""End-to-end regression tests for anima_train.py after its migration to schema-driven
config.

Covers:
    - parse_args, through the bridge-generated parser, accepts all historical CLI aliases
    - apply_yaml_config, when writing YAML fields into args, follows "CLI explicit wins"
      semantics
    - config/train_template.yaml can be fully loaded, with all field types correct

Note: these tests import the anima_train module, which triggers a torch import (~3s),
so a module-scoped fixture imports it only once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def at():
    """import anima_train once and reuse it."""
    import importlib.util  # noqa: PLC0415
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_anima_train_for_test", repo_root / "runtime" / "anima_train.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# CLI aliases
# ---------------------------------------------------------------------------


def test_legacy_cli_aliases_still_work(at, monkeypatch: pytest.MonkeyPatch) -> None:
    """--transformer / --vae / --qwen / --t5-tokenizer / --lr from old scripts must
    still work."""
    monkeypatch.setattr(sys, "argv", [
        "anima_train.py",
        "--transformer", "/x/t.safetensors",
        "--vae", "/x/v.safetensors",
        "--qwen", "/x/q",
        "--t5-tokenizer", "/x/t5",
        "--lr", "5e-5",
    ])
    args = at.parse_args()
    assert args.transformer_path == "/x/t.safetensors"
    assert args.vae_path == "/x/v.safetensors"
    assert args.text_encoder_path == "/x/q"
    assert args.t5_tokenizer_path == "/x/t5"
    assert args.learning_rate == 5e-5


def test_args_has_t5_tokenizer_path_not_legacy_t5_tokenizer(
    at, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the args field name is t5_tokenizer_path, args.t5_tokenizer must
    never appear again.

    Why: anima_train.py's path resolution stage used to have
    `getattr(args, "t5_tokenizer", "")`, accessing the old name that had already been
    migrated away, which always returned "", overwriting the yaml/CLI-provided
    t5_tokenizer_path with an empty string -- causing T5Tokenizer to silently fall back
    to downloading google/t5-v1_1-xxl over the network -- which fails outright offline
    or on weak connections.
    """
    monkeypatch.setattr(sys, "argv", [
        "anima_train.py",
        "--t5-tokenizer", "/x/t5",
    ])
    args = at.parse_args()
    assert args.t5_tokenizer_path == "/x/t5"
    assert not hasattr(args, "t5_tokenizer"), (
        "args should not have a t5_tokenizer attribute -- the schema field is "
        "t5_tokenizer_path; any getattr(args, 't5_tokenizer', ...) would get the "
        "default value and wipe out the path"
    )


def test_cli_only_flags_present(at, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-only flags outside the schema must be preserved."""
    monkeypatch.setattr(sys, "argv", [
        "anima_train.py",
        "--auto-install",
        "--interactive",
        "--no-live-curve",
    ])
    args = at.parse_args()
    assert args.auto_install is True
    assert args.interactive is True
    assert args.no_live_curve is True


def test_deprecated_repeats_flags_silently_accepted(at, monkeypatch: pytest.MonkeyPatch) -> None:
    """--repeats / --reg-repeats are deprecated but still accepted (doesn't break old
    scripts)."""
    monkeypatch.setattr(sys, "argv", [
        "anima_train.py", "--repeats", "5", "--reg-repeats", "3",
    ])
    args = at.parse_args()
    assert args.repeats == 5
    assert args.reg_repeats == 3


def test_no_prefer_json_flips_default(at, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bridge automatically derives --prefer-json / --no-prefer-json from
    prefer_json: bool=True.

    Since Cut 1 / R1, parse_args returns a sparse namespace (only explicit keys);
    schema defaults are uniformly filled in by apply_yaml_config via TrainingConfig.
    """
    monkeypatch.setattr(sys, "argv", ["anima_train.py", "--no-prefer-json"])
    assert at.parse_args().prefer_json is False
    monkeypatch.setattr(sys, "argv", ["anima_train.py"])
    sparse = at.parse_args()
    assert not hasattr(sparse, "prefer_json")  # not explicitly passed -> not in the sparse namespace
    assert at.apply_yaml_config(sparse, {}).prefer_json is True


# ---------------------------------------------------------------------------
# YAML merging (apply_yaml_config returns a new normalized namespace, no longer
# mutates in place)
# ---------------------------------------------------------------------------


def test_apply_yaml_overrides_defaults(at, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["anima_train.py"])
    args = at.apply_yaml_config(at.parse_args(), {"epochs": 99, "lora_rank": 64})
    assert args.epochs == 99
    assert args.lora_rank == 64


def test_cli_wins_over_yaml(at, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["anima_train.py", "--epochs", "3"])
    args = at.apply_yaml_config(at.parse_args(), {"epochs": 99})
    assert args.epochs == 3


def test_cli_explicit_default_wins_over_yaml(at, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicitly passing a value equal to the schema default still counts as
    explicit (SUPPRESS gives exact detection; the old "value == default" approximation
    had a blind spot here -- epochs defaults to 10)."""
    monkeypatch.setattr(sys, "argv", ["anima_train.py", "--epochs", "10"])
    args = at.apply_yaml_config(at.parse_args(), {"epochs": 99})
    assert args.epochs == 10


def test_unknown_yaml_keys_ignored(at, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["anima_train.py"])
    args = at.apply_yaml_config(at.parse_args(), {"this_key_doesnt_exist": 42})
    assert not hasattr(args, "this_key_doesnt_exist")


def test_invalid_yaml_combo_exits(at, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutually exclusive combinations fail fast through the unified loading path
    (before Cut 1, the trainer silently let this through)."""
    monkeypatch.setattr(sys, "argv", ["anima_train.py"])
    with pytest.raises(SystemExit):
        at.apply_yaml_config(
            at.parse_args(),
            {"infonoise_enabled": True, "loss_weighting": "min_snr"},
        )


# ---------------------------------------------------------------------------
# config/train_template.yaml was removed together with the CLI workflow (Studio now
# uses a preset pool instead); the two fixture tests for it were removed accordingly.
# In Studio mode, yaml has its absolute path injected by fork_preset_for_version,
# covered in test_version_config / test_presets_io etc.
