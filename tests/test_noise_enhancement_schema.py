"""Regression tests for schema.noise_enhancement_type + migrate_noise_enhancement_type.

Aligned with kohya-ss/sd-scripts PR #477 (raise error when both noise_offset and
multires): noise_offset and pyramid noise are governed by a single type field on the
Anima side, and the schema/migration layer force-clears the opposing field (lesson
from kohya_ss issue #2599: hiding it in the UI doesn't clear the value -- the
serialization layer must enforce mutual exclusion).
"""
from __future__ import annotations

import pytest

from studio.schema import TrainingConfig, migrate_noise_enhancement_type


# ---------------------------------------------------------------------------
# migrate_noise_enhancement_type (dict-level helper)
# ---------------------------------------------------------------------------


def test_migrate_legacy_only_offset() -> None:
    """Only noise_offset > 0 set -> type=offset."""
    out = migrate_noise_enhancement_type({"noise_offset": 0.05})
    assert out["noise_enhancement_type"] == "offset"
    assert out["pyramid_noise_iters"] == 0


def test_migrate_legacy_only_pyramid() -> None:
    """Only pyramid_noise_iters > 0 set -> type=pyramid, noise_offset cleared."""
    out = migrate_noise_enhancement_type({"pyramid_noise_iters": 3})
    assert out["noise_enhancement_type"] == "pyramid"
    assert out["noise_offset"] == 0.0


def test_migrate_legacy_neither() -> None:
    """Neither set / both 0 -> type=none."""
    out = migrate_noise_enhancement_type({})
    assert out["noise_enhancement_type"] == "none"


def test_migrate_legacy_both_set_pyramid_wins() -> None:
    """Legacy buggy config: both > 0 -> pyramid wins, offset cleared.

    Reason: the normalization at the end of Anima's old make_noise diluted noise_offset's
    constant bias, so pyramid was the one that actually took effect.
    """
    out = migrate_noise_enhancement_type({
        "noise_offset": 0.05,
        "pyramid_noise_iters": 3,
    })
    assert out["noise_enhancement_type"] == "pyramid"
    assert out["noise_offset"] == 0.0


# ---------------------------------------------------------------------------
# runtime defense-in-depth: noise_params_from_args dispatches by type (audit #3)
# ---------------------------------------------------------------------------


def test_noise_params_dispatch_by_type() -> None:
    """The runtime consumer goes by noise_enhancement_type -- for args built outside the
    normal config path (e.g. an old-format pause snapshot), the leftover opposing param is
    ignored, preventing offset+pyramid from stacking."""
    from argparse import Namespace

    from training.noise import noise_params_from_args

    both = Namespace(
        noise_enhancement_type="offset",
        noise_offset=0.05, pyramid_noise_iters=3, pyramid_noise_discount=0.5,
    )
    assert noise_params_from_args(both) == (0.05, 0, 0.5)

    both.noise_enhancement_type = "pyramid"
    assert noise_params_from_args(both) == (0.0, 3, 0.5)

    both.noise_enhancement_type = "none"
    assert noise_params_from_args(both) == (0.0, 0, 0.5)


def test_migrate_explicit_type_offset_clears_pyramid() -> None:
    """Explicit type=offset -> pyramid_noise_iters is force-cleared (lesson from issue #2599)."""
    out = migrate_noise_enhancement_type({
        "noise_enhancement_type": "offset",
        "noise_offset": 0.05,
        "pyramid_noise_iters": 3,  # stale leftover, must be cleared
    })
    assert out["noise_enhancement_type"] == "offset"
    assert out["noise_offset"] == 0.05
    assert out["pyramid_noise_iters"] == 0


def test_migrate_explicit_type_pyramid_clears_offset() -> None:
    out = migrate_noise_enhancement_type({
        "noise_enhancement_type": "pyramid",
        "noise_offset": 0.05,  # stale leftover
        "pyramid_noise_iters": 3,
    })
    assert out["noise_enhancement_type"] == "pyramid"
    assert out["noise_offset"] == 0.0
    assert out["pyramid_noise_iters"] == 3


def test_migrate_explicit_type_none_clears_both() -> None:
    out = migrate_noise_enhancement_type({
        "noise_enhancement_type": "none",
        "noise_offset": 0.05,
        "pyramid_noise_iters": 3,
    })
    assert out["noise_enhancement_type"] == "none"
    assert out["noise_offset"] == 0.0
    assert out["pyramid_noise_iters"] == 0


def test_migrate_idempotent() -> None:
    once = migrate_noise_enhancement_type({"pyramid_noise_iters": 3})
    twice = migrate_noise_enhancement_type(dict(once))
    assert once == twice


def test_migrate_non_dict_passthrough() -> None:
    """Non-dict input is returned unchanged (compat with pydantic model_validator(mode='before'))."""
    assert migrate_noise_enhancement_type(None) is None
    assert migrate_noise_enhancement_type("notadict") == "notadict"
    assert migrate_noise_enhancement_type(42) == 42


def test_migrate_str_number_coerced() -> None:
    """yaml occasionally reads a number as a str; must not crash, treat as 0."""
    out = migrate_noise_enhancement_type({"noise_offset": "0.05"})
    assert out["noise_enhancement_type"] == "offset"


def test_migrate_invalid_number_treated_as_zero() -> None:
    """Bad values (non-numeric str / None) -> treated as 0, must not raise."""
    out = migrate_noise_enhancement_type({
        "noise_offset": "abc",
        "pyramid_noise_iters": None,
    })
    assert out["noise_enhancement_type"] == "none"


# ---------------------------------------------------------------------------
# pydantic schema -- TrainingConfig constructs correctly and stays mutually exclusive after migrate
# ---------------------------------------------------------------------------


def test_schema_default_is_none() -> None:
    t = TrainingConfig()
    assert t.noise_enhancement_type == "none"
    assert t.noise_offset == 0.0
    assert t.pyramid_noise_iters == 0


def test_schema_legacy_offset_yaml() -> None:
    """Legacy yaml { noise_offset: 0.05 } -> type=offset, pyramid cleared."""
    t = TrainingConfig(noise_offset=0.05)  # type: ignore[call-arg]
    assert t.noise_enhancement_type == "offset"
    assert t.noise_offset == 0.05
    assert t.pyramid_noise_iters == 0


def test_schema_legacy_pyramid_yaml() -> None:
    t = TrainingConfig(pyramid_noise_iters=3)  # type: ignore[call-arg]
    assert t.noise_enhancement_type == "pyramid"
    assert t.noise_offset == 0.0
    assert t.pyramid_noise_iters == 3


def test_schema_legacy_both_set_pyramid_wins() -> None:
    """Legacy yaml with both set -> pyramid wins, offset cleared (same defense as issue #2599)."""
    t = TrainingConfig(noise_offset=0.05, pyramid_noise_iters=3)  # type: ignore[call-arg]
    assert t.noise_enhancement_type == "pyramid"
    assert t.noise_offset == 0.0
    assert t.pyramid_noise_iters == 3


def test_schema_explicit_offset_clears_pyramid() -> None:
    """Explicit type=offset but yaml still has a leftover pyramid field -> cleared."""
    t = TrainingConfig(
        noise_enhancement_type="offset",
        noise_offset=0.05,
        pyramid_noise_iters=3,  # stale leftover
    )
    assert t.noise_enhancement_type == "offset"
    assert t.noise_offset == 0.05
    assert t.pyramid_noise_iters == 0


def test_schema_explicit_pyramid_clears_offset() -> None:
    t = TrainingConfig(
        noise_enhancement_type="pyramid",
        noise_offset=0.05,  # stale leftover
        pyramid_noise_iters=3,
    )
    assert t.noise_enhancement_type == "pyramid"
    assert t.noise_offset == 0.0
    assert t.pyramid_noise_iters == 3


def test_schema_explicit_none_clears_both() -> None:
    t = TrainingConfig(
        noise_enhancement_type="none",
        noise_offset=0.05,
        pyramid_noise_iters=3,
    )
    assert t.noise_enhancement_type == "none"
    assert t.noise_offset == 0.0
    assert t.pyramid_noise_iters == 0


def test_schema_invalid_type_rejected() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TrainingConfig(noise_enhancement_type="multires")  # type: ignore[arg-type]


@pytest.mark.parametrize("t_value", ["none", "offset", "pyramid"])
def test_schema_all_types_validate(t_value: str) -> None:
    t = TrainingConfig(noise_enhancement_type=t_value)  # type: ignore[arg-type]
    assert t.noise_enhancement_type == t_value
