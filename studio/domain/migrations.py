"""Legacy yaml schema migration functions — old field name → new field name mapping.

Called from two places:
  1. Each model's `@model_validator(mode='before')` — cleans up data before
     pydantic validation
  2. Explicitly, once, before `apply_yaml_config` in subprocesses such as
     runtime/anima_train.py — because argparse_bridge.merge_yaml_into_namespace
     doesn't go through the pydantic validator, so it needs this fallback layer

Note: does NOT use `from __future__ import annotations` — under Pydantic v2 +
Python 3.12+'s deferred evaluation, that would turn typing._SpecialForm into a
schema key and raise AttributeError.
"""
from typing import Any

# Built-in HTTP monitor server fields retired in PP6.1 — the values have long
# had no effect, and the schema fields were removed. Historically every dump
# wrote this set of defaults into the yaml, so when reading old configs they
# must be silently dropped, and must NOT go through _tolerant_validate's
# dropped_fields notice (otherwise every old config.yaml / preset would pop a
# compatibility banner on open). TrainingConfig's extra="ignore" and
# argparse_bridge skipping unknown keys already guarantee no error; this set
# only serves to quiet the dropped_fields noise.
RETIRED_MONITOR_KEYS = frozenset({"no_monitor", "monitor_host", "monitor_port", "no_browser"})


def migrate_legacy_save_keys(data: Any) -> Any:
    """Rename the old config's save_every / save_state_every to their unit-suffixed names.

    save_every       → save_every_epochs   (epoch-based)
    save_state_every → save_state_every_steps (step-based)

    Idempotent; if the new name already exists, the equivalent old name is discarded.
    """
    if not isinstance(data, dict):
        return data
    for legacy, new in (("save_every", "save_every_epochs"),
                         ("save_state_every", "save_state_every_steps")):
        if legacy in data:
            if new in data:
                data.pop(legacy)
            else:
                data[new] = data.pop(legacy)
    return data


def migrate_noise_enhancement_type(data: Any) -> Any:
    """Align with kohya: `noise_offset` and pyramid noise are mutually exclusive; controlled by a single type field.

    Two steps:
      1. When old yaml has no `noise_enhancement_type`, derive it from the
         existing fields:
            pyramid_noise_iters > 0   → "pyramid"
            noise_offset > 0          → "offset"
            both 0                    → "none"
         For the historical buggy config (both > 0): choose "pyramid". Reason:
         the normalization at the end of Anima's old `make_noise` implementation
         (`noise = cur / cur.std().clamp(...)`) dilutes noise_offset's constant
         bias, so pyramid is what actually dominates in practice — deriving to
         pyramid best matches what users would subjectively observe.
      2. Force the opposite group's field to zero — lesson from kohya_ss issue
         #2599: the fields must be mutually exclusive at the serialization
         layer; a hidden UI field is not the same as a cleared value, otherwise
         a leftover yaml value would leak into training.
         (History note: the argparse path used to bypass the pydantic
         validator; since blade 1 / R1, both the trainer and Studio go through
         TrainingConfig construction, so this migration now applies uniformly
         to both paths. The runtime make_noise side additionally has
         noise_params_from_args dispatching by type as defense in depth.)

    Idempotent: if `noise_enhancement_type` was already given explicitly, it's respected as-is.
    """
    if not isinstance(data, dict):
        return data
    if "noise_enhancement_type" not in data:
        pyramid = _coerce_num(data.get("pyramid_noise_iters", 0))
        offset = _coerce_num(data.get("noise_offset", 0.0))
        if pyramid > 0:
            data["noise_enhancement_type"] = "pyramid"
        elif offset > 0:
            data["noise_enhancement_type"] = "offset"
        else:
            data["noise_enhancement_type"] = "none"
    t = data["noise_enhancement_type"]
    if t == "offset":
        data["pyramid_noise_iters"] = 0
    elif t == "pyramid":
        data["noise_offset"] = 0.0
    elif t == "none":
        data["noise_offset"] = 0.0
        data["pyramid_noise_iters"] = 0
    return data


def _coerce_num(v: Any) -> float:
    """yaml may read a number as str / None; convert leniently to float for type derivation."""
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
