"""Pure computation for model-family switching (multi-model P4-3, C5).

"Switching family" is not a plain field edit: the 4 weight paths must be
recomputed for the target family, family-flavor fields (sampler / scheduler /
timestep, etc.) must reset to the target family's defaults, and capability
fields unsupported by the target family must be turned off — otherwise we'd
persist a broken config with `model_family: krea2` plus anima paths (#419 debt
C2), or a custom t5 path could silently vanish via show_when pruning (C3).

This module only does pure dict → dict computation plus a change list; path
recomputation depends on the services layer (dependency direction is domain ←
services, never reversed), so the caller injects `path_defaults`. Nothing is
persisted here — the frontend confirms first, then goes through the normal
save path.
"""
from typing import Any

from .common import FAMILY_CONFIG_DEFAULTS, capability_violations
from .training import TrainingConfig


def _flavor_keys() -> set:
    """Family-flavor fields = the union of all families' config_defaults keys.

    These fields' meaning drifts by family (e.g. krea2's sampler=euler /
    timestep=krea2_shift), so they're uniformly reset on switch: use the
    target family's overlay value if it has one, otherwise fall back to the
    schema default.
    """
    keys: set = set()
    for defaults in FAMILY_CONFIG_DEFAULTS.values():
        keys |= set(defaults)
    return keys


def switch_family(
    config: dict[str, Any],
    target: str,
    path_defaults: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Explicitly switch a TrainingConfig dict to the `target` family.

    Args:
        config: the current config (dict, some fields may be missing)
        target: the target family id (caller has already validated it)
        path_defaults: the target family's 4 weight paths (produced by the
            services layer's `default_paths_for_new_version(family=target)`)

    Returns:
        (the fully switched config dict, a change list). The change list is
        per field: `{"field", "from", "to"}`, containing only fields whose
        value actually changed; `from` is None when the field wasn't present
        in the original config.
    """
    schema_defaults = TrainingConfig().model_dump(mode="python")
    new: dict[str, Any] = dict(config)
    new["model_family"] = target
    # Normalize paths to forward slashes (same normalization used when persisting
    # to yaml); if the only difference from the old value is slash style, it's the
    # same file — keep the original value so we don't produce a spurious change
    # row (str(Path) emits backslashes on Windows).
    for key, value in path_defaults.items():
        normalized = str(value).replace("\\", "/") if value else value
        old = config.get(key)
        if old is not None and str(old).replace("\\", "/") == normalized:
            new[key] = old
        else:
            new[key] = normalized

    overlay = FAMILY_CONFIG_DEFAULTS.get(target, {})
    for key in sorted(_flavor_keys()):
        new[key] = overlay.get(key, schema_defaults.get(key))

    # Explicitly turn capability fields unsupported by the target family
    # (navit / sra / leap / tag semantics, etc.) back off to their schema
    # default — otherwise the capability validator would reject the whole config.
    for field in capability_violations(target, new):
        new[field] = schema_defaults.get(field)

    changes = [
        {"field": key, "from": config.get(key), "to": new[key]}
        for key in sorted(new)
        if config.get(key) != new[key]
    ]
    return new, changes
