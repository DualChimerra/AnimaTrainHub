"""Preset file I/O -- validated with pydantic, persisted with PyYAML.

Storage location: `studio_data/presets/{name}.yaml`
Name whitelist: `[A-Za-z0-9_-]+`, to prevent path traversal and illegal Windows characters.

History: before PP0 this was called configs_io / studio_data/configs/. `configs_io` is now
a thin shell over this module.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ...domain.config_prune import prune_inactive_fields
from ...domain.config_rules import apply_disable_rule_fixes
from ...domain.migrations import RETIRED_MONITOR_KEYS
from ...paths import REPO_ROOT, USER_PRESETS_DIR
from ...schema import TrainingConfig

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# Global model path fields. Always absolute paths when writing to yaml + showing in the UI; when reading an old yaml with a
# relative path, _absolutize_model_paths falls back to converting it to absolute (faithful to the historical CWD=REPO_ROOT resolution
# semantics), so downstream code can safely assume these 4 fields are absolute paths.
_MODEL_PATH_FIELDS = (
    "transformer_path",
    "vae_path",
    "text_encoder_path",
    "t5_tokenizer_path",
)


from studio.domain.errors import DomainError


class PresetError(DomainError):
    """Preset I/O error.

    PR-2 C3 added a DomainError base -- the handler auto-translates it into the dual-write envelope.
    The existing raise PresetError("xxx") form is unchanged; http_status / code are overridden case by case by the
    router or when C4/C5 refine things (currently uses default = 400 / preset.error).
    """
    default_code = "preset.error"


_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _absolutize_model_paths(data: dict[str, Any]) -> dict[str, Any]:
    """Normalizes the 4 model path fields: relative path -> absolute under REPO_ROOT; separators unified to POSIX `/`.

    - A relative path in an old yaml (schema fallback) -> converted to an absolute path based on REPO_ROOT
      (faithful to the historical supervisor cwd=REPO_ROOT resolution semantics)
    - On Windows `str(Path)` gives backslashes (`G:\\foo`), while PathPicker gives POSIX
      (`G:/foo`); mixing them would make the same field look inconsistent depending on its source; unifying with `as_posix()`
      makes yaml writes + UI display always use `/`
    - Cross-platform bundle import: a Windows drive-letter path (`G:/...`) on POSIX has
      `Path.is_absolute()` return False, so it would be mistakenly treated as relative and joined onto REPO_ROOT, becoming
      `<repo>/G:/...`. This uses an extra regex to recognize a drive-letter prefix as absolute, to avoid silently mangling it.
      (The path is still unresolvable on a different machine, but keeping it as-is lets the UI/logs trace back to the original source.)
    Doesn't touch the yaml file; the next save naturally writes the normalized form.
    """
    for f in _MODEL_PATH_FIELDS:
        v = data.get(f)
        if isinstance(v, str) and v:
            if _WIN_DRIVE_RE.match(v):
                data[f] = v.replace("\\", "/")
                continue
            p = Path(v)
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            data[f] = p.as_posix()
    return data


def _validate_name(name: str) -> None:
    if not NAME_PATTERN.fullmatch(name):
        raise PresetError(
            f'Invalid preset name "{name}"; use letters, digits, underscore, or hyphen',
            code="preset.name_invalid",
            details={"name": name},
            http_status=400,
        )


def _preset_path(name: str, base: Path | None = None) -> Path:
    _validate_name(name)
    return (base or USER_PRESETS_DIR) / f"{name}.yaml"


def preset_path(name: str, base: Path | None = None) -> Path:
    """Public version of `_preset_path`, for end-to-end file downloads (server code shouldn't touch the private `_` helper)."""
    return _preset_path(name, base)


def parse_preset_bytes(raw: bytes, filename: str) -> tuple[dict[str, Any], str]:
    """Parses .yaml/.yml/.json upload content + pydantic validation, returns (config_dict, suggested_name).

    Doesn't write to disk -- the caller decides the final saved name (the frontend's confirm flow lets the user rename before saving).
    yaml.safe_load is a superset of JSON, so .json files can be consumed directly too.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PresetError(
            "Preset file is not valid UTF-8",
            code="preset.invalid",
            details={"reason": str(exc)},
            http_status=400,
        ) from exc
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise PresetError(
            f"Preset file could not be parsed: {exc}",
            code="preset.invalid",
            details={"reason": str(exc)},
            http_status=400,
        ) from exc
    if not isinstance(data, dict):
        raise PresetError(
            "Preset file format is invalid",
            code="preset.invalid",
            http_status=400,
        )
    cfg, _, _ = _tolerant_validate(data)
    stem = re.sub(r"\.(ya?ml|json)$", "", filename, flags=re.I)
    suggested = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-") or "imported"
    return _absolutize_model_paths(cfg.model_dump(mode="python")), suggested


def list_presets(base: Path | None = None) -> list[dict[str, Any]]:
    """Returns `[{name, path, updated_at}]`, sorted by modified time descending."""
    base = base or USER_PRESETS_DIR
    if not base.exists():
        return []
    items: list[dict[str, Any]] = []
    for p in base.glob("*.yaml"):
        items.append({
            "name": p.stem,
            "path": str(p),
            "updated_at": p.stat().st_mtime,
        })
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items


def _tolerant_validate(raw: dict[str, Any]) -> tuple[TrainingConfig, list[str], list[str]]:
    known = set(TrainingConfig.model_fields)
    data = dict(raw)

    if "attention_backend" not in data:
        if data.get("flash_attn") is True:
            data["attention_backend"] = "flash_attn"
        elif data.get("xformers") is True:
            data["attention_backend"] = "xformers"
        elif data.get("flash_attn") is False or data.get("xformers") is False:
            data["attention_backend"] = "none"
    data.pop("flash_attn", None)
    data.pop("xformers", None)
    # Retired monitor-server keys: old dumps all wrote these at some point; silently discard them without surfacing them in the dropped notice.
    for key in RETIRED_MONITOR_KEYS:
        data.pop(key, None)

    dropped = sorted(k for k in data if k not in known)
    data = {k: v for k, v in data.items() if k in known}
    try:
        return TrainingConfig.model_validate(data), dropped, []
    except ValidationError:
        pass

    defaults = TrainingConfig()
    defaulted: list[str] = []
    # disable_when rule fixup (blade 2 / R2 v2): a pinned/forbidden value that violates the rule gets fixed (fixed value =
    # disable_value, aligned with the frontend's takeover semantics; includes gate-first "turn off InfoNoise first to
    # preserve the user's investment in loss_weighting etc." -- a generalization of the historical InfoNoise-only shim).
    # Runs before the per-field fallback: a rule violation is a model-level error in pydantic (loc=()),
    # which the per-field fallback can't locate.
    data, rule_fixed = apply_disable_rule_fixes(data, TrainingConfig)
    defaulted.extend(rule_fixed)

    max_rounds = len(data) + 1
    for _ in range(max_rounds):
        try:
            cfg = TrainingConfig.model_validate(data)
            return cfg, dropped, sorted(defaulted)
        except ValidationError as exc:
            bad_fields = {
                e["loc"][0] for e in exc.errors() if e.get("loc")
            }
            if not bad_fields:
                # Remaining model-level errors = section 6.4's intentionally hand-kept validation (range / navit prerequisites),
                # with no declarative fix strategy -- reject directly per product semantics.
                raise PresetError(
                    f"Preset validation failed: {exc}",
                    code="preset.invalid",
                    details={"reason": str(exc)},
                    http_status=400,
                ) from exc
            for f in bad_fields:
                data[f] = getattr(defaults, f)
                defaulted.append(str(f))

    cfg = TrainingConfig.model_validate(data)
    return cfg, dropped, sorted(defaulted)


def read_preset(name: str, base: Path | None = None) -> dict[str, Any]:
    """Reads and tolerantly validates a preset. Unknown fields are dropped, invalid values fall back to defaults."""
    path = _preset_path(name, base)
    if not path.exists():
        raise PresetError(
            f'Preset "{name}" not found',
            code="preset.not_found",
            details={"name": name},
            http_status=404,
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise PresetError(
            "Preset file format is invalid",
            code="preset.invalid",
            details={"name": name},
            http_status=400,
        )
    cfg, _, _ = _tolerant_validate(raw)
    return _absolutize_model_paths(cfg.model_dump(mode="python"))


def read_preset_with_warnings(
    name: str, base: Path | None = None
) -> tuple[dict[str, Any], list[str], list[str]]:
    path = _preset_path(name, base)
    if not path.exists():
        raise PresetError(
            f'Preset "{name}" not found',
            code="preset.not_found",
            details={"name": name},
            http_status=404,
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise PresetError(
            "Preset file format is invalid",
            code="preset.invalid",
            details={"name": name},
            http_status=400,
        )
    cfg, dropped, defaulted = _tolerant_validate(raw)
    return _absolutize_model_paths(cfg.model_dump(mode="python")), dropped, defaulted


def render_config_yaml(dumped: dict[str, Any]) -> str:
    """Trimmed config dict -> yaml text. The single serialization exit point shared by
    disk writes (write_preset / write_version_config) and the preview endpoint (POST /api/schema/preview-yaml) --
    the basis for R4's "preview matches reality"; there is only this one place params are serialized."""
    return yaml.safe_dump(
        dumped, allow_unicode=True, sort_keys=False, default_flow_style=False
    )


def preview_config_yaml_text(raw: dict[str, Any]) -> str:
    """Current form config -> yaml text that exactly matches the file that would be written on save (R4).

    The tolerant semantics match saving: the fixed/trimmed shape IS "what the file looks like after clicking save".
    Pure computation, doesn't write to disk.
    """
    cfg, _, _ = _tolerant_validate(raw)
    return render_config_yaml(prune_inactive_fields(cfg.model_dump(mode="python")))


def write_preset(name: str, data: dict[str, Any], base: Path | None = None) -> Path:
    """Validates then writes to disk; any unknown field or type mismatch is rejected.

    Normalizes the 4 model fields before saving: relative path -> absolute (based on REPO_ROOT).
    Ensures the yaml is always written with absolute paths, avoiding a mix of old-format (relative) and new-format (absolute).
    """
    path = _preset_path(name, base)
    try:
        cfg = TrainingConfig.model_validate(data)
    except ValidationError as exc:
        raise PresetError(
            f"Preset validation failed: {exc}",
            code="preset.invalid",
            details={"reason": str(exc)},
            http_status=400,
        ) from exc
    # Fields whose show_when is false are trimmed before writing to disk (invisible in the UI = doesn't take effect); when read_preset runs,
    # pydantic fills missing fields back in with defaults, so the API still returns a complete config to the frontend.
    dumped = prune_inactive_fields(
        _absolutize_model_paths(cfg.model_dump(mode="python"))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_config_yaml(dumped), encoding="utf-8")
    return path


def delete_preset(name: str, base: Path | None = None) -> None:
    path = _preset_path(name, base)
    if not path.exists():
        raise PresetError(
            f'Preset "{name}" not found',
            code="preset.not_found",
            details={"name": name},
            http_status=404,
        )
    path.unlink()


def duplicate_preset(src: str, dst: str, base: Path | None = None) -> Path:
    src_path = _preset_path(src, base)
    dst_path = _preset_path(dst, base)
    if not src_path.exists():
        raise PresetError(
            f'Source preset "{src}" not found',
            code="preset.not_found",
            details={"name": src},
            http_status=404,
        )
    if dst_path.exists():
        raise PresetError(
            f'Preset "{dst}" already exists',
            code="preset.exists",
            details={"name": dst},
            http_status=409,
        )
    dst_path.write_bytes(src_path.read_bytes())
    return dst_path
