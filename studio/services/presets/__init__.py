"""Preset bidirectional flow (PP6.2).

`fork_preset_for_version` -- copies the global preset into the version's private config, immediately
applying project-specific fields (data_dir / output_dir / output_name etc.).

`save_version_config_as_preset` -- exports the version's private config back out to the global
preset pool; project-specific fields are reset to schema defaults (no project data leaks out).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import io as presets_io
from ... import secrets
from ...schema import TrainingConfig
from .. import models as model_downloader
from .. import version_config


def _auto_sync_paths() -> bool:
    """Reads settings.models.auto_sync_paths (default ON).

    ON  -> fork overwrites the 4 model fields with the global Settings values; on "save as preset" these 4 fields
          are reset to the Settings defaults (no locally customized paths leak out).
    OFF -> fork respects the preset's values; "save as preset" keeps the preset's absolute paths as-is.
    """
    try:
        return bool(secrets.load().models.auto_sync_paths)
    except Exception:
        return True


def fork_preset_for_version(
    src_preset_name: str,
    project: dict[str, Any],
    version: dict[str, Any],
) -> dict[str, Any]:
    return fork_preset_for_version_with_warnings(
        src_preset_name, project, version
    )[0]


def fork_preset_for_version_with_warnings(
    src_preset_name: str,
    project: dict[str, Any],
    version: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Copies a global preset into the version's private config, returning compatibility warnings."""
    src, dropped, defaulted = presets_io.read_preset_with_warnings(src_preset_name)
    new_data = deepcopy(src)
    new_data.update(version_config.project_specific_overrides(project, version))
    if _auto_sync_paths():
        # Takes the path per the family declared in the preset -- overwriting with a krea2 preset's paths on an anima config would corrupt it
        family = str(new_data.get("model_family") or "anima")
        new_data.update(model_downloader.default_paths_for_new_version(family=family))
    version_config.write_version_config(
        project, version, new_data, force_project_overrides=True
    )
    return version_config.read_version_config(project, version), dropped, defaulted


def save_version_config_as_preset(
    project: dict[str, Any],
    version: dict[str, Any],
    target_preset_name: str,
    *, overwrite: bool = False,
) -> dict[str, Any]:
    """version private config -> global preset.

    1. Read the version's private config
    2. Reset project-specific fields to TrainingConfig defaults (no project data leaks out)
    3. **Optionally** reset the 4 model fields to the current Settings defaults (controlled by a toggle):
       - toggle ON: reset to the current Settings absolute values computed by `default_paths_for_new_version()`,
         avoiding leaking "my machine's custom paths" into the preset pool.
       - toggle OFF: keep the absolute paths from the version yaml as-is (values the standalone-model user set deliberately).
    4. Write `presets/{target_preset_name}.yaml`
    Returns the preset dict as finally written to disk.
    """
    src = version_config.read_version_config(project, version)
    cleaned = deepcopy(src)
    defaults = TrainingConfig().model_dump()
    for f in version_config.PROJECT_SPECIFIC_FIELDS:
        cleaned[f] = defaults.get(f)
    if _auto_sync_paths():
        family = str(cleaned.get("model_family") or "anima")
        cleaned.update(model_downloader.default_paths_for_new_version(family=family))

    target_path = presets_io._preset_path(target_preset_name)  # validates the name is legal
    if target_path.exists() and not overwrite:
        raise presets_io.PresetError(
            f'Preset "{target_preset_name}" already exists',
            code="preset.exists",
            details={"name": target_preset_name},
            http_status=409,
        )
    presets_io.write_preset(target_preset_name, cleaned)
    return presets_io.read_preset(target_preset_name)
