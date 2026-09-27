"""Version private config (PP6.2).

Each version has its own yaml training config, stored at
`studio_data/projects/{id}-{slug}/versions/{label}/config.yaml`.
This is **completely independent** from the global `studio_data/presets/{name}.yaml` -- when the user "switches preset",
it's copied in from the global one, and "save as preset" exports it back out the other way; edits to the private config never flow back into
the preset pool.

Schema validation reuses `TrainingConfig` (the same model as presets).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .presets.io import _absolutize_model_paths, _tolerant_validate
from ..domain.config_prune import prune_inactive_fields
from ..schema import TrainingConfig
from .projects.versions import version_dir
from .projects import projects as _projects


from studio.domain.errors import DomainError


class VersionConfigError(DomainError):
    """version private config I/O error.

    PR-2 C3 added a DomainError base -- the handler auto-translates it into the dual-write envelope.
    """
    default_code = "version_config.error"


CONFIG_FILENAME = "config.yaml"


# ---------------------------------------------------------------------------
# Project-specific fields (PP6 spec section on key conventions)
# ---------------------------------------------------------------------------

PROJECT_SPECIFIC_FIELDS: frozenset[str] = frozenset({
    "data_dir",
    "reg_data_dir",
    "output_dir",
    "output_name",
    "resume_lora",
    "resume_state",
    "trigger_word",
})

# Resume fields. Different from the rest of PROJECT_SPECIFIC_FIELDS: those are **derived** (uniquely determined by
# project + version, always recomputing to the same value), while these two are **explicitly set by the user**, and recomputing
# can only get None. They should be cleared on fork preset / copy from a source version (to prevent paths leaking across projects),
# but must be preserved on an idempotent write-back (syncing global model paths at enqueue time) -- otherwise the resume point the user just set would be
# wiped the instant they click "start training", with no error, and training would silently start from scratch.
RESUME_FIELDS: frozenset[str] = frozenset({
    "resume_lora",
    "resume_state",
})


def project_specific_overrides(
    project: dict[str, Any], version: dict[str, Any], *,
    reset_resume: bool = True,
) -> dict[str, Any]:
    """Computes project-specific field values from project + version.

    `data_dir` / `output_dir` / `output_name` are always deterministically filled in;
    `reg_data_dir` is only filled when the reg set exists (meta.json), otherwise empty (letting the trainer use its default).
    `resume_lora` / `resume_state` default to empty -- the user explicitly PUTs to change them when resuming training;
    when `reset_resume=False`, these two keys aren't returned, so the caller keeps the current value in config.
    `trigger_word` comes from the version table (written by Step 4 Tagging), keeping the yaml and caption
    in sync; the runtime's bootstrap_phase injects the trigger into sample_prompt based on it.
    """
    pid = int(project["id"])
    slug = str(project["slug"])
    label = str(version["label"])
    vdir = version_dir(pid, slug, label)
    overrides: dict[str, Any] = {
        "data_dir": str(vdir / "train"),
        "output_dir": str(vdir / "output"),
        "output_name": f"{slug}_{label}",
        "resume_lora": None,
        "resume_state": None,
        "trigger_word": str(version.get("trigger_word") or ""),
    }
    reg_meta = vdir / "reg" / "meta.json"
    if reg_meta.exists():
        overrides["reg_data_dir"] = str(vdir / "reg")
    else:
        overrides["reg_data_dir"] = None
    if not reset_resume:
        for f in RESUME_FIELDS:
            overrides.pop(f, None)
    return overrides


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------


def version_config_path(project: dict[str, Any], version: dict[str, Any]) -> Path:
    pid = int(project["id"])
    slug = str(project["slug"])
    label = str(version["label"])
    return version_dir(pid, slug, label) / CONFIG_FILENAME


def has_version_config(project: dict[str, Any], version: dict[str, Any]) -> bool:
    return version_config_path(project, version).exists()


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def read_version_config(
    project: dict[str, Any], version: dict[str, Any]
) -> dict[str, Any]:
    """Reads the version's private config; raises VersionConfigError if it doesn't exist."""
    cfg, _, _ = read_version_config_with_warnings(project, version)
    return cfg


def read_version_config_with_warnings(
    project: dict[str, Any], version: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Reads the version's private config, also returning the (dropped, defaulted) field lists produced by tolerant validation.

    Used by the GET endpoint to pass compat info through to the frontend (the top banner hint). When InfoNoise's old config
    mutual-exclusion is auto-turned-off by _tolerant_validate, "infonoise_enabled" shows up in
    defaulted.
    """
    p = version_config_path(project, version)
    if not p.exists():
        raise VersionConfigError(
            "Training configuration is not set for this version",
            code="version.config_missing",
        )
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise VersionConfigError(
            "Training configuration is invalid",
            code="version.config_invalid",
        )
    cfg, dropped, defaulted = _tolerant_validate(raw)
    # overlay runs before absolutize -- override values and yaml values go through the same path-normalization pipeline
    data = apply_global_path_overlay(cfg.model_dump(mode="python"))
    return _absolutize_model_paths(data), dropped, defaulted


#: The 4 model path fields managed by the global settings when auto_sync_paths=ON (the Train page's corresponding
#: fields are locked and marked "Auto - global settings" -- this read-path overlay is what makes that badge true:
#: after the global selected / selected_te changes, an existing version's display and derived chain
#: (training enqueue / reg / eval) follow immediately, instead of staying frozen at the creation-time snapshot).
GLOBAL_MODEL_PATH_FIELDS = (
    "transformer_path", "vae_path", "text_encoder_path", "t5_tokenizer_path",
)


def apply_global_path_overlay(data: dict[str, Any]) -> dict[str, Any]:
    """Overrides the 4 model path fields with the current global settings when auto_sync_paths=ON.

    The same read-path semantics as the "override on write" used by fork / save_preset / bundle import;
    when OFF (a standalone-model user), returns as-is. Fails silently, returning the original value (a read shouldn't fail
    just because secrets / catalog raised an exception).
    """
    try:
        from . import models as model_downloader
        from .presets import _auto_sync_paths

        if not _auto_sync_paths():
            return data
        family = str(data.get("model_family") or "anima")
        data.update(model_downloader.default_paths_for_new_version(family=family))
    except Exception:
        pass
    return data


def write_version_config(
    project: dict[str, Any], version: dict[str, Any], data: dict[str, Any],
    *, force_project_overrides: bool = True, reset_resume: bool = True,
) -> Path:
    """Writes the version's private config.

    `force_project_overrides=True` (default): uses `project_specific_overrides` to
    forcibly override PROJECT_SPECIFIC_FIELDS, preventing the user from bypassing the frontend's disabled state to change paths.

    `reset_resume=False`: path fields are still force-refreshed as usual, but RESUME_FIELDS keeps the
    current value from `data` -- used for the idempotent "read it back out then write it back in" sync (see enqueue_version_training).
    """
    payload = dict(data)
    if force_project_overrides:
        payload.update(
            project_specific_overrides(project, version, reset_resume=reset_resume)
        )
    cfg, _, _ = _tolerant_validate(payload)
    # Fields whose show_when is false are trimmed before writing to disk (invisible in the UI = doesn't take effect); on read, pydantic
    # fills missing fields back in with the schema defaults, so the API still returns a complete config to the frontend.
    dumped = prune_inactive_fields(cfg.model_dump(mode="python"))
    p = version_config_path(project, version)
    p.parent.mkdir(parents=True, exist_ok=True)
    # The serialization exit point is unified as render_config_yaml -- the preview endpoint and disk writes share the same path (R4)
    from .presets.io import render_config_yaml

    p.write_text(render_config_yaml(dumped), encoding="utf-8")
    return p


def delete_version_config(
    project: dict[str, Any], version: dict[str, Any]
) -> bool:
    """Deletes the version's private config. Returns True if it was deleted, False if there was none to begin with."""
    p = version_config_path(project, version)
    if p.exists():
        p.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_project_and_version(
    conn, project_id: int, version_id: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convenience: reads project + version from the db; raises VersionConfigError if the version doesn't belong to this project."""
    from ..services.projects import versions as _versions
    p = _projects.get_project(conn, project_id)
    if not p:
        raise VersionConfigError(
            "Project not found", code="project.not_found",
            details={"id": project_id}, http_status=404,
        )
    v = _versions.get_version(conn, version_id)
    if not v or v["project_id"] != project_id:
        raise VersionConfigError(
            "Version not found", code="version.not_found",
            details={"id": version_id}, http_status=404,
        )
    return p, v
