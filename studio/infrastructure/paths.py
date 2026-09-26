"""Path constants and directory initialization used internally by Studio."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

# PR-7: this file moved from studio/paths.py to studio/infrastructure/paths.py, one extra
# level of nesting; REPO_ROOT needs to go up one more level (__file__ -> infrastructure/ ->
# studio/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# Already exists on the training side. `OUTPUT_DIR` is only a fallback for the `/samples/{name}`
# endpoint (when there's no task_id); CLI users get `./output/samples/...`, while Studio-mode
# samples land in `studio_data/projects/{id}-{slug}/versions/{label}/output/samples/`.
# After PP6.1 the global `monitor_data/` was retired (monitoring state is per-task now), so the
# constant / directory creation for it is no longer kept.
OUTPUT_DIR = REPO_ROOT / "output"

# Landing directory when the user explicitly chooses "export to a local directory".
DATA_EXPORTS = REPO_ROOT / "data_exports"


# Studio persistence (SQLite + user-saved presets + task logs).
#
# The location is customizable: a pointer file at the repo root, `studio_data_location.json`
# ({"path": "..."}), points to a custom directory. The pointer must live **outside**
# studio_data -- secrets.json / the db both live inside studio_data, so storing the location's
# own config inside it would be a chicken-and-egg problem. The pointer is read once at module
# import time and stays fixed for the process; after a migration (/api/studio-data/migrate)
# writes the pointer, the server needs a restart to pick it up (cli.py's restart loop spawns a
# new process via subprocess, and paths gets re-evaluated).
DEFAULT_STUDIO_DATA = REPO_ROOT / "studio_data"
STUDIO_DATA_POINTER = REPO_ROOT / "studio_data_location.json"


def resolve_studio_data(pointer_file: Path | None = None) -> Path:
    """Resolves the studio_data location: env override -> pointer file -> default location.

    This fork: the `ALS_STUDIO_DATA` env var (an absolute path) has the highest priority --
    injected by cloud (Colab / Kaggle) notebooks, to put the working directory on the local
    fast disk and sync to Google Drive separately (symlinking studio_data directly onto the
    Drive FUSE mount makes SQLite and small-file writes unreliable: lost projects/presets,
    corrupted zips, etc -- see the docs / Colab notebook comments for details).

    Pointer file (the upstream /api/studio-data/migrate mechanism): the target must be an
    existing absolute-path directory -- if the drive letter isn't mounted / the directory was
    deleted, falls back to the default location (which still retains the old pre-migration
    data and remains usable), only logging a warning rather than raising.
    """
    env = os.environ.get("ALS_STUDIO_DATA", "").strip()
    if env:
        return Path(env).expanduser()
    ptr = pointer_file if pointer_file is not None else STUDIO_DATA_POINTER
    try:
        if ptr.is_file():
            raw = json.loads(ptr.read_text("utf-8"))
            target = Path(str(raw.get("path", "")))
            if target.is_absolute() and target.is_dir():
                return target
            logging.getLogger(__name__).warning(
                "studio_data pointer target is invalid (missing or not an absolute path), falling back to the default location: %s", target,
            )
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to parse the studio_data pointer file, falling back to the default location: %s", ptr, exc_info=True,
        )
    return DEFAULT_STUDIO_DATA


STUDIO_DATA = resolve_studio_data()
STUDIO_DB = STUDIO_DATA / "studio.db"
USER_PRESETS_DIR = STUDIO_DATA / "presets"
USER_CONFIGS_DIR = USER_PRESETS_DIR  # compat alias (to be removed along with configs_io after PP0)
# LOGS_DIR: the flat directory holding all task logs from the pre-task-scoped layout.
# New tasks use `tasks/<id>/run.log` (see task_log_path); this directory is kept only for
# reading old tasks (nothing new is written here).
LOGS_DIR = STUDIO_DATA / "logs"
THUMB_CACHE_DIR = STUDIO_DATA / "thumb_cache"

# Root of the task-scoped archive. Each task gets its own subdirectory, decoupled from version,
# so deleting a version doesn't take the task's history (loss / params / samples / logs) with it.
# Subdirectory conventions (snapshot/ was introduced by task_snapshot.py, ADR-0007 SS11.7):
#   tasks/<id>/snapshot/config.yaml   <- config frozen when the task was enqueued
#   tasks/<id>/monitor/state.json     <- training monitor state (loss/LR/sample index)
#   tasks/<id>/samples/*.png          <- training sample images
#   tasks/<id>/run.log                <- worker subprocess stdout/stderr
TASKS_DIR = STUDIO_DATA / "tasks"

# React frontend
WEB_DIR = REPO_ROOT / "studio" / "web"
WEB_DIST = WEB_DIR / "dist"

# Announcement board data source (docs/announcements/<id>.md + <id>.en.md), see
# docs/todo/announcement-center.md. Lives under docs/ in the repo, shipped via git along with
# each release.
ANNOUNCEMENTS_DIR = REPO_ROOT / "docs" / "announcements"


def migrate_configs_to_presets() -> None:
    """Older versions put yaml files in studio_data/configs/; this renames that directory to
    presets/ in place. Only runs when presets/ doesn't already exist; never overwrites the
    user's newer data."""
    old = STUDIO_DATA / "configs"
    if old.exists() and not USER_PRESETS_DIR.exists():
        old.rename(USER_PRESETS_DIR)


def ensure_dirs() -> None:
    """Creates the required directories on first run."""
    STUDIO_DATA.mkdir(parents=True, exist_ok=True)
    migrate_configs_to_presets()
    for d in (USER_PRESETS_DIR, LOGS_DIR, DATA_EXPORTS):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Task-scoped path helpers
# ---------------------------------------------------------------------------
#
# None of these helpers mkdir -- callers mkdir(parents=True, exist_ok=True) as needed.
# Kept consistent with the existing convention in task_snapshot.snapshot_dir: snapshot_dir(id)
# is also just a subdirectory of `tasks/<id>/snapshot/`; this group of helpers provides the
# three sibling dirs monitor/samples/log, which together make up a task's full archive.

def task_dir(task_id: int) -> Path:
    """`studio_data/tasks/<task_id>/` -- root of the task's archive."""
    return TASKS_DIR / str(int(task_id))


def task_monitor_state_path(task_id: int) -> Path:
    """`tasks/<task_id>/monitor/state.json` -- the training monitor state file.

    Replaces the old paths:
    - `versions/<v>/monitor/task_<id>/state.json` (PP6.1, v0.5.0+, still read for compat)
    - `versions/<v>/monitor_state.json` (pre-PP6.1, still read for compat)
    - `studio_data/monitors/task_<id>/state.json` (fallback for no version_id, still read for compat)
    """
    return task_dir(task_id) / "monitor" / "state.json"


def task_samples_dir(task_id: int) -> Path:
    """`tasks/<task_id>/samples/` -- training sample images.

    Written by runtime: `runtime/training/phases/bootstrap.py` points ctx.sample_dir here.
    Read by the API: first candidate in `studio/api/routers/samples.py`.
    """
    return task_dir(task_id) / "samples"


def task_eval_dir(task_id: int) -> Path:
    """`tasks/<task_id>/eval/` -- training task scoped LoRA eval artifacts."""
    return task_dir(task_id) / "eval"


def task_log_path(task_id: int) -> Path:
    """`tasks/<task_id>/run.log` -- worker subprocess stdout/stderr.

    Replaces the old path `studio_data/logs/<task_id>.log` (still read for compat).
    `project_jobs`' logs (`studio_data/jobs/<job_id>.log`) are a separate system, untouched here.
    """
    return task_dir(task_id) / "run.log"


# ---------------------------------------------------------------------------
# Path traversal protection
# ---------------------------------------------------------------------------
#
# Lesson learned: early endpoints only used a literal blacklist
# (`"/" in name or "\\" in name or ".." in name`) to guard against traversal. A literal `..`
# check false-positives on legitimate filenames containing an ASCII ellipsis (a Pixiv title like
# `"kore..."`), but removing it would make defense-in-depth too thin. This is unified around a
# resolve() + relative_to() containment check instead: it both allows `..` to appear as ordinary
# filename characters and guarantees the final joined path can never escape base.

def validate_path_component(name: str) -> None:
    """Validates a single path component: rejects empty strings / path separators / an
    absolute-path prefix.

    A literal `..` is allowed through on its own (the containment check is the real defense),
    so a legitimate filename containing an ellipsis, like `"cafe..."`.txt`, can still pass.
    """
    if not name:
        raise ValueError("path component is empty")
    if "/" in name or "\\" in name:
        raise ValueError(f"path component contains separator: {name!r}")
    # A Windows drive letter (C:\...) or a POSIX absolute path (/foo) is always treated as an
    # invalid component
    if Path(name).is_absolute():
        raise ValueError(f"path component is absolute: {name!r}")


def safe_join(base: Path, *parts: str) -> Path:
    """Joins parts under base and runs a containment check.

    Each part goes through `validate_path_component`; after joining + resolve(), the result
    must still be within `base.resolve()`'s subtree, otherwise raises ValueError.

    Returns the resolved absolute Path. The caller does exists() / is_file() checks as needed.
    """
    for p in parts:
        validate_path_component(p)
    base_resolved = base.resolve()
    candidate = base_resolved.joinpath(*parts).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes base: {candidate} not in {base_resolved}") from exc
    return candidate
