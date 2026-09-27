"""Task config snapshot -- ADR-0007 section 11.7.

When a task is enqueued, the version config.yaml at that moment is frozen into
``studio_data/tasks/{task_id}/snapshot/config.yaml``.

Design notes:
- **Only freezes config**, not caption / images / reg set (cross-OS export is fine, disk cost is KB-scale)
- Mental separation in the UI: the task detail page has its own [Linked config] tab, **clicking a task never jumps to the version config edit page**
  -> so the user understands config is a historical snapshot, while caption / images are the version's current state
- Freeze timing: when a new task is enqueued; an old task still gets a freeze backfilled when the supervisor starts it
- A new task that fails to freeze doesn't get enqueued; backfilling the freeze for an old task at startup is still non-blocking

From the user's perspective: "click a task's detail [Linked config] to see what params it ran with then, and hit
'Apply this config' to jump to step 7 (training phase) with a prefill -> edit -> train = a new task" (the section 11.7 flow).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

import yaml

from ..paths import task_dir

SNAPSHOT_CONFIG_FILENAME = "config.yaml"


def snapshot_dir(task_id: int) -> Path:
    """``studio_data/tasks/{task_id}/snapshot/``.

    A sibling of monitor/ samples/ run.log; together they make up the task's complete archive.
    The path is derived from `paths.task_dir`, from the same source as the other task-scoped helpers;
    tests only need to monkeypatch `paths.TASKS_DIR` once to isolate all of it.
    """
    return task_dir(task_id) / "snapshot"


def snapshot_config_path(task_id: int) -> Path:
    return snapshot_dir(task_id) / SNAPSHOT_CONFIG_FILENAME


def has_snapshot(task_id: int) -> bool:
    return snapshot_config_path(task_id).exists()


def freeze_config(task_id: int, source: Path) -> Path:
    """Copies the source yaml to ``snapshot_config_path(task_id)``, returning the destination path.

    Calling it again overwrites (the same-task_id restart scenario). Raises FileNotFoundError if source doesn't exist.
    """
    if not source.exists():
        raise FileNotFoundError(f"snapshot source not found: {source}")
    dst = snapshot_config_path(task_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # When a new task is enqueued, config_path already points at this snapshot. The supervisor
    # still calls this function at startup for compat with old tasks, in which case source == dst,
    # so it should just be reused directly rather than letting shutil.copy2 raise SameFileError.
    if dst.exists() and source.samefile(dst):
        return dst
    shutil.copy2(source, dst)
    return dst


def read_snapshot_config(task_id: int) -> Optional[dict[str, Any]]:
    """Reads the task config snapshot; returns None if it doesn't exist.

    Returns ``{"yaml": raw_text, "config": parsed_dict}`` -- so the UI can both show the raw yaml
    (read-only monaco) and prefill the training config form.
    """
    p = snapshot_config_path(task_id)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    return {"yaml": text, "config": parsed}
