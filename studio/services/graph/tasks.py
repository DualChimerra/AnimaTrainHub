"""Read-only bridge from the Graph to the training queue.

The Graph never changes queue state. It only reads which training tasks
exist, the config each one was frozen with, and its sample images — enough to
recognise a task that matches a card or an empty slot and to copy samples
into a card.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from ... import db
from ...infrastructure.paths import task_samples_dir
from .. import task_snapshot

logger = logging.getLogger(__name__)

#: (task_id) -> (mtime_ns, parsed config). Snapshots never change once frozen
#: except on a restart of the same task, which the mtime check catches.
_CFG_CACHE: dict[int, tuple[int, dict[str, Any]]] = {}

_MAX_TASKS = 1000

#: (task_id) -> (mtime_ns, total_steps) read from the monitor state.
_TOTAL_CACHE: dict[int, tuple[int, Optional[int]]] = {}
_TOTAL_RE = re.compile(rb'"total_steps"\s*:\s*(\d+)')


def total_steps(task_id: int, monitor_state_path: Optional[str]) -> Optional[int]:
    """Total optimizer steps the trainer planned, from the monitor state.

    Epoch-based runs keep ``max_steps: 0`` in their config, so this is the
    only place the real total lives. The state file can be megabytes of loss
    history, so the number is picked out with a regex instead of parsing it.
    """
    if not monitor_state_path:
        return None
    p = Path(monitor_state_path)
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        return None
    hit = _TOTAL_CACHE.get(task_id)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        m = _TOTAL_RE.search(p.read_bytes())
    except OSError:
        return None
    val = int(m.group(1)) if m else None
    _TOTAL_CACHE[task_id] = (mtime, val)
    return val


def task_config(task_id: int, config_path: Optional[str] = None) -> Optional[dict[str, Any]]:
    p = task_snapshot.snapshot_config_path(task_id)
    if not p.exists() and config_path:
        p = Path(config_path)
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        return None
    hit = _CFG_CACHE.get(task_id)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        parsed = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - a broken yaml just means "unknown config"
        logger.info("graph: unreadable config for task %s", task_id, exc_info=True)
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    _CFG_CACHE[task_id] = (mtime, parsed)
    return parsed


def list_train_tasks(keys: list[str]) -> list[dict[str, Any]]:
    """Every training task, newest first, with only the config keys asked for."""
    with db.connection_for() as conn:
        rows = conn.execute(
            "SELECT t.id, t.name, t.status, t.created_at, t.started_at, t.finished_at, "
            "t.project_id, t.version_id, t.config_path, t.monitor_state_path, t.note, "
            "p.title AS project_title, v.label AS version_label "
            "FROM tasks t "
            "LEFT JOIN projects p ON p.id = t.project_id "
            "LEFT JOIN versions v ON v.id = t.version_id "
            "WHERE COALESCE(t.task_type, 'train') = 'train' "
            "ORDER BY t.id DESC LIMIT ?",
            (_MAX_TASKS,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        cfg = task_config(int(r["id"]), r["config_path"]) or {}
        subset = {k: cfg[k] for k in keys if k in cfg}
        if "max_steps" in keys and not cfg.get("max_steps"):
            total = total_steps(int(r["id"]), r["monitor_state_path"])
            if total:
                subset["max_steps"] = total
        out.append({
            "id": r["id"],
            "name": r["name"],
            "status": r["status"],
            "created_at": r["created_at"],
            "started_at": r["started_at"],
            "finished_at": r["finished_at"],
            "project_id": r["project_id"],
            "version_id": r["version_id"],
            "project_title": r["project_title"],
            "version_label": r["version_label"],
            "note": r["note"] or "",
            "has_config": bool(cfg),
            "config": subset,
        })
    return out


def task_row(task_id: int) -> Optional[dict[str, Any]]:
    with db.connection_for() as conn:
        row = conn.execute(
            "SELECT id, config_path, monitor_state_path FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    return dict(row) if row else None


def sample_file(task_id: int, filename: str) -> Optional[Path]:
    """Resolve a task's sample by name through the same directories the
    sample proxy searches."""
    from ...api.routers.samples import _sample_dirs

    row = task_row(task_id)
    if not row:
        return None
    dirs = (_sample_dirs(row["monitor_state_path"], task_id)
            if row["monitor_state_path"] else [task_samples_dir(task_id)])
    for d in dirs:
        p = d / filename
        if p.is_file():
            return p
    return None


def monitor_summary(task_id: int) -> dict[str, Any]:
    """total_steps / total_epochs / sample records from the monitor state
    (only what sample steps need; the loss history is dropped)."""
    import json

    row = task_row(task_id)
    path = row and row.get("monitor_state_path")
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: data.get(k) for k in ("total_steps", "total_epochs", "samples")}
