"""Health check / system status / training monitor state reads (extracted from server.py in PR-5).

3 routes:
    GET /api/health         health check (includes app version)
    GET /api/system/stats   topbar system resources (CPU / RAM / GPU / VRAM) for cold start
    GET /api/state          per-task monitor_state.json; the monitoring page fetches this once on
                            cold start, then gets incremental updates via SSE
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import responses as _responses
from ... import db
from ...services import system_stats

router = APIRouter()


_TASK_STATE_SELECT = """
    SELECT
        t.monitor_state_path,
        t.id AS task_id,
        t.project_id,
        t.version_id,
        p.slug AS project_slug,
        v.label AS version_label
    FROM tasks t
    LEFT JOIN projects p ON p.id = t.project_id
    LEFT JOIN versions v ON v.id = t.version_id
"""


@router.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    return {"status": "ok", "version": request.app.version}


@router.get("/api/system/stats")
def get_system_stats() -> dict[str, Any]:
    """For the topbar system-resources widget (CPU/RAM/GPU/VRAM). The frontend polls every 2-3s."""
    return system_stats.stats_to_json(system_stats.collect_stats())


@router.get("/api/state")
def get_state(task_id: Optional[int] = None, max_points: int = 0) -> JSONResponse:
    """Read the training monitor's state.json (PP6.1 rework — per-task).

    If `task_id` is given → look up the file at tasks.monitor_state_path; if
    missing / the file doesn't exist → return EMPTY_STATE, no error.
    If `task_id` is not given → prefer the running task; if none is running,
    fall back to the **most recent** task (done / failed / canceled) that has
    a monitor_state_path, so the monitoring page can still show the last
    training run's curves after it finishes. If there's nothing at all,
    return EMPTY_STATE.

    `max_points` (default 0 = no downsampling) — PR #37 originally defaulted
    this to 1000; this PR changes the default to 0: cold start is a one-off
    HTTP call, and users routinely have 10k+ step runs, so it's worth a
    bigger payload to give them the full history. Callers who want
    downsampling can pass max_points=N explicitly (kept around for future
    lightweight use cases like thumbnails / previews).

    The old global `monitor_data/state.json` path has been retired (PP6.1).
    """
    target_path: Optional[Path] = None
    target_task: dict[str, Any] | None = None
    if task_id is not None:
        with db.connection_for() as conn:
            row = conn.execute(
                _TASK_STATE_SELECT + " WHERE t.id = ?",
                (task_id,),
            ).fetchone()
        if row and row["monitor_state_path"]:
            target_path = Path(row["monitor_state_path"])
            target_task = dict(row)
    else:
        # No task_id given: look for a running task first; if none, fall back to the most recently finished task
        with db.connection_for() as conn:
            row = conn.execute(
                _TASK_STATE_SELECT +
                " WHERE t.status = 'running' AND t.monitor_state_path IS NOT NULL "
                "ORDER BY t.started_at DESC LIMIT 1"
            ).fetchone()
            if not (row and row["monitor_state_path"]):
                row = conn.execute(
                    _TASK_STATE_SELECT +
                    " WHERE t.monitor_state_path IS NOT NULL "
                    "ORDER BY COALESCE(t.finished_at, t.started_at, t.created_at) DESC "
                    "LIMIT 1"
                ).fetchone()
        if row and row["monitor_state_path"]:
            target_path = Path(row["monitor_state_path"])
            target_task = dict(row)

    if target_path is None or not target_path.exists():
        return JSONResponse(_responses.EMPTY_STATE)
    try:
        data = json.loads(target_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"failed to read state: {exc}")

    # Server-side downsampling of losses / lr_history (the samples cap of 50 is already handled on the train_monitor side)
    if max_points and max_points > 0:
        from runtime.train_monitor import _downsample_uniform
        if isinstance(data.get("losses"), list):
            data["losses"] = _downsample_uniform(data["losses"], max_points)
        if isinstance(data.get("lr_history"), list):
            data["lr_history"] = _downsample_uniform(data["lr_history"], max_points)
        if isinstance(data.get("optimizer_metrics_history"), list):
            data["optimizer_metrics_history"] = _downsample_uniform(data["optimizer_metrics_history"], max_points)

    _attach_eval_context(data, target_task)
    return JSONResponse(data)


def _attach_eval_context(data: Any, task: dict[str, Any] | None) -> None:
    if not isinstance(data, dict) or not task:
        return
    project_id = task.get("project_id")
    version_id = task.get("version_id")
    if project_id is None or version_id is None:
        return
    data.setdefault("task_id", task.get("task_id"))
    data.setdefault("project_id", project_id)
    data.setdefault("version_id", version_id)
    if task.get("project_slug") is not None:
        data.setdefault("project_slug", task.get("project_slug"))
    if task.get("version_label") is not None:
        data.setdefault("version_label", task.get("version_label"))
