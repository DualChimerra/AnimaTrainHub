"""Queue task snapshot (extracted from server.py in PR-6 commit 6).

Routes:
    GET /api/queue/{task_id}/snapshot/config    ADR-0007 §11.7: the config frozen when the task was enqueued
    GET /api/queue/{task_id}/snapshot/config/file    full training config download

History: this file used to have /api/queue/export + /api/queue/import (queue JSON
sharing from the preset-pool era). Removed in R-5 — modern task configs are
private to a version, so export was always empty and import always a no-op;
the UI button was already removed beforehand (PR #359 feedback round).
Project-level sharing is now covered by project bundle import/export.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response

from .... import db
from ....domain.errors import NotFoundError
from ....services import task_snapshot
from ....services.graph import tasks as graph_tasks

router = APIRouter()


@router.get("/api/queue/{task_id}/snapshot/config")
def get_task_snapshot_config(task_id: int) -> dict[str, Any]:
    """ADR-0007 §11.7: return the config frozen when the task was enqueued.

    Returns ``{"yaml": str, "config": dict}``. Task doesn't exist / no
    snapshot → 404. Used by the UI's [Linked Config] tab.
    """
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise NotFoundError("Task not found", code="task.not_found", details={"task_id": task_id})
    data = task_snapshot.read_snapshot_config(task_id)
    if data is None:
        raise NotFoundError(
            "No saved configuration for this task",
            code="task.snapshot_not_found", details={"task_id": task_id},
        )
    return data


@router.get("/api/queue/{task_id}/snapshot/config/file")
def download_task_config(task_id: int) -> Response:
    """Download the full config using the same source and format as the Graph."""
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise NotFoundError("Task not found", code="task.not_found", details={"task_id": task_id})
    text = graph_tasks.task_config_yaml(task_id, task.get("config_path"))
    if text is None:
        raise NotFoundError(
            "No saved configuration for this task",
            code="task.snapshot_not_found", details={"task_id": task_id},
        )
    return Response(
        content=text, media_type="application/x-yaml",
        headers={"Content-Disposition": f'attachment; filename="task_{task_id}.yaml"'},
    )
