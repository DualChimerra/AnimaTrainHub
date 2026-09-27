"""Task log reads (extracted from server.py in PR-6 commit 1).

1 route:
    GET /api/logs/{task_id}    reads tasks/<id>/run.log (falls back to
                               LOGS_DIR/<id>.log for old tasks), stripping
                               worker EVENT lines
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ...paths import LOGS_DIR, task_log_path

router = APIRouter()


def read_task_log(task_id: int) -> str:
    """Full task log text (with __EVENT__: protocol lines stripped out); returns "" if the file doesn't exist.

    New tasks use tasks/<id>/run.log; old tasks live at
    studio_data/logs/<id>.log — there's no migration script (the DB doesn't
    record a log path either), so we just fall back based on which one
    exists. Other routers (training.py's reg-prior replay endpoint) reuse
    this same helper.
    """
    p = task_log_path(task_id)
    if not p.exists():
        p = LOGS_DIR / f"{task_id}.log"
    if not p.exists():
        return ""
    raw = p.read_text(encoding="utf-8", errors="replace")
    return "".join(
        ln for ln in raw.splitlines(keepends=True)
        if not ln.startswith("__EVENT__:")
    )


@router.get("/api/logs/{task_id}")
def get_log(task_id: int) -> dict[str, Any]:
    text = read_task_log(task_id)
    return {"task_id": task_id, "content": text, "size": len(text.encode("utf-8"))}
