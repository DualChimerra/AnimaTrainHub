"""task terminal state -> version.status mapping (extracted from supervisor.py in PR-4).

ADR-0007 §11.3-B: task terminal states are mapped independently, no lying about status.
The three task_status values done/failed/canceled each map to their VersionStatus;
paused doesn't go through this function (§11.3-A: while task=paused the version stays
"training", the UI derives its own display state).
"""
from __future__ import annotations

from typing import Any

from .. import db


def _maybe_finalize_version(
    conn: Any, task_id: int, task_status: str = "done"
) -> None:
    """task terminal state -> push version.status (ADR-0007 §11.3-B).

    task_status mapping:
    - done -> completed (+ backfill output_lora_path)
    - failed -> failed (+ write last_failure_reason from task.error_msg)
    - canceled -> canceled

    paused doesn't go through this function (§11.3-A: while task=paused the version stays
    "training", the UI derives its own display state).
    """
    from ..services.projects import versions as _versions
    task_row = db.get_task(conn, task_id)
    if not task_row:
        return
    vid = task_row.get("version_id")
    pid = task_row.get("project_id")
    if not (vid and pid):
        return
    v = _versions.get_version(conn, int(vid))
    if not v:
        return
    from ..services.projects import projects as _projects
    p = _projects.get_project(conn, int(pid))
    if not p:
        return

    # ADR-0007 §11.3-B: task terminal states are mapped independently, no lying about status
    new_status_map = {
        "done":     _versions.VersionStatus.COMPLETED,
        "failed":   _versions.VersionStatus.FAILED,
        "canceled": _versions.VersionStatus.CANCELED,
    }
    new_status = new_status_map.get(task_status)
    if new_status is None:
        return  # unknown task_status (e.g. paused / running) -- leave version untouched

    fields: dict[str, Any] = {"status": new_status}

    if task_status == "done":
        output_name = f"{p['slug']}_{v['label']}"
        vdir = _versions.version_dir(int(pid), p["slug"], v["label"])
        candidate = vdir / "output" / f"{output_name}_final.safetensors"
        if candidate.exists():
            fields["output_lora_path"] = str(candidate)
    elif task_status == "failed":
        err = task_row.get("error_msg")
        if err:
            fields["last_failure_reason"] = str(err)

    _versions.update_version(conn, int(vid), **fields)
