"""Data job DAO -- since R-3, backed by the unified tasks ledger.

The interface keeps the shape it's had since pp2 (create_job / get_job / list_jobs / mark_* / ...),
so callers (workers / eval service / API routes / supervisor) need zero changes; underneath it all reads/writes
the tasks table (`task_type` = job kind, `params` JSON is the _v17 column).

The old `project_jobs` table is frozen as read-only legacy: _v18 marked leftover pending/running rows canceled;
Q-R2 decided not to display or migrate old rows (remapping IDs would corrupt eval run references and log paths).

Row-shape compat: the returned dict has these injected on top of the tasks row
    kind     = task_type
    log_path = tasks/<id>/run.log (paths.task_log_path, where worker stdout lands)
params_decoded is provided uniformly by db._row_to_dict (same convention as the old DAO).
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from ...infrastructure import db
from ...infrastructure.paths import STUDIO_DATA, task_log_path

# Old project_jobs's separate log directory (studio_data/jobs/<id>.log). Since R-3, new job logs
# go to tasks/<id>/run.log; this constant is only referenced by legacy read-only paths and old test fixtures.
JOB_LOGS_DIR = STUDIO_DATA / "jobs"

# Full kind set (authoritative source is db.JOB_TASK_TYPES; an order-preserving tuple for SQL IN, a frozenset for validation).
JOB_TASK_TYPES: tuple[str, ...] = db.JOB_TASK_TYPES
VALID_KINDS: frozenset[str] = frozenset(JOB_TASK_TYPES)
VALID_STATUSES: frozenset[str] = frozenset({
    "pending", "running", "done", "failed", "canceled"
})
TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "canceled"})
# 0.17 P-G data-job read-only zones: live / history segments (ordered tuples for SQL IN + pagination).
LIVE_STATUSES: tuple[str, ...] = ("running", "pending")
HISTORY_STATUSES: tuple[str, ...] = ("done", "failed", "canceled")


from studio.domain.errors import DomainError


class JobError(DomainError):
    """Data job business error.

    PR-2 C3 added a DomainError base -- the handler auto-translates it into the dual-write envelope.
    """
    default_code = "job.error"


def log_path_for(job_id: int) -> Path:
    """Job log path. Since R-3 = tasks/<id>/run.log (same layout as GPU tasks)."""
    return task_log_path(job_id)


# --- This fork: job-result sidecar file (added/skipped summary for the upload background task) ---------

def result_path_for(job_id: int) -> Path:
    """Sidecar JSON file for a job's result (e.g. upload's added/skipped summary)."""
    JOB_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return JOB_LOGS_DIR / f"{job_id}.result.json"


def write_result(job_id: int, payload: dict[str, Any]) -> None:
    """Called by the worker: writes the result to the sidecar file, for the status endpoint to read back to the frontend."""
    import json
    result_path_for(job_id).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def read_result(job_id: int) -> Optional[dict[str, Any]]:
    """Reads the job-result sidecar file; returns None if missing / unparseable."""
    import json
    path = result_path_for(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def as_job(task: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """tasks row -> job dict (injects kind / log_path compat fields), mutated in place and returned."""
    if not task:
        return None
    task["kind"] = task.get("task_type") or "train"
    task["log_path"] = str(task_log_path(int(task["id"])))
    return task


def create_job(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    kind: str,
    params: dict[str, Any],
    version_id: Optional[int] = None,
) -> dict[str, Any]:
    if kind not in VALID_KINDS:
        raise JobError(f"Invalid kind: {kind!r}")
    jid = db.create_task(
        conn,
        name=kind,
        config_name=kind,
        task_type=kind,
        params=params,
        project_id=project_id,
        version_id=version_id,
    )
    return as_job(db.get_task(conn, jid)) or {}


def get_job(conn: sqlite3.Connection, jid: int) -> Optional[dict[str, Any]]:
    task = db.get_task(conn, jid)
    if not task or (task.get("task_type") or "train") not in VALID_KINDS:
        return None
    return as_job(task)


def list_jobs(
    conn: sqlite3.Connection,
    *,
    project_id: Optional[int] = None,
    version_id: Optional[int] = None,
    kind: Optional[str] = None,
    status: Optional[str] = None,
) -> list[dict[str, Any]]:
    if kind is not None:
        type_clause = "task_type = ?"
        params: list[Any] = [kind]
    else:
        type_clause = f"task_type IN ({','.join('?' for _ in JOB_TASK_TYPES)})"
        params = list(JOB_TASK_TYPES)
    sql = f"SELECT * FROM tasks WHERE {type_clause}"
    if project_id is not None:
        sql += " AND project_id = ?"
        params.append(project_id)
    if version_id is not None:
        sql += " AND version_id = ?"
        params.append(version_id)
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY id DESC"
    return [as_job(db._row_to_dict(r)) or {} for r in conn.execute(sql, params)]


def list_pending_fifo(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """For supervisor dispatch: pending data jobs ordered by `priority DESC, created_at ASC`
    (same FIFO semantics as GPU tasks, D-R3 makes them peers; the hardcoded dispatch_order was removed)."""
    placeholders = ",".join("?" for _ in JOB_TASK_TYPES)
    rows = conn.execute(
        f"SELECT * FROM tasks WHERE status = 'pending' "
        f"AND task_type IN ({placeholders}) "
        "ORDER BY priority DESC, created_at ASC",
        JOB_TASK_TYPES,
    )
    return [as_job(db._row_to_dict(r)) or {} for r in rows]


def _escape_like(q: str) -> str:
    """Escapes LIKE metacharacters (% _ \\), per the same convention as db._build_task_filter."""
    return q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _jobs_filter(
    statuses: tuple[str, ...], kind: Optional[str], q: Optional[str] = None,
) -> tuple[str, list[Any]]:
    placeholders = ",".join("?" for _ in statuses)
    where = f" WHERE status IN ({placeholders})"
    params: list[Any] = list(statuses)
    if kind:
        where += " AND task_type = ?"
        params.append(kind)
    else:
        where += f" AND task_type IN ({','.join('?' for _ in JOB_TASK_TYPES)})"
        params.extend(JOB_TASK_TYPES)
    if q:
        # jobs itself has no name -- search by the owning project's title / slug (pushed down into SQL to keep total accurate).
        like = f"%{_escape_like(q)}%"
        where += (
            " AND project_id IN (SELECT id FROM projects "
            "WHERE title LIKE ? ESCAPE '\\' OR slug LIKE ? ESCAPE '\\')"
        )
        params.extend([like, like])
    return where, params


def count_jobs(
    conn: sqlite3.Connection,
    *,
    statuses: tuple[str, ...],
    kind: Optional[str] = None,
    q: Optional[str] = None,
) -> int:
    """0.17 P-G -- counts by status group (+ optional kind / project-name search), for the history pagination total."""
    where, params = _jobs_filter(statuses, kind, q)
    row = conn.execute(f"SELECT COUNT(*) FROM tasks{where}", params).fetchone()
    return int(row[0])


def list_jobs_page(
    conn: sqlite3.Connection,
    *,
    statuses: tuple[str, ...],
    kind: Optional[str] = None,
    q: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """0.17 P-G -- fetches jobs by status group, `id DESC`; limit=None means no pagination (the live group is unpaginated)."""
    where, params = _jobs_filter(statuses, kind, q)
    sql = f"SELECT * FROM tasks{where} ORDER BY id DESC"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params = [*params, int(limit), int(offset)]
    return [as_job(db._row_to_dict(r)) or {} for r in conn.execute(sql, params)]


def count_active(
    conn: sqlite3.Connection, *, version_id: int, kind: str
) -> int:
    """Count of pending/running jobs of the given kind under this version (used by the phase gate)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM tasks "
        "WHERE version_id = ? AND task_type = ? AND status IN ('pending', 'running')",
        (version_id, kind),
    ).fetchone()
    return int(row[0])


def latest_for(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    kind: str,
    version_id: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Fetches the most recent job of the given kind for this project (+ optional version)."""
    sql = "SELECT * FROM tasks WHERE project_id = ? AND task_type = ?"
    params: list[Any] = [project_id, kind]
    if version_id is not None:
        sql += " AND version_id = ?"
        params.append(version_id)
    sql += " ORDER BY id DESC LIMIT 1"
    row = conn.execute(sql, params).fetchone()
    return as_job(db._row_to_dict(row))


def mark_running(
    conn: sqlite3.Connection, jid: int, *, pid: Optional[int] = None
) -> None:
    db.update_task(conn, jid, status="running", started_at=time.time(), pid=pid)


def mark_done(conn: sqlite3.Connection, jid: int) -> None:
    db.update_task(conn, jid, status="done", finished_at=time.time())


def mark_failed(conn: sqlite3.Connection, jid: int, error_msg: str) -> None:
    db.update_task(
        conn, jid, status="failed", finished_at=time.time(), error_msg=error_msg
    )


def mark_canceled(conn: sqlite3.Connection, jid: int) -> None:
    db.update_task(conn, jid, status="canceled", finished_at=time.time())


def update_status(
    conn: sqlite3.Connection,
    jid: int,
    status: str,
    *,
    error_msg: Optional[str] = None,
) -> None:
    if status not in VALID_STATUSES:
        raise JobError(f"Invalid status: {status!r}")
    if status == "running":
        mark_running(conn, jid)
    elif status == "done":
        mark_done(conn, jid)
    elif status == "failed":
        mark_failed(conn, jid, error_msg or "unknown")
    elif status == "canceled":
        mark_canceled(conn, jid)
    elif status == "pending":
        db.update_task(
            conn, jid, status="pending",
            started_at=None, finished_at=None, pid=None, error_msg=None,
        )


def cleanup_orphan_running(conn: sqlite3.Connection) -> int:
    """Startup orphan cleanup. Jobs are now merged into tasks, handled uniformly by the supervisor's task-orphan reaping;
    this only mops up the legacy project_jobs table (already frozen since _v18; kept here idempotently as extra insurance)."""
    cur = conn.execute(
        "UPDATE project_jobs SET status = 'failed', finished_at = ?, "
        "error_msg = 'supervisor restart; orphan job' "
        "WHERE status = 'running'",
        (time.time(),),
    )
    conn.commit()
    return cur.rowcount
