"""SQLite persistence for the task queue.

Only stores the task index; config still lives in YAML files as the source of truth
(task.config_name points at studio_data/configs/{config_name}.yaml).
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from .paths import STUDIO_DB


def _journal_mode() -> str:
    """SQLite journal mode. Defaults to WAL (fastest on a local disk).

    Can be overridden via the `ALS_SQLITE_JOURNAL` env var (e.g. `TRUNCATE` / `DELETE`). For
    cloud deployments that periodically rsync the whole studio_data to Google Drive, `TRUNCATE`
    is recommended: WAL maintains extra `-wal` / `-shm` sidecar files, which are prone to tearing
    or lost commits when synced/copied on a FUSE mount; TRUNCATE mode's single `studio.db` file
    is safer to copy.
    """
    mode = os.environ.get("ALS_SQLITE_JOURNAL", "").strip().upper()
    valid = {"WAL", "TRUNCATE", "DELETE", "PERSIST", "MEMORY", "OFF"}
    return mode if mode in valid else "WAL"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    config_name  TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    priority     INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    pid          INTEGER,
    exit_code    INTEGER,
    output_dir   TEXT,
    error_msg    TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_queue
    ON tasks(status, priority DESC, created_at ASC);
"""

VALID_STATUSES = {"pending", "running", "done", "failed", "canceled", "paused", "scheduled"}
# `paused` doesn't count as terminal -- the task is paused but resumable, not a finished state.
TERMINAL_STATUSES = {"done", "failed", "canceled"}
# The two ordered status groups used to partition the queue page (0.17 P-A/P-E): live = in
# progress + waiting, history = finished. Kept as a tuple to preserve order for SQL
# `status IN (...)` + pagination.
# `scheduled` (0.17 P-B scheduled tasks) is part of live -- shown in the queue page's 4th
# section; the dispatcher only looks at pending, and the supervisor's tick promotes due ones
# (promote_due_scheduled).
LIVE_STATUSES = ("running", "paused", "pending", "scheduled")
HISTORY_STATUSES = ("done", "failed", "canceled")
# Valid values for tasks.task_type. R-2/R-3 ledger unification: the tasks table is now the
# unified ledger for all work items. Resource-tier ownership lives in supervisor/resources.py
# (infrastructure doesn't depend back on supervisor, so it's listed flat here; tests have a
# sync assertion to guard against drift).
GPU_TASK_TYPES = ("train", "reg_ai", "generate")
# Data-job kinds (written to tasks starting with R-3). The GPU view of /api/queue excludes
# these by default until R-5's tiering lands (including a no-group compat path, to keep
# Topbar/Overview/Monitor from mistaking a data job for a training task).
JOB_TASK_TYPES = (
    "download", "preprocess", "tag", "reg_build",
    "eval_samples", "eval_clip", "eval_dino", "eval_tag", "eval_ccip",
    # This fork: large zip uploads go through a background job (Cloudflare 524 workaround)
    "upload",
)
VALID_TASK_TYPES = GPU_TASK_TYPES + JOB_TASK_TYPES


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    """Opens a connection; the caller is responsible for closing it (prefer
    `with connection_for(...)`)."""
    db_path = path or STUDIO_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA journal_mode={_journal_mode()}")
    # Concurrent writes across multiple threads (supervisor + HTTP worker): wait up to 30s for
    # a lock before raising, so "database is locked" doesn't abruptly interrupt an enqueue /
    # status update.
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def connection_for(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(path: Optional[Path] = None) -> None:
    """Creates the base tables + upgrades the schema to the latest version (tracked via
    PRAGMA user_version)."""
    from .migrations import apply_all

    with connection_for(path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()
        apply_all(conn)


# ---------------------------------------------------------------------------
# DAO
# ---------------------------------------------------------------------------


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    if not row:
        return None
    out = dict(row)
    # R-2: decodes params (kind-specific params JSON, _v17) alongside the raw value, so
    # consumers don't need to re-parse it (same params_decoded convention as the project_jobs DAO).
    if isinstance(out.get("params"), str):
        try:
            import json as _json
            out["params_decoded"] = _json.loads(out["params"])
        except Exception:
            out["params_decoded"] = None
    return out


def create_task(
    conn: sqlite3.Connection,
    *,
    name: str,
    config_name: str,
    priority: int = 0,
    scheduled_at: Optional[float] = None,
    task_type: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    project_id: Optional[int] = None,
    version_id: Optional[int] = None,
    commit: bool = True,
) -> int:
    """Creates a pending (or scheduled) task.

    ``commit=False`` is only for enqueue endpoints that need to write a task-scoped config
    snapshot within the same transaction; other callers keep the original immediate-commit
    behavior.

    R-2 ledger unification: the tasks table now holds all work items. `task_type` defaults to
    'train' (compat for old callers); data-job kinds (download/tag/...) are stored with
    `params` (kind-specific params JSON) + project_id/version_id. The write-path switch
    (services moving from create_job to here) happens in R-3.
    """
    if task_type is not None and task_type not in VALID_TASK_TYPES:
        raise ValueError(f"invalid task_type: {task_type!r}")
    # ADR-0009 PR-1 C6: stores the current ContextVar trace_id when the task is created (already
    # bound by TraceIdMiddleware at the moment of the HTTP request). Falls back to bg-{uuid} to
    # mark a background trigger (CLI / tests / supervisor spawning it directly). The supervisor
    # dispatcher later reads this column -> injects it into the worker subprocess's env, so the
    # worker log lines up with the user's request trace_id.
    from .logging import get_trace_id, new_trace_id
    request_trace_id = get_trace_id() or f"bg-{new_trace_id()}"
    # 0.17 P-B: with scheduled_at set -> created as scheduled, promoted to pending by the
    # supervisor's tick once due. A time in the past is also fine to create -- the next tick
    # (<=1s) promotes it naturally, no special-casing needed.
    status = "scheduled" if scheduled_at is not None else "pending"
    import json as _json
    cur = conn.execute(
        "INSERT INTO tasks(name, config_name, status, priority, created_at, "
        "request_trace_id, scheduled_at, task_type, params, project_id, version_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, 'train'), ?, ?, ?)",
        (name, config_name, status, priority, time.time(), request_trace_id,
         scheduled_at, task_type,
         _json.dumps(params) if params is not None else None,
         project_id, version_id),
    )
    if commit:
        conn.commit()
    return int(cur.lastrowid)


def get_task(conn: sqlite3.Connection, task_id: int) -> Optional[dict[str, Any]]:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _row_to_dict(row)


def filter_out_task_types(
    items: list[dict[str, Any]], excluded: tuple[str, ...]
) -> list[dict[str, Any]]:
    """commit 15: strips the given task_type(s) out of a task list (defaults task_type to
    'train' for compat)."""
    return [
        t for t in items
        if (t.get("task_type") or "train") not in excluded
    ]


def list_tasks(
    conn: sqlite3.Connection, status: Optional[str] = None
) -> list[dict[str, Any]]:
    if status:
        sql = (
            "SELECT * FROM tasks WHERE status = ? "
            "ORDER BY priority DESC, created_at ASC"
        )
        params: tuple = (status,)
    else:
        sql = "SELECT * FROM tasks ORDER BY priority DESC, created_at ASC"
        params = ()
    return [_row_to_dict(r) or {} for r in conn.execute(sql, params)]


def _escape_like(s: str) -> str:
    """Escapes LIKE metacharacters (\\ % _), used together with `ESCAPE '\\'` so the search
    matches literally."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_task_filter(
    *,
    statuses: tuple[str, ...],
    exclude_types: tuple[str, ...],
    q: Optional[str],
    types: tuple[str, ...] = (),
) -> tuple[str, list[Any]]:
    """Builds the WHERE clause + params (no ORDER/LIMIT), shared by count_tasks / list_tasks_page.

    - statuses: `status IN (...)` (order-preserving tuple)
    - types: `COALESCE(task_type,'train') IN (...)` -- 0.17 P-F type filter (positive inclusion)
    - exclude_types: `COALESCE(task_type,'train') NOT IN (...)` -- old rows' NULL falls back to
      'train' so they aren't wrongly excluded (keeps pagination totals accurate)
    - q: substring match on name / config_name + the owning project's title/slug
      (LIKE + ESCAPE, metacharacters escaped). R-5: a data job's name=kind has no search value,
      so it's matched via the project-name subquery instead; GPU tasks get project-name search
      as a side benefit
    """
    clauses: list[str] = []
    params: list[Any] = []
    if statuses:
        clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
        params.extend(statuses)
    if types:
        clauses.append(
            f"COALESCE(task_type, 'train') IN ({','.join('?' for _ in types)})"
        )
        params.extend(types)
    if exclude_types:
        clauses.append(
            f"COALESCE(task_type, 'train') NOT IN ({','.join('?' for _ in exclude_types)})"
        )
        params.extend(exclude_types)
    if q:
        like = f"%{_escape_like(q)}%"
        clauses.append(
            "(name LIKE ? ESCAPE '\\' OR config_name LIKE ? ESCAPE '\\' "
            "OR project_id IN (SELECT id FROM projects "
            "WHERE title LIKE ? ESCAPE '\\' OR slug LIKE ? ESCAPE '\\'))"
        )
        params.extend([like, like, like, like])
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def count_tasks(
    conn: sqlite3.Connection,
    *,
    statuses: tuple[str, ...],
    exclude_types: tuple[str, ...] = (),
    q: Optional[str] = None,
    types: tuple[str, ...] = (),
) -> int:
    """Total count of tasks matching statuses (+ optional types / exclude_types / q), for
    pagination totals."""
    where, params = _build_task_filter(
        statuses=statuses, exclude_types=exclude_types, q=q, types=types
    )
    row = conn.execute(f"SELECT COUNT(*) AS n FROM tasks{where}", params).fetchone()
    return int(row["n"]) if row else 0


def list_tasks_page(
    conn: sqlite3.Connection,
    *,
    statuses: tuple[str, ...],
    exclude_types: tuple[str, ...] = (),
    q: Optional[str] = None,
    types: tuple[str, ...] = (),
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Fetches tasks matching statuses (+ optional types / exclude_types / q), `id DESC`
    (most recent first).

    limit=None means no pagination (the live group is fetched in full); a limit adds
    `LIMIT ? OFFSET ?` (used for the history group).
    """
    where, params = _build_task_filter(
        statuses=statuses, exclude_types=exclude_types, q=q, types=types
    )
    sql = f"SELECT * FROM tasks{where} ORDER BY id DESC"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params = [*params, int(limit), int(offset)]
    return [_row_to_dict(r) or {} for r in conn.execute(sql, params)]


def promote_due_scheduled(
    conn: sqlite3.Connection, now: Optional[float] = None
) -> list[int]:
    """0.17 P-B: promotes due scheduled tasks to pending, returns the list of promoted ids.

    Called once per supervisor tick (1s); scheduled_at is kept, not cleared (records the
    original planned time). The caller is responsible for publishing
    task_state_changed(pending) for each returned id.
    """
    ts = time.time() if now is None else now
    ids = [
        int(r["id"]) for r in conn.execute(
            "SELECT id FROM tasks WHERE status = 'scheduled' AND scheduled_at <= ? "
            "ORDER BY id ASC",
            (ts,),
        )
    ]
    if ids:
        conn.executemany(
            "UPDATE tasks SET status = 'pending' WHERE id = ? AND status = 'scheduled'",
            [(i,) for i in ids],
        )
        conn.commit()
    return ids


def next_pending(conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM tasks WHERE status = 'pending' "
        "ORDER BY priority DESC, created_at ASC LIMIT 1"
    ).fetchone()
    return _row_to_dict(row)


def update_task(
    conn: sqlite3.Connection, task_id: int, **fields: Any
) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    params = list(fields.values()) + [task_id]
    conn.execute(f"UPDATE tasks SET {cols} WHERE id = ?", params)
    conn.commit()


def delete_task(conn: sqlite3.Connection, task_id: int) -> int:
    cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    return cur.rowcount


def reorder(
    conn: sqlite3.Connection, ordered_ids: list[int]
) -> None:
    """Rewrites priority to match the given id order (first = highest). Only affects
    pending tasks."""
    base = len(ordered_ids)
    for i, tid in enumerate(ordered_ids):
        conn.execute(
            "UPDATE tasks SET priority = ? WHERE id = ? AND status = 'pending'",
            (base - i, tid),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# queue_settings -- a kv table that persists across restarts. Introduced by ADR 0006 PR-2.
# ---------------------------------------------------------------------------

_QUEUE_HELD_KEY = "queue.held"


def get_queue_held(conn: sqlite3.Connection) -> bool:
    """The queue-hold switch (ADR SS3.2). Defaults to False (not held)."""
    row = conn.execute(
        "SELECT value FROM queue_settings WHERE key = ?", (_QUEUE_HELD_KEY,)
    ).fetchone()
    if row is None:
        return False
    return str(row[0]).lower() == "true"


def set_queue_held(conn: sqlite3.Connection, held: bool) -> None:
    """Writes the hold switch. Value is serialized as the literal "true" / "false"."""
    conn.execute(
        "INSERT INTO queue_settings(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (_QUEUE_HELD_KEY, "true" if held else "false"),
    )
    conn.commit()
