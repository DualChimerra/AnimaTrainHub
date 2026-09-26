"""v2 -> v3: adds a `monitor_state_path` column to tasks (PP6.1).

Each training task has its own monitor state file path (per-version, falling back to per-task).
The `/api/state?task_id=N` endpoint uses this column to locate the file.
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(
        conn, "tasks", "monitor_state_path", "monitor_state_path TEXT"
    )
