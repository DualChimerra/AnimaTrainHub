"""v5 -> v6: ADR 0006 PR-2 -- pause/resume backend skeleton.

New columns:

- `paused_state_path`: `.pt` path at the time of pausing (pause_step_<N>.pt)
- `paused_config_path`: config snapshot path at the time of pausing (pause_step_<N>.config.json)
- `paused_step`: global_step snapshot (used by the picker / UI hints)
- `paused_at`: UNIX seconds, used for the "paused at step N ..." display

New table `queue_settings`: single-row kv storage, persists across restarts. Currently just one
key `queue.held` (true/false); the dispatcher checks it to decide whether to skip this round of
scheduling (ADR SS3.2).

DDL design principle: all paused_* columns are NULLABLE (all NULL when a task was never paused),
no backfill needed. queue_settings is a brand-new table, empty when an old DB upgrades.
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "paused_state_path",
                           "paused_state_path TEXT")
    _add_column_if_missing(conn, "tasks", "paused_config_path",
                           "paused_config_path TEXT")
    _add_column_if_missing(conn, "tasks", "paused_step",
                           "paused_step INTEGER")
    _add_column_if_missing(conn, "tasks", "paused_at",
                           "paused_at REAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS queue_settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )
