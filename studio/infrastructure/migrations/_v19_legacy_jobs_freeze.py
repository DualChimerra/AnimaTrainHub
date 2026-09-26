"""v17 -> v18: freezes the legacy project_jobs table (companion to the R-3 write-path flip).

Starting with R-3, data jobs are written to tasks (task_type = kind) instead, and the
supervisor no longer dispatches from project_jobs. Any pending / running rows leftover at the
moment of upgrade would otherwise stay stuck in that state forever (nothing dispatches them
anymore) -- so they're marked canceled once, with a reason noted.

The rest of the old table's rows are kept read-only as legacy (Q-R2: not migrated, not
displayed, to avoid an ID remap polluting eval run references and old log paths).
"""
from __future__ import annotations

import sqlite3
import time


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE project_jobs SET status = 'canceled', finished_at = ?, "
        "error_msg = 'superseded by unified ledger (0.17 R-3)' "
        "WHERE status IN ('pending', 'running')",
        (time.time(),),
    )
