"""v14 -> v15: adds scheduled_at to tasks (0.17 P-B scheduled tasks).

A scheduled task = a distinct `scheduled` status (not a subset of pending): enqueuing with a
future time -> status='scheduled' + scheduled_at; the supervisor's per-second tick promotes it
to pending once due, after which it follows the existing scheduling path (wait for a slot ->
running -> terminal). The dispatcher only looks at pending, so scheduled tasks are naturally
invisible to dispatch -- no change needed to next_pending / _next_pending_task_in.

- scheduled_at: planned start time (unix seconds). Kept as a record after promotion to pending
  (so "originally scheduled for 3:00, actually started early by hand" stays traceable).

NULLABLE, no backfill needed; non-scheduled tasks are always NULL.
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "scheduled_at", "scheduled_at REAL")
