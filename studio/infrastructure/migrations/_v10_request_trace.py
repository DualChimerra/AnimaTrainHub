"""v9 -> v10: adds a request_trace_id column to tasks (ADR-0009 PR-1 C6, trace_id threaded across processes).

When an API endpoint enqueues a task, it writes the contextvar trace_id (the ID at the moment
of the HTTP request); when the supervisor dispatcher picks the task up, it reads this column ->
injects it into the worker subprocess's env -> the worker bootstrap binds the contextvar.

Without this: the trace_id at the moment the user clicked "start training" and the trace_id at
the moment the dispatcher spawns it in the background would be two different IDs, and the
trace_id chain would break right at the spawn step. The trace in the user's screenshot/toast
wouldn't match the trace in the worker log -- defeating the whole point of introducing trace_id
in PR-1 C5.

Round 2 review SS4.4 stressed: "this must land together with PR-LOG-3 or the trace_id chain
is broken".
"""
from __future__ import annotations

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE tasks ADD COLUMN request_trace_id TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists (migration is tolerant of this).
        pass
