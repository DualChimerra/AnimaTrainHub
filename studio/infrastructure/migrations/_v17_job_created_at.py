"""v15 -> v16: adds created_at to project_jobs (0.17 P-G data-job detail page).

project_jobs has only had started_at / finished_at since the table was created; enqueue time
was never recorded -- the data-job detail page needs to show "enqueued at" (to match the task
detail summary).

NULLABLE, no backfill: the enqueue moment for old jobs is simply unknowable (started_at only
proves "when it started running"); the UI shows "--" for NULL. New jobs get it written by
create_job.
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "project_jobs", "created_at", "created_at REAL")
