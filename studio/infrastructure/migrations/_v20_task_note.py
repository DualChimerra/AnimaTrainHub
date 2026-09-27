"""v19 -> v20: tasks.note -- queue task note.

Adds column `tasks.note TEXT` (NULL / empty string = no note). Users right-click any task on
the queue page and write a one-liner (e.g. "tried alpha=16 this time", "swapped dataset to v3");
the note travels with the task: a badge shows on the queue list row, and the task detail page's
overview shows it in full at the top, editable.

Pure UI metadata -- doesn't participate in scheduling and isn't included in the config snapshot.
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "note", "note TEXT")
