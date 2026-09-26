"""v11 -> v12: projects.archived_at -- project archiving (soft-hide).

Adds column `projects.archived_at REAL` (NULL = not archived). Archived projects don't show in
the projects page's default list; the UI's old "delete" entry now archives first, and only
actually deletes when clicked again from the archive view. Archiving doesn't touch updated_at --
the sort position stays the same after restoring.
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(
        conn, "projects", "archived_at", "archived_at REAL"
    )
