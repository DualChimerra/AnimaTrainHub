"""v20 -> v21: projects.kind -- 'lora' (regular pipeline) or 'edit' (source/target pairs).

Edit projects reuse versions, configs, the queue and the monitor; only the dataset
step differs (pairs instead of curated images) and training runs with edit_mode.
"""
from __future__ import annotations

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE projects ADD COLUMN kind TEXT NOT NULL DEFAULT 'lora'")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
