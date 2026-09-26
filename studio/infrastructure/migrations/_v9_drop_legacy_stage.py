"""v8 -> v9: physically drops the projects.stage and versions.stage columns (ADR-0007 PR-5, destructive).

PR-2 v8 added status / phase / last_failure_reason; PR-3 / PR-5 commits 1/2/3 removed all code
that read/wrote stage. This migration is the destructive follow-up that pulls the two old
columns out of the DB.

**Explicitly breaks the existing convention in `studio/migrations/__init__.py`** ("rewriting
existing columns is not allowed"). This exception is documented in ADR-0007 SS Consequences.

SQLite < 3.35 doesn't support ``ALTER TABLE DROP COLUMN``, so this uniformly uses the
recreate-table pattern:
- CREATE TABLE {t}_new (without the stage column)
- INSERT INTO _new (cols) SELECT cols FROM the original table
- DROP the original table / RENAME _new -> the original name
- rebuild indexes

Idempotency: on a second run the `stage` column no longer exists, so ``SELECT stage FROM``
raises sqlite3.OperationalError; the migration framework won't rerun it anyway (this function
is skipped once PRAGMA user_version has advanced past it).
"""
from __future__ import annotations

import sqlite3


_PROJECTS_NEW_SCHEMA = """
CREATE TABLE projects_new (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    slug              TEXT UNIQUE NOT NULL,
    title             TEXT NOT NULL,
    active_version_id INTEGER,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    note              TEXT
);
"""

_PROJECTS_COLS = "id, slug, title, active_version_id, created_at, updated_at, note"


_VERSIONS_NEW_SCHEMA = """
CREATE TABLE versions_new (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id           INTEGER NOT NULL,
    label                TEXT NOT NULL,
    config_name          TEXT,
    created_at           REAL NOT NULL,
    output_lora_path     TEXT,
    note                 TEXT,
    trigger_word         TEXT NOT NULL DEFAULT '',
    status               TEXT NOT NULL DEFAULT 'preparing',
    phase                TEXT NOT NULL DEFAULT 'curating',
    last_failure_reason  TEXT,
    UNIQUE(project_id, label),
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);
"""

_VERSIONS_COLS = (
    "id, project_id, label, config_name, created_at, output_lora_path, note, "
    "trigger_word, status, phase, last_failure_reason"
)


def migrate(conn: sqlite3.Connection) -> None:
    _drop_column_via_recreate(
        conn, "projects", _PROJECTS_NEW_SCHEMA, _PROJECTS_COLS,
        ("CREATE INDEX IF NOT EXISTS idx_projects_slug ON projects(slug)",),
    )
    _drop_column_via_recreate(
        conn, "versions", _VERSIONS_NEW_SCHEMA, _VERSIONS_COLS,
        ("CREATE INDEX IF NOT EXISTS idx_versions_project ON versions(project_id)",),
    )


def _drop_column_via_recreate(
    conn: sqlite3.Connection,
    table: str,
    new_schema: str,
    cols: str,
    indexes: tuple[str, ...] = (),
) -> None:
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.executescript(new_schema)
        conn.execute(f"INSERT INTO {table}_new ({cols}) SELECT {cols} FROM {table}")
        conn.execute(f"DROP TABLE {table}")
        conn.execute(f"ALTER TABLE {table}_new RENAME TO {table}")
        for idx_sql in indexes:
            conn.execute(idx_sql)
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
