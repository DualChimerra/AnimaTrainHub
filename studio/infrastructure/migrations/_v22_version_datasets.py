"""v21 -> v22: version_datasets -- which shared datasets a version curates from.

Shared datasets are `projects` rows with kind='dataset' (their images live in
`studio_data/datasets/{id}-{slug}/download/`). A version links any number of
them; curation offers their images next to the project's own download/. Both
foreign keys cascade: deleting the version or the dataset drops the link, the
version's train/ copies stay untouched.
"""
from __future__ import annotations

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS version_datasets (
            version_id  INTEGER NOT NULL REFERENCES versions(id) ON DELETE CASCADE,
            dataset_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            position    INTEGER NOT NULL DEFAULT 0,
            added_at    REAL NOT NULL,
            PRIMARY KEY (version_id, dataset_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_version_datasets_dataset "
        "ON version_datasets(dataset_id)"
    )
    conn.commit()
