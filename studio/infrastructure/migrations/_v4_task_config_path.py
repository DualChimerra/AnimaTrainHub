"""v3 -> v4: adds a `config_path` column to tasks (PP6.3).

After PP6.3, a training task's yaml no longer comes from the global `presets/{config_name}.yaml`,
but from the version's private config (`projects/{id}-{slug}/versions/{label}/config.yaml`).
The supervisor prefers `tasks.config_path`, falling back to `_configs_dir / config_name.yaml`
when absent (compat with old tasks).
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "config_path", "config_path TEXT")
