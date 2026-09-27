"""v4 -> v5: adds a task_type column to tasks (distinguishes train / reg_ai / generate).

- train (default): existing training tasks, run runtime/anima_train.py. All old tasks fall back
  to this.
- reg_ai: prior generation (the base model generates a counterpart image for each training
  image to build the regularization set), runs runtime/anima_reg_ai.py. Introduced in PR-9 commit 3.
- generate: test generation (user manually runs a prompt to preview results), runs
  runtime/anima_generate.py. Introduced in PR-9 commit 5.

DEFAULT 'train' makes every existing task auto-classify as training when an old DB upgrades,
no backfill needed.
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(
        conn, "tasks", "task_type",
        "task_type TEXT NOT NULL DEFAULT 'train'",
    )
