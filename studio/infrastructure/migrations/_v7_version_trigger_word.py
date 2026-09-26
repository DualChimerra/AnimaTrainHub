"""v6 -> v7: versions.trigger_word -- the project-level trigger word, filled in at Step 4 (Tagging).

Adds column `versions.trigger_word TEXT DEFAULT ''`. Empty string means the trigger word is
disabled. The tag worker prepends it as the first tag when writing captions; version_config
injects it into the private yaml, and runtime bootstrap_phase also injects it into
sample_prompt/sample_prompts.
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(
        conn, "versions", "trigger_word",
        "trigger_word TEXT NOT NULL DEFAULT ''"
    )
