"""v11 -> v12: removes the `tagging` value from VersionPhase (the auto-tagging step was removed).

VersionPhase goes from 6 -> 5:

    curating -> preprocessing -> editing -> regularizing -> ready

phase is a TEXT column (added in _v8), no schema change needed; this just moves any existing
version still sitting at `tagging` forward to the next mandatory phase, `editing` (the old
tagging step was skippable, and skipping it led straight into editing -- consistent with the
old semantics, so users don't lose progress).

Idempotent: when there are no phase='tagging' rows, the UPDATE affects 0 rows, no side effects.
"""
from __future__ import annotations

import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE versions SET phase = 'editing' WHERE phase = 'tagging'"
    )
