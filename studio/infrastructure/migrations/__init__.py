"""Schema migrations: apply SQL upgrades in order, progress tracked via PRAGMA user_version.

`db.init_db()` first executescripts the base SCHEMA (v1, defines the tasks table), then calls
`apply_all()` to push user_version to the latest. To add a migration, append a callback to the
end of MIGRATIONS and user_version auto-increments.

Conventions:
- Any ALTER TABLE must tolerate "column already exists" (IF NOT EXISTS doesn't apply to ADD
  COLUMN, so a try/except covers it); this way upgrading an old DB doesn't fail on a duplicate ADD.
- Rewriting existing columns is normally not allowed; only add columns / tables / indexes.
- **v9 exception**: ADR-0007 PR-5 destructively drops the stage column (recreate-table mode),
  explicitly documented in ADR-0007 SS Consequences.
"""
from __future__ import annotations

import sqlite3
from typing import Callable

from ._v2_projects import migrate as _migrate_v2
from ._v3_monitor_state import migrate as _migrate_v3
from ._v4_task_config_path import migrate as _migrate_v4
from ._v5_task_type import migrate as _migrate_v5
from ._v6_pause_resume import migrate as _migrate_v6
from ._v7_version_trigger_word import migrate as _migrate_v7
from ._v8_version_status_phase import migrate as _migrate_v8
from ._v9_drop_legacy_stage import migrate as _migrate_v9
from ._v10_request_trace import migrate as _migrate_v10
from ._v11_preprocessing_phase import migrate as _migrate_v11
from ._v12_drop_tagging_phase import migrate as _migrate_v12
from ._v13_project_archived import migrate as _migrate_v13
from ._v14_last_state import migrate as _migrate_v14
from ._v15_generate_meta import migrate as _migrate_v15
from ._v16_scheduled_at import migrate as _migrate_v16
from ._v17_job_created_at import migrate as _migrate_v17
from ._v18_unified_ledger import migrate as _migrate_v18
from ._v19_legacy_jobs_freeze import migrate as _migrate_v19
from ._v20_task_note import migrate as _migrate_v20

Migration = Callable[[sqlite3.Connection], None]

# List index is the version number (1-based). v1 = base SCHEMA (not in this list).
MIGRATIONS: list[Migration] = [
    _migrate_v2,  # v2: projects / versions / project_jobs + extra fields on tasks
    _migrate_v3,  # v3: tasks.monitor_state_path (PP6.1 per-version monitor)
    _migrate_v4,  # v4: tasks.config_path (PP6.3 private config path)
    _migrate_v5,  # v5: tasks.task_type (PR-9 distinguishes train / reg_ai / generate)
    _migrate_v6,  # v6: tasks.paused_* columns + queue_settings table (ADR 0006 PR-2)
    _migrate_v7,  # v7: versions.trigger_word (trigger word field)
    _migrate_v8,  # v8: versions.status / phase / last_failure_reason (ADR-0007)
    _migrate_v9,  # v9: drop projects.stage + versions.stage (ADR-0007 PR-5, destructive)
    _migrate_v10, # v10: tasks.request_trace_id (ADR-0009 PR-1 C6, trace_id threaded across processes)
    _migrate_v11, # v11: versions.phase gains the preprocessing value (ADR-0010 companion; backfills non-empty curating+train -> preprocessing)
    _migrate_v12, # v12: removes the tagging phase (auto-tagging removed); existing tagging -> editing
    # Note: this fork's v12 differs in meaning from upstream v12 (upstream has no drop_tagging).
    # Upstream v12-v18 are shifted to v13-v19 in this fork; content is byte-identical, only the
    # numbering shifted.
    _migrate_v13, # upstream v12: projects.archived_at (soft-hide project archiving)
    _migrate_v14, # upstream v13: tasks.last_state_* columns (ADR 0006 Addendum 2, terminal-resume)
    _migrate_v15, # upstream v14: tasks.generate_params / generate_cover (0.17 P-I forward-write)
    _migrate_v16, # upstream v15: tasks.scheduled_at (0.17 P-B scheduled tasks, pairs with the new scheduled state)
    _migrate_v17, # upstream v16: project_jobs.created_at (0.17 P-G enqueue time for the data-job detail page)
    _migrate_v18, # upstream v17: tasks.params (R-2 ledger merge -- tasks absorbs data-job kind params)
    _migrate_v19, # upstream v18: freezes legacy project_jobs (R-3 write-path flip; leftover pending/running -> canceled)
    _migrate_v20, # v20: tasks.note (queue task note, written via right-click, editable in the detail page)
]


def current_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def apply_all(conn: sqlite3.Connection) -> int:
    """Push user_version to len(MIGRATIONS) + 1 (base = 1). Returns the final version number."""
    target = len(MIGRATIONS) + 1
    cur = current_version(conn)
    if cur == 0:
        # Brand-new DB / old DB never set user_version: v1 was already created by SCHEMA
        cur = 1
        conn.execute("PRAGMA user_version = 1")
    while cur < target:
        migration = MIGRATIONS[cur - 1]  # cur=1 -> MIGRATIONS[0] pushes to v2
        migration(conn)
        cur += 1
        conn.execute(f"PRAGMA user_version = {cur}")
    conn.commit()
    return cur
