"""v16 -> v17: adds params to tasks (0.17 R-2, first step of the ledger unification).

Resource-tier model (docs/design/queue-resource-model-0.17.md SS5 R-2): project_jobs' nine
kinds of data jobs will be unified into the tasks table (the write-path switch happens in R-3),
so tasks needs to hold the job's params JSON (kind-specific params: tagger config / download
search tags / reg-build options / etc).

- params: kind-specific params JSON (same semantics as project_jobs.params). Always NULL for
  train/reg_ai (their config lives in the config_path yaml); generate's param snapshot has its
  own generate_params from _v14 (a forward-write for the timeline, not reused here).

NULLABLE, no backfill. The old project_jobs table stays read-only (Q-R2: old rows are not
migrated or displayed).
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "params", "params TEXT")
