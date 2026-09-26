"""v7 -> v8: adds status / phase / last_failure_reason to versions -- ADR-0007 SS11.3-B.

Splits the master `versions.stage` single enum field into two orthogonal fields:

- `status` (5-value enum): preparing / training / completed / failed / canceled
- `phase`  (5-value enum, only meaningful when status=preparing):
  curating / tagging / editing / regularizing / ready
- `last_failure_reason` TEXT: training failure reason (used for UI derivation, nullable)

This migration only adds fields and backfills from the mapping table, it does **not drop**
the `versions.stage` column (that happens in v9). During the migration window the backend
dual-writes old and new fields (PR-3), keeping old frontends compatible.

Mapping table (ADR-0007 SS11.3-B):

  master versions.stage -> new status / phase
  ------------------------------------------------------------
  curating         -> status=preparing, phase=curating
  tagging          -> status=preparing, phase=tagging
  regularizing     -> status=preparing, phase=regularizing
  ready            -> status=preparing, phase=ready
  training         -> derived from the latest task:
                       done     -> completed
                       failed   -> failed
                       canceled -> canceled
                       running / pending / paused -> training
                       no task found -> preparing + phase=ready (dirty-data fallback)
  done             -> status=completed, phase=ready
  unknown stage    -> preparing + phase=ready (fallback)

phase has no business meaning when status != preparing, but the field is kept required to
keep the schema simple -- it's uniformly set to ready, meaning "the preparing phase is done".
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


# stage -> (status, phase) static mapping (training is an exception, needs a task lookup)
_STATIC_STAGE_MAP: dict[str, tuple[str, str]] = {
    "curating":     ("preparing", "curating"),
    "tagging":      ("preparing", "tagging"),
    "regularizing": ("preparing", "regularizing"),
    "ready":        ("preparing", "ready"),
    "done":         ("completed", "ready"),
}

# When the old stage='training', derive it from the latest task.status
_TASK_STATUS_MAP: dict[str, str] = {
    "done":     "completed",
    "failed":   "failed",
    "canceled": "canceled",
    "pending":  "training",
    "running":  "training",
    "paused":   "training",
}


def migrate(conn: sqlite3.Connection) -> None:
    _add_columns(conn)
    _backfill(conn)


def _add_columns(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(
        conn, "versions", "status",
        "status TEXT NOT NULL DEFAULT 'preparing'",
    )
    _add_column_if_missing(
        conn, "versions", "phase",
        "phase TEXT NOT NULL DEFAULT 'curating'",
    )
    _add_column_if_missing(
        conn, "versions", "last_failure_reason",
        "last_failure_reason TEXT",
    )


def _backfill(conn: sqlite3.Connection) -> None:
    """Translate the existing versions.stage into status + phase, per the ADR SS11.3-B mapping table."""
    rows = conn.execute("SELECT id, stage FROM versions").fetchall()
    for vid, stage in rows:
        if stage in _STATIC_STAGE_MAP:
            status, phase = _STATIC_STAGE_MAP[stage]
        elif stage == "training":
            status, phase = _derive_from_latest_task(conn, vid)
        else:
            # Unknown stage (should never happen in theory) -> fallback
            status, phase = "preparing", "ready"
        conn.execute(
            "UPDATE versions SET status = ?, phase = ? WHERE id = ?",
            (status, phase, vid),
        )
    conn.commit()


def _derive_from_latest_task(
    conn: sqlite3.Connection, version_id: int
) -> tuple[str, str]:
    """For the training stage, derive status from the latest task; phase lands on 'ready' (meaningless once terminal)."""
    row = conn.execute(
        "SELECT status FROM tasks "
        "WHERE version_id = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (version_id,),
    ).fetchone()
    if row is None:
        # Dirty-data fallback: stage=training but no associated task
        return ("preparing", "ready")
    task_status = str(row[0]) if row[0] else ""
    new_status = _TASK_STATUS_MAP.get(task_status, "preparing")
    return (new_status, "ready")
