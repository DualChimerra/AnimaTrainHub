"""v12 -> v13: ADR 0006 Addendum 2 -- terminal-resume (failed/canceled tasks become resumable).

New columns (tasks):

- `last_state_path`: `.pt` path of the most recent end-of-epoch auto backup
  (auto_epoch_state.pt, a single file that gets overwritten each time)
- `last_config_path`: matching config snapshot path (auto_epoch_state.config.json)
- `last_state_epoch` / `last_state_step`: epoch / global_step of the backup point
  (used for the UI's "resume from epoch N" hint)

Written by the supervisor when it receives an `auto_epoch_backup_written` event (previously
only kept in an in-memory slot field, lost as soon as the process/machine died). Once in the
DB, the resume point path for a failed (crash / shutdown) or canceled task survives restarts,
and the resume endpoint relaxes its status gate based on it.

Backfill (existing terminal tasks): starting with Addendum 2, auto backups moved to the task's
own archive at `studio_data/tasks/<id>/state/`, but tasks that ran before this migration wrote
to the old layout `<version>/output/state/task_<id>/auto_epoch_state.pt`. For failed/canceled
tasks attached to a project, this probes the old location by the version-directory convention;
if found, backfills last_state_* -- the resume path only trusts whatever's recorded in the DB,
so old and new layouts both work.

Left as NULL when not found (not one-click resumable, as expected): tasks predating Addendum 1
never had auto backup at all; a resume point whose version was deleted vanished along with the
directory; a detached task (no project/version, enqueued directly against a global preset) has
its output_dir only in yaml, possibly edited by the user since -- the value at write time isn't
reliable, so backfill is skipped for these. The epoch/step columns aren't backfilled either
(would require torch.load to recover; they're just a UI hint field, not worth it).

DDL design principle: everything NULLABLE, works with no backfill required.
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def _backfill_from_legacy_layout(conn: sqlite3.Connection) -> None:
    """Probe the old layout and backfill last_state_*. Idempotent: only touches rows where
    last_state_path IS NULL."""
    # Lazy import: avoids a hard dependency on the services layer when the migration module
    # loads; tests that monkeypatch services.projects.PROJECTS_DIR before running migrate get
    # the correct path (same pattern as _v11).
    from ...services.projects import projects as _projects

    rows = conn.execute(
        "SELECT t.id, p.id, p.slug, v.label "
        "FROM tasks t "
        "JOIN projects p ON t.project_id = p.id "
        "JOIN versions v ON t.version_id = v.id "
        "WHERE t.status IN ('failed', 'canceled') "
        "AND t.last_state_path IS NULL"
    ).fetchall()
    for tid, pid, slug, label in rows:
        try:
            vdir = _projects.project_dir(int(pid), str(slug)) / "versions" / str(label)
        except (TypeError, ValueError):
            continue
        state_dir = vdir / "output" / "state" / f"task_{int(tid)}"
        pt = state_dir / "auto_epoch_state.pt"
        if not pt.is_file():
            continue
        cfg = state_dir / "auto_epoch_state.config.json"
        conn.execute(
            "UPDATE tasks SET last_state_path = ?, last_config_path = ? "
            "WHERE id = ?",
            (str(pt), str(cfg) if cfg.is_file() else None, int(tid)),
        )
    conn.commit()


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "last_state_path",
                           "last_state_path TEXT")
    _add_column_if_missing(conn, "tasks", "last_config_path",
                           "last_config_path TEXT")
    _add_column_if_missing(conn, "tasks", "last_state_epoch",
                           "last_state_epoch INTEGER")
    _add_column_if_missing(conn, "tasks", "last_state_step",
                           "last_state_step INTEGER")
    _backfill_from_legacy_layout(conn)
