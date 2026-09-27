"""ADR-0007 section 11.3-C / section 6.9: derive_status_from_tasks + reconcile_version_status tests."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from studio import db
from studio.services.projects import projects, versions


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    pdir = tmp_path / "projects"
    monkeypatch.setattr(projects, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    return {"db": dbfile}


def _make_version(isolated, label: str = "v1") -> dict:
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label=label)
    return v


def _insert_task(isolated, version_id: int, project_id: int, status: str,
                 created_at: float | None = None) -> int:
    with db.connection_for(isolated["db"]) as conn:
        cur = conn.execute(
            "INSERT INTO tasks(name, config_name, status, project_id, version_id, created_at) "
            "VALUES (?, 'cfg', ?, ?, ?, ?)",
            (f"task-{status}", status, project_id, version_id, created_at or time.time()),
        )
        conn.commit()
        return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# derive_status_from_tasks
# ---------------------------------------------------------------------------


def test_derive_no_tasks_returns_preparing(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        assert versions.derive_status_from_tasks(conn, v["id"]) == "preparing"


def test_derive_pending_task_returns_training(isolated) -> None:
    v = _make_version(isolated)
    _insert_task(isolated, v["id"], v["project_id"], "pending")
    with db.connection_for(isolated["db"]) as conn:
        assert versions.derive_status_from_tasks(conn, v["id"]) == "training"


def test_derive_running_task_returns_training(isolated) -> None:
    v = _make_version(isolated)
    _insert_task(isolated, v["id"], v["project_id"], "running")
    with db.connection_for(isolated["db"]) as conn:
        assert versions.derive_status_from_tasks(conn, v["id"]) == "training"


def test_derive_paused_task_returns_training(isolated) -> None:
    """section 11.3-A: when task=paused, version stays training; the UI derives and shows a pause icon."""
    v = _make_version(isolated)
    _insert_task(isolated, v["id"], v["project_id"], "paused")
    with db.connection_for(isolated["db"]) as conn:
        assert versions.derive_status_from_tasks(conn, v["id"]) == "training"


def test_derive_done_task_returns_completed(isolated) -> None:
    v = _make_version(isolated)
    _insert_task(isolated, v["id"], v["project_id"], "done")
    with db.connection_for(isolated["db"]) as conn:
        assert versions.derive_status_from_tasks(conn, v["id"]) == "completed"


def test_derive_failed_task_returns_failed(isolated) -> None:
    v = _make_version(isolated)
    _insert_task(isolated, v["id"], v["project_id"], "failed")
    with db.connection_for(isolated["db"]) as conn:
        assert versions.derive_status_from_tasks(conn, v["id"]) == "failed"


def test_derive_canceled_task_returns_canceled(isolated) -> None:
    v = _make_version(isolated)
    _insert_task(isolated, v["id"], v["project_id"], "canceled")
    with db.connection_for(isolated["db"]) as conn:
        assert versions.derive_status_from_tasks(conn, v["id"]) == "canceled"


def test_derive_active_takes_priority_over_terminal(isolated) -> None:
    """Old task done + new task running -> training (active takes priority)."""
    v = _make_version(isolated)
    now = time.time()
    _insert_task(isolated, v["id"], v["project_id"], "done", created_at=now - 100)
    _insert_task(isolated, v["id"], v["project_id"], "running", created_at=now)
    with db.connection_for(isolated["db"]) as conn:
        assert versions.derive_status_from_tasks(conn, v["id"]) == "training"


def test_derive_picks_latest_terminal_task(isolated) -> None:
    """Multiple terminal-state tasks -> take the most recent by created_at."""
    v = _make_version(isolated)
    now = time.time()
    _insert_task(isolated, v["id"], v["project_id"], "failed", created_at=now - 100)
    _insert_task(isolated, v["id"], v["project_id"], "done", created_at=now - 50)
    _insert_task(isolated, v["id"], v["project_id"], "canceled", created_at=now)
    with db.connection_for(isolated["db"]) as conn:
        assert versions.derive_status_from_tasks(conn, v["id"]) == "canceled"


# ---------------------------------------------------------------------------
# reconcile_version_status
# ---------------------------------------------------------------------------


def test_reconcile_noop_when_consistent(isolated) -> None:
    """status matches derived -> was_corrected=False."""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        ver, corrected = versions.reconcile_version_status(conn, v["id"])
    assert ver is not None
    assert ver["status"] == "preparing"  # no task -> preparing
    assert corrected is False


def test_reconcile_corrects_stale_status(isolated) -> None:
    """Version status=training but there's no longer an active task -> derives preparing -> gets corrected."""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        # manually write a "lying" status
        versions.update_version(conn, v["id"], status="training")
        # reconcile should correct it
        ver, corrected = versions.reconcile_version_status(conn, v["id"])
    assert ver is not None
    assert ver["status"] == "preparing"  # derived with no task
    assert corrected is True


def test_reconcile_corrects_when_task_terminal_but_version_still_training(isolated) -> None:
    """task done but version is still stuck on training (supervisor missed a write) -> corrected to completed."""
    v = _make_version(isolated)
    _insert_task(isolated, v["id"], v["project_id"], "done")
    with db.connection_for(isolated["db"]) as conn:
        versions.update_version(conn, v["id"], status="training")
        ver, corrected = versions.reconcile_version_status(conn, v["id"])
    assert ver is not None
    assert ver["status"] == "completed"
    assert corrected is True


def test_reconcile_returns_none_for_missing_version(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        ver, corrected = versions.reconcile_version_status(conn, 99999)
    assert ver is None
    assert corrected is False
