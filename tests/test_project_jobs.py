"""PP2 — project_jobs DAO + log_path + status transitions."""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db
from studio.services.projects import jobs as project_jobs, projects


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    # create a parent project so the FK is valid
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
    return {"db": dbfile, "project_id": p["id"]}


def test_create_job_assigns_log_path(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn,
            project_id=isolated['project_id'],
            kind="download",
            params={"tag": "x", "count": 5},
        )
    assert job["status"] == "pending"
    # R-3 ledger merge: job logs use the same tasks/<id>/run.log layout as GPU tasks
    assert job["log_path"].replace("\\", "/").endswith(f"{job['id']}/run.log")
    assert job["params_decoded"] == {"tag": "x", "count": 5}


def test_create_rejects_unknown_kind(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(project_jobs.JobError):
            project_jobs.create_job(
                conn, project_id=isolated['project_id'], kind="bogus", params={}
            )


def test_create_accepts_preprocess_kind(isolated) -> None:
    """preprocess added to VALID_KINDS (the unified job kind for upscale / crop / inpaint)."""
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn,
            project_id=isolated["project_id"],
            kind="preprocess",
            params={"mode": "all", "model": "4x-AnimeSharp"},
        )
    assert job["status"] == "pending"
    assert job["kind"] == "preprocess"


def test_status_transitions(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn, project_id=isolated['project_id'], kind="download", params={}
        )
        project_jobs.mark_running(conn, job["id"], pid=1234)
        running = project_jobs.get_job(conn, job["id"])
        assert running["status"] == "running"
        assert running["pid"] == 1234
        assert running["started_at"] is not None
        project_jobs.mark_done(conn, job["id"])
        done = project_jobs.get_job(conn, job["id"])
        assert done["status"] == "done"
        assert done["finished_at"] is not None


def test_mark_failed_sets_error_msg(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn, project_id=isolated['project_id'], kind="download", params={}
        )
        project_jobs.mark_failed(conn, job["id"], "boom")
        got = project_jobs.get_job(conn, job["id"])
    assert got["status"] == "failed"
    assert got["error_msg"] == "boom"


def test_list_pending_fifo_picks_oldest_first(isolated) -> None:
    """R-3: dispatch order = priority DESC, created_at ASC (same FIFO semantics as GPU tasks)."""
    with db.connection_for(isolated["db"]) as conn:
        a = project_jobs.create_job(conn, project_id=isolated['project_id'], kind="download", params={})
        b = project_jobs.create_job(conn, project_id=isolated['project_id'], kind="download", params={})
        db.update_task(conn, a["id"], created_at=100.0)
        db.update_task(conn, b["id"], created_at=200.0)
        pending = project_jobs.list_pending_fifo(conn)
    assert [j["id"] for j in pending] == [a["id"], b["id"]]


def test_latest_for_returns_most_recent(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        a = project_jobs.create_job(conn, project_id=isolated['project_id'], kind="download", params={})
        b = project_jobs.create_job(conn, project_id=isolated['project_id'], kind="download", params={})
        latest = project_jobs.latest_for(conn, project_id=isolated['project_id'], kind="download")
    assert latest and latest["id"] == b["id"]


def test_cleanup_orphan_running_only_touches_legacy_table(isolated) -> None:
    """R-3: jobs have been merged into tasks (orphans are now reaped uniformly by the supervisor task);
    cleanup_orphan_running only backstops the old legacy project_jobs table."""
    with db.connection_for(isolated["db"]) as conn:
        # new job (tasks table) running -- cleanup doesn't touch it
        job = project_jobs.create_job(
            conn, project_id=isolated['project_id'], kind="download", params={}
        )
        project_jobs.mark_running(conn, job["id"])
        # leftover running row in the old table -- cleanup marks it failed
        conn.execute(
            "INSERT INTO project_jobs(project_id, version_id, kind, params, status) "
            "VALUES (?, NULL, 'download', '{}', 'running')",
            (isolated["project_id"],),
        )
        conn.commit()
        n = project_jobs.cleanup_orphan_running(conn)
        got = project_jobs.get_job(conn, job["id"])
    assert n == 1
    assert got["status"] == "running", "new jobs in the tasks table should not be touched by legacy cleanup"


def test_list_jobs_filters(isolated) -> None:
    pid = isolated["project_id"]
    with db.connection_for(isolated["db"]) as conn:
        # second project, to verify filtering by project_id
        other = projects.create_project(conn, title="Other")
        project_jobs.create_job(conn, project_id=pid, kind="download", params={})
        b = project_jobs.create_job(conn, project_id=pid, kind="download", params={})
        project_jobs.create_job(conn, project_id=other["id"], kind="download", params={})
        project_jobs.mark_done(conn, b["id"])
        pending = project_jobs.list_jobs(conn, project_id=pid, status="pending")
    assert len(pending) == 1
    assert pending[0]["status"] == "pending"
