from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest

from studio import db
from studio.services import task_snapshot
from studio.services.projects import projects, versions
from studio.supervisor import Supervisor, _maybe_finalize_version


def _wait_for(predicate, timeout: float = 8.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from studio.infrastructure import paths as _paths
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(db, "STUDIO_DB", db_path)
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "studio_data" / "tasks")
    monkeypatch.setattr(_paths, "LOGS_DIR", tmp_path / "logs")

    logs = tmp_path / "logs"
    configs = tmp_path / "configs"
    logs.mkdir()
    configs.mkdir()
    (configs / "fake.yaml").write_text("epochs: 1\nlr: 0.001\n", encoding="utf-8")

    with db.connection_for(db_path) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="baseline")

    return {
        "db": db_path,
        "logs": logs,
        "configs": configs,
        "project": p,
        "version": v,
    }


def _create_versioned_task(env, name: str = "t1") -> int:
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name=name, config_name="fake")
        db.update_task(
            conn, tid,
            project_id=env["project"]["id"],
            version_id=env["version"]["id"],
        )
    return tid


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_finalize_done_sets_completed(env) -> None:
    tid = _create_versioned_task(env)
    with db.connection_for(env["db"]) as conn:
        _maybe_finalize_version(conn, tid, "done")
        v = versions.get_version(conn, env["version"]["id"])
    assert v["status"] == "completed"


def test_finalize_failed_sets_failed_and_writes_reason(env) -> None:
    tid = _create_versioned_task(env)
    with db.connection_for(env["db"]) as conn:
        db.update_task(conn, tid, status="failed", error_msg="OOM at step 500")
        _maybe_finalize_version(conn, tid, "failed")
        v = versions.get_version(conn, env["version"]["id"])
    assert v["status"] == "failed"
    assert v["last_failure_reason"] == "OOM at step 500"


def test_finalize_canceled_sets_canceled(env) -> None:
    tid = _create_versioned_task(env)
    with db.connection_for(env["db"]) as conn:
        _maybe_finalize_version(conn, tid, "canceled")
        v = versions.get_version(conn, env["version"]["id"])
    assert v["status"] == "canceled"


def test_finalize_paused_does_not_change_version(env) -> None:
    tid = _create_versioned_task(env)
    with db.connection_for(env["db"]) as conn:
        versions.update_version(conn, env["version"]["id"], status="training")
        _maybe_finalize_version(conn, tid, "paused")
        v = versions.get_version(conn, env["version"]["id"])
    assert v["status"] == "training"


def test_finalize_no_version_id_is_noop(env) -> None:
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="orphan", config_name="fake")
        _maybe_finalize_version(conn, tid, "done")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_done_task_sets_version_completed_and_creates_snapshot(env) -> None:
    def fast_cmd(task: dict[str, Any], cfg: Path) -> list[str]:
        return [sys.executable, "-c", "import sys; sys.exit(0)"]

    sup = Supervisor(
        cmd_builder=fast_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )

    tid = _create_versioned_task(env)
    sup.start()
    try:
        assert _wait_for(
            lambda: db.get_task(_open(env["db"]), tid)["status"] == "done",
            timeout=10,
        )
    finally:
        sup.stop()

    with db.connection_for(env["db"]) as conn:
        v = versions.get_version(conn, env["version"]["id"])
    assert v["status"] == "completed"

    snap = task_snapshot.snapshot_config_path(tid)
    assert snap.exists()
    assert "epochs: 1" in snap.read_text(encoding="utf-8")


def test_failed_task_sets_version_failed(env) -> None:
    def fail_cmd(task: dict[str, Any], cfg: Path) -> list[str]:
        return [sys.executable, "-c", "import sys; sys.exit(1)"]

    sup = Supervisor(
        cmd_builder=fail_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )

    tid = _create_versioned_task(env)
    sup.start()
    try:
        assert _wait_for(
            lambda: db.get_task(_open(env["db"]), tid)["status"] == "failed",
            timeout=10,
        )
    finally:
        sup.stop()

    with db.connection_for(env["db"]) as conn:
        v = versions.get_version(conn, env["version"]["id"])
    assert v["status"] == "failed"
    assert v["last_failure_reason"] is not None


def test_spawn_sets_version_training_then_terminal(env) -> None:
    def slow_cmd(task: dict[str, Any], cfg: Path) -> list[str]:
        return [sys.executable, "-c", "import time; time.sleep(0.3)"]

    sup = Supervisor(
        cmd_builder=slow_cmd,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )

    tid = _create_versioned_task(env)
    sup.start()
    try:
        assert _wait_for(
            lambda: db.get_task(_open(env["db"]), tid)["status"] == "running",
            timeout=10,
        )
        with db.connection_for(env["db"]) as conn:
            v = versions.get_version(conn, env["version"]["id"])
        assert v["status"] == "training"

        assert _wait_for(
            lambda: db.get_task(_open(env["db"]), tid)["status"] == "done",
            timeout=10,
        )
    finally:
        sup.stop()

    with db.connection_for(env["db"]) as conn:
        v = versions.get_version(conn, env["version"]["id"])
    assert v["status"] == "completed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open(db_path: Path):
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn
