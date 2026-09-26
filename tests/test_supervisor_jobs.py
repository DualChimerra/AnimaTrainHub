from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from studio import db
from studio.services.projects import jobs as project_jobs, projects
from studio.supervisor import Supervisor


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from studio.infrastructure import paths as _paths
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(_paths, "LOGS_DIR", tmp_path / "logs")
    return {"db": dbfile, "logs": tmp_path / "logs", "tasks": tmp_path / "tasks"}


def _wait_until(pred, timeout: float = 5.0, step: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(step)
    return False


def _setup_project(isolated) -> dict:
    with db.connection_for(isolated["db"]) as conn:
        return projects.create_project(conn, title="P")


def test_eval_metrics_not_blocked_by_earlier_eval_samples(isolated) -> None:
    from unittest.mock import MagicMock
    p = _setup_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        project_jobs.create_job(
            conn, project_id=p["id"], kind="eval_samples",
            params={"run_id": "run-later"},
        )
        clip = project_jobs.create_job(
            conn, project_id=p["id"], kind="eval_clip",
            params={"run_id": "run-ready"},
        )

    from studio.services.inference import daemon as _daemon_mod
    fake = MagicMock(); fake.is_model_loaded = False; fake.is_busy = False
    _daemon_mod._INSTANCE = fake  # type: ignore[attr-defined]
    try:
        sup = Supervisor(
            on_event=lambda _e: None,
            db_path=isolated["db"], logs_dir=isolated["logs"],
        )
        spawned: list = []
        sup._spawn_job = lambda slot, job: spawned.append(job)  # type: ignore
        data_slot = next(s for s in sup._slots if s.name == "data")
        sup._dispatch_data(data_slot)
    finally:
        _daemon_mod._INSTANCE = None  # type: ignore[attr-defined]

    assert [j["id"] for j in spawned] == [clip["id"]], \
        "the light-tier metric is not blocked by an earlier-queued exclusive-tier eval_samples"


def test_download_job_runs_in_parallel_with_training_task(isolated, tmp_path) -> None:
    p = _setup_project(isolated)
    events: list[dict] = []
    task_sleep = lambda t, _cfg: [
        sys.executable, "-c", "import time; time.sleep(0.5)"
    ]
    job_sleep = lambda j: [sys.executable, "-c", "import time; time.sleep(0.5)"]
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "fake.yaml").write_text("epochs: 1\n", encoding="utf-8")

    sup = Supervisor(
        on_event=events.append,
        cmd_builder=task_sleep,
        job_cmd_builder=job_sleep,
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        configs_dir=configs,
        poll_interval=0.05,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(conn, name="t1", config_name="fake")
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="download", params={}
        )
    sup.start()
    try:
        assert _wait_until(
            lambda: any(
                e.get("type") == "task_state_changed" and e.get("status") == "running"
                for e in events
            )
            and any(
                e.get("type") == "job_state_changed" and e.get("status") == "running"
                for e in events
            ),
            timeout=5.0,
        ), "the task and the download job did not both enter running (still serialized?)"
    finally:
        sup.stop(timeout=10.0)


def test_light_job_deferred_during_training_when_disabled(
    isolated, tmp_path, monkeypatch,
) -> None:
    from studio import secrets as _sec
    secrets_file = tmp_path / "secrets.json"
    monkeypatch.setattr(_sec, "SECRETS_FILE", secrets_file)
    sec = _sec.Secrets()
    sec.queue.light_tasks_during_train = False
    _sec.save(sec)

    p = _setup_project(isolated)
    events: list[dict] = []
    task_sleep = lambda t, _cfg: [
        sys.executable, "-c", "import time; time.sleep(0.6)"
    ]
    job_quick = lambda j: [sys.executable, "-c", "print('tag done')"]
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "fake.yaml").write_text("epochs: 1\n", encoding="utf-8")

    sup = Supervisor(
        on_event=events.append,
        cmd_builder=task_sleep,
        job_cmd_builder=job_quick,
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        configs_dir=configs,
        poll_interval=0.05,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(conn, name="t1", config_name="fake")
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="tag", params={}
        )
    sup.start()
    try:
        assert _wait_until(
            lambda: any(
                e.get("type") == "task_state_changed" and e.get("status") == "running"
                for e in events
            ),
            timeout=5.0,
        )
        time.sleep(0.3)
        with db.connection_for(isolated["db"]) as conn:
            job_now = project_jobs.get_job(conn, job["id"])
        assert job_now["status"] == "pending", \
            f"tag job should not start while training, current status={job_now['status']}"
        assert _wait_until(
            lambda: project_jobs.get_job(
                db.connect(isolated["db"]), job["id"]
            )["status"] == "done",
            timeout=10.0,
        )
    finally:
        sup.stop(timeout=10.0)


def test_light_job_runs_during_training_by_default(isolated, tmp_path, monkeypatch) -> None:
    from studio import secrets as _sec
    secrets_file = tmp_path / "secrets.json"
    monkeypatch.setattr(_sec, "SECRETS_FILE", secrets_file)
    _sec.save(_sec.Secrets())

    p = _setup_project(isolated)
    events: list[dict] = []
    task_sleep = lambda t, _cfg: [
        sys.executable, "-c", "import time; time.sleep(0.5)"
    ]
    job_sleep = lambda j: [sys.executable, "-c", "import time; time.sleep(0.5)"]
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "fake.yaml").write_text("epochs: 1\n", encoding="utf-8")

    sup = Supervisor(
        on_event=events.append,
        cmd_builder=task_sleep,
        job_cmd_builder=job_sleep,
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        configs_dir=configs,
        poll_interval=0.05,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(conn, name="t1", config_name="fake")
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="tag", params={}
        )
    sup.start()
    try:
        assert _wait_until(
            lambda: any(
                e.get("type") == "task_state_changed" and e.get("status") == "running"
                for e in events
            )
            and any(
                e.get("type") == "job_state_changed" and e.get("status") == "running"
                for e in events
            ),
            timeout=5.0,
        ), "the tag job is still deferred after the switch was turned on"
    finally:
        sup.stop(timeout=10.0)


def test_job_lifecycle_done(isolated) -> None:
    p = _setup_project(isolated)
    events: list[dict] = []
    sup = Supervisor(
        on_event=events.append,
        job_cmd_builder=lambda j: [sys.executable, "-c", "print('hello'); print('bye')"],
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        poll_interval=0.05,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="download", params={}
        )
    sup.start()
    try:
        assert _wait_until(
            lambda: any(
                e.get("type") == "job_state_changed" and e.get("status") == "done"
                for e in events
            )
        )
    finally:
        sup.stop(timeout=5.0)

    with db.connection_for(isolated["db"]) as conn:
        finished = project_jobs.get_job(conn, job["id"])
    assert finished["status"] == "done"
    assert finished["finished_at"] is not None

    log_lines = [e for e in events if e.get("type") == "job_log_appended"]
    assert any("hello" in (e.get("text") or "") for e in log_lines)


def test_job_lifecycle_failed(isolated) -> None:
    p = _setup_project(isolated)
    events: list[dict] = []
    sup = Supervisor(
        on_event=events.append,
        job_cmd_builder=lambda j: [sys.executable, "-c", "import sys; sys.exit(2)"],
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        poll_interval=0.05,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="download", params={}
        )
    sup.start()
    try:
        assert _wait_until(
            lambda: any(
                e.get("type") == "job_state_changed" and e.get("status") == "failed"
                for e in events
            )
        )
    finally:
        sup.stop(timeout=5.0)
    with db.connection_for(isolated["db"]) as conn:
        finished = project_jobs.get_job(conn, job["id"])
    assert finished["status"] == "failed"
    assert "exit code 2" in (finished["error_msg"] or "")


def test_cancel_pending_job(isolated) -> None:
    p = _setup_project(isolated)
    sup = Supervisor(
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        poll_interval=10.0,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="download", params={}
        )
    assert sup.cancel(job["id"]) is True
    with db.connection_for(isolated["db"]) as conn:
        got = project_jobs.get_job(conn, job["id"])
    assert got["status"] == "canceled"


def test_worker_event_marker_lines_publish_typed_events(isolated) -> None:
    p = _setup_project(isolated)
    events: list[dict] = []
    cmd = [
        sys.executable, "-c",
        'import json; print("hello"); '
        'print("__EVENT__:preprocess_progress:" + json.dumps({"idx":1,"total":3,"status":"done"})); '
        'print("__EVENT__:preprocess_progress:" + json.dumps({"idx":2,"total":3,"status":"done"}))',
    ]
    sup = Supervisor(
        on_event=events.append,
        job_cmd_builder=lambda _j: cmd,
        db_path=isolated["db"],
        logs_dir=isolated["logs"],
        poll_interval=0.05,
        terminate_grace=2.0,
    )
    with db.connection_for(isolated["db"]) as conn:
        job = project_jobs.create_job(
            conn, project_id=p["id"], kind="download", params={}
        )
    sup.start()
    try:
        assert _wait_until(
            lambda: any(
                e.get("type") == "job_state_changed" and e.get("status") == "done"
                for e in events
            )
        )
    finally:
        sup.stop(timeout=5.0)

    progress = [e for e in events if e.get("type") == "preprocess_progress"]
    assert len(progress) == 2
    assert progress[0]["idx"] == 1 and progress[0]["total"] == 3
    assert progress[1]["idx"] == 2
    assert progress[0]["job_id"] == job["id"]
    assert progress[0]["project_id"] == p["id"]

    log_texts = [e.get("text", "") for e in events if e.get("type") == "job_log_appended"]
    assert any("hello" in t for t in log_texts)
    assert not any("__EVENT__" in t for t in log_texts)
