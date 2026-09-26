from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from studio import db
from studio.services.projects import jobs as project_jobs
from studio.services.inference import daemon as _daemon_mod
from studio.supervisor import Supervisor


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from studio.infrastructure import paths as _paths
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    logs = tmp_path / "logs"
    configs = tmp_path / "configs"
    tasks = tmp_path / "tasks"
    logs.mkdir()
    configs.mkdir()
    monkeypatch.setattr(_paths, "TASKS_DIR", tasks)
    monkeypatch.setattr(_paths, "LOGS_DIR", logs)
    return {"db": db_path, "logs": logs, "configs": configs, "tasks": tasks}


@pytest.fixture
def fake_daemon():
    fake = MagicMock()
    fake.is_model_loaded = False
    fake.is_busy = False
    fake.state = "stopped"
    _daemon_mod._INSTANCE = fake  # type: ignore[attr-defined]
    yield fake
    _daemon_mod._INSTANCE = None  # type: ignore[attr-defined]


@pytest.fixture
def fake_secrets(monkeypatch):
    cfg = MagicMock()
    cfg.queue.light_tasks_during_train = True
    monkeypatch.setattr(
        "studio.supervisor._secrets.load", lambda: cfg
    )
    return cfg


def _make_train_task(env: dict) -> int:
    cfg_path = env["configs"] / "t.yaml"
    cfg_path.write_text("epochs: 1\n", encoding="utf-8")
    with db.connection_for(env["db"]) as conn:
        return db.create_task(conn, name="t", config_name="t")


def _make_tag_job(env: dict, *, slug: str = "p") -> int:
    with db.connection_for(env["db"]) as conn:
        from studio.services.projects import projects, versions
        p = projects.create_project(conn, title="P", slug=slug)
        v = versions.create_version(conn, project_id=p["id"], label="v1")
        job = project_jobs.create_job(
            conn,
            project_id=p["id"],
            version_id=v["id"],
            kind="tag",
            params={},
        )
        return int(job["id"])




def test_yield_no_daemon_loaded(env, fake_daemon, fake_secrets):
    fake_daemon.is_model_loaded = False
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    assert sup._maybe_yield_daemon() is False
    fake_daemon.request_unload.assert_not_called()


def test_yield_daemon_idle_but_loaded(env, fake_daemon, fake_secrets):
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = False
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    assert sup._maybe_yield_daemon() is True
    fake_daemon.request_unload.assert_called_once()


def test_yield_daemon_busy_no_unload(env, fake_daemon, fake_secrets):
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = True
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    assert sup._maybe_yield_daemon() is True
    fake_daemon.request_unload.assert_not_called()


def test_yield_ignores_light_switch(env, fake_daemon, fake_secrets):
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = False
    fake_secrets.queue.light_tasks_during_train = True
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    assert sup._maybe_yield_daemon() is True
    fake_daemon.request_unload.assert_called_once()




def test_dispatch_train_skipped_while_daemon_loaded(env, fake_daemon, fake_secrets):
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = False

    spawned: list[Any] = []
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    monkey_spawn = sup._spawn_task
    sup._spawn_task = lambda slot, task: spawned.append(task)  # type: ignore

    _make_train_task(env)
    train_slot = next(s for s in sup._slots if s.name == "train")
    sup._dispatch_exclusive_tasks(train_slot)
    assert spawned == [], "train task should not be spawned while daemon holds GPU"
    fake_daemon.request_unload.assert_called_once()


def test_dispatch_train_proceeds_after_daemon_unloaded(env, fake_daemon, fake_secrets):
    fake_daemon.is_model_loaded = False
    fake_daemon.is_busy = False

    spawned: list[Any] = []
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    sup._spawn_task = lambda slot, task: spawned.append(task)  # type: ignore

    _make_train_task(env)
    train_slot = next(s for s in sup._slots if s.name == "train")
    sup._dispatch_exclusive_tasks(train_slot)
    assert len(spawned) == 1
    fake_daemon.request_unload.assert_not_called()




def test_dispatch_data_tag_job_yields_to_daemon_load_when_disabled(
    env, fake_daemon, fake_secrets,
):
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = False
    fake_secrets.queue.light_tasks_during_train = False

    spawned_jobs: list[Any] = []
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    sup._spawn_job = lambda slot, job: spawned_jobs.append(job)  # type: ignore

    _make_tag_job(env)
    data_slot = next(s for s in sup._slots if s.name == "data")
    sup._dispatch_data(data_slot)
    assert spawned_jobs == []
    fake_daemon.request_unload.assert_called_once()


def test_dispatch_data_tag_job_runs_alongside_daemon_by_default(
    env, fake_daemon, fake_secrets,
):
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = False

    spawned_jobs: list[Any] = []
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    sup._spawn_job = lambda slot, job: spawned_jobs.append(job)  # type: ignore

    _make_tag_job(env)
    data_slot = next(s for s in sup._slots if s.name == "data")
    sup._dispatch_data(data_slot)
    assert len(spawned_jobs) == 1
    fake_daemon.request_unload.assert_not_called()


def test_dispatch_data_download_not_blocked_by_daemon(env, fake_daemon, fake_secrets):
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = False

    spawned_jobs: list[Any] = []
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    sup._spawn_job = lambda slot, job: spawned_jobs.append(job)  # type: ignore

    with db.connection_for(env["db"]) as conn:
        from studio.services.projects import projects, versions
        p = projects.create_project(conn, title="P2", slug="p2")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
        project_jobs.create_job(
            conn,
            project_id=p["id"],
            version_id=v["id"],
            kind="download",
            params={},
        )

    data_slot = next(s for s in sup._slots if s.name == "data")
    sup._dispatch_data(data_slot)
    assert len(spawned_jobs) == 1, "download should run regardless of daemon state"
    fake_daemon.request_unload.assert_not_called()
