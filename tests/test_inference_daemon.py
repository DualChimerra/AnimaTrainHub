from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from studio import db
from studio.services.inference import daemon as _daemon_mod
from studio.services.inference.daemon import (
    InferenceDaemon,
    STATE_BUSY,
    STATE_IDLE,
    STATE_STOPPED,
    reset_daemon_for_test,
)



_MOCK_DAEMON = textwrap.dedent(
    """
    import base64, json, sys, os, time
    sys.stdout.write(json.dumps({"id":"_evt","kind":"ready"}) + "\\n")
    sys.stdout.flush()
    fake_b64 = base64.b64encode(b"FAKE-PNG").decode("ascii")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        action = msg.get("action")
        rid = msg.get("id", "")
        if action == "ping":
            sys.stdout.write(json.dumps({"id":rid,"kind":"pong"}) + "\\n")
            sys.stdout.flush()
        elif action == "generate":
            tid = msg.get("task_id", 0)
            sys.stdout.write(json.dumps({"id":rid,"kind":"started","task_id":tid}) + "\\n")
            sys.stdout.flush()
            if msg.get("config", {}).get("wait_for_cancel"):
                continue
            # emits 1 image: bytes go through the protocol's b64 field (commit 10), no disk write
            sys.stdout.write(json.dumps({"id":rid,"kind":"image_done","task_id":tid,"filename":"fake.png","path":"/anima_gen_%d/fake.png" % tid,"step":1,"total":1,"image_b64":fake_b64,"byte_size":8}) + "\\n")
            sys.stdout.flush()
            sys.stdout.write(json.dumps({"id":rid,"kind":"done","task_id":tid}) + "\\n")
            sys.stdout.flush()
        elif action == "cancel":
            target = msg.get("target_id") or rid
            sys.stdout.write(json.dumps({"id":target,"kind":"canceled","task_id":0}) + "\\n")
            sys.stdout.flush()
        elif action == "unload":
            sys.stdout.write(json.dumps({"id":"_evt","kind":"unloaded"}) + "\\n")
            sys.stdout.flush()
        elif action == "crash":
            os._exit(1)
    """
).strip()


@pytest.fixture
def mock_daemon_script(tmp_path: Path) -> Path:
    p = tmp_path / "mock_daemon.py"
    p.write_text(_MOCK_DAEMON, encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _reset_daemon():
    reset_daemon_for_test()
    yield
    reset_daemon_for_test()


def _wait_for(predicate, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False




def test_daemon_starts_and_reaches_idle(mock_daemon_script: Path) -> None:
    d = InferenceDaemon(script_path=mock_daemon_script)
    d.start()
    try:
        assert d.state == STATE_IDLE
        assert d.is_alive
    finally:
        d.stop()
    assert d.state == STATE_STOPPED


def test_submit_task_runs_to_done(mock_daemon_script: Path, tmp_path: Path) -> None:
    from studio.services.inference import disk_cache as generate_cache

    generate_cache.init(tmp_path / "cache")
    d = InferenceDaemon(script_path=mock_daemon_script)
    d.start()
    events: list[dict[str, Any]] = []
    try:
        d.submit_task(
            task_id=42, config={"prompts": ["a"]}, output_dir="/tmp/x",
            on_event=events.append,
        )
        assert d.state == STATE_BUSY
        assert _wait_for(
            lambda: any(e.get("kind") == "done" for e in events), timeout=3
        ), f"events={events}"
        assert _wait_for(lambda: d.state == STATE_IDLE, timeout=2)
    finally:
        d.stop()
    kinds = [e.get("kind") for e in events]
    assert "started" in kinds
    assert "image_done" in kinds
    assert "done" in kinds
    for e in events:
        assert e.get("task_id") == 42

    assert generate_cache.get_image(42, "fake.png") == b"FAKE-PNG"
    image_done_events = [e for e in events if e.get("kind") == "image_done"]
    assert image_done_events
    for e in image_done_events:
        assert "image_b64" not in e
    for e in image_done_events:
        assert e.get("delivery") == "cache"
    generate_cache.clear_all()


def test_submit_task_delivery_disk_when_save_to_disk_true(
    mock_daemon_script: Path,
    tmp_path: Path,
) -> None:
    from studio.services.inference import disk_cache as generate_cache

    generate_cache.init(tmp_path / "cache")
    d = InferenceDaemon(script_path=mock_daemon_script)
    d.start()
    events: list[dict[str, Any]] = []
    try:
        d.submit_task(
            task_id=43,
            config={"prompts": ["a"], "save_test_images_at_dispatch": True},
            output_dir="/tmp/x",
            on_event=events.append,
        )
        assert _wait_for(
            lambda: any(e.get("kind") == "done" for e in events), timeout=3
        ), f"events={events}"
    finally:
        d.stop()
    image_done_events = [e for e in events if e.get("kind") == "image_done"]
    assert image_done_events
    for e in image_done_events:
        assert e.get("delivery") == "disk"
    generate_cache.clear_all()


def test_daemon_crash_emits_error(mock_daemon_script: Path) -> None:
    d = InferenceDaemon(script_path=mock_daemon_script)
    d.start()
    events: list[dict[str, Any]] = []
    try:
        with d._lock:  # type: ignore[attr-defined]
            d._req_seq += 1
            req_id = "task-99-x"
            from studio.services.inference.daemon import _ActiveTask
            d._active = _ActiveTask(
                task_id=99, request_id=req_id, on_event=events.append,
            )
            d._state = STATE_BUSY
            stdin = d._proc.stdin  # type: ignore[union-attr]
        stdin.write(json.dumps({"id": req_id, "action": "crash"}) + "\n")
        stdin.flush()
        assert _wait_for(
            lambda: any(e.get("kind") == "error" for e in events), timeout=3
        ), f"events={events}"
    finally:
        d.stop()
    assert d.state == STATE_STOPPED


def test_global_listener_receives_events(mock_daemon_script: Path) -> None:
    d = InferenceDaemon(script_path=mock_daemon_script)
    seen: list[dict[str, Any]] = []
    d.add_global_listener(seen.append)
    d.start()
    try:
        assert _wait_for(
            lambda: any(e.get("kind") == "ready" for e in seen), timeout=2
        )
    finally:
        d.stop()
    assert _wait_for(
        lambda: any(e.get("kind") == "stopped" for e in seen), timeout=2
    ), f"seen={seen}"




def _make_generate_task(env: dict, *, cfg_overrides: dict[str, Any] | None = None) -> int:
    cfg_dir = env["configs"]
    cfg_path = cfg_dir / "gen.json"
    cfg = {
        "transformer_path": "/x", "vae_path": "/y", "text_encoder_path": "/z",
        "prompts": ["a"], "output_dir": str(env["configs"] / "out"),
    }
    if cfg_overrides:
        cfg.update(cfg_overrides)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="g", config_name="gen", priority=0)
        db.update_task(conn, tid, task_type="generate", config_path=str(cfg_path))
    return tid


def _patch_singleton(d: InferenceDaemon, monkeypatch) -> None:
    _daemon_mod._INSTANCE = d  # type: ignore[attr-defined]


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


def _task_status(db_path: Path, task_id: int) -> str:
    with db.connection_for(db_path) as conn:
        t = db.get_task(conn, task_id)
    return (t or {}).get("status", "?")


def test_supervisor_dispatches_generate_to_daemon(env, mock_daemon_script, monkeypatch):
    from studio.supervisor import Supervisor

    d = InferenceDaemon(script_path=mock_daemon_script)
    _patch_singleton(d, monkeypatch)

    events: list[dict[str, Any]] = []
    sup = Supervisor(
        on_event=events.append,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )

    tid = _make_generate_task(env)
    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "done", timeout=10
        ), f"final status={_task_status(env['db'], tid)}; events={events}"
    finally:
        sup.stop()

    statuses = [e["status"] for e in events if e.get("task_id") == tid]
    assert "running" in statuses
    assert "done" in statuses

    daemon_evts = [e for e in events if e.get("type") == "daemon_state_changed"]
    assert daemon_evts, "expected daemon_state_changed events; got none"
    busy_states = [e["busy"] for e in daemon_evts]
    assert True in busy_states
    assert False in busy_states


def test_supervisor_cancel_pending_generate(env, mock_daemon_script, monkeypatch):
    from studio.supervisor import Supervisor

    d = InferenceDaemon(script_path=mock_daemon_script)
    _patch_singleton(d, monkeypatch)
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    tid = _make_generate_task(env)
    assert sup.cancel(tid) is True
    assert _task_status(env["db"], tid) == "canceled"


def test_supervisor_cancel_running_generate_keeps_daemon_alive(env, mock_daemon_script, monkeypatch):
    from studio.supervisor import Supervisor

    d = InferenceDaemon(script_path=mock_daemon_script)
    _patch_singleton(d, monkeypatch)
    events: list[dict[str, Any]] = []
    sup = Supervisor(
        on_event=events.append,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    tid = _make_generate_task(env, cfg_overrides={"wait_for_cancel": True})

    sup.start()
    try:
        assert _wait_for(lambda: _task_status(env["db"], tid) == "running", timeout=5)
        assert d.is_alive
        assert d.state == STATE_BUSY

        assert sup.cancel(tid) is True
        assert _wait_for(lambda: _task_status(env["db"], tid) == "canceled", timeout=5)
        assert d.is_alive
        assert d.state == STATE_IDLE
    finally:
        sup.stop()

    daemon_evts = [e for e in events if e.get("type") == "daemon_state_changed"]
    assert daemon_evts
    assert any(e.get("busy") is False for e in daemon_evts)


def test_supervisor_train_dispatch_skips_generate(env, monkeypatch):
    from studio.supervisor import Supervisor

    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    tid_gen = _make_generate_task(env)
    assert sup._next_pending_task_in(("train", "reg_ai")) is None
    found = sup._next_pending_task_in(("generate",))
    assert found is not None and found["id"] == tid_gen


def test_dispatch_generate_skips_task_without_config_path(env, monkeypatch):
    from studio.supervisor import Supervisor

    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    submitted: list[int] = []
    monkeypatch.setattr(sup, "_submit_to_daemon", lambda t: submitted.append(t["id"]))
    train_slot = next(s for s in sup._slots if s.name == "train")

    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="g", config_name="gen", priority=0)
        db.update_task(conn, tid, task_type="generate")

    sup._dispatch_exclusive_tasks(train_slot)
    assert submitted == [], "a generate task with config_path=NULL should not be submitted"
    assert _task_status(env["db"], tid) == "pending", "should remain pending, not failed"

    with db.connection_for(env["db"]) as conn:
        db.update_task(conn, tid, config_path=str(env["configs"] / "gen.json"))
    sup._dispatch_exclusive_tasks(train_slot)
    assert submitted == [tid]




def test_task_timeout_kills_daemon_and_emits_error(
    mock_daemon_script: Path, tmp_path: Path,
) -> None:
    d = InferenceDaemon(script_path=mock_daemon_script)
    d.start()
    assert _wait_for(lambda: d.state == "idle")
    with d._lock:
        d._task_timeout_seconds = 0.5

    events: list[dict] = []
    d.submit_task(
        task_id=99, config={"wait_for_cancel": True},
        output_dir=str(tmp_path), on_event=events.append,
    )
    assert _wait_for(
        lambda: any(e.get("kind") == "error" for e in events), timeout=6.0,
    )
    assert not d.is_alive
    assert d.state == "stopped"


def test_task_timer_cleared_on_normal_done(
    mock_daemon_script: Path, tmp_path: Path,
) -> None:
    d = InferenceDaemon(script_path=mock_daemon_script)
    d.start()
    assert _wait_for(lambda: d.state == "idle")
    with d._lock:
        d._task_timeout_seconds = 30.0

    events: list[dict] = []
    d.submit_task(
        task_id=100, config={}, output_dir=str(tmp_path), on_event=events.append,
    )
    assert _wait_for(lambda: any(e.get("kind") == "done" for e in events))
    with d._lock:
        assert d._task_timer is None
    assert d.is_alive
    d.stop()
