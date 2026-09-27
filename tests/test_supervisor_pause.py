from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from studio import db
from studio.supervisor import _Slot, Supervisor


@pytest.fixture
def env(tmp_path: Path):
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    logs = tmp_path / "logs"
    configs = tmp_path / "configs"
    logs.mkdir()
    configs.mkdir()
    return {"db": db_path, "logs": logs, "configs": configs}


def _new_sup(env) -> Supervisor:
    return Supervisor(
        on_event=lambda _: None,
        cmd_builder=lambda *_: ["echo"],
        db_path=env["db"],
        logs_dir=env["logs"],
        configs_dir=env["configs"],
        poll_interval=10,
    )




def test_slot_has_pause_fields_with_safe_defaults() -> None:
    s = _Slot()
    assert s.pause_pending is False
    assert s.pause_state_path is None
    assert s.pause_config_path is None
    assert s.pause_step is None
    assert s.train_loop_started is False


def test_slot_reset_clears_pause_fields() -> None:
    s = _Slot()
    s.pause_pending = True
    s.pause_state_path = "/x/y.pt"
    s.pause_config_path = "/x/y.config.json"
    s.pause_step = 100
    s.train_loop_started = True
    s.reset()
    assert s.pause_pending is False
    assert s.pause_state_path is None
    assert s.pause_config_path is None
    assert s.pause_step is None
    assert s.train_loop_started is False




def _populate_running(env, **fields) -> int:
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        update = {"status": "running", "started_at": time.time()}
        update.update(fields)
        db.update_task(conn, tid, **update)
    return tid


def _make_slot_with_proc(tid: int) -> _Slot:
    slot = _Slot(name="train")
    slot.kind = "task"
    slot.id = tid
    slot.proc = MagicMock()
    slot.log_fp = None
    slot.tailer = None
    slot.state_poller = None
    return slot


def test_finish_slot_paused_when_pause_pending_and_state_path(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.pause_pending = True
    slot.pause_state_path = str(env["db"].parent / "pause_step_100.pt")
    slot.pause_config_path = str(env["db"].parent / "pause_step_100.config.json")
    slot.pause_step = 100
    expected_state = slot.pause_state_path
    expected_config = slot.pause_config_path

    sup._finish_slot(slot, rc=0)

    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task is not None
    assert task["status"] == "paused"
    assert task["paused_state_path"] == expected_state
    assert task["paused_config_path"] == expected_config
    assert task["paused_step"] == 100
    assert task["paused_at"] is not None


def test_finish_slot_canceled_when_pause_pending_but_no_state_path(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.pause_pending = True
    slot.cancel_pending = True

    sup._finish_slot(slot, rc=0)

    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task is not None
    assert task["status"] == "canceled"


def test_finish_slot_canceled_takes_precedence_over_rc(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.cancel_pending = True

    sup._finish_slot(slot, rc=0)
    assert _read_status(env["db"], tid) == "canceled"


def test_finish_slot_done_when_rc_zero(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)

    sup._finish_slot(slot, rc=0)
    assert _read_status(env["db"], tid) == "done"


def test_finish_slot_failed_when_rc_nonzero(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)

    sup._finish_slot(slot, rc=7)
    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task["status"] == "failed"
    assert task["exit_code"] == 7


def _read_status(db_path: Path, tid: int) -> str:
    with db.connection_for(db_path) as conn:
        t = db.get_task(conn, tid)
    return str(t["status"]) if t else ""




def test_pause_returns_false_for_unknown_task(env) -> None:
    sup = _new_sup(env)
    ok, reason = sup.pause(99999)
    assert ok is False
    assert "not found" in reason


def test_pause_returns_false_for_pending_task(env) -> None:
    sup = _new_sup(env)
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
    ok, reason = sup.pause(tid)
    assert ok is False
    assert "not running" in reason


def test_pause_returns_false_when_train_loop_not_started(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.train_loop_started = False
    sup._slots = [slot]

    ok, reason = sup.pause(tid)
    assert ok is False
    assert "train loop not started" in reason


def test_pause_succeeds_when_train_loop_started(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.train_loop_started = True
    sup._slots = [slot]

    ok, _reason = sup.pause(tid)
    assert ok is True
    assert slot.pause_pending is True
    slot.proc.send_signal.assert_called_once()


def test_pause_rejects_when_already_pausing(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.train_loop_started = True
    slot.pause_pending = True
    sup._slots = [slot]

    ok, reason = sup.pause(tid)
    assert ok is False
    assert "already pending" in reason


def test_pause_rejects_when_cancel_pending(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.train_loop_started = True
    slot.cancel_pending = True
    sup._slots = [slot]

    ok, reason = sup.pause(tid)
    assert ok is False
    assert "canceled" in reason


# ---- cancel paused task → canceled ------------------------------------------


def test_cancel_paused_task_changes_to_canceled_keeps_files(env, tmp_path) -> None:
    sup = _new_sup(env)
    state_pt = tmp_path / "auto_epoch_state.pt"
    state_cfg = tmp_path / "auto_epoch_state.config.json"
    state_pt.write_bytes(b"fake")
    state_cfg.write_text("{}", encoding="utf-8")

    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(
            conn, tid,
            status="paused",
            paused_state_path=str(state_pt),
            paused_config_path=str(state_cfg),
            last_state_path=str(state_pt),
            last_config_path=str(state_cfg),
            paused_step=100,
            paused_at=time.time(),
        )

    assert sup.cancel(tid) is True

    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task["status"] == "canceled"
    assert task["paused_state_path"] is None
    assert task["last_state_path"] == str(state_pt)
    assert state_pt.exists()
    assert state_cfg.exists()


def test_cancel_paused_task_robust_to_missing_files(env) -> None:
    sup = _new_sup(env)
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(
            conn, tid,
            status="paused",
            paused_state_path="/nonexistent/path.pt",
            paused_config_path="/nonexistent/path.config.json",
            paused_step=100,
        )
    assert sup.cancel(tid) is True
    assert _read_status(env["db"], tid) == "canceled"




def test_queue_held_returns_db_value(env) -> None:
    sup = _new_sup(env)
    assert sup._queue_held() is False
    with db.connection_for(env["db"]) as conn:
        db.set_queue_held(conn, True)
    assert sup._queue_held() is True
