from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from studio.services.inference.daemon import (
    InferenceDaemon,
    STATE_BUSY,
    STATE_IDLE,
    STATE_STOPPED,
    STATE_UNLOADING,
)


def _fake_proc() -> SimpleNamespace:
    return SimpleNamespace(stdin=None, poll=lambda: None)


def _make_daemon_in_state(
    *,
    state: str = STATE_IDLE,
    model_loaded: bool = True,
    with_proc: bool = True,
) -> InferenceDaemon:
    d = InferenceDaemon()
    if with_proc:
        d._proc = _fake_proc()  # type: ignore[assignment]
    d._state = state
    d._model_loaded = model_loaded
    return d


def _wait_until(predicate, timeout=2.0, interval=0.01) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False




def test_timer_does_not_arm_when_timeout_zero() -> None:
    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(0)
    assert d._idle_timer is None


def test_timer_does_not_arm_when_model_not_loaded() -> None:
    d = _make_daemon_in_state(model_loaded=False)
    d.set_idle_timeout_seconds(5.0)
    assert d._idle_timer is None


def test_timer_does_not_arm_when_busy() -> None:
    d = _make_daemon_in_state(state=STATE_BUSY)
    d.set_idle_timeout_seconds(5.0)
    assert d._idle_timer is None


def test_timer_does_not_arm_when_no_proc() -> None:
    d = _make_daemon_in_state(with_proc=False)
    d.set_idle_timeout_seconds(5.0)
    assert d._idle_timer is None


def test_timer_arms_when_idle_and_loaded() -> None:
    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(60.0)
    try:
        assert d._idle_timer is not None
        assert d._idle_timer.is_alive()
    finally:
        if d._idle_timer is not None:
            d._idle_timer.cancel()




def test_timer_fires_request_unload(monkeypatch: pytest.MonkeyPatch) -> None:
    d = _make_daemon_in_state()
    calls: list[None] = []

    def fake_unload() -> None:
        calls.append(None)

    monkeypatch.setattr(d, "request_unload", fake_unload)
    d.set_idle_timeout_seconds(0.05)

    assert _wait_until(lambda: len(calls) >= 1, timeout=1.0)
    assert len(calls) == 1


def test_timer_does_not_fire_if_state_changed_before_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = _make_daemon_in_state()
    calls: list[None] = []
    monkeypatch.setattr(d, "request_unload", lambda: calls.append(None))
    d.set_idle_timeout_seconds(0.05)
    with d._lock:
        d._state = STATE_BUSY
    time.sleep(0.2)
    assert calls == []


def test_set_idle_timeout_zero_cancels_existing_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = _make_daemon_in_state()
    calls: list[None] = []
    monkeypatch.setattr(d, "request_unload", lambda: calls.append(None))
    d.set_idle_timeout_seconds(0.1)
    assert d._idle_timer is not None
    d.set_idle_timeout_seconds(0)
    assert d._idle_timer is None
    time.sleep(0.2)
    assert calls == []




def test_busy_then_idle_rearms_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(60.0)
    first_timer = d._idle_timer
    assert first_timer is not None

    with d._lock:
        d._state = STATE_BUSY
        d._reschedule_idle_timer_locked()
    assert d._idle_timer is None

    with d._lock:
        d._state = STATE_IDLE
        d._reschedule_idle_timer_locked()
    try:
        assert d._idle_timer is not None
        assert d._idle_timer is not first_timer
    finally:
        if d._idle_timer is not None:
            d._idle_timer.cancel()


def test_unloaded_event_cancels_timer() -> None:
    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(60.0)
    assert d._idle_timer is not None
    with d._lock:
        d._state = STATE_IDLE
        d._model_loaded = False
        d._reschedule_idle_timer_locked()
    assert d._idle_timer is None


def test_unloading_state_cancels_timer() -> None:
    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(60.0)
    assert d._idle_timer is not None
    with d._lock:
        d._state = STATE_UNLOADING
        d._reschedule_idle_timer_locked()
    assert d._idle_timer is None


def test_stopped_cancels_timer() -> None:
    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(60.0)
    assert d._idle_timer is not None
    with d._lock:
        d._proc = None
        d._state = STATE_STOPPED
        d._model_loaded = False
        d._reschedule_idle_timer_locked()
    assert d._idle_timer is None




def test_sync_idle_timeout_from_secrets_reads_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio.infrastructure import secrets as _secrets

    fake = _secrets.Secrets()
    fake.generate.idle_timeout_minutes = 7
    monkeypatch.setattr(_secrets, "load", lambda: fake)

    d = _make_daemon_in_state()
    d.sync_idle_timeout_from_secrets()
    assert d._idle_timeout_seconds == 7 * 60.0
    if d._idle_timer is not None:
        d._idle_timer.cancel()


def test_sync_idle_timeout_zero_minutes_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio.infrastructure import secrets as _secrets

    fake = _secrets.Secrets()
    fake.generate.idle_timeout_minutes = 0
    monkeypatch.setattr(_secrets, "load", lambda: fake)

    d = _make_daemon_in_state()
    d.sync_idle_timeout_from_secrets()
    assert d._idle_timeout_seconds == 0
    assert d._idle_timer is None


def test_sync_idle_timeout_handles_secrets_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio.infrastructure import secrets as _secrets

    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(120.0)
    assert d._idle_timeout_seconds == 120.0

    def boom() -> Any:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(_secrets, "load", boom)
    d.sync_idle_timeout_from_secrets()
    assert d._idle_timeout_seconds == 120.0
    if d._idle_timer is not None:
        d._idle_timer.cancel()
