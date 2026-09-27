"""PR-1 C7 -- supervisor error_msg write-back + malformed event SSE + event_bus warn verification.

3 fixes from the B audit's P1 findings:
  - B-1.6: _tail_log_for_error_msg + _finish_slot append into db.tasks.error_msg
  - B-4.4: _on_task_log malformed event → SSE event_malformed
  - B-1.5: event_bus._safe_put QueueFull → logger.warning
"""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import pytest


# ── B-1.6: _tail_log_for_error_msg ─────────────────────────────────────


def test_tail_log_missing_file_returns_empty(tmp_path: Path) -> None:
    from studio.supervisor.core import _tail_log_for_error_msg
    assert _tail_log_for_error_msg(tmp_path / "nope.log") == ""


def test_tail_log_picks_traceback_section(tmp_path: Path) -> None:
    """When the tail has a Traceback, prefer that section (with the full stack)."""
    from studio.supervisor.core import _tail_log_for_error_msg
    log = tmp_path / "42.log"
    log.write_text(
        "[start] tagging 100 images\n"
        "[progress] 50/100\n"
        "[error] inference crashed on image 47\n"
        'Traceback (most recent call last):\n'
        '  File "wd14.py", line 184, in _infer_one\n'
        '    out = session.run(...)\n'
        'RuntimeError: ONNX op crash\n',
        encoding="utf-8",
    )
    out = _tail_log_for_error_msg(log)
    assert "Traceback" in out
    assert "RuntimeError" in out
    assert "ONNX op crash" in out


def test_tail_log_no_traceback_uses_last_lines(tmp_path: Path) -> None:
    """No Traceback string -> take the last N lines (default 12)."""
    from studio.supervisor.core import _tail_log_for_error_msg
    log = tmp_path / "x.log"
    lines = [f"line {i}" for i in range(30)]
    log.write_text("\n".join(lines), encoding="utf-8")
    out = _tail_log_for_error_msg(log, max_lines=5)
    out_lines = out.strip().splitlines()
    assert out_lines == ["line 25", "line 26", "line 27", "line 28", "line 29"]


def test_tail_log_truncates_to_max_chars(tmp_path: Path) -> None:
    from studio.supervisor.core import _tail_log_for_error_msg
    log = tmp_path / "x.log"
    log.write_text("Traceback (most recent call last):\n" + ("x" * 2000), encoding="utf-8")
    out = _tail_log_for_error_msg(log, max_chars=400)
    assert len(out) <= 400
    assert out.startswith("...")


# ── B-1.5: event_bus _safe_put QueueFull warn ──────────────────────────


def test_safe_put_logs_warning_on_queue_full(caplog: pytest.LogCaptureFixture) -> None:
    """QueueFull is no longer silently dropped; a WARNING line with the event type is logged."""
    from studio.infrastructure.event_bus import _safe_put

    async def _run():
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        q.put_nowait({"type": "first"})
        with caplog.at_level(logging.WARNING, logger="studio.infrastructure.event_bus"):
            _safe_put(q, {"type": "task_state_changed", "task_id": 42})
        warnings = [r for r in caplog.records
                    if r.name == "studio.infrastructure.event_bus" and r.levelname == "WARNING"]
        assert warnings, "QueueFull must logger.warning, not fail silently"
        assert "task_state_changed" in warnings[-1].getMessage()
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_run())


# ── B-4.4: malformed event SSE warn ────────────────────────────────────


def test_malformed_event_publishes_sse_event_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker writing a malformed __EVENT__: payload -> supervisor _on_task_log catches it and publishes
    event_malformed so the frontend can see it (previously it was silently dropped, leaving the UI pause button permanently greyed out)."""
    from studio.supervisor.core import Supervisor
    from unittest.mock import MagicMock

    events = []
    sup = Supervisor(on_event=events.append, db_path=tmp_path / "studio.db")
    sup._logs_dir = tmp_path
    slot = MagicMock()
    slot.id = 42

    callback = sup._make_task_log_callback(slot, 42)
    # feed in a malformed event line (payload isn't valid JSON)
    callback('__EVENT__:pause_state:{not json}')

    malformed = [e for e in events if e["type"] == "event_malformed"]
    assert malformed, f"should publish event_malformed; actual events: {events}"
    assert malformed[0]["task_id"] == 42
    assert "pause_state" in malformed[0]["raw_preview"]


def test_well_formed_event_does_not_publish_event_malformed(
    tmp_path: Path,
) -> None:
    """A normal event should not trigger event_malformed."""
    from studio.supervisor.core import Supervisor
    from unittest.mock import MagicMock

    events = []
    sup = Supervisor(on_event=events.append, db_path=tmp_path / "studio.db")
    sup._logs_dir = tmp_path
    slot = MagicMock()
    slot.id = 7
    slot.pause_state_path = None

    callback = sup._make_task_log_callback(slot, 7)
    callback('__EVENT__:pause_state:{"state_path": "/x.bin", "step": 100}')

    malformed = [e for e in events if e["type"] == "event_malformed"]
    assert not malformed
