from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from studio.infrastructure.logging import (
    HumanConsoleFormatter,
    JsonLineFormatter,
    STUDIO_LOG_NAME,
    _NOISY_LOGGERS,
    _reset_for_tests,
    reconfigure_console_utf8,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_logging(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANIMA_LOGGING_NO_BOOTSTRAP", raising=False)
    _reset_for_tests()
    saved_handlers = list(logging.getLogger().handlers)
    saved_level = logging.getLogger().level
    saved_excepthook = sys.excepthook
    yield
    _reset_for_tests()
    logging.getLogger().handlers = saved_handlers
    logging.getLogger().level = saved_level
    sys.excepthook = saved_excepthook


# ── JsonLineFormatter ─────────────────────────────────────────────────────


def test_json_formatter_emits_10_required_fields() -> None:
    fmt = JsonLineFormatter("webui")
    rec = logging.LogRecord(
        name="studio.test", level=logging.INFO, pathname="/x.py", lineno=1,
        msg="hello %s", args=("world",), exc_info=None,
    )
    out = json.loads(fmt.format(rec))
    assert set(out.keys()) >= {"ts", "level", "process", "pid", "trace_id", "logger", "msg"}
    assert out["level"] == "INFO"
    assert out["process"] == "webui"
    assert out["logger"] == "studio.test"
    assert out["msg"] == "hello world"
    assert out["trace_id"] is None


def test_json_formatter_includes_exception_when_present() -> None:
    fmt = JsonLineFormatter("worker:tag/42")
    try:
        raise ValueError("boom")
    except ValueError:
        rec = logging.LogRecord(
            name="studio.x", level=logging.ERROR, pathname="/x.py", lineno=1,
            msg="failed", args=(), exc_info=sys.exc_info(),
        )
    out = json.loads(fmt.format(rec))
    assert "exc" in out
    assert out["exc"]["type"] == "ValueError"
    assert out["exc"]["message"] == "boom"
    assert "Traceback" in out["exc"]["traceback"]


def test_json_formatter_includes_extra_user_fields() -> None:
    fmt = JsonLineFormatter("webui")
    rec = logging.LogRecord(
        name="studio.x", level=logging.INFO, pathname="/x.py", lineno=1,
        msg="tagged", args=(), exc_info=None,
    )
    rec.image_path = "/path/img.png"
    rec.image_idx = 47
    out = json.loads(fmt.format(rec))
    assert "extra" in out
    assert out["extra"]["image_path"] == "/path/img.png"
    assert out["extra"]["image_idx"] == 47


def test_json_formatter_ts_is_iso_with_z() -> None:
    fmt = JsonLineFormatter("webui")
    rec = logging.LogRecord(
        name="x", level=logging.INFO, pathname="/x.py", lineno=1,
        msg="", args=(), exc_info=None,
    )
    out = json.loads(fmt.format(rec))
    assert out["ts"].endswith("Z")
    assert "T" in out["ts"]


# ── HumanConsoleFormatter ─────────────────────────────────────────────────


def test_human_formatter_includes_level_logger_msg() -> None:
    fmt = HumanConsoleFormatter()
    rec = logging.LogRecord(
        name="studio.api.foo", level=logging.WARNING, pathname="/x.py", lineno=1,
        msg="warn msg", args=(), exc_info=None,
    )
    out = fmt.format(rec)
    assert "WARNI" in out  # %(levelname)-5s = WARNI (truncated)
    assert "studio.api.foo" in out
    assert "warn msg" in out


# ── setup_logging ─────────────────────────────────────────────────────────


def test_setup_logging_writes_to_studio_log(tmp_path: Path) -> None:
    setup_logging("webui", log_dir=tmp_path, console=False)
    logger = logging.getLogger("studio.test_setup_smoke")
    logger.info("smoke")
    for h in logging.getLogger().handlers:
        h.flush()

    log_file = tmp_path / STUDIO_LOG_NAME
    assert log_file.exists()
    line = log_file.read_text(encoding="utf-8").strip()
    out = json.loads(line)
    assert out["process"] == "webui"
    assert out["logger"] == "studio.test_setup_smoke"
    assert out["msg"] == "smoke"


def test_setup_logging_is_idempotent_for_same_process(tmp_path: Path) -> None:
    setup_logging("webui", log_dir=tmp_path, console=False)
    handlers_count = len(logging.getLogger().handlers)
    setup_logging("webui", log_dir=tmp_path, console=False)
    setup_logging("webui", log_dir=tmp_path, console=False)
    assert len(logging.getLogger().handlers) == handlers_count, (
        "calling setup_logging again for the same process should not stack handlers"
    )


def test_setup_logging_different_process_replaces_handlers(tmp_path: Path) -> None:
    setup_logging("webui", log_dir=tmp_path, console=False)
    count_a = len(logging.getLogger().handlers)
    setup_logging("worker:tag/1", log_dir=tmp_path, console=False)
    count_b = len(logging.getLogger().handlers)
    assert count_b == count_a, "a different process name should also install only one set of handlers (clear then reinstall)"


def test_setup_logging_silences_noisy_libs(tmp_path: Path) -> None:
    for n in _NOISY_LOGGERS:
        logging.getLogger(n).setLevel(logging.NOTSET)
    setup_logging("webui", log_dir=tmp_path, console=False, level="INFO")
    for n in _NOISY_LOGGERS:
        assert logging.getLogger(n).level == logging.WARNING, (
            f"{n} should be silenced to WARNING, actually {logging.getLogger(n).level}"
        )


def test_setup_logging_takes_over_uvicorn_loggers(tmp_path: Path) -> None:
    uv = logging.getLogger("uvicorn.access")
    fake_h = logging.StreamHandler()
    uv.handlers = [fake_h]
    uv.propagate = False

    setup_logging("webui", log_dir=tmp_path, console=False)

    assert uv.handlers == [], "uvicorn's built-in handlers should be cleared"
    assert uv.propagate is True, "uvicorn logger should propagate so root can take over"


def test_setup_logging_console_false_no_console_handler(tmp_path: Path) -> None:
    setup_logging("webui", log_dir=tmp_path, console=False)
    handlers = logging.getLogger().handlers
    stream_handlers = [h for h in handlers if isinstance(h, logging.StreamHandler)
                       and not isinstance(h, logging.handlers.RotatingFileHandler)]
    from concurrent_log_handler import ConcurrentRotatingFileHandler
    pure_stream = [h for h in stream_handlers if not isinstance(h, ConcurrentRotatingFileHandler)]
    assert pure_stream == [], "console=False should not install any stderr handler"


def test_setup_logging_installs_sys_excepthook(tmp_path: Path) -> None:
    original = sys.excepthook
    setup_logging("webui", log_dir=tmp_path, console=False)
    assert sys.excepthook is not original, "sys.excepthook should be replaced"


def test_setup_logging_excepthook_preserves_keyboardinterrupt(tmp_path: Path,
                                                                caplog: pytest.LogCaptureFixture) -> None:
    setup_logging("webui", log_dir=tmp_path, console=False)
    with caplog.at_level(logging.CRITICAL, logger="studio.unhandled"):
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt:
            etype, evalue, etb = sys.exc_info()
            sys.excepthook(etype, evalue, etb)
    critical_records = [r for r in caplog.records if r.name == "studio.unhandled"]
    assert critical_records == [], "KeyboardInterrupt should not route to logger.critical"


def test_setup_logging_extra_handlers_attached(tmp_path: Path) -> None:
    extra = logging.StreamHandler()
    setup_logging("worker:tag/1", log_dir=tmp_path, console=False, extra_handlers=[extra])
    assert extra in logging.getLogger().handlers


def test_reconfigure_console_utf8_does_not_crash() -> None:
    reconfigure_console_utf8()
