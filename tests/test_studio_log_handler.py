"""PR-1 C2 -- minimal tests for the infrastructure/logging.py skeleton.

Only covers make_studio_log_handler as exposed by C2; setup_logging isn't tested (that's C3).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def test_import_logging_module_does_not_touch_root(tmp_path: Path) -> None:
    """Importing studio.infrastructure.logging must not install a handler / change excepthook (would break import_smoke)."""
    root_handlers_before = len(logging.getLogger().handlers)
    excepthook_before = sys.excepthook

    import studio.infrastructure.logging as _slog  # noqa: F401

    assert len(logging.getLogger().handlers) == root_handlers_before, (
        "import must not add a handler to the root logger; the C2 skeleton stage must stay inert"
    )
    assert sys.excepthook is excepthook_before, (
        "import must not touch sys.excepthook; that's only installed by an explicit C3 setup_logging() call"
    )


def test_make_studio_log_handler_creates_file_on_first_emit(tmp_path: Path) -> None:
    """The default formatter is JsonLineFormatter (a C3 upgrade; C2 used a plain Formatter).

    Locks in: delayed file creation, JSON-line format (containing process / level / logger / msg),
    and unchanged constant values (STUDIO_LOG_NAME / MAX_BYTES / BACKUP_COUNT).
    """
    import json
    from studio.infrastructure.logging import (
        STUDIO_LOG_BACKUP_COUNT,
        STUDIO_LOG_MAX_BYTES,
        STUDIO_LOG_NAME,
        make_studio_log_handler,
    )

    h = make_studio_log_handler(log_dir=tmp_path, process="test-c2")
    try:
        log_file = tmp_path / STUDIO_LOG_NAME
        # delay=True: the file is only created on the first emit
        assert not log_file.exists(), "the file should not be created before any emit when delay=True"

        lg = logging.getLogger("studio.test_pr1_c2")
        lg.setLevel(logging.INFO)
        lg.addHandler(h)
        try:
            lg.info("c2 skeleton smoke")
            h.flush()
        finally:
            lg.removeHandler(h)

        assert log_file.exists()
        line = log_file.read_text(encoding="utf-8").strip()
        out = json.loads(line)
        assert out["process"] == "test-c2"
        assert out["level"] == "INFO"
        assert out["logger"] == "studio.test_pr1_c2"
        assert out["msg"] == "c2 skeleton smoke"

        # locks in the default config constant values (C3 must not change these casually)
        assert STUDIO_LOG_NAME == "studio.log"
        assert STUDIO_LOG_MAX_BYTES == 50 * 1024 * 1024
        assert STUDIO_LOG_BACKUP_COUNT == 5
    finally:
        h.close()


def test_make_studio_log_handler_uses_concurrent_handler_if_available() -> None:
    """Uses concurrent-log-handler when available (Windows cross-process file lock support);
    otherwise falls back to the stdlib RotatingFileHandler -- neither path may crash."""
    from studio.infrastructure.logging import _RotatingHandler

    # verify the type is a RotatingFileHandler subclass or itself (either satisfies this)
    import logging.handlers as _h
    assert (
        _RotatingHandler is _h.RotatingFileHandler
        or issubclass(_RotatingHandler, _h.BaseRotatingHandler)
    ), f"_RotatingHandler must be a (Concurrent)RotatingFileHandler; got {_RotatingHandler!r}"
