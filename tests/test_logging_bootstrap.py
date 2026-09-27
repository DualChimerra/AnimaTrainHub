from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def reset(monkeypatch: pytest.MonkeyPatch):
    from studio.infrastructure.logging import _reset_for_tests
    monkeypatch.delenv("ANIMA_LOGGING_NO_BOOTSTRAP", raising=False)
    _reset_for_tests()
    saved = list(logging.getLogger().handlers)
    saved_level = logging.getLogger().level
    saved_excepthook = sys.excepthook
    yield
    _reset_for_tests()
    logging.getLogger().handlers = saved
    logging.getLogger().level = saved_level
    sys.excepthook = saved_excepthook


# ── lifespan ────────────────────────────────────────────────────────────


def test_webui_lifespan_calls_setup_logging(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIMA_LOG_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from studio import server

    with TestClient(server.app) as client:
        client.get("/api/health")

    logging.getLogger("test.lifespan").info("force-emit to open file handler")

    assert (tmp_path / "studio.log").exists(), (
        "lifespan should call setup_logging('webui'), which installs a studio.log file handler"
    )


# ── cli.main ────────────────────────────────────────────────────────────


def test_cli_main_calls_setup_logging_without_file_handler() -> None:
    from studio import cli
    from studio.infrastructure import logging as _slog

    captured = {}
    real_setup = _slog.setup_logging

    def spy(process, **kwargs):
        captured["process"] = process
        captured["kwargs"] = kwargs
        _slog._CONFIGURED_PROCESSES.add(process)

    with patch.object(_slog, "setup_logging", side_effect=spy):
        with patch.object(cli, "cmd_build", return_value=0):
            rc = cli.main(["build"])

    assert rc == 0
    assert captured["process"] == "cli:build", f"unexpected process: {captured}"
    assert captured["kwargs"].get("file") is False, (
        "CLI must have file=False (no studio.log write, round 2 §1.3 decision)"
    )
    assert captured["kwargs"].get("console") is True


# ── workers/_base.worker_main ────────────────────────────────────────────


def test_worker_main_calls_setup_logging_without_file_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio.workers import _base
    from studio.infrastructure import logging as _slog

    captured = {}

    def spy(process, **kwargs):
        captured["process"] = process
        captured["kwargs"] = kwargs
        _slog._CONFIGURED_PROCESSES.add(process)

    monkeypatch.setattr(sys, "argv", ["tag_worker.py", "--job-id", "42"])

    with patch.object(_slog, "setup_logging", side_effect=spy):
        with pytest.raises(SystemExit) as excinfo:
            _base.worker_main(lambda job_id: 0)
    assert excinfo.value.code == 0

    assert captured["process"] == "worker:tag/42", f"unexpected process: {captured}"
    assert captured["kwargs"].get("file") is False
    assert captured["kwargs"].get("console") is True


def test_worker_main_respects_process_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio.workers import _base
    from studio.infrastructure import logging as _slog

    captured = {}

    def spy(process, **kwargs):
        captured["process"] = process
        _slog._CONFIGURED_PROCESSES.add(process)

    monkeypatch.setattr(sys, "argv", ["download_worker.py", "--job-id", "7"])
    monkeypatch.setenv("ANIMA_PROCESS_NAME", "worker:custom/99")

    with patch.object(_slog, "setup_logging", side_effect=spy):
        with pytest.raises(SystemExit):
            _base.worker_main(lambda job_id: 0)

    assert captured["process"] == "worker:custom/99", (
        "ANIMA_PROCESS_NAME env should take priority over argv inference"
    )


def test_worker_kind_from_argv_strips_worker_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.workers import _base
    for argv, expected in [
        (["tag_worker.py", "--job-id", "1"], "tag"),
        (["preprocess_worker.py"], "preprocess"),
        (["download_worker.py"], "download"),
        (["reg_build_worker.py"], "reg_build"),
        (["weird_script.py"], "weird_script"),
    ]:
        monkeypatch.setattr(sys, "argv", argv)
        assert _base._worker_kind_from_argv() == expected


# ── conftest fixture self-check ────────────────────────────────────────


def test_anima_logging_no_bootstrap_env_blocks_setup_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio.infrastructure.logging import _CONFIGURED_PROCESSES, setup_logging
    monkeypatch.setenv("ANIMA_LOGGING_NO_BOOTSTRAP", "1")
    setup_logging("test-process-name", log_dir=tmp_path, console=False)
    assert "test-process-name" not in _CONFIGURED_PROCESSES, (
        "when env is set it should no-op, not enter the sentinel branch"
    )
    assert not (tmp_path / "studio.log").exists(), "should not write a file"


def test_anima_log_dir_env_overrides_paths_logs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studio.infrastructure.logging import _resolve_log_dir
    monkeypatch.setenv("ANIMA_LOG_DIR", str(tmp_path / "custom"))
    assert _resolve_log_dir() == tmp_path / "custom"

    monkeypatch.delenv("ANIMA_LOG_DIR", raising=False)
    from studio.infrastructure.paths import LOGS_DIR
    assert _resolve_log_dir() == LOGS_DIR
