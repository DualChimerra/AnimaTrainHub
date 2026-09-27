from __future__ import annotations

import logging

import pytest


def test_studio_loggers_propagate_to_root() -> None:
    import studio
    from studio import supervisor as _sup  # noqa: F401
    from studio.services import system_stats as _ss  # noqa: F401

    failed = []
    for name, lg in logging.Logger.manager.loggerDict.items():
        if not name.startswith("studio."):
            continue
        if isinstance(lg, logging.PlaceHolder):
            continue
        if not lg.propagate:
            failed.append(name)
    assert not failed, f"the following studio.* loggers have propagate=False, caplog would miss them: {failed}"


def test_logger_call_via_caplog_works(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("studio.test_log_baseline")
    with caplog.at_level(logging.INFO, logger="studio.test_log_baseline"):
        logger.info("baseline info message")
        logger.warning("baseline warning message")
        try:
            raise ValueError("baseline exc")
        except ValueError:
            logger.exception("baseline exception with stack")

    messages = [r.getMessage() for r in caplog.records]
    assert "baseline info message" in messages
    assert "baseline warning message" in messages
    assert "baseline exception with stack" in messages
    exc_records = [r for r in caplog.records if r.exc_info]
    assert len(exc_records) >= 1, "logger.exception must carry exc_info (stack info)"


def test_logger_getlogger_name_uses_dunder_name() -> None:
    from pathlib import Path
    import re

    studio_root = Path(__file__).resolve().parent.parent / "studio"
    bad = []
    pattern = re.compile(r"logging\.getLogger\(([^)]+)\)")
    for py in studio_root.rglob("*.py"):
        rel = str(py.relative_to(studio_root)).replace("\\", "/")
        if rel.startswith("web/"):
            continue
        if rel == "infrastructure/logging.py":
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in pattern.finditer(text):
            arg = m.group(1).strip()
            if arg not in ("__name__", "", "f\"{__name__}\""):
                if arg.startswith('"uvicorn') or arg.startswith("'uvicorn"):
                    continue
                if arg.startswith('"studio.client') or arg.startswith("'studio.client"):
                    continue
                bad.append(f"{py.relative_to(studio_root)}: getLogger({arg})")
    assert not bad, (
        "getLogger inside studio/ should uniformly use __name__; the following violations must be fixed or given a silence-list comment:\n"
        + "\n".join(bad)
    )
