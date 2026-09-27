"""Entry point for the unified logging system (ADR-0009).

C2: skeleton (make_studio_log_handler)
C3: full setup_logging + JsonLineFormatter + HumanConsoleFormatter + silence list
    + uvicorn handler takeover + utf8 reconfigure (this file)
C5: ContextVar trace_id + Filter (to be added)
C6: cross-process env / db.tasks.request_trace_id (to be added)

Usage rules (ADR-0009 SS Future Dev Guide, short version):
    Every process entry point (webui / cli / worker) calls `setup_logging(process=...)` once
    at the top.
    Business code modules do `logger = logging.getLogger(__name__)` at module scope and call
    `.info/.warning/.exception`.
    Don't call setup_logging at import time (breaks the import smoke test).

`process` identifier convention:
    "webui"                 webui server
    "cli:<subcmd>"          CLI run / dev / build / test
    "worker:<kind>/<id>"    worker subprocess (kind=download/tag/preprocess/reg_build)
    "client"                frontend reporting (only used starting with the C-series PR-3)
"""
from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import logging.handlers
import os
import sys
import traceback as _traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from .paths import LOGS_DIR

try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler as _RotatingHandler
except ImportError:
    _RotatingHandler = logging.handlers.RotatingFileHandler

STUDIO_LOG_NAME = "studio.log"
STUDIO_LOG_MAX_BYTES = 50 * 1024 * 1024
STUDIO_LOG_BACKUP_COUNT = 5

# Cross-process / cross-surface naming convention for trace_id (ADR-0009 SS3.1)
TRACE_HEADER = "X-Trace-Id"           # HTTP request/response header
TRACE_ENV = "ANIMA_TRACE_ID"          # child process env (injected by the C6 supervisor)
PROCESS_ENV = "ANIMA_PROCESS_NAME"    # preset process name for the child process

# ContextVar -- auto-propagates across async/thread within the same process (C5)
# job_id / task_id will be used after C6 / C7
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "studio_trace_id", default=None
)
_job_id_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "studio_job_id", default=None
)
_task_id_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "studio_task_id", default=None
)

# Silence list for third-party library loggers -- these libraries spew heavy INFO noise when
# root level=INFO. The silence list must be explicit, so missing an entry doesn't 10x stderr
# (B audit N.1 / A round2 SS5.2). Applied at the end of setup_logging.
_NOISY_LOGGERS = (
    "asyncio",
    "urllib3",
    "httpcore",
    "httpx",
    "PIL",
    "matplotlib",
    "modelscope",
    "huggingface_hub",
    "filelock",
)

# ── trace_id public API ────────────────────────────────────────────────────


def new_trace_id() -> str:
    """Generates a 24-char hex trace_id (uuid4 hex[:24]).

    24 characters: (a) the last 8 characters are easy to copy from a user's screenshot;
    (b) clearly distinguishable from the supervisor's background-spawn `bg-{new_trace_id()}`
    (28 characters); (c) simpler than a 26-char ULID, no extra dependency needed.
    """
    return uuid.uuid4().hex[:24]


def bind_trace_id(trace_id: str) -> contextvars.Token:
    """Sets the current ContextVar trace_id, returns a token for reset.

    Usage (middleware / worker bootstrap):
        token = bind_trace_id(tid)
        try:
            ...do work...
        finally:
            reset_trace_id(token)
    """
    return _trace_id_var.set(trace_id)


def reset_trace_id(token: contextvars.Token) -> None:
    _trace_id_var.reset(token)


def get_trace_id() -> Optional[str]:
    """The current ContextVar trace_id; returns None if never bound."""
    return _trace_id_var.get()


def bind_job_id(job_id: int) -> contextvars.Token:
    return _job_id_var.set(job_id)


def bind_task_id(task_id: int) -> contextvars.Token:
    return _task_id_var.set(task_id)


def get_job_id() -> Optional[int]:
    return _job_id_var.get()


def get_task_id() -> Optional[int]:
    return _task_id_var.get()


# ── ContextFilter ────────────────────────────────────────────────────────


class ContextFilter(logging.Filter):
    """Reads ContextVars and injects them into the LogRecord, so JsonLineFormatter can pick
    them up directly.

    Uses a Filter rather than a LogRecord factory: a factory would be a global change, whereas
    a filter just gets attached to root -- a clean rollback (removing the filter in a C5 PR
    restores the old behavior without touching the LogRecord class).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = _trace_id_var.get()
        if not hasattr(record, "job_id"):
            jid = _job_id_var.get()
            if jid is not None:
                record.job_id = jid
        if not hasattr(record, "task_id"):
            tid = _task_id_var.get()
            if tid is not None:
                record.task_id = tid
        return True


# Module-level sentinel -- calling setup_logging again with the same process name doesn't
# re-attach handlers. Benefits supervisor restarts / test reloads / a worker entry point being
# called twice.
_CONFIGURED_PROCESSES: set[str] = set()


# ── Formatter ─────────────────────────────────────────────────────────────


class JsonLineFormatter(logging.Formatter):
    """JSON line output, 10 fixed fields (ADR-0009 SS2.3).

    Top-level fields: ts / level / process / pid / trace_id / logger / msg / job_id / task_id / exc.
    Optional nested field: extra (user fields passed via logger.x(..., extra={...})).
    """

    def __init__(self, process: str) -> None:
        super().__init__()
        self._process = process

    def format(self, record: logging.LogRecord) -> str:
        # stdlib formatTime uses time.strftime which doesn't support %f; build the ms part
        # ourselves via datetime
        ts = (
            _dt.datetime.fromtimestamp(record.created, tz=_dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )
        out: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "process": self._process,
            "pid": record.process,
            "trace_id": getattr(record, "trace_id", None),  # injected by the C5 Filter
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # optional contextvar fields
        for k in ("job_id", "task_id"):
            v = getattr(record, k, None)
            if v is not None:
                out[k] = v
        # exc_info -> structured
        if record.exc_info:
            etype, evalue, etb = record.exc_info
            out["exc"] = {
                "type": etype.__name__ if etype else "Unknown",
                "message": str(evalue),
                "traceback": "".join(_traceback.format_exception(etype, evalue, etb)),
            }
        # extra (a user's logger.x(..., extra={"k": "v"}) lands in record.__dict__)
        # distinguishes native fields + the ones we inject + user extra
        _builtin = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
            "message", "asctime", "trace_id", "job_id", "task_id",
        }
        extra = {k: v for k, v in record.__dict__.items() if k not in _builtin}
        if extra:
            out["extra"] = extra
        return json.dumps(out, ensure_ascii=False, default=str)


class HumanConsoleFormatter(logging.Formatter):
    """Human-readable console format, for CLI / dev terminal use.

    `2026-05-28 14:32:18.453 INFO  studio.api.routers.queue: queued task=42`
    """

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ── Public API ────────────────────────────────────────────────────────────


def reconfigure_console_utf8() -> None:
    """Windows consoles default to cp932/cp936, so writing non-ASCII text / emoji raises
    UnicodeEncodeError. Forces stdout/stderr into UTF-8 + replace mode so the logger never
    raises.

    Called as the first step inside setup_logging, covering all process entry points (webui /
    cli / worker); workers/_base.py:reconfigure_console_utf8 (B audit 4.6 -- half the workers
    were missing this) is covered by this fallback.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass


def make_studio_log_handler(
    log_dir: Path | None = None,
    *,
    process: str = "webui",
    formatter: logging.Formatter | None = None,
) -> logging.Handler:
    """Creates a rotating handler that writes to studio.log (doesn't attach it to root;
    that's the caller's decision).

    Args:
        log_dir: directory to write to; defaults to paths.LOGS_DIR
        process: process identifier written into JsonLineFormatter; defaults to "webui"
        formatter: custom formatter; defaults to JsonLineFormatter(process)
    """
    target = log_dir or LOGS_DIR
    target.mkdir(parents=True, exist_ok=True)
    handler = _RotatingHandler(
        str(target / STUDIO_LOG_NAME),
        maxBytes=STUDIO_LOG_MAX_BYTES,
        backupCount=STUDIO_LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(formatter or JsonLineFormatter(process))
    return handler


def setup_logging(
    process: str,
    *,
    log_dir: Path | None = None,
    level: str = "INFO",
    console: str | bool = "auto",
    file: bool = True,
    extra_handlers: list[logging.Handler] | None = None,
) -> None:
    """Install the global logging config. Called once per process entry point (webui / cli / worker).

    Idempotent: repeated calls with the same `process` name are a noop (prevents handler pile-up on
    supervisor restarts / test reloads).
    Behavior:
      1. reconfigure_console_utf8 (Windows encoding fallback)
      2. clear existing root logger handlers (prevents pile-up)
      3. (optional) install a RotatingFileHandler at {log_dir}/studio.log (JSON lines, 50MB x 5)
      4. install the console handler (auto/json/bool tri-state, see below)
      5. install extra_handlers (worker uses this to add a jobs/<id>.log handler)
      6. silence third-party libraries (asyncio/urllib3/PIL/...) -> WARNING
      7. take over uvicorn.access / uvicorn.error loggers (route through our root handler instead of uvicorn's default)
      8. install sys.excepthook -> logger.critical logs uncaught exceptions
      9. install threading.excepthook -> same, for threads (Python 3.8+ only)

    Args:
        process: process identifier ("webui" / "cli:run" / "worker:tag/42" / ...)
        log_dir: directory studio.log is written to; defaults to env ANIMA_LOG_DIR, else paths.LOGS_DIR
        level: root logger level ("DEBUG" / "INFO" / "WARNING" / "ERROR")
        console: "auto" = Human if stderr.isatty() else JSON;
                 "json" forces JSON; True forces Human; False installs no console handler
        file: whether to install a file handler at studio.log; defaults to True for webui;
              worker / cli should pass False (worker writes to stdout, single-writer via supervisor redirect;
              CLI's ~5s lifetime makes disk logging low-value). Worker was switched to True after 0.13.x ADR-0009 §"progressive dual-write" paid off this debt
        extra_handlers: extra handlers to add to root (worker uses this to add jobs/<id>.log)
    """
    # pytest's global fixture sets ANIMA_LOGGING_NO_BOOTSTRAP=1 so that application-code bootstrap
    # calls are all noops, avoiding caplog pollution / repeated handler installs; the test for
    # setup_logging itself, tests/test_logging_setup.py, does its own monkeypatch.delenv to lift this.
    if os.environ.get("ANIMA_LOGGING_NO_BOOTSTRAP"):
        return
    if process in _CONFIGURED_PROCESSES:
        return
    _CONFIGURED_PROCESSES.add(process)

    reconfigure_console_utf8()

    root = logging.getLogger()
    # Clear default / accumulated handlers (matters when a pytest fixture calls this repeatedly)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 1. file handler -- JSON lines to studio.log (not installed for worker / cli)
    if file:
        effective_log_dir = log_dir or _resolve_log_dir()
        root.addHandler(make_studio_log_handler(log_dir=effective_log_dir, process=process))

    # 2. console handler
    console_handler = _build_console_handler(console, process)
    if console_handler is not None:
        root.addHandler(console_handler)

    # ContextFilter -- installed on each handler rather than the logger.
    # stdlib Logger.filter is only called once at the top of Logger.handle; when a child logger
    # propagates to root, root.filter is **not** called, only root.handlers[*].emit is. So the filter
    # must be installed on the handlers, to guarantee every record gets ContextVar injection when it passes through a handler.
    _ctx_filter = ContextFilter()
    for h in root.handlers:
        h.addFilter(_ctx_filter)

    # 3. extra handlers (worker job log etc.) -- also gets ContextFilter installed
    for h in extra_handlers or ():
        h.addFilter(ContextFilter())
        root.addHandler(h)

    # 4. silence third-party libraries (prevents a 10x stderr noise spike once root=INFO)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # 5. take over uvicorn loggers -- keep the access log in the same format/trace_id as application logs
    #    (B audit cross-cutting issue C / A round2 §4.1 blind spot 2)
    for uname in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        u = logging.getLogger(uname)
        u.handlers = []        # clear uvicorn's own StreamHandler
        u.propagate = True     # let the root handler take over

    # 6. route uncaught exceptions into the logger
    _install_excepthooks()


def _resolve_log_dir() -> Path:
    """Prefer the ANIMA_LOG_DIR env var, else paths.LOGS_DIR.

    pytest conftest sets ANIMA_LOG_DIR=tmp_path_factory so tests don't write into the repo's studio_data/.
    """
    env = os.environ.get("ANIMA_LOG_DIR", "").strip()
    if env:
        return Path(env)
    return LOGS_DIR


def _build_console_handler(console: str | bool, process: str) -> logging.Handler | None:
    if console is False:
        return None
    if console == "auto":
        use_json = not sys.stderr.isatty()
    elif console == "json":
        use_json = True
    else:  # True or anything else
        use_json = False
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonLineFormatter(process) if use_json else HumanConsoleFormatter())
    return h


_EXCEPTHOOKS_INSTALLED = False


def _install_excepthooks() -> None:
    """Install sys.excepthook + threading.excepthook -> logger.critical.

    Idempotent (only installed once even with multiple setup_logging calls).
    """
    global _EXCEPTHOOKS_INSTALLED
    if _EXCEPTHOOKS_INSTALLED:
        return
    _EXCEPTHOOKS_INSTALLED = True

    _orig_excepthook = sys.excepthook

    def _excepthook(etype, evalue, etb):
        # KeyboardInterrupt still goes through the default hook, so Ctrl+C isn't swallowed
        if issubclass(etype, KeyboardInterrupt):
            _orig_excepthook(etype, evalue, etb)
            return
        logging.getLogger("studio.unhandled").critical(
            "unhandled exception in main thread", exc_info=(etype, evalue, etb)
        )

    sys.excepthook = _excepthook

    # Python 3.8+ has threading.excepthook
    try:
        import threading
        _orig_thread_hook = threading.excepthook

        def _thread_excepthook(args: Any) -> None:
            logging.getLogger("studio.unhandled").critical(
                "unhandled exception in thread %s", args.thread.name if args.thread else "?",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        threading.excepthook = _thread_excepthook
    except (AttributeError, ImportError):
        pass


def _reset_for_tests() -> None:
    """Test hook: clears the sentinel + ContextVars so multiple tests can each call setup_logging independently.

    Should not be called from production code. Used by fixtures:
        from studio.infrastructure.logging import _reset_for_tests
        _reset_for_tests()
    """
    global _EXCEPTHOOKS_INSTALLED
    _CONFIGURED_PROCESSES.clear()
    _EXCEPTHOOKS_INSTALLED = False
    # ContextVars leak across tests (same thread / async context), clear them explicitly
    _trace_id_var.set(None)
    _job_id_var.set(None)
    _task_id_var.set(None)
    # sys.excepthook is not restored (tests shouldn't depend on that either)


# The encoding fallback is an unconditional constant, exported for compat with old imports in workers/_base.py
__all__ = [
    "STUDIO_LOG_NAME", "STUDIO_LOG_MAX_BYTES", "STUDIO_LOG_BACKUP_COUNT",
    "TRACE_HEADER", "TRACE_ENV", "PROCESS_ENV",
    "JsonLineFormatter", "HumanConsoleFormatter", "ContextFilter",
    "make_studio_log_handler", "setup_logging", "reconfigure_console_utf8",
    "new_trace_id", "bind_trace_id", "reset_trace_id", "get_trace_id",
    "bind_job_id", "bind_task_id", "get_job_id", "get_task_id",
    "_reset_for_tests",
]
