"""FastAPI lifespan + migration of import-time side effects (extracted from server.py in PR-5).

Key change in PR-5: moved `ensure_dirs()` + `db.init_db()` from the top level
of server.py (import-time side effects) into lifespan startup — so that
`from studio.server import app` no longer triggers filesystem initialization,
making it easier for tests / tools to import without writing to disk.

Startup phase:
    1. Install the Windows ProactorEventLoop ConnectionResetError silencing filter
    2. ensure_dirs() + db.init_db()  <- new location as of PR-5
    3. Sweep up leftover generate tempdirs (guards against leaks after a supervisor crash)
    4. Background-download TAEFlux (intermediate-step preview, ~1.6MB, doesn't block the server)
    5. Bind the event bus to the loop + configure SSE connection callbacks
    6. Start the Supervisor + write it to app.state.supervisor
    7. Start the SystemStatsSampler

Shutdown phase:
    1. Cancel any pending SSE disconnect timer
    2. SystemStatsSampler.stop
    3. Supervisor.stop (includes daemon stop + graceful subprocess termination)
    4. disk_cache.clear_all (deletes the session directory + the key goes away with the process)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI

from .. import db
from ..infrastructure.event_bus import bus
from ..infrastructure.logging import setup_logging
from ..paths import ensure_dirs
from ..supervisor import Supervisor
from ..services import tunnel

logger = logging.getLogger(__name__)


def _install_proactor_disconnect_filter(loop: asyncio.AbstractEventLoop) -> None:
    """Swallows cosmetic ConnectionResetError noise from Windows + asyncio Proactor.

    Python's asyncio has a class of issue on Windows similar to
    [bpo-44291](https://github.com/python/cpython/issues/87691): when the
    remote TCP connection is forcibly closed (user closes the tab / refreshes
    / SSE reconnects, WinError 10054 / 10053), `_ProactorBasePipeTransport._call_connection_lost`
    goes through `socket.shutdown()` and raises `ConnectionResetError` /
    `ConnectionAbortedError`. The callback doesn't catch these two expected
    errors, so asyncio's default handler dumps a traceback to stderr. The
    server is completely fine — it's just log noise from a meaningless stack
    trace.

    Precise filtering: only silently swallow when the exception is
    ConnectionResetError / ConnectionAbortedError AND the handle's repr
    contains `_call_connection_lost`; all other asyncio exceptions still go
    to the default handler. Only installed on Windows; other platforms use
    SelectorEventLoop and don't have this bug.
    """
    if os.name != "nt":
        return

    def _filter(loop_: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
            handle = context.get("handle")
            if handle and "_call_connection_lost" in repr(handle):
                return
        loop_.default_exception_handler(context)

    loop.set_exception_handler(_filter)


class _CancelledAsgiNoiseFilter(logging.Filter):
    """Swallows cosmetic CancelledError noise from uvicorn when it cancels
    connections during shutdown.

    The uvicorn startup args include timeout_graceful_shutdown (see
    api/main.py): once that times out, uvicorn cancels the remaining
    connection tasks (long-lived SSE connections like /api/events). When the
    CancelledError bubbles up from starlette into h11's run_asgi, it gets
    caught by `except BaseException` and logged as an ERROR with a full
    traceback under "Exception in ASGI application" — but this cancellation
    was actively requested by the shutdown flow, not an application error, so
    a single Ctrl+C floods the log with two big blocks of fake errors.

    Precise filtering: only swallow records whose message matches that exact
    text AND whose exception type is CancelledError; all other ASGI
    exceptions are logged as usual.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not str(record.msg).startswith("Exception in ASGI application"):
            return True
        exc = record.exc_info[1] if record.exc_info else None
        return not isinstance(exc, asyncio.CancelledError)


_cancelled_asgi_filter = _CancelledAsgiNoiseFilter()


def _install_uvicorn_cancelled_asgi_filter() -> None:
    """Attached idempotently to the uvicorn.error logger (lifespan can run
    startup repeatedly in tests)."""
    uvicorn_logger = logging.getLogger("uvicorn.error")
    if _cancelled_asgi_filter not in uvicorn_logger.filters:
        uvicorn_logger.addFilter(_cancelled_asgi_filter)


@asynccontextmanager
async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
    """On startup, bind the event bus to the current loop and start the
    supervisor; on shutdown, stop the supervisor."""
    # PR-1 C4: unified logging entry point (ADR-0009). Called first so that
    # logs emitted by ensure_dirs / db.init_db themselves also make it into
    # studio.log. setup_logging mkdirs LOGS_DIR itself and doesn't need to
    # wait for ensure_dirs. No-op when env ANIMA_LOGGING_NO_BOOTSTRAP=1 (test mode).
    setup_logging("webui", level=os.environ.get("ANIMA_LOG_LEVEL", "INFO"))

    # Install the Windows ProactorEventLoop ConnectionResetError filter (see the helper docstring)
    _install_proactor_disconnect_filter(asyncio.get_running_loop())
    # Install the CancelledError log filter for SSE connection cancellation during shutdown (see the class docstring)
    _install_uvicorn_cancelled_asgi_filter()

    # PR-5: import-time side effects moved from the top of server.py — now
    # they hit disk only once the app actually starts, making it easier for
    # tests / tools to import without touching the filesystem.
    ensure_dirs()
    db.init_db()

    # Sweep up leftover test-generation tempdirs (guards against supervisor
    # crashes leaking anima_gen_* directories)
    from ..services.inference.core import cleanup_stale_generate_tempdirs
    from ..services.inference import disk_cache as generate_cache
    from ..services import models as _md
    from ..services import system_stats
    from ..infrastructure.paths import STUDIO_DATA
    cleanup_stale_generate_tempdirs()

    # Encrypted disk cache init: startup_clean wipes leftover session-*
    # directories (ones a previous SIGKILL / power loss / normal shutdown
    # failed to delete), then opens a new session (random session_id +
    # aes_key; the key disappears with the process -> leftover files become
    # unreadable garbage bytes that disk scanning tools can't identify).
    generate_cache.init(STUDIO_DATA / ".cache" / "generate")

    # Background download of TAEFlux (intermediate-step preview): starts
    # alongside the server; a failed download doesn't block the server. If
    # already downloaded, this is a no-op; while downloading, the user can
    # use other features normally — the preview feature just isn't active
    # until the download finishes.
    def _bg_download_taeflux() -> None:
        try:
            if _md.taeflux_available():
                return
            logger.info("background-downloading TAEFlux (~1.6MB)…")
            ok = _md.download_taeflux(on_log=lambda m: logger.info("[taeflux] %s", m))
            if not ok:
                logger.warning("taeflux background download failed; preview disabled until manual install")
        except Exception:
            logger.exception("taeflux background download crashed")
    threading.Thread(target=_bg_download_taeflux, name="taeflux-bg-download", daemon=True).start()

    # Background download of the autocomplete tag list (~3.5MB CSV): only
    # when there is no usable active.json (missing, or the old translation
    # table format). A failure just logs a warning; the next start retries.
    def _bg_download_tag_dict() -> None:
        from ..infrastructure import tag_dictionary as _td
        try:
            if _td.load_active() is not None:
                return
            logger.info("background-downloading the tag list (~3.5MB)…")
            _td.download_default()
        except Exception as exc:
            logger.warning(
                "tag list background download failed (%s); it is retried on the next start",
                exc,
            )
    threading.Thread(target=_bg_download_tag_dict, name="tag-dict-bg-download", daemon=True).start()

    bus.attach_loop(asyncio.get_running_loop())

    # commit 11: clear the generate cache after an SSE client disconnects, with a 30s buffer.
    # Guards against refresh/brief blips: on reconnect (_on_first_subscribe), the timer is canceled.
    _disconnect_timer: dict[str, Optional[threading.Timer]] = {"t": None}

    def _on_last_unsubscribe() -> None:
        # Don't reset an existing timer (when multiple clients each unsubscribe, only the last one matters)
        if _disconnect_timer["t"] is not None:
            return
        timer = threading.Timer(30.0, _flush_cache)
        timer.daemon = True
        _disconnect_timer["t"] = timer
        timer.start()

    def _on_first_subscribe() -> None:
        timer = _disconnect_timer.get("t")
        if timer is not None:
            timer.cancel()
            _disconnect_timer["t"] = None

    def _flush_cache() -> None:
        n = generate_cache.total_count()
        if n:
            generate_cache.clear_all()
            logger.info("flushed generate disk cache (%d images) after SSE idle", n)
        _disconnect_timer["t"] = None

    bus.set_connection_callbacks(
        on_first_subscribe=_on_first_subscribe,
        on_last_unsubscribe=_on_last_unsubscribe,
    )

    # Remote access: the tunnel gate relay runs on this loop; then open the
    # public link right away if the user switched autostart on.
    tunnel.bind_loop(asyncio.get_running_loop())
    tunnel.autostart(int(os.environ.get("ALS_STUDIO_PORT") or 8765))

    sup = Supervisor(on_event=bus.publish)
    sup.start()
    app_.state.supervisor = sup

    # PR #37: system stats SSE — a background sampler collects every 2.5s and
    # calls bus.publish. The frontend only does a single cold-start GET on
    # mount, to avoid cloud deployments getting hammered by each client
    # polling independently.
    def _publish_system_stats(payload: dict[str, Any]) -> None:
        bus.publish({"type": "system_stats_updated", "payload": payload})

    sys_sampler = system_stats.SystemStatsSampler(_publish_system_stats)
    sys_sampler.start()
    app_.state.system_stats_sampler = sys_sampler

    try:
        yield
    finally:
        # Cancel any pending disconnect timer — no need for the delay during shutdown
        timer = _disconnect_timer.get("t")
        if timer is not None:
            timer.cancel()
        sys_sampler.stop()
        sup.stop()
        # A stray cloudflared would keep a public URL alive pointing at a port
        # that no longer serves anything.
        tunnel.shutdown()
        # Shutdown wipes the entire session directory + the aes_key goes away
        # with the process (even if rmtree fails, leftover files without the
        # key are just garbage bytes; the next startup_clean covers it)
        generate_cache.clear_all()
