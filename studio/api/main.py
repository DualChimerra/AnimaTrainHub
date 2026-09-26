"""`anima-studio` / `python -m studio.server` uvicorn entry point (extracted from server.py in PR-5).

The uvicorn startup string still points at `studio.server:app` — all 130
route decorators in the legacy server.py register onto `api.app.app` at
import time, and `from .api.app import app` at the top of server.py
re-exports that same object.
"""
from __future__ import annotations


def main() -> None:
    import argparse
    import uvicorn

    # Third-party caches are collected into `<repo>/.cache/` (this fork).
    # When cli.py starts the server, it has already set this during its own
    # import phase, inherited by the subprocess; what's overridden here is
    # the **direct** `python -m studio.server` entry point — training
    # subprocesses are spawned by this process, so the cache env vars must be
    # in place at this layer, otherwise base-model downloads and the HF cache
    # would still land on the system drive. Repeated calls are idempotent
    # (existing values are not overwritten).
    from ..infrastructure import local_cache
    from ..infrastructure.paths import REPO_ROOT

    local_cache.apply(REPO_ROOT)

    parser = argparse.ArgumentParser(description="AnimaTrainHub daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--reload", action="store_true", help="dev mode (auto-reload on edit)"
    )
    args = parser.parse_args()

    # Remote-access autostart runs inside the app lifespan, which has no other
    # way to learn the port it is being served on.
    import os
    os.environ["ALS_STUDIO_PORT"] = str(args.port)

    # ADR 0012: the SPA entry point is at the root path / (no longer under the /studio subpath).
    print(f"[AnimaTrainHub] http://{args.host}:{args.port}/")
    uvicorn.run(
        "studio.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        # While a browser is open, the /api/events SSE long connection never
        # closes on its own, so graceful shutdown's default indefinite wait
        # gets stuck on "Waiting for connections to close"; also, py3.12+'s
        # Server.wait_closed() waits for every active connection, and even a
        # second Ctrl+C's force_exit can't break out of it (the transport
        # isn't force-closed). Give graceful shutdown a cap: after the
        # timeout, uvicorn cancels the remaining connection tasks -> the
        # connection closes -> lifespan finishes normally (supervisor /
        # daemon stop gracefully).
        timeout_graceful_shutdown=3,
    )
