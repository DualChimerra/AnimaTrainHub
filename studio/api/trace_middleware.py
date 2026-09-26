"""TraceIdMiddleware (ADR-0009 §3.2, PR-1 C5).

Pure ASGI middleware (doesn't inherit from BaseHTTPMiddleware), because:
  - BaseHTTPMiddleware uses starlette's built-in anyio.Stream wrapping, and
    ContextVar hopping across threads has known issues in starlette 0.36+
  - Pure ASGI grabs receive/send directly, so the ContextVar stays stable for
    the lifetime of the request

At the start of every HTTP request:
  1. Read the X-Trace-Id header; if absent, call new_trace_id()
  2. bind_trace_id -> every logger.x call within the request scope picks it up automatically
  3. Write X-Trace-Id back into response.headers (so the frontend's client.ts can store it in an atom)
  4. reset_trace_id when the request ends

Why middleware and not a router: a router's scope is path-after-match; a
request that hits 404 or fails before that point never gets a trace_id.
Middleware is the outermost ASGI layer, so every response passes through it.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from ..infrastructure.logging import (
    TRACE_HEADER,
    bind_trace_id,
    new_trace_id,
    reset_trace_id,
)

_TRACE_HEADER_LOWER = TRACE_HEADER.lower().encode("ascii")


class TraceIdMiddleware:
    """ASGI middleware; only handles the http scope, websocket / lifespan pass through untouched."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Read the header (scope["headers"] is List[(bytes, bytes)])
        trace_id: str | None = None
        for k, v in scope["headers"]:
            if k.lower() == _TRACE_HEADER_LOWER:
                try:
                    trace_id = v.decode("ascii", "replace").strip()
                except Exception:
                    trace_id = None
                if trace_id:
                    break
        if not trace_id:
            trace_id = new_trace_id()

        # Write trace_id into scope["state"] so the fallback Exception handler
        # at the outer ServerErrorMiddleware layer can pick it up (the
        # contextvar is reset in the finally block, so by the time the outer
        # handler runs, the contextvar is already empty). The DomainError
        # handler sits inside ExceptionMiddleware, where the contextvar is
        # still usable; the fallback handler must rely on scope state.
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["trace_id"] = trace_id

        token = bind_trace_id(trace_id)

        async def send_wrapper(message):
            # Write the X-Trace-Id header back onto the response
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # Avoid duplicates: strip any existing header with the same
                # name first (rare, but starlette may add trace_id_middleware nesting)
                headers = [(k, v) for k, v in headers if k.lower() != _TRACE_HEADER_LOWER]
                headers.append((_TRACE_HEADER_LOWER, trace_id.encode("ascii", "replace")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            reset_trace_id(token)
