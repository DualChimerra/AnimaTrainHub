"""Frontend error-reporting endpoint (ADR-0009 §5.2 / PR-3 C1).

POST /api/client-errors  — browser-side errors caught by the frontend's
ErrorBoundary / window.onerror / unhandledrejection are reported here.

The body is an arbitrary dict — the frontend is free to add fields. The
backend only recognizes:
    kind           "react.boundary" | "window.error" | "unhandledrejection" | "manual"
    message        string
    stack          string?           Error.stack
    componentStack string?           the React boundary's component stack
    source         string?           script URL (window.error)
    line / col     int?              window.error line/column number
    url            string            location.href
    user_agent     string            navigator.userAgent
    client_ts      string            ISO8601
    app_version    string
    build_hash     string?
    trace_id_last_4xx  string?       trace_id of the frontend's last 4xx response
                                     (lets devs look up the server-side context for that failure)

Returns **204 No Content** — always succeeds; a failed report must never
cascade into a frontend UI error (the frontend swallows it silently).

Per-IP rate limiting: an in-memory token bucket, 10 requests / minute. Over
the limit, requests are silently dropped (still returns 204) to avoid an
avalanche from ad-blockers / offline users / network flapping.

Logged via the `studio.client` logger → studio.log JSON line. For developers:
    jq 'select(.logger == "studio.client")' studio_data/logs/studio.log
"""
from __future__ import annotations

import logging
import time
from collections import deque
from threading import Lock
from typing import Any, Deque, Dict

from fastapi import APIRouter, Request, Response

logger = logging.getLogger("studio.client")
router = APIRouter()

_RATE_LIMIT_WINDOW_SECONDS = 60.0
_RATE_LIMIT_MAX_PER_WINDOW = 10
_ip_buckets: Dict[str, Deque[float]] = {}
_bucket_lock = Lock()


def _rate_limit_ok(ip: str, *, now: float | None = None) -> bool:
    """True = within limit, report accepted; False = over 10/min, rejected (silently 204).

    The `now` parameter is only for tests, to override the current time.
    """
    t = now if now is not None else time.monotonic()
    cutoff = t - _RATE_LIMIT_WINDOW_SECONDS
    with _bucket_lock:
        bucket = _ip_buckets.setdefault(ip, deque())
        # Drop entries outside the window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_MAX_PER_WINDOW:
            return False
        bucket.append(t)
        return True


def _client_ip(request: Request) -> str:
    """IP fallback priority: first X-Forwarded-For entry → request.client.host → 'unknown'.

    Single-machine deployments have every client share 127.0.0.1 and thus
    share the rate limit (10/min is still plenty for a single user); reverse
    proxy deployments rely on X-Forwarded-For.
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@router.post("/api/client-errors", status_code=204)
async def report_client_error(request: Request) -> Response:
    """Frontend error report. **Always returns 204** — a failed report must not cascade into the frontend UI."""
    ip = _client_ip(request)
    if not _rate_limit_ok(ip):
        # Over the limit, silently drop; log an occasional INFO line so we're not totally blind
        logger.info("client_errors rate-limit drop ip=%s", ip)
        return Response(status_code=204)

    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        # Non-JSON body — don't report it
        logger.warning("client_errors malformed body from ip=%s", ip)
        return Response(status_code=204)

    if not isinstance(body, dict):
        return Response(status_code=204)

    kind = str(body.get("kind") or "manual")[:64]
    message = str(body.get("message") or "(no message)")[:1000]
    # Pull out the recognized fields; the rest goes into extra
    extra: Dict[str, Any] = {
        "client_kind": kind,
        "client_ip": ip,
        "client_url": str(body.get("url") or "")[:500],
        "client_user_agent": str(body.get("user_agent") or "")[:300],
        "client_app_version": str(body.get("app_version") or "")[:64],
        "client_build_hash": str(body.get("build_hash") or "")[:32],
        "client_ts": str(body.get("client_ts") or "")[:32],
    }
    # Optional stack / componentStack — truncated to a sane length so the log file doesn't blow up
    for k in ("stack", "componentStack", "source"):
        v = body.get(k)
        if v:
            extra[f"client_{k}"] = str(v)[:4000]
    for k in ("line", "col"):
        v = body.get(k)
        if isinstance(v, int):
            extra[f"client_{k}"] = v
    if body.get("trace_id_last_4xx"):
        extra["client_trace_id_last_4xx"] = str(body["trace_id_last_4xx"])[:64]

    # logger.error goes to studio.log; JsonLineFormatter flattens the extra dict onto .extra
    logger.error("[%s] %s", kind, message, extra=extra)
    return Response(status_code=204)


def _reset_rate_limit_for_tests() -> None:
    """Test hook — clears ip_buckets so each test is isolated."""
    with _bucket_lock:
        _ip_buckets.clear()
