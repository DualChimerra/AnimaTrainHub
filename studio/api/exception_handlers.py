"""Unified exception handler registration (ADR-0009 §4 / PR-2 C2).

4 handlers (as of Phase 3: error responses only send the `error` envelope,
the legacy `detail` field has been removed):

  1. DomainError -> `{"error": {"code", "message", "trace_id", "details"?}}`.
     4xx doesn't log a stack trace; only 5xx calls logger.exception (ADR-0009 §4.1).

  2. RequestValidationError -> keeps starlette's default `{"detail": [...]}`
     unchanged (pydantic body validation failures — the frontend has
     dedicated handling for this; this is the only path that still keeps
     `detail`). The middleware already adds the X-Trace-Id header
     automatically.

  3. HTTPException (backstop) -> adds `{"error": {...}}` for HTTPExceptions
     that haven't been migrated / come from the framework (code=`http.<status>`);
     dict/list detail goes into error.details.

  4. Exception fallback -> 500 + `{"error": {...}}`, with a sanitized message:
        {"error": {"code": "internal.server_error",
                   "message": "Internal Server Error (see trace_id in server log)",
                   "trace_id": "..."}}
     The raw traceback does **not** go into the response (to prevent leaks);
     it goes to studio.log so developers can grep by trace_id.

ADR-0009 § error envelope progressive migration (complete):
  Phase 1 (0.12.0): dual-write fills both detail + error — shipped
  Phase 2 (0.15.0): backstop handler makes body.error universal + ~330 raise
    sites migrated to DomainError with semantic codes + frontend looks up
    errors.* i18n by code — implemented
  Phase 3 (0.15.0): removed the legacy detail key, error responses only send
    error (except the RequestValidationError 422 list) — this change
  See docs/todo/error-envelope-detail-key-removal.md for details
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..domain.errors import DomainError
from ..infrastructure.logging import get_trace_id

logger = logging.getLogger(__name__)


def _trace_id_from(req: Optional[Request]) -> Optional[str]:
    """Prefer request.scope state (written by TraceIdMiddleware, still usable
    across outer handlers); fall back to the contextvar (same process, same
    scope). The fallback handler runs at the ServerErrorMiddleware layer,
    where the contextvar has already been reset — it must rely on scope state.
    """
    if req is not None:
        state = req.scope.get("state") if hasattr(req, "scope") else None
        if state and state.get("trace_id"):
            return state["trace_id"]
    return get_trace_id()


def _error_envelope(
    *, message: str, code: str,
    details: Optional[Dict[str, Any]] = None,
    req: Optional[Request] = None,
) -> Dict[str, Any]:
    """A single error envelope (ADR-0009 Phase 3: the legacy `detail` key has
    been removed, only error is sent)."""
    err: Dict[str, Any] = {
        "code": code,
        "message": message,
        "trace_id": _trace_id_from(req),
    }
    if details:
        err["details"] = details
    return {"error": err}


async def _domain_error_handler(req: Request, exc: DomainError) -> JSONResponse:
    # 4xx business errors use info level (not an exceptional path, it's part
    # of the contract); only 5xx uses exception level.
    if exc.http_status >= 500:
        logger.exception("domain error %s: %s", exc.code, exc.message)
    else:
        logger.info("domain error %s: %s", exc.code, exc.message)
    return JSONResponse(
        status_code=exc.http_status,
        content=_error_envelope(
            message=exc.message, code=exc.code, details=exc.details, req=req,
        ),
    )


async def _request_validation_handler(
    _req: Request, exc: RequestValidationError,
) -> JSONResponse:
    # pydantic's default detail is list[dict]; kept as-is (the frontend has
    # dedicated handling for it). Not dual-written because body validation
    # isn't a DomainError, so we don't force it into the envelope.
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


async def _http_exception_handler(
    req: Request, exc: StarletteHTTPException,
) -> JSONResponse:
    """ADR-0009 Phase 2/3: adds the error envelope to bare HTTPExceptions too,
    so body.error covers every error response (as of Phase 3, only error is
    sent — no more dual-writing the legacy detail). The frontend always reads
    body.error.code -> i18n.

    - detail is a str -> message=detail, code=`http.<status>` (a fallback
      without a semantic code; endpoints already migrated to DomainError
      carry a semantic code and don't go through here — what's left is
      mostly framework / not-yet-migrated code).
    - detail is a dict/list (rare, no longer produced after the business
      logic migration) -> goes into error.details to preserve structure,
      message falls back to dict.message/error.
    exc.headers is preserved (e.g. 401 WWW-Authenticate).
    """
    detail = exc.detail
    err: Dict[str, Any] = {
        "code": f"http.{exc.status_code}",
        "message": "Request failed",
        "trace_id": _trace_id_from(req),
    }
    if isinstance(detail, str):
        err["message"] = detail
    elif isinstance(detail, dict):
        err["message"] = str(detail.get("message") or detail.get("error") or "Request failed")
        err["details"] = detail
    elif detail is not None:
        err["details"] = {"detail": detail}
    return JSONResponse(
        status_code=exc.status_code, content={"error": err},
        headers=getattr(exc, "headers", None),
    )


async def _fallback_handler(req: Request, exc: Exception) -> JSONResponse:
    # Uncaught exception — logged via logger.exception with the full
    # traceback + trace_id for developers to inspect; the response body is
    # sanitized and excludes the traceback to prevent leaks.
    logger.exception(
        "unhandled exception in %s %s", req.method, req.url.path,
    )
    return JSONResponse(
        status_code=500,
        content=_error_envelope(
            message="Internal Server Error (see trace_id in server log)",
            code="internal.server_error",
            req=req,
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Called once at app.py startup.

    Order doesn't matter (FastAPI matches by the most specific exception
    type). HTTPException registers the backstop handler (ADR-0009 Phase 2):
    bare HTTPExceptions not yet migrated to DomainError also get the error
    envelope added, with detail kept as-is so the existing shape isn't broken.
    """
    app.add_exception_handler(DomainError, _domain_error_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, _fallback_handler)
