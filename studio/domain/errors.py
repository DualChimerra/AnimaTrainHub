"""Unified exception hierarchy (ADR-0009 §4 / PR-2 C1).

The business layer (`studio.services.*`) raises DomainError subclasses;
FastAPI's api-layer exception_handler translates them into a unified JSON
envelope for the frontend. Routers no longer try/except-translate into
HTTPException (PR-2 C4/C5 is migrating the existing ~380 call sites gradually).

Why this lives in `domain/` and not `api/`:
  - services depending back on api would be an anti-pattern (ADR-0008's 4-layer
    architecture is services -> domain, never services -> api). Putting the
    base class in domain/errors.py lets services raise directly without
    breaking the layering.
  - the api layer only registers the exception_handler; it doesn't own the
    error class definitions.
  - no fastapi dependency here (pure Python Exception subclasses);
    api/exception_handlers.py is the one that imports FastAPI / Request /
    JSONResponse.

Message conventions (ADR-0009 §4.1 + the B-audit cross-D finding):
  - the `message` field is **English** -- the frontend looks up `code` in an
    i18n table to render a localized string; `message` is only the English
    fallback display. This avoids reviving the ADR-0008 §cross-D
    Chinese-string-matching trap through DomainError.
  - the `code` field is named **domain.action** (`preset.not_found` /
    `curation.duplicate`).
  - short-term (0.12.0), existing service errors like PresetError still have
    Chinese messages (C3 only adds the base class, doesn't touch messages),
    but **new code must** use an English message + i18n code.

Usage:
    from studio.domain.errors import NotFoundError, PresetNotFoundError

    if not preset_exists(name):
        raise PresetNotFoundError(f"preset {name!r} does not exist",
                                  details={"name": name})

The handler translates this automatically into (post-C2):
    HTTP 404
    Headers: X-Trace-Id: <id>
    Body: {
      "detail": "preset 'foo' does not exist",   # legacy contract
      "error": {
        "code": "preset.not_found",
        "message": "preset 'foo' does not exist",
        "trace_id": "<id>",
        "details": {"name": "foo"}
      }
    }
"""
from typing import Any, ClassVar, Dict, Optional


class DomainError(Exception):
    """Base class for business exceptions. Subclasses specify `http_status` + `default_code`."""

    http_status: ClassVar[int] = 400
    default_code: ClassVar[str] = "domain.error"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        http_status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.details = details or {}
        if http_status is not None:
            self.http_status = http_status

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, code={self.code!r})"


# -- 5 core subclasses (ADR-0009 §C round2 §1.1 decision: "5 core, not 7 all at once") --


class NotFoundError(DomainError):
    """Resource doesn't exist -- 404."""
    http_status = 404
    default_code = "not_found"


class ValidationError(DomainError):
    """Request field / business rule validation failed -- 422.

    `details` usually holds `{"field": "...", "reason": "..."}` for
    field-level hints in the frontend. Distinct from fastapi's
    RequestValidationError: that one is a pydantic body-parsing failure; this
    class is a business-layer judgment (e.g. "epoch must be > 0").
    """
    http_status = 422
    default_code = "validation"


class ConflictError(DomainError):
    """Resource conflict (name collision, state conflict) -- 409."""
    http_status = 409
    default_code = "conflict"


class AuthError(DomainError):
    """Not authenticated -- 401. The webui is currently single-user with no
    auth; this class is reserved for a future multi-user scenario."""
    http_status = 401
    default_code = "auth"


class ForbiddenError(DomainError):
    """Authenticated but not authorized -- 403. Reserved, same as AuthError."""
    http_status = 403
    default_code = "forbidden"


# -- Common subclass aliases (make service-side raises more readable) --


class InvalidPathError(ValidationError):
    """Path escapes the allowed root / has an invalid component -- 400 (not
    422, because path is part of the URL).

    Raised by `_safe_join_or_400`.
    """
    default_code = "path.invalid"
    http_status = 400


class PresetNotFoundError(NotFoundError):
    default_code = "preset.not_found"


class PresetNameInvalidError(ValidationError):
    default_code = "preset.name_invalid"
    http_status = 400


class PresetConflictError(ConflictError):
    default_code = "preset.exists"


__all__ = [
    # base class
    "DomainError",
    # 5 core subclasses
    "NotFoundError", "ValidationError", "ConflictError",
    "AuthError", "ForbiddenError",
    # preset / path aliases
    "InvalidPathError",
    "PresetNotFoundError", "PresetNameInvalidError", "PresetConflictError",
]
