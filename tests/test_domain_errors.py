"""PR-2 C1 -- basic tests for the DomainError hierarchy.

No dependency on fastapi / handler (that's C2); only verifies the class
hierarchy + fields.
"""
from __future__ import annotations

import pytest

from studio.domain.errors import (
    AuthError,
    ConflictError,
    DomainError,
    ForbiddenError,
    InvalidPathError,
    NotFoundError,
    PresetConflictError,
    PresetNameInvalidError,
    PresetNotFoundError,
    ValidationError,
)


# -- base class ------------------------------------------------------------


def test_domain_error_default_fields() -> None:
    e = DomainError("something broke")
    assert e.message == "something broke"
    assert e.code == "domain.error"
    assert e.details == {}
    assert e.http_status == 400
    assert str(e) == "something broke"


def test_domain_error_custom_code_and_details() -> None:
    e = DomainError(
        "x is bad", code="my.custom", details={"field": "x"}, http_status=418,
    )
    assert e.code == "my.custom"
    assert e.details == {"field": "x"}
    assert e.http_status == 418


def test_domain_error_inherits_from_exception() -> None:
    assert issubclass(DomainError, Exception)
    with pytest.raises(DomainError):
        raise DomainError("boom")


def test_domain_error_repr_contains_code() -> None:
    e = DomainError("x", code="my.code")
    assert "my.code" in repr(e)
    assert "DomainError" in repr(e)


# -- 5 core subclasses -------------------------------------------------


@pytest.mark.parametrize("cls,expected_status,expected_code", [
    (NotFoundError, 404, "not_found"),
    (ValidationError, 422, "validation"),
    (ConflictError, 409, "conflict"),
    (AuthError, 401, "auth"),
    (ForbiddenError, 403, "forbidden"),
])
def test_core_subclass_status_and_code(cls, expected_status, expected_code) -> None:
    e = cls("msg")
    assert e.http_status == expected_status
    assert e.code == expected_code
    assert isinstance(e, DomainError)


# -- preset / path aliases ---------------------------------------------


def test_preset_not_found_is_404_with_preset_code() -> None:
    e = PresetNotFoundError("preset 'foo' missing")
    assert isinstance(e, NotFoundError)
    assert e.http_status == 404
    assert e.code == "preset.not_found"


def test_preset_name_invalid_is_400() -> None:
    """name_invalid is 400, not 422 -- name is part of the URL path, and URL validation conventionally returns 400."""
    e = PresetNameInvalidError("preset name contains '/'")
    assert e.http_status == 400
    assert e.code == "preset.name_invalid"


def test_preset_conflict_is_409() -> None:
    e = PresetConflictError("preset 'foo' already exists")
    assert isinstance(e, ConflictError)
    assert e.http_status == 409
    assert e.code == "preset.exists"


def test_invalid_path_is_400() -> None:
    e = InvalidPathError("path '../etc' escapes safe root")
    assert isinstance(e, ValidationError)
    assert e.http_status == 400
    assert e.code == "path.invalid"


# -- no fastapi dependency -----------------------------------------------


def test_domain_errors_module_does_not_import_fastapi() -> None:
    """The domain/ layer must not depend back on api (ADR-0008 / ADR-0009 Section 4)."""
    import studio.domain.errors as _e
    import sys
    # Check the module doesn't directly import fastapi (can't 100% block transitive
    # contamination via submodules, but catches a direct import)
    src = open(_e.__file__, encoding="utf-8").read()
    for forbidden in ("import fastapi", "from fastapi"):
        assert forbidden not in src, (
            f"domain/errors.py should not import fastapi (services raise via domain); "
            f"found: {forbidden!r}"
        )


def test_subclass_can_override_per_instance() -> None:
    """A subclass can override http_status per-instance (rare but legal)."""
    e = NotFoundError("x", http_status=410)  # Gone instead of 404
    assert e.http_status == 410
    assert e.code == "not_found"  # code stays unchanged
