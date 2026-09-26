"""PR-2 C3 -- verify that all 11 service errors inherit from DomainError.

Once C4/C5 remove the router try/except, the precondition for a service raise to be
translated directly into an envelope by the handler is isinstance(exc, DomainError). This test locks the inheritance chain so future PRs can't break it.
"""
from __future__ import annotations

import pytest

from studio.domain.errors import DomainError
from studio.services.data_io.train_io import TrainIOError
from studio.services.dataset.browse import BrowseError
from studio.services.dataset.curation import CurationError
from studio.services.preprocess.core import PreprocessError
from studio.services.preprocess.duplicates import DuplicateFinderError
from studio.services.presets.io import PresetError
from studio.services.projects.jobs import JobError
from studio.services.projects.projects import ProjectError
from studio.services.projects.versions import VersionError
from studio.services.tagging.caption_snapshot import SnapshotError
from studio.services.version_config import VersionConfigError


@pytest.mark.parametrize("cls,expected_code", [
    (PresetError, "preset.error"),
    (ProjectError, "project.error"),
    (VersionError, "version.error"),
    (JobError, "job.error"),
    (CurationError, "curation.error"),
    (TrainIOError, "train_io.error"),
    (VersionConfigError, "version_config.error"),
    (DuplicateFinderError, "duplicate.error"),
    (PreprocessError, "preprocess.error"),
    (BrowseError, "browse.error"),
    (SnapshotError, "snapshot.error"),
])
def test_service_error_inherits_domain_with_code(cls, expected_code) -> None:
    assert issubclass(cls, DomainError), (
        f"{cls.__name__} must inherit from DomainError so the exception handler catches it automatically"
    )
    e = cls("test message")
    assert isinstance(e, DomainError)
    assert e.code == expected_code
    assert e.http_status == 400  # default; overridden case-by-case by the router or at raise time


def test_raise_service_error_caught_by_handler_via_isinstance(tmp_path) -> None:
    """End-to-end: raising PresetError gets translated into an envelope by the DomainError handler via the webui app."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from studio.api.exception_handlers import register_exception_handlers
    from studio.api.trace_middleware import TraceIdMiddleware

    a = FastAPI()
    a.add_middleware(TraceIdMiddleware)
    register_exception_handlers(a)

    @a.get("/raise_preset_err")
    def _r():
        raise PresetError("preset 'x' missing")

    c = TestClient(a)
    resp = c.get("/raise_preset_err")
    assert resp.status_code == 400
    body = resp.json()
    assert "detail" not in body  # Phase 3: only send the error envelope
    assert body["error"]["code"] == "preset.error"
    assert body["error"]["message"] == "preset 'x' missing"
    assert body["error"]["trace_id"] is not None


def test_raise_with_override_http_status() -> None:
    """A service can still raise PresetError("x", http_status=404) to have the handler translate it to 404.

    This is commonly used during C4/C5 router migration -- e.g. read_preset raises
    PresetError("...", http_status=404, code="preset.not_found") when the preset doesn't exist.
    """
    e = PresetError("missing", http_status=404, code="preset.not_found")
    assert e.http_status == 404
    assert e.code == "preset.not_found"
    # but isinstance is still both PresetError and DomainError
    assert isinstance(e, PresetError)
    assert isinstance(e, DomainError)
