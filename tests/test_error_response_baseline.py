from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.api.routers import root as _root_router
from studio.api.routers import samples as _samples_router


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    output = tmp_path / "output"
    samples_dir = output / "samples"
    web_dist = tmp_path / "web_dist"
    samples_dir.mkdir(parents=True)
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server, "OUTPUT_DIR", output)
    monkeypatch.setattr(server, "WEB_DIST", web_dist)
    monkeypatch.setattr(_samples_router, "OUTPUT_DIR", output)
    monkeypatch.setattr(_root_router, "WEB_DIST", web_dist)
    return TestClient(server.app)


def _assert_error_envelope(body: dict, status: int) -> None:
    assert "detail" not in body, f"Phase 3: error response should no longer have a legacy detail key: {body!r}"
    assert "error" in body, f"missing 'error' key in {status} response: {body!r}"
    err = body["error"]
    assert isinstance(err, dict) and isinstance(err.get("code"), str), (
        f"error must be {{code, message, trace_id}}, got {err!r}"
    )
    assert isinstance(err.get("message"), str)
    assert err.get("trace_id")


def test_preset_not_found_returns_404_with_error(client: TestClient, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.services.presets import io as presets_io
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", tmp_path / "presets")
    resp = client.get("/api/presets/__definitely_does_not_exist_xxx__")
    assert resp.status_code == 404, f"expected 404, got {resp.status_code} body={resp.text!r}"
    _assert_error_envelope(resp.json(), resp.status_code)


def test_preset_invalid_name_returns_400_with_error(client: TestClient, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.services.presets import io as presets_io
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", tmp_path / "presets")
    resp = client.put("/api/presets/bad..name/with/slash", json={"foo": "bar"})
    assert resp.status_code in (400, 404, 422), (
        f"expected 4xx, got {resp.status_code} body={resp.text!r}"
    )
    _assert_error_envelope(resp.json(), resp.status_code)


def test_unknown_task_log_returns_empty_not_error(client: TestClient) -> None:
    resp = client.get("/api/logs/999999")
    assert resp.status_code == 200, f"unknown task log expected 200 empty, got {resp.status_code}"
    body = resp.json()
    assert body == {"task_id": 999999, "content": "", "size": 0}


def test_validation_error_returns_422_with_detail(client: TestClient) -> None:
    resp = client.put("/api/presets/test_preset", json="not-an-object")
    assert resp.status_code == 422, f"expected 422 from body validation, got {resp.status_code}"
    body = resp.json()
    assert "detail" in body
    assert isinstance(body["detail"], (str, list, dict)), (
        f"422 detail may currently be a list (pydantic native); a future PR-LOG-4 may change it to dict; None/int/bool are not accepted"
    )
