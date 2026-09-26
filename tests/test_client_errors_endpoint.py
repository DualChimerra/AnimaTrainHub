from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from studio.api.routers.client_errors import _reset_rate_limit_for_tests, router

    _reset_rate_limit_for_tests()
    a = FastAPI()
    a.include_router(router)
    return TestClient(a)




def test_post_valid_body_returns_204(client: TestClient,
                                       caplog: pytest.LogCaptureFixture) -> None:
    body = {
        "kind": "react.boundary",
        "message": "Cannot read properties of undefined",
        "stack": "TypeError: ... at TrainingMonitor (foo.tsx:42)",
        "componentStack": "\n at TrainingMonitor\n at App",
        "url": "http://localhost:8765/studio/monitor/42",
        "user_agent": "Mozilla/5.0 ...",
        "client_ts": "2026-05-28T14:33:01.090Z",
        "app_version": "0.12.0",
        "build_hash": "abc1234",
    }
    with caplog.at_level(logging.ERROR, logger="studio.client"):
        resp = client.post("/api/client-errors", json=body)

    assert resp.status_code == 204
    assert resp.content == b""

    records = [r for r in caplog.records if r.name == "studio.client"]
    assert len(records) == 1
    rec = records[0]
    assert rec.levelname == "ERROR"
    assert "Cannot read properties" in rec.getMessage()
    assert "[react.boundary]" in rec.getMessage()
    assert getattr(rec, "client_kind", None) == "react.boundary"
    assert getattr(rec, "client_url", None) == "http://localhost:8765/studio/monitor/42"
    assert getattr(rec, "client_app_version", None) == "0.12.0"
    assert getattr(rec, "client_build_hash", None) == "abc1234"
    assert "TypeError" in getattr(rec, "client_stack", "")
    assert "TrainingMonitor" in getattr(rec, "client_componentStack", "")


def test_post_minimal_body_still_204(client: TestClient) -> None:
    resp = client.post("/api/client-errors", json={"message": "test"})
    assert resp.status_code == 204


def test_post_empty_dict_returns_204(client: TestClient) -> None:
    resp = client.post("/api/client-errors", json={})
    assert resp.status_code == 204


def test_post_non_json_body_silently_swallows(
    client: TestClient, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="studio.client"):
        resp = client.post("/api/client-errors", content=b"not json at all")
    assert resp.status_code == 204
    assert any("malformed" in r.getMessage() for r in caplog.records
               if r.name == "studio.client")


def test_post_non_dict_json_returns_204(client: TestClient) -> None:
    resp = client.post("/api/client-errors", json=["a", "b"])
    assert resp.status_code == 204
    resp2 = client.post("/api/client-errors", json="just a string")
    assert resp2.status_code == 204




def test_long_fields_truncated(client: TestClient,
                                 caplog: pytest.LogCaptureFixture) -> None:
    huge_stack = "x" * 10000
    huge_msg = "y" * 5000
    with caplog.at_level(logging.ERROR, logger="studio.client"):
        client.post("/api/client-errors", json={
            "message": huge_msg, "stack": huge_stack,
        })
    rec = [r for r in caplog.records if r.name == "studio.client"][0]
    assert len(rec.getMessage()) < len(huge_msg) + 100  # truncated to 1000
    assert len(getattr(rec, "client_stack", "")) <= 4000




def test_rate_limit_kicks_in_after_10_per_ip(client: TestClient,
                                                caplog: pytest.LogCaptureFixture) -> None:
    from studio.api.routers.client_errors import _reset_rate_limit_for_tests
    _reset_rate_limit_for_tests()

    with caplog.at_level(logging.INFO, logger="studio.client"):
        for i in range(10):
            resp = client.post("/api/client-errors", json={"message": f"err {i}"})
            assert resp.status_code == 204
        error_records = [r for r in caplog.records
                         if r.name == "studio.client" and r.levelname == "ERROR"]
        assert len(error_records) == 10

        caplog.clear()
        resp = client.post("/api/client-errors", json={"message": "over the limit"})
        assert resp.status_code == 204
        error_after = [r for r in caplog.records
                       if r.name == "studio.client" and r.levelname == "ERROR"]
        assert error_after == [], "rate-limit should not produce an ERROR record afterwards"
        info_records = [r for r in caplog.records
                        if r.name == "studio.client" and r.levelname == "INFO"]
        assert any("rate-limit drop" in r.getMessage() for r in info_records)


def test_rate_limit_per_ip_isolated(client: TestClient) -> None:
    from studio.api.routers.client_errors import _reset_rate_limit_for_tests
    _reset_rate_limit_for_tests()

    for i in range(10):
        client.post("/api/client-errors",
                    json={"message": f"a {i}"},
                    headers={"X-Forwarded-For": "1.2.3.4"})
    resp = client.post("/api/client-errors",
                        json={"message": "b"},
                        headers={"X-Forwarded-For": "5.6.7.8"})
    assert resp.status_code == 204


def test_rate_limit_window_recovers(client: TestClient,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.api.routers import client_errors as ce
    ce._reset_rate_limit_for_tests()

    fake_t = [1000.0]
    monkeypatch.setattr(ce.time, "monotonic", lambda: fake_t[0])

    for _ in range(10):
        client.post("/api/client-errors", json={"message": "x"})

    assert ce._rate_limit_ok("testclient") is False

    fake_t[0] += 61.0
    assert ce._rate_limit_ok("testclient") is True




def test_endpoint_registered_on_webui_app() -> None:
    from studio.api.app import app

    from ._route_helpers import iter_leaf_routes

    paths = {r.path for r in iter_leaf_routes(app.routes) if hasattr(r, "path")}
    assert "/api/client-errors" in paths
