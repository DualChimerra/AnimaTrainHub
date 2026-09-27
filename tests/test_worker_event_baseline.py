from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.api.routers import logs as _logs_router
from studio.paths import LOGS_DIR
from studio.supervisor.cmd_builder import _EVENT_MARKER


def test_event_marker_string_is_double_underscore_event_colon() -> None:
    assert _EVENT_MARKER == "__EVENT__:", (
        f"_EVENT_MARKER changed: {_EVENT_MARKER!r}. Changing this value breaks worker IPC entirely; "
        "must be kept in sync with runtime/training/snapshot.py:EVENT_MARKER + workers/preprocess_worker.py + api/routers/logs.py"
    )


def test_logs_router_filters_event_lines(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.infrastructure import paths as _paths
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(_logs_router, "LOGS_DIR", tmp_path)

    fake_log = tmp_path / "42.log"
    fake_log.write_text(
        "[start] tagging 100 images\n"
        "__EVENT__:pause_state:{\"is_pausable\":true}\n"
        "tagged 50/100\n"
        "__EVENT__:progress:{\"step\":50}\n"
        "[done] tagged 100 images\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(server.db, "STUDIO_DB", tmp_path / "studio.db")
    db.init_db(tmp_path / "studio.db")
    client = TestClient(server.app)

    resp = client.get("/api/logs/42")
    assert resp.status_code == 200
    body = resp.json()
    content = body["content"]
    assert "[start] tagging 100 images" in content
    assert "tagged 50/100" in content
    assert "[done] tagged 100 images" in content
    assert "__EVENT__:" not in content, "logs router must filter out EVENT lines; a user seeing the IPC protocol string is a bug"
    assert "pause_state" not in content
    assert "progress" not in content


def test_logs_router_prefers_task_scoped_path(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.infrastructure import paths as _paths
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(_logs_router, "LOGS_DIR", tmp_path / "legacy_logs")

    (tmp_path / "tasks" / "99" / "run.log").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "tasks" / "99" / "run.log").write_text("NEW\n", encoding="utf-8")
    (tmp_path / "legacy_logs").mkdir()
    (tmp_path / "legacy_logs" / "99.log").write_text("OLD\n", encoding="utf-8")

    monkeypatch.setattr(server.db, "STUDIO_DB", tmp_path / "studio.db")
    db.init_db(tmp_path / "studio.db")
    client = TestClient(server.app)
    body = client.get("/api/logs/99").json()
    assert body["content"] == "NEW\n"


def test_logs_router_falls_back_to_legacy_logs_dir(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.infrastructure import paths as _paths
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(_logs_router, "LOGS_DIR", tmp_path / "legacy_logs")

    (tmp_path / "legacy_logs").mkdir()
    (tmp_path / "legacy_logs" / "55.log").write_text("OLD_ONLY\n", encoding="utf-8")

    monkeypatch.setattr(server.db, "STUDIO_DB", tmp_path / "studio.db")
    db.init_db(tmp_path / "studio.db")
    client = TestClient(server.app)
    body = client.get("/api/logs/55").json()
    assert body["content"] == "OLD_ONLY\n"


def test_event_marker_unused_task_returns_empty_not_error(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.infrastructure import paths as _paths
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(_logs_router, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(server.db, "STUDIO_DB", tmp_path / "studio.db")
    db.init_db(tmp_path / "studio.db")
    client = TestClient(server.app)

    resp = client.get("/api/logs/9999")
    assert resp.status_code == 200
    assert resp.json() == {"task_id": 9999, "content": "", "size": 0}
