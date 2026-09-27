"""Directory browse endpoint tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import server
from studio.domain.errors import InvalidPathError, NotFoundError
from studio.services.dataset import browse


def _setup_tree(root: Path) -> None:
    (root / "configs").mkdir()
    (root / "models").mkdir()
    (root / "models" / "anima.safetensors").write_bytes(b"x")
    (root / "README.md").write_text("hi", encoding="utf-8")


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake = tmp_path / "repo"
    fake.mkdir()
    _setup_tree(fake)
    monkeypatch.setattr(server, "REPO_ROOT", fake)
    monkeypatch.setattr(browse, "REPO_ROOT", fake)
    # PR-5 moved /api/browse to api/routers/browse.py; the handler uses its own imported REPO_ROOT
    from studio.api.routers import browse as _browse_router
    monkeypatch.setattr(_browse_router, "REPO_ROOT", fake)
    return fake


def test_list_dir_returns_sorted_entries(fake_repo: Path) -> None:
    result = browse.list_dir(fake_repo)
    names = [e["name"] for e in result["entries"]]
    # directories sort before files
    assert names == ["configs", "models", "README.md"]
    types = [e["type"] for e in result["entries"]]
    assert types == ["dir", "dir", "file"]
    assert result["selected"] is None


def test_list_dir_returns_posix_paths(fake_repo: Path) -> None:
    """path / parent always use forward slashes, to avoid mixing in Windows backslashes when concatenated on the frontend."""
    result = browse.list_dir(fake_repo / "models")
    assert "\\" not in result["path"]
    assert result["path"].endswith("/repo/models")


def test_list_dir_rejects_outside_repo(fake_repo: Path, tmp_path: Path) -> None:
    """The lib layer still blocks outside paths by default; the server side only allows them with an explicit opt-in."""
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(InvalidPathError, match="Invalid path"):
        browse.list_dir(outside)


def test_list_dir_allows_outside_when_opted_in(fake_repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    result = browse.list_dir(outside, allow_outside_repo=True)
    assert result["entries"] == []


def test_list_dir_missing_path(fake_repo: Path) -> None:
    with pytest.raises(NotFoundError, match="Path not found"):
        browse.list_dir(fake_repo / "nope")


def test_list_dir_file_path_falls_back_to_parent(fake_repo: Path) -> None:
    """When given a file path, falls back to the parent directory and tells the frontend what to highlight via the selected field."""
    result = browse.list_dir(fake_repo / "models" / "anima.safetensors")
    assert result["path"].endswith("/repo/models")
    assert result["selected"] == "anima.safetensors"
    names = [e["name"] for e in result["entries"]]
    assert "anima.safetensors" in names


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_api_browse_default(fake_repo: Path) -> None:
    client = TestClient(server.app)
    resp = client.get("/api/browse")
    assert resp.status_code == 200
    # returned paths are consistently POSIX-style (no backslashes)
    assert "\\" not in resp.json()["path"]
    assert resp.json()["path"].endswith("/repo")
    names = [e["name"] for e in resp.json()["entries"]]
    assert "configs" in names


def test_api_browse_relative_path(fake_repo: Path) -> None:
    client = TestClient(server.app)
    resp = client.get("/api/browse?path=models")
    assert resp.status_code == 200
    names = [e["name"] for e in resp.json()["entries"]]
    assert names == ["anima.safetensors"]


def test_api_browse_allows_outside(fake_repo: Path, tmp_path: Path) -> None:
    """PathPicker allows browsing outside absolute paths (used on the settings/preset pages to pick a model on a data drive)."""
    client = TestClient(server.app)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.safetensors").write_bytes(b"x")
    resp = client.get(f"/api/browse?path={outside}")
    assert resp.status_code == 200
    names = [e["name"] for e in resp.json()["entries"]]
    assert names == ["model.safetensors"]


def test_api_browse_file_path_falls_back(fake_repo: Path) -> None:
    """Passing a file path no longer 404s; it falls back to the parent directory and sets selected."""
    client = TestClient(server.app)
    target = fake_repo / "models" / "anima.safetensors"
    resp = client.get(f"/api/browse?path={target}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["selected"] == "anima.safetensors"
    assert body["path"].endswith("/repo/models")
