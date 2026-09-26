"""Autocomplete tag list: parsing / upload / download / API endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio.infrastructure import tag_dictionary as td
from studio import server


@pytest.fixture
def tag_dict_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every read and write lands in tmp_path/tag_dictionary/, per test."""
    d = tmp_path / "tag_dictionary"
    monkeypatch.setattr(td, "TAG_DICT_DIR", d)
    monkeypatch.setattr(td, "ACTIVE_JSON", d / "active.json")
    monkeypatch.setattr(td, "SOURCE_FILE", d / "source.csv")
    return d


@pytest.fixture
def client(tag_dict_dir: Path) -> TestClient:  # noqa: ARG001 (fixture chains the patch)
    return TestClient(server.app)


class _FakeResp:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content = content
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


DANBOORU_CSV = (
    '1girl,0,6008644,"1girls,sole_female"\n'
    'highres,5,5256195,"high_res,high_resolution,hires"\n'
    'long_hair,0,4350743,"/lh,longhair"\n'
    "short_hair,0,2261608,\n"
).encode("utf-8")


# ── parsing ────────────────────────────────────────────────────────────────

def test_parse_danbooru_csv_keeps_file_order() -> None:
    assert td.parse_csv(DANBOORU_CSV.decode()) == ["1girl", "highres", "long hair", "short hair"]


def test_parse_plain_list() -> None:
    assert td.parse_csv("cat ears\nsmile\n") == ["cat ears", "smile"]


def test_parse_underscore_to_space() -> None:
    assert td.parse_csv("looking_at_viewer") == ["looking at viewer"]


def test_parse_skips_blank_comments_and_header() -> None:
    assert td.parse_csv("name,category,count\n\n# comment\nsmile,0,1\n") == ["smile"]


def test_parse_drops_duplicates() -> None:
    assert td.parse_csv("smile\nsmile,0,5\nblush\n") == ["smile", "blush"]


def test_parse_truncates_at_max_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td, "MAX_ENTRIES", 2)
    assert td.parse_csv("a\nb\nc\n") == ["a", "b"]


# ── upload ─────────────────────────────────────────────────────────────────

def test_apply_uploaded_writes_active_and_returns_meta(tag_dict_dir: Path) -> None:
    meta = td.apply_uploaded(b"smile\nblush\n", "mine.txt")
    assert meta["kind"] == "user"
    assert meta["entry_count"] == 2
    assert meta["source_name"] == "mine.txt"
    raw = json.loads((tag_dict_dir / "active.json").read_text(encoding="utf-8"))
    assert raw["tags"] == ["smile", "blush"]
    assert (tag_dict_dir / "source.csv").read_bytes() == b"smile\nblush\n"


def test_apply_uploaded_rejects_oversize(tag_dict_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    monkeypatch.setattr(td, "MAX_BYTES", 4)
    with pytest.raises(ValueError):
        td.apply_uploaded(b"smile\n", "x.txt")


def test_apply_uploaded_rejects_zero_entries(tag_dict_dir: Path) -> None:  # noqa: ARG001
    with pytest.raises(ValueError):
        td.apply_uploaded(b"# only a comment\n", "x.txt")


def test_apply_uploaded_rejects_non_utf8(tag_dict_dir: Path) -> None:  # noqa: ARG001
    with pytest.raises(ValueError):
        td.apply_uploaded(b"\xff\xfe\xfa", "x.txt")


# ── default download ───────────────────────────────────────────────────────

def test_download_default_writes_active(tag_dict_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td.requests, "get", lambda url, timeout: _FakeResp(DANBOORU_CSV))
    meta = td.download_default()
    assert meta["kind"] == "default"
    assert meta["entry_count"] == 4
    loaded = td.load_active()
    assert loaded is not None
    tags, _ = loaded
    assert tags[:2] == ["1girl", "highres"]
    assert (tag_dict_dir / "source.csv").exists()


def test_download_default_removes_legacy_files(tag_dict_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tag_dict_dir.mkdir(parents=True)
    (tag_dict_dir / "source.sqlite").write_bytes(b"old")
    monkeypatch.setattr(td.requests, "get", lambda url, timeout: _FakeResp(DANBOORU_CSV))
    td.download_default()
    assert not (tag_dict_dir / "source.sqlite").exists()


def test_download_default_propagates_http_error(tag_dict_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    monkeypatch.setattr(td.requests, "get", lambda url, timeout: _FakeResp(b"", 404))
    with pytest.raises(RuntimeError):
        td.download_default()


def test_download_default_rejects_oversize(tag_dict_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    monkeypatch.setattr(td, "MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(td.requests, "get", lambda url, timeout: _FakeResp(b"x" * 11))
    with pytest.raises(RuntimeError):
        td.download_default()


def test_download_default_rejects_empty_parse(tag_dict_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    monkeypatch.setattr(td.requests, "get", lambda url, timeout: _FakeResp(b"# nothing\n"))
    with pytest.raises(RuntimeError):
        td.download_default()


# ── active.json ────────────────────────────────────────────────────────────

def test_load_active_returns_none_when_missing(tag_dict_dir: Path) -> None:  # noqa: ARG001
    assert td.load_active() is None


def test_load_active_returns_none_when_corrupt(tag_dict_dir: Path) -> None:
    tag_dict_dir.mkdir(parents=True)
    (tag_dict_dir / "active.json").write_text("{not json", encoding="utf-8")
    assert td.load_active() is None


def test_load_active_treats_old_translation_table_as_missing(tag_dict_dir: Path) -> None:
    """The old format ({"entries": {tag: [translation]}}) is re-downloaded."""
    tag_dict_dir.mkdir(parents=True)
    (tag_dict_dir / "active.json").write_text(
        json.dumps({"meta": {}, "entries": {"smile": ["x"]}}), encoding="utf-8",
    )
    assert td.load_active() is None


# ── API ────────────────────────────────────────────────────────────────────

def test_get_meta_endpoint_uninitialized(client: TestClient) -> None:
    r = client.get("/api/tag-dictionary/meta")
    assert r.status_code == 200
    assert r.json() == {"loaded": False, "meta": None}


def test_get_data_endpoint_404_when_uninitialized(client: TestClient) -> None:
    assert client.get("/api/tag-dictionary/data").status_code == 404


def test_upload_endpoint_persists(client: TestClient) -> None:
    r = client.post(
        "/api/tag-dictionary/upload",
        files={"file": ("tags.txt", b"smile\nblush\n", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["meta"]["entry_count"] == 2
    data = client.get("/api/tag-dictionary/data").json()
    assert data["tags"] == ["smile", "blush"]


def test_upload_endpoint_rejects_garbage(client: TestClient) -> None:
    r = client.post(
        "/api/tag-dictionary/upload",
        files={"file": ("tags.txt", b"# nothing\n", "text/plain")},
    )
    assert r.status_code == 400
