from __future__ import annotations

import os
from pathlib import Path

import pytest

from studio.services.inference import disk_cache




def _png_bytes(payload: bytes = b"hello-png-content") -> bytes:
    return b"\x89PNG\r\n\x1a\n" + payload


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "generate_cache"


@pytest.fixture
def cache(cache_root: Path):
    sc = disk_cache.SessionCache(root=cache_root, max_count=100, max_bytes=10**9)
    sc.ensure_dir()
    yield sc
    sc.clear_all()




def test_put_get_roundtrip(cache: disk_cache.SessionCache) -> None:
    png = _png_bytes(b"abc")
    cache.put(1, "a.png", png, {"mode": "single", "prompts": ["test"]}, mode="single")
    assert cache.get_image(1, "a.png") == png


def test_get_missing_returns_none(cache: disk_cache.SessionCache) -> None:
    assert cache.get_image(99, "nope.png") is None


def test_put_writes_different_ciphertext_each_time(cache: disk_cache.SessionCache) -> None:
    png = _png_bytes(b"same")
    cache.put(1, "x.png", png, {}, mode="single")
    blob1 = next(iter(cache._index.values())).file_path.read_bytes()
    cache.put(1, "x.png", png, {}, mode="single")
    blob2 = next(iter(cache._index.values())).file_path.read_bytes()
    assert blob1 != blob2


def test_overwrite_same_key_deletes_old_file(cache: disk_cache.SessionCache) -> None:
    cache.put(1, "a.png", _png_bytes(b"v1"), {}, mode="single")
    old_path = next(iter(cache._index.values())).file_path
    cache.put(1, "a.png", _png_bytes(b"v2"), {}, mode="single")
    assert not old_path.exists()
    assert cache.total_count() == 1
    assert cache.get_image(1, "a.png") == _png_bytes(b"v2")




def test_on_disk_blob_has_no_png_magic(cache: disk_cache.SessionCache) -> None:
    cache.put(1, "a.png", _png_bytes(b"secret-payload" * 100), {}, mode="single")
    file_path = next(iter(cache._index.values())).file_path
    blob = file_path.read_bytes()
    assert not blob.startswith(b"\x89PNG"), "leaked PNG magic bytes"
    assert not blob[16:].startswith(b"\x89PNG"), "leaked PNG magic bytes after nonce"


def test_on_disk_blob_is_high_entropy(cache: disk_cache.SessionCache) -> None:
    cache.put(1, "a.png", _png_bytes(b"\x00" * 4096), {}, mode="single")
    file_path = next(iter(cache._index.values())).file_path
    blob = file_path.read_bytes()
    counts = [0] * 256
    for b in blob:
        counts[b] += 1
    max_share = max(counts) / len(blob)
    assert max_share < 0.05, f"ciphertext byte distribution unexpectedly uniform: max share {max_share:.3f}"


def test_filename_has_no_extension_hint(cache: disk_cache.SessionCache) -> None:
    cache.put(1, "a.png", _png_bytes(), {}, mode="single")
    file_path = next(iter(cache._index.values())).file_path
    assert file_path.suffix == ".bin", f"unexpected extension {file_path.suffix}"
    assert "a.png" not in file_path.name


# ---------------------------------------------------------------- list_index


def test_list_index_single_groups_by_task(cache: disk_cache.SessionCache) -> None:
    cache.put(1, "a.png", _png_bytes(b"a"), {"mode": "single"}, mode="single")
    cache.put(1, "b.png", _png_bytes(b"b"), {"mode": "single"}, mode="single")
    cache.put(2, "c.png", _png_bytes(b"c"), {"mode": "single"}, mode="single")
    idx = cache.list_index()
    assert len(idx) == 2
    by_task = {e["taskId"]: e for e in idx}
    assert by_task[1]["filenames"] == ["a.png", "b.png"]
    assert by_task[2]["filenames"] == ["c.png"]
    assert all("samples" not in e for e in idx)


def test_list_index_xy_includes_samples(cache: disk_cache.SessionCache) -> None:
    snap = {"mode": "xy", "xy_draft": {"x": {"axis": "cfg", "raw": "3,5"}, "y": None}}
    cache.put(
        7, "xy_x00_y00.png", _png_bytes(b"00"), snap,
        mode="xy", xy_info={"xi": 0, "yi": 0, "xv": "3", "yv": None},
    )
    cache.put(
        7, "xy_x01_y00.png", _png_bytes(b"01"), snap,
        mode="xy", xy_info={"xi": 1, "yi": 0, "xv": "5", "yv": None},
    )
    idx = cache.list_index()
    assert len(idx) == 1
    e = idx[0]
    assert e["mode"] == "xy"
    assert e["taskId"] == 7
    assert len(e["samples"]) == 2
    assert e["samples"][0]["xy"]["xi"] == 0
    assert e["samples"][1]["xy"]["xi"] == 1


def test_list_index_sorted_desc_by_created_at(cache: disk_cache.SessionCache, monkeypatch) -> None:
    times = iter([1000.0, 2000.0, 3000.0])
    monkeypatch.setattr("studio.services.inference.disk_cache.time.time", lambda: next(times))
    cache.put(1, "a.png", _png_bytes(), {}, mode="single")
    cache.put(2, "b.png", _png_bytes(), {}, mode="single")
    cache.put(3, "c.png", _png_bytes(), {}, mode="single")
    idx = cache.list_index()
    assert [e["taskId"] for e in idx] == [3, 2, 1]


# ---------------------------------------------------------------- drop / clear


def test_drop_task_removes_files_and_index(cache: disk_cache.SessionCache) -> None:
    cache.put(1, "a.png", _png_bytes(), {}, mode="single")
    cache.put(1, "b.png", _png_bytes(), {}, mode="single")
    cache.put(2, "c.png", _png_bytes(), {}, mode="single")
    files_before = list(cache.session_dir.iterdir())
    assert len(files_before) == 3
    n = cache.drop_task(1)
    assert n == 2
    files_after = list(cache.session_dir.iterdir())
    assert len(files_after) == 1
    assert cache.get_image(1, "a.png") is None
    assert cache.get_image(2, "c.png") is not None


def test_clear_all_empties_session_dir(cache: disk_cache.SessionCache) -> None:
    cache.put(1, "a.png", _png_bytes(), {}, mode="single")
    cache.put(1, "b.png", _png_bytes(), {}, mode="single")
    cache.clear_all()
    assert cache.total_count() == 0
    assert cache.total_bytes() == 0
    assert cache.session_dir.exists()
    assert list(cache.session_dir.iterdir()) == []


# ---------------------------------------------------------------- startup_clean


def test_startup_clean_removes_all_session_dirs(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    (root / "session-aaaaaaaa").mkdir()
    (root / "session-bbbbbbbb").mkdir()
    (root / "session-aaaaaaaa" / "file1.bin").write_bytes(b"garbage")
    (root / "other_dir").mkdir()
    n = disk_cache.startup_clean(root)
    assert n == 2
    assert not (root / "session-aaaaaaaa").exists()
    assert not (root / "session-bbbbbbbb").exists()
    assert (root / "other_dir").exists()


def test_startup_clean_on_nonexistent_root(tmp_path: Path) -> None:
    assert disk_cache.startup_clean(tmp_path / "doesnotexist") == 0


# ---------------------------------------------------------------- LRU


def test_lru_evicts_by_count(cache_root: Path) -> None:
    sc = disk_cache.SessionCache(root=cache_root, max_count=3, max_bytes=10**9)
    sc.ensure_dir()
    for i in range(5):
        sc.put(1, f"img{i}.png", _png_bytes(bytes([i])), {}, mode="single")
    assert sc.total_count() == 3
    assert sc.get_image(1, "img0.png") is None
    assert sc.get_image(1, "img1.png") is None
    assert sc.get_image(1, "img4.png") is not None
    assert len(list(sc.session_dir.iterdir())) == 3
    sc.clear_all()


def test_lru_evicts_by_bytes(cache_root: Path) -> None:
    sc = disk_cache.SessionCache(root=cache_root, max_count=10**6, max_bytes=80)
    sc.ensure_dir()
    sc.put(1, "a.png", b"hello", {}, mode="single")
    sc.put(1, "b.png", b"hello", {}, mode="single")
    sc.put(1, "c.png", b"hello", {}, mode="single")
    sc.put(1, "d.png", b"hello", {}, mode="single")
    assert sc.get_image(1, "a.png") is None
    assert sc.get_image(1, "d.png") == b"hello"
    sc.clear_all()


def test_lru_get_marks_recent(cache_root: Path) -> None:
    sc = disk_cache.SessionCache(root=cache_root, max_count=3, max_bytes=10**9)
    sc.ensure_dir()
    sc.put(1, "a.png", _png_bytes(b"a"), {}, mode="single")
    sc.put(1, "b.png", _png_bytes(b"b"), {}, mode="single")
    sc.put(1, "c.png", _png_bytes(b"c"), {}, mode="single")
    assert sc.get_image(1, "a.png") == _png_bytes(b"a")
    sc.put(1, "d.png", _png_bytes(b"d"), {}, mode="single")
    assert sc.get_image(1, "b.png") is None
    assert sc.get_image(1, "a.png") == _png_bytes(b"a")
    sc.clear_all()


def test_configure_shrink_evicts(cache_root: Path) -> None:
    sc = disk_cache.SessionCache(root=cache_root, max_count=10, max_bytes=10**9)
    sc.ensure_dir()
    for i in range(5):
        sc.put(1, f"i{i}.png", _png_bytes(bytes([i])), {}, mode="single")
    assert sc.total_count() == 5
    sc.configure(max_count=2)
    assert sc.total_count() == 2
    assert sc.get_image(1, "i4.png") is not None
    assert sc.get_image(1, "i0.png") is None
    sc.clear_all()




def test_different_session_cannot_decrypt(cache_root: Path) -> None:
    sc1 = disk_cache.SessionCache(root=cache_root, max_count=10, max_bytes=10**9)
    sc1.ensure_dir()
    sc1.put(1, "secret.png", _png_bytes(b"sensitive"), {}, mode="single")
    file_path = next(iter(sc1._index.values())).file_path
    blob = file_path.read_bytes()
    sc2 = disk_cache.SessionCache(root=cache_root, max_count=10, max_bytes=10**9)
    original = _png_bytes(b"sensitive")
    try:
        result = disk_cache._decrypt_and_strip(sc2.aes_key, blob)
    except (ValueError, Exception):
        pass
    else:
        assert result != original
    sc1.clear_all()


# ---------------------------------------------------------------- module-level


def test_module_init_clears_stale_session_dirs(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    (root / "session-stale1").mkdir()
    (root / "session-stale2").mkdir()
    monkeypatch.setattr(disk_cache, "_session", None)
    sc = disk_cache.init(root)
    try:
        assert not (root / "session-stale1").exists()
        assert not (root / "session-stale2").exists()
        assert sc.session_dir.exists()
    finally:
        sc.clear_all()
        monkeypatch.setattr(disk_cache, "_session", None)


def test_module_get_session_raises_when_not_initialized(monkeypatch) -> None:
    monkeypatch.setattr(disk_cache, "_session", None)
    with pytest.raises(RuntimeError, match="not initialized"):
        disk_cache.get_session()


def test_module_clear_all_safe_when_uninitialized(monkeypatch) -> None:
    monkeypatch.setattr(disk_cache, "_session", None)
    disk_cache.clear_all()
    assert disk_cache.total_count() == 0
    assert disk_cache.total_bytes() == 0
