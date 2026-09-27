from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from studio.services.preprocess import manifest as pm


DEFAULT_FOLDER = "1_data"


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    (tmp_path / "download").mkdir()
    (tmp_path / "preprocess").mkdir()
    return tmp_path


def _train_subfolder(
    project_dir: Path, label: str = "v1", folder: str = DEFAULT_FOLDER
) -> Path:
    d = project_dir / "versions" / label / "train" / folder
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_legacy(project_dir: Path, images: dict) -> None:
    pm.manifest_path(project_dir).write_text(
        json.dumps({"images": images}, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_train_manifest(project_dir: Path, label: str = "v1") -> dict:
    return json.loads(
        pm.train_manifest_path(project_dir, label).read_text(encoding="utf-8")
    )


def _rel(name: str, folder: str = DEFAULT_FOLDER) -> str:
    return f"{folder}/{name}"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_ensure_returns_existing_without_rewrite(project_dir: Path) -> None:
    _train_subfolder(project_dir)
    existing = {
        "version": 2,
        "images": {
            _rel("keep.png"): {"origin": "keep.jpg", "mtime": 999, "size": 42}
        },
    }
    target = pm.train_manifest_path(project_dir, "v1")
    target.write_text(json.dumps(existing), encoding="utf-8")
    _write_legacy(project_dir, {"keep.png": {"origin": "DIFFERENT.jpg"}})

    returned = pm.ensure_train_manifest(project_dir, "v1")

    assert returned == target
    assert _read_train_manifest(project_dir) == existing


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_no_legacy_no_train_writes_empty(project_dir: Path) -> None:
    target = pm.ensure_train_manifest(project_dir, "v1")
    assert target.exists()
    assert _read_train_manifest(project_dir) == {"version": 2, "images": {}}
    assert target.parent.exists()


def test_no_legacy_with_train_files_writes_empty(project_dir: Path) -> None:
    sub = _train_subfolder(project_dir)
    (sub / "foo.png").write_bytes(b"\x89PNG")

    pm.ensure_train_manifest(project_dir, "v1")
    assert _read_train_manifest(project_dir) == {"version": 2, "images": {}}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_legacy_full_match_rebuilds(project_dir: Path) -> None:
    sub = _train_subfolder(project_dir)
    (sub / "X.png").write_bytes(b"x")
    (sub / "Y.png").write_bytes(b"y")
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
        "Y.png": {"origin": "Y.jpg", "mtime": 200, "size": 2000},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert data["version"] == 2
    assert data["images"] == {
        _rel("X.png"): {"origin": "X.jpg", "mtime": 100, "size": 1000},
        _rel("Y.png"): {"origin": "Y.jpg", "mtime": 200, "size": 2000},
    }


def test_legacy_entry_missing_in_train_is_skipped(project_dir: Path) -> None:
    sub = _train_subfolder(project_dir)
    (sub / "X.png").write_bytes(b"x")
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
        "Y.png": {"origin": "Y.jpg", "mtime": 200, "size": 2000},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert set(data["images"].keys()) == {_rel("X.png")}


def test_train_file_not_in_legacy_is_skipped(project_dir: Path) -> None:
    sub = _train_subfolder(project_dir)
    (sub / "X.png").write_bytes(b"x")
    (sub / "Z.png").write_bytes(b"z")
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert set(data["images"].keys()) == {_rel("X.png")}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_multi_crop_fan_out_preserved(project_dir: Path) -> None:
    sub = _train_subfolder(project_dir)
    (sub / "Y_c0.png").write_bytes(b"c0")
    (sub / "Y_c1.png").write_bytes(b"c1")
    _write_legacy(project_dir, {
        "Y_c0.png": {"origin": "Y.jpg", "mtime": 200, "size": 800},
        "Y_c1.png": {"origin": "Y.jpg", "mtime": 200, "size": 850},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert data["images"][_rel("Y_c0.png")]["origin"] == "Y.jpg"
    assert data["images"][_rel("Y_c1.png")]["origin"] == "Y.jpg"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_duplicate_removed_legacy_entry_skipped(project_dir: Path) -> None:
    sub = _train_subfolder(project_dir)
    (sub / "X.png").write_bytes(b"x")
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
        "dup.jpg": {"kind": pm.DUPLICATE_REMOVED_KIND, "origin": "dup.jpg"},
    })
    (sub / "dup.jpg").write_bytes(b"dup")

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert set(data["images"].keys()) == {_rel("X.png")}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_corrupted_legacy_treated_as_missing(project_dir: Path) -> None:
    sub = _train_subfolder(project_dir)
    (sub / "X.png").write_bytes(b"x")
    pm.manifest_path(project_dir).write_text("not json", encoding="utf-8")

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert data == {"version": 2, "images": {}}


def test_legacy_invalid_shape_treated_as_missing(project_dir: Path) -> None:
    sub = _train_subfolder(project_dir)
    (sub / "X.png").write_bytes(b"x")
    pm.manifest_path(project_dir).write_text(json.dumps(["array"]), encoding="utf-8")

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert data == {"version": 2, "images": {}}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_idempotent_second_call_no_rewrite(project_dir: Path) -> None:
    sub = _train_subfolder(project_dir)
    (sub / "X.png").write_bytes(b"x")
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    target = pm.train_manifest_path(project_dir, "v1")
    first_mtime = target.stat().st_mtime_ns

    pm.ensure_train_manifest(project_dir, "v1")
    assert target.stat().st_mtime_ns == first_mtime


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_non_image_files_in_train_ignored(project_dir: Path) -> None:
    sub = _train_subfolder(project_dir)
    (sub / "X.png").write_bytes(b"x")
    (sub / "X.txt").write_text("caption")
    (sub / "manifest.json.tmp").write_text("{}")
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
        "X.txt": {"origin": "X.txt"},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert set(data["images"].keys()) == {_rel("X.png")}


def test_train_root_files_ignored(project_dir: Path) -> None:
    train_root = project_dir / "versions" / "v1" / "train"
    train_root.mkdir(parents=True)
    (train_root / "stray.png").write_bytes(b"stray")
    _write_legacy(project_dir, {"stray.png": {"origin": "stray.jpg"}})

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert data == {"version": 2, "images": {}}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_concurrent_ensure_yields_single_manifest(project_dir: Path) -> None:
    sub = _train_subfolder(project_dir)
    (sub / "X.png").write_bytes(b"x")
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
    })

    results: list[Path] = []
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            results.append(pm.ensure_train_manifest(project_dir, "v1"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(set(map(str, results))) == 1
    data = _read_train_manifest(project_dir)
    assert data["version"] == 2
    assert data["images"][_rel("X.png")]["origin"] == "X.jpg"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_multiple_versions_independent(project_dir: Path) -> None:
    v1_sub = _train_subfolder(project_dir, "v1")
    v2_sub = _train_subfolder(project_dir, "v2")
    (v1_sub / "X.png").write_bytes(b"x")
    (v2_sub / "Y.png").write_bytes(b"y")
    _write_legacy(project_dir, {
        "X.png": {"origin": "X.jpg", "mtime": 100, "size": 1000},
        "Y.png": {"origin": "Y.jpg", "mtime": 200, "size": 2000},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    pm.ensure_train_manifest(project_dir, "v2")

    v1_data = _read_train_manifest(project_dir, "v1")
    v2_data = _read_train_manifest(project_dir, "v2")
    assert set(v1_data["images"].keys()) == {_rel("X.png")}
    assert set(v2_data["images"].keys()) == {_rel("Y.png")}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_cross_subfolder_same_filename_both_entry(project_dir: Path) -> None:
    sub_a = _train_subfolder(project_dir, folder="1_data")
    sub_b = _train_subfolder(project_dir, folder="5_extra")
    (sub_a / "shared.png").write_bytes(b"a")
    (sub_b / "shared.png").write_bytes(b"b")
    _write_legacy(project_dir, {
        "shared.png": {"origin": "shared.jpg", "mtime": 100, "size": 500},
    })

    pm.ensure_train_manifest(project_dir, "v1")
    data = _read_train_manifest(project_dir)
    assert set(data["images"].keys()) == {
        "1_data/shared.png",
        "5_extra/shared.png",
    }
    assert data["images"]["1_data/shared.png"]["origin"] == "shared.jpg"
    assert data["images"]["5_extra/shared.png"]["origin"] == "shared.jpg"
