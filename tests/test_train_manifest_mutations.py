from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from studio.services.preprocess import manifest as pm


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    (tmp_path / "download").mkdir()
    (tmp_path / "preprocess").mkdir()
    (tmp_path / "versions" / "v1" / "train").mkdir(parents=True)
    return tmp_path


def _train_path(project_dir: Path, name: str, label: str = "v1") -> Path:
    return project_dir / "versions" / label / "train" / name


def _download_path(project_dir: Path, name: str) -> Path:
    return project_dir / "download" / name


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# train_load / train_get_entry / train_all_processed
# ---------------------------------------------------------------------------


def test_load_creates_empty_via_ensure(project_dir: Path) -> None:
    m = pm.train_load(project_dir, "v1")
    assert m == {"version": 2, "images": {}}
    assert pm.train_manifest_path(project_dir, "v1").exists()


def test_get_entry_missing_returns_none(project_dir: Path) -> None:
    assert pm.train_get_entry(project_dir, "v1", "missing.png") is None


def test_all_processed_filters_duplicate_removed(project_dir: Path) -> None:
    pm.train_add_processed(project_dir, "v1", "A.png", {"origin": "A.jpg"})
    _train_path(project_dir, "B.png").write_bytes(b"b")
    pm.train_mark_duplicate_removed(project_dir, "v1", ["B.png"])

    processed = pm.train_all_processed(project_dir, "v1")
    assert set(processed) == {"A.png"}


def test_duplicate_removed_origins_collects_all(project_dir: Path) -> None:
    _train_path(project_dir, "X.png").write_bytes(b"x")
    _train_path(project_dir, "Y.png").write_bytes(b"y")
    pm.train_mark_duplicate_removed(project_dir, "v1", ["X.png", "Y.png"])

    origins = pm.train_duplicate_removed_origins(project_dir, "v1")
    assert origins == {"X.png", "Y.png"}


# ---------------------------------------------------------------------------
# train_add_processed
# ---------------------------------------------------------------------------


def test_add_processed_stores_minimal_schema(project_dir: Path) -> None:
    _train_path(project_dir, "X.png").write_bytes(b"\x89PNG" + b"x" * 100)
    pm.train_add_processed(project_dir, "v1", "X.png", {
        "origin": "X.jpg",
        "model": "RealESRGAN_x4",
        "scale": 4,
        "action": "upscale",
        "src_size": [512, 512],
        "mtime": 1731000000,
    })

    m = pm.train_load(project_dir, "v1")
    entry = m["images"]["X.png"]
    assert entry["origin"] == "X.jpg"
    assert entry["mtime"] == 1731000000
    assert entry["size"] == 104  # stat'd train/X.png
    assert "model" not in entry
    assert "scale" not in entry
    assert "action" not in entry


def test_add_processed_size_fallback_stat_train(project_dir: Path) -> None:
    _train_path(project_dir, "Y.png").write_bytes(b"y" * 42)
    pm.train_add_processed(project_dir, "v1", "Y.png", {"origin": "Y.jpg"})

    entry = pm.train_get_entry(project_dir, "v1", "Y.png")
    assert entry is not None
    assert entry["size"] == 42


def test_add_processed_origin_fallback_to_name(project_dir: Path) -> None:
    _train_path(project_dir, "Z.jpg").write_bytes(b"z")
    pm.train_add_processed(project_dir, "v1", "Z.jpg", {})

    entry = pm.train_get_entry(project_dir, "v1", "Z.jpg")
    assert entry is not None
    assert entry["origin"] == "Z.jpg"


# ---------------------------------------------------------------------------
# train_replace_with_crops
# ---------------------------------------------------------------------------


def test_replace_with_crops_removes_source_and_writes_fan_out(
    project_dir: Path,
) -> None:
    _train_path(project_dir, "X.png").write_bytes(b"x" * 10)
    pm.train_add_processed(project_dir, "v1", "X.png", {"origin": "X.jpg"})

    # multi-crop: X.png → X_c0.png / X_c1.png
    pm.train_replace_with_crops(
        project_dir, "v1",
        source_name="X.png",
        outputs=[
            {"name": "X_c0.png", "origin": "X.jpg", "mtime": 1, "size": 100},
            {"name": "X_c1.png", "origin": "X.jpg", "mtime": 1, "size": 110},
        ],
    )

    m = pm.train_load(project_dir, "v1")
    assert "X.png" not in m["images"]
    assert m["images"]["X_c0.png"]["origin"] == "X.jpg"
    assert m["images"]["X_c1.png"]["origin"] == "X.jpg"
    assert m["images"]["X_c1.png"]["size"] == 110


def test_replace_with_crops_origin_fallback_to_source(
    project_dir: Path,
) -> None:
    _train_path(project_dir, "raw.png").write_bytes(b"r")
    pm.train_add_processed(project_dir, "v1", "raw.png", {"origin": "raw.png"})

    pm.train_replace_with_crops(
        project_dir, "v1",
        source_name="raw.png",
        outputs=[{"name": "raw_c0.png", "size": 50}],
    )

    entry = pm.train_get_entry(project_dir, "v1", "raw_c0.png")
    assert entry is not None
    assert entry["origin"] == "raw.png"


# ---------------------------------------------------------------------------
# train_mark_duplicate_removed
# ---------------------------------------------------------------------------


def test_mark_duplicate_removed_on_existing_processed(project_dir: Path) -> None:
    _train_path(project_dir, "A.png").write_bytes(b"a")
    _train_path(project_dir, "A.txt").write_text("tag", encoding="utf-8")
    pm.train_add_processed(project_dir, "v1", "A.png", {"origin": "A.jpg"})

    result = pm.train_mark_duplicate_removed(project_dir, "v1", ["A.png"])
    assert result == {"removed": ["A.png"], "missing": [], "skipped": []}

    entry = pm.train_get_entry(project_dir, "v1", "A.png")
    assert entry is not None
    assert entry["kind"] == pm.DUPLICATE_REMOVED_KIND
    assert entry["origin"] == "A.jpg"
    assert not _train_path(project_dir, "A.png").exists()
    assert not _train_path(project_dir, "A.txt").exists()


def test_mark_duplicate_removed_on_unrecorded_train_file(
    project_dir: Path,
) -> None:
    _train_path(project_dir, "B.png").write_bytes(b"b" * 5)

    result = pm.train_mark_duplicate_removed(project_dir, "v1", ["B.png"])
    assert result == {"removed": ["B.png"], "missing": [], "skipped": []}

    entry = pm.train_get_entry(project_dir, "v1", "B.png")
    assert entry is not None
    assert entry["origin"] == "B.png"
    assert entry["size"] == 5


def test_mark_duplicate_removed_missing_returns_missing(project_dir: Path) -> None:
    result = pm.train_mark_duplicate_removed(project_dir, "v1", ["ghost.png"])
    assert result["missing"] == ["ghost.png"]
    assert result["removed"] == []


def test_mark_duplicate_removed_already_marked_skipped(project_dir: Path) -> None:
    _train_path(project_dir, "C.png").write_bytes(b"c")
    pm.train_mark_duplicate_removed(project_dir, "v1", ["C.png"])

    result = pm.train_mark_duplicate_removed(project_dir, "v1", ["C.png"])
    assert result["skipped"] == ["C.png"]
    assert result["removed"] == []


# ---------------------------------------------------------------------------
# train_restore_duplicate_removed
# ---------------------------------------------------------------------------


def test_restore_duplicate_removed_unwinds_mark(project_dir: Path) -> None:
    (project_dir / "download").mkdir(exist_ok=True)
    (project_dir / "download" / "D.png").write_bytes(b"d-original")
    (project_dir / "download" / "D.txt").write_text("tag", encoding="utf-8")
    _train_path(project_dir, "D.png").write_bytes(b"d-train")
    pm.train_mark_duplicate_removed(project_dir, "v1", ["D.png"])
    assert not _train_path(project_dir, "D.png").exists()

    result = pm.train_restore_duplicate_removed(project_dir, "v1", ["D.png"])
    assert result == {"restored": ["D.png"], "missing": [], "no_origin": []}
    assert pm.train_get_entry(project_dir, "v1", "D.png") is None
    assert _train_path(project_dir, "D.png").read_bytes() == b"d-original"
    assert _train_path(project_dir, "D.txt").read_text(encoding="utf-8") == "tag"


def test_restore_duplicate_removed_no_origin(project_dir: Path) -> None:
    _train_path(project_dir, "F.png").write_bytes(b"f")
    pm.train_add_processed(project_dir, "v1", "F.png", {"origin": "F.jpg"})
    pm.train_mark_duplicate_removed(project_dir, "v1", ["F.png"])

    result = pm.train_restore_duplicate_removed(project_dir, "v1", ["F.png"])
    assert result == {"restored": [], "missing": [], "no_origin": ["F.png"]}
    entry = pm.train_get_entry(project_dir, "v1", "F.png")
    assert entry is not None and entry["kind"] == pm.DUPLICATE_REMOVED_KIND


def test_restore_duplicate_removed_missing_when_not_marked(
    project_dir: Path,
) -> None:
    _train_path(project_dir, "E.png").write_bytes(b"e")
    pm.train_add_processed(project_dir, "v1", "E.png", {"origin": "E.jpg"})

    result = pm.train_restore_duplicate_removed(project_dir, "v1", ["E.png"])
    assert result == {"restored": [], "missing": ["E.png"], "no_origin": []}
    entry = pm.train_get_entry(project_dir, "v1", "E.png")
    assert entry is not None
    assert entry.get("kind", "processed") == "processed"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_restore_copies_from_download_overwriting_train(
    project_dir: Path,
) -> None:
    _download_path(project_dir, "X.jpg").write_bytes(b"orig" * 10)
    _train_path(project_dir, "X.png").write_bytes(b"upscaled" * 100)
    pm.train_add_processed(project_dir, "v1", "X.png", {"origin": "X.jpg"})

    result = pm.train_restore(project_dir, "v1", ["X.png"])

    assert result == {"restored": ["X.png"], "missing": [], "no_origin": []}
    assert _train_path(project_dir, "X.jpg").read_bytes() == b"orig" * 10
    assert not _train_path(project_dir, "X.png").exists()
    assert pm.train_get_entry(project_dir, "v1", "X.png") is None
    entry = pm.train_get_entry(project_dir, "v1", "X.jpg")
    assert entry is not None
    assert entry["origin"] == "X.jpg"
    assert entry["size"] == 40


def test_restore_collapses_fan_out_to_single_origin(
    project_dir: Path,
) -> None:
    _download_path(project_dir, "a.jpg").write_bytes(b"orig" * 10)
    sub = project_dir / "versions" / "v1" / "train" / "1_data"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "a_0.png").write_bytes(b"crop0")
    (sub / "a_1.png").write_bytes(b"crop1")
    pm.train_replace_with_crops(
        project_dir, "v1",
        source_name="1_data/a.jpg",
        outputs=[
            {"name": "1_data/a_0.png", "origin": "a.jpg", "size": 5},
            {"name": "1_data/a_1.png", "origin": "a.jpg", "size": 5},
        ],
    )

    result = pm.train_restore(project_dir, "v1", ["1_data/a_0.png"])
    assert result["restored"] == ["1_data/a_0.png"]
    assert result["no_origin"] == [] and result["missing"] == []
    assert (sub / "a.jpg").read_bytes() == b"orig" * 10
    assert not (sub / "a_0.png").exists()
    assert not (sub / "a_1.png").exists()
    assert pm.train_get_entry(project_dir, "v1", "1_data/a_0.png") is None
    assert pm.train_get_entry(project_dir, "v1", "1_data/a_1.png") is None
    new_entry = pm.train_get_entry(project_dir, "v1", "1_data/a.jpg")
    assert new_entry is not None and new_entry["origin"] == "a.jpg"


def test_restore_fan_out_batch_handles_siblings_once(
    project_dir: Path,
) -> None:
    _download_path(project_dir, "a.jpg").write_bytes(b"orig")
    sub = project_dir / "versions" / "v1" / "train" / "1_data"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "a_0.png").write_bytes(b"c0")
    (sub / "a_1.png").write_bytes(b"c1")
    pm.train_replace_with_crops(
        project_dir, "v1",
        source_name="1_data/a.jpg",
        outputs=[
            {"name": "1_data/a_0.png", "origin": "a.jpg", "size": 2},
            {"name": "1_data/a_1.png", "origin": "a.jpg", "size": 2},
        ],
    )

    result = pm.train_restore(
        project_dir, "v1", ["1_data/a_0.png", "1_data/a_1.png"]
    )
    assert sorted(result["restored"]) == ["1_data/a_0.png", "1_data/a_1.png"]
    assert result["missing"] == [] and result["no_origin"] == []


def test_restore_missing_when_no_manifest_entry(project_dir: Path) -> None:
    result = pm.train_restore(project_dir, "v1", ["ghost.png"])
    assert result == {"restored": [], "missing": ["ghost.png"], "no_origin": []}


def test_restore_no_origin_when_download_missing(project_dir: Path) -> None:
    _train_path(project_dir, "Y.png").write_bytes(b"y")
    pm.train_add_processed(project_dir, "v1", "Y.png", {"origin": "Y.jpg"})

    result = pm.train_restore(project_dir, "v1", ["Y.png"])

    assert result == {"restored": [], "missing": [], "no_origin": ["Y.png"]}
    assert _train_path(project_dir, "Y.png").read_bytes() == b"y"


def test_restore_mixed_results(project_dir: Path) -> None:
    _download_path(project_dir, "OK.jpg").write_bytes(b"ok")
    _train_path(project_dir, "OK.png").write_bytes(b"old-ok")
    _train_path(project_dir, "NoOrig.png").write_bytes(b"x")
    pm.train_add_processed(project_dir, "v1", "OK.png", {"origin": "OK.jpg"})
    pm.train_add_processed(
        project_dir, "v1", "NoOrig.png", {"origin": "missing.jpg"}
    )

    result = pm.train_restore(
        project_dir, "v1", ["OK.png", "NoOrig.png", "Ghost.png"]
    )

    assert result["restored"] == ["OK.png"]
    assert result["no_origin"] == ["NoOrig.png"]
    assert result["missing"] == ["Ghost.png"]


# ---------------------------------------------------------------------------
# train_clear_all
# ---------------------------------------------------------------------------


def test_clear_all_empties_manifest_keeps_train_files(project_dir: Path) -> None:
    _train_path(project_dir, "P.png").write_bytes(b"p")
    pm.train_add_processed(project_dir, "v1", "P.png", {"origin": "P.jpg"})

    pm.train_clear_all(project_dir, "v1")

    m = pm.train_load(project_dir, "v1")
    assert m == {"version": 2, "images": {}}
    assert _train_path(project_dir, "P.png").exists()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_multi_version_isolation(project_dir: Path) -> None:
    (project_dir / "versions" / "v2" / "train").mkdir(parents=True)
    (project_dir / "versions" / "v2" / "train" / "Q.png").write_bytes(b"q")

    pm.train_add_processed(project_dir, "v1", "v1-only.png", {"origin": "x.jpg"})
    pm.train_add_processed(project_dir, "v2", "Q.png", {"origin": "Q.jpg"})

    v1 = pm.train_load(project_dir, "v1")
    v2 = pm.train_load(project_dir, "v2")
    assert set(v1["images"]) == {"v1-only.png"}
    assert set(v2["images"]) == {"Q.png"}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_train_mutations_dont_touch_project_manifest(project_dir: Path) -> None:
    pm.manifest_path(project_dir).write_text(
        json.dumps({"images": {"legacy.png": {"origin": "legacy.jpg"}}}),
        encoding="utf-8",
    )
    legacy_mtime = pm.manifest_path(project_dir).stat().st_mtime_ns

    _train_path(project_dir, "X.png").write_bytes(b"x")
    pm.train_add_processed(project_dir, "v1", "X.png", {"origin": "X.jpg"})
    pm.train_mark_duplicate_removed(project_dir, "v1", ["X.png"])
    pm.train_clear_all(project_dir, "v1")

    assert pm.manifest_path(project_dir).stat().st_mtime_ns == legacy_mtime
    legacy = _read(pm.manifest_path(project_dir))
    assert legacy == {"images": {"legacy.png": {"origin": "legacy.jpg"}}}
