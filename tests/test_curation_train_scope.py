from __future__ import annotations

from pathlib import Path

import pytest

from studio import db
from studio.services.dataset import curation
from studio.services.preprocess import manifest as preprocess_manifest
from studio.services.projects import projects, versions


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
    return {"db": dbfile, "p": p, "v": v}


def _pdir(env) -> Path:
    return projects.project_dir(env["p"]["id"], env["p"]["slug"])


def _dl(env, name: str, blob: bytes = b"img") -> Path:
    f = _pdir(env) / "download" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(blob)
    return f


def _train(env, folder: str = "1_data") -> Path:
    return _pdir(env) / "versions" / env["v"]["label"] / "train" / folder


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_copy_download_to_train_writes_manifest_entry(env) -> None:
    _dl(env, "X.jpg", blob=b"orig" * 10)
    with db.connection_for(env["db"]) as conn:
        result = curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["X.jpg"], dest_folder="1_data",
        )
    assert result == {"copied": ["X.jpg"], "skipped": [], "missing": []}
    assert (_train(env) / "X.jpg").read_bytes() == b"orig" * 10
    entry = preprocess_manifest.train_get_entry(
        _pdir(env), env["v"]["label"], "1_data/X.jpg"
    )
    assert entry is not None
    assert entry["origin"] == "X.jpg"
    assert entry["size"] == 40


def test_copy_download_to_train_copies_caption(env) -> None:
    _dl(env, "Y.jpg")
    (_pdir(env) / "download" / "Y.txt").write_text("a tag, b tag")
    (_pdir(env) / "download" / "Y.json").write_text('{"k":1}')
    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["Y.jpg"], dest_folder="1_data",
        )
    assert (_train(env) / "Y.txt").read_text() == "a tag, b tag"
    assert (_train(env) / "Y.json").read_text() == '{"k":1}'


def test_copy_download_to_train_skips_existing(env) -> None:
    _dl(env, "Z.jpg")
    dst = _train(env)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "Z.jpg").write_bytes(b"existing")

    with db.connection_for(env["db"]) as conn:
        result = curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["Z.jpg"], dest_folder="1_data",
        )
    assert result == {"copied": [], "skipped": ["Z.jpg"], "missing": []}
    assert (dst / "Z.jpg").read_bytes() == b"existing"


def test_copy_download_to_train_missing(env) -> None:
    with db.connection_for(env["db"]) as conn:
        result = curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["ghost.jpg"], dest_folder="1_data",
        )
    assert result == {"copied": [], "skipped": [], "missing": ["ghost.jpg"]}


def test_copy_download_to_train_multiple_folders_indep_entries(env) -> None:
    _dl(env, "shared.jpg", blob=b"s")
    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["shared.jpg"], dest_folder="1_data",
        )
        curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["shared.jpg"], dest_folder="5_extra",
        )
    m = preprocess_manifest.train_load(_pdir(env), env["v"]["label"])
    assert "1_data/shared.jpg" in m["images"]
    assert "5_extra/shared.jpg" in m["images"]
    assert m["images"]["1_data/shared.jpg"]["origin"] == "shared.jpg"
    assert m["images"]["5_extra/shared.jpg"]["origin"] == "shared.jpg"


def test_copy_download_to_train_no_preprocess_branch(env) -> None:
    import json
    _dl(env, "X.jpg", blob=b"raw")
    pre = _pdir(env) / "preprocess"
    pre.mkdir(parents=True, exist_ok=True)
    (pre / "X.png").write_bytes(b"upscaled-residue")
    preprocess_manifest.manifest_path(_pdir(env)).write_text(
        json.dumps({"images": {"X.png": {"origin": "X.jpg"}}}),
        encoding="utf-8",
    )

    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            files=["X.jpg"], dest_folder="1_data",
        )

    assert (_train(env) / "X.jpg").read_bytes() == b"raw"
    entry = preprocess_manifest.train_get_entry(
        _pdir(env), env["v"]["label"], "1_data/X.jpg"
    )
    assert entry["origin"] == "X.jpg"


def test_copy_download_to_train_invalid_folder_rejected(env) -> None:
    _dl(env, "X.jpg")
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(curation.CurationError):
            curation.copy_download_to_train(
                conn, env["p"]["id"], env["v"]["id"],
                files=["X.jpg"], dest_folder="../escape",
            )


def test_copy_download_to_train_invalid_filename_rejected(env) -> None:
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(curation.CurationError):
            curation.copy_download_to_train(
                conn, env["p"]["id"], env["v"]["id"],
                files=["../../etc/passwd"], dest_folder="1_data",
            )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_list_train_dedupes_by_origin_on_fan_out(env) -> None:
    train_sub = _train(env, "1_data")
    train_sub.mkdir(parents=True, exist_ok=True)
    (train_sub / "X_c0.png").write_bytes(b"c0")
    (train_sub / "X_c1.png").write_bytes(b"c1")
    preprocess_manifest.train_replace_with_crops(
        _pdir(env), env["v"]["label"],
        source_name="1_data/X.jpg",
        outputs=[
            {"name": "1_data/X_c0.png", "origin": "X.jpg", "mtime": 1, "size": 10},
            {"name": "1_data/X_c1.png", "origin": "X.jpg", "mtime": 1, "size": 10},
        ],
    )

    with db.connection_for(env["db"]) as conn:
        view = curation.curation_view(conn, env["p"]["id"], env["v"]["id"])
    assert [e["name"] for e in view["right"]["1_data"]] == ["X.jpg"]
    assert view["right"]["1_data"][0]["origin"] == "X.jpg"


def test_list_train_excludes_duplicate_removed_after_physical_delete(env) -> None:
    train_sub = _train(env, "1_data")
    train_sub.mkdir(parents=True, exist_ok=True)
    (train_sub / "Y.jpg").write_bytes(b"y")
    preprocess_manifest.train_add_processed(
        _pdir(env), env["v"]["label"], "1_data/Y.jpg", {"origin": "Y.jpg"},
    )
    preprocess_manifest.train_mark_duplicate_removed(
        _pdir(env), env["v"]["label"], ["1_data/Y.jpg"],
    )
    assert not (train_sub / "Y.jpg").exists()

    with db.connection_for(env["db"]) as conn:
        view = curation.curation_view(conn, env["p"]["id"], env["v"]["id"])
    assert "1_data" not in view["right"] or view["right"]["1_data"] == []


def test_remove_from_train_deletes_all_fan_out_derivatives(env) -> None:
    train_sub = _train(env, "1_data")
    train_sub.mkdir(parents=True, exist_ok=True)
    (train_sub / "X_c0.png").write_bytes(b"c0")
    (train_sub / "X_c1.png").write_bytes(b"c1")
    preprocess_manifest.train_replace_with_crops(
        _pdir(env), env["v"]["label"],
        source_name="1_data/X.jpg",
        outputs=[
            {"name": "1_data/X_c0.png", "origin": "X.jpg", "mtime": 1, "size": 10},
            {"name": "1_data/X_c1.png", "origin": "X.jpg", "mtime": 1, "size": 10},
        ],
    )

    with db.connection_for(env["db"]) as conn:
        res = curation.remove_from_train(
            conn, env["p"]["id"], env["v"]["id"], "1_data", ["X.jpg"],
        )
    assert res["removed"] == ["X.jpg"]
    assert not (train_sub / "X_c0.png").exists()
    assert not (train_sub / "X_c1.png").exists()
    m = preprocess_manifest.train_load(_pdir(env), env["v"]["label"])
    assert "1_data/X_c0.png" not in m["images"]
    assert "1_data/X_c1.png" not in m["images"]
