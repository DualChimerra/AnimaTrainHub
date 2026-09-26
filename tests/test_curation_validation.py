from __future__ import annotations

from pathlib import Path

import pytest

from studio import db
from studio.services.dataset import curation
from studio.services.projects import projects, versions


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
    return {"db": dbfile, "p": p, "v": v}


def _dl(env, name: str, blob: bytes = b"img") -> Path:
    pdir = projects.project_dir(env["p"]["id"], env["p"]["slug"])
    f = pdir / "download" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(blob)
    return f


def _val_dir(env, folder: str = "1_data") -> Path:
    return (
        versions.version_dir(env["p"]["id"], env["p"]["slug"], env["v"]["label"])
        / "validation"
        / folder
    )


def _pid_vid(env) -> tuple[int, int]:
    return env["p"]["id"], env["v"]["id"]


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


def test_copy_lands_in_fixed_folder_with_caption(env) -> None:
    _dl(env, "1.png")
    (projects.project_dir(env["p"]["id"], env["p"]["slug"]) / "download" / "1.txt").write_text(
        "a cat", encoding="utf-8"
    )
    pid, vid = _pid_vid(env)
    with db.connection_for(env["db"]) as conn:
        r = curation.copy_download_to_validation(conn, pid, vid, ["1.png"])
    assert r["copied"] == ["1.png"]
    assert (_val_dir(env) / "1.png").exists()
    assert (_val_dir(env) / "1.txt").read_text(encoding="utf-8") == "a cat"
    assert (projects.project_dir(env["p"]["id"], env["p"]["slug"]) / "download" / "1.png").exists()


def test_copy_missing_reported(env) -> None:
    pid, vid = _pid_vid(env)
    with db.connection_for(env["db"]) as conn:
        r = curation.copy_download_to_validation(conn, pid, vid, ["ghost.png"])
    assert r["missing"] == ["ghost.png"]
    assert r["copied"] == []


def test_copy_skips_already_in_validation(env) -> None:
    _dl(env, "1.png")
    pid, vid = _pid_vid(env)
    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_validation(conn, pid, vid, ["1.png"])
        r = curation.copy_download_to_validation(conn, pid, vid, ["1.png"])
    assert r["skipped"] == ["1.png"]
    assert r["copied"] == []


def test_copy_skips_name_in_train_no_leak(env) -> None:
    _dl(env, "1.png")
    pid, vid = _pid_vid(env)
    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_train(conn, pid, vid, ["1.png"], "5_concept")
        r = curation.copy_download_to_validation(conn, pid, vid, ["1.png"])
    assert r["skipped"] == ["1.png"]
    assert r["copied"] == []
    assert not (_val_dir(env) / "1.png").exists()


# ---------------------------------------------------------------------------
# list / view
# ---------------------------------------------------------------------------


def test_list_validation_flattens_folders(env) -> None:
    _dl(env, "1.png")
    pid, vid = _pid_vid(env)
    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_validation(conn, pid, vid, ["1.png"])
        other = _val_dir(env, "5_concept")
        other.mkdir(parents=True, exist_ok=True)
        (other / "2.png").write_bytes(b"img")
        items = curation.list_validation(conn, pid, vid)
    names = {(it["name"], it["folder"]) for it in items}
    assert names == {("1.png", "1_data"), ("2.png", "5_concept")}


def test_validation_view_left_excludes_train_and_validation(env) -> None:
    _dl(env, "1.png")
    _dl(env, "2.png")
    _dl(env, "3.png")
    pid, vid = _pid_vid(env)
    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_train(conn, pid, vid, ["1.png"], "5_concept")
        curation.copy_download_to_validation(conn, pid, vid, ["2.png"])
        view = curation.curation_validation_view(conn, pid, vid)
    left_names = {e["name"] for e in view["left"]}
    assert left_names == {"3.png"}
    right_names = {e["name"] for e in view["right"]}
    assert right_names == {"2.png"}
    assert view["val_total"] == 1


def test_train_view_left_excludes_validation(env) -> None:
    _dl(env, "1.png")
    _dl(env, "2.png")
    pid, vid = _pid_vid(env)
    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_validation(conn, pid, vid, ["1.png"])
        view = curation.curation_view(conn, pid, vid)
    left_names = {e["name"] for e in view["left"]}
    assert left_names == {"2.png"}


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def test_remove_by_folder_and_name(env) -> None:
    _dl(env, "1.png")
    (projects.project_dir(env["p"]["id"], env["p"]["slug"]) / "download" / "1.txt").write_text(
        "x", encoding="utf-8"
    )
    pid, vid = _pid_vid(env)
    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_validation(conn, pid, vid, ["1.png"])
        r = curation.remove_from_validation(
            conn, pid, vid, [{"folder": "1_data", "name": "1.png"}]
        )
    assert r["removed"] == ["1.png"]
    assert not (_val_dir(env) / "1.png").exists()
    assert not (_val_dir(env) / "1.txt").exists()


def test_remove_missing_reported(env) -> None:
    pid, vid = _pid_vid(env)
    with db.connection_for(env["db"]) as conn:
        r = curation.remove_from_validation(
            conn, pid, vid, [{"folder": "1_data", "name": "ghost.png"}]
        )
    assert r["missing"] == ["ghost.png"]
    assert r["removed"] == []
