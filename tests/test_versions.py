from __future__ import annotations

from pathlib import Path

import pytest

from studio import db
from studio.services.projects import projects, versions


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    pdir = tmp_path / "projects"
    monkeypatch.setattr(projects, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    return {"db": dbfile}


def _new_project(isolated, title: str = "P1") -> dict:
    with db.connection_for(isolated["db"]) as conn:
        return projects.create_project(conn, title=title)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_version_builds_tree_and_activates(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    assert vdir.exists()
    for sub in ("train", "reg", "output"):
        assert (vdir / sub).is_dir()
    assert not (vdir / "samples").exists()
    assert (vdir / "version.json").exists()
    with db.connection_for(isolated["db"]) as conn:
        p2 = projects.get_project(conn, p["id"])
    assert p2 and p2["active_version_id"] == v["id"]


def test_create_version_rejects_invalid_label(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        for bad in ("has space", "../escape", "name/sub", "中文", ".", "..", "..."):
            with pytest.raises(versions.VersionError, match="label"):
                versions.create_version(conn, project_id=p["id"], label=bad)


def test_create_version_rejects_duplicate_label(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.create_version(conn, project_id=p["id"], label="baseline")
        with pytest.raises(versions.VersionError, match="already exists"):
            versions.create_version(conn, project_id=p["id"], label="baseline")


# ---------------------------------------------------------------------------
# fork
# ---------------------------------------------------------------------------


def test_fork_copies_train_tree(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        src = versions.create_version(conn, project_id=p["id"], label="baseline")
    src_train = versions.version_dir(p["id"], p["slug"], "baseline") / "train"
    folder = src_train / "5_concept"
    folder.mkdir()
    (folder / "001.png").write_bytes(b"fakepng")
    (folder / "001.txt").write_text("tag1, tag2", encoding="utf-8")

    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.create_version(
            conn,
            project_id=p["id"],
            label="forked",
            fork_from_version_id=src["id"],
        )
    new_folder = (
        versions.version_dir(p["id"], p["slug"], "forked")
        / "train" / "5_concept"
    )
    assert (new_folder / "001.png").read_bytes() == b"fakepng"
    assert (new_folder / "001.txt").read_text(encoding="utf-8") == "tag1, tag2"
    assert v2["config_name"] == src["config_name"]


def test_fork_full_copy_includes_reg_config_unlocked(isolated, monkeypatch) -> None:
    from studio.services import version_config
    from studio.schema import TrainingConfig

    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        src = versions.create_version(conn, project_id=p["id"], label="baseline")
    src_vdir = versions.version_dir(p["id"], p["slug"], "baseline")

    (src_vdir / "train" / "5_concept").mkdir()
    (src_vdir / "train" / "5_concept" / "001.png").write_bytes(b"trainpng")

    (src_vdir / "reg" / "1_data").mkdir(parents=True)
    (src_vdir / "reg" / "meta.json").write_text(
        '{"target": 100}', encoding="utf-8"
    )
    (src_vdir / "reg" / "1_data" / "r.png").write_bytes(b"regpng")

    src_cfg = TrainingConfig().model_dump()
    version_config.write_version_config(p, src, src_cfg)

    (src_vdir / ".unlocked.json").write_text(
        '{"fields": ["resume_lora"]}', encoding="utf-8"
    )

    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.create_version(
            conn,
            project_id=p["id"],
            label="forked",
            fork_from_version_id=src["id"],
        )
    new_vdir = versions.version_dir(p["id"], p["slug"], "forked")

    assert (new_vdir / "train" / "5_concept" / "001.png").read_bytes() == b"trainpng"
    assert (new_vdir / "reg" / "meta.json").exists()
    assert (new_vdir / "reg" / "1_data" / "r.png").read_bytes() == b"regpng"
    assert (new_vdir / "config.yaml").exists()
    assert (new_vdir / ".unlocked.json").read_text(encoding="utf-8") == (
        '{"fields": ["resume_lora"]}'
    )

    new_cfg = version_config.read_version_config(p, v2)
    assert new_cfg["data_dir"] == str(new_vdir / "train")
    assert new_cfg["reg_data_dir"] == str(new_vdir / "reg")
    assert new_cfg["output_dir"] == str(new_vdir / "output")
    assert new_cfg["output_name"] == f"{p['slug']}_forked"


def test_fork_stage_done_resets_to_ready(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        src = versions.create_version(conn, project_id=p["id"], label="baseline")
        versions.update_version(conn, src["id"], status="completed")
        v2 = versions.create_version(
            conn,
            project_id=p["id"],
            label="forked",
            fork_from_version_id=src["id"],
        )
    assert v2["status"] == "preparing"
    assert v2["phase"] == "curating"


def test_fork_rejects_alien_source(isolated) -> None:
    a = _new_project(isolated, title="A")
    b = _new_project(isolated, title="B")
    with db.connection_for(isolated["db"]) as conn:
        src = versions.create_version(conn, project_id=a["id"], label="baseline")
        with pytest.raises(versions.VersionError, match="copy from"):
            versions.create_version(
                conn,
                project_id=b["id"],
                label="x",
                fork_from_version_id=src["id"],
            )


# ---------------------------------------------------------------------------
# delete + active reassign
# ---------------------------------------------------------------------------


def test_delete_active_version_reassigns(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v1 = versions.create_version(conn, project_id=p["id"], label="v1")
        v2 = versions.create_version(conn, project_id=p["id"], label="v2")
        versions.activate_version(conn, v2["id"])
        versions.delete_version(conn, v2["id"])
        p2 = projects.get_project(conn, p["id"])
    assert p2 and p2["active_version_id"] == v1["id"]


def test_delete_last_version_clears_active(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v1 = versions.create_version(conn, project_id=p["id"], label="only")
        versions.delete_version(conn, v1["id"])
        p2 = projects.get_project(conn, p["id"])
    assert p2 and p2["active_version_id"] is None


def test_delete_removes_dir(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
        src = versions.version_dir(p["id"], p["slug"], "baseline")
        assert src.exists()
        versions.delete_version(conn, v["id"])
    assert not src.exists()


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_for_version_counts_train_and_reg(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v = versions.create_version(conn, project_id=p["id"], label="v1")
    vdir = versions.version_dir(p["id"], p["slug"], "v1")
    (vdir / "train" / "5_concept").mkdir(parents=True)
    (vdir / "train" / "5_concept" / "a.png").write_bytes(b"x")
    (vdir / "train" / "5_concept" / "b.png").write_bytes(b"x")
    (vdir / "reg" / "1_data").mkdir(parents=True)
    (vdir / "reg" / "1_data" / "r.png").write_bytes(b"x")
    stats = versions.stats_for_version(p, v)
    assert stats["train_image_count"] == 2
    assert stats["reg_image_count"] == 1
    folder_names = {f["name"] for f in stats["train_folders"]}
    assert folder_names == {"1_data", "5_concept"}
    assert {f["name"]: f["image_count"] for f in stats["train_folders"]} == {
        "1_data": 0,
        "5_concept": 2,
    }
    assert stats["has_output"] is False
    assert stats["validation_image_count"] == 0
    assert stats["validation_tagged_count"] == 0


def test_stats_for_version_counts_validation(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v = versions.create_version(conn, project_id=p["id"], label="v1")
    vdir = versions.version_dir(p["id"], p["slug"], "v1")
    val = vdir / "validation" / "1_data"
    val.mkdir(parents=True)
    (val / "a.png").write_bytes(b"x")
    (val / "b.png").write_bytes(b"x")
    (val / "b.txt").write_text("tag", encoding="utf-8")
    stats = versions.stats_for_version(p, v)
    assert stats["validation_image_count"] == 2
    assert stats["validation_tagged_count"] == 1
    assert stats["train_image_count"] == 0
    assert stats["tagged_image_count"] == 0


def test_create_version_provisions_default_train_folder(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.create_version(conn, project_id=p["id"], label="v1")
    vdir = versions.version_dir(p["id"], p["slug"], "v1")
    assert (vdir / "train" / versions.DEFAULT_TRAIN_FOLDER).is_dir()
