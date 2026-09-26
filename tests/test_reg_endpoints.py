from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio import db, secrets, server
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.services.reg import builder as reg_builder


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return {"db": dbfile}


@pytest.fixture
def client(env) -> TestClient:
    server.app.state.supervisor = None
    return TestClient(server.app)


def _make(client: TestClient) -> tuple[int, int]:
    p = client.post("/api/projects", json={"title": "P"}).json()
    return p["id"], p["versions"][0]["id"]


def _seed_train(
    client: TestClient, pid: int, vid: int, folder: str, files: dict[str, list[str]]
) -> Path:
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    train = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "train"
    d = train / folder
    d.mkdir(parents=True, exist_ok=True)
    for name, tags in files.items():
        img = Image.new("RGB", (32, 32), (255, 0, 0))
        img.save(d / name, "PNG")
        (d / name).with_suffix(".txt").write_text(", ".join(tags), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# preview-tags
# ---------------------------------------------------------------------------


def test_preview_tags_returns_top_freq(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {
        "a.png": ["1girl", "solo", "blue_hair"],
        "b.png": ["1girl", "long_hair"],
        "c.png": ["1girl", "outdoor"],
    })
    r = client.get(f"/api/projects/{pid}/versions/{vid}/reg/preview-tags?top=3")
    assert r.status_code == 200
    items = r.json()["items"]
    assert items[0]["tag"] == "1girl"
    assert items[0]["count"] == 3
    assert len(items) == 3


def test_preview_tags_empty_train(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/{vid}/reg/preview-tags")
    assert r.status_code == 200
    assert r.json()["items"] == []


# ---------------------------------------------------------------------------
# GET /reg
# ---------------------------------------------------------------------------


def test_get_reg_when_not_exists(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/{vid}/reg")
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is False
    assert body["meta"] is None
    assert body["image_count"] == 0


def test_get_reg_when_exists_returns_meta_and_files(
    client: TestClient, tmp_path: Path
) -> None:
    pid, vid = _make(client)
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    rdir = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg"
    rdir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (16, 16), (0, 0, 255))
    img.save(rdir / "100.png", "PNG")
    img.save(rdir / "200.png", "PNG")
    meta = reg_builder.RegMeta(
        generated_at=1.0, based_on_version="baseline", api_source="gelbooru",
        target_count=5, actual_count=2, source_tags=["1girl"],
        excluded_tags=[], blacklist_tags=[], failed_tags=[],
        train_tag_distribution={"1girl": 5}, auto_tagged=True,
    )
    reg_builder.write_meta(rdir, meta)

    r = client.get(f"/api/projects/{pid}/versions/{vid}/reg")
    body = r.json()
    assert body["exists"] is True
    assert body["image_count"] == 2
    assert sorted(body["files"]) == ["100.png", "200.png"]
    assert body["meta"]["actual_count"] == 2
    assert body["meta"]["auto_tagged"] is True


# ---------------------------------------------------------------------------
# POST /reg/build
# ---------------------------------------------------------------------------


def test_start_reg_build_requires_train_data(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build", json={}
    )
    assert r.status_code == 400


def test_start_reg_build_creates_job(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {
        "1.png": ["1girl"],
    })
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"excluded_tags": ["character_x"], "auto_tag": False},
    )
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["kind"] == "reg_build"
    assert job["status"] == "pending"
    p_dict = json.loads(job["params"])
    assert p_dict["target_count"] is None
    assert p_dict["build_mode"] == "flat"
    assert p_dict["excluded_tags"] == ["character_x"]
    assert p_dict["auto_tag"] is False
    assert p_dict["skip_similar"] is True
    assert p_dict["postprocess_method"] == "smart"


def test_start_reg_build_passes_through_advanced_params(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"1.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={
            "auto_tag": False,
            "skip_similar": False,
            "aspect_ratio_filter_enabled": True,
            "min_aspect_ratio": 0.6,
            "max_aspect_ratio": 1.8,
            "postprocess_method": "stretch",
            "postprocess_max_crop_ratio": 0.2,
        },
    )
    assert r.status_code == 200
    p_dict = json.loads(r.json()["params"])
    assert "batch_size" not in p_dict
    assert p_dict["skip_similar"] is False
    assert p_dict["aspect_ratio_filter_enabled"] is True
    assert p_dict["min_aspect_ratio"] == 0.6
    assert p_dict["postprocess_method"] == "stretch"
    assert p_dict["postprocess_max_crop_ratio"] == 0.2


def test_start_reg_build_rejects_bad_postprocess_method(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"1.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"postprocess_method": "weird", "auto_tag": False},
    )
    assert r.status_code == 400


def test_start_reg_build_rejects_bad_max_crop(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"1.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"postprocess_max_crop_ratio": 0.9, "auto_tag": False},
    )
    assert r.status_code == 400


def test_start_reg_build_rejects_bad_api_source(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"1.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"api_source": "pixiv"},
    )
    assert r.status_code == 400


def test_start_reg_build_passes_through_incremental(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"1.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"incremental": True, "auto_tag": False},
    )
    assert r.status_code == 200
    p_dict = json.loads(r.json()["params"])
    assert p_dict["incremental"] is True


# ---------------------------------------------------------------------------
# DELETE /reg
# ---------------------------------------------------------------------------


def test_delete_reg_removes_content(client: TestClient) -> None:
    pid, vid = _make(client)
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    rdir = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg"
    rdir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(rdir / "1.png", "PNG")

    r = client.delete(f"/api/projects/{pid}/versions/{vid}/reg")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    assert rdir.exists()
    assert not (rdir / "1.png").exists()


def test_delete_reg_when_empty(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.delete(f"/api/projects/{pid}/versions/{vid}/reg")
    assert r.status_code == 200
    assert r.json()["deleted"] is False


# ---------------------------------------------------------------------------
# GET /reg/caption
# ---------------------------------------------------------------------------


def test_get_reg_caption_returns_tags(client: TestClient) -> None:
    pid, vid = _make(client)
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    rdir = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg" / "5_concept"
    rdir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(rdir / "100.png", "PNG")
    (rdir / "100.txt").write_text("a, b, c", encoding="utf-8")
    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/reg/caption?path=5_concept/100.png"
    )
    assert r.status_code == 200
    assert r.json()["tags"] == ["a", "b", "c"]


def test_get_reg_caption_rejects_path_traversal(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/reg/caption?path=../train/x.png"
    )
    assert r.status_code == 400


def test_get_reg_caption_404_for_missing(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/reg/caption?path=ghost.png"
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# A1 — POST /reg/delete-files
# ---------------------------------------------------------------------------


def _seed_reg(
    client: TestClient,
    pid: int,
    vid: int,
    folder: str,
    files: dict[str, list[str]],
) -> Path:
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    base = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg"
    d = base / folder if folder else base
    d.mkdir(parents=True, exist_ok=True)
    for name, tags in files.items():
        Image.new("RGB", (8, 8)).save(d / name, "PNG")
        if tags:
            (d / name).with_suffix(".txt").write_text(
                ", ".join(tags), encoding="utf-8"
            )
    return base


def _write_meta(rdir: Path, actual: int) -> None:
    meta = reg_builder.RegMeta(
        generated_at=0.0,
        based_on_version="v1",
        api_source="gelbooru",
        target_count=actual,
        actual_count=actual,
        source_tags=[],
        excluded_tags=[],
        blacklist_tags=[],
        failed_tags=[],
        train_tag_distribution={},
        auto_tagged=False,
    )
    reg_builder.write_meta(rdir, meta)


def test_delete_reg_files_removes_image_and_caption(client: TestClient) -> None:
    pid, vid = _make(client)
    rdir = _seed_reg(
        client, pid, vid, "5_concept",
        {"100.png": ["a", "b"], "101.png": ["c"], "102.png": []},
    )
    _write_meta(rdir, actual=3)

    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/delete-files",
        json={"relative_paths": ["5_concept/100.png", "5_concept/101.png"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    assert set(body["deleted"]) == {"5_concept/100.png", "5_concept/101.png"}
    assert not (rdir / "5_concept" / "100.png").exists()
    assert not (rdir / "5_concept" / "100.txt").exists()
    assert not (rdir / "5_concept" / "101.png").exists()
    assert (rdir / "5_concept" / "102.png").exists()


def test_delete_reg_files_updates_meta_actual_count(client: TestClient) -> None:
    pid, vid = _make(client)
    rdir = _seed_reg(
        client, pid, vid, "5_concept",
        {"100.png": ["a"], "101.png": ["b"], "102.png": ["c"]},
    )
    _write_meta(rdir, actual=3)

    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/delete-files",
        json={"relative_paths": ["5_concept/100.png"]},
    )
    assert r.status_code == 200
    meta = reg_builder.read_meta(rdir)
    assert meta is not None
    assert meta.actual_count == 2


def test_delete_reg_files_writes_deleted_ids_json(client: TestClient) -> None:
    pid, vid = _make(client)
    rdir = _seed_reg(
        client, pid, vid, "5_concept",
        {"42.png": ["a"], "99.png": ["b"]},
    )

    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/delete-files",
        json={"relative_paths": ["5_concept/42.png", "5_concept/99.png"]},
    )
    assert r.status_code == 200
    p = rdir / ".deleted_ids.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert set(data) == {"42", "99"}


def test_delete_reg_files_rejects_path_traversal(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_reg(client, pid, vid, "5_concept", {"100.png": []})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/delete-files",
        json={"relative_paths": ["../train/secret.png"]},
    )
    assert r.status_code == 400


def test_delete_reg_files_empty_list_400(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/delete-files",
        json={"relative_paths": []},
    )
    assert r.status_code == 400


def test_delete_reg_files_silently_skips_missing(client: TestClient) -> None:
    pid, vid = _make(client)
    rdir = _seed_reg(client, pid, vid, "5_concept", {"100.png": []})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/delete-files",
        json={"relative_paths": ["5_concept/100.png", "5_concept/ghost.png"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["deleted"] == ["5_concept/100.png"]
    data = json.loads((rdir / ".deleted_ids.json").read_text(encoding="utf-8"))
    assert data == ["100"]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_start_reg_build_default_auto_tag_kind_is_wd14(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"a.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    assert (job.get("params_decoded") or {}).get("auto_tag_kind") == "wd14"


def test_start_reg_build_accepts_cltagger(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"a.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"auto_tag_kind": "cltagger"},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    with db.connection_for() as conn:
        job = project_jobs.get_job(conn, job_id)
    assert (job.get("params_decoded") or {}).get("auto_tag_kind") == "cltagger"


def test_start_reg_build_rejects_invalid_build_mode(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"a.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"build_mode": "weird"},
    )
    assert r.status_code == 422


def test_start_reg_build_rejects_zero_target_count(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"a.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"target_count": 0},
    )
    assert r.status_code == 422


def test_start_reg_build_accepts_mirror_and_target_count(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"a.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"build_mode": "mirror", "target_count": 50},
    )
    assert r.status_code == 200
    p_dict = json.loads(r.json()["params"])
    assert p_dict["build_mode"] == "mirror"
    assert p_dict["target_count"] == 50


def test_start_reg_build_rejects_llm_auto_tag_kind(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_concept", {"a.png": ["x"]})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/reg/build",
        json={"auto_tag_kind": "llm"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# A4 — POST /reg/dedup-purge
# ---------------------------------------------------------------------------


def test_dedup_purge_empty_reg_returns_zero(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(f"/api/projects/{pid}/versions/{vid}/reg/dedup-purge")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scanned"] == 0
    assert body["groups"] == 0
    assert body["count"] == 0


def test_dedup_purge_no_duplicates_keeps_all(client: TestClient) -> None:
    pid, vid = _make(client)
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    rdir = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg" / "5_concept"
    rdir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (255, 0, 0)).save(rdir / "1.png", "PNG")
    Image.new("RGB", (96, 128), (0, 255, 0)).save(rdir / "2.png", "PNG")

    r = client.post(f"/api/projects/{pid}/versions/{vid}/reg/dedup-purge")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scanned"] == 2
    assert body["count"] == 0
    assert (rdir / "1.png").exists()
    assert (rdir / "2.png").exists()


def test_dedup_purge_deletes_identical_copies(client: TestClient) -> None:
    pid, vid = _make(client)
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    base = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "reg"
    rdir = base / "5_concept"
    rdir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (128, 128), (255, 128, 64))
    img.save(rdir / "100.png", "PNG")
    img.save(rdir / "200.png", "PNG")
    meta = reg_builder.RegMeta(
        generated_at=0.0, based_on_version="v1", api_source="gelbooru",
        target_count=2, actual_count=2, source_tags=[], excluded_tags=[],
        blacklist_tags=[], failed_tags=[], train_tag_distribution={},
        auto_tagged=False,
    )
    reg_builder.write_meta(base, meta)

    r = client.post(f"/api/projects/{pid}/versions/{vid}/reg/dedup-purge")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scanned"] == 2
    assert body["groups"] >= 1
    assert body["count"] == 1
    remaining = sorted(p.name for p in rdir.glob("*.png"))
    assert len(remaining) == 1
    p = base / ".deleted_ids.json"
    assert p.exists()
    deleted_stems = set(json.loads(p.read_text(encoding="utf-8")))
    assert deleted_stems <= {"100", "200"}
    assert len(deleted_stems) == 1
    m2 = reg_builder.read_meta(base)
    assert m2.actual_count == 1
