"""Shared dataset library: datasets as standalone pools, curation from several
sources, validation origins, cache routes and the cache folder listing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio import db, server
from studio.services.dataset import library
from studio.services.preprocess import manifest as preprocess_manifest
from studio.services.projects import projects, versions


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    return tmp_path


@pytest.fixture
def client(env) -> TestClient:
    server.app.state.supervisor = None
    return TestClient(server.app)


def _png(path: Path, color: str = "#88aacc") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path)
    return path


def _dataset(client: TestClient, title: str, files: dict[str, str]) -> dict:
    d = client.post("/api/library/datasets", json={"title": title}).json()
    ddir = Path(d["path"])
    for name, color in files.items():
        _png(ddir / "download" / name, color)
        (ddir / "download" / f"{Path(name).stem}.txt").write_text(f"tags of {name}", encoding="utf-8")
    return d


def _project(client: TestClient) -> tuple[int, int]:
    p = client.post("/api/projects", json={"title": "Proj"}).json()
    return p["id"], p["versions"][0]["id"]


def _vdir(pid: int, vid: int) -> Path:
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    return versions.version_dir(p["id"], p["slug"], v["label"])


# ---------------------------------------------------------------------------
# library basics
# ---------------------------------------------------------------------------


def test_dataset_lives_in_datasets_folder_and_lists(client: TestClient, env: Path) -> None:
    d = _dataset(client, "Foxes", {"a.png": "#ff0000", "b.png": "#00ff00"})
    assert Path(d["path"]).parent == env / "datasets"
    assert d["kind"] == "dataset"

    items = client.get("/api/library/datasets").json()["items"]
    assert [i["title"] for i in items] == ["Foxes"]
    row = items[0]
    assert row["image_count"] == 2
    assert row["image_bytes"] > 0
    assert set(row["covers"]) == {"a.png", "b.png"}
    assert row["used_by"] == []
    # The project routes take the dataset id: file listing + thumbnails work as-is.
    files = client.get(f"/api/projects/{d['id']}/files").json()
    assert files["count"] == 2
    assert client.get(f"/api/projects/{d['id']}/thumb", params={"name": "a.png"}).status_code == 200


def test_dataset_routes_reject_plain_projects(client: TestClient) -> None:
    pid, _ = _project(client)
    assert client.get(f"/api/library/datasets/{pid}").status_code == 404


# ---------------------------------------------------------------------------
# curation from several sources
# ---------------------------------------------------------------------------


def test_curation_pool_merges_linked_datasets(client: TestClient) -> None:
    d1 = _dataset(client, "One", {"001.png": "#111111", "002.png": "#222222"})
    d2 = _dataset(client, "Two", {"001.png": "#333333"})
    pid, vid = _project(client)

    r = client.put(f"/api/projects/{pid}/versions/{vid}/datasets", json={"dataset_ids": [d1["id"], d2["id"]]})
    assert [i["id"] for i in r.json()["items"]] == [d1["id"], d2["id"]]

    view = client.get(f"/api/projects/{pid}/versions/{vid}/curation").json()
    names = sorted(e["name"] for e in view["left"])
    assert names == sorted([f"@{d1['id']}/001.png", f"@{d1['id']}/002.png", f"@{d2['id']}/001.png"])
    # New projects have no own download/ images -> only the datasets are sources.
    assert [s["id"] for s in view["sources"]] == [d1["id"], d2["id"]]
    assert view["download_total"] == 3

    # Pool keys work as thumbnail names on the project route.
    thumb = client.get(f"/api/projects/{pid}/thumb", params={"name": f"@{d2['id']}/001.png", "raw": 1})
    assert thumb.status_code == 200


def test_copy_same_file_name_from_two_datasets(client: TestClient) -> None:
    d1 = _dataset(client, "One", {"001.png": "#111111"})
    d2 = _dataset(client, "Two", {"001.png": "#333333"})
    pid, vid = _project(client)
    client.put(f"/api/projects/{pid}/versions/{vid}/datasets", json={"dataset_ids": [d1["id"], d2["id"]]})
    k1, k2 = f"@{d1['id']}/001.png", f"@{d2['id']}/001.png"

    r = client.post(f"/api/projects/{pid}/versions/{vid}/curation/copy",
                    json={"files": [k1, k2], "dest_folder": "1_data"}).json()
    assert r["copied"] == [k1, k2]

    train = _vdir(pid, vid) / "train" / "1_data"
    assert (train / "001.png").is_file()
    assert (train / f"001-d{d2['id']}.png").is_file()
    # caption follows, named after the copy
    assert (train / f"001-d{d2['id']}.txt").read_text(encoding="utf-8") == "tags of 001.png"

    view = client.get(f"/api/projects/{pid}/versions/{vid}/curation").json()
    assert view["left"] == []
    right = {e["name"]: e for e in view["right"]["1_data"]}
    assert set(right) == {k1, k2}
    assert right[k2]["source"] == d2["id"]
    assert right[k2]["file"] == f"001-d{d2['id']}.png"

    # copying again is a no-op
    again = client.post(f"/api/projects/{pid}/versions/{vid}/curation/copy",
                        json={"files": [k1], "dest_folder": "1_data"}).json()
    assert again["skipped"] == [k1]

    # remove by pool key drops exactly that copy
    rm = client.post(f"/api/projects/{pid}/versions/{vid}/curation/remove",
                     json={"folder": "1_data", "files": [k2]}).json()
    assert rm["removed"] == [k2]
    assert not (train / f"001-d{d2['id']}.png").exists()
    assert (train / "001.png").is_file()


def test_validation_from_dataset_stays_out_of_pool(client: TestClient) -> None:
    d = _dataset(client, "One", {"a.png": "#111111", "b.png": "#222222"})
    pid, vid = _project(client)
    client.put(f"/api/projects/{pid}/versions/{vid}/datasets", json={"dataset_ids": [d["id"]]})
    key = f"@{d['id']}/a.png"

    r = client.post(f"/api/projects/{pid}/versions/{vid}/curation/validation/copy", json={"files": [key]}).json()
    assert r["copied"] == [key]
    val = client.get(f"/api/projects/{pid}/versions/{vid}/curation/validation").json()
    assert [e["name"] for e in val["right"]] == ["a.png"]
    assert val["right"][0]["origin"] == key
    assert [e["name"] for e in val["left"]] == [f"@{d['id']}/b.png"]

    client.post(f"/api/projects/{pid}/versions/{vid}/curation/validation/remove",
                json={"items": [{"folder": "1_data", "name": "a.png"}]})
    assert not (_vdir(pid, vid) / "validation" / library.VALIDATION_ORIGINS).exists()


def test_own_download_still_works_next_to_datasets(client: TestClient) -> None:
    """Old projects: own download/ images keep bare names and stay a source."""
    d = _dataset(client, "One", {"x.png": "#111111"})
    pid, vid = _project(client)
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    _png(projects.project_dir(p["id"], p["slug"]) / "download" / "own.png")
    client.put(f"/api/projects/{pid}/versions/{vid}/datasets", json={"dataset_ids": [d["id"]]})

    view = client.get(f"/api/projects/{pid}/versions/{vid}/curation").json()
    assert sorted(e["name"] for e in view["left"]) == sorted(["own.png", f"@{d['id']}/x.png"])
    assert view["sources"][0] == {"id": None, "title": None, "total": 1}

    client.post(f"/api/projects/{pid}/versions/{vid}/curation/copy",
                json={"files": ["own.png"], "dest_folder": "1_data"})
    entry = preprocess_manifest.train_load(projects.project_dir(p["id"], p["slug"]), "v1")["images"]["1_data/own.png"]
    assert entry["origin"] == "own.png"


# ---------------------------------------------------------------------------
# versions, deletion, import
# ---------------------------------------------------------------------------


def test_new_version_inherits_datasets(client: TestClient) -> None:
    d = _dataset(client, "One", {"a.png": "#111111"})
    pid, vid = _project(client)
    client.put(f"/api/projects/{pid}/versions/{vid}/datasets", json={"dataset_ids": [d["id"]]})
    v2 = client.post(f"/api/projects/{pid}/versions", json={"label": "v2"}).json()
    linked = client.get(f"/api/projects/{pid}/versions/{v2['id']}/datasets").json()["items"]
    assert [i["id"] for i in linked] == [d["id"]]


def test_delete_dataset_keeps_train_copies(client: TestClient) -> None:
    d = _dataset(client, "One", {"a.png": "#111111"})
    pid, vid = _project(client)
    client.put(f"/api/projects/{pid}/versions/{vid}/datasets", json={"dataset_ids": [d["id"]]})
    client.post(f"/api/projects/{pid}/versions/{vid}/curation/copy",
                json={"files": [f"@{d['id']}/a.png"], "dest_folder": "1_data"})
    usage = client.get(f"/api/library/datasets/{d['id']}").json()["used_by"]
    assert [u["version_id"] for u in usage] == [vid]

    assert client.delete(f"/api/library/datasets/{d['id']}").status_code == 200
    assert not Path(d["path"]).exists()
    assert (_vdir(pid, vid) / "train" / "1_data" / "a.png").is_file()
    assert client.get(f"/api/projects/{pid}/versions/{vid}/datasets").json()["items"] == []
    view = client.get(f"/api/projects/{pid}/versions/{vid}/curation").json()
    assert [e["name"] for e in view["right"]["1_data"]] == [f"@{d['id']}/a.png"]


def test_create_dataset_from_project_images(client: TestClient) -> None:
    pid, _ = _project(client)
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    pdl = projects.project_dir(p["id"], p["slug"]) / "download"
    _png(pdl / "one.png")
    (pdl / "one.booru.txt").write_text("booru", encoding="utf-8")

    d = client.post("/api/library/datasets", json={"title": "From project", "from_project_id": pid}).json()
    assert d["imported"]["added"] == ["one.png"]
    assert (Path(d["path"]) / "download" / "one.booru.txt").is_file()
    assert (pdl / "one.png").is_file()  # the project keeps its files


# ---------------------------------------------------------------------------
# training cache
# ---------------------------------------------------------------------------


def test_cache_routes_cover_dataset_images_only(client: TestClient) -> None:
    d = _dataset(client, "One", {"a.png": "#111111"})
    pid, vid = _project(client)
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    _png(projects.project_dir(p["id"], p["slug"]) / "download" / "own.png")
    client.put(f"/api/projects/{pid}/versions/{vid}/datasets", json={"dataset_ids": [d["id"]]})
    client.post(f"/api/projects/{pid}/versions/{vid}/curation/copy",
                json={"files": [f"@{d['id']}/a.png", "own.png"], "dest_folder": "1_data"})

    path = library.write_cache_routes(p, v)
    assert path is not None
    routes = json.loads(path.read_text(encoding="utf-8"))["routes"]
    assert routes == {"1_data/a.png": str(Path(d["path"]) / "cache")}

    # Nothing routed -> stale file removed
    client.post(f"/api/projects/{pid}/versions/{vid}/curation/remove",
                json={"folder": "1_data", "files": [f"@{d['id']}/a.png"]})
    assert library.write_cache_routes(p, v) is None
    assert not path.exists()


def test_cache_groups_listing_and_clear(client: TestClient) -> None:
    d = _dataset(client, "One", {"a.png": "#111111"})
    group = Path(d["path"]) / "cache" / "latents" / "wan21-f8c16-v1"
    group.mkdir(parents=True)
    (group / "abc_64x48.npz").write_bytes(b"x" * 100)
    (group / "group.json").write_text(json.dumps({"kind": "latents", "family": "anima", "last_used_at": 5}), encoding="utf-8")
    text = Path(d["path"]) / "cache" / "text" / "qwen-v1"
    text.mkdir(parents=True)
    (text / "k.safetensors").write_bytes(b"y" * 10)

    cache = client.get(f"/api/library/datasets/{d['id']}/cache").json()
    assert cache["total_files"] == 2
    assert cache["total_bytes"] == 110
    lat = next(g for g in cache["groups"] if g["kind"] == "latents")
    assert lat["family"] == "anima" and lat["files"] == 1

    r = client.delete(f"/api/library/datasets/{d['id']}/cache", params={"kind": "latents", "group": "wan21-f8c16-v1"})
    assert r.json()["freed_bytes"] == 100
    assert not group.exists() and text.exists()
    assert client.delete(f"/api/library/datasets/{d['id']}/cache", params={"kind": "../x"}).status_code == 400
    client.delete(f"/api/library/datasets/{d['id']}/cache")
    assert client.get(f"/api/library/datasets/{d['id']}/cache").json()["groups"] == []
