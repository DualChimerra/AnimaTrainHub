"""Graph (training results board): storage + HTTP surface."""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio import server
from studio.infrastructure import paths as _paths
from studio.services.graph import store


@pytest.fixture(autouse=True)
def _tmp_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(_paths, "STUDIO_DATA", tmp_path)


@pytest.fixture()
def client():
    return TestClient(server.app)


def _png(w=12, h=8, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 10, 10)).save(buf, format=fmt)
    return buf.getvalue()


def test_new_board_gets_default_params(client):
    r = client.post("/api/graph/boards", json={"name": "Style A"})
    assert r.status_code == 200
    params = r.json()["board"]["params"]
    ids = [p["id"] for p in params]
    assert ids[0] == "model" and "lr" in ids and "snr_mean" in ids
    snr = next(p for p in params if p["id"] == "snr_mean")
    assert snr["condition"] == {"param": "sampling", "options": ["style"]}


def test_empty_board_and_copy(client):
    a = client.post("/api/graph/boards", json={"name": "A", "empty": True}).json()
    assert a["board"]["params"] == []
    base = client.post("/api/graph/boards", json={"name": "B"}).json()
    c = client.post("/api/graph/boards", json={"name": "C", "copy_params_from": base["board"]["id"]}).json()
    assert c["board"]["params"] == base["board"]["params"]
    assert client.post("/api/graph/boards", json={"name": "  "}).status_code == 422


def test_run_lifecycle_and_unset_values(client):
    bid = client.post("/api/graph/boards", json={"name": "B"}).json()["board"]["id"]
    r = client.post(f"/api/graph/boards/{bid}/runs",
                    json={"name": "r1", "values": {"lr": 0.0001, "dora": "no", "x": None}, "rating": 7})
    run = r.json()
    assert run["values"] == {"lr": 0.0001, "dora": "no"}  # None = unset, never stored
    assert run["rating"] == 3 and run["status"] == "planned"
    run = client.patch(f"/api/graph/runs/{run['id']}", json={"status": "done", "favorite": True}).json()
    assert run["status"] == "done" and run["favorite"] is True
    assert client.patch(f"/api/graph/runs/{run['id']}", json={"status": "weird"}).status_code == 422


def test_images_upload_copy_and_duplicate_skips_images(client):
    bid = client.post("/api/graph/boards", json={"name": "B"}).json()["board"]["id"]
    rid = client.post(f"/api/graph/boards/{bid}/runs", json={"name": "r"}).json()["id"]
    files = [("files", ("a.png", _png(), "image/png")), ("files", ("b.webp", _png(fmt="WEBP"), "image/webp"))]
    r = client.post(f"/api/graph/runs/{rid}/images", files=files, data={"meta": json.dumps({"step": 400})})
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert [i["step"] for i in items] == [400, 400]
    assert items[0]["width"] == 12 and items[0]["height"] == 8
    f = client.get(f"/api/graph/images/{items[0]['id']}/file")
    assert f.status_code == 200 and f.content[:4] == b"\x89PNG"

    bad = client.post(f"/api/graph/runs/{rid}/images", files=[("files", ("x.txt", b"hello", "text/plain"))])
    assert bad.status_code == 422

    dup = client.post(f"/api/graph/runs/{rid}/duplicate").json()
    assert dup["images"] == [] and dup["id"] != rid

    img = client.patch(f"/api/graph/images/{items[0]['id']}", json={"seed": "42", "rating": 2, "prompt": "p"}).json()
    assert img["seed"] == 42 and img["rating"] == 2
    assert client.delete(f"/api/graph/images/{items[0]['id']}").status_code == 200
    assert len(client.get(f"/api/graph/boards/{bid}").json()["runs"][0]["images"]) == 1


def test_remap_treats_scientific_and_decimal_as_same(client):
    bid = client.post("/api/graph/boards", json={"name": "B"}).json()["board"]["id"]
    client.post(f"/api/graph/boards/{bid}/runs", json={"values": {"lr": 1e-4}})
    client.post(f"/api/graph/boards/{bid}/runs", json={"values": {"lr": 2e-5}})
    r = client.post(f"/api/graph/boards/{bid}/remap",
                    json={"param_id": "lr", "mapping": [{"from": 0.0001, "to": None}]})
    assert r.json()["changed"] == 1
    values = [run["values"] for run in client.get(f"/api/graph/boards/{bid}").json()["runs"]]
    assert values == [{}, {"lr": 2e-5}]


def test_delete_board_removes_files(client, tmp_path):
    bid = client.post("/api/graph/boards", json={"name": "B"}).json()["board"]["id"]
    rid = client.post(f"/api/graph/boards/{bid}/runs", json={}).json()["id"]
    client.post(f"/api/graph/runs/{rid}/images", files=[("files", ("a.png", _png(), "image/png"))])
    assert (tmp_path / "graph" / "images" / str(bid)).exists()
    assert client.delete(f"/api/graph/boards/{bid}").status_code == 200
    assert not (tmp_path / "graph" / "images" / str(bid)).exists()
    assert client.get(f"/api/graph/boards/{bid}").status_code == 404


def test_describe_samples_rotates_prompts():
    cfg = {"sample_prompts": ["a", "b"], "sample_seed": 10}
    info = store.describe_samples(
        ["step_400.png", "step_200.png", "step_0_baseline_1.png", "step_600.png", "epoch_2.png"], cfg)
    assert info["step_200.png"] == {"step": 200, "prompt": "a", "seed": 10, "epoch": None}
    assert info["step_400.png"]["prompt"] == "b"
    assert info["step_600.png"]["prompt"] == "a"
    assert info["step_0_baseline_1.png"] == {"step": 0, "prompt": "b", "seed": 11, "epoch": None}
    assert info["epoch_2.png"]["epoch"] == 2 and info["epoch_2.png"]["step"] is None


def test_config_download_404_without_link(client):
    bid = client.post("/api/graph/boards", json={"name": "B"}).json()["board"]["id"]
    rid = client.post(f"/api/graph/boards/{bid}/runs", json={}).json()["id"]
    assert client.get(f"/api/graph/runs/{rid}/config").status_code == 404


def test_epoch_samples_get_steps_from_monitor():
    mon = {"total_steps": 3000, "total_epochs": 30,
           "samples": [{"path": "C:/x/samples/epoch_9.png", "step": 905}]}
    info = store.describe_samples(["epoch_9.png", "epoch_10.png", "vae_roundtrip.png"], {}, mon)
    assert info["epoch_9.png"]["step"] == 905          # recorded step wins
    assert info["epoch_10.png"]["step"] == 1000        # total_steps / total_epochs
    assert info["epoch_10.png"]["epoch"] == 10
    assert not store.is_sample_name("vae_roundtrip.png")
