"""Edit projects: pairs API, phase gate, derived config fields."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, secrets, server
from studio.services import version_config
from studio.services.dataset import edit_pairs
from studio.services.projects import jobs as project_jobs, projects, versions


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return TestClient(server.app)


def _png(color=(200, 30, 30), size=(64, 48)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    image = Image.new("RGB", size, color)
    image.paste((0, 0, 0), (10, 10, 30, 30))
    image.save(buf, format="PNG")
    return buf.getvalue()


def _edit_project(client: TestClient) -> tuple[dict, dict, str]:
    p = client.post("/api/projects", json={"title": "Back view", "kind": "edit"}).json()
    v = p["versions"][0]
    return p, v, f"/api/projects/{p['id']}/versions/{v['id']}"


def _upload(client, base, role, *files):
    return client.post(f"{base}/pairs/upload", data={"role": role, "folder": "1_data"},
                       files=[("files", (name, data)) for name, data in files])


def test_kind_is_stored_and_regular_projects_stay_lora(client):
    p, _v, _base = _edit_project(client)
    assert p["kind"] == "edit"
    regular = client.post("/api/projects", json={"title": "Plain"}).json()
    assert regular["kind"] == "lora"
    assert client.post("/api/projects", json={"title": "X", "kind": "bogus"}).status_code == 422
    plain = regular["versions"][0]
    assert client.get(f"/api/projects/{regular['id']}/versions/{plain['id']}/pairs").status_code == 422


def test_pairs_upload_caption_phase_and_delete(client):
    p, v, base = _edit_project(client)
    body = client.post(f"{base}/advance-phase").json()
    assert body["advanced"] is False and "example" in body["reason"]

    _upload(client, base, "target", ("a.png", _png()), ("b.png", _png()), ("a.txt", b"Turn her around."))
    _upload(client, base, "source", ("a.jpg", _png((0, 0, 200))), ("stray.png", _png()))
    listing = client.get(f"{base}/pairs").json()
    assert listing["recipe"] == "text"
    by_stem = {item["stem"]: item for item in listing["items"]}
    assert by_stem["a"]["source"] == "a.jpg" and by_stem["a"]["caption"] == "Turn her around."
    assert by_stem["a"]["missing"] == []
    assert by_stem["b"]["missing"] == ["source", "caption"]
    assert by_stem["stray"]["missing"] == ["target", "caption"]

    body = client.post(f"{base}/advance-phase").json()
    assert body["advanced"] is False and "incomplete" in body["reason"]

    client.post(f"{base}/pairs/delete", json={"items": [{"folder": "1_data", "stem": "b"},
                                                        {"folder": "1_data", "stem": "stray"}]})
    assert client.put(f"{base}/pairs/1_data/a/caption", json={"caption": "Show her from behind."}).status_code == 200
    listing = client.get(f"{base}/pairs").json()
    assert listing["count"] == 1 and listing["complete"] == 1

    body = client.post(f"{base}/advance-phase").json()
    assert body["advanced"] is True and body["new_phase"] == "ready"
    thumb = client.get(f"/api/projects/{p['id']}/versions/{v['id']}/thumb",
                       params={"bucket": "control", "folder": "1_data", "name": "a.jpg"})
    assert thumb.status_code == 200


def test_image_prompt_recipe_needs_second_input_and_sets_config(client):
    p, v, base = _edit_project(client)
    assert client.put(f"{base}/pairs/recipe", json={"recipe": "image"}).status_code == 200
    _upload(client, base, "target", ("a.png", _png()))
    _upload(client, base, "source", ("a.png", _png((0, 0, 200))))
    item = client.get(f"{base}/pairs").json()["items"][0]
    assert item["missing"] == ["source2"]  # text is optional here
    _upload(client, base, "source2", ("a.png", _png((0, 200, 0))))
    assert client.get(f"{base}/pairs").json()["complete"] == 1
    assert client.post(f"{base}/advance-phase").json()["advanced"] is True
    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    assert (vdir / "train" / "1_data" / "a.txt").exists()  # empty caption so the trainer keeps it
    overrides = version_config.project_specific_overrides(p, v)
    assert overrides["edit_control_dir_2"] == str(vdir / "control2")
    # Removing one slot.
    client.post(f"{base}/pairs/delete", json={"slots": [{"role": "source2", "folder": "1_data", "stem": "a"}]})
    assert client.get(f"{base}/pairs").json()["items"][0]["missing"] == ["source2"]


def test_batch_instruction(client):
    _p, _v, base = _edit_project(client)
    _upload(client, base, "target", ("a.png", _png()), ("a.txt", b"1girl, solo"))
    assert client.post(f"{base}/pairs/instruction", json={"instruction": "Color the sketch.",
                                                          "mode": "prepend"}).json()["updated"] == 1
    # Prepending twice doesn't stack the same instruction.
    client.post(f"{base}/pairs/instruction", json={"instruction": "Color the sketch.", "mode": "prepend"})
    item = client.get(f"{base}/pairs").json()["items"][0]
    assert item["caption"] == "Color the sketch.\n1girl, solo"


def test_only_two_recipes(client):
    _p, _v, base = _edit_project(client)
    assert client.put(f"{base}/pairs/recipe", json={"recipe": "sketch"}).status_code == 422
    assert client.put(f"{base}/pairs/recipe", json={"recipe": "image"}).json() == {"recipe": "image"}


def test_config_overrides_turn_edit_mode_on_for_edit_projects(client):
    p, v, _base = _edit_project(client)
    overrides = version_config.project_specific_overrides(p, v)
    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    assert overrides["edit_mode"] is True
    assert overrides["edit_control_dir"] == str(vdir / "control")
    assert overrides["reg_data_dir"] is None
    regular = client.post("/api/projects", json={"title": "Plain"}).json()
    plain = version_config.project_specific_overrides(regular, regular["versions"][0])
    assert plain["edit_mode"] is False and plain["edit_control_dir"] is None


def test_bundle_round_trip_keeps_kind_and_sources(client, tmp_path):
    from studio.services.data_io import train_io

    p, v, base = _edit_project(client)
    _upload(client, base, "target", ("a.png", _png()), ("a.txt", b"Show her back."))
    _upload(client, base, "source", ("a.png", _png((0, 0, 200))))
    client.put(f"{base}/pairs/recipe", json={"recipe": "image"})
    _upload(client, base, "source2", ("a.png", _png((0, 200, 0))))
    dest = tmp_path / "bundle.zip"
    with db.connection_for() as conn:
        out = train_io.export_bundle(conn, v["id"], dest, train_io.BundleOptions())
        assert out["manifest"]["source"]["kind"] == "edit"
        imported = train_io.import_bundle(conn, dest, tmp_path / "presets")
    assert imported["project"]["kind"] == "edit"
    new_v = imported["version"]
    listing = client.get(f"/api/projects/{imported['project']['id']}/versions/{new_v['id']}/pairs").json()
    assert listing["recipe"] == "image"
    assert listing["complete"] == 1 and listing["items"][0]["source2"] == "a.png"


def test_generate_reference_upload_and_family_gate(client, tmp_path, monkeypatch):
    from studio.api.routers import generate as generate_router
    from studio.domain import GenerateConfig

    monkeypatch.setattr(generate_router, "REFERENCE_DIR", tmp_path / "refs")
    res = client.post("/api/generate/reference", files={"image": ("src.png", _png())})
    assert res.status_code == 200
    body = res.json()
    assert body["width"] == 64 and len(body["id"]) == 40
    assert client.get(f"/api/generate/reference/{body['id']}").status_code == 200
    assert client.post("/api/generate/reference", files={"image": ("x.png", b"nope")}).status_code == 400
    GenerateConfig(reference_image="x.png")
    with pytest.raises(ValueError):
        GenerateConfig(model_family="krea2", sampler_name="euler", reference_image="x.png")


def test_finishing_examples_creates_recommended_config(client):
    p, v, base = _edit_project(client)
    _upload(client, base, "target", ("a.png", _png()), ("a.txt", b"Show her back."))
    _upload(client, base, "source", ("a.png", _png((0, 0, 200))))
    assert client.post(f"{base}/advance-phase").json()["advanced"] is True
    cfg = client.get(f"{base}/config").json()["config"]
    expected = {"edit_mode": True, "lora_rank": 32, "learning_rate": 2e-5, "optimizer_type": "adamw",
                "epochs": 12, "grad_accum": 4, "resolution": [1024], "lr_scheduler": "cosine_with_warmup",
                "lr_scheduler_warmup_steps": 100, "shuffle_caption": False, "flip_augment": False,
                "sample_prompt": "", "sample_prompts": []}
    assert {k: cfg[k] for k in expected} == expected
    assert cfg["eval_validation_enabled"] is False

    # "Back to recommended" rewrites a changed config.
    client.put(f"{base}/config", json={**cfg, "lora_rank": 8, "epochs": 3})
    reset = client.post(f"{base}/pairs/recommended-config").json()["config"]
    assert reset["lora_rank"] == 32 and reset["epochs"] == 12 and reset["edit_mode"] is True
