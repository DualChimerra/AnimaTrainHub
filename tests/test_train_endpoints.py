from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.services.projects import projects, versions
from studio.schema import TrainingConfig


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from studio.infrastructure import paths as _paths
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    from studio.services.presets import io as presets_io
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", presets_dir)
    return {"db": dbfile, "presets": presets_dir}


@pytest.fixture
def client(env) -> TestClient:
    server.app.state.supervisor = None
    return TestClient(server.app)


def _make(client: TestClient) -> tuple[int, int]:
    p = client.post("/api/projects", json={"title": "P"}).json()
    return p["id"], p["versions"][0]["id"]


def _seed_preset(env, name: str, **overrides) -> None:
    from studio.services.presets import io as presets_io
    base = TrainingConfig().model_dump()
    base.update(overrides)
    presets_io.write_preset(name, base)


# ---------------------------------------------------------------------------
# GET /config
# ---------------------------------------------------------------------------


def test_get_config_returns_no_config_initially(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/{vid}/config")
    assert r.status_code == 200
    body = r.json()
    assert body["has_config"] is False
    assert body["config"] is None
    assert "data_dir" in body["project_specific_fields"]
    assert "output_name" in body["project_specific_fields"]


def test_get_config_no_config_returns_project_specific_defaults(
    client: TestClient,
) -> None:
    pid, vid = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/{vid}/config")
    body = r.json()
    assert body["has_config"] is False
    defaults = body.get("project_specific_defaults")
    assert defaults is not None, "缺 project_specific_defaults"
    assert defaults["data_dir"].endswith("train")
    assert defaults["output_dir"].endswith("output")
    assert defaults["output_name"], "output_name 不能为空"
    assert defaults["reg_data_dir"] is None
    assert defaults["resume_lora"] is None
    assert defaults["transformer_path"], "transformer_path 应填默认模型路径"
    assert defaults["vae_path"], "vae_path 应填"
    assert defaults["text_encoder_path"]
    assert defaults["t5_tokenizer_path"]


def test_get_config_defaults_picks_up_reg_when_meta_exists(
    client: TestClient, env
) -> None:
    pid, vid = _make(client)
    with db.connection_for(env["db"]) as conn:
        p = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    assert p is not None and v is not None
    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    (vdir / "reg").mkdir(parents=True, exist_ok=True)
    (vdir / "reg" / "meta.json").write_text('{"target_count": 50}', encoding="utf-8")

    r = client.get(f"/api/projects/{pid}/versions/{vid}/config")
    defaults = r.json()["project_specific_defaults"]
    assert defaults["reg_data_dir"] == str(vdir / "reg")


def test_get_config_returns_defaults_when_has_config(
    client: TestClient, env
) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl", lora_rank=64)
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    with db.connection_for(env["db"]) as conn:
        p = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    assert p is not None and v is not None
    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    (vdir / "reg" / "meta.json").write_text('{"target_count": 50}', encoding="utf-8")

    r = client.get(f"/api/projects/{pid}/versions/{vid}/config")
    body = r.json()
    assert body["has_config"] is True
    defaults = body.get("project_specific_defaults")
    assert defaults is not None, "has_config=True 时也应返回 project_specific_defaults"
    assert defaults["reg_data_dir"] == str(vdir / "reg")
    assert defaults["transformer_path"]


def test_get_config_for_unknown_version_404(client: TestClient) -> None:
    pid, _ = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/9999/config")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /config/from_preset
# ---------------------------------------------------------------------------


def test_fork_preset_writes_version_config(client: TestClient, env) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl", lora_rank=64)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["from_preset"] == "tpl"
    cfg = body["config"]
    assert cfg["lora_rank"] == 64
    assert cfg["data_dir"].endswith("train")
    v = client.get(f"/api/projects/{pid}/versions/{vid}").json()
    assert v["config_name"] == "tpl"


def test_fork_preset_unknown_404(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "nope"},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# PUT /config
# ---------------------------------------------------------------------------


def test_put_config_keeps_user_values(client: TestClient, env) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    cfg = client.get(f"/api/projects/{pid}/versions/{vid}/config").json()["config"]
    cfg["output_name"] = "custom_lora"
    cfg["resume_lora"] = "/tmp/some/lora.safetensors"
    cfg["lora_rank"] = 96
    r = client.put(f"/api/projects/{pid}/versions/{vid}/config", json=cfg)
    assert r.status_code == 200
    body = r.json()
    assert body["config"]["output_name"] == "custom_lora"
    assert body["config"]["resume_lora"] == "/tmp/some/lora.safetensors"
    assert body["config"]["lora_rank"] == 96


def test_fork_preset_still_forces_project_overrides(
    client: TestClient, env
) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl", data_dir="/wrong/path", output_name="wrong_name")
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    assert r.status_code == 200
    cfg = client.get(f"/api/projects/{pid}/versions/{vid}/config").json()["config"]
    assert cfg["data_dir"].endswith("train")
    assert cfg["output_name"] != "wrong_name"


def test_put_config_tolerates_invalid_values(client: TestClient, env) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    cfg = client.get(f"/api/projects/{pid}/versions/{vid}/config").json()["config"]
    cfg["lora_rank"] = 0
    r = client.put(f"/api/projects/{pid}/versions/{vid}/config", json=cfg)
    assert r.status_code == 200
    assert r.json()["config"]["lora_rank"] == 32


# ---------------------------------------------------------------------------
# POST /config/save_as_preset
# ---------------------------------------------------------------------------


def test_save_as_preset_clears_project_fields(client: TestClient, env) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl", lora_rank=128)
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/save_as_preset",
        json={"name": "my-tuned"},
    )
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["saved_preset"] == "my-tuned"
    assert saved["config"]["data_dir"] == "./dataset"
    assert saved["config"]["lora_rank"] == 128

    from studio.services.presets import io as presets_io
    presets = {p["name"] for p in presets_io.list_presets()}
    assert "my-tuned" in presets


def test_save_as_preset_existing_409(client: TestClient, env) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/save_as_preset",
        json={"name": "tpl"},
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "preset.exists"
    r2 = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/save_as_preset",
        json={"name": "tpl", "overwrite": True},
    )
    assert r2.status_code == 200


def test_save_as_preset_without_config_400(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/save_as_preset",
        json={"name": "x"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_enqueue_without_config_400(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(f"/api/projects/{pid}/versions/{vid}/queue")
    assert r.status_code == 400


def test_enqueue_creates_task_with_ids_and_config_path(
    client: TestClient, env
) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl", lora_rank=64)
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    r = client.post(f"/api/projects/{pid}/versions/{vid}/queue")
    assert r.status_code == 200, r.text
    task = r.json()
    assert task["status"] == "pending"
    assert task["project_id"] == pid
    assert task["version_id"] == vid
    from studio.services import task_snapshot
    expected = task_snapshot.snapshot_config_path(task["id"])
    assert task["config_path"] == str(expected)
    assert expected.exists()


def test_each_enqueued_task_keeps_model_selected_at_that_moment(
    client: TestClient, env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from studio.services import task_snapshot, version_config

    pid, vid = _make(client)
    _seed_preset(env, "tpl")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )

    selected = {"path": (tmp_path / "model-a.safetensors").as_posix()}

    def overlay(data):
        result = dict(data)
        result["transformer_path"] = selected["path"]
        return result

    monkeypatch.setattr(version_config, "apply_global_path_overlay", overlay)

    first = client.post(f"/api/projects/{pid}/versions/{vid}/queue").json()
    first_snapshot = task_snapshot.read_snapshot_config(first["id"])
    assert first_snapshot is not None
    assert first_snapshot["config"]["transformer_path"] == selected["path"]

    with db.connection_for(env["db"]) as conn:
        db.update_task(conn, first["id"], status="done")

    selected["path"] = (tmp_path / "model-b.safetensors").as_posix()
    second = client.post(f"/api/projects/{pid}/versions/{vid}/queue").json()
    second_snapshot = task_snapshot.read_snapshot_config(second["id"])
    assert second_snapshot is not None
    assert second_snapshot["config"]["transformer_path"] == selected["path"]

    # Первая задача не изменилась после переключения.
    first_snapshot = task_snapshot.read_snapshot_config(first["id"])
    assert first_snapshot is not None
    assert first_snapshot["config"]["transformer_path"].endswith("model-a.safetensors")


def test_enqueue_rejects_active_task(client: TestClient, env) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    r1 = client.post(f"/api/projects/{pid}/versions/{vid}/queue")
    assert r1.status_code == 200
    r2 = client.post(f"/api/projects/{pid}/versions/{vid}/queue")
    assert r2.status_code == 409


def test_enqueue_with_schedule_creates_scheduled_task(
    client: TestClient, env
) -> None:
    import time as _time
    pid, vid = _make(client)
    _seed_preset(env, "tpl")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    future = _time.time() + 3600
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/queue",
        json={"scheduled_at": future},
    )
    assert r.status_code == 200, r.text
    task = r.json()
    assert task["status"] == "scheduled"
    assert task["scheduled_at"] == pytest.approx(future)
    assert task["project_id"] == pid
    assert task["config_path"] and task["config_path"].endswith("config.yaml")


def test_enqueue_rejects_when_scheduled_active(client: TestClient, env) -> None:
    import time as _time
    pid, vid = _make(client)
    _seed_preset(env, "tpl")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    r1 = client.post(
        f"/api/projects/{pid}/versions/{vid}/queue",
        json={"scheduled_at": _time.time() + 3600},
    )
    assert r1.status_code == 200
    assert client.post(f"/api/projects/{pid}/versions/{vid}/queue").status_code == 409
    r3 = client.post(
        f"/api/projects/{pid}/versions/{vid}/queue",
        json={"scheduled_at": _time.time() + 7200},
    )
    assert r3.status_code == 409
