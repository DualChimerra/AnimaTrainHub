from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from studio.api.routers import logs as _logs_router
    from studio.api.routers.queue import lifecycle as _queue_lifecycle
    from studio.infrastructure import paths as _paths

    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    presets = tmp_path / "presets"
    logs = tmp_path / "logs"
    presets.mkdir()
    logs.mkdir()
    (presets / "good.yaml").write_text("epochs: 1\n", encoding="utf-8")

    monkeypatch.setattr(server, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server, "USER_PRESETS_DIR", presets)
    monkeypatch.setattr(server, "LOGS_DIR", logs)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(_queue_lifecycle, "USER_PRESETS_DIR", presets)
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(_logs_router, "LOGS_DIR", logs)
    return tmp_path


class _StubSupervisor:
    def __init__(self) -> None:
        self.canceled: list[int] = []
        self.current_task_id: int | None = None
    def cancel(self, task_id: int) -> bool:
        with db.connection_for() as conn:
            task = db.get_task(conn, task_id)
            if not task or task["status"] not in ("pending", "scheduled", "running"):
                return False
            db.update_task(conn, task_id, status="canceled")
        self.canceled.append(task_id)
        return True
    def is_task_pausable(self, task_id: int) -> bool:
        return False


@pytest.fixture
def client(isolated: Path) -> TestClient:
    server.app.state.supervisor = _StubSupervisor()
    return TestClient(server.app)


# ---------------------------------------------------------------------------


def test_empty_queue(client: TestClient) -> None:
    resp = client.get("/api/queue")
    assert resp.status_code == 200
    assert resp.json()["items"] == []


def test_enqueue_and_get(client: TestClient) -> None:
    resp = client.post("/api/queue", json={"config_name": "good", "name": "task1"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["config_name"] == "good"
    assert data["status"] == "pending"
    tid = data["id"]
    from studio.services import task_snapshot
    assert data["config_path"] == str(task_snapshot.snapshot_config_path(tid))
    assert task_snapshot.snapshot_config_path(tid).exists()

    got = client.get(f"/api/queue/{tid}")
    assert got.status_code == 200
    assert got.json()["id"] == tid


def test_enqueue_missing_config_404(client: TestClient) -> None:
    resp = client.post("/api/queue", json={"config_name": "ghost"})
    assert resp.status_code == 404


def test_filter_by_status(client: TestClient) -> None:
    client.post("/api/queue", json={"config_name": "good", "name": "a"})
    items = client.get("/api/queue?status=pending").json()["items"]
    assert len(items) == 1
    assert client.get("/api/queue?status=done").json()["items"] == []


def test_invalid_status_400(client: TestClient) -> None:
    resp = client.get("/api/queue?status=banana")
    assert resp.status_code == 400


def test_cancel_pending(client: TestClient) -> None:
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    resp = client.post(f"/api/queue/{tid}/cancel")
    assert resp.status_code == 200
    with db.connection_for() as conn:
        assert db.get_task(conn, tid)["status"] == "canceled"


def test_cancel_already_terminal_400(client: TestClient) -> None:
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    with db.connection_for() as conn:
        db.update_task(conn, tid, status="done")
    resp = client.post(f"/api/queue/{tid}/cancel")
    assert resp.status_code == 400


def test_retry_terminal_creates_new(client: TestClient) -> None:
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    with db.connection_for() as conn:
        db.update_task(conn, tid, status="failed")
    resp = client.post(f"/api/queue/{tid}/retry")
    assert resp.status_code == 200
    new_id = resp.json()["id"]
    assert new_id != tid
    assert resp.json()["status"] == "pending"


def test_retry_running_400(client: TestClient) -> None:
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    with db.connection_for() as conn:
        db.update_task(conn, tid, status="running")
    resp = client.post(f"/api/queue/{tid}/retry")
    assert resp.status_code == 400


def test_retry_copies_full_training_context(client: TestClient) -> None:
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    with db.connection_for() as conn:
        db.update_task(
            conn,
            tid,
            status="failed",
            config_path="/abs/path/to/version_private.yaml",
            project_id=42,
            version_id=99,
        )
    new = client.post(f"/api/queue/{tid}/retry").json()
    assert new["config_path"] == "/abs/path/to/version_private.yaml"
    assert new["project_id"] == 42
    assert new["version_id"] == 99
    assert new["status"] == "pending"
    assert new.get("started_at") is None
    assert new.get("finished_at") is None
    assert new.get("error_msg") is None
    assert new.get("monitor_state_path") is None


def test_retry_gets_its_own_frozen_config(client: TestClient) -> None:
    from studio.services import task_snapshot

    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    with db.connection_for() as conn:
        db.update_task(conn, tid, status="failed")

    new = client.post(f"/api/queue/{tid}/retry").json()
    expected = task_snapshot.snapshot_config_path(new["id"])
    assert new["config_path"] == str(expected)
    assert expected.read_text(encoding="utf-8") == "epochs: 1\n"


def test_outputs_list_with_files(
    client: TestClient, isolated, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.projects import projects as projects_mod, versions as versions_mod
    monkeypatch.setattr(projects_mod, "PROJECTS_DIR", isolated / "projects")
    with db.connection_for() as conn:
        p = projects_mod.create_project(conn, title="P")
        v = versions_mod.create_version(conn, project_id=p["id"], label="v1")
        tid = db.create_task(conn, name="t", config_name="good")
        db.update_task(conn, tid, status="done", project_id=p["id"], version_id=v["id"])
    out_dir = (
        versions_mod.version_dir(p["id"], p["slug"], v["label"]) / "output"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lora_final.safetensors").write_bytes(b"x" * 100)
    (out_dir / "training_state_step100.pt").write_bytes(b"y" * 50)
    state_dir = out_dir / "state" / f"task_{tid}"
    state_dir.mkdir(parents=True)
    (state_dir / "training_state_epoch2.pt").write_bytes(b"z" * 25)
    (state_dir / "notes.txt").write_text("ignore", encoding="utf-8")
    other_state_dir = out_dir / "state" / "task_999"
    other_state_dir.mkdir(parents=True)
    (other_state_dir / "training_state_epoch20.pt").write_bytes(b"other")
    wandb_dir = out_dir / "wandb" / "wandb" / "run-20260522_133229-rw752qai" / "files"
    wandb_dir.mkdir(parents=True)
    (wandb_dir / "wandb-metadata.json").write_text("{}", encoding="utf-8")
    samples_dir = out_dir / "samples"
    samples_dir.mkdir()
    (samples_dir / "step_0_baseline_0.png").write_bytes(b"png")

    resp = client.get(f"/api/queue/{tid}/outputs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == tid
    assert body["exists"] is True
    paths = sorted(f["path"] for f in body["files"])
    assert paths == [
        "lora_final.safetensors",
        "state/task_%d/training_state_epoch2.pt" % tid,
        "training_state_step100.pt",
    ]
    by_path = {f["path"]: f for f in body["files"]}
    assert by_path["lora_final.safetensors"]["is_lora"] is True
    assert by_path["lora_final.safetensors"]["kind"] == "lora"
    assert by_path["lora_final.safetensors"]["size"] == 100
    assert by_path["training_state_step100.pt"]["is_lora"] is False
    assert by_path["training_state_step100.pt"]["kind"] == "training_state"
    nested_state = by_path["state/task_%d/training_state_epoch2.pt" % tid]
    assert nested_state["name"] == "training_state_epoch2.pt"
    assert nested_state["kind"] == "training_state"
    assert body["supports_open_folder"] is False


def test_outputs_list_no_version(client: TestClient) -> None:
    with db.connection_for() as conn:
        tid = db.create_task(conn, name="t", config_name="good")
        db.update_task(conn, tid, status="done")
    body = client.get(f"/api/queue/{tid}/outputs").json()
    assert body["output_dir"] is None
    assert body["exists"] is False
    assert body["files"] == []


def test_download_output_file(
    client: TestClient, isolated, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.projects import projects as projects_mod, versions as versions_mod
    monkeypatch.setattr(projects_mod, "PROJECTS_DIR", isolated / "projects")
    with db.connection_for() as conn:
        p = projects_mod.create_project(conn, title="P")
        v = versions_mod.create_version(conn, project_id=p["id"], label="v1")
        tid = db.create_task(conn, name="t", config_name="good")
        db.update_task(conn, tid, status="done", project_id=p["id"], version_id=v["id"])
    out_dir = versions_mod.version_dir(p["id"], p["slug"], v["label"]) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lora_final.safetensors").write_bytes(b"BLOB")
    state_dir = out_dir / "state" / f"task_{tid}"
    state_dir.mkdir(parents=True)
    (state_dir / "training_state_epoch2.pt").write_bytes(b"STATE")

    resp = client.get(f"/api/queue/{tid}/output/lora_final.safetensors")
    assert resp.status_code == 200
    assert resp.content == b"BLOB"

    resp = client.get(f"/api/queue/{tid}/output/state/task_{tid}/training_state_epoch2.pt")
    assert resp.status_code == 200
    assert resp.content == b"STATE"
    assert "attachment" in resp.headers.get("content-disposition", "").lower()


def test_download_outputs_zip(
    client: TestClient, isolated, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import zipfile
    from studio.services.projects import projects as projects_mod, versions as versions_mod
    monkeypatch.setattr(projects_mod, "PROJECTS_DIR", isolated / "projects")
    with db.connection_for() as conn:
        p = projects_mod.create_project(conn, title="P")
        v = versions_mod.create_version(conn, project_id=p["id"], label="v1")
        tid = db.create_task(conn, name="t", config_name="good")
        db.update_task(conn, tid, status="done", project_id=p["id"], version_id=v["id"])
    out_dir = versions_mod.version_dir(p["id"], p["slug"], v["label"]) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lora_final.safetensors").write_bytes(b"AAA")
    (out_dir / "training_state_step100.pt").write_bytes(b"BB")
    state_dir = out_dir / "state" / f"task_{tid}"
    state_dir.mkdir(parents=True)
    (state_dir / "training_state_epoch2.pt").write_bytes(b"CC")
    wandb_dir = out_dir / "wandb" / "wandb" / "run-20260522_133229-rw752qai" / "files"
    wandb_dir.mkdir(parents=True)
    (wandb_dir / "requirements.txt").write_text("wandb", encoding="utf-8")
    samples_dir = out_dir / "samples"
    samples_dir.mkdir()
    (samples_dir / "vae_roundtrip.png").write_bytes(b"png")

    resp = client.get(f"/api/queue/{tid}/outputs.zip")
    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "application/zip"
    expected_name = f"{p['slug']}-{v['label']}_outputs.zip"
    assert expected_name in resp.headers.get("content-disposition", "")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = sorted(zf.namelist())
        assert names == [
            "lora_final.safetensors",
            "state/task_%d/training_state_epoch2.pt" % tid,
            "training_state_step100.pt",
        ]
        assert zf.read("lora_final.safetensors") == b"AAA"
        assert zf.read("training_state_step100.pt") == b"BB"
        assert zf.read("state/task_%d/training_state_epoch2.pt" % tid) == b"CC"


def test_list_task_outputs_returns_archive_basename(
    client: TestClient, isolated, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.projects import projects as projects_mod, versions as versions_mod
    monkeypatch.setattr(projects_mod, "PROJECTS_DIR", isolated / "projects")
    with db.connection_for() as conn:
        p = projects_mod.create_project(conn, title="P")
        v = versions_mod.create_version(conn, project_id=p["id"], label="v1")
        bound = db.create_task(conn, name="t", config_name="good")
        db.update_task(conn, bound, status="done", project_id=p["id"], version_id=v["id"])
        legacy = db.create_task(conn, name="old", config_name="good")

    resp = client.get(f"/api/queue/{bound}/outputs")
    assert resp.status_code == 200
    assert resp.json()["archive_basename"] == f"{p['slug']}-{v['label']}"

    resp = client.get(f"/api/queue/{legacy}/outputs")
    assert resp.status_code == 200
    assert resp.json()["archive_basename"] is None


def test_download_outputs_zip_partial(
    client: TestClient, isolated, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import zipfile
    from studio.services.projects import projects as projects_mod, versions as versions_mod
    monkeypatch.setattr(projects_mod, "PROJECTS_DIR", isolated / "projects")
    with db.connection_for() as conn:
        p = projects_mod.create_project(conn, title="P")
        v = versions_mod.create_version(conn, project_id=p["id"], label="v1")
        tid = db.create_task(conn, name="t", config_name="good")
        db.update_task(conn, tid, status="done", project_id=p["id"], version_id=v["id"])
    out_dir = versions_mod.version_dir(p["id"], p["slug"], v["label"]) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ep_001.safetensors").write_bytes(b"E1")
    (out_dir / "ep_002.safetensors").write_bytes(b"E2")
    (out_dir / "ep_003.safetensors").write_bytes(b"E3")
    state_dir = out_dir / "state" / f"task_{tid}"
    state_dir.mkdir(parents=True)
    (state_dir / "training_state_epoch2.pt").write_bytes(b"S2")

    resp = client.get(
        f"/api/queue/{tid}/outputs.zip?files=ep_001.safetensors,state/task_{tid}/training_state_epoch2.pt"
    )
    assert resp.status_code == 200
    expected_name = f"{p['slug']}-{v['label']}_outputs_selected.zip"
    assert expected_name in resp.headers.get("content-disposition", "")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = sorted(zf.namelist())
        nested = f"state/task_{tid}/training_state_epoch2.pt"
        assert names == ["ep_001.safetensors", nested]
        assert zf.read("ep_001.safetensors") == b"E1"
        assert zf.read(nested) == b"S2"


def test_download_outputs_zip_partial_missing_file_404(
    client: TestClient, isolated, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.projects import projects as projects_mod, versions as versions_mod
    monkeypatch.setattr(projects_mod, "PROJECTS_DIR", isolated / "projects")
    with db.connection_for() as conn:
        p = projects_mod.create_project(conn, title="P")
        v = versions_mod.create_version(conn, project_id=p["id"], label="v1")
        tid = db.create_task(conn, name="t", config_name="good")
        db.update_task(conn, tid, status="done", project_id=p["id"], version_id=v["id"])
    out_dir = versions_mod.version_dir(p["id"], p["slug"], v["label"]) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a.safetensors").write_bytes(b"A")

    resp = client.get(f"/api/queue/{tid}/outputs.zip?files=a.safetensors,ghost.safetensors")
    assert resp.status_code == 404


def test_download_outputs_zip_partial_blocks_traversal(
    client: TestClient, isolated, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.projects import projects as projects_mod, versions as versions_mod
    monkeypatch.setattr(projects_mod, "PROJECTS_DIR", isolated / "projects")
    with db.connection_for() as conn:
        p = projects_mod.create_project(conn, title="P")
        v = versions_mod.create_version(conn, project_id=p["id"], label="v1")
        tid = db.create_task(conn, name="t", config_name="good")
        db.update_task(conn, tid, status="done", project_id=p["id"], version_id=v["id"])
    out_dir = versions_mod.version_dir(p["id"], p["slug"], v["label"]) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a.safetensors").write_bytes(b"A")
    nested_dir = out_dir / "state" / f"task_{tid}"
    nested_dir.mkdir(parents=True)
    (nested_dir / "training_state_epoch2.pt").write_bytes(b"S2")

    safe = f"state/task_{tid}/training_state_epoch2.pt"
    resp = client.get(f"/api/queue/{tid}/outputs.zip", params={"files": safe})
    assert resp.status_code == 200

    for bad in ("../secret", "..\\secret", "/abs", "state/../secret"):
        resp = client.get(f"/api/queue/{tid}/outputs.zip", params={"files": bad})
        assert resp.status_code == 400, f"{bad!r} should be 400, got {resp.status_code}"



def test_delete_task_output_files(
    client: TestClient, isolated, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.projects import projects as projects_mod, versions as versions_mod
    monkeypatch.setattr(projects_mod, "PROJECTS_DIR", isolated / "projects")
    with db.connection_for() as conn:
        p = projects_mod.create_project(conn, title="P")
        v = versions_mod.create_version(conn, project_id=p["id"], label="v1")
        tid = db.create_task(conn, name="t", config_name="good")
        db.update_task(conn, tid, status="done", project_id=p["id"], version_id=v["id"])
    out_dir = versions_mod.version_dir(p["id"], p["slug"], v["label"]) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "keep.safetensors").write_bytes(b"K")
    (out_dir / "drop.safetensors").write_bytes(b"D")
    state_dir = out_dir / "state" / f"task_{tid}"
    state_dir.mkdir(parents=True)
    (state_dir / "training_state_epoch2.pt").write_bytes(b"S")

    resp = client.request(
        "DELETE",
        f"/api/queue/{tid}/outputs",
        json={"files": ["drop.safetensors", f"state/task_{tid}/training_state_epoch2.pt"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert sorted(body["deleted"]) == sorted([
        "drop.safetensors",
        f"state/task_{tid}/training_state_epoch2.pt",
    ])
    assert (out_dir / "keep.safetensors").exists()
    assert not (out_dir / "drop.safetensors").exists()
    assert not (state_dir / "training_state_epoch2.pt").exists()


def test_delete_task_output_files_missing_404(
    client: TestClient, isolated, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.projects import projects as projects_mod, versions as versions_mod
    monkeypatch.setattr(projects_mod, "PROJECTS_DIR", isolated / "projects")
    with db.connection_for() as conn:
        p = projects_mod.create_project(conn, title="P")
        v = versions_mod.create_version(conn, project_id=p["id"], label="v1")
        tid = db.create_task(conn, name="t", config_name="good")
        db.update_task(conn, tid, status="done", project_id=p["id"], version_id=v["id"])
    out_dir = versions_mod.version_dir(p["id"], p["slug"], v["label"]) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a.safetensors").write_bytes(b"A")

    resp = client.request(
        "DELETE",
        f"/api/queue/{tid}/outputs",
        json={"files": ["a.safetensors", "ghost.safetensors"]},
    )
    assert resp.status_code == 404
    assert (out_dir / "a.safetensors").exists()


def test_delete_task_output_files_blocks_traversal(
    client: TestClient, isolated, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.projects import projects as projects_mod, versions as versions_mod
    monkeypatch.setattr(projects_mod, "PROJECTS_DIR", isolated / "projects")
    with db.connection_for() as conn:
        p = projects_mod.create_project(conn, title="P")
        v = versions_mod.create_version(conn, project_id=p["id"], label="v1")
        tid = db.create_task(conn, name="t", config_name="good")
        db.update_task(conn, tid, status="done", project_id=p["id"], version_id=v["id"])
    out_dir = versions_mod.version_dir(p["id"], p["slug"], v["label"]) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a.safetensors").write_bytes(b"A")

    for bad in ("../secret", "/abs", "state/../secret"):
        resp = client.request("DELETE", f"/api/queue/{tid}/outputs", json={"files": [bad]})
        assert resp.status_code == 400, f"{bad!r} should be 400, got {resp.status_code}"


def test_download_outputs_zip_empty_dir_404(
    client: TestClient, isolated, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.projects import projects as projects_mod, versions as versions_mod
    monkeypatch.setattr(projects_mod, "PROJECTS_DIR", isolated / "projects")
    with db.connection_for() as conn:
        p = projects_mod.create_project(conn, title="P")
        v = versions_mod.create_version(conn, project_id=p["id"], label="v1")
        tid = db.create_task(conn, name="t", config_name="good")
        db.update_task(conn, tid, status="done", project_id=p["id"], version_id=v["id"])
    out_dir = versions_mod.version_dir(p["id"], p["slug"], v["label"]) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    resp = client.get(f"/api/queue/{tid}/outputs.zip")
    assert resp.status_code == 404


def test_download_output_blocks_traversal(client: TestClient) -> None:
    with db.connection_for() as conn:
        tid = db.create_task(conn, name="t", config_name="good")
    for bad in ("../etc.txt", "..\\etc.txt", ""):
        resp = client.get(f"/api/queue/{tid}/output/{bad}")
        assert resp.status_code != 200


def test_open_folder_blocks_non_loopback(client: TestClient) -> None:
    with db.connection_for() as conn:
        tid = db.create_task(conn, name="t", config_name="good")
    resp = client.post(f"/api/queue/{tid}/open-folder")
    assert resp.status_code == 403


def test_delete_only_terminal(client: TestClient) -> None:
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    assert client.delete(f"/api/queue/{tid}").status_code == 400
    with db.connection_for() as conn:
        db.update_task(conn, tid, status="done")
    assert client.delete(f"/api/queue/{tid}").status_code == 200
    assert client.get(f"/api/queue/{tid}").status_code == 404


def test_reorder(client: TestClient) -> None:
    a = client.post("/api/queue", json={"config_name": "good", "name": "a"}).json()["id"]
    b = client.post("/api/queue", json={"config_name": "good", "name": "b"}).json()["id"]
    resp = client.post("/api/queue/reorder", json={"ordered_ids": [b, a]})
    assert resp.status_code == 200
    items = client.get("/api/queue?status=pending").json()["items"]
    assert [i["id"] for i in items] == [b, a]



def _seed(status: str, *, name: str = "t", task_type: str = "train") -> int:
    with db.connection_for() as conn:
        tid = db.create_task(conn, name=name, config_name=name)
        db.update_task(conn, tid, status=status, task_type=task_type)
    return tid


def test_group_live_only_returns_live_statuses(client: TestClient) -> None:
    _seed("running", name="run")
    _seed("pending", name="pend")
    _seed("paused", name="pause")
    _seed("done", name="hist")
    items = client.get("/api/queue?group=live").json()["items"]
    assert {i["status"] for i in items} == {"running", "pending", "paused"}


def test_group_history_paginates_with_total(client: TestClient) -> None:
    ids = [_seed("done", name=f"d{i}") for i in range(5)]
    body = client.get("/api/queue?group=history&page=1&page_size=2").json()
    assert body["total"] == 5
    assert body["page"] == 1
    assert body["page_size"] == 2
    assert [i["id"] for i in body["items"]] == [ids[4], ids[3]]  # id DESC
    page3 = client.get("/api/queue?group=history&page=3&page_size=2").json()
    assert [i["id"] for i in page3["items"]] == [ids[0]]


def test_group_history_page_size_clamped(client: TestClient) -> None:
    _seed("done")
    body = client.get("/api/queue?group=history&page_size=9999").json()
    assert body["page_size"] == 100  # _MAX_PAGE_SIZE


def test_group_history_search_q(client: TestClient) -> None:
    _seed("done", name="alpha_lora")
    _seed("done", name="beta_run")
    body = client.get("/api/queue?group=history&q=alpha").json()
    assert [i["name"] for i in body["items"]] == ["alpha_lora"]
    assert body["total"] == 1


def test_group_history_status_subfilter(client: TestClient) -> None:
    _seed("done", name="d")
    _seed("failed", name="f")
    _seed("canceled", name="c")
    body = client.get("/api/queue?group=history&status=failed").json()
    assert [i["name"] for i in body["items"]] == ["f"]
    assert body["total"] == 1


def test_group_history_includes_all_types_by_default(client: TestClient) -> None:
    _seed("done", name="train")
    _seed("done", name="reg", task_type="reg_ai")
    _seed("done", name="gen", task_type="generate")
    body = client.get("/api/queue?group=history").json()
    assert {i["name"] for i in body["items"]} == {"train", "reg", "gen"}
    assert body["total"] == 3


def test_group_history_type_filter(client: TestClient) -> None:
    _seed("done", name="train")
    _seed("done", name="reg", task_type="reg_ai")
    _seed("done", name="gen", task_type="generate")
    body = client.get("/api/queue?group=history&types=generate").json()
    assert [i["name"] for i in body["items"]] == ["gen"]
    assert body["total"] == 1
    body2 = client.get("/api/queue?group=history&types=reg_ai,generate").json()
    assert {i["name"] for i in body2["items"]} == {"reg", "gen"}
    assert body2["total"] == 2


def test_group_live_type_filter(client: TestClient) -> None:
    _seed("running", name="train")
    _seed("pending", name="gen", task_type="generate")
    items = client.get("/api/queue?group=live&types=generate").json()["items"]
    assert [i["name"] for i in items] == ["gen"]


def test_invalid_type_filter_400(client: TestClient) -> None:
    resp = client.get("/api/queue?group=history&types=banana")
    assert resp.status_code == 400


def test_invalid_group_400(client: TestClient) -> None:
    resp = client.get("/api/queue?group=banana")
    assert resp.status_code == 400


def test_no_group_returns_all_backward_compat(client: TestClient) -> None:
    _seed("running")
    _seed("done")
    items = client.get("/api/queue").json()["items"]
    assert {i["status"] for i in items} == {"running", "done"}




def test_enqueue_with_scheduled_at_creates_scheduled(client: TestClient) -> None:
    import time as _time
    future = _time.time() + 3600
    resp = client.post(
        "/api/queue", json={"config_name": "good", "scheduled_at": future},
    )
    assert resp.status_code == 200, resp.text
    task = resp.json()
    assert task["status"] == "scheduled"
    assert task["scheduled_at"] == pytest.approx(future)


def test_scheduled_appears_in_group_live(client: TestClient) -> None:
    import time as _time
    client.post(
        "/api/queue",
        json={"config_name": "good", "scheduled_at": _time.time() + 3600},
    )
    items = client.get("/api/queue?group=live").json()["items"]
    assert [i["status"] for i in items] == ["scheduled"]


def test_start_now_promotes_scheduled_to_pending(client: TestClient) -> None:
    import time as _time
    future = _time.time() + 3600
    tid = client.post(
        "/api/queue", json={"config_name": "good", "scheduled_at": future},
    ).json()["id"]
    resp = client.post(f"/api/queue/{tid}/start_now")
    assert resp.status_code == 200
    task = client.get(f"/api/queue/{tid}").json()
    assert task["status"] == "pending"
    assert task["scheduled_at"] == pytest.approx(future)


def test_start_now_rejects_non_scheduled(client: TestClient) -> None:
    tid = client.post("/api/queue", json={"config_name": "good"}).json()["id"]
    resp = client.post(f"/api/queue/{tid}/start_now")
    assert resp.status_code == 409
    missing = client.post("/api/queue/9999/start_now")
    assert missing.status_code == 404


def test_cancel_scheduled_task(client: TestClient) -> None:
    import time as _time
    tid = client.post(
        "/api/queue",
        json={"config_name": "good", "scheduled_at": _time.time() + 3600},
    ).json()["id"]
    resp = client.post(f"/api/queue/{tid}/cancel")
    assert resp.status_code == 200
    assert client.get(f"/api/queue/{tid}").json()["status"] == "canceled"


def test_status_filter_accepts_scheduled(client: TestClient) -> None:
    import time as _time
    client.post(
        "/api/queue",
        json={"config_name": "good", "scheduled_at": _time.time() + 3600},
    )
    _seed("pending", name="p")
    resp = client.get("/api/queue?status=scheduled")
    assert resp.status_code == 200
    assert [i["status"] for i in resp.json()["items"]] == ["scheduled"]


def test_logs_missing_returns_empty(client: TestClient) -> None:
    resp = client.get("/api/logs/9999")
    assert resp.status_code == 200
    assert resp.json()["content"] == ""


def test_logs_returns_content(client: TestClient, isolated: Path) -> None:
    log_path = isolated / "logs" / "42.log"
    log_path.write_text("hello world\n", encoding="utf-8")
    resp = client.get("/api/logs/42")
    assert resp.status_code == 200
    assert resp.json()["content"] == "hello world\n"
