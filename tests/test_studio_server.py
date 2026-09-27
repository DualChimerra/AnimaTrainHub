from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import server


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    from studio import db
    from studio.api.routers import root as _root_router
    from studio.api.routers import samples as _samples_router
    from studio.services.projects import projects
    output = tmp_path / "output"
    samples_dir = output / "samples"
    web_dist = tmp_path / "web_dist"
    samples_dir.mkdir(parents=True)

    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server, "OUTPUT_DIR", output)
    monkeypatch.setattr(server, "WEB_DIST", web_dist)
    monkeypatch.setattr(_samples_router, "OUTPUT_DIR", output)
    monkeypatch.setattr(_root_router, "WEB_DIST", web_dist)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    return {
        "tmp": tmp_path,
        "db": dbfile,
        "output": output,
        "samples_dir": samples_dir,
        "web_dist": web_dist,
    }


@pytest.fixture
def client(isolated_paths: dict[str, Path]) -> TestClient:
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == server.app.version


def test_generate_sample_response_is_not_browser_cached(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    from studio.services.inference import disk_cache as generate_cache

    generate_cache.init(isolated_paths["tmp"] / ".cache" / "generate")

    try:
        generate_cache.cache_image(7, "sample.png", b"PNG", snapshot={"mode": "single"})
        resp = client.get("/api/generate/7/sample/sample.png")
        assert resp.status_code == 200
        assert resp.content == b"PNG"
        assert resp.headers["cache-control"] == "no-store"
    finally:
        generate_cache.drop_task(7)


# ---------------------------------------------------------------------------
# /api/state
# ---------------------------------------------------------------------------

def test_torch_status_proxies_service(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.runtime import torch as torch_setup
    monkeypatch.setattr(torch_setup, "current_status", lambda: {
        "installed": True,
        "version": "2.5.0+cpu",
        "cuda_build": "cpu",
        "cuda_available": False,
        "device_name": None,
        "cuda_detect": {"available": True, "driver_version": "555.86", "gpu_name": "RTX 5090"},
        "recommended_cu_tag": "cu128",
        "is_cpu_with_gpu": True,
        "is_cuda_build_unavailable": False,
    })
    resp = client.get("/api/torch/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_cpu_with_gpu"] is True
    assert body["recommended_cu_tag"] == "cu128"


def test_torch_reinstall_registers_marker_returns_pending(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from studio.services.runtime import pending_install, torch as torch_setup
    monkeypatch.setattr(pending_install, "STUDIO_DATA", tmp_path)
    monkeypatch.setattr(pending_install, "PENDING_MARKER", tmp_path / ".pending-pip-install.json")
    monkeypatch.setattr(torch_setup, "_decide_target_tag", lambda _t: "cu128")

    resp = client.post("/api/torch/reinstall", json={"target": "auto"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pending"] is True
    assert body["tag"] == "cu128"
    assert body["target"] == "auto"
    assert "studio.bat" in body["message"]
    assert (tmp_path / ".pending-pip-install.json").exists()


def test_torch_reinstall_invalid_target_returns_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.runtime import torch as torch_setup
    monkeypatch.setattr(
        torch_setup, "_decide_target_tag",
        lambda t: (_ for _ in ()).throw(ValueError(f"invalid target: {t!r}")),
    )
    resp = client.post("/api/torch/reinstall", json={"target": "xpu"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "install.target_invalid"
    assert body["error"]["details"]["target"] == "xpu"


def test_flash_attention_status_returns_env_and_candidates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.runtime import flash_attention as flash_attention_setup
    monkeypatch.setattr(flash_attention_setup, "current_status", lambda: {
        "installed": True, "version": "2.8.3"
    })
    monkeypatch.setattr(flash_attention_setup, "detect_env", lambda: {
        "python_tag": "cp311", "cuda_tag": "cu128", "cuda_ver": "12.8",
        "torch_tag": "torch2.5", "torch_ver": "2.5.0+cu128", "platform": "win_amd64",
    })
    monkeypatch.setattr(flash_attention_setup, "find_candidates", lambda _env: ([
        {
            "url": "https://x/wheel.whl",
            "name": "flash_attn-2.8.3+cu128torch2.5-cp311-cp311-win_amd64.whl",
            "score": 40,
            "notes": [],
            "usable": True,
            "tags": {"cuda": "cu128"},
        },
    ], None))

    resp = client.get("/api/flash-attention/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["installed"] is True
    assert body["version"] == "2.8.3"
    assert body["env"]["platform"] == "win_amd64"
    assert len(body["candidates"]) == 1
    c = body["candidates"][0]
    assert set(c.keys()) == {"url", "name", "notes", "usable"}
    assert body["fetch_error"] is None


def test_flash_attention_status_passes_fetch_error_through(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.runtime import flash_attention as flash_attention_setup
    monkeypatch.setattr(flash_attention_setup, "current_status", lambda: {
        "installed": False, "version": None,
    })
    monkeypatch.setattr(flash_attention_setup, "detect_env", lambda: {
        "python_tag": "cp311", "cuda_tag": None, "cuda_ver": None,
        "torch_tag": None, "torch_ver": None, "platform": "linux_x86_64",
    })
    monkeypatch.setattr(
        flash_attention_setup, "find_candidates",
        lambda _env: ([], "GitHub API error: API rate limit exceeded"),
    )
    resp = client.get("/api/flash-attention/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidates"] == []
    assert "rate limit" in body["fetch_error"]


def test_flash_attention_install_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.runtime import flash_attention as flash_attention_setup
    captured: dict = {}

    def fake_install(url):
        captured["url"] = url
        return {
            "installed": True, "version": "2.8.3",
            "url": url or "https://auto/wheel.whl",
            "stdout_tail": "Successfully installed",
            "restart_required": True,
        }

    monkeypatch.setattr(flash_attention_setup, "install", fake_install)
    resp = client.post("/api/flash-attention/install", json={"url": "https://x/manual.whl"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["installed"] is True
    assert body["restart_required"] is True
    assert captured["url"] == "https://x/manual.whl"


def test_flash_attention_install_url_null_uses_auto(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.runtime import flash_attention as flash_attention_setup
    captured: dict = {}

    def fake_install(url):
        captured["url"] = url
        return {"installed": True, "version": "2.8.3", "url": "auto",
                "stdout_tail": "", "restart_required": True}

    monkeypatch.setattr(flash_attention_setup, "install", fake_install)
    resp = client.post("/api/flash-attention/install", json={"url": None})
    assert resp.status_code == 200
    assert captured["url"] is None


def test_flash_attention_install_failure_returns_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.runtime import flash_attention as flash_attention_setup

    def boom(_url):
        raise RuntimeError("pip install failed:\nERROR: bad wheel")

    monkeypatch.setattr(flash_attention_setup, "install", boom)
    resp = client.post("/api/flash-attention/install", json={"url": "https://x/bad.whl"})
    assert resp.status_code == 500
    assert "bad wheel" in resp.json()["error"]["message"]


def test_state_missing_returns_empty(client: TestClient, isolated_paths: dict[str, Path]) -> None:
    resp = client.get("/api/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["losses"] == []
    assert body["lr_history"] == []
    assert body["step"] == 0
    assert body["epoch"] == 0
    assert body["start_time"] is None


def _make_task_with_state(
    isolated_paths: dict[str, Path], payload: dict | str | None
) -> int:
    from studio import db as _db
    state_dir = isolated_paths["tmp"] / "states"
    state_dir.mkdir(exist_ok=True)
    state_file = state_dir / "state.json"
    if payload is not None:
        state_file.write_text(
            json.dumps(payload) if isinstance(payload, dict) else payload,
            encoding="utf-8",
        )
    with _db.connection_for(isolated_paths["db"]) as conn:
        tid = _db.create_task(conn, name="t", config_name="x")
        _db.update_task(conn, tid, monitor_state_path=str(state_file))
    return tid


def test_state_by_task_id_returns_parsed_json(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    payload = {
        "losses": [{"step": 1, "loss": 0.5, "time": 100.0}],
        "lr_history": [{"step": 1, "lr": 1e-4}],
        "epoch": 2,
        "step": 42,
        "total_steps": 1000,
        "speed": 1.23,
        "samples": [],
        "start_time": 1700000000.0,
        "config": {"lora_rank": 32},
    }
    tid = _make_task_with_state(isolated_paths, payload)
    resp = client.get(f"/api/state?task_id={tid}")
    assert resp.status_code == 200
    assert resp.json() == payload


def test_state_includes_eval_context_for_bound_task(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    from studio import db as _db
    from studio.services.projects import projects, versions

    payload = {
        "losses": [],
        "lr_history": [],
        "epoch": 1,
        "step": 10,
        "total_steps": 20,
        "speed": 1.0,
        "samples": [],
        "start_time": 1700000000.0,
        "config": {},
    }
    tid = _make_task_with_state(isolated_paths, payload)
    with _db.connection_for(isolated_paths["db"]) as conn:
        project = projects.create_project(conn, title="Eval Monitor")
        version = versions.create_version(conn, project_id=project["id"], label="v1")
        _db.update_task(
            conn,
            tid,
            project_id=project["id"],
            version_id=version["id"],
        )

    resp = client.get(f"/api/state?task_id={tid}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["project_id"] == project["id"]
    assert body["project_slug"] == project["slug"]
    assert body["version_id"] == version["id"]
    assert body["version_label"] == "v1"
    assert body["task_id"] == tid
    assert body["step"] == 10


def test_state_corrupt_returns_500(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    tid = _make_task_with_state(isolated_paths, "this is not json")
    resp = client.get(f"/api/state?task_id={tid}")
    assert resp.status_code == 500


def test_state_unknown_task_returns_empty(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    resp = client.get("/api/state?task_id=99999")
    assert resp.status_code == 200
    assert resp.json()["losses"] == []


def test_state_running_task_used_when_no_task_id(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    payload = {"losses": [], "lr_history": [], "epoch": 0, "step": 7,
               "total_steps": 0, "speed": 0.0, "samples": [],
               "start_time": None, "config": {}}
    from studio import db as _db
    tid = _make_task_with_state(isolated_paths, payload)
    with _db.connection_for(isolated_paths["db"]) as conn:
        _db.update_task(conn, tid, status="running", started_at=1.0)
    resp = client.get("/api/state")
    assert resp.json()["step"] == 7


def test_state_max_points_downsamples_losses(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    losses = [{"step": i, "loss": 1.0 / (i + 1), "time": float(i)} for i in range(5000)]
    lr_history = [{"step": i, "lr": 1e-4} for i in range(5000)]
    optimizer_metrics_history = [{"step": i, "actual_lr": 1e-4, "d": 1e-4} for i in range(5000)]
    payload = {
        "losses": losses, "lr_history": lr_history,
        "optimizer_metrics_history": optimizer_metrics_history,
        "epoch": 0, "step": 4999,
        "total_steps": 5000, "speed": 0.0, "samples": [],
        "start_time": None, "config": {},
    }
    tid = _make_task_with_state(isolated_paths, payload)

    resp = client.get(f"/api/state?task_id={tid}&max_points=500")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["losses"]) == 500
    assert len(body["lr_history"]) == 500
    assert len(body["optimizer_metrics_history"]) == 500
    assert body["losses"][0]["step"] == 0
    assert body["losses"][-1]["step"] == 4999
    assert body["step"] == 4999
    assert body["total_steps"] == 5000


def test_state_max_points_zero_disables_downsample(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    losses = [{"step": i, "loss": 0.0} for i in range(100)]
    payload = {"losses": losses, "lr_history": [], "epoch": 0, "step": 99,
               "total_steps": 100, "speed": 0.0, "samples": [],
               "start_time": None, "config": {}}
    tid = _make_task_with_state(isolated_paths, payload)
    resp = client.get(f"/api/state?task_id={tid}&max_points=0")
    assert resp.status_code == 200
    assert len(resp.json()["losses"]) == 100


def test_state_default_returns_full_payload(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    losses = [{"step": i, "loss": 0.1} for i in range(10000)]
    payload = {
        "losses": losses, "lr_history": [], "epoch": 0, "step": 9999,
        "total_steps": 10000, "speed": 0.0, "samples": [],
        "start_time": None, "config": {},
    }
    tid = _make_task_with_state(isolated_paths, payload)
    resp = client.get(f"/api/state?task_id={tid}")
    assert resp.status_code == 200
    assert len(resp.json()["losses"]) == 10000


# ---------------------------------------------------------------------------
# /samples/{filename}
# ---------------------------------------------------------------------------

def test_sample_404_for_missing(client: TestClient) -> None:
    resp = client.get("/samples/does_not_exist.png")
    assert resp.status_code == 404


def test_sample_returns_file(client: TestClient, isolated_paths: dict[str, Path]) -> None:
    img_path = isolated_paths["samples_dir"] / "step_42.png"
    img_path.write_bytes(b"fake-png-bytes")
    resp = client.get("/samples/step_42.png")
    assert resp.status_code == 200
    assert resp.content == b"fake-png-bytes"


@pytest.mark.parametrize("bad", ["../secret.txt", "..\\secret.txt", "sub/dir.png", "sub\\dir.png"])
def test_sample_blocks_traversal(client: TestClient, bad: str) -> None:
    resp = client.get(f"/samples/{bad}")
    assert resp.status_code != 200


def test_sample_with_task_id_finds_in_output_samples(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    from studio import db as _db
    state_path = isolated_paths["tmp"] / "v1" / "monitor_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{}", encoding="utf-8")
    out_samples = state_path.parent / "output" / "samples"
    out_samples.mkdir(parents=True)
    (out_samples / "step_0_baseline_0.png").write_bytes(b"sample-bytes")

    with _db.connection_for(isolated_paths["db"]) as conn:
        tid = _db.create_task(conn, name="t", config_name="x")
        _db.update_task(conn, tid, monitor_state_path=str(state_path))

    resp = client.get(f"/samples/step_0_baseline_0.png?task_id={tid}")
    assert resp.status_code == 200, resp.text
    assert resp.content == b"sample-bytes"


def test_sample_with_task_id_finds_in_state_dir_samples(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    from studio import db as _db
    state_path = isolated_paths["tmp"] / "v2" / "monitor_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{}", encoding="utf-8")
    samples = state_path.parent / "samples"
    samples.mkdir()
    (samples / "step_5.png").write_bytes(b"old-layout")

    with _db.connection_for(isolated_paths["db"]) as conn:
        tid = _db.create_task(conn, name="t", config_name="x")
        _db.update_task(conn, tid, monitor_state_path=str(state_path))

    resp = client.get(f"/samples/step_5.png?task_id={tid}")
    assert resp.status_code == 200
    assert resp.content == b"old-layout"


# ---------------------------------------------------------------------------
# /
# ---------------------------------------------------------------------------

def test_root_serves_index_when_built(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    web_dist = isolated_paths["web_dist"]
    web_dist.mkdir(parents=True, exist_ok=True)
    (web_dist / "index.html").write_text("<!doctype html><title>anima</title>", encoding="utf-8")
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<title>anima</title>" in resp.text


def test_root_fallback_when_no_dist(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    assert not isolated_paths["web_dist"].exists()
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 200
    body = resp.json()
    assert "AnimaStudio" in body["message"]


def test_legacy_studio_path_redirects_to_root(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    resp = client.get("/studio/projects/1?tab=log", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/projects/1?tab=log"
    resp = client.get("/studio", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/"


# ---------------------------------------------------------------------------
# /api/system/restart (ADR 0002 / PR-A)
# ---------------------------------------------------------------------------

def test_uvicorn_run_bounds_graceful_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    from studio.api import main as main_mod

    captured: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: captured.update(kw))
    monkeypatch.setattr("sys.argv", ["anima-studio"])
    main_mod.main()
    timeout = captured.get("timeout_graceful_shutdown")
    assert isinstance(timeout, (int, float)) and timeout > 0


def test_cancelled_asgi_noise_filter_scope() -> None:
    import asyncio

    from studio.api.lifespan import _CancelledAsgiNoiseFilter

    f = _CancelledAsgiNoiseFilter()

    def record(msg: str, exc: BaseException | None) -> logging.LogRecord:
        return logging.LogRecord(
            name="uvicorn.error", level=logging.ERROR, pathname=__file__,
            lineno=1, msg=msg, args=(),
            exc_info=(type(exc), exc, None) if exc is not None else None,
        )

    assert not f.filter(
        record("Exception in ASGI application\n", asyncio.CancelledError())
    )
    assert f.filter(
        record("Exception in ASGI application\n", RuntimeError("boom"))
    )
    assert f.filter(record("Exception in ASGI application\n", None))
    assert f.filter(record("Cancel 2 running task(s)", asyncio.CancelledError()))


# ---------------------------------------------------------------------------
# /api/system/version (ADR 0002 / PR-B)
# ---------------------------------------------------------------------------

