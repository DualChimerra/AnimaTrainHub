from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin


def _png_bytes(color: tuple[int, int, int] = (0, 0, 0), size: tuple[int, int] = (8, 8)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _params(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "mode": "single",
        "prompts": ["1girl, anime"],
        "negative_prompt": "blurry",
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "cfg_scale": 7.0,
        "count": 1,
        "seed": 7,
        "loras": [],
        "xy_draft": None,
        "dataset_pick": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    from studio.api.routers import generate as _gen

    test_dir = tmp_path / "test"
    monkeypatch.setattr(_gen, "TEST_IMAGES_DIR", test_dir)

    class _FakeGenCfg:
        save_test_images = True

    class _FakeSecrets:
        generate = _FakeGenCfg()

    monkeypatch.setattr(_gen.secrets, "load", lambda: _FakeSecrets())

    from studio.api.exception_handlers import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(_gen.router)
    return TestClient(app, raise_server_exceptions=False), test_dir


def _open_png_text(path: Path) -> dict[str, str]:
    with Image.open(path) as img:
        img.load()
        return dict(img.text)


def _post_xy_save(
    tc: TestClient,
    composite_params: dict,
    cells: list[tuple[int, int, dict]],
    task_id: int | None = None,
):
    files: list[tuple[str, tuple[str, bytes, str]]] = [
        ("image", ("xy plot.png", _png_bytes(), "image/png")),
    ]
    for xi, yi, _cp in cells:
        files.append(("cells", (f"cell x{xi} y{yi}.png", _png_bytes(), "image/png")))
    manifest = [{"xi": xi, "yi": yi, "params": cp} for (xi, yi, cp) in cells]
    data = {
        "mode": "xy",
        "params": json.dumps(composite_params),
        "cells_manifest": json.dumps(manifest),
    }
    if task_id is not None:
        data["task_id"] = str(task_id)
    return tc.post("/api/generate/save", data=data, files=files)


def test_save_writes_anima_params_text_block(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params(seed=42))},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    saved = Path(r.json()["path"])
    assert saved.exists()

    text = _open_png_text(saved)
    assert "anima_params" in text
    parsed = json.loads(text["anima_params"])
    assert parsed["seed"] == 42
    assert parsed["schema_version"] == 2
    assert parsed["mode"] == "single"
    assert "created_at" in parsed


def test_save_writes_a1111_parameters_text_block(client) -> None:
    tc, _ = client
    p = _params(
        seed=42, steps=20, cfg_scale=7.0, width=1024, height=1024,
        prompts=["1girl, anime"], negative_prompt="blurry",
        loras=[{"name": "my-lora.safetensors", "scale": 0.8,
                "project_id": 12, "version_id": 34}],
    )
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(p)},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    saved = Path(r.json()["path"])

    text = _open_png_text(saved)
    assert "parameters" in text
    a1111 = text["parameters"]
    first_line = a1111.split("\n", 1)[0]
    assert "1girl, anime" in first_line
    assert "<lora:my-lora:0.8>" in first_line
    assert "Negative prompt: blurry" in a1111
    assert "Steps: 20" in a1111
    assert "CFG scale: 7.0" in a1111
    assert "Seed: 42" in a1111
    assert "Size: 1024x1024" in a1111


def test_save_does_not_write_sidecar(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    saved = Path(r.json()["path"])
    assert not saved.with_suffix(".json").exists()
    assert "sidecar" not in r.json()


def test_save_without_params_skips_metadata(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single"},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    saved = Path(r.json()["path"])
    text = _open_png_text(saved)
    assert "anima_params" not in text
    assert "parameters" not in text


def test_save_rejects_invalid_params_json(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": "not-json{"},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "generate.params_invalid"


def test_save_rejects_non_object_params(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": "[1, 2, 3]"},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "generate.params_invalid"


def test_disk_history_lists_entries_from_png_metadata(client) -> None:
    tc, _ = client
    for seed in (1, 2, 3):
        tc.post(
            "/api/generate/save",
            data={"mode": "single", "params": json.dumps(_params(seed=seed))},
            files={"image": (f"{seed}.png", _png_bytes(), "image/png")},
        )

    r = tc.get("/api/generate/disk/history")
    assert r.status_code == 200
    entries = r.json()["entries"]
    assert len(entries) == 3
    for a, b in zip(entries, entries[1:]):
        assert a["created_at"] >= b["created_at"]
    seeds = sorted(e["params"]["seed"] for e in entries)
    assert seeds == [1, 2, 3]
    assert all(e["image_url"].startswith("/api/generate/disk/image/") for e in entries)
    assert len({e["id"] for e in entries}) == 3


def test_disk_history_skips_png_without_anima_params(client) -> None:
    tc, test_dir = client
    single_dir = test_dir / "2026-01-01" / "single"
    single_dir.mkdir(parents=True)
    (single_dir / "image_0.png").write_bytes(_png_bytes())

    r = tc.get("/api/generate/disk/history")
    assert r.status_code == 200
    assert r.json()["entries"] == []


def test_disk_history_skips_png_with_only_a1111_block(client) -> None:
    tc, test_dir = client
    single_dir = test_dir / "2026-01-01" / "single"
    single_dir.mkdir(parents=True)
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "fake prompt\nSteps: 1, Sampler: x")
    img = Image.new("RGB", (8, 8))
    img.save(single_dir / "image_0.png", format="PNG", pnginfo=info)

    r = tc.get("/api/generate/disk/history")
    assert r.status_code == 200
    assert r.json()["entries"] == []


def test_disk_history_includes_xy_mode_entries(client) -> None:
    tc, _ = client
    _post_xy_save(
        tc,
        composite_params=_params(mode="xy", seed=99, xy_draft={
            "x": {"axis": "steps", "raw": "10, 20", "loraIndex": None},
            "y": None,
        }),
        cells=[
            (0, 0, _params(mode="single", seed=99, steps=10)),
            (1, 0, _params(mode="single", seed=99, steps=20)),
        ],
    )
    entries = tc.get("/api/generate/disk/history").json()["entries"]
    assert len(entries) == 1
    assert entries[0]["mode"] == "xy"


def test_disk_image_serves_saved_file(client) -> None:
    tc, _ = client
    tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params(seed=99))},
        files={"image": ("a.png", _png_bytes((255, 0, 0)), "image/png")},
    )
    listing = tc.get("/api/generate/disk/history").json()["entries"]
    assert len(listing) == 1
    url = listing[0]["image_url"]

    r = tc.get(url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 0
    assert "max-age" in r.headers.get("cache-control", "")


def test_disk_image_validates_inputs(client) -> None:
    tc, _ = client
    assert tc.get("/api/generate/disk/image/not-a-date/single/image_0.png").status_code == 400
    assert tc.get("/api/generate/disk/image/2026-01-01/bad/image_0.png").status_code == 400
    assert tc.get("/api/generate/disk/image/2026-01-01/single/image_0.jpg").status_code == 400
    assert tc.get("/api/generate/disk/image/2026-01-01/single/image_0.png").status_code == 404


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_save_uses_v2_filename(client) -> None:
    tc, _ = client
    r1 = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    r2 = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r1.json()["filename"] == "single image 1.png"
    assert r2.json()["filename"] == "single image 2.png"


def test_save_sets_task_id_from_form_field(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params()), "task_id": "42"},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    saved = Path(r.json()["path"])
    text = _open_png_text(saved)
    parsed = json.loads(text["anima_params"])
    assert parsed["task_id"] == 42


def test_save_atomic_write_no_tmp_remains(client, tmp_path) -> None:
    tc, test_dir = client
    tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    leftover = list((test_dir).rglob("*.tmp*"))
    assert leftover == [], f"leftover atomic-write tmp file(s): {leftover}"


def test_save_xy_skips_a1111_block(client) -> None:
    tc, _ = client
    r = _post_xy_save(
        tc,
        composite_params=_params(mode="xy", xy_draft={
            "x": {"axis": "steps", "raw": "20", "loraIndex": None},
            "y": None,
        }),
        cells=[(0, 0, _params(mode="single", steps=20))],
    )
    assert r.status_code == 200, r.text
    composite = Path(r.json()["composite"])
    composite_text = _open_png_text(composite)
    assert "anima_params" in composite_text
    assert "parameters" not in composite_text
    cell = Path(r.json()["cells"][0])
    cell_text = _open_png_text(cell)
    assert "anima_params" in cell_text
    assert "parameters" in cell_text


def test_disk_history_migrates_v1_to_v2(client, tmp_path) -> None:
    tc, test_dir = client
    v1_params = {
        "schema_version": 1,
        "mode": "single",
        "prompts": ["test"],
        "negative_prompt": "",
        "width": 8, "height": 8, "steps": 5, "cfg_scale": 4.0, "count": 1, "seed": 1,
        "lora_configs": [
            {"path": "G:/some/abs/path/my-lora.safetensors", "scale": 0.8,
             "project_id": 12, "version_id": 34}
        ],
    }
    from studio.api.routers import generate as _gen
    raw = _gen._inject_png_metadata(_png_bytes(), v1_params, mode="single")
    single_dir = test_dir / "2026-06-08" / "single"
    single_dir.mkdir(parents=True)
    (single_dir / "image_0.png").write_bytes(raw)

    r = tc.get("/api/generate/disk/history")
    entries = r.json()["entries"]
    assert len(entries) == 1
    params = entries[0]["params"]
    assert params["schema_version"] == 2
    assert "lora_configs" not in params
    assert params["loras"] == [
        {"name": "my-lora.safetensors", "scale": 0.8,
         "project_id": 12, "version_id": 34}
    ]


def test_disk_history_skips_tmp_files(client, tmp_path) -> None:
    tc, test_dir = client
    single_dir = test_dir / "2026-06-08" / "single"
    single_dir.mkdir(parents=True)
    (single_dir / "single image 1.tmp.png").write_bytes(_png_bytes())
    assert tc.get("/api/generate/disk/history").json()["entries"] == []


def test_disk_thumb_returns_png_with_etag(client) -> None:
    tc, _ = client
    tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    entry = tc.get("/api/generate/disk/history").json()["entries"][0]
    thumb_url = entry["thumb_url"]
    assert "/disk/thumb/" in thumb_url
    r = tc.get(thumb_url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers.get("etag")
    assert "max-age" in r.headers.get("cache-control", "")


def test_disk_thumb_validates_inputs(client) -> None:
    tc, _ = client
    assert tc.get("/api/generate/disk/thumb/not-a-date/single/foo.png").status_code == 400
    assert tc.get("/api/generate/disk/thumb/2026-06-08/bad/foo.png").status_code == 400
    assert tc.get("/api/generate/disk/thumb/2026-06-08/single/foo.png").status_code == 404


def test_disk_delete_removes_file(client, tmp_path) -> None:
    tc, _ = client
    tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    entry = tc.get("/api/generate/disk/history").json()["entries"][0]
    path = Path(entry["path"])
    assert path.is_file()
    encoded = entry["image_url"].rsplit("/", 1)[-1]
    delete_url = f"/api/generate/disk/{entry['date']}/{entry['mode']}/{encoded}"
    r = tc.delete(delete_url)
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["noop"] is False
    assert not path.is_file()
    r2 = tc.delete(delete_url)
    assert r2.status_code == 200
    assert r2.json()["noop"] is True


def test_disk_path_traversal_attack_blocked(client) -> None:
    tc, _ = client
    for evil in [
        "%2E%2E%2Fsecret.png",        # ../secret.png URL encoded
        "..%5Csecret.png",            # ..\secret.png URL encoded (Windows)
    ]:
        r = tc.get(f"/api/generate/disk/image/2026-06-08/single/{evil}")
        assert r.status_code in (400, 404), f"path={evil} got {r.status_code}"


def test_save_disabled_returns_403(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.api.routers import generate as _gen

    monkeypatch.setattr(_gen, "TEST_IMAGES_DIR", tmp_path / "test")

    class _FakeGenCfg:
        save_test_images = False

    class _FakeSecrets:
        generate = _FakeGenCfg()

    monkeypatch.setattr(_gen.secrets, "load", lambda: _FakeSecrets())

    from studio.api.exception_handlers import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(_gen.router)
    tc = TestClient(app, raise_server_exceptions=False)

    r = tc.post(
        "/api/generate/save",
        data={"mode": "single", "params": json.dumps(_params())},
        files={"image": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "generate.save_disabled"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _xy_snapshot(*, x_raw: str = "10, 20, 30", y_raw: str | None = "3, 5") -> dict:
    return _params(mode="xy", xy_draft={
        "x": {"axis": "steps", "raw": x_raw, "loraIndex": None},
        "y": {"axis": "cfg_scale", "raw": y_raw, "loraIndex": None} if y_raw else None,
    })


def test_save_xy_creates_folder_with_composite_and_cells(client) -> None:
    tc, test_dir = client
    r = _post_xy_save(
        tc,
        composite_params=_xy_snapshot(x_raw="10, 20", y_raw="3"),
        cells=[
            (0, 0, _params(mode="single", steps=10, cfg_scale=3.0)),
            (1, 0, _params(mode="single", steps=20, cfg_scale=3.0)),
        ],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    folder = Path(body["folder"])
    assert folder.is_dir()
    assert folder.name == "xy plot 1"
    assert (folder / "xy plot.png").is_file()
    assert (folder / "cell x0 y0.png").is_file()
    assert (folder / "cell x1 y0.png").is_file()
    assert not list(folder.parent.glob(".xy plot *.tmp"))


def test_save_xy_cell_carries_single_snapshot_and_xy_origin(client) -> None:
    tc, _ = client
    r = _post_xy_save(
        tc,
        composite_params=_xy_snapshot(x_raw="10, 20", y_raw=None),
        cells=[
            (0, 0, _params(mode="single", steps=10)),
            (1, 0, {**_params(mode="single", steps=20), "xy_origin": {
                "xi": 1, "yi": 0, "xv": "20", "yv": None,
                "x_axis": "steps", "y_axis": None,
            }}),
        ],
    )
    assert r.status_code == 200, r.text
    cell1 = Path(r.json()["cells"][1])
    text = _open_png_text(cell1)
    assert "anima_params" in text
    assert "parameters" in text
    parsed = json.loads(text["anima_params"])
    assert parsed["mode"] == "single"
    assert parsed["steps"] == 20
    assert parsed["xy_origin"]["xi"] == 1
    assert parsed["xy_origin"]["x_axis"] == "steps"


def test_save_xy_rejects_cell_xy_collision(client) -> None:
    tc, _ = client
    r = _post_xy_save(
        tc,
        composite_params=_xy_snapshot(),
        cells=[
            (0, 0, _params(mode="single")),
            (0, 0, _params(mode="single")),
        ],
    )
    assert r.status_code == 400
    assert "duplicate" in r.text


def test_save_xy_rejects_cell_count_mismatch(client) -> None:
    tc, _ = client
    r = tc.post(
        "/api/generate/save",
        data={
            "mode": "xy",
            "params": json.dumps(_xy_snapshot()),
            "cells_manifest": json.dumps([
                {"xi": 0, "yi": 0, "params": _params(mode="single")},
                {"xi": 1, "yi": 0, "params": _params(mode="single")},
            ]),
        },
        files=[
            ("image", ("xy plot.png", _png_bytes(), "image/png")),
            ("cells", ("cell x0 y0.png", _png_bytes(), "image/png")),
        ],
    )
    assert r.status_code == 400
    assert "length" in r.text


def test_save_xy_folder_index_skips_legacy_files(client) -> None:
    tc, test_dir = client
    xy_dir = test_dir / "2026-06-08" / "xy"
    xy_dir.mkdir(parents=True)
    (xy_dir / "xy plot 7.png").write_bytes(_png_bytes())

    from datetime import date as _date
    from studio.api.routers import generate as _gen
    class _FixedDate(_date):
        @classmethod
        def today(cls):
            return cls(2026, 6, 8)
    import datetime
    orig_date = _gen.date
    try:
        _gen.date = _FixedDate
        r = _post_xy_save(
            tc,
            composite_params=_xy_snapshot(x_raw="10", y_raw=None),
            cells=[(0, 0, _params(mode="single", steps=10))],
        )
    finally:
        _gen.date = orig_date
    assert r.status_code == 200
    assert Path(r.json()["folder"]).name == "xy plot 8"


def test_disk_history_xy_folder_entry_has_xy_meta(client) -> None:
    tc, _ = client
    _post_xy_save(
        tc,
        composite_params=_xy_snapshot(x_raw="10, 20", y_raw="3"),
        cells=[
            (0, 0, _params(mode="single", steps=10, cfg_scale=3.0)),
            (1, 0, _params(mode="single", steps=20, cfg_scale=3.0)),
        ],
    )
    entries = tc.get("/api/generate/disk/history").json()["entries"]
    assert len(entries) == 1
    e = entries[0]
    assert e["mode"] == "xy"
    assert e["folder"] == "xy plot 1"
    assert e["image_url"].endswith("xy%20plot.png")
    meta = e["xy_meta"]
    assert meta["x_axis"] == "steps"
    assert meta["y_axis"] == "cfg_scale"
    assert meta["x_values"] == ["10", "20"]
    assert meta["y_values"] == ["3"]
    assert len(meta["samples"]) == 2
    samples_by_pos = {(s["xy"]["xi"], s["xy"]["yi"]): s for s in meta["samples"]}
    assert samples_by_pos[(0, 0)]["xy"]["xv"] == "10"
    assert samples_by_pos[(1, 0)]["xy"]["xv"] == "20"
    assert "%20" in samples_by_pos[(0, 0)]["image_url"]


def test_disk_history_skips_legacy_xy_flat_files(client) -> None:
    tc, test_dir = client
    xy_dir = test_dir / "2026-06-08" / "xy"
    xy_dir.mkdir(parents=True)
    info = PngImagePlugin.PngInfo()
    info.add_text("anima_params", json.dumps(_params(mode="xy")))
    Image.new("RGB", (8, 8)).save(xy_dir / "xy plot 99.png", format="PNG", pnginfo=info)

    entries = tc.get("/api/generate/disk/history").json()["entries"]
    assert entries == []


def test_disk_history_skips_xy_folder_without_composite(client) -> None:
    tc, test_dir = client
    xy_dir = test_dir / "2026-06-08" / "xy" / "xy plot 1"
    xy_dir.mkdir(parents=True)
    info = PngImagePlugin.PngInfo()
    info.add_text("anima_params", json.dumps(_params(mode="single")))
    Image.new("RGB", (8, 8)).save(xy_dir / "cell x0 y0.png", format="PNG", pnginfo=info)

    entries = tc.get("/api/generate/disk/history").json()["entries"]
    assert entries == []


def test_disk_xy_folder_delete_removes_recursively(client) -> None:
    tc, _ = client
    r = _post_xy_save(
        tc,
        composite_params=_xy_snapshot(x_raw="10", y_raw=None),
        cells=[(0, 0, _params(mode="single", steps=10))],
    )
    folder = Path(r.json()["folder"])
    assert folder.is_dir()
    entry = tc.get("/api/generate/disk/history").json()["entries"][0]
    url = f"/api/generate/disk/{entry['date']}/xy/{entry['folder'].replace(' ', '%20')}"
    delete_r = tc.delete(url)
    assert delete_r.status_code == 200
    assert delete_r.json()["noop"] is False
    assert not folder.exists()
    assert tc.delete(url).json()["noop"] is True


def test_disk_xy_image_route_resolves_subpath(client) -> None:
    """GET /api/generate/disk/image/<date>/xy/<folder>/<filename> → cell PNG bytes."""
    tc, _ = client
    r = _post_xy_save(
        tc,
        composite_params=_xy_snapshot(x_raw="10", y_raw=None),
        cells=[(0, 0, _params(mode="single", steps=10))],
    )
    entry = tc.get("/api/generate/disk/history").json()["entries"][0]
    cell_url = entry["xy_meta"]["samples"][0]["image_url"]
    cell_r = tc.get(cell_url)
    assert cell_r.status_code == 200
    assert cell_r.headers["content-type"] == "image/png"
    assert len(cell_r.content) > 0
    composite_r = tc.get(entry["image_url"])
    assert composite_r.status_code == 200
    thumb_r = tc.get(entry["thumb_url"])
    assert thumb_r.status_code == 200


def test_disk_xy_path_traversal_blocked(client) -> None:
    tc, _ = client
    r1 = tc.get("/api/generate/disk/image/2026-06-08/xy/not%20a%20folder/cell.png")
    assert r1.status_code == 400
    r2 = tc.get("/api/generate/disk/image/bad-date/xy/xy%20plot%201/x.png")
    assert r2.status_code == 400
    r3 = tc.get("/api/generate/disk/image/2026-06-08/xy/xy%20plot%201/..%2Fsecret.png")
    assert r3.status_code in (400, 404)
    r4 = tc.delete("/api/generate/disk/2026-06-08/xy/bad%20folder")
    assert r4.status_code == 400


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _write_png_with_anima(path: Path, params: dict, *, zip: bool) -> None:
    info = PngImagePlugin.PngInfo()
    info.add_text("anima_params", json.dumps(params, ensure_ascii=False), zip=zip)
    Image.new("RGB", (8, 8)).save(path, format="PNG", pnginfo=info)


@pytest.mark.parametrize("zip_flag", [True, False])
def test_read_png_anima_params_ascii_roundtrip(tmp_path: Path, zip_flag: bool) -> None:
    from studio.api.routers import generate as _gen
    p = tmp_path / "a.png"
    params = _params(seed=123, prompts=["1girl, anime"])
    _write_png_with_anima(p, params, zip=zip_flag)
    assert _gen._read_png_anima_params(p) == params


def test_read_png_anima_params_cjk_uses_itxt(tmp_path: Path) -> None:
    from studio.api.routers import generate as _gen
    p = tmp_path / "a.png"
    params = _params(prompts=["1girl, white hair, red eyes"], negative_prompt="blurry")
    _write_png_with_anima(p, params, zip=True)
    out = _gen._read_png_anima_params(p)
    assert out == params
    with Image.open(p) as img:
        img.load()
        assert json.loads(img.text["anima_params"]) == out


def test_read_png_anima_params_missing_returns_none(tmp_path: Path) -> None:
    from studio.api.routers import generate as _gen
    p = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(p, format="PNG")
    assert _gen._read_png_anima_params(p) is None
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "fake a1111 block")
    Image.new("RGB", (8, 8)).save(p, format="PNG", pnginfo=info)
    assert _gen._read_png_anima_params(p) is None


def test_read_png_anima_params_non_png_returns_none(tmp_path: Path) -> None:
    from studio.api.routers import generate as _gen
    p = tmp_path / "a.png"
    p.write_bytes(b"not a png file at all")
    assert _gen._read_png_anima_params(p) is None
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert _gen._read_png_anima_params(p) is None
