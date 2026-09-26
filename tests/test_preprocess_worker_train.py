from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from studio.services.projects import projects
from studio.services.preprocess import manifest as pm
from studio.workers import preprocess_worker as worker


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    pdir = tmp_path / "projects" / "1-test"
    (pdir / "download").mkdir(parents=True)
    (pdir / "preprocess").mkdir(parents=True)
    train_root = pdir / "versions" / "v1" / "train"
    sub = train_root / "1_data"
    sub.mkdir(parents=True)
    return {
        "project": {"id": 1, "slug": "test"},
        "version": {"id": 10, "label": "v1"},
        "pdir": pdir,
        "sub": sub,
    }


def _silence(*_args, **_kwargs) -> None:
    pass


def _make_image(path: Path, size: tuple[int, int] = (200, 100), color=(255, 0, 0)) -> None:
    Image.new("RGB", size, color).save(path, format="PNG")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_swap_entry_removes_old_writes_new(env) -> None:
    sub = env["sub"]
    (sub / "X.jpg").write_bytes(b"jpg")
    (sub / "X.png").write_bytes(b"png" * 100)
    pm.train_add_processed(
        env["pdir"], "v1", "1_data/X.jpg", {"origin": "X.jpg"},
    )

    pm.train_swap_entry(
        env["pdir"], "v1",
        old_name="1_data/X.jpg",
        new_name="1_data/X.png",
        meta={"origin": "X.jpg"},
    )

    m = pm.train_load(env["pdir"], "v1")
    assert "1_data/X.jpg" not in m["images"]
    assert m["images"]["1_data/X.png"]["origin"] == "X.jpg"
    assert m["images"]["1_data/X.png"]["size"] == 300


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_crop_train_single_rect_overwrites_source(env) -> None:
    sub = env["sub"]
    _make_image(sub / "X.png", size=(200, 100))
    pm.train_add_processed(env["pdir"], "v1", "1_data/X.png", {"origin": "X.png"})

    rc = worker._run_crop_train(
        env["project"], env["version"],
        {"crops": {"1_data/X.png": [{"x": 0.2, "y": 0.0, "w": 0.5, "h": 1.0}]}},
        _silence, _silence,
    )
    assert rc == 0
    assert (sub / "X.png").is_file()
    with Image.open(sub / "X.png") as out:
        assert out.size == (100, 100)  # 0.5×200 = 100
    entry = pm.train_get_entry(env["pdir"], "v1", "1_data/X.png")
    assert entry is not None
    assert entry["origin"] == "X.png"


def test_crop_train_single_rect_replaces_jpg_without_doubling(env) -> None:
    sub = env["sub"]
    _make_image(sub / "X.jpg", size=(200, 100))
    (sub / "X.txt").write_text("existing caption", encoding="utf-8")

    rc = worker._run_crop_train(
        env["project"], env["version"],
        {"crops": {"1_data/X.jpg": [{"x": 0.2, "y": 0.0, "w": 0.5, "h": 1.0}]}},
        _silence, _silence,
    )
    assert rc == 0
    assert not (sub / "X.jpg").exists()
    assert (sub / "X.png").is_file()
    # Same stem, so the sidecar still belongs to X.png.
    assert (sub / "X.txt").read_text(encoding="utf-8") == "existing caption"
    assert sorted(p.name for p in sub.iterdir() if p.suffix.lower() in {".jpg", ".png"}) == ["X.png"]


# ---------------------------------------------------------------------------
# _run_crop_train: N>1 fan-out
# ---------------------------------------------------------------------------


def test_crop_train_fan_out_writes_multiple(env) -> None:
    sub = env["sub"]
    _make_image(sub / "Y.png", size=(200, 100))
    pm.train_add_processed(env["pdir"], "v1", "1_data/Y.png", {"origin": "Y.png"})

    rc = worker._run_crop_train(
        env["project"], env["version"],
        {"crops": {"1_data/Y.png": [
            {"x": 0.0, "y": 0.0, "w": 0.4, "h": 1.0},
            {"x": 0.5, "y": 0.0, "w": 0.4, "h": 1.0},
        ]}},
        _silence, _silence,
    )
    assert rc == 0
    assert (sub / "Y_c0.png").is_file()
    assert (sub / "Y_c1.png").is_file()
    assert not (sub / "Y.png").is_file()
    m = pm.train_load(env["pdir"], "v1")
    assert "1_data/Y.png" not in m["images"]
    assert "1_data/Y_c0.png" in m["images"]
    assert "1_data/Y_c1.png" in m["images"]
    assert m["images"]["1_data/Y_c0.png"]["origin"] == "Y.png"
    assert m["images"]["1_data/Y_c1.png"]["origin"] == "Y.png"


def test_crop_train_fan_out_removes_source_sidecars(env) -> None:
    sub = env["sub"]
    _make_image(sub / "Z.jpg", size=(200, 100))
    (sub / "Z.txt").write_text("old caption", encoding="utf-8")

    rc = worker._run_crop_train(
        env["project"], env["version"],
        {"crops": {"1_data/Z.jpg": [
            {"x": 0.0, "y": 0.0, "w": 0.4, "h": 1.0},
            {"x": 0.5, "y": 0.0, "w": 0.4, "h": 1.0},
        ]}},
        _silence, _silence,
    )
    assert rc == 0
    assert not (sub / "Z.jpg").exists()
    assert not (sub / "Z.txt").exists()
    assert (sub / "Z_c0.png").is_file()
    assert (sub / "Z_c1.png").is_file()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_crop_train_skips_when_source_missing(env) -> None:
    rc = worker._run_crop_train(
        env["project"], env["version"],
        {"crops": {"1_data/ghost.png": [{"x": 0, "y": 0, "w": 0.5, "h": 0.5}]}},
        _silence, _silence,
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_crop_train_rejects_invalid_rel_name(env) -> None:
    rc = worker._run_crop_train(
        env["project"], env["version"],
        {"crops": {"../escape/X.png": [{"x": 0, "y": 0, "w": 0.5, "h": 0.5}]}},
        _silence, _silence,
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_crop_train_origin_inherited_from_existing_entry(env) -> None:
    sub = env["sub"]
    _make_image(sub / "X_c0.png", size=(100, 100))
    pm.train_add_processed(
        env["pdir"], "v1", "1_data/X_c0.png", {"origin": "X.jpg"},
    )

    worker._run_crop_train(
        env["project"], env["version"],
        {"crops": {"1_data/X_c0.png": [
            {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0},
            {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0},
        ]}},
        _silence, _silence,
    )
    m = pm.train_load(env["pdir"], "v1")
    assert m["images"]["1_data/X_c0_c0.png"]["origin"] == "X.jpg"
    assert m["images"]["1_data/X_c0_c1.png"]["origin"] == "X.jpg"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_upscale_train_in_place_overwrites_jpg(env, monkeypatch) -> None:
    sub = env["sub"]
    (sub / "X.jpg").write_bytes(b"raw")
    pm.train_add_processed(env["pdir"], "v1", "1_data/X.jpg", {"origin": "X.jpg"})

    called: dict = {}

    def fake_upscale(src, dst, **kwargs):
        called["src"] = src
        called["dst"] = dst
        called["save_kwargs"] = kwargs.get("save_kwargs")
        Image.new("RGB", (400, 200), (0, 255, 0)).save(dst, format="JPEG", quality=95)
        return {
            "model": "fake", "scale": 4, "action": "upscale",
            "src_size": [200, 100], "dst_size": [400, 200],
        }

    monkeypatch.setattr(worker.upscaler, "upscale_file", fake_upscale)
    monkeypatch.setattr(worker.upscaler, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(worker.upscaler, "resolve_device", lambda d: type("Dev", (), {"type": "cpu"})())
    monkeypatch.setattr(worker.upscaler, "resolve_dtype", lambda *a, **k: "float32")
    fake_model = env["pdir"] / "fake-model.pth"
    fake_model.write_bytes(b"weight")
    monkeypatch.setattr(
        worker.model_downloader, "upscaler_target", lambda label: fake_model,
    )

    rc = worker._run_upscale_train(
        env["project"], env["version"], {"mode": "all"}, _silence, _silence,
    )
    assert rc == 0
    assert called["src"] == sub / "X.jpg"
    assert called["dst"] == sub / "X.jpg"
    assert called["save_kwargs"]["format"] == "JPEG"
    assert called["save_kwargs"]["quality"] == 95
    assert (sub / "X.jpg").is_file()
    assert not (sub / "X.png").exists()
    m = pm.train_load(env["pdir"], "v1")
    assert set(m["images"].keys()) == {"1_data/X.jpg"}
    assert m["images"]["1_data/X.jpg"]["origin"] == "X.jpg"
    assert m["images"]["1_data/X.jpg"]["processed"] is True


def test_upscale_train_in_place_overwrites_png(env, monkeypatch) -> None:
    sub = env["sub"]
    (sub / "X.png").write_bytes(b"old")
    pm.train_add_processed(env["pdir"], "v1", "1_data/X.png", {"origin": "X.png"})

    called: dict = {}

    def fake_upscale(src, dst, **kwargs):
        called["save_kwargs"] = kwargs.get("save_kwargs")
        Image.new("RGB", (400, 200), (0, 255, 0)).save(dst, format="PNG")
        return {"model": "fake", "scale": 4, "action": "upscale"}

    monkeypatch.setattr(worker.upscaler, "upscale_file", fake_upscale)
    monkeypatch.setattr(worker.upscaler, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(worker.upscaler, "resolve_device", lambda d: type("Dev", (), {"type": "cpu"})())
    monkeypatch.setattr(worker.upscaler, "resolve_dtype", lambda *a, **k: "float32")
    fake_model = env["pdir"] / "fake.pth"
    fake_model.write_bytes(b"w")
    monkeypatch.setattr(worker.model_downloader, "upscaler_target", lambda label: fake_model)

    rc = worker._run_upscale_train(
        env["project"], env["version"], {"mode": "all"}, _silence, _silence,
    )
    assert rc == 0
    assert called["save_kwargs"]["format"] == "PNG"
    m = pm.train_load(env["pdir"], "v1")
    assert set(m["images"].keys()) == {"1_data/X.png"}
    assert m["images"]["1_data/X.png"]["processed"] is True
    assert (sub / "X.png").is_file()
