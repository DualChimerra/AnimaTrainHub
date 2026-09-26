from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from studio.services.inference import upscaler


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class BicubicScaleModel(nn.Module):

    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


class StubDescriptor:

    def __init__(self, scale: int = 4):
        self.model = BicubicScaleModel(scale=scale)
        self.scale = scale
        self.input_channels = 3
        self.output_channels = 3


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch) -> StubDescriptor:
    stub = StubDescriptor(scale=4)

    def fake_load(model_path: Path, *, device=None, dtype=None):
        if device is not None:
            stub.model.to(device)
        if dtype is not None:
            stub.model.to(dtype)
        return stub

    monkeypatch.setattr(upscaler, "load_model", fake_load)
    return stub


@pytest.fixture(autouse=True)
def _clear_cache():
    upscaler.clear_cache()
    yield
    upscaler.clear_cache()


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_cpu_explicit() -> None:
    assert upscaler.resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert upscaler.resolve_device("auto").type == "cpu"


def test_resolve_device_cuda_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert upscaler.resolve_device("cuda").type == "cpu"


# ---------------------------------------------------------------------------
# resolve_dtype
# ---------------------------------------------------------------------------


def test_resolve_dtype_explicit() -> None:
    cpu = torch.device("cpu")
    assert upscaler.resolve_dtype("fp32", cpu) == torch.float32
    assert upscaler.resolve_dtype("fp16", cpu) == torch.float16
    assert upscaler.resolve_dtype("bf16", cpu) == torch.bfloat16


def test_resolve_dtype_auto_cpu() -> None:
    assert upscaler.resolve_dtype("auto", torch.device("cpu")) == torch.float32


def test_resolve_dtype_auto_cuda() -> None:
    assert upscaler.resolve_dtype("auto", torch.device("cuda")) == torch.float16


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_tiled_inference_matches_full_forward() -> None:
    model = BicubicScaleModel(scale=4)
    torch.manual_seed(0)
    img = torch.rand(1, 3, 64, 64)

    with torch.inference_mode():
        full = model(img)

    tiled = upscaler.tiled_inference(model, img, scale=4, tile_size=24, tile_pad=8)

    assert full.shape == tiled.shape == (1, 3, 256, 256)
    assert torch.allclose(full, tiled, atol=1e-4), (
        f"max diff = {(full - tiled).abs().max().item()}"
    )


def test_tiled_inference_no_tile_path() -> None:
    model = BicubicScaleModel(scale=2)
    img = torch.rand(1, 3, 32, 32)
    out = upscaler.tiled_inference(model, img, scale=2, tile_size=0)
    with torch.inference_mode():
        ref = model(img)
    assert torch.equal(out, ref)


def test_tiled_inference_handles_non_divisible_size() -> None:
    model = BicubicScaleModel(scale=4)
    img = torch.rand(1, 3, 70, 50)
    out = upscaler.tiled_inference(model, img, scale=4, tile_size=32, tile_pad=4)
    assert out.shape == (1, 3, 280, 200)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _write_test_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    import numpy as np

    w, h = size
    grad = np.linspace(0, 255, w, dtype=np.uint8)
    arr = np.tile(grad[None, :], (h, 1))
    rgb = np.stack([arr, arr // 2, 255 - arr], axis=-1)
    Image.fromarray(rgb).save(path, format="PNG")


def test_upscale_file_writes_png_and_returns_meta(
    tmp_path: Path, stub_model: StubDescriptor
) -> None:
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    _write_test_image(src, size=(32, 32))

    logs: list[str] = []
    meta = upscaler.upscale_file(
        src,
        dst,
        model_path=tmp_path / "dummy.pth",
        label="4x-AnimeSharp",
        tile_size=16,
        tile_pad=4,
        device="cpu",
        on_log=logs.append,
    )

    assert dst.exists()
    with Image.open(dst) as out:
        assert out.size == (128, 128)
        assert out.mode == "RGB"

    sidecar = dst.with_suffix(dst.suffix + ".preprocess.json")
    assert not sidecar.exists()

    assert meta["source"] == "in.png"
    assert meta["model"] == "4x-AnimeSharp"
    assert meta["scale"] == 4
    assert meta["src_size"] == [32, 32]
    assert meta["dst_size"] == [128, 128]
    assert meta["device"] == "cpu"
    assert "elapsed_seconds" in meta
    assert "mtime" in meta

    assert any("in.png" in line for line in logs)


def test_upscale_file_smart_skips_model_when_already_big(
    tmp_path: Path, stub_model: StubDescriptor
) -> None:
    src = tmp_path / "big.png"
    dst = tmp_path / "out.png"
    _write_test_image(src, size=(1024, 1024))

    meta = upscaler.upscale_file(
        src,
        dst,
        model_path=tmp_path / "dummy.pth",
        device="cpu",
        target_area=768 * 768,
    )
    assert meta["action"] == "resize"
    assert meta["scale"] == 1
    out_w, out_h = meta["dst_size"]
    assert abs(out_w * out_h - 768 * 768) <= 768


def test_upscale_file_smart_upscales_when_too_small(
    tmp_path: Path, stub_model: StubDescriptor
) -> None:
    src = tmp_path / "small.png"
    dst = tmp_path / "out.png"
    _write_test_image(src, size=(256, 256))

    meta = upscaler.upscale_file(
        src,
        dst,
        model_path=tmp_path / "dummy.pth",
        device="cpu",
        tile_size=128,
        target_area=1024 * 1024,
    )
    assert meta["action"] == "upscale+resize"
    assert meta["scale"] == 4
    out_w, out_h = meta["dst_size"]
    assert abs(out_w * out_h - 1024 * 1024) <= 1024


def test_upscale_file_target_area_none_keeps_old_behavior(
    tmp_path: Path, stub_model: StubDescriptor
) -> None:
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    _write_test_image(src, size=(64, 64))

    meta = upscaler.upscale_file(
        src,
        dst,
        model_path=tmp_path / "dummy.pth",
        device="cpu",
        tile_size=32,
        target_area=None,
    )
    assert meta["action"] == "upscale"
    assert meta["dst_size"] == [256, 256]


def test_resize_to_area_preserves_aspect() -> None:
    img = Image.new("RGB", (1920, 3360), color=(120, 60, 30))
    out = upscaler.resize_to_area(img, 1024 * 1024)
    w, h = out.size
    assert abs((w / h) - (1920 / 3360)) < 1e-3
    assert abs(w * h - 1024 * 1024) / (1024 * 1024) < 0.01


def test_upscale_file_prewarms_thumbnails(
    tmp_path: Path, stub_model: StubDescriptor, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services.dataset import thumb_cache

    monkeypatch.setattr(thumb_cache, "THUMB_CACHE_DIR", tmp_path / "thumbs")

    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    _write_test_image(src, size=(32, 32))

    upscaler.upscale_file(
        src,
        dst,
        model_path=tmp_path / "dummy.pth",
        device="cpu",
        tile_size=16,
        prewarm_thumb_sizes=[64, 128],
    )

    for sz in (64, 128):
        cached = thumb_cache.get_or_make_thumb(dst, sz)
        assert cached.exists()
        assert cached.parent == thumb_cache.THUMB_CACHE_DIR
        assert cached.read_bytes()[:2] == b"\xff\xd8"


def test_upscale_file_converts_rgba(
    tmp_path: Path, stub_model: StubDescriptor
) -> None:
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    rgba = Image.new("RGBA", (16, 16), (200, 100, 50, 128))
    rgba.save(src, format="PNG")

    upscaler.upscale_file(
        src,
        dst,
        model_path=tmp_path / "dummy.pth",
        device="cpu",
        tile_size=8,
    )
    with Image.open(dst) as out:
        assert out.mode == "RGB"
        assert out.size == (64, 64)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_load_model_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        upscaler.load_model(tmp_path / "nope.pth")


def test_load_model_caches_by_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "fake.pth"
    model_path.write_bytes(b"x")

    call_count = {"n": 0}

    class FakeLoader:
        def load_from_file(self, _path: str) -> StubDescriptor:
            call_count["n"] += 1
            return StubDescriptor(scale=4)

    import sys
    import types

    fake_spandrel = types.ModuleType("spandrel")
    fake_spandrel.ModelLoader = FakeLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spandrel", fake_spandrel)

    d1 = upscaler.load_model(model_path)
    d2 = upscaler.load_model(model_path)
    assert d1 is d2
    assert call_count["n"] == 1


def test_load_model_casts_to_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "fake.pth"
    model_path.write_bytes(b"x")

    class StubWithParams(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.interpolate(self.conv(x), scale_factor=4, mode="bicubic", align_corners=False)

    class FakeDescriptor:
        def __init__(self):
            self.model = StubWithParams()
            self.scale = 4
            self.input_channels = 3
            self.output_channels = 3

    class FakeLoader:
        def load_from_file(self, _path: str) -> FakeDescriptor:
            return FakeDescriptor()

    import sys
    import types

    fake_spandrel = types.ModuleType("spandrel")
    fake_spandrel.ModelLoader = FakeLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spandrel", fake_spandrel)

    d = upscaler.load_model(
        model_path, device=torch.device("cpu"), dtype=torch.float16
    )
    p = next(d.model.parameters())
    assert p.dtype == torch.float16


def test_load_model_without_spandrel_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "fake.pth"
    model_path.write_bytes(b"x")

    import sys
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "spandrel" or name.startswith("spandrel."):
            raise ImportError("no spandrel")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "spandrel", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="spandrel"):
        upscaler.load_model(model_path)
