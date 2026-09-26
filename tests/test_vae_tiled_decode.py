from __future__ import annotations

import logging

import pytest
import torch

from training.vae import (
    VAEWrapper,
    _cosine_blend_mask,
    _tile_starts,
)


# ---------------------------------------------------------------------------
# _tile_starts
# ---------------------------------------------------------------------------


def test_tile_starts_size_le_tile_returns_zero_only() -> None:
    assert _tile_starts(64, 64, 48) == [0]
    assert _tile_starts(32, 64, 48) == [0]


def test_tile_starts_appends_boundary_when_stride_doesnt_cover() -> None:
    assert _tile_starts(128, 64, 48) == [0, 48, 64]


def test_tile_starts_exact_stride_no_duplicate_tail() -> None:
    assert _tile_starts(160, 64, 48) == [0, 48, 96]


# ---------------------------------------------------------------------------
# _cosine_blend_mask
# ---------------------------------------------------------------------------


def test_cosine_blend_mask_shape_and_dtype() -> None:
    m = _cosine_blend_mask(512, 512, fade=128, device="cpu")
    assert m.shape == (1, 1, 1, 512, 512)
    assert m.dtype == torch.float32


def test_cosine_blend_mask_center_one_edges_clamped() -> None:
    m = _cosine_blend_mask(512, 512, fade=128, device="cpu")[0, 0, 0]
    assert m[256, 256].item() == pytest.approx(1.0, abs=1e-5)
    assert m[0, 0].item() == pytest.approx(1e-6, abs=1e-7)
    assert m[-1, -1].item() == pytest.approx(1e-6, abs=1e-7)
    assert m[0, -1].item() == pytest.approx(1e-6, abs=1e-7)


def test_cosine_blend_mask_symmetric() -> None:
    m = _cosine_blend_mask(256, 256, fade=64, device="cpu")[0, 0, 0]
    assert torch.allclose(m, m.flip(0), atol=1e-6)
    assert torch.allclose(m, m.flip(1), atol=1e-6)


def test_cosine_blend_mask_zero_fade_all_ones() -> None:
    m = _cosine_blend_mask(64, 64, fade=0, device="cpu")
    assert torch.all(m == 1.0)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class _RecordingModel:

    def __init__(self, oom_on_full: bool = False, oom_on_full_enc: bool = False):
        self.calls: list[tuple[int, ...]] = []
        self.enc_calls: list[tuple[int, ...]] = []
        self.oom_on_full = oom_on_full
        self.oom_on_full_enc = oom_on_full_enc

    def encode(self, pixels: torch.Tensor, scale) -> torch.Tensor:
        self.enc_calls.append(tuple(pixels.shape))
        b, _c, t, H, W = pixels.shape
        if self.oom_on_full_enc and H >= 1024 and len(self.enc_calls) == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM on full encode")
        z = torch.nn.functional.avg_pool3d(pixels, kernel_size=(1, 8, 8))  # [b,3,t,H/8,W/8]
        if z.shape[1] < 16:
            z = torch.nn.functional.pad(z, (0, 0, 0, 0, 0, 0, 0, 16 - z.shape[1]))  # 3ch→16ch
        return z

    def decode(self, z: torch.Tensor, scale) -> torch.Tensor:
        self.calls.append(tuple(z.shape))
        b, _c, t, h, w = z.shape
        upsampled = torch.nn.functional.interpolate(
            z[:, :3].reshape(b * t, 3, h, w),
            scale_factor=8,
            mode="nearest",
        ).reshape(b, 3, t, h * 8, w * 8)
        if self.oom_on_full and h >= 128 and len(self.calls) == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM on full decode")
        return upsampled


def _make_wrapper(model) -> VAEWrapper:
    mean = torch.zeros(16)
    std = torch.ones(16)
    return VAEWrapper(model, mean, std)


def test_decode_full_path_calls_model_once() -> None:
    model = _RecordingModel(oom_on_full=False)
    wrapper = _make_wrapper(model)
    z = torch.randn(1, 16, 1, 32, 32)
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 256, 256)
    assert len(model.calls) == 1


# ---------------------------------------------------------------------------
# VAEWrapper.decode OOM fallback
# ---------------------------------------------------------------------------


def test_decode_oom_fallback_invokes_tiled_decode() -> None:
    model = _RecordingModel(oom_on_full=True)
    wrapper = _make_wrapper(model)
    z = torch.zeros(1, 16, 1, 128, 128)  # 1024×1024 reg
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 1024, 1024)
    assert len(model.calls) == 1 + 9
    assert model.calls[0] == (1, 16, 1, 128, 128)
    for shape in model.calls[1:]:
        assert shape == (1, 16, 1, 64, 64)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_tiled_decode_reconstructs_full_for_linear_decoder() -> None:
    model = _RecordingModel(oom_on_full=False)
    wrapper = _make_wrapper(model)
    torch.manual_seed(0)
    z = torch.randn(1, 16, 1, 128, 128)

    full = wrapper.decode(z)
    tiled = wrapper._tiled_decode(z)

    assert tiled.shape == full.shape
    assert torch.allclose(tiled, full, atol=1e-4)


def test_tiled_decode_handles_small_input_without_tiling() -> None:
    model = _RecordingModel(oom_on_full=False)
    wrapper = _make_wrapper(model)
    z = torch.randn(1, 16, 1, 32, 32)  # < tile=64
    out = wrapper._tiled_decode(z)
    assert out.shape == (1, 3, 1, 256, 256)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_tiling_on_always_tiles_no_full_call() -> None:
    model = _RecordingModel(oom_on_full=False)
    wrapper = VAEWrapper(model, torch.zeros(16), torch.ones(16), tiling="on")
    z = torch.zeros(1, 16, 1, 128, 128)
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 1024, 1024)
    assert len(model.calls) == 9
    assert all(s == (1, 16, 1, 64, 64) for s in model.calls)


def test_tiling_off_uses_whole_image_then_oom_net() -> None:
    model = _RecordingModel(oom_on_full=True)
    wrapper = VAEWrapper(model, torch.zeros(16), torch.ones(16), tiling="off")
    z = torch.zeros(1, 16, 1, 128, 128)
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 1024, 1024)
    assert model.calls[0] == (1, 16, 1, 128, 128)
    assert len(model.calls) == 1 + 9


def test_est_decode_peak_scales_with_pixels_and_dtype() -> None:
    model = _RecordingModel()
    wrapper = _make_wrapper(model)
    z32 = torch.zeros(1, 16, 1, 128, 128, dtype=torch.float32)
    est_fp32 = wrapper._est_decode_peak_bytes(z32)
    assert est_fp32 == pytest.approx(11000 * 1024 * 1024, rel=1e-6)
    z16 = torch.zeros(1, 16, 1, 128, 128, dtype=torch.bfloat16)
    assert wrapper._est_decode_peak_bytes(z16) == pytest.approx(est_fp32 / 2, rel=1e-6)


def test_should_auto_tile_threshold_matches_measured_cliff() -> None:
    GB = 1024 ** 3
    total = int(31.8 * GB)
    model = _RecordingModel()
    w = _make_wrapper(model)

    def est(res, dtype):
        h = res // 8
        return w._est_decode_peak_bytes(torch.zeros(1, 16, 1, h, h, dtype=dtype))

    light = int(2.2 * GB)
    assert not VAEWrapper._should_auto_tile(light, est(1024, torch.float32), total)
    assert VAEWrapper._should_auto_tile(light, est(1536, torch.float32), total)
    assert not VAEWrapper._should_auto_tile(light, est(1536, torch.bfloat16), total)
    heavy = int(13 * GB)
    assert VAEWrapper._should_auto_tile(heavy, est(1536, torch.bfloat16), total)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_encode_full_path_calls_model_once() -> None:
    model = _RecordingModel()
    wrapper = _make_wrapper(model)
    px = torch.randn(1, 3, 1, 256, 256)
    z = wrapper.encode(px)
    assert z.shape == (1, 16, 1, 32, 32)
    assert len(model.enc_calls) == 1


def test_encode_tiling_on_tiles_in_pixel_space() -> None:
    model = _RecordingModel()
    wrapper = VAEWrapper(model, torch.zeros(16), torch.ones(16), tiling="on")
    px = torch.zeros(1, 3, 1, 1024, 1024)
    z = wrapper.encode(px)
    assert z.shape == (1, 16, 1, 128, 128)
    assert len(model.enc_calls) == 9
    assert all(s == (1, 3, 1, 512, 512) for s in model.enc_calls)


def test_encode_oom_fallback_invokes_tiled_encode() -> None:
    model = _RecordingModel(oom_on_full_enc=True)
    wrapper = VAEWrapper(model, torch.zeros(16), torch.ones(16), tiling="off")
    px = torch.zeros(1, 3, 1, 1024, 1024)
    z = wrapper.encode(px)
    assert z.shape == (1, 16, 1, 128, 128)
    assert model.enc_calls[0] == (1, 3, 1, 1024, 1024)
    assert len(model.enc_calls) == 1 + 9


def test_tiled_encode_reconstructs_full_for_linear_encoder() -> None:
    model = _RecordingModel()
    wrapper = _make_wrapper(model)
    torch.manual_seed(0)
    px = torch.randn(1, 3, 1, 1024, 1024)
    full = wrapper.model.encode(px, wrapper.scale)
    tiled = wrapper._tiled_encode(px)
    assert tiled.shape == full.shape
    assert torch.allclose(tiled, full, atol=1e-4)


def test_should_offload_for_whole_decode_false_on_cpu() -> None:
    wrapper = _make_wrapper(_RecordingModel())
    z = torch.zeros(1, 16, 1, 128, 128)  # CPU
    assert wrapper.should_offload_for_whole_decode(z) is False


def test_est_encode_peak_scales_with_pixels_and_dtype() -> None:
    wrapper = _make_wrapper(_RecordingModel())
    px32 = torch.zeros(1, 3, 1, 1024, 1024, dtype=torch.float32)
    est_fp32 = wrapper._est_encode_peak_bytes(px32)
    assert est_fp32 == pytest.approx(5500 * 1024 * 1024, rel=1e-6)
    px16 = torch.zeros(1, 3, 1, 1024, 1024, dtype=torch.bfloat16)
    assert wrapper._est_encode_peak_bytes(px16) == pytest.approx(est_fp32 / 2, rel=1e-6)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_log_once_same_key_logs_only_first(caplog) -> None:
    wrapper = _make_wrapper(_RecordingModel())
    logger = logging.getLogger("training.vae")
    with caplog.at_level(logging.INFO, logger="training.vae"):
        for i in range(200):
            wrapper._log_once("auto_encode", logger.info, "主动分块 #%d", i)
    hits = [r for r in caplog.records if "主动分块" in r.getMessage()]
    assert len(hits) == 1
    assert "#0" in hits[0].getMessage()


def test_log_once_distinct_keys_each_logged_once(caplog) -> None:
    wrapper = _make_wrapper(_RecordingModel())
    logger = logging.getLogger("training.vae")
    with caplog.at_level(logging.WARNING, logger="training.vae"):
        wrapper._log_once("auto_encode", logger.warning, "encode 分块")
        wrapper._log_once("auto_decode", logger.warning, "decode 分块")
        wrapper._log_once("auto_encode", logger.warning, "encode 分块")
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs == ["encode 分块", "decode 分块"]
