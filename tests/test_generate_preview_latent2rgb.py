from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from runtime.anima_daemon import (
    _PREVIEW_TARGET_PX,
    _decode_latent2rgb_preview,
    _preview_latent_spec,
)

_WAN21_LATENT_RGB_FACTORS = [list(r) for r in _preview_latent_spec().rgb_factors]
_WAN21_LATENT_RGB_BIAS = list(_preview_latent_spec().rgb_bias)


def test_wan21_factors_shape() -> None:
    assert len(_WAN21_LATENT_RGB_FACTORS) == 16
    assert all(len(row) == 3 for row in _WAN21_LATENT_RGB_FACTORS)
    assert len(_WAN21_LATENT_RGB_BIAS) == 3


def test_decode_returns_upscaled_rgb_image() -> None:
    latent = torch.randn(1, 16, 1, 16, 16)
    img = _decode_latent2rgb_preview(latent)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert max(img.size) == _PREVIEW_TARGET_PX
    assert img.size[0] == img.size[1]


def test_decode_matches_comfy_latent2rgb_math() -> None:
    h, w = _PREVIEW_TARGET_PX, _PREVIEW_TARGET_PX // 3
    latent = torch.randn(1, 16, 1, h, w)
    img = _decode_latent2rgb_preview(latent)
    assert img.size == (w, h)

    x0 = latent[0, :, 0].float()  # [16, H, W]
    factors = torch.tensor(_WAN21_LATENT_RGB_FACTORS)
    bias = torch.tensor(_WAN21_LATENT_RGB_BIAS)
    rgb = torch.einsum("chw,cr->hwr", x0, factors) + bias  # [H, W, 3]
    rgb = ((rgb + 1.0) / 2.0).clamp(0.0, 1.0)
    expected = (rgb.numpy() * 255).astype(np.uint8)
    assert np.array_equal(np.asarray(img), expected)


def test_decode_never_crashes_on_bad_input() -> None:
    assert _decode_latent2rgb_preview(torch.randn(3, 3)) is None
