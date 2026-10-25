"""Cheap pixel-space comparisons for ladder rungs (no extra models to download).

Each metric isolates one trait, so a ladder plot shows where that trait gets
replaced by the base model's defaults:

- palette:  1D Wasserstein distance of the L, a, b channel distributions
            (CIELAB units; ~2 is barely visible, >10 is a different palette)
- shapes:   correlation of Sobel edge maps at 256 px (lines, eyes, ears, contours)
- layout:   correlation of 16 px luminance thumbnails (pose / composition)
- detail:   high-frequency energy of the rung relative to the original
            (<1 = smoother, the "plastic" look; >1 = noisier)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _srgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """[3,H,W] sRGB in [0,1] -> [3,H,W] CIELAB (D65)."""
    c = rgb.clamp(0, 1).double()
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float64)
    xyz = torch.einsum("ij,jhw->ihw", m, lin)
    xyz = xyz / torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64).view(3, 1, 1)
    f = torch.where(xyz > (6 / 29) ** 3, xyz ** (1 / 3), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return torch.stack([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).double()


def _resize(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    scale = size / max(h, w)
    out = (max(1, round(h * scale)), max(1, round(w * scale)))
    return F.interpolate(x[None, None] if x.dim() == 2 else x[None], size=out,
                         mode="area")[0].squeeze(0)


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    denom = float(a.norm() * b.norm())
    return float((a * b).sum()) / denom if denom > 0 else 0.0


def _sobel(gray: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype)
    g = gray[None, None]
    gx = F.conv2d(g, kx[None, None], padding=1)
    gy = F.conv2d(g, kx.t()[None, None], padding=1)
    return (gx ** 2 + gy ** 2).sqrt()[0, 0]


def _high_freq_energy(gray: torch.Tensor) -> float:
    blur = F.avg_pool2d(gray[None, None], 5, stride=1, padding=2, count_include_pad=False)[0, 0]
    return float(((gray - blur) ** 2).mean())


def compare_images(original: torch.Tensor, rung: torch.Tensor) -> dict[str, float]:
    """Compare two [3,H,W] RGB tensors in [0,1] of the same size."""
    if original.shape != rung.shape:
        raise ValueError(f"shape mismatch: {tuple(original.shape)} vs {tuple(rung.shape)}")
    lab_o = _srgb_to_lab(_resize(original.double(), 256))
    lab_r = _srgb_to_lab(_resize(rung.double(), 256))
    q = torch.linspace(0.01, 0.99, 99, dtype=torch.float64)
    palette = float(sum((torch.quantile(lab_o[c].flatten(), q) - torch.quantile(lab_r[c].flatten(), q)).abs().mean()
                        for c in range(3)) / 3)
    gray_o, gray_r = _luma(original), _luma(rung)
    shapes = _corr(_sobel(_resize(gray_o, 256)), _sobel(_resize(gray_r, 256)))
    layout = _corr(_resize(gray_o, 16), _resize(gray_r, 16))
    hf_o = _high_freq_energy(_resize(gray_o, 512))
    detail = _high_freq_energy(_resize(gray_r, 512)) / hf_o if hf_o > 0 else 1.0
    return {"palette": palette, "shapes": shapes, "layout": layout, "detail": detail}


def effect_share(window_vs_full: dict[str, float], none_vs_full: dict[str, float]) -> dict[str, float]:
    """How much of a full LoRA's effect a partial application reproduces, per trait.

    1 = the partial result matches the full-LoRA result, 0 = no closer than
    the base model without the LoRA. Distances (palette) and similarities
    (shapes, layout) are both mapped onto "distance from the full result".
    """
    out = {}
    for name in ("palette", "shapes", "layout"):
        if name == "palette":
            gap_none, gap_win = none_vs_full[name], window_vs_full[name]
        else:
            gap_none, gap_win = 1 - none_vs_full[name], 1 - window_vs_full[name]
        out[name] = 1 - gap_win / gap_none if gap_none > 1e-9 else 1.0
    return out
