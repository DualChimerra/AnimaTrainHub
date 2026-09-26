"""Family-agnostic latent space constants (a shared external fact layer).

The latent space is a property of the VAE, not of the model family -- multiple
families can share the same space (D6: identical fingerprints auto-share the
npz cache). The space's definition (normalization stats / latent2rgb
projection coefficients) comes from a single upstream source of truth,
defined once here; family specs reference the same instance, making
"shared across families" a structural fact (identity) rather than
per-family copies kept equal by tests.

Wan2.1 space: defined by ComfyUI's `comfy/latent_formats.py` `Wan21`. The
Qwen-Image VAE reuses it (`comfy/supported_models.py`
`QwenImage.latent_format = Wan21`). Anima and Krea 2 share this space (D6/D17).
"""

from __future__ import annotations

# Relative import: the studio server imports this module indirectly via the
# `runtime.training.*` namespace (for bucket distribution previews), where
# sys.path doesn't include runtime/, so an absolute `training.*` import would
# raise ModuleNotFoundError.
from .spec import LatentSpec

# Linear projection coefficients for the fast latent2rgb preview ("blurry but
# recognizable"). Taken from ComfyUI's comfy/latent_formats.py
# `Wan21.latent_rgb_factors[_bias]` (GPL-3.0, see THIRD_PARTY_NOTICES).
_WAN21_RGB_FACTORS: tuple[tuple[float, float, float], ...] = (
    (-0.1299, -0.1692, 0.2932),
    (0.0671, 0.0406, 0.0442),
    (0.3568, 0.2548, 0.1747),
    (0.0372, 0.2344, 0.1420),
    (0.0313, 0.0189, -0.0328),
    (0.0296, -0.0956, -0.0665),
    (-0.3477, -0.4059, -0.2925),
    (0.0166, 0.1902, 0.1975),
    (-0.0412, 0.0267, -0.1364),
    (-0.1293, 0.0740, 0.1636),
    (0.0680, 0.3019, 0.1128),
    (0.0032, 0.0581, 0.0639),
    (-0.1251, 0.0927, 0.1699),
    (0.0060, -0.0633, 0.0005),
    (0.3477, 0.2275, 0.2950),
    (0.1984, 0.0913, 0.1861),
)
_WAN21_RGB_BIAS: tuple[float, float, float] = (-0.1835, -0.0868, -0.3360)

#: Qwen-Image VAE = Wan2.1 latent space, f8, 16ch.
WAN21_F8C16 = LatentSpec(
    fingerprint="wan21-f8c16",
    channels=16,
    spatial_stride=8,
    patch_spatial=2,
    patch_temporal=1,
    temporal=False,
    rgb_factors=_WAN21_RGB_FACTORS,
    rgb_bias=_WAN21_RGB_BIAS,
)
