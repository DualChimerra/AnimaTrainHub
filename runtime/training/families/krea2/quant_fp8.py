"""Krea2 inference-side fp8 weight support (bit-for-bit ComfyUI parity).

Two community/official fp8 forms (tmp/quant-inference-design.md):
- **fp8_scaled**: weights in F8_E4M3 + a per-layer F32 scalar
  ``{layer}.weight_scale``, with safetensors
  ``__metadata__._quantization_metadata`` declaring each layer's format (this
  is the form used by Comfy-Org's official Turbo fp8)
- **plain fp8 cast**: weights stored directly as fp8, no scale, no metadata

The numerics reproduce the local ComfyUI oracle bit-for-bit (the eager path
under v0.27.1 + torch2.7/cu128, design doc Sec 1):

    W_compute = W_fp8.to(input.dtype) [* scale.to(input.dtype)]
    y = F.linear(x, W_compute, bias)

i.e. comfy_kitchen's ``dequantize_per_tensor_fp8``
(backends/eager/quantization.py:59-63, where scale is **cast to the compute
dtype before multiplying**) + comfy's ``cast_bias_weight`` (target dtype =
input.dtype). Native ``_scaled_mm`` is not used (Comfy doesn't use it either
when ``--fast`` is empty, the default). The ``full_precision_matrix_mult``
layer flag only affects Comfy's choice of native matmul and makes no
difference to the dequant path -- it's parsed and then ignored.

The forward-patch approach is adapted from kohya-ss/musubi-tuner's
``apply_fp8_monkey_patch`` (Apache-2.0, see THIRD_PARTY_NOTICES); the dequant
formula matches Comfy (GPL-3.0).

After patching, weights have requires_grad=False; training (fp8_base) and
inference share this same patch, the base model is always frozen, and
gradients only flow through LoRA parameters.
"""

from __future__ import annotations

import json
import logging
from types import MethodType

import torch


logger = logging.getLogger(__name__)

_FP8_TORCH_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_SUPPORTED_FORMATS = {"float8_e4m3fn", "float8_e5m2"}


def parse_quantization_metadata(metadata: dict | None) -> dict[str, dict]:
    """Parse safetensors ``__metadata__._quantization_metadata`` (the Comfy protocol).

    Returns {layer name: layer config dict}; returns an empty dict if
    undeclared; raises if an unsupported format is declared (only the two fp8
    forms are supported -- scope decided by the user).
    """
    if not metadata:
        return {}
    raw = metadata.get("_quantization_metadata")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Krea2 checkpoint's _quantization_metadata is not valid JSON: {exc}")
    layers = parsed.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("Krea2 _quantization_metadata is missing the layers mapping")
    unsupported = {
        str(conf.get("format"))
        for conf in layers.values()
        if isinstance(conf, dict) and conf.get("format") not in _SUPPORTED_FORMATS
    }
    if unsupported:
        raise ValueError(
            f"Krea2 checkpoint declares an unsupported quantization format {sorted(unsupported)}; "
            f"inference currently only supports fp8 ({sorted(_SUPPORTED_FORMATS)}) -- "
            f"for int8/nvfp4/int4, use bf16 or fp8 weights instead"
        )
    return {str(layer): dict(conf) for layer, conf in layers.items()}


def _fp8_linear_forward(self, input: torch.Tensor) -> torch.Tensor:
    # ComfyUI parity: target dtype = input.dtype (cast_bias_weight :286-290);
    # scale is cast to the compute dtype before multiplying (ck eager
    # dequantize_per_tensor_fp8). bias is likewise cast to input.dtype by
    # cast_bias_weight -- in the DiT case bias is already input dtype (a
    # no-op, parity unaffected); in the TE fp8 (fp32 compute) case bias is
    # stored as fp16 and must be cast.
    weight = self.weight.to(input.dtype)
    scale = getattr(self, "weight_scale", None)
    if scale is not None:
        weight = weight * scale.to(input.dtype)
    bias = self.bias
    if bias is not None and bias.dtype != input.dtype:
        bias = bias.to(input.dtype)
    return torch.nn.functional.linear(input, weight, bias)


def patch_fp8_linears(
    model: torch.nn.Module,
    scales: dict[str, torch.Tensor | None],
    *,
    device: torch.device | None = None,
) -> int:
    """Attach a dequant forward to Linear layers holding fp8 weights; returns the patch count.

    ``scales``: layer name (e.g. ``blocks.0.attn.gate``) -> F32 scalar scale;
    pass None for the plain-cast form. scale is stored on the module as a
    non-persistent buffer (not in state_dict, so it doesn't break the
    existing serialization surface).

    ``device``: where to put scale. Defaults to following that layer's weight
    device. **Block-swap scenarios must pass the compute device explicitly**:
    a swapped-out layer's weight is currently on CPU, so following it would
    also put scale on CPU, but by forward time the weight has already been
    moved to GPU, and `weight * scale` would hit a device mismatch
    (scale.to() only changes dtype, not device). scale is a per-layer scalar,
    so keeping it resident on GPU is negligible overhead.
    """
    patched = 0
    modules = dict(model.named_modules())
    for layer, scale in scales.items():
        module = modules.get(layer)
        if module is None or not isinstance(module, torch.nn.Linear):
            raise ValueError(f"fp8-quantized layer doesn't exist in the model or isn't a Linear: {layer}")
        if module.weight.dtype not in _FP8_TORCH_DTYPES:
            raise ValueError(
                f"fp8 patch target layer's weight isn't fp8: {layer} ({module.weight.dtype})"
            )
        if scale is not None:
            module.register_buffer(
                "weight_scale",
                scale.to(
                    device=device if device is not None else module.weight.device,
                    dtype=torch.float32,
                ),
                persistent=False,
            )
        module.forward = MethodType(_fp8_linear_forward, module)
        patched += 1
    if patched:
        logger.info("Krea2 fp8 inference: attached dequant forward to %d Linear layers", patched)
    return patched


def model_has_fp8_layers(model: object) -> bool:
    """Used by the sampling/LoRA paths to check whether the base model is fp8-quantized.

    Anything that isn't an nn.Module (a test fake / not yet loaded) is always
    False -- an fp8 form can only come from a real loader's output.
    """
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return False
    return any(
        isinstance(m, torch.nn.Linear) and m.weight.dtype in _FP8_TORCH_DTYPES
        for m in modules()
    )
