"""Residency and forward for ComfyUI ``int8_tensorwise + convrot`` weights (MOSS design §5.4).

Family-neutral: MOSS and Krea2 both load ConvRot INT8 checkpoints through this module.

Format (comfy-kitchen 0.2.38 ``tensor/int8_utils.py`` /
``backends/eager/quantization.py``, Apache-2.0, see THIRD_PARTY_NOTICES):

- H = normalized regular Hadamard (kron expansion of a 4×4 base / √g), symmetric and self-inverse
- storage: ``W_rot = W @ Hᵀ`` (groups of g columns along the input dim), then row-wise int8;
  ``weight_scale`` is a row-wise F32 ``[out, 1]``
- dequantization: ``W = (q · s) @ Hᵀ`` (H is self-inverse)

Three forwards (``patch_int8_linear(..., forward=...)``; int8 stays resident as is):

- ``w8a16`` (training): ``y = ((x @ H) @ qᵀ) ⊙ sᵀ + b``. q holds integers within ±127, so the
  cast to the compute dtype is lossless; scale is applied per output column after the matmul,
  creating no fp32 temporary weight and not rounding scale to bf16 first. ``x @ H`` is an
  ordinary differentiable matmul, so autograd gives the input gradient naturally.
- ``w8a8`` (generation DiT, ComfyUI parity): bit-exact copy of comfy-kitchen eager ``int8_linear``:
  the rotated activation is dynamically quantized to int8 per row, multiplied with
  ``torch._int_mm``, then scaled back to the output dtype by ``x_scale × w_scale`` (fp32). The
  author's local ComfyUI (torch cu128) disables comfy-kitchen's cuda / triton backends, so eager
  is what actually runs.
- ``dequant`` (TE, ComfyUI parity): ComfyUI does not use W8A8 for the TE; each forward dequantizes
  the weight to the compute dtype and runs an ordinary ``F.linear`` (``full_precision_mm``).

Re-quantization after a LoRA merge (stochastic rounding): see :func:`requantize_convrot`.
"""

from __future__ import annotations

import json
import math
from types import MethodType

import torch
from torch import Tensor, nn
from torch.nn import functional as F

#: The only accepted Comfy per-layer quantization config
INT8_CONVROT_FORMAT = "int8_tensorwise"

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], Tensor] = {}


def hadamard(size: int, *, device=None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Normalized regular Hadamard matrix (built like comfy-kitchen ``_build_hadamard``).

    Elements are ±1/√size; for size a power of 4, √size is a power of 2, so bf16 / fp16 are exact per element.
    """
    if size < 4 or size & (size - 1) or (size.bit_length() - 1) % 2:
        raise ValueError(f"convrot group size must be a power of 4, got {size}")
    device = torch.device(device) if device is not None else torch.device("cpu")
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    # The cache is process-wide: if TE encoding first builds H inside inference_mode, it is an
    # inference tensor, and when the DiT training forward later reuses it (same device / dtype /
    # group size) autograd cannot save it and backward fails. Build outside inference_mode so the result is a normal tensor.
    with torch.inference_mode(False):
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
        h = h4
        while h.shape[0] < size:
            h = torch.kron(h, h4)
        out = (h / math.sqrt(size)).to(device=device, dtype=dtype)
    _HADAMARD_CACHE[key] = out
    return out


def parse_comfy_quant(blob: Tensor | bytes) -> dict:
    """Parse a ``{layer}.comfy_quant`` U8 blob; raise unless it is int8_tensorwise + convrot.

    ComfyUI-INT8-Fast writes the spec without ``format`` (``{"convrot": true, "convrot_groupsize": 256,
    "per_row": true}``); the int8 weight dtype already implies int8_tensorwise, so a missing format is accepted.
    """
    raw = bytes(blob.tolist()) if isinstance(blob, Tensor) else bytes(blob)
    try:
        conf = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"comfy_quant config is not valid JSON: {exc}") from exc
    if not isinstance(conf, dict) or conf.get("format", INT8_CONVROT_FORMAT) != INT8_CONVROT_FORMAT \
            or not conf.get("convrot"):
        raise ValueError(
            f"Only {INT8_CONVROT_FORMAT} + convrot int8 weights are supported, got {conf}"
        )
    group = int(conf.get("convrot_groupsize", 256))
    hadamard(group)  # validates the group size
    return {"format": INT8_CONVROT_FORMAT, "convrot": True, "convrot_groupsize": group}


def rotate_activation(x: Tensor, group: int) -> Tensor:
    """``x @ H`` (in groups of ``group`` along the last dim)."""
    features = x.shape[-1]
    if features % group:
        raise ValueError(f"Input dim {features} is not divisible by convrot group size {group}")
    h = hadamard(group, device=x.device, dtype=x.dtype)
    return (x.reshape(*x.shape[:-1], features // group, group) @ h).reshape(x.shape)


def _rotate_activation_comfy(x: Tensor, group: int) -> Tensor:
    """comfy-kitchen ``_rotate_activation``: a 3D batched matmul over ``(-1, groups, group size)``
    with H in ``x.dtype``. Mathematically the same as :func:`rotate_activation`; the shape is
    copied so cuBLAS picks the same algorithm (used by the W8A8 parity path)."""
    features = x.shape[-1]
    if features % group:
        raise ValueError(f"Input dim {features} is not divisible by convrot group size {group}")
    h = hadamard(group, device=x.device, dtype=x.dtype)
    return torch.matmul(x.reshape(-1, features // group, group), h).reshape(x.shape)


def _rotate_weight_comfy(w: Tensor, group: int) -> Tensor:
    """comfy-kitchen ``_rotate_weight``: ``W @ Hᵀ`` (grouped along the input dim), H in ``w.dtype``."""
    out_f, in_f = w.shape
    if in_f % group:
        raise ValueError(f"in_features {in_f} is not divisible by convrot group size {group}")
    h = hadamard(group, device=w.device, dtype=w.dtype)
    return torch.matmul(w.reshape(out_f, in_f // group, group), h.T).reshape(out_f, in_f)


def _round_int8(scaled: Tensor, seed: int = 0) -> Tensor:
    """comfy-kitchen eager ``_round_int8``: stochastic rounding when seed > 0 (``torch.rand`` of the
    same dtype, Generator on the device), otherwise round half to even. Rewrites ``scaled`` in place."""
    if seed > 0:
        generator = torch.Generator(device=scaled.device)
        generator.manual_seed(int(seed))
        scaled.add_(torch.rand(
            scaled.shape, dtype=scaled.dtype, layout=scaled.layout,
            device=scaled.device, generator=generator,
        ))
        return scaled.floor_().clamp_(-128.0, 127.0).to(torch.int8)
    return scaled.round_().clamp_(-128.0, 127.0).to(torch.int8)


def quantize_rowwise(x: Tensor, *, seed: int = 0) -> tuple[Tensor, Tensor]:
    """comfy-kitchen eager ``quantize_int8_rowwise``: row-wise amax/127 (fp32 scale, floor 1e-30);
    the division uses the scale cast to ``x.dtype`` (0 replaced by that dtype's tiny)."""
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    divisor = scale.to(device=x.device, dtype=x.dtype)
    divisor = torch.where(divisor == 0, torch.full_like(divisor, torch.finfo(x.dtype).tiny), divisor)
    return _round_int8(x / divisor, seed), scale


def int8_mm(a: Tensor, b: Tensor) -> Tensor:
    """comfy-kitchen eager ``_int8_matmul_accumulate``: zero-pads per Comfy's rules before
    ``torch._int_mm`` (on CUDA, M to a multiple of 32 and ≥32, K to a multiple of 8, N to a multiple
    of 8 / 32 on Turing). Integer accumulation is exact; padding only satisfies cuBLASLt shape limits."""
    def _round_up(value: int, alignment: int) -> int:
        return ((value + alignment - 1) // alignment) * alignment

    orig_m, k, orig_n = a.size(0), a.size(1), b.size(1)
    if orig_m == 0 or k == 0:
        return torch._int_mm(a, b)
    padded_m = _round_up(max(orig_m, 32), 32) if a.is_cuda else orig_m
    if padded_m != orig_m:
        a = torch.cat((a, a.new_zeros((padded_m - orig_m, k))), dim=0)
    padded_k = _round_up(k, 8)
    if padded_k != k:
        a = torch.cat((a, a.new_zeros((a.size(0), padded_k - k))), dim=1)
        b = torch.cat((b, b.new_zeros((padded_k - k, b.size(1)))), dim=0)
    turing = a.is_cuda and torch.cuda.get_device_capability(a.device) == (7, 5)
    padded_n = _round_up(orig_n, 32 if turing else 8)
    if padded_n != orig_n:
        b = torch.cat((b, b.new_zeros((b.size(0), padded_n - orig_n))), dim=1)
    out = torch._int_mm(a, b)
    if out.size(0) != orig_m or out.size(1) != orig_n:
        out = out[:orig_m, :orig_n]
    return out


def w8a8_linear(x: Tensor, qweight: Tensor, weight_scale: Tensor, group: int,
                bias: Tensor | None = None) -> Tensor:
    """Bit-exact copy of comfy-kitchen eager ``int8_linear`` (convrot); output dtype = ``x.dtype``."""
    out_dtype = x.dtype
    x = _rotate_activation_comfy(x, group)
    orig_shape = x.shape
    x_8, x_scale = quantize_rowwise(x.reshape(-1, x.shape[-1]))
    acc = int8_mm(x_8, qweight.to(device=x.device).T.contiguous())
    w_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(1, -1)
    # Comfy chunks by rows only to save memory; element-wise ops match the whole block: (x_scale·w_scale) first
    out = (acc.float() * (x_scale.float() * w_scale)).to(out_dtype)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype).reshape(1, -1)
    return out.reshape(*orig_shape[:-1], qweight.shape[0])


def requantize_convrot(weight: Tensor, group: int, *, seed: int) -> tuple[Tensor, Tensor]:
    """Re-quantize merged weights back to int8 (comfy-kitchen eager ``quantize_int8_convrot_weight``
    + ``stochastic_rounding=seed``): rotate in ``weight.dtype`` (fp16 in Comfy), row-wise amax,
    stochastic rounding. Returns ``(int8 [N, K], fp32 scale [N, 1])``."""
    return quantize_rowwise(_rotate_weight_comfy(weight, group), seed=seed)


def dequantize(qweight: Tensor, scale: Tensor, group: int) -> Tensor:
    """int8 + row-wise scale -> fp32 weight in the original basis (comfy-kitchen ``dequantize_int8_convrot_weight``:
    multiply by scale in fp32, then by Hᵀ per group; callers ``.to(dtype)`` as needed)."""
    w = qweight.float() * scale.float()
    out_f, in_f = w.shape
    h = hadamard(group, device=w.device, dtype=torch.float32)
    return (w.reshape(out_f, in_f // group, group) @ h.T).reshape(out_f, in_f)


def _int8_convrot_forward(self: nn.Linear, input: Tensor) -> Tensor:
    x = rotate_activation(input, self._convrot_groupsize)
    out = F.linear(x, self.weight.to(x.dtype))
    out = out * self.scale_weight.reshape(-1).to(device=out.device, dtype=out.dtype)
    if self.bias is not None:
        out = out + self.bias.to(out.dtype)
    return out


def _int8_convrot_w8a8_forward(self: nn.Linear, input: Tensor) -> Tensor:
    return w8a8_linear(input, self.weight, self.scale_weight, self._convrot_groupsize, self.bias)


def _int8_convrot_dequant_forward(self: nn.Linear, input: Tensor) -> Tensor:
    weight = dequantize(self.weight, self.scale_weight, self._convrot_groupsize).to(input.dtype)
    bias = None if self.bias is None else self.bias.to(input.dtype)
    return F.linear(input, weight, bias)


_INT8_FORWARDS = {
    "w8a16": _int8_convrot_forward,
    "w8a8": _int8_convrot_w8a8_forward,
    "dequant": _int8_convrot_dequant_forward,
}


def patch_int8_linear(module: nn.Linear, scale: Tensor, group: int, *,
                      forward: str = "w8a16") -> None:
    """Attach a forward to a Linear with int8 weights (three kinds, see the module docstring). ``scale`` is
    stored as a non-persistent buffer (not in state_dict)."""
    if forward not in _INT8_FORWARDS:
        raise ValueError(f"Unknown int8 forward {forward!r}, choose from {sorted(_INT8_FORWARDS)}")
    if not isinstance(module, nn.Linear):
        raise ValueError(f"int8 patch target is not a Linear: {type(module).__name__}")
    if module.weight.dtype != torch.int8:
        raise ValueError(f"int8 patch target weight is not int8: {module.weight.dtype}")
    if scale.shape not in ((module.out_features, 1), (module.out_features,)):
        raise ValueError(
            f"int8 weight_scale shape {tuple(scale.shape)} does not match out_features={module.out_features}"
        )
    if module.in_features % group:
        raise ValueError(f"in_features={module.in_features} is not divisible by convrot group size {group}")
    module.register_buffer("scale_weight", scale.to(dtype=torch.float32), persistent=False)
    module._convrot_groupsize = int(group)
    module.forward = MethodType(_INT8_FORWARDS[forward], module)


def _int8_convrot_embedding_forward(self: nn.Embedding, input: Tensor) -> Tensor:
    # Same as comfy-kitchen dequantize_embedding: take only the used rows, dequantize per row,
    # then un-rotate along the hidden dim (W = W_rot @ Hᵀ; H is symmetric, so this is the activation rotation)
    rows = self.weight[input].float() * self.scale_weight[input].float()
    return rotate_activation(rows, self._convrot_groupsize).to(self._int8_compute_dtype)


def patch_int8_embedding(module: nn.Embedding, scale: Tensor, group: int,
                         compute_dtype: torch.dtype) -> None:
    """Attach a row-wise dequantizing lookup to an Embedding with int8 weights (embed_tokens of the Comfy int8 TE)."""
    if not isinstance(module, nn.Embedding):
        raise ValueError(f"int8 embedding patch target is not an Embedding: {type(module).__name__}")
    if module.weight.dtype != torch.int8:
        raise ValueError(f"int8 embedding patch target weight is not int8: {module.weight.dtype}")
    if scale.shape not in ((module.num_embeddings, 1), (module.num_embeddings,)):
        raise ValueError(
            f"int8 embedding weight_scale shape {tuple(scale.shape)} does not match vocabulary size {module.num_embeddings}"
        )
    if module.embedding_dim % group:
        raise ValueError(f"embedding_dim={module.embedding_dim} is not divisible by convrot group size {group}")
    module.register_buffer("scale_weight", scale.reshape(-1, 1).to(dtype=torch.float32), persistent=False)
    module._convrot_groupsize = int(group)
    module._int8_compute_dtype = compute_dtype
    module.forward = MethodType(_int8_convrot_embedding_forward, module)


def model_has_int8_layers(model: object) -> bool:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return False
    return any(isinstance(m, nn.Linear) and m.weight.dtype == torch.int8 for m in modules())
