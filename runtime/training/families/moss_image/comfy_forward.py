"""ComfyUI-style inference forward for the MOSS-Image DiT (generation parity, MOSS design §5.9 / M9).

The ``modeling.moss`` forward follows the official upstream implementation (for training:
modulation in fp32, RoPE cast to bf16 before rotation, hand-written RMSNorm). It is
mathematically identical to ComfyUI's inference forward but rounds differently. This module
reuses the same weights and follows the inference branch of ComfyUI 0.41.0
``comfy/ldm/moss_image/model.py`` and the comfy-kitchen 0.2.38 eager backend op by op:

- modulated norm ``rms_adaln(x, w·(1+scale)−1, shift)``: ``w·(1+scale)−1`` is computed in bf16,
  then ``rms_norm(x) · (1+scale') + shift``
- q/k ``rms_rope_split_half``: ``rms_norm(x, weight)``, then split-half rotation with fp32
  frequencies (product promoted to fp32), result cast back to bf16
- RoPE frequencies: flux ``rope`` (computed in fp64, stored in fp32); time embedding: flux ``timestep_embedding``
- gated residual: in-place ``addcmul_`` (one rounding)
- linear layers: the model's own Linear (int8 layers already use the comfy-kitchen eager ``w8a8`` forward)

**Not bit-exact**: DiT attention. For Comfy-Org's MOSS files ComfyUI uses comfy-kitchen's
bundled INT8 SageAttention CUDA kernel (closed-source binary); here attention follows the
model's ``attention_backend`` through the flash-attn package / xformers, and PyTorch SDPA for
``none`` (copying Comfy's backend priority). ComfyUI runs the text refiner blocks through xformers.

Attribution in THIRD_PARTY_NOTICES (ComfyUI GPL-3.0 + comfy_kitchen Apache-2.0).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from modeling.moss_image.attention import AttentionParams, attention
SDPA = "torch"

def attention_bshd(q, k, v, backend, *, stage=None):
    mask = torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
    return attention((q, k, v), attn_params=AttentionParams(backend, mask)).unflatten(-1, q.shape[-2:])

_EPS = 1e-6
_ROPE_THETA = 10000


def _rms_norm(x: Tensor, weight: Tensor | None) -> Tensor:
    """comfy ``ops.RMSNorm`` (``F.rms_norm`` with the weight cast to the input dtype)."""
    if weight is None:
        return F.rms_norm(x, (x.shape[-1],), eps=_EPS)
    return F.rms_norm(x, weight.shape, weight=weight.to(x.dtype), eps=_EPS)


def _rms_adaln(x: Tensor, norm_weight: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """ComfyUI ``modulated_rms_norm`` -> comfy-kitchen eager ``rms_adaln``."""
    folded = norm_weight.to(x.dtype) * (1 + scale) - 1
    return F.rms_norm(x, x.shape[-1:], eps=_EPS) * (1 + folded) + shift


def _apply_rope_split_half(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """comfy-kitchen eager ``apply_rope_split_half1`` (copied line by line)."""
    t = x
    if torch.promote_types(x.dtype, freqs_cis.dtype) != freqs_cis.dtype:
        t = x.to(freqs_cis.dtype)
    if freqs_cis.ndim == 2:
        freqs_cis = freqs_cis.unsqueeze(0)
    diagonal = freqs_cis.diagonal(dim1=-2, dim2=-1).movedim(-1, -2)
    out = t.unflatten(-1, (2, -1)) * diagonal
    half = x.shape[-1] // 2
    out[..., 0, :].addcmul_(t[..., half:], freqs_cis[..., 0, 1])
    out[..., 1, :].addcmul_(t[..., :half], freqs_cis[..., 1, 0])
    return out.reshape(x.shape).type_as(x)


def _rms_rope_split_half(x: Tensor, pe: Tensor, weight: Tensor) -> Tensor:
    """comfy-kitchen eager ``_rms_rope1(split_half=True)``."""
    x_norm = F.rms_norm(x, (x.shape[-1],), weight=weight.to(x.dtype), eps=_EPS)
    return _apply_rope_split_half(x_norm, pe)


def _flux_rope(pos: Tensor, dim: int) -> Tensor:
    """ComfyUI ``ldm/flux/math.rope``: frequencies and phases in fp64, fp32 2×2 rotation matrices out."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (_ROPE_THETA ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32).double(), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.to(dtype=torch.float32)


def _rope(head_dim: int, h: int, w: int, text_len: int, device) -> Tensor:
    """ComfyUI ``MOSSImage.rope`` + flux ``EmbedND``: image first (f = text length, rows and
    columns centered), text after; returns ``[1, N, 1, head_dim/2, 2, 2]``."""
    ids = torch.zeros(h * w + text_len, 4, device=device, dtype=torch.float32)
    ids[:h * w, 1] = text_len
    ids[:h * w, 2] = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2).repeat_interleave(w)
    ids[:h * w, 3] = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2).repeat(h)
    ids[h * w:, 1] = torch.arange(text_len, device=device, dtype=torch.float32)
    ids = ids.unsqueeze(0)
    axis_dim = head_dim // 4
    emb = torch.cat([_flux_rope(ids[..., i], axis_dim) for i in range(4)], dim=-3).unsqueeze(1)
    return emb.transpose(1, 2).contiguous()


def _timestep_embedding(t: Tensor, dim: int = 256, max_period: int = 10000) -> Tensor:
    """flux ``timestep_embedding`` (``time_factor=1000``, fp32)."""
    t = 1000.0 * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding.to(t)


def _attention(q: Tensor, k: Tensor, v: Tensor, q_norm_w: Tensor, k_norm_w: Tensor,
               pe: Tensor, heads: int, backend: str = SDPA) -> Tensor:
    """ComfyUI ``moss_image.attention`` inference branch; kernels are described in the module docstring."""
    b, n, _ = q.shape
    q = _rms_rope_split_half(q.view(b, n, heads, -1), pe, q_norm_w)
    k = _rms_rope_split_half(k.view(b, n, heads, -1), pe, k_norm_w)
    if backend != SDPA:
        return attention_bshd(q, k, v.view(b, n, heads, -1), backend, stage="moss_comfy").flatten(2)
    q, k = q.flatten(2), k.flatten(2)
    dim_head = q.shape[-1] // heads
    # comfy attention_pytorch: reshape to heads, then transpose to (b, heads, n, d)
    q, k, v = (t.unsqueeze(3).reshape(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
    return _sdpa(q, k, v).transpose(1, 2).reshape(b, -1, heads * dim_head)


def _sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """comfy ``ops.scaled_dot_product_attention``: for inputs of at least 128K elements, pick the
    SDPA backend in the order FLASH -> CUDNN -> EFFICIENT -> MATH (torch's default order puts cuDNN
    after efficient, and the two differ numerically); small inputs call it directly."""
    if q.nelement() < 1024 * 128 or not q.is_cuda or _SDPA_PRIORITY is None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    from torch.nn.attention import sdpa_kernel

    with sdpa_kernel(_SDPA_PRIORITY, set_priority=True):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)


def _sdpa_priority():
    try:
        import inspect

        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return None
    if "set_priority" not in inspect.signature(sdpa_kernel).parameters:
        return None
    return [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


_SDPA_PRIORITY = _sdpa_priority()


def _situ(hidden: Tensor, gate: Tensor) -> Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _text_block(block, x: Tensor, pe: Tensor, backend: str) -> Tensor:
    q, k, v = block.qkv(_rms_norm(x, block.norm1.weight)).chunk(3, dim=-1)
    x = x + block.proj(_attention(q, k, v, block.q_norm.weight, block.k_norm.weight, pe, block.heads,
                                  backend))
    y = _rms_norm(x, block.norm2.weight)
    return x + block.ffn_down(_situ(block.ffn_up(y), block.ffn_gate(y)))


def _dit_block(block, x: Tensor, mod: tuple[Tensor, ...], pe: Tensor, backend: str) -> Tensor:
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
    y = _rms_adaln(x, block.norm1.weight, scale_a, shift_a)
    q, k, v = block.proj.q(y), block.proj.k(y), block.proj.v(y)
    out = _attention(q, k, v, block.q_norm.weight, block.k_norm.weight, pe, block.heads, backend)
    x = x.addcmul_(block.proj.o(out), gate_a)
    y = _rms_adaln(x, block.norm2.weight, scale_f, shift_f)
    hidden, gate = block.proj.ffn_up_gate(y).chunk(2, dim=-1)
    return x.addcmul_(block.proj.ffn_down(_situ(hidden, gate)), gate_f)


@torch.no_grad()
def comfy_forward(model, x: Tensor, timestep: Tensor, context: Tensor, *,
                  skip_layers=()) -> Tensor:
    """ComfyUI ``MOSSImage._forward``.

    ``x``: ``[B, 128, H/16, W/16]`` (ComfyUI's patchified latent layout, already in the compute
    dtype); ``timestep``: σ (fp32 ``[B]``); ``context``: ``[B, S, text_dim]``.
    ``skip_layers``: layers skipped by SkipLayerGuidanceDiTSimple (passed through unchanged).
    Returns the velocity prediction ``[B, 128, H/16, W/16]`` in the same dtype.
    """
    b, _, h, w = x.shape
    dtype, device = x.dtype, x.device
    # Image-tokens-only LoRA (experimental.py): this forward bypasses MossImage.forward and
    # its pre-hook, so publish the image length here
    mask_state = getattr(model, "_moss_token_mask_state", None)
    if mask_state is not None:
        mask_state["img_len"] = h * w
    heads = model.num_heads
    backend = getattr(model, "attn_mode", SDPA)
    pe = _rope(model.dim // heads, h, w, context.shape[1], device)

    vec = model.time_embedding(_timestep_embedding(timestep, model.freq_dim).to(dtype))
    mod = model.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    head_shift, head_scale = model.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)

    txt = model.text_embedder.proj(context)
    for block in model.text_embedder.blocks:
        txt = _text_block(block, txt, pe[:, h * w:], backend)
    hidden = torch.cat((model.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)
    del txt

    skip = frozenset(skip_layers or ())
    # Per-block modulation offsets (experimental.BlockModulation); empty for ordinary LoRAs
    block_mods = getattr(model, "_moss_block_mods", None) or ()
    for index, block in enumerate(model.layers):
        if index in skip:
            continue
        block_mod = mod
        for adapter in block_mods:
            delta = adapter.delta(index, vec).unbind(1)
            block_mod = tuple(m + d.unsqueeze(1).to(m.dtype) for m, d in zip(block_mod, delta))
        hidden = _dit_block(block, hidden, block_mod, pe, backend)

    out = model.final_linear_video(
        _rms_adaln(hidden[:, :h * w], model.final_norm_video.weight, head_scale, head_shift)
    )
    return out.transpose(1, 2).reshape(b, model.latent_channels, h, w)
