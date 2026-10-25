"""LoRA merge for the MOSS int8 convrot base (bit-exact ComfyUI parity, MOSS design §5.9 / M9).

The merge itself is family-neutral (``training.families.lora_int8_merge``); MOSS only adds its
merge units. Comfy-Org's MOSS files store qkv fused, and Comfy's key is ``layers.N.proj.qkv``:
the q/k/v LoRAs land on row slices of the fused weight (``narrow(0, i*dim, dim)``), and the
stochastic-rounding random numbers are drawn once for the fused shape. So q/k/v are concatenated
back, merged and re-quantized as one block, then split into this project's three q/k/v Linear layers.

Used only for inference (training uses W8A16 + LoRA bypass, see quant_int8). Attribution in THIRD_PARTY_NOTICES.
"""

from __future__ import annotations

import re

import torch
from torch import Tensor

from training.families.comfy_lora_merge import QuantizedMergeAdapter
from training.families.lora_int8_merge import merge_loras_into_int8_model as _merge_int8

_QKV_RE = re.compile(r"^(layers\.\d+\.proj)\.([qkv])$")


def _comfy_units(model: torch.nn.Module, layer_names: set[str]) -> dict[str, list[str]]:
    """LoRA-targeted Linear -> ComfyUI weight key (unit of merge / seed) and its members (in row order)."""
    fused = bool(getattr(model, "moss_fused_qkv", False))
    units: dict[str, list[str]] = {}
    for name in sorted(layer_names):
        m = _QKV_RE.match(name)
        if fused and m:
            units[f"{m.group(1)}.qkv"] = [f"{m.group(1)}.{p}" for p in ("q", "k", "v")]
        else:
            units[name] = [name]
    return units


def merge_loras_into_int8_model(
    model: torch.nn.Module,
    sources: list[tuple[dict[str, Tensor], float, str]],
    *,
    compute_dtype: torch.dtype = torch.float32,
    chunk_rows: int | None = None,
    keep_backup: bool = True,
) -> QuantizedMergeAdapter:
    """Merge several LoRAs into the MOSS int8 base (see ``training.families.lora_int8_merge``)."""
    return _merge_int8(
        model, sources, compute_dtype=compute_dtype, chunk_rows=chunk_rows,
        keep_backup=keep_backup, merge_units=_comfy_units,
    )
