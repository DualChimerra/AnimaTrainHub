"""ComfyUI semantics for merging LoRA into quantized bases (family-neutral layer).

ComfyUI never computes LoRA on quantized storage: at load time it merges LoRA into the weights
(``model_patcher.patch_weight_to_device`` -> ``comfy.lora.calculate_weight``),
then re-quantizes them in each storage format. This module holds only the format-independent
parts shared by several families:

- :func:`string_to_seed`: seed for stochastic rounding during re-quantization (``comfy.utils.string_to_seed``)
- :func:`group_lora_layers`: groups kohya / PEFT LoRA keys by layer
- :func:`apply_lora_delta`: ``weight += ((strength * alpha) * diff).type(dtype)``
  (the lora / loha / lokr forms of comfy ``weight_adapter``)
- :class:`QuantizedMergeAdapter`: backup / restore handle for merged weights

Each family's dequantization / re-quantization (Krea2 fp8, MOSS int8 convrot) stays in the
family. This code used to live in Krea2's ``lora_fp8_merge``; when MOSS was added it moved here
under the rule that a later family does not import an earlier one. Krea2 behavior is unchanged.

Attribution in THIRD_PARTY_NOTICES (ComfyUI GPL-3.0 + comfy_kitchen).
"""

from __future__ import annotations

import zlib

import torch
from torch import Tensor

LORA_PREFIX = "lora_unet_"
COMFY_PREFIX = "diffusion_model."
# PEFT/comfy key suffix -> kohya/lycoris naming (the common civitai form:
# ``diffusion_model.{dotted layer}.lora_A/lora_B``, lora_A=down, lora_B=up,
# usually without an alpha key; comfy treats a missing alpha as scale 1.0, matching the
# "alpha" default branch of apply_lora_delta)
_PEFT_SUFFIXES = (
    ("lora_A.weight", "lora_down.weight"),
    ("lora_B.weight", "lora_up.weight"),
    ("lora_down.weight", "lora_down.weight"),
    ("lora_up.weight", "lora_up.weight"),
    ("alpha", "alpha"),
    ("dora_scale", "dora_scale"),
)


def string_to_seed(key: str) -> int:
    """Equivalent of comfy.utils.string_to_seed: standard CRC-32 (0xEDB88320).

    Comfy's hand-written version XORs ord() of each character; layer names are ASCII, so this
    matches the utf-8 bytes and zlib.crc32 is bit-identical (unit-tested against the reference).
    """
    if not key.isascii():
        raise ValueError(f"seed key must be ASCII (comfy layer names): {key!r}")
    return zlib.crc32(key.encode("utf-8"))


def group_lora_layers(sd: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    """Group by layer, normalizing both key formats to {layer_underscored: {kohya_suffix: tensor}}:

    - kohya/lycoris: ``lora_unet_{underscored layer}.{suffix}`` (this app's training output)
    - PEFT/comfy: ``diffusion_model.{dotted layer}.{lora_A|lora_B|alpha}``
      (civitai / musubi / comfy ecosystem)
    """
    layers: dict[str, dict[str, Tensor]] = {}
    for key, tensor in sd.items():
        if key.startswith(LORA_PREFIX):
            name, _, suffix = key.partition(".")
            layers.setdefault(name[len(LORA_PREFIX):], {})[suffix] = tensor
        elif key.startswith(COMFY_PREFIX):
            rest = key[len(COMFY_PREFIX):]
            for peft_suffix, kohya_suffix in _PEFT_SUFFIXES:
                if rest.endswith("." + peft_suffix):
                    layer = rest[: -len(peft_suffix) - 1]
                    layers.setdefault(layer.replace(".", "_"), {})[kohya_suffix] = tensor
                    break
            else:
                raise ValueError(
                    f"Quantized-base merge cannot recognize the suffix of a comfy-style LoRA key: {key}"
                )
    return layers


def apply_lora_delta(
    w16: Tensor,
    tensors: dict[str, Tensor],
    strength: float,
    layer: str,
    source: str,
    compute_dtype: torch.dtype = torch.float32,
    chunk_rows: int | None = None,
) -> Tensor:
    """Merge one LoRA into one layer: diff in fp32/bf16, added in ``w16``'s precision (in place)."""
    device = w16.device
    if "dora_scale" in tensors:
        raise ValueError(
            f"Quantized-base merge does not support DoRA (weight_decompose) LoRA: {source} layer {layer}. "
            f"Use the bf16 base model instead."
        )

    if "lora_down.weight" in tensors:  # plain LoRA / LoCon Linear
        if "lora_mid.weight" in tensors:
            raise ValueError(f"Quantized-base merge does not support LoCon mid (tucker): {source} layer {layer}")
        mat1 = tensors["lora_up.weight"].to(device=device, dtype=compute_dtype)
        mat2 = tensors["lora_down.weight"].to(device=device, dtype=compute_dtype)
        alpha = float(tensors["alpha"]) / mat2.shape[0] if "alpha" in tensors else 1.0
        mat1 = mat1.flatten(start_dim=1)
        mat2 = mat2.flatten(start_dim=1)
        rows = int(mat1.shape[0])
        if chunk_rows is not None and 0 < int(chunk_rows) < rows:
            # A rank-r LoRA is small, but materializing its out×in dense delta
            # can consume almost 1 GiB for Krea2's tproj layer.  Compute row
            # slices and add them immediately so peak workspace is bounded by
            # chunk_rows×in instead of the full weight.  Only plain Linear LoRA
            # uses this path; LoHa/LoKr retain their exact legacy algorithms.
            w_rows = w16.reshape(rows, -1)
            for start in range(0, rows, int(chunk_rows)):
                end = min(start + int(chunk_rows), rows)
                lora_diff = torch.mm(mat1[start:end], mat2)
                w_rows[start:end] += (
                    (strength * alpha) * lora_diff
                ).type(w16.dtype)
                del lora_diff
        else:
            lora_diff = torch.mm(mat1, mat2).reshape(w16.shape)
            w16 += ((strength * alpha) * lora_diff).type(w16.dtype)
        return w16

    if "hada_w1_a" in tensors:  # LoHa
        if "hada_t1" in tensors or "hada_t2" in tensors:
            raise ValueError(f"Quantized-base merge does not support LoHa tucker (t1/t2): {source} layer {layer}")
        m1 = torch.mm(
            tensors["hada_w1_a"].to(device=device, dtype=compute_dtype),
            tensors["hada_w1_b"].to(device=device, dtype=compute_dtype),
        )
        m2 = torch.mm(
            tensors["hada_w2_a"].to(device=device, dtype=compute_dtype),
            tensors["hada_w2_b"].to(device=device, dtype=compute_dtype),
        )
        # dim follows comfy weight_adapter/loha.py: divisor = rank dimension of w1_b
        alpha = float(tensors["alpha"]) / tensors["hada_w1_b"].shape[0] if "alpha" in tensors else 1.0
        lora_diff = (m1 * m2).reshape(w16.shape)
        w16 += ((strength * alpha) * lora_diff).type(w16.dtype)
        return w16

    if "lokr_w1" in tensors or "lokr_w1_a" in tensors:  # LoKr
        if "lokr_t2" in tensors:
            raise ValueError(f"Quantized-base merge does not support LoKr tucker (t2): {source} layer {layer}")
        # dim follows comfy weight_adapter/lokr.py: set whenever w1/w2 is factorized;
        # if both are, the w2_b assignment comes last and wins; if both are full
        # matrices it stays None -> alpha factor 1.0
        dim = None
        if "lokr_w1" in tensors:
            w1 = tensors["lokr_w1"].to(device=device, dtype=compute_dtype)
        else:
            dim = tensors["lokr_w1_b"].shape[0]
            w1 = torch.mm(
                tensors["lokr_w1_a"].to(device=device, dtype=compute_dtype),
                tensors["lokr_w1_b"].to(device=device, dtype=compute_dtype),
            )
        if "lokr_w2" in tensors:
            w2 = tensors["lokr_w2"].to(device=device, dtype=compute_dtype)
        else:
            dim = tensors["lokr_w2_b"].shape[0]
            w2 = torch.mm(
                tensors["lokr_w2_a"].to(device=device, dtype=compute_dtype),
                tensors["lokr_w2_b"].to(device=device, dtype=compute_dtype),
            )
        if "alpha" in tensors and dim is not None:
            alpha = float(tensors["alpha"]) / dim
        else:
            alpha = 1.0
        lora_diff = torch.kron(w1, w2).reshape(w16.shape)
        w16 += ((strength * alpha) * lora_diff).type(w16.dtype)
        return w16

    raise ValueError(
        f"Quantized-base merge cannot recognize the LoRA layer form: {source} layer {layer} ({sorted(tensors)})"
    )


class QuantizedMergeAdapter:
    """Lifecycle handle for merged weights, a duck-typed member of the daemon adapters list.

    - ``detach()``: restores the original weights and scales bit for bit from CPU backups (swap / unload LoRA)
    - ``network = None``: makes lycoris-side multiplier setters a safe no-op
    - ``supports_hot_reload = False``: a merge has no resident network, so the daemon's hot
      weight swap is disabled (must go detach -> merge again)

    ``backup``: ``{Linear module name: (CPU copy of weight, CPU copy of weight_scale or None)}``.
    """

    network = None
    supports_hot_reload = False

    def __init__(self, model: torch.nn.Module,
                 backup: dict[str, tuple[Tensor, Tensor | None]],
                 *, can_restore: bool = True) -> None:
        self._model = model
        self._backup = backup
        #: Whether backups were kept. False = changing LoRA cannot restore in place; detach()
        #: returns False so the daemon falls back to reloading the model (saves a model-sized CPU backup).
        self._can_restore = can_restore

    def detach(self) -> bool:
        if not self._can_restore:
            # No backup: tell the caller it cannot restore, so it reloads the model for a clean state.
            # The model reference must be dropped too: _run_generate/_run_xy still hold this handle
            # in their local adapters variable, and without clearing it the old model stays pinned in
            # VRAM during the reload (XY swapping LoRA per cell could keep three models, straight OOM)
            self._model = None
            return False
        if not self._backup:
            return True
        modules = dict(self._model.named_modules())
        for layer, (weight_cpu, scale_cpu) in self._backup.items():
            module = modules[layer]
            module.weight.data.copy_(weight_cpu.to(module.weight.device))
            if scale_cpu is not None:
                module.scale_weight.copy_(scale_cpu.to(module.scale_weight.device))
        self._backup = {}
        self._model = None
        return True
