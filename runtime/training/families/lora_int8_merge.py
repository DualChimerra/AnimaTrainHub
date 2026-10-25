"""LoRA merge and write-back for ConvRot int8 bases (ComfyUI semantics, family-neutral layer).

ComfyUI does not run LoRA on an int8 base as a bypass at inference: it merges LoRA into the
weights at load time and re-quantizes them (the classic ModelPatcher path
``patch_weight_to_device``). This module reproduces it (oracle = the author's local ComfyUI
0.41.0 + comfy_kitchen 0.2.38 eager backend, MOSS design §5.9 / M9):

    W16  = dequant(q, s).to(fp16)                       # scale multiplied in fp32, Hᵀ per group
    W16 += ((strength * alpha/dim) * ΔW_fp32).to(fp16)  # per LoRA, comfy calculate_weight
    q', s' = requantize_convrot(W16, seed)              # rotate in fp16, row-wise amax/127, stochastic rounding

- fp16 = comfy ``lora_compute_dtype`` (should_use_fp16 on modern GPUs).
- seed = ``string_to_seed("diffusion_model.{comfy key}.weight")``. A merge unit is one ComfyUI
  weight; families whose Comfy files fuse several of this project's Linear layers into one
  weight (MOSS qkv) pass ``merge_units`` so the members are merged and re-quantized as one block.
- CUDA and CPU random sequences differ, so bit-exact parity holds only on the GPU (as for Krea2 fp8).

Used only for inference (training uses the LoRA bypass on the int8 forward). Shared by MOSS and
Krea2 under the "no cross-family imports" rule (families/README.md). Attribution in THIRD_PARTY_NOTICES.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
from torch import Tensor

from training.families.comfy_lora_merge import (
    LORA_PREFIX,
    QuantizedMergeAdapter,
    apply_lora_delta,
    group_lora_layers,
    string_to_seed,
)
from training.families.quant_int8 import dequantize, requantize_convrot

logger = logging.getLogger(__name__)

#: ComfyUI lora_compute_dtype (modern NVIDIA GPUs)
_MERGE_DTYPE = torch.float16

#: (model, LoRA-targeted int8 layer names) -> {ComfyUI weight key: member layers in row order}
MergeUnits = Callable[[torch.nn.Module, set[str]], dict[str, list[str]]]


def _single_layer_units(model: torch.nn.Module, layer_names: set[str]) -> dict[str, list[str]]:
    del model
    return {name: [name] for name in sorted(layer_names)}


def merge_loras_into_int8_model(
    model: torch.nn.Module,
    sources: list[tuple[dict[str, Tensor], float, str]],
    *,
    compute_dtype: torch.dtype = torch.float32,
    chunk_rows: int | None = None,
    keep_backup: bool = True,
    merge_units: MergeUnits | None = None,
) -> QuantizedMergeAdapter:
    """Merge several LoRAs into an int8 convrot base with ComfyUI semantics and write the re-quantized weights back.

    ``sources``: [(state_dict, strength, source name), ...], order = mount order = comfy patch
    order. ``compute_dtype``: precision of ΔW, fp32 = ComfyUI default (the ``lora_merge_precision``
    setting can choose bf16 to lower the peak). ``merge_units``: groups layers that share one
    ComfyUI weight (default: every layer is its own unit). Returns a restore handle holding CPU
    backups of the original int8 weights and scales.
    """
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"LoRA merge compute_dtype supports only fp32/bf16, got: {compute_dtype}")
    module_index = {
        name.replace(".", "_"): (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    # ComfyUI ``.diff`` patches (trained norm weights) can target any module with a weight
    weight_index = {
        name.replace(".", "_"): (name, module)
        for name, module in model.named_modules()
        if isinstance(getattr(module, "weight", None), torch.Tensor)
    }

    per_layer: dict[str, list[tuple[dict[str, Tensor], float, str]]] = {}
    # Layers that stay floating point in an int8 base (the text refiner and the output head /
    # modulations when quantizing on load, Krea2's high-precision first/last/tmlp/txtfusion):
    # merged directly, no re-quantization
    dense_layers: dict[str, list[tuple[dict[str, Tensor], float, str]]] = {}
    missing: list[str] = []
    for sd, strength, source in sources:
        grouped = group_lora_layers(sd)
        if not grouped:
            raise ValueError(f"LoRA file has no {LORA_PREFIX}* layers: {source}")
        for layer_key, tensors in grouped.items():
            entry = (weight_index if "diff" in tensors else module_index).get(layer_key)
            if entry is None:
                missing.append(f"{source}:{layer_key}")
                continue
            name, module = entry
            target = per_layer if module.weight.dtype == torch.int8 else dense_layers
            target.setdefault(name, []).append((tensors, strength, source))

    if missing and not per_layer and not dense_layers:
        raise ValueError(f"No LoRA layer matches the current model: {missing[:5]} ...")
    if missing:
        logger.warning(
            "int8 merge: %d LoRA layer(s) have no matching layer in the model and "
            "were skipped (%s%s) — those LoRA layers do not affect the output",
            len(missing), missing[:5], " ..." if len(missing) > 5 else "",
        )

    modules = dict(model.named_modules())
    # Compute device: under block swap the offloaded weights are CPU-pinned masters, while
    # weight_scale always lives on the compute device. Compute on the GPU; copy_() writes back
    # across devices into the masters (same as the Krea2 fp8 merge)
    merge_device = next(
        (p.device for p in model.parameters() if p.device.type == "cuda"), None,
    )
    # Like ComfyUI, the dequantized fp32 matmul does not use TF32 (off by default in torch; enforced here)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    backup: dict[str, tuple[Tensor, Tensor | None]] = {}
    units = (merge_units or _single_layer_units)(model, set(per_layer))
    try:
        for unit, members in units.items():
            lins = [modules[n] for n in members]
            groups = {int(m._convrot_groupsize) for m in lins}
            if len(groups) != 1:
                raise ValueError(f"int8 merge: members of {unit} have different convrot group sizes {sorted(groups)}")
            group = groups.pop()
            if keep_backup:
                for n, m in zip(members, lins):
                    backup[n] = (
                        m.weight.detach().to("cpu", copy=True),
                        m.scale_weight.detach().to("cpu", copy=True),
                    )
            device = merge_device or lins[0].weight.device
            q = torch.cat([m.weight.detach().to(device) for m in lins], dim=0)
            s = torch.cat([m.scale_weight.detach().to(device).reshape(-1, 1) for m in lins], dim=0)
            w16 = dequantize(q, s, group).to(_MERGE_DTYPE)
            del q, s
            row = 0
            for n, m in zip(members, lins):
                rows = int(m.out_features)
                for tensors, strength, source in per_layer.get(n, ()):
                    apply_lora_delta(
                        w16[row:row + rows], tensors, strength, n, source, compute_dtype, chunk_rows,
                    )
                row += rows
            seed = string_to_seed(f"diffusion_model.{unit}.weight")
            q_new, s_new = requantize_convrot(w16, group, seed=seed)
            del w16
            row = 0
            for m in lins:
                rows = int(m.out_features)
                m.weight.data.copy_(q_new[row:row + rows])
                m.scale_weight.copy_(s_new[row:row + rows].reshape(m.scale_weight.shape))
                row += rows
            del q_new, s_new
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for name, entries in dense_layers.items():
            module = modules[name]
            if keep_backup:
                backup[name] = (module.weight.detach().to("cpu", copy=True), None)
            device = merge_device or module.weight.device
            w = module.weight.detach().to(device=device, dtype=torch.float32)
            for tensors, strength, source in entries:
                if "diff" in tensors:
                    w += strength * tensors["diff"].to(device=device, dtype=torch.float32).reshape(w.shape)
                else:
                    apply_lora_delta(w, tensors, strength, name, source, compute_dtype, chunk_rows)
            module.weight.data.copy_(w.to(module.weight.dtype))
            del w
    except Exception:
        QuantizedMergeAdapter(model, backup).detach()
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32

    logger.info(
        "INT8 LoRA merge: %d adapters, %d int8 layers, %d floating-point layers",
        len(sources), len(per_layer), len(dense_layers),
    )
    logger.debug(
        "int8_merge: units=%d delta=%s backup=%s", len(units),
        str(compute_dtype).removeprefix("torch."),
        "kept (detach can restore)" if keep_backup else "none (reload to swap LoRA)",
    )
    return QuantizedMergeAdapter(model, backup, can_restore=keep_backup)
