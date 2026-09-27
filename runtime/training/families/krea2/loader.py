"""Strict single-file Krea2 checkpoint inspection and loading.

Accepts the Comfy/musubi single-file layout (not the diffusers sharded
layout). **Raw and Turbo have identical structure (same keys, same
shapes) -- this loader treats them identically and cannot, and doesn't try
to, tell them apart**; the "Turbo isn't recommended as a training base"
guard lives in the studio-side catalog variant's purpose metadata (P4-4),
not in the loading layer.

The meta-device + ``assign=True`` loading strategy was adapted from
kohya-ss/musubi-tuner (Apache-2.0):
Copyright 2026 Kohya S. and musubi-tuner contributors.
https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/krea2/krea2_utils.py

Fingerprint validation, prefix handling, and diagnostics are original to this
repository.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

from modeling.krea2 import KREA2_CONFIG, Krea2Config, SingleStreamDiT
from training.families.krea2.quant_fp8 import (
    parse_quantization_metadata,
    patch_fp8_linears,
)


logger = logging.getLogger(__name__)

_PREFIX_CANDIDATES = (
    "",
    "diffusion_model.",
    "model.diffusion_model.",
    "module.",
    "model.",
    "transformer.",
)
_FLOAT_DTYPES = {
    "BF16",
    "F16",
    "F32",
    "F64",
}
# fp8 weights stay resident as-is + Linear forward dequants layer by layer
# (quant_fp8). Both inference and training (fp8_base: base model frozen,
# LoRA params full precision, the kohya/musubi ecosystem's standard approach)
# are supported -- weights are never silently upcast back to bf16 (that would
# lose precision for zero VRAM benefit, A7'/C13).
_FP8_DTYPES = {"F8_E4M3", "F8_E5M2"}


@dataclass(frozen=True)
class Krea2CheckpointInfo:
    path: Path
    prefix: str
    key_count: int
    parameter_count: int


def _checkpoint_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Krea2 checkpoint does not exist: {resolved}")
    if resolved.is_dir():
        raise ValueError(
            "Krea2 loader needs a single-file raw.safetensors, not a "
            f"diffusers transformer shard directory: {resolved}"
        )
    if resolved.suffix.lower() != ".safetensors":
        raise ValueError(f"Krea2 checkpoint must be a .safetensors file: {resolved}")
    return resolved


def _expected_state(config: Krea2Config):
    with torch.device("meta"):
        model = SingleStreamDiT(config)
    return model, {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
    }


def _choose_prefix(source_keys: list[str], expected_keys: set[str]) -> str:
    best_prefix = ""
    best_score = -1
    for prefix in _PREFIX_CANDIDATES:
        normalized = {
            key[len(prefix):] if prefix and key.startswith(prefix) else key
            for key in source_keys
        }
        score = len(normalized & expected_keys)
        if score > best_score:
            best_prefix = prefix
            best_score = score
    if best_score <= 0:
        diffusers_hint = any(
            key.startswith(("transformer_blocks.", "img_in.", "text_fusion."))
            for key in source_keys
        )
        hint = (
            "; detected diffusers-style keys, use the raw.safetensors from the repo root instead"
            if diffusers_hint
            else ""
        )
        raise ValueError(f"Not a recognizable Krea2 checkpoint: zero structural-fingerprint matches{hint}")
    return best_prefix


def _normalize_key(key: str, prefix: str) -> str:
    return key[len(prefix):] if prefix and key.startswith(prefix) else key


def _format_key_delta(label: str, keys: set[str]) -> str:
    sample = ", ".join(sorted(keys)[:5])
    suffix = " ..." if len(keys) > 5 else ""
    return f"{label} {len(keys)}: {sample}{suffix}"


def _inspect(
    path: str | Path,
    config: Krea2Config,
    *,
    expected_shapes: dict[str, tuple[int, ...]] | None = None,
    allow_fp8: bool = False,
) -> tuple[Krea2CheckpointInfo, dict[str, str], dict[str, str]]:
    """Validate the checkpoint and return (info, weight key mapping, fp8 scale key mapping).

    When ``allow_fp8=True`` (the inference path): fp8 weights are accepted;
    ``{layer}.weight_scale`` F32 scalar keys are collected separately (the
    fp8_scaled form) and excluded from the key-set comparison.
    """
    checkpoint = _checkpoint_path(path)
    if expected_shapes is None:
        _, expected_shapes = _expected_state(config)
    expected_keys = set(expected_shapes)

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        quant_meta = (
            parse_quantization_metadata(handle.metadata()) if allow_fp8 else {}
        )
        all_source_keys = list(handle.keys())
        prefix = _choose_prefix(all_source_keys, expected_keys)
        # fp8_scaled's per-layer scale keys: after normalization they look like
        # blocks.N.xxx.weight_scale. Only stripped out when allow_fp8; callers
        # that explicitly pass allow_fp8=False still get them flagged as
        # "unexpected".
        scale_to_source: dict[str, str] = {}
        source_keys = []
        for source_key in all_source_keys:
            normalized = _normalize_key(source_key, prefix)
            if allow_fp8 and normalized.endswith(".weight_scale"):
                layer = normalized[: -len(".weight_scale")]
                if f"{layer}.weight" in expected_keys:
                    scale_to_source[layer] = source_key
                    continue
            source_keys.append(source_key)
        normalized_to_source: dict[str, str] = {}
        for source_key in source_keys:
            normalized = _normalize_key(source_key, prefix)
            if normalized in normalized_to_source:
                raise ValueError(f"Krea2 checkpoint key collision after prefix normalization: {normalized}")
            normalized_to_source[normalized] = source_key

        actual_keys = set(normalized_to_source)
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys
        errors = []
        if missing:
            errors.append(_format_key_delta("missing", missing))
        if unexpected:
            errors.append(_format_key_delta("unexpected", unexpected))

        shape_mismatches = []
        dtype_mismatches = []
        fp8_keys = []
        for normalized in sorted(expected_keys & actual_keys):
            source_key = normalized_to_source[normalized]
            tensor_slice = handle.get_slice(source_key)
            actual_shape = tuple(tensor_slice.get_shape())
            expected_shape = expected_shapes[normalized]
            if actual_shape != expected_shape:
                shape_mismatches.append(
                    f"{normalized}: {actual_shape} != {expected_shape}"
                )
            actual_dtype = tensor_slice.get_dtype()
            if actual_dtype in _FP8_DTYPES:
                fp8_keys.append(normalized)
            elif actual_dtype not in _FLOAT_DTYPES:
                dtype_mismatches.append(f"{normalized}: {actual_dtype}")
        if fp8_keys and not allow_fp8:
            raise ValueError(
                f"Krea2 checkpoint contains fp8 params ({len(fp8_keys)}, e.g. "
                f"{fp8_keys[0]}), but the caller requires full-precision weights."
            )
        # fp8_scaled consistency: a layer with a scale must have fp8 weights;
        # a layer declared in metadata with neither fp8 weights nor a scale is a malformed file
        if allow_fp8:
            fp8_set = set(fp8_keys)
            for layer in scale_to_source:
                if f"{layer}.weight" not in fp8_set:
                    errors.append(f"{layer} has weight_scale but its weight isn't fp8")
            for layer in quant_meta:
                if f"{layer}.weight" not in fp8_set:
                    errors.append(f"metadata declares quantized layer {layer} but its weight isn't fp8")
        if shape_mismatches:
            sample = "; ".join(shape_mismatches[:5])
            suffix = " ..." if len(shape_mismatches) > 5 else ""
            errors.append(f"{len(shape_mismatches)} shape mismatches: {sample}{suffix}")
        if dtype_mismatches:
            sample = "; ".join(dtype_mismatches[:5])
            suffix = " ..." if len(dtype_mismatches) > 5 else ""
            errors.append(f"{len(dtype_mismatches)} non-float tensors: {sample}{suffix}")
        if errors:
            raise ValueError("Krea2 checkpoint structural fingerprint mismatch; " + "; ".join(errors))

    parameter_count = sum(
        math_product(shape) for shape in expected_shapes.values()
    )
    return (
        Krea2CheckpointInfo(
            path=checkpoint,
            prefix=prefix,
            key_count=len(source_keys),
            parameter_count=parameter_count,
        ),
        normalized_to_source,
        scale_to_source,
    )


def math_product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def inspect_krea2_checkpoint(
    path: str | Path,
    *,
    config: Krea2Config = KREA2_CONFIG,
) -> Krea2CheckpointInfo:
    """Validate keys and shapes from the safetensors header without reading payloads.

    Both bf16 and fp8 (scaled / plain cast) are valid checkpoint forms.
    """
    info, _, _ = _inspect(path, config, allow_fp8=True)
    return info


def checkpoint_contains_fp8(path: str | Path) -> bool:
    """Lightweight probe: checks the safetensors header for fp8 weights (doesn't read the payload).

    Used for guardrails at training startup (combinations like an fp8 base
    model + grad_checkpoint disabled need to fail fast, not blow up after a
    13GB load). Returns False for non-safetensors / read failures -- the real
    structural check is the loader's job.
    """
    try:
        checkpoint = _checkpoint_path(path)
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            return any(
                handle.get_slice(key).get_dtype() in _FP8_DTYPES
                for key in handle.keys()
            )
    except Exception:
        return False


def _swapped_block_prefixes(config: Krea2Config, blocks_to_swap: int) -> tuple[str, ...]:
    """state_dict key prefixes of the swapped-out layers (the last N layers, same convention as PinnedBlockSwap)."""
    if blocks_to_swap <= 0:
        return ()
    first = max(config.layers - blocks_to_swap, 0)
    return tuple(f"blocks.{i}." for i in range(first, config.layers))


@lru_cache(maxsize=None)
def _swapped_param_counts(
    blocks_to_swap: int, config: Krea2Config,
) -> tuple[int, int]:
    """(swapped-out layer param count, full-model param count). Counts params
    on a meta model, so no disk reads and no VRAM use.

    Cached: block-swap preflight asks for the ratio once for each candidate
    value from 0..28, and without caching that would build 29 meta models.
    ``Krea2Config`` is a frozen dataclass (hashable), and the return value is
    two ints, so caching carries zero memory risk.
    """
    prefixes = _swapped_block_prefixes(config, blocks_to_swap)
    with torch.device("meta"):
        probe = SingleStreamDiT(config)
    swapped = total = 0
    for name, param in probe.named_parameters():
        total += param.numel()
        if prefixes and name.startswith(prefixes):
            swapped += param.numel()
    return swapped, total


def swapped_param_ratio(
    blocks_to_swap: int, *, config: Krea2Config = KREA2_CONFIG,
) -> float:
    """Fraction of the full model's parameters accounted for by the swapped-out
    layers -- **use this for the VRAM budget discount, not a byte count**.

    The discount must be dtype-independent: `check_load_budget`'s need comes
    from the weight file's actual size, and an fp8 checkpoint is only half
    the size of bf16. If the discount were computed in bf16 bytes, it would
    over-discount the fp8 case (need 13GB minus 11.3GB -> thinks only 1.7GB
    is needed, but 7.2GB is actually resident), making the guardrail
    meaningless. Multiplying the ratio by the file's actual size is correct
    for both precisions.
    """
    if blocks_to_swap <= 0:
        return 0.0
    swapped, total = _swapped_param_counts(blocks_to_swap, config)
    return (swapped / total) if total else 0.0


def estimate_swapped_bytes(
    blocks_to_swap: int,
    dtype: torch.dtype,
    *,
    config: Krea2Config = KREA2_CONFIG,
) -> int:
    """Byte count of the swapped-out layers' weights (**for the pinned-memory budget only**).

    Estimated here using the compute dtype, which overestimates for an fp8
    base model -- but for the pinned budget, overestimating is the safe
    direction (rejects early rather than letting the user hit a pinned-memory
    wall later). Don't use this for the VRAM discount; see
    ``swapped_param_ratio``'s docstring.
    """
    if blocks_to_swap <= 0:
        return 0
    swapped, _total = _swapped_param_counts(blocks_to_swap, config)
    return swapped * torch.empty(0, dtype=dtype).element_size()


def _swapped_bytes_from_checkpoint(
    checkpoint: Path,
    prefixes: tuple[str, ...],
    normalized_to_source: dict,
) -> int:
    """**Actual** byte count of the swapped-out layers in the checkpoint (header-only read, no data loading).

    Must use the actual dtype, not the compute dtype: an fp8 checkpoint is
    only half the size of bf16, and estimating from bf16 would put 28 layers
    at 22.6GB (actually 11.3GB), which on a 37.5GB machine would hit the 60%
    safety line and get **incorrectly rejected** -- exactly blocking B12's
    target configuration. Overestimating here isn't conservative, it's a
    false negative.

    Returns 0 when the header can't be read; the caller falls back to the
    compute-dtype-based estimate.
    """
    total = 0
    try:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            for normalized, source_key in normalized_to_source.items():
                if not normalized.startswith(prefixes):
                    continue
                slice_ = handle.get_slice(source_key)
                numel = 1
                for dim in slice_.get_shape():
                    numel *= dim
                total += numel * _dtype_size(slice_.get_dtype())
    except Exception:  # noqa: BLE001
        return 0
    return total


def _swapped_pinned_bytes(
    checkpoint: Path,
    prefixes: tuple[str, ...],
    normalized_to_source: dict,
    dtype: torch.dtype,
) -> int:
    """Byte count that will **actually be pinned** for the swapped-out layers
    (header-only read), used by ``PinnedPacker`` for its preallocation plan.

    Differs from ``_swapped_bytes_from_checkpoint`` (whose convention is the
    budget-guardrail figure = the checkpoint's raw bytes): this must mirror
    ``load_krea2_model``'s actual write-out rule -- fp8 tensors are pinned
    as-is (at checkpoint dtype), everything else is first cast to the
    compute dtype and then pinned (at ``dtype``). The planned number needs to
    be **exact**: overestimating wastes pinned memory (the packer
    preallocates it all at once based on this number), underestimating opens
    extra overflow chunks. Returns 0 when the header can't be read (the
    packer falls back to allocating chunks on demand, no worse than pinning
    tensor by tensor).
    """
    compute_size = torch.empty(0, dtype=dtype).element_size()
    total = 0
    try:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            for normalized, source_key in normalized_to_source.items():
                if not normalized.startswith(prefixes):
                    continue
                slice_ = handle.get_slice(source_key)
                numel = 1
                for dim in slice_.get_shape():
                    numel *= dim
                name = str(slice_.get_dtype()).upper()
                per_elem = _dtype_size(name) if name.startswith("F8_") else compute_size
                total += numel * per_elem
    except Exception:  # noqa: BLE001
        return 0
    return total


#: safetensors dtype string -> byte width (the header stores a string, not a torch.dtype)
_DTYPE_BYTES = {
    "F64": 8, "I64": 8,
    "F32": 4, "I32": 4,
    "F16": 2, "BF16": 2, "I16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}


def _dtype_size(name: str) -> int:
    return _DTYPE_BYTES.get(str(name).upper(), 2)


def _check_swap_budget(
    config: Krea2Config,
    blocks_to_swap: int,
    dtype: torch.dtype,
    normalized_to_source: dict,
    checkpoint: Path,
) -> None:
    """Check the memory-budget guardrail before writing swapped-out layers to
    pinned memory (B6: failure raises, no silent degradation).

    Called **before any weight is read** -- failure must be fail-fast, not
    blow up halfway through the copy.
    """
    from training.sysmem import check_pinned_budget

    prefixes = _swapped_block_prefixes(config, blocks_to_swap)
    need = _swapped_bytes_from_checkpoint(checkpoint, prefixes, normalized_to_source)
    if need <= 0:  # header couldn't be read: fall back to a compute-dtype estimate
        need = estimate_swapped_bytes(blocks_to_swap, dtype, config=config)
    check_pinned_budget(need, blocks=blocks_to_swap)


def load_krea2_model(
    path: str | Path,
    device: str | torch.device,
    dtype: torch.dtype,
    *,
    config: Krea2Config = KREA2_CONFIG,
    purpose: str = "train",
    blocks_to_swap: int = 0,
) -> SingleStreamDiT:
    """Strict-load a single-file Krea2 checkpoint into a frozen meta-created model.

    fp8 weights (both the plain-cast and fp8_scaled forms) are accepted for
    both inference and training: fp8 tensors stay resident in VRAM as-is, and
    the Linear forward dequants to the compute dtype layer by layer (ComfyUI
    parity, see quant_fp8). Training here means the kohya/musubi ecosystem's
    fp8_base semantics -- the base model is frozen with no gradients, LoRA
    params are full precision; the VRAM savings depend on grad checkpointing
    (the temporary dequantized weights get freed at the end of each recompute
    segment), and that constraint is enforced by a startup check in the
    trainer (phases/models). When ``blocks_to_swap`` > 0 (block swap, see
    docs/design/block-swap.md): the last N layers' weights are **loaded
    directly into CPU pinned memory**, never touching VRAM -- this is what
    makes "running K2 on a 12/16GB consumer card" possible; loading
    everything onto the GPU first and then moving it back down would still
    peak at the full model's size. Once loaded, these CPU tensors are taken
    over in place by ``training.block_swap.PinnedBlockSwap``.

    ``purpose`` currently doesn't affect loading behavior; kept for semantic
    annotation at call sites.
    """
    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError(f"Krea2 loader does not support dtype={dtype}")
    target_device = torch.device(device)
    if target_device.type == "meta":
        raise ValueError("Krea2 loader's target device cannot be meta")
    allow_fp8 = True

    model, expected_shapes = _expected_state(config)
    _, normalized_to_source, scale_to_source = _inspect(
        path,
        config,
        expected_shapes=expected_shapes,
        allow_fp8=allow_fp8,
    )

    state_dict = {}
    fp8_scales: dict[str, torch.Tensor | None] = {}
    checkpoint = _checkpoint_path(path)

    swapped_prefixes = _swapped_block_prefixes(config, blocks_to_swap)
    packer = None
    if swapped_prefixes:
        _check_swap_budget(
            config, blocks_to_swap, dtype, normalized_to_source, checkpoint,
        )
        # Swapped-out layers are NOT pinned tensor by tensor (the host
        # allocator rounds up to a power of 2, wasting up to 1.47x pinned
        # memory, which on a 32GB machine directly hits Windows' lockable-
        # memory ceiling). Instead, preallocate large power-of-2 chunks
        # sized to the exact total and slice views out of them -- allocation
        # only happens after the budget guardrail passes, and a failed
        # allocation fails fast.
        from training.block_swap import PinnedPacker

        packer = PinnedPacker(_swapped_pinned_bytes(
            checkpoint, swapped_prefixes, normalized_to_source, dtype,
        ))

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for normalized, source_key in normalized_to_source.items():
            tensor = handle.get_tensor(source_key)
            if not tensor.is_floating_point():
                raise ValueError(
                    f"Krea2 checkpoint contains a non-float param {source_key}: {tensor.dtype}"
                )
            # Swapped-out layers never go to the GPU: they land directly in CPU pinned memory (see docstring)
            swapped = normalized.startswith(swapped_prefixes) if swapped_prefixes else False
            if allow_fp8 and tensor.dtype in (
                torch.float8_e4m3fn, torch.float8_e5m2,
            ):
                # fp8 stays resident as-is (this is where the VRAM savings come from), no dtype cast
                state_dict[normalized] = (
                    packer.pin(tensor) if swapped
                    else tensor.to(device=target_device)
                )
                if normalized.endswith(".weight"):
                    layer = normalized[: -len(".weight")]
                    fp8_scales.setdefault(layer, None)
            elif swapped:
                state_dict[normalized] = packer.pin(tensor, dtype=dtype)
            else:
                state_dict[normalized] = tensor.to(
                    device=target_device,
                    dtype=dtype,
                )
        for layer, source_key in scale_to_source.items():
            fp8_scales[layer] = handle.get_tensor(source_key).to(
                device=target_device,
            )

    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    model.requires_grad_(False)
    if packer is not None:
        logger.info(
            "block swap pinned: %.2f GB of weights packed into %d chunks, %.2f GB actually locked (%d overflow chunks)",
            packer.packed_bytes / 1024 ** 3, packer.num_chunks,
            packer.allocated_bytes / 1024 ** 3, packer.overflow_chunks,
        )
    if fp8_scales:
        # scale always goes on the compute device: swapped-out layers' weights
        # are on CPU right now, and following them would cause a device
        # mismatch at forward time (see patch_fp8_linears docstring)
        patch_fp8_linears(model, fp8_scales, device=target_device)
    return model
