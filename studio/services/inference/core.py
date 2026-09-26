"""Inference core -- unified implementation for loading/merging multiple LoRAs.

Consumers:
  - runtime/anima_generate.py (standalone test generation, stacking multiple LoRAs)
  - runtime/anima_train.py's in-training sampling (moved here in PR-9 commit 7)
  - runtime/anima_reg_ai.py (prior generation -- does not call apply_loras, samples
    straight from the base model)

The author of PR #17 copy-pasted a LoRA-loading implementation into both
anima_generate.py and anima_reg_ai.py, which carried two P0 bugs:
  1. rank/alpha were hardcoded to 32/32 instead of reading the top-level
     ss_network_dim/ss_network_alpha -- LoRAs trained with dim != 32 would get
     the wrong shape or the wrong alpha scaling.
  2. With multiple LoRAs, tensors from different LoRAs were added directly into
     one state_dict and fed into a single LycorisNetwork -- but LoKr's
     lokr_w1/lokr_w2 are submatrices, and summing submatrices is not the same as
     summing weight deltas, producing wrong images.

This module fixes both issues in one place:
  - read_lora_meta(): reads rank/alpha from top-level metadata, algo/factor from
    ss_network_args
  - apply_loras(): injects a separate AnimaLycorisAdapter per LoRA, controlling
    each one's contribution via LycorisNetwork.multiplier=scale; at forward time
    the multiple hooks naturally accumulate deltas, which is equivalent to
    summing weight deltas.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# Temp output directory prefix for test-generation tasks. One anima_gen_{task_id}/
# per task. Product decision: images generated on the test page are not saved --
# the supervisor deletes the whole directory when the task ends; studio also
# sweeps and cleans up leftovers on startup (in case the supervisor crashed).
GENERATE_TEMP_PREFIX = "anima_gen_"


# Fallback values when metadata is missing. Matches AnimaLycorisAdapter's defaults.
_DEFAULT_RANK = 32
_DEFAULT_ALPHA = 16.0
_DEFAULT_ALGO = "lokr"
_DEFAULT_FACTOR = 8


@dataclass
class LoRASpec:
    """Load parameters for a single LoRA."""
    path: str
    scale: float = 1.0


@dataclass
class LoRAMeta:
    """LoRA training parameters parsed from safetensors metadata.

    weight_decompose / rs_lora must faithfully replay the training-side settings
    -- otherwise the inference-side network structure won't match the file:
    DoRA would be missing its dora_scale tensor (unexpected keys), and RS-LoRA
    would miscompute effective alpha as alpha/rank instead of alpha/sqrt(rank),
    cutting the strength by a factor of sqrt(rank).

    Likewise for lora_reg_dims (regex -> rank): during training, some layers'
    rank is overridden to a custom value by pattern. If inference reconstructs
    the network using the global rank, the overridden layers' lokr_w2_a/b shapes
    immediately mismatch the checkpoint.
    """
    rank: int
    alpha: float
    algo: str
    factor: int
    weight_decompose: bool = False
    rs_lora: bool = False
    lora_reg_dims: Optional[dict[str, int]] = None
    #: The model family this artifact belongs to (D13 tag); untagged legacy
    #: artifacts are grandfathered in as anima.
    model_family: str = "anima"
    #: Whether model_family came from an explicit metadata tag. External-ecosystem
    #: files (civitai / musubi / comfy-style) don't carry our tag -- the
    #: grandfathered value is display-only; the hard cross-family rejection only
    #: applies to explicitly tagged files, and untagged files fall back to the
    #: key-matching check done during injection/merge.
    family_explicit: bool = False


def read_lora_meta(path: str) -> LoRAMeta:
    """Read LoRA training parameters from the safetensors top-level metadata.

    Matches the convention AnimaLycorisAdapter.save() writes (utils/lycoris_adapter.py):
      - top-level metadata: ss_network_dim (rank), ss_network_alpha (alpha)
      - inside the ss_network_args JSON: algo, factor, weight_decompose, rs_lora, ...

    Falls back to defaults when fields are missing or parsing fails (rank=32,
    alpha=rank, algo=lokr, factor=8, weight_decompose=False, rs_lora=False).
    """
    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
    except Exception as e:
        logger.warning(f"Failed to read LoRA metadata {path}: {e}; using default parameters")
        return LoRAMeta(_DEFAULT_RANK, _DEFAULT_ALPHA, _DEFAULT_ALGO, _DEFAULT_FACTOR)

    try:
        ss_args = json.loads(meta.get("ss_network_args", "{}"))
        if not isinstance(ss_args, dict):
            ss_args = {}
    except (ValueError, TypeError):
        ss_args = {}

    rank = _DEFAULT_RANK
    if "ss_network_dim" in meta:
        try:
            rank = int(meta["ss_network_dim"])
        except (ValueError, TypeError):
            pass

    # The common convention when alpha isn't explicit is alpha=rank (keeps a 1.0x multiplier)
    alpha = float(rank)
    if "ss_network_alpha" in meta:
        try:
            alpha = float(meta["ss_network_alpha"])
        except (ValueError, TypeError):
            pass

    algo = str(ss_args.get("algo", _DEFAULT_ALGO))
    factor = _DEFAULT_FACTOR
    if "factor" in ss_args:
        try:
            factor = int(ss_args["factor"])
        except (ValueError, TypeError):
            pass

    weight_decompose = bool(ss_args.get("weight_decompose", False))
    rs_lora = bool(ss_args.get("rs_lora", False))

    # lora_reg_dims: training overrides some layers' rank via regex; inference must
    # rebuild with the same pattern, otherwise the overridden layers' shapes won't
    # match the checkpoint. Validated as dict[str, int]; other shapes are dropped.
    lora_reg_dims: Optional[dict[str, int]] = None
    raw_reg = ss_args.get("lora_reg_dims")
    if isinstance(raw_reg, dict) and raw_reg:
        parsed: dict[str, int] = {}
        for k, v in raw_reg.items():
            try:
                parsed[str(k)] = int(v)
            except (ValueError, TypeError):
                logger.warning(f"Skipping lora_reg_dims entry (rank is not an integer): {k!r}={v!r}")
        if parsed:
            lora_reg_dims = parsed

    return LoRAMeta(
        rank=rank,
        alpha=alpha,
        algo=algo,
        factor=factor,
        weight_decompose=weight_decompose,
        rs_lora=rs_lora,
        lora_reg_dims=lora_reg_dims,
        model_family=str(ss_args.get("model_family") or "anima"),
        family_explicit=bool(ss_args.get("model_family")),
    )


_PEFT_LORA_PREFIX = "diffusion_model."
_PEFT_SUFFIX_MAP = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
    "alpha": "alpha",
}


def _normalize_peft_lora_sd(
    sd: dict,
) -> Optional[tuple[dict, int, Optional[dict[str, int]]]]:
    """Normalize PEFT/comfy key format (the civitai ecosystem's) to the
    kohya/lycoris convention.

    ``diffusion_model.{dotted layer name}.lora_A/lora_B`` -> ``lora_unet_{layer
    name with underscores}.lora_down/lora_up.weight``; a missing alpha key means
    comfy's 1.0-scale semantics, so we fill in a per-layer ``alpha = rank`` (an
    alpha/rank-style loader then gets 1.0, matching the numeric value); rank is
    inferred from the lora_A tensor shape (these files have no ss_* metadata, so
    it can't be read from the header); mixed-rank layers go into lora_reg_dims.
    Returns (normalized sd, max_rank, reg_dims); returns None for non-pure-PEFT
    shapes (kohya/lycoris files pass through unchanged).
    """
    import torch  # noqa: PLC0415

    if not sd or not all(key.startswith(_PEFT_LORA_PREFIX) for key in sd):
        return None
    grouped: dict[str, dict[str, Any]] = {}
    for key, tensor in sd.items():
        rest = key[len(_PEFT_LORA_PREFIX):]
        for peft_suffix, kohya_suffix in _PEFT_SUFFIX_MAP.items():
            if rest.endswith("." + peft_suffix):
                layer = rest[: -len(peft_suffix) - 1]
                grouped.setdefault(layer, {})[kohya_suffix] = tensor
                break
        else:
            if rest.endswith(".dora_scale"):
                raise ValueError(
                    "DoRA LoRAs in PEFT form are not supported (dora_scale is non-linear); "
                    "use a version exported in the kohya format instead."
                )
            raise ValueError(f"Unrecognized PEFT LoRA key: {key}")

    normalized: dict[str, Any] = {}
    ranks: dict[str, int] = {}
    for layer, tensors in grouped.items():
        down = tensors.get("lora_down.weight")
        up = tensors.get("lora_up.weight")
        if down is None or up is None:
            raise ValueError(f"This PEFT LoRA layer is missing its lora_A/lora_B pair: {layer}")
        rank = int(down.shape[0])
        ranks[layer] = rank
        prefix = f"lora_unet_{layer.replace('.', '_')}"
        normalized[f"{prefix}.lora_down.weight"] = down
        normalized[f"{prefix}.lora_up.weight"] = up
        alpha = tensors.get("alpha")
        normalized[f"{prefix}.alpha"] = (
            alpha if alpha is not None else torch.tensor(float(rank))
        )

    max_rank = max(ranks.values())
    reg_dims = {
        f"lora_unet_{layer.replace('.', '_')}": rank
        for layer, rank in ranks.items()
        if rank != max_rank
    } or None
    return normalized, max_rank, reg_dims


def apply_loras(
    model: Any,
    specs: Sequence[LoRASpec],
    device: str,
    dtype: Any,
    family_id: str = "anima",
) -> list[Any]:
    """Inject a separate AnimaLycorisAdapter for each LoRA; forward-time hooks
    accumulate the deltas.

    dtype is the compute dtype for the LoRA network / tensors. ComfyUI's weight
    adapter computes the LoRA delta at fp32 intermediate precision; the test
    generation parity path should pass torch.float32.

    The multiplier field controls each LoRA's contribution weight (the caller's
    scale):
      - LycorisNetwork.multiplier is the global multiplier read at forward time
      - a per-lora-module fallback is also set (different lycoris versions read
        it from different places)

    Returns the adapter list -- the caller **must keep a reference** to it,
    otherwise once Python's GC runs, the LycorisNetwork inside each
    AnimaLycorisAdapter gets collected too, and the forward hooks on the model
    stop working (lycoris holds the network via a closure).
    """
    from safetensors import safe_open

    from utils.lycoris_adapter import AnimaLycorisAdapter

    # An fp8-quantized base model follows ComfyUI's merge semantics (dequant ->
    # add delta -> stochastic-rounding write-back, seed = CRC32 of the layer
    # name) -- injecting fp8 weights directly via a lycoris hook would either
    # crash on dtype or produce values inconsistent with Comfy. Currently only
    # the krea2 loader produces fp8 weights (the Anima loader rejects fp8).
    fp8_merge = False
    if specs:
        from training.families.krea2.quant_fp8 import model_has_fp8_layers  # noqa: PLC0415

        fp8_merge = model_has_fp8_layers(model)

    merge_sources: list[tuple[dict, float, str]] = []
    adapters: list[Any] = []
    for spec in specs:
        path = spec.path or ""
        if not path or not Path(path).exists():
            logger.warning(f"LoRA path does not exist, skipping: {path!r}")
            continue

        meta = read_lora_meta(path)
        # Cross-family fail-fast (A5, matches the training-side resume_lora
        # check): injecting a krea2 LoRA into an anima base model (or vice
        # versa) with the wrong preset silently produces a broken result where
        # every key misses. Hard-reject only on an **explicit tag** -- external-
        # ecosystem files (civitai/musubi/comfy-style) carry no model_family tag
        # of ours, so the grandfathered value can't be used to reject; untagged
        # files are allowed through and fall back to the key-matching check
        # during injection/merge below (an all-miss raises an error, not silent).
        if meta.family_explicit and meta.model_family != family_id:
            raise ValueError(
                f"This LoRA belongs to a different model family: {Path(path).name} is for "
                f"'{meta.model_family}', while the current base model family is '{family_id}'. "
                f"Use a LoRA from the same family, or switch the base model."
            )
        sd_raw: dict = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd_raw[k] = f.get_tensor(k)

        # Normalize external-ecosystem PEFT key format (common with civitai):
        # convert to kohya keys + infer rank/reg_dims from tensor shapes (these
        # files carry no ss_* metadata at all, so meta's rank=32 is just the
        # fallback default and can't be relied on)
        rank, alpha, algo = meta.rank, meta.alpha, meta.algo
        reg_dims = meta.lora_reg_dims
        peft = _normalize_peft_lora_sd(sd_raw)
        if peft is not None:
            sd_raw, rank, reg_dims = peft
            alpha = float(rank)   # per-layer alpha is already filled into sd; the global value is only for building the network
            algo = "lora"         # PEFT's two matrices = plain LoRA

        if fp8_merge:
            # merge is a weight-level linear operation, so it only honors
            # per-layer alpha/dim scaling. rs_lora artifacts need no special
            # case: lycoris bakes the sqrt(rank) correction into the per-layer
            # alpha key when saving (register_buffer("alpha", alpha*dim/sqrt(dim)),
            # same for locon/loha/lokr), so a standard alpha/dim merge already
            # yields alpha/sqrt(rank) -- numerically consistent with the bf16
            # injection path (network built with scale=alpha/sqrt(r)). DoRA
            # (column-norm normalization, non-linear) merge would need comfy's
            # weight_decompose semantics, which isn't implemented yet, so it
            # stays rejected.
            if meta.weight_decompose:
                raise ValueError(
                    f"An fp8-quantized base model cannot load a LoRA trained with DoRA (weight_decompose): "
                    f"LoRA: {Path(path).name}. Use a bf16 base model instead."
                )
            merge_sources.append((sd_raw, float(spec.scale), Path(path).name))
            continue
        from training.families import get_family  # noqa: PLC0415

        adapter = AnimaLycorisAdapter(
            preset=get_family(family_id).lora_preset(),
            algo=algo,
            rank=rank,
            alpha=alpha,
            factor=meta.factor,
            weight_decompose=meta.weight_decompose,
            rs_lora=meta.rs_lora,
            lora_reg_dims=reg_dims,
        )
        adapter.inject(model)
        if adapter.network is not None:
            adapter.network.to(device=device, dtype=dtype)

        if adapter.network is not None:
            adapter.network.multiplier = float(spec.scale)
            for lora in getattr(adapter.network, "loras", []):
                if hasattr(lora, "multiplier"):
                    lora.multiplier = float(spec.scale)

        sd = {k: v.to(device=device, dtype=dtype) for k, v in sd_raw.items()}

        result = adapter.load_state_dict(sd, strict=False)
        missing = len(getattr(result, "missing_keys", []) or [])
        unexpected = len(getattr(result, "unexpected_keys", []) or [])
        if sd and unexpected >= len(sd):
            # None of the keys were consumed by the LoRA network = a different-
            # family file or a key format this path doesn't support (this is the
            # content-matching fallback for untagged files, so it doesn't
            # silently produce an image with no LoRA effect)
            raise ValueError(
                f"This LoRA does not match the current base model: none of the keys in {Path(path).name} line up"
                f" (it may belong to another model family, or use a key format this path does not support yet)."
            )
        logger.info(
            f"Loaded LoRA: {Path(path).name} "
            f"(algo={algo}, rank={rank}, alpha={alpha}, "
            f"scale={spec.scale}; missing={missing}, unexpected={unexpected})"
        )
        adapters.append(adapter)

    if merge_sources:
        from training.families.krea2.lora_fp8_merge import (  # noqa: PLC0415
            merge_loras_into_fp8_model,
        )

        # A single handle covers all LoRAs (merge happens once); when the daemon
        # swaps LoRAs / changes scale, detach() restores from the backup and re-merges
        return [merge_loras_into_fp8_model(model, merge_sources)]
    return adapters


# ---------------------------------------------------------------------------
# Generate test-image output: temp directory management
# ---------------------------------------------------------------------------


def generate_tempdir(task_id: int) -> Path:
    """Temp output directory path for a single generate task.

    Lives under the system tempdir (isolated from studio_data); cleaned up when
    the task finishes.
    """
    return Path(tempfile.gettempdir()) / f"{GENERATE_TEMP_PREFIX}{task_id}"


def cleanup_generate_tempdir(task_id: int) -> None:
    """Clean up a single tempdir when the task ends. A missing directory is a
    no-op (safe to call even for non-generate tasks)."""
    d = generate_tempdir(task_id)
    if not d.exists():
        return
    try:
        shutil.rmtree(d)
        logger.info(f"cleaned generate tempdir: {d}")
    except OSError as e:
        logger.warning(f"failed to clean {d}: {e}")


def cleanup_stale_generate_tempdirs() -> None:
    """Sweep and clean up any leftover anima_gen_* directories at startup (in case
    the supervisor crashed and leaked one)."""
    parent = Path(tempfile.gettempdir())
    if not parent.exists():
        return
    for d in parent.glob(f"{GENERATE_TEMP_PREFIX}*"):
        if not d.is_dir():
            continue
        try:
            shutil.rmtree(d)
            logger.info(f"cleaned stale generate tempdir: {d}")
        except OSError as e:
            logger.warning(f"failed to clean stale {d}: {e}")
