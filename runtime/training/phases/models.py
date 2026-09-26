"""models_phase: paths + family-owned weights + LoRA injection.

Cached-varlen families may defer their large DiT until dataset captions have
been cached and the text encoder released. ``finish(ctx)`` closes that deferred
half immediately after ``text_cache`` and before optimizer construction.
"""

from __future__ import annotations

import logging
from pathlib import Path

from training import block_swap_preflight
from training.context import TrainingContext
from training.families import resolve_family
from training.families.anima import ANIMA_SPEC as _ANIMA_SPEC
from training.sysmem import log_vram
from training.model_loading import (
    find_diffusion_pipe_root,
    resolve_path_best_effort,
)


logger = logging.getLogger(__name__)


def _resolve_paths(ctx: TrainingContext) -> None:
    args = ctx.args
    ctx.repo_root = find_diffusion_pipe_root()
    logger.info("model code path: %s", ctx.repo_root)

    phases_dir = Path(__file__).resolve().parent
    training_dir = phases_dir.parent
    runtime_dir = training_dir.parent
    bases = [
        Path.cwd(),
        ctx.config_dir,
        ctx.config_dir.parent if ctx.config_dir else None,
        runtime_dir,
        runtime_dir.parent,
        ctx.repo_root,
        ctx.repo_root.parent,
    ]
    args.transformer_path = resolve_path_best_effort(args.transformer_path, bases)
    args.vae_path = resolve_path_best_effort(args.vae_path, bases)
    args.text_encoder_path = resolve_path_best_effort(args.text_encoder_path, bases)
    args.t5_tokenizer_path = resolve_path_best_effort(args.t5_tokenizer_path, bases)
    args.data_dir = resolve_path_best_effort(args.data_dir, bases)
    reg_data_dir = getattr(args, "reg_data_dir", "") or ""
    if reg_data_dir:
        args.reg_data_dir = resolve_path_best_effort(reg_data_dir, bases)


def _swap_vram_discount(ctx: TrainingContext) -> float:
    """The **fraction** of weights that stay off VRAM when block swap is on (budget discount, see check_load_budget).

    A fraction rather than bytes: fp8 and bf16 file sizes differ by 2x, so a
    byte-based discount would blow through the budget in the fp8 case. If a
    family hasn't implemented the estimate, returns 0 (the guard degrades to
    conservative, never wrongly lets something through).
    """
    blocks_to_swap = int(getattr(ctx.args, "blocks_to_swap", 0) or 0)
    if blocks_to_swap <= 0:
        return 0.0
    ratio_fn = getattr(ctx.family, "swapped_param_ratio", None)
    if ratio_fn is None:
        return 0.0
    try:
        # checkpoint_path: anima uses it to tell apart the 28/36-layer versions; krea2 has a single structure and ignores it
        return float(ratio_fn(
            blocks_to_swap,
            checkpoint_path=str(getattr(ctx.args, "transformer_path", "") or ""),
        ))
    except Exception:  # noqa: BLE001
        return 0.0


def _load_dit(ctx: TrainingContext) -> None:
    args = ctx.args
    backend = getattr(args, "attention_backend", "flash_attn")
    if backend == "none":
        logger.info(
            "attention_backend=none, neither flash_attn nor xformers enabled, using PyTorch SDPA"
        )
    logger.info("Loading Transformer...")
    extra = {}
    blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
    if blocks_to_swap > 0:
        # The capability bit is already gated by cap_gate on the schema side; a bare CLI /
        # old yaml can still carry it, so fail fast here rather than silently ignoring it
        # (otherwise the user thinks they saved VRAM when they didn't).
        if "block_swap" not in ctx.family.spec.capabilities:
            raise RuntimeError(
                f"model_family='{ctx.family.spec.family_id}' does not support block swap, "
                f"but blocks_to_swap={blocks_to_swap}. Please set it to 0."
            )
        # Swapped-out layers land directly in CPU pinned memory via the loader, never touching VRAM
        # (this is the whole premise of the 12/16GB targets).
        extra["blocks_to_swap"] = blocks_to_swap
        logger.info("block swap: the last %d layers will stay resident in RAM, not loaded into VRAM", blocks_to_swap)
    ctx.model = ctx.family.load_dit(
        args.transformer_path,
        ctx.device,
        ctx.dtype,
        attention_backend=backend,
        repo_root=ctx.repo_root,
        **extra,
    )
    # Return large-weight mmap cache pages to the system (13-26GB; a real hang-on-paging
    # case observed in the field, and training benefits from it too).
    from training.sysmem import trim_working_set

    trim_working_set()
    log_vram("after DiT load", ctx.device)


def _log_train_start_vram(ctx: TrainingContext) -> None:
    """VRAM baseline right before the training loop starts -- the reading point for judging blocks_to_swap's actual effect."""
    swap = getattr(ctx, "block_swap", None)
    if swap is not None:
        logger.info(
            "block swap active: swapped out %d/%d layers, pinned %.2fGB",
            swap.num_swap, swap.total, swap.pinned_bytes / 1024**3,
        )
    log_vram("before training start", ctx.device)


def _load_vae(ctx: TrainingContext) -> None:
    args = ctx.args
    logger.info("Loading VAE...")
    ctx.vae = ctx.family.load_vae(
        args.vae_path,
        ctx.device,
        ctx.vae_dtype,
        tiling=getattr(args, "vae_tiling", "auto"),
    )


def _load_text(ctx: TrainingContext) -> None:
    args = ctx.args
    logger.info("Loading text encoder...")
    ctx.text_stack = ctx.family.load_text(
        args.text_encoder_path,
        ctx.device,
        ctx.dtype,
        t5_tokenizer_path=args.t5_tokenizer_path,
        cache_enabled=bool(getattr(args, "text_encoder_cache", True)),
    )


def _setup_block_swap(ctx: TrainingContext) -> None:
    """Build and attach block swap (docs/design/block-swap.md cut 2).

    **Must run after LoRA injection**: LyCORIS ``apply_to()`` reads the base
    weight's shape to build adapters; at that point the swapped-out layers'
    weights are the CPU-pinned tensors the loader put down (shape/dtype intact,
    just not in VRAM), so injection works normally. Attaching before injecting
    would also work, but there's no reason to complicate the order.

    Attaching goes through a hook (``attach()``) without touching the model's
    forward loop -- krea2's loop lives inside parity-sensitive ``modeling/``,
    and anima's manually unrolled loop shouldn't be touched either. All four
    forward + backward hooks are mandatory (backward must fetch the weights
    back itself, since recompute doesn't trigger forward_hook; see doc §9.10
    and tests/test_block_swap_grad_fidelity.py).
    """
    blocks_to_swap = int(getattr(ctx.args, "blocks_to_swap", 0) or 0)
    if blocks_to_swap <= 0:
        return
    from training.block_swap import PinnedBlockSwap

    # Clamp to the total layer count: blocks_to_swap is a setting shared across
    # families/versions (krea2 has 28 layers, anima has 28/36), so out-of-range
    # values are treated as "swap everything" rather than failing (same convention as the loader).
    ctx.block_swap = PinnedBlockSwap(
        ctx.model.blocks, min(blocks_to_swap, len(ctx.model.blocks)), ctx.device,
    )
    ctx.block_swap.attach()
    log_vram("after block swap attach", ctx.device)


def _inject_adapter(ctx: TrainingContext) -> None:
    args = ctx.args
    logger.info("Injecting %s...", args.lora_type.upper())
    from training.adapters import build_adapter

    ctx.injector = build_adapter(args, preset=ctx.family.lora_preset())
    ctx.injector.metadata_extra = ctx.family.lora_metadata()
    ctx.injector.inject(ctx.model)

    if getattr(args, "resume_lora", "") and Path(args.resume_lora).exists():
        lora_family = _read_lora_family(args.resume_lora)
        if lora_family != ctx.family.spec.family_id:
            raise RuntimeError(
                f"resume_lora rejected across model families: {args.resume_lora} belongs to '{lora_family}', "
                f"current model_family='{ctx.family.spec.family_id}'"
            )
        ctx.injector.load(args.resume_lora)
        logger.info("continuing training from an existing LoRA: %s", args.resume_lora)

    _setup_block_swap(ctx)

    if getattr(args, "sra_enabled", False):
        from training.families.anima.sra_align import SRAAligner

        model_channels = ctx.model.model_channels
        block_idx = int(getattr(args, "sra_block", 4))
        num_blocks = len(ctx.model.blocks)
        if block_idx >= num_blocks:
            logger.warning(
                "sra_block=%d >= model block count (%d), clamped to %d",
                block_idx,
                num_blocks,
                num_blocks - 1,
            )
            block_idx = num_blocks - 1
        ctx.sra_aligner = SRAAligner(
            model=ctx.model,
            block_idx=block_idx,
            patch_spatial=ctx.model.patch_spatial,
            patch_temporal=ctx.model.patch_temporal,
            model_channels=model_channels,
            vae_channels=_ANIMA_SPEC.latent.channels,
            device=ctx.device,
            dtype=ctx.dtype,
            normalize=bool(getattr(args, "sra_normalize", True)),
        )


def _defer_dit_for_text_cache(ctx: TrainingContext) -> bool:
    return (
        ctx.family.spec.text.strategy == "cached_varlen"
        and bool(getattr(ctx.args, "text_encoder_cache", True))
    )


def _validate_fp8_base(ctx: TrainingContext) -> None:
    """Combined validation for an fp8 base model (fp8_base training) -- fail-fast before any large load.

    Probing only reads the safetensors header (millisecond-scale); non-fp8 base
    models pass through at zero cost. Right now only the krea2 loader accepts an
    fp8 checkpoint (the Anima loader rejects it on its own), but the probe itself
    is family-agnostic. Two hard constraints:

    - grad_checkpoint must be on: fp8's VRAM savings depend on the recompute
      segment freeing per-layer dequant temporaries; if off, autograd keeps all
      264 layers' bf16 copies resident, using more memory than plain bf16.
    - DoRA is unsupported: lycoris weight_decompose initialization reads the base
      weight's numeric values (norm), and a raw fp8 cast lacks the scale
      correction, giving incorrect numbers (consistent with the inference/merge rejection rule).
    """
    from training.families.krea2.loader import checkpoint_contains_fp8

    args = ctx.args
    if not checkpoint_contains_fp8(getattr(args, "transformer_path", "") or ""):
        return
    problems = []
    if not bool(getattr(args, "grad_checkpoint", True)):
        problems.append(
            "grad_checkpoint=false: an fp8 base model's per-layer dequant temporaries would be "
            "kept fully resident by autograd, using more VRAM than bf16. Please enable gradient checkpointing."
        )
    if bool(getattr(args, "lora_dora", False)):
        problems.append(
            "lora_dora=true: DoRA initialization reads the base weight's numeric values, which "
            "are incorrect under fp8 storage. Please disable DoRA or switch to a bf16 base model."
        )
    if problems:
        raise RuntimeError(
            "fp8 base model is incompatible with the current config:\n- " + "\n- ".join(problems)
        )
    logger.info(
        "fp8 base model detected: training with fp8_base semantics (weights stay resident in fp8, dequantized per layer on forward)"
    )


def run(ctx: TrainingContext) -> None:
    """Resolve paths and load either the complete stack or the cache-first half."""
    from training.sysmem import check_load_budget, guard_enabled_from_env

    if ctx.family is None:
        ctx.family = resolve_family(ctx.args)
    _resolve_paths(ctx)
    _validate_fp8_base(ctx)
    # block swap preflight: answer "can this blocks_to_swap value actually run"
    # before dataset scanning / latent cache / text encoding, and exit with a
    # recommended value if not. models.run is the 2nd phase after bootstrap,
    # the earliest point where every path is available.
    block_swap_preflight.run(ctx)

    if _defer_dit_for_text_cache(ctx):
        logger.info(
            "text cache enabled: loading VAE/Qwen3-VL first, will load the Transformer after caching and releasing the TE"
        )
        # Segmented budget: this segment only loads VAE + TE (the DiT gets its own budget in the finish segment).
        # The switch comes from Settings -> Training -> Training Params (injected by the supervisor via env, default on).
        check_load_budget(
            guard_enabled_from_env(),
            weight_paths=[getattr(ctx.args, "vae_path", ""),
                          getattr(ctx.args, "text_encoder_path", "")],
            stage="training model load (VAE/text encoder)",
            settings_hint="Settings -> Training -> Training Params",
        )
        _load_vae(ctx)
        _load_text(ctx)
        return

    # Preserve the historical Anima order. Storage-free Krea2 deliberately keeps
    # the DiT resident while its text encoder is loaded for per-batch encoding.
    check_load_budget(
        guard_enabled_from_env(),
        weight_paths=[
            getattr(ctx.args, "transformer_path", ""),
            getattr(ctx.args, "vae_path", ""),
            getattr(ctx.args, "text_encoder_path", ""),
        ],
        stage="training model load",
        vram_discount_ratio=_swap_vram_discount(ctx),
        settings_hint="Settings -> Training -> Training Params",
    )
    _load_dit(ctx)
    _load_vae(ctx)
    _load_text(ctx)
    _inject_adapter(ctx)
    _log_train_start_vram(ctx)


def finish(ctx: TrainingContext) -> None:
    """Load/inject a DiT deferred by cached text preparation; otherwise no-op."""
    if ctx.model is not None:
        return
    from training.sysmem import check_load_budget, guard_enabled_from_env

    logger.info("text cache done and text encoder released; continuing to load Transformer...")
    check_load_budget(
        guard_enabled_from_env(),
        weight_paths=[getattr(ctx.args, "transformer_path", "")],
        stage="training model load (Transformer)",
        vram_discount_ratio=_swap_vram_discount(ctx),
        settings_hint="Settings -> Training -> Training Params",
    )
    _load_dit(ctx)
    _inject_adapter(ctx)
    _log_train_start_vram(ctx)


def _read_lora_family(path) -> str:
    """Read artifact family; legacy unmarked safetensors grandfather to Anima."""
    import json

    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            meta = handle.metadata() or {}
        args = json.loads(meta.get("ss_network_args") or "{}")
        return str(args.get("model_family") or "anima")
    except Exception:
        return "anima"
