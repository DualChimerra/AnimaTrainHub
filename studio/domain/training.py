"""Main training config schema — the single pydantic v2 source of truth.

Full set of training parameters, aligned with config/train_template.yaml.

Later:
    - argparse is reverse-generated from studio.argparse_bridge (P2-B)
    - the frontend form reads /api/schema for auto-rendering
    - YAML configs are validated with TrainingConfig.model_validate(yaml_dict)

Convention: every field carries UI metadata via
`json_schema_extra={"group", "control", "show_when"?}`. The frontend
groups fields by `group` and conditionally shows them via `show_when`.

Note: do not use `from __future__ import annotations` — under Pydantic v2 +
Python 3.12+, deferred evaluation would treat typing._SpecialForm as a schema
key and raise AttributeError.
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .common import (
    AttentionBackend,
    FAMILY_CONFIG_DEFAULTS,
    FAMILY_SAMPLING,
    LEGACY_SAMPLING_FAMILIES,
    TIMESTEP_SAMPLING_OPTION_FAMILIES,
    _meta,
    cap_gate,
    capability_violations,
    option_gates,
    sampling_option_gates,
)
from .migrations import migrate_legacy_save_keys, migrate_noise_enhancement_type


class TrainingConfig(BaseModel):
    """Full set of training parameters, aligned with config/train_template.yaml.

    `extra="ignore"`: extra keys left over from cloud presets / legacy versions are silently dropped.
    """

    model_config = ConfigDict(extra="ignore")

    # ---------------------------------------------------------------- Model paths
    # These paths get replaced with **absolute paths** when Studio creates a
    # version (based on secrets.models.root + secrets.models.selected_anima), so
    # the absolute path the user sees in the yaml / Train page is always
    # unambiguous — no need to worry about relative-path anchoring.
    # The defaults here are only a fallback: for a bare CLI training run with
    # the yaml left completely unfilled, they resolve relative to the repo
    # (matching historical behavior).
    # Top-level decisions are never hidden in the advanced section (P4-3). The
    # frontend intercepts changes to this field as a "switch action": paths
    # are recalculated via /api/models/family-switch + family-specific fields
    # reset + only written after confirmation.
    model_family: Literal["anima", "krea2"] = Field(
        "anima",
        description="Model family. Determines which model implementation training uses; other fields show/hide automatically based on family capabilities. Switching recalculates model paths and resets family-specific defaults",
        json_schema_extra=_meta("model"),
    )
    transformer_path: str = Field(
        "models/diffusion_models/anima-base-v1.0.safetensors",
        description="Main diffusion model weights (.safetensors)",
        json_schema_extra=_meta("model", "path", cli_alias="--transformer"),
    )
    vae_path: str = Field(
        "models/vae/qwen_image_vae.safetensors",
        description="VAE weights (.safetensors)",
        json_schema_extra=_meta("model", "path", cli_alias="--vae"),
    )
    text_encoder_path: str = Field(
        "models/text_encoders",
        description="Qwen text encoder directory",
        json_schema_extra=_meta("model", "path", cli_alias="--qwen"),
    )
    t5_tokenizer_path: str = Field(
        "models/t5_tokenizer",
        description="T5 tokenizer directory (Anima family only)",
        json_schema_extra=_meta("model", "path", cli_alias="--t5-tokenizer",
                                show_when="model_family==anima"),
    )
    text_encoder_cache: bool = Field(
        True,
        description="Precompute and cache text conditioning, then free the text encoder (recommended, saves roughly 9GB VRAM); when off, text sidecars aren't read or written and Qwen3-VL stays resident, encoding per batch — suited to high-VRAM but disk-constrained cloud setups",
        json_schema_extra=_meta(
            "model", show_when=cap_gate("text_cache"), advanced=True,
        ),
    )

    # ----------------------------------------------------------------- Dataset
    data_dir: str = Field(
        "./dataset",
        description="Dataset directory (supports Kohya-style N_xxx subfolders for setting repeats)",
        json_schema_extra=_meta("dataset", "path"),
    )
    resolution: list[int] = Field(
        default=[1024],
        description="Training resolution(s). Multiple values allowed (comma-separated, e.g. 512, 768, 1024) — for folders without a resolution prefix, each image trains once per resolution; a single value is classic single-resolution training",
        json_schema_extra=_meta("dataset"),
    )
    aspect_ratio_limit: float = Field(
        2.0, ge=1.0, le=4.0,
        description="Maximum aspect ratio allowed for a bucket. 2.0 = widest 2:1, tallest 1:2. Larger values let longer/flatter images train at their native ratio, but extreme-ratio buckets have fewer images and lower effective short-edge resolution",
        json_schema_extra=_meta("dataset"),
    )
    # This fork: ARB bucket boundaries/granularity can be configured explicitly
    # (None = auto-derived from base_reso, matching upstream). With explicit
    # 512/2048/64, the bucket set matches the old fork byte-for-byte (the crop
    # page's trainBuckets.ts prediction depends on this stability).
    bucket_min_reso: Optional[int] = Field(
        None, ge=64, le=4096,
        description="Minimum pixels per side for ARB bucketing (multiple of step). Leave blank = auto-derived from resolution",
        json_schema_extra=_meta("dataset", advanced=True),
    )
    bucket_max_reso: Optional[int] = Field(
        None, ge=64, le=8192,
        description="Maximum pixels per side for ARB bucketing (multiple of step). Leave blank = auto-derived from resolution. Increasing this preserves more detail from larger source images, at the cost of more VRAM",
        json_schema_extra=_meta("dataset", advanced=True),
    )
    bucket_step: Optional[int] = Field(
        None, ge=8, le=256,
        description="ARB bucket width/height granularity (pixels). Leave blank = 64; VAE downsampling alignment usually requires a multiple of 8/16",
        json_schema_extra=_meta("dataset", advanced=True),
    )
    reg_data_dir: Optional[str] = Field(
        None,
        description="Regularization set directory (optional, guards against overfitting)",
        json_schema_extra=_meta("dataset", "path"),
    )
    reg_caption: Optional[str] = Field(
        None,
        description="Shared caption for the regularization set (leave blank to use each image's own .txt/.json)",
        json_schema_extra=_meta("dataset"),
    )
    reg_weight: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Regularization loss weight relative to the training set; 1.0 = equal weight, lower values weaken the reg set's influence",
        json_schema_extra=_meta("dataset"),
    )

    # -------------------------------------------------------------- Caption
    shuffle_caption: bool = Field(
        True,
        description="Enable tag shuffling (JSON mode shuffles within categories, TXT mode shuffles everything)",
        json_schema_extra=_meta("caption", show_when=cap_gate("caption_tag_ops")),
    )
    keep_tokens: int = Field(
        0, ge=0,
        description="Protect the first N tags from shuffling and dropout (TXT mode only;"
                    " in JSON mode this role is covered by meta.trigger and fixed fields)",
        json_schema_extra=_meta("caption", show_when=cap_gate("caption_tag_ops")),
    )
    flip_augment: bool = Field(
        True,
        description="Horizontal flip augmentation",
        json_schema_extra=_meta("caption"),
    )
    tag_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Random drop probability per tag during training (0 = off); nonzero values help generalization and reduce reliance on any single tag",
        json_schema_extra=_meta("caption", show_when=cap_gate("caption_tag_ops")),
    )
    prefer_json: bool = Field(
        True,
        description="Prefer JSON tag files (recommended, supports per-category shuffle)",
        json_schema_extra=_meta("caption"),
    )
    caption_comfy_encoding: bool = Field(
        True,
        description="Encode tags the ComfyUI way (same encoding pipeline as test generation and "
                    "sampling previews; recommended to keep on); turning it off falls back to the "
                    "old per-tag encoding, useful for A/B comparisons between old and new encoding, "
                    "or to continue training from a checkpoint saved under the old encoding",
        json_schema_extra=_meta("caption", advanced=True),
    )
    cache_latents: bool = Field(
        True,
        description="Cache VAE latents to speed up training",
        json_schema_extra=_meta(
            "system",
            disable_when="navit_packing==true",
            disable_value=True,
            disable_hint="NaViT packing budgets by latent token count, so pre-encoded caching is mandatory",
        ),
    )
    vae_cache_batch_size: int = Field(
        0, ge=0,
        description="Batch size for VAE latent cache encoding; 0 = follow the training batch size, set to 1 for one-at-a-time encoding if VRAM is tight",
        json_schema_extra=_meta("system", show_when="cache_latents==true", advanced=True),
    )
    # Grouped under "system" rather than "training": it only changes where the
    # weights live, not any training value — swapping out more layers has
    # zero effect on the resulting LoRA bit-for-bit (a pure resource knob,
    # like vae_tiling / cache_latents)
    blocks_to_swap: int = Field(
        0, ge=0,
        description="Block swapping: number of DiT layers swapped out to system RAM (0=off; Anima accepts 0-28, about 0.13GB per layer; "
                    "Krea 2 accepts 0-28, about 0.4GB per layer for the fp8 base model / 0.8GB for bf16). "
                    "Higher values use less VRAM and more RAM",
        json_schema_extra=_meta(
            "system",
            show_when=cap_gate("block_swap"),
            advanced=True,
        ),
    )
    block_swap_preflight: bool = Field(
        True,
        description="Block-swap preflight check: before training starts, estimates from the base model file size, free VRAM, and available RAM "
                    "whether the current blocks_to_swap will actually run; if not, it fails immediately with a suggested value, "
                    "instead of OOMing after the dataset and cache have already finished. Turn this off if the estimate produces false rejections",
        json_schema_extra=_meta(
            "system",
            show_when=cap_gate("block_swap"),
            advanced=True,
        ),
    )

    # --------------------------------------------------- NaViT / Patch-n-Pack
    # Block-diagonal packed training (Phase 2 data layer). All the following
    # fields default to off / behavior-neutral: with navit_packing=False, the
    # original ARB bucketing path runs, byte-for-byte identical.
    navit_packing: bool = Field(
        False,
        description="Enable NaViT / Patch-n-Pack block-diagonal packing: packs multiple differently-sized images"
                    " into one training sequence by token budget (zero padding), replacing fixed ARB bucket batching. Requires cache_latents + xformers installed",
        json_schema_extra=_meta(
            "system",
            advanced=True,
            show_when=cap_gate("navit"),
            disable_when="leap_enabled==true||infonoise_enabled==true||sra_enabled==true||lora_type==tlora",
            disable_hint="Mutually exclusive with LeapAlign / InfoNoise / SRA / T-LoRA (navit v1 isn't adapted for them) — disable those first",
        ),
    )
    navit_token_budget: int = Field(
        16384, ge=1,
        description="Token budget ceiling for a single packed sequence (sum of all image token counts ≤ this value)",
        json_schema_extra=_meta("system", show_when="navit_packing==true", advanced=True),
    )
    navit_max_images_per_pack: int = Field(
        0, ge=0,
        description="Max images per pack (0=unlimited, bound only by the token budget)",
        json_schema_extra=_meta("system", show_when="navit_packing==true", advanced=True),
    )
    navit_text_trim_padding: bool = Field(
        False,
        description="Trim cross-attention text padding (data-layer flag, only takes effect with navit_packing)",
        json_schema_extra=_meta("system", show_when="navit_packing==true", advanced=True),
    )
    navit_pack_strategy: Literal["next_fit", "ffd"] = Field(
        "next_fit",
        description="Packing strategy: next_fit=sequential greedy (fast, looser packing); ffd=First-Fit-Decreasing windowed (tighter packing, fewer steps)",
        json_schema_extra=_meta("system", show_when="navit_packing==true", advanced=True),
    )
    navit_pack_ffd_window: int = Field(
        256, ge=0,
        description="FFD window size (0=global FFD, fixed packs per epoch; >0=windowed FFD + reshuffle across epochs)",
        json_schema_extra=_meta("system", show_when="navit_packing==true", advanced=True),
    )
    navit_drop_last: bool = Field(
        False,
        description="Drop the last, not-fully-filled pack (keeps step counts aligned; False=keep short packs)",
        json_schema_extra=_meta("system", show_when="navit_packing==true", advanced=True),
    )
    navit_native_resolution: bool = Field(
        False,
        description="Size images at native resolution (floor-aligned to 16px, zero padding), bypassing ARB bucket quantization of per-image size; "
                    "oversized images are handled per navit_native_over_budget. Requires navit_packing + cache_latents",
        json_schema_extra=_meta(
            "system", show_when="navit_packing==true", advanced=True,
            # Without this declaration: turning off navit_packing after
            # enabling native hides this field via show_when, but the value
            # stays true → saving hits the hand-written validator's "native
            # requires packing" error, leaving the user stuck in the UI.
            # Declaring it as a disable rule lets the frontend's takeover
            # automatically pin it back to false when the gate closes, and
            # the tolerant loader self-heals the same way; an explicit
            # violation still fails fast (the guard surface is unchanged).
            disable_when="navit_packing!=true",
            disable_value=False,
            disable_hint="Native sizing only applies on the NaViT block-diagonal packing path; enable navit_packing first",
        ),
    )
    navit_native_over_budget: Literal["downscale", "fail"] = Field(
        "downscale",
        description="How to handle native token counts exceeding navit_token_budget (or the model's RoPE per-side limit): "
                    "downscale=proportionally downscale to fit (default, never OOMs or overflows the token budget); fail=error out, requiring a larger budget or cropping the dataset",
        json_schema_extra=_meta("system", show_when="navit_native_resolution==true", advanced=True),
    )
    cache_encode_tiled: bool = Field(
        False,
        description="Tiled cache encoding: for oversized images, VAE-encode in tiles by tile_px and blend the latents with feathering (peak VRAM scales with per-tile pixel count)",
        json_schema_extra=_meta("system", advanced=True),
    )
    cache_encode_tile_px: int = Field(
        1024, ge=64,
        description="Tile edge length in pixels for tiled encoding (must be a multiple of the VAE's 8x downsampling)",
        json_schema_extra=_meta("system", show_when="cache_encode_tiled==true", advanced=True),
    )
    cache_encode_tile_overlap: int = Field(
        128, ge=0,
        description="Tile overlap in pixels (feather seam width; 0=hard seam, no overlap)",
        json_schema_extra=_meta("system", show_when="cache_encode_tiled==true", advanced=True),
    )
    cache_encode_max_pixels: int = Field(
        0, ge=0,
        description="Max total pixels per encode call (including the flipped copy); 0=use the conservative built-in default of 4M. Also the tiling trigger threshold",
        json_schema_extra=_meta("system", show_when="cache_encode_tiled==true", advanced=True),
    )

    # ------------------------------------------------------------------- LoRA
    lora_type: Literal["lora", "lokr", "loha", "ortho", "tlora"] = Field(
        "lora",
        description="Adapter algorithm. lora: classic low-rank, general-purpose (default); lokr: Kronecker factorization, most parameter-efficient; loha: Hadamard product, higher expressiveness but more parameters; ortho: orthogonal parameterization, very few trainable parameters, resists overfitting — good for small character/subject datasets; tlora: rank shrinks as noise increases, purpose-built to resist overfitting on single-image/very-small subject datasets",
        json_schema_extra=_meta(
            "lora",
            # Option-level forbid (R2 v2): navit is a B=1 packed sequence with
            # per-image t; T-LoRA's rank mask matches timestep along the
            # batch dimension, so the dimensions don't line up
            option_disable_when={"tlora": "navit_packing==true"},
            disable_hint="T-LoRA's rank mask matches timestep along the batch dimension, which is incompatible with NaViT packing (B=1, per-image t)",
        ),
    )
    lora_rank: int = Field(
        32, ge=4,
        description="Rank: higher = more expressive but more parameters, more VRAM, and easier to overfit; lower = cheaper but easier to underfit. Common values: 8/16/32/64",
        json_schema_extra=_meta("lora"),
    )
    lora_alpha: float = Field(
        32.0, ge=0.0,
        description="Alpha: LoRA scaling factor — higher means a stronger LoRA effect. Usually equal to rank; when rs_lora is on, commonly set to √rank",
        json_schema_extra=_meta("lora"),
    )
    lokr_factor: int = Field(
        8, ge=2,
        description="LoKr matrix-factorization factor: higher = stronger compression, fewer parameters; lower = more parameters. Default of 8 suits most cases",
        json_schema_extra=_meta("lora", show_when="lora_type==lokr"),
    )
    tlora_min_rank: int = Field(
        1, ge=1,
        description="Minimum active rank T-LoRA keeps at high noise levels (matches the ControlGenAI/T-LoRA paper, default 1)",
        json_schema_extra=_meta("lora", show_when="lora_type==tlora", advanced=True),
    )
    tlora_alpha_rank_scale: float = Field(
        1.0, ge=0.0,
        description=(
            "T-LoRA power-law scaling (matches the official SDXL `alpha_rank_scale`): 1.0=linear schedule; "
            ">1 shifts high rank toward the low-noise end; <1 opens high rank earlier. "
            "Formula r=(1-t)^α·(rank-min_rank)+min_rank, where t is the noise level (0=clean, 1=noisy)"
        ),
        json_schema_extra=_meta("lora", show_when="lora_type==tlora", advanced=True),
    )
    tlora_use_ortho: bool = Field(
        True,
        description="T-LoRA-specific: stack OrthoLoRA orthogonal parameterization on top (the paper's full recipe, on by default); when off, plain T-LoRA is used",
        json_schema_extra=_meta("lora", show_when="lora_type==tlora", advanced=True),
    )
    lora_dora: bool = Field(
        False,
        description="DoRA: decomposes weights into direction + magnitude trained independently; usually converges more steadily, at a slight VRAM cost",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_rs: bool = Field(
        False,
        description="rs-LoRA: scale=α/√r instead of α/r, more stable training at high rank (>32)",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Random drop probability for LoRA input features: higher = stronger regularization, slower convergence; 0 = off",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_rank_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Random drop probability for LoRA's internal rank dimension (randomly activates a subset of ranks each step): higher = stronger regularization; 0 = off",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_module_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Random skip probability for an entire LoRA module (stochastic depth): with this probability, the module is skipped entirely for a step; higher = stronger regularization; 0 = off",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_reg_dims: Optional[dict[str, int]] = Field(
        None,
        description="Per-layer rank: a dict mapping regex → rank, overriding the default rank for modules whose full name matches (e.g. {\"lora_unet_.*double.*\": 16})",
        examples=[{"lora_unet_.*double.*": 16}],
        json_schema_extra=_meta("lora", "code", advanced=True),
    )

    # ------------------------------------------------------------------ Training
    epochs: int = Field(
        10, ge=1,
        description="Number of training epochs",
        json_schema_extra=_meta("training"),
    )
    max_steps: int = Field(
        0, ge=0,
        description="Max steps (0=unlimited)",
        json_schema_extra=_meta("training"),
    )
    batch_size: int = Field(
        1, ge=1,
        description="Batch size",
        json_schema_extra=_meta(
            "training",
            # Under navit, the DataLoader uses NavitPackBatchSampler (one
            # step = one token pack), so batch_size plays no part in
            # batching — without pinning this, users would think they're
            # controlling step count/VRAM when it does nothing (this once
            # caused an "expected 2520, got 5040" discrepancy). Side effect
            # of pinning to 1: vae_cache_batch_size=0's "follow batch_size"
            # becomes one-at-a-time encoding; set that field explicitly if
            # you need a larger cache batch.
            disable_when="navit_packing==true",
            disable_value=1,
            disable_hint="NaViT packing batches by token budget (navit_token_budget); batch_size plays no part in batching; "
                         "control the VAE cache encoding batch size explicitly via vae_cache_batch_size",
        ),
    )
    grad_checkpoint: bool = Field(
        True,
        description="Gradient checkpointing (saves VRAM, adds roughly 1/3 more compute)",
        json_schema_extra=_meta("training"),
    )
    grad_accum: int = Field(
        4, ge=1,
        description="Gradient accumulation steps (effective batch = batch_size × grad_accum)",
        json_schema_extra=_meta("training"),
    )
    learning_rate: float = Field(
        1e-4, gt=0.0,
        description="Learning rate. For Automagic this is the initial per-parameter learning rate, recommended 1e-6 (automatically rewritten when switching optimizer to automagic); for Lion, recommended AdamW lr / 3; for Prodigy / PPSF it must be 1.0",
        json_schema_extra=_meta(
            "training",
            cli_alias="--lr",
            disable_when="optimizer_type==prodigy||optimizer_type==prodigy_plus_schedulefree",
            disable_value=1.0,
            disable_hint="This optimizer manages its own learning rate",
        ),
    )
    lr_scheduler: Literal[
        "none",
        "cosine",
        "cosine_with_restart",
        "cosine_with_warmup",
        "cosine_cycles",
        "constant_then_cosine",
        "polynomial",
        "rex",
        "rex_annealing_warm_restarts",
    ] = Field(
        "none",
        description="Learning rate schedule (none = constant; Prodigy / PPSF / Automagic / SOAP-SF are fixed to none)",
        json_schema_extra=_meta(
            "training",
            disable_when="optimizer_type==automagic||optimizer_type==prodigy||optimizer_type==prodigy_plus_schedulefree||optimizer_type==soap_sf",
            disable_value="none",
            disable_hint="Adaptive / Schedule-Free optimizers are fixed to a constant learning rate",
        ),
    )
    lr_scheduler_t0: int = Field(
        500, ge=1,
        description="cosine_with_restart first restart period (in steps)",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_with_restart", advanced=True),
    )
    lr_scheduler_t_mult: float = Field(
        2.0, ge=1.0,
        description="cosine_with_restart period multiplier applied after each restart (>1 = growing periods)",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_with_restart", advanced=True),
    )
    lr_scheduler_eta_min: float = Field(
        1e-6, ge=0.0,
        description="Learning rate floor: cosine / polynomial / rex / RAWR schedules stop decreasing once they reach this value; usually much smaller than the initial lr (e.g. 1e-4 initial with 1e-6 floor)",
        json_schema_extra=_meta(
            "training",
            show_when="lr_scheduler==cosine||lr_scheduler==cosine_with_restart||lr_scheduler==cosine_with_warmup||lr_scheduler==polynomial||lr_scheduler==rex||lr_scheduler==rex_annealing_warm_restarts",
            advanced=True,
        ),
    )
    lr_scheduler_warmup_steps: int = Field(
        100, ge=0,
        description="Warmup steps for cosine_with_warmup / polynomial / rex (linear ramp from 0 to the peak lr); for RAWR the warmup repeats at the start of every cycle. 0 = no warmup",
        json_schema_extra=_meta(
            "training",
            show_when="lr_scheduler==cosine_with_warmup||lr_scheduler==polynomial||lr_scheduler==rex||lr_scheduler==rex_annealing_warm_restarts",
            advanced=True,
        ),
    )
    lr_scheduler_power: float = Field(
        1.0, gt=0.0,
        description="polynomial: decay power. 1 = linear decay to the floor; >1 drops faster early and flattens near the end; <1 holds the lr longer and drops at the end",
        json_schema_extra=_meta("training", show_when="lr_scheduler==polynomial"),
    )
    lr_scheduler_rex_d: float = Field(
        0.9, gt=0.0, le=1.0,
        description="rex / RAWR curve shape d: factor = z / ((1 - d) + d·z), z = remaining fraction. 0.5 = the REX paper; 0.9 (LoRA Easy Training Scripts default) holds the peak longer and drops more sharply at the end",
        json_schema_extra=_meta("training", show_when="lr_scheduler==rex||lr_scheduler==rex_annealing_warm_restarts", advanced=True),
    )
    lr_scheduler_cycle_multiplier: float = Field(
        1.0, gt=0.0,
        description="RAWR: each next cycle is this many times longer than the previous one (1 = equal cycles, 2 = doubling). Cycle lengths are fitted so all cycles fill the run",
        json_schema_extra=_meta("training", show_when="lr_scheduler==rex_annealing_warm_restarts"),
    )
    lr_scheduler_gamma: float = Field(
        0.9, gt=0.0, le=1.0,
        description="RAWR: peak lr multiplier after each restart (1 = every cycle restarts at the full lr; 0.9 = each peak 10% lower)",
        json_schema_extra=_meta("training", show_when="lr_scheduler==rex_annealing_warm_restarts"),
    )
    lr_scheduler_cycle_count: int = Field(
        3, ge=1, le=100,
        description="Number of peaks for cosine_cycles / RAWR; the cycles together fill the whole run (cosine_cycles splits it evenly)",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_cycles||lr_scheduler==rex_annealing_warm_restarts"),
    )
    lr_scheduler_cycle_max_lr: Optional[float] = Field(
        None, gt=0.0,
        description="Max LR shared across all cycles; leave blank to use learning_rate",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_cycles"),
    )
    lr_scheduler_cycle_min_lr: float = Field(
        0.0, ge=0.0,
        description="Min LR shared across all cycles; per-cycle list values override it",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_cycles"),
    )
    lr_scheduler_cycle_max_lrs: list[float] = Field(
        default_factory=list,
        description="Comma-separated max LR for each peak; empty = shared max for all. The number of values must match the number of peaks",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_cycles", advanced=True),
    )
    lr_scheduler_cycle_min_lrs: list[float] = Field(
        default_factory=list,
        description="Comma-separated min LR for each peak; empty = shared min for all. The number of values must match the number of peaks",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_cycles", advanced=True),
    )
    lr_scheduler_decay_start_ratio: float = Field(
        2.0 / 3.0, ge=0.0, lt=1.0,
        description="constant_then_cosine: fraction of training completed before the smooth decay begins; 0.6667 = the final third",
        json_schema_extra=_meta("training", show_when="lr_scheduler==constant_then_cosine"),
    )
    lr_scheduler_decay_min_lr: float = Field(
        0.0, ge=0.0,
        description="constant_then_cosine: LR at the very end of training",
        json_schema_extra=_meta("training", show_when="lr_scheduler==constant_then_cosine"),
    )
    optimizer_type: Literal["adamw", "adamw8bit", "automagic", "came", "lion", "prodigy", "prodigy_plus_schedulefree", "simplified_ademamix", "soap", "soap_sf"] = Field(
        "adamw",
        description="Optimizer. adamw standard baseline; adamw8bit same as AdamW but with state quantized to int8 (state VRAM ~1/4 of AdamW, hyperparameters carry over directly, requires bitsandbytes); automagic adaptive per-parameter lr (recommended lr=1e-6); came confidence-guided + factored second moment (lower state VRAM than AdamW, lr on the same scale as AdamW); lion roughly half AdamW's VRAM (recommended lr=AdamW lr / 3); prodigy / prodigy_plus_schedulefree adaptively estimates lr (set lr to 1.0); simplified_ademamix one raw-gradient momentum + current-gradient mix (lr ≈ AdamW lr × (1 − β1), e.g. 1e-6 at β1=0.99); soap Adam-in-Shampoo-eigenbasis second-order preconditioning (fits faster, lr on the same scale as AdamW); soap_sf SOAP + Schedule-Free (lr_scheduler fixed to none)",
        json_schema_extra=_meta("training"),
    )
    prodigy_d_coef: float = Field(
        1.0, ge=0.1, le=10.0,
        description="Overall scaling factor for Prodigy's estimated d; higher = larger effective lr. Raise it (2.0+) if underfitting, lower it (0.5) if overfitting or on a small dataset",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy"),
    )
    prodigy_safeguard_warmup: bool = Field(
        True,
        description="Prevent d from being pushed up by early high gradients during Prodigy warmup; on by default for stability",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy", advanced=True),
    )
    # ---------------------------- CAME-specific fields ----------------------------
    # CAME = Confidence-guided Adaptive Memory Efficient (Luo et al. 2023,
    # arxiv 2307.02047). lr uses real AdamW-scale values and can use a
    # regular lr_scheduler.
    came_beta1: float = Field(
        0.9, ge=0.0, lt=1.0,
        description="CAME β1 (update momentum EMA decay)",
        json_schema_extra=_meta("training", show_when="optimizer_type==came", advanced=True),
    )
    came_beta2: float = Field(
        0.999, ge=0.0, lt=1.0,
        description="CAME β2 (factored second-moment EMA decay)",
        json_schema_extra=_meta("training", show_when="optimizer_type==came", advanced=True),
    )
    came_beta3: float = Field(
        0.9999, ge=0.0, lt=1.0,
        description="CAME β3 (confidence/instability EMA decay; larger = smoother confidence estimate)",
        json_schema_extra=_meta("training", show_when="optimizer_type==came", advanced=True),
    )
    came_eps1: float = Field(
        1e-30, gt=0.0,
        description="CAME eps1 (second-moment regularizer, prevents division by zero)",
        json_schema_extra=_meta("training", show_when="optimizer_type==came", advanced=True),
    )
    came_eps2: float = Field(
        1e-16, gt=0.0,
        description="CAME eps2 (instability floor regularizer; increasing it flattens per-coordinate confidence differences and shrinks step size overall)",
        json_schema_extra=_meta("training", show_when="optimizer_type==came", advanced=True),
    )
    came_clip_threshold: float = Field(
        1.0, gt=0.0,
        description="CAME update RMS clipping threshold (same role as Adafactor's d)",
        json_schema_extra=_meta("training", show_when="optimizer_type==came", advanced=True),
    )
    lion_beta1: float = Field(
        0.9, ge=0.0, lt=1.0,
        description="Lion β1 (momentum interpolation coefficient)",
        json_schema_extra=_meta("training", show_when="optimizer_type==lion", advanced=True),
    )
    lion_beta2: float = Field(
        0.99, ge=0.0, lt=1.0,
        description="Lion β2 (momentum accumulation coefficient)",
        json_schema_extra=_meta("training", show_when="optimizer_type==lion", advanced=True),
    )
    # ------------------------- Simplified AdEMAMix-specific fields -------------------------
    # Morwani et al. 2025 (arxiv 2502.02431). The momentum sums raw gradients
    # (no 1-β1 factor), so lr is ~(1-β1) times the AdamW lr.
    ademamix_beta1: float = Field(
        0.99, ge=0.0, lt=1.0,
        description="Simplified AdEMAMix β1 (momentum decay; higher = longer gradient memory, and lr must shrink by (1 − β1))",
        json_schema_extra=_meta("training", show_when="optimizer_type==simplified_ademamix", advanced=True),
    )
    ademamix_beta2: float = Field(
        0.999, ge=0.0, lt=1.0,
        description="Simplified AdEMAMix β2 (second-moment EMA decay)",
        json_schema_extra=_meta("training", show_when="optimizer_type==simplified_ademamix", advanced=True),
    )
    ademamix_alpha: float = Field(
        0.0, ge=0.0,
        description="Simplified AdEMAMix α: weight of the current gradient added on top of the momentum (0 = momentum only)",
        json_schema_extra=_meta("training", show_when="optimizer_type==simplified_ademamix", advanced=True),
    )
    ademamix_beta1_warmup: int = Field(
        0, ge=0,
        description="Simplified AdEMAMix: steps over which β1 ramps up from the starting β1 to β1 (0 = off). Stabilizes the start with a high β1",
        json_schema_extra=_meta("training", show_when="optimizer_type==simplified_ademamix", advanced=True),
    )
    ademamix_min_beta1: float = Field(
        0.9, ge=0.0, lt=1.0,
        description="Simplified AdEMAMix: starting β1 for the β1 warmup",
        json_schema_extra=_meta("training", show_when="optimizer_type==simplified_ademamix", advanced=True),
    )
    automagic_variant: Literal["v1", "v2"] = Field(
        "v1",
        description="v1: per-element lr mask (classic, recommended); v2: per-param scalar lr + fused backward (experimental, saves VRAM; incompatible with grad_accum / grad_clip / fp16)",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic"),
    )
    automagic_min_lr: float = Field(
        1e-7, ge=0.0,
        description="Automagic per-parameter learning rate floor",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic", advanced=True),
    )
    automagic_max_lr: float = Field(
        1e-3, gt=0.0,
        description="Automagic per-parameter learning rate ceiling",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic", advanced=True),
    )
    automagic_lr_bump: float = Field(
        1e-6, ge=0.0,
        description="Step size Automagic uses to adjust the per-parameter learning rate on agreeing/disagreeing updates",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic", advanced=True),
    )
    automagic_beta2: float = Field(
        0.999, ge=0.0, lt=1.0,
        description="Automagic second-moment β2",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic", advanced=True),
    )
    automagic_eps: float = Field(
        1e-30, gt=0.0,
        description="Automagic numerical-stability epsilon",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic", advanced=True),
    )
    automagic_clip_threshold: float = Field(
        1.0, gt=0.0,
        description="Automagic update RMS clipping threshold",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic", advanced=True),
    )
    automagic_agreement_threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description="v2 sign-agreement ratio threshold: above this fraction the direction is considered consistent → raise lr",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic&&automagic_variant==v2", advanced=True),
    )
    # ---------------- ProdigyPlusScheduleFree (PPSF)-specific fields ----------------
    # When PPSF is selected, lr_scheduler must be none (Schedule-Free doesn't
    # need a scheduler; startup validation would otherwise be fatal). lr is
    # forced to 1.0 (overridden internally by the factory).
    ppsf_d_coef: float = Field(
        1.0, ge=0.1, le=10.0,
        description="Overall scaling factor for PPSF's estimated d; higher = larger effective lr. Raise it (2.0+) if underfitting, lower it (0.5) if overfitting or on a small dataset",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree"),
    )
    ppsf_prodigy_steps: int = Field(
        0, ge=0,
        description="Freeze the d estimate after step N; 0 = keep estimating throughout. Recommend 1/4-1/2 of total steps so lr stabilizes later in training",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_beta1: float = Field(
        0.9, ge=0.0, le=1.0,
        description="PPSF first-moment decay rate (β1): default 0.9; higher = smoother but slower to respond",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_beta2: float = Field(
        0.99, ge=0.0, le=1.0,
        description="PPSF second-moment decay rate (β2): default 0.99; higher = smoother gradient-variance estimate, slower to respond",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_split_groups: bool = Field(
        True,
        description="Estimate d separately per param group (lets each of LoRA's multiple parameter groups use its own suitable lr); on by default",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_split_groups_mean: bool = Field(
        False,
        description="When split_groups is on, average d across groups (recommended off for LoRA's multiple param groups)",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_use_speed: bool = Field(
        False,
        description="PPSF speed mode (experimental, may introduce instability)",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_fused_back_pass: bool = Field(
        False,
        description="Integrate PPSF with fused backward (enable when VRAM is tight; can significantly reduce VRAM use)",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_use_stableadamw: bool = Field(
        True,
        description="Enable PPSF's stable-AdamW-style normalization to prevent abnormal per-step gradient scale; on by default",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    # ------------------------- SOAP / SOAP-SF-specific fields -------------------------
    # SOAP = Adam in the Shampoo eigenbasis (Vyas et al. 2024, arxiv 2409.11321).
    # soap_sf = SOAP + Schedule-Free (arxiv 2405.15682); when soap_sf is
    # selected, lr_scheduler must be none (startup validation would otherwise
    # be fatal), and lr uses AdamW-scale values (not 1.0 like Prodigy).
    soap_beta1: float = Field(
        0.95, ge=0.0, lt=1.0,
        description="SOAP β1. For soap: Adam first-moment decay; for soap_sf: the Schedule-Free z↔x interpolation weight (not momentum). soap_sf commonly uses 0.9",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap||optimizer_type==soap_sf", advanced=True),
    )
    soap_beta2: float = Field(
        0.95, ge=0.0, lt=1.0,
        description="SOAP β2 (second-moment / eigenbasis covariance decay)",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap||optimizer_type==soap_sf", advanced=True),
    )
    soap_precondition_frequency: int = Field(
        10, ge=1,
        description="Refresh the Shampoo eigenbasis every N steps: higher = cheaper compute but a staler eigenbasis. Typical range 5-20",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap||optimizer_type==soap_sf", advanced=True),
    )
    soap_max_precond_dim: int = Field(
        10000, ge=1,
        description="Per-dimension threshold: an axis gets full-rank second-order preconditioning only if its size is ≤ this value, larger axes fall back to Adam. Setting this high (10000) lets even large feature dimensions get second-order treatment — the main source of the speedup; setting it low gives a SOAP-lite that saves VRAM",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap||optimizer_type==soap_sf", advanced=True),
    )
    soap_shampoo_beta: float = Field(
        -1.0, le=1.0,
        description="Shampoo covariance EMA decay; < 0 reuses β2 (recommended)",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap||optimizer_type==soap_sf", advanced=True),
    )
    soap_precond_in_state: bool = Field(
        True,
        description="Whether to store the recomputable Shampoo matrices (GG/Q) in the checkpoint. False=smaller checkpoints, cold-rebuilds the eigenbasis on resume (zero cost when training from scratch without resuming)",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap||optimizer_type==soap_sf", advanced=True),
    )
    soap_sf_weight_lr_power: float = Field(
        2.0, ge=0.0,
        description="Power applied to lr in the Schedule-Free Polyak weighting; higher favors steps with larger lr",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap_sf", advanced=True),
    )
    soap_sf_r: float = Field(
        0.0, ge=0.0,
        description="Power applied to the step index in the Schedule-Free Polyak weighting (0=uniform average; higher favors later iterates — x catches up to z faster in short training runs)",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap_sf", advanced=True),
    )
    soap_sf_warmup_steps: int = Field(
        0, ge=0,
        description="Schedule-Free linear lr warmup steps; SF generally doesn't need this, but a few steps can stabilize the early preconditioner estimate",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap_sf", advanced=True),
    )
    ema_enabled: bool = Field(
        False,
        description="[Weight EMA] Maintain an exponential moving average of the adapter weights during training, saved alongside normal checkpoints "
                    "as an extra *_ema.safetensors. The smoothed copy is more stable than any single point, overfitting degrades more gracefully, and "
                    "the \"which epoch is best\" guessing game mostly disappears. VRAM cost = one fp32 copy of the adapter parameters (small)",
        json_schema_extra=_meta("training", advanced=True),
    )
    ema_decay: float = Field(
        0.999, ge=0.9, le=0.99999,
        description="[Weight EMA] Smoothing factor: higher = longer window, steadier but more lagged. 0.999 ≈ the most recent 1000 update steps, "
                    "0.9999 ≈ the most recent 10000. Use 0.999 for LoRA runs of 2000-3000 total steps; lower to 0.99 for very short runs",
        json_schema_extra=_meta("training", show_when="ema_enabled==true", advanced=True),
    )
    ema_start_ratio: float = Field(
        0.0, ge=0.0, le=0.95,
        description="[Weight EMA] What fraction of total steps to wait before accumulating (0 = from the start). 0.3 means the volatile first 30% "
                    "isn't averaged in, only smoothing the part that's \"already decent\" — effectively an automatic version of \"average a few epochs from the plateau\"",
        json_schema_extra=_meta("training", show_when="ema_enabled==true", advanced=True),
    )
    weight_decay: float = Field(
        0.0, ge=0.0,
        description="Weight decay: higher = stronger suppression of weights, helps with overfitting; 0 = off. Common range 0.001-0.1; too high will break training",
        json_schema_extra=_meta("training", advanced=True),
    )
    kv_trim: bool = Field(
        False,
        description="Cross-attention KV trim: trims to the nearest bucket (64/128/256/512) by actual token count, reducing padding compute",
        json_schema_extra=_meta("system", advanced=True),
    )
    vae_tiling: Literal["auto", "on", "off"] = Field(
        "auto",
        description="VAE tiled decode: auto=automatically tile when available VRAM is tight (recommended); on=always tile (saves VRAM, ~30% slower); "
                    "off=full-image decode, falls back only on a genuine OOM. On high-VRAM cards, full-image decode near the VRAM ceiling can trigger a "
                    "system-memory fallback, degrading a single decode from under a second to over a hundred — auto avoids this",
        json_schema_extra=_meta("system", advanced=True),
    )
    noise_enhancement_type: Literal["none", "offset", "pyramid"] = Field(
        "none",
        description="Noise-enhancement mechanism (default none). offset adds a per-sample DC bias to the noise; pyramid layers low-frequency noise across multiple scales. The two mechanisms differ but both alter the low-frequency component, so they're mutually exclusive to avoid double-stacking. LoRA training defaults to none",
        json_schema_extra=_meta(
            "noise_augmentation",
            advanced=True,
            disable_when="infonoise_enabled==true",
            disable_hint="Noise enhancement is disabled while InfoNoise is enabled (mutually exclusive by schema)",
        ),
    )
    noise_offset: float = Field(
        0.0, ge=0.0, le=0.2,
        description="DC bias strength (0-0.2, 0=off). Shifts the noise mean away from 0, giving the model a chance to learn extreme-brightness scenes (pure black / pure white / strong contrast). Typical range 0.05-0.1; below 0.05 the noise field is nearly identical to baseline, above 0.1 the starting loss rises noticeably",
        json_schema_extra=_meta("noise_augmentation", show_when="noise_enhancement_type==offset", advanced=True),
    )
    pyramid_noise_iters: int = Field(
        0, ge=0, le=6,
        description="Number of pyramid noise layers (0-6, 0=off). Each layer injects noise at spatial // 2^(k+1) scale. Actual effect strength is set by pyramid_noise_discount — iters alone determines the frequency range covered, and when discount is low, the layer count matters little",
        json_schema_extra=_meta("noise_augmentation", show_when="noise_enhancement_type==pyramid", advanced=True),
    )
    pyramid_noise_discount: float = Field(
        0.5, ge=0.1, le=0.9,
        description="Per-layer relative decay factor (0.1-0.9). The core parameter controlling low-frequency strength: this implementation normalizes overall noise std to 1, so discount determines the low-frequency share. 0.1-0.4 normalizes to something close to standard Gaussian noise, roughly equivalent to off; 0.5-0.7 noticeably changes the low-frequency structure",
        json_schema_extra=_meta("noise_augmentation", show_when="noise_enhancement_type==pyramid", advanced=True),
    )
    timestep_sampling: Literal[
        "logit_normal",
        "uniform",
        "logit_normal_low",
        "mode",
        "mixed_uniform_low",
        "mixed_uniform_logit",
        "krea2_shift",
        "style_friendly",
        "dual_peak",
    ] = Field(
        "logit_normal",
        description="Sampling distribution. logit_normal is mid-range-weighted (SD3/Anima default); krea2_shift shifts dynamically by per-image token count; uniform is equal probability; mode is single-peak-shifted; mixed_* blends uniform with a biased end (ratio controlled by timestep_mix_low_prob); style_friendly samples directly on the log-SNR axis with a normal distribution, concentrating training on the high-noise window where style forms (arXiv 2411.14793, style-LoRA-specific, ignores timestep_shift)",
        json_schema_extra=_meta(
            "timestep_sampling",
            alt_description="[Timestep sampling] When InfoNoise is enabled, this acts as the warmup-period baseline — the adaptive CDF takes over for the main phase; when Leap is enabled, the leap path always uses U(0,1), and this field only applies to the (1-leap_ratio) fraction of standard steps",
            alt_description_when="infonoise_enabled==true||leap_enabled==true",
            advanced=True,
            # krea2_shift's mu interpolation is calibrated for K2 and is
            # UI-only for other families; mechanically the shared loop can
            # run for any family — the backend doesn't gate it (A1), and a
            # hand-edited yaml is honored
            option_show_when=option_gates(TIMESTEP_SAMPLING_OPTION_FAMILIES),
        ),
    )
    timestep_shift: float = Field(
        3.0, ge=0.1, le=10.0,
        description="Distribution shift inside logit-normal / mode: >1 shifts toward the high-noise end (coarse structure), <1 shifts toward the low-noise end (detail)",
        json_schema_extra=_meta(
            "timestep_sampling",
            # style_friendly's shift is fully determined by style_snr_mean;
            # stacking both would double-shift
            show_when="timestep_sampling!=uniform&&timestep_sampling!=style_friendly&&timestep_sampling!=dual_peak",
            alt_description="When InfoNoise is on, acts as the warmup-phase baseline shift — the adaptive CDF takes over for the main phase; when Leap is enabled, the leap path always uses U(0,1), and this field only applies to the (1-leap_ratio) fraction of standard steps",
            alt_description_when="infonoise_enabled==true||leap_enabled==true",
            advanced=True,
        ),
    )
    timestep_mix_low_prob: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Fraction of samples routed to the biased end under mixed_* modes: 0 = fully uniform; typical range 0.15-0.30",
        json_schema_extra=_meta(
            "timestep_sampling",
            show_when="timestep_sampling!=uniform&&timestep_sampling!=dual_peak",
            alt_description="With InfoNoise on + a mixed_* baseline, this is the warmup-phase mix ratio — the adaptive CDF takes over for the main phase; when Leap is enabled, the leap path always uses U(0,1), and this field only applies to the (1-leap_ratio) fraction of standard steps",
            alt_description_when="infonoise_enabled==true||leap_enabled==true",
            advanced=True,
        ),
    )
    timestep_schedule_shift: float = Field(
        1.0, ge=0.1, le=10.0,
        description="Extra σ-schedule shift applied to t after sampling: 1.0 = no shift; higher values bias the whole distribution toward the high-noise end. Differs from timestep_shift by acting on the final t rather than inside logit-normal",
        json_schema_extra=_meta(
            "timestep_sampling",
            alt_description="Only takes effect during InfoNoise's warmup period — the adaptive CDF takes over for the main phase; when Leap is enabled, the leap path always uses U(0,1), and this field only applies to the (1-leap_ratio) fraction of standard steps",
            alt_description_when="infonoise_enabled==true||leap_enabled==true",
            advanced=True,
            disable_when="infonoise_enabled==true",
            disable_hint="Schedule shift is disabled while InfoNoise is enabled (mutually exclusive by schema — only 1.0 is compatible)",
        ),
    )
    style_snr_mean: float = Field(
        -6.0, ge=-12.0, le=6.0,
        description="[Style-Friendly SNR] log-SNR sampling mean m: training t is sampled as λ~N(m, σ²), t=sigmoid(-λ/2). "
                    "Smaller values lean toward the high-noise end (style/lighting/composition), larger values lean toward the low-noise end (texture detail). "
                    "The FLUX/SD3.5 paper recipe uses -6 (median t≈0.95); raise to -4 ~ -3 to retain a bit more fine-detail refinement",
        json_schema_extra=_meta(
            "timestep_sampling",
            show_when="timestep_sampling==style_friendly",
            advanced=True,
        ),
    )
    style_snr_sigma: float = Field(
        2.0, gt=0.0, le=6.0,
        description="[Style-Friendly SNR] log-SNR sampling standard deviation σ: window width. Small = concentrated near the mean but narrow coverage, "
                    "easy to overfit that band; large = broad coverage but diluted. Paper recommends 2.0-3.0",
        json_schema_extra=_meta(
            "timestep_sampling",
            show_when="timestep_sampling==style_friendly",
            advanced=True,
        ),
    )
    dual_peak_peak1_position: float = Field(
        0.525, gt=0.0, lt=1.0, allow_inf_nan=False,
        description="First component peak in flow t: 0.525 = timestep 525/1000. The mixture maximum can move.",
        json_schema_extra=_meta(
            "timestep_sampling", show_when="timestep_sampling==dual_peak", advanced=True,
        ),
    )
    dual_peak_peak1_width: float = Field(
        0.55, gt=0.0, le=2.8, allow_inf_nan=False,
        description="First peak width (log-SNR standard deviation). Larger = wider, lower peak at the same weight.",
        json_schema_extra=_meta(
            "timestep_sampling", show_when="timestep_sampling==dual_peak", advanced=True,
        ),
    )
    dual_peak_peak1_weight: float = Field(
        0.15, ge=0.0, le=1.0, allow_inf_nan=False,
        description="First peak relative weight. All four weights are normalized together; zero disables this component.",
        json_schema_extra=_meta(
            "timestep_sampling", show_when="timestep_sampling==dual_peak", advanced=True,
        ),
    )
    dual_peak_peak2_position: float = Field(
        0.85, gt=0.0, lt=1.0, allow_inf_nan=False,
        description="Second component peak in flow t: 0.85 = timestep 850/1000. Defaults put the mixture maximum near 870.",
        json_schema_extra=_meta(
            "timestep_sampling", show_when="timestep_sampling==dual_peak", advanced=True,
        ),
    )
    dual_peak_peak2_width: float = Field(
        1.2, gt=0.0, le=2.8, allow_inf_nan=False,
        description="Second peak width (log-SNR standard deviation). Independent of the first peak.",
        json_schema_extra=_meta(
            "timestep_sampling", show_when="timestep_sampling==dual_peak", advanced=True,
        ),
    )
    dual_peak_peak2_weight: float = Field(
        0.35, ge=0.0, le=1.0, allow_inf_nan=False,
        description="Second peak relative weight. Normalized with the other three weights.",
        json_schema_extra=_meta(
            "timestep_sampling", show_when="timestep_sampling==dual_peak", advanced=True,
        ),
    )
    dual_peak_background_mean: float = Field(
        -3.2, ge=-12.0, le=6.0, allow_inf_nan=False,
        description="Broad Style-Friendly component log-SNR mean. Lower moves it toward high noise. This covers the whole interval, not only the right tail.",
        json_schema_extra=_meta(
            "timestep_sampling", show_when="timestep_sampling==dual_peak", advanced=True,
        ),
    )
    dual_peak_background_width: float = Field(
        2.2, gt=0.0, le=6.0, allow_inf_nan=False,
        description="Broad Style-Friendly component log-SNR standard deviation.",
        json_schema_extra=_meta(
            "timestep_sampling", show_when="timestep_sampling==dual_peak", advanced=True,
        ),
    )
    dual_peak_background_weight: float = Field(
        0.45, ge=0.0, le=1.0, allow_inf_nan=False,
        description="Broad background relative weight. Zero removes the broad component.",
        json_schema_extra=_meta(
            "timestep_sampling", show_when="timestep_sampling==dual_peak", advanced=True,
        ),
    )
    dual_peak_uniform_weight: float = Field(
        0.05, ge=0.0, le=1.0, allow_inf_nan=False,
        description="Uniform coverage relative weight. Default 0.05, normalized with other weights.",
        json_schema_extra=_meta(
            "timestep_sampling", show_when="timestep_sampling==dual_peak", advanced=True,
        ),
    )
    timestep_shift_resolution_aware: bool = Field(
        False,
        description="Applies a resolution correction to sampled t based on per-image token count (SD3-style s=sqrt(this image's token count / reference tier's token count), "
                    "where the reference tier is the first entry in resolution): images at the reference size are unchanged, larger images shift toward high noise, smaller ones toward low noise. "
                    "Under multi-resolution and NaViT native-resolution training, images of every size land at a noise level equivalent to the reference tier; has no effect under single-resolution training",
        json_schema_extra=_meta(
            "timestep_sampling",
            alt_description="When Leap is enabled, the leap path's t_k/t_j are not affected by this correction — it only applies to the (1-leap_ratio) fraction of standard steps",
            alt_description_when="leap_enabled==true",
            advanced=True,
        ),
    )
    infonoise_enabled: bool = Field(
        False,
        description="[InfoNoise] Enable adaptive timestep sampling: during training, the t distribution is automatically adjusted based on information content, focusing on the most informative training range",
        json_schema_extra=_meta(
            "timestep_sampling",
            advanced=True,
            disable_when=(
                "noise_enhancement_type!=none"
                "||loss_weighting!=none"
                "||loss_type==huber"
                "||timestep_schedule_shift!=1"
                "||leap_enabled==true"
                "||navit_packing==true"
            ),
            disable_hint="Cannot be enabled while a mutually exclusive field (noise_enhancement / loss_weighting / loss_type / schedule_shift / leap / navit) is non-default (mutually exclusive by schema)",
        ),
    )
    infonoise_K: int = Field(
        64, ge=16, le=256,
        description="[InfoNoise] Number of log-σ bins (16-256): higher = finer resolution but sparser samples per bin",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_N_warm: int = Field(
        0, ge=0,
        description="[InfoNoise] Warmup steps: 0 = automatically 1/5 of total steps (minimum 200)",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_M: int = Field(
        100, ge=10,
        description="[InfoNoise] Sampling-distribution refresh period: recompute every M steps. Higher = less compute overhead but more lag in the distribution update",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_B: int = Field(
        256, ge=32,
        description="[InfoNoise] FIFO buffer capacity per bin: higher = steadier average but slower to respond",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_beta: float = Field(
        0.9, ge=0.1, le=0.999,
        description="[InfoNoise] EMA weight for the new value (per the paper, β multiplies the new value, not the standard EMA direction): 0.9 means the new value carries 90% weight; higher = faster response to the latest distribution",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_N_min: int = Field(
        50, ge=1,
        description="[InfoNoise] Refresh trigger: minimum samples required per bin before recomputing the distribution (must be ≤ infonoise_B)",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_gate_pivot_c: float = Field(
        0.15, ge=0.0, le=10.0,
        description="[InfoNoise] Gate function pivot c: default 0.15 (the paper's §5 CIFAR-reported value, robust across datasets); set to 0 for adaptive selection (literal implementation of paper Eq 87); other positive values are custom c. Keep the default in most cases",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    loss_type: Literal["mse", "huber"] = Field(
        "mse",
        description="Training loss type. mse classic; huber robust to outliers (quadratic when |x|<δ, linear when |x|≥δ)",
        json_schema_extra=_meta(
            "loss",
            disable_when="infonoise_enabled==true||leap_enabled==true",
            disable_hint="Loss-type switching is disabled while InfoNoise / Leap is enabled (mutually exclusive by schema — only mse is compatible)",
        ),
    )
    huber_c: float = Field(
        0.15, ge=0.01, le=5.0,
        description="[Huber loss] Delta coefficient (controls the quadratic/linear transition point): higher = closer to MSE, lower = more tolerant of outliers. Typical range 0.1-0.3",
        json_schema_extra=_meta("loss", show_when="loss_type==huber", advanced=True),
    )
    loss_weighting: Literal["none", "min_snr", "detail_inv_t", "cosmap"] = Field(
        "none",
        description="Loss weighting scheme: none unweighted; min_snr suppresses weight at extreme timesteps; detail_inv_t emphasizes low-t detail; cosmap uses the SD3 cosine mapping",
        json_schema_extra=_meta(
            "loss",
            disable_when="infonoise_enabled==true||leap_enabled==true",
            disable_hint="Loss weighting is disabled while InfoNoise / Leap is enabled (mutually exclusive by schema — only none is compatible)",
        ),
    )
    masked_loss: bool = Field(
        False,
        description="Weight loss by the training mask: regions painted in the preprocessing mask page (grayscale 0=don't learn, 255=learn normally, in-between=partial weight) produce no gradient, and generation in that region is governed by the base model's prior; images without a mask are unaffected",
        json_schema_extra=_meta(
            "loss",
            show_when=cap_gate("masked_loss"),
            disable_when="leap_enabled==true||navit_packing==true",
            disable_hint="Leap (per-sample loss has no spatial dimension) / NaViT packing paths don't support masks (mutually exclusive by schema)",
        ),
    )
    min_snr_gamma: float = Field(
        5.0, ge=0.1, le=20.0,
        description="Min-SNR threshold: weight-suppression threshold for high-SNR, easy steps (low-t end). Default 5.0; smaller = stronger suppression",
        json_schema_extra=_meta("loss", show_when="loss_weighting==min_snr", advanced=True),
    )
    weight_cap_ratio: float = Field(
        0.0, ge=0.0, le=50.0,
        description="Max/min weight ratio ceiling within a batch: caps the influence of extreme weights. 0 = disabled; recommend 5 for small batches + Prodigy",
        json_schema_extra=_meta("loss", show_when="loss_weighting!=none", advanced=True),
    )
    detail_inv_t_min: float = Field(
        1.0, ge=1.0, le=20.0,
        description="detail_inv_t weight floor. Default 1.0; raising to 1.5 gives high-t steps slightly more weight too (<1.0 has no effect since 1/t≥1 always holds)",
        json_schema_extra=_meta("loss", show_when="loss_weighting==detail_inv_t", advanced=True),
    )
    detail_inv_t_max: float = Field(
        5.0, ge=0.1, le=50.0,
        description="detail_inv_t weight ceiling. Default 5.0; lower it (e.g. 3) to soften the detail boost, raise it (e.g. 8) for a more aggressive detail boost",
        json_schema_extra=_meta("loss", show_when="loss_weighting==detail_inv_t", advanced=True),
    )
    leap_enabled: bool = Field(
        False,
        description="[LeapAlign self-distillation] Enable two-step leap self-distillation (reward-model-free version): each step uses the real latent as x0, samples two per-sample timesteps (k>j), and takes two leap steps, loss=MSE(the two-step prediction of x̂0, real x0). Essentially a shortcut/consistency-style self-distillation. Cost: at leap_ratio=1.0, each step does 2 forward passes (~2x compute, and activation VRAM near 2x since both forward passes carry gradients). Mutually exclusive with InfoNoise / loss_weighting / loss_type=huber",
        json_schema_extra=_meta(
            "loss",
            advanced=True,
            show_when=cap_gate("leap"),
            disable_when="infonoise_enabled==true||loss_weighting!=none||loss_type==huber||navit_packing==true",
            disable_hint="Cannot be enabled while a mutually exclusive field (InfoNoise / loss_weighting / loss_type=huber / navit) is non-default (mutually exclusive by schema, forming a symmetric lock with the other side)",
        ),
    )
    leap_ratio: float = Field(
        0.6, ge=0.0, le=1.0,
        description="[LeapAlign mixed training] Fraction of steps that take the leap self-distillation path, the rest using traditional rectified flow: 1.0 pure leap (handles global structure); 0.0 pure traditional (handles fine detail sharpness); 0.6 lets leap handle most of the global alignment while traditional refinement handles the details. Both gradients accumulate on the same LoRA weights, each contributing its strength",
        json_schema_extra=_meta("loss", show_when="leap_enabled==true", advanced=True),
    )
    leap_variant: Literal["original", "sparse", "bridge", "lagrange"] = Field(
        "original",
        description="[LeapAlign/FlowBP] Trajectory self-distillation variant (unified form: analytically construct trajectory points + integrate along the trajectory to x̂0 + MSE(x̂0, real x0)): original=two-step leap + straight-through connector (K=2, 1 Jacobian, matches historical behavior, default); sparse=K-point Euler replay summing pure direct terms (FlowBP-Sparse, zero connector/zero Jacobian, K dense supervision points, most stable, at the cost of K× forward passes + K× VRAM, K controlled by leap_activation_k); bridge=two-step leap + Euler-reconstructed connector (FlowBP-Bridge, no straight-through bias); lagrange=two-segment leap with three-point Lagrange/Simpson integration per segment (FlowBP-Lagrange, 6× forward passes, per-segment integration error O(Δt²)→O(Δt⁵), paper §A.2). Note: under self-distillation the ground truth is an analytic straight-line interpolation point with no rollout noise, so the connector residual is undercut — bridge/lagrange gains over original narrow accordingly, and sparse is the only structurally different variant",
        json_schema_extra=_meta("loss", show_when="leap_enabled==true", advanced=True),
    )
    leap_activation_k: int = Field(
        3, ge=2, le=8,
        description="[FlowBP-Sparse] Activation set size K: samples K descending timesteps via stratified jitter over (0,1) and replays them with Euler, all K points carrying gradients. K directly determines VRAM/compute (K× forward passes + K× activation VRAM) and supervision density. 3 balances VRAM against density (a bit heavier than original's 2x); consumer cards with limited VRAM can use 2 (degrades to original's VRAM tier); 4+ gives denser supervision but may strain a 12GB card. Only applies to the sparse variant",
        json_schema_extra=_meta("loss", show_when="leap_enabled==true&&leap_variant==sparse", advanced=True),
    )
    leap_nested_grad_coe: float = Field(
        0.3, ge=0.0, le=1.0,
        description="[LeapAlign] Gradient discount α (paper Eq 9): scales the nested gradient of the second leap w.r.t. x_j. 0=drop the nested gradient entirely (most VRAM-efficient), 1=no discount (most complete gradient but prone to blowing up). Paper's optimum is 0.3. Applies to original/bridge/lagrange; sparse has zero connector/zero Jacobian and doesn't use this parameter",
        json_schema_extra=_meta("loss", show_when="leap_enabled==true&&leap_variant!=sparse", advanced=True),
    )
    leap_min_gap: float = Field(
        0.1, ge=0.01, le=0.9,
        description="[LeapAlign] Minimum gap between the two sampled timesteps (k,j): larger = bigger leap span, more aggressive self-distillation but more accumulated error. Typical range 0.1-0.3. Only applies to original/bridge/lagrange; sparse's activation set uses stratified jitter spread across (0,1) and doesn't use this field",
        json_schema_extra=_meta("loss", show_when="leap_enabled==true&&leap_variant!=sparse", advanced=True),
    )
    leap_traj_sim_weighting: bool = Field(
        False,
        description="[LeapAlign] Trajectory-similarity weighting (paper Eq 12): the closer a leap stays to the real path, the higher its weight, suppressing wild predictions from large-span leaps from dominating the loss. Off by default",
        json_schema_extra=_meta("loss", show_when="leap_enabled==true", advanced=True),
    )
    leap_traj_sim_min: float = Field(
        0.1, ge=1e-4,
        description="[LeapAlign] Trajectory-similarity weighting floor τ: prevents near-identical leap pairs from being over-amplified by 1/d. Smaller = more aggressive. Typical range 0.05-0.2",
        json_schema_extra=_meta("loss", show_when="leap_traj_sim_weighting==true", advanced=True),
    )

    # ----------------------------------------------------------- SRA v2 representation alignment
    dop_enabled: bool = Field(
        False,
        description="[DOP differential output preservation] Requires no extra regularization images: takes the same batch of training images, strips the "
                    "trigger word from the caption, and compares the prediction \"with the adapter on\" against \"with the adapter off,\" using the "
                    "difference as a penalty. This effectively teaches two things at once — with the trigger word: apply my style; without it: change "
                    "nothing. Since the two branches share identical content, \"copying dataset content\" earns no reward, specifically curing style "
                    "LoRAs that drag characters/backgrounds along into generations. Requires a non-empty trigger_word. "
                    "Cost: enabling it adds two extra forward passes per step (roughly 2-2.5x per-step time)",
        json_schema_extra=_meta(
            "loss",
            advanced=True,
            disable_when="leap_enabled==true||navit_packing==true",
            disable_hint="DOP v1 only supports the standard rectified-flow path (Leap has its own objective; "
                         "NaViT's per-image packing would need cross-attention repacking, not adapted in v1)",
        ),
    )
    dop_weight: float = Field(
        1.0, ge=0.0, le=100.0,
        description="[DOP] Weight of the preservation term relative to the main loss. Both are MSE in the same space, so 1.0 is equal weighting; "
                    "raise to 2-5 if style still leaks into trigger-word-free prompts, lower to 0.3-0.5 if style gets suppressed and fails to learn",
        json_schema_extra=_meta("loss", show_when="dop_enabled==true", advanced=True),
    )
    dop_ratio: float = Field(
        1.0, ge=0.0, le=1.0,
        description="[DOP] Fraction of steps it's enabled on (1.0=every step). Purely a speed/strength trade-off: 0.5 means half the steps do the "
                    "extra two forward passes while the other half run at normal speed, and constraint strength is halved accordingly",
        json_schema_extra=_meta("loss", show_when="dop_enabled==true", advanced=True),
    )
    sra_enabled: bool = Field(
        False,
        description="[SRA v2 representation alignment] Enable VAE Self-Representation Alignment: aligns an intermediate transformer block's hidden state to the clean VAE latent, speeding convergence and regularizing the representation. Adds only ~4% GFLOPs (a lightweight MLP), discarded automatically after training",
        json_schema_extra=_meta(
            "loss", advanced=True, show_when=cap_gate("sra"),
            # Audit #1 (design doc §10.1): leap steps skip SRA entirely
            # (guarded in loop.py) — at leap_ratio=1.0, SRA has zero effect
            # silently; the navit path is guarded the same way
            disable_when="leap_enabled==true||navit_packing==true",
            disable_hint="The Leap / NaViT paths skip SRA computation (per-image t / packed sequences aren't adapted for it) — has no effect even when enabled",
        ),
    )
    sra_block: int = Field(
        4, ge=1, le=35,
        description="[SRA v2] Which block layer to take the intermediate representation from for alignment (0-indexed). The paper recommends shallower layers work best",
        json_schema_extra=_meta("loss", show_when="sra_enabled==true", advanced=True),
    )
    sra_weight: float = Field(
        0.2, ge=0.0,
        description="[SRA v2] Alignment loss weight λ: align_loss is multiplied by this value before being added to the total loss. The trainer defaults to 0.2; too large will cause instability",
        json_schema_extra=_meta("loss", show_when="sra_enabled==true", advanced=True),
    )
    sra_normalize: bool = Field(
        True,
        description="[SRA v2] Apply per-sample z-score normalization to projected/target separately before computing smooth-L1 (same family of ideas as the paper's cosine ablation: magnitude-agnostic, aligning structure only). The original paper trains from scratch with SD-VAE (latents at roughly unit scale) so it skips normalization; this project's video-VAE latents are at a different scale, plus LoRA fine-tuning, so turning this off causes the align loss to run several orders of magnitude above the denoise loss and collapse quickly. Recommend keeping it on",
        json_schema_extra=_meta("loss", show_when="sra_enabled==true", advanced=True),
    )
    sra_decay_type: Literal["none", "linear", "cosine", "jump"] = Field(
        "linear",
        description="[SRA v2] Weight decay method: none=fixed throughout; linear=linearly decays to 0 from the start point; cosine=cosine-decays to 0 from the start point; jump=turns off immediately at the start point. Effective weight = sra_weight × decay factor",
        json_schema_extra=_meta("loss", show_when="sra_enabled==true", advanced=True),
    )
    sra_decay_start_ratio: float = Field(
        0.2, ge=0.0, le=1.0,
        description="[SRA v2] Decay start point (as a fraction of total training steps). linear/cosine hold full weight before this point; jump drops straight from sra_weight to 0 at this point",
        json_schema_extra=_meta("loss", show_when="sra_enabled==true&&sra_decay_type!=none", advanced=True),
    )
    sra_decay_end_ratio: float = Field(
        0.3, ge=0.0, le=1.0,
        description="[SRA v2] Decay end point (as a fraction of total training steps). linear/cosine reach 0 by this point; jump doesn't use this field",
        json_schema_extra=_meta("loss", show_when="sra_enabled==true&&sra_decay_type!=none&&sra_decay_type!=jump", advanced=True),
    )

    grad_clip_max_norm: float = Field(
        1.0, ge=0.0,
        description="Gradient clipping max norm: when the global gradient norm across all trainable parameters for this step exceeds this value, it's scaled down proportionally to prevent a single extreme-gradient step from destabilizing the model; the default 1.0 suits most cases, lower to 0.5 if bf16+DoRA/LoKr is unstable, 0=disabled",
        json_schema_extra=_meta(
            "training", advanced=True,
            # Audit #6 (design doc §10.1): automagic v2's fused backward
            # updates parameters in place, so clipping after the fact
            # silently does nothing (the optimizer only logs a one-line
            # warning); the 1.0 default is exactly the trap everyone falls
            # into — pinning the config layer to 0 makes it explicit that
            # "this combination has no clipping"
            disable_when="optimizer_type==automagic&&automagic_variant==v2",
            disable_value=0.0,
            disable_hint="Automagic v2's fused backward updates parameters in place, so gradient clipping cannot take effect",
        ),
    )

    mixed_precision: Literal["bf16", "fp16", "no"] = Field(
        "bf16",
        description="Training precision. bf16 recommended (same dynamic range as fp32, stable); fp16 uses the same VRAM but has a smaller dynamic range and gradients overflow more easily; no uses fp32, most stable but doubles VRAM use",
        json_schema_extra=_meta("system"),
    )

    attention_backend: AttentionBackend = Field(
        "flash_attn",
        description="Attention backend. none = default PyTorch SDPA; xformers saves more VRAM; flash_attn is fastest (requires an Ampere+ GPU)",
        json_schema_extra=_meta(
            "system",
            disable_when="navit_packing==true",
            disable_value="xformers",
            disable_hint="NaViT packing already forces xformers varlen (required for block-diagonal packing; install xformers)",
        ),
    )
    num_workers: int = Field(
        0, ge=0,
        description="Number of parallel data-loading threads; higher = faster loading but more RAM use. Must be 0 on Windows",
        json_schema_extra=_meta("system", advanced=True),
    )

    @field_validator("resolution", mode="before")
    @classmethod
    def _normalize_resolution(cls, v: Any) -> list[int]:
        """Scalar / list / legacy-config scalar → normalized to list[int].

        Each value is snapped to the nearest multiple of 64 (half-up, matching the
        frontend's `Math.round` and the dataset's `_parse_folder_meta`, to avoid
        off-center buckets) + clamped to [256, 4096], then **deduplicated while
        preserving order** (otherwise `[1000, 1024]` would both snap to 1024 →
        that tier gets trained twice via fan-out and double-counted in the
        histogram).
        """
        if v is None:
            return [1024]
        if isinstance(v, (int, float, str)):
            v = [v]
        out: list[int] = []
        seen: set[int] = set()
        for x in v:
            n = int((float(x) + 32) // 64) * 64  # round-half-up to /64
            n = max(256, min(4096, n))
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out or [1024]

    @model_validator(mode="after")
    def _validate_family_capabilities(self):
        # Second line of defense for multi-model PR-3 (first line = show_when
        # expanded at author time; third line = trainer bootstrap): catches
        # a hand-written yaml / bare CLI enabling a capability the current
        # family doesn't support
        bad = capability_violations(self.model_family, self.__dict__)
        if bad:
            raise ValueError(
                f"model_family='{self.model_family}' does not support these enabled fields: {bad}"
                f"(per-family capabilities are listed in studio/domain/common.py FAMILY_CAPABILITIES)"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _apply_family_config_defaults(cls, data: Any) -> Any:
        """Overlay ModelSpec defaults only when a family-specific field is absent."""
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        family = str(payload.get("model_family") or "anima")
        for field, value in FAMILY_CONFIG_DEFAULTS.get(family, {}).items():
            payload.setdefault(field, value)
        return payload

    @model_validator(mode="before")
    @classmethod
    def _migrate_save_keys(cls, data: Any) -> Any:
        return migrate_legacy_save_keys(data)

    @model_validator(mode="before")
    @classmethod
    def _migrate_fork_bucket_keys(cls, data: Any) -> Any:
        """This fork's legacy-config compatibility: ``bucket_max_ar`` → ``aspect_ratio_limit``.

        The old fork named the bucket aspect-ratio ceiling bucket_max_ar (same
        semantics as upstream's aspect_ratio_limit). When aspect_ratio_limit is
        given explicitly, it takes precedence.
        """
        if isinstance(data, dict):
            ar = data.pop("bucket_max_ar", None)
            if ar is not None and data.get("aspect_ratio_limit") is None:
                data["aspect_ratio_limit"] = ar
        return data

    @model_validator(mode="before")
    @classmethod
    def _migrate_noise_enhancement(cls, data: Any) -> Any:
        return migrate_noise_enhancement_type(data)

    @model_validator(mode="before")
    @classmethod
    def _coerce_sample_sampler_scheduler(cls, data: Any) -> Any:
        """Per-family handling of sampler/scheduler values outside the whitelist (multi-model P4-2, also settles debt from #419):

        - **Families with legacy corpora** (anima, predating the #256 Literal
          tightening): values outside the whitelist are always silently
          merged into the family default — preserving #256's migration
          contract as-is (historical values like euler must still let old
          configs load). The C1 pathology of "the UI offers an option that
          then gets silently rewritten" is eliminated at the root by the
          option_show_when gate: the UI no longer shows euler to anima.
        - **Families born in the Literal era** (krea2 onward): no legacy
          corpus. Junk values outside the Literal are still merged (for
          load robustness); **a value that's inside the Literal but that
          this family can't run raises an error** (both families' sample_image
          raise at the entry point, so the config layer fails fast earlier)
          instead of silently rewriting an explicit config.
        """
        if not isinstance(data, dict):
            return data
        family = str(data.get("model_family") or "anima")
        allowed = FAMILY_SAMPLING.get(family)
        if allowed is None:
            return data  # unknown family is caught by model_family's Literal validation elsewhere, not repeated here
        union: dict[str, tuple[str, ...]] = {}
        for spec in FAMILY_SAMPLING.values():
            for kind in ("samplers", "schedulers"):
                union[kind] = tuple(dict.fromkeys(union.get(kind, ()) + spec[kind]))
        legacy = family in LEGACY_SAMPLING_FAMILIES
        for field, kind in (
            ("sample_sampler_name", "samplers"),
            ("sample_scheduler", "schedulers"),
        ):
            value = data.get(field)
            if value is None or value in allowed[kind]:
                continue
            if legacy or value not in union[kind]:
                data[field] = allowed[kind][0]  # grandfather / junk value → family default
            else:
                raise ValueError(
                    f"{field}='{value}' does not apply to model_family='{family}'"
                    f" (available for this family: {', '.join(allowed[kind])})"
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def _pin_setdefaults(cls, data: Any) -> Any:
        """Construction-time setdefault for pin rules: "defaults follow the pin, only an explicit violation raises."

        `TrainingConfig(navit_packing=True)` sets unspecified pinned fields like
        attention_backend / cache_latents to their pinned values automatically
        (same semantics as the frontend's takeover); a value that's given
        explicitly and violates the pin is left for _enforce_disable_rules to
        fail fast — an explicit user configuration is never silently rewritten
        (replaces the old _coerce_navit_attention_backend's blanket coercion).
        """
        if not isinstance(data, dict):
            return data
        from .config_rules import apply_pin_setdefaults

        return apply_pin_setdefaults(data, cls)

    @model_validator(mode="after")
    def _enforce_disable_rules(self) -> "TrainingConfig":
        """Two-sided enforcement of disable_when declarations (Blade 2 / R2 v2, design doc §6).

        A field's disable_when + disable_value + disable_hint / option_disable_when
        is a single source of truth: frontend graying-out + takeover, this
        validator, the tolerant-load fixer (apply_disable_rule_fixes), and the
        R6 confirmation dialog all derive from the same declaration.
        Historically hand-written pairwise mutual-exclusion validators
        (prodigy×scheduler, infonoise×4, leap×3, navit×4, navit→cache_latents,
        navit→attention_backend coerce) are all replaced by this; the domain
        rationale lives in each field's disable_hint.
        """
        from .config_rules import disable_rule_violations

        violations = disable_rule_violations(self.__dict__, type(self))
        if violations:
            lines = []
            for v in violations:
                if v["kind"] == "pin":
                    lines.append(
                        f"{v['field']}={v['actual']!r} must be "
                        f"{v['expected']!r} under the current configuration: {v['hint']}"
                    )
                else:
                    lines.append(
                        f"{v['field']}={v['actual']!r} is not allowed under the current configuration: {v['hint']}"
                    )
            raise ValueError("A cross-field constraint is not satisfied — " + "; ".join(lines))
        return self

    @model_validator(mode="after")
    def _validate_dual_peak_weights(self) -> "TrainingConfig":
        if self.timestep_sampling == "dual_peak" and (
            self.dual_peak_peak1_weight + self.dual_peak_peak2_weight
            + self.dual_peak_background_weight + self.dual_peak_uniform_weight
        ) <= 0.0:
            raise ValueError("dual_peak: at least one mixture weight must be positive")
        return self

    @model_validator(mode="after")
    def _validate_custom_lr_schedulers(self) -> "TrainingConfig":
        if self.lr_scheduler == "cosine_cycles":
            count = self.lr_scheduler_cycle_count
            for name, values in (
                ("lr_scheduler_cycle_max_lrs", self.lr_scheduler_cycle_max_lrs),
                ("lr_scheduler_cycle_min_lrs", self.lr_scheduler_cycle_min_lrs),
            ):
                if values and len(values) != count:
                    raise ValueError(
                        f"{name} must be empty or contain exactly {count} values"
                    )

            common_max = self.lr_scheduler_cycle_max_lr or self.learning_rate
            max_lrs = self.lr_scheduler_cycle_max_lrs or [common_max] * count
            min_lrs = self.lr_scheduler_cycle_min_lrs or [self.lr_scheduler_cycle_min_lr] * count
            for index, (min_lr, max_lr) in enumerate(zip(min_lrs, max_lrs), start=1):
                if min_lr < 0.0 or max_lr <= 0.0:
                    raise ValueError(
                        f"cosine_cycles cycle {index}: min LR must be >= 0 and max LR > 0"
                    )
                if min_lr > max_lr:
                    raise ValueError(
                        f"cosine_cycles cycle {index}: min LR ({min_lr}) "
                        f"cannot exceed max LR ({max_lr})"
                    )

        if (
            self.lr_scheduler == "constant_then_cosine"
            and self.lr_scheduler_decay_min_lr > self.learning_rate
        ):
            raise ValueError(
                "lr_scheduler_decay_min_lr cannot exceed learning_rate for "
                "constant_then_cosine"
            )
        return self

    @model_validator(mode="after")
    def _validate_detail_inv_t_range(self) -> "TrainingConfig":
        """The detail_inv_t weighting curve requires min <= max; fail-fast replaces the historical silent swap."""
        if self.detail_inv_t_min > self.detail_inv_t_max:
            raise ValueError(
                f"detail_inv_t_min ({self.detail_inv_t_min}) cannot be greater than "
                f"detail_inv_t_max ({self.detail_inv_t_max})."
            )
        return self

    @model_validator(mode="after")
    def _validate_sra_decay_range(self) -> "TrainingConfig":
        """SRA linear/cosine decay requires start <= end; jump only reads start."""
        if self.sra_decay_type in {"linear", "cosine"} and self.sra_decay_start_ratio > self.sra_decay_end_ratio:
            raise ValueError(
                f"sra_decay_start_ratio ({self.sra_decay_start_ratio}) cannot be greater than "
                f"sra_decay_end_ratio ({self.sra_decay_end_ratio})."
            )
        return self

    @model_validator(mode="after")
    def _validate_infonoise_n_min_le_b(self) -> "TrainingConfig":
        """N_min > B means the adaptive distribution can never converge (FIFO capacity is insufficient to trigger a refresh)."""
        if self.infonoise_enabled and self.infonoise_N_min > self.infonoise_B:
            raise ValueError(
                f"infonoise_N_min ({self.infonoise_N_min}) cannot be greater than "
                f"infonoise_B ({self.infonoise_B}): above it the adaptive distribution can never converge."
            )
        return self

    @model_validator(mode="after")
    def _validate_navit_prerequisites(self) -> "TrainingConfig":
        """Non-pin/non-forbid prerequisite checks for NaViT packing (§6.4, kept hand-written).

        Mutually exclusive items (leap / infonoise / sra / tlora / cache_latents /
        attention_backend) and the native→packing prerequisite
        (navit_native_resolution's disable_when) are already enforced via
        disable_when / option_disable_when declarations through
        _enforce_disable_rules; domain rationale is in each field's
        disable_hint and docs/navit-packing.md section 3.
        """
        if self.navit_packing and self.navit_token_budget <= 0:
            raise ValueError(
                "navit_packing needs navit_token_budget set explicitly (>0, sized to your VRAM, "
                "see the VRAM table in docs/navit-packing.md)."
            )
        return self

    # ---------------------------------------------------------------- Output/saving
    output_dir: str = Field(
        "./output",
        description="Output directory",
        json_schema_extra=_meta("output", "path"),
    )
    output_name: str = Field(
        "anima_lora",
        description="Output filename prefix",
        json_schema_extra=_meta("output"),
    )
    save_every_epochs: int = Field(
        2, ge=0,
        description="Save every N epochs (0=disabled)",
        json_schema_extra=_meta("output"),
    )
    save_every_steps: int = Field(
        0, ge=0,
        description="Save every N steps (0=disabled)",
        json_schema_extra=_meta("output"),
    )
    save_state_every_epochs: int = Field(
        0, ge=0,
        description="Save full training state every N epochs (for resuming, 0=disabled)",
        json_schema_extra=_meta("output"),
    )
    save_state_every_steps: int = Field(
        0, ge=0,
        description="Save full training state every N steps (for resuming, 0=disabled)",
        json_schema_extra=_meta("output"),
    )
    seed: int = Field(
        42,
        description="Training random seed",
        json_schema_extra=_meta("output"),
    )
    resume_lora: Optional[str] = Field(
        None,
        description="Resume training from an existing LoRA (weights only)",
        json_schema_extra=_meta("output", "path"),
    )
    resume_state: Optional[str] = Field(
        None,
        description="Resume from a saved training state (full checkpoint resume)",
        json_schema_extra=_meta("output", "path"),
    )

    # -------------------------------------------------------------------- Sampling
    sample_every: int = Field(
        2, ge=0,
        description="Sample every N epochs (0=disabled)",
        json_schema_extra=_meta("sample"),
    )
    sample_steps: int = Field(
        0, ge=0,
        description="Sample every N steps (0=disabled)",
        json_schema_extra=_meta("sample"),
    )
    sample_infer_steps: int = Field(
        25, ge=1,
        description="Inference steps",
        json_schema_extra=_meta("sample"),
    )
    sample_cfg_scale: float = Field(
        4.0, ge=0.0,
        description="CFG Scale",
        json_schema_extra=_meta("sample"),
    )
    sample_sampler_name: Literal["er_sde", "dpmpp_3m_sde", "euler"] = Field(
        "er_sde",
        description="Sampler. er_sde / dpmpp_3m_sde match Anima's ComfyUI stack"
                    "(dpmpp_3m_sde uses BrownianTree noise, requires torchsde); "
                    "euler is Krea 2's FlowMatchEuler",
        json_schema_extra=_meta(
            "sample", option_show_when=sampling_option_gates("samplers"),
        ),
    )
    sample_scheduler: Literal["simple", "sgm_uniform"] = Field(
        "simple",
        description="Scheduler. Same name and meaning as ComfyUI; the sigma table is determined by the model family"
                    "(Krea 2's simple uses a fixed shift=1.15 convention)",
        json_schema_extra=_meta(
            "sample", option_show_when=sampling_option_gates("schedulers"),
        ),
    )
    sample_width: int = Field(
        0, ge=0,
        description="Sample width (0=follow resolution)",
        json_schema_extra=_meta("sample"),
    )
    sample_height: int = Field(
        0, ge=0,
        description="Sample height (0=follow resolution)",
        json_schema_extra=_meta("sample"),
    )
    sample_seed: int = Field(
        0,
        description="Sample seed (0=random)",
        json_schema_extra=_meta("sample"),
    )
    sample_negative_prompt: str = Field(
        "",
        description="Negative prompt",
        json_schema_extra=_meta("sample", "textarea"),
    )
    sample_prompt: str = Field(
        "newest, safe, 1girl, masterpiece, best quality",
        description="Single-prompt mode: all sample images during training share this prompt (ignored if sample_prompts is set)",
        json_schema_extra=_meta("sample", "textarea"),
    )
    sample_prompts: list[str] = Field(
        default_factory=list,
        description="Rotate through multiple prompts (takes priority over sample_prompt)",
        json_schema_extra=_meta("sample", "string-list"),
    )
    trigger_word: str = Field(
        "",
        description="Trigger word (version-level, written by the Step 4 Tagging page; empty string=disabled). "
                    "During training, bootstrap_phase automatically prepends it to sample_prompt / "
                    "sample_prompts so sample images reflect whether the LoRA is active.",
        json_schema_extra=_meta("sample", hidden=True),
    )

    # ----------------------------------------------------------- Post-training metric evaluation
    eval_validation_enabled: bool = Field(
        False,
        description="After training finishes, generate images from a held-out validation set and compute CLIP-T / CLIP-I / DINO-I metrics. "
                    "The validation set is split off from the training set and never trains — it lives in validation/, alongside train/.",
        json_schema_extra=_meta("eval_validation"),
    )
    eval_validation_split_ratio: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Fraction of the dataset randomly split into the validation set before training starts (0-1, 0=no auto split). "
                    "Recommend around 0.1. Fills proportionally: once the validation set reaches this ratio, no more is split off; on very small datasets, rounding may yield 0.",
        json_schema_extra=_meta("eval_validation", show_when="eval_validation_enabled==true"),
    )
    eval_validation_split_seed: int = Field(
        0, ge=0,
        description="Seed for the random validation split; fixing it makes the split reproducible.",
        json_schema_extra=_meta("eval_validation", show_when="eval_validation_enabled==true"),
    )

    # WandB: as of 0.18 the per-config override block was removed entirely —
    # wandb is an account/workflow-level setting, not something that varies
    # per project; writing api_key/entity/base_url into the yaml would leak
    # them in plaintext through preset sharing/bundle export/task snapshots.
    # Global config lives in Settings (secrets.WandBConfig), injected as
    # WANDB_* environment variables into the training process by the
    # supervisor; secrets are never written to any yaml.
    # Legacy wandb_* keys in old yaml files are dropped as unknown fields by
    # _tolerant_validate (with a dropped_fields notice); the argparse bridge
    # simply skips unknown keys.

    # ---------------------------------------------------------------- Monitoring/progress
    # This group is entirely hidden from Studio users (hidden=True) — Studio
    # runs training via subprocess with stdout redirected to the task log
    # (not a tty), so these "terminal experience" fields are meaningless to
    # web users; the monitor page uses monitor_state.json, which is unrelated
    # to these values. The fields stay in the schema only so bare-CLI users
    # can still override them by hand in yaml; when equal to the default,
    # config_prune won't persist them (hidden trimming), non-default
    # overrides are kept as usual.
    # The old HTTP monitor server fields (no_monitor / monitor_host /
    # monitor_port / no_browser) were removed along with the server; see
    # migrations.RETIRED_MONITOR_KEYS.
    loss_curve_steps: int = Field(
        100, ge=10,
        description="Terminal rich live-curve width (CLI terminal only, doesn't affect the Studio monitor page)",
        json_schema_extra=_meta("monitor", hidden=True),
    )
    # Defaults to True: Studio's subprocess stdout is a pipe, not a tty, and
    # rich still prints screen-clearing progress lines on a non-tty, making
    # the task log huge and hard to read; the plain log_every-throttled
    # branch is cleaner. Bare-CLI users who want the rich progress bar can
    # override with `no_progress: false` explicitly in yaml.
    no_progress: bool = Field(
        True,
        description="Disable the terminal rich progress bar and curve (CLI / log-file scenarios)",
        json_schema_extra=_meta("monitor", hidden=True),
    )
    log_every: int = Field(
        10, ge=1,
        description="Terminal log output interval (only takes effect when the rich progress bar is disabled)",
        json_schema_extra=_meta("monitor", hidden=True),
    )
