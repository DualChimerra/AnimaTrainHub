"""dataset_phase: build datasets + dataloader + VAE roundtrip self-check.

Extracted from main() L257-342 (ADR 0003 PR-B).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.context import TrainingContext
from training.dataset import (
    BucketBatchSampler,
    BucketManager,
    CachedLatentDataset,
    ImageDataset,
    MergedDataset,
    NavitPackBatchSampler,
    collate_fn,
    collate_fn_cached,
    collate_fn_navit_pack,
)


logger = logging.getLogger(__name__)


def _as_resolutions(value) -> list[int]:
    """Normalize the config's resolution into a list[int].

    A schema scalar, a schema list, or a hand-written YAML scalar can all show up
    (merge_yaml_into_namespace is a bare setattr and doesn't go through the
    pydantic validator), so we fall back here to always producing a non-empty list.
    """
    if isinstance(value, (list, tuple)):
        out = [int(v) for v in value]
        return out or [1024]
    return [int(value)]


def _rope_max_side_tokens(ctx) -> int:
    """The model's RoPE addressable patch-token ceiling per side (= max_img_h // patch_spatial).

    NaViT native-resolution mode uses this to cap each side at the data layer,
    avoiding a forward-pass fail-fast in ``_packed_rope_from_grid`` (which would
    waste a whole VAE cache pass for nothing). If unavailable (e.g. a unit test
    with no model) -> 0 (no early cap; the forward-pass fail-fast still backstops it).
    """
    pe = getattr(getattr(ctx, "model", None), "pos_embedder", None)
    try:
        return int(min(int(pe.max_h), int(pe.max_w)))
    except Exception:
        return 0


def _native_dataset_kwargs(args, ctx) -> dict:
    """Native-resolution kwargs for ImageDataset when navit_native_resolution is on; off -> empty dict (no behavior change)."""
    navit_packing = bool(getattr(args, "navit_packing", False))
    if not (navit_packing and bool(getattr(args, "navit_native_resolution", False))):
        return {}
    kwargs = dict(
        native_resolution=True,
        native_token_budget=int(getattr(args, "navit_token_budget", 0) or 0),
        native_over_budget=str(getattr(args, "navit_native_over_budget", "downscale") or "downscale"),
        native_max_side_tokens=_rope_max_side_tokens(ctx),
    )
    logger.info(
        "[navit-native] native resolution enabled: per-image floor-aligned to 16px, zero padding, "
        "bypasses ARB bucket quantization; over-budget strategy=%s, token budget=%d, "
        "RoPE per-side cap=%s tokens.",
        kwargs["native_over_budget"], kwargs["native_token_budget"],
        kwargs["native_max_side_tokens"] or "unlimited",
    )
    return kwargs


def run(ctx: TrainingContext) -> None:
    """
    - Main dataset / regularization dataset + per-folder repeat
    - cache_latents wraps CachedLatentDataset
    - MergedDataset chains the main set + regularization set
    - Windows num_workers > 0 falls back to 0 (multiprocess spawn crashes easily)
    - BucketBatchSampler / DataLoader
    - VAE encode-decode roundtrip self-check (vae_roundtrip.png)
    """
    args = ctx.args

    # Multi-resolution: args.resolution can be a scalar or a list (hand-written YAML can
    # also be a scalar), normalized here into a list; the base tier takes the first item,
    # BucketManagers for other tiers are built by ImageDataset as needed.
    res_list = _as_resolutions(args.resolution)
    ar_limit = float(getattr(args, "aspect_ratio_limit", 2.0))
    base_reso = res_list[0]

    # NaViT native-resolution kwargs (navit_native_resolution); off -> empty dict, no behavior change.
    native_kwargs = _native_dataset_kwargs(args, ctx)

    # masked loss (B2): when enabled, the data layer loads a {stem}.mask sidecar next to
    # each image. The first version of the NaViT packing path doesn't support this (a
    # per-image packed loss has no batch grid, see SS5 decision) -- warn and disable it,
    # to avoid writing mask keys into the npz for nothing.
    load_masks = bool(getattr(args, "masked_loss", False))
    if load_masks and bool(getattr(args, "navit_packing", False)):
        logger.warning(
            "[masked-loss] NaViT packing path doesn't support masked loss yet: masks ignored for this run"
        )
        load_masks = False
        args.masked_loss = False

    # Dataset
    # This fork: bucket_min_reso / bucket_max_reso / bucket_step can be set explicitly in
    # the training config (None = auto-derived from base_reso, matching upstream). When
    # explicitly 512/2048/64, this matches the old fork's ARB bucket set byte-for-byte
    # (the crop page's trainBuckets.ts prediction depends on this stability).
    _bucket_kwargs = {}
    for _cfg_key, _kw in (
        ("bucket_min_reso", "min_reso"),
        ("bucket_max_reso", "max_reso"),
        ("bucket_step", "step"),
    ):
        _v = getattr(args, _cfg_key, None)
        if _v is not None:
            _bucket_kwargs[_kw] = int(_v)
    ctx.bucket_mgr = BucketManager(
        base_reso, aspect_ratio_limit=ar_limit, **_bucket_kwargs
    )
    ctx.base_dataset = ImageDataset(
        args.data_dir, base_reso, ctx.bucket_mgr,
        shuffle_caption=args.shuffle_caption,
        keep_tokens=args.keep_tokens,
        flip_augment=args.flip_augment,
        tag_dropout=args.tag_dropout,
        prefer_json=args.prefer_json,
        resolutions=res_list,
        aspect_ratio_limit=ar_limit,
        load_masks=load_masks,
        **native_kwargs,
    )
    ctx.dataset = ctx.base_dataset

    if load_masks:
        # Count how many images have a mask (decision 5: switch on but zero masks just logs, doesn't error)
        n_masked = sum(
            1 for s in ctx.base_dataset.samples
            if ctx.base_dataset._mask_path_for(s["image"]).is_file()
        )
        if n_masked > 0:
            logger.info(
                "[masked-loss] enabled: %d/%d images have a mask (the rest learn on the full image as usual)",
                n_masked, len(ctx.base_dataset.samples),
            )
        else:
            logger.info(
                "[masked-loss] switch is on but the training set has no mask files at all (no .mask sidecar) "
                "-- this run is effectively as if it were off"
            )

    # Regularization dataset (Kohya-style, anti-overfitting)
    reg_data_dir = getattr(args, "reg_data_dir", "") or ""
    ctx.reg_dataset = None
    if reg_data_dir:
        if not Path(reg_data_dir).exists():
            logger.warning(f"regularization dataset path doesn't exist, skipped: {reg_data_dir}")
        elif len(ctx.base_dataset) == 0:
            logger.warning("main dataset is empty, regularization set skipped")
        else:
            reg_caption = (getattr(args, "reg_caption", "") or "").strip()
            reg_base = ImageDataset(
                reg_data_dir, base_reso, ctx.bucket_mgr,
                shuffle_caption=args.shuffle_caption,
                keep_tokens=args.keep_tokens,
                flip_augment=args.flip_augment,
                tag_dropout=0.0,  # regularization sets usually don't use dropout
                prefer_json=args.prefer_json,
                caption_override=reg_caption if reg_caption else None,
                resolutions=res_list,
                aspect_ratio_limit=ar_limit,
                **native_kwargs,
            )
            if len(reg_base) == 0:
                # Don't wire in an empty regularization set: wrapping it in CachedLatentDataset
                # would print a confusing "all 0 images cached" log, and it's a pure no-op in
                # MergedDataset anyway.
                logger.info(f"regularization dataset is empty (no valid images), skipped: {reg_data_dir}")
            else:
                ctx.reg_dataset = reg_base
                reg_weight = float(getattr(args, "reg_weight", 1.0) or 1.0)
                cap_preview = f", caption=\"{reg_caption[:50]}{'...' if len(reg_caption) > 50 else ''}\"" if reg_caption else ""
                weight_info = f", weight={reg_weight}" if reg_weight != 1.0 else ""
                logger.info(f"regularization dataset: {reg_data_dir} ({len(reg_base)} samples, per-folder repeat{weight_info}){cap_preview}")

    # Cache VAE latents (before repeat)
    ctx.use_cached = getattr(args, "cache_latents", False)
    if ctx.use_cached:
        # 0 = follow the training batch size (matches kohya GUI's VAE batch size semantics)
        cache_batch_size = int(getattr(args, "vae_cache_batch_size", 0) or 0)
        if cache_batch_size <= 0:
            cache_batch_size = int(getattr(args, "batch_size", 1) or 1)
        ctx.dataset = CachedLatentDataset(
            ctx.dataset, ctx.vae, ctx.device, ctx.vae_dtype,
            cache_batch_size=cache_batch_size,
            encode_tiled=getattr(args, "cache_encode_tiled", False),
            encode_tile_px=getattr(args, "cache_encode_tile_px", 1024),
            encode_tile_overlap=getattr(args, "cache_encode_tile_overlap", 128),
            encode_max_pixels=getattr(args, "cache_encode_max_pixels", 0),
            label="training set",
        )
    if ctx.reg_dataset is not None and ctx.use_cached:
        ctx.reg_dataset = CachedLatentDataset(
            ctx.reg_dataset, ctx.vae, ctx.device, ctx.vae_dtype,
            cache_batch_size=cache_batch_size,
            encode_tiled=getattr(args, "cache_encode_tiled", False),
            encode_tile_px=getattr(args, "cache_encode_tile_px", 1024),
            encode_tile_overlap=getattr(args, "cache_encode_tile_overlap", 128),
            encode_max_pixels=getattr(args, "cache_encode_max_pixels", 0),
            label="regularization set",
        )

    # repeat: both the main and regularization datasets use Kohya-style per-folder-name repeat
    # (e.g. 5_concept), so no global repeat is needed
    if ctx.reg_dataset is not None:
        reg_weight = float(getattr(args, "reg_weight", 1.0) or 1.0)
        ctx.dataset = MergedDataset(ctx.dataset, ctx.reg_dataset, reg_weight=reg_weight)

    if args.num_workers > 0 and os.name == "nt":
        logger.warning("num_workers > 0 crashes easily on Windows: forced to 0 (avoids multiprocess spawn issues)")
        args.num_workers = 0

    if getattr(args, "navit_packing", False):
        # NaViT / Patch-n-Pack block-diagonal packing: fits multiple differently-sized images
        # into one training sequence within a token budget (zero padding), replacing ARB
        # fixed-bucket batching. Requires cache_latents.
        batch_sampler = NavitPackBatchSampler(
            ctx.dataset,
            token_budget=int(getattr(args, "navit_token_budget", 16384) or 16384),
            max_images_per_pack=int(getattr(args, "navit_max_images_per_pack", 0) or 0),
            shuffle=True,
            seed=getattr(args, "seed", 42),
            drop_last=getattr(args, "navit_drop_last", False),
            strategy=getattr(args, "navit_pack_strategy", "next_fit"),
            # Not `or 256`: 0 is a valid value (global FFD, fixed packs per epoch), which `or` would swallow as falsy.
            ffd_window=int(getattr(args, "navit_pack_ffd_window", 256)),
        )
        ctx.dataloader = DataLoader(
            ctx.dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn_navit_pack,
            num_workers=args.num_workers,
        )
    elif ctx.use_cached:
        # drop_last=False: a bucket's leftover tail shorter than batch_size becomes a short
        # batch instead of dropped images. Matches kohya sd-scripts / ostris ai-toolkit;
        # diffusion uses LayerNorm/GroupNorm, which is insensitive to dynamic batch sizes,
        # and loop.py also reads bs dynamically from latents.shape[0].
        batch_sampler = BucketBatchSampler(
            ctx.dataset, batch_size=args.batch_size,
            drop_last=False, shuffle=True,
            seed=getattr(args, "seed", 42),
        )
        ctx.dataloader = DataLoader(
            ctx.dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn_cached,
            num_workers=args.num_workers,
        )
    else:
        # The non-cached path must also batch by bucket: collate_fn uses torch.stack to
        # combine pixel_values, and mixing different bucket sizes in one batch (different
        # aspect ratios under ARB -> different H x W) raises RuntimeError. BucketBatchSampler
        # relies on ImageDataset.bucket_for_index to group same-size samples into the same
        # batch (the cached path already did this; the non-cached path used to miss it ->
        # guaranteed crash at bs>1). drop_last=False matches the cached path.
        batch_sampler = BucketBatchSampler(
            ctx.dataset, batch_size=args.batch_size,
            drop_last=False, shuffle=True,
            seed=getattr(args, "seed", 42),
        )
        ctx.dataloader = DataLoader(
            ctx.dataset, batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )

    # Pre-training self-check: VAE encode->decode roundtrip (quickly rules out VAE/scale/shape issues)
    try:
        if len(ctx.base_dataset) > 0:
            from PIL import Image
            item0 = ctx.base_dataset[0]
            pixels0 = item0["pixel_values"].unsqueeze(0).to(ctx.device, dtype=ctx.dtype)  # [1,3,H,W]
            with torch.no_grad():
                # Both encode/decode go through VAEWrapper (with auto/on tiling), avoiding a
                # whole-image op on large images triggering a system-memory fallback hang.
                z0 = ctx.vae.encode(pixels0.unsqueeze(2))                        # [1,16,1,h,w]
                recon0 = ctx.vae.decode(z0).squeeze(2)                           # [1,3,H,W]
                recon0 = (recon0.clamp(-1, 1) + 1) / 2
            arr0 = (recon0[0].permute(1, 2, 0).detach().cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(arr0).save(ctx.sample_dir / "vae_roundtrip.png")
            logger.info("VAE roundtrip self-check saved: samples/vae_roundtrip.png")
    except Exception as e:
        logger.warning(f"VAE roundtrip self-check failed (if samples are still noise, fix this first): {e}")
