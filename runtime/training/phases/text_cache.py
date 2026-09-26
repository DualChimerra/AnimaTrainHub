"""text_cache_phase: build the per-image caption sidecar plan and hand it to
ModelFamily for encoding.

Fixed position: after dataset, before optimizer. dataset still only produces
``captions``; cache hits and encoded tensor shapes are entirely up to the
family. Anima's "online" strategy is a zero-cost no-op here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from training.context import TrainingContext
from training.text_cache import TextCacheEntry


logger = logging.getLogger(__name__)


def _image_dataset(dataset):
    """Find the underlying dataset with samples + caption resolver inside a latent/repeat wrapper."""

    seen = set()
    current = dataset
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "samples") and hasattr(current, "caption_for_sample"):
            return current
        next_dataset = getattr(current, "base_image_dataset", None)
        if next_dataset is None:
            next_dataset = getattr(current, "base_dataset", None)
        if next_dataset is None:
            next_dataset = getattr(current, "dataset", None)
        current = next_dataset
    return None


def _collect_entries(ctx: TrainingContext) -> list[TextCacheEntry]:
    """Dedup main + regularization sets by image; resolution fan-out/repeat never re-encodes text."""

    by_image: dict[str, TextCacheEntry] = {}
    for dataset in (ctx.base_dataset, ctx.reg_dataset):
        base = _image_dataset(dataset)
        if base is None:
            continue
        for sample in base.samples:
            image = Path(sample["image"])
            caption = base.caption_for_sample(sample)
            key = str(image)
            existing = by_image.get(key)
            if existing is not None:
                if existing.caption != caption:
                    raise ValueError(
                        f"same image resolved to two different captions, cannot build text cache: {image}"
                    )
                continue
            by_image[key] = TextCacheEntry.for_image(image, caption)
    return list(by_image.values())


def _extra_prompts(args) -> list[str]:
    prompts = getattr(args, "sample_prompts", None) or []
    if isinstance(prompts, str):
        prompts = [prompts]
    out = [str(p) for p in prompts if p is not None]
    single = getattr(args, "sample_prompt", "") or ""
    if single and single not in out:
        out.append(str(single))
    sampling_enabled = bool(
        int(getattr(args, "sample_steps", 0) or 0)
        or int(getattr(args, "sample_every", 0) or 0)
    )
    if sampling_enabled and not out:
        out.append("1girl, masterpiece")
    # The CFG unconditional branch also needs encoding; an empty string is a valid, meaningful prompt.
    out.append(str(getattr(args, "sample_negative_prompt", "") or ""))
    return list(dict.fromkeys(out))


def run(ctx: TrainingContext) -> None:
    strategy = ctx.family.spec.text.strategy
    if strategy == "online":
        ctx.family.prepare_text_cache([], [])
        return
    if strategy != "cached_varlen":  # the registry normally catches this first; kept here as an actionable error
        raise ValueError(f"unknown text strategy: {strategy!r}")

    if not bool(getattr(ctx.args, "text_encoder_cache", True)):
        logger.info(
            "[text-cache] cache disabled: no sidecar scan/read/write, text encoder stays resident and encodes per batch"
        )
        ctx.family.prepare_text_cache(
            [],
            [],
            text=ctx.text_stack,
            device=ctx.device,
            dtype=ctx.dtype,
        )
        return

    entries = _collect_entries(ctx)
    captions = [entry.caption for entry in entries]
    extras = _extra_prompts(ctx.args)
    # The aggregated prompt cache belongs to the task archive (tasks/<id>/.text-cache/), not the
    # dataset's train/ dir: putting it there would get mistaken for a concept folder during
    # dataset scanning. Plain CLI runs (no task archive) fall back to output_dir, which is also
    # outside the dataset scan range.
    cache_root = ctx.task_archive_dir or ctx.output_dir
    logger.info(
        "[text-cache] pre-caching %d image captions + %d sample/negative prompts (varlen)",
        len(entries), len(extras),
    )
    ctx.family.prepare_text_cache(
        captions,
        extras,
        cache_entries=entries,
        cache_root=cache_root,
        text=ctx.text_stack,
        device=ctx.device,
        dtype=ctx.dtype,
    )
