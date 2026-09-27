#!/usr/bin/env python3
"""Prior generation -- the base model generates the counterpart image for each
training image's tags, forming a regularization set.

The design comes from DreamBooth prior preservation: the training loss sees
both "what the LoRA has learned to produce" and "what the base model produces
on its own", so the LoRA only learns the difference and doesn't pollute the base concept.

**No LoRA is applied** -- adding one would overwrite the very prior we're trying to preserve.

Usage:
    python runtime/anima_reg_ai.py --config reg_ai_config.json [--monitor-state-file state.json]

Logic:
  1. Scan every image + caption in the train directory
  2. For each image, first copy the training image's same-named tag file to the reg output image's same-named tag file
  3. Read tags from the reg-side tag file, drop excluded ones, join into a prompt
     following Anima's space-separated tag convention, and rewrite the reg-side
     tag file to hold the actual prompt (JSON keeps its standard JSON shape)
  4. Output to reg/{matching subfolder}/{stem}_ai_{seed}.png (mirrors the train subdirectory structure)
  5. reg/meta.json writes generation_method="ai_base", api_source=""
     (shares the reg_builder.RegMeta schema with booru pulls, so it never gets overwritten by a name clash)

incremental=True: skips images in the reg subfolder that already start with train_stem (for resuming after a restart).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# anima_train + train_monitor are both in the same runtime/ directory, so _THIS_DIR is enough.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _REPO_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import anima_train as _T  # noqa: E402

# Reuses reg_builder.RegMeta (PR-9 commit 2 added the generation_method field) +
# clear_reg_dir (the same implementation used by the booru full-mode build entry
# point, keeping behavior/semantics tied to the booru reg path).
from studio.services.reg.builder import (  # noqa: E402
    RegMeta,
    clear_reg_dir,
    read_meta,
    write_meta,
)
from studio.services.tagging.caption_format import (  # noqa: E402
    caption_json_to_tags,
    caption_json_to_text,
    normalize_caption_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("anima_reg_ai")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anima prior generation (base model produces the reg set)")
    p.add_argument("--config", required=True)
    p.add_argument("--monitor-state-file", default="")
    return p.parse_args()


CAPTION_SUFFIXES = (".json", ".txt", ".caption")


def _tag_key(tag: str) -> str:
    """Canonical key for matching/excluding tags.

    Anima prompts should use spaces, not underscores.  We still treat spaces and
    underscores as equivalent for exclude matching so older UI/booru-style
    excluded tags continue to work.
    """
    return " ".join(str(tag or "").strip().lower().replace("_", " ").split())


def _dedupe_tags(tags: list[str]) -> list[str]:
    """Deduplicate while keeping the original tag text and order."""
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        text = str(tag or "").strip()
        key = _tag_key(text)
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _prompt_tag(tag: str) -> str:
    """Normalize one tag for Anima text encoders: lowercase, spaces, no underscores."""
    return _tag_key(tag)


def _read_json_tags(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []

    tags: list[str] = []
    meta = data.get("meta")
    if isinstance(meta, dict):
        trigger = meta.get("trigger")
        if isinstance(trigger, str) and trigger.strip():
            tags.append(trigger.strip())
    tags.extend(caption_json_to_tags(data))
    return _dedupe_tags(tags)


def _read_text_tags(caption_path: Path) -> list[str]:
    raw = caption_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return []
    if "," in raw:
        return [t.strip() for t in raw.split(",") if t.strip()]
    return [t.strip() for t in raw.split() if t.strip()]


def _caption_candidates_for_image(img_path: Path) -> list[Path]:
    return [
        p for suffix in CAPTION_SUFFIXES
        if (p := img_path.with_suffix(suffix)).exists()
    ]


def _read_tags_from_caption(caption_path: Path) -> list[str]:
    if caption_path.suffix == ".json":
        return _read_json_tags(caption_path)
    return _read_text_tags(caption_path)


def _caption_path_for_image(img_path: Path) -> Path | None:
    """Return the first readable sidecar caption path, preferring JSON."""
    for p in _caption_candidates_for_image(img_path):
        try:
            _read_tags_from_caption(p)
            return p
        except Exception as e:
            logger.warning("caption read failed %s: %s", p, e)
    return None


def _read_tags(img_path: Path) -> list[str]:
    """Read the caption next to the image, returning the raw tag list (not normalized).

    JSON caption is preferred, TXT/CAPTION as a fallback; matches the same JSON
    semantics used by the training dataset / tag editor, so prior generation
    doesn't fail to get a prompt when JSON tagging was chosen at Step 4.
    """
    caption_path = _caption_path_for_image(img_path)
    if caption_path is None:
        return []
    return _read_tags_from_caption(caption_path)


def _copy_caption_for_reg(train_img_path: Path, out_img_path: Path) -> Path | None:
    """Copy train sidecar caption to the generated reg image sidecar path."""
    src = _caption_path_for_image(train_img_path)
    if src is None:
        return None
    suffix = ".txt" if src.suffix == ".caption" else src.suffix
    dst = out_img_path.with_suffix(suffix)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def _build_prompt_from_caption(caption_path: Path, excluded_tags: set[str]) -> str:
    """Build an Anima prior prompt from a reg-side caption file."""
    return ", ".join(_prompt_tags_from_caption(caption_path, excluded_tags))


def _prompt_tags_from_caption(caption_path: Path, excluded_tags: set[str]) -> list[str]:
    """Return normalized prompt tags from a caption file."""
    prompt_tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in _read_tags_from_caption(caption_path):
        tag = _prompt_tag(raw_tag)
        if not tag or tag in excluded_tags or tag in seen:
            continue
        seen.add(tag)
        prompt_tags.append(tag)
    return prompt_tags


def _filter_normalized_caption(
    data: dict, excluded_tags: set[str]
) -> dict:
    """Drop excluded tags + meta.trigger from a normalized standard-shape caption.

    The reg side never carries a trigger: the base prior doesn't know the LoRA
    handle, and this also prevents the reg sidecar from getting a trigger
    injected again when the training side's caption_utils.load_and_build_caption reads it.

    Scalar fields (count/character/series/artist) are split on commas, filtered
    tag-by-tag against excluded, and joined back, so a single excluded item can
    still match inside a combined value like "1girl, 1boy".
    """
    src_tags = data.get("tags") or {}

    def _keep_list(values: list[str]) -> list[str]:
        return [t for t in values if _tag_key(t) not in excluded_tags]

    def _keep_scalar(value: str) -> str:
        kept = [
            t.strip() for t in str(value or "").split(",")
            if t.strip() and _tag_key(t) not in excluded_tags
        ]
        return ", ".join(kept)

    meta = {
        k: v for k, v in (data.get("meta") or {}).items() if k != "trigger"
    }
    return {
        "meta": meta,
        "tags": {
            "quality": _keep_list(src_tags.get("quality") or []),
            "count": _keep_scalar(src_tags.get("count") or ""),
            "character": _keep_scalar(src_tags.get("character") or ""),
            "series": _keep_scalar(src_tags.get("series") or ""),
            "artist": _keep_scalar(src_tags.get("artist") or ""),
            "appearance": _keep_list(src_tags.get("appearance") or []),
            "tags": _keep_list(src_tags.get("tags") or []),
            "environment": _keep_list(src_tags.get("environment") or []),
            "nl": str(src_tags.get("nl") or "").strip(),
        },
    }


def _rewrite_json_caption_for_prompt(caption_path: Path, excluded_tags: set[str]) -> str:
    """Normalize -> filter excluded + drop trigger -> write back standard shape.

    The reg sidecar is a derived artifact, so it's always written in the
    standard shape produced by caption_format.normalize_caption_json; the
    training side's caption_utils.load_and_build_caption reads reg JSON through
    the same normalize entry point, so there's no need to preserve the
    user-side's original documented_full / simplified shape -- which also
    removes the mirrored maintenance cost of 4 separate shape filters.
    """
    raw = json.loads(caption_path.read_text(encoding="utf-8"))
    normalized = normalize_caption_json(raw if isinstance(raw, dict) else {})
    filtered = _filter_normalized_caption(normalized, excluded_tags)
    prompt = caption_json_to_text(filtered)
    caption_path.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return prompt


def _peek_prompt_for_image(train_img_path: Path, excluded_tags: set[str]) -> str | None:
    """Pure read-only derivation of the prompt (used for batch pre-encoding) -- copies or rewrites nothing.

    Byte-for-byte identical to what the loop's _copy_caption_for_reg +
    _rewrite_caption_for_prompt would produce (same source file, same rule),
    guaranteeing an exact hit against the online LRU during generation. Returns
    None on a source lookup / read failure (that entry falls back to lazy in-loop encoding).
    """
    src = _caption_path_for_image(train_img_path)
    if src is None:
        return None
    try:
        if src.suffix == ".json":
            raw = json.loads(src.read_text(encoding="utf-8"))
            normalized = normalize_caption_json(raw if isinstance(raw, dict) else {})
            return caption_json_to_text(
                _filter_normalized_caption(normalized, excluded_tags)
            )
        return _build_prompt_from_caption(src, excluded_tags)
    except Exception:
        return None


def _rewrite_caption_for_prompt(caption_path: Path, excluded_tags: set[str]) -> str:
    """Persist the reg sidecar caption that corresponds to the generated image."""
    if caption_path.suffix == ".json":
        return _rewrite_json_caption_for_prompt(caption_path, excluded_tags)

    prompt = _build_prompt_from_caption(caption_path, excluded_tags)
    caption_path.write_text(prompt, encoding="utf-8")
    return prompt


def _normalize(tag: str) -> str:
    return _tag_key(tag)


def _reg_subfolder(train_subfolder: str, repeat: int) -> str:
    """train subfolder name -> reg subfolder name, rewriting the Kohya prefix by `repeat`.

    The reg set's repeat is independent of train -- it no longer mirrors train's
    `N_` prefix verbatim (that would make reg follow train's repeat, so a
    2_data training folder would see reg twice per epoch). Instead, this always
    uses the repeat the user picked in the UI while keeping the label:
      - "" (train root)          -> "{repeat}_data"
      - "2_data" / "data"        -> "{repeat}_data"
      - "nested/5_concept"       -> "nested/{repeat}_concept" (only the last segment's prefix is rewritten)
    """
    if not train_subfolder:
        return f"{repeat}_data"
    parent, _, leaf = train_subfolder.rpartition("/")
    m = re.match(r"^\d+_(.*)$", leaf)
    label = m.group(1) if m else leaf
    new_leaf = f"{repeat}_{label}"
    return f"{parent}/{new_leaf}" if parent else new_leaf


def _scan_train(train_dir: Path) -> list[dict]:
    """Scan the train directory, returning an info list for each image.

    Element: {"subfolder": str, "stem": str, "img": Path, "tags": list[str]}
    subfolder="" means the train root directory.
    """
    entries: list[dict] = []
    train_dir = train_dir.resolve()

    def _scan(folder: Path, sub: str) -> None:
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                entries.append({
                    "subfolder": sub,
                    "stem": f.stem,
                    "img": f,
                    "tags": _read_tags(f),
                })
            elif f.is_dir():
                child = f.name if not sub else f"{sub}/{f.name}"
                _scan(f, child)

    _scan(train_dir, "")
    return entries


def _already_has_reg(reg_sub: Path, train_stem: str) -> bool:
    """incremental: skip if the reg subdirectory already has an image starting with train_stem."""
    if not reg_sub.exists():
        return False
    for f in reg_sub.iterdir():
        if (
            f.is_file()
            and f.stem.startswith(train_stem)
            and f.suffix.lower() in IMAGE_EXTS
        ):
            return True
    return False


def _write_meta_final(
    reg_dir: Path,
    entries: list[dict],
    excluded_tags: set,
    incremental: bool,
    actual_count: int,
) -> None:
    """Write reg/meta.json using reg_builder.RegMeta (generation_method='ai_base').

    Shares its schema with booru pulls; api_source is left empty (prior
    generation has no booru source). incremental_runs is incremented by 1 on
    top of any existing meta (matches PP5.1 booru behavior).
    """
    prior = read_meta(reg_dir)
    runs = (prior.incremental_runs + 1) if (incremental and prior) else 0

    tag_dist: Counter = Counter()
    for e in entries:
        tag_dist.update(e["tags"])

    meta = RegMeta(
        generated_at=time.time(),
        based_on_version="",
        api_source="",  # prior generation has no booru source
        target_count=len(entries),
        actual_count=actual_count + (prior.actual_count if (incremental and prior) else 0),
        source_tags=[],
        excluded_tags=sorted(excluded_tags),
        blacklist_tags=[],
        failed_tags=[],
        train_tag_distribution=dict(tag_dist.most_common(50)),
        auto_tagged=False,
        incremental_runs=runs,
        generation_method="ai_base",
    )
    write_meta(reg_dir, meta)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"config file does not exist: {cfg_path}")
        sys.exit(1)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    train_dir = Path(cfg["train_dir"])
    reg_dir = Path(cfg["reg_dir"])
    excluded_tags: set = {_normalize(t) for t in cfg.get("excluded_tags", [])}
    negative_prompt: str = cfg.get("negative_prompt", "")
    width: int = int(cfg.get("width", 1024))
    height: int = int(cfg.get("height", 1024))
    steps: int = int(cfg.get("steps", 25))
    cfg_scale: float = float(cfg.get("cfg_scale", 4.0))
    sampler_name: str = cfg.get("sampler_name", "er_sde")
    scheduler: str = cfg.get("scheduler", "simple")
    base_seed: int = int(cfg.get("seed", 0))
    incremental: bool = bool(cfg.get("incremental", False))
    repeat: int = max(1, int(cfg.get("repeat", 1)))
    mixed_precision: str = cfg.get("mixed_precision", "bf16")
    backend: str = cfg.get("attention_backend", "flash_attn")
    use_flash = (backend == "flash_attn")
    use_xformers = (backend == "xformers")

    transformer_path: str = cfg["transformer_path"]
    vae_path: str = cfg["vae_path"]
    text_encoder_path: str = cfg["text_encoder_path"]
    t5_tokenizer_path: str = cfg.get("t5_tokenizer_path", "")

    # monitor (falls back to reg/.monitor_state.json, matching anima_generate's behavior)
    state_file = args.monitor_state_file or str(reg_dir / "monitor_state.json")
    _update_monitor = None
    try:
        from train_monitor import set_state_file, update_monitor
        set_state_file(state_file)
        update_monitor(config={"type": "reg_ai"})
        _update_monitor = update_monitor
    except Exception as e:
        logger.warning(f"monitor init failed: {e}")

    if not train_dir.exists():
        logger.error(f"train directory does not exist: {train_dir}")
        sys.exit(1)

    entries = _scan_train(train_dir)
    if not entries:
        logger.error("train directory has no images")
        sys.exit(1)

    logger.info(f"train has {len(entries)} images total")

    if incremental:
        to_generate = [
            e for e in entries
            if not _already_has_reg(
                reg_dir / _reg_subfolder(e["subfolder"], repeat),
                e["stem"],
            )
        ]
        logger.info(f"incremental mode: {len(to_generate)}/{len(entries)} images need generating")
    else:
        to_generate = entries

    if not to_generate:
        logger.info("every image already has a matching regularization image, nothing to generate")
        _write_meta_final(reg_dir, entries, excluded_tags, incremental, 0)
        return

    # Load the base model (no LoRA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float32

    repo_root = _T.find_diffusion_pipe_root()
    bases = [Path.cwd(), _THIS_DIR, repo_root]
    transformer_path = _T.resolve_path_best_effort(transformer_path, bases)
    vae_path = _T.resolve_path_best_effort(vae_path, bases)
    text_encoder_path = _T.resolve_path_best_effort(text_encoder_path, bases)
    if t5_tokenizer_path:
        t5_tokenizer_path = _T.resolve_path_best_effort(t5_tokenizer_path, bases)

    family = _T.resolve_family(cfg)  # D8'
    from training.sysmem import (
        check_load_budget, gpu_free_bytes_global, guard_enabled_from_env,
    )

    # AI prior generation is a heavy-load task in the same exclusive tier as
    # training, so it shares the training side's watermark protection switch
    # (Settings -> Training -> Training Params, injected by the supervisor via env, default on).
    check_load_budget(
        guard_enabled_from_env(),
        weight_paths=[transformer_path, vae_path, text_encoder_path],
        stage="regularization generation model load",
        settings_hint="Settings -> Training -> Training Params",
    )
    logger.info("Loading VAE...")
    vae = family.load_vae(vae_path, device, dtype,
                          tiling=str(cfg.get("vae_tiling", "auto")))

    logger.info("Loading text encoder...")
    # The family's opaque text stack isn't unpacked; caching is off for ad-hoc prompts (cached_varlen families keep the TE resident)
    text_stack = family.load_text(
        text_encoder_path, device, dtype,
        t5_tokenizer_path=t5_tokenizer_path or None,
        purpose="generate",
        cache_enabled=False,
    )

    # TE-first + batched pre-encoding (krea2): reg prompts are a different
    # caption per image (each used only once), and the LRU capacity (64) can't
    # hold them all -- so this works batch by batch: at the start of each batch,
    # the TE loads onto the GPU, encodes 64 entries -> fully releases -> every
    # image generated within the batch then hits the LRU. The first batch is
    # encoded before the DiT loads (zero co-residency); for batch 2+ the TE
    # briefly co-resides with the DiT while loading, and pre-encoding is skipped
    # (falling back to the lazy per-image path) if VRAM is insufficient. The
    # anima text stack has no such API and is skipped entirely.
    _PRECACHE_BATCH = 64

    def _precache_batch(batch: list[dict], first: bool) -> None:
        precache = getattr(text_stack, "precache_online_prompts", None)
        if not callable(precache):
            return
        if not first:
            free = gpu_free_bytes_global()
            # Upper bound on the TE's VRAM footprint when loaded (bf16 ~11GB;
            # fp8 smaller) -- skip when there isn't enough, falling back to
            # sample_image's internal per-image path.
            if free is not None and free < 12 * 1024**3:
                logger.info("insufficient VRAM headroom, skipping pre-encoding for this batch (falling back to per-image encoding)")
                return
        prompts = [
            p for p in (
                _peek_prompt_for_image(entry["img"], excluded_tags)
                for entry in batch
            ) if p
        ]
        if not prompts:
            return
        # negative is a single fixed entry but gets evicted once a batch fills the LRU -- include it every batch to keep it hit
        prompts.append(str(negative_prompt or ""))
        try:
            encoded = precache(prompts)
            release = getattr(text_stack, "release_model", None)
            if callable(release):
                release()
            if encoded:
                logger.info("pre-encoded %d captions for this batch; TE released", encoded)
        except Exception:
            logger.exception("batch pre-encoding failed; falling back to lazy per-image encoding")

    if to_generate:
        _precache_batch(to_generate[:_PRECACHE_BATCH], first=True)

    logger.info("Loading Transformer...")
    model = family.load_dit(
        transformer_path, device, dtype,
        attention_backend=("flash_attn" if use_flash else "none"), repo_root=repo_root,
        purpose="generate",
    )
    if use_xformers:
        _T.enable_xformers(model)

    model.eval()

    if not incremental:
        logger.info("full mode: clearing old reg content")
        clear_reg_dir(reg_dir)

    # Generation loop
    total = len(to_generate)
    actual_count = 0

    for idx, entry in enumerate(to_generate):
        if idx and idx % _PRECACHE_BATCH == 0:
            _precache_batch(
                to_generate[idx:idx + _PRECACHE_BATCH], first=False,
            )
        seed = (base_seed + idx) if base_seed != 0 else random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)
        random.seed(seed)

        reg_sub = reg_dir / _reg_subfolder(entry["subfolder"], repeat)
        reg_sub.mkdir(parents=True, exist_ok=True)

        out_name = f"{entry['stem']}_ai_{seed}.png"
        out_path = reg_sub / out_name
        caption_path = _copy_caption_for_reg(entry["img"], out_path)
        if caption_path is None:
            logger.warning(f"[{idx + 1}/{total}] {entry['img'].name} has no tag file, skipping")
            continue

        try:
            prompt = _rewrite_caption_for_prompt(caption_path, excluded_tags)
        except Exception as e:
            logger.warning(f"[{idx + 1}/{total}] {caption_path.name} read failed, skipping: {e}")
            caption_path.unlink(missing_ok=True)
            continue

        if not prompt:
            logger.warning(f"[{idx + 1}/{total}] {entry['img'].name} has no tags left after filtering, skipping")
            caption_path.unlink(missing_ok=True)
            continue

        logger.info(f"[{idx + 1}/{total}] {entry['img'].name} -> {out_name}")
        logger.info(f"  caption: {caption_path.name}")
        logger.info(f"  prompt: {prompt[:80]}")

        try:
            img = family.sample_image(
                model, vae, text_stack,
                prompt,
                height=height,
                width=width,
                steps=steps,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                sampler_name=sampler_name,
                scheduler=scheduler,
                device=device,
                dtype=dtype,
                vram_policy="auto",
            )
            img.save(out_path)
            actual_count += 1
            logger.info(f"  saved: {out_path}")
            if _update_monitor:
                _update_monitor(sample_path=str(out_path), step=idx + 1)
        except Exception as e:
            logger.error(f"  generation failed: {e}")
            if not out_path.exists():
                caption_path.unlink(missing_ok=True)

    _write_meta_final(reg_dir, entries, excluded_tags, incremental, actual_count)
    logger.info(f"done: {actual_count}/{total} images")


if __name__ == "__main__":
    main()
