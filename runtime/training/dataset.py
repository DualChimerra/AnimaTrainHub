"""Dataset & collate: ARB bucketing + ImageDataset + regularization-set merge + cached latent.

NaViT / Patch-n-Pack block-diagonal packing (Phase 2 data layer):
- Tiled VAE encode: delegated to VAEWrapper._tiled_encode (cache_encode_tiled)
- Token-budget packing NavitPackBatchSampler + pack_indices_by_budget / pack_indices_ffd_windowed
- CachedLatentDataset extension: tiled encode, token_count_for_index
- collate_fn_navit_pack -- heterogeneous-latent per-image list collate

Extracted from the original runtime/anima_train.py L1144-1675 + L1939-1962 (ADR 0003 PR-A).

Public API:
- BucketManager / ImageDataset / RepeatDataset / MergedDataset
- BucketBatchSampler / CachedLatentDataset
- collate_fn / collate_fn_cached -- DataLoader collate
- NavitPackBatchSampler / collate_fn_navit_pack -- block-diagonal packing
"""

from __future__ import annotations

import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Relative import: this module is reused by the studio server as
# `runtime.training.dataset` (bucket distribution preview); there sys.path only
# has the repo root, so an absolute `training.*` import would raise
# ModuleNotFoundError.
from .families.anima import ANIMA_SPEC

# -- single source of truth for latent spec (multi-model PR-1) -----------------
# Bridge constant from the single-family era; from PR-2b onward ctx.family.spec
# is passed to each call site instead.
_ANIMA_LATENT = ANIMA_SPEC.latent
#: latent npz cache layout version (bump when the key set / tensor layout
#: changes; a mismatch triggers delete-and-re-encode)
LATENT_CACHE_LAYOUT_VERSION = 1
#: Grandfather value for legacy caches with no fingerprint key: historically
#: only Wan21/Qwen-Image ever produced this VAE's cache (04-synthesis D12),
#: which avoids a full re-encode on upgrade.
_LEGACY_CACHE_FINGERPRINT = "wan21-f8c16"


logger = logging.getLogger(__name__)


@dataclass
class NativeFitImagePlan:
    """Planning result for NaViT's native fixed size (``navit_native_resolution``).

    ``width``/``height`` are the final pixel size fed to the VAE (a multiple of
    ``align``). The pixel path reuses ``ImageDataset``'s existing resize-cover +
    center-crop (zero padding), so the valid region always fills the entire
    latent (which is the precondition for the navit cache path carrying no mask).
    """
    source_width: int
    source_height: int
    width: int
    height: int
    token_w: int
    token_h: int
    token_count: int
    was_downscaled: bool


def plan_native_fit_image(
    width: int,
    height: int,
    *,
    max_tokens: int = 0,
    max_side_tokens: int = 0,
    align: int = _ANIMA_LATENT.align_px,
    over_budget: str = "downscale",
) -> NativeFitImagePlan:
    """Plan a single image's native fixed size (floor-aligned to ``align``, with
    optional over-budget downscale).

    Ported from an early fork of the same family, ``anima-lora-train``
    (the floor branch of ``trainer/data.py::plan_native_fit_image`` plus the
    proportional-scale math of ``plan_multiscale_copy``; same GPL-3.0 lineage,
    see the PR notes). Unlike that earlier upstream version, this function
    **actually implements** downscale (the earlier version raised
    NotImplementedError for any non-fail strategy).

    Alignment unit: ``align = patch_spatial(2) x vae_downsample(8) = 16px``.

    - Normal case (native tokens <= ``max_tokens`` and each side <=
      ``max_side_tokens``): floor each side to a multiple of ``align`` (dropping
      up to align-1 px), no scaling, aspect ratio barely changes.
    - Over budget (token count exceeds ``max_tokens`` or a side exceeds
      ``max_side_tokens``):
        * ``over_budget="downscale"`` (default): scale proportionally to satisfy
          both limits, then floor each axis to ``align``
          (``floor(a)*floor(b) <= a*b`` guarantees staying within budget). The
          caller then does resize-cover + center-crop to trim the <=align-1 px
          overflow from flooring -> exact alignment, zero padding.
        * ``over_budget="fail"``: raise ``ValueError`` directly (requires
          raising the budget / cropping on the dataset side).

    ``max_tokens`` / ``max_side_tokens`` of 0 means that dimension is unbounded.
    """
    W, H = int(width), int(height)
    if W <= 0 or H <= 0:
        raise ValueError(f"image dimensions must be positive, got {W}x{H}")
    align = max(1, int(align))
    max_tokens = max(0, int(max_tokens or 0))
    max_side = max(0, int(max_side_tokens or 0))

    # Native floor grid (in patch-token units)
    gw, gh = W // align, H // align
    if gw <= 0 or gh <= 0:
        raise ValueError(
            f"image {W}x{H} smaller than one align unit ({align}px); cannot form a token"
        )

    over_side = bool(max_side and (gw > max_side or gh > max_side))
    over_budget_tokens = bool(max_tokens and gw * gh > max_tokens)

    if not over_side and not over_budget_tokens:
        return NativeFitImagePlan(
            source_width=W, source_height=H,
            width=gw * align, height=gh * align,
            token_w=gw, token_h=gh, token_count=gw * gh,
            was_downscaled=False,
        )

    strategy = (over_budget or "downscale").lower()
    if strategy == "fail":
        reasons = []
        if over_budget_tokens:
            reasons.append(f"{gw * gh} tokens > navit_token_budget={max_tokens}")
        if over_side:
            reasons.append(
                f"side {max(gw, gh)} tokens > RoPE per-side limit {max_side}"
                f" (~{max_side * align}px)"
            )
        raise ValueError(
            f"[navit-native] image {W}x{H} exceeds native size limit ({'; '.join(reasons)})."
            "Raise navit_token_budget / increase max_img_h*max_img_w, or set "
            "navit_native_over_budget to downscale (default, auto proportional downscale)."
        )
    if strategy != "downscale":
        raise ValueError(
            f"unknown navit_native_over_budget={over_budget!r}; expected downscale or fail"
        )

    # downscale: scale proportionally to satisfy both the per-side limit and the token budget
    s = 1.0
    if over_side:
        s = min(s, max_side / float(max(gw, gh)))
    if max_tokens and gw * gh > max_tokens:
        s = min(s, math.sqrt(max_tokens / float(gw * gh)))
    ngw = max(1, int(gw * s))
    ngh = max(1, int(gh * s))
    # Rounding fallback after flooring: a side / the budget can still be pushed
    # over by max(1,.), clamp back to the primary axis
    if max_side:
        ngw, ngh = min(ngw, max_side), min(ngh, max_side)
    if max_tokens and ngw * ngh > max_tokens:
        if ngw >= ngh:
            ngw = max(1, max_tokens // ngh)
        else:
            ngh = max(1, max_tokens // ngw)
    return NativeFitImagePlan(
        source_width=W, source_height=H,
        width=ngw * align, height=ngh * align,
        token_w=ngw, token_h=ngh, token_count=ngw * ngh,
        was_downscaled=True,
    )


class BucketManager:
    """ARB bucket manager.

    SYNC WITH ``studio/web/src/lib/trainBuckets.ts``. The crop page on the web
    UI predicts trainer buckets to pre-align cluster crops so the trainer
    doesn't re-resize them -- that prediction depends on a TS port of this
    class. Any change to the algorithm or to the default parameters
    (``base_reso``, ``step``, the 0.1 area tolerance, the
    ``aspect_ratio_limit`` R, the min/max derivation) MUST land in both files
    in the same commit, or the frontend's predicted bucket != trainer's actual
    bucket and crops will silently degrade.

    The bucket set is a pure function of ``(base_reso, aspect_ratio_limit,
    step)``:

    - ``aspect_ratio_limit`` (R, default 2.0) symmetrically caps the widest
      bucket at R:1 and the tallest at 1:R.
    - ``min_reso`` / ``max_reso`` are the edge-length search bounds. When not
      given they are **derived** from ``(base_reso, R)`` -- at constant area
      base^2 the most extreme bucket has edges ``base*sqrt(R) x base/sqrt(R)``, so the
      bounds round outward to ``~ base/sqrt(R)`` and ``~ base*sqrt(R)`` (one ``step`` of
      margin so quantization never clips; the area band + AR cap do the real
      cut). Passing them explicitly (tests / special cases) overrides the
      derivation. The old hard-wired 512/2048 degrade at small base -- e.g.
      base=512 left only the 512x512 square, killing all AR variety -- which is
      why the bounds now scale with base.

    See ``docs/design/preprocess-crop-design.md`` SS7 for the crop UX policy and
    ``docs/design/multi-resolution-training-design.md`` SS6 for the derivation.
    """
    def __init__(self, base_reso=1024, min_reso=None, max_reso=None, step=64,
                 aspect_ratio_limit=2.0):
        self.base_reso = base_reso
        self.aspect_ratio_limit = aspect_ratio_limit
        self.step = step
        if min_reso is None or max_reso is None:
            span = math.sqrt(aspect_ratio_limit)
            derived_min = max(step, int(math.floor(base_reso / span / step) * step) - step)
            derived_max = int(math.ceil(base_reso * span / step) * step) + step
            if min_reso is None:
                min_reso = derived_min
            if max_reso is None:
                max_reso = derived_max
        self.min_reso = min_reso
        self.max_reso = max_reso
        self.buckets = self._generate(min_reso, max_reso, step, base_reso, aspect_ratio_limit)

    def _generate(self, min_r, max_r, step, base, ar_limit):
        # Keep algorithm identical to trainBuckets.generateBuckets() in TS:
        #   - double loop over (w, h) in [min_r, max_r] step `step`
        #   - area within +-10% of base^2 (the 0.1 below)
        #   - max AR ratio <= ar_limit (R)
        # Default-param consumers (base=1024, R=2.0) should see exactly the same
        # 37 buckets on both sides -- covered by
        # `studio/web/src/lib/trainBuckets.test.ts` asserting count == 37.
        buckets = []
        base_area = base * base
        for w in range(min_r, max_r + 1, step):
            for h in range(min_r, max_r + 1, step):
                if abs(w * h - base_area) / base_area > 0.1:
                    continue
                if max(w/h, h/w) > ar_limit:
                    continue
                buckets.append((w, h))
        return buckets

    def get_bucket(self, w, h):
        # Snap by ABSOLUTE AR distance -- not relative. The TS port
        # `trainBuckets.snapToBucket()` mirrors this exactly. Multiple buckets
        # may share the same aspect ratio under the +-10% area band (e.g.
        # 1472^2/1536^2/1600^2 when base=1536); in that tie, prefer the bucket
        # whose area is closest to base^2 so exact-square inputs land on the
        # configured base square instead of the first smaller square.
        aspect = w / h
        base_area = self.base_reso * self.base_reso
        best = (self.base_reso, self.base_reso)
        best_score = (float("inf"), float("inf"))
        for bw, bh in self.buckets:
            score = (abs(aspect - bw/bh), abs(bw * bh - base_area))
            if score < best_score:
                best_score = score
                best = (bw, bh)
        return best


class ImageDataset(Dataset):
    """
    Image dataset.

    Supports two caption formats:
    1. JSON file (preferred) - supports categorized shuffle
    2. TXT file (fallback) - classic shuffle
    """
    # Keep in sync with studio/datasets.py:IMAGE_EXTS (anima_train.py is a
    # standalone CLI script that does not import the studio package; when one
    # changes, update the other too).
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    def __init__(self, data_dir, resolution=1024, bucket_mgr=None,
                 shuffle_caption=False, keep_tokens=0, flip_augment=False,
                 tag_dropout=0.0, prefer_json=True, caption_override=None,
                 resolutions=None, aspect_ratio_limit=2.0,
                 native_resolution=False, native_token_budget=0,
                 native_over_budget="downscale", native_max_side_tokens=0,
                 native_align=_ANIMA_LATENT.align_px, load_masks=False):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        # Multi-resolution: bucket_mgr is the manager for the base resolution
        # (backward compatible -- the single-ARB path still uses it, including
        # None -> square-bucket semantics). Managers for non-base resolutions
        # (folder px overrides / other entries in the config list) are built
        # on demand in bucket_mgrs by _bucket_mgr_for. Each sample's
        # target_reso decides which bucket set it uses; when target_reso is
        # unset (or == base) it goes through bucket_mgr.
        self.bucket_mgr = bucket_mgr
        self.aspect_ratio_limit = aspect_ratio_limit
        self.resolutions = [int(r) for r in resolutions] if resolutions else [resolution]
        self.bucket_mgrs = {}
        # NaViT native fixed size (navit_native_resolution, opt-in): when
        # enabled, each image is floor-aligned to a 16px native fixed size
        # (see _target_size_for / plan_native_fit_image), completely bypassing
        # ARB bucket quantization; when disabled (default) the fields below
        # are inert and the code path is byte-identical to the bucket path.
        self.native_resolution = bool(native_resolution)
        self.native_token_budget = int(native_token_budget or 0)
        self.native_over_budget = str(native_over_budget or "downscale").lower()
        self.native_max_side_tokens = int(native_max_side_tokens or 0)
        self.native_align = int(native_align or _ANIMA_LATENT.align_px)
        if self.native_resolution and len(self.resolutions) > 1:
            logger.warning(
                "[navit-native] multi-resolution fan-out is meaningless under "
                "navit_native_resolution (every entry for the same image produces the same "
                "native size): ignoring extra resolution entries %s, processing as a single "
                "native copy.",
                self.resolutions[1:],
            )
            # Actually collapse the fan-out (before _scan): without this the
            # same image would still be duplicated per entry, each encoding
            # an npz with identical content (implicit epoch xN + cache
            # time/disk xN).
            self.resolutions = self.resolutions[:1]
        self.shuffle_caption = shuffle_caption
        self.keep_tokens = keep_tokens
        self.flip_augment = flip_augment
        self.tag_dropout = tag_dropout
        self.prefer_json = prefer_json
        self.caption_override = caption_override  # regularization set: uniform caption, e.g. "1girl, solo"
        # masked loss (B2): load the {stem}.mask sidecar next to the image,
        # apply the same geometric transform as the image, then area-downsample
        # to latent resolution as the loss spatial weight.
        self.load_masks = bool(load_masks)
        self._mask_warned: set = set()

        # Try to import caption_utils (direct import to bypass __init__.py)
        self.caption_utils = None
        if prefer_json:
            try:
                import importlib.util
                import sys

                # Load caption_utils.py directly (after ADR 0003 PR-A, utils/
                # lives at the repo root, not runtime/utils/; __file__ is
                # runtime/training/dataset.py, so we need to go up three
                # parents to reach the repo root).
                utils_path = Path(__file__).parent.parent.parent / "utils" / "caption_utils.py"
                if utils_path.exists():
                    spec = importlib.util.spec_from_file_location("caption_utils", utils_path)
                    caption_module = importlib.util.module_from_spec(spec)
                    sys.modules["caption_utils"] = caption_module
                    spec.loader.exec_module(caption_module)

                    self.caption_utils = {
                        "load_and_build": caption_module.load_and_build_caption,
                        "load_json": caption_module.load_caption_json,
                        "normalize": caption_module.normalize_caption_json,
                        "build": caption_module.build_caption_from_json,
                    }
                    logger.info("JSON caption mode enabled (categorized shuffle)")
                else:
                    logger.warning(f"caption_utils.py not found: {utils_path}")
            except Exception as e:
                logger.warning(f"failed to load caption_utils: {e}, falling back to TXT mode")

        self.samples = self._scan()
        json_count = sum(1 for s in self.samples if s.get("json_path"))
        txt_count = len(self.samples) - json_count
        unique_count = len(set(id(s) for s in self.samples))
        logger.info(f"dataset: {unique_count} images -> {len(self.samples)} samples (including repeats) (JSON: {json_count}, TXT: {txt_count})")
        self._preflight_json_captions()
        self.bucket_for_index = self._build_bucket_for_index()

    def _build_bucket_for_index(self):
        """Pre-scan each image's size and compute each sample's bucket (tw, th)
        so BucketBatchSampler can batch by bucket.

        Required for the non-cached path: ``collate_fn`` uses ``torch.stack``
        to stack a batch's pixel_values; mixing different bucket sizes in a
        batch would crash. ``BucketBatchSampler`` relies on
        ``dataset.bucket_for_index`` to group same-size samples into the same
        batch. The cached path doesn't read this (``CachedLatentDataset``
        builds its own from the npz latent shape and exposes it as the outer
        wrapper), but this pre-scan is cheap too (only reads image headers).

        Multi-resolution: each sample goes through the manager for its
        ``target_reso`` (``_bucket_mgr_for``); the same image lands in
        different buckets at different resos, so we dedupe by
        ``(image, target_reso)``; each image is opened only once and its size
        reused. An entry with no manager (base entry with
        ``bucket_mgr=None``, i.e. unbucketed) maps to None -> sampler falls
        back to plain slicing.
        """
        from PIL import Image
        dims: dict = {}     # image path -> (w, h)
        by_key: dict = {}   # (image path, target_reso) -> bucket (tw, th)
        out = []
        for s in self.samples:
            target_reso = s.get("target_reso")
            # Not native and this entry is unbucketed (base square bucket) ->
            # None (sampler falls back to plain slicing), keeping the old path
            # unchanged
            if not self.native_resolution and self._bucket_mgr_for(target_reso) is None:
                out.append(None)
                continue
            path = str(s["image"])
            # native ignores target_reso (native size depends only on the
            # source image); the bucket path still keys by target_reso
            key = (path, None if self.native_resolution else target_reso)
            if key not in by_key:
                if path not in dims:
                    try:
                        with Image.open(s["image"]) as im:
                            dims[path] = (im.width, im.height)
                    except Exception:
                        dims[path] = None
                wh = dims[path]
                by_key[key] = self._target_size_for(*wh, target_reso=target_reso) if wh else None
            out.append(by_key[key])
        return out

    def _target_size_for(self, img_w, img_h, target_reso=None):
        """Target pixel size ``(tw, th)`` for one image (a multiple of 16),
        shared by the fixed-size path and cache validation so they use one
        common rule.

        - ``native_resolution=False`` (default): goes through the ARB bucket
          ``_bucket_mgr_for(target_reso).get_bucket``; when that entry is
          unbucketed (base square bucket, ``bucket_mgr=None``) -> returns
          ``None`` (caller falls back to square-bucket semantics). Byte-identical
          to the pre-change behavior.
        - ``native_resolution=True``: goes through ``plan_native_fit_image``
          (native floor-16 + over-budget downscale), completely ignoring
          ``target_reso`` and buckets.
        """
        if self.native_resolution:
            plan = plan_native_fit_image(
                img_w, img_h,
                max_tokens=self.native_token_budget,
                max_side_tokens=self.native_max_side_tokens,
                align=self.native_align,
                over_budget=self.native_over_budget,
            )
            return (plan.width, plan.height)
        mgr = self._bucket_mgr_for(target_reso)
        if mgr is None:
            return None
        return mgr.get_bucket(img_w, img_h)

    @staticmethod
    def _parse_folder_meta(name: str) -> tuple[int | None, int, str]:
        """Parse a folder name ``[Npx_][R_]label`` -> ``(reso_override, repeat, label)``.

        Token order (all optional): ``\\d+px`` resolution prefix -> ``\\d+``
        repeat prefix -> the remainder is the label.

        - ``1024px_2_data`` -> ``(1024, 2, 'data')``
        - ``768px_concept`` -> ``(768, 1, 'concept')``
        - ``1024px_data``   -> ``(1024, 1, 'data')``
        - ``5_concept`` (Kohya style, backward compatible) -> ``(None, 5, 'concept')``
        - ``concept``       -> ``(None, 1, 'concept')``

        The resolution value snaps to the nearest multiple of 64 (half-up) and
        clamps to ``[256, 4096]`` (matching the schema validator and the
        frontend's ``Math.round``, to avoid off-center buckets / cross-language
        rounding drift).
        SYNC WITH ``parseFolderMeta`` in ``studio/web/src/lib/folderMeta.ts`` --
        the two parsers must stay in sync.
        """
        reso: int | None = None
        repeat = 1
        rest = name
        m = re.match(r"^(\d+)px_(.*)$", rest)
        if m:
            raw = int(m.group(1))
            reso = max(256, min(4096, (raw + 32) // 64 * 64))  # round-half-up, matches JS Math.round
            rest = m.group(2)
        m = re.match(r"^(\d+)_(.*)$", rest)
        if m:
            repeat = max(int(m.group(1)), 1)
            rest = m.group(2)
        return reso, repeat, rest

    @staticmethod
    def _parse_repeats_from_dir(name: str) -> int:
        """Parse a Kohya-style repeat count from a folder name, e.g.
        '5_concept' -> 5 (kept for compatibility with old callers)."""
        return ImageDataset._parse_folder_meta(name)[1]

    def _bucket_mgr_for(self, reso):
        """Get the BucketManager for a given reso.

        The base resolution (reso is None or == self.resolution) goes through
        self.bucket_mgr -- keeping the old path unchanged (including
        bucket_mgr=None -> square bucket). Other resolutions get a manager
        built on demand and cached in bucket_mgrs.
        """
        if reso is None or reso == self.resolution:
            return self.bucket_mgr
        mgr = self.bucket_mgrs.get(reso)
        if mgr is None:
            mgr = BucketManager(reso, aspect_ratio_limit=self.aspect_ratio_limit)
            self.bucket_mgrs[reso] = mgr
        return mgr

    def _make_sample(self, img_path):
        """Build a sample dict for one image; return None if no caption is found"""
        sample = {"image": img_path}
        json_path = img_path.with_suffix(".json")
        if self.prefer_json and json_path.exists():
            sample["json_path"] = json_path
            sample["txt_path"] = None
        else:
            txt_path = img_path.with_suffix(".txt")
            if not txt_path.exists():
                txt_path = img_path.with_suffix(".caption")
            if not txt_path.exists():
                return None
            sample["json_path"] = None
            sample["txt_path"] = txt_path
        return sample

    def _scan(self):
        """Scan the dataset directory, supporting Kohya-style repeat + multi-resolution.

        Folder name ``[Npx_][R_]label``::

            dataset/
            |-- 5_new/          <- repeat 5, uses config's resolutions
            |-- 1024px_2_hires/ <- repeat 2, fixed 1024 (overrides the list, no fan-out)
            `-- old/            <- repeat 1, uses config's resolutions

        - With an ``Npx_`` prefix -> that folder fixes its resolution at N,
          overriding the config list, no fan-out.
        - Without a px prefix -> uses ``self.resolutions``; when the list has
          more than one entry, each image gets one copy per entry (fan-out).

        Each unique image expands into ``repeat x that folder's resolution
        count`` samples, each carrying a ``target_reso``.
        """
        unique = []  # (sample_dict, repeat, resos)
        folder_info = []  # (name, repeat, resos, count) for logging

        # Root-level images (repeat=1, no px -> uses resolutions)
        root_count = 0
        for p in sorted(self.data_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in self.EXTS:
                s = self._make_sample(p)
                if s:
                    unique.append((s, 1, self.resolutions))
                    root_count += 1
        if root_count:
            folder_info.append(("(root)", 1, self.resolutions, root_count))

        # Subfolders (parse px override + repeat)
        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            reso_override, repeats, _label = self._parse_folder_meta(subdir.name)
            resos = [reso_override] if reso_override else self.resolutions
            count = 0
            for img_path in sorted(subdir.rglob("*")):
                if img_path.suffix.lower() not in self.EXTS:
                    continue
                s = self._make_sample(img_path)
                if s:
                    unique.append((s, repeats, resos))
                    count += 1
            if count:
                folder_info.append((subdir.name, repeats, resos, count))

        # Expand: repeat x resolution fan-out; each expanded sample carries a target_reso.
        # The repeat copies for the same (image, reso) share one dict; different
        # resos each get their own copy carrying their own target_reso.
        samples = []
        for s, repeat, resos in unique:
            for target_reso in resos:
                item = dict(s)
                item["target_reso"] = target_reso
                for _ in range(repeat):
                    samples.append(item)

        # Log: repeat x resolution per folder
        for name, rep, resos, cnt in folder_info:
            reso_str = "/".join(str(r) for r in resos)
            logger.info(
                f"  folder {name}: {cnt} images x repeat {rep} x resolution[{reso_str}] "
                f"= {cnt * rep * len(resos)} samples"
            )

        return samples

    def _preflight_json_captions(self):
        """Pre-check all JSON captions before training starts; reject training
        outright on any build failure (fail-fast).

        JSON samples have no .txt fallback (when prefer_json hits in
        ``_make_sample``, txt_path=None), so a caption build failure would
        silently degrade to an empty caption in ``__getitem__`` -- not even
        the trigger word survives, and the whole LoRA trains blank while
        training runs to completion as if nothing were wrong (#345).
        Rather than flooding the training log with a per-sample warning,
        report everything clearly once before training starts and abort.

        shuffle=False + dropout=0 guarantees the pre-check is deterministic and
        doesn't consume any random state.
        When caption_override provides a global override, caption files aren't
        read, so this check is skipped.
        """
        if self.caption_override is not None:
            return
        json_paths = []
        seen = set()
        for s in self.samples:
            jp = s.get("json_path")
            if jp and jp not in seen:
                seen.add(jp)
                json_paths.append(jp)
        if not json_paths:
            return
        if self.caption_utils is None:
            raise ValueError(
                f"the dataset has {len(json_paths)} JSON captions, but caption_utils failed to "
                f"load (see the warning above); these images would train with an empty caption, "
                f"so training has been rejected."
            )
        bad = []
        for jp in json_paths:
            try:
                caption = self.caption_utils["load_and_build"](
                    jp, shuffle=False, tag_dropout=0.0
                )
            except Exception:
                caption = None
            if caption is None:
                bad.append(jp)
        if bad:
            preview = "\n".join(f"  - {p}" for p in bad[:5])
            more = f"\n  ...and {len(bad)} more" if len(bad) > 5 else ""
            raise ValueError(
                f"{len(bad)} JSON captions failed to parse; the corresponding images would "
                f"train with an empty caption (not even a trigger word), so training has been "
                f"rejected. Please check or re-tag these files on the tagging page:\n{preview}{more}"
            )

    def _process_caption_txt(self, caption):
        """Process a TXT caption: Kohya-semantics keep_tokens + shuffle + tag_dropout.

        The keep_tokens prefix participates in neither shuffling nor dropout
        (same semantics as Kohya -- dropout possibly discarding the trigger
        word is a known ecosystem behavior; protection relies on the user
        explicitly configuring keep_tokens, not implicit value-based
        protection); the remaining tags are shuffled first, then each is
        independently dropped out with no safety net.
        """
        if not caption:
            return ""
        if "," in caption:
            tags = [t.strip() for t in caption.split(",")]
        else:
            tags = caption.split()

        kept = tags[:self.keep_tokens]
        rest = tags[self.keep_tokens:]
        if self.shuffle_caption:
            random.shuffle(rest)
        if self.tag_dropout > 0:
            rest = [t for t in rest if random.random() > self.tag_dropout]

        return ", ".join(kept + rest)

    def _process_caption_json(self, json_path):
        """Process a JSON caption: categorized shuffle"""
        if self.caption_utils is None:
            return None

        try:
            # Goes through caption_utils' authoritative pipeline (load ->
            # detect standard format -> normalize -> build). Earlier this
            # duplicated that check locally using `"tags" in raw_json`, which
            # only checks whether the key exists -- that would misdetect the
            # simplified form written by Studio tagging,
            # {"tags": [list], "meta": {trigger}}, as the standard format and
            # feed it straight to build, which crashes calling .get() on a
            # list (#345). load_and_build correctly branches on
            # isinstance(tags, dict): the list form goes through normalize to
            # move it into tags.tags, reusing a single source to avoid this
            # logic drifting apart again.
            return self.caption_utils["load_and_build"](
                json_path,
                shuffle=self.shuffle_caption,
                tag_dropout=self.tag_dropout,
            )
        except Exception as e:
            logger.warning(f"failed to process JSON {json_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def _mask_path_for(self, img_path) -> Path:
        """Studio training mask sidecar path: ``{stem}.mask`` next to the
        image, same directory and stem.

        Mirrors the on-disk convention in
        studio/services/preprocess/masks.py (extension is always .mask,
        content is grayscale PNG bytes) -- the stem excludes the extension, so
        X.jpg and X.png share the same mask.
        """
        img_path = Path(img_path)
        return img_path.parent / f"{img_path.stem}.mask"

    def _load_mask_image(self, img_path, size):
        """Load and validate a mask (grayscale L, size must equal the image's
        current size).

        Fail-safe (design SS2): missing file -> None (learn the whole image
        normally); size mismatch / unreadable -> warn once + None, **never
        crash training**.
        """
        from PIL import Image
        mp = self._mask_path_for(img_path)
        if not mp.is_file():
            return None
        try:
            with Image.open(mp) as raw:
                raw.load()
                mask = raw.convert("L") if raw.mode != "L" else raw.copy()
        except Exception as e:  # noqa: BLE001
            if str(mp) not in self._mask_warned:
                self._mask_warned.add(str(mp))
                logger.warning(f"[masked-loss] mask unreadable, treating as no mask: {mp} ({e})")
            return None
        if mask.size != tuple(size):
            if str(mp) not in self._mask_warned:
                self._mask_warned.add(str(mp))
                logger.warning(
                    "[masked-loss] mask size %sx%s does not match image %sx%s, treating as no "
                    "mask (redraw the mask after editing the image externally): %s",
                    mask.size[0], mask.size[1], size[0], size[1], mp,
                )
            return None
        return mask

    def caption_for_sample(self, sample) -> str:
        """Resolve a sample's final caption without opening the image.

        Training's ``__getitem__``, the latent cache wrapper, and the Phase 2
        text-cache pre-scan all share this single source of truth, avoiding
        the JSON/TXT/override priority drifting apart across three places.
        Caption shuffle/dropout is still decided by the existing processing
        functions; the cached_varlen family forbids these random operations at
        the registry layer, so the pre-cache and the training batch end up
        with exactly the same deterministic text.
        """
        caption = None
        if self.caption_override is not None:
            caption = self.caption_override
        elif sample.get("json_path"):
            caption = self._process_caption_json(sample["json_path"])

        if caption is None and sample.get("txt_path"):
            caption = sample["txt_path"].read_text(encoding="utf-8").strip()
            caption = self._process_caption_txt(caption)

        return "" if caption is None else str(caption)

    def __getitem__(self, idx):
        # Default path: DataLoader can't pass extra arguments, so flip_augment
        # decides whether to flip randomly. When CachedLatentDataset wants
        # explicit flip control it calls get_with_flip(idx, flip=...) directly,
        # encoding each image once with flip=False and once with flip=True
        # during the cache stage, to avoid baking randomness into the npz
        # (Kohya-style dual-copy latent).
        flip = self.flip_augment and random.random() > 0.5
        return self.get_with_flip(idx, flip=flip)

    def get_with_flip(self, idx, *, flip: bool):
        """``__getitem__`` with explicit flip control.

        flip=True/False: force-flip / don't flip, the caller decides; used for
        the dual-copy cache encode. flip is decoupled from self.flip_augment --
        it doesn't read self.flip_augment or consume any random state.
        """
        import numpy as np
        from PIL import Image
        sample = self.samples[idx]
        img = Image.open(sample["image"]).convert("RGB")

        # Get the caption (a regularization set can uniformly override via caption_override)
        caption = self.caption_for_sample(sample)

        # Target size: ARB bucket (native off) or native floor-16 fixed size
        # (native on). Always goes through _target_size_for; when it returns
        # None (base entry with no bucketing) falls back to a target_reso
        # square bucket.
        target_reso = sample.get("target_reso")
        size = self._target_size_for(img.width, img.height, target_reso)
        if size is not None:
            tw, th = size
        else:
            tw = th = target_reso or self.resolution

        # masked loss: the mask is loaded at the original size (validated ==
        # image size), then goes through exactly the same geometric transform
        # as the image below (NEAREST to avoid grayscale interpolation
        # contamination), and finally area-downsampled to latent /8
        # (BOX = block mean, SS9 decision 4).
        mask_img = None
        if self.load_masks:
            mask_img = self._load_mask_image(sample["image"], (img.width, img.height))

        # Resize and crop
        scale = max(tw / img.width, th / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        img = img.resize((nw, nh), Image.LANCZOS)

        left = (nw - tw) // 2
        top = (nh - th) // 2
        img = img.crop((left, top, left + tw, top + th))

        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        mask_tensor = None
        if mask_img is not None:
            m = mask_img.resize((nw, nh), Image.NEAREST)
            m = m.crop((left, top, left + tw, top + th))
            if flip:
                m = m.transpose(Image.FLIP_LEFT_RIGHT)
            _vs = _ANIMA_LATENT.spatial_stride
            m = m.resize((max(1, tw // _vs), max(1, th // _vs)), Image.BOX)
            # Grayscale 255=learn / 0=don't learn -> [0,1] loss spatial weight
            mask_tensor = torch.from_numpy(
                np.array(m).astype(np.float32) / 255.0
            )

        # Convert to tensor [-1, 1]
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return {"pixel_values": tensor, "caption": caption, "mask": mask_tensor}


class RepeatDataset(Dataset):
    """Kohya-style dataset repeat"""
    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class MergedDataset(Dataset):
    """Merge the main dataset with a regularization dataset (Kohya-style reg)"""
    def __init__(self, main_dataset, reg_dataset, reg_weight: float = 1.0):
        self.main_dataset = main_dataset
        self.reg_dataset = reg_dataset
        self.reg_weight = float(reg_weight)
        self._main_len = len(main_dataset)
        self._reg_len = len(reg_dataset)

        # Build bucket_for_index for BucketBatchSampler
        self.bucket_for_index = self._build_bucket_for_index()

    def _get_cached_dataset(self, d):
        if hasattr(d, "bucket_for_index"):
            return d
        if hasattr(d, "dataset"):
            return self._get_cached_dataset(d.dataset)
        return None

    def _build_bucket_for_index(self):
        main_cached = self._get_cached_dataset(self.main_dataset)
        reg_cached = self._get_cached_dataset(self.reg_dataset)
        buckets = []
        if main_cached and main_cached.bucket_for_index:
            main_base_len = len(main_cached.bucket_for_index)
            for idx in range(self._main_len):
                b = main_cached.bucket_for_index[idx % main_base_len]
                buckets.append(b if b is not None else (0, 0))
        else:
            buckets.extend([(0, 0)] * self._main_len)
        if reg_cached and reg_cached.bucket_for_index:
            reg_base_len = len(reg_cached.bucket_for_index)
            for idx in range(self._reg_len):
                b = reg_cached.bucket_for_index[idx % reg_base_len]
                buckets.append(b if b is not None else (0, 0))
        else:
            buckets.extend([(0, 0)] * self._reg_len)
        return buckets

    def __len__(self):
        return self._main_len + self._reg_len

    def __getitem__(self, idx):
        if idx < self._main_len:
            item = self.main_dataset[idx]
            item["loss_weight"] = 1.0
            item["is_reg"] = False
            return item
        item = self.reg_dataset[idx - self._main_len]
        item["loss_weight"] = self.reg_weight
        item["is_reg"] = True
        return item


class BucketBatchSampler:
    """Batch sampler that groups samples by bucket so latents in each batch have the same size."""
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._cached_dataset = self._get_cached_dataset(dataset)
        self._base_len = len(self._cached_dataset) if self._cached_dataset else 0

    def _get_cached_dataset(self, d):
        if hasattr(d, "bucket_for_index"):
            return d
        if hasattr(d, "dataset"):
            return self._get_cached_dataset(d.dataset)
        return None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        # Under ARB the actual batch count = sum_bucket f(n_b, bs); using the
        # global n would be biased (each bucket has its own remainder).
        # Without bucket info, fall back to the global formula (linear
        # DataLoader behavior).
        if self._cached_dataset is None:
            n = len(self.dataset)
            if self.drop_last:
                return n // self.batch_size
            return (n + self.batch_size - 1) // self.batch_size
        counts = {}
        for idx in range(len(self.dataset)):
            base_idx = idx % self._base_len
            bucket = self._cached_dataset.bucket_for_index[base_idx]
            if bucket is None:
                bucket = (0, 0)
            counts[bucket] = counts.get(bucket, 0) + 1
        total = 0
        for n in counts.values():
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        if self._cached_dataset is None:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch
            return

        bucket_to_indices = {}
        for idx in range(len(self.dataset)):
            base_idx = idx % self._base_len
            bucket = self._cached_dataset.bucket_for_index[base_idx]
            if bucket is None:
                bucket = (0, 0)
            bucket_to_indices.setdefault(bucket, []).append(idx)

        buckets = list(bucket_to_indices.keys())
        if self.shuffle:
            rng.shuffle(buckets)
        for bucket in buckets:
            indices = bucket_to_indices[bucket]
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch


# ============================================================ Tiled VAE encode
# cache_encode_tiled: slice oversized images into pixel tiles, encode each
# tile, then feather-blend the results on the latent grid.
# Peak VRAM drops from being proportional to the full image's pixel count to
# being proportional to a single tile's pixel count.

_CACHE_ENCODE_MAX_PIXELS = 4 * 1024 * 1024


class CachedLatentDataset(Dataset):
    """Kohya-style npz-file-cached dataset.

    When flip_augment + cache_latents are both on, uses Kohya's dual-copy
    latent mode:
      - During the cache stage, each image is encoded twice (flip=False /
        flip=True), stored under the npz's `latent` / `latent_flipped` keys
        respectively
      - During training, __getitem__ picks the flipped version with 50%
        probability
    Older versions silently baked "the random flip from the cache stage" into
    the npz, permanently disabling flip augmentation and permanently
    mirror-contaminating 50% of the data; the current version detects a
    missing latent_flipped key via _is_cache_valid and auto-re-encodes to fix it.
    """

    #: Latent spec (fingerprinted into the cache criteria / written to disk).
    #: Class-level default = Anima; overridable via __init__.
    #: From PR-2b onward it's passed in via ctx.family.spec.latent.
    latent_spec = _ANIMA_LATENT

    def __init__(self, base_dataset, vae, device, dtype, cache_dir=None, cache_batch_size=1,
                 encode_tiled=False, encode_tile_px=1024, encode_tile_overlap=128,
                 encode_max_pixels=0, latent_spec=None, label=""):
        import numpy as np
        if latent_spec is not None:
            self.latent_spec = latent_spec
        # Log label (e.g. "training set"/"regularization set"): the main set
        # and the regularization set each build one instance; without a label
        # the cache log would print two indistinguishable "checking VAE
        # latent cache..." lines.
        self.label = str(label or "")
        self.base_dataset = base_dataset
        self.base_image_dataset = self._get_base_image_dataset(base_dataset)
        self.np = np
        # Get the underlying dataset's samples list
        self.samples = self._get_base_samples(base_dataset)
        # When the same image fans out to multiple resolutions, the npz must
        # be split per file (otherwise different-resolution latents would
        # overwrite each other). Only images that genuinely appear under more
        # than one target_reso use the r{reso} naming; single-resolution
        # images keep img.npz, leaving existing caches untouched.
        _resos_per_img: dict[str, set] = {}
        for s in self.samples:
            _resos_per_img.setdefault(str(s["image"]), set()).add(s.get("target_reso"))
        self._multi_reso = {img for img, rs in _resos_per_img.items() if len(rs) > 1}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.bucket_for_index = []
        self.token_count_for_index = []  # read by the NaViT packer (empty by default, filled by _fill)
        self.cache_batch_size = max(1, int(cache_batch_size or 1))
        # Whether the cache needs the dual-copy latent -- depends on the
        # underlying ImageDataset.flip_augment
        self.flip_augment = bool(
            getattr(self.base_image_dataset, "flip_augment", False)
        )
        # masked loss (B2): when the underlying ImageDataset.load_masks is
        # enabled, the mask is downsampled during the cache encode stage and
        # stored in the npz too (mask / mask_flipped keys, mirroring latent_flipped)
        self.load_masks = bool(
            getattr(self.base_image_dataset, "load_masks", False)
        )
        # cache_encode_tiled (opt-in): oversized images switch to tiled
        # encode + latent feather-blend; peak VRAM proportional to a single
        # tile's pixel count. Images within the threshold keep the old path
        # (byte-identical).
        self.encode_tiled = bool(encode_tiled)
        self.encode_tile_px = int(encode_tile_px or 1024)
        self.encode_tile_overlap = int(encode_tile_overlap or 128)
        self.encode_max_pixels = (
            int(encode_max_pixels) if int(encode_max_pixels or 0) > 0
            else _CACHE_ENCODE_MAX_PIXELS
        )
        self._build_cache(vae, device, dtype)

    def _get_base_samples(self, dataset):
        """Get the underlying ImageDataset's samples"""
        if hasattr(dataset, "samples"):
            return dataset.samples
        elif hasattr(dataset, "dataset"):
            return self._get_base_samples(dataset.dataset)
        return []

    def _get_base_image_dataset(self, dataset):
        if hasattr(dataset, "samples") and hasattr(dataset, "bucket_mgr"):
            return dataset
        if hasattr(dataset, "dataset"):
            return self._get_base_image_dataset(dataset.dataset)
        return None

    def _expected_bucket_size(self, img_path, target_reso=None):
        base = self.base_image_dataset
        if base is None:
            return None
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                # Uses the same fixed-size rule as ImageDataset.get_with_flip
                # (bucket or native) so cache validation checks against the
                # actual encode size. Under native this yields the native
                # floor-16 size.
                if hasattr(base, "_target_size_for"):
                    size = base._target_size_for(img.width, img.height, target_reso)
                    if size is not None:
                        return size
                elif hasattr(base, "_bucket_mgr_for"):
                    mgr = base._bucket_mgr_for(target_reso)
                    if mgr:
                        return mgr.get_bucket(img.width, img.height)
                resolution = int(target_reso or getattr(base, "resolution"))
                return (resolution, resolution)
        except Exception:
            return None

    def _get_npz_path(self, img_path, target_reso=None):
        """The npz cache path for an image.

        Single-resolution image -> ``img.npz`` (leaves existing caches
        untouched); an image fanned out to multiple resolutions ->
        ``img.r{reso}.npz``, so different-resolution latents don't overwrite
        each other.
        """
        img_path = Path(img_path)
        if target_reso is not None and str(img_path) in getattr(self, "_multi_reso", set()):
            return img_path.with_suffix(f".r{int(target_reso)}.npz")
        return img_path.with_suffix(".npz")

    def _is_cache_valid(self, img_path, npz_path, target_reso=None):
        """Check whether the cache is valid (the image hasn't been modified,
        and the format is compatible with the current flip_augment setting).

        - Missing `latent` key / an incompatible cache from another model ->
          delete and re-encode
        - latent fingerprint / layout version mismatch (VAE latent space
          changed, or the cache layout was upgraded) -> delete and re-encode;
          a legacy cache with no fingerprint key is grandfathered as
          wan21-f8c16 (the only VAE that historically produced caches,
          avoiding a full re-encode on upgrade, 04-synthesis D12)
        - flip_augment=True and the npz is missing the `latent_flipped` key ->
          invalid, re-encode (an old single-copy cache is exactly the
          "flip permanently baked in" contaminated state, and must be
          re-encoded to fix it)
        - flip_augment=False and the npz has `latent_flipped` -> still
          considered valid (the dual-copy cache is a superset of flip mode;
          turning flip off just reads latent without wasting anything)
        - bucket size mismatch -> invalid
        - When masked loss is enabled, three mask-sidecar/npz consistency
          checks (SS9 decision 3, missing any one means training on a stale
          mask): mask exists but the npz has no `mask` key (newly drawn);
          mask mtime newer than the npz (redrawn); mask deleted but the npz
          still has a `mask` key (cleared).
          These checks are skipped when load_masks is off (a cache with a
          mask key is a superset, nothing wasted).
        """
        if not npz_path.exists():
            return False
        if npz_path.stat().st_mtime < img_path.stat().st_mtime:
            return False
        mask_path = None
        if getattr(self, "load_masks", False) and self.base_image_dataset is not None:
            mask_path = self.base_image_dataset._mask_path_for(img_path)
        try:
            with self.np.load(npz_path) as data:
                if "latent" not in data.files:
                    npz_path.unlink()
                    logger.debug(f"deleted incompatible cache: {npz_path.name}")
                    return False
                cache_fp = (
                    str(data["latent_fingerprint"].item())
                    if "latent_fingerprint" in data.files
                    else _LEGACY_CACHE_FINGERPRINT
                )
                cache_ver = (
                    int(data["layout_version"])
                    if "layout_version" in data.files
                    else LATENT_CACHE_LAYOUT_VERSION
                )
                if (cache_fp != self.latent_spec.fingerprint
                        or cache_ver != LATENT_CACHE_LAYOUT_VERSION):
                    npz_path.unlink()
                    logger.debug(
                        f"deleted cache with mismatched fingerprint: {npz_path.name} "
                        f"({cache_fp} v{cache_ver} != "
                        f"{self.latent_spec.fingerprint} v{LATENT_CACHE_LAYOUT_VERSION})"
                    )
                    return False
                if getattr(self, "flip_augment", False) and "latent_flipped" not in data.files:
                    return False
                expected_bucket = self._expected_bucket_size(img_path, target_reso)
                if expected_bucket is not None:
                    if "bucket_w" not in data.files or "bucket_h" not in data.files:
                        return False
                    if (int(data["bucket_w"]), int(data["bucket_h"])) != expected_bucket:
                        return False
                if mask_path is not None:
                    has_mask_file = mask_path.is_file()
                    has_mask_key = "mask" in data.files
                    if has_mask_file != has_mask_key:
                        return False
                    if has_mask_file and npz_path.stat().st_mtime < mask_path.stat().st_mtime:
                        return False
                    if (
                        has_mask_key
                        and getattr(self, "flip_augment", False)
                        and "mask_flipped" not in data.files
                    ):
                        return False
        except Exception:
            try:
                npz_path.unlink()
            except Exception:
                pass
            return False
        return True

    def _build_cache(self, vae, device, dtype):
        """Build/load the npz cache.

        Per-folder repeat (the 5_concept prefix) makes the same image appear
        repeated N times in samples; multi-resolution fan-out also makes the
        same image appear multiple times under different target_reso values.
        The npz destination is decided by `_get_npz_path(img, target_reso)` --
        a single-resolution image uses `img.npz`, an image fanned out to
        multiple resolutions uses `img.r{reso}.npz`, split per file. Dedupe by
        npz_path so each (image, reso) is encoded at most once; otherwise the
        same npz would be repeatedly overwritten N times (doubled again under
        flip_augment).
        """
        tag = f" ({self.label})" if self.label else ""
        logger.info(f"checking VAE latent cache{tag}...")
        to_encode = []
        seen_npz = set()
        unique_total = 0
        for i, sample in enumerate(self.samples):
            img_path = sample["image"]
            npz_path = self._get_npz_path(
                img_path, sample.get("target_reso"))
            if npz_path in seen_npz:
                continue
            seen_npz.add(npz_path)
            unique_total += 1
            if not self._is_cache_valid(img_path, npz_path, sample.get("target_reso")):
                to_encode.append(i)

        if to_encode:
            logger.info(f"need to encode {len(to_encode)}/{unique_total} images{tag}...")
            self._encode_and_save(to_encode, vae, device, dtype)
        else:
            logger.info(f"all {unique_total} images already cached{tag}")

        self._fill_bucket_for_index()

    def _fill_bucket_for_index(self):
        """Fill bucket_for_index for all samples (needed for BucketBatchSampler).
        Uses latent spatial shape (h, w) as grouping key so batches have consistent tensor sizes.

        Also fills token_count_for_index (NaViT packer reads it; patch_spatial=2)."""
        self.bucket_for_index = [None] * len(self.samples)
        self.token_count_for_index = [0] * len(self.samples)
        patch_spatial = _ANIMA_LATENT.patch_spatial
        for i in range(len(self.samples)):
            npz_path = self._get_npz_path(
                self.samples[i]["image"], self.samples[i].get("target_reso"))
            if not npz_path.exists():
                continue
            with self.np.load(npz_path) as data:
                latent = data["latent"]
                s = latent.shape
            if len(s) == 5:
                _, _, _, h, w = s
            else:
                _, _, h, w = s
            self.bucket_for_index[i] = (int(h), int(w))
            self.token_count_for_index[i] = (int(h) // patch_spatial) * (int(w) // patch_spatial)

    def _encode_and_save(self, indices, vae, device, dtype):
        """Encode images and save as npz.

        When flip_augment=True, each image is encoded twice (flip=False /
        flip=True), stored under the `latent` / `latent_flipped` keys
        respectively; during training __getitem__ picks one at random.
        When flip_augment=False, only encodes once, storing `latent`.

        Groups by actual bucket size and batches into the VAE; different
        sizes can't be stacked, so they're accumulated separately.
        When cache_encode_tiled=True, images over the pixel budget switch to
        tiled encode + latent feather-blend.
        """
        base_img = self.base_image_dataset
        want_flip = self.flip_augment and base_img is not None
        pending = {}
        encoded_count = 0

        def _encode_pixels(pixel_tensors):
            pixels = torch.stack(pixel_tensors, dim=0).to(device, dtype=dtype)
            with torch.inference_mode():
                # Goes through VAEWrapper.encode (including auto/on tiling), so
                # large images/large batches won't hit a VRAM cliff
                latents = vae.encode(pixels.unsqueeze(2))
            return latents.detach().cpu().float()

        def _encode_tiled_single(pixel_tensor):
            """Tiled encode for a single image (when cache_encode_tiled is over the pixel budget)."""
            pixels = pixel_tensor.unsqueeze(0).to(device, dtype=dtype).unsqueeze(2)  # [1,C,1,H,W]
            with torch.inference_mode():
                # Uses VAEWrapper's tiled encode directly (tile size
                # configurable): a single layer of tiling plus unified cosine
                # feathering, avoiding double tiling from wrapping another
                # vae.encode call around it.
                lat = vae._tiled_encode(
                    pixels, self.encode_tile_px, self.encode_tile_overlap
                )
            return lat.detach().cpu().float()[0]

        def _flush(bucket_key):
            nonlocal encoded_count
            batch = pending.pop(bucket_key, [])
            if not batch:
                return

            h, w = int(batch[0]["bucket_h"]), int(batch[0]["bucket_w"])
            use_tiled = (
                getattr(self, "encode_tiled", False)
                and h > 0 and w > 0
                and h * w > self.encode_max_pixels
            )

            def _mask_kwargs(entry):
                """masked loss: the mask has already been downsampled to
                latent resolution inside get_with_flip, so it's stored in the
                npz as-is (an image with no mask writes no key -- the key's
                presence is the cache-invalidation criterion)."""
                out = {}
                if entry.get("mask") is not None:
                    out["mask"] = entry["mask"].numpy()
                if entry.get("mask_flipped") is not None:
                    out["mask_flipped"] = entry["mask_flipped"].numpy()
                return out

            if use_tiled:
                logger.info(
                    "[cache-tiled] %dx%d over the pixel budget, tiled encode (tile=%d overlap=%d)",
                    w, h, self.encode_tile_px, self.encode_tile_overlap,
                )
                for entry in batch:
                    lat = _encode_tiled_single(entry["pixels"])
                    lat_f = _encode_tiled_single(entry["pixels_flipped"]) if want_flip else None
                    npz_kwargs = {"latent": lat.numpy()}
                    if lat_f is not None:
                        npz_kwargs["latent_flipped"] = lat_f.numpy()
                    npz_kwargs.update(_mask_kwargs(entry))
                    _entry_sample = self.samples[entry["index"]]
                    npz_path = self._get_npz_path(
                        _entry_sample["image"], _entry_sample.get("target_reso"))
                    self.np.savez(
                        npz_path,
                        bucket_w=entry["bucket_w"],
                        bucket_h=entry["bucket_h"],
                        latent_fingerprint=self.latent_spec.fingerprint,
                        layout_version=LATENT_CACHE_LAYOUT_VERSION,
                        **npz_kwargs,
                    )
                    encoded_count += 1
                    if encoded_count % 10 == 0 or encoded_count == len(indices):
                        logger.info(f"  encoding progress: {encoded_count}/{len(indices)}")
                return

            latents = _encode_pixels([entry["pixels"] for entry in batch])
            if want_flip:
                latents_flipped = _encode_pixels([entry["pixels_flipped"] for entry in batch])
            else:
                latents_flipped = [None] * len(batch)

            for n, entry in enumerate(batch):
                npz_kwargs = {"latent": latents[n].numpy()}
                if want_flip:
                    npz_kwargs["latent_flipped"] = latents_flipped[n].numpy()
                npz_kwargs.update(_mask_kwargs(entry))

                _entry_sample = self.samples[entry["index"]]
                npz_path = self._get_npz_path(
                    _entry_sample["image"], _entry_sample.get("target_reso"))
                self.np.savez(
                    npz_path,
                    bucket_w=entry["bucket_w"],
                    bucket_h=entry["bucket_h"],
                    latent_fingerprint=self.latent_spec.fingerprint,
                    layout_version=LATENT_CACHE_LAYOUT_VERSION,
                    **npz_kwargs,
                )
                encoded_count += 1
                if encoded_count % 10 == 0 or encoded_count == len(indices):
                    logger.info(f"  encoding progress: {encoded_count}/{len(indices)}")

        logger.info(f"VAE cache batch size: {self.cache_batch_size}")
        for i in indices:
            if base_img is not None:
                # Explicit flip control, to avoid baking randomness into the npz
                item = base_img.get_with_flip(i, flip=False)
            else:
                item = self.base_dataset[i]
            pixels = item["pixel_values"]
            _, ph, pw = pixels.shape
            bucket_w, bucket_h = pw, ph

            pixels_flipped = None
            mask_flipped = None
            if want_flip:
                item_f = base_img.get_with_flip(i, flip=True)
                pixels_flipped = item_f["pixel_values"]
                mask_flipped = item_f.get("mask")

            bucket_key = (bucket_h, bucket_w)
            pending.setdefault(bucket_key, []).append({
                "index": i,
                "pixels": pixels,
                "pixels_flipped": pixels_flipped,
                "mask": item.get("mask"),
                "mask_flipped": mask_flipped,
                "bucket_w": bucket_w,
                "bucket_h": bucket_h,
            })
            if len(pending[bucket_key]) >= self.cache_batch_size:
                _flush(bucket_key)

        for bucket_key in list(pending):
            _flush(bucket_key)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npz_path = self._get_npz_path(
            sample["image"], sample.get("target_reso"))
        data = self.np.load(npz_path)
        # When flip_augment=True and the npz has latent_flipped, picks the
        # mirrored version with 50% probability, matching the flip
        # probability of the non-cached path ImageDataset.__getitem__.
        # Without a latent_flipped key (a single-copy cache when
        # flip_augment=False) only latent is read.
        use_flip = (
            self.flip_augment
            and "latent_flipped" in data.files
            and random.random() > 0.5
        )
        latent_key = "latent_flipped" if use_flip else "latent"
        latent = torch.from_numpy(data[latent_key])

        # masked loss: the mask keeps the same flip choice as the latent
        # (a mismatch would apply the weight to the mirrored position).
        # No key in the npz = this image has no mask (collate fills all 1s).
        mask = None
        if getattr(self, "load_masks", False):
            mask_key = "mask_flipped" if use_flip and "mask_flipped" in data.files else "mask"
            if mask_key in data.files:
                mask = torch.from_numpy(data[mask_key])

        # Get a reference to base_dataset (handling possible nesting)
        base = self.base_dataset
        while hasattr(base, "dataset"):
            base = base.dataset

        # Process the caption (shares the same source of truth as
        # ImageDataset / the text-cache pre-scan)
        if hasattr(base, "caption_for_sample"):
            caption = base.caption_for_sample(sample)
        else:  # a third-party wrapper that isn't ImageDataset: keep the pre-Phase-2 duck-type behavior
            caption = None
            if getattr(base, "caption_override", None) is not None:
                caption = base.caption_override
            elif sample.get("json_path") and hasattr(base, "_process_caption_json"):
                caption = base._process_caption_json(sample["json_path"])

            if caption is None and sample.get("txt_path"):
                caption = sample["txt_path"].read_text(encoding="utf-8").strip()
                if hasattr(base, "_process_caption_txt"):
                    caption = base._process_caption_txt(caption)

            if caption is None:
                caption = ""

        return {
            "latent": latent,
            "caption": caption,
            "mask": mask,
            # navit collate needs the per-image image path
            "image": str(sample["image"]),
        }


def _stack_masks(batch, h, w):
    """masked loss: only emit a mask batch when at least one sample has one
    (zero overhead when none do).

    Samples with no mask are filled with all 1s (learn normally); same-bucket
    membership guarantees consistent mask spatial size.
    """
    if not any(b.get("mask") is not None for b in batch):
        return None
    return torch.stack([
        b["mask"] if b.get("mask") is not None else torch.ones(h, w)
        for b in batch
    ])


def collate_fn(batch):
    """DataLoader collate"""
    pixels = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["caption"] for b in batch]
    result = {"pixel_values": pixels, "captions": captions}
    masks = _stack_masks(
        batch,
        pixels.shape[-2] // _ANIMA_LATENT.spatial_stride,
        pixels.shape[-1] // _ANIMA_LATENT.spatial_stride,
    )
    if masks is not None:
        result["masks"] = masks
    if "loss_weight" in batch[0]:
        result["loss_weight"] = torch.tensor([b["loss_weight"] for b in batch], dtype=torch.float32)
        result["is_reg"] = torch.tensor([b["is_reg"] for b in batch], dtype=torch.bool)
    return result


def collate_fn_cached(batch):
    """DataLoader collate for cached latents"""
    latents = torch.stack([b["latent"] for b in batch])
    captions = [b["caption"] for b in batch]
    result = {"latents": latents, "captions": captions}
    masks = _stack_masks(batch, latents.shape[-2], latents.shape[-1])
    if masks is not None:
        result["masks"] = masks
    if "loss_weight" in batch[0]:
        result["loss_weight"] = torch.tensor([b["loss_weight"] for b in batch], dtype=torch.float32)
        result["is_reg"] = torch.tensor([b["is_reg"] for b in batch], dtype=torch.bool)
    return result


# =================================================== NaViT / Patch-n-Pack packing
# Token-budget packer + block-diagonal collate: pack images with different
# token counts into one training sequence (zero padding).


def pack_indices_by_budget(token_counts, token_budget, order, max_images_per_pack=0):
    """Greedy next-fit packing: place sample indices into packs whose total
    token count is <= budget.

    NaViT block-diagonal packing has no padding; a pack's cost = the sum of
    its images' token counts. ``order`` is an already-shuffled index
    sequence; an image whose own token count exceeds the budget gets its own
    pack (caller warns).
    The result covers each index in ``order`` exactly once, preserving order.
    """
    packs = []
    cur, cur_sum = [], 0
    cap = int(max_images_per_pack or 0)
    budget = int(token_budget)
    for idx in order:
        n = int(token_counts[idx])
        over_budget = bool(cur) and (cur_sum + n > budget)
        over_count = cap > 0 and len(cur) >= cap
        if over_budget or over_count:
            packs.append(cur)
            cur, cur_sum = [], 0
        cur.append(idx)
        cur_sum += n
    if cur:
        packs.append(cur)
    return packs


def pack_indices_ffd_windowed(token_counts, token_budget, order,
                              max_images_per_pack=0, window=0):
    """Windowed First-Fit-Decreasing packing: run FFD within windows of the
    (already-shuffled) ``order``.

    Classic FFD (sort by size descending, place each item into the first bin
    it fits) packs tighter than next-fit -- fewer, fuller packs => fewer
    optimizer steps, less wasted token budget (see NeMo sequence-packing /
    ICLR'23 "Efficient Sequence Packing"). Cost: a fully global descending
    sort would group the same images together every epoch (size order is
    fixed), weakening batch diversity for small-data SGD.

    The fix is ``window``: ``order`` is split into contiguous windows of size
    ``window``, and FFD runs within each window. Since ``order`` is reshuffled
    every epoch, window membership (and thus grouping) varies across epochs,
    while the within-window descending sort still recovers most of the
    packing benefit. ``window<=0`` means a single global window (maximum
    packing, but the packs are fixed every epoch -- suitable only for
    single-pass data).

    Covers each index in ``order`` exactly once. An image over budget on its
    own gets its own pack (same as next-fit).
    """
    budget = int(token_budget)
    cap = int(max_images_per_pack or 0)
    win = int(window or 0)
    order = list(order)
    if win <= 0:
        windows = [order]
    else:
        windows = [order[i:i + win] for i in range(0, len(order), win)]

    packs = []
    for w in windows:
        items = sorted(w, key=lambda i: int(token_counts[i]), reverse=True)
        bins = []  # each: [list_of_indices, summed_tokens]
        for idx in items:
            n = int(token_counts[idx])
            placed = False
            for b in bins:
                over_count = cap > 0 and len(b[0]) >= cap
                if (not over_count) and (b[1] + n <= budget):
                    b[0].append(idx)
                    b[1] += n
                    placed = True
                    break
            if not placed:
                bins.append([[idx], n])
        packs.extend(b[0] for b in bins)
    return packs


def _lookup_token_count_walk(d, idx):
    """Resolve a sample's token count by walking the dataset wrapper (the
    free-function version of the NaViT packer)."""
    main = getattr(d, "main_dataset", None)
    reg = getattr(d, "reg_dataset", None)
    if main is not None and reg is not None:
        ml = getattr(d, "_main_len", len(main))
        if idx < ml:
            return _lookup_token_count_walk(main, idx)
        return _lookup_token_count_walk(reg, idx - ml)
    inner = getattr(d, "dataset", None)
    if inner is not None and inner is not d and not isinstance(inner, list):
        return _lookup_token_count_walk(inner, idx % len(inner))
    counts = getattr(d, "token_count_for_index", None)
    if counts is not None and len(counts) > 0:
        return int(counts[idx % len(counts)])
    inner = getattr(d, "base_dataset", None)
    if inner is not None and inner is not d:
        return _lookup_token_count_walk(inner, idx % len(inner))
    return 0


def _walk_attr_list(dataset, attr):
    """Look up the leaf dataset's per-index list attribute ``attr`` through a
    single-chain wrapper (RepeatDataset/CachedLatentDataset), mapping via
    ``% len`` onto ``len(dataset)``.
    Returns None for a MergedDataset (two branches) or when the attribute
    doesn't exist."""
    cur = dataset
    for _ in range(12):
        if getattr(cur, "main_dataset", None) is not None and getattr(cur, "reg_dataset", None) is not None:
            return None  # MergedDataset: not a single chain
        v = getattr(cur, attr, None)
        if v is not None and len(v) > 0:
            n = len(dataset)
            return [v[i % len(v)] for i in range(n)]
        nxt = getattr(cur, "dataset", None)
        if nxt is None or nxt is cur or isinstance(nxt, list):
            nxt = getattr(cur, "base_dataset", None)
        if nxt is None or nxt is cur:
            return None
        cur = nxt
    return None


def dataset_token_counts(dataset, patch_spatial=_ANIMA_LATENT.patch_spatial):
    """Per-index token counts for NaViT packing.

    Prefers the already-filled ``token_count_for_index`` (filled by
    CachedLatentDataset). If that field is all-zero or absent, derives it from
    the cached latent shape ``bucket_for_index = (h, w)`` (latent px) as
    ``(h // patch_spatial) * (w // patch_spatial)`` -- i.e. the post-patchify
    token count.
    If neither is available, falls back to a per-index walk (returning 0 ->
    the packer will fail-fast).
    """
    counts = _walk_attr_list(dataset, "token_count_for_index")
    if counts is not None and any(int(c) > 0 for c in counts):
        return [int(c) for c in counts]

    shapes = _walk_attr_list(dataset, "bucket_for_index")
    if shapes is not None:
        ps = max(1, int(patch_spatial))
        derived = []
        for s in shapes:
            if not s:
                derived.append(0)
                continue
            h, w = int(s[0]), int(s[1])
            derived.append((h // ps) * (w // ps))
        if any(c > 0 for c in derived):
            return derived

    return [int(_lookup_token_count_walk(dataset, i) or 0) for i in range(len(dataset))]


class NavitPackBatchSampler:
    """Produces dataset index packs for NaViT/Patch-n-Pack block-diagonal training.

    Each produced list is one packed training sequence: the sum of its
    images' token counts is <= ``token_budget``, and the whole pack becomes
    one zero-padding block-diagonal forward pass. This decouples "images per
    step" from any single image's shape -- images with different token counts
    and aspect ratios can share a pack, and even a small dataset can fill a
    large effective batch.
    """

    def __init__(self, dataset, token_budget, max_images_per_pack=0,
                 shuffle=True, seed=42, drop_last=False,
                 strategy="next_fit", ffd_window=256):
        self.dataset = dataset
        self.token_budget = int(token_budget)
        self.max_images_per_pack = int(max_images_per_pack or 0)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.strategy = str(strategy or "next_fit").lower()
        if self.strategy not in ("next_fit", "ffd"):
            raise ValueError(
                f"navit pack strategy must be 'next_fit' or 'ffd', got {strategy!r}"
            )
        self.ffd_window = int(ffd_window or 0)
        self.epoch = 0
        self.token_counts = dataset_token_counts(dataset)
        self._cached_packs = None
        # Fail-fast: an all-zero token count means no image's size could be
        # resolved (neither token_count_for_index nor bucket_for_index is
        # usable/all-zero). Without this check, `cur_sum + 0 > budget` would
        # never trigger -> the whole dataset would pack into one
        # ~500k-token sequence -> OOM.
        if not self.token_counts or not any(int(c) > 0 for c in self.token_counts):
            raise RuntimeError(
                "[NavitPack] could not resolve the token count for any sample "
                "(neither token_count_for_index nor bucket_for_index is usable/all-zero). "
                "NaViT packing requires a cached dataset (cache_latents=true) "
                "to obtain each image's latent shape."
            )
        mx = max(self.token_counts) if self.token_counts else 0
        if self.token_counts and self.token_budget < mx:
            logger.warning(
                "[NavitPack] token_budget=%d < largest single-image token count=%d: that image "
                "will get its own pack, possibly exceeding the budget and causing OOM. "
                "Recommend token_budget >= the largest single-image token count.",
                self.token_budget, mx,
            )
        logger.info(
            "[NavitPack] dataset_len=%d token_budget=%d max_images_per_pack=%s "
            "strategy=%s ffd_window=%s (token count range %d..%d)",
            len(self.token_counts), self.token_budget,
            self.max_images_per_pack or "unbounded", self.strategy,
            (self.ffd_window or "global") if self.strategy == "ffd" else "-",
            min(self.token_counts) if self.token_counts else 0, mx,
        )
        if self.strategy == "ffd" and self.ffd_window <= 0:
            logger.warning(
                "[NavitPack] strategy=ffd with ffd_window<=0 (global FFD): the packs will be "
                "exactly the same every epoch (fixed size order), weakening batch diversity for "
                "small datasets. For multi-epoch training, set a positive window."
            )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._cached_packs = None

    def _build_packs(self):
        order = list(range(len(self.token_counts)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        if self.strategy == "ffd":
            packs = pack_indices_ffd_windowed(
                self.token_counts, self.token_budget, order,
                self.max_images_per_pack, self.ffd_window,
            )
        else:
            packs = pack_indices_by_budget(
                self.token_counts, self.token_budget, order, self.max_images_per_pack
            )
        if self.drop_last and len(packs) > 1:
            last_sum = sum(self.token_counts[i] for i in packs[-1])
            if last_sum < self.token_budget:
                packs = packs[:-1]
        return packs

    def __iter__(self):
        packs = self._build_packs()
        self._cached_packs = packs
        for pack in packs:
            yield pack

    def __len__(self):
        if self._cached_packs is None:
            self._cached_packs = self._build_packs()
        return len(self._cached_packs)


def collate_fn_navit_pack(batch):
    """NaViT pack collate.

    The cached latents within one pack have different spatial shapes and
    can't be stacked, so they're kept as a list. The training loop patchifies
    each image into tokens, concatenates the tokens and the per-image RoPE
    grid, encodes the captions, concatenates the corresponding
    ``text_seqlens``, and then calls ``forward_packed_navit``.
    """
    latents = [b["latent"] for b in batch]        # each [C, T, h_i, w_i]
    captions = [b["caption"] for b in batch]
    images = [b.get("image", "") for b in batch]
    result = {
        "navit_latents": latents,
        "captions": captions,
        "images": images,
    }
    # Regularization-set downweighting: aligned with collate_fn_cached --
    # passes loss_weight / is_reg through for the training loop to apply on
    # the per-image loss (the navit path also respects the batch's loss_weight).
    if "loss_weight" in batch[0]:
        result["loss_weight"] = torch.tensor(
            [b["loss_weight"] for b in batch], dtype=torch.float32
        )
        result["is_reg"] = torch.tensor([b["is_reg"] for b in batch], dtype=torch.bool)
    return result
