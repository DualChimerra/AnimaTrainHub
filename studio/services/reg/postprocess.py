"""Regularization set aspect-ratio clustering post-process (PP5.5; fixed 2026-04-28).

**Purpose**: center-crop images with similar aspect ratio (ar) to a shared ar,
so more images land in the same bucket during ARB training -> fewer buckets.
**Only ar is aligned, resolution is never forced to match** - the training
dataloader resizes each image to its bucket's resolution anyway, so the reg
set only needs consistent ar; resolution keeps the original image (no
upscale, no blur).

Adapted from `regex_dataset_builder.py`'s postprocess block into a library;
fixed two bugs in the source script on 2026-04-28:
1. The KMeans features mixed in log(width), causing images with the same ar
   but very different resolutions to land in different clusters; now uses
   only [aspect_ratio]
2. In smart mode, when ar was exactly equal it fell through to the stretch
   formula, computing the resize ratio as if it were crop_ratio; smart now
   always only crops ar and never resizes, and crop_ratio is computed purely
   from the ar difference

Algorithm:
- min_cluster_size = 2 (< 2 -> no clustering, everything goes into cluster 0)
- features: only [aspect_ratio] (z-score normalized)
- KMeans(random_state=42, n_init=10), incrementing k from 1 up to max_k =
  len(images), looking for the first solution where every cluster's max_crop <= max_crop_ratio
- if no K satisfies the constraint -> leave everything unchanged (returns None)
- merge similar clusters: abs aspect diff < 0.02 OR relative diff < 5%, and
  the merge still satisfies the max_crop constraint
- inplace = True always (PP5.5 decision: no backup)

method semantics:
- `smart` (default): only center-crops to target_ar, **keeps the original
  resolution** (recommended; matching ar means the same ARB bucket, without
  blurring small images via upscale)
- `stretch`: directly stretches to target_w x target_h (distorts)
- `crop`: center-crops to target_ar first, then resizes to target_w x target_h

User-facing entry point: `postprocess(reg_dir, *, method='smart',
max_crop_ratio=0.1)`, returns a summary dict. Never raises, whether it fails or can't find a K.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from ...services.dataset.scan import IMAGE_EXTS

ProgressFn = Callable[[str], None]

VALID_METHODS = {"smart", "stretch", "crop"}


# ---------------------------------------------------------------------------
# image collection
# ---------------------------------------------------------------------------


@dataclass
class _ImageInfo:
    path: Path
    width: int
    height: int
    aspect_ratio: float


def _collect_images(reg_dir: Path) -> list[_ImageInfo]:
    """Recursively scan reg_dir for every image, deduped (keeps the first when lowercase filenames collide)."""
    out: list[_ImageInfo] = []
    seen_lower: set[str] = set()
    for f in sorted(reg_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        key = f.name.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        try:
            with Image.open(f) as im:
                w, h = im.size
        except Exception:
            continue
        if not w or not h or h <= 0:
            continue
        out.append(_ImageInfo(path=f, width=w, height=h, aspect_ratio=w / h))
    return out


# ---------------------------------------------------------------------------
# crop ratio
# ---------------------------------------------------------------------------


def calculate_crop_ratio(
    img_w: int, img_h: int, target_w: int, target_h: int, method: str = "smart"
) -> float:
    """Estimated "cost" for each of the three methods (smart / stretch /
    crop), used during clustering to check max_crop.

    - smart: computed purely from the ar difference, crop_ratio = 1 -
      min(orig_ar, target_ar) / max(orig_ar, target_ar). Returns 0 for equal
      ar; doesn't consider absolute resolution (fixes the source script bug).
    - stretch: max(|w_diff|/w, |h_diff|/h), measuring the stretch magnitude.
    - crop: an ar difference similar to smart, but since resize_and_crop
      actually also resizes to target_w x target_h, factoring in the resize
      dimension is reasonable too; kept as the source script's formula.
    """
    if not img_w or not img_h or not target_w or not target_h:
        return 1.0
    if method == "smart":
        orig_ar = img_w / img_h
        target_ar = target_w / target_h
        big = max(orig_ar, target_ar)
        small = min(orig_ar, target_ar)
        return 1.0 - small / big if big > 0 else 0.0
    if method == "stretch":
        wr = abs(img_w - target_w) / max(img_w, 1)
        hr = abs(img_h - target_h) / max(img_h, 1)
        return max(wr, hr)
    if method == "crop":
        original_ar = img_w / img_h
        target_ar = target_w / target_h
        if original_ar > target_ar:
            crop_w = img_h * target_ar
            return (img_w - crop_w) / img_w if img_w > 0 else 0.0
        crop_h = img_w / target_ar
        return (img_h - crop_h) / img_h if img_h > 0 else 0.0
    # default
    wr = abs(img_w - target_w) / max(img_w, 1)
    hr = abs(img_h - target_h) / max(img_h, 1)
    return max(wr, hr)


# ---------------------------------------------------------------------------
# target resolution (median, then adjusted to the target aspect)
# ---------------------------------------------------------------------------


def _determine_target_resolution(cluster: list[_ImageInfo]) -> tuple[int, int]:
    widths = [i.width for i in cluster]
    heights = [i.height for i in cluster]
    return int(np.median(widths)), int(np.median(heights))


def _adjusted_target_for_cluster(
    cluster: list[_ImageInfo],
) -> tuple[int, int, float]:
    """Returns (target_w, target_h, target_ar) - median resolution, adjusted to the median AR."""
    target_ar = float(np.median([i.aspect_ratio for i in cluster]))
    tw_med, th_med = _determine_target_resolution(cluster)
    ar_med = tw_med / th_med if th_med > 0 else 1.0
    if abs(ar_med - target_ar) > 0.01:
        if ar_med > target_ar:
            tw = int(th_med * target_ar)
            th = th_med
        else:
            tw = tw_med
            th = int(tw_med / target_ar)
    else:
        tw, th = tw_med, th_med
    return tw, th, target_ar


def _max_crop_in_cluster(cluster: list[_ImageInfo], method: str) -> tuple[float, int, int]:
    tw, th, _ = _adjusted_target_for_cluster(cluster)
    return (
        max(
            calculate_crop_ratio(i.width, i.height, tw, th, method)
            for i in cluster
        ),
        tw,
        th,
    )


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


def _merge_same_aspect_ratio_clusters(
    clusters: dict[int, list[_ImageInfo]],
    max_crop_ratio: float,
    method: str,
) -> dict[int, list[_ImageInfo]]:
    """Merge clusters with similar aspect ratio (abs < 0.02 OR relative < 5%, and the merge still satisfies the constraint)."""
    if len(clusters) <= 1:
        return clusters
    info: dict[int, dict[str, Any]] = {}
    for cid, imgs in clusters.items():
        info[cid] = {
            "images": imgs,
            "target_ar": float(np.median([i.aspect_ratio for i in imgs])),
        }
    merged: dict[int, list[_ImageInfo]] = {}
    used: set[int] = set()
    new_id = 0
    sorted_ids = sorted(info.items(), key=lambda x: x[1]["target_ar"])

    for cid1, info1 in sorted_ids:
        if cid1 in used:
            continue
        bucket = [info1]
        used.add(cid1)
        target_ar = info1["target_ar"]
        changed = True
        while changed:
            changed = False
            for cid2, info2 in sorted_ids:
                if cid2 in used:
                    continue
                ar2 = info2["target_ar"]
                ar_diff_abs = abs(target_ar - ar2)
                ar_diff_rel = ar_diff_abs / max(target_ar, ar2, 0.001)
                if ar_diff_abs >= 0.02 and ar_diff_rel >= 0.05:
                    continue
                # try merging: compute max_crop for the current bucket + info2 together
                merged_imgs: list[_ImageInfo] = []
                for b in bucket:
                    merged_imgs.extend(b["images"])
                merged_imgs.extend(info2["images"])
                mc, _, _ = _max_crop_in_cluster(merged_imgs, method)
                if mc <= max_crop_ratio:
                    bucket.append(info2)
                    used.add(cid2)
                    target_ar = float(np.median([i.aspect_ratio for i in merged_imgs]))
                    changed = True
                    break
        out_imgs: list[_ImageInfo] = []
        for b in bucket:
            out_imgs.extend(b["images"])
        merged[new_id] = out_imgs
        new_id += 1
    # anything left over (shouldn't happen in theory)
    for cid, imgs in clusters.items():
        if cid not in used:
            merged[new_id] = imgs
            new_id += 1
    return merged


def cluster_by_resolution(
    images: list[_ImageInfo], max_crop_ratio: float, method: str = "smart"
) -> Optional[dict[int, list[_ImageInfo]]]:
    """Increment k from 1, find the first one satisfying max_crop <= limit, then merge.

    Features are only [aspect_ratio] (fixes a source script bug: the
    original [ar, log(width)] would split images with the same ar but very
    different resolution into different clusters, defeating the purpose of ARB bucketing).

    Returns None when no solution satisfies the constraint - the caller should leave files untouched.
    """
    if len(images) < 2:
        return {0: list(images)} if images else None

    features = np.array([[i.aspect_ratio] for i in images], dtype=float)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0, 1, std)
    normalized = (features - mean) / std

    max_k = len(images)
    for k in range(1, max_k + 1):
        if k >= len(images):
            continue
        try:
            if k == 1:
                test_clusters: dict[int, list[_ImageInfo]] = {0: list(images)}
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(normalized)
                test_clusters = defaultdict(list)
                for idx, img in enumerate(images):
                    test_clusters[int(labels[idx])].append(img)
                test_clusters = dict(test_clusters)
        except Exception:
            continue

        all_valid = True
        for imgs in test_clusters.values():
            mc, _, _ = _max_crop_in_cluster(imgs, method)
            if mc > max_crop_ratio:
                all_valid = False
                break
        if all_valid:
            return _merge_same_aspect_ratio_clusters(
                test_clusters, max_crop_ratio, method
            )
    return None


# ---------------------------------------------------------------------------
# resize / crop
# ---------------------------------------------------------------------------


def resize_and_crop_image(
    image_path: Path, target_w: int, target_h: int, output_path: Path, method: str
) -> bool:
    """Actual on-disk behavior for the three methods (smart / stretch / crop). Returns False on failure.

    smart mode only center-crops to target_ar, **keeping the original
    resolution** (no resize, no upscale), since the ARB training dataloader
    resizes per-bucket anyway. stretch / crop keep the source script's
    behavior (forced to exactly target_w x target_h).
    """
    try:
        with Image.open(image_path) as img:
            ow, oh = img.size
            original_ar = ow / oh if oh > 0 else 1.0
            target_ar = target_w / target_h if target_h > 0 else 1.0

            if method == "smart":
                # only center-crop to target_ar; keep as much of the original resolution as possible.
                if abs(original_ar - target_ar) < 1e-6:
                    img.save(output_path, quality=95)
                    return True
                if original_ar > target_ar:
                    crop_w = max(1, int(round(oh * target_ar)))
                    left = (ow - crop_w) // 2
                    cropped = img.crop((left, 0, left + crop_w, oh))
                else:
                    crop_h = max(1, int(round(ow / target_ar)))
                    top = (oh - crop_h) // 2
                    cropped = img.crop((0, top, ow, top + crop_h))
                cropped.save(output_path, quality=95)
            elif method == "stretch":
                resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                resized.save(output_path, quality=95)
            elif method == "crop":
                if original_ar > target_ar:
                    crop_w = int(oh * target_ar)
                    left = (ow - crop_w) // 2
                    cropped = img.crop((left, 0, left + crop_w, oh))
                else:
                    crop_h = int(ow / target_ar)
                    top = (oh - crop_h) // 2
                    cropped = img.crop((0, top, ow, top + crop_h))
                resized = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
                resized.save(output_path, quality=95)
            else:
                return False
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def postprocess(
    reg_dir: Path,
    *,
    method: str = "smart",
    max_crop_ratio: float = 0.1,
    on_progress: ProgressFn = print,
    cancel_event: Optional[threading.Event] = None,
) -> dict[str, Any]:
    """Run resolution-clustering post-process over every image in reg_dir (always inplace).

    Returns:
        {
            "clusters": int | None,        # None = no K satisfying the constraint was found
            "processed": int,              # number of images actually changed (excludes already-matching sizes)
            "skipped": int,                # already matching size / skipped
            "method": str,
            "max_crop_ratio": float,
            "target_resolutions": [(w, h, count), ...],
        }

    Never raises - on failure, clusters=None / processed=0.
    """
    if method not in VALID_METHODS:
        on_progress(f"[postprocess] Invalid method: {method}, skipping")
        return {
            "clusters": None, "processed": 0, "skipped": 0,
            "method": method, "max_crop_ratio": max_crop_ratio,
            "target_resolutions": [],
        }

    if not reg_dir.exists():
        on_progress(f"[postprocess] {reg_dir} does not exist, skipping")
        return {
            "clusters": None, "processed": 0, "skipped": 0,
            "method": method, "max_crop_ratio": max_crop_ratio,
            "target_resolutions": [],
        }

    on_progress(f"[postprocess] Collecting images (method={method}, max_crop={max_crop_ratio})")
    images = _collect_images(reg_dir)
    if not images:
        on_progress("[postprocess] No images, skipping")
        return {
            "clusters": None, "processed": 0, "skipped": 0,
            "method": method, "max_crop_ratio": max_crop_ratio,
            "target_resolutions": [],
        }
    on_progress(f"[postprocess] {len(images)} images total")

    clusters = cluster_by_resolution(images, max_crop_ratio, method)
    if clusters is None:
        on_progress(
            f"[postprocess] No K satisfies max_crop <= {max_crop_ratio}, leaving unchanged"
        )
        return {
            "clusters": None, "processed": 0, "skipped": len(images),
            "method": method, "max_crop_ratio": max_crop_ratio,
            "target_resolutions": [],
        }

    on_progress(f"[postprocess] {len(clusters)} clusters - details:")
    # per-cluster details (matches the source script's log format)
    for cid in sorted(clusters.keys()):
        cluster = clusters[cid]
        tw, th, tar = _adjusted_target_for_cluster(cluster)
        widths = [i.width for i in cluster]
        heights = [i.height for i in cluster]
        ars = [i.aspect_ratio for i in cluster]
        max_crop = max(
            calculate_crop_ratio(i.width, i.height, tw, th, method) for i in cluster
        )
        on_progress(f"  Cluster {cid}: {len(cluster)} images")
        on_progress(f"    Target resolution: {tw}x{th} (aspect ratio: {tar:.3f})")
        on_progress(
            f"    Average resolution: {int(np.mean(widths))}x{int(np.mean(heights))}"
        )
        on_progress(
            f"    Aspect ratio range: {min(ars):.3f} - {max(ars):.3f} "
            f"(average: {np.mean(ars):.3f})"
        )
        on_progress(f"    Max crop ratio: {max_crop * 100:.1f}%")
        on_progress(
            f"    Resolution range: {min(widths)}x{min(heights)} to "
            f"{max(widths)}x{max(heights)}"
        )

    processed = 0
    skipped = 0
    targets: list[tuple[int, int, int]] = []
    for cid in sorted(clusters.keys()):
        if cancel_event and cancel_event.is_set():
            on_progress("[postprocess] [cancel] Cancelled by user")
            break
        cluster = clusters[cid]
        tw, th, target_ar = _adjusted_target_for_cluster(cluster)
        on_progress(
            f"[postprocess] Processing cluster {cid} ({len(cluster)} images) -> "
            f"{'ar=' + format(target_ar, '.3f') if method == 'smart' else f'{tw}x{th}'}"
        )
        targets.append((tw, th, len(cluster)))
        for info in cluster:
            if cancel_event and cancel_event.is_set():
                break
            # smart only aligns ar and keeps the original resolution, so the
            # skip condition is checked against ar; stretch / crop are still
            # checked against the target resolution.
            if method == "smart":
                if abs(info.aspect_ratio - target_ar) < 1e-6:
                    skipped += 1
                    continue
            else:
                if info.width == tw and info.height == th:
                    skipped += 1
                    continue
            ok = resize_and_crop_image(info.path, tw, th, info.path, method)
            if ok:
                processed += 1
            else:
                skipped += 1
                on_progress(f"  x resize failed: {info.path.name}")

    on_progress(
        f"[postprocess] Done: processed={processed}, skipped={skipped}, "
        f"clusters={len(clusters)}"
    )
    return {
        "clusters": len(clusters),
        "processed": processed,
        "skipped": skipped,
        "method": method,
        "max_crop_ratio": max_crop_ratio,
        "target_resolutions": targets,
    }
