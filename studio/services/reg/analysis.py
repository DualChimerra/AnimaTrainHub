"""reg dataset analysis / scoring primitives (extracted from builder.py's 1108 lines in PR-3.9).

Only "looks + computes" -- doesn't touch disk / doesn't write meta / doesn't drive the main loop. builder.py's main flow
(_build_for_subfolder / _build_inner / build) calls into this module to compose the end-to-end "greedy search
+ score-based picking + write to disk" logic.

Public (used by builder.py's main flow):
    analyze_dataset_structure   scans train_dir for tag frequency / resolution / aspect ratio stats
    collect_source_image_ids    scans the source dataset's post_id (avoids clashing images with train)
    collect_existing_reg_per_subfolder  scans existing reg / used for incremental mode
    calculate_tag_similarity    scoring: negative MSE (tag frequency vector distance)
    calculate_resolution_similarity  scoring: aspect 0.6 + resolution 0.4
    calculate_missing_tags      computes the priority queue of "which tags are still missing"
    check_aspect_ratio          whether a single image's aspect ratio is within the filter's allowed range
    find_best_match             picks the highest score (tag + resolution combined) among a batch of posts

Semi-public (used by builder.py + tests):
    _normalize_tags             lowercase + space->_ + dedup while preserving order
    analyze_tags_in_file        reads a single image's caption + normalizes
    _search_with_filters        search_posts + local blacklist / id exclusion filtering
    _IMAGE_EXT_NODOT            datasets.IMAGE_EXTS with the dot stripped (for comparison against booru file_ext)

All thresholds / formulas match the source script regex_dataset_builder.py:
- tag similarity sigmoid coefficient 0.1; resolution scoring aspect 0.6 + resolution 0.4
- final score = tag_score + resolution_score * 0.1 (resolution acts as a tie-breaker)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import requests
from PIL import Image

from ...services.dataset.scan import IMAGE_EXTS
from ..booru import api as booru_api, pool as booru_pool
from ..dataset import tagedit


ProgressFn = Callable[[str], None]

# IMAGE_EXTS uses the ".xxx" form in datasets.py; here we need the dotless form (for comparison against file_ext)
_IMAGE_EXT_NODOT = {e.lstrip(".") for e in IMAGE_EXTS}


# ---------------------------------------------------------------------------
# tag analysis
# ---------------------------------------------------------------------------


def _normalize_tags(raw: list[str]) -> list[str]:
    """Lowercase + space->underscore + dedup while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        n = t.lower().strip().replace(" ", "_")
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def analyze_tags_in_file(image_path: Path) -> list[str]:
    """Reads an image's corresponding caption (.txt or .json), returns the normalized tag list.

    Reuses `tagedit.read_tags`, then normalizes case / spaces / dedup.
    """
    raw = tagedit.read_tags(image_path)
    return _normalize_tags(raw)


def analyze_dataset_structure(
    dataset_path: Path, on_progress: ProgressFn = print
) -> dict[str, Any]:
    """Scans sub-folders + the root directory, computing tag frequency / resolution / aspect ratio stats.

    The returned structure matches the source script:
    ```
    {
        "subfolders": {name: {"images": [...], "tag_freq": Counter, "image_count": int}},
        "total_images": int,
        "global_tag_freq": Counter,
        "global_tag_weights": {tag: count/total_images},
        "resolutions": [(w, h), ...],
        "aspect_ratios": [...],
        "median_resolution": (w, h) | None,
        "median_aspect_ratio": float | None,
        "resolution_std": (sw, sh) | None,
    }
    ```
    """
    structure: dict[str, Any] = {
        "subfolders": {},
        "total_images": 0,
        "global_tag_freq": Counter(),
        "global_tag_weights": {},
        "resolutions": [],
        "aspect_ratios": [],
    }

    def _scan_folder(folder: Path, key: str) -> None:
        data = {"images": [], "tag_freq": Counter(), "image_count": 0}
        for img in sorted(folder.iterdir()):
            if not img.is_file():
                continue
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            tags = analyze_tags_in_file(img)
            if not tags:
                continue
            w, h, ar = None, None, None
            try:
                with Image.open(img) as im:
                    w, h = im.size
                    if h > 0:
                        ar = w / h
            except Exception as exc:
                on_progress(f"    Warning: could not read image size {img.name}: {exc}")
            data["images"].append({
                "image": img.name,
                "tags": tags,
                "width": w,
                "height": h,
                "aspect_ratio": ar,
            })
            data["tag_freq"].update(tags)
            structure["global_tag_freq"].update(tags)
            data["image_count"] += 1
            structure["total_images"] += 1
            if w and h:
                structure["resolutions"].append((w, h))
                if ar:
                    structure["aspect_ratios"].append(ar)
        if data["image_count"] > 0:
            structure["subfolders"][key] = data
            on_progress(
                f"  [{key or '<root>'}] {data['image_count']} images, "
                f"{len(data['tag_freq'])} distinct tags"
            )

    # Images directly at the root
    has_root_imgs = any(
        f.is_file() and f.suffix.lower() in IMAGE_EXTS
        for f in dataset_path.iterdir()
    )
    if has_root_imgs:
        _scan_folder(dataset_path, "")

    # Sub-folders
    for sub in sorted(dataset_path.iterdir()):
        if sub.is_dir():
            _scan_folder(sub, sub.name)

    # Global weights
    if structure["total_images"] > 0:
        for tag, count in structure["global_tag_freq"].items():
            structure["global_tag_weights"][tag] = (
                count / structure["total_images"]
            )

    # Resolution stats (median + std dev)
    if structure["resolutions"]:
        res_arr = np.array(structure["resolutions"])
        structure["median_resolution"] = (
            int(np.median(res_arr[:, 0])),
            int(np.median(res_arr[:, 1])),
        )
        structure["resolution_std"] = (
            float(np.std(res_arr[:, 0])),
            float(np.std(res_arr[:, 1])),
        )
        ar_arr = np.array(structure["aspect_ratios"])
        structure["median_aspect_ratio"] = float(np.median(ar_arr))
    else:
        structure["median_resolution"] = None
        structure["resolution_std"] = None
        structure["median_aspect_ratio"] = None

    return structure


def collect_source_image_ids(source_path: Path) -> set[str]:
    """Recursively collects the file stem (= post_id) for every image in the source dataset.

    Source script convention: booru-downloaded filenames are `{post_id}.{ext}`, so the stem is the ID.
    """
    ids: set[str] = set()
    for img in source_path.rglob("*"):
        if not img.is_file():
            continue
        if img.suffix.lower().lstrip(".") not in _IMAGE_EXT_NODOT:
            continue
        ids.add(img.stem)
    return ids


def collect_existing_reg_per_subfolder(
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    """PP5.1 -- scans existing reg images, aggregating (ids, tags, count) per sub-folder.

    Returns {subfolder_name: {"ids": set[str], "tags": list[list[str]], "count": int}}
    subfolder_name == "" means the root of output_dir.
    """
    out: dict[str, dict[str, Any]] = {}
    if not output_dir.exists():
        return out

    def _ensure(key: str) -> dict[str, Any]:
        if key not in out:
            out[key] = {"ids": set(), "tags": [], "count": 0}
        return out[key]

    for img in output_dir.rglob("*"):
        if not img.is_file():
            continue
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img.relative_to(output_dir)
        # sub-folder = first path segment (if present)
        sub_key = rel.parts[0] if len(rel.parts) > 1 else ""
        bucket = _ensure(sub_key)
        bucket["ids"].add(img.stem)
        bucket["tags"].append(analyze_tags_in_file(img))
        bucket["count"] += 1
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def calculate_tag_similarity(
    target_weights: dict[str, float],
    candidate_tags: list[str],
    current_weights: dict[str, float],
    target_count: int,
) -> float:
    """Matches the source script: negative MSE.

    Both `target_weights` and `current_weights` are measured as **document frequency** (doc frequency):
    how many images a given tag appears in / total image count, in [0, 1]. `target_weights` comes from analyzing
    train captions; `current_weights` comes from images already downloaded into reg, using the same formula as
    `target_weights` (each image accumulates `1/target_count` per tag) -- once enough images have been pulled to reach
    `target_count`, K images containing a given tag will accumulate `current_weights[tag]` to `K / target_count`,
    exactly the doc frequency of that tag within the reg set, symmetric in magnitude with `target_weights`.

    This is **not** reverse-IDF: it's measured at the same magnitude as target, so accumulation has no asymmetric bias.
    Side effect: a densely-tagged image (e.g. a 50-tag booru post) contributes more MSE terms by itself, so candidate
    selection is slightly biased toward images with more tags -- but this matches train's doc frequency distribution, so it
    doesn't create a mismatch (target is also computed by doc frequency).

    If changed to "normalized distribution matching" (candidate side using `1/(target_count * len(post_tags))`),
    `target_weights` would also have to be recomputed the same way -- otherwise it would create an asymmetric IDF bias.
    This PR doesn't touch the contract, just documents it (see `tmp/reg_algorithm_research.md` Smell D
    + owner discussion 2026-05-30).
    """
    new_weights = dict(current_weights)
    for tag in candidate_tags:
        new_weights[tag] = new_weights.get(tag, 0) + (1 / target_count)
    score = 0.0
    all_tags = set(target_weights.keys()) | set(candidate_tags)
    for tag in all_tags:
        score += (target_weights.get(tag, 0) - new_weights.get(tag, 0)) ** 2
    return -score


def calculate_resolution_similarity(
    post_w: int,
    post_h: int,
    target_resolution: tuple[int, int],
    target_aspect_ratio: float,
    resolution_std: Optional[tuple[float, float]] = None,
) -> float:
    """Matches the source script: aspect ratio 0.6 + resolution 0.4."""
    if not post_w or not post_h or not target_resolution or not target_aspect_ratio:
        return 0.0
    post_ar = post_w / post_h if post_h > 0 else 1.0
    aspect_diff = abs(post_ar - target_aspect_ratio) / max(target_aspect_ratio, 0.001)
    aspect_score = 1.0 / (1.0 + aspect_diff * 10)

    tw, th = target_resolution
    if resolution_std and resolution_std[0] > 0 and resolution_std[1] > 0:
        width_score = 1.0 / (1.0 + abs(post_w - tw) / (resolution_std[0] * 2))
        height_score = 1.0 / (1.0 + abs(post_h - th) / (resolution_std[1] * 2))
    else:
        width_score = 1.0 / (1.0 + abs(post_w - tw) / max(tw, 1) * 10)
        height_score = 1.0 / (1.0 + abs(post_h - th) / max(th, 1) * 10)

    resolution_score = (width_score + height_score) / 2
    return aspect_score * 0.6 + resolution_score * 0.4


def calculate_missing_tags(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    blacklist_tags: set[str],
    failed_tags: set[str],
) -> list[tuple[str, float]]:
    missing: list[tuple[str, float]] = []
    for tag, tw in target_weights.items():
        if tag in blacklist_tags or tag in failed_tags:
            continue
        diff = tw - current_weights.get(tag, 0.0)
        if diff > 0:
            missing.append((tag, diff))
    missing.sort(key=lambda x: x[1], reverse=True)
    return missing


def check_aspect_ratio(
    w: Optional[int],
    h: Optional[int],
    *,
    enabled: bool,
    min_ar: float,
    max_ar: float,
) -> bool:
    if not enabled:
        return True
    if not w or not h or h == 0:
        return False
    ar = w / h
    return min_ar <= ar <= max_ar


# PR-3 (Smell C fix): normalize tag_score and res_score to the same magnitude before weighting.
# - tag_score = -MSE, magnitude ~ [-len(target_weights), 0]
# - res_score ∈ [0, 1]
# The old formula `tag_score + res_score * 0.1` let tag dominate but gave res an unjustified
# 0.1 coefficient -- actual behavior: a tag difference of 0.001 could already be flipped by res, inconsistent with the
# docstring's "tie-breaker". The new formula normalizes tag_score / len(target_weights) to
# ~[-1, 0], the same magnitude as res_score, weighted 0.7:0.3. 0.7 keeps tag dominant,
# 0.3 lets res actually influence ranking (the owner cares about ARB bucket consistency).
# See `tmp/reg_algorithm_research.md` Smell C + owner's caption-disentanglement
# discussion decision (2026-05-30).
TAG_SCORE_WEIGHT = 0.7
RES_SCORE_WEIGHT = 0.3


def find_best_match(
    posts: list[dict[str, Any]],
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    target_count: int,
    *,
    api_source: str,
    skip_similar: bool,
    target_resolution: Optional[tuple[int, int]] = None,
    target_aspect_ratio: Optional[float] = None,
    resolution_std: Optional[tuple[float, float]] = None,
    source_image_ids: Optional[set[str]] = None,
    aspect_ratio_filter_enabled: bool = False,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> tuple[Optional[dict[str, Any]], float]:
    if source_image_ids is None:
        source_image_ids = set()

    candidates = posts[::2] if skip_similar else posts
    best_post = None
    best_score = float("-inf")

    # tag_score normalization denominator: the size of the tag set participating in MSE is approximately
    # target_weights's size (candidate_tags contributes only a small non-overlapping part). Dividing by it normalizes
    # tag_score to ~[-1, 0], to then weight against res_score's [0, 1].
    tag_denom = max(1, len(target_weights))

    for post in candidates:
        post_id, _, _, _ = booru_api.post_fields(post, api_source)
        if post_id and post_id in source_image_ids:
            continue
        pw, ph = booru_api.post_dimensions(post, api_source)
        if not check_aspect_ratio(
            pw, ph,
            enabled=aspect_ratio_filter_enabled,
            min_ar=min_aspect_ratio,
            max_ar=max_aspect_ratio,
        ):
            continue
        post_tags = booru_api.post_tag_list(post, api_source)
        tag_score = calculate_tag_similarity(
            target_weights, post_tags, current_weights, target_count
        )
        tag_norm = tag_score / tag_denom
        res_score = 0.0
        if target_resolution and target_aspect_ratio and pw and ph:
            res_score = calculate_resolution_similarity(
                pw, ph, target_resolution, target_aspect_ratio, resolution_std
            )
        final_score = TAG_SCORE_WEIGHT * tag_norm + RES_SCORE_WEIGHT * res_score
        if final_score > best_score:
            best_score = final_score
            best_post = post
    return best_post, best_score


# ---------------------------------------------------------------------------
# search wrapper (with local filtering)
# ---------------------------------------------------------------------------


def _search_with_filters(
    tags: list[str],
    *,
    api_source: str,
    user_id: str,
    api_key: str,
    username: str,
    blacklist_tags: set[str],
    exclude_ids: set[str],
    page: int = 1,
    limit: int = 100,
    client: Optional[booru_pool.BooruClient] = None,
) -> list[dict[str, Any]]:
    """Search + local filtering (blacklist / already-excluded IDs / missing id or url).

    PP9: `client` goes through the unified pool (API token bucket); if not passed, calls the underlying layer directly (compat with old tests).
    """
    norm = _normalize_tags(tags)
    query = " ".join(norm)
    try:
        if client is not None:
            posts = client.search_posts(
                api_source,
                query,
                page=page,
                limit=limit,
                user_id=user_id,
                api_key=api_key,
                username=username,
            )
        else:
            posts = booru_api.search_posts(
                api_source,
                query,
                page=page,
                limit=limit,
                user_id=user_id,
                api_key=api_key,
                username=username,
            )
    except requests.RequestException:
        return []

    out: list[dict[str, Any]] = []
    for post in posts:
        pid, file_url, _, _ = booru_api.post_fields(post, api_source)
        if not pid or not file_url:
            continue
        if pid in exclude_ids:
            continue
        # Local blacklist filtering
        if blacklist_tags:
            ptags = booru_api.post_tag_list(post, api_source)
            if any(t in blacklist_tags for t in ptags):
                continue
        out.append(post)
    return out
