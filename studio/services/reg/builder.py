"""Regularization dataset builder (PP5).

Adapted from the standalone `C:/Users/Mei/Desktop/SD/danbooru/dev/regex_dataset_builder.py`
script into a library: dropped input() / JSON config file, all parameters go
through `RegBuildOptions`; progress is pushed back to the caller via
`on_progress(line)` (the worker forwards it to the log + bus.publish).

**Logic must match the source script exactly** - every threshold / constant /
decision is carried over as-is:
- Descending tag-count sequence: 10 -> 5 -> 3 -> 2 -> 1 (capped at max_search_tags)
- Up to 3 different offsets tried per tag count
- failed_tags: once a single-tag search fails, it's never retried
- invalid_tag_combinations: a search found results but nothing from this batch got downloaded (already in the source dataset / didn't qualify)
- max_rounds = 50; max_consecutive_failures = 5
- find_best_match's skip_similar takes even indices (`posts[::2]`)
- tag similarity uses a sigmoid with coefficient 0.1; resolution score is aspect 0.6 + resolution 0.4
- final score = tag_score + resolution_score * 0.1 (resolution acts as a tie-breaker)
- 80% completion rate counts as success
- 0.5s after each image, 1s after each batch

Out of scope (-> PP5.5): resolution K-means clustering post-process, cropping to a uniform per-cluster resolution.

Since PR-3.9: pure analysis / scoring / search-filter functions moved to
`analysis.py`; this file keeps the main flow (_build_for_subfolder /
_build_inner / build) + RegBuildOptions / RegMeta / meta CRUD. analysis's 11
public + semi-public names are re-exported into this module's namespace via
the `from .analysis import ...` below, to keep the old
`from studio.services.reg.builder import X` / `reg_builder.X` import path
working (tests make heavy use of attribute access).
"""
from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ...services.dataset.scan import IMAGE_EXTS
from ..booru import api as booru_api, pool as booru_pool
from .analysis import (
    _IMAGE_EXT_NODOT,
    _normalize_tags,
    _search_with_filters,
    analyze_dataset_structure,
    analyze_tags_in_file,
    calculate_missing_tags,
    calculate_resolution_similarity,
    calculate_tag_similarity,
    check_aspect_ratio,
    collect_existing_reg_per_subfolder,
    collect_source_image_ids,
    find_best_match,
)


ProgressFn = Callable[[str], None]

VIDEO_EXTS = {
    "mp4", "webm", "avi", "mov", "mkv", "flv", "wmv", "mpg", "mpeg", "m4v",
}


# ---------------------------------------------------------------------------
# options & meta
# ---------------------------------------------------------------------------


@dataclass
class RegBuildOptions:
    """All parameters for building a regularization set.

    `target_count=None` -> use train's total image count (matches the source script's default).
    """

    train_dir: Path
    output_dir: Path

    # API credentials
    api_source: str = "gelbooru"
    user_id: str = ""
    api_key: str = ""
    username: str = ""

    # limits
    target_count: Optional[int] = None
    max_search_tags: int = 20  # gelbooru default 20; danbooru free 2 / gold 6 / platinum 12
    # batch_size = the step inside the search loop for "recompute missing_weight every N downloads"; unrelated to the train subfolder mirroring
    batch_size: int = 5

    # tags
    excluded_tags: list[str] = field(default_factory=list)  # project-specific (character names etc.)
    blacklist_tags: list[str] = field(default_factory=list)  # global blacklist

    # image selection strategy
    skip_similar: bool = True
    aspect_ratio_filter_enabled: bool = False
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0

    # output to disk
    save_tags: bool = False  # PP5 default False (auto_tag goes through WD14)
    convert_to_png: bool = True
    remove_alpha_channel: bool = False

    # post-processing
    auto_tag: bool = True  # whether to run the tagger after pulling reg images
    auto_tag_kind: str = "wd14"  # A3 - tagger kind; constrained to VALID_TAGGER_NAMES, UI currently exposes wd14/cltagger
    auto_dedup: bool = True  # A4 - auto dedup after build + top-up loop if short
    # B1 (PR-2): build mode.
    # - "mirror": the old source-script behavior, mirroring train's subfolders, each pulled independently to its own image count.
    # - "flat": all images go into a single `1_data/` bucket; target_count decides the total.
    build_mode: str = "flat"
    based_on_version: str = ""  # meta only, doesn't affect logic


@dataclass
class RegMeta:
    generated_at: float
    based_on_version: str
    api_source: str
    target_count: int
    actual_count: int
    source_tags: list[str]            # search tags actually used (deduped)
    excluded_tags: list[str]
    blacklist_tags: list[str]
    failed_tags: list[str]            # tags whose search failed
    train_tag_distribution: dict[str, int]  # train tag frequency (top 50)
    auto_tagged: bool
    # A3 - name of the tagger auto_tag actually ran ("wd14" / "cltagger" / ...).
    # None = never ran, or meta from before it ran. Old meta (missing this
    # field) is read as None. auto_tagged=True with auto_tag_kind=None is
    # treated as "unknown tagger" (data from an older version).
    auto_tag_kind: Optional[str] = None
    # B1 (PR-2): which mode this reg set was generated with (mirror / flat).
    # Old meta (missing this field) = mirror (the default before PR-1). The
    # frontend uses this to infer the current structure when switching modes,
    # and blocks the switch with a prompt to clear first if inconsistent.
    build_mode: str = "mirror"
    incremental_runs: int = 0         # how many top-up runs have happened (PP5.1)
    # PP5.5 - post-process summary (postprocessed_at=None means it never ran or failed)
    postprocessed_at: Optional[float] = None
    postprocess_clusters: Optional[int] = None
    postprocess_method: Optional[str] = None
    postprocess_max_crop_ratio: Optional[float] = None
    # How this set was generated: "scrape" = pulled from a booru (default,
    # compatible with old meta), "ai_base" = the base model generated
    # counter-examples from train's tags as a regularization set (prior-based
    # generation). This field exists so the api_source field doesn't get
    # polluted by a fake source like "ai_generated": when
    # generation_method="scrape", api_source is "gelbooru"|"danbooru"; when
    # generation_method="ai_base", api_source stays empty (there's no
    # meaningful source).
    generation_method: str = "scrape"


# ---------------------------------------------------------------------------
# main loops
# ---------------------------------------------------------------------------


def _build_for_subfolder(
    subfolder_name: str,
    subfolder_data: dict[str, Any],
    target_weights: dict[str, float],
    output_dir: Path,
    *,
    opts: RegBuildOptions,
    blacklist_tags: set[str],
    failed_tags: set[str],
    source_tags_used: set[str],
    source_image_ids: set[str],
    target_resolution: Optional[tuple[int, int]],
    target_aspect_ratio: Optional[float],
    resolution_std: Optional[tuple[float, float]],
    total_target_count: int,
    total_downloaded_so_far: int,
    on_progress: ProgressFn,
    cancel_event: Optional[threading.Event],
    pre_existing: Optional[dict[str, Any]] = None,  # PP5.1
    client: Optional[booru_pool.BooruClient] = None,  # PP9
    deleted_ids: Optional[set[str]] = None,  # A2 - booru IDs the user deleted from the UI
) -> tuple[bool, int]:
    """Batch loop for a single subfolder. Returns (success_reached_80pct, actual_downloaded_count)."""
    label = subfolder_name or "<root>"
    on_progress(f"\n===== Subfolder {label} =====")

    target_count = subfolder_data["image_count"]
    remaining_quota = total_target_count - total_downloaded_so_far
    if remaining_quota <= 0:
        on_progress(f"  Reached the total limit of {total_target_count}, skipping {label}")
        return False, 0
    target_count = min(target_count, remaining_quota)
    on_progress(f"  Target {target_count} images, batch size {opts.batch_size}, up to {opts.max_search_tags} tags")

    if subfolder_name == "":
        out_sub = output_dir
    else:
        out_sub = output_dir / subfolder_name
    out_sub.mkdir(parents=True, exist_ok=True)

    current_weights: dict[str, float] = defaultdict(float)
    downloaded_count = 0
    downloaded_ids: set[str] = set()
    skipped = 0
    failed = 0

    # A2 - merge booru IDs the user deleted from the UI into downloaded_ids, so
    # the search stage's `exclude_ids=downloaded_ids` automatically excludes
    # them, preventing an incremental top-up from pulling them back in. Not
    # counted in downloaded_count since they're no longer on disk.
    if deleted_ids:
        downloaded_ids.update(deleted_ids)
        on_progress(
            f"  [a2] Excluding {len(deleted_ids)} previously deleted booru IDs (from reg/.deleted_ids.json)"
        )

    # PP5.1 - incremental: count existing images as "already downloaded" for
    # the starting point + accumulate current_weights from them
    if pre_existing and pre_existing.get("count"):
        existing_count = int(pre_existing["count"])
        downloaded_count = min(existing_count, target_count)
        for pid in pre_existing.get("ids") or set():
            downloaded_ids.add(str(pid))
        for tags in pre_existing.get("tags") or []:
            for t in tags:
                current_weights[t] += 1 / target_count
        on_progress(
            f"  [incremental] Reusing {existing_count} existing images (starting point {downloaded_count}/{target_count})"
        )
        if downloaded_count >= target_count:
            on_progress("  [incremental] Existing images already meet the target, nothing to top up")
            return True, downloaded_count

    batch_round = 0
    max_rounds = 50
    consecutive_failures = 0
    max_consecutive_failures = 5
    invalid_tag_combinations: set[tuple[str, ...]] = set()

    while downloaded_count < target_count and batch_round < max_rounds:
        if cancel_event and cancel_event.is_set():
            on_progress("  [cancel] Cancelled by user")
            return False, downloaded_count

        batch_round += 1
        batch_remaining = min(opts.batch_size, target_count - downloaded_count)
        on_progress(f"\n  ----- batch {batch_round} ({downloaded_count}/{target_count}) -----")

        missing_tags = calculate_missing_tags(
            target_weights, current_weights, blacklist_tags, failed_tags
        )
        if not missing_tags:
            on_progress("  All tags have reached their target weight")
            break

        available_tags = [t for t, _ in missing_tags if t not in failed_tags]
        if not available_tags:
            on_progress(f"  All missing tags failed to search: {list(failed_tags)}")
            break

        info_preview = ", ".join(
            f"{t}(missing {w:.2f})" for t, w in missing_tags[:5]
        )
        on_progress(f"  Most missing: {info_preview}")

        # descending tag counts: 10 -> 5 -> 3 -> 2 -> 1
        tag_counts_seq = [10, 5, 3, 2, 1]
        tag_counts_seq = [min(tc, opts.max_search_tags) for tc in tag_counts_seq]
        tag_counts_seq = list(dict.fromkeys(tag_counts_seq))  # dedupe, keep order

        posts: list[dict[str, Any]] = []
        search_tags: list[str] = []
        tried_combinations: set[tuple[str, ...]] = set()
        all_tags_failed = False

        for tag_count in tag_counts_seq:
            if len(available_tags) < tag_count:
                continue
            max_attempts = min(3, len(available_tags) - tag_count + 1)
            for offset in range(max_attempts):
                if offset + tag_count > len(available_tags):
                    break
                cand_tags = available_tags[offset:offset + tag_count]
                comb_key = tuple(sorted(cand_tags))
                if comb_key in invalid_tag_combinations:
                    if offset == 0:
                        on_progress(f"    Skipping invalid combination: {cand_tags}")
                    continue
                if comb_key in tried_combinations:
                    continue
                tried_combinations.add(comb_key)

                on_progress(f"    Searching with {tag_count} tags: {cand_tags}")
                posts = _search_with_filters(
                    cand_tags,
                    api_source=opts.api_source,
                    user_id=opts.user_id,
                    api_key=opts.api_key,
                    username=opts.username,
                    blacklist_tags=blacklist_tags,
                    exclude_ids=downloaded_ids,
                    page=1,
                    limit=100,
                    client=client,
                )
                if posts:
                    search_tags = cand_tags
                    for t in cand_tags:
                        source_tags_used.add(t)
                    break
                if tag_count == 1:
                    failed_tags.add(cand_tags[0])
                    on_progress(f"    x tag '{cand_tags[0]}' search failed, adding to the skip list")
                    remaining_avail = [
                        t for t, _ in missing_tags if t not in failed_tags
                    ]
                    if not remaining_avail:
                        on_progress(f"  All missing tags have now failed: {list(failed_tags)}")
                        all_tags_failed = True
                        break
            if posts or all_tags_failed:
                break

        if not posts:
            consecutive_failures += 1
            on_progress(
                f"  No matches (consecutive failures {consecutive_failures}/{max_consecutive_failures})"
            )
            # check: if every possible combination has been marked invalid, exit
            all_invalid = True
            for tc in tag_counts_seq:
                if len(available_tags) < tc:
                    continue
                tk = tuple(sorted(available_tags[:tc]))
                if tk not in invalid_tag_combinations:
                    all_invalid = False
                    break
            if all_invalid and invalid_tag_combinations:
                on_progress("  Every combination is marked invalid, stopping search")
                break
            if consecutive_failures >= max_consecutive_failures:
                on_progress(f"  Stopping after {consecutive_failures} consecutive failures")
                break
            continue
        consecutive_failures = 0
        on_progress(f"    {len(posts)} candidates")

        # download this batch from the candidates
        batch_downloaded = 0
        attempts = 0
        max_attempts = len(posts)

        while batch_downloaded < batch_remaining and attempts < max_attempts:
            if cancel_event and cancel_event.is_set():
                on_progress("  [cancel] Cancelled by user")
                return False, downloaded_count

            attempts += 1
            best_post, score = find_best_match(
                posts,
                target_weights,
                current_weights,
                target_count,
                api_source=opts.api_source,
                skip_similar=opts.skip_similar,
                target_resolution=target_resolution,
                target_aspect_ratio=target_aspect_ratio,
                resolution_std=resolution_std,
                source_image_ids=source_image_ids,
                aspect_ratio_filter_enabled=opts.aspect_ratio_filter_enabled,
                min_aspect_ratio=opts.min_aspect_ratio,
                max_aspect_ratio=opts.max_aspect_ratio,
            )
            if not best_post:
                break
            posts.remove(best_post)

            pid, file_url, file_ext, _ = booru_api.post_fields(
                best_post, opts.api_source
            )
            if not pid or not file_url:
                skipped += 1
                continue
            if pid in source_image_ids:
                on_progress(f"    Skipping (already in source): {pid}")
                skipped += 1
                downloaded_ids.add(pid)
                continue
            ext_lower = (file_ext or "").lower()
            if ext_lower in VIDEO_EXTS:
                on_progress(f"    Skipping (video): {pid} .{file_ext}")
                skipped += 1
                downloaded_ids.add(pid)
                continue
            if ext_lower not in _IMAGE_EXT_NODOT:
                on_progress(f"    Skipping (not an image): {pid} .{file_ext}")
                skipped += 1
                downloaded_ids.add(pid)
                continue
            pw, ph = booru_api.post_dimensions(best_post, opts.api_source)
            if not check_aspect_ratio(
                pw, ph,
                enabled=opts.aspect_ratio_filter_enabled,
                min_ar=opts.min_aspect_ratio,
                max_ar=opts.max_aspect_ratio,
            ):
                ar_v = pw / ph if pw and ph else 0
                on_progress(f"    Skipping (aspect ratio {ar_v:.2f}): {pid}")
                skipped += 1
                downloaded_ids.add(pid)
                continue

            ext = "png" if opts.convert_to_png else (file_ext or "jpg")
            image_path = out_sub / f"{pid}.{ext}"
            txt_path = out_sub / f"{pid}.txt"

            if image_path.exists():
                try:
                    image_path.unlink()
                except Exception as exc:
                    on_progress(f"    Warning: could not delete {image_path.name}: {exc}")

            try:
                if client is not None:
                    final = client.download_image(
                        file_url,
                        image_path,
                        convert_to_png=opts.convert_to_png,
                        remove_alpha_channel=opts.remove_alpha_channel,
                        referer=booru_api.default_base_url(opts.api_source) + "/",
                        username=opts.username,
                    )
                else:
                    final = booru_api.download_image(
                        file_url,
                        image_path,
                        convert_to_png=opts.convert_to_png,
                        remove_alpha_channel=opts.remove_alpha_channel,
                        referer=booru_api.default_base_url(opts.api_source) + "/",
                        username=opts.username,
                    )
            except Exception as exc:
                on_progress(f"    x Download failed: {pid} ({exc})")
                if image_path.exists():
                    try:
                        image_path.unlink()
                    except Exception:
                        pass
                failed += 1
                downloaded_ids.add(pid)
                continue

            post_tags = booru_api.post_tag_list(best_post, opts.api_source)
            if opts.save_tags and post_tags:
                # captions always use space form (consistent with WD14/CLTagger
                # output and the training set). The underscore form is just
                # booru's wire format, only used for matching / querying -
                # post_tags below still accumulates into current_weights with
                # underscores untouched. Tag-form convention: user-visible /
                # caption = spaces; underscores only cross the booru boundary.
                txt_path.write_text(
                    ", ".join(t.replace("_", " ") for t in post_tags),
                    encoding="utf-8",
                )

            for tag in post_tags:
                current_weights[tag] += 1 / target_count

            downloaded_count += 1
            batch_downloaded += 1
            downloaded_ids.add(pid)
            matched = [t for t in post_tags if t in target_weights][:5]
            on_progress(
                f"    [{downloaded_count}/{target_count}] OK {pid} "
                f"score={score:.4f} matched={matched}"
            )

            if (
                total_target_count is not None
                and total_downloaded_so_far + downloaded_count >= total_target_count
            ):
                on_progress(f"  Reached the total limit of {total_target_count}")
                break

            # PP9 - removed the hard 0.5s sleep per image; rate is controlled by BooruClient's token bucket
            if cancel_event and cancel_event.is_set():
                on_progress("  [cancel] Cancelled by user")
                return False, downloaded_count

        on_progress(f"  Downloaded this batch: {batch_downloaded}")

        if batch_downloaded == 0 and posts and search_tags:
            invalid_tag_combinations.add(tuple(sorted(search_tags)))
            on_progress(f"  Combination {search_tags} found candidates but downloaded none, marking invalid")
            continue
        elif batch_downloaded > 0 and search_tags:
            invalid_tag_combinations.discard(tuple(sorted(search_tags)))

        if downloaded_count < target_count:
            if cancel_event:
                if cancel_event.wait(1.0):
                    on_progress("  [cancel] Cancelled by user")
                    return False, downloaded_count
            else:
                time.sleep(1.0)

    on_progress(
        f"\n  Subfolder {label} done: {downloaded_count}/{target_count} "
        f"(skipped={skipped} failed={failed})"
    )
    success = downloaded_count >= target_count * 0.8
    return success, downloaded_count


def build(
    opts: RegBuildOptions,
    *,
    on_progress: ProgressFn = print,
    cancel_event: Optional[threading.Event] = None,
    incremental: bool = False,
    client: Optional[booru_pool.BooruClient] = None,
) -> RegMeta:
    """Main flow for building a regularization set. Returns a RegMeta (returns partial metadata even if cancelled mid-way).

    Source script logic:
    1. analyze_dataset_structure(train_dir)
    2. collect_source_image_ids (avoid colliding with train)
    3. auto-blacklist: add based_on_version as a tag to a temporary blacklist (avoid pulling the same artist's fan art)
    4. allocate the target count across subfolders proportionally, looping _build_for_subfolder
    5. write meta.json

    PP5.1: when `incremental=True`, existing images under output_dir are kept
    as the "already downloaded" starting point, `current_weights` is
    accumulated from their existing captions, and only the gap is topped up;
    the old meta's `incremental_runs + 1` is written back.
    """
    on_progress(f"[reg] api={opts.api_source} train={opts.train_dir}")

    if not opts.train_dir.exists():
        raise FileNotFoundError(f"The train directory does not exist: {opts.train_dir}")
    if opts.api_source == "gelbooru" and not (opts.user_id and opts.api_key):
        raise ValueError("gelbooru needs user_id + api_key (set secrets.gelbooru in Settings)")
    if opts.api_source == "danbooru" and not (opts.username and opts.api_key):
        raise ValueError("danbooru needs username + api_key (set secrets.danbooru in Settings)")

    # PP9 - build a client if none was passed (rate-controlled by secrets.download.*), close it when done
    owns_client = False
    if client is None:
        try:
            from .. import secrets as _secrets
            d = _secrets.load().download
            cfg = booru_pool.BooruPoolConfig(
                parallel_workers=d.parallel_workers,
                api_rate_per_sec=d.api_rate_per_sec,
                cdn_rate_per_sec=d.cdn_rate_per_sec,
            )
        except Exception:  # noqa: BLE001
            cfg = booru_pool.BooruPoolConfig()
        client = booru_pool.BooruClient(cfg)
        owns_client = True

    try:
        return _build_inner(
            opts,
            client=client,
            on_progress=on_progress,
            cancel_event=cancel_event,
            incremental=incremental,
        )
    finally:
        if owns_client:
            client.close()


def _build_inner(
    opts: RegBuildOptions,
    *,
    client: booru_pool.BooruClient,
    on_progress: ProgressFn,
    cancel_event: Optional[threading.Event],
    incremental: bool,
) -> RegMeta:
    structure = analyze_dataset_structure(opts.train_dir, on_progress)
    if structure["total_images"] == 0:
        raise ValueError(f"The train directory has no captioned images: {opts.train_dir}")

    source_image_ids = collect_source_image_ids(opts.train_dir)
    on_progress(f"[reg] {len(source_image_ids)} source image IDs collected, will be avoided")

    # tag sets
    blacklist_tags = set(_normalize_tags(opts.blacklist_tags))
    excluded = set(_normalize_tags(opts.excluded_tags))
    blacklist_tags |= excluded
    # auto-blacklist: based_on_version
    if opts.based_on_version:
        ver_tag = opts.based_on_version.lower().strip().replace(" ", "_")
        if ver_tag and ver_tag not in blacklist_tags:
            blacklist_tags.add(ver_tag)
            on_progress(f"[reg] Auto-added to blacklist: {ver_tag}")

    failed_tags: set[str] = set()
    source_tags_used: set[str] = set()

    # target count
    total_target = (
        opts.target_count
        if opts.target_count and opts.target_count > 0
        else structure["total_images"]
    )
    # B1 (PR-2): build_mode decides the subfolder allocation
    # - mirror: old source-script behavior, mirrors train's subfolders, each pulled independently to its own count
    # - flat: all images go into a single 1_data/ bucket, satisfied by total_target in one go
    if opts.build_mode == "flat":
        subfolders_plan: dict[str, dict[str, Any]] = {
            "1_data": {"image_count": total_target}
        }
        on_progress(
            f"[reg] Flat mode: target {total_target} images, single bucket 1_data/"
        )
    else:
        subfolders_plan = structure["subfolders"]
        on_progress(
            f"[reg] Mirror mode: target {total_target} images (train total "
            f"{structure['total_images']}), mirroring {len(subfolders_plan)} subfolders"
        )

    # output directory
    opts.output_dir.mkdir(parents=True, exist_ok=True)

    # PP5.1 - scan existing images when incremental
    pre_existing_per_sub: dict[str, dict[str, Any]] = {}
    prior_meta: Optional[RegMeta] = None
    if incremental:
        pre_existing_per_sub = collect_existing_reg_per_subfolder(opts.output_dir)
        prior_meta = read_meta(opts.output_dir)
        existing_total = sum(b["count"] for b in pre_existing_per_sub.values())
        on_progress(
            f"[reg] Incremental mode: {existing_total} existing images across "
            f"{len(pre_existing_per_sub)} subfolders"
        )

    # A2 - booru IDs the user deleted from the UI (across all subfolders)
    # should be excluded regardless of incremental: on a fresh build
    # .deleted_ids.json has already been cleared by DELETE /reg; this set is
    # only non-empty during incremental, to avoid a top-up pulling deleted
    # images back in.
    deleted_ids = read_deleted_ids(opts.output_dir)
    if deleted_ids:
        on_progress(f"[reg] {len(deleted_ids)} previously deleted booru IDs (A2 exclusion)")

    target_resolution = structure.get("median_resolution")
    target_aspect_ratio = structure.get("median_aspect_ratio")
    resolution_std = structure.get("resolution_std")

    total_downloaded = 0
    success_subfolder_count = 0
    for sub_name, sub_data in subfolders_plan.items():
        try:
            ok, dled = _build_for_subfolder(
                sub_name,
                sub_data,
                structure["global_tag_weights"],
                opts.output_dir,
                opts=opts,
                blacklist_tags=blacklist_tags,
                failed_tags=failed_tags,
                source_tags_used=source_tags_used,
                source_image_ids=source_image_ids,
                target_resolution=target_resolution,
                target_aspect_ratio=target_aspect_ratio,
                resolution_std=resolution_std,
                total_target_count=total_target,
                total_downloaded_so_far=total_downloaded,
                on_progress=on_progress,
                cancel_event=cancel_event,
                pre_existing=pre_existing_per_sub.get(sub_name),
                client=client,
                deleted_ids=deleted_ids,
            )
            if ok:
                success_subfolder_count += 1
            total_downloaded += dled
            if total_downloaded >= total_target:
                on_progress(f"[reg] Reached the overall target of {total_target}, stopping remaining subfolders")
                break
        except Exception as exc:
            on_progress(f"[reg] Error in subfolder {sub_name}: {exc}")
            import traceback
            on_progress(traceback.format_exc())

    # write meta
    top_dist = dict(structure["global_tag_freq"].most_common(50))
    # when incremental, failed_tags / source_tags / auto_tagged / runs are all merged with the old meta
    if incremental and prior_meta is not None:
        merged_failed = sorted(set(failed_tags) | set(prior_meta.failed_tags))
        merged_source = sorted(set(source_tags_used) | set(prior_meta.source_tags))
        merged_excluded = sorted(set(excluded) | set(prior_meta.excluded_tags))
        runs = prior_meta.incremental_runs + 1
    else:
        merged_failed = sorted(failed_tags)
        merged_source = sorted(source_tags_used)
        merged_excluded = sorted(excluded)
        runs = 0
    meta = RegMeta(
        generated_at=time.time(),
        based_on_version=opts.based_on_version,
        api_source=opts.api_source,
        target_count=total_target,
        actual_count=total_downloaded,
        source_tags=merged_source,
        excluded_tags=list(merged_excluded),
        blacklist_tags=sorted(blacklist_tags),
        failed_tags=merged_failed,
        train_tag_distribution=top_dist,
        auto_tagged=False,  # overwritten by the worker once auto_tag finishes
        incremental_runs=runs,
        build_mode=opts.build_mode,
    )
    write_meta(opts.output_dir, meta)

    on_progress(
        f"[reg] Done: {total_downloaded}/{total_target}"
        f" images ({success_subfolder_count}/{len(subfolders_plan)} "
        f"subfolders reached 80%)"
    )
    return meta


# ---------------------------------------------------------------------------
# meta IO
# ---------------------------------------------------------------------------


META_FILENAME = "meta.json"


def meta_path(reg_dir: Path) -> Path:
    return reg_dir / META_FILENAME


def write_meta(reg_dir: Path, meta: RegMeta) -> Path:
    import json
    p = meta_path(reg_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


def read_meta(reg_dir: Path) -> Optional[RegMeta]:
    import json
    p = meta_path(reg_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return RegMeta(**data)
    except Exception:
        return None


def update_meta_auto_tagged(
    reg_dir: Path, auto_tagged: bool, kind: Optional[str] = None,
) -> None:
    """Rewrite meta.auto_tagged after auto_tag finishes, and record which tagger was used.

    When `kind=None`, only `auto_tagged` is touched, `auto_tag_kind` is left
    alone (compatibility with old callers). New callers should pass kind
    alongside it.
    """
    m = read_meta(reg_dir)
    if m is None:
        return
    m.auto_tagged = auto_tagged
    if kind is not None:
        m.auto_tag_kind = kind if auto_tagged else None
    write_meta(reg_dir, m)


# ---------------------------------------------------------------------------
# A2 - user-deleted blacklist (reg/.deleted_ids.json)
# ---------------------------------------------------------------------------


DELETED_IDS_FILENAME = ".deleted_ids.json"


def deleted_ids_path(reg_dir: Path) -> Path:
    return reg_dir / DELETED_IDS_FILENAME


def read_deleted_ids(reg_dir: Path) -> set[str]:
    """Read reg/.deleted_ids.json; missing / corrupt returns an empty set."""
    import json
    p = deleted_ids_path(reg_dir)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x) for x in data if x}
    except Exception:
        pass
    return set()


def clear_reg_dir(reg_dir: Path) -> None:
    """Clear everything under reg/ (images, subfolders, meta,
    `.deleted_ids.json`, etc.), keeping the empty directory itself.

    Used by the full-mode build entry point: the user's intent is "start from
    zero", so `.deleted_ids.json` is cleared too (if the user wants to keep
    their deleted preferences, they should choose incremental mode instead).

    Matches the behavior of `DELETE /api/projects/{pid}/versions/{vid}/reg` -
    that endpoint also does iterdir + rmtree/unlink on children.
    """
    import shutil
    if not reg_dir.exists():
        return
    for child in reg_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def append_deleted_ids(reg_dir: Path, new_ids: list[str]) -> None:
    """Append new_ids (booru ID = filename stem) to reg/.deleted_ids.json, deduped, order preserved.

    The DELETE /reg endpoint clears this file too when it clears reg/ (it
    globs and deletes all children), so a fresh build never sees the
    previous round's deletion blacklist.
    """
    import json
    if not new_ids:
        return
    p = deleted_ids_path(reg_dir)
    existing = read_deleted_ids(reg_dir)
    merged = sorted(existing | {str(x) for x in new_ids if x})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_meta_postprocess(
    reg_dir: Path,
    *,
    when: Optional[float],
    clusters: Optional[int],
    method: Optional[str],
    max_crop_ratio: Optional[float],
) -> None:
    """PP5.5 - rewrite meta's post-process fields once post-processing finishes."""
    m = read_meta(reg_dir)
    if m is None:
        return
    m.postprocessed_at = when
    m.postprocess_clusters = clusters
    m.postprocess_method = method
    m.postprocess_max_crop_ratio = max_crop_ratio
    write_meta(reg_dir, m)


# ---------------------------------------------------------------------------
# preview helper (used by the GET /reg/preview-tags endpoint)
# ---------------------------------------------------------------------------


def preview_train_tag_distribution(
    train_dir: Path, top: int = 20
) -> list[tuple[str, int]]:
    """Lightweight scan of train's tag frequency, returns the top N. Doesn't read image dimensions (fast)."""
    counter: Counter[str] = Counter()
    if not train_dir.exists():
        return []
    for img in train_dir.rglob("*"):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        counter.update(analyze_tags_in_file(img))
    return counter.most_common(top)
