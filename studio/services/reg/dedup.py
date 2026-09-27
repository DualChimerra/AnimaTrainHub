"""A4 -- reg-set dedup helper (PR-1).

Applies the similarity-grouping algorithm from `services/preprocess/duplicates.py` to reg/,
returning the relative paths to delete in each group; deletion (including .txt +
`.deleted_ids.json` + meta update) is handled by `purge_paths`.

Two callers:
- API endpoint `POST /reg/dedup-purge` -- triggered manually by the user in RegPreview
- worker reg_build_worker -- when `auto_dedup=True`, runs automatically after build and loops
  with incremental top-up until the target count is reached

Does no path-traversal validation: callers are responsible for ensuring `relative_paths` are
valid relative paths inside rdir (worker-generated ones are always valid; the endpoint path goes through `_safe_join_or_400`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..dataset.scan import IMAGE_EXTS
from . import builder as reg_builder


def scan_for_dedup(reg_dir: Path) -> list[str]:
    """Run similarity grouping on reg/, returning the **non-kept** relative paths in each group.

    Each group keeps `group[0]` (duplicate_finder sorts by file size + pixel count, the first
    item is treated as the "recommended keep"); the rest are treated as "recommended delete". No duplicate groups returns [].

    Uses the default `DuplicateOptions()` -- per the owner's decision (2026-05-30): the reg-set quality
    bar is lower than train, so parameter tuning isn't exposed.
    """
    if not reg_dir.exists():
        return []
    from ..preprocess import duplicates as duplicate_finder

    sources: list[tuple[str, Path]] = []
    for f in reg_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            try:
                rel = f.relative_to(reg_dir).as_posix()
            except ValueError:
                continue
            sources.append((rel, f))
    if not sources:
        return []

    options = duplicate_finder.DuplicateOptions()
    try:
        infos = duplicate_finder.build_all_image_infos(sources, options)
        groups, _pair_metrics, _stats = duplicate_finder.group_similar_images(
            infos, options,
        )
    except duplicate_finder.DuplicateFinderError:
        return []

    to_delete: list[str] = []
    for group in groups:
        if len(group) <= 1:
            continue
        for item in group[1:]:
            to_delete.append(item.name)
    return to_delete


def purge_paths(reg_dir: Path, relative_paths: list[str]) -> dict[str, Any]:
    """Delete images under reg/ by relative path + same-named .txt caption, update meta.actual_count,
    and append the booru ID (file stem) to `reg/.deleted_ids.json`.

    Nonexistent path / non-image: silently skipped. **No** traversal validation is done -- callers are responsible.

    Returns `{deleted: [rel...], count: int}`.
    """
    deleted: list[str] = []
    deleted_booru_ids: list[str] = []
    for rel in relative_paths:
        if not rel:
            continue
        parts = [p for p in rel.replace("\\", "/").split("/") if p]
        if not parts:
            continue
        target = reg_dir.joinpath(*parts)
        if not target.exists() or target.suffix.lower() not in IMAGE_EXTS:
            continue
        booru_id = target.stem
        try:
            target.unlink()
            txt = target.with_suffix(".txt")
            if txt.exists():
                txt.unlink()
        except OSError:
            continue
        deleted.append(rel)
        deleted_booru_ids.append(booru_id)

    if deleted_booru_ids:
        reg_builder.append_deleted_ids(reg_dir, deleted_booru_ids)

    meta = reg_builder.read_meta(reg_dir)
    if meta is not None and deleted:
        meta.actual_count = max(0, meta.actual_count - len(deleted))
        reg_builder.write_meta(reg_dir, meta)

    return {"deleted": deleted, "count": len(deleted)}
