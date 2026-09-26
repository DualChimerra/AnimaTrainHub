"""Preprocess status manifest (single JSON file, per version).

Design: see [ADR 0010](../../docs/adr/0010-preprocess-train-scope.md)
(supersedes ADR 0004 - the old project-level preprocess/manifest.json is now
read-only, kept only as a fallback source for ensure_train_manifest to
migrate old projects; nothing mutates it anymore).

In short
--------
`projects/{id}-{slug}/versions/{label}/train/manifest.json` records, for that
version's train/ directory, each image's origin + status.

Schema (as written) - deliberately minimal:

    {
      "images": {
        "1_data/X.png":    {"origin": "X.png",  "mtime": ..., "size": ..., "processed": true},
        "1_data/Y_c0.png": {"origin": "Y.png",  "mtime": ..., "size": ...},
        "1_data/Y_c1.png": {"origin": "Y.png",  "mtime": ..., "size": ...}
      }
    }

Fields:
- entry key = POSIX-style relative path under train/, `"{folder}/{filename}"`
- `origin` = the source filename this image traces back to in `download/` (multi-crop derivatives share an origin)
- `processed` = whether it went through upscale/crop (the worker sets True; a plain curate copy doesn't set it)
- `kind: "duplicate_removed"` marks a manually reviewed skip; the physical file in train/ is not deleted

An image with no manifest entry is an implicit original (its train/ file was just copied in by the curate stage).

Concurrent writes
-----------------
Single server process, no cross-process writers: a `threading.Lock` serializes all in-process mutations.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import masks as train_masks

MANIFEST_NAME = "manifest.json"
DUPLICATE_REMOVED_KIND = "duplicate_removed"

# In-process serialization lock. Every mutation must go `with _LOCK:`; reads don't need it (json.load is atomic).
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def manifest_path(project_dir: Path) -> Path:
    return project_dir / "preprocess" / MANIFEST_NAME


# ---------------------------------------------------------------------------
# read / write
# ---------------------------------------------------------------------------


def _empty_manifest() -> dict[str, Any]:
    return {"images": {}}


def load(project_dir: Path) -> dict[str, Any]:
    """Read the manifest; missing or corrupt -> empty manifest (never raises).

    A single read takes no lock - `json.load` is atomic, so the worst case is
    reading a stale version, never a half-written one. `_atomic_write` uses
    tmp+rename so the rename itself is atomic.
    """
    path = manifest_path(project_dir)
    if not path.exists():
        return _empty_manifest()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("images"), dict):
            return _empty_manifest()
        return raw
    except (OSError, json.JSONDecodeError):
        # Don't raise on corruption - the next write overwrites it with something valid; callers consistently see an empty manifest until then
        return _empty_manifest()


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """Atomic tmp+rename write. Same-partition write + os.replace guarantees readers always see complete JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)  # cross-platform atomic rename


# ---------------------------------------------------------------------------
# Resolver - unified downstream entry point
# ---------------------------------------------------------------------------


def entry_origin(entry: dict[str, Any], fallback_name: str) -> str:
    """Extract origin from an entry (the filename under download/{...} it points to).

    Falls back to the entry's own key if `origin` is missing (1:1 same-name assumption).
    """
    return entry.get("origin") or fallback_name


def is_duplicate_removed_entry(entry: Optional[dict[str, Any]]) -> bool:
    """Whether a manifest entry was manually reviewed and confirmed as a skip during dedup."""
    return bool(entry and entry.get("kind") == DUPLICATE_REMOVED_KIND)


def resolve(project_dir: Path, name: str) -> Optional[Path]:
    """Given a product filename (e.g. `foo.png`), return the disk path it actually refers to.

    Implicit original -> `download/{name}` (even if the file doesn't exist; the resolver doesn't check existence)
    Has a manifest entry -> `preprocess/{name}`

    Callers check existence with `.exists()` as needed - this way listing images only needs one stat, no duplicates.
    """
    m = load(project_dir)
    entry = m["images"].get(name)
    if entry is None:
        return project_dir / "download" / name
    return project_dir / "preprocess" / name


def resolve_origin(project_dir: Path, download_name: str) -> list[Path]:
    """Reverse resolve: given a download/{name}, list every derivative product under preprocess/.

    - Manifest has processed entries with `origin == download_name` -> return them [preprocess/X]
    - Only duplicate_removed entries trace back to this origin -> return [] (downstream skips it)
    - No matching entry -> fall back to [download/download_name] (implicit original)
    """
    m = load(project_dir)
    removed = False
    matches: list[Path] = []
    for name, entry in m["images"].items():
        if entry_origin(entry, name) != download_name:
            continue
        if is_duplicate_removed_entry(entry):
            removed = True
            continue
        matches.append(project_dir / "preprocess" / name)
    if matches:
        return matches
    if removed:
        return []
    return [project_dir / "download" / download_name]


def get_entry(project_dir: Path, name: str) -> Optional[dict[str, Any]]:
    """Read a single entry (None if missing). Used by the thumb endpoint's resolve_origin fallback."""
    m = load(project_dir)
    return m["images"].get(name)


# ---------------------------------------------------------------------------
# ADR 0010 - per-version train/ manifest (fallback rebuild)
#
# The current model writes preprocess products to versions/{label}/train/,
# with status recorded in the manifest.json alongside it. This section only
# exposes the fallback entry point: the first time a version's train
# manifest is accessed, it's implicitly rebuilt from the old project-level
# preprocess/manifest.json.
#
# See docs/adr/0010-preprocess-train-scope.md + docs/design/preprocess-train-scope-plan.md
# section 3.2. The rebuild only reads the old manifest's metadata, it doesn't
# copy image bytes (train/ already holds the processed products, copied in by
# the curate stage; the only thing the new model loses is the origin
# back-reference).
# ---------------------------------------------------------------------------

TRAIN_MANIFEST_VERSION = 2


def train_manifest_path(project_dir: Path, version_label: str) -> Path:
    return project_dir / "versions" / version_label / "train" / MANIFEST_NAME


def _scan_train_images(train_dir: Path) -> set[str]:
    """Collect image relative paths (POSIX form) from train_dir's immediate sub-folders.

    LoRA training uses a repeat-folder layout: `train/1_data/X.png`, not
    `train/X.png` (`{N_label}/{image}` is what dataset_config.toml parses).
    Manifest entry keys use the POSIX relative path to disambiguate across
    folders (the same filename can legitimately appear in more than one
    folder).

    Images placed directly at the root are ignored (shouldn't happen, but
    defensive); non-image files (caption .txt / anything else) are also ignored.
    """
    from ..dataset.scan import IMAGE_EXTS

    if not train_dir.exists():
        return set()
    rel_paths: set[str] = set()
    for sub in train_dir.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                rel_paths.add(f"{sub.name}/{f.name}")
    return rel_paths


def _build_train_manifest_from_legacy(
    legacy: dict[str, Any], train_rel_paths: set[str]
) -> dict[str, Any]:
    """Extract origin relationships for images that actually exist under train/, from the old project-level manifest.

    The old manifest's entry names are flat product names (e.g. `X.png`, no
    folder prefix). The new train/ layout puts them in a sub-folder (e.g.
    `1_data/X.png`). Matching rule:

    - Index by filename (the last path segment)
    - If an old entry's name matches any train file with the same name, use
      that train relative path as the new key
    - If the same name appears in multiple sub-folders, add one entry per
      matching relative path (conservative - let the user decide in the UI;
      rare but valid)

    Skipped: (1) old entries with no matching filename under train/, (2) old
    duplicate_removed entries (manual dedup-review state doesn't migrate
    across models; the new model tracks it per-version independently).
    """
    by_filename: dict[str, list[str]] = {}
    for rel in train_rel_paths:
        nm = rel.rsplit("/", 1)[-1]
        by_filename.setdefault(nm, []).append(rel)

    images: dict[str, Any] = {}
    for name, entry in legacy.get("images", {}).items():
        if not isinstance(entry, dict):
            continue
        if is_duplicate_removed_entry(entry):
            continue
        rels = by_filename.get(name)
        if not rels:
            continue
        for rel in rels:
            images[rel] = {
                "origin": entry_origin(entry, name),
                "mtime": entry.get("mtime", 0),
                "size": entry.get("size", 0),
            }
    return {"version": TRAIN_MANIFEST_VERSION, "images": images}


def ensure_train_manifest(project_dir: Path, version_label: str) -> Path:
    """Idempotent: ensures versions/{label}/train/manifest.json exists; returns its path.

    Fallback rebuild rules (see ADR 0010 "Decision" section):

    1. Target already exists -> return it directly (O(1) stat, no overhead on the hot path)
    2. Doesn't exist + the old `preprocess/manifest.json` exists -> rebuild a
       v2-schema manifest by matching train/'s actual filenames against the
       old entries' origins
    3. Old manifest also missing / corrupt -> write an empty v2 manifest

    The train/ directory is also **created** if missing (it may still be
    empty the first time this version is accessed).

    Every train manifest read path should go through this first (defensive;
    the idempotent cost is one stat). Forking a version
    (`versions.py:create_version`) also calls it explicitly to protect
    against a corrupt source manifest.

    PR-1 scope: this function + tests. **Wiring it into call sites is PR-2
    scope** (done alongside slimming down the manifest module, across all
    read/write entry points).
    """
    target = train_manifest_path(project_dir, version_label)
    if target.exists():
        return target

    train_dir = target.parent
    legacy_path = manifest_path(project_dir)  # old project-level manifest

    with _LOCK:
        # Double-check after acquiring the lock (someone else may have just created it)
        if target.exists():
            return target

        train_dir.mkdir(parents=True, exist_ok=True)

        # Collect images under train/ (immediate sub-folders, LoRA repeat-folder layout)
        train_rel_paths = _scan_train_images(train_dir)

        # Read the old manifest (missing / corrupt -> empty, same semantics as load())
        legacy: dict[str, Any]
        if legacy_path.exists():
            try:
                raw = json.loads(legacy_path.read_text(encoding="utf-8"))
                legacy = raw if isinstance(raw, dict) else {}
            except (OSError, json.JSONDecodeError):
                legacy = {}
        else:
            legacy = {}

        manifest = _build_train_manifest_from_legacy(legacy, train_rel_paths)
        _atomic_write(target, manifest)
        return target


# ---------------------------------------------------------------------------
# ADR 0010 - train-scope manifest API
#
# The old project-scope API has been removed; this section is the sole
# mutation API today.
#
# Key semantics:
# - the manifest lives at `versions/{label}/train/manifest.json`
# - entry keys are **POSIX relative paths** (e.g. `"1_data/X.png"`),
#   expressing the LoRA repeat-folder layout (`train/{N_label}/{image}`);
#   same-name images in different folders get independent entries
# - `train_restore(name)` = copy `download/{entry.origin}` back over
#   `train/{name}` (it does not delete the entry; see ADR 0010's Restore
#   semantics section); if the origin file is missing, the name goes on the
#   no_origin list
# - `train_add_processed` falls back to stat'ing `train/{name}` for size
# - every train_xxx mutation calls ensure_train_manifest first (defensive, idempotent)
#
# Still using the single module-level `_LOCK` (version writes are infrequent, a single lock is fine).
# ---------------------------------------------------------------------------


def _train_dir(project_dir: Path, version_label: str) -> Path:
    return project_dir / "versions" / version_label / "train"


def _empty_train_manifest() -> dict[str, Any]:
    return {"version": TRAIN_MANIFEST_VERSION, "images": {}}


def _read_train_target(target: Path) -> dict[str, Any]:
    """Read a train manifest file (target already known to exist); corrupt -> empty v2 manifest.

    Same design as the old `load()` - doesn't raise on corruption, gets
    overwritten on the next write. Used internally by callers alongside
    `ensure_train_manifest` (callers have already ensured target exists).
    """
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("images"), dict):
            return raw
    except (OSError, json.JSONDecodeError):
        pass
    return _empty_train_manifest()


# ---- read ----------------------------------------------------------------


def train_load(project_dir: Path, version_label: str) -> dict[str, Any]:
    """Read the train manifest; rebuilds via fallback if missing (see ADR 0010's fallback rebuild section).

    Returns the full manifest dict `{"version": 2, "images": {...}}`.
    """
    target = ensure_train_manifest(project_dir, version_label)
    return _read_train_target(target)


def train_get_entry(
    project_dir: Path, version_label: str, name: str
) -> Optional[dict[str, Any]]:
    return train_load(project_dir, version_label)["images"].get(name)


def train_all_processed(
    project_dir: Path, version_label: str
) -> dict[str, dict[str, Any]]:
    """Entries that are not duplicate_removed."""
    m = train_load(project_dir, version_label)
    return {
        name: entry
        for name, entry in m["images"].items()
        if not is_duplicate_removed_entry(entry)
    }


def train_duplicate_removed(
    project_dir: Path, version_label: str
) -> dict[str, dict[str, Any]]:
    m = train_load(project_dir, version_label)
    return {
        name: entry
        for name, entry in m["images"].items()
        if is_duplicate_removed_entry(entry)
    }


def train_duplicate_removed_origins(
    project_dir: Path, version_label: str
) -> set[str]:
    return {
        entry_origin(entry, name)
        for name, entry in train_duplicate_removed(
            project_dir, version_label
        ).items()
    }


# ---- mutation (must be `with _LOCK`) -------------------------------------


def train_add_processed(
    project_dir: Path,
    version_label: str,
    name: str,
    meta: dict[str, Any],
) -> None:
    """Record one processed image (train scope).

    Schema: keeps `origin / mtime / size / processed`, drops any other
    fields (model/scale/action/...). Size falls back to stat'ing `train/{name}`.

    `processed: bool` (ADR 0010 fixup 2026-06-04): the worker passes
    `meta["processed"] = True` after finishing an upscale/crop; a plain
    curate-stage `copy_download_to_train` doesn't pass it (defaults to False,
    field omitted). The frontend uses this field to draw the "processed"
    badge; see ADR 0010's section on inferring status from field differences.
    """
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    with _LOCK:
        m = _read_train_target(target)
        origin = meta.get("origin") or name
        entry: dict[str, Any] = {
            "origin": origin,
            "mtime": meta.get("mtime", time.time()),
        }
        if "size" in meta:
            entry["size"] = meta["size"]
        else:
            png = _train_dir(project_dir, version_label) / name
            try:
                entry["size"] = png.stat().st_size
            except OSError:
                entry["size"] = 0
        if meta.get("processed"):
            entry["processed"] = True
        m["images"][name] = entry
        _atomic_write(target, m)


def train_replace_with_crops(
    project_dir: Path,
    version_label: str,
    *,
    source_name: str,
    outputs: list[dict[str, Any]],
) -> None:
    """Multi-crop fan-out: replace `source_name` with N crop-product entries.

    Same operation as the old `replace_with_crops` - finds every old entry
    whose origin matches source_name, plus source_name itself, deletes them
    all, and writes N new entries (origin carried over from the old entry, or
    falling back to source_name).

    Disk files (train/{name}.png etc.) are the caller's responsibility; this
    function only touches the manifest.
    """
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    with _LOCK:
        m = _read_train_target(target)
        to_remove = {source_name}
        for nm, entry in m["images"].items():
            if entry_origin(entry, nm) == source_name:
                to_remove.add(nm)
        for nm in to_remove:
            m["images"].pop(nm, None)
        now = time.time()
        for o in outputs:
            entry = {
                "origin": o.get("origin") or source_name,
                "mtime": o.get("mtime", now),
                "size": int(o.get("size", 0)),
            }
            # a crop derivative is inherently a processing operation (ADR 0010 fixup)
            if o.get("processed", True):
                entry["processed"] = True
            m["images"][o["name"]] = entry
        _atomic_write(target, m)


def train_mark_duplicate_removed(
    project_dir: Path,
    version_label: str,
    names: list[str],
) -> dict[str, list[str]]:
    """Dedup removal (train scope): physically deletes train/{name} + its
    caption sidecar, and rewrites the manifest entry as a `kind=duplicate_removed`
    tombstone (used by the overview page's "removed" tab + `train_restore_duplicate_removed`).

    Downstream tagging / training scan `train/` directly, so physically
    deleting the file guarantees it no longer shows up in the caption queue
    or dataset_config listing.

    Each version is reviewed independently (the manifest is per-version).
    Forking a version copies the whole tree (ADR 0007 `_copytree("train")`)
    including physical files only - the tombstone is copied along with it
    since the manifest also lives under train/.
    """
    removed: list[str] = []
    missing: list[str] = []
    skipped: list[str] = []
    now = time.time()
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    train_dir = _train_dir(project_dir, version_label)
    with _LOCK:
        m = _read_train_target(target)
        for name in names:
            entry = m["images"].get(name)
            if is_duplicate_removed_entry(entry):
                skipped.append(name)
                continue
            src = train_dir / name
            if entry is not None:
                origin = entry_origin(entry, name)
                size = int(entry.get("size", 0) or 0)
            elif src.is_file():
                origin = name
                try:
                    size = src.stat().st_size
                except OSError:
                    size = 0
            else:
                missing.append(name)
                continue
            # physically delete the image + caption sidecar + mask sidecar
            if src.is_file():
                try:
                    src.unlink()
                except OSError:
                    pass
            for ext in (".txt", ".json"):
                sidecar = src.with_suffix(ext)
                if sidecar.is_file():
                    try:
                        sidecar.unlink()
                    except OSError:
                        pass
            train_masks.delete_mask(train_dir, name)
            m["images"][name] = {
                "kind": DUPLICATE_REMOVED_KIND,
                "origin": origin,
                "mtime": now,
                "size": size,
            }
            removed.append(name)
        _atomic_write(target, m)
    return {"removed": removed, "missing": missing, "skipped": skipped}


def train_restore_duplicate_removed(
    project_dir: Path,
    version_label: str,
    names: list[str],
) -> dict[str, list[str]]:
    """Undo a dedup removal: copy the image + caption back from
    `download/{entry.origin}` over `train/{name}`, and delete the manifest entry.

    Returns three groups:
    - `restored`: successfully restored (the download original exists and was copied over)
    - `missing`: name has no entry, or the entry isn't duplicate_removed
    - `no_origin`: the entry is duplicate_removed but `download/{origin}` is
      physically missing - the entry is kept, caller should prompt the user
      to import the original externally
    """
    import shutil

    restored: list[str] = []
    missing: list[str] = []
    no_origin: list[str] = []
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    download_dir = project_dir / "download"
    train_dir = _train_dir(project_dir, version_label)
    with _LOCK:
        m = _read_train_target(target)
        for name in names:
            entry = m["images"].get(name)
            if not is_duplicate_removed_entry(entry):
                missing.append(name)
                continue
            origin = entry_origin(entry, name)
            src = download_dir / origin
            if not src.is_file():
                no_origin.append(name)
                continue
            dst = train_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except OSError:
                no_origin.append(name)
                continue
            # caption follows along: download/{origin_stem}.{ext} -> train/{name_stem}.{ext}
            origin_stem = Path(origin).stem
            for ext in (".txt", ".json"):
                cap_src = download_dir / f"{origin_stem}{ext}"
                if cap_src.is_file():
                    try:
                        shutil.copy2(cap_src, dst.with_suffix(ext))
                    except OSError:
                        pass
            del m["images"][name]
            restored.append(name)
        _atomic_write(target, m)
    return {"restored": restored, "missing": missing, "no_origin": no_origin}


def train_restore(
    project_dir: Path,
    version_label: str,
    names: list[str],
) -> dict[str, list[str]]:
    """Restore: copy `download/{entry.origin}` back over `train/{folder}/{origin}`.

    Multi-crop fan-out is collapsed: if `name` is a member of a fan-out group
    (multiple entries in the same folder sharing an origin), the whole group
    is restored down to a single `train/{folder}/{origin}` image; sibling
    physical files, manifest entries, and caption sidecars are all cleaned up
    together. This guarantees restoring a_0/a_1 never leaves you with "two
    copies of the same image A".

    Caption follows along: copied from `download/{origin_stem}.{ext}` to
    `train/{folder}/{origin_stem}.{ext}` (`.txt` / `.json`).

    Returns three groups:
    - `restored`: the *input* names that were successfully restored (other
      siblings in a fan-out group are listed individually here too, even
      though they were cleaned up as a side effect, so the UI can reconcile)
    - `missing`: name has no manifest entry (and wasn't already cleaned up as
      a sibling earlier in this batch)
    - `no_origin`: entry exists but `download/{origin}` is physically missing
      - the UI should offer the user three choices (drag in a replacement /
        keep the processed version / remove from train)
    """
    import shutil

    restored: list[str] = []
    missing: list[str] = []
    no_origin: list[str] = []
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    download_dir = project_dir / "download"
    train_dir = _train_dir(project_dir, version_label)
    # siblings already handled within this batch (avoid duplicate copies / false missing)
    handled: set[str] = set()
    with _LOCK:
        m = _read_train_target(target)
        for name in names:
            if name in handled:
                restored.append(name)
                continue
            entry = m["images"].get(name)
            if entry is None:
                missing.append(name)
                continue
            origin = entry_origin(entry, name)
            folder = name.rsplit("/", 1)[0] if "/" in name else ""
            src = download_dir / origin
            if not src.is_file():
                no_origin.append(name)
                continue
            # find the fan-out group: every entry in the same folder sharing this origin
            group: list[str] = [
                k for k, e in m["images"].items()
                if (k.rsplit("/", 1)[0] if "/" in k else "") == folder
                and entry_origin(e, k) == origin
            ]
            dst_rel = f"{folder}/{origin}" if folder else origin
            dst = train_dir / dst_rel
            # delete sibling physical files + captions (leave dst itself alone - the copy below overwrites it)
            for sib in group:
                if sib == dst_rel:
                    continue
                sib_path = train_dir / sib
                if sib_path.is_file():
                    try:
                        sib_path.unlink()
                    except OSError:
                        pass
                for ext in (".txt", ".json"):
                    side = sib_path.with_suffix(ext)
                    if side.is_file():
                        try:
                            side.unlink()
                        except OSError:
                            pass
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except OSError:
                no_origin.append(name)
                continue
            # caption sidecar follows along
            origin_stem = Path(origin).stem
            for ext in (".txt", ".json"):
                cap_src = download_dir / f"{origin_stem}{ext}"
                if cap_src.is_file():
                    try:
                        shutil.copy2(cap_src, dst.with_suffix(ext))
                    except OSError:
                        pass
            # mask sidecar: restore means going back to the download original,
            # so the whole group's masks are invalidated (D8 - deleted even if
            # the size happens to still match; predictability wins)
            train_masks.delete_masks_for(train_dir, {*group, dst_rel})
            # manifest: delete the whole group + write a new entry at dst_rel
            for sib in group:
                m["images"].pop(sib, None)
            try:
                st = src.stat()
                m["images"][dst_rel] = {
                    "origin": origin,
                    "mtime": int(st.st_mtime),
                    "size": st.st_size,
                }
            except OSError:
                m["images"][dst_rel] = {"origin": origin}
            restored.append(name)
            handled.update(group)
        _atomic_write(target, m)
    return {"restored": restored, "missing": missing, "no_origin": no_origin}


def train_swap_entry(
    project_dir: Path,
    version_label: str,
    old_name: str,
    new_name: str,
    meta: dict[str, Any],
) -> None:
    """Atomically replace a train manifest entry: delete `old_name`, write `new_name`.

    Used by the worker when an upscale output's extension changes (e.g.
    src=`1_data/X.jpg` -> dst=`1_data/X.png`), so the manifest doesn't keep a
    dangling old entry.

    `meta` follows the same rules as `train_add_processed` - only
    origin/mtime/size are kept, everything else is dropped. Size falls back
    to stat'ing `train/{new_name}`.
    """
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    with _LOCK:
        m = _read_train_target(target)
        m["images"].pop(old_name, None)
        origin = meta.get("origin") or new_name
        entry: dict[str, Any] = {
            "origin": origin,
            "mtime": meta.get("mtime", time.time()),
        }
        if "size" in meta:
            entry["size"] = meta["size"]
        else:
            png = _train_dir(project_dir, version_label) / new_name
            try:
                entry["size"] = png.stat().st_size
            except OSError:
                entry["size"] = 0
        if meta.get("processed"):
            entry["processed"] = True
        m["images"][new_name] = entry
        _atomic_write(target, m)


def train_remove_entries(
    project_dir: Path,
    version_label: str,
    names: list[str],
) -> int:
    """Bulk-delete train manifest entries (by entry key). Returns the number actually removed.

    Used by `curation.remove_from_train`: when a user deletes a download
    original from train, the caller looks up the N derived relative paths by
    origin (multi-crop fan-out), and a single pop-and-write is more efficient
    than mutating one entry at a time.
    """
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    removed = 0
    with _LOCK:
        m = _read_train_target(target)
        for name in names:
            if m["images"].pop(name, None) is not None:
                removed += 1
        if removed:
            _atomic_write(target, m)
    return removed


def train_clear_all(project_dir: Path, version_label: str) -> None:
    """Clear this version's train manifest state - only clears the manifest
    file, does **not** touch physical files under train/.

    Different semantics from the old `clear_all` - that one deleted the
    physical preprocess/ PNG products; under the current model, train/ *is*
    the training data itself, so deleting physical files would mean deleting
    the training set, which "clearing preprocess status" should never
    trigger. If a caller wants to fully redo preprocessing, it should call
    `train_restore` on each image instead (restoring the download original),
    not this function.

    This function exists for the extreme case where the manifest is
    corrupted beyond reading, to "zero and rebuild".
    """
    ensure_train_manifest(project_dir, version_label)
    target = train_manifest_path(project_dir, version_label)
    with _LOCK:
        _atomic_write(target, _empty_train_manifest())
