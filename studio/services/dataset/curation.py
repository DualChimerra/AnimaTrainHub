"""Curation operations (PP3): backend logic for the download / train dual panel.

- `download/` is always the full project-level backup and is never deleted
- The left-side candidates = the `download/` listing (**each image resolves its
  actual byte path via `preprocess_manifest.resolve()`**, which may be
  download/{name} or preprocess/{name}.png -- transparent to the frontend, see
  ADR 0004)
- Copy / remove only touch the copies under `versions/{label}/train/{folder}/`
- Filenames are diffed as sets: left = download minus all-train, right = train
  grouped by folder
- Subfolders follow the Kohya-style N_xxx convention (still parsed via
  dataset.parse_repeat during PP4 / PP6 training)
- Each image is returned as `{name, mtime}` (mtime in unix seconds); sort order
  is left to the frontend based on user preference -- the backend only
  guarantees a stable output ordered by name

Constraints:
- Path traversal is not allowed (both folder and filename are validated)
- When copying, any same-stem .txt / .json metadata is carried along (if present)
- When removing, the same-stem metadata is deleted too; download/ is never touched
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from ..projects import projects, versions
from .scan import IMAGE_EXTS
from ..preprocess import manifest as preprocess_manifest
from ..preprocess import masks as train_masks

# Kohya: optional `N_` prefix + letters (no bare digits / empty labels like `5_`)
_FOLDER_PATTERN = re.compile(r"^([0-9]+_)?[A-Za-z][A-Za-z0-9_-]*$")
# Filename safety: only a bare filename (with extension) is allowed, no path separators
_FILE_PATTERN = re.compile(r"^[^\\/]+$")


from studio.domain.errors import DomainError


class CurationError(DomainError):
    """Curation business errors (invalid path / not found / conflict).

    PR-2 C3 adds the DomainError base -- the handler automatically converts it
    into the dual-write envelope.
    """
    default_code = "curation.error"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _validate_folder(name: str) -> None:
    if not _FOLDER_PATTERN.fullmatch(name):
        raise CurationError(
            f'Invalid folder name: "{name}"',
            code="curation.folder_name_invalid", details={"name": name},
        )


def _validate_filename(name: str) -> None:
    if not _FILE_PATTERN.fullmatch(name) or ".." in name:
        raise CurationError(
            f'Invalid file name: "{name}"',
            code="curation.file_name_invalid", details={"name": name},
        )


def _project_dir(conn, project_id: int) -> tuple[dict[str, Any], Path]:
    p = projects.get_project(conn, project_id)
    if not p:
        raise CurationError(
            "Project not found", code="project.not_found",
            details={"id": project_id}, http_status=404,
        )
    return p, projects.project_dir(p["id"], p["slug"])


def _resolve_version_dir(conn, project_id: int, version_id: int) -> tuple[
    dict[str, Any], dict[str, Any], Path
]:
    """Returns (project, version, version root dir); raises 404 if project/version
    doesn't exist."""
    p = projects.get_project(conn, project_id)
    if not p:
        raise CurationError(
            "Project not found", code="project.not_found",
            details={"id": project_id}, http_status=404,
        )
    v = versions.get_version(conn, version_id)
    if not v or v["project_id"] != project_id:
        raise CurationError(
            "Version not found", code="version.not_found",
            details={"id": version_id}, http_status=404,
        )
    return p, v, versions.version_dir(p["id"], p["slug"], v["label"])


def _version_train_dir(conn, project_id: int, version_id: int) -> tuple[
    dict[str, Any], dict[str, Any], Path
]:
    p, v, vdir = _resolve_version_dir(conn, project_id, version_id)
    return p, v, vdir / "train"


def _version_validation_dir(conn, project_id: int, version_id: int) -> tuple[
    dict[str, Any], dict[str, Any], Path
]:
    p, v, vdir = _resolve_version_dir(conn, project_id, version_id)
    return p, v, vdir / "validation"


def _list_image_entries(d: Path) -> list[dict[str, Any]]:
    """List images in a directory -> `[{name, mtime}, ...]`, stably ordered by name.

    mtime comes from the disk stat, in unix seconds (float); once the frontend has
    it, it can freely re-sort by id / name / mtime.
    """
    if not d.exists():
        return []
    entries: list[dict[str, Any]] = []
    for f in d.iterdir():
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            mtime = f.stat().st_mtime
        except OSError:
            mtime = 0.0
        entries.append({"name": f.name, "mtime": mtime})
    entries.sort(key=lambda e: e["name"])
    return entries


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


def list_download(conn, project_id: int) -> list[dict[str, Any]]:
    """Left-side candidate list on the curation page = physical images in
    `download/`, one row each.

    ADR 0010 fixup (2026-06-04): curation is decoupled from preprocessing
    derivatives. The original ADR 0004 design expanded multi-crop derivatives via
    the manifest (X.jpg -> shown as X_c0.png + X_c1.png), but under the new model,
    list_train dedupes by origin (fan-out collapses into one row X.jpg); a mismatch
    between the left/right name spaces broke the `used` exclusion -> images already
    added to train would reappear on the left -> the user re-selects them ->
    `copy_to_train` sees the destination already physically exists -> skip with an
    error.

    New behavior: list_download only lists physical images in download/ (unaware
    of manifest derivatives); `name` shares the same namespace as the `origin`
    returned by list_train (the download filename), so the `used` exclusion works
    correctly. Preprocessing derivatives are only exposed to the user on the
    Preprocess Overview page.

    `duplicate_removed` isn't filtered here either (per the PR-4 fixup decision --
    dedup has moved down into train scope).
    """
    _, pdir = _project_dir(conn, project_id)
    download_dir = pdir / "download"

    entries: list[dict[str, Any]] = []
    if download_dir.exists():
        for f in sorted(download_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            entries.append({"name": f.name, "mtime": mtime})
    return entries


def list_train(
    conn, project_id: int, version_id: int
) -> dict[str, list[dict[str, Any]]]:
    """Train subfolders -> `[{name, mtime, origin}, ...]` (deduped by origin).

    ADR 0010 fixup: the curation page's right-side train area shows "which
    download originals the user selected while curating", decoupled from
    post-preprocessing state:

    - **Deduped** by manifest entry.origin: multi-crop fan-out derivatives
      (X_c0.png + X_c1.png sharing origin=X.jpg) are shown as a single entry
    - The displayed set is determined by a physical iterdir: once
      duplicate_removed images are physically deleted, they naturally no longer
      appear in curation; to view / restore them, use the "Deleted" tab on the
      overview page instead
    - The returned `name` uses **origin** (the download filename), so it aligns
      with `copy_download_to_train` / `remove_from_train`'s name semantics, which
      are scoped to download

    `mtime` uses the physical file's mtime; sorting by time on the frontend stays
    stable. Legacy project fallback: after ensure_train_manifest rebuilds it, the
    same path is used.
    """
    p, v, train = _version_train_dir(conn, project_id, version_id)
    if not train.exists():
        return {}
    pdir = projects.project_dir(p["id"], p["slug"])
    preprocess_manifest.ensure_train_manifest(pdir, v["label"])
    tm = preprocess_manifest.train_load(pdir, v["label"])
    entries = tm.get("images", {})

    out: dict[str, list[dict[str, Any]]] = {}
    for sub in sorted(train.iterdir()):
        if not sub.is_dir():
            continue
        # The physical directory determines the displayed set (physically-deleted
        # duplicate_removed images don't appear), plus compatibility with the old
        # path (copy_to_train doesn't write to the manifest, but physical images
        # are still picked up by the scan). The manifest is only used to look up
        # origin -> dedup by origin (collapsing multi-crop fan-out into one row).
        items_by_origin: dict[str, dict[str, Any]] = {}
        for raw in _list_image_entries(sub):
            rel = f"{sub.name}/{raw['name']}"
            entry = entries.get(rel, {})
            origin = preprocess_manifest.entry_origin(entry, raw["name"])
            if origin in items_by_origin:
                continue
            items_by_origin[origin] = {
                "name": origin,
                "origin": origin,
                "mtime": raw["mtime"],
            }
        out[sub.name] = sorted(items_by_origin.values(), key=lambda e: e["name"])
    return out


def list_validation(
    conn, project_id: int, version_id: int
) -> list[dict[str, Any]]:
    """Flattens images from every subfolder under validation/ into a single flat
    list (both manually added images and ones moved in by auto-split), each entry
    `{name, mtime, folder}`.

    validation has no manifest (the held-out set is read-only, and identity is
    distinguished by directory location -- see the eval_validation module), so
    this doesn't look up origin or dedupe; `name` is just the physical filename.
    `folder` lets the frontend locate thumbnails (needed by the version thumb's
    validation bucket) and perform precise deletion (a multi-select may span
    different auto-split repeat folders).
    """
    _, _, val = _version_validation_dir(conn, project_id, version_id)
    if not val.exists():
        return []
    out: list[dict[str, Any]] = []
    for sub in sorted(p for p in val.iterdir() if p.is_dir()):
        for e in _list_image_entries(sub):
            out.append({"name": e["name"], "mtime": e["mtime"], "folder": sub.name})
    return out


def _used_names(train: dict[str, list[dict[str, Any]]],
                val: list[dict[str, Any]]) -> set[str]:
    """The set of download names already assigned to train (by origin) or validation.

    Held-out requires that an image can't be in both train and validation at once,
    or eval would be measuring memorization rather than generalization. The
    left-column candidates subtract this set from download; both buckets draw from
    the same pool.
    """
    used = {e["name"] for files in train.values() for e in files}
    used |= {e["name"] for e in val}
    return used


def curation_view(conn, project_id: int, version_id: int) -> dict[str, Any]:
    """Used by the frontend: left = download minus train minus validation, right =
    train grouped by folder.

    Each file is returned as `{name, mtime}`; the frontend uses mtime to offer
    "sort by time". The actual byte path on the left is decided by the resolver
    (processed images use the preprocess/ copy, unprocessed ones use the
    original); the frontend fetches it through the project thumbnail endpoint and
    is unaware of the difference.

    left also subtracts validation: after training, auto-split moves images into
    validation/, but those images still remain in download/ -- the old logic would
    let them resurface as train left-column candidates -> they could get re-added
    to train -> overlapping with validation and leaking held-out data. Subtracting
    validation both fixes this and lets the train / validation curation views
    share the same candidate pool.
    """
    left = list_download(conn, project_id)
    train = list_train(conn, project_id, version_id)
    val = list_validation(conn, project_id, version_id)
    used = _used_names(train, val)
    return {
        "left": [e for e in left if e["name"] not in used],
        "right": train,
        # download_total keeps its historical meaning: total left-side candidate
        # count (kept for API compatibility)
        "download_total": len(left),
        "train_total": sum(len(v) for v in train.values()),
        "folders": list(train.keys()),
    }


def curation_validation_view(
    conn, project_id: int, version_id: int
) -> dict[str, Any]:
    """Curation view for the validation set: left = download minus train minus
    validation (same pool as the training set), right = the flat validation list.

    Symmetric with `curation_view`; the only difference is that right is flat (no
    folder concept, see `list_validation`). Used by the frontend to render the
    right column in validation-set mode.
    """
    left = list_download(conn, project_id)
    train = list_train(conn, project_id, version_id)
    val = list_validation(conn, project_id, version_id)
    used = _used_names(train, val)
    return {
        "left": [e for e in left if e["name"] not in used],
        "right": val,
        "download_total": len(left),
        "val_total": len(val),
    }


# ---------------------------------------------------------------------------
# copy / remove
# ---------------------------------------------------------------------------


_META_EXTS = (".txt", ".json")


def copy_download_to_train(
    conn,
    project_id: int,
    version_id: int,
    files: list[str],
    dest_folder: str,
) -> dict[str, list[str]]:
    """ADR 0010 train scope (PR-2 step C): plain download -> train copy + writes a
    train manifest entry. A simplified replacement for `copy_to_train`; the old one
    is removed in PR-3.

    Differences from the old `copy_to_train`:

    - **The preprocess-derivative branch is removed** -- bytes always come from
      `download/{name}`
    - Writes a train manifest entry with key = `f"{dest_folder}/{name}"`,
      origin = name (at the curate stage the image is still the unprocessed
      original; a later in-place preprocess pass in train/ updates the entry)
    - Caption files (.txt/.json) are still copied from `download/{stem}.{ext}` to
      `train/{dest_folder}/{stem}.{ext}`
    - Does not consume or know about preprocess derivatives (under the new model,
      multi-crop fan-out happens during the preprocess phase, after curation)

    `files` is a flat list of image names from the download pool, without a folder
    prefix.
    """
    _validate_folder(dest_folder)
    p, v, train = _version_train_dir(conn, project_id, version_id)
    pdir = projects.project_dir(p["id"], p["slug"])
    download_dir = pdir / "download"
    dst_dir = train / dest_folder
    dst_dir.mkdir(parents=True, exist_ok=True)
    preprocess_manifest.ensure_train_manifest(pdir, v["label"])

    copied: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    for name in files:
        _validate_filename(name)
        src = download_dir / name
        if not src.exists():
            missing.append(name)
            continue
        dst = dst_dir / name
        if dst.exists():
            skipped.append(name)
            continue
        shutil.copy2(src, dst)
        # carry along caption metadata
        stem = Path(name).stem
        for ext in _META_EXTS:
            sm = download_dir / f"{stem}{ext}"
            if sm.exists():
                try:
                    shutil.copy2(sm, dst_dir / f"{stem}{ext}")
                except OSError:
                    pass
        # write the train manifest entry, key = "{folder}/{name}"
        rel = f"{dest_folder}/{name}"
        meta: dict[str, Any] = {"origin": name}
        try:
            st = dst.stat()
            meta["mtime"] = st.st_mtime
            meta["size"] = st.st_size
        except OSError:
            pass
        preprocess_manifest.train_add_processed(pdir, v["label"], rel, meta)
        copied.append(name)
    return {"copied": copied, "skipped": skipped, "missing": missing}


def remove_from_train(
    conn,
    project_id: int,
    version_id: int,
    folder: str,
    files: list[str],
) -> dict[str, list[str]]:
    """Delete every train derivative of a download original from train/{folder}/,
    plus any same-stem metadata; download is left untouched.

    ADR 0010 fixup (2026-06-04): `files` is a list of **origin names** (download
    filenames), matching the `name` field returned by list_train. This function
    looks up the train manifest for every entry whose origin matches, and deletes
    their physical train files + manifest entries + same-stem caption
    (.txt/.json). This way, deleting one row deletes every derivative of that
    original within train (multi-crop fan-out is cleaned up together).
    """
    _validate_folder(folder)
    p, v, train = _version_train_dir(conn, project_id, version_id)
    pdir = projects.project_dir(p["id"], p["slug"])
    fdir = train / folder
    preprocess_manifest.ensure_train_manifest(pdir, v["label"])
    tm = preprocess_manifest.train_load(pdir, v["label"])
    entries = tm.get("images", {})

    # origin -> [rel paths in this folder]
    by_origin: dict[str, list[str]] = {}
    for rel, entry in entries.items():
        if "/" not in rel:
            continue
        f, filename = rel.split("/", 1)
        if f != folder:
            continue
        origin = preprocess_manifest.entry_origin(entry, filename)
        by_origin.setdefault(origin, []).append(rel)

    removed: list[str] = []
    missing: list[str] = []
    rels_to_pop: list[str] = []
    for origin_name in files:
        _validate_filename(origin_name)
        rels = by_origin.get(origin_name, [])
        if not rels:
            # not recorded in the manifest -> fall back to deleting fdir / origin_name
            # directly (a same-name scenario from legacy projects)
            pp = fdir / origin_name
            if pp.exists():
                pp.unlink()
                for ext in _META_EXTS:
                    mp = pp.with_suffix(ext)
                    if mp.exists():
                        try:
                            mp.unlink()
                        except OSError:
                            pass
                train_masks.delete_mask(train, f"{folder}/{origin_name}")
                removed.append(origin_name)
            else:
                missing.append(origin_name)
            continue
        # delete every derivative's physical file + each derivative stem's metadata
        # + mask sidecar
        for rel in rels:
            _, filename = rel.split("/", 1)
            pp = fdir / filename
            if pp.exists():
                try:
                    pp.unlink()
                except OSError:
                    pass
            for ext in _META_EXTS:
                mp = pp.with_suffix(ext)
                if mp.exists():
                    try:
                        mp.unlink()
                    except OSError:
                        pass
            train_masks.delete_mask(train, rel)
        rels_to_pop.extend(rels)
        removed.append(origin_name)

    if rels_to_pop:
        preprocess_manifest.train_remove_entries(
            pdir, v["label"], rels_to_pop,
        )
    return {"removed": removed, "missing": missing}


# ---------------------------------------------------------------------------
# validation copy / remove (manual maintenance of the held-out validation set)
#
# Symmetric with train's copy/remove, but validation has no manifest (identity
# in the held-out set is distinguished by directory location), so it's simpler:
# plain physical copy / delete, with no manifest writes and no origin dedup.
# Manually added images always land in the fixed `validation/1_data/` folder
# (reusing DEFAULT_TRAIN_FOLDER, matching the common auto-split destination); the
# UI doesn't expose the folder concept.
# ---------------------------------------------------------------------------


def copy_download_to_validation(
    conn,
    project_id: int,
    version_id: int,
    files: list[str],
) -> dict[str, list[str]]:
    """Copy from download -> `validation/1_data/` (along with caption .txt/.json).

    `files` is a flat list of image names from the download pool (no folder
    prefix). Any name already present in train (by origin) or validation is
    always skipped -- to prevent held-out leakage, matching the same exclusion
    set used for `curation_view`'s left column. Validation captions are used as
    generation prompts by eval (see eval_samples), so the sidecar files must be
    copied along with them.
    """
    p, v, val = _version_validation_dir(conn, project_id, version_id)
    pdir = projects.project_dir(p["id"], p["slug"])
    download_dir = pdir / "download"
    dst_dir = val / versions.DEFAULT_TRAIN_FOLDER
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Already-assigned set (train origins union validation names) -- same
    # exclusion rule as curation_view.
    train = list_train(conn, project_id, version_id)
    existing_val = list_validation(conn, project_id, version_id)
    used = _used_names(train, existing_val)

    copied: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    for name in files:
        _validate_filename(name)
        if name in used:
            skipped.append(name)
            continue
        src = download_dir / name
        if not src.exists():
            missing.append(name)
            continue
        dst = dst_dir / name
        if dst.exists():
            skipped.append(name)
            continue
        shutil.copy2(src, dst)
        stem = Path(name).stem
        for ext in _META_EXTS:
            sm = download_dir / f"{stem}{ext}"
            if sm.exists():
                try:
                    shutil.copy2(sm, dst_dir / f"{stem}{ext}")
                except OSError:
                    pass
        used.add(name)
        copied.append(name)
    return {"copied": copied, "skipped": skipped, "missing": missing}


def remove_from_validation(
    conn,
    project_id: int,
    version_id: int,
    items: list[dict[str, str]],
) -> dict[str, list[str]]:
    """Delete the given images from validation/ (along with same-stem captions);
    download is left untouched.

    `items` is `[{"folder": ..., "name": ...}]`: a multi-select may span
    different auto-split repeat folders, so deletion is located precisely by
    (folder, name) rather than by name across all folders.
    """
    _, _, val = _version_validation_dir(conn, project_id, version_id)
    removed: list[str] = []
    missing: list[str] = []
    for item in items:
        folder = item.get("folder", "")
        name = item.get("name", "")
        _validate_folder(folder)
        _validate_filename(name)
        pp = val / folder / name
        if not pp.exists():
            missing.append(name)
            continue
        try:
            pp.unlink()
        except OSError:
            missing.append(name)
            continue
        for ext in _META_EXTS:
            mp = pp.with_suffix(ext)
            if mp.exists():
                try:
                    mp.unlink()
                except OSError:
                    pass
        removed.append(name)
    return {"removed": removed, "missing": missing}


# ---------------------------------------------------------------------------
# folder ops
# ---------------------------------------------------------------------------


def create_folder(conn, project_id: int, version_id: int, name: str) -> Path:
    _validate_folder(name)
    _, _, train = _version_train_dir(conn, project_id, version_id)
    target = train / name
    if target.exists():
        raise CurationError(
            f'Folder "{name}" already exists',
            code="curation.folder_exists", details={"name": name},
        )
    target.mkdir(parents=True, exist_ok=False)
    return target


def rename_folder(
    conn, project_id: int, version_id: int, name: str, new_name: str
) -> Path:
    _validate_folder(name)
    _validate_folder(new_name)
    if name == new_name:
        return _version_train_dir(conn, project_id, version_id)[2] / name
    _, _, train = _version_train_dir(conn, project_id, version_id)
    src = train / name
    dst = train / new_name
    if not src.exists():
        raise CurationError(
            f'Folder "{name}" not found',
            code="curation.folder_not_found", details={"name": name},
            http_status=404,
        )
    if dst.exists():
        raise CurationError(
            f'Folder "{new_name}" already exists',
            code="curation.folder_exists", details={"name": new_name},
        )
    src.rename(dst)
    return dst


def _version_reg_dir(conn, project_id: int, version_id: int) -> tuple[
    dict[str, Any], dict[str, Any], Path
]:
    p = projects.get_project(conn, project_id)
    if not p:
        raise CurationError(f"Project not found: id={project_id}")
    v = versions.get_version(conn, version_id)
    if not v or v["project_id"] != project_id:
        raise CurationError(f"Version not found: id={version_id}")
    reg_dir = versions.version_dir(p["id"], p["slug"], v["label"]) / "reg"
    return p, v, reg_dir


def rename_reg_folder(
    conn, project_id: int, version_id: int, name: str, new_name: str
) -> Path:
    """Rename a subfolder under reg/ (e.g. changing a Kohya repeat prefix from
    2_data to 1_data).

    Structurally the same as rename_folder (train/), just operating on reg/
    instead. The UI uses it at the "reg images generated" step, to match the
    renaming experience from Step 1's train folders.
    """
    _validate_folder(name)
    _validate_folder(new_name)
    if name == new_name:
        return _version_reg_dir(conn, project_id, version_id)[2] / name
    _, _, reg = _version_reg_dir(conn, project_id, version_id)
    src = reg / name
    dst = reg / new_name
    if not src.exists():
        raise CurationError(f"Folder not found: {name}")
    if dst.exists():
        raise CurationError(f"The target already exists: {new_name}")
    src.rename(dst)
    return dst


def delete_folder(conn, project_id: int, version_id: int, name: str) -> None:
    """Delete the entire subfolder along with its train copies inside; download is
    left untouched."""
    _validate_folder(name)
    _, _, train = _version_train_dir(conn, project_id, version_id)
    target = train / name
    if not target.exists():
        raise CurationError(
            f'Folder "{name}" not found',
            code="curation.folder_not_found", details={"name": name},
            http_status=404,
        )
    shutil.rmtree(target)


# ---------------------------------------------------------------------------
# stage hint
# ---------------------------------------------------------------------------


def has_train_images(
    conn, project_id: int, version_id: int
) -> bool:
    """Whether this version's train/ already has any images (in any subfolder)."""
    _, _, train = _version_train_dir(conn, project_id, version_id)
    if not train.exists():
        return False
    for sub in train.iterdir():
        if sub.is_dir() and any(
            f.is_file() and f.suffix.lower() in IMAGE_EXTS
            for f in sub.iterdir()
        ):
            return True
    return False
