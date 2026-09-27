"""Preprocessing business layer: listing / status / starting jobs / restore.

Phase one only does "upscaling", but the directory contract and interfaces already reserve room for crop / inpaint.

Data model (ADR 0004)
-------------------
`projects/{id}-{slug}/preprocess/manifest.json` is the single source of truth for status:

    {"images": {"bar.png": {"kind": "processed", "model": "...", "scale": 4, ...}}}

- not recorded in the manifest -> default = use the original image in download/
- `kind: processed` -> preprocess/{name}.png is a modified copy

Downstream consumers (curation / thumbnail / copy_to_train) get the actual file path via
`studio.services.preprocess_manifest.resolve()`; this module is only responsible for
**listing image status + starting jobs + restore**.

Output filename rule: always `{src_stem}.png`. When source images share a stem but differ in extension
(e.g. both `cat.jpg` and `cat.png` exist) -- the later-processed one overwrites the former, with a warning logged.

Job scheduling
--------
preprocess is a GPU-bound job kind, using the DATA slot:
- light tier: while training is running, gated by the `queue.light_tasks_during_train` switch (on by default)
- if the daemon is holding VRAM -> triggers yielding (_maybe_yield_daemon), waits for the next tick

Doesn't reuse download_worker's concurrency design -- serial processing is enough; once the model is loaded on GPU,
a single image takes 1-3s (4x, 512px input, cuda). The benefit of batch concurrency doesn't outweigh the VRAM risk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from ...services.projects import jobs as project_jobs, projects
from ...services.dataset.scan import IMAGE_EXTS
from . import manifest as preprocess_manifest
from . import masks as train_masks


PREPROCESS_KIND = "preprocess"
# Within the same kind, params['stage'] dispatches to different worker branches. Default 'upscale' for compat with
# legacy jobs (treated as upscale when params lacks stage).
STAGE_UPSCALE = "upscale"
STAGE_CROP = "crop"
DEFAULT_MODEL = "4x-AnimeSharp"
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_PAD = 16
DEFAULT_DEVICE = "auto"
# Target area for LoRA training buckets. 1024^2 = 1048576 px is a common SDXL/Flux/Anima bucket; the user
# can pick 768^2/1024^2/1536^2/2048^2 in the UI, or a custom side length.
DEFAULT_TARGET_AREA = 1024 * 1024

PRODUCT_SUFFIX = ".png"
# Minimum normalized side length for a crop box; smaller than this on the canvas doesn't count as valid (avoids accidental zero-pixel images)
MIN_CROP_NORM = 0.02


from studio.domain.errors import DomainError, InvalidPathError, NotFoundError, ValidationError


class PreprocessError(DomainError):
    """Preprocessing business error (project not found / invalid params / invalid filename).

    PR-2 C3 added a DomainError base -- the handler auto-translates it into the dual-write envelope.
    """
    default_code = "preprocess.error"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def project_paths(p: dict[str, Any]) -> tuple[Path, Path]:
    """Returns `(download_dir, preprocess_dir)`; existence is not guaranteed."""
    pdir = projects.project_dir(p["id"], p["slug"])
    return pdir / "download", pdir / "preprocess"


def project_root(p: dict[str, Any]) -> Path:
    """Project root directory (manifest paths are based on this)."""
    return projects.project_dir(p["id"], p["slug"])


def product_path_for(preprocess_dir: Path, source_name: str) -> Path:
    """`download/foo.webp` -> `preprocess/foo.png`."""
    stem = Path(source_name).stem
    return preprocess_dir / f"{stem}{PRODUCT_SUFFIX}"


# ---------------------------------------------------------------------------
# Listing / status (based on the manifest)
# ---------------------------------------------------------------------------


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _download_images(download: Path) -> list[Path]:
    if not download.exists():
        return []
    return sorted([f for f in download.iterdir() if _is_image(f)])


# ---------------------------------------------------------------------------
# Target selection + start
# ---------------------------------------------------------------------------


_SAFE_NAME_FORBIDDEN = ("/", "\\", "..")


def _validate_name(name: str) -> None:
    if not name or any(t in name for t in _SAFE_NAME_FORBIDDEN):
        raise InvalidPathError("Invalid path", details={"name": name})


def _validate_rel_name(name: str) -> None:
    """ADR 0010 train-scope name validation: must look like `"folder/image"` (POSIX form).

    Strictly 2 segments; rejects `..` / backslashes / absolute paths / empty segments, to prevent path traversal.
    """
    if not name:
        raise InvalidPathError("Invalid path", details={"name": name})
    if "\\" in name or name.startswith("/"):
        raise InvalidPathError("Invalid path", details={"name": name})
    parts = name.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1] or ".." in parts:
        raise InvalidPathError("Invalid path", details={"name": name})


def _validate_rect(rect: dict[str, Any]) -> dict[str, float]:
    """Normalize + clamp a crop rect. Invalid -> raises PreprocessError."""
    try:
        x = float(rect["x"])
        y = float(rect["y"])
        w = float(rect["w"])
        h = float(rect["h"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(
            "Invalid crop region",
            code="preprocess.crop_rect_invalid", http_status=400,
        ) from exc
    # Clamp to [0,1], but still enforce the w/h lower-bound check
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    if w < MIN_CROP_NORM or h < MIN_CROP_NORM:
        raise ValidationError(
            "Crop region is too small",
            code="preprocess.crop_too_small", http_status=400,
        )
    return {"x": x, "y": y, "w": w, "h": h}


# ---------------------------------------------------------------------------
# ADR 0010 -- train-scope listing / status / job
#
# New code uses the *_train family:
#
# - list_train_images:   lists all images in train/ + manifest metadata
# - summary_train:       brief stats for the train scope
# - resolve_targets_train / start_job_train / start_crop_job_train: job creation
# - list_crop_workspace_train / list_duplicate_removed_workspace_train: sub-page workspaces
# - restore_products_train: calls manifest.train_restore (semantics: copy download -> train)
# ---------------------------------------------------------------------------


def version_train_dir(p: dict[str, Any], version_label: str) -> Path:
    return project_root(p) / "versions" / version_label / "train"


def _train_images_listing(train_dir: Path) -> list[tuple[str, Path]]:
    """Recursively collects `(rel_path, full_path)` from train_dir's first-level sub-folders (LoRA repeat folders).

    rel_path is in POSIX form `"{folder}/{image}"`, matching the manifest entry key.
    Images placed directly at the root of train_dir are ignored (LoRA training only reads inside sub-folders).
    Output is stable, sorted lexicographically by rel_path.
    """
    if not train_dir.exists():
        return []
    out: list[tuple[str, Path]] = []
    for sub in train_dir.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if not _is_image(f):
                continue
            out.append((f"{sub.name}/{f.name}", f))
    out.sort(key=lambda t: t[0])
    return out


def list_train_images(
    p: dict[str, Any], version_label: str
) -> list[dict[str, Any]]:
    """Lists all images under `versions/{vlabel}/train/` + manifest entry metadata.

    Replaces the old binary concept of `list_pending + list_processed` -- under the new model, train/ IS the "training
    grid", with no pending/processed distinction (status is inferred implicitly from field differences, see ADR 0010
    section "Manifest schema v2").

    Returns `[{name, mtime, size, w, h, origin, source, orphan, duplicate_removed,
    model, scale, action, target_area, src_size, dst_size, elapsed_seconds}]`:
    - `origin / source`: both filled with `entry.origin` (kept for compat with the old frontend field name source)
    - `orphan`: `download/{origin}` is missing (restore will report no_origin)
    - `duplicate_removed`: bool (defaults to False; the UI distinguishes "included in training" vs "skipped during review")
    - legacy schema pass-through fields (model/scale/...) are always None for new entries; the frontend tolerates this

    Edge case: the manifest marks duplicate_removed but the file under train/ was physically deleted (deleted externally by the user) ->
    a stale entry is still reported; the UI tolerates w/h being None.
    """
    from PIL import Image

    pdir = project_root(p)
    download_dir = pdir / "download"
    train_dir = version_train_dir(p, version_label)

    m = preprocess_manifest.train_load(pdir, version_label)
    entries = m["images"]

    download_names = (
        {f.name for f in _download_images(download_dir)}
        if download_dir.exists() else set()
    )

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rel, f in _train_images_listing(train_dir):
        seen.add(rel)
        entry = entries.get(rel, {})
        origin = preprocess_manifest.entry_origin(entry, rel)
        st = f.stat()
        w: Optional[int] = None
        h: Optional[int] = None
        try:
            with Image.open(f) as im:
                w, h = im.size
        except (OSError, ValueError):
            pass
        is_dup = preprocess_manifest.is_duplicate_removed_entry(entry)
        items.append({
            "name": rel,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "w": w, "h": h,
            "origin": origin,
            "source": origin,
            "orphan": origin not in download_names,
            "duplicate_removed": is_dup,
            # ADR 0010 fixup (2026-06-04): reads the manifest entry.processed field directly
            "processed": not is_dup and _is_processed(entry),
            "model": entry.get("model"),
            "scale": entry.get("scale"),
            "action": entry.get("action"),
            "target_area": entry.get("target_area"),
            "src_size": entry.get("src_size"),
            "dst_size": entry.get("dst_size"),
            "elapsed_seconds": entry.get("elapsed_seconds"),
        })

    # stale duplicate_removed entry (present in the manifest, missing from train/ on disk)
    for name, entry in sorted(entries.items()):
        if name in seen:
            continue
        if not preprocess_manifest.is_duplicate_removed_entry(entry):
            continue
        origin = preprocess_manifest.entry_origin(entry, name)
        items.append({
            "name": name,
            "mtime": float(entry.get("mtime", 0.0) or 0.0),
            "size": int(entry.get("size", 0) or 0),
            "w": None, "h": None,
            "origin": origin,
            "source": origin,
            "orphan": origin not in download_names,
            "duplicate_removed": True,
            "processed": False,
            "model": None, "scale": None, "action": None,
            "target_area": None, "src_size": None, "dst_size": None,
            "elapsed_seconds": None,
        })

    return items


def summary_train(p: dict[str, Any], version_label: str) -> dict[str, Any]:
    """Brief stats for the train scope.

    `image_count` = number of physical images in train/ + entries only marked duplicate_removed in the manifest
    and already physically deleted (a rare stale entry, still counted for display).
    """
    pdir = project_root(p)
    train_dir = version_train_dir(p, version_label)
    m = preprocess_manifest.train_load(pdir, version_label)
    physical = {rel for rel, _ in _train_images_listing(train_dir)}
    soft_removed_only = {
        name for name, entry in m["images"].items()
        if preprocess_manifest.is_duplicate_removed_entry(entry)
        and name not in physical
    }
    return {"image_count": len(physical) + len(soft_removed_only)}


def resolve_targets_train(
    p: dict[str, Any], version_label: str, *,
    mode: str, names: Optional[Iterable[str]] = None,
) -> list[str]:
    """Returns the list of image names to process in the current train/ grid, based on mode + names.

    mode='all' / 'all_force' -> all images in train/
    mode='selected'          -> the intersection of the given names with what actually exists in train/
    """
    train_dir = version_train_dir(p, version_label)
    if not train_dir.exists() and mode != "selected":
        return []
    existing = {rel for rel, _ in _train_images_listing(train_dir)}

    if mode in ("all", "all_force"):
        return sorted(existing)
    if mode == "selected":
        if not names:
            raise ValidationError(
                "No images selected",
                code="preprocess.selection_empty", http_status=400,
            )
        chosen: list[str] = []
        for n in names:
            _validate_rel_name(n)
            if n in existing:
                chosen.append(n)
        return sorted(set(chosen))
    raise ValidationError(
        f"Invalid preprocess mode: {mode}",
        code="preprocess.mode_invalid", details={"mode": mode}, http_status=400,
    )


def start_job_train(
    conn, *,
    project_id: int,
    version_id: int,
    mode: str = "all",
    names: Optional[list[str]] = None,
    model: str = DEFAULT_MODEL,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_pad: int = DEFAULT_TILE_PAD,
    device: str = DEFAULT_DEVICE,
    target_area: Optional[int] = DEFAULT_TARGET_AREA,
) -> dict[str, Any]:
    """train scope preprocess job. The worker gets the version label via job.version_id,
    then lists sources and writes outputs from `versions/{label}/train/` (worker changed in PR-2 step D).
    """
    p = projects.get_project(conn, project_id)
    if not p:
        raise NotFoundError(
            "Project not found",
            code="project.not_found", details={"id": project_id},
        )
    if mode not in ("all", "selected", "all_force"):
        raise ValidationError(
            f"Invalid preprocess mode: {mode}",
            code="preprocess.mode_invalid", details={"mode": mode}, http_status=400,
        )
    if mode == "selected" and not names:
        raise ValidationError(
            "No images selected",
            code="preprocess.selection_empty", http_status=400,
        )
    if names:
        for n in names:
            _validate_rel_name(n)

    params: dict[str, Any] = {
        "stage": STAGE_UPSCALE,
        "mode": mode,
        "model": model,
        "tile_size": int(tile_size),
        "tile_pad": int(tile_pad),
        "device": device,
        "target_area": int(target_area) if target_area else None,
    }
    if names:
        params["names"] = list(names)

    return project_jobs.create_job(
        conn,
        project_id=project_id,
        version_id=version_id,
        kind=PREPROCESS_KIND,
        params=params,
    )


def start_crop_job_train(
    conn, *,
    project_id: int,
    version_id: int,
    crops: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """train scope crop job. The source filenames in `crops` are the current filenames under `train/`."""
    p = projects.get_project(conn, project_id)
    if not p:
        raise NotFoundError(
            "Project not found",
            code="project.not_found", details={"id": project_id},
        )
    if not isinstance(crops, dict) or not crops:
        raise ValidationError(
            "No crop regions provided",
            code="preprocess.crops_required", http_status=400,
        )
    sanitized: dict[str, list[dict[str, Any]]] = {}
    for name, rects in crops.items():
        _validate_rel_name(name)
        if not isinstance(rects, list) or not rects:
            raise ValidationError(
                "Invalid crop region",
                code="preprocess.crop_rect_invalid",
                details={"name": name}, http_status=400,
            )
        out_rects: list[dict[str, Any]] = []
        for r in rects:
            if not isinstance(r, dict):
                raise ValidationError(
                    "Invalid crop region",
                    code="preprocess.crop_rect_invalid",
                    details={"name": name}, http_status=400,
                )
            clean = _validate_rect(r)
            label = r.get("label")
            if label is not None:
                clean["label"] = str(label)[:64]
            out_rects.append(clean)
        sanitized[name] = out_rects

    params = {"stage": STAGE_CROP, "crops": sanitized}
    return project_jobs.create_job(
        conn,
        project_id=project_id,
        version_id=version_id,
        kind=PREPROCESS_KIND,
        params=params,
    )


def _is_processed(entry: dict[str, Any]) -> bool:
    """ADR 0010 status inference (2026-06-04 fixup): reads the manifest entry's
    `processed` field directly.

    The worker writes `processed: True` after upscale/crop completes; curate copying the original doesn't write it
    (defaults to False). Legacy entries (without a `processed` field) are always treated as unprocessed --
    the user re-running preprocess upgrades them to the new field.
    """
    return bool(entry.get("processed", False))


def list_crop_workspace_train(
    p: dict[str, Any], version_label: str
) -> list[dict[str, Any]]:
    """Crop-page workspace (train scope): all images in train/, with pixel dimensions + processed flag.

    See `_is_processed` for the determination logic. Images marked duplicate_removed are skipped (so users can't
    crop an already soft-deleted image).
    """
    from PIL import Image

    pdir = project_root(p)
    train_dir = version_train_dir(p, version_label)
    download_dir = pdir / "download"
    m = preprocess_manifest.train_load(pdir, version_label)
    entries = m["images"]
    removed_origins = preprocess_manifest.train_duplicate_removed_origins(
        pdir, version_label
    )

    items: list[dict[str, Any]] = []
    for rel, f in _train_images_listing(train_dir):
        entry = entries.get(rel, {})
        if preprocess_manifest.is_duplicate_removed_entry(entry):
            continue
        origin = preprocess_manifest.entry_origin(entry, rel)
        if origin in removed_origins:
            continue
        try:
            with Image.open(f) as im:
                w, h = im.size
        except (OSError, ValueError):
            continue
        st = f.stat()
        mask_info = train_masks.mask_stat(train_dir, rel)
        items.append({
            "name": rel,
            "source": origin,
            "w": w, "h": h,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "processed": _is_processed(entry),
            # Training mask sidecar: None when there's no mask. The frontend uses it to draw a corner badge and decide
            # whether to GET the mask (the value also doubles as a cache-buster).
            "mask_mtime": mask_info["mtime"] if mask_info else None,
        })
    return items


def list_duplicate_removed_workspace_train(
    p: dict[str, Any], version_label: str
) -> list[dict[str, Any]]:
    """train scope soft-delete workspace (the "Deleted" tab).

    `mark_duplicate_removed` has already deleted the physical file at train/{name}; this function scans manifest tombstones,
    reading thumbnail metadata live from `download/{origin}`; the frontend thumbnail also goes through the download bucket.
    """
    from PIL import Image

    pdir = project_root(p)
    download_dir = pdir / "download"
    removed = preprocess_manifest.train_duplicate_removed(pdir, version_label)

    items: list[dict[str, Any]] = []
    for name in sorted(removed.keys()):
        entry = removed[name]
        origin = preprocess_manifest.entry_origin(entry, name)
        src = download_dir / origin
        if not src.is_file():
            items.append({
                "name": name,
                "source": origin,
                "w": None, "h": None,
                "mtime": float(entry.get("mtime", 0.0) or 0.0),
                "size": int(entry.get("size", 0) or 0),
            })
            continue
        try:
            with Image.open(src) as im:
                w, h = im.size
        except (OSError, ValueError):
            w, h = None, None
        st = src.stat()
        items.append({
            "name": name,
            "source": origin,
            "w": w, "h": h,
            "mtime": st.st_mtime,
            "size": st.st_size,
        })
    return items


def restore_products_train(
    p: dict[str, Any], version_label: str, names: Iterable[str],
) -> dict[str, list[str]]:
    """train scope restore: copies from `download/{entry.origin}` over `train/{name}`.

    Returns three groups: `{restored, missing, no_origin}`. See ADR 0010 section "Restore semantics".
    `no_origin` = the file is physically missing from download; the UI should offer the user three options (drag in a replacement / keep /
    remove from train) instead of hiding the failure.
    """
    pdir = project_root(p)
    name_list: list[str] = []
    for raw in names:
        _validate_rel_name(raw)
        name_list.append(raw)
    return preprocess_manifest.train_restore(pdir, version_label, name_list)


def inpaint_save_train(
    p: dict[str, Any], version_label: str, *, name: str, data: bytes,
) -> dict[str, Any]:
    """Inpaint-whole-image save (train scope): the image exported from the frontend canvas overwrites `train/{name}`.

    The output follows the same `{folder}/{stem}.png` convention as crop; if the source wasn't .png, the old source file is deleted
    (the caption sidecar is left alone since the stem doesn't change). The manifest reuses
    train_replace_with_crops's single-output path (delete the old entry + write a new one,
    processed=True); origin carries over from the old entry.

    The uploaded image must match the existing source's dimensions exactly -- inpainting is pixel-by-pixel editing, so a size mismatch means the frontend's
    stroke-replay target is misaligned; reject outright instead of silently accepting it.
    """
    import io
    import os
    import time

    from PIL import Image

    _validate_rel_name(name)
    pdir = project_root(p)
    train_dir = version_train_dir(p, version_label)
    src_path = train_dir / name
    if not src_path.is_file():
        raise NotFoundError(
            "Image not found in train set",
            code="preprocess.inpaint_source_missing", details={"name": name},
        )
    try:
        with Image.open(src_path) as im:
            src_w, src_h = im.size
    except (OSError, ValueError) as exc:
        raise ValidationError(
            "Source image is unreadable",
            code="preprocess.inpaint_source_unreadable",
            details={"name": name}, http_status=400,
        ) from exc

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # PIL raises inconsistent types on decode failure, normalize to 400
        raise ValidationError(
            "Uploaded image is not a valid image file",
            code="preprocess.inpaint_image_invalid",
            details={"name": name}, http_status=400,
        ) from exc
    if img.size != (src_w, src_h):
        raise ValidationError(
            "Uploaded image size does not match the source image",
            code="preprocess.inpaint_size_mismatch",
            details={
                "name": name,
                "expected": [src_w, src_h],
                "got": [img.size[0], img.size[1]],
            },
            http_status=400,
        )
    if img.mode != "RGB":
        img = img.convert("RGB")

    folder, filename = name.split("/", 1)
    out_rel = f"{folder}/{Path(filename).stem}{PRODUCT_SUFFIX}"
    out_path = train_dir / out_rel

    # origin carries over from the existing manifest entry, otherwise falls back to the source filename (matches crop worker)
    existing = preprocess_manifest.train_get_entry(pdir, version_label, name)
    origin = (
        preprocess_manifest.entry_origin(existing, filename)
        if existing is not None else filename
    )

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    img.save(tmp_path, format="PNG", optimize=False)
    os.replace(tmp_path, out_path)

    if out_rel != name:
        try:
            src_path.unlink()
        except OSError:
            pass

    try:
        st = out_path.stat()
        size, mtime = st.st_size, st.st_mtime
    except OSError:
        size, mtime = 0, time.time()

    preprocess_manifest.train_replace_with_crops(
        pdir, version_label,
        source_name=name,
        outputs=[{"name": out_rel, "origin": origin, "size": size, "mtime": mtime}],
    )

    try:
        from studio.services.dataset import thumb_cache
        thumb_cache.prewarm_from_image(out_path, img, [256, 768])
    except Exception:  # noqa: BLE001
        pass

    return {
        "name": out_rel, "origin": origin,
        "mtime": mtime, "size": size, "w": src_w, "h": src_h,
    }


# ---------------------------------------------------------------------------
# Training mask sidecar (PR-B B1, see services/preprocess/masks.py)
# ---------------------------------------------------------------------------


def _mask_source_size(
    p: dict[str, Any], version_label: str, name: str,
) -> tuple[Path, tuple[int, int]]:
    """Validates the rel name + that the source image exists, returns (train_dir, source image size)."""
    from PIL import Image

    _validate_rel_name(name)
    train_dir = version_train_dir(p, version_label)
    src_path = train_dir / name
    if not src_path.is_file():
        raise NotFoundError(
            "Image not found in train set",
            code="preprocess.mask_source_missing", details={"name": name},
        )
    try:
        with Image.open(src_path) as im:
            return train_dir, im.size
    except (OSError, ValueError) as exc:
        raise ValidationError(
            "Source image is unreadable",
            code="preprocess.mask_source_unreadable",
            details={"name": name}, http_status=400,
        ) from exc


def mask_save_train(
    p: dict[str, Any], version_label: str, *, name: str, data: bytes,
) -> dict[str, Any]:
    """Writes the training mask (grayscale PNG, size must match the source image's current size)."""
    train_dir, size = _mask_source_size(p, version_label, name)
    return train_masks.write_mask(train_dir, name, data, expected_size=size)


def mask_delete_train(
    p: dict[str, Any], version_label: str, *, name: str,
) -> dict[str, Any]:
    """Deletes the training mask (= restores normal full-image learning). Returns ok even if the mask doesn't exist."""
    _validate_rel_name(name)
    train_dir = version_train_dir(p, version_label)
    return {"deleted": train_masks.delete_mask(train_dir, name)}


def mask_file_train(
    p: dict[str, Any], version_label: str, *, name: str,
) -> Optional[Path]:
    """Mask file path (returns None if it doesn't exist). Used by the GET endpoint."""
    _validate_rel_name(name)
    train_dir = version_train_dir(p, version_label)
    return train_masks.mask_file(train_dir, name)
