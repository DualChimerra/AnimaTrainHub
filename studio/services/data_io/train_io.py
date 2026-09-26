"""PP7 -- export / import of the training set (the tagged train/ folder).

Cloud Studio instances are easy to lose; users often spend hours hand-tuning
captions during step 4 (tag editing), and it's the "tagged train/" that they
really don't want to lose. This module packs it into a downloadable zip, and
supports importing it back as a new project.

zip layout (schema_version 1, the legacy train.zip):
    {slug}-{label}.train.zip
    |-- manifest.json    # source / stats
    `-- train/
        `-- {N}_data/    # N is kept as-is
            |-- *.png/...
            `-- *.txt    # optional

zip layout (schema_version 2, bundle.zip):
    {slug}-{label}.bundle.zip
    |-- manifest.json    # source / stats / includes
    |-- train/           # optional: training set
    |   `-- {N}_data/
    |       |-- *.png/...
    |       `-- *.txt    # included or not, per train_captions
    |-- reg/             # optional: regularization set
    |   |-- meta.json    # from reg/meta.json
    |   `-- {folder}/
    |       |-- *.png/...
    |       `-- *.txt    # included or not, per reg_captions
    `-- presets/         # optional: presets
        `-- {name}.yaml

Import semantics: always creates a new project + source.version_label
(falls back to v1 if missing/invalid; never merges into an existing project),
with slug conflicts automatically suffixed with -imported-{ts}.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ...services.projects import projects, versions
from ..dataset.scan import IMAGE_EXTS
from ...paths import safe_join

SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 2
MANIFEST_NAME = "manifest.json"
TRAIN_PREFIX = "train/"
REG_PREFIX = "reg/"
PRESETS_PREFIX = "presets/"
CAPTION_EXTS = {".txt"}
# VAE latent cache (CachedLatentDataset's {name}.npz / {name}.r{reso}.npz, sitting
# right next to the image)
LATENT_CACHE_EXT = ".npz"
# Phase 2 text cache: an image sidecar. The prompt aggregation cache belongs to the
# task's own file tree (tasks/<id>/.text-cache/), not the version tree, and is not
# bundled. We keep reusing the existing train_latent_cache / reg_latent_cache bundle
# options -- both express "carry along a rebuildable training cache" and don't change
# the default export size.
TEXT_CACHE_SIDECAR_SUFFIX = ".text.safetensors"
# Training mask sidecar (train/{folder}/{stem}.mask, alongside the image; see
# services/preprocess/masks.py). Its arcname mirrors the image's two-part path.
MASK_SUFFIX = ".mask"


VERSION_CONFIG_ARC = "presets/config.yaml"


@dataclass
class BundleOptions:
    train: bool = True
    train_captions: bool = True
    reg: bool = False
    reg_captions: bool = False
    # True = export this version's private config.yaml (with path fields stripped)
    # as a portable training config
    include_config: bool = False
    # True = also bundle the corresponding directory's VAE latent cache (*.npz), so
    # re-encoding isn't needed after import
    train_latent_cache: bool = False
    reg_latent_cache: bool = False
    # True = also bundle training mask sidecars (train/{folder}/{stem}.mask, the
    # masked-loss data plane)
    train_masks: bool = False


from studio.domain.errors import DomainError, NotFoundError


class TrainIOError(DomainError):
    """Business errors from the export / import process.

    PR-2 C3 adds the DomainError base -- the handler automatically converts it
    into the dual-write envelope.
    """
    default_code = "train_io.error"


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def export_train(
    conn: sqlite3.Connection, version_id: int, dest: Path
) -> dict[str, Any]:
    """Package a version's train/ + manifest.json into dest (a zip path).

    dest's parent directory must already exist; uses ZIP_STORED (PNG/jpg are
    already compressed).
    Returns {"manifest": {...}, "size_bytes": int}.
    """
    v = versions.get_version(conn, version_id)
    if not v:
        raise NotFoundError(
            "Version not found", code="version.not_found",
            details={"id": version_id},
        )
    p = projects.get_project(conn, v["project_id"])
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found",
            details={"id": v["project_id"]},
        )

    train_dir = versions.version_dir(p["id"], p["slug"], v["label"]) / "train"
    if not train_dir.exists():
        raise TrainIOError(
            "No images to export", code="dataset.export_empty", http_status=400,
        )

    concepts: list[dict[str, Any]] = []
    image_count = 0
    tagged_count = 0
    payload: list[tuple[Path, str]] = []  # (abs_path, arcname)

    for sub in sorted(train_dir.iterdir()):
        if not sub.is_dir():
            continue
        cnt = 0
        for f in sorted(sub.iterdir()):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS:
                cnt += 1
                arc = f"{TRAIN_PREFIX}{sub.name}/{f.name}"
                payload.append((f, arc))
                if f.with_suffix(".txt").exists():
                    tagged_count += 1
            elif ext in CAPTION_EXTS:
                arc = f"{TRAIN_PREFIX}{sub.name}/{f.name}"
                payload.append((f, arc))
        if cnt > 0:
            concepts.append({"folder": sub.name, "image_count": cnt})
            image_count += cnt

    if image_count == 0:
        raise TrainIOError(
            "No images to export", code="dataset.export_empty", http_status=400,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": time.time(),
        "source": {
            "title": p["title"],
            "version_label": v["label"],
            "slug": p["slug"],
        },
        "stats": {
            "image_count": image_count,
            "tagged_count": tagged_count,
            "untagged_count": image_count - tagged_count,
            "concepts": concepts,
        },
    }

    with zipfile.ZipFile(
        dest, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
        for src, arc in payload:
            zf.write(src, arcname=arc)

    return {"manifest": manifest, "size_bytes": dest.stat().st_size}


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


def _safe_arc(name: str) -> Optional[str]:
    """Zip-slip protection: rejects absolute paths / .. segments / anything without
    the train/ prefix.

    Returns the relative "{folder}/{filename}" part with the train/ prefix
    stripped; returns None if invalid. Directory-only entries (ending in /) also
    return None.
    """
    if not name or name.endswith("/"):
        return None
    norm = name.replace("\\", "/")
    if norm.startswith("/") or ".." in norm.split("/"):
        return None
    if not norm.startswith(TRAIN_PREFIX):
        return None
    inner = norm[len(TRAIN_PREFIX) :]
    parts = inner.split("/")
    # Must be exactly {folder}/{filename}; deeper nesting ({folder}/sub/x.png) is rejected
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return inner


def _read_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    try:
        with zf.open(MANIFEST_NAME) as fh:
            return json.loads(fh.read().decode("utf-8"))
    except KeyError as exc:
        raise TrainIOError(
            "Import file is invalid: missing manifest",
            code="dataset.import_invalid", http_status=400,
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TrainIOError(
            f"Import file is invalid: {exc}",
            code="dataset.import_invalid",
            details={"reason": str(exc)}, http_status=400,
        ) from exc


def _resolve_slug_conflict(
    conn: sqlite3.Connection, base: str
) -> str:
    """On a slug conflict, append a -imported-{ts} suffix; if still conflicting,
    append -{n}."""
    if not conn.execute(
        "SELECT 1 FROM projects WHERE slug = ?", (base,)
    ).fetchone():
        return base
    candidate = f"{base}-imported-{int(time.time())}"
    n = 1
    final = candidate
    while conn.execute(
        "SELECT 1 FROM projects WHERE slug = ?", (final,)
    ).fetchone():
        n += 1
        final = f"{candidate}-{n}"
    return final


def import_train(
    conn: sqlite3.Connection, zip_path: Path
) -> dict[str, Any]:
    """Extract a zip into a newly created project + v1, stage=tagging.

    Returns {"project": {...}, "version": {...}, "stats": {...}}.
    """
    if not zip_path.exists():
        raise NotFoundError(
            "File not found", code="file.not_found", http_status=404,
        )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest = _read_manifest(zf)
            source = manifest.get("source") or {}
            title = (source.get("title") or "imported").strip() or "imported"
            base_slug = projects.slugify(source.get("slug") or title)

            # First pass over valid entries; error out immediately on empty content
            # to avoid creating an empty project
            entries: list[tuple[zipfile.ZipInfo, str]] = []
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if info.filename == MANIFEST_NAME:
                    continue
                inner = _safe_arc(info.filename)
                if inner is None:
                    raise TrainIOError(
                        "Import file contains an invalid path",
                        code="dataset.import_invalid", http_status=400,
                    )
                entries.append((info, inner))

            if not entries:
                raise TrainIOError(
                    "Import file contains nothing to import",
                    code="dataset.import_empty", http_status=400,
                )

            # Create the project + v1 (via the DAO, which builds the directory tree
            # automatically)
            slug = _resolve_slug_conflict(conn, base_slug)
            note = f"imported from {title!r}"
            p = projects.create_project(conn, title=title, slug=slug, note=note)
            v = versions.create_version(
                conn, project_id=p["id"], label="v1"
            )

            vdir = versions.version_dir(p["id"], p["slug"], v["label"])
            train_dir = vdir / "train"
            seen_folders: set[str] = set()

            for info, inner in entries:
                folder, filename = inner.split("/", 1)
                # An extra layer of filename safety + containment check
                if filename.startswith("."):
                    raise TrainIOError(
                        "Import file contains an invalid filename",
                        code="dataset.import_invalid", http_status=400,
                    )
                try:
                    target = safe_join(train_dir, folder, filename)
                except ValueError as exc:
                    raise TrainIOError(
                        "Import file contains an invalid filename",
                        code="dataset.import_invalid",
                        details={"reason": str(exc)}, http_status=400,
                    ) from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                seen_folders.add(folder)
                with zf.open(info) as src, target.open("wb") as dst:
                    while True:
                        chunk = src.read(64 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

            # Count tagged_count only after everything has been written -- zip entry
            # order isn't guaranteed, so counting while writing could undercount
            # (e.g. getting 0 if .png comes before .txt)
            image_count = 0
            tagged_count = 0
            for folder in seen_folders:
                fdir = train_dir / folder
                if not fdir.exists():
                    continue
                for f in fdir.iterdir():
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                        image_count += 1
                        if f.with_suffix(".txt").exists():
                            tagged_count += 1

            # ADR-0007 PR-5: stage is no longer advanced automatically; the phase
            # cursor is advanced explicitly by the user via PhaseHeaderNav
            v = versions.get_version(conn, v["id"])
            p = projects.get_project(conn, p["id"])
            assert v is not None and p is not None

    except zipfile.BadZipFile as exc:
        raise TrainIOError(
            "Import file is corrupt", code="dataset.import_corrupt",
            http_status=400,
        ) from exc

    stats = {
        "image_count": image_count,
        "tagged_count": tagged_count,
        "untagged_count": image_count - tagged_count,
        "concepts": sorted(seen_folders),
    }
    return {"project": p, "version": v, "stats": stats}


# ---------------------------------------------------------------------------
# bundle export / import (schema_version 2)
# ---------------------------------------------------------------------------


def _collect_train(
    train_dir: Path, include_captions: bool, include_latent_cache: bool = False,
    include_masks: bool = False,
) -> tuple[list[tuple[Path, str]], dict[str, Any]]:
    """Scan the train/ directory and return (payload, stats_dict).

    When include_masks is set, also collects the `{stem}.mask` sidecar found
    alongside each image.
    """
    payload: list[tuple[Path, str]] = []
    concepts: list[dict[str, Any]] = []
    image_count = 0
    tagged_count = 0
    latent_cache_count = 0
    text_cache_count = 0
    mask_count = 0

    if not train_dir.exists():
        return payload, {
            "image_count": 0, "tagged_count": 0, "concepts": [],
            "latent_cache_count": 0, "text_cache_count": 0, "mask_count": 0,
        }

    for sub in sorted(train_dir.iterdir()):
        if not sub.is_dir():
            continue
        cnt = 0
        for f in sorted(sub.iterdir()):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS:
                cnt += 1
                payload.append((f, f"{TRAIN_PREFIX}{sub.name}/{f.name}"))
                if f.with_suffix(".txt").exists():
                    tagged_count += 1
                    if include_captions:
                        txt = f.with_suffix(".txt")
                        payload.append((txt, f"{TRAIN_PREFIX}{sub.name}/{txt.name}"))
            elif ext == LATENT_CACHE_EXT and include_latent_cache:
                latent_cache_count += 1
                payload.append((f, f"{TRAIN_PREFIX}{sub.name}/{f.name}"))
            elif include_latent_cache and f.name.endswith(
                TEXT_CACHE_SIDECAR_SUFFIX
            ):
                text_cache_count += 1
                payload.append((f, f"{TRAIN_PREFIX}{sub.name}/{f.name}"))
            elif ext == MASK_SUFFIX and include_masks:
                mask_count += 1
                payload.append((f, f"{TRAIN_PREFIX}{sub.name}/{f.name}"))
            elif ext in CAPTION_EXTS and include_captions:
                # .txt is already included from the image branch above; skip here
                # to avoid duplicates
                pass
        if cnt > 0:
            concepts.append({"folder": sub.name, "image_count": cnt})
            image_count += cnt

    return payload, {
        "image_count": image_count,
        "tagged_count": tagged_count,
        "concepts": concepts,
        "latent_cache_count": latent_cache_count,
        "text_cache_count": text_cache_count,
        "mask_count": mask_count,
    }


def _collect_reg(
    reg_dir: Path, include_captions: bool, include_latent_cache: bool = False
) -> tuple[list[tuple[Path, str]], dict[str, Any]]:
    """Scan the reg/ directory and return (payload, stats_dict)."""
    payload: list[tuple[Path, str]] = []
    image_count = 0
    latent_cache_count = 0
    text_cache_count = 0

    if not reg_dir.exists():
        return payload, {
            "image_count": 0, "latent_cache_count": 0, "text_cache_count": 0,
        }

    # meta.json
    meta = reg_dir / "meta.json"
    if meta.exists():
        payload.append((meta, f"{REG_PREFIX}meta.json"))

    for item in sorted(reg_dir.iterdir()):
        if item.name == "meta.json":
            continue
        if item.is_dir():
            for f in sorted(item.iterdir()):
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext in IMAGE_EXTS:
                    image_count += 1
                    payload.append((f, f"{REG_PREFIX}{item.name}/{f.name}"))
                    if include_captions and f.with_suffix(".txt").exists():
                        txt = f.with_suffix(".txt")
                        payload.append((txt, f"{REG_PREFIX}{item.name}/{txt.name}"))
                elif ext == LATENT_CACHE_EXT and include_latent_cache:
                    latent_cache_count += 1
                    payload.append((f, f"{REG_PREFIX}{item.name}/{f.name}"))
                elif (
                    include_latent_cache
                    and f.name.endswith(TEXT_CACHE_SIDECAR_SUFFIX)
                ):
                    text_cache_count += 1
                    payload.append((f, f"{REG_PREFIX}{item.name}/{f.name}"))
                elif ext in CAPTION_EXTS and include_captions:
                    pass  # already included from the image branch
        elif item.is_file() and item.suffix.lower() in IMAGE_EXTS:
            # An image placed directly under reg/'s root (non-standard but
            # supported for compatibility). Its neighboring npz isn't collected:
            # a bare reg/{file} entry is only ever allowed through
            # _safe_arc_bundle for meta.json.
            image_count += 1
            payload.append((item, f"{REG_PREFIX}{item.name}"))

    return payload, {
        "image_count": image_count,
        "latent_cache_count": latent_cache_count,
        "text_cache_count": text_cache_count,
    }


def export_bundle(
    conn: sqlite3.Connection,
    version_id: int,
    dest: Path,
    opts: BundleOptions,
) -> dict[str, Any]:
    """Package bundle.zip to dest according to BundleOptions.

    At least one of train / reg / include_config must be selected, otherwise
    raises TrainIOError.
    Returns {"manifest": {...}, "size_bytes": int}.
    """
    if not opts.train and not opts.reg and not opts.include_config:
        raise TrainIOError(
            "Select at least one item to export (training set, regularization "
            "set, or training configuration)",
            code="dataset.export_nothing_selected", http_status=400,
        )

    v = versions.get_version(conn, version_id)
    if not v:
        raise NotFoundError(
            "Version not found", code="version.not_found",
            details={"id": version_id},
        )
    p = projects.get_project(conn, v["project_id"])
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found",
            details={"id": v["project_id"]},
        )

    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    payload: list[tuple[Path, str]] = []
    in_memory: list[tuple[str, str]] = []  # (content_str, arcname) for small generated files

    # --- train section ---
    train_stats: dict[str, Any] = {}
    if opts.train:
        tp, train_stats = _collect_train(
            vdir / "train", opts.train_captions, opts.train_latent_cache,
            include_masks=opts.train_masks,
        )
        if not tp:
            raise TrainIOError(
                "No images to export", code="dataset.export_empty",
                http_status=400,
            )
        payload.extend(tp)

    # --- reg section ---
    reg_stats: dict[str, Any] = {}
    if opts.reg:
        rp, reg_stats = _collect_reg(
            vdir / "reg", opts.reg_captions, opts.reg_latent_cache
        )
        payload.extend(rp)

    # --- the version's private training config ---
    # Exports the version's own config.yaml with PROJECT_SPECIFIC_FIELDS (path
    # fields) stripped out, keeping portable content like hyperparameters. On
    # import, write_version_config(force_project_overrides=True) automatically
    # fills in the correct paths for the target environment.
    config_included = False
    if opts.include_config:
        from .. import version_config as _vc
        from ...domain.config_prune import prune_inactive_fields
        import yaml as _yaml
        try:
            # read_version_config backfills every default field via pydantic; to
            # match the on-disk yaml, which only carries active fields, we prune
            # once more here so the bundle's config does the same.
            cfg = _vc.read_version_config(p, v)
            portable = prune_inactive_fields(
                {k: v_ for k, v_ in cfg.items() if k not in _vc.PROJECT_SPECIFIC_FIELDS}
            )
            in_memory.append((
                _yaml.safe_dump(portable, allow_unicode=True, sort_keys=False, default_flow_style=False),
                VERSION_CONFIG_ARC,
            ))
            config_included = True
        except _vc.VersionConfigError:
            # the version hasn't had training parameters configured yet; skip it
            # (not an error, just omitted from the bundle)
            pass

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "exported_at": time.time(),
        "source": {
            "title": p["title"],
            "version_label": v["label"],
            "slug": p["slug"],
            "preset_name": v.get("config_name"),
        },
        "includes": {
            "train": opts.train,
            "train_captions": opts.train_captions,
            "reg": opts.reg,
            "reg_captions": opts.reg_captions,
            "config": config_included,
            "train_latent_cache": opts.train_latent_cache,
            "reg_latent_cache": opts.reg_latent_cache,
            "train_masks": opts.train_masks,
        },
        "stats": {
            "train_image_count": train_stats.get("image_count", 0),
            "train_tagged_count": train_stats.get("tagged_count", 0),
            "reg_image_count": reg_stats.get("image_count", 0),
            "config_included": config_included,
            "latent_cache_count": (
                train_stats.get("latent_cache_count", 0)
                + reg_stats.get("latent_cache_count", 0)
            ),
            "text_cache_count": (
                train_stats.get("text_cache_count", 0)
                + reg_stats.get("text_cache_count", 0)
            ),
            "train_mask_count": train_stats.get("mask_count", 0),
        },
    }

    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
        for content, arc in in_memory:
            zf.writestr(arc, content)
        for src, arc in payload:
            zf.write(src, arcname=arc)

    return {"manifest": manifest, "size_bytes": dest.stat().st_size}


def _safe_arc_bundle(name: str) -> Optional[tuple[str, str]]:
    """Zip-slip protection for bundles.

    Returns (section, inner); section is "train" | "reg" | "preset".
    Returns None if invalid.
    """
    if not name or name.endswith("/"):
        return None
    norm = name.replace("\\", "/")
    if norm.startswith("/") or ".." in norm.split("/"):
        return None

    if norm.startswith(TRAIN_PREFIX):
        inner = norm[len(TRAIN_PREFIX):]
        parts = inner.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        return ("train", inner)

    if norm.startswith(REG_PREFIX):
        inner = norm[len(REG_PREFIX):]
        # reg/meta.json -- a single path segment
        if inner == "meta.json":
            return ("reg", inner)
        parts = inner.split("/")
        # reg/{folder}/{file} -- two segments
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        return ("reg", inner)

    if norm.startswith(PRESETS_PREFIX):
        inner = norm[len(PRESETS_PREFIX):]
        parts = inner.split("/")
        if len(parts) != 1 or not inner or not inner.endswith(".yaml"):
            return None
        return ("preset", inner)

    return None


def _bundle_source_version_label(source: dict[str, Any]) -> str:
    """The manifest is untrusted input -- validation goes through the same rule set
    as versions (including rejecting "." / ".."); falls back to v1 on invalid
    input rather than raising, so old / hand-edited bundles can still be
    imported."""
    raw = str(source.get("version_label") or "v1").strip()
    return raw if versions.is_valid_label(raw) else "v1"


def _bundle_source_preset_name(source: dict[str, Any]) -> Optional[str]:
    raw = source.get("preset_name") or source.get("config_name")
    if raw is None:
        return None
    name = str(raw).strip()
    return name or None


def _restore_preset_name(
    conn: sqlite3.Connection,
    version_id: int,
    preset_name: Optional[str],
    presets_base: Path,
) -> Optional[dict[str, Any]]:
    """Only backfills config_name if the preset genuinely exists (either already
    present locally, or just extracted from the bundle), avoiding a version that
    points at a nonexistent preset; preset_path already validates the name, so an
    invalid name (e.g. containing path separators) is treated as nonexistent.
    Returns the updated version; returns None if nothing changed."""
    if not preset_name:
        return None
    from ..presets import io as _presets_io
    try:
        if not _presets_io.preset_path(preset_name, presets_base).exists():
            return None
    except _presets_io.PresetError:
        return None
    return versions.update_version(conn, version_id, config_name=preset_name)


def import_bundle(
    conn: sqlite3.Connection,
    zip_path: Path,
    presets_base: Path,
) -> dict[str, Any]:
    """Import from a bundle.zip (schema_version 1 or 2), creating a new project +
    source version.

    v1 (the legacy train.zip): equivalent to import_train.
    v2: handles train / reg / presets separately according to manifest.includes.
    Returns {"project": {...}, "version": {...}, "stats": {...}}.
    """
    if not zip_path.exists():
        raise NotFoundError(
            "File not found", code="file.not_found", http_status=404,
        )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest = _read_manifest(zf)
            schema_ver = manifest.get("schema_version", 1)

            if schema_ver == 1:
                # Legacy format: delegate to the existing logic
                pass  # fallthrough to import_train below

            source = manifest.get("source") or {}
            title = (source.get("title") or "imported").strip() or "imported"
            base_slug = projects.slugify(source.get("slug") or title)
            version_label = _bundle_source_version_label(source)
            preset_name = _bundle_source_preset_name(source)

            if schema_ver == 1:
                # v1: only train/ entries, using the existing safety checks
                entries_v1: list[tuple[zipfile.ZipInfo, str]] = []
                for info in zf.infolist():
                    if info.is_dir() or info.filename == MANIFEST_NAME:
                        continue
                    inner = _safe_arc(info.filename)
                    if inner is None:
                        raise TrainIOError(
                            "Import file contains an invalid path",
                            code="dataset.import_invalid", http_status=400,
                        )
                    entries_v1.append((info, inner))
                if not entries_v1:
                    raise TrainIOError(
                        "Import file contains nothing to import",
                        code="dataset.import_empty", http_status=400,
                    )

                slug = _resolve_slug_conflict(conn, base_slug)
                p = projects.create_project(conn, title=title, slug=slug,
                                             note=f"imported from {title!r}")
                v = versions.create_version(conn, project_id=p["id"], label=version_label)
                v = _restore_preset_name(conn, v["id"], preset_name, presets_base) or v
                vdir = versions.version_dir(p["id"], p["slug"], v["label"])
                train_dir = vdir / "train"
                seen_folders: set[str] = set()
                for info, inner in entries_v1:
                    folder, filename = inner.split("/", 1)
                    if filename.startswith("."):
                        raise TrainIOError(
                            "Import file contains an invalid filename",
                            code="dataset.import_invalid", http_status=400,
                        )
                    target = safe_join(train_dir, folder, filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    seen_folders.add(folder)
                    with zf.open(info) as src, target.open("wb") as dst:
                        _copy_chunks(src, dst)

                img_cnt, tag_cnt = _count_train(train_dir, seen_folders)
                # ADR-0007 PR-5: stage is no longer advanced automatically
                v = versions.get_version(conn, v["id"])
                p = projects.get_project(conn, p["id"])
                assert v is not None and p is not None
                return {
                    "project": p, "version": v,
                    "stats": {
                        "train_image_count": img_cnt,
                        "train_tagged_count": tag_cnt,
                        "reg_image_count": 0,
                        "config_imported": False,
                        "preset_count": 0,
                    },
                }

            # v2 bundle
            includes = manifest.get("includes") or {}
            train_entries: list[tuple[zipfile.ZipInfo, str]] = []
            reg_entries: list[tuple[zipfile.ZipInfo, str]] = []
            preset_entries: list[tuple[zipfile.ZipInfo, str]] = []

            for info in zf.infolist():
                if info.is_dir() or info.filename == MANIFEST_NAME:
                    continue
                result = _safe_arc_bundle(info.filename)
                if result is None:
                    raise TrainIOError(
                        "Import file contains an invalid path",
                        code="dataset.import_invalid", http_status=400,
                    )
                section, inner = result
                if section == "train":
                    train_entries.append((info, inner))
                elif section == "reg":
                    reg_entries.append((info, inner))
                elif section == "preset":
                    preset_entries.append((info, inner))

            if not train_entries and not reg_entries and not preset_entries:
                raise TrainIOError(
                    "Import file contains nothing to import",
                    code="dataset.import_empty", http_status=400,
                )

            slug = _resolve_slug_conflict(conn, base_slug)
            p = projects.create_project(conn, title=title, slug=slug,
                                         note=f"imported from {title!r}")
            v = versions.create_version(conn, project_id=p["id"], label=version_label)
            vdir = versions.version_dir(p["id"], p["slug"], v["label"])

            # --- write out train ---
            seen_train: set[str] = set()
            npz_targets: list[Path] = []
            if train_entries:
                train_dir = vdir / "train"
                for info, inner in train_entries:
                    folder, filename = inner.split("/", 1)
                    if filename.startswith("."):
                        raise TrainIOError(
                            "Import file contains an invalid filename",
                            code="dataset.import_invalid", http_status=400,
                        )
                    target = safe_join(train_dir, folder, filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    seen_train.add(folder)
                    with zf.open(info) as src, target.open("wb") as dst:
                        _copy_chunks(src, dst)
                    if target.suffix.lower() == LATENT_CACHE_EXT:
                        npz_targets.append(target)

            # --- write out reg ---
            reg_image_count = 0
            if reg_entries:
                reg_dir = vdir / "reg"
                for info, inner in reg_entries:
                    if inner == "meta.json":
                        target = reg_dir / "meta.json"
                        target.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        parts = inner.split("/", 1)
                        if len(parts) != 2:
                            raise TrainIOError(
                                "Import file contains an invalid path",
                                code="dataset.import_invalid", http_status=400,
                            )
                        folder, filename = parts
                        if filename.startswith("."):
                            raise TrainIOError(
                                "Import file contains an invalid filename",
                                code="dataset.import_invalid", http_status=400,
                            )
                        target = safe_join(reg_dir, folder, filename)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.suffix.lower() in IMAGE_EXTS:
                            reg_image_count += 1
                        elif target.suffix.lower() == LATENT_CACHE_EXT:
                            npz_targets.append(target)
                    with zf.open(info) as src, target.open("wb") as dst:
                        _copy_chunks(src, dst)

            # Latent cache mtime fixup: the training side's _is_cache_valid requires
            # the npz mtime to be >= the image's mtime, but extracting entries in
            # dictionary order often writes the npz before the same-named image
            # (".npz" sorts before ".png") -> the cache would be judged stale and
            # re-encoded. After extraction, touch every npz to the current time.
            for t in npz_targets:
                os.utime(t, None)

            # --- write out the version's training config / other presets ---
            # presets/config.yaml -> applied to the new version (force_project_overrides
            # fills in paths automatically)
            # other .yaml files -> written to the global presets directory (conflicts
            # get a _imported_{n} suffix)
            config_imported = False
            preset_count = 0
            if preset_entries:
                from .. import version_config as _vc
                import yaml as _yaml
                for info, inner in preset_entries:
                    if inner.startswith("."):
                        continue
                    if inner == "config.yaml":
                        try:
                            raw = json.loads(zf.read(info))
                        except (json.JSONDecodeError, ValueError):
                            try:
                                raw = _yaml.safe_load(zf.read(info)) or {}
                            except Exception:
                                raw = {}
                        if isinstance(raw, dict):
                            # The 4 global model path fields aren't in
                            # PROJECT_SPECIFIC_FIELDS, so an absolute path from the
                            # source machine carried in the bundle (e.g.
                            # `G:/models/...`) would be written as-is, and when
                            # imported on another machine `_absolutize_model_paths`
                            # would mangle it into `<repo>/G:/...`. To match
                            # fork_preset_for_version: when auto_sync_paths is ON,
                            # overwrite the 4 fields with this machine's global
                            # values; when OFF, respect the bundle's content (drive-
                            # letter detection was already fixed in
                            # _absolutize_model_paths, so it no longer mis-prepends
                            # the prefix on POSIX).
                            from .. import presets as _presets_svc
                            from .. import models as _md
                            if _presets_svc._auto_sync_paths():
                                family = str(raw.get("model_family") or "anima")
                                try:
                                    raw.update(_md.default_paths_for_new_version(
                                        family=family))
                                except ValueError:
                                    # bundle from an unknown family: don't override
                                    # paths, let write_version_config's schema
                                    # validation below reject it
                                    pass
                            try:
                                _vc.write_version_config(p, v, raw, force_project_overrides=True)
                                config_imported = True
                            except _vc.VersionConfigError:
                                pass  # invalid config, skip without interrupting the import
                    else:
                        from ..presets import io as _presets_io

                        try:
                            config, _ = _presets_io.parse_preset_bytes(zf.read(info), inner)
                        except _presets_io.PresetError:
                            continue
                        target_name = Path(inner).stem
                        target = _presets_io.preset_path(target_name, presets_base)
                        if target.exists():
                            n = 1
                            while True:
                                candidate_name = f"{target_name}_imported_{n}"
                                candidate = _presets_io.preset_path(candidate_name, presets_base)
                                if not candidate.exists():
                                    target_name = candidate_name
                                    break
                                n += 1
                        _presets_io.write_preset(target_name, config, presets_base)
                        preset_count += 1

            # Backfill config_name only after preset extraction is complete: only
            # record it if this machine genuinely has the preset (either it already
            # existed, or it was just extracted from the bundle), to avoid a
            # dangling reference.
            v = _restore_preset_name(conn, v["id"], preset_name, presets_base) or v

            img_cnt, tag_cnt = _count_train(vdir / "train", seen_train) if train_entries else (0, 0)

            # ADR-0007 PR-5: stage is no longer advanced automatically
            v = versions.get_version(conn, v["id"])
            p = projects.get_project(conn, p["id"])
            assert v is not None and p is not None

    except zipfile.BadZipFile as exc:
        raise TrainIOError(
            "Import file is corrupt", code="dataset.import_corrupt",
            http_status=400,
        ) from exc

    return {
        "project": p,
        "version": v,
        "stats": {
            "train_image_count": img_cnt,
            "train_tagged_count": tag_cnt,
            "reg_image_count": reg_image_count,
            "config_imported": config_imported,
            "preset_count": preset_count,
        },
    }


def _copy_chunks(src: Any, dst: Any) -> None:
    while True:
        chunk = src.read(64 * 1024)
        if not chunk:
            break
        dst.write(chunk)


def _count_train(train_dir: Path, folders: set[str]) -> tuple[int, int]:
    img = 0
    tagged = 0
    for folder in folders:
        fdir = train_dir / folder
        if not fdir.exists():
            continue
        for f in fdir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                img += 1
                if f.with_suffix(".txt").exists():
                    tagged += 1
    return img, tagged
