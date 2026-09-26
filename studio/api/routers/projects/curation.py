"""File management + curation + deduplication (PR-6.5 commit 4, extracted from server.py).

routes:
    POST /api/projects/{pid}/files/delete                                download/ file + caption metadata
    GET  /api/projects/{pid}/files                                       download/ listing
    GET  /api/projects/{pid}/thumb                                       thumbnail (includes manifest resolve)
    GET  /api/projects/{pid}/versions/{vid}/jobs/latest                  hydrate latest job + log
    GET  /api/projects/{pid}/versions/{vid}/curation                     curation_view (train/ content + download/ remainder)
    POST /api/projects/{pid}/versions/{vid}/preprocess/duplicates/scan   train-scope dedup scan + SSE progress
    POST /api/projects/{pid}/versions/{vid}/preprocess/duplicates/apply  mark manifest duplicate_removed
    POST /api/projects/{pid}/versions/{vid}/curation/copy                download -> train/{folder}
    POST /api/projects/{pid}/versions/{vid}/curation/remove              train/{folder} -> delete
    POST /api/projects/{pid}/versions/{vid}/curation/folder              create/rename/delete folder
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...errors import _safe_join_or_400
from ...responses import _thumb_response
from ....domain.errors import DomainError, NotFoundError, ValidationError
from ...schemas.curation import (
    CopyRequest,
    CopyValidationRequest,
    DeleteFilesRequest,
    DuplicateApplyRequest,
    DuplicateScanRequest,
    FolderOp,
    RemoveRequest,
    RemoveValidationRequest,
)
from ._shared import _publish_project_state
from .... import db
from ....services.projects import jobs as project_jobs, projects, versions
from ....services.dataset import curation, scan as datasets
from ....infrastructure.event_bus import bus
from ....services.preprocess import duplicates as duplicate_finder, manifest as preprocess_manifest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/projects/{pid}/files/delete")
def delete_project_files(
    pid: int, body: DeleteFilesRequest,
) -> dict[str, Any]:
    """Delete the given files from a project's `download/` (including same-stem caption metadata).

    Metadata naming convention:
    - booru downloads write `{stem}.booru.txt`
    - tag/caption flows may write `{stem}.txt` or `{stem}.json`
    All of them are cleaned up together; extensions that don't exist are silently skipped.
    """
    if not body.names:
        return {"deleted": [], "missing": []}
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found", details={"id": pid},
        )
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    if not pdir.exists():
        return {"deleted": [], "missing": list(body.names)}

    META_EXTS = (".booru.txt", ".txt", ".json")
    deleted: list[str] = []
    missing: list[str] = []
    for name in body.names:
        f = _safe_join_or_400(pdir, name)
        if not f.exists() or not f.is_file():
            missing.append(name)
            continue
        try:
            f.unlink()
        except OSError as exc:
            raise DomainError(
                f'Failed to delete "{name}": {exc}',
                code="dataset.delete_failed",
                details={"name": name, "reason": str(exc)},
                http_status=500,
            ) from exc
        # Clean up same-stem metadata (best-effort, failures are just logged)
        stem = f.stem
        for ext in META_EXTS:
            m = pdir / f"{stem}{ext}"
            if m.exists():
                try:
                    m.unlink()
                except OSError as exc:
                    logger.warning("failed to delete metadata %s: %s", m, exc)
        deleted.append(name)
    return {"deleted": deleted, "missing": missing}


@router.get("/api/projects/{pid}/files")
def list_files(pid: int, bucket: str = "download") -> dict[str, Any]:
    if bucket != "download":
        raise ValidationError(
            "Unsupported image set",
            code="dataset.image_set_invalid",
            details={"bucket": bucket}, http_status=400,
        )
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found", details={"id": pid},
        )
    pdir = projects.project_dir(p["id"], p["slug"]) / "download"
    items: list[dict[str, Any]] = []
    if pdir.exists():
        for f in sorted(pdir.iterdir()):
            if f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS:
                st = f.stat()
                items.append({
                    "name": f.name,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                    "has_meta": f.with_suffix(".booru.txt").exists(),
                })
    return {"items": items, "count": len(items)}


@router.get("/api/projects/{pid}/thumb")
def project_thumb(
    pid: int,
    bucket: str = "download",
    name: str = "",
    size: int = 256,
    raw: int = 0,
) -> FileResponse:
    """Thumbnail: defaults to 256px JPEG (cached); size=0 -> original image.

    Two buckets:
      - `bucket=download` (default): `name` is the original filename under download/.
        The backend decides the actual byte path via
        `preprocess_manifest.resolve_origin()`: unprocessed -> download/{name},
        processed -> the first origin-matching derivative under preprocess/. The
        frontend calling "by download name" doesn't need to be aware of preprocessing.
      - `bucket=preprocess`: `name` is the **actual product filename** under preprocess/
        (including the _c0 / _c1 suffix for multi-crop derivatives). Fetched directly by
        filename, **does not** go through resolve_origin -- after multi-crop, multiple
        products share the same origin, so always landing on [0] via origin would be a
        bug. The crop / overview pages should use this path for precise addressing.

    `raw=1` (bucket=download only): skips resolve_origin, forces reading the raw
    download/{name} bytes. Used by the "compare preview" scenario: the left pane must
    always show the download original, never hijacked by a preprocess derivative.

    Cache path: `studio_data/thumb_cache/{sha1(src+mtime+size)}.jpg`.
    Automatically invalidated when the source file's mtime changes (hash changes).
    """
    if bucket not in ("download", "preprocess"):
        raise ValidationError(
            "Unsupported image set",
            code="dataset.image_set_invalid",
            details={"bucket": bucket}, http_status=400,
        )
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found", details={"id": pid},
        )
    pdir = projects.project_dir(p["id"], p["slug"])

    if bucket == "preprocess":
        # Direct addressing — no resolve. Path traversal guard against the
        # actual preprocess/ dir (any filename including _c0/_c1 derivatives).
        _safe_join_or_400(pdir / "preprocess", name)
        f = pdir / "preprocess" / name
    elif raw:
        # bucket=download + raw=1: bypass resolve_origin, hand back the
        # untouched download/{name} bytes. Used by the processed-tab compare
        # preview left pane (need the original, not the derivative).
        _safe_join_or_400(pdir / "download", name)
        f = pdir / "download" / name
    else:
        # bucket=download — historical behavior: address by download name,
        # resolve to first preprocess product if any (1:1 / multi-crop cases).
        # duplicate_removed origins: resolve_origin returns [] but the original
        # file in download/ still exists; the Download page must keep showing
        # it (a soft delete is not the same as invisible). Fall back to
        # download/{name} like any other un-resolved origin.
        _safe_join_or_400(pdir / "download", name)
        candidates = preprocess_manifest.resolve_origin(pdir, name)
        f = candidates[0] if candidates else (pdir / "download" / name)
        # Curation passes multi-crop derivative names (X_c0.png) through this
        # endpoint with bucket=download. resolve_origin only matches by origin,
        # not by entry key, so derivatives miss → f points at a non-existent
        # download/X_c0.png. Fall back: if the name IS a preprocess entry key,
        # serve preprocess/{name} directly. Filename was already safety-checked
        # against download/, same validation applies to preprocess/.
        if not f.exists() and preprocess_manifest.get_entry(pdir, name) is not None:
            f = pdir / "preprocess" / name

    if not f.exists() or f.suffix.lower() not in datasets.IMAGE_EXTS:
        logger.info("thumb not found: pid=%s bucket=%s name=%s -> %s", pid, bucket, name, f)
        raise NotFoundError(
            "Thumbnail not found", code="dataset.thumbnail_not_found",
        )
    return _thumb_response(f, size)


# ---------------------------------------------------------------------------
# /api/projects/{pid}/versions/{vid}/jobs/latest (hydrate)
# /api/jobs/{jid} / log / cancel were moved to api/routers/jobs.py in PR-6 commit 2;
# this endpoint stays in the projects subpackage because its path lives under /api/projects/.
# ---------------------------------------------------------------------------

_HYDRATABLE_JOB_KINDS = {
    "download",
    "tag",
    "reg_build",
    "eval_samples",
    "eval_clip",
    "eval_dino",
    "eval_tag",
    "eval_ccip",
}


@router.get("/api/projects/{pid}/versions/{vid}/jobs/latest")
def get_latest_version_job(pid: int, vid: int, kind: str) -> dict[str, Any]:
    """Used to hydrate on page refresh: returns the most recent job of the given kind
    for this version, plus its full log.

    The Tagging / Regularization pages previously only knew the jid after startBuild in
    the current session, and lost it on refresh; this gives the frontend a starting point
    to re-lock onto the jid on mount + replay the historical log, with SSE taking over
    from there. `job` may be running / pending / completed; the frontend decides whether
    to keep waiting for events based on status.
    """
    if kind not in _HYDRATABLE_JOB_KINDS:
        raise HTTPException(400, f"unknown kind: {kind}")
    with db.connection_for() as conn:
        job = project_jobs.latest_for(
            conn, project_id=pid, kind=kind, version_id=vid,
        )
    if not job:
        return {"job": None, "log": ""}
    log_path = Path(job.get("log_path") or "")
    log = ""
    if log_path.exists():
        try:
            log = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            log = ""
    return {"job": job, "log": log}


# ---------------------------------------------------------------------------
# /api/projects/{pid}/versions/{vid}/curation  (PP3)
# ---------------------------------------------------------------------------


@router.get("/api/projects/{pid}/versions/{vid}/curation")
def get_curation(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        return curation.curation_view(conn, pid, vid)


# ---------------------------------------------------------------------------
# Manual maintenance of the held-out validation set (symmetric with train curation, but
# the right column is flat, no folders)
# ---------------------------------------------------------------------------


@router.get("/api/projects/{pid}/versions/{vid}/curation/validation")
def get_curation_validation(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        return curation.curation_validation_view(conn, pid, vid)


@router.post("/api/projects/{pid}/versions/{vid}/curation/validation/copy")
def copy_to_validation(
    pid: int, vid: int, body: CopyValidationRequest,
) -> dict[str, Any]:
    """Copy download -> validation/1_data/ (caption included); names already present in
    train/validation are skipped to prevent leakage."""
    with db.connection_for() as conn:
        return curation.copy_download_to_validation(conn, pid, vid, body.files)


@router.post("/api/projects/{pid}/versions/{vid}/curation/validation/remove")
def remove_from_validation(
    pid: int, vid: int, body: RemoveValidationRequest,
) -> dict[str, Any]:
    with db.connection_for() as conn:
        return curation.remove_from_validation(
            conn, pid, vid, [item.model_dump() for item in body.items],
        )


# ---------------------------------------------------------------------------
# ADR 0010 -- train-scope duplicates endpoint
#
# Scope narrowed to versions/{label}/train/, corresponding to scan_train_duplicates /
# apply_train_duplicate_removals.
# ---------------------------------------------------------------------------


def _resolve_pv_or_404_dup(
    pid: int, vid: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        if not p:
            raise NotFoundError(
                "Project not found", code="project.not_found", details={"id": pid},
            )
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found", details={"id": vid},
            )
    return p, v


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/duplicates/scan")
def scan_preprocess_duplicates_train(
    pid: int, vid: int, body: DuplicateScanRequest,
) -> dict[str, Any]:
    """ADR 0010 train-scope duplicate scan. sources = versions/{label}/train/ (skipping
    entries already marked duplicate_removed in the manifest); the return structure
    matches the old endpoint, with the `target` field changed from `"preprocess"` to
    `"train"`."""
    _resolve_pv_or_404_dup(pid, vid)
    with db.connection_for() as conn:
        try:
            options = duplicate_finder.options_from_payload(body.model_dump())
            last_progress_at = 0.0

            def publish_progress(payload: dict[str, Any]) -> None:
                nonlocal last_progress_at
                now = time.monotonic()
                if now - last_progress_at < 1.0:
                    return
                last_progress_at = now
                bus.publish({
                    "type": "duplicate_scan_progress",
                    "project_id": pid,
                    "version_id": vid,
                    "status": "running",
                    **payload,
                })

            bus.publish({
                "type": "duplicate_scan_progress",
                "project_id": pid,
                "version_id": vid,
                "status": "running",
                "text": "Scanning duplicate candidates in train set...",
            })
            result = duplicate_finder.scan_train_duplicates(
                conn, pid, vid, options, on_progress=publish_progress,
            )
            bus.publish({
                "type": "duplicate_scan_progress",
                "project_id": pid,
                "version_id": vid,
                "status": "done",
                "total_images": result["total_images"],
                "group_count": result["group_count"],
                "candidate_count": result["candidate_count"],
                "crop_relation_count": result.get("crop_relation_count", 0),
                "elapsed_seconds": result["elapsed_seconds"],
                "text": (
                    f"Scanned {result['total_images']} train images; "
                    f"found {result['group_count']} groups / "
                    f"{result['candidate_count']} candidates, "
                    f"{result.get('crop_relation_count', 0)} crop relations."
                ),
            })
            return result
        except curation.CurationError as exc:
            bus.publish({
                "type": "duplicate_scan_progress",
                "project_id": pid,
                "version_id": vid,
                "status": "failed",
                "text": str(exc),
            })
            raise
        except duplicate_finder.DuplicateFinderError as exc:
            bus.publish({
                "type": "duplicate_scan_progress",
                "project_id": pid,
                "version_id": vid,
                "status": "failed",
                "text": str(exc),
            })
            raise


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/duplicates/apply")
def apply_preprocess_duplicates_train(
    pid: int, vid: int, body: DuplicateApplyRequest,
) -> dict[str, Any]:
    """ADR 0010 train scope: marks per-version review status into the train manifest.
    `names` are train-relative paths (`"1_data/X.png"`). Physical files are untouched."""
    _resolve_pv_or_404_dup(pid, vid)
    with db.connection_for() as conn:
        result = duplicate_finder.apply_train_duplicate_removals(
            conn, pid, vid, names=body.names,
        )
        project = projects.get_project(conn, pid)
    if project:
        _publish_project_state(project)
    return result


@router.post("/api/projects/{pid}/versions/{vid}/curation/copy")
def copy_to_train(
    pid: int, vid: int, body: CopyRequest,
) -> dict[str, Any]:
    """ADR 0010 fixup (2026-06-04): endpoint switched to the simplified
    `copy_download_to_train` -- a pure download -> train copy + writing a train manifest
    entry. No longer branches on "preprocess derivative vs. download original", and no
    longer checks the old `duplicate_removed` (PR-4 decided dedupe moves down to train
    scope, so the old marker doesn't affect Curation operations). The frontend's Curation
    picking a download original to add to train just works directly.
    """
    with db.connection_for() as conn:
        result = curation.copy_download_to_train(
            conn, pid, vid, body.files, body.dest_folder,
        )
    return result


@router.post("/api/projects/{pid}/versions/{vid}/curation/remove")
def remove_from_train(
    pid: int, vid: int, body: RemoveRequest,
) -> dict[str, Any]:
    with db.connection_for() as conn:
        result = curation.remove_from_train(
            conn, pid, vid, body.folder, body.files,
        )
    return result


@router.post("/api/projects/{pid}/versions/{vid}/curation/folder")
def folder_op(
    pid: int, vid: int, body: FolderOp,
) -> dict[str, Any]:
    with db.connection_for() as conn:
        if body.op == "create":
            p = curation.create_folder(conn, pid, vid, body.name)
            return {"path": str(p)}
        if body.op == "rename":
            if not body.new_name:
                raise ValidationError(
                    "A new name is required to rename a folder",
                    code="curation.new_name_required", http_status=400,
                )
            p = curation.rename_folder(
                conn, pid, vid, body.name, body.new_name,
            )
            return {"path": str(p)}
        if body.op == "delete":
            curation.delete_folder(conn, pid, vid, body.name)
            return {"deleted": body.name}
        raise ValidationError(
            f"Unknown folder operation: {body.op}",
            code="curation.op_invalid", details={"op": body.op}, http_status=400,
        )
