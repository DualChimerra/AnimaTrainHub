"""Image fetching + preprocessing (extracted from server.py in PR-6.5 commit 3).

14 routes:

  download / upload (5)
    POST /api/projects/{pid}/download/estimate    estimate via booru count API
    POST /api/projects/{pid}/download             start a booru download job
    POST /api/projects/{pid}/upload               local multi-file upload (single image / zip)
    POST /api/projects/{pid}/upload-from-path     import single image / zip from a server-visible path
    GET  /api/projects/{pid}/download/status      most recent download job + log_tail

  preprocessing (13)
    POST /api/projects/{pid}/preprocess/start
    GET  /api/projects/{pid}/preprocess/status
    GET  /api/projects/{pid}/preprocess/files
    GET  /api/projects/{pid}/preprocess/duplicates/removed
    GET  /api/projects/{pid}/preprocess/crop/workspace
    POST /api/projects/{pid}/preprocess/crop
    POST /api/projects/{pid}/versions/{vid}/preprocess/inpaint/save
    GET  /api/projects/{pid}/versions/{vid}/preprocess/mask
    PUT  /api/projects/{pid}/versions/{vid}/preprocess/mask
    DELETE /api/projects/{pid}/versions/{vid}/preprocess/mask
    POST /api/projects/{pid}/preprocess/files/reset
    POST /api/projects/{pid}/preprocess/files/restore
    GET  /api/projects/{pid}/preprocess/thumb     [Deprecated] kept for old URL compatibility

Note: duplicates scan / apply (a preprocess sub-domain) belongs to commit 4 (curation), not this file.
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from ...errors import _validate_component_or_400  # noqa: F401  reserved for future use
from ....domain.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from ...schemas.ingestion import (
    DownloadRequest,
    EstimateRequest,
    PreprocessCropRequest,
    PreprocessRestoreRequest,
    PreprocessStartRequest,
    UploadFromPathBody,
)
from ._shared import _publish_job_state, _publish_project_state
from ....infrastructure.event_bus import bus
from .... import db, secrets
from ....services.projects import jobs as project_jobs, projects, versions
from ....paths import REPO_ROOT
from ....services.preprocess import core as preprocess_svc
from ....services import model_downloader
from ....services.booru import downloader
from ....services.preprocess import manifest as preprocess_manifest

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# /api/projects/{pid}/download + /api/projects/{pid}/files + /api/jobs/*  (PP2)
# ---------------------------------------------------------------------------


@router.post("/api/projects/{pid}/download/estimate")
def estimate_download(pid: int, body: EstimateRequest) -> dict[str, Any]:
    """Call the booru's count API to estimate hits, then let the user decide the count.

    Returns -1 for unknown (the API doesn't support an exact count); the frontend treats
    that as "download all".
    """
    if body.api_source not in {"gelbooru", "danbooru"}:
        raise ValidationError(
            f"Unsupported image source: {body.api_source}",
            code="download.source_unsupported",
            details={"source": body.api_source}, http_status=400,
        )
    if not body.tag.strip():
        raise ValidationError(
            "Tag is required", code="download.tag_required", http_status=400,
        )
    if not secrets.has_credentials_for(body.api_source):
        raise ValidationError(
            f"No {body.api_source} credentials configured; add them on the Settings page",
            code="download.credentials_missing",
            details={"source": body.api_source}, http_status=400,
        )
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise NotFoundError(
                "Project not found", code="project.not_found", details={"id": pid},
            )
    sec = secrets.load()
    if body.api_source == "danbooru":
        opts = downloader.DownloadOptions(
            tag=body.tag.strip(),
            count=1,
            api_source="danbooru",
            username=sec.danbooru.username,
            api_key=sec.danbooru.api_key,
            exclude_tags=list(sec.download.exclude_tags),
        )
    else:
        opts = downloader.DownloadOptions(
            tag=body.tag.strip(),
            count=1,
            api_source="gelbooru",
            user_id=sec.gelbooru.user_id,
            api_key=sec.gelbooru.api_key,
            exclude_tags=list(sec.download.exclude_tags),
        )
    count = downloader.estimate(opts)
    return {
        "tag": body.tag.strip(),
        "api_source": body.api_source,
        "exclude_tags": list(sec.download.exclude_tags),
        "effective_query": opts.effective_tag_query(),
        "count": count,
    }


@router.post("/api/projects/{pid}/download")
def start_download(pid: int, body: DownloadRequest) -> dict[str, Any]:
    if not body.tag.strip():
        raise ValidationError(
            "Tag is required", code="download.tag_required", http_status=400,
        )
    if body.count < 1:
        raise ValidationError(
            "Download count must be at least 1",
            code="download.count_invalid", http_status=400,
        )
    if body.api_source not in {"gelbooru", "danbooru"}:
        raise ValidationError(
            f"Unsupported image source: {body.api_source}",
            code="download.source_unsupported",
            details={"source": body.api_source}, http_status=400,
        )
    if not secrets.has_credentials_for(body.api_source):
        raise ValidationError(
            f"No {body.api_source} credentials configured; add them on the Settings page",
            code="download.credentials_missing",
            details={"source": body.api_source}, http_status=400,
        )

    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise NotFoundError(
                "Project not found", code="project.not_found", details={"id": pid},
            )
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            kind="download",
            params={
                "tag": body.tag.strip(),
                "count": body.count,
                "api_source": body.api_source,
            },
        )
    _publish_job_state(job)
    return job


_UPLOAD_CHUNK = 1 << 20  # 1 MiB streaming write chunk


def _staging_root() -> Path:
    """Upload staging root = STUDIO_DATA/uploads.

    Derived from ``project_jobs.JOB_LOGS_DIR.parent`` (rather than the STUDIO_DATA constant
    directly), so that when tests monkeypatch JOB_LOGS_DIR the staging dir follows it into
    tmp too, instead of polluting the repo.
    """
    return project_jobs.JOB_LOGS_DIR.parent / "uploads"


def _safe_name(name: str) -> str:
    """Strip path segments, keep only the basename (prevents path traversal)."""
    return (name or "").replace("\\", "/").rsplit("/", 1)[-1]


async def _stage_upload_files(files: list[UploadFile], staging_dir: Path) -> int:
    """Stream uploaded files to staging_dir (chunked, never loaded fully into memory). Returns the file count written."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        name = _safe_name(f.filename or "")
        if not name:
            continue
        dest = staging_dir / name
        with dest.open("wb") as out:
            while True:
                chunk = await f.read(_UPLOAD_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
        n += 1
    return n


def _publish_upload_log(pid: int, line: str) -> None:
    """Publish one upload-stage log line to SSE subscribers (shown in the frontend's TaskLogDrawer).

    `accept_many`'s on_log callback fires every 25 images / 5s / on a slow image (throttled,
    to avoid flooding). Coexists with logger.info: the former is for the user, the latter goes
    to studio.log for debugging.
    """
    bus.publish({"type": "project_upload_log", "project_id": pid, "line": line})


def _publish_upload_state(pid: int, status: str) -> None:
    """Publish an upload state transition (running / done / failed) to SSE subscribers.

    LogSource.status uses this to drive TaskLogDrawer's badge + auto-expand (opens
    automatically while live; stays expanded but stops auto-opening once terminal).
    """
    bus.publish({"type": "project_upload_state", "project_id": pid, "status": status})


@router.post("/api/projects/{pid}/upload")
async def upload_local_files(
    pid: int, files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """Local upload: single image / zip / matching .txt caption -> handled by a background job.

    The endpoint just **streams the uploaded files to a staging dir** and immediately returns
    an upload job (sub-second); the actual unzip / convert_to_png / caption pairing runs in
    upload_worker in the background. This way a large zip doesn't stall inside the synchronous
    request and trip Cloudflare's 100s timeout (524). The frontend polls `upload/status` for
    progress + results.
    """
    if not files:
        raise ValidationError(
            "No files uploaded", code="dataset.no_files", http_status=400,
        )
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise HTTPException(404, f"Project not found: id={pid}")

    staging_dir = _staging_root() / f"{pid}_{uuid.uuid4().hex}"
    try:
        staged = await _stage_upload_files(files, staging_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    if staged == 0:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(400, "No valid filenames")

    with db.connection_for() as conn:
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            kind="upload",
            params={"staging_dir": str(staging_dir), "source": "upload"},
        )
    _publish_job_state(job)
    return job


@router.post("/api/projects/{pid}/upload-from-path")
def upload_local_file_from_path(pid: int, body: UploadFromPathBody) -> dict[str, Any]:
    """Import a single image / zip from a server-visible path -> background upload job (no copy, doesn't delete the original)."""
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found", details={"id": pid},
        )
    src = Path(body.path)
    if not src.is_absolute():
        src = (REPO_ROOT / src).resolve()
    else:
        src = src.resolve()
    if not src.exists():
        raise NotFoundError(
            f"Path not found: {body.path}",
            code="path.not_found", details={"path": body.path},
        )
    if not src.is_file():
        raise HTTPException(400, "Please select a file")

    with db.connection_for() as conn:
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            kind="upload",
            params={"paths": [str(src)], "source": "path"},
        )
    _publish_job_state(job)
    return job


@router.get("/api/projects/{pid}/upload/status")
def upload_status(pid: int) -> dict[str, Any]:
    """Most recent upload job + log_tail + result (added/skipped).

    The frontend polls this after uploading bytes: shows "processing" before the job reaches
    a terminal state, then on done pops the added/skipped summary from result and refreshes
    the gallery; on failed shows error_msg.
    """
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise HTTPException(404, f"Project not found: id={pid}")
        job = project_jobs.latest_for(conn, project_id=pid, kind="upload")
    if not job:
        return {"job": None, "log_tail": "", "result": None}
    log_path = Path(job.get("log_path") or "")
    tail = ""
    if log_path.exists():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            tail = "\n".join(text.splitlines()[-50:])
        except Exception:
            tail = ""
    result = (
        project_jobs.read_result(job["id"])
        if job.get("status") == "done"
        else None
    )
    return {"job": job, "log_tail": tail, "result": result}


@router.get("/api/projects/{pid}/download/status")
def download_status(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise NotFoundError(
                "Project not found", code="project.not_found", details={"id": pid},
            )
        job = project_jobs.latest_for(conn, project_id=pid, kind="download")
    if not job:
        return {"job": None, "log_tail": ""}
    log_path = Path(job.get("log_path") or "")
    tail = ""
    if log_path.exists():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            tail = "\n".join(text.splitlines()[-50:])
        except Exception:
            tail = ""
    return {"job": job, "log_tail": tail}


# ---------------------------------------------------------------------------
# ADR 0010 -- train-scope preprocess endpoint group
#
# `/api/projects/{pid}/versions/{vid}/preprocess/*` -- scope narrowed to the train set,
# calls the *_train service functions.
# ---------------------------------------------------------------------------


def _resolve_pv_or_404(pid: int, vid: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch (project, version), validating both exist and vid belongs to pid."""
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


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/start")
def start_preprocess_train(
    pid: int, vid: int, body: PreprocessStartRequest,
) -> dict[str, Any]:
    """ADR 0010 train scope: run upscale on versions/{label}/train/{folder}/.

    Same body schema + validation as the old `start_preprocess`; the worker looks at
    job.version_id and dispatches to _run_upscale_train.
    """
    if body.mode not in ("all", "selected", "all_force"):
        raise ValidationError(
            f"Invalid preprocess mode: {body.mode}",
            code="preprocess.mode_invalid",
            details={"mode": body.mode}, http_status=400,
        )
    if body.tile_size <= 0:
        raise ValidationError(
            "Tile size must be greater than 0",
            code="preprocess.tile_size_invalid", http_status=400,
        )
    if body.device not in ("auto", "cuda", "cpu"):
        raise ValidationError(
            f"Invalid device: {body.device}",
            code="preprocess.device_invalid",
            details={"device": body.device}, http_status=400,
        )
    if body.target_area is not None and (
        body.target_area < 256 * 256 or body.target_area > 4096 * 4096
    ):
        raise ValidationError(
            f"Target area is out of range: {body.target_area}",
            code="preprocess.target_area_out_of_range",
            details={"value": body.target_area}, http_status=400,
        )
    try:
        target = model_downloader.upscaler_target(body.model)
    except ValueError as exc:
        raise NotFoundError(
            f'Upscaler "{body.model}" not found',
            code="upscaler.not_found", details={"name": body.model},
        ) from exc
    if not target.exists():
        raise ConflictError(
            f'Upscaler weights for "{body.model}" are not downloaded; '
            "download them under Settings → Preprocess",
            code="upscaler.not_downloaded", details={"name": body.model},
        )

    p, v = _resolve_pv_or_404(pid, vid)
    with db.connection_for() as conn:
        job = preprocess_svc.start_job_train(
            conn,
            project_id=pid,
            version_id=vid,
            mode=body.mode,
            names=body.names,
            model=body.model,
            tile_size=body.tile_size,
            tile_pad=body.tile_pad,
            device=body.device,
            target_area=body.target_area,
        )
    _publish_job_state(job)
    return job


@router.get("/api/projects/{pid}/versions/{vid}/preprocess/status")
def preprocess_status_train(pid: int, vid: int) -> dict[str, Any]:
    """Latest train-scope preprocess job + log tail + train summary."""
    p, v = _resolve_pv_or_404(pid, vid)
    with db.connection_for() as conn:
        job = project_jobs.latest_for(
            conn, project_id=pid, version_id=vid,
            kind=preprocess_svc.PREPROCESS_KIND,
        )
    log_tail = ""
    if job:
        log_path = Path(job.get("log_path") or "")
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                log_tail = "\n".join(text.splitlines()[-50:])
            except Exception:  # noqa: BLE001
                log_tail = ""
    return {
        "job": job,
        "log_tail": log_tail,
        "summary": preprocess_svc.summary_train(p, v["label"]),
    }


@router.get("/api/projects/{pid}/versions/{vid}/preprocess/files")
def list_preprocess_files_train(pid: int, vid: int) -> dict[str, Any]:
    """train scope: list all images under versions/{label}/train/ + manifest metadata.

    Under the new model, the list_pending / list_processed binary concept is gone (see ADR
    0010 §Manifest schema v2); a unified `images` list is returned instead, and the frontend
    renders status badges based on differences in entry fields. The response still includes
    `summary`, matching the old endpoint.
    """
    p, v = _resolve_pv_or_404(pid, vid)
    return {
        "images": preprocess_svc.list_train_images(p, v["label"]),
        "summary": preprocess_svc.summary_train(p, v["label"]),
    }


@router.get("/api/projects/{pid}/versions/{vid}/preprocess/duplicates/removed")
def list_duplicate_removed_train(pid: int, vid: int) -> dict[str, Any]:
    """train scope: list manifest entries flagged by dedup review for the "Removed" tab."""
    p, v = _resolve_pv_or_404(pid, vid)
    return {
        "images": preprocess_svc.list_duplicate_removed_workspace_train(
            p, v["label"]
        ),
    }


@router.get("/api/projects/{pid}/versions/{vid}/preprocess/crop/workspace")
def list_crop_workspace_train_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """train scope: crop-page working set = all of train/{folder}/{image} + pixel size +
    processed flag."""
    p, v = _resolve_pv_or_404(pid, vid)
    return {"images": preprocess_svc.list_crop_workspace_train(p, v["label"])}


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/crop")
def start_preprocess_crop_train(
    pid: int, vid: int, body: PreprocessCropRequest,
) -> dict[str, Any]:
    """train scope: create a crop job. `crops`'s source filenames are train-relative paths
    (`"1_data/X.png"`, matching the `name` returned by list_crop_workspace_train)."""
    if not body.crops:
        raise ValidationError(
            "No crop regions provided",
            code="preprocess.crops_required", http_status=400,
        )
    _resolve_pv_or_404(pid, vid)
    crops_payload: dict[str, list[dict[str, Any]]] = {
        name: [r.model_dump() for r in rects]
        for name, rects in body.crops.items()
    }
    with db.connection_for() as conn:
        job = preprocess_svc.start_crop_job_train(
            conn, project_id=pid, version_id=vid, crops=crops_payload,
        )
    _publish_job_state(job)
    return job


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/inpaint/save")
async def inpaint_save_train_endpoint(
    pid: int, vid: int,
    name: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """train scope inpaint save: frontend exports the full canvas (PNG) to overwrite `train/{name}`.

    Written to disk synchronously (no job); the output is always `{folder}/{stem}.png`, and
    the manifest is marked processed=True (see core.inpaint_save_train). PIL encoding a large
    image can take seconds, so it runs in a threadpool to avoid blocking the event loop.
    """
    p, v = _resolve_pv_or_404(pid, vid)
    data = await file.read()
    res = await run_in_threadpool(
        preprocess_svc.inpaint_save_train, p, v["label"], name=name, data=data,
    )
    _publish_project_state(p)
    return res


@router.get("/api/projects/{pid}/versions/{vid}/preprocess/mask")
def get_mask_train_endpoint(pid: int, vid: int, name: str) -> FileResponse:
    """Training mask sidecar (grayscale PNG, same size as the source image). No mask -> 404
    (the frontend uses this to distinguish "never painted"). The frontend appends a
    mask_mtime cache-buster, so this only needs no-cache."""
    p, v = _resolve_pv_or_404(pid, vid)
    path = preprocess_svc.mask_file_train(p, v["label"], name=name)
    if path is None:
        raise NotFoundError(
            "Mask not found", code="preprocess.mask_not_found",
            details={"name": name},
        )
    return FileResponse(
        path, media_type="image/png",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


@router.put("/api/projects/{pid}/versions/{vid}/preprocess/mask")
async def put_mask_train_endpoint(
    pid: int, vid: int,
    name: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Write the training mask (grayscale PNG exported from the frontend's mask layer). Synchronous disk write, no job."""
    p, v = _resolve_pv_or_404(pid, vid)
    data = await file.read()
    res = await run_in_threadpool(
        preprocess_svc.mask_save_train, p, v["label"], name=name, data=data,
    )
    _publish_project_state(p)
    return res


@router.delete("/api/projects/{pid}/versions/{vid}/preprocess/mask")
def delete_mask_train_endpoint(pid: int, vid: int, name: str) -> dict[str, Any]:
    """Delete the training mask (= this image reverts to normal full-image training)."""
    p, v = _resolve_pv_or_404(pid, vid)
    res = preprocess_svc.mask_delete_train(p, v["label"], name=name)
    _publish_project_state(p)
    return res


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/files/reset")
def reset_preprocess_files_train(pid: int, vid: int) -> dict[str, Any]:
    """train scope: clear train manifest state (**does not touch** the physical files in
    train/, see ADR 0010 §train_clear_all decision). list_train_images downstream can still
    list the physical images; only the entry metadata is gone. The UI falls back to the
    "unprocessed" status badge.
    """
    p, v = _resolve_pv_or_404(pid, vid)
    pdir = projects.project_dir(p["id"], p["slug"])
    preprocess_manifest.train_clear_all(pdir, v["label"])
    _publish_project_state(p)
    return {"ok": True}


@router.post("/api/projects/{pid}/versions/{vid}/preprocess/files/restore")
def restore_preprocess_files_train(
    pid: int, vid: int, body: PreprocessRestoreRequest,
) -> dict[str, Any]:
    """train scope restore: copy from `download/{entry.origin}` back over `train/{name}`.
    Returns three groups, `{restored, missing, no_origin}` (see ADR 0010 §Restore semantics);
    `no_origin` feeds the frontend's three-option UI [drag in a replacement / keep / remove].
    """
    if not body.names:
        return {"restored": [], "missing": [], "no_origin": []}
    p, v = _resolve_pv_or_404(pid, vid)
    res = preprocess_svc.restore_products_train(p, v["label"], body.names)
    if res["restored"]:
        _publish_project_state(p)
    return res


