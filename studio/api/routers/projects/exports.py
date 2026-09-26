"""Train / bundle export + cross-project import (extracted from server.py in PR-6.5 commit 2).

6 routes:
    GET  /api/projects/{pid}/versions/{vid}/train.zip           zip train/ + manifest.json on the fly
    GET  /api/projects/{pid}/versions/{vid}/bundle.zip          zip a bundle on the fly per options (schema v2)
    POST /api/projects/{pid}/versions/{vid}/export-bundle       zip into data_exports/
    POST /api/projects/import-bundle                            import from a PathPicker path / data_exports
    POST /api/projects/import-bundle/upload                     import via uploaded zip
    POST /api/projects/import-train                             upload a training set zip -> new project + v1

train.zip / bundle.zip / export-bundle use ZIP_STORED (PNG/jpg are already compressed, re-compressing
just wastes CPU). On success/failure they publish version_train_zip_ready / _failed +
version_bundle_zip_ready / _failed -- the frontend's <a> direct link triggers the download, and the
SSE event clears the app-side "zipping..." state.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from ...errors import _data_export_path, _export_result, _unique_data_export_path
from ....domain.errors import NotFoundError, ValidationError
from ...schemas.exports import BundleImportBody, BundleOptionsBody
from ._shared import (
    _project_payload,
    _publish_project_state,
    _publish_version_state,
)
from .... import db
from ....services.projects import projects, versions
from ....infrastructure.event_bus import bus
from ....paths import DATA_EXPORTS, REPO_ROOT, USER_PRESETS_DIR
from ....services.data_io import train_io

router = APIRouter()


@router.get("/api/projects/{pid}/versions/{vid}/train.zip")
def export_version_train_zip(
    pid: int, vid: int, background: BackgroundTasks,
) -> FileResponse:
    """Zip the version's train/ + manifest.json for a one-off download.

    Implementation: write to a temp file, then FileResponse; BackgroundTasks cleans up after
    the response is sent. Uses ZIP_STORED like outputs.zip (PNG/jpg are already compressed,
    re-compressing just wastes CPU).

    On success/failure, publishes version_train_zip_ready / _failed -- the frontend uses an
    <a> direct link to trigger the download (native browser progress bar), and the SSE event
    clears the app-side "zipping..." state + pops a toast on failure. Same pattern as
    outputs.zip.
    """
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found", details={"id": vid},
            )
        p = projects.get_project(conn, pid)
        assert p is not None

        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            train_io.export_train(conn, vid, tmp_path)
        except train_io.TrainIOError as exc:
            tmp_path.unlink(missing_ok=True)
            bus.publish({
                "type": "version_train_zip_failed",
                "project_id": pid,
                "version_id": vid,
                "error": str(exc),
            })
            raise
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            bus.publish({
                "type": "version_train_zip_failed",
                "project_id": pid,
                "version_id": vid,
                "error": str(exc),
            })
            raise

    bus.publish({
        "type": "version_train_zip_ready",
        "project_id": pid,
        "version_id": vid,
    })
    background.add_task(lambda: tmp_path.unlink(missing_ok=True))
    archive_name = f"{p['slug']}-{v['label']}.train.zip"
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=archive_name,
        background=background,
    )


@router.get("/api/projects/{pid}/versions/{vid}/bundle.zip")
def export_version_bundle(
    pid: int,
    vid: int,
    background: BackgroundTasks,
    train: bool = True,
    train_captions: bool = True,
    reg: bool = False,
    reg_captions: bool = False,
    include_config: bool = False,
    train_latent_cache: bool = False,
    reg_latent_cache: bool = False,
    train_masks: bool = False,
) -> FileResponse:
    """Zip a bundle.zip on the fly per options (schema_version 2) and hand it to the browser to download."""
    opts = train_io.BundleOptions(
        train=train,
        train_captions=train_captions,
        reg=reg,
        reg_captions=reg_captions,
        include_config=include_config,
        train_latent_cache=train_latent_cache,
        reg_latent_cache=reg_latent_cache,
        train_masks=train_masks,
    )

    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found", details={"id": vid},
            )
        p = projects.get_project(conn, pid)
        assert p is not None

        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            train_io.export_bundle(conn, vid, tmp_path, opts)
        except train_io.TrainIOError as exc:
            tmp_path.unlink(missing_ok=True)
            bus.publish({"type": "version_bundle_zip_failed", "project_id": pid, "version_id": vid, "error": str(exc)})
            raise
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            bus.publish({"type": "version_bundle_zip_failed", "project_id": pid, "version_id": vid, "error": str(exc)})
            raise

    bus.publish({"type": "version_bundle_zip_ready", "project_id": pid, "version_id": vid})
    background.add_task(lambda: tmp_path.unlink(missing_ok=True))
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=f"{p['slug']}-{v['label']}.bundle.zip",
        background=background,
    )


@router.post("/api/projects/{pid}/versions/{vid}/export-bundle")
def export_version_bundle_to_data_exports(
    pid: int,
    vid: int,
    body: BundleOptionsBody,
) -> dict[str, Any]:
    """Zip a bundle.zip per options and save it to data_exports/."""
    opts = body.to_options()
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found", details={"id": vid},
            )
        p = projects.get_project(conn, pid)
        assert p is not None
        DATA_EXPORTS.mkdir(parents=True, exist_ok=True)
        dest = _unique_data_export_path(f"{p['slug']}-{v['label']}.bundle.zip")
        try:
            train_io.export_bundle(conn, vid, dest, opts)
        except train_io.TrainIOError as exc:
            dest.unlink(missing_ok=True)
            bus.publish({"type": "version_bundle_zip_failed", "project_id": pid, "version_id": vid, "error": str(exc)})
            raise
        except Exception as exc:
            dest.unlink(missing_ok=True)
            bus.publish({"type": "version_bundle_zip_failed", "project_id": pid, "version_id": vid, "error": str(exc)})
            raise

    bus.publish({"type": "version_bundle_zip_ready", "project_id": pid, "version_id": vid})
    return _export_result(dest)


def _bundle_import_payload(result: dict[str, Any]) -> dict[str, Any]:
    p = result["project"]
    _publish_project_state(p)
    _publish_version_state(result["version"])
    return {
        "project": _project_payload(p),
        "version": result["version"],
        "stats": result["stats"],
    }


def _import_bundle_from_path(dest: Path, original: str) -> dict[str, Any]:
    if not dest.exists():
        raise NotFoundError(
            f"Path not found: {original}",
            code="path.not_found", details={"path": original},
        )
    if not dest.is_file():
        raise ValidationError(
            "Selected path is not a file",
            code="path.not_a_file", http_status=400,
        )
    if dest.suffix.lower() != ".zip":
        raise ValidationError(
            "Select a .zip file",
            code="file.ext_invalid", details={"types": ".zip"}, http_status=400,
        )
    with db.connection_for() as conn:
        result = train_io.import_bundle(conn, dest, USER_PRESETS_DIR)
    return _bundle_import_payload(result)


@router.post("/api/projects/import-bundle")
def import_bundle_zip(body: BundleImportBody) -> dict[str, Any]:
    """Import a bundle from a PathPicker path or a data_exports filename (supports both v1/v2)."""
    if body.filename:
        return _import_bundle_from_path(_data_export_path(body.filename), body.filename)
    assert body.path is not None
    dest = Path(body.path)
    if not dest.is_absolute():
        dest = (REPO_ROOT / dest).resolve()
    else:
        dest = dest.resolve()
    return _import_bundle_from_path(dest, body.path)


@router.post("/api/projects/import-bundle/upload")
async def import_bundle_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a bundle zip -> create a new project + version.

    `_import_bundle_from_path` is a synchronous zip-extract + manifest-validate + write-to-disk
    operation, so it must run in run_in_threadpool or it will stall the event loop (see
    ingestion.upload_local_files for details).
    """
    if not file.filename:
        raise ValidationError(
            "No files uploaded", code="dataset.no_files", http_status=400,
        )
    if Path(file.filename).suffix.lower() != ".zip":
        raise ValidationError(
            "Select a .zip file",
            code="file.ext_invalid", details={"types": ".zip"}, http_status=400,
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.close()
        return await run_in_threadpool(
            _import_bundle_from_path, Path(tmp.name), file.filename,
        )
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass


@router.post("/api/projects/import-train")
async def import_train_zip(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a training set zip -> create a new project + v1 (stage=tagging), return the new project.

    train_io.import_train() is a synchronous zip-extract + db-write + image-write operation,
    so it runs in run_in_threadpool (see ingestion.upload_local_files for the root-cause
    explanation).
    """
    if not file.filename:
        raise ValidationError(
            "No files uploaded", code="dataset.no_files", http_status=400,
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        # UploadFile is already a SpooledTemporaryFile internally; large files spill to a temp disk file
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.close()
        tmp_path = Path(tmp.name)

        def _do_import() -> dict[str, Any]:
            with db.connection_for() as conn:
                return train_io.import_train(conn, tmp_path)

        result = await run_in_threadpool(_do_import)
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass

    p = result["project"]
    _publish_project_state(p)
    _publish_version_state(result["version"])
    return {
        "project": _project_payload(p),
        "version": result["version"],
        "stats": result["stats"],
    }
