"""Checkpoint soup: merge several trained adapters into one file.

8 routes:
    GET    /api/soup/sources                 uploads + merged outputs in one call
    POST   /api/soup/upload                  bring your own .safetensors
    DELETE /api/soup/uploads/{name}          remove an upload
    DELETE /api/soup/outputs/{name}          remove a merged file
    GET    /api/soup/outputs/{name}/download save the merged file locally
    POST   /api/soup/inspect                 read the inputs + compatibility verdict
    POST   /api/soup/merge                   write the merged file

The merge itself is synchronous: adapter files are tens of MB, so a background
task would add a queue, a poll loop and a failure mode for no benefit. FastAPI
runs these ``def`` handlers in a threadpool, so the event loop stays free.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse

from ...domain.errors import ValidationError
from ..schemas.soup import SoupInspectBody, SoupMergeBody
from ...services import soup as soup_svc

router = APIRouter()
logger = logging.getLogger(__name__)

#: Refuse an upload that cannot plausibly be an adapter before buffering it all.
MAX_UPLOAD_BYTES = 4 * 1024 ** 3


def _fail(exc: soup_svc.SoupError) -> ValidationError:
    """SoupError is always something the user can fix → 422 with the reason."""
    return ValidationError(str(exc), details={"reason": str(exc)})


@router.get("/api/soup/sources")
def list_sources() -> dict[str, Any]:
    """Everything mergeable that is not tied to a project version.

    Project checkpoints come from the existing ``/api/projects/{pid}/lora_ckpts``
    endpoint — the picker shows both lists side by side.
    """
    soup_svc.ensure_dirs()
    return {"uploads": soup_svc.list_uploads(), "outputs": soup_svc.list_outputs()}


@router.post("/api/soup/upload")
async def upload_source(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValidationError("The file is larger than 4 GB.")
    try:
        return soup_svc.save_upload(file.filename or "upload", data)
    except soup_svc.SoupError as exc:
        raise _fail(exc) from exc


@router.delete("/api/soup/uploads/{name}")
def delete_upload(name: str) -> dict[str, Any]:
    try:
        soup_svc.delete_upload(name)
    except soup_svc.SoupError as exc:
        raise _fail(exc) from exc
    return {"ok": True}


@router.delete("/api/soup/outputs/{name}")
def delete_output(name: str) -> dict[str, Any]:
    try:
        soup_svc.delete_output(name)
    except soup_svc.SoupError as exc:
        raise _fail(exc) from exc
    return {"ok": True}


@router.get("/api/soup/outputs/{name}/download")
def download_output(name: str) -> FileResponse:
    try:
        path = soup_svc.output_path(name)
    except soup_svc.SoupError as exc:
        raise _fail(exc) from exc
    if not path.is_file():
        raise ValidationError("File not found.")
    return FileResponse(str(path), filename=path.name,
                        media_type="application/octet-stream")


@router.post("/api/soup/inspect")
def inspect_sources(body: SoupInspectBody) -> dict[str, Any]:
    """What the picker shows before the merge button becomes clickable."""
    try:
        return soup_svc.compatibility(body.paths)
    except soup_svc.SoupError as exc:
        raise _fail(exc) from exc


@router.post("/api/soup/merge")
def merge_sources(body: SoupMergeBody) -> dict[str, Any]:
    try:
        return soup_svc.merge(
            [{"path": i.path, "weight": i.weight} for i in body.inputs],
            body.name,
            method=body.method,
            overwrite=body.overwrite,
        )
    except soup_svc.SoupError as exc:
        raise _fail(exc) from exc
