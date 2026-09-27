"""Tag list endpoints (English tags for autocomplete).

4 routes:
    GET  /api/tag-dictionary/meta     meta of the current list (the frontend pings it on start)
    GET  /api/tag-dictionary/data     the whole list (~100k tags) for in-memory autocomplete
    POST /api/tag-dictionary/upload   multipart csv/txt upload that replaces the list
    POST /api/tag-dictionary/reset    download the default list again
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from ...domain.errors import DomainError, NotFoundError, ValidationError
from ...infrastructure import tag_dictionary as td

router = APIRouter()


@router.get("/api/tag-dictionary/meta")
def get_meta() -> dict[str, Any]:
    meta = td.get_meta()
    return {"loaded": meta is not None, "meta": meta}


@router.get("/api/tag-dictionary/data")
def get_data() -> JSONResponse:
    loaded = td.load_active()
    if loaded is None:
        raise NotFoundError(
            "Tag dictionary is not loaded", code="tag.dictionary_not_loaded",
        )
    tags, meta = loaded
    # Cached for 5 minutes; after an upload / reset the frontend compares
    # meta.downloaded_at and reloads on its own.
    return JSONResponse(
        content={"tags": tags, "meta": meta},
        headers={"Cache-Control": "public, max-age=300"},
    )


@router.post("/api/tag-dictionary/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    try:
        meta = td.apply_uploaded(content, file.filename or "user-upload")
    except ValueError as exc:
        raise ValidationError(
            f"Invalid tag dictionary file: {exc}",
            code="tag.dictionary_upload_invalid",
            details={"reason": str(exc)}, http_status=400,
        ) from exc
    return {"loaded": True, "meta": meta}


@router.post("/api/tag-dictionary/reset")
def reset() -> dict[str, Any]:
    try:
        meta = td.reset_to_default()
    except RuntimeError as exc:
        raise DomainError(
            f"Failed to download the default tag dictionary: {exc}",
            code="tag.dictionary_download_failed",
            details={"reason": str(exc)}, http_status=502,
        ) from exc
    return {"loaded": True, "meta": meta}
