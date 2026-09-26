"""Upscaler switching (extracted from server.py in PR-6 commit 2).

1 route:
    POST /api/upscalers/select  switch the default upscaler (preset / custom filename / local path)

The old custom-download endpoint (POST /api/upscalers/download_custom) has been
replaced by unified source candidates (POST /api/model-sources/upscaler registers
a candidate + the generic /api/models/download triggers it, with model_id=upscaler_custom).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..schemas.models import UpscalerSelectRequest
from ... import secrets
from ...domain.errors import InvalidPathError, NotFoundError, ValidationError
from ...services import models as model_downloader

router = APIRouter()


@router.post("/api/upscalers/select")
def select_upscaler(body: UpscalerSelectRequest) -> dict[str, Any]:
    """Switch the default upscaler. Writes to secrets.models.selected_upscaler.

    Accepts a preset label or an existing local custom filename; an invalid
    value (neither a preset nor present in the upscalers/ directory) returns 400.
    """
    label = body.label.strip()
    if not label:
        raise ValidationError(
            "Upscaler name is required", code="upscaler.label_required", http_status=400,
        )
    valid = label in model_downloader.UPSCALER_VARIANTS
    if not valid:
        # Custom filename: must already exist on disk
        try:
            target = model_downloader.upscaler_target(label)
        except ValueError as exc:
            raise InvalidPathError("Invalid path", details={"reason": str(exc)}) from exc
        if not target.exists():
            raise NotFoundError(
                f'Upscaler "{label}" not found',
                code="upscaler.not_found", details={"name": label},
            )
    cur = secrets.load()
    new_models = cur.models.model_copy(update={"selected_upscaler": label})
    new = cur.model_copy(update={"models": new_models})
    secrets.save(new)
    return {"selected": label}
