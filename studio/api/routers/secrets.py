"""Global credentials / service configuration (extracted from server.py in PR-6 commit 2).

4 routes:
    GET /api/secrets    masked secrets snapshot (sensitive fields like API keys are masked)
    PUT /api/secrets    update secrets, returns the new masked snapshot
    GET /api/secrets/wandb/presets/{id}/export   download a wandb preset yaml (with the real api_key)
    POST /api/secrets/wandb/presets/import       import a wandb preset (yaml/json upload)
"""
from __future__ import annotations

import logging
import re
from typing import Any

import yaml
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response
from pydantic import ValidationError

from ... import secrets
from ...domain.errors import DomainError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/secrets")
def get_secrets() -> dict[str, Any]:
    return secrets.to_masked_dict(secrets.load())


@router.put("/api/secrets")
def put_secrets(body: dict[str, Any]) -> dict[str, Any]:
    new = secrets.update(body)
    # When the user changes generate.idle_timeout_minutes in Settings, sync it
    # to the running daemon immediately — otherwise it wouldn't take effect
    # until the next generation dispatch. Safe to call even if the daemon
    # hasn't started yet (it just no-ops and picks it up on the next dispatch).
    try:
        from ...services.inference.daemon import get_daemon
        get_daemon().sync_idle_timeout_from_secrets()
    except Exception:
        logger.warning("failed to sync idle_timeout to daemon", exc_info=True)
    return secrets.to_masked_dict(new)


@router.get("/api/secrets/wandb/presets/{preset_id}/export")
def export_wandb_preset(preset_id: str) -> Response:
    """Download a single wandb preset's yaml. **Contains the real api_key**
    (this is an explicit user export action that bypasses the masking done by
    GET /api/secrets) — keep the file safe."""
    preset = secrets.get_wandb_preset(preset_id)
    if preset is None:
        raise DomainError(
            f"WandB preset {preset_id!r} not found",
            code="secrets.wandb_preset_not_found",
            details={"id": preset_id},
            http_status=404,
        )
    text = yaml.safe_dump(
        preset.model_dump(), allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    # preset.id is normalized by the validator to [A-Za-z0-9_-], so it's safe to use directly in a filename
    return Response(
        content=text,
        media_type="application/yaml",
        headers={
            "Content-Disposition": f'attachment; filename="wandb-preset-{preset.id}.yaml"'
        },
    )


@router.post("/api/secrets/wandb/presets/import")
async def import_wandb_preset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a yaml/json file to import a wandb preset; a name collision gets an automatic suffix and becomes the current selection."""
    raw = await file.read()
    try:
        data = yaml.safe_load(raw.decode("utf-8"))  # yaml is a superset of json
    except Exception as exc:
        raise DomainError(
            f"WandB preset file could not be parsed: {exc}",
            code="secrets.wandb_preset_invalid",
            http_status=400,
        ) from exc
    stem = re.sub(r"\.(ya?ml|json)$", "", file.filename or "", flags=re.I)
    try:
        new, preset = secrets.import_wandb_preset(data, fallback_label=stem)
    except (ValueError, ValidationError) as exc:
        raise DomainError(
            f"WandB preset is invalid: {exc}",
            code="secrets.wandb_preset_invalid",
            http_status=400,
        ) from exc
    return {
        "id": preset.id,
        "label": preset.label,
        "secrets": secrets.to_masked_dict(new),
    }
