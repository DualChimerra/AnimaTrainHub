"""studio_data storage location — query / migrate to a custom directory.

3 routes:
    GET  /api/studio-data/info            current/default location + full scan (file count/bytes)
    POST /api/studio-data/migrate         validate + start a background copy thread (progress via SSE)
    GET  /api/studio-data/migrate_status  migration status snapshot (fallback for modal reopen / missed SSE events)

Migration protocol: once the copy finishes, writes the `studio_data_location.json`
pointer at the repo root — **takes effect after a server restart** (paths.STUDIO_DATA
is resolved at import time; cli.py's restart loop spawns a fresh process that
re-resolves it). Data at the old location is kept, not deleted. Progress events:
`studio_data_migrate_progress` / `_done`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..schemas.studio_data import StudioDataMigrateRequest
from ...domain.errors import ConflictError, ValidationError
from ...infrastructure.paths import DEFAULT_STUDIO_DATA, STUDIO_DATA
from ...services import studio_data as svc
from ..deps import _check_no_running_tasks

router = APIRouter()


@router.get("/api/studio-data/info")
def studio_data_info(scan: bool = True) -> dict[str, Any]:
    """Current / default location; when scan=true, includes a full scan (large
    directories can take a few seconds — the frontend confirmation modal shows
    a loading state while it waits; the Settings page only shows the path and
    uses scan=false to avoid the disk scan)."""
    return {
        "current": str(STUDIO_DATA),
        "default": str(DEFAULT_STUDIO_DATA),
        "is_custom": STUDIO_DATA.resolve() != DEFAULT_STUDIO_DATA.resolve(),
        "scan": svc.scan_studio_data() if scan else None,
    }


@router.post("/api/studio-data/migrate")
def studio_data_migrate(body: StudioDataMigrateRequest) -> dict[str, Any]:
    """Start a migration. Constraint: no running task (if training keeps writing files during the copy, you'd end up copying half-written data)."""
    _check_no_running_tasks()
    try:
        svc.start_migration(Path(body.target))
    except ValueError as exc:
        raise ValidationError(
            f"Invalid target location: {exc}",
            code="studio_data.target_invalid",
            details={"reason": str(exc)}, http_status=422,
        ) from exc
    except RuntimeError as exc:
        raise ConflictError(
            "A data migration is already in progress",
            code="studio_data.migration_busy",
        ) from exc
    return {"ok": True}


@router.get("/api/studio-data/migrate_status")
def studio_data_migrate_status() -> dict[str, Any]:
    return svc.migration_status()
