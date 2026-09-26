"""Models-root storage location — query / migrate to a custom directory (mirrors studio_data, no restart needed).

3 routes:
    GET  /api/models-root/info            current/default location + full scan (file count/bytes)
    POST /api/models-root/migrate         validate + start a background copy thread (progress via SSE)
    GET  /api/models-root/migrate_status  migration status snapshot (fallback for modal reopen / missed SSE events)

Migration protocol: once the copy finishes, updates `secrets.models.root`,
**taking effect immediately** (`models_root()` reads the secret live — no
pointer file, no restart needed). The old directory is kept, not deleted.
Progress events: `models_root_migrate_progress` / `_done`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..schemas.models_storage import ModelsRootMigrateRequest
from ...domain.errors import ConflictError, ValidationError
from ...services import models_storage as svc
from ...services.models import models_root
from ..deps import _check_no_running_tasks

router = APIRouter()


@router.get("/api/models-root/info")
def models_root_info(scan: bool = True) -> dict[str, Any]:
    """Current / default location; when scan=true, includes a full scan (large
    directories can take a few seconds — the frontend confirmation modal shows
    a loading state while it waits; the Settings page only shows the path and
    uses scan=false to avoid the disk scan)."""
    current = models_root()
    default = svc.default_models_root()
    return {
        "current": str(current),
        "default": str(default),
        "is_custom": current.resolve() != default.resolve(),
        "scan": svc.scan_models_root() if scan else None,
    }


@router.post("/api/models-root/migrate")
def models_root_migrate(body: ModelsRootMigrateRequest) -> dict[str, Any]:
    """Start a migration. Constraint: no running task (training would keep reading
    weights during the copy / a move's semantics aren't safe mid-training).

    If the target already has a non-empty models/ dir and on_conflict wasn't
    passed → 409 `target_conflict` (details include stats on the target's
    existing files + how many share a name); the frontend shows a
    "skip/overwrite/cancel" prompt and resends with on_conflict.
    """
    _check_no_running_tasks()
    try:
        svc.start_migration(Path(body.target), on_conflict=body.on_conflict)
    except svc.TargetConflictError as exc:
        # Note: must come before except ValueError (it's a subclass of ValueError)
        raise ConflictError(
            "Target already contains a non-empty models directory",
            code="models_root.target_conflict",
            details=exc.details,
        ) from exc
    except ValueError as exc:
        raise ValidationError(
            f"Invalid target location: {exc}",
            code="models_root.target_invalid",
            details={"reason": str(exc)}, http_status=422,
        ) from exc
    except RuntimeError as exc:
        raise ConflictError(
            "A models-root migration is already in progress",
            code="models_root.migration_busy",
        ) from exc
    return {"ok": True}


@router.get("/api/models-root/migrate_status")
def models_root_migrate_status() -> dict[str, Any]:
    return svc.migration_status()
