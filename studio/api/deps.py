"""Shared dependency helpers (extracted from server.py starting with PR-6).

Helpers shared across routers: getting the Supervisor instance, validating
version / project records, etc. These will eventually move to FastAPI's
`Depends(...)` style, but for now they stay as plain function calls to keep
this change behavior-neutral.
"""
from __future__ import annotations

from typing import Optional

from ..domain.errors import DomainError, ValidationError
from ..supervisor import Supervisor


def _check_no_running_tasks() -> None:
    """Precondition for restart / migration / asset deletion: every task must
    be done / failed / canceled / pending.

    If any task is running, raise 422 immediately with the task list, so the
    frontend can show the user a friendly prompt ("pause the following tasks
    first"). This fork: upstream originally kept this in routers/system.py
    (the self-update router, removed in this fork); it has been moved to
    deps.py to be shared by models_storage / studio_data.
    """
    from .. import db  # noqa: PLC0415 — late import to avoid a circular import
    with db.connection_for() as conn:
        running = db.list_tasks(conn, status="running")
    if running:
        raise ValidationError(
            "Tasks are running; cancel or wait for them to finish first",
            code="system.tasks_running",
            details={
                "tasks": [
                    {
                        "id": t["id"],
                        "name": t.get("name", ""),
                        "task_type": t.get("task_type", "train"),
                    }
                    for t in running
                ],
            },
            http_status=422,
        )


def _supervisor() -> Supervisor:
    """Get the Supervisor from app.state. Returns 503 if lifespan startup
    hasn't finished yet.

    This helper does a late import to avoid a three-way circular import
    between `api/app.py`, `api/routers/*`, and `api/deps.py` — routers are
    still initializing when app.py includes them, so importing api.app at
    that point would technically get `app` (since `app = FastAPI(...)` has
    already executed), but the circular relationship isn't healthy.
    """
    from .app import app
    sup: Optional[Supervisor] = getattr(app.state, "supervisor", None)
    if sup is None:
        raise DomainError(
            "The service is still starting; try again in a moment",
            code="system.starting", http_status=503,
        )
    return sup


def _resolve_model_paths(
    base_model: Optional[str] = None, *, family: str = "anima"
) -> dict[str, str]:
    """Resolve default base-model paths (shared by prior generation / test
    image generation).

    Uses the same resolution logic as creating a new training version
    (`default_paths_for_new_version`): when the user switches the selected
    base model under Settings -> Models (an official variant or a registered
    local custom `.safetensors`), it also affects the primary weights path
    used here — which is what makes "test-generate on a fine-tuned checkpoint"
    possible.

    A non-empty `base_model` overrides the base model for this request only
    (e.g. when the "base model" dropdown on the prior generation / test page
    is set to something other than the default) — only the transformer
    weights are swapped, all other paths still follow the global settings.

    `family` is taken by the caller from the request / config; after the
    generate-side request schema gained model_family (P4-4), this function
    was renamed to drop the anima prefix.
    """
    from ..services.models import default_paths_for_new_version
    return default_paths_for_new_version(base_model, family=family)
