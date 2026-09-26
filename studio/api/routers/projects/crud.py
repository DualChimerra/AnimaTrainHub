"""Projects + Versions CRUD + phase advancement + ckpts (PR-6.5 commit 1, extracted from server.py).

18 routes:
    GET    /api/projects                                       list (enriched with active version; includes archived rows)
    POST   /api/projects                                       create (optional initial version)
    GET    /api/projects/{pid}                                 get (includes versions list)
    PATCH  /api/projects/{pid}                                 update
    DELETE /api/projects/{pid}                                 delete
    POST   /api/projects/{pid}/archive                         archive (soft-hide, reversible)
    POST   /api/projects/{pid}/unarchive                       unarchive
    GET    /api/projects/{pid}/versions                        list versions
    POST   /api/projects/{pid}/versions                        create version
    GET    /api/projects/{pid}/versions/{vid}                  get version
    PATCH  /api/projects/{pid}/versions/{vid}                  update version
    DELETE /api/projects/{pid}/versions/{vid}                  delete version
    POST   /api/projects/{pid}/versions/{vid}/activate         activate
    POST   /api/projects/{pid}/versions/{vid}/advance-phase    ADR-0007 §11.5-A
    POST   /api/projects/{pid}/versions/{vid}/skip-phase       ADR-0007 §11.5-B
    GET    /api/projects/{pid}/versions/{vid}/lora_ckpts       LoRA picker second tier (XY ckpt axis)
    GET    /api/projects/{pid}/state_ckpts                     resume_state picker (grouped by version)
    GET    /api/projects/{pid}/lora_ckpts                      resume_lora picker (grouped by version)
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter

from ....domain.errors import NotFoundError
from ...schemas.projects import (
    ProjectCreate,
    ProjectUpdate,
    VersionCreate,
    VersionUpdate,
)
from ._shared import (
    _project_payload,
    _publish_project_state,
    _publish_version_state,
    _version_dir_or_404,
)
from .... import db
from ....services.projects import projects, versions, phase as versions_phase

router = APIRouter()


@router.get("/api/projects")
def list_projects_endpoint() -> dict[str, Any]:
    """ADR-0007 §11.8-E: enrich with active version label + status, for the card's
    top-right badge.

    Since v12 also carries archived_at (archived rows aren't filtered here -- project
    counts are small, so splitting archived/active is left to the frontend, saving a
    round trip when switching views) + active_version_phase (while preparing, the badge
    shows the current phase, e.g. "preparing - tagging").
    """
    with db.connection_for() as conn:
        rows = projects.list_projects(conn)
        enriched: list[dict[str, Any]] = []
        for r in projects.projects_with_stats(rows):
            r["active_version_label"] = None
            r["active_version_status"] = None
            r["active_version_phase"] = None
            avid = r.get("active_version_id")
            if avid:
                av = versions.get_version(conn, int(avid))
                if av:
                    r["active_version_label"] = av["label"]
                    r["active_version_status"] = versions.get_status(av)
                    r["active_version_phase"] = versions.get_phase(av)
            enriched.append(r)
    return {"items": enriched}


@router.post("/api/projects")
def create_project_endpoint(body: ProjectCreate) -> dict[str, Any]:
    with db.connection_for() as conn:
        p = projects.create_project(
            conn, title=body.title, slug=body.slug, note=body.note
        )
        if body.initial_version_label:
            # Project is already created; if version creation fails, VersionError
            # carries a code and bubbles straight up to the global handler
            versions.create_version(
                conn, project_id=p["id"], label=body.initial_version_label
            )
        p = projects.get_project(conn, p["id"])
    assert p is not None
    _publish_project_state(p)
    return _project_payload(p)


@router.get("/api/projects/{pid}")
def get_project_endpoint(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
    if not p:
        raise NotFoundError(
            "Project not found", code="project.not_found", details={"id": pid},
        )
    return _project_payload(p)


@router.patch("/api/projects/{pid}")
def patch_project_endpoint(pid: int, body: ProjectUpdate) -> dict[str, Any]:
    fields = body.model_dump(exclude_unset=True)
    with db.connection_for() as conn:
        # ProjectError (including _must_get's project.not_found) carries a code and bubbles straight up
        p = projects.update_project(conn, pid, **fields)
    _publish_project_state(p)
    return _project_payload(p)


@router.post("/api/projects/{pid}/archive")
def archive_project_endpoint(pid: int) -> dict[str, Any]:
    """Archive: soft-hidden from the list, directory / versions / tasks are left untouched, reversible any time."""
    with db.connection_for() as conn:
        p = projects.set_archived(conn, pid, True)
    _publish_project_state(p)
    return _project_payload(p)


@router.post("/api/projects/{pid}/unarchive")
def unarchive_project_endpoint(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        p = projects.set_archived(conn, pid, False)
    _publish_project_state(p)
    return _project_payload(p)


@router.delete("/api/projects/{pid}")
def delete_project_endpoint(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        projects.delete_project(conn, pid)
    return {"deleted": pid}


# Versions ------------------------------------------------------------------


@router.get("/api/projects/{pid}/versions")
def list_versions_endpoint(pid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise NotFoundError(
                "Project not found", code="project.not_found",
                details={"id": pid},
            )
        vs = versions.list_versions(conn, pid)
        p = projects.get_project(conn, pid)
    assert p is not None
    return {
        "items": [
            {**v, "stats": versions.stats_for_version(p, v)} for v in vs
        ]
    }


@router.get("/api/projects/{pid}/versions/{vid}/lora_ckpts")
def list_version_lora_ckpts(pid: int, vid: int) -> dict[str, Any]:
    """List all .safetensors under version output/ (step / epoch / final), used for the
    LoRA picker's second tier (XY ckpt axis + switching ckpt in single-image mode)."""
    p, v, vdir = _version_dir_or_404(pid, vid)
    return {"items": versions.list_lora_ckpts(vdir)}


@router.get("/api/projects/{pid}/state_ckpts")
def list_project_state_ckpts(pid: int) -> dict[str, Any]:
    """List training_state_step*.pt across all of a project's versions, grouped by version.

    Used by the Train page's resume_state field "browse this project" picker: the user
    sees a semantic file list grouped by version, and once one is picked the frontend
    writes the absolute path into the field.
    """
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        if not p:
            raise NotFoundError(
                "Project not found", code="project.not_found",
                details={"id": pid},
            )
        return {"groups": versions.list_project_state_ckpts(conn, p)}


@router.get("/api/projects/{pid}/lora_ckpts")
def list_project_lora_ckpts(pid: int) -> dict[str, Any]:
    """List LoRA ckpts (.safetensors) across all of a project's versions, grouped by version.

    Used by the Train page's resume_lora field "browse this project" picker.
    """
    with db.connection_for() as conn:
        p = projects.get_project(conn, pid)
        if not p:
            raise NotFoundError(
                "Project not found", code="project.not_found",
                details={"id": pid},
            )
        return {"groups": versions.list_project_lora_ckpts(conn, p)}


@router.post("/api/projects/{pid}/versions")
def create_version_endpoint(pid: int, body: VersionCreate) -> dict[str, Any]:
    with db.connection_for() as conn:
        if not projects.get_project(conn, pid):
            raise NotFoundError(
                "Project not found", code="project.not_found",
                details={"id": pid},
            )
        # VersionError (label_invalid / label_exists / fork_source_invalid etc.)
        # carries a code and bubbles straight up to the global DomainError handler
        v = versions.create_version(
            conn,
            project_id=pid,
            label=body.label,
            fork_from_version_id=body.fork_from_version_id,
            note=body.note,
        )
    _publish_version_state(v)
    return v


@router.get("/api/projects/{pid}/versions/{vid}")
def get_version_endpoint(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        p = projects.get_project(conn, pid)
    if not v or v["project_id"] != pid:
        raise NotFoundError(
            "Version not found", code="version.not_found", details={"id": vid},
        )
    assert p is not None
    return {**v, "stats": versions.stats_for_version(p, v)}


@router.patch("/api/projects/{pid}/versions/{vid}")
def patch_version_endpoint(
    pid: int, vid: int, body: VersionUpdate,
) -> dict[str, Any]:
    fields = body.model_dump(exclude_unset=True)
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found",
                details={"id": vid},
            )
        # VersionError (label_invalid / label_exists etc.) carries a code and bubbles straight up
        v = versions.update_version(conn, vid, **fields)
    _publish_version_state(v)
    return v


@router.delete("/api/projects/{pid}/versions/{vid}")
def delete_version_endpoint(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found",
                details={"id": vid},
            )
        versions.delete_version(conn, vid)
    return {"deleted": vid}


@router.post("/api/projects/{pid}/versions/{vid}/activate")
def activate_version_endpoint(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found",
                details={"id": vid},
            )
        versions.activate_version(conn, vid)
        p = projects.get_project(conn, pid)
    assert p is not None
    _publish_project_state(p)
    # Thin response: does not return the full _project_payload (per-version filesystem
    # stats can take seconds on a multi-version project). The frontend's optimistic
    # update already holds the new value; full data converges via project_state_changed -> reload.
    return {"active_version_id": vid}


# ---------------------------------------------------------------------------
# Phase cursor advance / skip -- ADR-0007 §11.5-A / §11.5-B
# ---------------------------------------------------------------------------


def _phase_advance_payload(
    advanced: bool, result: versions_phase.CheckResult,
    new_phase: Optional[str], version: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "advanced": advanced,
        "ok": result.ok,
        "reason": result.reason,
        "new_phase": new_phase,
        "version": version,
    }


@router.post("/api/projects/{pid}/versions/{vid}/advance-phase")
def advance_phase_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """Advance the phase cursor -- called by the "next step" button (ADR-0007 §11.5-A).

    On success -> cursor++ + return the new phase + publish version_state_changed.
    On failure -> ok=False + reason (frontend toast), cursor unchanged.
    """
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found",
                details={"id": vid},
            )
        advanced, result, new_phase = versions_phase.advance_phase(conn, vid)
        v_after = versions.get_version(conn, vid)
    if advanced and v_after is not None:
        _publish_version_state(v_after)
    return _phase_advance_payload(advanced, result, new_phase, v_after)


@router.post("/api/projects/{pid}/versions/{vid}/skip-phase")
def skip_phase_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """Skip a skippable phase (currently only regularizing; ADR-0007 §11.5-A)."""
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found",
                details={"id": vid},
            )
        advanced, result, new_phase = versions_phase.skip_phase(conn, vid)
        v_after = versions.get_version(conn, vid)
    if advanced and v_after is not None:
        _publish_version_state(v_after)
    return _phase_advance_payload(advanced, result, new_phase, v_after)
