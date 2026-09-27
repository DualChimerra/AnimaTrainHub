"""Shared helpers for the projects/ subpackage (PR-6.5, extracted from server.py).

Only serves sub-routers inside the projects subpackage; routers outside the projects
domain should not import from here (avoids reverse / lateral coupling).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .... import db
from ....domain.errors import NotFoundError
from ....services.projects import projects, versions
from ....infrastructure.event_bus import bus
from ....services import version_config


def _project_payload(p: dict[str, Any]) -> dict[str, Any]:
    """External detail payload: the project itself + versions[] including stats + download stats."""
    out = dict(p)
    out.update(projects.stats_for_project(p))
    with db.connection_for() as conn:
        vs = versions.list_versions(conn, p["id"])
    out["versions"] = [
        {**v, "stats": versions.stats_for_version(p, v)} for v in vs
    ]
    return out


def _publish_project_state(p: dict[str, Any]) -> None:
    bus.publish({
        "type": "project_state_changed",
        "project_id": p["id"],
    })


def _publish_version_state(v: dict[str, Any]) -> None:
    bus.publish({
        "type": "version_state_changed",
        "project_id": v["project_id"],
        "version_id": v["id"],
        "status": versions.get_status(v),
        "phase": versions.get_phase(v),
    })


def _publish_job_state(job: dict[str, Any]) -> None:
    bus.publish({
        "type": "job_state_changed",
        "job_id": job["id"],
        "project_id": job["project_id"],
        "version_id": job.get("version_id"),
        "kind": job["kind"],
        "status": job["status"],
    })


def _version_dir_or_404(pid: int, vid: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Returns (project, version, version_dir)."""
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found",
                details={"id": vid},
            )
        p = projects.get_project(conn, pid)
    assert p is not None
    return p, v, versions.version_dir(p["id"], p["slug"], v["label"])


def _version_train_dir_or_404(pid: int, vid: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Returns (project, version, version_dir/train)."""
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        if not v or v["project_id"] != pid:
            raise NotFoundError(
                "Version not found", code="version.not_found",
                details={"id": vid},
            )
        p = projects.get_project(conn, pid)
    assert p is not None
    return p, v, versions.version_dir(p["id"], p["slug"], v["label"]) / "train"


def _project_and_version_or_404(
    pid: int, vid: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolves the (project, version) pair for the version_config domain.

    VersionConfigError (project/version not_found, already carrying code +
    http_status=404) bubbles straight up to the global DomainError handler.
    """
    with db.connection_for() as conn:
        return version_config.get_project_and_version(conn, pid, vid)


def _reg_dir(vdir: Path) -> Path:
    """reg root directory -- subfolders directly mirror the train subfolder (matches the
    source script, no 1_general intermediate layer)."""
    return vdir / "reg"
