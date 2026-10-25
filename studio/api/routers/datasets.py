"""Shared dataset library.

routes:
    GET    /api/library/datasets                                   list (stats, covers, cache size, usage)
    POST   /api/library/datasets                                   create (optionally from a project's images)
    GET    /api/library/datasets/{did}                             one dataset (same payload as a list row)
    PATCH  /api/library/datasets/{did}                             rename / note
    DELETE /api/library/datasets/{did}                             delete folder + row (trainings keep their copies)
    POST   /api/library/datasets/{did}/import-project              add a project's own download/ images
    GET    /api/library/datasets/{did}/cache                       cache folders (what / size / last use)
    DELETE /api/library/datasets/{did}/cache                       clear cache (?kind=&group= narrows it)
    GET    /api/projects/{pid}/versions/{vid}/datasets     datasets this version curates from
    PUT    /api/projects/{pid}/versions/{vid}/datasets     replace that list

Images, uploads, booru downloads and thumbnails go through the project routes
with the dataset id (a dataset is a projects row with kind='dataset').
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter

from ...domain.errors import NotFoundError
from ..schemas.datasets import DatasetCreate, DatasetImport, DatasetUpdate, VersionDatasetsPut
from ... import db
from ...infrastructure.event_bus import bus
from ...services.dataset import library
from ...services.projects import projects, versions

router = APIRouter()


def _publish(dataset_id: int) -> None:
    # Same event the project pages listen to: datasets are project rows.
    bus.publish({"type": "project_state_changed", "project_id": dataset_id})


@router.get("/api/library/datasets")
def list_datasets() -> dict[str, Any]:
    with db.connection_for() as conn:
        return {"items": library.list_datasets(conn)}


@router.post("/api/library/datasets")
def create_dataset(body: DatasetCreate) -> dict[str, Any]:
    with db.connection_for() as conn:
        d = library.create_dataset(conn, title=body.title, note=body.note)
        imported: Optional[dict[str, list[str]]] = None
        if body.from_project_id is not None:
            imported = library.import_project_images(conn, d["id"], body.from_project_id)
        out = library.summarize(conn, library.get_dataset(conn, d["id"]))
    _publish(d["id"])
    if imported is not None:
        out["imported"] = imported
    return out


@router.get("/api/library/datasets/{did}")
def get_dataset(did: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        return library.summarize(conn, library.get_dataset(conn, did))


@router.patch("/api/library/datasets/{did}")
def patch_dataset(did: int, body: DatasetUpdate) -> dict[str, Any]:
    fields = body.model_dump(exclude_unset=True)
    with db.connection_for() as conn:
        library.get_dataset(conn, did)
        projects.update_project(conn, did, **fields)
        out = library.summarize(conn, library.get_dataset(conn, did))
    _publish(did)
    return out


@router.delete("/api/library/datasets/{did}")
def delete_dataset(did: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        library.delete_dataset(conn, did)
    _publish(did)
    return {"deleted": did}


@router.post("/api/library/datasets/{did}/import-project")
def import_project(did: int, body: DatasetImport) -> dict[str, Any]:
    with db.connection_for() as conn:
        result = library.import_project_images(conn, did, body.project_id)
    _publish(did)
    return result


@router.get("/api/library/datasets/{did}/cache")
def dataset_cache(did: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        d = library.get_dataset(conn, did)
    groups = library.cache_groups(d)
    return {
        "groups": groups,
        "total_files": sum(g["files"] for g in groups),
        "total_bytes": sum(g["bytes"] for g in groups),
        "path": str(projects.project_dir(d["id"], d["slug"]) / library.CACHE_DIRNAME),
    }


@router.delete("/api/library/datasets/{did}/cache")
def clear_dataset_cache(did: int, kind: Optional[str] = None, group: Optional[str] = None) -> dict[str, Any]:
    with db.connection_for() as conn:
        d = library.get_dataset(conn, did)
    freed = library.clear_cache(d, kind, group)
    _publish(did)
    return {"freed_bytes": freed}


# ---------------------------------------------------------------------------
# version <-> datasets
# ---------------------------------------------------------------------------


def _version_or_404(conn, pid: int, vid: int) -> dict[str, Any]:
    v = versions.get_version(conn, vid)
    if not v or v["project_id"] != pid:
        raise NotFoundError("Version not found", code="version.not_found", details={"id": vid})
    return v


@router.get("/api/projects/{pid}/versions/{vid}/datasets")
def get_version_datasets(pid: int, vid: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        _version_or_404(conn, pid, vid)
        return {"items": [library.summarize(conn, d) for d in library.linked_datasets(conn, vid)]}


@router.put("/api/projects/{pid}/versions/{vid}/datasets")
def put_version_datasets(pid: int, vid: int, body: VersionDatasetsPut) -> dict[str, Any]:
    with db.connection_for() as conn:
        v = _version_or_404(conn, pid, vid)
        linked = library.set_linked_datasets(conn, vid, body.dataset_ids)
        items = [library.summarize(conn, d) for d in linked]
    bus.publish({
        "type": "version_state_changed",
        "project_id": pid,
        "version_id": vid,
        "status": versions.get_status(v),
        "phase": versions.get_phase(v),
    })
    return {"items": items}
