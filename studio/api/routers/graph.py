"""Graph: store and compare LoRA training results.

Boards
    GET    /api/graph/boards                     list
    POST   /api/graph/boards                     create (default / empty / copy params)
    GET    /api/graph/boards/{bid}               board + every card and image
    PATCH  /api/graph/boards/{bid}               name / note / params / views
    DELETE /api/graph/boards/{bid}
    POST   /api/graph/boards/{bid}/remap         rewrite one parameter's value across cards
Cards (runs)
    POST   /api/graph/boards/{bid}/runs
    PATCH  /api/graph/runs/{rid}
    POST   /api/graph/runs/{rid}/duplicate       copy settings, never images
    DELETE /api/graph/runs/{rid}
    GET    /api/graph/runs/{rid}/config          training config as .yaml (task snapshot → version)
Images
    POST   /api/graph/runs/{rid}/images          multipart upload (PNG / JPEG / WebP)
    POST   /api/graph/runs/{rid}/import-samples  copy samples of a queue task into the card
    PATCH  /api/graph/images/{iid}
    DELETE /api/graph/images/{iid}
    GET    /api/graph/images/{iid}/file          original, or ?w=N thumbnail
Queue (read-only)
    GET    /api/graph/tasks?keys=a,b             training tasks + the config keys asked for
    GET    /api/graph/tasks/{tid}/samples        sample files with step / prompt / seed
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import yaml
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse, Response

from ...domain.errors import NotFoundError, ValidationError
from ...services.graph import store, tasks as graph_tasks
from ..responses import _thumb_response

router = APIRouter()
logger = logging.getLogger(__name__)


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except store.GraphNotFound as exc:
        raise NotFoundError(str(exc), code="graph.not_found") from exc
    except store.GraphError as exc:
        raise ValidationError(str(exc), code="graph.invalid",
                              details={"reason": str(exc)}) from exc


# ── boards ──────────────────────────────────────────────────────────────────

@router.get("/api/graph/boards")
def list_boards() -> dict[str, Any]:
    return {"items": _call(store.list_boards)}


@router.post("/api/graph/boards")
def create_board(body: dict[str, Any]) -> dict[str, Any]:
    src = body.get("copy_params_from")
    return _call(store.create_board, body.get("name", ""),
                 copy_params_from=int(src) if src not in (None, "") else None,
                 empty=bool(body.get("empty")))


@router.get("/api/graph/boards/{bid}")
def get_board(bid: int) -> dict[str, Any]:
    return _call(store.get_board, bid)


@router.patch("/api/graph/boards/{bid}")
def update_board(bid: int, body: dict[str, Any]) -> dict[str, Any]:
    return _call(store.update_board, bid, body)


@router.delete("/api/graph/boards/{bid}")
def delete_board(bid: int) -> dict[str, Any]:
    _call(store.delete_board, bid)
    return {"ok": True}


@router.post("/api/graph/boards/{bid}/remap")
def remap(bid: int, body: dict[str, Any]) -> dict[str, Any]:
    mapping = body.get("mapping")
    if not isinstance(mapping, list):
        raise ValidationError("mapping must be a list")
    return {"changed": _call(store.remap_values, bid, str(body.get("param_id", "")), mapping)}


# ── runs ────────────────────────────────────────────────────────────────────

@router.post("/api/graph/boards/{bid}/runs")
def create_run(bid: int, body: dict[str, Any]) -> dict[str, Any]:
    return _call(store.create_run, bid, body)


@router.patch("/api/graph/runs/{rid}")
def update_run(rid: int, body: dict[str, Any]) -> dict[str, Any]:
    return _call(store.update_run, rid, body)


@router.post("/api/graph/runs/{rid}/duplicate")
def duplicate_run(rid: int, body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _call(store.duplicate_run, rid, (body or {}).get("overrides"))


@router.delete("/api/graph/runs/{rid}")
def delete_run(rid: int) -> dict[str, Any]:
    _call(store.delete_run, rid)
    return {"ok": True}


@router.get("/api/graph/runs/{rid}/config")
def run_config(rid: int) -> Response:
    """The config the card's training actually used.

    Prefers the linked task's frozen snapshot (exactly what ran); falls back
    to the linked version's current config. Cards without either get a 404 —
    the UI then offers a fragment built from the card's own parameters.
    """
    run = _call(store.get_run, rid)
    text: Optional[str] = None
    if run.get("task_id"):
        row = graph_tasks.task_row(int(run["task_id"]))
        if row:
            cfg = graph_tasks.task_config(int(run["task_id"]), row["config_path"])
            if cfg:
                text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    if text is None and run.get("project_id") and run.get("version_id"):
        from ...services import version_config
        from ...services.projects import projects, versions
        from ... import db

        with db.connection_for() as conn:
            p = projects.get_project(conn, int(run["project_id"]))
            v = versions.get_version(conn, int(run["version_id"]))
        if p and v and version_config.has_version_config(p, v):
            try:
                cfg = version_config.read_version_config(p, v)
                text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
            except version_config.VersionConfigError:
                text = None
    if text is None:
        raise NotFoundError("This card has no linked training config", code="graph.no_config")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (run.get("name") or f"run_{rid}"))
    return Response(
        content=text, media_type="application/x-yaml",
        headers={"Content-Disposition": f'attachment; filename="{safe or "config"}.yaml"'},
    )


# ── images ──────────────────────────────────────────────────────────────────

@router.post("/api/graph/runs/{rid}/images")
async def upload_images(
    rid: int,
    files: list[UploadFile] = File(...),
    meta: str = Form(""),
) -> dict[str, Any]:
    """``meta`` is an optional JSON object applied to every uploaded file
    (e.g. the step the user picked before dropping)."""
    try:
        common = json.loads(meta) if meta else {}
    except ValueError:
        common = {}
    if not isinstance(common, dict):
        common = {}
    added: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for f in files:
        data = await f.read()
        try:
            added.append(_call(store.add_image, rid, data, {**common, "source": f.filename or ""}))
        except ValidationError as exc:
            errors.append({"name": f.filename or "", "reason": exc.message})
    if not added and errors:
        raise ValidationError(errors[0]["reason"], code="graph.invalid", details={"errors": errors})
    return {"items": added, "errors": errors}


@router.post("/api/graph/runs/{rid}/import-samples")
def import_samples(rid: int, body: dict[str, Any]) -> dict[str, Any]:
    task_id = int(body.get("task_id") or 0)
    names = [str(n) for n in (body.get("filenames") or [])]
    if not task_id or not names:
        raise ValidationError("Pick a task and at least one sample")
    row = graph_tasks.task_row(task_id)
    if not row:
        raise NotFoundError("Task not found", code="graph.task_not_found")
    cfg = graph_tasks.task_config(task_id, row["config_path"]) or {}
    info = store.describe_samples(names, cfg, graph_tasks.monitor_summary(task_id))
    existing = {i["source"] for i in _call(store.get_run, rid)["images"]}
    added: list[dict[str, Any]] = []
    skipped = 0
    for n in names:
        source = f"task:{task_id}/{n}"
        if source in existing:
            skipped += 1
            continue
        p = graph_tasks.sample_file(task_id, n)
        if p is None:
            skipped += 1
            continue
        meta = dict(info.get(n) or {})
        epoch = meta.pop("epoch", None)
        if epoch is not None:
            meta["comment"] = f"epoch {epoch}"
        meta["source"] = source
        added.append(_call(store.add_image, rid, p.read_bytes(), meta))
    return {"items": added, "skipped": skipped}


@router.patch("/api/graph/images/{iid}")
def update_image(iid: int, body: dict[str, Any]) -> dict[str, Any]:
    return _call(store.update_image, iid, body)


@router.delete("/api/graph/images/{iid}")
def delete_image(iid: int) -> dict[str, Any]:
    _call(store.delete_image, iid)
    return {"ok": True}


@router.get("/api/graph/images/{iid}/file")
def image_file(iid: int, w: Optional[int] = None) -> FileResponse:
    p = _call(store.image_path, iid)
    # Stored files are write-once (uuid names), so the URL is a stable key.
    return _thumb_response(p, w if (w and w > 0) else 0, immutable=True)


# ── queue bridge ────────────────────────────────────────────────────────────

@router.get("/api/graph/tasks")
def list_tasks(keys: str = "") -> dict[str, Any]:
    wanted = [k for k in (s.strip() for s in keys.split(",")) if k][:200]
    return {"items": graph_tasks.list_train_tasks(wanted)}


@router.get("/api/graph/tasks/{tid}/samples")
def task_samples(tid: int) -> dict[str, Any]:
    from .samples import list_task_samples

    listing = list_task_samples(tid)
    row = graph_tasks.task_row(tid)
    cfg = (graph_tasks.task_config(tid, row["config_path"]) if row else None) or {}
    listed = [i for i in listing.get("items", []) if store.is_sample_name(i["filename"])]
    info = store.describe_samples([i["filename"] for i in listed], cfg, graph_tasks.monitor_summary(tid))
    items = [{**i, **(info.get(i["filename"]) or {})} for i in listed]
    return {"items": items, "total": len(items)}
