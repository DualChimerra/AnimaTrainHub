"""tag + captions + reg + version_config + enqueue training + version thumb
(extracted from server.py in PR-6.5 commit 5).

23 routes:

  tagging (1)
    POST /api/projects/{pid}/versions/{vid}/tag

  captions (8)
    GET /captions    PUT/GET /captions/{folder}/{filename}
    POST /captions/snapshot  GET /captions/snapshots
    POST /captions/snapshots/{sid}/restore  DELETE /captions/snapshots/{sid}
    POST /captions/commit  POST /captions/batch

  reg (5)
    GET /reg/preview-tags   GET /reg   POST /reg/build
    GET /reg/caption   DELETE /reg

  reg_ai (3)
    POST /reg/generate-prior   GET /reg/generate-prior/latest
    GET /reg/generate-prior/{task_id}

  version_config (4)
    GET /config  PUT /config  POST /config/from_preset  POST /config/save_as_preset

  training launch (1)
    POST /api/projects/{pid}/versions/{vid}/queue

  version thumb (1)
    GET /api/projects/{pid}/versions/{vid}/thumb
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...deps import _resolve_model_paths
from ...errors import _safe_join_or_400
from ..logs import read_task_log
from ...responses import _thumb_response
from ...schemas.queue import ScheduleTrainingRequest
from ...schemas.training import (
    BatchOp,
    CaptionEdit,
    CommitRequest,
    FromPresetRequest,
    RegAiRequest,
    RegBuildRequest,
    RegDeleteFilesRequest,
    RegRenameFolderRequest,
    SaveAsPresetRequest,
)
from ._shared import (
    _project_and_version_or_404,
    _publish_job_state,
    _publish_version_state,
    _reg_dir,
    _version_dir_or_404,
    _version_train_dir_or_404,
)
from .... import db
from ....domain.errors import (
    ConflictError,
    InvalidPathError,
    NotFoundError,
    ValidationError,
)
from ....services.presets import io as presets_io
from ....services.projects import jobs as project_jobs, projects, versions
from ....services.dataset import scan as datasets
from ....domain import RegAiConfig
from ....infrastructure.event_bus import bus
from ....paths import STUDIO_DATA, safe_join
from ....services import model_downloader, task_snapshot, version_config
from ....services import presets as preset_flow
from ....services.tagging import caption_snapshot
from ....services.reg import builder as reg_builder, dedup as reg_dedup
from ....services.dataset import tagedit
from ....services.dataset import curation as dataset_curation

router = APIRouter()

# Valid name when tagging scope picks a single train folder (Kohya style, blocks path
# traversal); same rule as dataset.curation._FOLDER_PATTERN.
_TAG_SCOPE_FOLDER_RE = re.compile(r"^([0-9]+_)?[A-Za-z][A-Za-z0-9_-]*$")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Captions read/write + snapshots + commit/batch
# ---------------------------------------------------------------------------


@router.get("/api/projects/{pid}/versions/{vid}/captions")
def list_captions_endpoint(
    pid: int, vid: int, folder: Optional[str] = None, full: bool = False,
) -> dict[str, Any]:
    _, _, train = _version_train_dir_or_404(pid, vid)
    if folder is None:
        return {"folder": None, "items": tagedit.list_all_captions(train, full=full)}
    _safe_join_or_400(train, folder)
    return {
        "folder": folder,
        "items": tagedit.list_captions_in_folder(train, folder, full=full),
    }


@router.get("/api/projects/{pid}/versions/{vid}/captions/{folder}/{filename}")
def get_caption_endpoint(
    pid: int, vid: int, folder: str, filename: str,
) -> dict[str, Any]:
    _, _, train = _version_train_dir_or_404(pid, vid)
    _safe_join_or_400(train, folder, filename)
    try:
        return tagedit.read_one(train, folder, filename)
    except FileNotFoundError as exc:
        raise NotFoundError(
            "Image not found",
            code="image.not_found",
            details={"folder": folder, "filename": filename},
        ) from exc


@router.put("/api/projects/{pid}/versions/{vid}/captions/{folder}/{filename}")
def put_caption_endpoint(
    pid: int, vid: int, folder: str, filename: str, body: CaptionEdit,
) -> dict[str, Any]:
    _, _, train = _version_train_dir_or_404(pid, vid)
    _safe_join_or_400(train, folder, filename)
    try:
        return tagedit.write_one(train, folder, filename, body.tags)
    except FileNotFoundError as exc:
        raise NotFoundError(
            "Image not found",
            code="image.not_found",
            details={"folder": folder, "filename": filename},
        ) from exc


@router.post("/api/projects/{pid}/versions/{vid}/captions/snapshot")
def create_caption_snapshot(pid: int, vid: int) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    return caption_snapshot.create_snapshot(vdir)


@router.get("/api/projects/{pid}/versions/{vid}/captions/snapshots")
def list_caption_snapshots(pid: int, vid: int) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    return {"items": caption_snapshot.list_snapshots(vdir)}


@router.post("/api/projects/{pid}/versions/{vid}/captions/snapshots/{sid}/restore")
def restore_caption_snapshot(pid: int, vid: int, sid: str) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    return caption_snapshot.restore_snapshot(vdir, sid)


@router.delete("/api/projects/{pid}/versions/{vid}/captions/snapshots/{sid}")
def delete_caption_snapshot(pid: int, vid: int, sid: str) -> dict[str, Any]:
    _, _, vdir = _version_dir_or_404(pid, vid)
    caption_snapshot.delete_snapshot(vdir, sid)
    return {"deleted": sid}


@router.post("/api/projects/{pid}/versions/{vid}/captions/commit")
def commit_captions(pid: int, vid: int, body: CommitRequest) -> dict[str, Any]:
    """Write multiple captions in one shot; auto-creates a snapshot restore point before writing."""
    _, _, vdir = _version_dir_or_404(pid, vid)
    train = vdir / "train"
    snap = caption_snapshot.create_snapshot(vdir)
    written = 0
    skipped: list[str] = []
    for it in body.items:
        try:
            img = safe_join(train, it.folder, it.name)
        except ValueError:
            skipped.append(f"{it.folder}/{it.name}")
            continue
        if not img.exists():
            skipped.append(f"{it.folder}/{it.name}")
            continue
        tagedit.write_tags(img, it.tags)
        written += 1
    return {"snapshot": snap, "written": written, "skipped": skipped}


@router.post("/api/projects/{pid}/versions/{vid}/captions/batch")
def batch_caption_endpoint(
    pid: int, vid: int, body: BatchOp,
) -> dict[str, Any]:
    _, _, train = _version_train_dir_or_404(pid, vid)
    op = body.op
    scope = body.scope
    if op == "add":
        n = tagedit.add_tags(
            scope, train, body.tags or [],
            position="front" if body.position == "front" else "back",
        )
        return {"op": op, "affected": n}
    if op == "remove":
        return {"op": op, "affected": tagedit.remove_tags(scope, train, body.tags or [])}
    if op == "replace":
        if not body.old or not body.new:
            raise ValidationError(
                "Replace requires both the old and new tag",
                code="tag.replace_args_required", http_status=400,
            )
        return {"op": op, "affected": tagedit.replace_tag(scope, train, body.old, body.new)}
    if op == "dedupe":
        return {"op": op, "affected": tagedit.dedupe(scope, train)}
    if op == "stats":
        return {"op": op, "items": tagedit.stats(scope, train, top=max(1, body.top))}
    raise ValidationError(
        f'Unknown operation: "{op}"',
        code="tag.op_invalid", details={"op": op}, http_status=400,
    )


# ---------------------------------------------------------------------------
# /api/projects/{pid}/versions/{vid}/reg  (PP5)
# ---------------------------------------------------------------------------


@router.get("/api/projects/{pid}/versions/{vid}/reg/preview-tags")
def reg_preview_tags(pid: int, vid: int, top: int = 20) -> dict[str, Any]:
    """Return the top-N tag frequency for train (doesn't actually build reg). Used by the UI's "exclude tag" checkboxes."""
    _, _, vdir = _version_dir_or_404(pid, vid)
    train = vdir / "train"
    items = reg_builder.preview_train_tag_distribution(train, top=max(1, top))
    return {"items": [{"tag": t, "count": c} for t, c in items]}


@router.get("/api/projects/{pid}/versions/{vid}/reg")
def get_reg_status(pid: int, vid: int) -> dict[str, Any]:
    """Return the reg set's status (meta + image count + filename list)."""
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    if not rdir.exists():
        return {"exists": False, "meta": None, "image_count": 0, "files": []}
    images: list[str] = []
    for f in sorted(rdir.rglob("*")):
        if f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS:
            try:
                rel = f.relative_to(rdir).as_posix()
            except ValueError:
                continue
            images.append(rel)
    meta = reg_builder.read_meta(rdir)
    meta_dict = None
    if meta is not None:
        meta_dict = asdict(meta)
    return {
        "exists": bool(images) or meta is not None,
        "meta": meta_dict,
        "image_count": len(images),
        "files": images,
    }


# A3 -- this round of reg auto-tag only exposes wd14 / cltagger in the UI. The underlying
# VALID_TAGGER_NAMES includes LLM / JoyCaption, but they're slow/expensive at reg's volume
# (>train), left for a separate PR; the 422 check is a safety net against a contributor
# passing them by mistake.
_REG_TAGGER_ALLOWED = {"wd14", "cltagger"}


@router.post("/api/projects/{pid}/versions/{vid}/reg/build")
def start_reg_build(pid: int, vid: int, body: RegBuildRequest) -> dict[str, Any]:
    if body.api_source not in {"gelbooru", "danbooru"}:
        raise ValidationError(
            "Image source must be Gelbooru or Danbooru",
            code="reg.api_source_invalid", http_status=400,
        )
    if body.postprocess_method not in {"smart", "stretch", "crop"}:
        raise ValidationError(
            "Postprocess method must be smart, stretch, or crop",
            code="reg.postprocess_method_invalid", http_status=400,
        )
    if not (0.05 <= body.postprocess_max_crop_ratio <= 0.5):
        raise ValidationError(
            "Max crop ratio must be between 0.05 and 0.5",
            code="reg.crop_ratio_out_of_range", http_status=400,
        )
    if body.aspect_ratio_filter_enabled and not (
        0.0 < body.min_aspect_ratio < body.max_aspect_ratio
    ):
        raise ValidationError(
            "Min aspect ratio must be less than max aspect ratio, "
            "and both must be greater than 0",
            code="reg.aspect_ratio_invalid", http_status=400,
        )
    if body.auto_tag_kind not in _REG_TAGGER_ALLOWED:
        _allowed = sorted(_REG_TAGGER_ALLOWED)
        raise ValidationError(
            f"Auto-tag mode must be one of: {_allowed}",
            code="reg.auto_tag_kind_invalid", details={"allowed": _allowed},
            http_status=422,
        )
    # B1 -- validate build_mode + target_count
    if body.build_mode not in {"mirror", "flat"}:
        raise ValidationError(
            "Build mode must be mirror or flat",
            code="reg.build_mode_invalid", http_status=422,
        )
    if body.target_count is not None and body.target_count <= 0:
        raise ValidationError(
            "Target count must be greater than 0 or empty",
            code="reg.target_count_invalid", http_status=422,
        )
    _, v, vdir = _version_dir_or_404(pid, vid)
    train = vdir / "train"
    has_image = train.exists() and any(
        f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS
        for f in train.rglob("*")
    )
    if not has_image:
        raise ValidationError(
            "Add images to the training set first",
            code="train.no_images", http_status=400,
        )

    with db.connection_for() as conn:
        job = project_jobs.create_job(
            conn,
            project_id=pid,
            version_id=vid,
            kind="reg_build",
            params={
                "version_id": vid,
                "excluded_tags": list(body.excluded_tags),
                "auto_tag": bool(body.auto_tag),
                "auto_tag_kind": body.auto_tag_kind,
                "auto_dedup": bool(body.auto_dedup),
                "build_mode": body.build_mode,
                "target_count": body.target_count,
                "api_source": body.api_source,
                "incremental": bool(body.incremental),
                "skip_similar": bool(body.skip_similar),
                "aspect_ratio_filter_enabled": bool(body.aspect_ratio_filter_enabled),
                "min_aspect_ratio": float(body.min_aspect_ratio),
                "max_aspect_ratio": float(body.max_aspect_ratio),
                "postprocess_method": body.postprocess_method,
                "postprocess_max_crop_ratio": float(body.postprocess_max_crop_ratio),
            },
        )
    _publish_job_state(job)
    return job


@router.get("/api/projects/{pid}/versions/{vid}/reg/caption")
def get_reg_caption(pid: int, vid: int, path: str) -> dict[str, Any]:
    """Read the caption for a single image in the reg set. `path` is relative to reg/ (may include subfolders)."""
    if not path:
        raise InvalidPathError("Invalid path", code="path.invalid")
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    # path may contain `/` subdirectories; split on the separator and hand the parts to
    # safe_join for component validation + containment
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    img = _safe_join_or_400(rdir, *parts)
    if not img.exists() or img.suffix.lower() not in datasets.IMAGE_EXTS:
        raise NotFoundError(
            "Image not found", code="image.not_found", details={"path": path},
        )
    return {"path": path, "tags": tagedit.read_tags(img)}


@router.post("/api/projects/{pid}/versions/{vid}/reg/generate-prior")
def reg_generate_prior(pid: int, vid: int, body: RegAiRequest) -> dict[str, Any]:
    """Start a prior-generation task -- the base model generates a reference image for each
    train image's tags, inverted from the caption.

    The model family follows this version's training config (prior generation is a
    version-level operation, not a per-request choice): defaults to anima if there's no
    config or it isn't declared.
    """
    project, ver = _project_and_version_or_404(pid, vid)
    family = "anima"
    if version_config.has_version_config(project, ver):
        try:
            vc = version_config.read_version_config(project, ver)
            family = str(vc.get("model_family") or "anima")
        except version_config.VersionConfigError:
            pass  # a broken config doesn't block prior generation; falls back to anima
    model_paths = _resolve_model_paths(body.base_model, family=family)
    _, _, vdir = _version_dir_or_404(pid, vid)
    train = vdir / "train"
    has_image = train.exists() and any(
        f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS
        for f in train.rglob("*")
    )
    if not has_image:
        raise ValidationError(
            "Add images to the training set first",
            code="train.no_images", http_status=400,
        )

    # "Most recent click wins": clicking generate again abandons all older reg_ai tasks
    # (pending + running) for this same version and only runs the newly created one.
    # Otherwise, when the UI badge gets stuck (SSE dies, see anima-phase-cursor-sse-desync),
    # repeated clicks pile up a stack of reg_ai tasks; the supervisor runs them serially
    # (created_at ASC), so the newest click lands at the back of the queue -> looks like "the
    # new one is queued with an empty log while the old task is still generating images".
    # Cancel the old ones before creating the new one:
    #   - pending -> supervisor.cancel marks it canceled directly
    #   - running -> async SIGTERM; once the slot exits, the supervisor naturally picks up the
    #     new task
    try:
        sup: Optional[Any] = _supervisor()
    except HTTPException:
        sup = None  # no supervisor in tests / during startup: fall back to db for pending below
    with db.connection_for() as conn:
        stale = [
            t for t in db.list_tasks(conn)
            if t.get("task_type") == "reg_ai"
            and t.get("version_id") == vid
            and t.get("status") in ("pending", "running")
        ]
    for t in stale:
        tid = int(t["id"])
        if sup is not None:
            sup.cancel(tid)
        elif t.get("status") == "pending":
            with db.connection_for() as conn:
                db.update_task(conn, tid, status="canceled", finished_at=time.time())
            bus.publish({"type": "task_state_changed", "task_id": tid, "status": "canceled"})

    rdir = _reg_dir(vdir)
    rdir.mkdir(parents=True, exist_ok=True)

    from ....domain.common import FAMILY_SAMPLING
    from ....services.runtime.xformers import detect_attention_backend
    sampling = FAMILY_SAMPLING[family]
    cfg = RegAiConfig(
        **model_paths,
        model_family=family,
        train_dir=str(train),
        reg_dir=str(rdir),
        excluded_tags=list(body.excluded_tags),
        negative_prompt=body.negative_prompt,
        width=body.width,
        height=body.height,
        steps=body.steps,
        cfg_scale=body.cfg_scale,
        sampler_name=body.sampler_name or sampling["samplers"][0],
        scheduler=body.scheduler or sampling["schedulers"][0],
        seed=body.seed,
        incremental=body.incremental,
        repeat=body.repeat,
        mixed_precision=body.mixed_precision,
        attention_backend=detect_attention_backend(),
    )

    cfg_dir = STUDIO_DATA / "reg_ai_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name=f"reg-prior p{pid}v{vid}", config_name="reg_ai", priority=0,
        )
        db.update_task(
            conn, task_id, task_type="reg_ai", project_id=pid, version_id=vid,
        )

    cfg_path = cfg_dir / f"reg_ai_{task_id}.json"
    cfg_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    with db.connection_for() as conn:
        db.update_task(conn, task_id, config_path=str(cfg_path))
        task = db.get_task(conn, task_id)

    bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "pending"})
    return task or {"id": task_id}


@router.get("/api/projects/{pid}/versions/{vid}/reg/generate-prior/latest")
def get_latest_reg_prior_task(pid: int, vid: int) -> dict[str, Any]:
    """For page hydration: returns this version's most recent AI-prior task + its full log."""
    with db.connection_for() as conn:
        row = conn.execute(
            """
            SELECT * FROM tasks
            WHERE project_id = ? AND version_id = ? AND task_type = 'reg_ai'
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (pid, vid),
        ).fetchone()
    task = dict(row) if row else None
    return {
        "task": task,
        "log": read_task_log(int(task["id"])) if task else "",
    }


@router.get("/api/projects/{pid}/versions/{vid}/reg/generate-prior/{task_id}")
def get_reg_prior_task(pid: int, vid: int, task_id: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task or task.get("task_type") != "reg_ai":
        raise NotFoundError(
            "Task not found", code="task.not_found", details={"id": task_id},
        )
    return task


@router.post("/api/projects/{pid}/versions/{vid}/reg/folder")
def reg_rename_folder(
    pid: int, vid: int, body: RegRenameFolderRequest,
) -> dict[str, Any]:
    """Rename a subfolder under reg/ (e.g. changing the Kohya repeat prefix 2_data -> 1_data).

    Mirrors the train folder rename in Step 1 (Curation): lets the user tweak the reg repeat
    count at the "reg images generated" step without manually renaming on the Colab
    filesystem.
    """
    with db.connection_for() as conn:
        try:
            p = dataset_curation.rename_reg_folder(
                conn, pid, vid, body.name, body.new_name,
            )
        except dataset_curation.CurationError as exc:
            raise HTTPException(400, str(exc)) from exc
    return {"path": str(p)}


@router.delete("/api/projects/{pid}/versions/{vid}/reg")
def delete_reg(pid: int, vid: int) -> dict[str, Any]:
    """Clear the contents of reg/ (including meta.json + all subfolders), keeping the empty
    directory itself.

    `versions.create_version` always creates an empty reg/; "exists" is defined as having
    meta or images.
    """
    import shutil as _shutil
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    has_content = rdir.exists() and (
        (rdir / "meta.json").exists()
        or any(
            f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS
            for f in rdir.rglob("*")
        )
    )
    if not has_content:
        return {"deleted": False, "reason": "reg empty"}
    try:
        for child in rdir.iterdir():
            if child.is_dir():
                _shutil.rmtree(child)
            else:
                child.unlink()
    except OSError as exc:
        raise HTTPException(500, f"Delete failed: {exc}") from exc
    return {"deleted": True}


@router.post("/api/projects/{pid}/versions/{vid}/reg/delete-files")
def delete_reg_files(
    pid: int, vid: int, body: RegDeleteFilesRequest,
) -> dict[str, Any]:
    """Bulk-delete images from the reg set by relative path (including same-name .txt
    captions), update meta.actual_count, and append the deleted booru IDs (filename stems)
    to reg/.deleted_ids.json.

    When incrementally topping up, the builder reads this file and adds the IDs to the
    search exclude list, so the same booru post doesn't get pulled back in.

    body.relative_paths is a list of paths relative to reg/, which may span subfolders;
    an out-of-bounds path raises 400 via _safe_join_or_400.
    """
    if not body.relative_paths:
        raise ValidationError(
            "No files selected", code="reg.paths_empty", http_status=400,
        )
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    if not rdir.exists():
        raise NotFoundError(
            "Regularization set not found", code="reg.not_found",
        )

    # Endpoint entry point: every path goes through _safe_join_or_400 to block traversal;
    # valid ones are converted back to a path relative to rdir and fed to
    # reg_dedup.purge_paths. Paths coming from the worker via dedup.scan_for_dedup are
    # guaranteed valid, so the dedup module itself doesn't re-check for traversal.
    validated_rels: list[str] = []
    for rel in body.relative_paths:
        if not rel:
            continue
        parts = [p for p in rel.replace("\\", "/").split("/") if p]
        if not parts:
            continue
        target = _safe_join_or_400(rdir, *parts)
        validated_rels.append(target.relative_to(rdir).as_posix())

    return reg_dedup.purge_paths(rdir, validated_rels)


@router.post("/api/projects/{pid}/versions/{vid}/reg/dedup-purge")
def dedup_purge_reg(pid: int, vid: int) -> dict[str, Any]:
    """A4 -- scan the reg set with preprocess dedup's default params, automatically deleting
    the "recommended for deletion" items in each group (group[0] is kept, the rest are
    deleted + written to .deleted_ids.json + meta decremented).

    Manual user entry point; the worker runs the same reg_dedup module automatically after
    build when auto_dedup=True. reg's quality bar is lower than train's, so no review panel
    is shown.

    Returns synchronously; slow for large image counts (O(n^2)).
    """
    _, _, vdir = _version_dir_or_404(pid, vid)
    rdir = _reg_dir(vdir)
    if not rdir.exists():
        raise NotFoundError(
            "Regularization set not found", code="reg.not_found",
        )

    to_delete = reg_dedup.scan_for_dedup(rdir)
    scanned = sum(
        1 for f in rdir.rglob("*")
        if f.is_file() and f.suffix.lower() in datasets.IMAGE_EXTS
    )
    if not to_delete:
        return {"scanned": scanned, "groups": 0, "deleted": [], "count": 0}

    result = reg_dedup.purge_paths(rdir, to_delete)
    return {
        "scanned": scanned,
        # "groups" is approximated by the count of items pending deletion (each group has
        # >= 1 image deleted); scan_for_dedup doesn't expose the total group count, and the
        # user only cares how many were deleted anyway.
        "groups": len(to_delete),
        "deleted": result["deleted"],
        "count": result["count"],
    }


# ---------------------------------------------------------------------------
# /api/projects/{pid}/versions/{vid}/config  (PP6.2 training config -- version-private)
# ---------------------------------------------------------------------------


@router.get("/api/projects/{pid}/versions/{vid}/config")
def get_version_config_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """Read the version's private config; returns has_config=false / config=null if it doesn't exist.

    Returns `project_specific_defaults` regardless of has_config -- the project-specific
    values the backend auto-injects when forking a preset (project paths + global model
    paths + reg detection results). The frontend's "+ new preset" button can be clicked even
    when the version already has a config (replacing the current preset), so this hint must
    always be returned, independent of has_config.
    """
    project, ver = _project_and_version_or_404(pid, vid)
    psf = sorted(version_config.PROJECT_SPECIFIC_FIELDS)

    def _psd(family: str) -> dict[str, Any]:
        return {
            **version_config.project_specific_overrides(project, ver),
            **model_downloader.default_paths_for_new_version(family=family),
        }

    if not version_config.has_version_config(project, ver):
        return {
            "has_config": False,
            "config": None,
            "project_specific_fields": psf,
            "project_specific_defaults": _psd("anima"),
        }
    try:
        cfg, dropped, defaulted = version_config.read_version_config_with_warnings(project, ver)
    except version_config.VersionConfigError as exc:
        raise ValidationError(
            f"Training configuration is invalid: {exc}",
            code="version.config_invalid", details={"reason": str(exc)},
            http_status=422,
        ) from exc
    # The path hint follows the version's declared family (a krea2 version shouldn't get
    # anima path defaults)
    psd = _psd(str(cfg.get("model_family") or "anima"))
    return {
        "has_config": True,
        "config": cfg,
        "project_specific_fields": psf,
        "project_specific_defaults": psd,
        "dropped_fields": dropped,
        "defaulted_fields": defaulted,
    }


@router.get("/api/projects/{pid}/versions/{vid}/train-estimate")
def get_train_estimate_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """Pre-training estimate: whether there's enough VRAM + how long it'll take based on
    historically measured speed.

    The VRAM side reuses training/block_swap_preflight.evaluate -- the same arithmetic as the
    guardrail that actually blocks training, so this panel can never disagree with it. The
    speed side takes the most recently measured it/s recorded by this project's training
    monitor; with no history it returns None and the frontend shows nothing (a made-up number
    is worse than a blank).
    """
    project, ver, _train_dir = _version_train_dir_or_404(pid, vid)
    cfg: dict[str, Any] = {}
    if version_config.has_version_config(project, ver):
        try:
            cfg, _dropped, _defaulted = version_config.read_version_config_with_warnings(project, ver)
        except version_config.VersionConfigError:
            cfg = {}

    from studio.infrastructure.db import connection_for
    from studio.services import train_estimate

    with connection_for() as conn:
        return train_estimate.estimate(conn, pid, cfg)


@router.get("/api/projects/{pid}/versions/{vid}/bucket-distribution")
def get_bucket_distribution_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """Preview of the training set's ARB bucket distribution (computed with the real
    BucketManager, matching actual training bucket-for-bucket).

    Buckets each image using the version config's resolution list + aspect_ratio_limit +
    per-folder px/repeat, returning the buckets and effective sample count per resolution
    tier. Uses schema defaults with no config; an empty dataset returns empty groups.
    """
    project, ver, train_dir = _version_train_dir_or_404(pid, vid)
    resolutions: list[int] = [1024]
    ar_limit = 2.0
    prefer_json = True
    cfg: dict[str, Any] = {}
    if version_config.has_version_config(project, ver):
        try:
            cfg, _dropped, _defaulted = version_config.read_version_config_with_warnings(project, ver)
            r = cfg.get("resolution")
            if isinstance(r, list) and r:
                resolutions = [int(x) for x in r]
            elif isinstance(r, (int, float)):
                resolutions = [int(r)]
            ar_limit = float(cfg.get("aspect_ratio_limit", 2.0) or 2.0)
            prefer_json = bool(cfg.get("prefer_json", True))
        except version_config.VersionConfigError:
            cfg = {}
    groups = versions.compute_bucket_histogram(train_dir, resolutions, ar_limit, prefer_json)

    # NaViT packing mode: includes a real packing-simulation estimate of pack count (the
    # frontend's step-count formula can't just divide by batch_size in this mode -- batching
    # is instead driven by the token budget, see NavitPackBatchSampler).
    navit: dict[str, Any] | None = None
    if cfg.get("navit_packing"):
        dirs: list[Path] = [train_dir]
        reg_dir = str(cfg.get("reg_data_dir") or "")
        if reg_dir and Path(reg_dir).exists():
            dirs.append(Path(reg_dir))
        navit = versions.compute_navit_pack_estimate(
            dirs, resolutions, ar_limit, prefer_json,
            native_resolution=bool(cfg.get("navit_native_resolution", False)),
            token_budget=int(cfg.get("navit_token_budget", 16384) or 16384),
            max_images_per_pack=int(cfg.get("navit_max_images_per_pack", 0) or 0),
            strategy=str(cfg.get("navit_pack_strategy", "next_fit") or "next_fit"),
            ffd_window=int(cfg.get("navit_pack_ffd_window", 256) or 0),
            drop_last=bool(cfg.get("navit_drop_last", False)),
            over_budget=str(cfg.get("navit_native_over_budget", "downscale") or "downscale"),
            seed=int(cfg.get("seed", 42) or 42),
        )
    # Authoritative source for the effective sample count. The frontend used to scan folders
    # and count images itself, but the trainer only accepts images **with a caption**
    # (compute_bucket_histogram mirrors this rule) -- if the dataset has even a few untagged
    # images, the step count shown before starting would be higher than actual. Computing it
    # twice in two places is bound to disagree, so the backend computes it here once and the
    # frontend just copies it.
    def _effective(directory: Path | None) -> int:
        if directory is None or not directory.exists():
            return 0
        return sum(
            int(b.get("count") or 0)
            for group in versions.compute_bucket_histogram(
                directory, resolutions, ar_limit, prefer_json
            )
            for b in group.get("buckets", [])
        )

    reg_dir_str = str(cfg.get("reg_data_dir") or "")
    reg_path = Path(reg_dir_str) if reg_dir_str else None
    train_samples = _effective(train_dir)
    reg_samples = _effective(reg_path)

    return {
        "resolutions": resolutions, "aspect_ratio_limit": ar_limit,
        "groups": groups, "navit": navit,
        "effective_samples": train_samples + reg_samples,
        "train_samples": train_samples,
        "reg_samples": reg_samples,
    }


@router.put("/api/projects/{pid}/versions/{vid}/config")
def put_version_config_endpoint(
    pid: int, vid: int, body: dict[str, Any],
) -> dict[str, Any]:
    """Write the version's private config directly (full replace).

    PP10.4: project-specific fields (data_dir / output_dir / output_name, etc.) are **not**
    force-overwritten. They're already pre-filled when forking a preset; the user can freely
    change them on the Train page (e.g. `resume_lora` to continue training, or a custom
    output_name). If it breaks, switch presets again to get back to defaults.
    """
    project, ver = _project_and_version_or_404(pid, vid)
    try:
        version_config.write_version_config(
            project, ver, body, force_project_overrides=False,
        )
        cfg = version_config.read_version_config(project, ver)
    except version_config.VersionConfigError as exc:
        raise ValidationError(
            f"Training configuration is invalid: {exc}",
            code="version.config_invalid", details={"reason": str(exc)},
            http_status=400,
        ) from exc
    return {"has_config": True, "config": cfg}


@router.post("/api/projects/{pid}/versions/{vid}/config/from_preset")
def fork_preset_for_version_endpoint(
    pid: int, vid: int, body: FromPresetRequest,
) -> dict[str, Any]:
    """Copy a global preset into the version's private config (applying project-specific fields)."""
    project, ver = _project_and_version_or_404(pid, vid)
    try:
        cfg, dropped, defaulted = preset_flow.fork_preset_for_version_with_warnings(
            body.name, project, ver
        )
    except version_config.VersionConfigError as exc:
        raise ValidationError(
            f"Training configuration is invalid: {exc}",
            code="version.config_invalid", details={"reason": str(exc)},
            http_status=400,
        ) from exc
    # Sync versions.config_name = the source preset name (informational only)
    with db.connection_for() as conn:
        versions.update_version(conn, vid, config_name=body.name)
    return {
        "has_config": True,
        "config": cfg,
        "from_preset": body.name,
        "dropped_fields": dropped,
        "defaulted_fields": defaulted,
    }


@router.post("/api/projects/{pid}/versions/{vid}/config/save_as_preset")
def save_version_config_as_preset_endpoint(
    pid: int, vid: int, body: SaveAsPresetRequest,
) -> dict[str, Any]:
    """Version's private config -> global preset (strips project-specific fields)."""
    project, ver = _project_and_version_or_404(pid, vid)
    try:
        cfg = preset_flow.save_version_config_as_preset(
            project, ver, body.name, overwrite=body.overwrite,
        )
    except version_config.VersionConfigError as exc:
        raise ValidationError(
            f"Training configuration is invalid: {exc}",
            code="version.config_invalid", details={"reason": str(exc)},
            http_status=400,
        ) from exc
    return {"saved_preset": body.name, "config": cfg}


def _ensure_dop_trigger(
    project: dict[str, Any], ver: dict[str, Any], cfg: dict[str, Any],
) -> dict[str, Any]:
    """DOP needs a trigger word; make sure the version has one before queueing.

    The trigger lives on the version row and the write below forces it into the
    yaml, so a value typed straight into the yaml (or left over from an older
    config) would be wiped and the run would die at startup. Order of preference:
    the version's own value, the one already in the yaml, and finally the one the
    captions carry (``trigger_detect.confident_trigger``). The chosen value is
    saved on the version so the UI shows it. Nothing found → fail here with an
    actionable message instead of a traceback minutes into the run.
    """
    if not cfg.get("dop_enabled") or str(ver.get("trigger_word") or "").strip():
        return ver
    from ....services import trigger_detect

    word = str(cfg.get("trigger_word") or "").strip()
    if not word:
        vdir = versions.version_dir(project["id"], project["slug"], ver["label"])
        word = trigger_detect.confident_trigger(vdir / "train") or ""
    if not word:
        raise ValidationError(
            "DOP is enabled but this version has no trigger word. Enter the trigger "
            "word on the Train page (it must appear in the captions), or turn DOP off.",
            code="version.dop_trigger_missing", http_status=400,
        )
    with db.connection_for() as conn:
        updated = versions.update_version(conn, int(ver["id"]), trigger_word=word)
    logger.info("DOP: version %s trigger_word set to %r before queueing", ver["id"], word)
    return updated


@router.get("/api/projects/{pid}/versions/{vid}/trigger-detect")
def detect_trigger_endpoint(pid: int, vid: int) -> dict[str, Any]:
    """Suggest the trigger word from the captions already in train/."""
    from ....services import trigger_detect

    _project, ver, train_dir = _version_train_dir_or_404(pid, vid)
    return {**trigger_detect.detect(train_dir), "current": ver.get("trigger_word") or ""}


@router.post("/api/projects/{pid}/versions/{vid}/queue")
def enqueue_version_training(
    pid: int, vid: int, body: Optional[ScheduleTrainingRequest] = None,
) -> dict[str, Any]:
    """PP6.3 -- enqueue a version for training.

    Validation:
    - the version has training parameters configured (version_config exists)
    - this version has no active task (pending / running / scheduled)

    0.17 P-B: if body carries scheduled_at (unix seconds), the task is created as scheduled
    and the supervisor promotes it to pending once due; with no body / no such field, it's
    immediately pending (original behavior).
    """
    project, ver = _project_and_version_or_404(pid, vid)
    if not version_config.has_version_config(project, ver):
        raise ValidationError(
            "Training configuration is not set for this version",
            code="version.config_missing", http_status=400,
        )
    cfg_path = version_config.version_config_path(project, ver)
    # auto_sync_paths=ON: syncs the global model paths to disk at enqueue time -- the trainer
    # subprocess reads the yaml directly (without going through studio's read-side overlay),
    # so once global selected / selected_te changes, training must use the current value (the
    # Train page locks these fields and promises "auto - global settings"). When OFF, read
    # doesn't overlay, so the write-back is idempotent.
    #
    # reset_resume=False: this is a "read it out and write it back" operation, not a
    # new/fork -- force-resetting resume_lora / resume_state would silently wipe the resume
    # point the user just set the instant they click "start training" (no error), and
    # training would start from scratch.
    try:
        synced = version_config.read_version_config(project, ver)
        ver = _ensure_dop_trigger(project, ver, synced)
        version_config.write_version_config(
            project, ver, synced, force_project_overrides=True, reset_resume=False,
        )
    except version_config.VersionConfigError:
        pass  # a broken config is reported by downstream validation, not interrupted here
    scheduled_at = body.scheduled_at if body else None

    with db.connection_for() as conn:
        # Whether this version already has an active GPU task (R-5: after the ledger merge,
        # tasks also holds data jobs; a pending tagging job shouldn't block training enqueue --
        # only check the GPU task types)
        active = conn.execute(
            "SELECT id, status FROM tasks "
            "WHERE version_id = ? AND status IN ('pending', 'running', 'scheduled') "
            "AND COALESCE(task_type, 'train') IN ('train', 'reg_ai', 'generate') "
            "LIMIT 1",
            (vid,),
        ).fetchone()
        if active:
            raise ConflictError(
                "This version already has a running task; "
                "wait for it to finish or cancel it",
                code="version.has_active_task",
                details={"task_id": active["id"], "status": active["status"]},
            )

        # Create the task
        slug = project["slug"]
        label = ver["label"]
        task_name = f"{slug}_{label}"
        config_name = ver["config_name"] or f"proj_{pid}_{label}"  # informational
        status = "scheduled" if scheduled_at is not None else "pending"
        # ADR-0009 PR-1 C6: same as db.create_task -- stash the ContextVar trace_id
        from studio.infrastructure.logging import get_trace_id, new_trace_id
        req_tid = get_trace_id() or f"bg-{new_trace_id()}"
        cur = conn.execute(
            "INSERT INTO tasks(name, config_name, status, priority, created_at, "
            "project_id, version_id, config_path, request_trace_id, scheduled_at) "
            "VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?)",
            (task_name, config_name, status, time.time(), pid, vid,
             str(cfg_path), req_tid, scheduled_at),
        )
        tid = int(cur.lastrowid)
        # The task's model and other training parameters are frozen the moment it's
        # enqueued. Subsequently switching the global model or editing the version config
        # only affects new tasks.
        frozen_cfg = task_snapshot.freeze_config(tid, cfg_path)
        conn.execute(
            "UPDATE tasks SET config_path = ? WHERE id = ?",
            (str(frozen_cfg), tid),
        )
        conn.commit()
        # ADR-0007 PR-5: version.status is pushed to training by the supervisor in
        # _spawn_task; project has no stage, so we no longer advance it here.
        task = db.get_task(conn, tid)
    bus.publish({
        "type": "task_state_changed",
        "task_id": tid,
        "status": status,
    })
    return task or {}


# Version-level thumbnail: bucket = train | reg | samples (PP3 added train; reg/samples left for PP4-5)
@router.get("/api/projects/{pid}/versions/{vid}/thumb")
def version_thumb(
    pid: int,
    vid: int,
    bucket: str = "train",
    folder: str = "",
    name: str = "",
    size: int = 256,
) -> FileResponse:
    if bucket not in {"train", "reg", "samples", "validation"}:
        raise InvalidPathError("Invalid path", code="path.invalid")
    with db.connection_for() as conn:
        v = versions.get_version(conn, vid)
        p = projects.get_project(conn, pid)
    if not v or not p or v["project_id"] != pid:
        raise NotFoundError(
            "Version not found", code="version.not_found", details={"id": vid},
        )
    vdir = versions.version_dir(p["id"], p["slug"], v["label"]) / bucket
    if bucket in {"train", "reg", "validation"}:
        if not folder:
            raise InvalidPathError("Invalid path", code="path.invalid")
        f = _safe_join_or_400(vdir, folder, name)
    else:
        f = _safe_join_or_400(vdir, name)
    if not f.exists() or f.suffix.lower() not in datasets.IMAGE_EXTS:
        logger.info(
            "version thumb 404: pid=%s vid=%s bucket=%s folder=%s name=%s -> %s",
            pid, vid, bucket, folder, name, f,
        )
        raise NotFoundError("Image not found", code="image.not_found")
    return _thumb_response(f, size)
