"""Version data model + physical directories + fork training tree + activate.

A Version is a Pipeline "experiment unit": each version independently
maintains its own train/ reg/ output/. The label is user-chosen (semantic
names like baseline / high-lr), unique within a project, and immutable (it's
a path anchor).

Deletion: rmtree the version directory + DELETE the db row. Unrecoverable.
If the deleted version was active, it's automatically reassigned to "the
most recently created remaining version".
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from . import projects
from ...services.dataset.scan import IMAGE_EXTS

# ADR-0007 section 11.3-B: the versions state machine uses two orthogonal
# fields, status and phase. The old single `stage` field was removed in PR-5
# (PR-5 commit 2 dropped VALID_STAGES / advance_stage).


class VersionStatus:
    """Version runtime state machine (5 enum values, ADR-0007 section 11.3-B)."""

    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    VALUES: frozenset[str] = frozenset({
        PREPARING, TRAINING, COMPLETED, FAILED, CANCELED,
    })


class VersionPhase:
    """Version preparation cursor; only meaningful while status=preparing (ADR-0007 sections 11.3-B / 11.5-A).

    Order: curating -> preprocessing -> editing -> regularizing -> ready.
    preprocessing / regularizing can be skipped (SKIPPABLE); the rest are mandatory.

    ADR 0010 added the preprocessing phase (after curating): the user does
    fine-grained processing (upscale / crop / dedup) on the train set after
    picking images; skippable = accept the default upscale algorithm at
    training time.

    The old auto-tagging step (the old tagging phase) has been removed;
    captions now come from uploaded .txt files, manual entry (editing), or
    training without captions. Old tagging data was migrated to editing.
    """

    CURATING = "curating"
    PREPROCESSING = "preprocessing"
    EDITING = "editing"
    REGULARIZING = "regularizing"
    READY = "ready"

    ORDER: tuple[str, ...] = (
        CURATING, PREPROCESSING, EDITING, REGULARIZING, READY,
    )
    VALUES: frozenset[str] = frozenset(ORDER)
    SKIPPABLE: frozenset[str] = frozenset({PREPROCESSING, REGULARIZING})


def get_status(v: dict[str, Any]) -> str:
    """Read version.status; None / missing field falls back to preparing."""
    return str(v.get("status") or VersionStatus.PREPARING)


def get_phase(v: dict[str, Any]) -> str:
    """Read version.phase; None / missing field falls back to curating."""
    return str(v.get("phase") or VersionPhase.CURATING)


# ---------------------------------------------------------------------------
# ADR-0007 sections 11.3-C / 6.9: version.status derivation + consistency check
# ---------------------------------------------------------------------------


_TASK_TO_VERSION_STATUS: dict[str, str] = {
    "done":     VersionStatus.COMPLETED,
    "failed":   VersionStatus.FAILED,
    "canceled": VersionStatus.CANCELED,
}


def derive_status_from_tasks(
    conn: sqlite3.Connection, version_id: int
) -> str:
    """Derive version.status per ADR section 11.3-C:

    - has an active task (pending / running / paused / scheduled) -> training
    - no active task, look at the most recent terminal task -> completed / failed / canceled
    - never had a task -> preparing

    0.17 P-B: scheduled (a scheduled task not yet due) is treated the same as
    pending - the version is already claimed by that task (the enqueue
    endpoint would 409), and the status must reflect that.
    R-5: after the ledger merge, the tasks table also holds data jobs
    (tag/download/eval...), so the derivation only looks at GPU task types -
    otherwise a pending tagging job would bump the version to "training".
    """
    row = conn.execute(
        "SELECT 1 FROM tasks "
        "WHERE version_id = ? AND status IN ('pending', 'running', 'paused', 'scheduled') "
        "AND COALESCE(task_type, 'train') IN ('train', 'reg_ai', 'generate') "
        "LIMIT 1",
        (version_id,),
    ).fetchone()
    if row:
        return VersionStatus.TRAINING

    row = conn.execute(
        "SELECT status FROM tasks "
        "WHERE version_id = ? AND status IN ('done', 'failed', 'canceled') "
        "AND COALESCE(task_type, 'train') IN ('train', 'reg_ai', 'generate') "
        "ORDER BY created_at DESC LIMIT 1",
        (version_id,),
    ).fetchone()
    if row:
        return _TASK_TO_VERSION_STATUS.get(str(row[0]), VersionStatus.PREPARING)

    return VersionStatus.PREPARING


def reconcile_version_status(
    conn: sqlite3.Connection, version_id: int
) -> tuple[Optional[dict[str, Any]], bool]:
    """Read a version + correct any status mismatch; returns (version, was_corrected).

    Safety net for ADR section 6.9: during the dual-write transition period,
    when the supervisor occasionally misses a write, this function lets any
    read path self-heal.
    - computes derive_status_from_tasks
    - mismatch against the stored value -> logs a warning + UPDATE + returns
      the corrected version + True
    - match -> returns (version, False) directly
    - version doesn't exist -> (None, False)

    This function doesn't emit SSE (kept as a pure db operation); callers
    decide whether to publish based on was_corrected.
    """
    import logging
    logger = logging.getLogger(__name__)

    v = get_version(conn, version_id)
    if not v:
        return None, False

    derived = derive_status_from_tasks(conn, version_id)
    stored = get_status(v)
    if stored == derived:
        return v, False

    logger.warning(
        "version %d status mismatch: stored=%r derived=%r -> correcting",
        version_id, stored, derived,
    )
    update_version(conn, version_id, status=derived)
    return get_version(conn, version_id), True

# The label must be path-safe: letters / digits / underscore / hyphen / dot.
# An all-dots label ("." / "..") would make version_dir resolve outside
# versions/ (".." == the project root, so delete_version would rmtree the
# whole project) - this must be rejected.
_VALID_LABEL = re.compile(r"^(?!\.+$)[A-Za-z0-9_.-]+$")


def is_valid_label(label: str) -> bool:
    """Version label validation, reusable by external input sources (e.g. a bundle manifest)."""
    return bool(_VALID_LABEL.fullmatch(label))


from studio.domain.errors import DomainError


class VersionError(DomainError):
    """Version business error.

    PR-2 C3 added the DomainError base - the handler auto-translates it into the dual-write envelope.
    """
    default_code = "version.error"


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def version_dir(project_id: int, slug: str, label: str) -> Path:
    return projects.project_dir(project_id, slug) / "versions" / label


def _natural_key(s: str) -> list[Any]:
    """Natural-sort key: numeric runs in the string compare as ints, so a_5 < a_60.

    re.split(r'(\\d+)', 'a_60') -> ['a_', '60', '']
    converted to ['a_', 60, ''], compared element-wise against the same
    conversion of 'a_5' -> ['a_', 5, ''].
    """
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_lora_ckpts(vdir: Path) -> list[dict[str, Any]]:
    """Scan versions/{label}/output/*.safetensors, listing every LoRA checkpoint file.

    anima_train output naming convention (runtime/anima_train.py:2434, 2464):
      - {output_name}_step{N}.safetensors    (saved by step)
      - {output_name}_epoch{N}.safetensors   (saved by epoch)
      - {output_name}_final.safetensors      (training finished)

    Returns {kind, value, label, path, mtime} for each checkpoint:
      - kind: 'step' | 'epoch' | 'final' | 'other'
      - value: int (step/epoch number; 0 for final/other)
      - label: for display, "step 2476" / "epoch 5" / "final" / the filename
      - path: absolute path string
      - mtime: modification timestamp (frontend shows newest first)
    Sort order: final first -> step number descending -> epoch number
    descending -> everything else by natural-sort label ascending (so a_5 <
    a_60, avoiding lexical order putting a_60 before a_9, or mtime order
    scrambling the user's expectation).
    """
    output_dir = vdir / "output"
    if not output_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for f in output_dir.glob("*.safetensors"):
        if not f.is_file():
            continue
        name = f.stem  # strip .safetensors
        kind = "other"
        value = 0
        label = name
        # match *_step{N}
        m = re.search(r"_step(\d+)$", name)
        if m:
            kind = "step"
            value = int(m.group(1))
            label = f"step {value}"
        else:
            m = re.search(r"_epoch(\d+)$", name)
            if m:
                kind = "epoch"
                value = int(m.group(1))
                label = f"epoch {value}"
            elif name.endswith("_final"):
                kind = "final"
                label = "final"
        try:
            mtime = f.stat().st_mtime
        except OSError:
            mtime = 0.0
        items.append({
            "kind": kind, "value": value, "label": label,
            "path": str(f), "mtime": mtime,
        })

    # sort: final on top; step/epoch by value descending; other by natural-sort label ascending
    kind_order = {"final": 0, "step": 1, "epoch": 2, "other": 3}

    def _sort_key(x: dict[str, Any]) -> tuple[Any, ...]:
        ko = kind_order.get(x["kind"], 9)
        if x["kind"] in ("step", "epoch"):
            return (ko, -x["value"], [], -x["mtime"])
        # final / other: value is always 0, sort by natural-sort label ascending (mainly benefits "other")
        return (ko, 0, _natural_key(x["label"]), -x["mtime"])

    items.sort(key=_sort_key)
    return items


_STATE_FILE_RE = re.compile(r"training_state_(step|epoch)(\d+)\.pt$")


def list_state_ckpts(vdir: Path) -> list[dict[str, Any]]:
    """Scan a version's output/ for every resume-training state file.

    Scans two locations (ADR 0006 PR-1 path migration):
      - old path: ``output/training_state_step{N}.pt`` (pre-PR-1 leftovers)
      - new path: ``output/state/task_<TID>/training_state_step{N}.pt`` (PR-1+)

    Both granularities are checked (PR-1 also fixed a bug where the old scan missed epoch files):
      - step  ->  ``training_state_step{N}.pt``    label "step N"
      - epoch ->  ``training_state_epoch{N}.pt``   label "epoch N"

    Pause files (PR-2+'s ``pause_step_<N>.pt``) are **excluded** - the picker
    shouldn't expose a mid-pause state. The naming prefix filters them out naturally.

    Returns [{step, label, path, mtime}], step entries descending; epoch
    entries are sorted separately by their own step-like int, interleaved
    around the step entries - the UI can re-sort by mtime/step as it likes.
    """
    output_dir = vdir / "output"
    if not output_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    # avoid the same relative path appearing twice (shouldn't collide in
    # theory, but glob overlap + symlinks as a safety net)
    candidates: list[Path] = []
    candidates.extend(output_dir.glob("training_state_*.pt"))
    state_root = output_dir / "state"
    if state_root.exists():
        candidates.extend(state_root.glob("task_*/training_state_*.pt"))
    for f in candidates:
        if not f.is_file():
            continue
        m = _STATE_FILE_RE.search(f.name)
        if not m:
            continue
        key = str(f.resolve())
        if key in seen:
            continue
        seen.add(key)
        kind = m.group(1)  # "step" or "epoch"
        n = int(m.group(2))
        try:
            mtime = f.stat().st_mtime
        except OSError:
            mtime = 0.0
        items.append({
            "step": n if kind == "step" else 0,
            "label": f"{kind} {n}",
            "path": str(f),
            "mtime": mtime,
            "_kind": kind,  # for internal sorting, stripped before returning
            "_n": n,
        })
    # step entries first (descending by step), then epoch entries (descending by epoch).
    items.sort(key=lambda x: (0 if x["_kind"] == "step" else 1, -x["_n"]))
    for it in items:
        it.pop("_kind", None)
        it.pop("_n", None)
    return items


def list_project_state_ckpts(
    conn: sqlite3.Connection, project: dict[str, Any]
) -> list[dict[str, Any]]:
    """List every version's state.pt files across a project, grouped by version (used by the Train page's resume_state picker).

    Returns [{version_id, label, items: [{step, label, path, mtime}, ...]}],
    versions ordered by `created_at` ascending, items by step descending. A
    version with no output .pt files keeps its group with an empty items list.
    """
    pid = int(project["id"])
    slug = str(project["slug"])
    groups: list[dict[str, Any]] = []
    for v in list_versions(conn, pid):
        vdir = version_dir(pid, slug, str(v["label"]))
        groups.append({
            "version_id": int(v["id"]),
            "label": str(v["label"]),
            "items": list_state_ckpts(vdir),
        })
    return groups


def list_project_lora_ckpts(
    conn: sqlite3.Connection, project: dict[str, Any]
) -> list[dict[str, Any]]:
    """List every version's LoRA checkpoints (.safetensors) across a project, grouped by version (used by the resume_lora picker).

    Returns [{version_id, label, items: [{kind, value, label, path, mtime}, ...]}],
    versions ordered by `created_at` ascending; items sorted by list_lora_ckpts's
    own ordering (final -> step desc -> epoch desc -> other).
    """
    pid = int(project["id"])
    slug = str(project["slug"])
    groups: list[dict[str, Any]] = []
    for v in list_versions(conn, pid):
        vdir = version_dir(pid, slug, str(v["label"]))
        groups.append({
            "version_id": int(v["id"]),
            "label": str(v["label"]),
            "items": list_lora_ckpts(vdir),
        })
    return groups


def _write_version_json(v: dict[str, Any], pdir_label_path: Path) -> None:
    pdir_label_path.mkdir(parents=True, exist_ok=True)
    (pdir_label_path / "version.json").write_text(
        json.dumps(v, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Default training subfolder: Kohya-style N_label, repeat=1.
# Created by default so users landing on the Curation page can immediately
# copy images in, without first having to "+ create a new folder".
DEFAULT_TRAIN_FOLDER = "1_data"


def _ensure_version_tree(vdir: Path) -> None:
    # samples/ is no longer created here: sample images are now task-scoped
    # archives (studio_data/tasks/<id>/samples/); anything under the version
    # tree is just historical data from old tasks, and read compatibility is
    # handled by samples.py's multi-candidate resolution.
    for sub in ("train", "reg", "output"):
        (vdir / sub).mkdir(parents=True, exist_ok=True)
    (vdir / "train" / DEFAULT_TRAIN_FOLDER).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def _row_to_version(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return dict(row) if row else None


def get_version(
    conn: sqlite3.Connection, version_id: int
) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM versions WHERE id = ?", (version_id,)
    ).fetchone()
    return _row_to_version(row)


def _must_get(conn: sqlite3.Connection, version_id: int) -> dict[str, Any]:
    v = get_version(conn, version_id)
    if not v:
        raise VersionError(
            "Version not found", code="version.not_found",
            details={"id": version_id}, http_status=404,
        )
    return v


def list_versions(
    conn: sqlite3.Connection, project_id: int
) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM versions WHERE project_id = ? ORDER BY created_at ASC",
            (project_id,),
        )
    ]


def create_version(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    label: str,
    fork_from_version_id: Optional[int] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """Label validation: only [A-Za-z0-9_.-]+; unique within the project.

    When fork_from_version_id is given, fully copies the source version's
    user-generated artifacts:
        train/, reg/, config.yaml, .unlocked.json (PP10.4)
    Output artifacts (output/, samples/, monitor_state.json) are never copied.
    After copying config.yaml, it's immediately rewritten once to force
    data_dir / reg_data_dir / output_dir / output_name to the new version's paths.

    ADR-0007 PR-5: forking no longer inherits stage / status / phase; a new
    version always starts from the preparing / curating defaults. After
    forking, the user continues from the curation phase.
    """
    p = projects.get_project(conn, project_id)
    if not p:
        raise VersionError(
            "Project not found", code="project.not_found",
            details={"id": project_id}, http_status=404,
        )
    if not _VALID_LABEL.fullmatch(label):
        raise VersionError(
            f'Invalid version label "{label}"; use letters, digits, '
            "underscore, hyphen, or dot",
            code="version.label_invalid", details={"name": label},
            http_status=400,
        )
    # uniqueness
    if conn.execute(
        "SELECT 1 FROM versions WHERE project_id = ? AND label = ?",
        (project_id, label),
    ).fetchone():
        raise VersionError(
            f'Version label "{label}" already exists',
            code="version.label_exists", details={"name": label},
            http_status=400,
        )

    src_config_name: Optional[str] = None
    if fork_from_version_id is not None:
        src = get_version(conn, fork_from_version_id)
        if not src or src["project_id"] != project_id:
            raise VersionError(
                "The version to copy from was not found in this project",
                code="version.fork_source_invalid",
                details={"id": fork_from_version_id}, http_status=404,
            )
        src_config_name = src["config_name"]

    now = time.time()
    cur = conn.execute(
        "INSERT INTO versions(project_id, label, config_name, created_at, note) "
        "VALUES (?, ?, ?, ?, ?)",
        (project_id, label, src_config_name, now, note),
    )
    conn.commit()
    vid = int(cur.lastrowid)

    vdir = version_dir(project_id, p["slug"], label)
    _ensure_version_tree(vdir)

    if fork_from_version_id is not None:
        src = _must_get(conn, fork_from_version_id)
        src_vdir = version_dir(project_id, p["slug"], src["label"])
        # train / reg: recursively copy the directory (only if it exists)
        for sub in ("train", "reg"):
            src_sub = src_vdir / sub
            if src_sub.exists():
                _copytree(src_sub, vdir / sub)
        # config.yaml + .unlocked.json: single-file copy
        for fname in ("config.yaml", ".unlocked.json"):
            src_file = src_vdir / fname
            if src_file.exists():
                shutil.copy2(src_file, vdir / fname)
        # After copying config.yaml over, data_dir / reg_data_dir /
        # output_dir / output_name still point at the source version;
        # rewrite once with force_project_overrides=True to fix them to the
        # new version's paths. reg_data_dir is auto-detected by
        # project_specific_overrides based on whether the new version's
        # reg/meta.json exists - it follows what actually got copied.
        v_for_rewrite = _must_get(conn, vid)
        new_cfg_path = vdir / "config.yaml"
        if new_cfg_path.exists():
            from .. import version_config as _vc  # deferred import to avoid a cycle
            try:
                cfg = _vc.read_version_config(p, v_for_rewrite)
                _vc.write_version_config(
                    p, v_for_rewrite, cfg, force_project_overrides=True
                )
            except _vc.VersionConfigError:
                # a corrupt source config shouldn't block creating the new version; the user can switch presets on the Train page
                pass

    v = _must_get(conn, vid)
    _write_version_json(v, vdir)

    # first version in a project -> auto-set as active
    if p.get("active_version_id") is None:
        projects.update_project(conn, project_id, active_version_id=vid)

    return v


def _copytree(src: Path, dst: Path) -> None:
    """Recursively copy a directory (including subfolders and same-named metadata files).

    Hardlinks are more restricted on Windows, so this always does a plain
    copy (see PP1's notes on this). Generalized from _copytree_train since
    PP10.1 - train / reg both use this now.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for sub in src.iterdir():
        target = dst / sub.name
        if sub.is_dir():
            _copytree(sub, target)
        else:
            shutil.copy2(sub, target)


_UPDATABLE = {
    "note", "config_name", "output_lora_path", "trigger_word",
    "status", "phase", "last_failure_reason",
}


def update_version(
    conn: sqlite3.Connection, version_id: int, **fields: Any
) -> dict[str, Any]:
    v = _must_get(conn, version_id)
    keep = {k: val for k, val in fields.items() if k in _UPDATABLE}
    if "status" in keep and keep["status"] not in VersionStatus.VALUES:
        raise VersionError(f"Invalid status: {keep['status']!r}")
    if "phase" in keep and keep["phase"] not in VersionPhase.VALUES:
        raise VersionError(f"Invalid phase: {keep['phase']!r}")
    if not keep:
        return v
    cols = ", ".join(f"{k} = ?" for k in keep)
    params: list[Any] = list(keep.values()) + [version_id]
    conn.execute(f"UPDATE versions SET {cols} WHERE id = ?", params)
    conn.commit()
    v = _must_get(conn, version_id)
    p = projects.get_project(conn, v["project_id"])
    if p:
        _write_version_json(v, version_dir(p["id"], p["slug"], v["label"]))
    return v


def delete_version(conn: sqlite3.Connection, version_id: int) -> None:
    """rmtree the version directory + DELETE the db row; reassigns active automatically if needed. Unrecoverable."""
    v = _must_get(conn, version_id)
    p = projects.get_project(conn, v["project_id"])
    if p:
        src = version_dir(p["id"], p["slug"], v["label"])
        if src.exists():
            shutil.rmtree(src, ignore_errors=True)

        if p.get("active_version_id") == version_id:
            # pick whichever remaining version has the newest created_at; clear if none left
            row = conn.execute(
                "SELECT id FROM versions WHERE project_id = ? AND id != ? "
                "ORDER BY created_at DESC LIMIT 1",
                (v["project_id"], version_id),
            ).fetchone()
            new_active = int(row[0]) if row else None
            projects.update_project(
                conn, v["project_id"], active_version_id=new_active
            )

    conn.execute("DELETE FROM versions WHERE id = ?", (version_id,))
    conn.commit()


def activate_version(
    conn: sqlite3.Connection, version_id: int
) -> dict[str, Any]:
    """Set the given version as the project's active_version. Returns the updated version."""
    v = _must_get(conn, version_id)
    projects.update_project(conn, v["project_id"], active_version_id=version_id)
    return v


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def _scan_caption_dataset(root: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Scan root/<folder>/ for image count and captioned count (.txt / .json sidecar).

    train/ and validation/ share the same layout (validation mirrors train's
    subfolder structure), so they share this scan. Returns (folders, total, tagged).
    """
    folders: list[dict[str, Any]] = []
    total = 0
    tagged = 0
    if root.exists():
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                cnt = 0
                for f in sub.iterdir():
                    if not (f.is_file() and f.suffix.lower() in IMAGE_EXTS):
                        continue
                    cnt += 1
                    if f.with_suffix(".txt").exists() or f.with_suffix(".json").exists():
                        tagged += 1
                folders.append({"name": sub.name, "image_count": cnt})
                total += cnt
    return folders, total, tagged


def stats_for_version(p: dict[str, Any], v: dict[str, Any]) -> dict[str, Any]:
    """train / validation image and tagged counts / reg count / whether output exists."""
    vdir = version_dir(p["id"], p["slug"], v["label"])
    train_folders, train_total, tagged_total = _scan_caption_dataset(vdir / "train")
    _, val_total, val_tagged = _scan_caption_dataset(vdir / "validation")
    reg_dir = vdir / "reg"
    reg_total = 0
    reg_meta_exists = False
    if reg_dir.exists():
        # reg/{train-subfolder-mirror}/{post_id}.png - scanned recursively (matches the source script)
        for f in reg_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                reg_total += 1
        reg_meta_exists = (reg_dir / "meta.json").exists()
    output_dir = vdir / "output"
    has_output = output_dir.exists() and any(output_dir.iterdir())
    return {
        "train_image_count": train_total,
        "tagged_image_count": tagged_total,
        "train_folders": train_folders,
        "validation_image_count": val_total,
        "validation_tagged_count": val_tagged,
        "reg_image_count": reg_total,
        "reg_meta_exists": reg_meta_exists,
        "has_output": has_output,
    }


def compute_bucket_histogram(
    train_dir: Path,
    resolutions: list[int],
    aspect_ratio_limit: float = 2.0,
    prefer_json: bool = True,
) -> list[dict[str, Any]]:
    """Compute the training set's ARB bucket distribution using the **real** BucketManager (matches actual training bucket-for-bucket).

    Scan rules mirror the trainer's ``ImageDataset._scan`` / ``_make_sample``,
    to avoid the preview disagreeing with the actual training count:
    - loose images at the root count with repeat=1 + the config's resolution list;
    - subfolders are scanned **recursively** (``rglob``), with px override / repeat parsed from the folder name;
    - **only images with a caption count** (trainer drops anything without a ``.json``/``.txt``/``.caption``).

    Each image fans out into a bucket per resolution; count = the number of
    effective samples (repeat times the number of resolution tiers). Reuses
    the runtime's ``BucketManager`` + ``_parse_folder_meta`` instead of
    introducing a third copy of the bucketing algorithm. Returns ``[{reso,
    buckets: [{w, h, count}]}]``, resolutions ascending, buckets by count descending.
    """
    from runtime.training.dataset import BucketManager, ImageDataset
    from PIL import Image

    train_dir = Path(train_dir)
    base_resos = [int(r) for r in resolutions]
    mgrs: dict[int, Any] = {}

    def mgr_for(reso: int):
        if reso not in mgrs:
            mgrs[reso] = BucketManager(int(reso), aspect_ratio_limit=aspect_ratio_limit)
        return mgrs[reso]

    def has_caption(img_path: Path) -> bool:
        # mirrors _make_sample: prefer_json and .json exists -> json; otherwise needs .txt or .caption.
        if prefer_json and img_path.with_suffix(".json").exists():
            return True
        return img_path.with_suffix(".txt").exists() or img_path.with_suffix(".caption").exists()

    hist: dict[int, dict[tuple[int, int], int]] = {}

    def add_image(img_path: Path, repeat: int, resos: list[int]) -> None:
        if not has_caption(img_path):
            return
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            return
        for target_reso in resos:
            bw, bh = mgr_for(target_reso).get_bucket(w, h)
            bmap = hist.setdefault(int(target_reso), {})
            bmap[(bw, bh)] = bmap.get((bw, bh), 0) + repeat

    if train_dir.exists():
        # loose images at the root: repeat=1, no px prefix -> use the config's resolution list
        for p in sorted(train_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                add_image(p, 1, base_resos)
        # subfolders: recursive scan + px override / repeat parsing
        for sub in sorted(train_dir.iterdir()):
            if not sub.is_dir():
                continue
            reso_override, repeat, _label = ImageDataset._parse_folder_meta(sub.name)
            resos = [reso_override] if reso_override else base_resos
            for f in sorted(sub.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    add_image(f, repeat, resos)

    out: list[dict[str, Any]] = []
    for reso in sorted(hist):
        buckets = [
            {"w": w, "h": h, "count": c}
            for (w, h), c in sorted(hist[reso].items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        ]
        out.append({"reso": reso, "buckets": buckets})
    return out


class _NavitTokenStub:
    """Minimal dataset shell that hands ``NavitPackBatchSampler`` a list of token counts.

    ``dataset_token_counts`` discovers token counts through the
    ``token_count_for_index`` attribute, so packing (including
    shuffle/strategy/drop_last) goes through the exact same code path as the
    **real packer**, not a second copy of the algorithm.
    """

    def __init__(self, counts: list[int]) -> None:
        self.token_count_for_index = counts

    def __len__(self) -> int:
        return len(self.token_count_for_index)


def compute_navit_pack_estimate(
    data_dirs: list[Path],
    resolutions: list[int],
    aspect_ratio_limit: float = 2.0,
    prefer_json: bool = True,
    *,
    native_resolution: bool = False,
    token_budget: int = 16384,
    max_images_per_pack: int = 0,
    strategy: str = "next_fit",
    ffd_window: int = 256,
    drop_last: bool = False,
    over_budget: str = "downscale",
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate the number of packs per epoch for NaViT packing mode (= the numerator of optimizer steps/epoch).

    Scan rules share the same source as ``compute_bucket_histogram``
    (mirroring ``ImageDataset._scan``: loose root images, then subfolders
    sorted+rglob, only images with a caption count, repeat expansion); the
    per-image token count and packing both reuse the runtime's real
    implementation:

    - ``native_resolution=True``: ``plan_native_fit_image`` (floor-16 +
      over-budget downscale), multi-resolution fan-out collapses to a single
      tier (mirrors ``ImageDataset.__init__``);
    - otherwise, token count is derived from the ARB bucket size
      (``(w//16)*(h//16)``, matching the convention
      ``dataset_token_counts`` derives from the latent shape);
    - packing goes through the real ``NavitPackBatchSampler`` (same
      shuffle(seed)+strategy+drop_last), so the epoch-0 pack count matches
      the training log's ``dataset_len``/steps exactly; later epochs drift by
      a few steps due to reshuffling, so this is still described externally
      as an "estimate".

    ``data_dirs`` is passed as ``[train_dir]`` or ``[train_dir, reg_dir]``
    (reg participates in the same packing pool, matching ``MergedDataset``'s
    main+reg concatenation order).

    Known bias: the model's per-side RoPE token cap (read from the
    pos_embedder at training time) isn't available here, so it's treated as
    unbounded - this only affects extreme oversized images whose single side
    exceeds cap x16 px (the budget cap still applies).

    Returns ``{packs_per_epoch, samples, avg_images_per_pack, token_min,
    token_max, token_budget, strategy, native, downscaled, sizes}``;
    ``sizes`` is only non-empty under native (a histogram of native sizes
    ``[{w,h,count}]``, count includes repeat, sorted by count descending).
    """
    from runtime.training.dataset import (
        ImageDataset,
        NavitPackBatchSampler,
        plan_native_fit_image,
    )
    from PIL import Image

    base_resos = [int(r) for r in resolutions]
    if native_resolution and len(base_resos) > 1:
        # mirrors ImageDataset.__init__: fan-out is meaningless under native, collapse to a single tier
        base_resos = base_resos[:1]
    mgrs: dict[int, Any] = {}

    def mgr_for(reso: int):
        if reso not in mgrs:
            from runtime.training.dataset import BucketManager
            mgrs[reso] = BucketManager(int(reso), aspect_ratio_limit=aspect_ratio_limit)
        return mgrs[reso]

    def has_caption(img_path: Path) -> bool:
        if prefer_json and img_path.with_suffix(".json").exists():
            return True
        return img_path.with_suffix(".txt").exists() or img_path.with_suffix(".caption").exists()

    token_counts: list[int] = []
    size_hist: dict[tuple[int, int], int] = {}
    downscaled = 0

    def add_image(img_path: Path, repeat: int, resos: list[int]) -> None:
        nonlocal downscaled
        if not has_caption(img_path):
            return
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            return
        if native_resolution:
            try:
                plan = plan_native_fit_image(
                    w, h, max_tokens=token_budget, max_side_tokens=0,
                    over_budget=over_budget,
                )
            except ValueError:
                # an over-budget image with over_budget="fail": training would
                # fail-fast; the estimate skips it and doesn't count it
                # (better than a 500 taking down the whole distribution panel)
                return
            token_counts.extend([plan.token_count] * repeat)
            key = (plan.width, plan.height)
            size_hist[key] = size_hist.get(key, 0) + repeat
            if plan.was_downscaled:
                downscaled += repeat
        else:
            # mirrors _scan's expansion order: reso fan-out outer, repeat inner
            for target_reso in resos:
                bw, bh = mgr_for(target_reso).get_bucket(w, h)
                # same convention as dataset_token_counts's latent-shape derivation:
                # (px/8 latent) // patch_spatial(2) -> px // 16
                token_counts.extend([(bw // 16) * (bh // 16)] * repeat)

    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            continue
        for p in sorted(data_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                add_image(p, 1, base_resos)
        for sub in sorted(data_dir.iterdir()):
            if not sub.is_dir():
                continue
            reso_override, repeat, _label = ImageDataset._parse_folder_meta(sub.name)
            resos = [reso_override] if reso_override else base_resos
            for f in sorted(sub.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    add_image(f, repeat, resos)

    if not token_counts:
        return {
            "packs_per_epoch": 0, "samples": 0, "avg_images_per_pack": 0,
            "token_min": 0, "token_max": 0, "token_budget": int(token_budget),
            "strategy": str(strategy), "native": bool(native_resolution),
            "downscaled": 0, "sizes": [],
        }

    sampler = NavitPackBatchSampler(
        _NavitTokenStub(token_counts),
        token_budget=int(token_budget),
        max_images_per_pack=int(max_images_per_pack or 0),
        shuffle=True,
        seed=int(seed),
        drop_last=bool(drop_last),
        strategy=str(strategy or "next_fit"),
        ffd_window=int(ffd_window or 0),
    )
    packs = len(sampler)
    samples = len(token_counts)
    return {
        "packs_per_epoch": packs,
        "samples": samples,
        "avg_images_per_pack": round(samples / packs, 1) if packs else 0,
        "token_min": min(token_counts),
        "token_max": max(token_counts),
        "token_budget": int(token_budget),
        "strategy": str(strategy),
        "native": bool(native_resolution),
        "downscaled": downscaled,
        "sizes": [
            {"w": w, "h": h, "count": c}
            for (w, h), c in sorted(size_hist.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        ],
    }
