"""Queue task lifecycle (extracted from server.py in PR-6 commit 6).

14 routes:
    GET   /api/queue                 list (hides generate / reg_ai by default)
    POST  /api/queue                 enqueue (by preset name, can carry scheduled_at)
    POST  /api/queue/{task_id}/start_now  manually bump a scheduled task to pending (0.17 P-B)
    GET   /api/queue/hold            query queue-hold state + pending count waiting to resume
    POST  /api/queue/hold            hold the queue (dispatcher stops pulling new tasks)
    POST  /api/queue/release         resume scheduling
    POST  /api/queue/reorder         reorder by an id list
    GET   /api/queue/{task_id}       task DB row (including is_pausable / is_resumable signals)
    PUT   /api/queue/{task_id}/note    task note (v20 tasks.note; written from right-click / detail page)
    POST  /api/queue/{task_id}/cancel
    POST  /api/queue/{task_id}/pause   ADR 0006 §4.1
    POST  /api/queue/{task_id}/resume  ADR 0006 §6 path A + Addendum 2 (paused/failed/canceled)
    POST  /api/queue/{task_id}/retry   copies config_path / project_id / version_id into a new task
    DELETE /api/queue/{task_id}       only a terminal task can be deleted
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from ...deps import _supervisor
from ...schemas.queue import EnqueueRequest, ReorderRequest, TaskNoteBody
from .... import db
from ....domain.errors import (
    ConflictError,
    DomainError,
    NotFoundError,
    PresetNotFoundError,
    ValidationError,
)
from ....infrastructure.event_bus import bus
from ....paths import USER_PRESETS_DIR, task_dir
from ....services import task_snapshot
from ....supervisor.resources import (
    JOB_KIND_RESOURCE_CLASS,
    RESOURCE_EXCLUSIVE,
    TASK_TYPE_RESOURCE_CLASS,
)

router = APIRouter()

# R-5 tier view: the GPU view = the exclusive tier (train/reg_ai/generate/eval_samples),
# the data view = the light + io tiers. Derived from resources.py's tier mapping to
# prevent drift.
EXCLUSIVE_VIEW_TYPES: tuple[str, ...] = tuple(TASK_TYPE_RESOURCE_CLASS) + tuple(
    k for k, c in JOB_KIND_RESOURCE_CLASS.items() if c == RESOURCE_EXCLUSIVE
)
DATA_VIEW_TYPES: tuple[str, ...] = tuple(
    k for k, c in JOB_KIND_RESOURCE_CLASS.items() if c != RESOURCE_EXCLUSIVE
)
_RESOURCE_CLASS_TYPES = {"exclusive": EXCLUSIVE_VIEW_TYPES, "data": DATA_VIEW_TYPES}

# Starting states allowed for resume (ADR 0006 Addendum 2). done is excluded -- that means
# retraining, which goes through retry / ResumeFieldPicker.
RESUMABLE_STATUSES = ("paused", "failed", "canceled")


def _resume_source_paths(task: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Get the resume point (state_path, config_path) based on status.

    paused uses paused_* (written by the pause flow, points at auto_epoch_state.pt);
    failed/canceled use last_* (the supervisor writes these to DB on every
    auto_epoch_backup_written event per epoch, ADR Addendum 2); other statuses have no
    resume point.
    """
    status = task.get("status")
    if status == "paused":
        return task.get("paused_state_path"), task.get("paused_config_path")
    if status in ("failed", "canceled"):
        return task.get("last_state_path"), task.get("last_config_path")
    return None, None


def _is_resumable(task: dict[str, Any]) -> bool:
    """Signal for showing/hiding the UI's "resume training" button. Kept consistent with
    the resume endpoint's admission conditions: the state file must exist; if the config
    snapshot path is set, that file must exist too (strict freeze).
    """
    state_path, config_path = _resume_source_paths(task)
    if not state_path or not Path(state_path).exists():
        return False
    if config_path and not Path(config_path).exists():
        return False
    return True


def _enrich_tasks(items: list[dict[str, Any]]) -> None:
    """Inject is_pausable (ADR 0006 PR-4 §8.1) + is_resumable (Addendum 2) into every row. Mutates in place."""
    try:
        sup = _supervisor()
        for it in items:
            it["is_pausable"] = sup.is_task_pausable(int(it["id"]))
    except (HTTPException, DomainError):
        for it in items:
            it["is_pausable"] = False
    for it in items:
        it["is_resumable"] = _is_resumable(it)


# 0.17 P-E -- upper bound on page_size for history pagination, to prevent one huge pull.
_MAX_PAGE_SIZE = 100

# Note length cap -- the UI is a "one-line sticky note", not a log. Truncate rather than 422 on overflow.
_MAX_NOTE_LEN = 500


@router.get("/api/queue")
def list_queue(
    status: Optional[str] = None,
    include_generate: bool = False,
    group: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    q: Optional[str] = None,
    types: Optional[str] = None,
    resource_class: Optional[str] = None,
) -> dict[str, Any]:
    """Queue task list.

    - No `group` passed: returns everything, **still hiding generate/reg_ai** (compatible
      with Overview / Generate / Topbar / Monitor's `listQueue()` -- they rely on this default
      to avoid treating generate as a training task, or self-locking).
    - `group=live`: in-progress + waiting (running/paused/pending), unpaginated, `{items}`.
    - `group=history`: finished (done/failed/canceled), paginated:
      `{items, total, page, page_size}`; `status` sub-filters by terminal state, `q` searches
      name/config_name.

    0.17 P-F: **live/history no longer hide generate/reg_ai** (the queue page needs all types
    visible); instead `types` (comma-separated train/reg_ai/generate) does a positive type
    filter, pushed down into SQL to keep the pagination total accurate.
    """
    if status and status not in db.VALID_STATUSES:
        raise ValidationError(
            f"Unsupported status filter: {status}",
            code="queue.status_filter_invalid", details={"status": status},
            http_status=400,
        )
    if group is not None and group not in ("live", "history"):
        raise ValidationError(
            f"Unsupported queue group: {group}",
            code="queue.group_invalid", details={"group": group},
            http_status=400,
        )
    type_tuple: tuple[str, ...] = ()
    if types:
        type_tuple = tuple(t for t in (s.strip() for s in types.split(",")) if t)
        bad = [t for t in type_tuple if t not in db.VALID_TASK_TYPES]
        if bad:
            raise ValidationError(
                f"Unsupported task type filter: {','.join(bad)}",
                code="queue.type_filter_invalid", details={"types": bad},
                http_status=400,
            )
    # R-5 tier-view param: explicit types takes priority (a single type already implies a
    # tier); otherwise expand by tier.
    if resource_class is not None:
        if resource_class not in _RESOURCE_CLASS_TYPES:
            raise ValidationError(
                f"Unsupported resource class: {resource_class}",
                code="queue.resource_class_invalid",
                details={"resource_class": resource_class}, http_status=400,
            )
        if not type_tuple:
            type_tuple = _RESOURCE_CLASS_TYPES[resource_class]

    # R-3 ledger merge: data jobs landed in the tasks table. The GPU view (this endpoint)
    # excludes JOB_TASK_TYPES by default, pre-dating R-5 tiering -- doesn't exclude them when
    # an explicit types filter is given (caller specifies its own).
    if group is None:
        # Preserve old behavior: everything + hide generate/reg_ai (callers with no group
        # rely on this default) + hide data jobs (protects Topbar/Overview/Monitor from
        # treating them as training tasks).
        with db.connection_for() as conn:
            items = db.list_tasks(conn, status=status)
        items = db.filter_out_task_types(items, db.JOB_TASK_TYPES)
        if not include_generate:
            items = db.filter_out_task_types(items, ("generate", "reg_ai"))
        _enrich_tasks(items)
        return {"items": items}

    exclude = () if type_tuple else db.JOB_TASK_TYPES
    if group == "live":
        with db.connection_for() as conn:
            items = db.list_tasks_page(
                conn, statuses=db.LIVE_STATUSES, q=q, types=type_tuple,
                exclude_types=exclude,
            )
        _enrich_tasks(items)
        return {"items": items}

    # group == "history" -- paginated
    statuses = (
        (status,) if status in db.HISTORY_STATUSES else db.HISTORY_STATUSES
    )
    page = max(1, page)
    page_size = min(_MAX_PAGE_SIZE, max(1, page_size))
    offset = (page - 1) * page_size
    with db.connection_for() as conn:
        total = db.count_tasks(
            conn, statuses=statuses, q=q, types=type_tuple, exclude_types=exclude,
        )
        items = db.list_tasks_page(
            conn, statuses=statuses, q=q, types=type_tuple, exclude_types=exclude,
            limit=page_size, offset=offset,
        )
    _enrich_tasks(items)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("/api/queue")
def enqueue(body: EnqueueRequest) -> dict[str, Any]:
    cfg_path = USER_PRESETS_DIR / f"{body.config_name}.yaml"
    if not cfg_path.exists():
        raise PresetNotFoundError(
            f'Preset "{body.config_name}" not found',
            code="preset.not_found", details={"name": body.config_name},
        )
    name = body.name or body.config_name
    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name=name, config_name=body.config_name, priority=body.priority,
            scheduled_at=body.scheduled_at,
            commit=False,
        )
        frozen_cfg = task_snapshot.freeze_config(task_id, cfg_path)
        conn.execute(
            "UPDATE tasks SET config_path = ? WHERE id = ?",
            (str(frozen_cfg), task_id),
        )
        conn.commit()
        task = db.get_task(conn, task_id)
    bus.publish({
        "type": "task_state_changed",
        "task_id": task_id,
        "status": task["status"] if task else "pending",
    })
    return task or {"id": task_id}


@router.post("/api/queue/{task_id}/start_now")
def start_scheduled_now(task_id: int) -> dict[str, Any]:
    """0.17 P-B -- manually bump a scheduled task early: immediately flip to pending and join scheduling.

    scheduled_at is kept as a record (so "originally planned for 3:00, started early
    manually" stays traceable); only tasks in the scheduled state can be bumped, else 409.
    """
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
        if not task:
            raise NotFoundError("Task not found", code="task.not_found", details={"task_id": task_id})
        if task["status"] != "scheduled":
            raise ConflictError(
                "Only scheduled tasks can be started early",
                code="task.not_scheduled", details={"status": task["status"]},
            )
        db.update_task(conn, task_id, status="pending")
    bus.publish(
        {"type": "task_state_changed", "task_id": task_id, "status": "pending"}
    )
    return {"task_id": task_id, "status": "pending"}


@router.get("/api/queue/hold")
def get_queue_hold() -> dict[str, Any]:
    """View the current queue-hold state + the number of pending tasks waiting for scheduling to resume (used by the UI banner)."""
    with db.connection_for() as conn:
        held = db.get_queue_held(conn)
        pending = db.list_tasks(conn, status="pending")
    return {"held": held, "pending_waiting": len(pending)}


@router.post("/api/queue/hold")
def hold_queue() -> dict[str, Any]:
    """Hold the queue: the dispatcher stops pulling new tasks. Already-running tasks are unaffected (ADR §3.2).

    "Also pause running tasks" is split into two steps by the frontend modal: call this
    endpoint first, then call `/api/queue/{id}/pause` separately. The backend doesn't combine
    these into one operation.
    """
    with db.connection_for() as conn:
        db.set_queue_held(conn, True)
    bus.publish({"type": "queue_hold_changed", "held": True})
    return {"held": True}


@router.post("/api/queue/release")
def release_queue() -> dict[str, Any]:
    """Resume scheduling: the dispatcher goes back to pulling pending tasks by priority + created_at."""
    with db.connection_for() as conn:
        db.set_queue_held(conn, False)
    bus.publish({"type": "queue_hold_changed", "held": False})
    return {"held": False}


@router.post("/api/queue/reorder")
def reorder_queue(body: ReorderRequest) -> dict[str, Any]:
    with db.connection_for() as conn:
        db.reorder(conn, body.ordered_ids)
    return {"reordered": len(body.ordered_ids)}


@router.get("/api/queue/{task_id}")
def get_queue_item(task_id: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise NotFoundError("Task not found", code="task.not_found", details={"task_id": task_id})
    # ADR 0006 PR-4 -- the is_pausable signal lets the UI decide whether to show the pause
    # button (§8.1). Only computed when the supervisor is actually running; defaults to
    # False when idle (tests / during startup).
    try:
        task["is_pausable"] = _supervisor().is_task_pausable(task_id)
    except (HTTPException, DomainError):
        task["is_pausable"] = False
    task["is_resumable"] = _is_resumable(task)
    return task


@router.put("/api/queue/{task_id}/note")
def set_task_note(task_id: int, body: TaskNoteBody) -> dict[str, Any]:
    """Write / clear a task note (v20 tasks.note).

    Written by right-clicking any row on the queue page; also editable from the task detail
    page's overview. Pure UI metadata: a task in any state can be edited (even a long-finished
    historical task can get a note like "this config worked best").
    Empty / whitespace-only -> clears it (stores NULL).
    """
    raw = (body.note or "").strip()
    note = raw[:_MAX_NOTE_LEN] or None
    with db.connection_for() as conn:
        if not db.get_task(conn, task_id):
            raise NotFoundError(
                "Task not found", code="task.not_found", details={"task_id": task_id}
            )
        db.update_task(conn, task_id, note=note)
        task = db.get_task(conn, task_id)
    # The queue page / detail page may be open at the same time -- reuse task_state_changed
    # to refresh both.
    bus.publish({
        "type": "task_state_changed",
        "task_id": task_id,
        "status": (task or {}).get("status"),
    })
    return task or {"id": task_id, "note": note}


@router.post("/api/queue/{task_id}/cancel")
def cancel_task(task_id: int) -> dict[str, Any]:
    if not _supervisor().cancel(task_id):
        # The task may have already finished / not be under supervisor control
        with db.connection_for() as conn:
            task = db.get_task(conn, task_id)
        if not task:
            raise NotFoundError("Task not found", code="task.not_found", details={"task_id": task_id})
        if task["status"] in db.TERMINAL_STATUSES:
            raise ValidationError(
                "This task has already finished",
                code="task.already_finished", http_status=400,
            )
        raise ConflictError(
            "Cannot cancel this task in its current state", code="task.cancel_rejected",
        )
    return {"task_id": task_id, "canceled": True}


@router.post("/api/queue/{task_id}/pause")
def pause_task(task_id: int) -> dict[str, Any]:
    """Pause a running task (ADR §4.1 / §4.3).

    Async: returns immediately; the UI's modal subscribes to SSE to watch save progress.
    Once the supervisor receives `__EVENT__:pause_state` from the subprocess, it writes
    status as paused and publishes task_state_changed.
    """
    ok, reason = _supervisor().pause(task_id)
    if not ok:
        # Distinguish client error (404/409) from state-machine rejection (409)
        with db.connection_for() as conn:
            task = db.get_task(conn, task_id)
        if not task:
            raise NotFoundError("Task not found", code="task.not_found", details={"task_id": task_id})
        raise ConflictError(
            "Cannot pause this task right now",
            code="task.pause_rejected",
            details={"reason": reason} if reason else None,
        )
    return {"task_id": task_id, "pause_pending": True}


@router.post("/api/queue/{task_id}/resume")
def resume_task(task_id: int) -> dict[str, Any]:
    """Resume a paused / failed / canceled task (ADR 0006 §6 path A + Addendum 2).

    Flow:
      1. Validate status is in RESUMABLE_STATUSES + the resume-point file exists
         (paused reads paused_*; failed/canceled read last_*, i.e. the end-of-epoch auto
         backup)
      2. task -> pending; for failed/canceled, copy last_* into paused_* -- reuses the
         paused pipeline, so cmd_builder (--resume-state injection) / bootstrap_phase
         (sibling .config.json snapshot freeze) need zero changes downstream
      3. the supervisor naturally picks it up on its next _tick
      4. once the subprocess's load_training_state succeeds it emits `resume_state_loaded`
         -> the supervisor's `_clear_pause_fields` clears the db fields (the files themselves
         are kept, Addendum 2)
      5. failed/canceled had finalized the version to failed/canceled -- once revived,
         reconcile derives it back to training and publishes version_state_changed

    Missing file -> 409 (ADR §5.5: guides the user to start a new task via
    ResumeFieldPicker instead). done cannot be resumed (that means retraining, via
    retry / ResumeFieldPicker).
    """
    corrected_version: Optional[dict[str, Any]] = None
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
        if not task:
            raise NotFoundError("Task not found", code="task.not_found", details={"task_id": task_id})
        status = task["status"]
        if status not in RESUMABLE_STATUSES:
            raise ConflictError(
                "This task cannot be resumed",
                code="task.not_resumable", details={"status": status},
            )
        state_path, config_path = _resume_source_paths(task)
        if not state_path or not Path(state_path).exists():
            raise ConflictError(
                "The saved training state is missing; start a new run instead",
                code="task.resume_state_missing",
            )
        if config_path and not Path(config_path).exists():
            # A missing snapshot isn't fatal on its own (bootstrap_phase would fall back to the
            # args.config yaml), but resume semantics would drift; per ADR §5.7's strict
            # freeze principle, refuse to continue.
            raise ConflictError(
                "The saved training state is missing; start a new run instead",
                code="task.resume_state_missing",
            )
        fields: dict[str, Any] = dict(
            status="pending",
            started_at=None,
            finished_at=None,
            exit_code=None,
            error_msg=None,
        )
        if status in ("failed", "canceled"):
            # ADR Addendum 2 decision 3: move the resume point into paused_* to reuse the
            # existing pipeline.
            fields["paused_state_path"] = state_path
            fields["paused_config_path"] = config_path
            fields["paused_step"] = task.get("last_state_step")
        db.update_task(conn, task_id, **fields)
        if status in ("failed", "canceled") and task.get("version_id"):
            # ADR Addendum 2 decision 5: derive version status back to training from
            # failed/canceled (the task is now pending). _write_task_running_to_db will write
            # it again at dispatch time; changing it here first is just to keep the UI
            # immediately consistent.
            from ....services.projects import versions as _versions
            v, was_corrected = _versions.reconcile_version_status(
                conn, int(task["version_id"])
            )
            if was_corrected and v:
                corrected_version = v
    bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "pending"})
    if corrected_version is not None:
        from ....services.projects import versions as _versions
        bus.publish({
            "type": "version_state_changed",
            "project_id": corrected_version["project_id"],
            "version_id": corrected_version["id"],
            "status": _versions.get_status(corrected_version),
            "phase": _versions.get_phase(corrected_version),
        })
    return {"task_id": task_id, "status": "pending"}


@router.post("/api/queue/{task_id}/retry")
def retry_task(task_id: int) -> dict[str, Any]:
    """Re-enqueue a finished task: copy its full training context to create a new task.

    Fields that must be copied (introduced in PP6.1+; the old retry only copied
    name/config_name/priority, which made the supervisor fall back to the old legacy path
    of using the global preset instead of the version's private config, causing retry
    parameters to diverge from the original task):
    - config_path: absolute path to the version's private config
    - project_id / version_id: used for monitor_state_path resolution and stage advancement

    Not copied: status / pid / *_at / exit_code / error_msg / monitor_state_path (all are
    artifacts of the "last run"; the new task starts from pending and the supervisor
    re-resolves them).
    """
    with db.connection_for() as conn:
        original = db.get_task(conn, task_id)
        if not original:
            raise NotFoundError("Task not found", code="task.not_found", details={"task_id": task_id})
        if original["status"] not in db.TERMINAL_STATUSES:
            raise ValidationError(
                "Only finished tasks can be retried",
                code="task.not_retryable", http_status=400,
            )
        new_id = db.create_task(
            conn,
            name=original["name"],
            config_name=original["config_name"],
            priority=original["priority"],
            commit=False,
        )
        copy_fields: dict[str, Any] = {}
        # R-3: task_type / params must be copied together -- data-job-type tasks rely on
        # them to route to the right worker and restore parameters when re-run (missing them
        # would degrade to a train task running the wrong script).
        for k in ("config_path", "project_id", "version_id", "task_type", "params"):
            if original.get(k) is not None:
                copy_fields[k] = original[k]
        source_cfg = original.get("config_path")
        if source_cfg and Path(source_cfg).exists():
            copy_fields["config_path"] = str(
                task_snapshot.freeze_config(new_id, Path(source_cfg))
            )
        if copy_fields:
            cols = ", ".join(f"{key} = ?" for key in copy_fields)
            conn.execute(
                f"UPDATE tasks SET {cols} WHERE id = ?",
                [*copy_fields.values(), new_id],
            )
        conn.commit()
        new_task = db.get_task(conn, new_id)
    bus.publish(
        {"type": "task_state_changed", "task_id": new_id, "status": "pending"}
    )
    return new_task or {"id": new_id}


@router.delete("/api/queue/{task_id}")
def delete_queue_item(task_id: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
        if not task:
            raise NotFoundError("Task not found", code="task.not_found", details={"task_id": task_id})
        if task["status"] not in db.TERMINAL_STATUSES:
            raise ValidationError(
                "Only finished tasks can be deleted",
                code="task.not_deletable", http_status=400,
            )
        db.delete_task(conn, task_id)
    # task-scoped artifacts (snapshot/config.yaml / monitor/state.json / samples/ / run.log)
    # share the task DB row's lifecycle -- deleting the task cleans them up too. Old tasks
    # scattered at studio_data/logs/<id>.log / studio_data/monitors/task_<id>/ are left
    # untouched (no migration script written; kept as old-version compatibility test data).
    import shutil
    tdir = task_dir(task_id)
    if tdir.exists():
        shutil.rmtree(tdir, ignore_errors=True)
    # ADR Addendum 2 decision 4: **old-layout** resume points (<version>/output/state/task_<id>/,
    # leftovers from before this Addendum) aren't under task_dir, so they're cleaned
    # separately; the new layout (tasks/<id>/state/) is already covered by the task_dir
    # rmtree above. Only act when the directory name exactly equals task_<id> -- guards
    # against accidentally deleting another directory if a DB path is malformed.
    for col in ("last_state_path", "paused_state_path"):
        p = task.get(col)
        if not p:
            continue
        state_dir = Path(p).parent
        if state_dir.name == f"task_{task_id}" and state_dir.exists():
            shutil.rmtree(state_dir, ignore_errors=True)
    return {"deleted": task_id}
