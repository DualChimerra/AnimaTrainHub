"""Supervisor main class -- PR-4 extracted from supervisor.py (zero behavior change).

Design highlights:
    - Single process, serial execution (at most one worker at a time, avoiding the
      complexity of multiple tasks fighting over the GPU)
    - Scheduling priority: project_jobs (download/tag/reg_build) > training tasks
      -- so data-prep work never gets stuck behind training
    - Each task gets its own log file:
        * task: studio_data/logs/{task_id}.log
        * job:  studio_data/jobs/{job_id}.log
      while a job runs, a LogTailer publishes log deltas as job_log_appended SSE
    - Cancellation uses SIGTERM (Unix) / CTRL_BREAK_EVENT (Windows), then kill after
      a 30-second timeout
    - Startup recovery: on restart, any orphaned task/job left with status='running'
      is marked failed
    - Tests can inject a cmd_builder in place of the real worker invocation

The main class is **not split up** (kept as a single ~1100-line class): all 37
methods read/write shared self fields (`_slots / _daemon_* / _stop / _thread /
_db_path`), so state coupling is very high and there's no clean sub-domain
boundary -- splitting into Mixins/helper classes would actually increase future
maintenance cost (see the decision log in tmp/0.11.0_planning.md PR-4). Leaf
helpers (_Slot / default cmd builder / _maybe_finalize_version /
_kill_process_tree) have been moved to sibling modules; this file only keeps the
Supervisor class body.
"""
from __future__ import annotations

import itertools
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .. import db, secrets as _secrets
from ..services import eval_auto, eval_validation
from ..services.runtime import xformers as _xformers_svc
from ..services.projects import jobs as project_jobs
from ..infrastructure.log_tail import LogTailer, MonitorStatePoller
from ..paths import (
    LOGS_DIR,
    REPO_ROOT,
    STUDIO_DATA,
    STUDIO_DB,
    USER_PRESETS_DIR,
    task_dir,
    task_log_path,
)
from ..services.inference.daemon import (
    InferenceDaemon,
    STATE_STOPPED as _DAEMON_STOPPED,
    get_daemon,
)
from .resources import (
    RESOURCE_EXCLUSIVE,
    RESOURCE_LIGHT,
    job_resource_class,
)
from .cmd_builder import (
    _EVENT_MARKER,
    CmdBuilder,
    EventCallback,
    JobCmdBuilder,
    _default_cmd_builder,
    _default_job_cmd_builder,
    _resolve_monitor_state_path,
)
from .finalizer import _maybe_finalize_version
from .process import _kill_process_tree
from .slot import SLOT_DATA, SLOT_TRAIN, _Slot

logger = logging.getLogger(__name__)


def _tail_log_for_error_msg(log_path: Path, max_lines: int = 12, max_chars: int = 800) -> str:
    """B-1.6: upgrade a failed task's db.error_msg from "exit code 1" to a traceback excerpt.

    Strategy: read the last N lines of jobs/<id>.log; find the last occurrence of
    'Traceback' and cut from there; otherwise fall back to the last N lines. Truncate
    to max_chars to fit the UI display width.

    On failure, returns "" (caller falls back to the "exit code N" default).
    """
    try:
        if not log_path.exists():
            return ""
        text = log_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if not lines:
            return ""
        tb_start = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("Traceback"):
                tb_start = i
                break
        snippet_lines = lines[tb_start:] if tb_start is not None else lines[-max_lines:]
        out = "\n".join(snippet_lines).strip()
        if len(out) > max_chars:
            out = "..." + out[-(max_chars - 3):]
        return out
    except Exception:
        logger.exception("tail log %s failed", log_path)
        return ""


class Supervisor:
    POLL_INTERVAL = 1.0
    TERMINATE_GRACE = 30.0

    def __init__(
        self,
        *,
        on_event: Optional[EventCallback] = None,
        cmd_builder: Optional[CmdBuilder] = None,
        job_cmd_builder: Optional[JobCmdBuilder] = None,
        db_path: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
        configs_dir: Optional[Path] = None,
        poll_interval: Optional[float] = None,
        terminate_grace: Optional[float] = None,
    ) -> None:
        self._on_event: EventCallback = on_event or (lambda _evt: None)
        self._cmd_builder: CmdBuilder = cmd_builder or _default_cmd_builder
        self._job_cmd_builder: JobCmdBuilder = (
            job_cmd_builder or _default_job_cmd_builder
        )
        self._db_path = db_path or STUDIO_DB
        self._logs_dir = logs_dir or LOGS_DIR
        self._configs_dir = configs_dir or USER_PRESETS_DIR
        self._poll = poll_interval if poll_interval is not None else self.POLL_INTERVAL
        self._grace = terminate_grace if terminate_grace is not None else self.TERMINATE_GRACE

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # PP10.2.b: two slots. TRAIN slot only runs tasks, DATA slot only runs project_jobs.
        # download always runs in parallel with training; tag / reg_build are deferred
        # by default while training is active.
        self._slots: list[_Slot] = [
            _Slot(name=SLOT_TRAIN),
            _Slot(name=SLOT_DATA),
        ]
        self._log_seq = itertools.count()

        # commit 9: generate tasks go through the daemon and don't occupy a _Slot;
        # tracked via separate fields instead. The daemon runs one task at a time;
        # the model is lazily loaded and reused across tasks.
        self._daemon_lock = threading.Lock()
        self._daemon_active_task_id: Optional[int] = None
        self._daemon_state_poller: Optional[MonitorStatePoller] = None
        self._daemon_cancel_pending: bool = False
        self._daemon_listener_registered = False
        # 0.17 item1: generate tasks run through the daemon with no run.log, so LogTab
        # is empty. When dispatching, we open that task's run.log and persist the
        # daemon's log output for the duration of the run + emit task_log_appended
        # (the daemon runs serially, one at a time, so ownership is unambiguous);
        # closed at finalize. Both the log thread and the supervisor thread touch it,
        # so it is always accessed under _daemon_lock.
        self._daemon_log_fp: Optional[Any] = None

    # ------------------------------------------------------------------ Control
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="studio-supervisor", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for slot in self._slots:
            if slot.busy:
                self._terminate_slot(slot)
        # Stop the inference daemon (if running). Failure here doesn't block
        # supervisor shutdown.
        try:
            get_daemon().stop(timeout=timeout)
        except Exception:
            logger.exception("inference daemon stop failed")
        if self._thread:
            self._thread.join(timeout=timeout)

    def _find_slot(self, *, kind: str, id: int) -> Optional[_Slot]:
        for slot in self._slots:
            if slot.kind == kind and slot.id == id:
                return slot
        return None

    def cancel(self, task_id: int) -> bool:
        """Cancel a task: pending/scheduled -> status=canceled immediately; running ->
        signal asynchronously and return right away.

        ADR 0006 PR-2: a paused task can also be canceled, moving status directly from
        paused to canceled.
        ADR Addendum 2: the resume checkpoint file is kept (still resumable after
        cancellation); only the paused_* fields are cleared.

        The key point of the async path: **it must not block the web request thread**.
        The supervisor main loop naturally polls proc.poll(), picks up the exit code,
        and goes through `_finish_slot`, which writes status=canceled. If the process
        hasn't exited after the 30s background grace timer, it force-kills the whole
        process tree.
        """
        with db.connection_for(self._db_path) as conn:
            task = db.get_task(conn, task_id)
            if not task:
                return False
            if task["status"] in ("pending", "scheduled"):
                db.update_task(
                    conn, task_id, status="canceled", finished_at=time.time()
                )
                self._on_event(
                    {"type": "task_state_changed", "task_id": task_id, "status": "canceled"}
                )
                return True
        if task["status"] == "paused":
            # The process has already exited, so no signal is needed -- clear the
            # paused_* fields, then separately write status=canceled + finished_at.
            # ADR Addendum 2: the resume checkpoint file is kept, so it's still
            # resumable after cancellation (the last_state_* fields remain).
            # Deliberately done outside the `with` block: _clear_pause_fields opens
            # its own connection internally, to avoid nesting.
            self._clear_pause_fields(task_id)
            with db.connection_for(self._db_path) as conn:
                db.update_task(
                    conn, task_id,
                    status="canceled",
                    finished_at=time.time(),
                )
            self._on_event(
                {"type": "task_state_changed", "task_id": task_id, "status": "canceled"}
            )
            return True
        if task["status"] == "running":
            # R-5: after the ledger merge, a "running" entry may actually be a data
            # job (DATA slot, kind="job") -- cancellation is unified through this
            # entry point, with the same SIGTERM semantics as cancel_job.
            slot = (
                self._find_slot(kind="task", id=task_id)
                or self._find_slot(kind="job", id=task_id)
            )
            if slot is not None:
                self._signal_terminate_async(slot)
                return True
            with self._daemon_lock:
                is_daemon_task = self._daemon_active_task_id == task_id
                if is_daemon_task:
                    self._daemon_cancel_pending = True
            if is_daemon_task:
                if get_daemon().cancel_active_task(task_id):
                    return True
                logger.warning("daemon cancel request missed; task_id=%s", task_id)
            return True
        return False

    def is_task_pausable(self, task_id: int) -> bool:
        """ADR Section 8.1 + Addendum 1: the UI's is_pausable signal.

        Conditions: the task is running on a slot, the `train_loop_started` event has
        been received, **`last_auto_epoch_state_path` has been set** (i.e. the first
        epoch's auto backup has been written), and there is no pause/cancel pending.
        If any condition fails, the UI should hide the pause button.

        ADR 0006 Addendum 1: disabling pause before the first epoch finishes is a key
        safeguard -- without an auto_epoch_state.pt yet, pressing pause would make the
        supervisor fall back to canceling with no recoverable progress, so the UI
        hides the button outright to prevent misclicks.
        """
        slot = self._find_slot(kind="task", id=task_id)
        if slot is None:
            return False
        return (
            slot.proc is not None
            and slot.train_loop_started
            and slot.last_auto_epoch_state_path is not None
            and not slot.pause_pending
            and not slot.cancel_pending
        )

    def pause(self, task_id: int) -> tuple[bool, str]:
        """Pause a running task: send a soft signal so handle_interrupt can save state
        before exiting.

        Returns (success, reason_if_failed).

        ADR Section 8.1 defense-in-depth: by the time the API calls this method, the UI
        should already have hidden the pause button based on the SSE `is_pausable`
        field; this method re-validates the train_loop_started signal server-side and
        rejects if it's not ready / status isn't running / the task doesn't exist.

        Non-blocking: calling `_signal_pause_async` returns immediately. The child
        process emits an event -> `_on_task_log` updates the slot -> the child process
        exits -> `_finish_slot` marks it paused. The UI's modal subscribes to SSE to
        watch progress (ADR Section 4.3).
        """
        with db.connection_for(self._db_path) as conn:
            task = db.get_task(conn, task_id)
        if not task:
            return False, "task not found"
        if task["status"] != "running":
            return False, f"task status is {task['status']!r}, not running"
        slot = self._find_slot(kind="task", id=task_id)
        if slot is None:
            return False, "task not on a slot (generate-on-daemon not supported)"
        if not slot.train_loop_started:
            return False, "train loop not started yet, retry after a few seconds"
        if slot.pause_pending:
            return False, "pause already pending"
        if slot.cancel_pending:
            return False, "task is being canceled"
        self._signal_pause_async(slot)
        return True, ""

    @property
    def current_task_id(self) -> Optional[int]:
        for slot in self._slots:
            if slot.kind == "task":
                return slot.id
        return None

    @property
    def current_job_id(self) -> Optional[int]:
        for slot in self._slots:
            if slot.kind == "job":
                return slot.id
        return None

    # -------------------------------------------------------------- Main loop
    def _loop(self) -> None:
        try:
            self._reconcile_orphans()
        except Exception:
            logger.exception("reconcile failed")
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("supervisor tick failed")
            self._stop.wait(self._poll)

    def _reconcile_orphans(self) -> None:
        # ADR 0006 PR-2 compatibility note: list_tasks(status="running") filters
        # strictly by status, so paused tasks (status='paused') never end up in this
        # loop by construction -- a paused task survives a supervisor restart with its
        # state unchanged (ADR Section 8.4).
        with db.connection_for(self._db_path) as conn:
            for t in db.list_tasks(conn, status="running"):
                logger.info("orphan running task %d -> failed", t["id"])
                db.update_task(
                    conn,
                    t["id"],
                    status="failed",
                    finished_at=time.time(),
                    pid=None,
                    error_msg="supervisor restart while task was running",
                )
                self._on_event(
                    {
                        "type": "task_state_changed",
                        "task_id": t["id"],
                        "status": "failed",
                    }
                )
            n = project_jobs.cleanup_orphan_running(conn)
            if n:
                logger.info("orphan running jobs -> failed: %d", n)

    def _tick(self) -> None:
        # 0) 0.17 P-B: promote due scheduled tasks to pending so the dispatch step
        #    below can see them. We ignore queue_held here -- hold semantics mean
        #    "stop dispatching new work", not "stop clarifying status"; a promoted
        #    task that becomes pending is still blocked by hold just the same.
        self._promote_due_scheduled()

        # 1) Reap first: poll every busy slot once, and run _finish_slot for any that
        #    exited.
        for slot in self._slots:
            if not slot.busy:
                continue
            assert slot.proc is not None
            rc = slot.proc.poll()
            if rc is not None:
                self._finish_slot(slot, rc)

        # 2) Dispatch work to idle slots (by slot responsibility). R-1: generate is now
        #    folded into the exclusive dispatch path (unified FIFO + centralized
        #    admission), so there's no longer a separate step 3.
        for slot in self._slots:
            if slot.busy:
                continue
            if slot.name == SLOT_TRAIN:
                self._dispatch_exclusive_tasks(slot)
            elif slot.name == SLOT_DATA:
                self._dispatch_data(slot)

    def _promote_due_scheduled(self) -> None:
        """0.17 P-B: promote a task whose scheduled_at is due -> pending + publish a
        state event."""
        try:
            with db.connection_for(self._db_path) as conn:
                promoted = db.promote_due_scheduled(conn)
        except Exception:
            logger.exception("promote_due_scheduled failed")
            return
        for tid in promoted:
            logger.info("scheduled task %d due -> pending", tid)
            self._on_event(
                {"type": "task_state_changed", "task_id": tid, "status": "pending"}
            )

    # ---- Pending task selection -----------------------------------------------
    def _next_pending_task_in(self, types: tuple[str, ...]) -> Optional[dict[str, Any]]:
        """Find the first task in the pending queue matching one of the given task_types."""
        with db.connection_for(self._db_path) as conn:
            pending = db.list_tasks(conn, status="pending")
        for t in pending:
            tt = t.get("task_type") or "train"
            if tt in types:
                return t
        return None

    # ---- R-1 resource tier admission (docs/design/queue-resource-model-0.17.md Section 3) ----

    def _daemon_active(self) -> bool:
        """Whether the daemon has an active generate task (submitted but not yet finalized)."""
        with self._daemon_lock:
            return self._daemon_active_task_id is not None

    def _data_slot_exclusive_busy(self) -> bool:
        """Whether the DATA slot is currently running an exclusive-tier job
        (eval_samples, base-model-level VRAM)."""
        for slot in self._slots:
            if (
                slot.name == SLOT_DATA and slot.busy
                and slot.job_kind is not None
                and job_resource_class(slot.job_kind) == RESOURCE_EXCLUSIVE
            ):
                return True
        return False

    def _exclusive_busy(self) -> bool:
        """Whether the system as a whole has any exclusive-tier work running (the
        precondition for the "at most 1 concurrent" admission rule).

        Checks each of the three execution slots: the TRAIN slot (train/reg_ai), the
        daemon (active generate), and the DATA slot (eval_samples). This is the shared
        root fix for L1 (generate and training missing a mutual-exclusion backend
        guard) and L2 (training not yielding to a running eval_samples job).
        """
        return (
            self._train_busy()
            or self._daemon_active()
            or self._data_slot_exclusive_busy()
        )

    def _dispatch_exclusive_tasks(self, slot: _Slot) -> None:
        """Unified dispatch for exclusive-tier work (tasks table: train / reg_ai /
        generate share one FIFO).

        D-R3 equal-priority FIFO: there's no priority ordering among the three types;
        the head of the queue is picked by `priority DESC, created_at ASC`, and a
        running item is never preempted. Routing: train/reg_ai -> TRAIN slot child
        process; generate -> daemon (the daemon is one of the executors for the
        exclusive tier, not a separate lane).

        eval_samples (project_jobs table), before the R-3 ledger merge, is dispatched
        by `_dispatch_data`, but shares the same `_exclusive_busy` admission gate --
        during the transition period, the cross-table ordering favors the tasks side
        grabbing any opening first; after R-3 everything moves into one shared FIFO.

        ADR 0006 PR-2: skip this dispatch pass when queue_held=True (ADR Section 3.2).
        """
        if self._queue_held():
            return
        if self._exclusive_busy():
            return
        task = self._next_pending_task_in(
            ("train", "reg_ai", "generate", "eval_samples")
        )
        if task is None:
            return
        ttype = task.get("task_type") or "train"
        if ttype == "generate":
            # enqueue_generate first does create_task(pending), then writes
            # config.json and sets config_path -- between those two steps, the task is
            # already pending but config_path is still NULL. Don't submit it yet (the
            # daemon would report "config not found"); wait for the next tick.
            # FIFO semantics: we don't skip past it to grab a later task (the window
            # is under 1s).
            if not task.get("config_path"):
                return
            self._submit_to_daemon(task)
            return
        if ttype == "eval_samples":
            # R-3: an exclusive-tier data job. It queues under the same FIFO as
            # train/generate (D-R3 cross-type equal priority), and executes on the
            # DATA slot (a worker child process). If the DATA slot is occupied by a
            # light job, wait for it to finish (light jobs are always short) rather
            # than skipping ahead.
            data_slot = next(
                (s for s in self._slots if s.name == SLOT_DATA), None
            )
            if data_slot is None or data_slot.busy:
                return
            if self._maybe_yield_daemon():
                return
            self._spawn_job(data_slot, project_jobs.as_job(dict(task)) or task)
            return
        # train / reg_ai: the daemon's resident model holds an exclusive lease, which
        # must be revoked before spawning (unload frees VRAM). The case where the
        # daemon is running a generate is already caught by _exclusive_busy; this only
        # handles the idle-but-loaded lease.
        if self._maybe_yield_daemon():
            return  # the daemon is still holding VRAM; wait for the next tick to dispatch
        self._spawn_task(slot, task)

    def _queue_held(self) -> bool:
        """ADR Section 3.2 queue hold switch, persisted across supervisor restarts (db kv)."""
        try:
            with db.connection_for(self._db_path) as conn:
                return db.get_queue_held(conn)
        except Exception:
            logger.exception("failed to read queue_held")
            return False  # on read failure, default to allowing dispatch (fail open)

    def _maybe_yield_daemon(self) -> bool:
        """If the daemon is holding VRAM, trigger an unload; the caller should skip
        this dispatch pass.

        R-1: the daemon's resident model counts as an exclusive lease. Before
        dispatching exclusive-tier work (train / reg_ai / eval_samples), that lease
        must be revoked -- **no setting exempts this anymore** (the old
        allow_gpu_during_train setting used to let "training + resident base model"
        coexist, which was part of L3). The conservative path used when the light-tier
        setting is off also reuses this function.

        Return value:
          - True: the daemon is still holding VRAM (running a generate, or an unload
                  request was just sent); the caller should not dispatch and should
                  recheck on the next tick
          - False: the daemon isn't holding the GPU (not started, or already
                  unloaded); safe to dispatch immediately
        """
        daemon = get_daemon()
        if not daemon.is_model_loaded:
            return False
        if daemon.is_busy:
            # Don't force-interrupt a generate the user explicitly triggered; wait for
            # it to finish
            return True
        try:
            daemon.request_unload()
            logger.info("requested daemon unload to yield GPU")
        except Exception:
            logger.exception("daemon unload request failed")
        return True

    def _dispatch_data(self, slot: _Slot) -> None:
        """DATA slot: runs project_jobs. R-1 admission by resource tier (fixes L2/L3):

        - io (download): always allowed (only constrained by queue_held)
        - light (tag / preprocess / reg_build / eval metrics): always allowed when no
          exclusive work is running (an idle resident model on the daemon is harmless
          -- it's small); when exclusive work is running, it depends on
          `queue.light_tasks_during_train` (on by default). When that setting is
          **off**, the conservative path matches the old default: it also requires
          the daemon's lease to have been released
        - exclusive (eval_samples, base-model-level): same rules as train -- only
          dispatched once there's no exclusive work running and the daemon's lease has
          been revoked, **ignoring the light-tier setting** (fixes L3)

        ADR 0006 PR-2: skip this dispatch pass when queue_held=True, including
        download. Semantically, hold means "pause the whole queue from dispatching new
        work", regardless of tier.
        """
        if self._queue_held():
            return
        exclusive_busy = self._exclusive_busy()
        light_parallel = self._light_tasks_during_train()
        with db.connection_for(self._db_path) as conn:
            pending = project_jobs.list_pending_fifo(conn)
        for job in pending:
            cls = job_resource_class(job["kind"])
            if cls == RESOURCE_EXCLUSIVE:
                # eval_samples goes through the unified exclusive FIFO
                # (_dispatch_exclusive_tasks queues it alongside train/generate at
                # equal priority); this function only handles light + io.
                continue
            if cls == RESOURCE_LIGHT:
                if exclusive_busy and not light_parallel:
                    continue
                if not light_parallel and self._maybe_yield_daemon():
                    continue  # conservative mode: wait for the daemon to unload
            self._spawn_job(slot, job)
            return

    def _train_busy(self) -> bool:
        for slot in self._slots:
            if slot.name == SLOT_TRAIN and slot.busy:
                return True
        return False

    def _light_tasks_during_train(self) -> bool:
        """R-1: whether light-tier jobs are allowed while exclusive work is running
        (on by default; falls back to the schema default on read failure)."""
        try:
            return bool(_secrets.load().queue.light_tasks_during_train)
        except Exception:
            return False

    # -------------------------------------------------------------- Child processes
    def _spawn_task(self, slot: _Slot, task: dict[str, Any]) -> None:
        # ADR-0009 PR-1 C6 trace_id propagation across processes:
        #   1) task.request_trace_id is stored by the API endpoint when the task is
        #      enqueued (the contextvar bound by TraceIdMiddleware at the moment of
        #      the HTTP request); for old tasks / ones without it, fall back to
        #      bg-{uuid}
        #   2) bind it to a ContextVar so every logger.x call throughout
        #      _spawn_task carries it
        #   3) inject ANIMA_TRACE_ID / ANIMA_PROCESS_NAME env vars into the worker
        #      child process
        from ..infrastructure.logging import (
            PROCESS_ENV, TRACE_ENV,
            bind_trace_id, new_trace_id, reset_trace_id,
        )
        trace_id = task.get("request_trace_id") or f"bg-{new_trace_id()}"
        kind = task.get("task_type") or "train"
        process_name = f"worker:{kind}/{task['id']}"
        _trace_token = bind_trace_id(trace_id)
        try:
            cfg_path = self._resolve_task_config_path(task)
            if not cfg_path.exists():
                self._fail_task_config_missing(task, cfg_path)
                return

            self._freeze_task_snapshot(int(task["id"]), cfg_path)

            # Task-scoped file layout: monitor state always lives at
            # tasks/<id>/monitor/state.json, decoupled from the version (it used to
            # live at versions/<label>/monitor/task_<id>/state.json, where deleting a
            # version would also wipe out the task's history)
            monitor_state_path = _resolve_monitor_state_path(task)
            # Inject it into the task dict up front for cmd_builder to use, and for
            # persisting to the db
            task = dict(task)
            task["monitor_state_path"] = str(monitor_state_path)

            # Task-scoped file layout: logs live at tasks/<id>/run.log, alongside
            # monitor / samples / snapshot. Old tasks that ran under
            # studio_data/logs/<id>.log are still read via the fallback in logs.py;
            # we no longer write new files there.
            log_path = task_log_path(task["id"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fp = open(log_path, "wb")

            # Before training: if the task has validation metrics enabled with a
            # split ratio set, carve out a held-out set from train/ into validation/
            # (moved, not used for training). Failure here doesn't block training,
            # only gets logged.
            self._maybe_split_validation(task, cfg_path, log_fp)

            cmd = self._cmd_builder(task, cfg_path)
            # ADR 0006 PR-1: injecting LORA_TASK_ID makes the training child process
            # write user-cycle saves under output_dir/state/task_<TID>/, avoiding
            # cross-task overwrites within the same version. As of Addendum 2,
            # auto_epoch_state.pt is instead written under the task's own file tree
            # at tasks/<id>/state/ -- the child process derives the path from
            # --monitor-state-file (bootstrap, same as samples/).
            # ADR-0009 PR-1 C6: TRACE_ENV + PROCESS_ENV let the worker bootstrap pick
            # these up.
            proc = self._popen(cmd, log_fp, extra_env={
                "LORA_TASK_ID": str(task["id"]),
                TRACE_ENV: trace_id,
                PROCESS_ENV: process_name,
            })

            slot.proc = proc
            slot.kind = "task"
            slot.id = task["id"]
            slot.log_fp = log_fp
            slot.cancel_pending = False

            tid = task["id"]

            # PP6.4 -- log tail -> SSE (replaces the frontend's old 2s log polling)
            slot.tailer = LogTailer(log_path, self._make_task_log_callback(slot, tid))
            slot.tailer.start()

            # PP6.4 -> PR #37: monitor_state.json changes -> SSE monitor_progress (delta protocol)
            slot.state_poller = MonitorStatePoller(
                monitor_state_path, self._make_monitor_callback(tid)
            )
            slot.state_poller.start()

            self._write_task_running_to_db(task, proc.pid, monitor_state_path)

            self._on_event(
                {
                    "type": "task_state_changed",
                    "task_id": task["id"],
                    "status": "running",
                }
            )
            logger.info(
                "started task %d on slot=%s (pid=%d)", task["id"], slot.name, proc.pid
            )
        finally:
            reset_trace_id(_trace_token)

    def _resolve_task_config_path(self, task: dict[str, Any]) -> Path:
        """PP6.3: prefer task.config_path (the version's own absolute config path);
        fall back to the old path, _configs_dir / {config_name}.yaml, if unset.
        """
        explicit_cfg = task.get("config_path")
        if explicit_cfg:
            return Path(explicit_cfg)
        return self._configs_dir / f"{task['config_name']}.yaml"

    def _maybe_split_validation(
        self, task: dict[str, Any], cfg_path: Path, log_fp: Any
    ) -> None:
        """Before training, carve out a held-out validation set from train/ into
        validation/ (moved).

        Only takes effect when the task has eval_validation enabled with ratio>0;
        tops up proportionally, leaves things alone once enough images are present,
        and never moves images back. Failure here only gets logged, never blocks
        training.
        """
        try:
            with db.connection_for(self._db_path) as conn:
                summary = eval_validation.split_for_task(conn, task, cfg_path)
        except Exception:
            logger.exception("validation split failed for task=%s", task.get("id"))
            return
        if summary and summary.get("moved"):
            try:
                log_fp.write(
                    f"[eval-validation] moved {summary['moved']} image(s) to "
                    f"validation/ (train={summary['train']}, "
                    f"validation={summary['validation']})\n".encode("utf-8")
                )
                log_fp.flush()
            except Exception:
                pass

    def _fail_task_config_missing(
        self, task: dict[str, Any], cfg_path: Path
    ) -> None:
        """Mark the task failed and publish an event when its config file is missing."""
        explicit_cfg = task.get("config_path")
        with db.connection_for(self._db_path) as conn:
            now = time.time()
            db.update_task(
                conn,
                task["id"],
                status="failed",
                started_at=now,
                finished_at=now,
                error_msg=(
                    f"config not found: {cfg_path}"
                    if explicit_cfg
                    else f"preset not found: {task['config_name']}"
                ),
            )
        self._on_event(
            {
                "type": "task_state_changed",
                "task_id": task["id"],
                "status": "failed",
            }
        )

    def _freeze_task_snapshot(self, task_id: int, cfg_path: Path) -> None:
        """ADR-0007 Section 11.7: for backward compatibility with old tasks, freeze the
        config at startup if it wasn't frozen already.

        New tasks are already frozen at enqueue time; this call is an idempotent
        no-op for those. A freeze failure here for an old task doesn't block it from
        starting.
        """
        try:
            from ..services import task_snapshot
            task_snapshot.freeze_config(task_id, cfg_path)
        except Exception:
            logger.exception(
                "task %s config snapshot freeze failed (non-fatal)", task_id
            )

    def _make_task_log_callback(
        self, slot: _Slot, tid: int
    ) -> Callable[[str], None]:
        """LogTailer callback: recognize the __EVENT__: protocol -> mirror state onto
        the slot + publish SSE; plain lines -> task_log_appended.

        ADR 0006 PR-2: the training worker talks to the supervisor via the
        __EVENT__: protocol (pause_state / train_loop_started /
        auto_epoch_backup_written / resume_state_loaded). Mirrors the jobs side's
        _on_line path.
        """
        def _on_task_log(line: str) -> None:
            if line.startswith(_EVENT_MARKER):
                try:
                    rest = line[len(_EVENT_MARKER):]
                    evt_type, payload_str = rest.split(":", 1)
                    import json as _json
                    payload = _json.loads(payload_str) if payload_str else {}
                except Exception:
                    # B-4.4: silently dropping a malformed event means the UI's
                    # pause_state never arrives -> the pause button stays grayed out
                    # forever. logger.exception goes to studio.log; the
                    # event_malformed SSE event lets the frontend surface it (without
                    # blocking the task).
                    logger.exception("malformed event marker: %r", line[:200])
                    self._on_event({
                        "type": "event_malformed",
                        "task_id": tid,
                        "raw_preview": line[:200],
                    })
                    return  # don't forward this as a log line
                # State machine mirroring (see ADR Section 8.1 / _on_line / Addendum 1 Supervisor section)
                if evt_type == "pause_state":
                    # ADR Addendum 1 plan Delta: state_path being None/empty means the
                    # pause happened within the first epoch -> falls through
                    # _finish_slot's cancel branch (empty pause_state_path downgrades
                    # to canceled).
                    slot.pause_state_path = str(payload.get("state_path") or "")
                    slot.pause_config_path = str(payload.get("config_path") or "")
                    slot.pause_step = payload.get("step")
                elif evt_type == "train_loop_started":
                    slot.train_loop_started = True
                elif evt_type == "auto_epoch_backup_written":
                    # ADR 0006 Addendum 1: loop.py emits this once at the end of each
                    # epoch -> marks the slot field -> the is_pausable upgrade
                    # condition is met -> SSE unlocks the UI's pause button.
                    slot.last_auto_epoch_state_path = str(payload.get("state_path") or "") or None
                    slot.last_auto_epoch_config_path = str(payload.get("config_path") or "") or None
                    # ADR Addendum 2: also persist to the db. Slot fields are
                    # in-memory state that's lost the moment the process or machine
                    # dies; once persisted to the db, the resume checkpoint path can
                    # still be looked up afterward whether the task later ends up
                    # failed (crash/shutdown) or canceled, and the resume endpoint
                    # uses that to allow resuming.
                    self._persist_last_state(tid, payload)
                elif evt_type == "resume_state_loaded":
                    # ADR Section 5.5 / PR-3: once the training child process's
                    # load_training_state succeeds, the paused_* fields have already
                    # been consumed, so clear the db fields to avoid staleness.
                    # ADR Addendum 2: **the file is no longer deleted** --
                    # auto_epoch_state.pt is a single file that gets overwritten each
                    # time rather than accumulating, so deleting it would instead
                    # create a window with "no checkpoint between resume and the end
                    # of the next epoch".
                    self._clear_pause_fields(tid)
                elif evt_type == "eval_training_finished":
                    slot.eval_training_finished_payload = dict(payload)
                self._on_event({
                    "type": evt_type,
                    "task_id": tid,
                    **payload,
                })
                return
            self._on_event({
                "type": "task_log_appended",
                "task_id": tid,
                "text": line,
                "seq": next(self._log_seq),
            })
        return _on_task_log

    def _queue_auto_eval_after_training(
        self, tid: int, payload: dict[str, Any]
    ) -> None:
        try:
            with db.connection_for(self._db_path) as conn:
                task = db.get_task(conn, tid)
                if not task:
                    return
                queued = eval_auto.queue_training_finished_eval(conn, task, payload)
        except Exception:
            logger.exception("after-training auto eval enqueue failed for task=%s", tid)
            return
        if not queued:
            return
        for job, run in queued:
            self._on_event({
                "type": "eval_auto_sample_queued",
                "task_id": tid,
                "job_id": job.get("id"),
                "project_id": job.get("project_id"),
                "version_id": job.get("version_id"),
                "run_id": run.get("run_id"),
                "checkpoint": run.get("checkpoint"),
            })
        self._on_event({
            "type": "eval_auto_after_training_queued",
            "task_id": tid,
            "count": len(queued),
        })

    def _make_monitor_callback(
        self, tid: int
    ) -> Callable[[dict[str, Any]], None]:
        """MonitorStatePoller callback: publish deltas from monitor_state.json as
        SSE monitor_progress (PR #37 delta protocol).

        payload is a delta (appended_losses/lr/samples plus the latest
        step/speed/...); the client fetches a snapshot once via GET /api/state on
        first load, then continuously merges these deltas on top.
        """
        def _on_state_delta(delta: dict[str, Any]) -> None:
            self._on_event({
                "type": "monitor_progress",
                "task_id": tid,
                "delta": delta,
            })
        return _on_state_delta

    def _write_task_running_to_db(
        self, task: dict[str, Any], pid: int, monitor_state_path: Path
    ) -> None:
        """DB writes after a task spawns: task.status=running + version.status=training
        (ADR-0007 Section 11.3-B dual write).
        """
        with db.connection_for(self._db_path) as conn:
            db.update_task(
                conn,
                task["id"],
                status="running",
                started_at=time.time(),
                pid=pid,
                monitor_state_path=str(monitor_state_path),
            )
            vid = task.get("version_id")
            if vid:
                try:
                    from ..services.projects import versions as _versions
                    _versions.update_version(
                        conn, int(vid),
                        status=_versions.VersionStatus.TRAINING,
                    )
                except Exception:
                    logger.exception(
                        "version.status=training write failed for task %s",
                        task["id"],
                    )

    def _spawn_job(self, slot: _Slot, job: dict[str, Any]) -> None:
        log_path = Path(job.get("log_path") or project_jobs.log_path_for(job["id"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # The worker itself opens the log in append mode; the supervisor just
        # attaches stdout to the same file
        log_fp = open(log_path, "ab")

        cmd = self._job_cmd_builder(job)
        proc = self._popen(cmd, log_fp)

        with db.connection_for(self._db_path) as conn:
            project_jobs.mark_running(conn, job["id"], pid=proc.pid)

        slot.proc = proc
        slot.kind = "job"
        slot.id = job["id"]
        slot.job_kind = job["kind"]  # R-1: used by _exclusive_busy to classify the tier
        slot.log_fp = log_fp
        slot.cancel_pending = False

        jid = job["id"]
        pid_ = job["project_id"]
        vid = job.get("version_id")
        kind = job["kind"]

        slot.tailer = LogTailer(
            log_path, self._make_job_log_callback(jid, pid_, vid, kind)
        )
        slot.tailer.start()

        self._on_event({
            "type": "job_state_changed",
            "job_id": jid,
            "project_id": pid_,
            "version_id": vid,
            "kind": kind,
            "status": "running",
        })
        logger.info(
            "started job %d on slot=%s (kind=%s, pid=%d)",
            jid, slot.name, kind, proc.pid,
        )

    def _make_job_log_callback(
        self,
        jid: int,
        pid_: Optional[int],
        vid: Optional[int],
        kind: str,
    ) -> Callable[[str], None]:
        """LogTailer callback: recognize the __EVENT__: protocol and publish typed
        SSE; plain lines -> job_log_appended.

        Structured event marker: the worker writes `__EVENT__:type:json_payload`,
        which the supervisor turns into a typed SSE event (not added to the job log).
        Lighter than building a dedicated IPC channel, and more reliable than having
        the frontend grep the log text. job_id / project_id are injected by the
        supervisor.
        """
        def _on_line(line: str) -> None:
            if line.startswith(_EVENT_MARKER):
                try:
                    rest = line[len(_EVENT_MARKER):]
                    evt_type, payload_str = rest.split(":", 1)
                    import json as _json
                    payload = _json.loads(payload_str) if payload_str else {}
                    self._on_event({
                        "type": evt_type,
                        "job_id": jid,
                        "project_id": pid_,
                        "version_id": vid,
                        "kind": kind,
                        **payload,
                    })
                except Exception:
                    logger.exception("malformed event marker: %r", line[:200])
                return  # don't forward this as a log line

            self._on_event({
                "type": "job_log_appended",
                "job_id": jid,
                "project_id": pid_,
                "version_id": vid,
                "kind": kind,
                "text": line,
                "seq": next(self._log_seq),
            })
        return _on_line

    # ----------------------------------------------- Daemon path (commit 9)
    def _submit_to_daemon(self, task: dict[str, Any]) -> None:
        """Push a generate task to the inference daemon.

        A parallel entry point to _spawn_task; there's no _Slot concept here -- the
        daemon manages its own active task.
        """
        import json as _json

        task_id = int(task["id"])
        cfg_path_str = task.get("config_path")
        cfg_path = Path(cfg_path_str) if cfg_path_str else None
        if cfg_path is None or not cfg_path.exists():
            self._fail_daemon_task(
                task_id, f"config not found: {cfg_path_str or '<none>'}",
            )
            return

        try:
            cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            self._fail_daemon_task(task_id, f"failed to read config: {e}")
            return

        # output_dir: taken from cfg if present (enqueue_generate writes it as
        # anima_gen_{tid}); falling back here is also fine
        output_dir = (
            cfg.get("output_dir")
            or str(STUDIO_DATA / "monitors" / f"task_{task_id}")
        )

        # monitor_state.json: the daemon writes this file, and the supervisor starts
        # a poller to push it over SSE
        monitor_state_path = _resolve_monitor_state_path(task)
        cfg["__monitor_state_file"] = str(monitor_state_path)

        daemon = get_daemon()
        if daemon.state == _DAEMON_STOPPED:
            try:
                daemon.start()
            except Exception as e:
                logger.exception("daemon start failed")
                self._fail_daemon_task(task_id, f"daemon start failed: {e}")
                return

        # Sync the idle timeout from secrets into the daemon right after spawning, so
        # the model doesn't stay resident after the first task finishes generating;
        # if the user changes the value in settings, the next dispatch will re-read
        # it too.
        daemon.sync_idle_timeout_from_secrets()

        if not self._daemon_listener_registered:
            daemon.add_global_listener(self._on_daemon_global_event)
            daemon.add_log_listener(self._on_daemon_log_line)
            self._daemon_listener_registered = True

        with self._daemon_lock:
            self._daemon_active_task_id = task_id
            self._daemon_cancel_pending = False
            # 0.17 item1: open this generate task's run.log (LogTab reads /api/logs ->
            # tasks/<id>/run.log). The daemon runs serially, one at a time, so all of
            # its log output during this run belongs to this task.
            try:
                lp = task_log_path(task_id)
                lp.parent.mkdir(parents=True, exist_ok=True)
                self._daemon_log_fp = open(lp, "ab")
            except Exception:
                logger.exception("open daemon task log failed")
                self._daemon_log_fp = None

        # poller: the daemon writes monitor_state.json -> SSE monitor_progress (delta protocol)
        def _on_state_delta(delta: dict[str, Any]) -> None:
            self._on_event({
                "type": "monitor_progress",
                "task_id": task_id,
                "delta": delta,
            })

        self._daemon_state_poller = MonitorStatePoller(monitor_state_path, _on_state_delta)
        self._daemon_state_poller.start()

        with db.connection_for(self._db_path) as conn:
            db.update_task(
                conn,
                task_id,
                status="running",
                started_at=time.time(),
                monitor_state_path=str(monitor_state_path),
            )
        self._on_event({
            "type": "task_state_changed",
            "task_id": task_id,
            "status": "running",
        })

        try:
            daemon.submit_task(
                task_id=task_id,
                config=cfg,
                output_dir=output_dir,
                on_event=self._on_daemon_task_event,
            )
            logger.info("submitted generate task %d to daemon", task_id)
            self._emit_daemon_state()
        except Exception as e:
            logger.exception("daemon submit failed")
            self._on_daemon_task_event({
                "kind": "error",
                "task_id": task_id,
                "message": f"daemon submit failed: {e}",
            })

    def _on_daemon_task_event(self, event: dict[str, Any]) -> None:
        """Task-level event pushed back by the daemon (image_done / done / error / preview_step)."""
        kind = event.get("kind")
        tid = int(event.get("task_id") or 0)
        if kind == "started":
            self._emit_daemon_state()
            return
        if kind in ("image_done", "image_error"):
            return
        if kind == "phase":
            # Generation phase (load/clip/sample/vae) -> the frontend progress bar
            # covers the non-sampling phases too (so it no longer sticks at 0%/100%)
            self._on_event({
                "type": "generate_phase",
                "task_id": tid,
                "name": event.get("name"),
            })
            return
        if kind == "preview_step":
            # commit 14: intermediate step progress + optional preview. step/total are
            # always present; image_b64 depends on
            # settings.preview_every_n_steps + whether TAEFlux is available
            self._on_event({
                "type": "generate_preview_step",
                "task_id": tid,
                "step": event.get("step"),
                "total": event.get("total"),
                "image_b64": event.get("image_b64"),
            })
            return
        if kind == "image_started":
            # Multiple images (XY grid or count>1): which image number is currently
            # in progress
            self._on_event({
                "type": "generate_image_started",
                "task_id": tid,
                "batch_idx": event.get("batch_idx"),
                "batch_total": event.get("batch_total"),
                "total_steps": event.get("total_steps"),
            })
            return
        if kind == "done":
            self._finalize_daemon_task(tid, status="done")
            self._emit_daemon_state()
        elif kind == "canceled":
            self._finalize_daemon_task(tid, status="canceled")
            self._emit_daemon_state()
        elif kind == "error":
            self._finalize_daemon_task(
                tid, status="failed", error_msg=str(event.get("message") or "daemon error"),
            )
            self._emit_daemon_state()

    def _on_daemon_log_line(self, entry: dict[str, Any]) -> None:
        """Daemon stderr delta lines -> SSE daemon_log_line (used by the frontend log drawer).

        0.17 item1: also appends to the current active generate task's run.log +
        emits task_log_appended (for live LogTab updates). The file write happens
        under the lock to avoid a race with finalize's close; the SSE emit happens
        outside the lock.
        """
        line = entry.get("line")
        self._on_event({
            "type": "daemon_log_line",
            "ts": entry.get("ts"),
            "seq": entry.get("seq"),
            "line": line,
        })
        with self._daemon_lock:
            tid = self._daemon_active_task_id
            fp = self._daemon_log_fp
            if tid is not None and fp is not None and isinstance(line, str):
                try:
                    fp.write((line + "\n").encode("utf-8", errors="replace"))
                    fp.flush()
                except Exception:
                    logger.exception("write daemon task log failed")
            else:
                tid = None
        if tid is not None and isinstance(line, str):
            self._on_event({"type": "task_log_appended", "task_id": tid, "text": line})

    def _on_daemon_global_event(self, event: dict[str, Any]) -> None:
        """Process-level daemon event (loaded / unloaded / stopped)."""
        kind = event.get("kind")
        if kind in ("loaded", "unloaded"):
            self._emit_daemon_state()
            return
        if kind == "stopped":
            with self._daemon_lock:
                tid = self._daemon_active_task_id
                cancel_pending = self._daemon_cancel_pending
            if tid is not None:
                if cancel_pending:
                    self._finalize_daemon_task(tid, status="canceled")
                else:
                    self._finalize_daemon_task(
                        tid, status="failed",
                        error_msg=f"daemon exited (rc={event.get('rc')})",
                    )
            self._emit_daemon_state()

    def _emit_daemon_state(self) -> None:
        """commit 13: broadcast the daemon's current state to SSE subscribers (the
        frontend status pill)."""
        daemon = get_daemon()
        with self._daemon_lock:
            active_tid = self._daemon_active_task_id
        try:
            self._on_event({
                "type": "daemon_state_changed",
                "state": daemon.state,
                "model_loaded": daemon.is_model_loaded,
                "busy": daemon.is_busy,
                "active_task_id": active_tid,
            })
        except Exception:
            logger.exception("emit daemon state failed")

    def _finalize_daemon_task(
        self,
        task_id: int,
        *,
        status: str,
        error_msg: Optional[str] = None,
    ) -> None:
        """Final cleanup for a task on the daemon: write db status + stop the poller +
        clear the active-task marker.

        As of commit 10, the images themselves live in the server's in-memory cache
        (not on disk), so they aren't cleaned up here -- that's left to client
        disconnect / LRU / lifespan (commit 11). This only cleans up the task's small
        on-disk leftovers:
          - anima_gen_{tid}/config.json + the empty directory
          - monitors/task_{tid}/state.json (if the fallback path was ever written to)
        """
        with self._daemon_lock:
            if self._daemon_active_task_id == task_id:
                self._daemon_active_task_id = None
                self._daemon_cancel_pending = False
            poller = self._daemon_state_poller
            self._daemon_state_poller = None
            log_fp = self._daemon_log_fp
            self._daemon_log_fp = None
        if poller is not None:
            try:
                poller.stop()
            except Exception:
                pass
        if log_fp is not None:  # 0.17 item1: close this task's run.log
            try:
                log_fp.close()
            except Exception:
                pass

        fields: dict[str, Any] = {
            "status": status,
            "finished_at": time.time(),
            "pid": None,
        }
        if error_msg:
            fields["error_msg"] = error_msg
        with db.connection_for(self._db_path) as conn:
            db.update_task(conn, task_id, **fields)

        try:
            from ..services.inference.core import cleanup_generate_tempdir
            cleanup_generate_tempdir(task_id)
        except Exception as e:
            logger.warning("cleanup generate tempdir failed: %s", e)

        self._on_event({
            "type": "task_state_changed",
            "task_id": task_id,
            "status": status,
        })
        logger.info("daemon task %d finished: %s", task_id, status)

    def _fail_daemon_task(self, task_id: int, msg: str) -> None:
        """A generate task failing before it was even handed to the daemon (e.g. missing config)."""
        with self._daemon_lock:
            if self._daemon_active_task_id == task_id:
                self._daemon_active_task_id = None
            log_fp = self._daemon_log_fp
            self._daemon_log_fp = None
        if log_fp is not None:  # 0.17 item1: close run.log as a fallback (this path usually isn't hit)
            try:
                log_fp.close()
            except Exception:
                pass
        with db.connection_for(self._db_path) as conn:
            db.update_task(
                conn, task_id,
                status="failed",
                started_at=time.time(),
                finished_at=time.time(),
                error_msg=msg,
            )
        self._on_event({
            "type": "task_state_changed",
            "task_id": task_id,
            "status": "failed",
        })

    # ---- Child process helpers -----------------------------------------------------------
    def _popen(
        self,
        cmd: list[str],
        log_fp: Any,
        extra_env: Optional[dict[str, str]] = None,
    ) -> subprocess.Popen:
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        # Windows defaults stdout to the ANSI codepage; any worker writing non-ASCII text / emoji
        # would trigger a UnicodeEncodeError, and logging's default backslashreplace
        # would turn it into \uXXXX, filling the task log with garbled text. Give
        # every child process a UTF-8 + unbuffered fallback here.
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Cut down on underlying libraries' loading progress bars (safetensors /
        # transformers / accelerate, etc. print hundreds of lines like
        # `Loading weights: NN%|...` when stdout is piped, drowning out the user's own
        # training log). Only silences "loading progress" output, doesn't affect
        # logger.error or training step logging.
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.setdefault("TRANSFORMERS_VERBOSITY", "error")
        env.setdefault("DIFFUSERS_VERBOSITY", "error")
        env.setdefault("ACCELERATE_DISABLE_RICH", "1")
        # xformers's triton probe would dump a harmless ImportError traceback into
        # the task log (no official triton wheel on Windows), which the failure
        # summary would mistake for the real failure reason; this app's xformers path
        # doesn't use triton kernels, so short-circuit it unconditionally.
        _xformers_svc.disable_triton_probe(env)
        # Training-side memory/VRAM watermark guard switch (Settings -> Training, off
        # by default). The runtime-side env default is "on" (a safety fallback for
        # running the CLI directly), so it only needs to be passed explicitly when
        # turned off; read by the training and reg-AI child processes -- the
        # generation daemon uses its own cfg.ram_guard and is unaffected.
        try:
            if not _secrets.load().training.ram_guard:
                env.setdefault("LORA_RAM_GUARD", "0")
        except Exception:
            logger.exception("failed to load training ram_guard setting")
        try:
            wandb_cfg = _secrets.load().wandb
            if wandb_cfg.enabled:
                # 0.18 presets: enabled is the top-level master switch; the remaining
                # fields are read from the currently selected preset.
                wb = wandb_cfg.active
                env.setdefault("WANDB_ENABLED", "1")
                env.setdefault("WANDB_MODE", wb.mode)
                env.setdefault("WANDB_LOG_SAMPLES", "1" if wb.log_samples else "0")
                env.setdefault("WANDB_SAMPLE_MAX_SIDE", str(wb.sample_max_side))
                env.setdefault("WANDB_SAMPLE_EVERY_N_STEPS", str(wb.sample_every_n_steps))
                env.setdefault("WANDB_UPLOAD_MODEL", "1" if wb.upload_model else "0")
                env.setdefault("WANDB_UPLOAD_MODEL_POLICY", wb.upload_model_policy)
                env.setdefault("WANDB_UPLOAD_STATE_MANUAL", "1" if wb.upload_state_manual else "0")
                env.setdefault("WANDB_UPLOAD_STATE_MANUAL_POLICY", wb.upload_state_manual_policy)
                env.setdefault("WANDB_UPLOAD_STATE_AUTO", "1" if wb.upload_state_auto else "0")
                env.setdefault("WANDB_UPLOAD_STATE_AUTO_POLICY", wb.upload_state_auto_policy)
                if wb.api_key:
                    env.setdefault("WANDB_API_KEY", wb.api_key)
                if wb.project:
                    env.setdefault("WANDB_PROJECT", wb.project)
                if wb.entity:
                    env.setdefault("WANDB_ENTITY", wb.entity)
                if wb.base_url:
                    env.setdefault("WANDB_BASE_URL", wb.base_url)
        except Exception:
            logger.exception("failed to load wandb settings")
        if extra_env:
            env.update(extra_env)
        return subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            creationflags=creationflags,
            env=env,
        )

    def _finish_slot(self, slot: _Slot, rc: int) -> None:
        kind = slot.kind
        cid = slot.id
        assert cid is not None and kind is not None
        if slot.log_fp:
            try:
                slot.log_fp.close()
            except Exception:
                pass
        if slot.tailer:
            try:
                slot.tailer.stop()
            except Exception:
                pass
        if slot.state_poller:
            try:
                slot.state_poller.stop()
            except Exception:
                pass

        # ADR 0006 PR-2 + Addendum 1 three-way branch (originally a two-way branch:
        # canceled vs done/failed).
        # paused takes highest priority -- pause_pending=True and the child process
        # emitted pause_state (with both state_path / config_path present) means the
        # pause genuinely succeeded.
        # ADR Addendum 1 plan Delta: pause_pending=True but pause_state_path empty
        # means the pause happened within the first epoch, or the child process
        # exited before it could emit (slow IO / exception / force kill) -> downgrade
        # to canceled (the fallback behind the ADR Section 4.3 modal's "force cancel,
        # discard progress").
        if slot.pause_pending and slot.pause_state_path:
            status = "paused"
        elif slot.pause_pending or slot.cancel_pending:
            status = "canceled"
        elif rc == 0:
            status = "done"
        else:
            status = "failed"

        if kind == "task":
            with db.connection_for(self._db_path) as conn:
                fields: dict[str, Any] = {
                    "status": status,
                    "exit_code": rc,
                    "finished_at": time.time(),
                    "pid": None,
                }
                if status == "failed":
                    # B-1.6: tail the last 12 lines of the task's run.log
                    # (prioritizing a Traceback if present) and append it to
                    # error_msg, so the UI's task list can show the root cause
                    # directly without digging through the trace every time. New
                    # tasks use tasks/<id>/run.log; old tasks fall back to the legacy
                    # logs/<id>.log.
                    new_log = task_log_path(cid)
                    tail_src = new_log if new_log.exists() else self._logs_dir / f"{cid}.log"
                    tail = _tail_log_for_error_msg(tail_src)
                    fields["error_msg"] = (
                        f"exit code {rc}\n{tail}" if tail else f"exit code {rc}"
                    )
                elif status == "paused":
                    fields["paused_state_path"] = slot.pause_state_path
                    fields["paused_config_path"] = slot.pause_config_path
                    fields["paused_step"] = slot.pause_step
                    fields["paused_at"] = time.time()
                db.update_task(conn, cid, **fields)
                # ADR-0007 Section 11.3-B: a task's terminal state (done/failed/canceled)
                # maps independently onto version.status. paused doesn't map (the task
                # can still be resumed, Section 11.3-A).
                if status in ("done", "failed", "canceled"):
                    _maybe_finalize_version(conn, cid, status)
            # As of commit 10, generate tasks go through the daemon rather than
            # SLOT_TRAIN, so this _finish_slot path only ever handles train / reg_ai;
            # generate tempdir cleanup is no longer needed here (moved to
            # _finalize_daemon_task).
            self._on_event(
                {"type": "task_state_changed", "task_id": cid, "status": status}
            )
            logger.info("task %d finished: %s (rc=%d)", cid, status, rc)
            if status == "done" and slot.eval_training_finished_payload is not None:
                self._queue_auto_eval_after_training(
                    cid,
                    slot.eval_training_finished_payload,
                )
        else:  # job
            with db.connection_for(self._db_path) as conn:
                if status == "done":
                    project_jobs.mark_done(conn, cid)
                elif status == "canceled":
                    project_jobs.mark_canceled(conn, cid)
                else:
                    # B-1.6: same idea as for tasks -- tail the job log and append it
                    # to error_msg.
                    # R-3: job logs have already moved to tasks/<id>/run.log as part
                    # of the ledger merge.
                    tail = _tail_log_for_error_msg(project_jobs.log_path_for(cid))
                    err_msg = f"exit code {rc}\n{tail}" if tail else f"exit code {rc}"
                    project_jobs.mark_failed(conn, cid, err_msg)
                job = project_jobs.get_job(conn, cid)
            self._on_event({
                "type": "job_state_changed",
                "job_id": cid,
                "project_id": job["project_id"] if job else None,
                "version_id": job.get("version_id") if job else None,
                "kind": job["kind"] if job else None,
                "status": status,
            })
            logger.info("job %d finished: %s (rc=%d)", cid, status, rc)

        slot.reset()

    def _terminate_slot(self, slot: _Slot) -> None:
        """Synchronously terminate the child process on a given slot (only used by
        supervisor.stop()).

        For cancellation on the web request path, use `_signal_terminate_async`
        instead, to avoid blocking the request thread for 30 seconds.
        """
        if not slot.proc:
            return
        slot.cancel_pending = True
        proc = slot.proc
        self._send_terminate_signal(proc)
        try:
            proc.wait(timeout=self._grace)
        except subprocess.TimeoutExpired:
            logger.warning(
                "%s %s on slot=%s did not exit in %.0fs, killing process tree",
                slot.kind, slot.id, slot.name, self._grace,
            )
            _kill_process_tree(proc.pid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _signal_terminate_async(self, slot: _Slot) -> None:
        """Non-blocking: send the soft termination signal and start a background grace
        timer that force-kills the process tree.

        The web request thread returns immediately, so reload() isn't blocked for 30
        seconds by a cancel request. The supervisor main loop polls proc.poll() every
        POLL_INTERVAL seconds, and as soon as the process exits, `_finish_slot` sets
        status to canceled and publishes the event.
        """
        if not slot.proc:
            return
        slot.cancel_pending = True
        proc = slot.proc
        self._send_terminate_signal(proc)

        grace = self._grace

        def _grace_then_kill_tree() -> None:
            # Can't use proc.wait() here -- it would race with the supervisor main
            # loop's poll; poll in a loop instead
            deadline = time.time() + grace
            while time.time() < deadline:
                if proc.poll() is not None:
                    return
                time.sleep(0.5)
            if proc.poll() is None:
                logger.warning(
                    "proc %d did not exit in %.0fs, killing process tree",
                    proc.pid, grace,
                )
                _kill_process_tree(proc.pid)

        threading.Thread(
            target=_grace_then_kill_tree,
            name=f"cancel-grace-{proc.pid}",
            daemon=True,
        ).start()

    @staticmethod
    def _send_terminate_signal(proc: subprocess.Popen) -> None:
        """The soft termination signal used by cancel.

        ADR 0006 PR-2: on Windows we no longer send CTRL_BREAK_EVENT -- it collides
        with the pause signal (pause occupies CTRL_BREAK_EVENT), and cancel is
        inherently meant to be a hard interrupt anyway, so it goes straight to
        taskkill /T /F. POSIX has no such conflict and keeps using SIGTERM.

        `_signal_terminate_async` still runs a grace timer afterward, but on Windows
        the process is already dead via taskkill by the time it fires, so the first
        poll in the grace loop just returns -- no time wasted.
        """
        try:
            if os.name == "nt":
                _kill_process_tree(proc.pid)
            else:
                proc.terminate()
        except Exception:
            logger.exception("send terminate signal failed")

    @staticmethod
    def _send_pause_signal(proc: subprocess.Popen) -> None:
        """The soft signal used by pause -- caught by the child process's
        handle_interrupt, which saves state.

        Windows: `CTRL_BREAK_EVENT` is delivered to the CREATE_NEW_PROCESS_GROUP
        child process group, and Python maps it to SIGBREAK (sig=21), caught by the
        handler registered during the resume phase.
        POSIX: `SIGINT` -- kept separate from SIGTERM so it doesn't collide with
        cancel, which uses SIGTERM.

        This signal path was verified via a spike (decision recorded in ADR 0006).
        """
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                proc.send_signal(signal.SIGINT)
        except Exception:
            logger.exception("send pause signal failed")

    def _signal_pause_async(self, slot: _Slot) -> None:
        """Non-blocking: send the pause signal, with no grace-period force-kill.

        The key difference from `_signal_terminate_async`: pause **never times out
        and downgrades**. Per ADR Section 4.3, the 30s threshold is handled by the
        UI's modal deciding the next step (wait another 30s / force-cancel and
        discard progress / terminate the task) -- the supervisor never proactively
        kills the process; that kill decision comes from the cancel API (called after
        the user picks an option in the modal).
        """
        if not slot.proc:
            return
        slot.pause_pending = True
        self._send_pause_signal(slot.proc)

    def _persist_last_state(self, task_id: int, payload: dict[str, Any]) -> None:
        """Write the checkpoint info from auto_epoch_backup_written into the tasks row
        (ADR Addendum 2).

        One UPDATE per epoch, negligible overhead; failures here are only logged, not
        raised -- training must not be affected by a DB write error, and if one epoch
        is missed it'll just write again on the next one.
        """
        state_path = str(payload.get("state_path") or "") or None
        if not state_path:
            return
        try:
            with db.connection_for(self._db_path) as conn:
                db.update_task(
                    conn, task_id,
                    last_state_path=state_path,
                    last_config_path=str(payload.get("config_path") or "") or None,
                    last_state_epoch=payload.get("epoch"),
                    last_state_step=payload.get("step"),
                )
        except Exception:
            logger.exception("task %s last_state persist failed", task_id)

    def _clear_pause_fields(self, task_id: int) -> None:
        """Clear the db `paused_*` fields (ADR Section 5.5 / Addendum 2 revision).

        Call sites:
          - the resume_state_loaded event (after cmd_builder successfully loads state)
          - cancel while paused -> canceled

        As of Addendum 2, **the checkpoint file is no longer deleted here**:
        auto_epoch_state.pt is a single file that gets overwritten rather than
        accumulating, so keeping it means a canceled task can still be resumed, and
        immediately still has a checkpoint again right after resuming. File cleanup
        has been unified into DELETE task (lifecycle.delete_queue_item).
        Deliberately **does not change status** -- the caller decides what status to write.
        """
        with db.connection_for(self._db_path) as conn:
            task = db.get_task(conn, task_id)
            if not task:
                return
            db.update_task(
                conn, task_id,
                paused_state_path=None,
                paused_config_path=None,
                paused_step=None,
                paused_at=None,
            )
