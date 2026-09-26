"""Resident daemon for test generation: reuses the loaded model to avoid a 30-60s
reload on every generate.

Design highlights:
  - The daemon is a resident subprocess (runtime/anima_daemon.py), managed by the
    InferenceDaemon class inside the server process; JSON-over-stdio protocol,
    with stderr routed to logs
  - Lazy spawn: only started when the first generate task arrives; once started it
    stays alive until the server shuts down, the user manually unloads it, or it
    yields the GPU (commit 12)
  - Runs one task at a time (the queue is fed by the supervisor; the daemon itself
    doesn't queue), returning to idle once done
  - Protocol (line-delimited JSON):
      stdin  -> {"id": "<req_id>", "action": "generate"|"unload"|"ping", ...}
      stdout -> {"id": "<req_id>"|"_evt", "kind": "started"|"image_done"|
                 "done"|"error"|"loaded"|"unloaded", ...}
  - The image_done event payload includes base64 PNG bytes (as of commit 10); the
    reader decodes it into generate_cache, then forwards a "slimmed down" version
    of the event (with b64 stripped) to the supervisor callback, keeping the large
    payload out of the logging/SSE pipeline
  - A reader thread dispatches stdout events back to the callback; the caller (the
    supervisor) registers the callback
"""
from __future__ import annotations

import base64
import collections
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ...paths import REPO_ROOT
from ..runtime import xformers as _xformers_svc
from . import disk_cache as generate_cache

logger = logging.getLogger(__name__)

# Daemon state machine
STATE_STOPPED = "stopped"      # child process not started / has exited
STATE_STARTING = "starting"    # spawning, ready signal not yet received
STATE_IDLE = "idle"            # daemon alive and waiting for commands; model may or may not be loaded
STATE_BUSY = "busy"            # daemon is currently running a task
STATE_UNLOADING = "unloading"  # received the unload command, waiting for the unloaded event


# Path to the daemon process script
_DAEMON_SCRIPT = REPO_ROOT / "runtime" / "anima_daemon.py"


EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class _ActiveTask:
    """The task the daemon is currently running (or one just submitted that hasn't
    received a started event yet)."""
    task_id: int
    request_id: str
    on_event: EventCallback
    # Decision #15: freeze secrets.generate.save_test_images when the task starts,
    # to avoid a mid-task setting flip leaving one task half in cache, half on
    # disk. enqueueGenerate writes cfg.save_test_images_at_dispatch ->
    # submit_task reads it and stores it here -> _handle_image_done uses it to
    # decide the SSE delivery sub-field.
    save_to_disk: bool = False
    # The GenerateParamsSnapshot dict built by the frontend; on image_done it's
    # tucked into the encrypted cache payload header alongside the PNG bytes, and
    # returned to the frontend at list_index time to backfill the history rail.
    # Passed through via config.json: route -> supervisor -> daemon.submit_task ->
    # here.
    params_snapshot: dict[str, Any] = field(default_factory=dict)
    # 'single' | 'xy'; used by the frontend history rail for grouping, derived
    # from params_snapshot.mode
    mode: str = "single"
    started_at: float = field(default_factory=time.time)


class InferenceDaemon:
    """Server-side proxy for the test-generation daemon. Thread-safe.

    Usage pattern (singleton):
        d = InferenceDaemon()
        d.start()
        d.submit_task(task_id=42, config={...}, on_event=cb)
        # ...wait for cb to receive the done event
        d.stop()

    Event dicts received by `on_event` look like:
        {"kind": "started", "task_id": 42}
        {"kind": "image_done", "task_id": 42, "filename": "gen_0000_p0_c0_s42.png",
                                "path": "/tmp/anima_gen_42/..."}
        {"kind": "done", "task_id": 42}
        {"kind": "error", "task_id": 42, "message": "..."}
    """

    READY_TIMEOUT = 30.0  # max wait for the ready signal after the child process starts importing
    UNLOAD_TIMEOUT = 60.0  # max wait for the unloaded event after requesting unload

    def __init__(self, *, script_path: Optional[Path] = None) -> None:
        self._script = script_path or _DAEMON_SCRIPT
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._state: str = STATE_STOPPED
        self._model_loaded: bool = False
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._active: Optional[_ActiveTask] = None
        self._req_seq = 0
        # Global listeners (for daemon state changes: loaded / unloaded / process crash)
        self._global_listeners: list[EventCallback] = []
        # Daemon stderr ring buffer + incremental listeners (used by the UI drawer,
        # persists across multiple start/stop cycles)
        self._log_lock = threading.Lock()
        self._log_buffer: collections.deque[dict[str, Any]] = collections.deque(maxlen=2000)
        self._log_seq = 0
        self._log_listeners: list[EventCallback] = []
        # Idle timeout: auto-unload to free VRAM after the daemon has been idle
        # (with the model loaded) for N seconds. 0 = disabled. The supervisor
        # injects this after spawn via sync_idle_timeout_from_secrets(); the
        # router also re-syncs it once after PUT /api/secrets.
        self._idle_timeout_seconds: float = 0.0
        self._idle_timer: Optional[threading.Timer] = None
        # Task timeout fallback (based on user reports of a stuck generate
        # requiring a full machine restart): if a task hasn't finished N seconds
        # after starting, hard-kill the daemon process (protocol-level cancel is
        # useless once it's truly stuck). The reader thread hitting EOF ->
        # _handle_proc_exit automatically marks it error and resets state.
        # 0 = disabled (default).
        self._task_timeout_seconds: float = 0.0
        self._task_timer: Optional[threading.Timer] = None

    # ---------------------------------------------------------------- State
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def is_busy(self) -> bool:
        return self.state == STATE_BUSY

    @property
    def is_alive(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def is_model_loaded(self) -> bool:
        """Whether the model is in VRAM (used by the commit 12 GPU-yield check)."""
        with self._lock:
            return self._model_loaded

    def add_global_listener(self, cb: EventCallback) -> None:
        with self._lock:
            self._global_listeners.append(cb)

    # --------------------------------------------------------------- idle auto-unload
    def set_idle_timeout_seconds(self, seconds: float) -> None:
        """Set the timeout (seconds) for auto-unloading the daemon when idle. 0 = disabled.

        The timer only runs while the daemon is idle + the model is loaded + the
        process is alive; entering busy / the model being unloaded / the process
        dying all auto-cancel it. Callers don't need to worry about this.
        """
        secs = max(0.0, float(seconds))
        with self._lock:
            if self._idle_timeout_seconds == secs:
                return
            self._idle_timeout_seconds = secs
            self._reschedule_idle_timer_locked()

    def sync_idle_timeout_from_secrets(self) -> None:
        """Read the idle / task timeout settings from secrets.generate and apply them.

        On failure (corrupt file / missing field), falls back to leaving the
        current value unchanged and logs a warning.
        """
        try:
            # Local import to avoid a module-layer cycle between services/inference
            # and infrastructure
            from ...infrastructure import secrets as _secrets
            gen = _secrets.load().generate
            minutes = int(gen.idle_timeout_minutes)
            task_minutes = int(getattr(gen, "task_timeout_minutes", 0) or 0)
        except Exception:
            logger.warning(
                "failed to read timeouts from secrets; keeping current values",
                exc_info=True,
            )
            return
        self.set_idle_timeout_seconds(max(0, minutes) * 60.0)
        with self._lock:
            self._task_timeout_seconds = max(0, task_minutes) * 60.0

    def _reschedule_idle_timer_locked(self) -> None:
        """Reset the idle timer based on current state. **Must be called while
        holding self._lock.**

        Cancels the old timer; starts a new one when timeout>0 + IDLE + model
        loaded + process alive. In every other case (including BUSY / UNLOADING /
        STOPPED / model not loaded), it only cancels without restarting.
        """
        old = self._idle_timer
        if old is not None:
            try:
                old.cancel()
            except Exception:
                pass
            self._idle_timer = None
        if (
            self._idle_timeout_seconds > 0
            and self._state == STATE_IDLE
            and self._model_loaded
            and self._proc is not None
        ):
            timer = threading.Timer(self._idle_timeout_seconds, self._on_idle_timeout)
            timer.daemon = True
            timer.name = "inference-daemon-idle-timer"
            self._idle_timer = timer
            timer.start()

    def _cancel_task_timer_locked(self) -> None:
        """Cancel the task timeout timer. **Must be called while holding self._lock.**"""
        if self._task_timer is not None:
            try:
                self._task_timer.cancel()
            except Exception:
                pass
            self._task_timer = None

    def _on_task_timeout(self, req_id: str) -> None:
        """Task timeout fallback: if this same task is still running, hard-kill the
        daemon process.

        In a truly stuck scenario (a hung page / GPU hang), protocol-level cancel
        is useless -- only a process-level kill works; the reader thread then hits
        EOF -> _handle_proc_exit marks it error and resets state, and the next
        task auto-respawns the daemon. The task might have just finished right as
        this fires -- re-verify by request_id before killing.
        """
        with self._lock:
            active = self._active
            proc = self._proc
            timeout = self._task_timeout_seconds
            if (
                active is None or active.request_id != req_id
                or self._state != STATE_BUSY or proc is None
            ):
                return
        logger.warning(
            "generate task %s exceeded timeout (%.0fs); killing daemon process",
            active.task_id, timeout,
        )
        try:
            proc.kill()
        except Exception:
            logger.exception("task-timeout kill failed")

    def _on_idle_timeout(self) -> None:
        """Idle timer expiry callback: triggers unload if still idle+loaded.

        State may have already changed by the moment this fires (another thread
        just called submit_task / a manual unload); re-check before calling
        request_unload, to avoid a redundant protocol message.
        """
        with self._lock:
            should_unload = (
                self._state == STATE_IDLE
                and self._model_loaded
                and self._proc is not None
            )
            timeout = self._idle_timeout_seconds
        if not should_unload:
            return
        logger.info("daemon idle for %.0fs; auto-unloading model", timeout)
        try:
            self.request_unload()
        except Exception:
            logger.exception("auto unload from idle timer failed")

    # --------------------------------------------------------------- Lifecycle
    def start(self) -> None:
        """Spawn the daemon child process; returns immediately if already running."""
        with self._lock:
            if self._state != STATE_STOPPED:
                return
            self._state = STATE_STARTING

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.setdefault("TRANSFORMERS_VERBOSITY", "error")
        env.setdefault("DIFFUSERS_VERBOSITY", "error")
        # xformers's triton probe would dump a harmless ImportError traceback into
        # the daemon log drawer; this app's xformers path doesn't use triton
        # kernels, so short-circuit it unconditionally.
        _xformers_svc.disable_triton_probe(env)

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        cmd = [sys.executable, str(self._script)]
        logger.info("spawning inference daemon: %s", " ".join(cmd))
        self._append_log(f"$ {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(REPO_ROOT),
                env=env,
                creationflags=creationflags,
                bufsize=1,
                text=True,
                encoding="utf-8",
            )
        except Exception:
            with self._lock:
                self._state = STATE_STOPPED
            logger.exception("failed to spawn daemon")
            raise

        with self._lock:
            self._proc = proc
            # reader thread handles stdout (the protocol)
            self._reader_thread = threading.Thread(
                target=self._read_stdout_loop,
                args=(proc,),
                name="inference-daemon-stdout",
                daemon=True,
            )
            self._reader_thread.start()
            # stderr thread forwards log lines to this process's logger
            self._stderr_thread = threading.Thread(
                target=self._read_stderr_loop,
                args=(proc,),
                name="inference-daemon-stderr",
                daemon=True,
            )
            self._stderr_thread.start()

        # wait for ready
        deadline = time.time() + self.READY_TIMEOUT
        while time.time() < deadline:
            with self._lock:
                if self._state == STATE_IDLE:
                    return
                if self._state == STATE_STOPPED:
                    raise RuntimeError("daemon exited before ready")
            time.sleep(0.05)
        raise TimeoutError(f"daemon not ready in {self.READY_TIMEOUT}s")

    def stop(self, timeout: float = 10.0) -> None:
        """Shut down the daemon child process. Graceful first, then force-kill."""
        with self._lock:
            proc = self._proc
            if proc is None:
                self._state = STATE_STOPPED
                return
        # close stdin -> the daemon's main loop exits on EOF
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("daemon didn't exit in %.1fs, killing", timeout)
            try:
                proc.kill()
                proc.wait(timeout=3.0)
            except Exception:
                pass
        with self._lock:
            self._proc = None
            self._state = STATE_STOPPED
            self._model_loaded = False
            self._active = None
            self._reschedule_idle_timer_locked()

    # ----------------------------------------------------------------- Submission
    def submit_task(
        self,
        *,
        task_id: int,
        config: dict[str, Any],
        output_dir: str,
        on_event: EventCallback,
    ) -> str:
        """Submit a generate task to the daemon. The daemon must be idle.

        Returns request_id. The command is sent synchronously; subsequent events
        are pushed asynchronously via on_event.
        """
        with self._lock:
            if self._state != STATE_IDLE:
                raise RuntimeError(
                    f"daemon not ready to accept task (state={self._state})"
                )
            self._req_seq += 1
            req_id = f"task-{task_id}-{self._req_seq}"
            save_to_disk = bool(config.get("save_test_images_at_dispatch", False))
            snapshot = config.get("_anima_params_snapshot_") or {}
            if not isinstance(snapshot, dict):
                snapshot = {}
            mode = str(snapshot.get("mode") or "single")
            if mode not in ("single", "xy"):
                mode = "single"
            self._active = _ActiveTask(
                task_id=task_id, request_id=req_id, on_event=on_event,
                save_to_disk=save_to_disk,
                params_snapshot=snapshot,
                mode=mode,
            )
            self._state = STATE_BUSY
            self._reschedule_idle_timer_locked()
            # task timeout fallback timer (0 = disabled)
            self._cancel_task_timer_locked()
            if self._task_timeout_seconds > 0:
                timer = threading.Timer(
                    self._task_timeout_seconds, self._on_task_timeout, args=[req_id],
                )
                timer.daemon = True
                timer.name = "inference-daemon-task-timer"
                self._task_timer = timer
                timer.start()
            assert self._proc is not None and self._proc.stdin is not None
            stdin = self._proc.stdin

        # snapshot is a server-internal protocol field; not passed to the daemon
        # child process (to avoid the downstream config schema validation
        # rejecting an unknown field; the underscore prefix already signals
        # "server-only").
        if "_anima_params_snapshot_" in config:
            config = {k: v for k, v in config.items() if k != "_anima_params_snapshot_"}
        msg = {
            "id": req_id,
            "action": "generate",
            "task_id": task_id,
            "config": config,
            "output_dir": output_dir,
        }
        try:
            stdin.write(json.dumps(msg) + "\n")
            stdin.flush()
        except Exception as e:
            logger.exception("failed to send task to daemon")
            with self._lock:
                self._state = STATE_IDLE
                self._active = None
                self._reschedule_idle_timer_locked()
            raise RuntimeError(f"daemon write failed: {e}") from e
        return req_id

    def cancel_active_task(self, task_id: int) -> bool:
        """Request cancellation of the current generate task; the daemon stays
        resident and the cached model isn't unloaded."""
        with self._lock:
            active = self._active
            if self._state != STATE_BUSY or active is None or active.task_id != task_id:
                return False
            assert self._proc is not None and self._proc.stdin is not None
            stdin = self._proc.stdin
            req_id = active.request_id

        try:
            stdin.write(json.dumps({
                "id": f"cancel-{task_id}",
                "action": "cancel",
                "target_id": req_id,
            }) + "\n")
            stdin.flush()
        except Exception:
            logger.exception("failed to send cancel")
            return False
        return True

    def request_unload(self) -> None:
        """Notify the daemon to unload the model (freeing VRAM). Once done, the
        daemon pushes an unloaded event.

        Not exposed to the frontend as of commit 9; reserved for commit 12's GPU
        yielding / commit 13's manual unload.
        """
        with self._lock:
            if self._state == STATE_STOPPED:
                return
            if self._state == STATE_BUSY:
                logger.warning("unload requested while busy; ignored")
                return
            assert self._proc is not None and self._proc.stdin is not None
            stdin = self._proc.stdin
            self._state = STATE_UNLOADING
            self._reschedule_idle_timer_locked()
        try:
            stdin.write(json.dumps({"id": "_unload", "action": "unload"}) + "\n")
            stdin.flush()
        except Exception:
            logger.exception("failed to send unload")

    # ----------------------------------------------------------- Internal readers
    def _read_stdout_loop(self, proc: subprocess.Popen) -> None:
        """Read daemon stdout lines -> parse JSON -> dispatch."""
        assert proc.stdout is not None
        try:
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("daemon stdout non-JSON: %r", line[:200])
                    continue
                self._handle_event(msg)
        except Exception:
            logger.exception("daemon stdout reader crashed")
        finally:
            self._handle_proc_exit(proc)

    def _read_stderr_loop(self, proc: subprocess.Popen) -> None:
        """daemon stderr -> ring buffer + log listeners.

        Not printed to the terminal -- viewed through the UI drawer instead
        (/api/generate/daemon/logs fetches history, daemon_log_line SSE pushes
        deltas). Keeps the terminal quiet; open the drawer when you need it.

        B-4.5: if the reader crashes, the daemon keeps running but its stderr
        stops being consumed -> the UI drawer stays empty forever and daemon
        OOM / model-loading errors become completely invisible. Fix: auto-restart
        once after a crash; if the restart also blows up, mark STOPPED and emit a
        warning event. proc still alive + reader dead is the worst kind of
        observability hole.
        """
        assert proc.stderr is not None
        attempt = 0
        while attempt < 2 and proc.poll() is None:
            attempt += 1
            try:
                for raw_line in proc.stderr:
                    line = raw_line.rstrip()
                    if line:
                        self._append_log(line)
                # normal EOF (proc exited, stderr closed) -- exit the loop
                return
            except Exception:
                logger.exception(
                    "daemon stderr reader crashed (attempt %d/2)", attempt
                )
                if attempt < 2 and proc.poll() is None:
                    # brief backoff, then restart this loop
                    time.sleep(0.5)
                    continue
        # both attempts crashed and proc is still alive -> the daemon is now unobservable
        if proc.poll() is None:
            logger.error(
                "daemon stderr reader gave up after 2 attempts; daemon (pid=%d) "
                "is still running but its stderr is unmonitored",
                proc.pid,
            )
            for cb in list(self._log_listeners):
                try:
                    cb({"ts": time.time(), "seq": -1,
                        "line": "[stderr reader stopped -- daemon log no longer captured]"})
                except Exception:
                    logger.exception("daemon log listener failed during stderr-down emit")

    # ----------------------------------------------------------- log buffer
    def _append_log(self, line: str) -> None:
        """Receive one line of daemon stderr -> ring buffer + push to listeners
        (thread-safe)."""
        entry = {"ts": time.time(), "line": line}
        with self._log_lock:
            self._log_buffer.append(entry)
            seq = self._log_seq
            self._log_seq += 1
            listeners = list(self._log_listeners)
        entry_out = {**entry, "seq": seq}
        for cb in listeners:
            try:
                cb(entry_out)
            except Exception:
                logger.exception("daemon log listener failed")

    def read_logs(self, since_seq: int = 0, limit: int = 2000) -> dict[str, Any]:
        """Return ring buffer history. When since_seq>0, only returns lines newer
        than that seq (delta)."""
        with self._log_lock:
            # entries stored in the buffer don't carry a seq; derive it by working
            # backward from buffer end = _log_seq - 1
            total = self._log_seq
            start_seq = max(0, total - len(self._log_buffer))
            entries = []
            for i, item in enumerate(self._log_buffer):
                s = start_seq + i
                if s < since_seq:
                    continue
                entries.append({**item, "seq": s})
        if limit and len(entries) > limit:
            entries = entries[-limit:]
        return {"entries": entries, "next_seq": total}

    def add_log_listener(self, cb: EventCallback) -> None:
        """Register a daemon log delta listener; cb(entry) receives {ts, line, seq}."""
        with self._log_lock:
            self._log_listeners.append(cb)

    def _handle_event(self, msg: dict[str, Any]) -> None:
        """Dispatch a protocol message. Task events route to _active.on_event;
        global events go to the listeners."""
        kind = msg.get("kind")
        msg_id = msg.get("id")

        if msg_id == "_evt":
            # a daemon global state event
            with self._lock:
                if kind == "ready":
                    self._state = STATE_IDLE
                    self._model_loaded = False
                elif kind == "loaded":
                    self._model_loaded = True  # state stays IDLE
                elif kind == "unloaded":
                    self._state = STATE_IDLE
                    self._model_loaded = False
                # `loaded` entering idle+loaded -> start the idle timer; `unloaded`
                # (model gone) -> cancel it
                if kind in ("ready", "loaded", "unloaded"):
                    self._reschedule_idle_timer_locked()
            for cb in list(self._global_listeners):
                try:
                    cb(msg)
                except Exception:
                    logger.exception("global listener failed")
            return

        # task events
        with self._lock:
            active = self._active
        if active is None or active.request_id != msg_id:
            logger.warning("event for unknown request: %s", msg_id)
            return

        # commit 10: image_done carries base64 PNG -> goes into the cache, a
        # slimmed-down version (without b64) is forwarded
        # commit 14: preview_step carries base64 JPEG -> passed straight through
        #   to the callback (not cached; the frontend's SSE displays it
        #   immediately as <img src="data:...">  as the current step's preview;
        #   done/the final image replaces it)
        # Decision #14: image_done gets a `delivery: 'disk' | 'cache'` sub-field
        # so the frontend knows whether to POST /api/generate/save to persist it
        # to disk, or just add a CacheEntry directly (cache). Both still route
        # through the cache as a staging point (in persistent mode, the cache is
        # temporary storage before the disk write; once the frontend's disk write
        # succeeds, the user can manually clear the cache entry or let LRU evict
        # it naturally).
        forward_msg = msg
        if kind == "image_done" and "image_b64" in msg:
            filename = msg.get("filename") or ""
            xy_info = msg.get("xy") if isinstance(msg.get("xy"), dict) else None
            try:
                data = base64.b64decode(msg["image_b64"])
                generate_cache.cache_image(
                    active.task_id, filename, data,
                    snapshot=active.params_snapshot,
                    mode=active.mode,
                    xy_info=xy_info,
                )
            except Exception:
                logger.exception("cache_image failed for %s", filename)
            forward_msg = {k: v for k, v in msg.items() if k != "image_b64"}
            forward_msg["delivery"] = "disk" if active.save_to_disk else "cache"

        # For done/error/canceled, flip state before invoking the callback -- so
        # that if the callback queries is_busy/state, it sees the correct IDLE
        # state (commit 13's daemon_state_changed depends on this ordering)
        if kind in ("done", "error", "canceled"):
            with self._lock:
                self._active = None
                self._state = STATE_IDLE
                self._cancel_task_timer_locked()
                # task finished, back to idle; if the model is still loaded, restart
                # the idle countdown
                self._reschedule_idle_timer_locked()

        try:
            active.on_event({**forward_msg, "task_id": active.task_id})
        except Exception:
            logger.exception("task on_event handler failed")

    def _handle_proc_exit(self, proc: subprocess.Popen) -> None:
        """Handle child process exit: mark STOPPED + push error to the active task
        + notify listeners."""
        rc = proc.wait()
        logger.warning("inference daemon exited rc=%d", rc)
        with self._lock:
            self._proc = None
            prev_state = self._state
            self._state = STATE_STOPPED
            self._model_loaded = False
            active = self._active
            self._active = None
            listeners = list(self._global_listeners)
            self._cancel_task_timer_locked()
            self._reschedule_idle_timer_locked()

        if active is not None and prev_state != STATE_UNLOADING:
            try:
                active.on_event({
                    "kind": "error",
                    "task_id": active.task_id,
                    "message": f"daemon exited unexpectedly (rc={rc})",
                })
            except Exception:
                logger.exception("error handler failed")

        for cb in listeners:
            try:
                cb({"id": "_evt", "kind": "stopped", "rc": rc})
            except Exception:
                logger.exception("listener failed on proc exit")


# Singleton handle; initialized on server startup (lazy spawn)
_INSTANCE: Optional[InferenceDaemon] = None
_INSTANCE_LOCK = threading.Lock()


def get_daemon() -> InferenceDaemon:
    """Return the singleton daemon instance (lazily constructed)."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = InferenceDaemon()
        return _INSTANCE


def reset_daemon_for_test() -> None:
    """For tests: clear the singleton so the next test gets a clean instance."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            try:
                _INSTANCE.stop(timeout=3.0)
            except Exception:
                pass
        _INSTANCE = None
