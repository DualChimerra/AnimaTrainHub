"""Log file tailer: pushes bytes appended to a log file incrementally to a callback.

Used by the supervisor to track worker subprocess logs, publishing line by line to SSE.

PP6.4: adds MonitorStatePoller -- watches monitor_state.json's mtime and, on change,
publishes the whole state to SSE subscribers (replaces the frontend's 1Hz polling of /api/state).
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

# C++ libraries (onnxruntime being a typical example) sometimes write ANSI-colored logs
# directly to the worker process's fd 2; the frontend's <pre> doesn't parse ANSI, so it renders
# garbage like `[1;31m...`. On Windows it can also stuff in UTF-16-style NUL bytes, making an
# ASCII line look like it has spaces between every character. Strip both uniformly at the tail
# stage so the frontend gets clean text.
_ANSI_CSI_RE = re.compile(r"\x1b\[[\d;?]*[A-Za-z]")


class LogTailer:
    """Polls a log file and sends newly appended bytes to `on_line(line)`, line by line.

    Thread-safe; call start/stop once each; never raises (IO failures retry silently).
    """

    def __init__(
        self,
        path: Path,
        on_line: Callable[[str], None],
        *,
        poll_interval: float = 0.3,
    ) -> None:
        self._path = path
        self._on_line = on_line
        self._poll = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0
        self._buffer = ""

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"log-tail-{self._path.name}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        # Wrap-up: flush the remaining buffer as the last line
        if self._buffer.strip():
            try:
                self._on_line(self._buffer.rstrip("\r\n"))
            finally:
                self._buffer = ""

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._read_chunk()
            except Exception:
                # IO exceptions are swallowed here to avoid stalling the supervisor
                pass
            self._stop.wait(self._poll)
        # One more flush before exiting, to catch output written right at the end
        try:
            self._read_chunk()
        except Exception:
            pass

    def _read_chunk(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, "rb") as f:
            f.seek(self._offset)
            chunk = f.read()
            if not chunk:
                return
            self._offset += len(chunk)
        raw = chunk.decode("utf-8", errors="replace")
        # Strip ANSI CSI escapes + NUL bytes (a side effect of C++ libraries like onnxruntime
        # writing directly to fd 2)
        cleaned = _ANSI_CSI_RE.sub("", raw).replace("\x00", "")
        text = self._buffer + cleaned
        # Split into lines; keep an incomplete trailing segment in the buffer to join next time
        lines = text.split("\n")
        self._buffer = lines.pop()
        for line in lines:
            self._on_line(line.rstrip("\r"))


class MonitorStatePoller:
    """Polls monitor_state.json's mtime and, on change, **builds an incremental delta** to
    push to the callback.

    Protocol (reworked in PR #37): earlier versions pushed the full state every time (the
    losses/lr arrays were retransmitted in full at every step -- a single push for a 2000-step
    training run could be ~200KB). That's O(N^2) waste over a WAN in cloud deployments.

    New design: the poller keeps last_step / last_loss_count / last_lr_count /
    last_optimizer_metrics_count / last_sample_count, and each time only packages "what's new
    since the last publish" into a delta:

        {
          "step": 234, "total_steps": 2000,
          "epoch": 3, "total_epochs": 10,
          "speed": 1.2, "start_time": 1234567890.0,
          "appended_losses": [{step, loss, time}],   # may be an empty array
          "appended_lr":     [{step, lr}],
          "appended_optimizer_metrics": [{step, actual_lr, d, ...}],
          "appended_samples":[{path, step, time, xy?}],
          "config": {...},        # only included when changed (first push / config change)
        }

    Throttle rules (a mix of step + time):
    - poll_interval 0.5s to probe mtime
    - min_publish_interval 1.0s hard floor -- even if training does a step every 100ms, this
      never pushes faster than 1Hz; accumulated loss points get batched into the next push
    - skips the push when "step unchanged + no new samples + config unchanged", to avoid an
      empty delta (e.g. training only updated a derived metric like speed)
    """

    def __init__(
        self,
        path: Path,
        on_delta: Callable[[dict[str, Any]], None],
        *,
        poll_interval: float = 0.5,
        min_publish_interval: float = 1.0,
    ) -> None:
        self._path = path
        self._on_delta = on_delta
        self._poll = poll_interval
        self._min_pub = min_publish_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_mtime: float = 0.0
        self._last_publish_at: float = 0.0

        # Incremental tracking
        self._last_step: int = -1
        self._last_loss_count: int = 0
        self._last_lr_count: int = 0
        self._last_optimizer_metrics_count: int = 0
        self._last_sample_count: int = 0
        self._last_config: dict[str, Any] | None = None

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"monitor-state-{self._path.name}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._check_once()
            except Exception:
                # IO/parse exceptions retry silently, to avoid stalling the supervisor
                pass
            self._stop.wait(self._poll)
        # Read once more before exiting, to catch the final state right at the end; force=True
        # bypasses the throttle
        try:
            self._check_once(force=True)
        except Exception:
            pass

    def _check_once(self, *, force: bool = False) -> None:
        if not self._path.exists():
            return
        mtime = self._path.stat().st_mtime
        if mtime <= self._last_mtime:
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Half-written JSON / temporarily locked -> try again next round
            return
        self._last_mtime = mtime

        # -- throttle: bail out before the min publish interval, check again next round
        # (picking up anything accumulated meanwhile) --
        now = time.time()
        if not force and (now - self._last_publish_at) < self._min_pub:
            return

        # -- compute the delta --
        step = int(data.get("step", 0) or 0)
        losses = data.get("losses") or []
        lr_hist = data.get("lr_history") or []
        optimizer_metrics = data.get("optimizer_metrics_history") or []
        samples = data.get("samples") or []
        config = data.get("config") or {}

        appended_losses = losses[self._last_loss_count:]
        appended_lr = lr_hist[self._last_lr_count:]
        appended_optimizer_metrics = optimizer_metrics[self._last_optimizer_metrics_count:]
        appended_samples = samples[self._last_sample_count:]
        config_changed = config != self._last_config

        has_progress = (
            step != self._last_step
            or appended_losses
            or appended_lr
            or appended_optimizer_metrics
            or appended_samples
            or config_changed
        )
        if not (force or has_progress):
            return

        delta: dict[str, Any] = {
            "step": step,
            "total_steps": int(data.get("total_steps", 0) or 0),
            "epoch": int(data.get("epoch", 0) or 0),
            "total_epochs": int(data.get("total_epochs", 0) or 0),
            "speed": float(data.get("speed", 0.0) or 0.0),
            "start_time": data.get("start_time"),
            "appended_losses": appended_losses,
            "appended_lr": appended_lr,
            "appended_optimizer_metrics": appended_optimizer_metrics,
            "appended_samples": appended_samples,
        }
        if config_changed:
            delta["config"] = config

        # Push + advance the cursor
        self._last_step = step
        self._last_loss_count = len(losses)
        self._last_lr_count = len(lr_hist)
        self._last_optimizer_metrics_count = len(optimizer_metrics)
        self._last_sample_count = len(samples)
        self._last_config = config
        self._last_publish_at = now

        self._on_delta(delta)
