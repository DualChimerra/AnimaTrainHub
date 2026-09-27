"""Training monitor state writer (PP6.1 rework).

History: this used to be a dual-track HTTP server + JSON file setup. The Studio
frontend has its own monitor page now, so the HTTP server was useless and has been
removed. This file's only job now is writing training progress (loss / lr / samples)
to a JSON file, whose path is set by `set_state_file(path)`.

API:
- `set_state_file(path)` -- set the write path (called once at app startup); if unset, save_state is a silent no-op
- `update_monitor(...)` -- called from the training loop, updates in-memory state and saves to disk
- `restore_monitor_state(...)` -- restores historical curves on checkpoint resume
- `get_state()` -- read the current state (a copy, so external code can't mutate it)
- `_downsample_uniform(points, n)` -- utility: uniform downsampling for frontend display

State structure: losses / lr_history / optimizer_metrics_history / samples / epoch /
total_epochs / step / total_steps / speed / start_time / config (total_epochs was added
later during PP6.x; the frontend falls back to 0 when it's missing from old state).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


# Global state (in-memory)
MONITOR_STATE: dict[str, Any] = {
    "losses": [],
    "lr_history": [],
    "optimizer_metrics_history": [],
    "epoch": 0,
    "total_epochs": 0,
    "step": 0,
    "total_steps": 0,
    "speed": 0.0,
    "samples": [],
    "start_time": None,
    "config": {},
}

# File output path; None = don't write to disk (save_state is a silent no-op)
_state_file: Optional[Path] = None


def set_state_file(path: Optional[Path | str]) -> None:
    """Configure the state JSON output path. None means don't write to disk.

    Ensures the parent directory exists; if a state file already exists at the same
    path it's left alone (checkpoint resume loads history via
    `restore_monitor_state`, this function doesn't read it).
    """
    global _state_file
    if path is None:
        _state_file = None
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _state_file = p


def reset_monitor() -> None:
    """Clear the in-memory MONITOR_STATE. Must be called whenever the daemon reuses
    a process across tasks, otherwise the previous task's samples/step/loss would
    leak into the next task."""
    MONITOR_STATE.update({
        "losses": [], "lr_history": [], "optimizer_metrics_history": [],
        "samples": [],
        "epoch": 0, "total_epochs": 0,
        "step": 0, "total_steps": 0,
        "speed": 0.0,
        "start_time": None,
        "config": {},
    })


def save_state() -> None:
    """Write the current MONITOR_STATE to _state_file (if configured). Failures are silently swallowed."""
    if _state_file is None:
        return
    try:
        with open(_state_file, "w", encoding="utf-8") as f:
            json.dump(MONITOR_STATE, f)
    except Exception:
        pass


def update_monitor(
    loss=None, lr=None, epoch=None, total_epochs=None, step=None,
    total_steps=None, speed=None, sample_path=None, config=None,
    xy=None, optimizer_metrics=None,
):
    """Update the monitor state. Updates step/epoch and other metadata first, then appends loss/lr points.

    `xy`: an optional dict {xi, yi, xv, yv}, used only for the generate XY matrix
    task. The frontend's PreviewXYGrid uses it to place cells."""
    if epoch is not None:
        MONITOR_STATE["epoch"] = epoch
    if total_epochs is not None:
        MONITOR_STATE["total_epochs"] = total_epochs
    if step is not None:
        MONITOR_STATE["step"] = step
    if total_steps is not None:
        MONITOR_STATE["total_steps"] = total_steps
    if speed is not None:
        MONITOR_STATE["speed"] = speed

    if loss is not None:
        MONITOR_STATE["losses"].append(
            {"step": MONITOR_STATE["step"], "loss": loss, "time": time.time()}
        )
        if len(MONITOR_STATE["losses"]) > 50000:
            MONITOR_STATE["losses"] = MONITOR_STATE["losses"][-50000:]

    if lr is not None:
        MONITOR_STATE["lr_history"].append(
            {"step": MONITOR_STATE["step"], "lr": lr}
        )
        if len(MONITOR_STATE["lr_history"]) > 50000:
            MONITOR_STATE["lr_history"] = MONITOR_STATE["lr_history"][-50000:]

    if optimizer_metrics is not None:
        point = {"step": MONITOR_STATE["step"]}
        for key, value in dict(optimizer_metrics).items():
            try:
                point[key] = float(value)
            except (TypeError, ValueError):
                continue
        MONITOR_STATE["optimizer_metrics_history"].append(point)
        if len(MONITOR_STATE["optimizer_metrics_history"]) > 50000:
            MONITOR_STATE["optimizer_metrics_history"] = MONITOR_STATE["optimizer_metrics_history"][-50000:]

    if sample_path is not None:
        sample = {
            "path": str(sample_path),
            "step": MONITOR_STATE["step"],
            "time": time.time(),
        }
        if xy is not None:
            sample["xy"] = xy
        MONITOR_STATE["samples"].append(sample)
        if len(MONITOR_STATE["samples"]) > 50:
            MONITOR_STATE["samples"] = MONITOR_STATE["samples"][-50:]

    if config is not None:
        MONITOR_STATE["config"] = config

    if MONITOR_STATE["start_time"] is None:
        MONITOR_STATE["start_time"] = time.time()

    save_state()


def get_state() -> dict[str, Any]:
    """Read the current state (a shallow copy; the list contents are still shared, callers must not mutate in place)."""
    return MONITOR_STATE.copy()


def restore_monitor_state(
    losses=None, lr_history=None, epoch=None, total_epochs=None, step=None,
    total_steps=None, start_time=None, config=None, optimizer_metrics_history=None,
):
    """Checkpoint resume: load the historical curves from the save file back into in-memory state, then persist."""
    if losses is not None:
        MONITOR_STATE["losses"] = losses
    if lr_history is not None:
        MONITOR_STATE["lr_history"] = lr_history
    if optimizer_metrics_history is not None:
        MONITOR_STATE["optimizer_metrics_history"] = optimizer_metrics_history
    if epoch is not None:
        MONITOR_STATE["epoch"] = epoch
    if total_epochs is not None:
        MONITOR_STATE["total_epochs"] = total_epochs
    if step is not None:
        MONITOR_STATE["step"] = step
    if total_steps is not None:
        MONITOR_STATE["total_steps"] = total_steps
    if start_time is not None:
        MONITOR_STATE["start_time"] = start_time
    if config is not None:
        MONITOR_STATE["config"] = config
    save_state()


def _downsample_uniform(points: list[Any], target_points: int) -> list[Any]:
    """Uniformly downsample to target_points (keeping first and last), suitable for displaying long loss/lr sequences."""
    if not isinstance(target_points, int) or target_points <= 0:
        return points
    n = len(points)
    if n <= target_points:
        return points
    if target_points == 1:
        return [points[-1]]
    step = (n - 1) / (target_points - 1)
    out = []
    for i in range(target_points):
        idx = round(i * step)
        out.append(points[idx])
    return out


def reset_state() -> None:
    """For testing: reset in-memory state back to its initial values."""
    MONITOR_STATE.clear()
    MONITOR_STATE.update({
        "losses": [],
        "lr_history": [],
        "optimizer_metrics_history": [],
        "epoch": 0,
        "total_epochs": 0,
        "step": 0,
        "total_steps": 0,
        "speed": 0.0,
        "samples": [],
        "start_time": None,
        "config": {},
    })
