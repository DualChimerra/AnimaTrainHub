"""Default cmd builder + worker EVENT protocol constants (extracted from supervisor.py in PR-4).

`Supervisor.__init__` accepts `cmd_builder` / `job_cmd_builder` injection
parameters for easy test replacement; this module implements the default
versions that supervisor uses, which run the real runtime/anima_train.py /
workers modules.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from ..paths import REPO_ROOT, task_monitor_state_path

# Since R-1, scheduling admission no longer uses this set (it now goes
# through resources.py's tier model: exclusive / light / io, see
# docs/design/queue-resource-model-0.17.md). Kept here as the derived fact of
# "GPU-hungry job kinds" (= non-io tier), for documentation-style assertions.
from .resources import JOB_KIND_RESOURCE_CLASS as _JOB_CLASS, RESOURCE_IO as _IO

GPU_BOUND_JOB_KINDS: frozenset[str] = frozenset(
    k for k, c in _JOB_CLASS.items() if c != _IO
)

# The structured event marker from worker to supervisor. A worker writes
#   __EVENT__:my_event_type:{"foo":1,"bar":"x"}
# to stdout; supervisor recognizes it in _on_line and publishes it as a typed
# SSE event (job_id / project_id injected automatically), skipping job_log.
# Lighter weight than building a dedicated IPC channel.
_EVENT_MARKER = "__EVENT__:"


EventCallback = Callable[[dict[str, Any]], None]
CmdBuilder = Callable[[dict[str, Any], Path], list[str]]
JobCmdBuilder = Callable[[dict[str, Any]], list[str]]


def _default_cmd_builder(task: dict[str, Any], config_path: Path) -> list[str]:
    """Route to the corresponding script based on task_type.

    train (default / legacy task): runtime/anima_train.py
    reg_ai: runtime/anima_reg_ai.py (prior generation)
    generate: goes through inference_daemon, **not** this cmd_builder --
        supervisor dispatches it directly to the daemon in
        _dispatch_exclusive_tasks. The fallback to anima_generate.py here
        only exists so a future test that injects a cmd_builder doesn't hit
        a KeyError -- this path is never actually taken at runtime
        (_next_pending_task_in only picks train/reg_ai in dispatch_train).
    """
    task_type = task.get("task_type") or "train"
    if task_type == "reg_ai":
        script = REPO_ROOT / "runtime" / "anima_reg_ai.py"
    elif task_type == "generate":
        script = REPO_ROOT / "runtime" / "anima_generate.py"  # fallback, never hit on the normal path
    else:
        script = REPO_ROOT / "runtime" / "anima_train.py"
    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(config_path),
    ]
    msp = task.get("monitor_state_path")
    if msp:
        cmd.extend(["--monitor-state-file", str(msp)])
    # ADR 0006 PR-3: reviving a paused task -> inject --resume-state so
    # anima_train's resume_phase loads the state; the adjacent
    # .config.json snapshot is auto-detected by bootstrap_phase, which
    # freezes the args (ADR §5.7).
    paused_state = task.get("paused_state_path")
    if paused_state:
        cmd.extend(["--resume-state", str(paused_state)])
    return cmd


def _resolve_monitor_state_path(task: dict[str, Any]) -> Path:
    """决定 task 的 monitor_state.json 落盘路径。

    一律落 `studio_data/tasks/<id>/monitor/state.json`，跟 version 解耦：
    - 删 version 不再带走 task 历史（loss 曲线 / 采样图）
    - 同一 version 多次跑的 task 各自独立档案，用户可以拉出来对比

    历史路径仅保留**读**兼容（老 task DB monitor_state_path 列保留旧值，
    读端按值取，新 task 不再写这些路径）：
    - `versions/<label>/monitor/task_<id>/state.json` —— PP6.1（v0.5.0+）
    - `versions/<label>/monitor_state.json` —— pre-PP6.1
    - `studio_data/monitors/task_<id>/state.json` —— 无 version_id 兜底
    """
    return task_monitor_state_path(task["id"])


def _default_job_cmd_builder(job: dict[str, Any]) -> list[str]:
    """默认按 kind 选 worker 模块。"""
    kind = job["kind"]
    return [
        sys.executable,
        "-m",
        f"studio.workers.{kind}_worker",
        "--job-id",
        str(job["id"]),
    ]
