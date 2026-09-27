"""Task-scheduling daemon thread -- split apart in PR-4.

The original 1431-line `studio.supervisor` single file was split into this
subpackage by responsibility:

    slot.py         _Slot dataclass + SLOT_TRAIN/DATA constants
    cmd_builder.py  the default cmd builder + monitor_state_path + worker EVENT protocol constants
    finalizer.py    task terminal state -> version.status mapping (ADR-0007 §11.3-B)
    process.py      _kill_process_tree (cross-platform process-tree kill)
    core.py         the Supervisor main class (kept as one class, not split further -- state is tightly coupled, see the PR-4 decision log)

This `__init__.py` is a compat shim: it re-exports every public name at the
package top level, so old import paths like
`from studio.supervisor import Supervisor / _Slot / _default_cmd_builder
/ _maybe_finalize_version` keep working transparently.

monkeypatch path compat: tests use
`monkeypatch.setattr("studio.supervisor._secrets.load", X)` and
`monkeypatch.setattr("studio.supervisor.subprocess.Popen", X)`, which rely on
the `_secrets` / `subprocess` attributes existing on the `studio.supervisor`
module object. Re-exporting them from core.py below makes the lookup hit the
real module singleton (Python module objects are singletons, so the patch
also affects calls inside core.py).
"""
from __future__ import annotations

from .cmd_builder import (
    _EVENT_MARKER,
    GPU_BOUND_JOB_KINDS,
    CmdBuilder,
    EventCallback,
    JobCmdBuilder,
    _default_cmd_builder,
    _default_job_cmd_builder,
    _resolve_monitor_state_path,
)
from .core import Supervisor, _secrets, subprocess
from .finalizer import _maybe_finalize_version
from .process import _kill_process_tree
from .slot import SLOT_DATA, SLOT_TRAIN, _Slot

__all__ = [
    "Supervisor",
    "_Slot",
    "SLOT_TRAIN",
    "SLOT_DATA",
    "GPU_BOUND_JOB_KINDS",
    "EventCallback",
    "CmdBuilder",
    "JobCmdBuilder",
    "_EVENT_MARKER",
    "_default_cmd_builder",
    "_default_job_cmd_builder",
    "_resolve_monitor_state_path",
    "_maybe_finalize_version",
    "_kill_process_tree",
    # the 2 below are exposed only for monkeypatch path compat; not part of the business API
    "_secrets",
    "subprocess",
]
