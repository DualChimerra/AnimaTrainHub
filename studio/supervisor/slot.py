"""Supervisor execution slots (extracted from supervisor.py in PR-4).

As of PP10.2.a, changed from a single "_current_*" field set to "list[_Slot]"; 10.2.b
split this into two slots:
  - TRAIN slot: runs only training tasks (db.tasks table)
  - DATA  slot: runs only project_jobs (download / tag / reg_build)
download always runs alongside training; tag / reg_build depend on a settings toggle.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Optional

from ..infrastructure.log_tail import LogTailer, MonitorStatePoller

# Slot name constants
SLOT_TRAIN = "train"
SLOT_DATA = "data"


@dataclass
class _Slot:
    """One execution slot inside the supervisor. Each slot runs at most 1 subprocess."""
    name: str = "main"
    proc: Optional[subprocess.Popen] = None
    kind: Optional[str] = None  # "task" | "job"
    id: Optional[int] = None
    # R-1 resource admission: the kind of the DATA slot's current job (recorded at spawn
    # time, so _exclusive_busy can check "is the running job an exclusive tier
    # (eval_samples)" without a DB query on every tick).
    job_kind: Optional[str] = None
    log_fp: Optional[Any] = None
    tailer: Optional[LogTailer] = None
    state_poller: Optional[MonitorStatePoller] = None
    cancel_pending: bool = False
    # ADR 0006 PR-2 pause/resume backend ----------------------------------
    pause_pending: bool = False
    # `__EVENT__:pause_state` payload -- after handle_interrupt writes the .pt + snapshot,
    # the subprocess emits it over stdout; _on_line catches it and fills these three
    # fields, which _finish_slot then uses to mark the task paused. pause_pending=True but
    # these fields missing -> subprocess exited before it could emit -> treated as a
    # cancel fallback (ADR §4.3 "force cancel, save progress" modal).
    pause_state_path: Optional[str] = None
    pause_config_path: Optional[str] = None
    pause_step: Optional[int] = None
    # ADR §8.1 is_pausable signal -- pausing is only allowed after the resume phase emits
    # `train_loop_started`. The UI side uses this over SSE to unlock the pause button, and
    # the API side rejects premature pause requests as defense-in-depth.
    train_loop_started: bool = False
    # ADR 0006 Addendum 1: these two fields are filled in once the end-of-epoch auto
    # backup completes. The is_pausable upgrade condition requires
    # `last_auto_epoch_state_path is not None` -- the button stays fully hidden before the
    # first epoch finishes, so a user can't pause into a state with nothing to resume from.
    last_auto_epoch_state_path: Optional[str] = None
    last_auto_epoch_config_path: Optional[str] = None
    eval_training_finished_payload: Optional[dict[str, Any]] = None

    @property
    def busy(self) -> bool:
        return self.proc is not None

    def reset(self) -> None:
        self.proc = None
        self.kind = None
        self.id = None
        self.job_kind = None
        self.log_fp = None
        self.tailer = None
        self.state_poller = None
        self.cancel_pending = False
        self.pause_pending = False
        self.pause_state_path = None
        self.pause_config_path = None
        self.pause_step = None
        self.train_loop_started = False
        self.last_auto_epoch_state_path = None
        self.last_auto_epoch_config_path = None
        self.eval_training_finished_payload = None
