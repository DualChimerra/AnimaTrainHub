"""Resource tier model (0.17 R-1; design in docs/design/queue-resource-model-0.17.md §3).

The queue's primary goal is **VRAM concurrency admission**: while training (or any GPU-
exclusive work) is running, what else can share the GPU at the same time. Tier is a static
property of the work item, not of an ownership table:

- ``exclusive``: base-model-scale VRAM (several GB), at most 1 system-wide at a time.
  train / reg_ai / generate / eval_samples (eval sampling shares generate's base-model
  stack, runs as an independent subprocess). The daemon's resident model is treated as
  holding one revocable exclusive lease.
- ``light``: small models (tens to hundreds of MB). Always admitted when no exclusive job
  is running; when one is running, admitted per the
  `secrets.queue.light_tasks_during_train` toggle (on by default).
- ``io``: doesn't touch the GPU, always admitted (subject only to queue hold constraints).

New task types / job kinds must declare their tier explicitly here; an undeclared kind is
conservatively treated as ``exclusive`` (never runs alongside training -- better slow than
broken).
"""
from __future__ import annotations

RESOURCE_EXCLUSIVE = "exclusive"
RESOURCE_LIGHT = "light"
RESOURCE_IO = "io"

# tasks table task_type -> tier (all three current types are exclusive).
TASK_TYPE_RESOURCE_CLASS: dict[str, str] = {
    "train": RESOURCE_EXCLUSIVE,
    "reg_ai": RESOURCE_EXCLUSIVE,
    "generate": RESOURCE_EXCLUSIVE,
}

# project_jobs kind -> tier.
JOB_KIND_RESOURCE_CLASS: dict[str, str] = {
    "download": RESOURCE_IO,
    "preprocess": RESOURCE_LIGHT,   # spandrel upscaling, small model (D-R1)
    "tag": RESOURCE_LIGHT,          # WD14 / CLTagger ONNX
    "reg_build": RESOURCE_LIGHT,
    "eval_samples": RESOURCE_EXCLUSIVE,  # shares generate's base-model stack (D-R2)
    "eval_clip": RESOURCE_LIGHT,
    "eval_dino": RESOURCE_LIGHT,
    "eval_tag": RESOURCE_LIGHT,
    "eval_ccip": RESOURCE_LIGHT,
    # This fork: unpacking large uploaded zips (pure disk IO, can run alongside training)
    "upload": RESOURCE_IO,
}


def job_resource_class(kind: str) -> str:
    """job kind -> tier; an unknown kind conservatively falls back to exclusive (never
    runs alongside training)."""
    return JOB_KIND_RESOURCE_CLASS.get(kind, RESOURCE_EXCLUSIVE)


def task_resource_class(task_type: str | None) -> str:
    """task_type -> tier; old rows with NULL fall back to train (=exclusive)."""
    return TASK_TYPE_RESOURCE_CLASS.get(task_type or "train", RESOURCE_EXCLUSIVE)
