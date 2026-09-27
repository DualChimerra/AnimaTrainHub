"""State snapshot helpers for pause / resume (ADR 0006 PR-2).

Three things live here:

  - `build_pause_state_path(state_dir, step)`: builds the pause `.pt` path
  - `write_config_snapshot(path, args, sample_prompts)`: serializes every arg
    training is actually using at the moment of pause into JSON (see ADR §5.7)
  - `emit_event(event_type, payload)`: writes an `__EVENT__:...` line to stdout,
    which the supervisor's `_on_line` recognizes and publishes as an SSE typed event

None of these three import the training pipeline, deliberately kept as an
independent module so the supervisor / spike scripts / tests can all reuse
them, without context.py growing ever more bloated.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


# Aligned with studio/supervisor.py:49's _EVENT_MARKER protocol; changing this
# constant is a cross-process breaking change, so it lives as the sole literal at the top of this module.
EVENT_MARKER = "__EVENT__:"


def build_pause_state_path(state_dir: Path, step: int) -> Path:
    """The pause `.pt` file path (ADR §5.1).

    The `pause_` prefix distinguishes it from PR-1's periodic save
    `training_state_step<N>.pt`; the same-named `.config.json` is the snapshot (see `write_config_snapshot`).
    """
    return state_dir / f"pause_step_{step}.pt"


def build_pause_config_path(state_dir: Path, step: int) -> Path:
    """The pause config snapshot file path (same prefix as the state `.pt`, `.config.json` suffix)."""
    return state_dir / f"pause_step_{step}.config.json"


# ADR 0006 Addendum 1: the overwrite-style single-file path used for the epoch auto backup.
# The name carries no step / epoch N suffix -- it is **overwrite-style**: each new
# epoch overwrites the previous file before writing. Fully independent from the
# user-opt `save_state_every_epochs` output training_state_epoch{N}.pt (a
# multi-file historical archive); all three file kinds coexist under <state_dir>/task_<TID>/.

def build_auto_epoch_state_path(state_dir: Path) -> Path:
    """The auto epoch backup state file path (a single overwrite-style file).

    ADR 0006 Addendum 1 plan Delta: **forcibly** writes one overwrite-style
    state file at the end of every epoch as a pause fallback. The pause signal
    triggers `handle_interrupt`, which only emits + exits without writing to
    disk itself -- resume uses this auto backup instead.
    """
    return state_dir / "auto_epoch_state.pt"


def build_auto_epoch_config_path(state_dir: Path) -> Path:
    """The auto epoch backup config snapshot path (same prefix as the state `.pt`, `.config.json` suffix)."""
    return state_dir / "auto_epoch_state.config.json"


def _jsonify(value: Any) -> Any:
    """Convert objects found in args / sample_prompts into a json.dump-able form.

    Coverage: Path -> str, set -> list, everything else passes through
    unchanged. argparse.Namespace's own vars() output is always primitives +
    Path; anything else falls back to repr().
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonify(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    return repr(value)  # fallback, never raises


def write_config_snapshot(
    path: Path,
    args: Any,  # argparse.Namespace or dict
    sample_prompts: list[str] | None = None,
) -> None:
    """Freeze every parameter training is actually using at the moment of pause into JSON (ADR §5.7).

    Resume strictly builds the new args from the snapshot, fully decoupled from
    any config / preset / yaml changes the user makes afterward. The snapshot
    excludes the wandb run id (already finished) and the monitor's live state
    (already dumped inside the `.pt` file).

    `sample_prompts` comes from `ctx.sample_prompts` (runtime state), not an
    args field, so it's passed separately.
    """
    if hasattr(args, "__dict__"):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        args_dict = {"_args_repr": repr(args)}

    payload = {
        "version": 1,  # leaves a hook for a future schema migration
        "args": {k: _jsonify(v) for k, v in args_dict.items()},
        "sample_prompts": list(sample_prompts) if sample_prompts else [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # ADR 0006 Addendum 2: atomic tmp + os.replace write, same as save_training_state,
    # so a power loss hitting the write window can't leave a half-written json
    # (resume freezes strictly, and a corrupt snapshot just refuses to restore).
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def emit_event(event_type: str, payload: dict[str, Any] | None = None) -> None:
    """Write a `__EVENT__:type:json` protocol line to stdout for the supervisor.

    flush=True is essential -- otherwise the child process's stdout buffer
    holds it back until several KB accumulate, delaying delivery on the pause
    path; if the supervisor's `_on_line` doesn't catch the event in time, it wrongly declares a timeout.
    """
    body = json.dumps(payload or {}, ensure_ascii=False)
    sys.stdout.write(f"{EVENT_MARKER}{event_type}:{body}\n")
    sys.stdout.flush()
