"""Pre-flight estimate for a training run: will it fit, and how long will it take.

Both numbers already existed somewhere in the codebase but never reached the
person about to press "Start training":

* **Memory.** ``training/block_swap_preflight.evaluate`` is a pure function that
  answers "does this ``blocks_to_swap`` fit in the free VRAM / RAM" — but it only
  ran inside the trainer, i.e. after queueing, after the dataset was cached, and
  in the worst case after an OOM. Here we call the very same function with the
  same inputs so the form can answer it up front. Same arithmetic means the panel
  can never disagree with the guard that actually stops the run.

* **Time.** There is no way to know the speed of a run that has not happened, and
  guessing from FLOP counts is how estimates end up off by 3×. Instead we take the
  measured ``it/s`` from this project's most recent training task (the monitor
  writes it to ``tasks/<id>/monitor/state.json``) and multiply by the planned step
  count. That is honest: it is a measurement, it says which run it came from, and
  before the first run it returns ``None`` and the UI shows nothing rather than a
  fabricated number.

Everything here is read-only and best-effort: any missing piece degrades to
``None`` for that one field instead of failing the request.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

def _load_monitor_speed(task_id: int) -> Optional[float]:
    """Read ``it/s`` out of a task's monitor state; None when unavailable."""
    from studio.infrastructure.paths import task_monitor_state_path

    try:
        path = task_monitor_state_path(int(task_id))
        if not path.exists():
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    speed = state.get("speed") if isinstance(state, dict) else None
    try:
        value = float(speed)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def measured_speed(conn, project_id: int) -> Optional[dict[str, Any]]:
    """Most recent usable ``it/s`` measurement for this project.

    Walks finished training tasks newest-first and returns the first one that
    actually recorded a speed. Returns the task id alongside so the UI can say
    where the number came from — an estimate you cannot trace is an estimate you
    cannot sanity-check.
    """
    from studio.infrastructure import db

    try:
        tasks = db.list_tasks(conn)
    except Exception:  # noqa: BLE001 — estimate must never break the page
        return None

    candidates = [
        t for t in tasks
        if int(t.get("project_id") or 0) == int(project_id)
        and str(t.get("type") or "train") == "train"
        and str(t.get("status") or "") in ("done", "failed", "canceled", "paused", "running")
    ]
    candidates.sort(key=lambda t: str(t.get("started_at") or t.get("created_at") or ""), reverse=True)

    for task in candidates:
        speed = _load_monitor_speed(int(task.get("id") or 0))
        if speed is not None:
            return {
                "it_per_s": speed,
                "task_id": int(task.get("id") or 0),
                "task_name": str(task.get("config_name") or task.get("name") or ""),
            }
    return None


def memory_fit(cfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Run the block-swap preflight arithmetic against the current machine.

    Returns ``None`` when the verdict cannot be reached (no CUDA, unreadable
    checkpoint, non-Anima family) — the caller then simply shows nothing, which
    is the correct answer for "we do not know".
    """
    try:
        from training import block_swap_preflight, sysmem
        from training.families import get_family
    except Exception:  # noqa: BLE001 — runtime deps are optional in the API process
        return None

    transformer = str(cfg.get("transformer_path") or "")
    if not transformer or not Path(transformer).exists():
        return None

    try:
        family = get_family(str(cfg.get("model_family") or "anima"))
        if family is None or "block_swap" not in family.spec.capabilities:
            return None
        ratio_fn = getattr(family, "swapped_param_ratio", None)
        blocks_fn = getattr(family, "swappable_blocks", None)
        if ratio_fn is None or blocks_fn is None:
            return None

        # Same call shape as block_swap_preflight.run(): the block count and the
        # swapped-parameter ratio both depend on the checkpoint (Anima 2B has 28
        # layers, 14B has 36), so the path is a required argument.
        total_blocks = int(blocks_fn(checkpoint_path=transformer))
        file_bytes = sysmem._file_bytes([transformer])
        free_vram = sysmem.gpu_free_bytes_global()
        avail_ram = sysmem.available_ram_bytes()
        result = block_swap_preflight.evaluate(
            file_bytes=file_bytes,
            blocks_to_swap=int(cfg.get("blocks_to_swap", 0) or 0),
            total_blocks=total_blocks,
            ratio_fn=lambda n: float(ratio_fn(n, checkpoint_path=transformer)),
            free_vram_bytes=free_vram,
            avail_ram_bytes=avail_ram,
            vram_base_bytes=sysmem._VRAM_BASE_BYTES,
            pinned_limit_bytes=(
                sysmem.pinned_safe_limit(avail_ram) if avail_ram is not None else None
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("memory estimate unavailable: %s", exc)
        return None

    if not result.checked:
        return None
    current = result.current
    return {
        "ok": bool(result.ok),
        "vram_need_bytes": int(current.vram_need) if current else None,
        "ram_need_bytes": int(current.pinned_need) if current else None,
        "free_vram_bytes": free_vram,
        "avail_ram_bytes": avail_ram,
        "recommended_blocks_to_swap": result.recommended,
        "blocks_to_swap": int(cfg.get("blocks_to_swap", 0) or 0),
        "total_blocks": total_blocks,
    }


def estimate(conn, project_id: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Everything the training form wants to show before the run starts."""
    return {
        "speed": measured_speed(conn, project_id),
        "memory": memory_fit(cfg),
    }
