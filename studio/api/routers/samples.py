"""Sample image proxy (extracted from server.py in PR-6 commit 1).

2 routes:
    GET /samples/{filename}        with task_id, resolves against multiple candidate dirs
                                   derived from monitor_state_path; without it, falls back
                                   to the global OUTPUT_DIR/samples/; optional ?w=N thumbnail
    GET /api/queue/{task_id}/samples  a task's sample image manifest (used by the queue
                                      page's inline sample strip)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse

from .. import errors as _errors
from ..responses import _thumb_response
from ... import db
from ...domain.errors import NotFoundError
from ...paths import OUTPUT_DIR, task_samples_dir
from ...services.dataset.scan import IMAGE_EXTS

router = APIRouter()
logger = logging.getLogger(__name__)

# Cap on the queue page's per-row inline sample strip. The monitor state caps itself at
# 50 (train_monitor.py); here, scanning disk can see the full image history, so this cap of
# the same order of magnitude guards against an extra-long run dumping hundreds at once.
_MAX_LIST = 200

_EPOCH_RE = re.compile(r"^epoch_(\d+)", re.IGNORECASE)
_STEP_RE = re.compile(r"^step_(\d+)", re.IGNORECASE)


def _sample_dirs(monitor_state_path: str, task_id: int) -> list[Path]:
    """Candidate sample directories for a task, ordered new -> old layout.

    - **New (task-scoped)** `studio_data/tasks/<task_id>/samples/`
    - `samples/` alongside `monitor_state.json` (compatibility with pre-PP6.1 v0.5.0+ tasks;
      when the state file lives at versions/<v>/monitor/task_<id>/, samples are there too)
    - `output/samples/` alongside `monitor_state.json` (pre-PP6.1 legacy tasks;
      sample_dir = output_dir/samples, and output_dir is usually versions/{label}/output)
    - `output/<any subdirectory>/samples/` (fallback in case anima_train used a different
      output name)
    """
    monitor_dir = Path(monitor_state_path).parent
    dirs = [
        task_samples_dir(task_id),
        monitor_dir / "samples",
        monitor_dir / "output" / "samples",
    ]
    output_root = monitor_dir / "output"
    if output_root.is_dir():
        for sub in sorted(output_root.iterdir()):
            if sub.is_dir():
                dirs.append(sub / "samples")
    return dirs


def _monitor_state_path(task_id: int) -> Optional[str]:
    with db.connection_for() as conn:
        row = conn.execute(
            "SELECT monitor_state_path FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    if not row or not row["monitor_state_path"]:
        return None
    return str(row["monitor_state_path"])


def _marks(filename: str) -> tuple[Optional[int], Optional[int]]:
    """Parse (epoch, step) from a filename. `epoch_3_xxx.png` / `step_1200_xxx.png`."""
    ep = _EPOCH_RE.match(filename)
    st = _STEP_RE.match(filename)
    return (
        int(ep.group(1)) if ep else None,
        int(st.group(1)) if st else None,
    )


@router.get("/samples/{filename}")
def get_sample(
    filename: str,
    task_id: Optional[int] = None,
    w: Optional[int] = None,
) -> FileResponse:
    """Sample image proxy.

    With `?task_id=N` -> looks through `_sample_dirs()`'s candidate directories one by one.
    Without task_id -> falls back to the global OUTPUT_DIR/samples/ (compatibility with old
    training runs launched directly from the CLI).

    With `?w=N` -> goes through thumb_cache to generate an N px thumbnail (used by the
    monitor page's thumbnail strip); without it -> returns the original image. Both paths go
    through _thumb_response's weak etag + no-cache, so the browser can just hit a 304,
    avoiding the "a failed response during a restart window gets cached forever" problem.
    """
    _errors._validate_component_or_400(filename)

    resolved: Optional[Path] = None
    if task_id is not None:
        state_path = _monitor_state_path(task_id)
        if not state_path:
            raise NotFoundError("Sample image not found", code="sample.not_found")
        candidates = [d / filename for d in _sample_dirs(state_path, task_id)]
        for p in candidates:
            if p.exists():
                resolved = p
                break
        if resolved is None:
            logger.info(
                "sample 404: task_id=%s file=%s tried=%s",
                task_id, filename, [str(p) for p in candidates],
            )
            raise NotFoundError("Sample image not found", code="sample.not_found")
    else:
        path = OUTPUT_DIR / "samples" / filename
        if not path.exists():
            raise NotFoundError("Sample image not found", code="sample.not_found")
        resolved = path

    # With w -> thumbnail; w<=0 or missing -> original image. Reuses thumb_cache, which
    # writes .jpg to disk. Task-scoped sample image content is immutable (filenames carry
    # epoch/step, and retraining gets a new task_id), so the URL
    # (`/samples/{file}?task_id=N&w=W`) is a stable, unique key -> long immutable cache, and
    # reopening the browser skips the round trip entirely (saves one 304 RTT per image in
    # cloud-tunnel scenarios). The no-task_id fallback (CLI OUTPUT_DIR/samples, where
    # same-named files may be overwritten) keeps no-cache revalidation.
    immutable = task_id is not None
    size = w if (w is not None and w > 0) else 0
    return _thumb_response(resolved, size, immutable=immutable)


@router.get("/api/queue/{task_id}/samples")
def list_task_samples(task_id: int) -> dict[str, Any]:
    """A task's sample image manifest, sorted by mtime ascending (training timeline).

    Used by the queue page's per-row inline sample strip + lightbox. Deliberately **scans
    disk** instead of reading monitor_state.json:
    - works for finished tasks too (the samples array is still in the state, but reading the
      whole state file just to get 50 paths is too heavy -- the losses array for a 10k-step
      run can be several MB);
    - scanning disk sees the full image history, not limited by the monitor's cap of 50.

    Task doesn't exist / no monitor_state_path / directory not created yet -> `{"items": []}`,
    no error (the queue has a bunch of non-training tasks, the frontend requests per row, and
    a 404 would just spam red errors in the console).
    """
    state_path = _monitor_state_path(task_id)
    if not state_path:
        return {"items": [], "total": 0}

    seen: set[str] = set()
    found: list[tuple[float, str, int]] = []  # (mtime, filename, size)
    for d in _sample_dirs(state_path, task_id):
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            if f.name in seen:
                continue  # same-name file in old/new layout: first-matched dir wins (same order as get_sample)
            seen.add(f.name)
            try:
                stat = f.stat()
            except OSError:
                continue
            found.append((stat.st_mtime, f.name, stat.st_size))

    found.sort(key=lambda r: (r[0], r[1]))
    total = len(found)
    # Over the cap, keep the **most recent** batch (the user wants to see what training looks like most recently).
    if total > _MAX_LIST:
        found = found[-_MAX_LIST:]

    items = []
    for mtime, name, size in found:
        epoch, step = _marks(name)
        items.append({
            "filename": name,
            "mtime": mtime,
            "size": size,
            "epoch": epoch,
            "step": step,
        })
    return {"items": items, "total": total}
