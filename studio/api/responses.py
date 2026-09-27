"""Shared response constants / response factories (extracted from server.py starting with PR-5)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.responses import FileResponse

from ..services.dataset import thumb_cache

# The empty state /api/state returns when task_id doesn't exist / there's no
# task / state.json is missing, so the frontend monitor page can render
# stably (no error, and doesn't just show "loading").
EMPTY_STATE: dict[str, Any] = {
    "losses": [],
    "lr_history": [],
    "epoch": 0,
    "total_epochs": 0,
    "step": 0,
    "total_steps": 0,
    "speed": 0.0,
    "samples": [],
    "start_time": None,
    "config": {},
}


def _thumb_response(src: Path, size: int, immutable: bool = False) -> FileResponse:
    """Unified thumbnail response: weak etag (based on src mtime+size).

    Defaults to `no-cache, must-revalidate`: an earlier version used
    `max-age=86400`, which made the browser remember every response for 24h,
    including failed responses from a restart transition period — from the
    user's perspective, "images won't load after a restart." Switching to
    etag + no-cache means the browser sends a conditional request every time,
    and a hit resolves via a 304 in a few ms.

    `immutable=True`: for content that never changes once written (and whose
    URL is already a stable, unique key — e.g. training sample images with
    `?task_id=N` plus a filename that includes epoch/step). In that case,
    `max-age=1y, immutable` lets the browser **never revalidate at all** —
    when accessed over a cloudflared tunnel in the cloud, this saves a 304
    round trip per image, so sample images on the monitor page load nearly
    instantly and don't reload on reopen. The restart-transition concern
    doesn't apply here: this header is only sent when `.exists()` hits and an
    actual image is returned; 404 / failure responses carry no cache header.

    PR-6: extracted from server.py into api/responses.py, shared by the
    samples router and the project_thumb in server.py (which stayed in
    server.py until PR-6.5).
    """
    out = thumb_cache.get_or_make_thumb(src, size)
    try:
        mtime_ns = out.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    etag = f'W/"{mtime_ns}-{size}"'
    cache_control = (
        "public, max-age=31536000, immutable"
        if immutable
        else "no-cache, must-revalidate"
    )
    return FileResponse(
        out,
        headers={
            "Cache-Control": cache_control,
            "ETag": etag,
        },
    )
