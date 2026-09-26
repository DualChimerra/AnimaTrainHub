"""In-process server memory cache for test-generation images (commit 10 + LRU
fallback in commit 11).

Design:
  - the daemon pushes PNG bytes back over stdout JSON (base64-encoded)
  - InferenceDaemon's reader decodes them and calls cache_image(task_id, filename, bytes)
  - HTTP `GET /api/generate/{tid}/sample/{fn}` reads from here, never touches disk
  - closing/restarting the server clears memory automatically; a hard kill leaves
    nothing behind either

Cleanup triggers (commit 11):
  1. LRU: cap of 200 images / 500MB (whichever triggers first, ordered by write
     access order; reads count too)
  2. SSE client disconnect + 30s buffer (timer hooked into server.py's lifespan)
  3. lifespan shutdown -> clear_all
  4. supervisor actively calls drop_task (task failure/cancellation etc.; the
     interface is still kept for that)
"""
from __future__ import annotations

import collections
import threading
from typing import Optional

# Default cap: 200 images OR 500MB (whichever triggers first). Adjustable via configure().
DEFAULT_MAX_COUNT = 200
DEFAULT_MAX_BYTES = 500 * 1024 * 1024

# (task_id, filename) -> PNG bytes; OrderedDict keeps access order (oldest first)
_CACHE: "collections.OrderedDict[tuple[int, str], bytes]" = collections.OrderedDict()
_LOCK = threading.RLock()
_BYTES_TOTAL = 0  # current total cached bytes
_max_count = DEFAULT_MAX_COUNT
_max_bytes = DEFAULT_MAX_BYTES


def configure(*, max_count: Optional[int] = None, max_bytes: Optional[int] = None) -> None:
    """Adjust the caps dynamically (at startup or in tests). Enforced immediately after."""
    global _max_count, _max_bytes
    with _LOCK:
        if max_count is not None:
            _max_count = int(max_count)
        if max_bytes is not None:
            _max_bytes = int(max_bytes)
        _enforce_limits_locked()


def cache_image(task_id: int, filename: str, data: bytes) -> None:
    """Called on the daemon's image_done. A repeat task_id+filename overwrites the
    entry; triggers LRU eviction."""
    global _BYTES_TOTAL
    with _LOCK:
        key = (task_id, filename)
        if key in _CACHE:
            _BYTES_TOTAL -= len(_CACHE[key])
            del _CACHE[key]
        _CACHE[key] = data
        _BYTES_TOTAL += len(data)
        _enforce_limits_locked()


def get_image(task_id: int, filename: str) -> Optional[bytes]:
    """HTTP image fetch. On a hit, returns the bytes and moves the entry to the end
    (marks it as most-recently-used for LRU)."""
    with _LOCK:
        key = (task_id, filename)
        if key not in _CACHE:
            return None
        _CACHE.move_to_end(key)
        return _CACHE[key]


def list_filenames(task_id: int) -> list[str]:
    """List all filenames currently cached for this task (alphabetical order)."""
    with _LOCK:
        return sorted(fn for (tid, fn) in _CACHE if tid == task_id)


def drop_task(task_id: int) -> int:
    """Delete all cache entries for this task; returns how many were removed."""
    global _BYTES_TOTAL
    with _LOCK:
        keys = [k for k in _CACHE if k[0] == task_id]
        for k in keys:
            _BYTES_TOTAL -= len(_CACHE[k])
            del _CACHE[k]
        return len(keys)


def total_count() -> int:
    """Current number of images in the cache (not the number of tasks)."""
    with _LOCK:
        return len(_CACHE)


def total_bytes() -> int:
    """Current bytes occupied by the cache."""
    with _LOCK:
        return _BYTES_TOTAL


def clear_all() -> None:
    """Called on server lifespan shutdown; also used by tests."""
    global _BYTES_TOTAL
    with _LOCK:
        _CACHE.clear()
        _BYTES_TOTAL = 0


def _enforce_limits_locked() -> None:
    """Evict the oldest entries until max_count and max_bytes are satisfied. Caller
    must already hold the lock."""
    global _BYTES_TOTAL
    while _CACHE and (len(_CACHE) > _max_count or _BYTES_TOTAL > _max_bytes):
        _, data = _CACHE.popitem(last=False)  # oldest
        _BYTES_TOTAL -= len(data)
