"""Image thumbnail cache (PP3 polish).

Why this exists as its own module: previously `/api/.../thumb` served the
original image directly and the frontend just shrank it with CSS. When
download/ has hundreds of multi-MB PNGs, the browser keeps decoding large
images, and scrolling / hovering to switch previews gets janky. Here
thumbnails are pre-generated into `studio_data/thumb_cache/{sha1}.jpg`
(hash = src path + mtime + size), and subsequent requests just return the
cached version.

Design:
- The cache key includes the source file's mtime, so replacing the source
  auto-invalidates it (the hash changes)
- Thread-safe: writes to .tmp first, then renames
- size=0 means "don't resize" — returns the source path directly
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from ...paths import THUMB_CACHE_DIR

logger = logging.getLogger(__name__)

# In-process lock: prevents two concurrent requests from generating the same thumbnail at once (half-written file).
_LOCKS_LOCK = threading.Lock()
_KEY_LOCKS: dict[str, threading.Lock] = {}

# Pillow 9.1+ moved LANCZOS under Image.Resampling; older versions can still use Image.LANCZOS.
_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]


def _key_lock(key: str) -> threading.Lock:
    with _LOCKS_LOCK:
        lk = _KEY_LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _KEY_LOCKS[key] = lk
        return lk


def _key_for(src: Path, size: int) -> Optional[str]:
    """Cache key: sha1(abs_path|mtime_ns|size).

    Returns None when stat fails — no longer falls back to mtime=0, since
    that would make Windows's occasional stat failures (antivirus scanning /
    file locks) cause every affected image to share the same cache key,
    mixing up thumbnails. Callers getting None should skip the cache and
    generate a temporary thumbnail directly (or fall back to the original
    image).
    """
    try:
        mtime = src.stat().st_mtime_ns
    except OSError as exc:
        logger.warning("thumb cache: stat failed for %s: %s", src, exc)
        return None
    payload = f"{src.resolve()}|{mtime}|{size}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def get_or_make_thumb(src: Path, size: int) -> Path:
    """Return a thumbnail path usable directly as a FileResponse.

    size <= 0 -> serve the original image as-is (no resize). Generation
    failure -> falls back to the original image.
    """
    if size <= 0:
        return src
    if not src.exists():
        return src
    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _key_for(src, size)
    if key is None:
        # stat failed: don't cache, don't mix up thumbnails; return the
        # original image directly (Cache-Control: no-cache lets the browser
        # retry next time, and once stat recovers it'll get a proper thumbnail)
        return src
    out = THUMB_CACHE_DIR / f"{key}.jpg"
    if out.exists():
        return out

    lock = _key_lock(key)
    with lock:
        if out.exists():
            return out
        tmp = out.with_suffix(out.suffix + ".tmp")
        try:
            # All pixel operations must complete while the file is still open:
            # ImageOps.exif_transpose returns the original lazy image directly
            # for images with no orientation, and Image.open is lazy; once the
            # with block exits and the file handle closes, a later
            # thumbnail/save triggering the lazy load would fail, getting
            # swallowed by except and falling back to the source image
            # (several MB), leaving the frontend still loading the large
            # image with janky scrolling.
            with Image.open(src) as raw:
                img = ImageOps.exif_transpose(raw) or raw
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail((size, size), _RESAMPLE)
                img.save(tmp, "JPEG", quality=80, optimize=True)
            os.replace(tmp, out)
        except Exception as exc:
            logger.warning(
                "thumb generation failed for %s (size=%d): %s", src, size, exc
            )
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return src
    return out


def prewarm_from_image(
    src: Path, image: Image.Image, sizes: list[int]
) -> list[Path]:
    """Write multiple thumbnail sizes directly to the cache from a PIL Image already in memory, saving decode work on first view.

    Mainly used by the upscaler: the upscaled PIL Image is already in memory,
    so rather than waiting for the user's first visit to re-read the PNG +
    decode + resize (a multi-MB 4x PNG takes 1-3s on CPU), it's better to
    generate both 256 / 768 up front during the worker stage.

    `src` determines the cache key — it must be the real path of the
    upscaled output file (same hash calculation as get_or_make_thumb).
    `image` should be RGB; other modes are auto-converted.
    """
    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    base = image
    if base.mode != "RGB":
        base = base.convert("RGB")
    for size in sizes:
        if size <= 0:
            continue
        key = _key_for(src, size)
        if key is None:
            continue
        out = THUMB_CACHE_DIR / f"{key}.jpg"
        if out.exists():
            written.append(out)
            continue
        lock = _key_lock(key)
        with lock:
            if out.exists():
                written.append(out)
                continue
            tmp = out.with_suffix(out.suffix + ".tmp")
            try:
                thumb = base.copy()
                thumb.thumbnail((size, size), _RESAMPLE)
                thumb.save(tmp, "JPEG", quality=80, optimize=True)
                os.replace(tmp, out)
                written.append(out)
            except Exception as exc:
                logger.warning(
                    "thumb prewarm failed for %s (size=%d): %s", src, size, exc
                )
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    return written


def clear_cache() -> int:
    """Delete all .jpg files under the cache directory; return the number deleted."""
    if not THUMB_CACHE_DIR.exists():
        return 0
    n = 0
    for p in THUMB_CACHE_DIR.glob("*.jpg"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n
