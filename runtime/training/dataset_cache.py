"""Shared dataset cache: VAE latents and caption encodings stored inside the
source dataset's folder, so every training that uses the same images reuses them.

Studio writes ``{data_dir}/.cache_routes.json`` when it enqueues a training run:

    {"version": 1, "routes": {"1_data/001.png": "<dataset>/cache", ...}}

Only images curated from a shared dataset have a route; everything else (the
project's own download/, reg images, CLI runs without the file) keeps the
classic sidecar next to the image. The runtime never imports studio -- this
file is the whole contract.

Layout under a cache root (one directory per format, so switching model,
resolution or text encoder writes new files next to the old ones instead of
replacing them):

    latents/<fingerprint>-v<layout>/<sha>_<w>x<h>.npz
    text/<fingerprint>-v<format>/<caption key>.safetensors
    <kind>/<group>/group.json      what the group is for, shown by Studio

Entries are addressed by the bytes of the training image (sha256), not by its
path: two versions that copied the same original share one cache entry, and a
cropped / upscaled copy gets its own.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)

ROUTES_FILENAME = ".cache_routes.json"
GROUP_META = "group.json"
LATENT_KIND = "latents"
TEXT_KIND = "text"

_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def group_name(fingerprint: str, version: int) -> str:
    """Directory-safe name of one cache format (fingerprint + layout version)."""
    slug = _SLUG.sub("-", str(fingerprint)).strip("-") or "default"
    return f"{slug}-v{int(version)}"


class SharedCacheRouter:
    """Maps training images to their dataset cache root and builds entry paths."""

    def __init__(self, data_dir, routes: dict[str, str]):
        self.data_dir = Path(data_dir)
        self._routes = {str(k): Path(v) for k, v in routes.items() if v}
        self._hashes: dict[str, str] = {}
        self._lock = threading.Lock()
        self._touched: set[Path] = set()

    @classmethod
    def load(cls, data_dir) -> Optional["SharedCacheRouter"]:
        """Router for ``data_dir``, or None when Studio routed nothing there."""
        if not data_dir:
            return None
        path = Path(data_dir) / ROUTES_FILENAME
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("[dataset-cache] unreadable %s, using sidecar caches: %s", path, exc)
            return None
        routes = payload.get("routes") if isinstance(payload, dict) else None
        if not isinstance(routes, dict) or not routes:
            return None
        return cls(data_dir, routes)

    def __len__(self) -> int:
        return len(self._routes)

    def root_for(self, image_path) -> Optional[Path]:
        try:
            rel = Path(image_path).resolve().relative_to(self.data_dir.resolve()).as_posix()
        except ValueError:
            return None
        return self._routes.get(rel)

    def content_hash(self, image_path) -> str:
        key = str(image_path)
        with self._lock:
            cached = self._hashes.get(key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with open(image_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        value = digest.hexdigest()[:32]
        with self._lock:
            self._hashes[key] = value
        return value

    def latent_path(
        self, image_path, *, fingerprint: str, layout_version: int, bucket: tuple[int, int],
    ) -> Optional[Path]:
        """``bucket`` is (width, height) of the encoded pixels."""
        root = self.root_for(image_path)
        if root is None:
            return None
        w, h = int(bucket[0]), int(bucket[1])
        group = root / LATENT_KIND / group_name(fingerprint, layout_version)
        return group / f"{self.content_hash(image_path)}_{w}x{h}.npz"

    def text_path(self, image_path, *, cache_key: str, text_fingerprint: str, format_version: int) -> Optional[Path]:
        root = self.root_for(image_path)
        if root is None:
            return None
        group = root / TEXT_KIND / group_name(text_fingerprint, format_version)
        return group / f"{cache_key[:40]}.safetensors"

    def touch_group(self, group_dir: Path, meta: dict[str, Any]) -> None:
        """Write/refresh ``group.json`` once per run (what it is + when it was last used)."""
        group_dir = Path(group_dir)
        if group_dir in self._touched:
            return
        self._touched.add(group_dir)
        path = group_dir / GROUP_META
        current: dict[str, Any] = {}
        if path.is_file():
            try:
                current = json.loads(path.read_text(encoding="utf-8")) or {}
            except (OSError, ValueError):
                current = {}
        now = time.time()
        current.update(meta)
        current.setdefault("created_at", now)
        current["last_used_at"] = now
        try:
            group_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.debug("[dataset-cache] cannot write %s: %s", path, exc)
