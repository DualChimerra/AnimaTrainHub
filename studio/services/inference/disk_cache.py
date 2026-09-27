"""Encrypted on-disk cache for test-generation outputs (replaces the old cache.py's pure in-memory dict).

Why move to disk: the in-memory cache ate resident RAM (a 200-image / 500MB
cap), and users doing long training sessions plus batch generation would hit
that ceiling. Moving to disk relieves the RAM pressure, and also makes room
for the cache to persist across a session (surviving a refresh / route
change without losing the history strip).

Why encrypt: when a user runs test generation on a cloud training machine,
some domestic cloud providers scan disks for sensitive images by filename /
extension / PNG magic bytes / content classification (CNN). A plaintext PNG
on disk, even for a few seconds, can get flagged; once encrypted, the file on
disk is high-entropy random bytes with no extension and no magic bytes, so
disk scanners can't identify it.

Threat model = defending against passive disk scanning: this does not defend
against a local attacker attaching to the process to dump the key, nor
against someone deliberately reading images through the app's own code -
neither of those would be stopped by this line of defense anyway, so adding
AEAD or real asymmetric crypto would be over-engineering.

Mechanism ("session fingerprint"):
  - on startup, the process generates a session_id (uuid4) + aes_key (32
    random bytes), kept **only in process memory**, never written to disk
  - cache directory: `studio_data/.cache/generate/session-<uuid>/`
  - one file per image, `<file_uuid>.bin`, no extension, self-contained:
    `[16B nonce][SHAKE-128 keystream XOR(payload)]`
    where payload = `[4B snapshot_len][snapshot_json_utf8][png_bytes]`
  - on startup, every `session-*` directory under root is rmtree'd (this also
    cleans up leftovers from a previous SIGKILL / power loss) - the new
    session_id is random, so it never collides with an old directory
  - on shutdown, the session directory is deleted, and the key disappears with the process

Leftover files after an unclean exit have no key to decrypt them - they're
just random bytes, unrecognizable to a disk scanner as an image. The next
startup's startup_clean sweeps them up anyway.

Crypto choice: pure-stdlib `hashlib.shake_128` keystream + XOR. SHAKE-128 is a
SHA-3 variant (NIST standard), implemented in C, encrypting/decrypting a
1.5MB PNG in milliseconds. No AEAD, but the threat model doesn't need
integrity - defeating magic-byte/entropy/CNN-based disk scanning is enough.
Zero extra dependencies sidesteps the `cryptography` library's DLL
compatibility headaches.

API mirrors cache.py (which this module replaces): put + get_image + list_filenames +
drop_task + clear_all + configure + list_index (new).

"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_COUNT = 200
DEFAULT_MAX_BYTES = 500 * 1024 * 1024
_SESSION_DIR_PREFIX = "session-"
_NONCE_LEN = 16
_SNAPSHOT_LEN_HEADER = 4  # big-endian uint32


@dataclass
class _Entry:
    """One entry in the index - file path + in-memory snapshot cache + size."""
    file_path: Path
    snapshot: dict[str, Any]
    created_at: float
    size: int  # size in bytes of the encrypted file (nonce + payload)
    mode: str  # 'single' | 'xy', used by the frontend's history strip grouping
    task_id: int
    filename: str
    # {xi, yi, xv, yv} carried by the daemon's image_done event in XY mode; None in single mode.
    # list_index reconstructs xyMeta.samples from this for the frontend's PreviewXYGrid.
    xy_info: Optional[dict[str, Any]] = None


@dataclass
class SessionCache:
    """Session-scoped encrypted disk cache, thread-safe.

    Each server process holds one instance (created at lifespan startup,
    cleared with clear_all() at shutdown). Leftover directories from a
    SIGKILL / power loss are cleaned up on the next startup by startup_clean().
    """
    root: Path
    max_count: int = DEFAULT_MAX_COUNT
    max_bytes: int = DEFAULT_MAX_BYTES

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    aes_key: bytes = field(default_factory=lambda: os.urandom(32))
    _lock: threading.RLock = field(default_factory=threading.RLock)
    # (task_id, filename) -> _Entry; OrderedDict maintains LRU order
    _index: "collections.OrderedDict[tuple[int, str], _Entry]" = field(
        default_factory=collections.OrderedDict,
    )
    _bytes_total: int = 0

    @property
    def session_dir(self) -> Path:
        return self.root / f"{_SESSION_DIR_PREFIX}{self.session_id}"

    def ensure_dir(self) -> None:
        """mkdir the session directory. Called once after lifespan init."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- put/get
    def put(
        self,
        task_id: int,
        filename: str,
        data: bytes,
        snapshot: dict[str, Any],
        *,
        mode: str = "single",
        xy_info: Optional[dict[str, Any]] = None,
    ) -> None:
        """Called on the daemon's image_done event; a repeat (task_id, filename) overwrites (deletes the old file, writes a new one).

        snapshot: the frontend-built GenerateParamsSnapshot dict, bundled
        into the encrypted payload header alongside the PNG; list_index()
        returns the same copy to the frontend, avoiding a second source of truth.

        xy_info: {xi, yi, xv, yv} carried by the daemon's image_done event in
        XY mode, used to reconstruct the PreviewXYGrid; None in single mode.
        """
        snap_bytes = json.dumps(snapshot, ensure_ascii=False).encode("utf-8")
        if len(snap_bytes) >= 2**32:
            raise ValueError("snapshot too large (>4GiB)")
        payload = len(snap_bytes).to_bytes(_SNAPSHOT_LEN_HEADER, "big") + snap_bytes + data
        nonce = os.urandom(_NONCE_LEN)
        ciphertext = _xor(payload, _keystream(self.aes_key, nonce, len(payload)))
        blob = nonce + ciphertext

        file_id = uuid.uuid4().hex
        file_path = self.session_dir / f"{file_id}.bin"
        # atomic write: write to tmp + rename. The session directory is exclusive to this process, so there's no concurrent write to the same name
        tmp = file_path.with_suffix(".bin.tmp")
        tmp.write_bytes(blob)
        tmp.replace(file_path)

        with self._lock:
            key = (task_id, filename)
            old = self._index.pop(key, None)
            if old is not None:
                self._bytes_total -= old.size
                _safe_unlink(old.file_path)
            entry = _Entry(
                file_path=file_path,
                snapshot=snapshot,
                created_at=time.time(),
                size=len(blob),
                mode=mode,
                task_id=task_id,
                filename=filename,
                xy_info=xy_info,
            )
            self._index[key] = entry
            self._bytes_total += entry.size
            self._enforce_limits_locked()

    def get_image(self, task_id: int, filename: str) -> Optional[bytes]:
        """Read an image: look up file_path -> read from disk -> decrypt -> strip the snapshot header -> return PNG bytes.

        A hit does move_to_end (treated as most-recently-used for LRU). If
        the file is gone (deleted externally / decrypt failure), returns None
        and also evicts the index entry, so it doesn't keep hanging around as
        a dead reference.
        """
        with self._lock:
            key = (task_id, filename)
            entry = self._index.get(key)
            if entry is None:
                return None
            self._index.move_to_end(key)
            file_path = entry.file_path
        # decrypt IO happens without holding the lock, so one large image doesn't stall the whole cache
        try:
            blob = file_path.read_bytes()
        except FileNotFoundError:
            with self._lock:
                self._index.pop(key, None)
                self._bytes_total -= entry.size
            logger.warning("disk cache file gone: %s", file_path)
            return None
        try:
            return _decrypt_and_strip(self.aes_key, blob)
        except Exception:
            logger.exception("decrypt failed for %s", file_path)
            return None

    # ---------------------------------------------------------------- list / delete
    def list_filenames(self, task_id: int) -> list[str]:
        with self._lock:
            return sorted(fn for (tid, fn) in self._index if tid == task_id)

    def list_index(self) -> list[dict[str, Any]]:
        """Used by the frontend's history strip via GET /api/generate/cache/index.

        Aggregated by task_id - multiple images from the same task (one per
        XY grid cell) merge into a single history entry. Return shape matches
        the frontend's CacheEntry adapter:
            { id, taskId, mode, createdAt (ms), filenames[], params, samples? }
        `samples` only exists when mode=xy, listing [{filename,
        xy:{xi,yi,xv,yv}}] for PreviewXYGrid to reconstruct the grid.
        createdAt is the most recent entry's timestamp within that task.
        Sorted by createdAt descending (newest first).
        """
        with self._lock:
            entries = list(self._index.values())

        by_task: dict[int, list[_Entry]] = collections.defaultdict(list)
        for e in entries:
            by_task[e.task_id].append(e)

        out: list[dict[str, Any]] = []
        for task_id, group in by_task.items():
            group_sorted = sorted(group, key=lambda e: e.filename)
            first = group_sorted[0]
            latest_created = max(e.created_at for e in group)
            item: dict[str, Any] = {
                "id": f"cache:{task_id}",
                "taskId": task_id,
                "mode": first.mode,
                "createdAt": int(latest_created * 1000),
                "filenames": [e.filename for e in group_sorted],
                "params": first.snapshot,
            }
            if first.mode == "xy":
                item["samples"] = [
                    {"filename": e.filename, "xy": e.xy_info}
                    for e in group_sorted
                    if e.xy_info is not None
                ]
            out.append(item)

        out.sort(key=lambda x: x["createdAt"], reverse=True)
        return out

    def drop_task(self, task_id: int) -> int:
        with self._lock:
            keys = [k for k in self._index if k[0] == task_id]
            for k in keys:
                entry = self._index.pop(k)
                self._bytes_total -= entry.size
                _safe_unlink(entry.file_path)
            return len(keys)

    def total_count(self) -> int:
        with self._lock:
            return len(self._index)

    def total_bytes(self) -> int:
        with self._lock:
            return self._bytes_total

    def clear_all(self) -> None:
        """Delete the entire session directory + clear the index. Called at lifespan shutdown; also used by tests."""
        with self._lock:
            self._index.clear()
            self._bytes_total = 0
        if self.session_dir.exists():
            try:
                shutil.rmtree(self.session_dir)
            except OSError:
                logger.exception("clear_all: rmtree failed for %s", self.session_dir)
        # let subsequent put() calls keep working (recreate the dir)
        self.ensure_dir()

    def configure(
        self,
        *,
        max_count: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> None:
        """Dynamically adjust the caps. Enforced immediately after the change."""
        with self._lock:
            if max_count is not None:
                self.max_count = int(max_count)
            if max_bytes is not None:
                self.max_bytes = int(max_bytes)
            self._enforce_limits_locked()

    def _enforce_limits_locked(self) -> None:
        while self._index and (
            len(self._index) > self.max_count or self._bytes_total > self.max_bytes
        ):
            _, entry = self._index.popitem(last=False)  # oldest
            self._bytes_total -= entry.size
            _safe_unlink(entry.file_path)


# ---------------------------------------------------------------------------
# module-level singleton - created by lifespan init; tests may init their own instance explicitly
# ---------------------------------------------------------------------------

_session: Optional[SessionCache] = None
_session_lock = threading.Lock()


def init(
    root: Path,
    *,
    max_count: int = DEFAULT_MAX_COUNT,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> SessionCache:
    """Initialize the module singleton. Called once at lifespan startup.

    Runs startup_clean(root) first to remove leftover session-* directories, then creates the new session.
    """
    global _session
    with _session_lock:
        if _session is not None:
            # calling init() again with an existing session = the old session is no longer reachable, clear it before starting a new one
            _session.clear_all()
        startup_clean(root)
        sc = SessionCache(root=root, max_count=max_count, max_bytes=max_bytes)
        sc.ensure_dir()
        _session = sc
        logger.info("disk_cache initialized: session_id=%s dir=%s", sc.session_id, sc.session_dir)
        return sc


def get_session() -> SessionCache:
    """Get the current singleton. Raises if not yet init'd (avoids a lazy implicit init hiding call-order bugs)."""
    if _session is None:
        raise RuntimeError("disk_cache not initialized; call init() first")
    return _session


def startup_clean(root: Path) -> int:
    """rmtree every `session-*` directory under root.

    The new session_id is a random uuid, guaranteed never to collide with a
    historical one, so "delete every session-*" is always equivalent to
    "delete the stale ones". Returns how many directories were deleted.
    """
    if not root.exists():
        return 0
    n = 0
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith(_SESSION_DIR_PREFIX):
            try:
                shutil.rmtree(child)
                n += 1
            except OSError:
                logger.warning("startup_clean: rmtree failed for %s", child, exc_info=True)
    if n:
        logger.info("startup_clean: removed %d stale session dir(s) under %s", n, root)
    return n


# ---------------------------------------------------------------------------
# module-level shortcuts - replace the old cache.py functions of the same name, zero changes needed at call sites
# ---------------------------------------------------------------------------


def cache_image(
    task_id: int,
    filename: str,
    data: bytes,
    snapshot: Optional[dict[str, Any]] = None,
    *,
    mode: str = "single",
    xy_info: Optional[dict[str, Any]] = None,
) -> None:
    """Called on the daemon's image_done event (replaces the old cache.cache_image). snapshot defaults to {}."""
    get_session().put(
        task_id, filename, data, snapshot or {},
        mode=mode, xy_info=xy_info,
    )


def get_image(task_id: int, filename: str) -> Optional[bytes]:
    return get_session().get_image(task_id, filename)


def list_filenames(task_id: int) -> list[str]:
    return get_session().list_filenames(task_id)


def list_index() -> list[dict[str, Any]]:
    return get_session().list_index()


def drop_task(task_id: int) -> int:
    return get_session().drop_task(task_id)


def total_count() -> int:
    if _session is None:
        return 0
    return _session.total_count()


def total_bytes() -> int:
    if _session is None:
        return 0
    return _session.total_bytes()


def clear_all() -> None:
    """Called at lifespan shutdown; a no-op if the singleton was never init'd."""
    if _session is None:
        return
    _session.clear_all()


def configure(
    *,
    max_count: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> None:
    get_session().configure(max_count=max_count, max_bytes=max_bytes)


# ---------------------------------------------------------------------------
# Crypto helpers - SHAKE-128 keystream + XOR
# ---------------------------------------------------------------------------


def _keystream(key: bytes, nonce: bytes, n: int) -> bytes:
    """SHAKE-128(key || nonce).digest(n) - a keystream of arbitrary length.

    SHAKE-128 is the SHA-3 standard's extendable-output function (NIST FIPS
    202). Output for a given (key, nonce) pair is always deterministic, used
    here as the keystream for XOR stream encryption.
    """
    h = hashlib.shake_128()
    h.update(key)
    h.update(nonce)
    return h.digest(n)


def _xor(a: bytes, b: bytes) -> bytes:
    """Equal-length XOR. For 1.5MB, int.from_bytes is ~50x faster than a zip loop."""
    if len(a) != len(b):
        raise ValueError("xor lengths differ")
    if not a:
        return b""
    return (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).to_bytes(len(a), "big")


def _decrypt_and_strip(key: bytes, blob: bytes) -> bytes:
    """Reverse: blob -> decrypt payload -> strip the snapshot header -> return PNG bytes only."""
    if len(blob) < _NONCE_LEN + _SNAPSHOT_LEN_HEADER:
        raise ValueError("blob too short")
    nonce = blob[:_NONCE_LEN]
    ciphertext = blob[_NONCE_LEN:]
    payload = _xor(ciphertext, _keystream(key, nonce, len(ciphertext)))
    snap_len = int.from_bytes(payload[:_SNAPSHOT_LEN_HEADER], "big")
    if snap_len > len(payload) - _SNAPSHOT_LEN_HEADER:
        raise ValueError("snapshot len out of range")
    return payload[_SNAPSHOT_LEN_HEADER + snap_len:]


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("unlink failed for %s", p, exc_info=True)
