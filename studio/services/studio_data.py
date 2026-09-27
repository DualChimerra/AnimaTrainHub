"""studio_data directory migration -- scans size + copies to a custom location in the background (no ADR; a small feature).

Flow (frontend Settings -> System -> Storage location):
1. GET /api/studio-data/info       -- current/default location + a full scan (file count/byte count/top-level breakdown)
2. POST /api/studio-data/migrate   -- validates, then starts a background copy thread; progress goes over SSE
3. Copy completes -> writes the repo-root pointer file `studio_data_location.json` -> takes effect after a server restart

Design notes:
- **The target is a parent directory**: the user picks any directory, and data lands at `target/studio_data/` (the whole
  studio_data gets "moved into it"); the target itself isn't required to be empty -- only the landing subdirectory
  must not exist or be empty (never merged into existing data).
- **Copy only, never delete**: the old data is left as-is (the owner's decision); on failure, the half-copied landing
  directory is cleaned up (required to be empty/nonexistent before starting, so rmtree is safe), and the pointer is never written, so it's as if nothing happened.
- **sqlite consistency**: the server process may write to studio.db at any time while a request is in flight,
  so a plain copy could catch a half-written page mid-write. `.db` files go through the sqlite3 backup API (an online backup that gets a consistent snapshot);
  the corresponding `-wal` / `-shm` files are skipped (the backup output is self-contained).
- **Single-flight**: only one migration is allowed at a time (module-level lock + a status singleton).
"""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ..infrastructure.event_bus import bus
from ..infrastructure.paths import DEFAULT_STUDIO_DATA, STUDIO_DATA, STUDIO_DATA_POINTER

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL_SECONDS = 0.2

# The landing subdirectory name is fixed -- it doesn't follow the current location's directory name (an old-style migration's custom location might be named something else)
DATA_DIR_NAME = "studio_data"

Publish = Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_studio_data(root: Path | None = None) -> dict[str, Any]:
    """Fully scans studio_data: total file count / total byte count + a top-level entry breakdown (used to display the confirm modal).

    `-wal` / `-shm` are not counted (skipped during migration, see the module docstring). Returns all zeros if the directory doesn't exist.
    """
    base = root if root is not None else STUDIO_DATA
    entries: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    if not base.is_dir():
        return {"total_files": 0, "total_bytes": 0, "entries": []}
    for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
        files = 0
        size = 0
        if child.is_dir():
            for f in child.rglob("*"):
                if not f.is_file() or _skip_file(f):
                    continue
                files += 1
                try:
                    size += f.stat().st_size
                except OSError:
                    pass
        elif child.is_file():
            if _skip_file(child):
                continue
            files = 1
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
        entries.append({
            "name": child.name,
            "is_dir": child.is_dir(),
            "files": files,
            "bytes": size,
        })
        total_files += files
        total_bytes += size
    return {"total_files": total_files, "total_bytes": total_bytes, "entries": entries}


def _skip_file(p: Path) -> bool:
    """sqlite companion files are not copied: the backup API output is already a single consistent file."""
    return p.name.endswith(".db-wal") or p.name.endswith(".db-shm")


# ---------------------------------------------------------------------------
# Migration status (singleton)
# ---------------------------------------------------------------------------

@dataclass
class MigrationStatus:
    state: str = "idle"          # idle / running / done / error
    target: str = ""
    total_files: int = 0
    total_bytes: int = 0
    done_files: int = 0
    done_bytes: int = 0
    current_file: str = ""       # relative path, used for progress display
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_status = MigrationStatus()
_status_lock = threading.Lock()


def migration_status() -> dict[str, Any]:
    with _status_lock:
        return _status.as_dict()


def _set_status(**kw: Any) -> None:
    with _status_lock:
        for k, v in kw.items():
            setattr(_status, k, v)


# ---------------------------------------------------------------------------
# Validation + start
# ---------------------------------------------------------------------------

def validate_target(target: Path, *, source: Path | None = None) -> Path:
    """Validates the migration target, raising ValueError if invalid (the caller turns this into a 422); returns the actual landing directory.

    target is any directory the user picked; data lands at `target/studio_data/`, so target itself
    isn't required to be empty. Rules: absolute path; if target already exists it must be a directory; the landing directory can't be
    the current location; the landing directory and current location can't be nested inside each other (copying into your own subtree would recurse infinitely);
    the landing directory must not exist or must be empty (never merged into existing data).
    """
    src = (source if source is not None else STUDIO_DATA).resolve()
    if not target.is_absolute():
        raise ValueError("The target must be an absolute path")
    if target.exists() and not target.is_dir():
        raise ValueError("The target already exists and is not a directory")
    dst = target.resolve() / DATA_DIR_NAME
    if dst == src:
        raise ValueError("The target is the same as the current studio_data location")
    for a, b in ((dst, src), (src, dst)):
        try:
            a.relative_to(b)
        except ValueError:
            continue
        raise ValueError("The target and the current studio_data are nested inside each other")
    if dst.exists():
        if not dst.is_dir():
            raise ValueError(f"A file named {DATA_DIR_NAME} already exists in the target")
        if any(dst.iterdir()):
            raise ValueError(f"A non-empty {DATA_DIR_NAME} directory already exists in the target")
    return dst


def start_migration(
    target: Path,
    *,
    source: Path | None = None,
    publish: Publish = bus.publish,
    pointer_file: Path | None = None,
) -> None:
    """Validates then starts the background copy thread. Raises RuntimeError if a migration is already running (the caller turns this into a 409).

    target is the parent directory the user picked; the actual copy goes to `target/studio_data/` (validate_target's
    return value). The source / pointer_file params are for test injection only; production uses the defaults (the current
    STUDIO_DATA + the repo-root pointer).
    """
    src = (source if source is not None else STUDIO_DATA).resolve()
    ptr = pointer_file if pointer_file is not None else STUDIO_DATA_POINTER
    dst = validate_target(target, source=src)
    with _status_lock:
        if _status.state == "running":
            raise RuntimeError("A migration is already running")
        _status.state = "running"
        _status.target = str(dst)
        _status.total_files = 0
        _status.total_bytes = 0
        _status.done_files = 0
        _status.done_bytes = 0
        _status.current_file = ""
        _status.error = ""
    t = threading.Thread(
        target=_run_migration,
        args=(src, dst, publish, ptr),
        name="studio-data-migration",
        daemon=True,
    )
    t.start()


# ---------------------------------------------------------------------------
# Copy thread
# ---------------------------------------------------------------------------

def _run_migration(src: Path, dst: Path, publish: Publish, pointer_file: Path) -> None:
    try:
        files = [
            f for f in sorted(src.rglob("*"))
            if f.is_file() and not _skip_file(f)
        ]
        total_bytes = 0
        for f in files:
            try:
                total_bytes += f.stat().st_size
            except OSError:
                pass
        _set_status(total_files=len(files), total_bytes=total_bytes)

        dst.mkdir(parents=True, exist_ok=True)
        last_pub = 0.0
        done_files = 0
        done_bytes = 0
        for f in files:
            rel = f.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                size = f.stat().st_size
                if f.suffix == ".db":
                    _backup_sqlite(f, out)
                else:
                    shutil.copy2(f, out)
            except FileNotFoundError:
                # Deleted after the scan (e.g. a temp file) -- skip; progress may stall below 100%, harmless
                logger.info("File disappeared during migration, skipping: %s", rel)
                continue
            done_files += 1
            done_bytes += size
            now = time.monotonic()
            if now - last_pub >= PROGRESS_INTERVAL_SECONDS:
                last_pub = now
                _set_status(done_files=done_files, done_bytes=done_bytes, current_file=str(rel))
                publish({
                    "type": "studio_data_migrate_progress",
                    "done_files": done_files,
                    "total_files": len(files),
                    "done_bytes": done_bytes,
                    "total_bytes": total_bytes,
                    "current_file": str(rel),
                })

        pointer_file.write_text(
            json.dumps({"path": str(dst)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _set_status(state="done", done_files=done_files, done_bytes=done_bytes, current_file="")
        publish({
            "type": "studio_data_migrate_done",
            "ok": True,
            "target": str(dst),
            "done_files": done_files,
            "done_bytes": done_bytes,
        })
        logger.info("studio_data migration complete: %s -> %s (%d files), effective after restart", src, dst, done_files)
    except Exception as exc:
        logger.exception("studio_data migration failed: %s -> %s", src, dst)
        # dst is the target/studio_data landing directory, which was empty/nonexistent before starting
        # (guaranteed by validate_target), so clearing the whole tree just reverts to the pre-migration state; the user's target parent directory is untouched
        shutil.rmtree(dst, ignore_errors=True)
        _set_status(state="error", error=str(exc))
        publish({"type": "studio_data_migrate_done", "ok": False, "error": str(exc)})


def _backup_sqlite(src_db: Path, out: Path) -> None:
    """Uses an sqlite online backup to get a consistent snapshot; falls back to a plain copy for non-sqlite .db files."""
    try:
        with sqlite3.connect(str(src_db)) as conn, sqlite3.connect(str(out)) as dst_conn:
            conn.backup(dst_conn)
    except sqlite3.Error:
        out.unlink(missing_ok=True)
        shutil.copy2(src_db, out)
