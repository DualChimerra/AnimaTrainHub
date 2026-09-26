"""Models-root migration -- scans the size and copies it to a custom location in
the background (mirrors the studio_data migration).

Flow (frontend Settings -> System -> Storage location -> Models root):
1. GET  /api/models-root/info          -- current/default location + a full scan (file count/bytes/top-level breakdown)
2. POST /api/models-root/migrate       -- validates, then starts a background copy thread; progress goes over SSE
3. Copy finishes -> updates `secrets.models.root` -> **takes effect immediately**
   (`models_root()` reads the secret fresh every time, no restart needed --
   unlike studio_data's pointer-file-plus-restart approach)

Design points (consistent with `studio_data.py`):
- **The target is a parent directory**: the user picks any directory, and data
  lands at `target/models/`; the target itself is not required to be empty.
- **Copy only, never delete**: the old directory is left as-is (a deliberate
  choice; old absolute paths baked into existing versions' yaml files can still
  resolve).
- **Single-flight**: a module-level lock + a status singleton.
- Model weights have no sqlite/wal, so there's no `.db` backup branch.

Where this diverges from studio_data (issue #351): when the target directory
already has data, it no longer rejects unconditionally -- instead it raises
`TargetConflictError` so the frontend can pop up a "skip / overwrite / cancel"
choice. Model directories are often shared with image-generation tools, so a
merge migration that ends in "skip existing files" is equivalent to just
pointing the path there and filling in the missing files. A failed rollback in
merge mode **must not** rmtree (it would delete the user's existing data) --
instead each file is written atomically via a `.part` temp file, and on
failure the already-copied valid copies are left in place (skip is idempotent,
so a rerun just resumes); the secret only switches once the entire copy is done.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .. import secrets
from ..infrastructure.event_bus import bus
from ..infrastructure.paths import REPO_ROOT
from .models import models_root

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL_SECONDS = 0.2

# The landing subdirectory name is fixed -- it doesn't follow the current
# location's directory name (a custom location might be named anything).
DATA_DIR_NAME = "models"

Publish = Callable[[dict[str, Any]], None]


def default_models_root() -> Path:
    """The default location when `secrets.models.root` is unset (matches `models_root()`'s fallback)."""
    return REPO_ROOT / "models"


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_models_root(root: Path | None = None) -> dict[str, Any]:
    """Does a full scan of the models root: total file count / total bytes +
    top-level entry breakdown (for display in the confirmation modal).

    Returns all zeros when the directory doesn't exist (no model has been
    downloaded yet).
    """
    base = root if root is not None else models_root()
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
                if not f.is_file():
                    continue
                files += 1
                try:
                    size += f.stat().st_size
                except OSError:
                    pass
        elif child.is_file():
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
# Validation + startup
# ---------------------------------------------------------------------------

class TargetConflictError(ValueError):
    """The target directory already has data and no merge policy was given --
    the user must explicitly choose one of three options (skip/overwrite/cancel).

    Subclassing ValueError keeps old `except ValueError` fallbacks working;
    the router catches this class and converts it to 409 + code
    `models_root.target_conflict`, with `details` for the frontend's conflict dialog.
    """

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


def _conflict_details(src: Path, dst: Path) -> dict[str, Any]:
    """Stats for the conflict dialog: the target's existing file count/bytes +
    the count of files with the same name in the current root (which would be
    overwritten)."""
    existing_files = 0
    existing_bytes = 0
    for f in dst.rglob("*"):
        if not f.is_file():
            continue
        existing_files += 1
        try:
            existing_bytes += f.stat().st_size
        except OSError:
            pass
    same_name_files = 0
    if src.is_dir():
        for f in src.rglob("*"):
            if f.is_file() and (dst / f.relative_to(src)).exists():
                same_name_files += 1
    return {
        "target": str(dst),
        "existing_files": existing_files,
        "existing_bytes": existing_bytes,
        "same_name_files": same_name_files,
    }


def validate_target(
    target: Path, *, source: Path | None = None, on_conflict: str | None = None
) -> Path:
    """Validates the migration target, raising ValueError if invalid (the caller
    converts to 422); returns the actual landing directory.

    target is any directory the user picked; data lands at `target/models/`, so
    target itself isn't required to be empty. Rules: must be an absolute path;
    if target already exists it must be a directory; the landing directory
    can't equal the current location; the landing directory and the current
    location must not be nested inside each other. When the landing directory
    already has data, behavior depends on on_conflict: None -> raises
    TargetConflictError (frontend shows the three-way choice); "skip" /
    "overwrite" -> allows the merge to proceed.
    """
    if on_conflict not in (None, "skip", "overwrite"):
        raise ValueError(f"Unknown conflict policy {on_conflict!r}")
    src = (source if source is not None else models_root()).resolve()
    if not target.is_absolute():
        raise ValueError("The target must be an absolute path")
    if target.exists() and not target.is_dir():
        raise ValueError("The target already exists and is not a directory")
    dst = target.resolve() / DATA_DIR_NAME
    if dst == src:
        raise ValueError("The target is the same as the current models root")
    for a, b in ((dst, src), (src, dst)):
        try:
            a.relative_to(b)
        except ValueError:
            continue
        raise ValueError("The target and the current models root are nested inside each other")
    if dst.exists():
        if not dst.is_dir():
            raise ValueError(f"A file named {DATA_DIR_NAME} already exists in the target")
        if any(dst.iterdir()) and on_conflict is None:
            raise TargetConflictError(
                f"A non-empty {DATA_DIR_NAME} directory already exists in the target",
                _conflict_details(src, dst),
            )
    return dst


def start_migration(
    target: Path,
    *,
    source: Path | None = None,
    publish: Publish = bus.publish,
    on_conflict: str | None = None,
) -> None:
    """Validates and starts the background copy thread. Raises RuntimeError if a
    migration is already running (the caller converts to 409).

    on_conflict: the merge policy when the landing directory already has data
    ("skip" / "overwrite"); if None and there's a conflict, validate_target
    raises TargetConflictError. The source parameter is for test injection
    only; production uses the default (the current `models_root()`).
    """
    src = (source if source is not None else models_root()).resolve()
    dst = validate_target(target, source=src, on_conflict=on_conflict)
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
        args=(src, dst, publish, on_conflict),
        name="models-root-migration",
        daemon=True,
    )
    t.start()


def _update_models_root_secret(new_root: Path) -> None:
    """After a successful copy, points `secrets.models.root` at the new landing
    directory (takes effect immediately, no restart needed)."""
    cur = secrets.load()
    new_models = cur.models.model_copy(update={"root": str(new_root)})
    secrets.save(cur.model_copy(update={"models": new_models}))


# ---------------------------------------------------------------------------
# Copy thread
# ---------------------------------------------------------------------------

def _copy_atomic(src_file: Path, out: Path) -> None:
    """copy2 to a temp name in the same directory, then os.replace -- the target
    location only ever has complete files.

    Under merge/overwrite mode, a direct copy2-overwrite would smash the
    target's existing complete weight file into a half-written one if it
    failed; writing to a temp name first and then atomically replacing means
    the target either keeps the old file or switches to the new one entirely.
    The temp file is cleaned up on any failure.
    """
    part = out.with_name(out.name + ".studio-migrate.part")
    try:
        shutil.copy2(src_file, part)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, out)


def _run_migration(
    src: Path, dst: Path, publish: Publish, on_conflict: str | None = None
) -> None:
    # 开跑前记住 dst 是否已有数据：合并模式下 dst 的既有内容是用户的（可能与
    # 跑图工具共用目录），失败回滚绝不能整树 rmtree —— 据此选回滚策略。
    dst_preexisted = dst.is_dir() and any(dst.iterdir())
    try:
        files = [f for f in sorted(src.rglob("*")) if f.is_file()]
        if on_conflict == "skip":
            # 目标已有同名文件的不复制；进度总量只算真要复制的
            files = [f for f in files if not (dst / f.relative_to(src)).exists()]
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
                _copy_atomic(f, out)
            except FileNotFoundError:
                # 扫描后被删（如临时文件）—— 跳过，进度可能停在 <100%，无碍
                logger.info("迁移期间文件消失，跳过: %s", rel)
                continue
            done_files += 1
            done_bytes += size
            now = time.monotonic()
            if now - last_pub >= PROGRESS_INTERVAL_SECONDS:
                last_pub = now
                _set_status(done_files=done_files, done_bytes=done_bytes, current_file=str(rel))
                publish({
                    "type": "models_root_migrate_progress",
                    "done_files": done_files,
                    "total_files": len(files),
                    "done_bytes": done_bytes,
                    "total_bytes": total_bytes,
                    "current_file": str(rel),
                })

        # 复制完整 → 切 secret（立即生效）
        _update_models_root_secret(dst)
        _set_status(state="done", done_files=done_files, done_bytes=done_bytes, current_file="")
        publish({
            "type": "models_root_migrate_done",
            "ok": True,
            "target": str(dst),
            "done_files": done_files,
            "done_bytes": done_bytes,
        })
        logger.info(
            "模型根目录迁移完成: %s → %s（%d 文件），已更新 secrets.models.root（立即生效）",
            src, dst, done_files,
        )
    except Exception as exc:
        logger.exception("模型根目录迁移失败: %s → %s", src, dst)
        if dst_preexisted:
            # 合并模式：dst 的既有数据是用户的，绝不能整树删。已复制完成的文件
            # 都是完整有效副本（_copy_atomic 保证），留下无害；skip 幂等，重跑即续传。
            logger.info("目标目录原有数据，保留已复制文件，不回滚: %s", dst)
        else:
            # dst 开始前为空 / 不存在，整树清掉等于回到迁移前；
            # secret 未动；用户的 target 父目录不受影响。
            shutil.rmtree(dst, ignore_errors=True)
        _set_status(state="error", error=str(exc))
        publish({"type": "models_root_migrate_done", "ok": False, "error": str(exc)})
