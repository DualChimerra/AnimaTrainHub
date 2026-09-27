"""Caption snapshot (PP4) -- packs all caption files under train/ into a zip in caption_snapshots/.

Created when the user clicks the backup button on the (4) tag-editing page; snapshots can be
listed / restored / deleted.
Snapshot path: `<version_dir>/caption_snapshots/{ts}.zip`
Inside the zip, files are flattened as `<folder>/<filename>`, e.g. `1_data/a.txt`, `5_face/b.json`.
On restore, all existing *.txt / *.json under train/ are cleared first, then unpacked back in.
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Any

from ...paths import safe_join, validate_path_component

CAPTION_EXTS = (".txt", ".json")
SNAPSHOT_DIRNAME = "caption_snapshots"


from studio.domain.errors import DomainError, NotFoundError


class SnapshotError(DomainError):
    """Snapshot business error.

    PR-2 C3 added a DomainError base -- the handler auto-translates it into the dual-write envelope.
    """
    default_code = "snapshot.error"


def snapshot_root(version_dir: Path) -> Path:
    return version_dir / SNAPSHOT_DIRNAME


def _iter_caption_files(train_dir: Path):
    """Yield (rel_path_in_zip, abs_path) pairs for caption files."""
    if not train_dir.exists():
        return
    for sub in sorted(d for d in train_dir.iterdir() if d.is_dir()):
        for f in sorted(sub.iterdir()):
            if f.is_file() and f.suffix.lower() in CAPTION_EXTS:
                yield f"{sub.name}/{f.name}", f


def _snapshot_meta(zip_path: Path) -> dict[str, Any]:
    sid = zip_path.stem
    stat = zip_path.stat()
    file_count = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            file_count = sum(1 for n in z.namelist() if not n.endswith("/"))
    except zipfile.BadZipFile:
        file_count = -1
    return {
        "id": sid,
        "created_at": int(stat.st_mtime),
        "size": stat.st_size,
        "file_count": file_count,
    }


def create_snapshot(version_dir: Path) -> dict[str, Any]:
    """Pack all captions under train/ into a new zip. An empty train/ is allowed (produces an empty zip)."""
    train_dir = version_dir / "train"
    out_dir = snapshot_root(version_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sid = str(int(time.time() * 1000))
    zip_path = out_dir / f"{sid}.zip"
    # Extremely unlikely ts collision; suffix as a defensive fallback
    suffix = 0
    while zip_path.exists():
        suffix += 1
        zip_path = out_dir / f"{sid}_{suffix}.zip"
        if suffix > 9:
            raise SnapshotError(
                "Could not allocate a snapshot id; please retry",
                code="snapshot.id_collision",
            )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for arcname, src in _iter_caption_files(train_dir):
            z.write(src, arcname=arcname)

    return _snapshot_meta(zip_path)


def list_snapshots(version_dir: Path) -> list[dict[str, Any]]:
    out_dir = snapshot_root(version_dir)
    if not out_dir.exists():
        return []
    items = [
        _snapshot_meta(p)
        for p in out_dir.iterdir()
        if p.is_file() and p.suffix == ".zip"
    ]
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


def _resolve_snapshot(version_dir: Path, sid: str) -> Path:
    try:
        validate_path_component(sid)
    except ValueError as exc:
        raise SnapshotError(
            "Invalid snapshot id",
            code="snapshot.invalid", details={"id": sid, "reason": str(exc)},
            http_status=400,
        ) from exc
    try:
        p = safe_join(snapshot_root(version_dir), f"{sid}.zip")
    except ValueError as exc:
        raise SnapshotError(
            "Invalid snapshot id",
            code="snapshot.invalid", details={"id": sid, "reason": str(exc)},
            http_status=400,
        ) from exc
    if not p.exists():
        raise NotFoundError(
            "Snapshot not found",
            code="snapshot.not_found", details={"id": sid},
        )
    return p


def restore_snapshot(version_dir: Path, sid: str) -> dict[str, Any]:
    """Restore: delete all *.txt/*.json under train/ first, then unpack the snapshot. Image files are left untouched."""
    zip_path = _resolve_snapshot(version_dir, sid)
    train_dir = version_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # 1) Remove old captions
    removed = 0
    for sub in train_dir.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.is_file() and f.suffix.lower() in CAPTION_EXTS:
                f.unlink()
                removed += 1

    # 2) Unpack new ones (only into folders that already exist / will be created)
    written = 0
    skipped: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            name = member.filename
            if member.is_dir():
                continue
            if "/" not in name:
                skipped.append(name)
                continue
            folder, fname = name.split("/", 1)
            if Path(fname).suffix.lower() not in CAPTION_EXTS:
                skipped.append(name)
                continue
            try:
                target = safe_join(train_dir, folder, fname)
            except ValueError:
                skipped.append(name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            written += 1

    return {
        "id": sid,
        "removed_old": removed,
        "written": written,
        "skipped": skipped,
    }


def delete_snapshot(version_dir: Path, sid: str) -> None:
    zip_path = _resolve_snapshot(version_dir, sid)
    zip_path.unlink()
