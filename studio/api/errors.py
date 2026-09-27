"""HTTPException wrapper helpers (extracted from server.py in PR-5).

Wraps the paths module's ValueError / invalid-path exceptions uniformly into
HTTP 400 / path validation, plus data-export filename resolution (used by
download endpoints like /api/preset/export).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..domain.errors import ConflictError, InvalidPathError, ValidationError
from ..paths import DATA_EXPORTS, safe_join, validate_path_component


def _safe_join_or_400(base: Path, *parts: str) -> Path:
    """HTTPException version of safe_join. Wraps ValueError as 400."""
    try:
        return safe_join(base, *parts)
    except ValueError as exc:
        raise InvalidPathError("Invalid path", details={"reason": str(exc)}) from exc


def _validate_component_or_400(name: str) -> None:
    """HTTPException version of validate_path_component (for plain name
    validation that doesn't need a join)."""
    try:
        validate_path_component(name)
    except ValueError as exc:
        raise InvalidPathError("Invalid path", details={"reason": str(exc)}) from exc


def _data_export_path(filename: str, suffixes: tuple[str, ...] = (".zip",)) -> Path:
    path = _safe_join_or_400(DATA_EXPORTS, filename)
    allowed = tuple(s.lower() for s in suffixes)
    if allowed and path.suffix.lower() not in allowed:
        label = " / ".join(allowed)
        raise ValidationError(
            f"Select a {label} file",
            code="file.ext_invalid", details={"types": label}, http_status=400,
        )
    return path


def _unique_data_export_path(
    filename: str, suffixes: tuple[str, ...] = (".zip",)
) -> Path:
    base = _data_export_path(filename, suffixes)
    if not base.exists():
        return base
    stem = base.stem
    suffix = base.suffix
    for i in range(2, 1000):
        candidate = _data_export_path(f"{stem}-{i}{suffix}", suffixes)
        if not candidate.exists():
            return candidate
    raise ConflictError(
        f'Too many files named "{filename}"; rename and try again',
        code="export.name_conflict", details={"name": filename},
    )


def _export_result(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "filename": path.name,
        "path": str(path),
        "size": st.st_size,
        "mtime": st.st_mtime,
    }
