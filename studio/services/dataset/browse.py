"""Filesystem browsing: provides data for the frontend's "path picker" control.

Only returns directory info (doesn't read file contents). The allowed root
whitelist can be restricted by the caller, defaulting to under `REPO_ROOT`
only; when a path is passed explicitly, it's validated to be within the
whitelist or a valid location under an absolute path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ...paths import REPO_ROOT


from studio.domain.errors import (
    DomainError,
    ForbiddenError,
    InvalidPathError,
    NotFoundError,
    ValidationError,
)


class BrowseError(DomainError):
    """PR-2 C3 adds the DomainError base — the handler auto-translates it into the dual-write envelope."""
    default_code = "browse.error"


def list_dir(target: Path, *, allow_outside_repo: bool = False) -> dict[str, Any]:
    """List the subdirectories and filenames under the target directory.

    If target points to an existing file, fall back to its parent directory
    and tell the frontend which item to highlight via the `selected` field
    in the response (opening the picker to "the file's containing directory"
    is a common use case).

    All paths are returned in POSIX form (`/` separator), to avoid mixing
    Windows backslashes with the forward-slash style stored in yaml when the
    frontend concatenates paths.
    """
    target = target.resolve()
    if not allow_outside_repo:
        try:
            target.relative_to(REPO_ROOT.resolve())
        except ValueError:
            raise InvalidPathError("Invalid path")

    selected: str | None = None
    if target.exists() and not target.is_dir():
        selected = target.name
        target = target.parent

    if not target.exists():
        raise NotFoundError("Path not found", code="path.not_found")
    if not target.is_dir():
        raise ValidationError(
            "Path is not a directory",
            code="path.not_a_directory", http_status=400,
        )

    entries: list[dict[str, Any]] = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            entries.append({
                "name": child.name,
                "type": "dir" if is_dir else "file",
            })
    except PermissionError:
        raise ForbiddenError(
            "Permission denied", code="path.permission_denied",
        )

    parent = target.parent.as_posix() if target.parent != target else None
    return {
        "path": target.as_posix(),
        "parent": parent,
        "entries": entries,
        "selected": selected,
    }
