"""File browsing / dataset scanning / thumbnails (extracted from server.py in PR-5).

3 routes:
    GET /api/datasets             scan a dataset directory (default = repo_root/dataset)
    GET /api/browse               directory browsing (for PathPicker, allow_outside_repo=True)
    GET /api/datasets/thumbnail   single-image thumbnail (REPO_ROOT only)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse

from .. import errors as _errors
from ...domain.errors import InvalidPathError, NotFoundError
from ...services.dataset import browse, scan as datasets
from ...infrastructure.paths import REPO_ROOT

router = APIRouter()


@router.get("/api/datasets")
def get_datasets(path: str = "") -> dict[str, Any]:
    """Scan a dataset directory. `?path=` sets the root; default = repo_root/dataset."""
    root = Path(path) if path else REPO_ROOT / "dataset"
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    return datasets.scan_dataset_root(root)


@router.get("/api/browse")
def browse_dir(path: str = "") -> dict[str, Any]:
    """Directory browsing (for the frontend path picker). Default = REPO_ROOT.

    PathPicker is designed for users to pick external model paths (cloud machines
    keep models on a separate data disk), so allow_outside_repo=True here; the
    security boundary lives in list_dir itself (it only reads entry names + types,
    never returns file contents).
    """
    target = Path(path) if path else REPO_ROOT
    if not target.is_absolute():
        target = (REPO_ROOT / target).resolve()
    return browse.list_dir(target, allow_outside_repo=True)


@router.get("/api/datasets/thumbnail")
def get_dataset_thumbnail(folder: str, name: str) -> FileResponse:
    """Return a dataset thumbnail (actually the original image; the frontend scales it with CSS).

    `folder` can be an absolute or relative path (picked by the user in the
    dataset browser), `name` must be a single filename (no separators). The
    resolved path must stay inside REPO_ROOT.
    """
    _errors._validate_component_or_400(name)
    p = (Path(folder) / name).resolve()
    try:
        p.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise InvalidPathError("Invalid path", http_status=403) from None
    if not p.exists() or p.suffix.lower() not in datasets.IMAGE_EXTS:
        raise NotFoundError(
            "Thumbnail not found", code="dataset.thumbnail_not_found",
        )
    return FileResponse(p)
