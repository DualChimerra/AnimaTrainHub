"""Listing of the data_exports directory (extracted from server.py in PR-6 commit 1).

1 route:
    GET /api/data-exports    list the zip / yaml / json files under DATA_EXPORTS
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..errors import _export_result
from ...paths import DATA_EXPORTS

router = APIRouter()


@router.get("/api/data-exports")
def list_data_exports() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    DATA_EXPORTS.mkdir(parents=True, exist_ok=True)
    for path in sorted(
        DATA_EXPORTS.iterdir(),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    ):
        if not path.is_file() or path.suffix.lower() not in {".zip", ".yaml", ".yml", ".json"}:
            continue
        try:
            items.append(_export_result(path))
        except OSError:
            continue
    return items
