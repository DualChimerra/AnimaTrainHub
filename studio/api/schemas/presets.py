"""/api/presets/* request BaseModels (extracted from server.py inline models in PR-5)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DuplicateRequest(BaseModel):
    new_name: str


class PresetExportBody(BaseModel):
    config: dict[str, Any]


class PresetImportBody(BaseModel):
    filename: str


class PresetImportFromPathBody(BaseModel):
    path: str
