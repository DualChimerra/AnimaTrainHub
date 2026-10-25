"""/api/library/datasets request bodies (shared dataset library)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class DatasetCreate(BaseModel):
    title: str
    note: Optional[str] = None
    #: Optionally start from a project's own download/ images (hardlinked when possible).
    from_project_id: Optional[int] = None


class DatasetUpdate(BaseModel):
    title: Optional[str] = None
    note: Optional[str] = None


class DatasetImport(BaseModel):
    project_id: int


class VersionDatasetsPut(BaseModel):
    dataset_ids: list[int]
