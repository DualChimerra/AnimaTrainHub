"""files/curation/duplicates BaseModels (extracted from server.py in PR-6.5 commit 4)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ...services.preprocess import duplicates as duplicate_finder


class DeleteFilesRequest(BaseModel):
    names: list[str]


class CopyRequest(BaseModel):
    files: list[str]
    dest_folder: str


class RemoveRequest(BaseModel):
    folder: str
    files: list[str]


class CopyValidationRequest(BaseModel):
    """download -> validation copy (lands in the fixed validation/1_data/, no dest_folder)."""
    files: list[str]


class ValidationItem(BaseModel):
    folder: str
    name: str


class RemoveValidationRequest(BaseModel):
    """Exact delete by (folder, name) -- a multi-select may span different auto-split repeat folders."""
    items: list[ValidationItem]


class FolderOp(BaseModel):
    op: str  # "create" | "rename" | "delete"
    name: str
    new_name: Optional[str] = None


class DuplicateScanRequest(BaseModel):
    # The UI only exposes match scope + sensitivity; other threshold/perf params are fixed as duplicates.DEFAULT_*.
    match_scope: str = "both"
    sensitivity: str = duplicate_finder.DEFAULT_SENSITIVITY


class DuplicateApplyRequest(BaseModel):
    names: list[str]
