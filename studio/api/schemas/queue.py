"""/api/queue request BaseModels (extracted from server.py in PR-6 commit 6)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class EnqueueRequest(BaseModel):
    config_name: str
    name: Optional[str] = None
    priority: int = 0
    # 0.17 P-B: planned start time (unix seconds). If set -> task is created as scheduled and
    # the supervisor promotes it to pending once due; if omitted -> immediately pending (old behavior).
    scheduled_at: Optional[float] = None


class ScheduleTrainingRequest(BaseModel):
    """Optional body for POST /api/projects/{pid}/versions/{vid}/queue (0.17 P-B)."""
    scheduled_at: Optional[float] = None


class ReorderRequest(BaseModel):
    ordered_ids: list[int]


class ImportRequest(BaseModel):
    payload: dict[str, Any]


class ExportOutputsBody(BaseModel):
    files: Optional[list[str]] = None


class DeleteOutputsBody(BaseModel):
    files: list[str]


class TaskNoteBody(BaseModel):
    """PUT /api/queue/{task_id}/note -- queue task note (v20 tasks.note).

    `note` as an empty string / whitespace-only -> clears the note (stored as NULL).
    """
    note: Optional[str] = None
