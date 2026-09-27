"""Models root directory migration request model."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class ModelsRootMigrateRequest(BaseModel):
    """Target parent directory (absolute path; data lands in `target/models/`; target need not be empty).

    on_conflict: merge strategy when `target/models` already has data. If omitted, the backend
    returns 409 `models_root.target_conflict` (the frontend then shows a skip/overwrite/cancel
    choice and resends); "skip" -> only copy files missing from the target; "overwrite" -> files
    with the same name are overwritten by the copy from the current root.
    """
    target: str
    on_conflict: Optional[Literal["skip", "overwrite"]] = None
