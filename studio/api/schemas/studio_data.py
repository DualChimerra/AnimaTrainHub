"""studio_data migration request model."""
from __future__ import annotations

from pydantic import BaseModel


class StudioDataMigrateRequest(BaseModel):
    """Target parent directory (absolute path; data lands in `target/studio_data/`; target need not be empty)."""
    target: str
