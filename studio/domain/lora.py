"""LoRA loading parameters (shared by GenerateConfig and others).

Note: does NOT use `from __future__ import annotations` — under Pydantic v2 +
Python 3.12+'s deferred evaluation, that would turn typing._SpecialForm into a
schema key and raise AttributeError.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LoraEntry(BaseModel):
    """Loading parameters for a single LoRA. Shared by Generate / API to avoid a private definition in server.py."""

    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="Absolute path to the LoRA safetensors file")
    scale: float = Field(1.0, description="Contribution weight (multiplier); independent per LoRA when stacking multiple")
    # Associated version (chosen via the picker; absent for external files); the frontend uses vid to fetch the checkpoint list
    project_id: Optional[int] = Field(None, ge=1)
    version_id: Optional[int] = Field(None, ge=1)
