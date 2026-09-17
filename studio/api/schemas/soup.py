"""Request bodies for the checkpoint-soup endpoints."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SoupInspectBody(BaseModel):
    paths: list[str] = Field(..., min_length=1, max_length=32)


class SoupInput(BaseModel):
    path: str
    #: Relative contribution. In ``average`` mode these are rescaled to sum to 1,
    #: so only their ratio matters; in ``sum`` mode they are used as-is.
    weight: float = 1.0


class SoupMergeBody(BaseModel):
    inputs: list[SoupInput] = Field(..., min_length=2, max_length=12)
    name: str
    method: Literal["average", "sum"] = "average"
    overwrite: bool = False
