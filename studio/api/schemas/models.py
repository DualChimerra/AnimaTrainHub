"""/api/models/* + /api/upscalers/* request BaseModels (extracted from server.py in PR-6 commit 2)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ModelDownloadRequest(BaseModel):
    model_id: str           # "anima_main" | "anima_vae" | "qwen3" | "t5_tokenizer"
    variant: Optional[str] = None  # used by anima_main / krea2_main, ignored otherwise


class FamilySwitchRequest(BaseModel):
    """Preview computation for switching the model family in a training config (multi-model P4-3).
    Pure computation, nothing is written to disk."""
    target: str          # target family id ("anima" / "krea2")
    config: dict         # current config dict (some fields may be missing)


class ModelSourceCandidateRequest(BaseModel):
    """Add/remove request for a unified model source candidate (docs/design/model-source-unification.md #6).

    POST adds: kind=download needs repo (single-file assets also need filename); kind=local
    needs path. DELETE removes: matched only by identity key (download=(repo, filename),
    local=path).
    """
    kind: str                # "download" | "local"
    repo: str = ""
    filename: str = ""
    path: str = ""
    extra: dict[str, str] = Field(default_factory=dict)


class UpscalerSelectRequest(BaseModel):
    label: str   # preset key, custom filename, or local absolute path
