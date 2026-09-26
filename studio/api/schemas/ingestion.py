"""download / upload / preprocess BaseModels (extracted from server.py in PR-6.5 commit 3)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ...services.preprocess import core as preprocess_svc


class DownloadRequest(BaseModel):
    tag: str
    count: int = 20
    api_source: str = "gelbooru"


class EstimateRequest(BaseModel):
    tag: str
    api_source: str = "gelbooru"


class UploadFromPathBody(BaseModel):
    path: str


class PreprocessStartRequest(BaseModel):
    mode: str = "all"  # all | selected | all_force
    names: Optional[list[str]] = None
    model: str = preprocess_svc.DEFAULT_MODEL
    tile_size: int = preprocess_svc.DEFAULT_TILE_SIZE
    tile_pad: int = preprocess_svc.DEFAULT_TILE_PAD
    device: str = preprocess_svc.DEFAULT_DEVICE
    # target_area=None uses the plain 4x model; non-None uses smart mode (skip the model when
    # already big enough, otherwise LANCZOS-resize to the target)
    target_area: Optional[int] = preprocess_svc.DEFAULT_TARGET_AREA


class PreprocessRestoreRequest(BaseModel):
    """Restore a processed image: removes the manifest entry + deletes preprocess/{name} PNG.

    After restoring, the image goes back to the "implicit original" state -- downstream
    resolvers point back to download/. See ADR 0004.
    """
    names: list[str]


class CropRect(BaseModel):
    """Normalized crop rect [0..1]^4. x/y = top-left corner, w/h = width/height."""
    x: float
    y: float
    w: float
    h: float
    label: Optional[str] = None


class PreprocessCropRequest(BaseModel):
    """Crop job input: source filename -> one or more normalized rects.

    The source filename is the current filename under preprocess/ (or falls back to the
    download/ filename if preprocess/ has no matching entry). Each rect produces one PNG:
    N=1 overwrites stem.png; N>1 outputs stem_c0.png / stem_c1.png / ... and deletes the
    original stem.png.
    """
    crops: dict[str, list[CropRect]]
