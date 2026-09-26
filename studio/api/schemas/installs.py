"""Install / runtime endpoint request BaseModels (extracted from server.py in PR-6 commit 3).

Covers the wd14 / torch / flash-attention / llm-tagger domains. xformers has no request body.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class WD14InstallRequest(BaseModel):
    target: str = "auto"  # "auto" | "gpu" | "cpu" | "directml"


class TorchReinstallRequest(BaseModel):
    target: str = "auto"  # "auto" | "cu128" | "cu126" | "cu124" | "cu118" | "cpu"


class FlashAttnInstallRequest(BaseModel):
    url: Optional[str] = None  # None = auto-pick the best match from GitHub Releases


class LLMModelsRefreshRequest(BaseModel):
    # preset_id specifies which preset to update; if omitted, uses the current current_preset
    preset_id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: Optional[int] = None


class LLMConnectionTestRequest(BaseModel):
    preset_id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    timeout: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
