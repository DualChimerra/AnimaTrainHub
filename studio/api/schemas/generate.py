"""/api/generate request BaseModels (extracted from server.py in PR-6 commit 5)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel

from ...domain import AttentionBackend, LoraEntry, XYMatrixSpec


class GenerateRequest(BaseModel):
    prompts: list[str] = ["newest, safe, 1girl, masterpiece, best quality"]
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg_scale: float = 4.0
    sampler_name: Literal["er_sde", "dpmpp_3m_sde", "euler"] = "er_sde"
    scheduler: Literal["simple", "sgm_uniform"] = "simple"
    count: int = 1
    seed: int = 0
    lora_configs: list[LoraEntry] = []
    mixed_precision: str = "bf16"
    # Model family the base model belongs to (multi-model P4-4): determines path resolution /
    # daemon loading and the sampling stack; sampler is validated against a per-family whitelist
    # (GenerateConfig validator, cross-family -> 422)
    model_family: Literal["anima", "krea2"] = "anima"
    # Base model to use for this generation (official variant key or a registered local custom
    # path); None -> use the family's "selected" from Settings. Only swaps the transformer weights.
    base_model: Optional[str] = None
    # Text encoder variant to use for this generation (applies to krea2): None -> follow the
    # download center's selected TE (selected_te); explicit "bf16"/"fp8" overrides it for this
    # request (symmetric with base_model semantics).
    text_encoder: Optional[Literal["bf16", "fp8"]] = None
    # commit C: attention_backend now defaults from secrets.generate.attention_backend and the
    # frontend Generate page no longer sends this field; kept Optional for old-client compat /
    # ad-hoc overrides.
    attention_backend: Optional[AttentionBackend] = None
    # XY matrix: None = single-image mode; when set, the schema forces a single prompt + count=1
    xy_matrix: Optional[XYMatrixSpec] = None
    # GenerateParamsSnapshot dict built by the frontend (a prefs view: prompts/loras/xy_draft/
    # dataset_pick etc). The server doesn't interpret its structure, just passes it through to
    # the daemon -> on image_done it's stuffed into the encrypted cache payload header. Returned
    # to the frontend via /api/generate/cache/index as CacheEntry.params for refill. The
    # save_test_images=true disk-write path also uses this same snapshot when writing PNG
    # anima_params metadata.
    params_snapshot: Optional[dict[str, Any]] = None
