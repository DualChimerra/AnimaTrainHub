"""Prior generation schema — corresponds to the JSON config for runtime/anima_reg_ai.py.

Design comes from DreamBooth prior preservation: the training loss sees both
"what the LoRA has learned" and "what the base model originally looked like"
at the same time, so the LoRA only learns the difference. **No LoRA is
attached** here — attaching one would overwrite the very prior we're trying to
preserve.

Note: does NOT use `from __future__ import annotations` — under Pydantic v2 +
Python 3.12+'s deferred evaluation, that would turn typing._SpecialForm into a
schema key and raise AttributeError.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import AttentionBackend
from .generate import validate_sampling_for_family


class RegAiConfig(BaseModel):
    """JSON config for prior generation (corresponds to runtime/anima_reg_ai.py).

    Design comes from DreamBooth prior preservation: the training loss sees
    both "what the LoRA has learned" and "what the base model originally
    looked like" at the same time, so the LoRA only learns the difference.
    **No LoRA is attached** here — attaching one would overwrite the very
    prior we're trying to preserve.
    """

    model_config = ConfigDict(extra="forbid")

    # Model family (read server-side from the version config — prior generation
    # is a version-level operation; the family follows that version's training
    # config, it's not a per-request user choice)
    model_family: Literal["anima", "krea2"] = Field("anima")

    # Model paths (filled server-side from secrets)
    transformer_path: str = Field("")
    vae_path: str = Field("")
    text_encoder_path: str = Field("")
    t5_tokenizer_path: str = Field("")

    # Data directories (filled server-side)
    train_dir: str = Field("")
    reg_dir: str = Field("")

    # Generation controls
    excluded_tags: list[str] = Field(
        default_factory=list,
        description="Excluded tags (not included in the prompt)",
    )
    negative_prompt: str = Field("")
    width: int = Field(1024, ge=256, le=4096)
    height: int = Field(1024, ge=256, le=4096)
    steps: int = Field(25, ge=1, le=150)
    cfg_scale: float = Field(4.0, ge=0.0, le=20.0)
    sampler_name: Literal["er_sde", "dpmpp_3m_sde", "euler"] = Field("er_sde")
    scheduler: Literal["simple", "sgm_uniform"] = Field("simple")
    seed: int = Field(0, description="Random seed (0 = random)")
    incremental: bool = Field(
        False,
        description="Fill-in mode: skip images already present in the reg subfolder whose filenames start with train_stem (for resuming after a restart)",
    )
    # This fork: the reg subfolder's Kohya repeat prefix is configurable (independent of the train repeat)
    repeat: int = Field(
        1,
        ge=1,
        le=100,
        description=(
            "Kohya-style repeat prefix (N_label) for the reg subfolder. The reg set's "
            "repeat is independent of the train repeat; defaults to 1 (DreamBooth "
            "standard: each reg image is seen once per epoch)."
        ),
    )
    mixed_precision: str = Field("bf16")
    attention_backend: AttentionBackend = Field(
        "flash_attn",
        description="Attention backend: none (SDPA) / xformers / flash_attn",
    )

    @model_validator(mode="after")
    def _validate_sampler_family(self) -> "RegAiConfig":
        validate_sampling_for_family(
            self.model_family, self.sampler_name, self.scheduler)
        return self
