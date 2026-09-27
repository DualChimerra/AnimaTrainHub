"""Anima family architecture definition: Anima(MiniTrainDIT) + LLMAdapter + attention backend state machine."""

from modeling.anima.anima_modeling import (  # noqa: F401
    Anima,
    set_attention_backend,
    set_flash_attn_enabled,
    set_xformers_enabled,
)
