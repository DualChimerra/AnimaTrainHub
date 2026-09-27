"""Regression tests for the schema.attention_backend field."""
from __future__ import annotations

import pytest

from studio.schema import (
    AttentionBackend,  # noqa: F401  re-export smoke test
    GenerateConfig,
    RegAiConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# pydantic schema
# ---------------------------------------------------------------------------


def test_generate_config_default_is_flash_attn() -> None:
    """No field set -> defaults to flash_attn (consistent with the legacy flash_attn=True default)."""
    g = GenerateConfig(transformer_path="", vae_path="", text_encoder_path="")
    assert g.attention_backend == "flash_attn"


def test_generate_config_new_field() -> None:
    g = GenerateConfig(transformer_path="", vae_path="", text_encoder_path="",
                       attention_backend="xformers")
    assert g.attention_backend == "xformers"


@pytest.mark.parametrize("backend", ["none", "xformers", "flash_attn"])
def test_all_backends_validate(backend: str) -> None:
    """All three enum values validate."""
    g = GenerateConfig(transformer_path="", vae_path="", text_encoder_path="",
                       attention_backend=backend)  # type: ignore[arg-type]
    assert g.attention_backend == backend


def test_reg_ai_config_default_backend() -> None:
    r = RegAiConfig()
    assert r.attention_backend == "flash_attn"


def test_training_config_default_backend() -> None:
    t = TrainingConfig()
    assert t.attention_backend == "flash_attn"


def test_invalid_backend_rejected() -> None:
    """Invalid backend -> ValidationError."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GenerateConfig(transformer_path="", vae_path="", text_encoder_path="",
                       attention_backend="sdpa")  # type: ignore[arg-type]
