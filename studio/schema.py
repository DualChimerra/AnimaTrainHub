"""Re-export shim -- the real definitions live in studio.domain.* (split out
in PR-2 from the original 976-line single file).

This module used to hold every pydantic schema (TrainingConfig, etc). After
the 0.11.0 restructure:
  - training / generate / reg / LoRA / XY matrix classes moved to
    `studio.domain.*` submodules
  - this file stays as a compat shim, so `from studio.schema import X` still
    works
  - new code should use `from studio.domain import X` directly
"""
from .domain import (
    GROUP_ORDER,
    AttentionBackend,
    GenerateConfig,
    LoraEntry,
    RegAiConfig,
    TrainingConfig,
    XYAxisSpec,
    XYAxisType,
    XYMatrixSpec,
    _check_axis_values,
    _meta,
    migrate_legacy_save_keys,
    migrate_noise_enhancement_type,
)

__all__ = [
    "AttentionBackend",
    "GROUP_ORDER",
    "GenerateConfig",
    "LoraEntry",
    "RegAiConfig",
    "TrainingConfig",
    "XYAxisSpec",
    "XYAxisType",
    "XYMatrixSpec",
    "_check_axis_values",
    "_meta",
    "migrate_legacy_save_keys",
    "migrate_noise_enhancement_type",
]
