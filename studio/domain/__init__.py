"""studio.domain — single source of truth for the pydantic schema (no I/O).

Split out of the original studio/schema.py (976 lines). Structure:
  - common.py      _meta() helper, AttentionBackend, GROUP_ORDER
  - migrations.py  migrate_legacy_save_keys
  - training.py    TrainingConfig (largest class, 643 lines; this restructuring PR
                    keeps one class per file)
  - lora.py        LoraEntry
  - xy_matrix.py   XYAxisType, XYAxisSpec, XYMatrixSpec, _check_axis_values
  - generate.py    GenerateConfig
  - reg.py         RegAiConfig

Note: does NOT use `from __future__ import annotations` — under Pydantic v2 +
Python 3.12+'s deferred evaluation, that would turn typing._SpecialForm into a
schema key and raise AttributeError.
"""
from .common import AttentionBackend, GROUP_ORDER, _meta
from .comfy_parity import (
    force_comfy_parity_runtime_config,
    is_exact_ksampler_parity_backend,
)
from .generate import GenerateConfig
from .lora import LoraEntry
from .migrations import migrate_legacy_save_keys, migrate_noise_enhancement_type
from .reg import RegAiConfig
from .training import TrainingConfig
from .xy_matrix import XYAxisSpec, XYAxisType, XYMatrixSpec, _check_axis_values

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
    "force_comfy_parity_runtime_config",
    "is_exact_ksampler_parity_backend",
    "migrate_legacy_save_keys",
    "migrate_noise_enhancement_type",
]
