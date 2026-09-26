from __future__ import annotations

import pytest
import torch.nn as nn

from training.families.anima import ANIMA_SPEC
from training.families.anima.preset import ANIMA_PRESET


def test_preset_disables_conv():
    assert ANIMA_PRESET["enable_conv"] is False


def test_preset_targets_attention_and_mlp():
    names = ANIMA_PRESET["target_name"]
    for needle in ("q_proj", "k_proj", "v_proj", "output_proj", "mlp.layer1", "mlp.layer2"):
        assert any(needle in p for p in names), f"missing target {needle}"


def test_preset_excludes_llm_adapter():
    assert any("llm_adapter" in p for p in ANIMA_PRESET["exclude_name"])


def test_preset_uses_fnmatch():
    assert ANIMA_PRESET["use_fnmatch"] is True


def test_preset_prefix_matches_spec():
    assert ANIMA_PRESET["lora_prefix"] == "lora_unet"
    assert ANIMA_PRESET["lora_prefix"] == ANIMA_SPEC.lora.prefix


def test_lycoris_adapter_requires_explicit_preset():
    from utils.lycoris_adapter import LycorisAdapter

    adapter = LycorisAdapter(algo="lokr", rank=4, alpha=4.0, factor=8)
    with pytest.raises(ValueError):
        adapter.inject(nn.Linear(4, 4))


def test_lycoris_adapter_alias_kept():
    from utils.lycoris_adapter import AnimaLycorisAdapter, LycorisAdapter

    assert AnimaLycorisAdapter is LycorisAdapter
