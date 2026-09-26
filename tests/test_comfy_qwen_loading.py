"""comfy_qwen3 encoder checkpoint key filtering (select_encoder_state_dict).

Instantiating the full 28-layer encoder is too heavy, so the unit test only
covers the pure function: a missing key raises hard, and an extra key (e.g.
lm_head.weight from a non-tied-embeddings variant) is filtered out cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "runtime"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from training.families.anima.comfy_qwen import select_encoder_state_dict  # noqa: E402


_EXPECTED = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.norm.weight",
]


def _full_state_dict() -> dict:
    return {k: torch.zeros(1) for k in _EXPECTED}


def test_exact_match_passes_through() -> None:
    sd = _full_state_dict()
    out = select_encoder_state_dict(_EXPECTED, sd, source="ckpt")
    assert set(out.keys()) == set(_EXPECTED)


def test_extra_lm_head_is_filtered_not_fatal() -> None:
    sd = _full_state_dict()
    sd["lm_head.weight"] = torch.zeros(1)
    out = select_encoder_state_dict(_EXPECTED, sd, source="ckpt")
    assert "lm_head.weight" not in out
    assert set(out.keys()) == set(_EXPECTED)


def test_missing_key_raises_with_source() -> None:
    sd = _full_state_dict()
    del sd["model.norm.weight"]
    with pytest.raises(RuntimeError, match="missing keys.*model.norm.weight"):
        select_encoder_state_dict(_EXPECTED, sd, source="ckpt")
