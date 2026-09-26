"""Adapter build function for the LyCORIS backend (lokr / loha / lora).

Extracted from the LycorisAdapter instantiation logic in phases/models.py (ADR 0003 PR-C).

Dispatched by the BUILDERS dict in training/adapters/__init__.py:
    BUILDERS["lokr"] = build
    BUILDERS["loha"] = build
    BUILDERS["lora"] = build

The actual adapter class lives in utils/lycoris_adapter.py; this file is just a
lightweight "read args -> call constructor" wrapper.
"""

from __future__ import annotations

from typing import Any

from training.adapters.protocol import AdapterProtocol


def build(args, *, preset: dict[str, Any]) -> AdapterProtocol:
    """Instantiate a LycorisAdapter from args and an explicit family preset."""
    from utils.lycoris_adapter import LycorisAdapter
    return LycorisAdapter(
        preset=preset,
        algo=args.lora_type,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        factor=args.lokr_factor,
        dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
        rank_dropout=float(getattr(args, "lora_rank_dropout", 0.0) or 0.0),
        module_dropout=float(getattr(args, "lora_module_dropout", 0.0) or 0.0),
        weight_decompose=bool(getattr(args, "lora_dora", False)),
        rs_lora=bool(getattr(args, "lora_rs", False)),
        lora_reg_dims=getattr(args, "lora_reg_dims", None) or None,
    )
