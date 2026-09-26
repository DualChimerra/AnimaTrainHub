"""Model-loading infrastructure: prefix inference, safetensors reading, path
resolution, xformers / gradient checkpointing.

Extracted from the original runtime/anima_train.py L370-612 (ADR 0003 PR-A).
These are relatively low-level utils; the higher-level load_vae lives in
training.vae, and load_anima_model / load_text_encoders live in
families/anima/loader.

Public (used by the sister script):
- find_diffusion_pipe_root / resolve_path_best_effort / enable_xformers
- forward_with_optional_checkpoint (called by the train loop)

Internal:
- _strip_prefixes / _pick_best_prefix_remap -- automatic checkpoint key prefix inference
- _load_safetensors_state_dict / _load_weights_best_effort -- fault-tolerant loading
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


# ============================================================================
# xformers support
# ============================================================================

def enable_xformers(model):
    """Enable xformers memory-efficient attention on the model."""
    try:
        from xformers.ops import memory_efficient_attention  # noqa: F401
    except ImportError:
        logger.warning("xformers is not installed, skipping enable")
        return False

    enabled_count = 0
    module_switches = 0
    module_names = {
        cls.__module__
        for cls in type(model).__mro__
        if getattr(cls, "__module__", None)
    }
    # Since exec-load retirement, module identity is unique (multi-model PR-2a):
    # walking the MRO covers the real module names
    # (modeling.anima.cosmos_predict2_modeling / anima_modeling), no alias broadcast needed.
    for module_name in sorted(module_names):
        module = sys.modules.get(module_name)
        fn = getattr(module, "set_xformers_enabled", None) if module is not None else None
        if fn is None:
            continue
        try:
            if fn(True):
                module_switches += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to enable xformers module switch (%s): %s", module_name, exc)

    for name, module in model.named_modules():
        # Find attention modules and swap them in
        if hasattr(module, "set_use_memory_efficient_attention_xformers"):
            module.set_use_memory_efficient_attention_xformers(True)
            enabled_count += 1
        elif hasattr(module, "enable_xformers_memory_efficient_attention"):
            module.enable_xformers_memory_efficient_attention()
            enabled_count += 1

    if module_switches > 0 or enabled_count > 0:
        logger.info(
            "xformers enabled: module_switches=%d, module_hooks=%d",
            module_switches,
            enabled_count,
        )
        return True

    logger.warning("xformers is installed, but the current model has no xformers attention hook to enable")
    return False


# ============================================================================
# Model code / path resolution
# ============================================================================

def find_diffusion_pipe_root():
    """[deprecated shim] Returns the Anima model code directory (`modeling/anima/`).

    Model code now ships with the repo and goes through a normal import
    (exec-load and external diffusion-pipe checkout compatibility have been
    retired, multi-model PR-2a). The function name and return semantics are
    kept -- it's one of the 7 sister-script contract functions
    (docs/AGENTS.md Sec 3.2 "can add, cannot remove, cannot change signature");
    downstream still passes the return value through as
    `load_anima_model(..., repo_root=)`, but that parameter is now ignored.

    The `DIFFUSION_PIPE_ROOT` env var is no longer supported: a warning is
    logged once if it's set, and this check will be removed next release.
    """
    if os.environ.get("DIFFUSION_PIPE_ROOT"):
        logger.warning(
            "DIFFUSION_PIPE_ROOT is no longer supported (model code ships with "
            "the repo and uses a normal import); ignoring this variable"
        )
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "modeling" / "anima"


# ============================================================================
# Checkpoint key prefix inference + fault-tolerant loading
# ============================================================================

def _strip_prefixes(key: str, prefixes: list[str]) -> str:
    """Repeatedly strip prefixes (supports compound prefixes like module.model.)."""
    if not prefixes:
        return key
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p) :]
                changed = True
    return key


def _pick_best_prefix_remap(sd_keys: list[str], model_keys: set[str]) -> tuple[list[str], int]:
    """
    Pick the remap scheme, from a set of common prefix combinations, that
    matches the most model_keys.
    Returns (prefixes, matched_count).
    """
    candidates: list[tuple[str, list[str]]] = [
        ("none", []),
        ("net.", ["net."]),
        ("model.", ["model."]),
        ("module.", ["module."]),
        ("module.+model.", ["module.", "model."]),
        ("module.model.", ["module.model."]),
        ("diffusion_model.", ["diffusion_model."]),
        ("model.diffusion_model.", ["model.diffusion_model."]),
        ("transformer.", ["transformer."]),
        ("vae.", ["vae."]),
        ("first_stage_model.", ["first_stage_model."]),
        ("net.+model.", ["net.", "model."]),
        ("net.model.", ["net.model."]),
    ]

    best_prefixes: list[str] = []
    best_matched = -1
    for _name, prefixes in candidates:
        matched = 0
        for k in sd_keys:
            kk = _strip_prefixes(k, prefixes)
            if kk in model_keys:
                matched += 1
        if matched > best_matched:
            best_matched = matched
            best_prefixes = prefixes
    return best_prefixes, best_matched


def _load_safetensors_state_dict(path: Path) -> dict:
    from safetensors import safe_open

    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    return sd


def ensure_models_namespace(repo_root):
    """Make sure the models namespace is importable."""
    repo_root = Path(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(repo_root.parent) not in sys.path:
        sys.path.insert(0, str(repo_root.parent))


def resolve_path_best_effort(path_str: str, bases: list[Path]) -> str:
    """
    Try to resolve a relative path against several bases until it hits a path
    that actually exists.
    Mainly used so that models/* files can be found regardless of whether the
    process starts from the repo root or from the AnimaLoraToolkit directory.
    """
    if not path_str:
        return path_str

    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    # First try it as-is (relative to cwd)
    if p.exists():
        return str(p)

    # Try joining each base in turn
    for b in bases:
        if not b:
            continue
        try:
            cand = (Path(b) / p).resolve()
        except Exception:
            cand = Path(b) / p
        if cand.exists():
            return str(cand)

    # Common case: the config wrote AnimaLoraToolkit/xxx, but the process
    # already started from inside AnimaLoraToolkit
    parts = p.parts
    if parts and parts[0].lower() in ("animaloratoolkit", "anima_trainer", "anima-trainer"):
        p2 = Path(*parts[1:])
        if p2.exists():
            return str(p2)
        for b in bases:
            if not b:
                continue
            cand = Path(b) / p2
            if cand.exists():
                return str(cand)

    return path_str


def _load_weights_best_effort(model: torch.nn.Module, sd: dict, label: str) -> dict:
    """
    More robust weight loading:
    - automatically tries stripping common prefixes (model./module./...)
    - logs match rate, missing/unexpected keys
    - raises immediately if a critical module failed to load (avoids
      continuing to train while sampling produces pure noise)
    """
    model_keys = set(model.state_dict().keys())
    sd_keys = list(sd.keys())
    prefixes, matched = _pick_best_prefix_remap(sd_keys, model_keys)
    # Common path is no prefix remap. Reusing the original dict avoids building
    # another large key->tensor mapping while loading multi-GB checkpoints.
    remapped = sd if not prefixes else {_strip_prefixes(k, prefixes): v for k, v in sd.items()}

    incompatible = model.load_state_dict(remapped, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])

    matched_after = len(set(remapped.keys()) & model_keys)
    coverage = matched_after / max(1, len(model_keys))
    remap_name = "+".join(prefixes) if prefixes else "none"

    logger.info(
        f"{label} weight load: remap={remap_name}, matched {matched_after}/{len(model_keys)} ({coverage:.1%}), "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

    # Missing critical layers drives output to near-zero, so sampling would be pure noise
    critical_prefixes = ("x_embedder.", "blocks.", "final_layer.")
    critical_missing = [k for k in missing if k.startswith(critical_prefixes)]
    if coverage < 0.60 or len(critical_missing) > 0:
        preview_missing = ", ".join(critical_missing[:8])
        raise RuntimeError(
            f"{label} weights don't look like they loaded correctly (remap={remap_name}, coverage={coverage:.1%}). "
            f"Missing critical params: {preview_missing or 'N/A'}.\n"
            f"This usually means you picked the wrong .safetensors (not the full transformer/vae weights), "
            f"or the checkpoint key prefix doesn't match."
        )
    return {
        "remap": remap_name,
        "coverage": coverage,
        "missing": missing,
        "unexpected": unexpected,
    }


# Compatibility re-export (kept for the sister script/loop's existing import
# surface; moved to families/anima/forward.py in multi-model PR-2b)
from training.families.anima.forward import forward_with_optional_checkpoint  # noqa: E402,F401
