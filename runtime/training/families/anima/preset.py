"""Anima family's LoRA/LoKr target preset (multi-model PR-2b, moved from utils/lokr_preset.py).

"Algorithm is family-agnostic, target selection is family-specific"
(docs/design/multi-model/01 S7): LyCORIS's lokr/loha/lora math holds for any
stack of Linear layers; which layers to hit, what to exclude, and the saved
key-name prefix are family knowledge, and belong in families/.

After LycorisNetwork.apply_preset(ANIMA_PRESET), when injected into the Anima DiT:
- hits q/k/v/output_proj of self/cross attention
- hits layer1/layer2 of the MLP
- excludes llm_adapter (training this would break text comprehension)
- trains q/k normalization only when train_norm is enabled
- saved key-name prefix is lora_unet_* (ComfyUI ecosystem convention, matches spec.lora.prefix)
"""
from __future__ import annotations

from typing import Any

ANIMA_PRESET: dict[str, Any] = {
    "enable_conv": False,                    # The Anima DiT backbone + TE + LLM Adapter are all nn.Linear
    "target_module": [],                     # Don't match by module class
    "target_name": [
        "*q_proj", "*k_proj", "*v_proj", "*output_proj",
        "*mlp.layer1", "*mlp.layer2",
    ],
    "exclude_name": ["llm_adapter*"],
    "use_fnmatch": True,                     # Enable fnmatch (accepts * wildcards)
    "lora_prefix": "lora_unet",              # Keeps compatibility with ComfyUI's existing loading flow
    "module_algo_map": {},                   # Reserved for future per-module algorithm overrides
    "name_algo_map": {},                     # Same as above, overridden by layer name
    # Adapter-owned extension, removed before passing the preset to LyCORIS.
    "train_norm_names": [
        "blocks.*.self_attn.q_norm", "blocks.*.self_attn.k_norm",
        "blocks.*.cross_attn.q_norm", "blocks.*.cross_attn.k_norm",
    ],
}

ANIMA_TEXT_PRESET: dict[str, Any] = {
    # HF and Comfy wrappers both hold the backbone here. Strip this wrapper
    # so saved keys match ComfyUI's lora_te_layers_* mapping.
    "root_module": "model",
    "enable_conv": False,
    "target_module": [],
    "target_name": ["*self_attn.q_proj", "*self_attn.k_proj", "*self_attn.v_proj",
                    "*self_attn.o_proj", "*mlp.gate_proj", "*mlp.up_proj", "*mlp.down_proj"],
    "exclude_name": [],
    "use_fnmatch": True,
    "lora_prefix": "lora_te",
    "module_algo_map": {},
    "name_algo_map": {},
}
