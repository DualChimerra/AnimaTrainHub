"""Anima family's LoRA/LoKr target preset (multi-model PR-2b, moved from utils/lokr_preset.py).

"Algorithm is family-agnostic, target selection is family-specific"
(docs/design/multi-model/01 S7): LyCORIS's lokr/loha/lora math holds for any
stack of Linear layers; which layers to hit, what to exclude, and the saved
key-name prefix are family knowledge, and belong in families/.

After LycorisNetwork.apply_preset(ANIMA_PRESET), when injected into the Anima DiT:
- hits q/k/v/output_proj of self/cross attention
- hits layer1/layer2 of the MLP
- excludes llm_adapter (training this would break text comprehension)
- leaves norm layers untouched
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
}
