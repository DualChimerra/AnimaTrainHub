"""Domain-shared primitives: Field metadata helpers, the attention backend type, group order.

Shared by the model files training.py / generate.py / reg.py etc.

Note: does NOT use `from __future__ import annotations` -- under Pydantic v2 +
Python 3.12+, deferred evaluation would treat typing._SpecialForm as a schema
key and raise AttributeError.
"""
from typing import Any, Literal


def _meta(group: str, control: str = "auto", **extra: Any) -> dict[str, Any]:
    """Field's json_schema_extra payload -- the frontend uses this to bucket by group and show/hide via show_when."""
    return {"group": group, "control": control, **extra}


# --- Multi-model capability matrix (multi-model PR-3, 04-synthesis D5/D11) ---
# Single source of truth (config pipeline cut 1 / R3): runtime training/families'
# SPECS references this table directly (dependency direction runtime -> studio;
# this module is a zero-dependency pure-data leaf). studio/domain still doesn't
# import runtime (server startup doesn't depend on runtime's sys.path).
# tests/test_model_family_gating.py pins SPECS and this table to be identical
# (prevents them drifting apart as duplicated mirrors).
FAMILY_CAPABILITIES: dict[str, frozenset] = {
    "anima": frozenset({
        "navit", "sra", "leap", "compile_blocks",
        "caption_tag_ops", "online_text", "masked_loss", "block_swap",
    }),
    "krea2": frozenset({"masked_loss", "text_cache", "block_swap"}),
}

# Single source of truth for family default overlays (ModelSpec.config_defaults
# references this table directly, R3). Only overlays missing fields; never
# rewrites an explicit config. Consumer: the pydantic before-validator
# `_apply_family_config_defaults` (studio and the trainer now share this one
# loading path, R1).
FAMILY_CONFIG_DEFAULTS: dict[str, dict[str, Any]] = {
    "anima": {},
    "krea2": {
        "shuffle_caption": False,
        "keep_tokens": 0,
        "tag_dropout": 0.0,
        "text_encoder_cache": True,
        "attention_backend": "none",
        "timestep_sampling": "krea2_shift",
        "timestep_shift_resolution_aware": False,
        "sample_sampler_name": "euler",
        "sample_scheduler": "simple",
        "sample_infer_steps": 28,
        "sample_cfg_scale": 4.5,
    },
}

#: Value domain for the schema's model_family Literal (order == UI dropdown order)
MODEL_FAMILIES: tuple[str, ...] = tuple(FAMILY_CAPABILITIES)

# --- Sampling-side whitelist (multi-model P4-2, A13) ---
# Single source of truth (first item = family default): runtime
# SPECS[fam].sampling.samplers/schedulers references this table directly (R3,
# same dependency direction as the capability matrix).
# These two fields are a hard whitelist -- both families' sample_image raise
# at the entry point for values outside the list (anima: Comfy-parity check
# in families/anima/sampling.py; krea2: same in families/krea2/sampling.py),
# so a cross-family value errors out at the config layer (fail-fast).
FAMILY_SAMPLING: dict[str, dict[str, tuple[str, ...]]] = {
    "anima": {
        "samplers": ("er_sde", "dpmpp_3m_sde"),
        "schedulers": ("simple", "sgm_uniform"),
    },
    "krea2": {
        # Scheduler name matches ComfyUI: Comfy's "simple" scheduler, when
        # attached to Krea2 (ModelSamplingFlux), is exactly this project's
        # krea2 sigma convention -- shift is a model-level convention, not
        # part of the scheduler name (formerly called krea2_shift; legacy
        # values outside the Literal are merged back into "simple").
        "samplers": ("euler",),
        "schedulers": ("simple",),
    },
}

#: Families that existed before the Literal tightening (#256): legacy
#: sampler/scheduler values outside the whitelist are silently merged into
#: the family default per the #256 migration contract (grandfathered, same
#: treatment as D12's missing npz key / D13's untagged LoRA). Families born
#: in the Literal era have no legacy corpus -- any union value outside the
#: whitelist always errors (fail-fast with an actionable message), never
#: silently rewritten.
LEGACY_SAMPLING_FAMILIES: frozenset = frozenset({"anima"})

#: Options in timestep_sampling that are gated per family. All other options
#: are shared, generic loop implementations available to every family.
#: krea2_shift can technically run on any family (the loop supplies
#: token_counts via requires_token_counts), but its mu interpolation is
#: calibrated for K2 -- this only hides it in the UI as guidance, the backend
#: sets no gate (A1: identical code isn't restricted). When a 3rd
#: resolution-aware family reuses this strategy, just add it to the tuple.
TIMESTEP_SAMPLING_OPTION_FAMILIES: dict[str, tuple[str, ...]] = {
    "krea2_shift": ("krea2",),
}


def option_gates(option_families: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """Compile "option -> families that support it" into an option-level show_when expression map.

    The option-level counterpart of cap_gate: expanded at author time, zero
    new grammar for the three show_when evaluators. The result goes into the
    Field metadata's `option_show_when`; the frontend filters dropdown options
    by the current model_family. Options absent from the map are always visible.
    """
    return {
        opt: "||".join(f"model_family=={f}" for f in sorted(fams))
        for opt, fams in option_families.items()
    }


def sampling_option_gates(kind: str) -> dict[str, str]:
    """Derive the samplers / schedulers option-gate map from FAMILY_SAMPLING.

    Values supported by every family are ungated (always visible); the rest
    expand into an expression based on which families support them.
    """
    by_option: dict[str, list[str]] = {}
    for fam, spec in FAMILY_SAMPLING.items():
        for value in spec[kind]:
            by_option.setdefault(value, []).append(fam)
    all_families = set(FAMILY_SAMPLING)
    return option_gates({
        value: tuple(fams)
        for value, fams in by_option.items()
        if set(fams) != all_families
    })

#: Field -> required capability bit. Drives three lines of defense: show_when
#: (expanded via cap_gate at author time), the TrainingConfig validator, and
#: trainer bootstrap validation. A field only counts as "enabling" a
#: capability when its value is truthy/nonzero (a field left at its default,
#: off state is valid for any family).
FIELD_CAPABILITY_REQUIREMENTS: dict[str, str] = {
    "navit_packing": "navit",
    "sra_enabled": "sra",
    "leap_enabled": "leap",
    "masked_loss": "masked_loss",
    "shuffle_caption": "caption_tag_ops",
    "keep_tokens": "caption_tag_ops",
    "tag_dropout": "caption_tag_ops",
}


def cap_gate(capability: str) -> str:
    """Compile a capability bit into a show_when field-comparison expression (expanded at author time, zero new grammar at runtime).

    cap_gate("navit") -> "model_family==anima"; when a future 3rd family
    supports the capability, this automatically becomes
    "model_family==anima||model_family==foo" (only FAMILY_CAPABILITIES needs
    to change). Zero changes needed in any of the three show_when evaluators
    (frontend schema.ts / config_prune / YAML preview).
    """
    fams = sorted(f for f, caps in FAMILY_CAPABILITIES.items() if capability in caps)
    if not fams:
        raise ValueError(f"No model family supports the capability '{capability}'")
    return "||".join(f"model_family=={f}" for f in fams)


def capability_violations(model_family: str, values: dict) -> list[str]:
    """Return the list of field names that are "enabled but unsupported by the current family" (shared by the validator and bootstrap)."""
    caps = FAMILY_CAPABILITIES.get(str(model_family))
    if caps is None:
        return []  # an unknown family is already reported by Literal / resolve_family, no need to repeat here
    bad = []
    for field, cap in FIELD_CAPABILITY_REQUIREMENTS.items():
        if cap in caps:
            continue
        v = values.get(field)
        if isinstance(v, bool):
            active = v
        else:
            active = bool(v)  # nonzero int/float counts as enabled
        if active:
            bad.append(field)
    return bad


# One of three attention backends (replaces the old xformers / flash_attn bool pair)
AttentionBackend = Literal["none", "xformers", "flash_attn"]


# The frontend SchemaForm renders sections in this order.
# Each group: (key, label, default_collapsed). default_collapsed=True makes
# the frontend start it collapsed. The read-only model path group shows an
# "Auto - Global Settings" badge and is never collapsed.
GROUP_ORDER: list[tuple[str, str, bool]] = [
    ("model", "Model Path", False),
    ("dataset", "Dataset", False),
    ("caption", "Caption Processing", False),
    ("lora", "Network Settings", False),
    ("training", "Training Parameters", False),
    ("noise_augmentation", "Noise Augmentation", False),
    ("timestep_sampling", "Timestep Sampling", False),
    ("loss", "Loss", False),
    ("system", "System & Performance", False),
    ("output", "Output & Saving", False),
    ("sample", "Sampling", False),
    ("eval_validation", "Post-training Metrics", True),
    ("monitor", "Monitoring & Progress", False),
]
