"""Declarative constants for the Anima family (PR-1 only has ANIMA_SPEC; the behavior adapter lands with PR-2b)."""

from __future__ import annotations

# The capability matrix / family defaults / sampling whitelist have a single
# source in studio/domain/common.py (config pipeline knife 1 / R3): studio
# does not import runtime (the server's sys.path has no runtime/), while the
# reverse runtime -> studio direction is the existing dependency direction
# (bootstrap already imports studio.schema); common.py is a zero-dependency
# pure-data leaf module reachable from all three scenarios: server, bare CLI,
# and tests.
from studio.domain.common import (
    FAMILY_CAPABILITIES,
    FAMILY_CONFIG_DEFAULTS,
    FAMILY_SAMPLING,
)

# Relative import: the studio server imports this module indirectly via
# `runtime.training.dataset`, and over there sys.path has no runtime/, so an
# absolute `training.*` import would raise ModuleNotFoundError.
from ..latent_spaces import WAN21_F8C16
from ..spec import (
    ConstantShift,
    LoraOutputSpec,
    ModelSpec,
    SamplingDefaults,
    TextSpec,
)


ANIMA_SPEC = ModelSpec(
    family_id="anima",
    display_name="Anima",
    objective="rectified_flow",
    # Qwen-Image VAE = the Wan2.1 latent space; shares the same instance as
    # Krea 2 -> latent cache sharing across families (D6) is a structural fact.
    latent=WAN21_F8C16,
    text=TextSpec(
        # Online encoding per step (Qwen3-0.6B final layer + T5 IDs into LLMAdapter), no text cache
        strategy="online",
        max_seq_len=512,
        fingerprint="anima-qwen3-0.6b-t5xxl",
    ),
    sampling=SamplingDefaults(
        # Whitelist matches sampling.py's current Comfy KSampler parity; first item = family default
        samplers=FAMILY_SAMPLING["anima"]["samplers"],
        schedulers=FAMILY_SAMPLING["anima"]["schedulers"],
        default_sampler=FAMILY_SAMPLING["anima"]["samplers"][0],
        default_scheduler=FAMILY_SAMPLING["anima"]["schedulers"][0],
        default_steps=25,
        default_cfg=4.0,
        shift_policy=ConstantShift(shift=3.0),
    ),
    # D5: Anima = full capability set minus text_cache
    capabilities=FAMILY_CAPABILITIES["anima"],
    lora=LoraOutputSpec(prefix="lora_unet", preset_name="anima_full"),
    config_defaults=FAMILY_CONFIG_DEFAULTS["anima"],
)
