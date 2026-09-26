"""ModelSpec -- declarative constants for a model family (multi-model support PR-1).

Pure data, frozen, no behavior. Design source:
docs/design/multi-model/03-interface-evolution.md Sec 2.1 (field set is the v1
frozen surface) + 04-synthesis.md D11/D12/D17.

At the PR-1 stage this file handles two things:
- Consolidating the latent spec (z_dim / stride / patch / alignment unit /
  latent2rgb coefficients) that was previously scattered across
  dataset / sampling / timestep_sampling / phases into a single source;
- Providing a fingerprint for the latent npz cache (the fingerprint is the
  **identity of the latent space**, not the family name: Anima and Krea 2
  both use ``wan21-f8c16`` -> the cache is shared across families, D6).

The ModelFamily behavior interface (eight methods + convert_lora_state_dict)
lands in PR-2b.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Union


@dataclass(frozen=True)
class ConstantShift:
    """Fixed timestep shift (Anima: 3.0, matches Comfy parity sampling_settings)."""

    shift: float


@dataclass(frozen=True)
class ResolutionAwareShift:
    """Resolution-aware dynamic shift (Krea 2: base 0.5 -> max 1.15, interpolated by image_seq_len).

    Only affects the sampling-side sigma schedule and schema default overlay;
    the training-side t sampling goes through the timestep_samplers plugin
    (04-synthesis D14).
    """

    base_shift: float
    max_shift: float
    base_image_seq_len: int
    max_image_seq_len: int


ShiftPolicy = Union[ConstantShift, ResolutionAwareShift]


@dataclass(frozen=True)
class LatentSpec:
    #: Identity of the latent space (not the family name); part of the npz cache fingerprint key
    fingerprint: str
    #: z_dim (VAE latent channel count)
    channels: int
    #: VAE spatial downsampling factor (f8)
    spatial_stride: int
    #: DiT patchify spatial side length (2 -> 1 token = 16x16 px)
    patch_spatial: int
    patch_temporal: int
    #: Video rejection flag (03 Sec 3.2): always False in v1, checked at registry registration
    temporal: bool
    #: latent2rgb preview linear projection [channels][3] (D17, taken from ComfyUI's LatentFormat)
    rgb_factors: tuple[tuple[float, float, float], ...]
    rgb_bias: tuple[float, float, float]

    @property
    def align_px(self) -> int:
        """Pixel alignment unit for buckets / sampling sizes = spatial_stride x patch_spatial."""
        return self.spatial_stride * self.patch_spatial


@dataclass(frozen=True)
class TextSpec:
    #: Anima=online (encoded on-the-fly each step); K2=cached_varlen (varlen precache, D3)
    strategy: Literal["online", "cached_varlen"]
    max_seq_len: int
    #: TE fingerprint (part of the cache key for cached_varlen; online families just record it)
    fingerprint: str


@dataclass(frozen=True)
class SamplingDefaults:
    #: sampler / scheduler allow-list and defaults (schema overlay, consumed on the sampling side)
    samplers: tuple[str, ...]
    schedulers: tuple[str, ...]
    default_sampler: str
    default_scheduler: str
    default_steps: int
    default_cfg: float
    shift_policy: ShiftPolicy


@dataclass(frozen=True)
class LoraOutputSpec:
    #: Saved key-name prefix (both families use "lora_unet", 04-synthesis Sec 7.1)
    prefix: str
    #: lycoris preset identifier (goes into ss_network_args.preset)
    preset_name: str


@dataclass(frozen=True)
class ModelSpec:
    #: registry key & schema enum value, never renamed (persists into yaml / LoRA metadata / resume state)
    family_id: str
    display_name: str
    #: non-rectified-flow rejection flag (03 Sec 3.3): only legal value in v1
    objective: Literal["rectified_flow"]
    latent: LatentSpec
    text: TextSpec
    sampling: SamplingDefaults
    capabilities: frozenset[str]
    lora: LoraOutputSpec
    #: per-family default overlay merged into the initial yaml when a version is created (baked in at author time)
    config_defaults: Mapping[str, Any] = field(default_factory=dict)


#: Capability vocabulary (03 Sec 2.4). Adding a word is free; removing/changing
#: the meaning of one needs 04-synthesis review.
KNOWN_CAPABILITIES = frozenset({
    "navit", "sra", "leap", "compile_blocks",
    "caption_tag_ops", "online_text", "text_cache", "masked_loss",
    "block_swap",
})


def validate_spec(spec: ModelSpec) -> None:
    """Self-consistency check run at registry registration (violation -> ValueError, kills the process at startup)."""
    unknown = spec.capabilities - KNOWN_CAPABILITIES
    if unknown:
        raise ValueError(
            f"ModelSpec[{spec.family_id}] unknown capability flags: {sorted(unknown)}"
        )
    if spec.text.strategy == "cached_varlen" and "caption_tag_ops" in spec.capabilities:
        # Cross-field invariant (03 Sec 2.4): the cache key is a hash of the
        # caption content; tag shuffle/dropout would make the caption drift
        # every step, permanently missing the cache.
        raise ValueError(
            f"ModelSpec[{spec.family_id}] cached_varlen and caption_tag_ops are mutually exclusive"
        )
    if spec.text.strategy == "cached_varlen" and "text_cache" not in spec.capabilities:
        raise ValueError(
            f"ModelSpec[{spec.family_id}] cached_varlen requires the text_cache capability"
        )
    if spec.text.strategy == "cached_varlen" and not spec.text.fingerprint.strip():
        raise ValueError(
            f"ModelSpec[{spec.family_id}] cached_varlen requires a non-empty TE fingerprint"
        )
    if spec.latent.temporal:
        raise ValueError(
            f"ModelSpec[{spec.family_id}] temporal=True: v1 does not support video families (03 Sec 3.2)"
        )
    if len(spec.latent.rgb_factors) != spec.latent.channels:
        raise ValueError(
            f"ModelSpec[{spec.family_id}] rgb_factors row count "
            f"{len(spec.latent.rgb_factors)} != channels {spec.latent.channels}"
        )
    if any(len(row) != 3 for row in spec.latent.rgb_factors) or len(spec.latent.rgb_bias) != 3:
        raise ValueError(f"ModelSpec[{spec.family_id}] rgb_factors/bias must be RGB triples")
    if spec.sampling.default_sampler not in spec.sampling.samplers:
        raise ValueError(
            f"ModelSpec[{spec.family_id}] default_sampler is not in the allow-list {spec.sampling.samplers}"
        )
    if spec.sampling.default_scheduler not in spec.sampling.schedulers:
        raise ValueError(
            f"ModelSpec[{spec.family_id}] default_scheduler is not in the allow-list {spec.sampling.schedulers}"
        )
