"""Test-generation schema — corresponds to the JSON config for runtime/anima_generate.py.

LoRA loading goes through inference_core.apply_loras — each LoRA is injected
independently, with rank/alpha read from ss_network_args, correctly merging
multiple LoRAs.

Note: does NOT use `from __future__ import annotations` — under Pydantic v2 +
Python 3.12+'s deferred evaluation, that would turn typing._SpecialForm into a
schema key and raise AttributeError.
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import FAMILY_SAMPLING, AttentionBackend
from .lora import LoraEntry
from .xy_matrix import XYMatrixSpec, _check_axis_values


def validate_sampling_for_family(
    family: str, sampler_name: str, scheduler: str,
) -> None:
    """Strict, per-family validation of the sampler for generate / reg_ai (multi-model P4-4).

    Unlike the training config, these two configs are per-task, throwaway
    JSON — there's no legacy corpus, so no grandfathering is needed: an
    out-of-family value is rejected immediately with actionable messaging
    (both families' sample_image raise on out-of-whitelist values at the
    entry point anyway; this just moves that check earlier, to a 422).
    """
    allowed = FAMILY_SAMPLING.get(family)
    if allowed is None:
        return  # An unknown family is already rejected by model_family's Literal validation
    for label, value, kind in (
        ("sampler_name", sampler_name, "samplers"),
        ("scheduler", scheduler, "schedulers"),
    ):
        if value not in allowed[kind]:
            raise ValueError(
                f"{label}='{value}' does not apply to model_family='{family}'"
                f" (available for this family: {', '.join(allowed[kind])})"
            )


class GenerateConfig(BaseModel):
    """Parameters for a test-generation task. Corresponds to the JSON config for runtime/anima_generate.py.

    LoRA loading goes through inference_core.apply_loras — each LoRA is
    injected independently, with rank/alpha read from ss_network_args,
    correctly merging multiple LoRAs.
    """

    model_config = ConfigDict(extra="forbid")

    # Model family (filled server-side from the request; the daemon dispatches loading and the sampling stack based on this)
    model_family: Literal["anima", "krea2"] = Field("anima")
    # Distilled inference base model (Krea2 Turbo: 8 steps / guidance 0 / mu fixed at 1.15).
    # Injected server-side after detecting the official Turbo path via the catalog variant purpose
    distilled: bool = Field(False)

    # Model paths (filled server-side from secrets)
    transformer_path: str = Field("models/diffusion_models/anima-base-v1.0.safetensors")
    vae_path: str = Field("models/vae/qwen_image_vae.safetensors")
    text_encoder_path: str = Field("models/text_encoders")
    t5_tokenizer_path: str = Field("models/t5_tokenizer")

    # Generation parameters
    prompts: list[str] = Field(
        default_factory=lambda: ["newest, safe, 1girl, masterpiece, best quality"],
        description="List of positive prompts (each prompt generates `count` images)",
    )
    negative_prompt: str = Field("")
    width: int = Field(1024, ge=256, le=4096)
    height: int = Field(1024, ge=256, le=4096)
    steps: int = Field(25, ge=1, le=150)
    cfg_scale: float = Field(4.0, ge=0.0, le=20.0)
    sampler_name: Literal["er_sde", "dpmpp_3m_sde", "euler"] = Field("er_sde")
    scheduler: Literal["simple", "sgm_uniform"] = Field("simple")
    count: int = Field(1, ge=1, le=32, description="Number of images to generate per prompt")
    seed: int = Field(0, description="Random seed (0 = random)")

    # LoRA (multiple LoRAs are injected independently; multiplier=scale controls each one's contribution weight)
    lora_configs: list[LoraEntry] = Field(
        default_factory=list,
        description="List of LoRAs (each injected independently, multiplier=scale)",
    )

    # XY matrix (None = plain single-image mode; when set, anima_generate.py takes the XY loop branch)
    xy_matrix: Optional[XYMatrixSpec] = Field(
        None,
        description="XY matrix parameters; when set, prompts is limited to a single entry and count=1 (to avoid a combinatorial explosion)",
    )

    # Runtime
    output_dir: str = Field("", description="Output directory (server fills in a tempdir, cleaned up when the task ends)")
    mixed_precision: str = Field("bf16")
    vae_precision: Literal["bf16", "fp32"] = Field(
        "bf16",
        description="VAE decode precision: bf16 matches ComfyUI's default on modern GPUs; fp32 is full precision (triggers an offload to free VRAM before decoding)",
    )
    vae_tiling: Literal["auto", "on", "off"] = Field(
        "auto",
        description="VAE tiled decode: auto = tile automatically when VRAM is tight (recommended); on = always tile "
                    "(saves VRAM, about 30% slower); "
                    "off = decode the whole image, only falls back to tiling on an actual OOM. On high-VRAM cards, "
                    "full-image decode at fp32 / high resolution can nearly fill VRAM and "
                    "trigger a system-memory fallback that makes a single decode take over a hundred seconds; auto avoids this",
    )
    attention_backend: AttentionBackend = Field(
        "flash_attn",
        description="Attention backend: none (SDPA) / xformers / flash_attn",
    )
    vram_policy: Literal["auto", "save_vram", "performance"] = Field(
        "auto",
        description="VRAM policy (applies to krea2): auto = decide whether the text encoder and DiT yield to each other based on free VRAM (recommended); "
                    "save_vram = force sequential execution, lowest peak usage (a 16GB card can run fp8), a few extra seconds of CPU↔GPU transfer per image; "
                    "performance = keep everything resident in VRAM, highest peak usage, zero transfers",
    )
    ram_guard: bool = Field(
        False,
        description="RAM/VRAM headroom guard: before loading a large model, estimate the required system RAM and free GPU VRAM "
                    "from the actual weight file size, and abort with an error if either is insufficient "
                    "(including cases where other processes are using the VRAM); "
                    "off by default (the file-size-based estimate is conservative and has a high false-reject rate on well-equipped machines); "
                    "when off, loading proceeds even under insufficient resources, which may trigger system-wide paging stalls",
    )

    @model_validator(mode="after")
    def _validate_sampler_family(self) -> "GenerateConfig":
        validate_sampling_for_family(
            self.model_family, self.sampler_name, self.scheduler)
        return self

    @model_validator(mode="after")
    def _validate_xy(self) -> "GenerateConfig":
        """XY 与 prompts/count 互斥；axis lora_index 必须指向已存在的 lora_configs。"""
        if self.xy_matrix is None:
            return self
        if len(self.prompts) > 1:
            raise ValueError("xy_matrix cannot be combined with multiple prompts (the grid would explode) — XY needs a single prompt")
        if self.count != 1:
            raise ValueError("xy_matrix cannot be combined with count>1 — in XY mode each (x,y) cell produces one image")
        for label, axis in (("x", self.xy_matrix.x), ("y", self.xy_matrix.y)):
            if axis is None:
                continue
            _check_axis_values(axis)
            if axis.lora_index is not None and axis.lora_index >= len(self.lora_configs):
                raise ValueError(
                    f"xy_matrix.{label}.lora_index={axis.lora_index} is out of range (only "
                    f"{len(self.lora_configs)} lora_configs are configured)"
                )
        return self
