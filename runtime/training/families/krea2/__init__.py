"""Krea2 ModelSpec and ModelFamily implementation (multi-model Phase 3)."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch

from .preset import KREA2_PRESET
from .sampling import (
    KREA2_BASE_IMAGE_SEQ_LEN,
    KREA2_BASE_SHIFT,
    KREA2_MAX_IMAGE_SEQ_LEN,
    KREA2_MAX_SHIFT,
    KREA2_RAW_GUIDANCE,
    KREA2_RAW_STEPS,
    KREA2_SAMPLER,
    KREA2_SCHEDULER,
    Krea2SamplingCondition,
    prepare_sampling_condition,
    resolve_sampling_settings,
    sample_image,
)
from .text_encoding import (
    KREA2_TEXT_FINGERPRINT,
    Krea2TextCondition,
    Krea2TextStack,
    load_krea2_text_stack,
)
from ..latent_spaces import WAN21_F8C16
from ..spec import (
    LoraOutputSpec,
    ModelSpec,
    ConstantShift,
    SamplingDefaults,
    TextSpec,
)
# Single source of data (Cut 1 / R3): see the matching comment in anima/__init__.py for the dependency direction
from studio.domain.common import (
    FAMILY_CAPABILITIES,
    FAMILY_CONFIG_DEFAULTS,
    FAMILY_SAMPLING,
)


logger = logging.getLogger(__name__)


_GIB = 1024 ** 3
#: VRAM needed to move TE onto the GPU: fp16 weights 8.9GB + embed table cast to fp32 transiently ~1.5GB + buffer
_TE_LOAD_NEED_BYTES = int(11 * _GIB)
#: Official fp8_scaled single-file TE: Linear ~5GB + embed still fp16 (fp32 cast transient unchanged)
_TE_LOAD_NEED_BYTES_FP8 = int(7 * _GIB)


def _cuda_free_bytes(device) -> int | None:
    """Currently free VRAM on the target CUDA device; returns None for non-CUDA / a failed query."""
    try:
        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(dev)
        return int(free)
    except Exception:
        return None


def _sampling_headroom_bytes(height: int, width: int) -> int:
    # Sampling headroom is a rough estimate, not per-model accounting (user
    # decision in orchestration doc D1): 2GB base + 3GB x area ratio
    # (1024^2 -> 5GB, 1536^2 -> 8.75GB)
    area_ratio = (height * width) / (1024 * 1024)
    return int((2.0 + 3.0 * area_ratio) * _GIB)


def _should_yield_dit(policy: str, device,
                      need_bytes: int = _TE_LOAD_NEED_BYTES) -> bool:
    """Whether DiT should step down to CPU before the TE needs to move onto the
    GPU for encoding (comfy free_memory's "only yield if it won't fit"
    semantics). ``need_bytes`` depends on TE precision (halved for fp8)."""
    if policy == "performance":
        return False
    if policy == "save_vram":
        return True
    free = _cuda_free_bytes(device)
    return free is not None and free < need_bytes


def _should_offload_te(policy: str, device, height: int, width: int,
                       dit_yielded: bool) -> bool:
    """Whether to offload the TE to CPU before sampling. auto only offloads
    when sampling headroom is insufficient -- on 32GB fp8, all three fit
    resident with plenty of free VRAM, so from then on it's zero moves."""
    if policy == "performance":
        return False
    if policy == "save_vram":
        return True
    if dit_yielded:
        # DiT already yielded = VRAM can't fit all three resident, so DiT
        # must have exclusive use during sampling; also, DiT is still on CPU
        # right now so "free" would read artificially high and can't be used
        # as a signal
        return True
    free = _cuda_free_bytes(device)
    return free is None or free < _sampling_headroom_bytes(height, width)


KREA2_SPEC = ModelSpec(
    family_id="krea2",
    display_name="Krea 2",
    objective="rectified_flow",
    # Shares the Qwen-Image VAE / Wan2.1 latent space with Anima -- references the same instance (D6).
    latent=WAN21_F8C16,
    text=TextSpec(
        strategy="cached_varlen",
        max_seq_len=512,
        fingerprint=KREA2_TEXT_FINGERPRINT,
    ),
    sampling=SamplingDefaults(
        # Allow-list's single source is studio/domain/common.py FAMILY_SAMPLING
        # (Cut 1 / R3); constants like KREA2_SAMPLER still live in sampling.py
        # (shared with the generation-side parity code); consistency with the
        # single-source data is locked down by tests/test_model_family_gating.py
        samplers=FAMILY_SAMPLING["krea2"]["samplers"],
        schedulers=FAMILY_SAMPLING["krea2"]["schedulers"],
        default_sampler=KREA2_SAMPLER,
        default_scheduler=KREA2_SCHEDULER,
        default_steps=KREA2_RAW_STEPS,
        default_cfg=KREA2_RAW_GUIDANCE,
        # Comfy-parity convention: fixed mu=1.15 (matches ComfyUI's
        # ModelSamplingFlux; note this value is **mu (pre-exp)**, unlike
        # Anima's ConstantShift, which is a direct factor -- shift_policy's
        # meaning is up to the family to interpret). diffusers' resolution-
        # aware dynamic mu is kept as the non-default
        # build_krea2_sigmas(dynamic_mu=True) path.
        shift_policy=ConstantShift(shift=1.15),
    ),
    capabilities=FAMILY_CAPABILITIES["krea2"],
    lora=LoraOutputSpec(prefix="lora_unet", preset_name="krea2_full"),
    config_defaults=FAMILY_CONFIG_DEFAULTS["krea2"],
)


class Krea2Family:
    spec = KREA2_SPEC

    def load_dit(self, path, device, dtype, *,
                 attention_backend: str = "flash_attn", repo_root=None,
                 purpose: str = "train", blocks_to_swap: int = 0):
        from training.families.krea2.loader import load_krea2_model

        if attention_backend != "none":
            logger.info(
                "Krea2 currently always uses PyTorch SDPA; ignoring attention_backend=%s",
                attention_backend,
            )
        return load_krea2_model(
            path, device, dtype, purpose=purpose, blocks_to_swap=blocks_to_swap,
        )

    def swappable_blocks(self, *, checkpoint_path: str | None = None) -> int:
        """Upper bound on the number of swappable layers (= DiT backbone layer count).

        Needed by the block-swap preflight search when it looks for a
        recommended value. Like ``swapped_param_ratio``, this is a duck-typed
        optional method: if a family doesn't implement it, preflight is
        skipped (falls back to the old behavior).

        ``checkpoint_path`` is a cross-family protocol parameter (anima's
        layer count is determined by the checkpoint); krea2's structure is
        fixed (KREA2_CONFIG), so it's unused here.
        """
        del checkpoint_path
        from modeling.krea2 import KREA2_CONFIG

        return int(KREA2_CONFIG.layers)

    def swapped_param_ratio(self, blocks_to_swap: int, *,
                            checkpoint_path: str | None = None) -> float:
        """Fraction of the full model's parameters that the swapped-out layers
        account for (used for the VRAM budget discount; see the same-named
        function in loader).

        Deliberately a ratio, not a byte count -- fp8 and bf16 file sizes
        differ by 2x, and a byte-based discount would blow through the safety
        margin in the fp8 case.

        ``checkpoint_path`` is a cross-family protocol parameter (anima uses
        it to distinguish the 28/36-layer versions); krea2's structure is
        fixed (KREA2_CONFIG), so it's unused here.
        """
        del checkpoint_path
        from training.families.krea2.loader import swapped_param_ratio

        return swapped_param_ratio(blocks_to_swap)

    def load_vae(self, path, device, dtype, *, tiling: str = "auto"):
        from training.vae import load_vae

        return load_vae(path, device, dtype, None, tiling=tiling)

    def load_text(self, text_encoder_path, device, dtype, *,
                  t5_tokenizer_path: str = "", comfy_qwen: bool = False,
                  t5_fast: bool = False, purpose: str = "train",
                  cache_enabled: bool = True):
        if purpose == "generate":
            # Comfy parity (sd.py:258): for generation, TE is always fp16
            # storage + fp32 compute (text_encoder_dtype defaults to fp16 +
            # set_model_compute_dtype fp32), ignoring the caller's dtype --
            # the same fixed behavior as TE offload. TE precision is
            # determined by the directory shape text_encoder_path points to
            # (HF sharded = bf16 -> fp16; comfy single-file = official
            # fp8_scaled kept resident as-is).
            return load_krea2_text_stack(
                text_encoder_path,
                device=device,
                dtype=torch.float16,
                compute_dtype=torch.float32,
                cache_enabled=cache_enabled,
            )
        return load_krea2_text_stack(
            text_encoder_path,
            device=device,
            dtype=dtype,
            cache_enabled=cache_enabled,
        )

    def prepare_text_cache(self, captions: Iterable[str],
                           extra_prompts: Iterable[str], *, cache_entries=(),
                           cache_root=None, text=None, device=None,
                           dtype=None) -> None:
        if text is None:
            raise ValueError("Krea2 prepare_text_cache requires a Krea2TextStack")
        text.prepare_text_cache(
            captions,
            extra_prompts,
            cache_entries=cache_entries,
            cache_root=cache_root,
        )

    def encode_text_for_batch(self, text, dit, captions, device, dtype, *,
                              comfy_encoding: bool = True,
                              kv_trim: bool = True):
        return text.encode_text_for_batch(captions, device=device, dtype=dtype)

    def forward_train(self, dit, noisy, t, cond, *, use_checkpoint: bool = False):
        return dit(
            noisy,
            t,
            cond.context,
            attention_mask=cond.attention_mask,
            use_checkpoint=use_checkpoint,
        )

    def sample_image(self, model, vae, text, prompt, *,
                     height: int = 1024, width: int = 1024,
                     steps: int | None = None,
                     cfg_scale: float | None = None,
                     negative_prompt: str = "",
                     sampler_name: str | None = None,
                     scheduler: str | None = None,
                     distilled: bool = False,
                     device="cuda", dtype=None, step_callback=None,
                     phase_callback=None, seed: int | None = None,
                     vram_policy: str | None = None):
        # First resolve steps and guidance by Raw/Turbo (when steps/cfg aren't
        # given explicitly, use the family defaults: Raw 28 steps / 4.5, Turbo
        # 8 steps / 0.0 -- TDM distillation has no uncond, so at guidance=0
        # prepare skips encoding the negative and sampling skips the uncond
        # forward pass).
        resolved_steps, guidance = resolve_sampling_settings(
            distilled=distilled,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler,
        )
        # -- VRAM orchestration (when vram_policy=None, keep the old behavior;
        # callers like the training preview must never touch model: an
        # optimizer state reference would be broken by .to()).
        # Yield on demand (comfy free_memory semantics): if encoding needs to
        # move the TE onto the GPU and it won't fit, DiT steps down to CPU
        # first; if all prompts hit the online LRU, this whole step is skipped.
        dit_yielded = False
        if vram_policy is not None:
            needed = [prompt] + ([negative_prompt] if guidance > 0 else [])
            cached = getattr(text, "online_conditions_cached", None)
            te_resident = bool(getattr(text, "is_model_on_device", False))
            need_te_move = not te_resident and not (
                callable(cached) and cached(needed)
            )
            te_need = (
                _TE_LOAD_NEED_BYTES_FP8
                if getattr(text, "is_fp8_storage", False)
                else _TE_LOAD_NEED_BYTES
            )
            if need_te_move and _should_yield_dit(vram_policy, device, te_need):
                logger.info("krea2 VRAM orchestration: DiT yielding to CPU before encoding")
                model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                dit_yielded = True
        condition = prepare_sampling_condition(
            text,
            prompt,
            negative_prompt=negative_prompt,
            cfg_scale=guidance,
            device=device,
            dtype=dtype,
            phase_callback=phase_callback,
        )
        # TE handling before sampling: None = old behavior, always offload
        # (no-op in training cache mode); auto only offloads when sampling
        # headroom is insufficient -- when VRAM is plentiful, stays resident, zero moves.
        if vram_policy is None or _should_offload_te(
            vram_policy, device, height, width, dit_yielded,
        ):
            offload = getattr(text, "offload_model", None)
            if callable(offload):
                offload()
        if dit_yielded:
            model.to(torch.device(device))
        return sample_image(
            model,
            vae,
            condition,
            height=height,
            width=width,
            steps=resolved_steps,
            cfg_scale=guidance,
            sampler_name=sampler_name,
            scheduler=scheduler,
            distilled=distilled,
            device=device,
            dtype=dtype,
            step_callback=step_callback,
            phase_callback=phase_callback,
            seed=seed,
        )

    def lora_preset(self) -> dict[str, Any]:
        return KREA2_PRESET

    def lora_metadata(self) -> dict[str, str]:
        return {
            "model_family": self.spec.family_id,
            "preset": self.spec.lora.preset_name,
        }

    def convert_lora_state_dict(self, sd: dict) -> dict:
        return sd


__all__ = [
    "KREA2_SPEC",
    "Krea2Family",
    "Krea2SamplingCondition",
    "Krea2TextCondition",
    "Krea2TextStack",
    "load_krea2_text_stack",
    "prepare_sampling_condition",
    "sample_image",
]
