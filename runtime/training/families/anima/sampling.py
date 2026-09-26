"""Inference sampling: sigma schedule + ER-SDE solver + sample_image (shared by training/generation).

Extracted from the original runtime/anima_train.py L822-961 + L1677-1815 (ADR 0003 PR-A).

Public:
- sample_image -- shared entry point for training-time preview sampling + the generate CLI (called by the sister script)

Internal:
- _time_snr_shift / _flow_sigmas_simple -- matches ComfyUI ModelSamplingDiscreteFlow
- _default_noise_sampler / _sample_er_sde_const_x0 -- ER-SDE-Solver-3 implementation under CONST flow

Note: sample_t / make_noise / compute_loss_weight are sampling utilities for
the *training step* and are not in this module -- see
training.timestep_sampling / training.noise / training.loss_weighting.
"""

from __future__ import annotations

import logging
import sys
import torch
import torch.nn.functional as F

from training.families.anima import ANIMA_SPEC
from training.families.anima.text_encoding import (
    build_comfy_anima_conditioning_inputs,
    encode_qwen,
)

_ANIMA_LATENT = ANIMA_SPEC.latent


logger = logging.getLogger(__name__)


_COMFY_PARITY_SAMPLERS = {"dpmpp_3m_sde", "er_sde"}
_COMFY_PARITY_SCHEDULERS = {"sgm_uniform", "simple"}


def _resolve_parity_sampler_scheduler(sampler_name: str, scheduler: str) -> tuple[str, str]:
    sampler = str(sampler_name).lower().strip()
    sched = str(scheduler).lower().strip()
    if sampler not in _COMFY_PARITY_SAMPLERS or sched not in _COMFY_PARITY_SCHEDULERS:
        raise ValueError(f"unsupported Comfy parity sampler/scheduler: {sampler}+{sched}")
    return sampler, sched


def _set_model_xformers_enabled(model, enabled: bool) -> bool:
    """Toggle model-module xformers switches. Returns True if it had been enabled."""
    module_names = {
        cls.__module__
        for cls in type(model).__mro__
        if getattr(cls, "__module__", None)
    }
    # Module identity is unique now that exec-load has been retired (multi-model
    # PR-2a); the literal name here is a fallback that covers callers where
    # `model` is a wrapper/dummy whose MRO doesn't include the model module.
    module_names.add("modeling.anima.cosmos_predict2_modeling")

    was_enabled = False
    for module_name in sorted(module_names):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        was_enabled = bool(getattr(module, "_USE_XFORMERS", False)) or was_enabled
        fn = getattr(module, "set_xformers_enabled", None)
        if fn is not None:
            fn(enabled)
    return was_enabled


def _module_device(module) -> torch.device | None:
    if module is None or not hasattr(module, "parameters"):
        return None
    try:
        param = next(module.parameters())
    except StopIteration:
        return None
    except Exception:
        return None
    return param.device


def _decode_offload_targets(model, qwen_model) -> tuple:
    """Modules allowed to be offloaded during VAE decode.

    When block swap is active, the DiT must be skipped: the blanket
    ``.to(device)`` used to restore modules would move the swapped-out
    layers' pinned CPU master copy entirely back onto the GPU -- wasting the
    swap and hitting a peak usage equal to the full model (an immediate OOM
    on a small card); and once swapped, what's resident on the DiT is
    already down to a sliver, so offloading it is pointless anyway. The
    marker is set on the model by loader.place_model_for_block_swap.
    """
    if int(getattr(model, "blocks_to_swap", 0) or 0) > 0:
        return (qwen_model,)
    return (model, qwen_model)


def _offload_modules_for_vae_decode(*modules) -> list[tuple[object, torch.device]]:
    """Move large, inactive modules off GPU while fp32 VAE decode runs."""
    offloaded: list[tuple[object, torch.device]] = []
    for module in modules:
        device = _module_device(module)
        if device is None or device.type != "cuda" or not hasattr(module, "to"):
            continue
        module.to("cpu")
        offloaded.append((module, device))
    if offloaded and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return offloaded


def _restore_offloaded_modules(offloaded: list[tuple[object, torch.device]]) -> None:
    for module, device in offloaded:
        module.to(device)
    if offloaded and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_modules_to_device(device: str | torch.device, *modules) -> None:
    target = torch.device(device)
    moved = False
    for module in modules:
        current = _module_device(module)
        if current is None or current == target or not hasattr(module, "to"):
            continue
        module.to(target)
        moved = True
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _decode_vae(vae, latents: torch.Tensor) -> torch.Tensor:
    if hasattr(vae, "decode"):
        return vae.decode(latents)
    return vae.model.decode(latents, vae.scale)


def _time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    """ComfyUI ModelSamplingDiscreteFlow.time_snr_shift"""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def _flow_sigmas_simple(steps: int, *, shift: float = 3.0, timesteps: int = 1000, device: str = "cpu") -> torch.Tensor:
    """
    Reproduces ComfyUI:
    - supported_models.Anima's sampling_settings: shift=3.0, multiplier=1.0
    - ModelSamplingDiscreteFlow + simple_scheduler(model_sampling, steps)

    Returns: sigmas (steps+1,) float32, high to low, ending with 0.0.
    Note: ComfyUI's simple_scheduler returns the first item as 1.0 as-is;
    KSampler only applies offset_first_sigma_for_snr after entering the
    concrete sampler. Don't offset it early at the scheduler level, or the
    txt2img initial noise_scaling will differ from ComfyUI's.
    """
    ts = torch.arange(1, timesteps + 1, device=device, dtype=torch.float32) / float(timesteps)  # (0, 1]
    sigmas_full = _time_snr_shift(float(shift), ts)  # (0, 1]

    ss = len(sigmas_full) / float(steps)
    sigmas = [float(sigmas_full[-(1 + int(i * ss))]) for i in range(steps)]
    sigmas.append(0.0)
    sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
    return sigmas


def _flow_sigmas_sgm_uniform(steps: int, *, shift: float = 3.0, timesteps: int = 1000, multiplier: int = 1000, device: str = "cpu") -> torch.Tensor:
    """SGM uniform scheduler -- matches ComfyUI normal_scheduler(sgm=True) line for line.

    ModelSamplingDiscreteFlow semantics:
      sigma_max/min = the two ends of the sigma table = time_snr_shift(shift, {1, 1/timesteps})
      timestep(sigma)   = sigma * multiplier
      sigma(ts)     = time_snr_shift(shift, ts / multiplier)
    sgm branch: linspace(timestep(sigma_max), timestep(sigma_min), steps+1)[:-1],
    then map each back to sigma via sigma(), and append 0 at the end. Note
    this applies a "double shift" to sigma_max/sigma_min -- sigma is already
    the shifted value, timestep doesn't unshift it, and sigma() shifts it
    again. This is ComfyUI's existing behavior, deliberately reproduced here
    to match its output.
    """
    # the two ends of the sigma table (already shifted)
    sigma_max = float(_time_snr_shift(float(shift), torch.tensor(1.0)))
    sigma_min = float(_time_snr_shift(float(shift), torch.tensor(1.0 / timesteps)))
    start = sigma_max * multiplier  # timestep(sigma_max)
    end = sigma_min * multiplier    # timestep(sigma_min)
    tl = torch.linspace(start, end, steps + 1, device=device, dtype=torch.float32)[:-1]
    sigmas = _time_snr_shift(float(shift), tl / multiplier)  # sigma(ts)
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
    return sigmas


def _prepare_comfy_t2i_noise(
    shape: tuple[int, ...],
    sigmas: torch.Tensor,
    *,
    device: str | torch.device,
    seed: int | None,
) -> torch.Tensor:
    """ComfyUI txt2img initial noise for CONST flow.

    ComfyUI KSampler creates initial noise on CPU with a CPU generator seeded by
    the requested seed, then model_sampling.noise_scaling() moves the sample into
    the sampler path. For CONST + empty txt2img latent, noise_scaling is simply
    `sigma * noise`.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

    noise = torch.randn(
        shape,
        dtype=torch.float32,
        layout=torch.strided,
        device="cpu",
        generator=generator,
    )
    sigma0 = float(sigmas[0].detach().cpu()) if sigmas.numel() > 0 else 0.0
    return (noise * sigma0).to(device=device, dtype=torch.float32)


def _fix_comfy_empty_latent_channels(
    latent_image: torch.Tensor,
    *,
    latent_channels: int,
    latent_dimensions: int,
) -> torch.Tensor:
    """Mirror ComfyUI `fix_empty_latent_channels` for txt2img empty latents.

    ResolutionMaster commonly emits a 4-channel empty latent. ComfyUI KSampler
    repeats all-zero empty latents to the model latent channel count, then adds
    the temporal dimension for video/Anima-style latent formats.
    """
    if torch.count_nonzero(latent_image) == 0 and latent_image.shape[1] != latent_channels:
        repeats = (latent_channels + latent_image.shape[1] - 1) // latent_image.shape[1]
        latent_image = latent_image.repeat(1, repeats, *([1] * (latent_image.ndim - 2)))
        latent_image = latent_image.narrow(1, 0, latent_channels)

    if latent_dimensions == 3 and latent_image.ndim == 4:
        latent_image = latent_image.unsqueeze(2)
    return latent_image


def _prepare_comfy_ksampler_txt2img_latent(
    height: int,
    width: int,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build the same empty latent shape path as the target Comfy workflow."""
    latent = torch.zeros(
        # (1, 4, ...) is ResolutionMaster workflow parity (4ch empty latent ->
        # repeat to fill in); the 4 is not this model's latent channel count,
        # keep it as a literal
        (1, 4, height // _ANIMA_LATENT.spatial_stride, width // _ANIMA_LATENT.spatial_stride),
        device=device,
        dtype=torch.float32,
    )
    return _fix_comfy_empty_latent_channels(
        latent,
        latent_channels=_ANIMA_LATENT.channels,
        latent_dimensions=3,
    )


# The ER-SDE-Solver implementation + _default_noise_sampler have moved to
# training.inference_samplers.er_sde (ADR 0003 PR-C plugin registry).
# sample_image dispatches to them via build_inference_sampler.


@torch.no_grad()
def sample_image(
    model, vae, qwen_model, qwen_tokenizer, t5_tokenizer,
    prompt, height=1024, width=1024, steps=25, cfg_scale=4.0,
    negative_prompt=None,
    sampler_name: str = "er_sde",
    scheduler: str = "simple",
    device="cuda",
    dtype=torch.bfloat16,
    step_callback=None,
    phase_callback=None,
    seed: int | None = None,
):
    """Sample-generate an image (Comfy-style, the only path) -- shared by training preview / Generate / RegAI.

    Matches ComfyUI KSampler: raw prompt goes into Qwen, SDTokenizer-style T5
    weights, CFG batched into one forward, CPU-seeded initial noise. Exact
    parity only holds under the Generate runtime (comfy_qwen3 encoder +
    xformers); training preview / RegAI use HF Qwen, so they're Comfy-style
    rather than bit-for-bit identical.

    Args:
        negative_prompt: the negative prompt; None is equivalent to an empty
            string (matches ComfyUI: the negative prompt is exactly whatever
            was written in the workflow, with no implicit default string)
        sampler_name: er_sde / dpmpp_3m_sde
        scheduler: simple / sgm_uniform
    """
    import numpy as np
    from PIL import Image

    if str(device).startswith("cuda"):
        _move_modules_to_device(device, model, qwen_model)

    model.eval()

    sampler_name, scheduler = _resolve_parity_sampler_scheduler(sampler_name, scheduler)

    logger.info(f"[Debug] Sampling start. Prompt: {prompt[:50]}...")

    # Check VAE scale
    if isinstance(vae.scale, list) and len(vae.scale) == 2:
        m, s = vae.scale
        logger.info(f"[Debug] VAE scale: mean_shape={m.shape}, std_inv_shape={s.shape}")
        logger.info(f"[Debug] VAE scale values: mean={m.mean().item():.4f}, std_inv={s.mean().item():.4f}")

    # Matches ComfyUI: the negative prompt has no implicit default; None means empty.
    negative_prompt = "" if negative_prompt is None else str(negative_prompt)

    # Text encoding (CLIP/T5+Qwen) -- phase reporting lets the progress bar cover the non-sampling stages
    if phase_callback:
        phase_callback("clip")
    try:
        def build_cross(prompt_text: str):
            qwen_text, t5_ids, t5_attn, t5_w = build_comfy_anima_conditioning_inputs(
                t5_tokenizer,
                prompt_text,
                max_length=512,
            )
            qwen_embeds, qwen_attn = encode_qwen(
                qwen_model,
                qwen_tokenizer,
                [qwen_text],
                device,
                preserve_empty_text=True,
            )
            logger.info(f"[Debug] Qwen embeds: {qwen_embeds.shape}, mean={qwen_embeds.mean().item():.4f}")
            qwen_embeds = qwen_embeds.to(device=device, dtype=dtype)
            t5_ids = t5_ids.to(device)
            t5_attn = t5_attn.to(device)
            t5_w = t5_w.to(device, dtype=dtype)
            cross = model.preprocess_text_embeds(qwen_embeds, t5_ids, t5xxl_weights=t5_w)
            if cross.shape[1] < 512:
                cross = F.pad(cross, (0, 0, 0, 512 - cross.shape[1]))
            return cross

        # conditional (positive prompt)
        cross_cond = build_cross(prompt)

        # unconditional/negative prompt
        cross_uncond = build_cross(negative_prompt)

    except Exception as e:
        logger.error(f"[Debug] Encoding failed: {e}")
        raise e

    # sigmas (matches ComfyUI supported_models.Anima: shift=3.0, multiplier=1.0)
    lat_h = height // _ANIMA_LATENT.spatial_stride
    lat_w = width // _ANIMA_LATENT.spatial_stride
    _scheduler_builders = {
        "simple": _flow_sigmas_simple,
        "sgm_uniform": _flow_sigmas_sgm_uniform,
    }
    sched_fn = _scheduler_builders.get(str(scheduler).lower().strip())
    if sched_fn is None:
        # Already validated at the entry point by _resolve_parity_sampler_scheduler; this is a defensive fallback
        raise ValueError(
            f"unsupported Comfy parity sampler/scheduler: "
            f"{str(sampler_name).lower().strip()}+{str(scheduler).lower().strip()}"
        )
    sigmas = sched_fn(steps, shift=3.0, device=device)

    # Initialize noise (ComfyUI CONST.noise_scaling: x = sigma*noise + (1-sigma)*latent_image; txt2img latent_image=0)
    empty_latent = _prepare_comfy_ksampler_txt2img_latent(height, width, device="cpu")
    x = _prepare_comfy_t2i_noise(tuple(empty_latent.shape), sigmas, device=device, seed=seed)
    logger.info(f"[Debug] Latents init: {x.shape}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")

    pad_mask = torch.zeros(1, 1, lat_h, lat_w, device=device, dtype=dtype)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    # If NaN retry turned xformers off, it must be restored once sampling
    # finishes -- otherwise a single NaN would leave the whole rest of the
    # process running on SDPA (no longer exact parity) with the user none
    # the wiser.
    xformers_disabled_for_nan = False

    def denoise_fn(x_in: torch.Tensor, sigma_in: torch.Tensor) -> torch.Tensor:
        nonlocal xformers_disabled_for_nan
        if not torch.is_tensor(sigma_in):
            sigma_in = torch.tensor(float(sigma_in), device=x_in.device, dtype=torch.float32)
        sigma_5d = sigma_in.view(1, 1, 1, 1, 1).to(device=x_in.device, dtype=torch.float32)

        def _run_model_forward() -> torch.Tensor:
            with torch.autocast(device_type=device_type, dtype=dtype):
                # ComfyUI's CFGGuider batches negative/positive conds through one
                # model forward in the common txt2img path, then chunks outputs.
                # It also passes ModelSamplingDiscreteFlow.timestep(sigma) as
                # float32. For Anima multiplier=1.0, timestep == sigma.
                x_model = x_in.to(device=x_in.device, dtype=dtype)
                sigma_1d = sigma_in.reshape(1).to(device=x_in.device, dtype=torch.float32)
                if float(cfg_scale) == 1.0:
                    return model(
                        x_model,
                        sigma_1d.expand(x_model.shape[0]),
                        cross_cond,
                        padding_mask=pad_mask.expand(x_model.shape[0], -1, -1, -1).contiguous(),
                    )
                x_batch = torch.cat([x_model, x_model], dim=0)
                cross_batch = torch.cat([cross_uncond, cross_cond], dim=0)
                pad_batch = pad_mask.expand(x_batch.shape[0], -1, -1, -1).contiguous()
                sigma_batch = sigma_1d.expand(x_batch.shape[0])
                v_uncond, v_cond = model(
                    x_batch,
                    sigma_batch,
                    cross_batch,
                    padding_mask=pad_batch,
                ).chunk(2)
                return v_uncond + cfg_scale * (v_cond - v_uncond)

        v = _run_model_forward()

        if torch.isnan(v).any():
            if _set_model_xformers_enabled(model, False):
                xformers_disabled_for_nan = True
                logger.warning("xformers attention produced NaN; retrying denoise with SDPA fallback")
                v = _run_model_forward()
            if torch.isnan(v).any():
                raise RuntimeError("v contains NaN during sampling")

        # CONST(flow): denoised x0 = x - sigma * v
        return x_in - sigma_5d * v.float()

    sampler_name_l = str(sampler_name).lower().strip()
    logger.info(f"[Debug] Sampler={sampler_name_l}, Scheduler={scheduler}, steps={steps}, cfg={cfg_scale}")

    # PR-C: dispatched through the inference_samplers plugin registry; the whitelist was already validated at the entry point
    from training.inference_samplers import build_inference_sampler
    sampler_fn = build_inference_sampler(sampler_name_l)
    if sampler_fn is None:
        raise ValueError(
            f"unsupported Comfy parity sampler/scheduler: "
            f"{sampler_name_l}+{str(scheduler).lower().strip()}"
        )
    sampler_kwargs = {
        "seed": seed,
        "s_noise": 1.0,
        "max_stage": 3,
        "step_callback": step_callback,
    }
    if sampler_name_l == "dpmpp_3m_sde":
        sampler_kwargs["require_brownian_tree"] = True
    if phase_callback:
        phase_callback("sample")
    try:
        x = sampler_fn(denoise_fn, x, sigmas, **sampler_kwargs)
    finally:
        if xformers_disabled_for_nan:
            # The remaining steps for this image already ran on SDPA (keeps
            # it consistent within the image); reset the process-level
            # switch so the next image tries xformers again.
            _set_model_xformers_enabled(model, True)
            logger.warning("xformers re-enabled after per-image SDPA fallback (this image is not exact parity)")

    # VAE decode
    if phase_callback:
        phase_callback("vae")
    latents = x.to(device=device, dtype=dtype)
    logger.info(f"[Debug] Final latents: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
    del denoise_fn, x, cross_cond, cross_uncond, pad_mask, sigmas, empty_latent
    offloaded_modules: list[tuple[object, torch.device]] = []
    try:
        # VAEWrapper.decode picks whole-image vs. tiled decode based on
        # tiling(auto/on/off). Offloading is now driven by free VRAM and
        # unified with tiling (replacing the old "only offload for fp32" rule
        # -- dtype isn't an accurate proxy for VRAM pressure: bf16 with a
        # large image/resident models can OOM just the same): DiT+Qwen only
        # move to CPU when freeing VRAM would let the whole image decode in
        # one pass and do so quickly, to avoid pointlessly moving things to
        # host memory when it wouldn't clear the peak. See the decision logic
        # in VAEWrapper.should_offload_for_whole_decode.
        _should_offload = getattr(vae, "should_offload_for_whole_decode", None)
        if device_type == "cuda" and callable(_should_offload) and _should_offload(latents):
            logger.info("[Debug] VAE decode: VRAM is tight but the peak is just under the cliff, offloading inactive modules to decode the whole image at once")
            offloaded_modules = _offload_modules_for_vae_decode(
                *_decode_offload_targets(model, qwen_model)
            )
        images = _decode_vae(vae, latents)
        images = images.squeeze(2)  # [B,C,H,W]
        images = (images.clamp(-1, 1) + 1) / 2

        # convert to PIL
        img = images[0].permute(1, 2, 0).cpu().float().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img)

        del images, latents
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"[Debug] VAE decode failed: {e}")
        raise
    finally:
        if offloaded_modules:
            logger.info("[Debug] VAE decode: restoring offloaded modules after cleanup")
            _restore_offloaded_modules(offloaded_modules)

    model.train()
    return pil_image
