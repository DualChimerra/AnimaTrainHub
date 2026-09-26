#!/usr/bin/env python3
"""Test image generation -- standalone inference run (CLI usage, no longer called by the Studio server).

Usage:
    python runtime/anima_generate.py --config generate_config.json [--monitor-state-file state.json]

See studio.schema.GenerateConfig for the JSON config fields.

History / current place in the stack:
  - Early on, the server had the supervisor spawn this script as the worker for
    a generate task; every generation had to reload the model, taking 30-60s.
  - PR Phase 2 (commit 9+) switched to a resident inference_daemon
    (runtime/anima_daemon.py) + model reuse across tasks + images served from
    an in-memory cache instead of hitting disk. The server no longer spawns this script.
  - This file is kept around for CLI usage: the user runs generation directly
    from the command line, writing to disk at cfg.output_dir (a user-specified path, genuinely persisted).

Key implementation notes:
  - Multi-LoRA loading goes through studio.services.inference_core.apply_loras --
    each LoRA gets its own independently injected AnimaLycorisAdapter, with
    rank/alpha read from ss_network_args, and contribution weight controlled via
    multiplier=scale (fixes PR #17's bug of a hardcoded rank=32 plus directly
    adding LoKr submatrices, which produced broken images).
  - Progress is pushed via train_monitor over SSE; the frontend pulls single images by sample_path.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch

# anima_train + train_monitor are both in the same runtime/ directory, so _THIS_DIR is enough.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _REPO_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import anima_train as _T  # noqa: E402

from studio.domain.comfy_parity import force_comfy_parity_runtime_config  # noqa: E402
from studio.services.inference.core import LoRASpec, apply_loras  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("anima_generate")


def _torch_dtype_from_precision(value: str | None) -> torch.dtype:
    normalized = str(value or "fp32").lower().strip()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anima test image generation")
    p.add_argument("--config", required=True, help="path to the JSON config file")
    p.add_argument("--monitor-state-file", default="", help="path to the progress state file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"config file does not exist: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = force_comfy_parity_runtime_config(
        cfg,
        force_exact_ksampler_backend=False,
    )

    output_dir = Path(cfg.get("output_dir", "./generate_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts: list[str] = cfg.get("prompts") or ["newest, safe, 1girl, masterpiece, best quality"]
    negative_prompt: str = cfg.get("negative_prompt", "")
    width: int = int(cfg.get("width", 1024))
    height: int = int(cfg.get("height", 1024))
    steps: int = int(cfg.get("steps", 25))
    cfg_scale: float = float(cfg.get("cfg_scale", 4.0))
    sampler_name: str = cfg.get("sampler_name", "er_sde")
    scheduler: str = cfg.get("scheduler", "simple")
    count: int = max(1, int(cfg.get("count", 1)))
    base_seed: int = int(cfg.get("seed", 0))
    distilled: bool = bool(cfg.get("distilled", False))
    lora_configs: list[dict] = cfg.get("lora_configs", [])
    mixed_precision: str = cfg.get("mixed_precision", "bf16")
    vae_precision: str = cfg.get("vae_precision", mixed_precision)
    vram_policy: str = str(cfg.get("vram_policy") or "auto")
    text_encoder_backend: str = cfg.get("text_encoder_backend", "hf")
    t5_tokenizer_backend: str = cfg.get("t5_tokenizer_backend", "slow")
    backend: str = cfg.get("attention_backend", "none")
    use_flash = (backend == "flash_attn")
    use_xformers = (backend == "xformers")

    transformer_path: str = cfg["transformer_path"]
    vae_path: str = cfg["vae_path"]
    text_encoder_path: str = cfg["text_encoder_path"]
    t5_tokenizer_path: str = cfg.get("t5_tokenizer_path", "")

    # monitor
    state_file = args.monitor_state_file or str(output_dir / "monitor_state.json")
    _update_monitor = None
    try:
        from train_monitor import set_state_file, update_monitor
        set_state_file(state_file)
        update_monitor(config={
            "type": "generate",
            "prompts": len(prompts),
            "count": count,
            "steps": steps,
            "cfg_scale": cfg_scale,
        })
        _update_monitor = update_monitor
    except Exception as e:
        logger.warning(f"monitor init failed: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _torch_dtype_from_precision(mixed_precision)
    vae_dtype = _torch_dtype_from_precision(vae_precision)

    # Path resolution
    repo_root = _T.find_diffusion_pipe_root()
    bases = [Path.cwd(), _THIS_DIR, repo_root]
    transformer_path = _T.resolve_path_best_effort(transformer_path, bases)
    vae_path = _T.resolve_path_best_effort(vae_path, bases)
    text_encoder_path = _T.resolve_path_best_effort(text_encoder_path, bases)
    if t5_tokenizer_path:
        t5_tokenizer_path = _T.resolve_path_best_effort(t5_tokenizer_path, bases)

    family = _T.resolve_family(cfg)  # D8': bypass callers dispatch through family too
    logger.info("Loading VAE...")
    vae = family.load_vae(vae_path, device, vae_dtype,
                          tiling=str(cfg.get("vae_tiling", "auto")))

    logger.info("Loading text encoder...")
    # The family's opaque text stack isn't unpacked; caching is off for ad-hoc prompts (cached_varlen families keep the TE resident)
    text_stack = family.load_text(
        text_encoder_path, device, dtype,
        t5_tokenizer_path=t5_tokenizer_path or None,
        comfy_qwen=text_encoder_backend == "comfy_qwen3",
        t5_fast=t5_tokenizer_backend == "fast",
        purpose="generate",
        cache_enabled=False,
    )

    # TE-first ordering (krea2, same as the daemon): pre-encode all prompts
    # before loading the DiT and fully release the TE -- only one big model on
    # the GPU at a time. The prompt set is closed (schema guarantees a single
    # prompt for XY). The anima text stack (a tuple) has no such API and is
    # skipped naturally; the "performance" tier never releases.
    precache = getattr(text_stack, "precache_online_prompts", None)
    if callable(precache):
        try:
            encoded = precache([*[str(p) for p in prompts], negative_prompt])
        except Exception:
            logger.exception("prompt pre-encoding failed; falling back to lazy per-image encoding")
        else:
            if vram_policy != "performance":
                release = getattr(text_stack, "release_model", None)
                if callable(release):
                    release()
            if encoded:
                logger.info("krea2 pre-encoded %d prompts; TE released", encoded)

    logger.info("Loading Transformer...")
    model = family.load_dit(
        transformer_path, device, dtype,
        attention_backend=("flash_attn" if use_flash else "none"), repo_root=repo_root,
        purpose="generate",
    )
    if use_xformers and not _T.enable_xformers(model):
        raise RuntimeError(
            "Exact ComfyUI KSampler parity is guaranteed only with xformers, "
            "but xformers could not be enabled"
        )

    # Multi-LoRA: each one independently injected + multiplier=scale. adapters must
    # keep a reference alive, otherwise the forward hook stops working after GC
    # (lycoris holds the network via a closure).
    specs = [
        LoRASpec(path=str(lc.get("path", "")), scale=float(lc.get("scale", 1.0)))
        for lc in lora_configs
    ]
    _adapters = apply_loras(model, specs, device, torch.float32,  # noqa: F841 -- keep the reference alive
                            family_id=family.spec.family_id)

    model.eval()

    # XY matrix branch (schema already validated: when xy_matrix is set, prompts has a single entry + count=1)
    xy_matrix = cfg.get("xy_matrix")
    if xy_matrix is not None:
        _run_xy_matrix(
            xy_matrix=xy_matrix,
            base_specs=specs,
            adapters=_adapters,
            prompt=prompts[0],
            negative_prompt=negative_prompt,
            base_seed=base_seed,
            base_steps=steps,
            base_cfg_scale=cfg_scale,
            base_sampler=sampler_name,
            scheduler=scheduler,
            distilled=distilled,
            height=height,
            width=width,
            family=family, model=model, vae=vae, text=text_stack,
            device=device, dtype=dtype,
            output_dir=output_dir,
            update_monitor=_update_monitor,
            vram_policy=vram_policy,
        )
        logger.info("XY matrix generation complete")
        return

    # Generation loop
    total = count * len(prompts)
    logger.info(f"starting generation: {len(prompts)} prompts x {count} each = {total} images")

    img_idx = 0
    for pi, prompt in enumerate(prompts):
        for ci in range(count):
            seed = (base_seed + img_idx) if base_seed != 0 else random.randint(0, 2**31 - 1)
            torch.manual_seed(seed)
            random.seed(seed)

            logger.info(f"[{img_idx + 1}/{total}] seed={seed}  prompt={prompt[:60]}...")
            try:
                img = family.sample_image(
                    model, vae, text_stack,
                    prompt,
                    height=height,
                    width=width,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    negative_prompt=negative_prompt,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    distilled=distilled,
                    device=device,
                    dtype=dtype,
                    seed=seed,
                    vram_policy=vram_policy,
                )
                fname = f"gen_{img_idx:04d}_p{pi}_c{ci}_s{seed}.png"
                out_path = output_dir / fname
                img.save(out_path)
                logger.info(f"saved: {out_path}")
                if _update_monitor:
                    _update_monitor(sample_path=str(out_path), step=img_idx + 1)
            except Exception as e:
                logger.error(f"generation failed [{img_idx + 1}/{total}]: {e}")

            img_idx += 1

    logger.info("generation complete")


# ---------------------------------------------------------------------------
# XY matrix implementation -- loops over all images within a single task, avoiding N model-load costs
# ---------------------------------------------------------------------------


def _set_lora_multiplier(adapter, scale: float) -> None:
    """Change an adapter's multiplier in place, no re-injection needed.

    Matches the value-setting path inside inference_core.apply_loras:
    network.multiplier is the global multiplier read during forward;
    per-lora.multiplier is a fallback (different lycoris versions read it via different paths).
    """
    if adapter.network is None:
        return
    adapter.network.multiplier = float(scale)
    for lora in getattr(adapter.network, "loras", []):
        if hasattr(lora, "multiplier"):
            lora.multiplier = float(scale)


def _apply_axis(
    axis: dict,
    value,
    *,
    cur_steps: int, cur_cfg_scale: float, cur_seed: int, cur_sampler: str,
    base_specs, adapters,
) -> tuple[int, float, int, str]:
    """Update the fields derived from axis_type; lora_scale directly mutates every adapter.

    lora_ckpt is not handled here -- it requires reinjection, which
    _run_xy_matrix handles separately via the apply_loras reload path.

    Returns a (steps, cfg_scale, seed, sampler) 4-tuple (values that don't change pass through unchanged).
    """
    axis_type = axis["axis"]
    if axis_type == "steps":
        cur_steps = int(value)
    elif axis_type == "cfg_scale":
        cur_cfg_scale = float(value)
    elif axis_type == "seed":
        cur_seed = int(value)
    elif axis_type == "sampler_name":
        cur_sampler = str(value)
    elif axis_type == "lora_scale":
        # Global axis: every LoRA's multiplier gets set to the same cell value
        for ad in adapters:
            _set_lora_multiplier(ad, float(value))
    return cur_steps, cur_cfg_scale, cur_seed, cur_sampler


def _run_xy_matrix(
    *,
    xy_matrix: dict,
    base_specs: list,
    adapters: list,
    prompt: str,
    negative_prompt: str,
    base_seed: int,
    base_steps: int,
    base_cfg_scale: float,
    base_sampler: str,
    scheduler: str,
    height: int,
    width: int,
    family, model, vae, text,
    device: str, dtype,
    output_dir,
    update_monitor,
    distilled: bool = False,
    vram_policy: str = "auto",
) -> None:
    """Loop over (yi, xi) to produce an N x M grid of images.

    Design:
      - Each cell derives its parameters from base_* (preventing the previous
        cell's changes from leaking into the next); lora_scale is implemented by
        mutating adapter.multiplier, so every LoRA's multiplier must be reset
        back to base_specs[i].scale before each cell.
      - Filename `xy_x{xi:02d}_y{yi:02d}_s{seed}.png`; the frontend arranges the
        grid by (yi, xi).
      - update_monitor pushes sample_path + xy metadata; the frontend uses
        xy={xi,yi,xv,yv} to render cell labels + ordering.
      - base_seed=0 -> randomized once and shared across all cells (XY should
        only show the axis effect); axis=seed overrides per cell value.
    """
    x_spec = xy_matrix["x"]
    y_spec = xy_matrix.get("y")
    x_values = x_spec["values"]
    y_values = y_spec["values"] if y_spec else [None]

    # The lora_ckpt axis needs a detach + reinject of the LoRA between cells; the
    # CLI runner doesn't wire into the CACHE.apply_loras setup (only the daemon
    # has that), so it's rejected outright here. The production path via
    # runtime/anima_daemon.py:_run_xy already supports it.
    if x_spec.get("axis") == "lora_ckpt" or (y_spec and y_spec.get("axis") == "lora_ckpt"):
        raise NotImplementedError(
            "the lora_ckpt axis requires the daemon path (runtime/anima_daemon.py); "
            "the CLI runner anima_generate.py does not support hot-swapping LoRA files"
        )

    # An fp8 base model's LoRA is merged into the weights (no resident network),
    # so the lora_scale axis needs a per-cell detach + re-merge -- the daemon's
    # CACHE.apply_loras path already supports this (_cell_lora_configs); the CLI
    # runner has no cache management and, like the lora_ckpt axis, isn't wired up for it.
    if x_spec.get("axis") == "lora_scale" or (y_spec and y_spec.get("axis") == "lora_scale"):
        from training.families.krea2.quant_fp8 import model_has_fp8_layers

        if model_has_fp8_layers(model):
            raise NotImplementedError(
                "an fp8 base model's LoRA strength axis needs a per-cell re-merge; use the "
                "daemon path (runtime/anima_daemon.py); the CLI runner should use a bf16 base model instead."
            )

    if base_seed == 0:
        base_seed = random.randint(0, 2**31 - 1)
        logger.info(f"XY shared seed (randomized from cfg.seed=0): {base_seed}")

    base_scales = [float(s.scale) for s in base_specs]
    total = len(x_values) * len(y_values)
    logger.info(f"starting XY generation: {len(x_values)}x{len(y_values)} = {total} images")

    img_idx = 0
    for yi, yv in enumerate(y_values):
        for xi, xv in enumerate(x_values):
            # Reset every LoRA back to its base scale, so a previous cell's lora_scale change doesn't leak
            for i, s in enumerate(base_scales):
                if i < len(adapters):
                    _set_lora_multiplier(adapters[i], s)

            cur_steps = base_steps
            cur_cfg_scale = base_cfg_scale
            cur_seed = base_seed
            cur_sampler = base_sampler

            cur_steps, cur_cfg_scale, cur_seed, cur_sampler = _apply_axis(
                x_spec, xv,
                cur_steps=cur_steps, cur_cfg_scale=cur_cfg_scale,
                cur_seed=cur_seed, cur_sampler=cur_sampler,
                base_specs=base_specs, adapters=adapters,
            )
            if y_spec is not None and yv is not None:
                cur_steps, cur_cfg_scale, cur_seed, cur_sampler = _apply_axis(
                    y_spec, yv,
                    cur_steps=cur_steps, cur_cfg_scale=cur_cfg_scale,
                    cur_seed=cur_seed, cur_sampler=cur_sampler,
                    base_specs=base_specs, adapters=adapters,
                )

            torch.manual_seed(cur_seed)
            random.seed(cur_seed)

            logger.info(
                f"XY [{xi},{yi}] x={xv} y={yv} "
                f"steps={cur_steps} cfg={cur_cfg_scale} seed={cur_seed} sampler={cur_sampler}"
            )
            try:
                img = family.sample_image(
                    model, vae, text,
                    prompt,
                    height=height,
                    width=width,
                    steps=cur_steps,
                    cfg_scale=cur_cfg_scale,
                    negative_prompt=negative_prompt,
                    sampler_name=cur_sampler,
                    scheduler=scheduler,
                    distilled=distilled,
                    device=device,
                    dtype=dtype,
                    seed=cur_seed,
                    vram_policy=vram_policy,
                )
                fname = f"xy_x{xi:02d}_y{yi:02d}_s{cur_seed}.png"
                out_path = output_dir / fname
                img.save(out_path)
                logger.info(f"saved: {out_path}")
                if update_monitor:
                    update_monitor(
                        sample_path=str(out_path),
                        step=img_idx + 1,
                        xy={"xi": xi, "yi": yi, "xv": xv, "yv": yv},
                    )
            except Exception as e:
                logger.error(f"XY [{xi},{yi}] failed: {e}")

            img_idx += 1


if __name__ == "__main__":
    main()
