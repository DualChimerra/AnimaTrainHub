"""Image upscaling service (preprocessing stage).

Wraps spandrel + tiled inference into a single-file API:

- `load_model(path)`: cached loading of ESRGAN/RRDB weights (spandrel auto-detects the architecture)
- `tiled_inference(model, img, *, scale, tile_size, tile_pad)`: tiled forward
  pass + overlap stitching, keeping the VRAM ceiling linear in tile_size
- `upscale_file(src, dst, *, model_path, ...)`: the full file-level API,
  handling Pillow decoding, device selection, and writing the output;
  **returns a metadata dict, doesn't write a sidecar** (status is recorded
  centrally by preprocess_manifest at the worker entry point, see ADR 0004)

Design:
- the model is only instantiated on the first call to load_model; subsequent
  calls with the same path reuse the cache (a process-level dict, same
  approach as wd14_tagger)
- device strategy: device='auto' -> cuda if available, else cpu; an explicit
  'cuda' with no GPU auto-downgrades to cpu with a log warning (avoids
  crashing the subprocess outright)
- tile_size is in **input pixels**; a 4x model with tile=256 -> a 1024x1024 output tile.
  Peak VRAM is roughly `tile**2 * scale**2 * 4 bytes * batch * ~7x for intermediate tensors`,
  around 2-3GB at 256
- tile_pad removes stitching seams, default 16px overlap (spandrel's recommended value)
- unit tests exercise the full pipeline with a stub model, no real weights needed
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# The minimal protocol we need from spandrel's `ImageModelDescriptor`: only
# `.model` and `.scale`. Cache key = absolute path string. Note: swapping the
# weights file while keeping the same name leaves a stale cache entry - in
# practice weight files don't change after download, so this is acceptable.
# Call `clear_cache()` if you need to invalidate it.
_MODEL_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# device & model loading
# ---------------------------------------------------------------------------


def resolve_device(device: str = "auto") -> torch.device:
    """device='auto' -> cuda if available, else cpu. Explicit 'cuda' with no GPU downgrades."""
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("requested cuda but cuda not available; falling back to cpu")
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    """precision='auto' -> fp16 on cuda, fp32 on cpu.

    fp16 is visually indistinguishable for CNN-based vision models like
    ESRGAN/RRDB, but is usually 1.6-2x faster on GPU; ComfyUI also defaults
    to fp16. bf16 is available on sm_80+ (Ampere and later) and is friendlier
    to numeric range, but unsupported on RTX 20 series / Tesla.
    """
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    # auto
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_model(
    model_path: Path,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Any:
    """Load a spandrel ImageModelDescriptor, cached per process.

    The descriptor exposes:
        .model - torch.nn.Module, can be called directly
        .scale - integer upscale factor (4x-AnimeSharp is 4)
        .input_channels / .output_channels - usually 3

    Passing dtype casts the model's weights to it (fp16 typically gives a 1.6-2x speedup on GPU).
    """
    if not model_path.exists():
        raise FileNotFoundError(f"The model weights do not exist: {model_path}")

    key = str(model_path.resolve())
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        if device is not None:
            cached.model.to(device)
        if dtype is not None:
            cached.model.to(dtype)
        return cached

    try:
        from spandrel import ModelLoader
    except ImportError as exc:
        raise RuntimeError(
            "spandrel is not installed (pip install spandrel). Upscaling during preprocessing needs it."
        ) from exc

    descriptor = ModelLoader().load_from_file(str(model_path))
    descriptor.model.eval()
    if device is not None:
        descriptor.model.to(device)
    if dtype is not None:
        descriptor.model.to(dtype)
    _MODEL_CACHE[key] = descriptor
    return descriptor


def clear_cache() -> None:
    """Clear the model cache (used by tests / when switching device)."""
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------


def resize_to_area(img: Image.Image, target_area: int) -> Image.Image:
    """LANCZOS resize to `~target_area` pixels, preserving aspect ratio.

    new_W * new_H ~= target_area, keeping the W/H ratio. Not snapped to a
    multiple of 64 - Kohya will resize/crop again per-bucket internally, so
    snapping early would only limit which bucket it lands in. Returns a new PIL Image.
    """
    w, h = img.size
    if w <= 0 or h <= 0 or target_area <= 0:
        return img
    ratio = math.sqrt(target_area / (w * h))
    new_w = max(1, round(w * ratio))
    new_h = max(1, round(h * ratio))
    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), Image.LANCZOS)


# Lower bound for "already big enough, skip the model": when area >=
# target_area x SKIP_RATIO, just LANCZOS-resize directly. 0.95 = tolerate a
# 5% pixel shortfall via plain LANCZOS (a slight upsample) - the visual
# difference is imperceptible, and it saves a (costly) model inference pass.
# Below this threshold, the model is worth it for detail preservation.
SKIP_MODEL_RATIO = 0.95


def _img_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> float32 BCHW [0,1]. RGBA is converted to RGB (alpha is simply
    dropped - 4x-AnimeSharp doesn't handle a transparency channel; if the
    caller needs to preserve alpha, composite onto white first)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    import numpy as np

    arr = np.array(img, dtype=np.float32) / 255.0  # HWC
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # 1CHW
    return t


def _tensor_to_img(t: torch.Tensor) -> Image.Image:
    """1CHW or CHW float[0,1] -> PIL RGB. fp16/bf16 is cast back to fp32 before converting to numpy."""
    import numpy as np

    if t.dim() == 4:
        t = t.squeeze(0)
    # numpy has no direct bf16 support; fp16 converts but loses precision when multiplied by 255 - always cast back to fp32
    arr = t.clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()  # HWC
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def tiled_inference(
    model: Callable[[torch.Tensor], torch.Tensor],
    img: torch.Tensor,
    *,
    scale: int,
    tile_size: int = 256,
    tile_pad: int = 16,
) -> torch.Tensor:
    """Tiled forward pass: each tile's input is `tile_size+2*tile_pad`, and the padded region is trimmed off the output before stitching.

    img: 1CHW float[0,1] tensor (already on the target device)
    Returns a 1CHW float tensor, with H and W both x scale.

    With no tiling (tile_size <= 0), does a single full-image forward pass (saves IO overhead for small images / when VRAM allows).
    """
    if tile_size <= 0:
        with torch.inference_mode():
            return model(img)

    assert img.dim() == 4 and img.shape[0] == 1, f"need 1CHW tensor, got {img.shape}"
    _, c, h, w = img.shape
    out_h, out_w = h * scale, w * scale
    output = torch.zeros((1, c, out_h, out_w), dtype=img.dtype, device=img.device)

    # tile grid steps by tile_size; edge tiles are truncated when short
    n_y = (h + tile_size - 1) // tile_size
    n_x = (w + tile_size - 1) // tile_size

    with torch.inference_mode():
        for ty in range(n_y):
            for tx in range(n_x):
                y0 = ty * tile_size
                x0 = tx * tile_size
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)

                # padded region (used to remove stitching seams)
                py0 = max(y0 - tile_pad, 0)
                px0 = max(x0 - tile_pad, 0)
                py1 = min(y1 + tile_pad, h)
                px1 = min(x1 + tile_pad, w)

                tile = img[:, :, py0:py1, px0:px1]
                up = model(tile)

                # in the upscaled coordinate space, trim off the padded region, keeping only the tile's actual extent
                cut_top = (y0 - py0) * scale
                cut_left = (x0 - px0) * scale
                cut_bot = cut_top + (y1 - y0) * scale
                cut_right = cut_left + (x1 - x0) * scale
                core = up[:, :, cut_top:cut_bot, cut_left:cut_right]

                output[
                    :,
                    :,
                    y0 * scale : y1 * scale,
                    x0 * scale : x1 * scale,
                ] = core

    return output


# ---------------------------------------------------------------------------
# file-level API
# ---------------------------------------------------------------------------


def upscale_file(
    src: Path,
    dst: Path,
    *,
    model_path: Path,
    label: str = "4x-AnimeSharp",
    tile_size: int = 256,
    tile_pad: int = 16,
    device: str = "auto",
    precision: str = "auto",
    target_area: Optional[int] = None,
    on_log: Callable[[str], None] = lambda _l: None,
    prewarm_thumb_sizes: Optional[list[int]] = None,
    save_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Read src -> upscale intelligently -> write dst (PNG). Returns a metadata dict.

    target_area controls behavior (aimed at "just good enough" for LoRA training):
      - target_area=None: pure 4x model upscale (compatible with the old path)
      - target_area=N, src area >= N x SKIP_MODEL_RATIO: skip the model, LANCZOS-resize straight to ~N
      - target_area=N, src area < N x SKIP_MODEL_RATIO: model 4x -> LANCZOS-resize to ~N

    The skip-model path is "just resize the big image", a few hundred ms; the
    model path is the tens-of-seconds-costly one. Most training set images
    are already big enough for 1024^2/1536^2, so preprocessing only actually
    invokes the model on a minority of small images.

    Returns metadata (picked up by the worker to write into the manifest, see ADR 0004):
        {source, model, scale, action, target_area, tile_size, tile_pad,
         device, dtype, src_size, dst_size, elapsed_seconds, mtime}
    action: 'resize' | 'upscale' | 'upscale+resize'

    Silently downgrades to cpu when `device='cuda'` but no GPU is available.
    """
    t_start = time.monotonic()

    # 1) read the image first to decide which path to take (avoids loading the model needlessly)
    with Image.open(src) as raw:
        raw.load()
        if raw.mode != "RGB":
            raw = raw.convert("RGB")
        src_img = raw.copy()  # still needed after leaving the with block
    src_size = src_img.size  # (W, H)
    src_area = src_img.width * src_img.height
    skip_model = (
        target_area is not None
        and src_area >= int(target_area * SKIP_MODEL_RATIO)
    )

    if skip_model:
        # already big enough - LANCZOS-resize straight to the target area, bypassing the model
        out_img = resize_to_area(src_img, int(target_area))  # type: ignore[arg-type]
        action = "resize"
        scale = 1  # this path never touched the model, recorded as 1 = not upscaled
        dev = resolve_device(device)
        dtype = resolve_dtype(precision, dev)
    else:
        # go through the model upscale
        dev = resolve_device(device)
        dtype = resolve_dtype(precision, dev)
        descriptor = load_model(model_path, device=dev, dtype=dtype)
        scale = int(descriptor.scale)
        tensor = _img_to_tensor(src_img).to(dev, dtype=dtype)
        out_tensor = tiled_inference(
            descriptor.model,
            tensor,
            scale=scale,
            tile_size=tile_size,
            tile_pad=tile_pad,
        )
        out_img = _tensor_to_img(out_tensor)
        if target_area is not None:
            out_img = resize_to_area(out_img, int(target_area))
            action = "upscale+resize"
        else:
            action = "upscale"

    dst.parent.mkdir(parents=True, exist_ok=True)
    # `save_kwargs` decides the output format (defaults to lossless PNG,
    # matching historical behavior). The worker passes the format matching
    # src's extension, to preserve caption/dataset_config's dependency on the
    # extension (ADR 0010 fixup: never change the extension). An in-place
    # overwrite (src == dst) goes through tmp+rename to avoid a partial write
    # the training framework could read mid-write.
    final_save_kwargs: dict[str, Any] = (
        {"format": "PNG", "optimize": False}
        if save_kwargs is None
        else dict(save_kwargs)
    )
    tmp = dst.with_suffix(dst.suffix + ".upscale.tmp")
    out_img.save(tmp, **final_save_kwargs)
    import os as _os
    _os.replace(tmp, dst)

    # While the PIL Image is still in memory, pre-generate its thumbnail into the cache.
    # Without this, a user's first grid view decodes each PNG one at a time
    # (1-3s each, minutes for 200 images), whereas the worker has already
    # paid the decode cost here - spending a fraction of a second more to
    # generate the thumb is nearly free spread across the batch.
    if prewarm_thumb_sizes:
        try:
            from ..dataset import thumb_cache
            thumb_cache.prewarm_from_image(dst, out_img, prewarm_thumb_sizes)
        except Exception as exc:  # noqa: BLE001 - a thumb prewarm failure shouldn't affect the upscale itself
            on_log(f"   warning: thumb prewarm failed: {exc}")

    elapsed = time.monotonic() - t_start

    meta = {
        "source": src.name,
        "model": label,
        "action": action,
        "scale": scale,
        "target_area": target_area,
        "tile_size": tile_size,
        "tile_pad": tile_pad,
        "device": str(dev),
        "dtype": str(dtype).replace("torch.", ""),
        "src_size": list(src_size),
        "dst_size": list(out_img.size),
        "elapsed_seconds": round(elapsed, 3),
        "mtime": time.time(),
    }
    on_log(
        f"   OK [{action}] {src.name} -> {dst.name}  "
        f"{src_size[0]}x{src_size[1]} -> {out_img.size[0]}x{out_img.size[1]}  "
        f"({elapsed:.1f}s)"
    )
    return meta
