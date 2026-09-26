"""Anima Transformer / VAE / text encoder loading (public API, imported directly by sister scripts).

Extracted from the original runtime/anima_train.py L614-775 (ADR 0003 PR-A).

Public (called by anima_daemon / anima_generate / anima_reg_ai via anima_train.X):
- load_anima_model -- Anima Transformer + flash_attn toggle + checkpoint-inferred config
- load_vae -- WAN VAE + normalization wrapper
- load_text_encoders -- Qwen + T5 tokenizer

Internal:
- ensure_models_namespace -- adds the model code directory to sys.path
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import torch

from training.model_loading import (
    _load_safetensors_state_dict,
    _load_weights_best_effort,
)


logger = logging.getLogger(__name__)


class VAEWrapper:
    """WAN VAE + normalization params + auto fallback from full-image decode
    OOM to tiled decode.

    Issue #200: running a 1024x1024 reg generation on an 8GB-class small
    card, with transformer/Qwen/T5 resident on GPU, leaves ~1GB, and
    full-image decode's working memory fills it up -> OOM. This class
    `try`s the full image first at the `decode()` entry point, and only
    falls back to cosine-blend tiled decode on a CUDA OOM (each tile is 64
    latent / 512 pixels, single-tile working peak is about 75MB).

    Follow-up (the VRAM-cliff fix): both `decode()` / `encode()` now decide
    via `tiling` (auto/on/off), where auto proactively tiles once "already
    used + estimated peak" crosses a total-VRAM threshold -- not just
    waiting for OOM. On a large-VRAM card, a full-image op that nearly fills
    VRAM can trigger WDDM VRAM paging, degrading a single op from <1s to
    hundreds of seconds, without throwing a clean OOM -- a reactive fallback
    can't save it, so it has to be proactive.

    Callers should go through `wrapper.encode(pixels)` / `wrapper.decode(z)`
    (which include the tiling decision), not call
    `wrapper.model.encode/decode(...)` directly (which bypasses tiling).
    """

    # Tile geometry: tile=512px / overlap=128px / 4-stage VAE 8x upsample
    _TILE_LATENT = 64
    _STRIDE_LATENT = 48
    _UPSAMPLE = 8

    # Full-image decode peak VRAM ~= _DECODE_PEAK_BYTES_PER_OUT_PX x (elem/4) x output pixels x B x T.
    # Calibrated from tools/spike/vae_stress.py (RTX 5090 / WAN VAE dim=96): fp32 1024^2~=10.4G,
    # 1536^2~=22.6G; bf16 halves it. Using 11000 (measured ~9.8k) leaves ~12% headroom.
    _DECODE_PEAK_BYTES_PER_OUT_PX = 11000
    # Full-image encode peak ~= _ENCODE_PEAK_BYTES_PER_IN_PX x (elem/4) x input pixels x B x T.
    # Calibrated from tools/spike/vae_stress.py (fp32): 1024^2~=5.5G, 1536^2~=11.6G, 2048^2~=20.3G.
    _ENCODE_PEAK_BYTES_PER_IN_PX = 5500
    # auto: tile whenever "currently used + estimated peak" exceeds this
    # fraction of total VRAM. The cliff sits at ~50%, not at full VRAM --
    # fp32 1536^2 (peak 22.6G / total 31.8G = 71%) degrades to ~190s from
    # WDDM VRAM paging even though it "fits"; fp32 1024^2 (10.4G / 33%) is
    # fine. 0.5 lets the fast path (fp32<=1024 / bf16<=1536) go full-image,
    # while routing anything that would hit the cliff (large images, fp32,
    # or stacked with resident models) to tiling.
    _TILE_VRAM_FRACTION = 0.5

    def __init__(self, model, mean, std, tiling: str = "auto"):
        self.model = model
        self.mean = mean
        self.std = std
        self.scale = [mean, 1.0 / std]
        # VAE weight precision (mean/std share model's dtype). encode/decode
        # cast their input to this at the entry point, so that fp16 training
        # + fp32 VAE with a caller passing fp16 latent/pixel doesn't get a
        # dtype mismatch.
        self.dtype = mean.dtype
        # Tiling mode:
        #   auto (default) = estimated from free VRAM, proactively tiles
        #                     when the full-image peak approaches available VRAM;
        #   on             = always tiled (saves VRAM, about 30% slower);
        #   off            = full image, only falls back to tiling on a real OOM (old behavior).
        # auto fixes the freeze where a large-VRAM card's near-full
        # full-image decode triggers a "system memory fallback" ->
        # degrading a single decode from <1s to hundreds of seconds
        # (a reactive OOM fallback can't save it because no clean OOM is thrown).
        self.tiling = str(tiling or "auto").lower().strip()
        # Dedup for tiling-decision / OOM-fallback logging: latent caching
        # calls encode per bucket (200 images of different sizes -> 200
        # calls), and logging every one would spam the log. Each kind of
        # event is logged only once per wrapper instance.
        self._logged_once: set[str] = set()

    def _log_once(self, key: str, log_fn, msg: str, *args) -> None:
        """Only emits the log once per ``key`` for a given wrapper instance (silent thereafter).

        During caching, the transformer/Qwen are already resident on GPU
        (models_phase runs before dataset_phase), so free VRAM is low ->
        auto decides to tile for almost every image. Tiling itself is
        correct here (a full-image op would push usage over the WDDM VRAM
        paging cliff); it's just that logging it per image adds no
        information, hence the dedup.
        """
        if key in self._logged_once:
            return
        self._logged_once.add(key)
        log_fn(msg, *args)

    def _est_decode_peak_bytes(self, z) -> int:
        b, _c, t, H, W = z.shape
        out_px = (H * self._UPSAMPLE) * (W * self._UPSAMPLE)
        elem = z.element_size()  # 2=bf16/fp16, 4=fp32
        return int(self._DECODE_PEAK_BYTES_PER_OUT_PX * (elem / 4.0) * out_px * b * max(1, t))

    def _est_encode_peak_bytes(self, pixels) -> int:
        b, _c, t, H, W = pixels.shape  # H/W are pixel resolution
        in_px = H * W
        elem = pixels.element_size()
        return int(self._ENCODE_PEAK_BYTES_PER_IN_PX * (elem / 4.0) * in_px * b * max(1, t))

    @classmethod
    def _should_auto_tile(cls, used_bytes: int, est_peak_bytes: int, total_bytes: int) -> bool:
        """auto decision: whether currently-used + estimated decode peak crosses the tiling threshold of total VRAM."""
        return (used_bytes + est_peak_bytes) > total_bytes * cls._TILE_VRAM_FRACTION

    def should_offload_for_whole_decode(self, z) -> bool:
        """Whether it's worth moving inactive modules (DiT/Qwen) to CPU to
        free VRAM before sampling decode.

        True only when "VRAM is tight (would tile right now) **and** the
        peak is still under the cliff": in that case, freeing resident
        modules is enough to do a full-image decode and keep parity / no
        tile seams. False once the peak is already over the cliff (e.g. fp32
        1536^2) -- freeing VRAM can't save the full image then (the cliff is
        a fraction of total VRAM, independent of free), so tiling is the
        more memory-efficient choice.
        """
        if not (torch.is_tensor(z) and z.is_cuda and torch.cuda.is_available()):
            return False
        free, total = torch.cuda.mem_get_info()
        est = self._est_decode_peak_bytes(z)
        return self._should_auto_tile(total - free, est, total) and est < total * self._TILE_VRAM_FRACTION

    def decode(self, z):
        """latent -> pixel; decides full-image / tiled based on self.tiling.

        z: ``[b, 16, t, H, W]`` latent
        return: ``[b, 3, t, H*8, W*8]`` (matches the underlying `WanVAE_.decode`, not clamped)
        """
        z = z.to(self.dtype)  # align to VAE weight precision (fp16 training / fp32 VAE; no-op when dtypes already match)
        if self.tiling == "on":
            return self._tiled_decode(z)

        if self.tiling == "auto" and z.is_cuda and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            est = self._est_decode_peak_bytes(z)
            if self._should_auto_tile(used, est, total):
                # debug level: during caching the model is resident ->
                # free is low -> tiling almost every image is expected;
                # not shown by default, to avoid "used X > total VRAM" being
                # misread as insufficient VRAM.
                self._log_once(
                    "auto_decode", logger.debug,
                    "VAE decode proactively tiled (tiling=auto): used %.1fG + estimated peak %.1fG > total VRAM %.1fG x %.2f",
                    used / 1024 ** 3, est / 1024 ** 3, total / 1024 ** 3, self._TILE_VRAM_FRACTION,
                )
                return self._tiled_decode(z)

        # off, or auto decided VRAM is sufficient: full image, OOM fallback still kept (a safety net for small-VRAM cards).
        try:
            return self.model.decode(z, self.scale)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log_once(
                "oom_decode", logger.warning,
                "VAE full-image decode OOM, falling back to tiled decode "
                "(tile=%dpx, overlap=%dpx) (won't repeat this log again)",
                self._TILE_LATENT * self._UPSAMPLE,
                (self._TILE_LATENT - self._STRIDE_LATENT) * self._UPSAMPLE,
            )
            return self._tiled_decode(z)

    def encode(self, pixels):
        """pixel -> latent; decides full-image / tiled based on self.tiling.

        pixels: ``[b, 3, t, H, W]`` (H/W are pixel resolution, must be a multiple of 8)
        return: ``[b, 16, t, H/8, W/8]`` latent (matches `WanVAE_.encode`)
        """
        pixels = pixels.to(self.dtype)  # align to VAE weight precision (same as decode; no-op when dtypes already match)
        if self.tiling == "on":
            return self._tiled_encode(pixels)

        if self.tiling == "auto" and pixels.is_cuda and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            est = self._est_encode_peak_bytes(pixels)
            if self._should_auto_tile(used, est, total):
                # debug level: during caching the model is resident ->
                # free is low -> tiling almost every image is expected;
                # not shown by default, to avoid "used X > total VRAM" being
                # misread as insufficient VRAM.
                self._log_once(
                    "auto_encode", logger.debug,
                    "VAE encode proactively tiled (tiling=auto): used %.1fG + estimated peak %.1fG > total VRAM %.1fG x %.2f",
                    used / 1024 ** 3, est / 1024 ** 3, total / 1024 ** 3, self._TILE_VRAM_FRACTION,
                )
                return self._tiled_encode(pixels)

        try:
            return self.model.encode(pixels, self.scale)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log_once(
                "oom_encode", logger.warning,
                "VAE full-image encode OOM, falling back to tiled encode "
                "(tile=%dpx, overlap=%dpx) (won't repeat this log again)",
                self._TILE_LATENT * self._UPSAMPLE,
                (self._TILE_LATENT - self._STRIDE_LATENT) * self._UPSAMPLE,
            )
            return self._tiled_encode(pixels)

    @torch.no_grad()
    def _tiled_decode(self, z):
        """Tile independently along H/W and decode each, stitched back with cosine blend.

        - tile=64 latent, stride=48, overlap=16 latent (128 px); the last
          tile's start is clamped to (size - tile) to prevent overrun
        - a small image (H or W < tile) uses ``eff_h/eff_w = min(tile, size)``, equivalent to a single full-image tile
        - blending uses a raised-cosine edge ramp, with the accumulator in fp32 to prevent bf16 precision loss
        - the mask's minimum is clamped to 1e-6 to prevent divide-by-zero in wsum at the corners
        """
        b, _c, t, H, W = z.shape
        tile = self._TILE_LATENT
        stride = self._STRIDE_LATENT
        up = self._UPSAMPLE

        eff_h = min(tile, H)
        eff_w = min(tile, W)
        hs = _tile_starts(H, eff_h, stride)
        ws = _tile_starts(W, eff_w, stride)

        tile_h_px = eff_h * up
        tile_w_px = eff_w * up
        overlap_px = (tile - stride) * up
        H_px, W_px = H * up, W * up

        acc = torch.zeros(b, 3, t, H_px, W_px, dtype=torch.float32, device=z.device)
        wsum = torch.zeros(1, 1, 1, H_px, W_px, dtype=torch.float32, device=z.device)

        mask = _cosine_blend_mask(tile_h_px, tile_w_px, fade=overlap_px, device=z.device)

        for hi in hs:
            for wi in ws:
                z_tile = z[:, :, :, hi:hi + eff_h, wi:wi + eff_w]
                img_tile = self.model.decode(z_tile, self.scale).float()
                hp, wp = hi * up, wi * up
                acc[:, :, :, hp:hp + tile_h_px, wp:wp + tile_w_px] += img_tile * mask
                wsum[:, :, :, hp:hp + tile_h_px, wp:wp + tile_w_px] += mask

        return (acc / wsum).to(z.dtype)

    @torch.no_grad()
    def _tiled_encode(self, pixels, tile_px=None, overlap_px=None):
        """Tile independently along H/W and encode each, stitched back in latent space with cosine blend.

        - default tile=512px / stride=384px / overlap=128px (= decode's 64/48/16 latent x8)
        - ``tile_px`` / ``overlap_px`` (pixels), when not None, override the
          defaults, for cache tiling to pass in via config; both must be an
          integer multiple of the VAE downsample factor (``up``) (latent tile boundaries must land on whole cells)
        - the pixel start is always a multiple of 8 -> the latent position is
          an integer (_TILE/_STRIDE x 8 and H/W are all divisible by 8)
        - blending happens in latent space, with the accumulator in fp32
        - encode is non-linear, so stitching is an approximation
          (overlap+cosine smooths the tile boundaries), which is good enough for latent caching
        """
        b, _c, t, H, W = pixels.shape
        up = self._UPSAMPLE
        if tile_px is None:
            tile_px = self._TILE_LATENT * up
            stride_px = self._STRIDE_LATENT * up
        else:
            tile_px = int(tile_px)
            ov_px = int(overlap_px) if overlap_px is not None else (self._TILE_LATENT - self._STRIDE_LATENT) * up
            for _name, _v in (("tile_px", tile_px), ("overlap_px", ov_px)):
                if _v % up != 0:
                    raise ValueError(
                        f"_tiled_encode requires {_name}={_v} to be an integer multiple of the VAE downsample factor {up}"
                        " (latent tile boundaries must land on whole cells)."
                    )
            if not (0 <= ov_px < tile_px):
                raise ValueError(f"overlap_px={ov_px} must satisfy 0 <= overlap < tile_px={tile_px}")
            stride_px = tile_px - ov_px

        eff_h = min(tile_px, H)
        eff_w = min(tile_px, W)
        hs = _tile_starts(H, eff_h, stride_px)
        ws = _tile_starts(W, eff_w, stride_px)

        lat_h, lat_w = H // up, W // up
        tile_lh, tile_lw = eff_h // up, eff_w // up
        overlap_lat = (tile_px - stride_px) // up

        # encode output channels = z_dim(16). Reading the real channel count from the first tile would be more robust, but WAN VAE fixes it at 16.
        acc = torch.zeros(b, 16, t, lat_h, lat_w, dtype=torch.float32, device=pixels.device)
        wsum = torch.zeros(1, 1, 1, lat_h, lat_w, dtype=torch.float32, device=pixels.device)

        mask = _cosine_blend_mask(tile_lh, tile_lw, fade=overlap_lat, device=pixels.device)

        for hi in hs:
            for wi in ws:
                px_tile = pixels[:, :, :, hi:hi + eff_h, wi:wi + eff_w]
                z_tile = self.model.encode(px_tile, self.scale).float()
                lh, lw = hi // up, wi // up
                acc[:, :, :, lh:lh + tile_lh, lw:lw + tile_lw] += z_tile * mask
                wsum[:, :, :, lh:lh + tile_lh, lw:lw + tile_lw] += mask

        return (acc / wsum).to(pixels.dtype)


def _tile_starts(size: int, tile: int, stride: int) -> list[int]:
    """List of tile start positions at a fixed stride; appends ``size - tile`` if the tail is left uncovered."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


def _cosine_blend_mask(h: int, w: int, *, fade: int, device) -> torch.Tensor:
    """2D raised-cosine mask; a 0->1 ramp within `fade` pixels of the edge, 1 in the center.

    The final mask is clamp_min(1e-6)'d as a whole to prevent divide-by-zero
    in wsum at the corners -- a 2D corner is 1D x 1D, so a 1e-6 clamp on the
    1D ramps would become 1e-12 at a 2D corner, too close to fp32 denormal;
    clamping is done uniformly after the 2D multiply instead.
    """
    def ramp_1d(n: int) -> torch.Tensor:
        m = torch.ones(n, dtype=torch.float32, device=device)
        if fade > 0:
            t = torch.linspace(math.pi, 2 * math.pi, fade, device=device)
            r = torch.cos(t) * 0.5 + 0.5
            m[:fade] = r
            m[-fade:] = r.flip(0)
        return m
    mh = ramp_1d(h)
    mw = ramp_1d(w)
    mask_2d = (mh[:, None] * mw[None, :]).clamp_min(1e-6)
    return mask_2d[None, None, None, :, :]


def load_vae(vae_path, device, dtype, repo_root, *, tiling: str = "auto"):
    """Load the VAE. ``tiling`` is passed straight through to VAEWrapper (auto/on/off)."""
    # Normal import (exec-load retired, multi-model PR-2a); repo_root parameter kept but no longer used
    from modeling.wan.vae2_1 import WanVAE_ as WanVAE

    cfg = dict(
        dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True], dropout=0.0,
    )

    model = WanVAE(**cfg).eval().requires_grad_(False)

    sd = _load_safetensors_state_dict(Path(vae_path))
    _load_weights_best_effort(model, sd, label="VAE")
    model = model.to(device=device, dtype=dtype)

    # VAE normalization parameters
    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ], dtype=dtype, device=device)
    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ], dtype=dtype, device=device)

    logger.info("VAE loading complete")
    return VAEWrapper(model, mean, std, tiling=tiling)
