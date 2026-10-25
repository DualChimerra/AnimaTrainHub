"""Experimental MOSS adapter options: tone head, timestep modulation, image-tokens-only LoRA,
per-block norms and per-block modulation.

- Tone head: LoRA on ``final_linear_video`` (hidden -> 128 latent channels) and
  ``video_head_modulation`` (per-timestep scale/shift before it). These turn the
  transformer result into latent colors, so the adapter gets a direct handle on palette
  and contrast.
- Timestep modulation: LoRA on ``shared_block_modulation``, the one per-timestep
  shift/scale/gate shared by all 42 blocks. Its input is the timestep embedding only, so
  the adapter learns a different "volume" per noise level.
- Image tokens only: MOSS is single-stream, the same Linear layers process image and text
  tokens ([image N, text L] per sequence). The LoRA diff is zeroed on text positions and
  the text refiner is left untouched, so the prompt is read exactly as by the base model.
  Needs the LyCORIS bypass forward (diff computed separately from the base layer).
- Per-block norms: train the norm1 / norm2 weights of every block — a per-block,
  per-channel gain in front of attention and the FFN (the AdaIN-style "volume" of each
  feature at each depth). Exported as ComfyUI ``.diff`` keys through TrainNorm.
- Per-block modulation: MOSS shares one timestep modulation across all blocks (Anima has
  its own per block). A zero-initialised low-rank map from the timestep embedding gives
  every block its own offset to shift / scale / gate. New parameters, so it only runs
  where this module is attached (training, training samples, the studio's Generate).
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable

import torch
from torch import nn

logger = logging.getLogger(__name__)

TONE_HEAD_MODULES = ("final_linear_video", "video_head_modulation")
TIME_MODULATION_MODULES = ("shared_block_modulation",)
IMAGE_TOKENS_ONLY_KEY = "moss_image_tokens_only"
_BLOCK_LORA_PREFIX = "lora_unet_layers_"
BLOCK_NORM_NAMES = ["layers.*.norm1", "layers.*.norm2"]
_BLOCK_NORM_KEY_RE = re.compile(r"^lora_unet_layers_\d+_norm[12]\.diff$")
BLOCK_MOD_PREFIX = "moss_block_mod."


def _get(args: Any, key: str, default: Any) -> Any:
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def adapt_preset(preset: dict[str, Any], args: Any) -> dict[str, Any]:
    """Return the MOSS LyCORIS preset extended / narrowed by the experimental options."""
    preset = dict(preset)
    names = list(preset.get("target_name", []))
    if _get(args, "moss_tone_head", False):
        names += [n for n in TONE_HEAD_MODULES if n not in names]
    if _get(args, "moss_time_modulation", False):
        names += [n for n in TIME_MODULATION_MODULES if n not in names]
    preset["target_name"] = names
    if _get(args, IMAGE_TOKENS_ONLY_KEY, False):
        preset["target_module"] = [m for m in preset.get("target_module", []) if m != "MossImageTextBlock"]
    if _get(args, "moss_block_norms", False):
        preset["train_norm_names"] = list(BLOCK_NORM_NAMES)
    return preset


def extra_reg_dims(args: Any) -> dict[str, int]:
    """Per-module ranks for the extra targets (merged into lora_reg_dims, saved in the LoRA metadata)."""
    dims: dict[str, int] = {}
    if _get(args, "moss_tone_head", False):
        rank = int(_get(args, "moss_tone_head_rank", 16))
        dims.update({f"lora_unet_{n}": rank for n in TONE_HEAD_MODULES})
    if _get(args, "moss_time_modulation", False):
        rank = int(_get(args, "moss_time_modulation_rank", 4))
        dims.update({f"lora_unet_{n}": rank for n in TIME_MODULATION_MODULES})
    return dims


def preset_for_state_dict(preset: dict[str, Any], keys) -> dict[str, Any]:
    """Inference: extend the preset with the extra targets a LoRA file actually contains."""
    keys = list(keys)
    preset = dict(preset)
    names = list(preset.get("target_name", []))
    for module in TONE_HEAD_MODULES + TIME_MODULATION_MODULES:
        if module not in names and any(k.startswith(f"lora_unet_{module}.") for k in keys):
            names.append(module)
    preset["target_name"] = names
    if any(_BLOCK_NORM_KEY_RE.match(k) for k in keys):
        preset["train_norm_names"] = list(BLOCK_NORM_NAMES)
    return preset


def install_image_token_mask(model: torch.nn.Module, network: Any) -> int:
    """Zero the LoRA diff on text positions of every main-block LoRA module.

    The image length comes from a forward pre-hook on the MOSS model (x: [B, C, H, W] ->
    N = H * W). It is kept after the forward on purpose: gradient checkpointing recomputes
    the blocks during backward, after forward() has returned.
    """
    state = getattr(model, "_moss_token_mask_state", None)
    if state is None:
        state = {"img_len": None}

        def _pre_hook(_module, args, kwargs):
            x = args[0] if args else kwargs.get("x")
            if isinstance(x, torch.Tensor) and x.dim() == 4:
                state["img_len"] = int(x.shape[-2] * x.shape[-1])

        model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        model._moss_token_mask_state = state

    patched = 0
    for lora in getattr(network, "loras", []) or []:
        name = getattr(lora, "lora_name", "") or ""
        if not name.startswith(_BLOCK_LORA_PREFIX) or getattr(lora, "_moss_image_only", False):
            continue
        if not getattr(lora, "bypass_mode", False):
            raise ValueError(f"image-tokens-only LoRA needs the bypass forward, {name} is not in bypass mode")
        original = lora.bypass_forward_diff

        def _image_only_diff(x, scale=1, *, _original=original):
            diff = _original(x, scale=scale)
            n = state["img_len"]
            if n is not None and diff.dim() == 3 and diff.shape[1] > n:
                keep = torch.arange(diff.shape[1], device=diff.device) < n
                diff = diff * keep.to(diff.dtype).view(1, -1, 1)
            return diff

        lora.bypass_forward_diff = _image_only_diff
        lora._moss_image_only = True
        patched += 1
    logger.info("MOSS image-tokens-only LoRA: masked text positions in %d block layers", patched)
    return patched


class BlockModulation:
    """Per-block low-rank offset to MOSS's shared timestep modulation.

    delta_i(t) = up_i @ down_i @ time_embedding(t), reshaped to [B, 6, D] (shift / scale /
    gate for attention and FFN) and added to the shared modulation entering block i.
    ``up`` starts at zero, so the model is unchanged until training moves it.
    Saved as ``moss_block_mod.down`` [blocks, rank, dim] and ``moss_block_mod.up``
    [blocks, 6 * dim, rank] in the LoRA file; ComfyUI ignores these keys.
    """

    def __init__(self, num_blocks: int, dim: int, rank: int, multiplier: Callable[[], float],
                 *, device: torch.device | str = "cpu", trainable: bool = True) -> None:
        down = torch.empty(num_blocks, rank, dim, device=device, dtype=torch.float32)
        for block in down:
            nn.init.kaiming_uniform_(block, a=math.sqrt(5))
        self.down = nn.Parameter(down, requires_grad=trainable)
        self.up = nn.Parameter(torch.zeros(num_blocks, 6 * dim, rank, device=device, dtype=torch.float32),
                               requires_grad=trainable)
        self._multiplier = multiplier
        self._handles: list[Any] = []
        self._model: nn.Module | None = None

    @classmethod
    def from_state_dict(cls, sd: dict[str, torch.Tensor], multiplier: Callable[[], float],
                        *, device: torch.device | str = "cpu") -> "BlockModulation":
        down, up = sd[BLOCK_MOD_PREFIX + "down"], sd[BLOCK_MOD_PREFIX + "up"]
        mod = cls(down.shape[0], down.shape[2], down.shape[1], multiplier, device=device, trainable=False)
        mod.load(sd)
        return mod

    # ------------------------------------------------------------------ math
    def delta(self, index: int, time: torch.Tensor) -> torch.Tensor:
        """Offset for block ``index``: [B, 6, D] in float32."""
        t = time.float().to(self.down.device)
        hidden = t @ self.down[index].transpose(0, 1)
        out = hidden @ self.up[index].transpose(0, 1)
        return out.view(t.shape[0], 6, -1) * float(self._multiplier())

    # ------------------------------------------------------------------ hooks
    def attach(self, model: nn.Module) -> None:
        state: dict[str, torch.Tensor | None] = {"time": None}

        def _capture_time(_module, _inputs, output):
            state["time"] = output

        self._handles.append(model.time_embedding.register_forward_hook(_capture_time))
        for index, block in enumerate(model.layers):
            def _pre_hook(_module, args, kwargs, _index=index):
                time = state["time"]
                if time is None or float(self._multiplier()) == 0.0:
                    return None
                args = list(args)
                if len(args) > 2:
                    args[2] = args[2] + self.delta(_index, time).to(args[2].dtype)
                else:
                    kwargs["modulation"] = kwargs["modulation"] + self.delta(_index, time).to(
                        kwargs["modulation"].dtype)
                return tuple(args), kwargs

            self._handles.append(block.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        # comfy_forward (Generate) walks the blocks without calling them: it reads this list
        mods = getattr(model, "_moss_block_mods", None)
        if mods is None:
            mods = []
            model._moss_block_mods = mods
        mods.append(self)
        self._model = model

    # ------------------------------------------------------------------ adapter-extra protocol
    def parameters(self) -> list[nn.Parameter]:
        return [self.down, self.up]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {BLOCK_MOD_PREFIX + "down": self.down.detach(), BLOCK_MOD_PREFIX + "up": self.up.detach()}

    def load(self, sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if BLOCK_MOD_PREFIX + "up" not in sd:
            # A file without these keys means "no per-block modulation" (Generate hot-swapping
            # to an ordinary LoRA, or resuming from one): back to the zero-effect init
            with torch.no_grad():
                self.up.zero_()
        for name in ("down", "up"):
            key = BLOCK_MOD_PREFIX + name
            if key in sd:
                param = getattr(self, name)
                if tuple(sd[key].shape) != tuple(param.shape):
                    raise ValueError(f"{key}: shape {tuple(sd[key].shape)} does not match {tuple(param.shape)}")
                with torch.no_grad():
                    param.copy_(sd[key].to(device=param.device, dtype=param.dtype))
        return {k: v for k, v in sd.items() if not k.startswith(BLOCK_MOD_PREFIX)}

    def detach(self) -> bool:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if self._model is not None:
            mods = getattr(self._model, "_moss_block_mods", [])
            if self in mods:
                mods.remove(self)
            self._model = None
        return True


class BlockModulationHandle:
    """Generate-side handle for a merged (quantized) base: no LyCORIS network to hang on."""

    network = None
    supports_hot_reload = False

    def __init__(self, module: BlockModulation, scale: float) -> None:
        self._module = module
        self._scale = float(scale)
        module._multiplier = lambda: self._scale

    def set_multiplier(self, scale: float) -> None:
        self._scale = float(scale)

    def detach(self) -> bool:
        return self._module.detach()


def has_block_modulation(keys) -> bool:
    return any(k.startswith(BLOCK_MOD_PREFIX) for k in keys)


def attach_block_modulation_from_state_dict(model: nn.Module, sd: dict[str, torch.Tensor],
                                            multiplier: Callable[[], float]) -> BlockModulation:
    device = next(model.parameters()).device
    mod = BlockModulation.from_state_dict(sd, multiplier, device=device)
    mod.attach(model)
    return mod
