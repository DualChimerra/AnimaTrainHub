"""Krea2 Qwen3-VL text conditioning and variable-length cache lifecycle.

The prompt template, selected hidden layers, interior-padding gather, and output
layout are adapted from kohya-ss/musubi-tuner's Krea2 text encoder and cache
implementation at commit 8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6.
Copyright 2026 Kohya S. and musubi-tuner contributors. Apache-2.0.
https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/krea2/krea2_encoder.py
https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/krea2_cache_text_encoder_outputs.py

This repository loads the official sharded Hugging Face directory, uses the
shared ``TextCacheStore`` sidecar protocol, and owns the lazy load/release
lifecycle needed to keep the 4B text encoder out of VRAM after pre-caching.
"""

from __future__ import annotations

import gc
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from ...text_cache import TextCacheEntry, TextCacheStore


logger = logging.getLogger(__name__)

KREA2_TEXT_FINGERPRINT = "qwen3-vl-4b-instruct-krea2-12x2560-v1"
KREA2_MAX_LENGTH = 512
KREA2_SELECTED_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
KREA2_TEXT_WIDTH = 2560

_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_PREFIX_TOKENS = 34
_SUFFIX_TOKENS = 5
_CACHE_TENSOR_KEY = "context"


@dataclass(frozen=True)
class Krea2TextCondition:
    """Padded Krea2 text condition consumed by the DiT."""

    context: Tensor
    attention_mask: Tensor


def gather_valid_text(hidden_states: Tensor, attention_mask: Tensor) -> list[Tensor]:
    """Gather valid tokens in order, including suffix tokens after interior padding."""

    if hidden_states.ndim < 3:
        raise ValueError("Krea2 hidden_states must be at least (B, seq, features)")
    if attention_mask.ndim != 2:
        raise ValueError("Krea2 attention_mask must be (B, seq)")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError(
            "Krea2 hidden_states and attention_mask batch/seq dims disagree: "
            f"{tuple(hidden_states.shape[:2])} != {tuple(attention_mask.shape)}"
        )

    mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
    gathered = [hidden_states[index][mask[index]] for index in range(mask.shape[0])]
    if any(item.shape[0] == 0 for item in gathered):
        raise ValueError("Krea2 text condition cannot have zero valid tokens")
    return gathered


def pad_text_conditions(
    contexts: Sequence[Tensor],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Krea2TextCondition:
    """Right-pad variable-length ``(seq, layers, width)`` tensors for one batch."""

    if not contexts:
        raise ValueError("Krea2 text batch cannot be empty")
    shape = tuple(contexts[0].shape[1:])
    if contexts[0].ndim != 3 or any(
        item.ndim != 3 or tuple(item.shape[1:]) != shape or item.shape[0] == 0
        for item in contexts
    ):
        raise ValueError("Krea2 context must be non-empty (seq, layers, width) tensors with matching layer count/width")

    target = torch.device(device)
    max_length = max(item.shape[0] for item in contexts)
    padded = torch.zeros(
        len(contexts), max_length, *shape, device=target, dtype=dtype,
    )
    mask = torch.zeros(len(contexts), max_length, device=target, dtype=torch.bool)
    for index, item in enumerate(contexts):
        length = item.shape[0]
        padded[index, :length].copy_(item.to(device=target, dtype=dtype))
        mask[index, :length] = True
    return Krea2TextCondition(context=padded, attention_mask=mask)


def _cast_linear_forward(self, input: Tensor) -> Tensor:
    # ComfyUI manual_cast semantics (ops.py cast_bias_weight): weights stay
    # resident at low precision, cast to input.dtype for the forward compute.
    # fp16->fp32 cast is exact, so per-layer casting is bit-identical to
    # upcasting the whole model -- the real tradeoff is VRAM (8.9GB vs 17.8GB).
    weight = self.weight.to(input.dtype)
    bias = self.bias.to(input.dtype) if self.bias is not None else None
    return torch.nn.functional.linear(input, weight, bias)


def _cast_embedding_forward(self, input: Tensor) -> Tensor:
    # comfy ops Embedding: cast the weight to compute dtype, then look up
    # (row-select and cast commute, numerically equivalent) -- this is how
    # the activation stream enters the compute-dtype domain at the source.
    return torch.nn.functional.embedding(
        input,
        self.weight.to(self._krea2_compute_dtype),
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse,
    )


def patch_manual_cast(model: torch.nn.Module, compute_dtype: torch.dtype) -> int:
    """Patch equivalent to ComfyUI's manual_cast (sd.py:258 ``set_model_compute_dtype``).

    Once the Embedding output enters the compute-dtype domain, every Linear
    casts its low-precision weight to input.dtype (=compute dtype) for the
    compute, layer by layer; RMSNorm / rotary need no patch -- in the
    transformers implementation, the fp32 activation stream combined with
    torch's type promotion is numerically equivalent to Comfy's weight-cast
    semantics. Returns the number of patched modules.
    """
    from training.families.krea2.quant_fp8 import _FP8_TORCH_DTYPES  # noqa: PLC0415

    patched = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.dtype in _FP8_TORCH_DTYPES:
                # fp8_scaled layers already have a dequant forward attached
                # (cast to input.dtype + multiply by scale = the fp8 version of
                # manual_cast); overwriting it would lose the scale
                continue
            module.forward = MethodType(_cast_linear_forward, module)
            patched += 1
        elif isinstance(module, torch.nn.Embedding):
            module._krea2_compute_dtype = compute_dtype
            module.forward = MethodType(_cast_embedding_forward, module)
            patched += 1
    if patched:
        logger.info(
            "Krea2 TE manual_cast: %d modules computing in %s (weights stay resident at storage dtype)",
            patched, compute_dtype,
        )
    return patched


#: Text-side key prefixes for comfy's single-file TE -> HF
#: Qwen3VLForConditionalGeneration keys. The comfy packaging (Comfy-Org
#: qwen3vl_4b_fp8_scaled) mounts language_model directly under ``model.``;
#: the visual side (model.visual.*) matches on both sides with no mapping needed.
_COMFY_TE_TEXT_PREFIXES = ("model.layers.", "model.embed_tokens.", "model.norm.")


def _comfy_te_single_file(model_path: Path) -> Path | None:
    """Return the weights file if the directory is comfy's single-file TE layout; return None for an HF sharded layout."""
    if (model_path / "model.safetensors.index.json").exists():
        return None
    candidates = sorted(model_path.glob("*.safetensors"))
    return candidates[0] if len(candidates) == 1 else None


def _load_comfy_single_file_te(
    model_path: Path,
    weights_file: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    """Load a Qwen3-VL model in comfy's single-file layout (the official fp8_scaled form).

    Small config/tokenizer files still come from the directory (downloaded
    together with the fp8 entry by the download hub); weight keys get one
    prefix mapping (the text side gains ``language_model.``), the
    ``comfy_quant`` config blob is discarded, and the ``weight_scale`` F32
    scalars are collected and attached via patch_fp8_linears as a dequant
    forward (exactly like the DiT's fp8_scaled). fp8 weights stay resident
    as-is; non-quantized keys (embed/norm/visual) are cast to the storage
    dtype. lm_head is tied to embed -- the file doesn't contain that key, so
    tie_weights() re-ties it after loading.
    """
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    from training.families.krea2.quant_fp8 import (  # noqa: PLC0415
        _FP8_TORCH_DTYPES,
        patch_fp8_linears,
    )

    logger.info("Loading Krea2 Qwen3-VL (comfy single-file form): %s", weights_file)
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    try:
        from accelerate import init_empty_weights
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Qwen3-VL single-file loading requires accelerate (a transformers companion dependency)"
        ) from exc

    # init_empty_weights by default only puts parameters on meta; buffers
    # (rotary inv_freq etc., non-persistent, not in the weights file) are
    # constructed for real on CPU with correct values -- a plain
    # torch.device("meta") context would also meta-ize buffers, which can't
    # be recovered after loading (this used to crash offload's .to("cpu") on a meta tensor).
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)

    state_dict: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    with safe_open(str(weights_file), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.endswith(".comfy_quant"):
                continue
            mapped = key
            for prefix in _COMFY_TE_TEXT_PREFIXES:
                if key.startswith(prefix):
                    mapped = "model.language_model." + key[len("model."):]
                    break
            tensor = handle.get_tensor(key)
            if mapped.endswith(".weight_scale"):
                layer = mapped[: -len(".weight_scale")]
                scales[layer] = tensor.to(device=device)
                continue
            if tensor.dtype in _FP8_TORCH_DTYPES:
                state_dict[mapped] = tensor.to(device=device)
            else:
                state_dict[mapped] = tensor.to(device=device, dtype=dtype)

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    unexpected = list(result.unexpected_keys)
    missing = [k for k in result.missing_keys if k != "lm_head.weight"]
    if missing or unexpected:
        raise ValueError(
            f"Qwen3-VL comfy single-file key mismatch: missing {missing[:5]}, "
            f"unexpected {unexpected[:5]}"
        )
    # lm_head is tied to embed (tie_word_embeddings) -- the file doesn't
    # contain that key; transformers' tie_weights() doesn't re-tie it for a
    # meta-constructed + assign-loaded model, so point it back at embed
    # manually (zero-copy).
    out_emb = model.get_output_embeddings()
    if out_emb is not None and out_emb.weight.device.type == "meta":
        out_emb.weight = model.get_input_embeddings().weight
    # This check covers parameters + buffers (a missed buffer once left a
    # leftover meta rotary inv_freq: encoding happened to still run, but
    # offloading the whole model with .to("cpu") crashed on hitting it)
    leftover_meta = [
        name for name, tensor in [
            *model.named_parameters(), *model.named_buffers(),
        ]
        if tensor.device.type == "meta"
    ]
    if leftover_meta:
        raise ValueError(
            f"Qwen3-VL single-file load still has unmaterialized parameters: {leftover_meta[:5]}"
        )
    # Buffers are constructed on CPU (init_empty_weights only meta-izes
    # parameters) -- move them to the target device; parameters are already
    # assigned in place, so .to() is a no-op for them
    model.to(device)
    if scales:
        patch_fp8_linears(model, scales)
        logger.info("Qwen3-VL fp8_scaled: attached a dequant forward to %d layers", len(scales))
    return model.eval().requires_grad_(False)


def _default_model_loader(
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError(
            "The installed transformers does not include Qwen3VLForConditionalGeneration; "
            "please install a version that supports Qwen3-VL"
        ) from exc

    if not model_path.is_dir():
        raise ValueError(f"Krea2 text encoder must be a Hugging Face model directory: {model_path}")
    single = _comfy_te_single_file(model_path)
    if single is not None:
        return _load_comfy_single_file_te(model_path, single, device, dtype)
    logger.info("Loading Krea2 Qwen3-VL text encoder: %s", model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": str(device)},
    )
    return model.eval().requires_grad_(False)


def _load_tokenizer(model_path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("Krea2 text encoding requires transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    prefix_tokens = tokenizer(_PROMPT_PREFIX, add_special_tokens=False)["input_ids"]
    suffix_tokens = tokenizer(_PROMPT_SUFFIX, add_special_tokens=False)["input_ids"]
    if len(prefix_tokens) != _PREFIX_TOKENS or len(suffix_tokens) != _SUFFIX_TOKENS:
        raise ValueError(
            "Krea2 tokenizer is not compatible with the Qwen3-VL-4B-Instruct prompt template: "
            f"prefix={len(prefix_tokens)}, suffix={len(suffix_tokens)}"
        )
    return tokenizer


class Krea2TextStack:
    """Lazy Qwen3-VL conditioner with cached and storage-free online modes."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype | None = None,
        cache_enabled: bool = True,
        tokenizer=None,
        model_loader: Callable[[Path, torch.device, torch.dtype], Any] | None = None,
        text_fingerprint: str = KREA2_TEXT_FINGERPRINT,
        max_length: int = KREA2_MAX_LENGTH,
        selected_layers: Sequence[int] = KREA2_SELECTED_LAYERS,
        hidden_width: int = KREA2_TEXT_WIDTH,
        cache_batch_size: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.dtype = dtype
        # None = compute dtype follows the storage dtype (current training
        # path); the generate path passes fp32 + dtype=fp16 to replicate
        # ComfyUI's "fp16 storage + fp32 compute" convention (sd.py:258)
        self.compute_dtype = compute_dtype
        # Whether the directory is the official fp8_scaled single-file form
        # (comfy layout) -- affects the orchestration layer's VRAM budget
        # decision for loading the TE (~5GB weights vs fp16 8.9GB) and the training text cache fingerprint
        self.is_fp8_storage = _comfy_te_single_file(self.model_path) is not None
        self.cache_enabled = bool(cache_enabled)
        self.tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(self.model_path)
        self._model_loader = model_loader or _default_model_loader
        self._model = None
        self._offloaded = False
        # Embeddings encoded by an fp8 TE differ from bf16 at quantization
        # level -- distinguishing the fingerprint prevents mixing cache
        # sources (changing TE precision -> fingerprint changes -> a full,
        # one-time sidecar re-encode)
        if self.is_fp8_storage:
            text_fingerprint = f"{text_fingerprint}-tefp8"
        self.store = TextCacheStore(text_fingerprint)
        self.max_length = int(max_length)
        self.selected_layers = tuple(int(index) for index in selected_layers)
        self.hidden_width = int(hidden_width)
        self.cache_batch_size = int(cache_batch_size)
        self._caption_entries: dict[str, list[TextCacheEntry]] = {}
        self._prompt_captions: list[str] = []
        self._cache_root: Path | None = None
        # In-memory prompt->context LRU for online mode (generate) (same
        # semantics as Comfy's conditioning node cache): a hit means the TE
        # is never touched. Stored on CPU, keeping the original dtype (one
        # fp32 entry is about 63MB, capacity 16 is about 1GB RAM) -- casting
        # would break "the first image and a cache-hit image are
        # bit-identical". The cached mode (training) doesn't use this path.
        self._online_lru: OrderedDict[str, Tensor] = OrderedDict()
        self._online_lru_capacity = 16

        if self.max_length <= 0 or not self.selected_layers or self.hidden_width <= 0:
            raise ValueError("Krea2 text encoding config must be positive and selected_layers cannot be empty")
        if self.cache_batch_size <= 0:
            raise ValueError("Krea2 cache_batch_size must be positive")

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_model_on_device(self) -> bool:
        """Whether the TE currently resides on the target device (used by the orchestration layer to decide whether it needs to free VRAM to move it there)."""
        return self._model is not None and not self._offloaded

    def _online_lru_get(self, caption: str) -> Tensor | None:
        context = self._online_lru.get(caption)
        if context is not None:
            self._online_lru.move_to_end(caption)
        return context

    def _online_lru_put(self, caption: str, context: Tensor) -> None:
        self._online_lru[caption] = context
        self._online_lru.move_to_end(caption)
        while len(self._online_lru) > self._online_lru_capacity:
            self._online_lru.popitem(last=False)

    def online_conditions_cached(self, captions: Sequence[str]) -> bool:
        """Whether this batch of captions all hit the online LRU (an
        orchestration-layer peek: an all-hit means the TE doesn't need to go
        to GPU for the encoding stage, so the on-demand yield decision can be
        skipped entirely)."""
        return (
            not self.cache_enabled
            and all(str(caption) in self._online_lru for caption in captions)
        )

    def precache_online_prompts(self, captions: Sequence[str]) -> int:
        """Task-level pre-encode: encode this batch of captions into the
        online LRU (stored on CPU); returns the count newly encoded.

        The prompt set for XY / multi-prompt generate is closed before the
        task starts -- encode everything up front, then offload_model(), so
        the TE uses zero VRAM during sampling (the inference-side version of
        training's two-stage loading). Doesn't apply to the cached mode
        (training), no-op there. LRU capacity is raised to fit this batch's
        needs (capped at 64, about 30MB CPU RAM per entry); anything beyond
        that falls back to the per-cell lazy path.
        """
        if self.cache_enabled:
            return 0
        unique = list(dict.fromkeys(str(caption) for caption in captions))
        self._online_lru_capacity = max(
            self._online_lru_capacity, min(len(unique), 64),
        )
        missing = [c for c in unique[:64] if c not in self._online_lru]
        step = max(1, self.cache_batch_size)
        for start in range(0, len(missing), step):
            chunk = missing[start:start + step]
            for caption, context in zip(chunk, self._encode_many(chunk)):
                self._online_lru_put(caption, context.detach().to("cpu"))
        return len(missing)

    def ensure_model(self):
        if self._model is None:
            self._model = self._model_loader(self.model_path, self.device, self.dtype)
            if self.compute_dtype is not None and self.compute_dtype != self.dtype:
                patch_manual_cast(self._model, self.compute_dtype)
            # Return the TE weight file's (5-18GB) mmap cache pages to the system (a real machine hit a paging freeze from this)
            from training.sysmem import trim_working_set  # noqa: PLC0415

            trim_working_set()
        elif self._offloaded:
            # Was offloaded to CPU before the last sample (see offload_model) -- move back to the target device
            self._model.to(self.device)
            self._offloaded = False
        return self._model

    def offload_model(self) -> None:
        """Move the TE to CPU to free VRAM for the DiT (Comfy parity:
        free_memory's "unload CLIP to offload_device after encoding" semantics).

        In the generate scenario, DiT (26.3GB bf16) + Qwen3-VL (8.9GB)
        resident together is about 35GB, over the supported 32GB floor --
        the DiT must have the device to itself before sampling. The next
        prompt moves it back via ensure_model (GPU<->CPU takes seconds, far
        faster than reloading from disk after a release).
        """
        if self._model is None or self._offloaded:
            return
        self._model.to("cpu")
        self._offloaded = True
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_model(self) -> None:
        """Release the cached-mode TE before the 12.9B DiT occupies the device."""

        if self._model is None:
            return
        self._model = None
        self._offloaded = False
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _tokenize(self, captions: Sequence[str]) -> tuple[Tensor, Tensor]:
        text = [_PROMPT_PREFIX + str(caption) for caption in captions]
        suffix = [_PROMPT_SUFFIX] * len(text)
        if not self.cache_enabled:
            # Online mode (generate) does not truncate: a prompt longer than
            # training's 512-token convention goes into the model in full --
            # the quality consequences are the user's call (no blocking, no
            # silently dropping the tail). The varlen chain
            # (gather_valid_text / DiT text_len) has no structural length
            # constraint; just pad to the longest in the batch.
            encoded = self.tokenizer(
                text,
                truncation=False,
                return_length=False,
                return_overflowing_tokens=False,
                padding="longest",
                return_tensors="pt",
            )
        else:
            # Training's cached mode keeps the official fixed 512-token
            # training convention (cache fingerprint semantics unchanged)
            encoded = self.tokenizer(
                text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length + _PREFIX_TOKENS - _SUFFIX_TOKENS,
                return_tensors="pt",
            )
        suffix_encoded = self.tokenizer(suffix, return_tensors="pt")
        input_ids = torch.cat(
            [encoded["input_ids"], suffix_encoded["input_ids"]], dim=1,
        ).to(self.device, non_blocking=True)
        mask = torch.cat(
            [encoded["attention_mask"].bool(), suffix_encoded["attention_mask"].bool()],
            dim=1,
        ).to(self.device, non_blocking=True)
        return input_ids, mask

    def _encode_many(self, captions: Sequence[str]) -> list[Tensor]:
        if not captions:
            return []
        model = self.ensure_model()
        input_ids, mask = self._tokenize(captions)
        backbone = getattr(model, "model", model)
        with torch.inference_mode():
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None or max(self.selected_layers) >= len(hidden_states):
            count = 0 if hidden_states is None else len(hidden_states)
            raise RuntimeError(
                f"Qwen3-VL returned too few hidden_states layers: {count}, "
                f"need index {max(self.selected_layers)}"
            )
        stacked = torch.stack(
            [hidden_states[index] for index in self.selected_layers], dim=2,
        )[:, _PREFIX_TOKENS:]
        cropped_mask = mask[:, _PREFIX_TOKENS:]
        contexts = gather_valid_text(stacked, cropped_mask)
        for context in contexts:
            self._validate_context(context, source="Qwen3-VL output")
        return contexts

    def _encode_in_chunks(
        self, captions: Sequence[str], *, log_progress: bool = False,
    ) -> dict[str, Tensor]:
        # log_progress is only turned on during precaching (matching the
        # VAE cache's "encoding progress" convention); miss repair during
        # training also goes through this function, and we don't want it
        # spamming the log there for one or two entries.
        encoded: dict[str, Tensor] = {}
        unique = list(dict.fromkeys(str(caption) for caption in captions))
        next_mark = 10
        for start in range(0, len(unique), self.cache_batch_size):
            chunk = unique[start:start + self.cache_batch_size]
            contexts = self._encode_many(chunk)
            for caption, context in zip(chunk, contexts):
                encoded[caption] = context.detach().cpu().contiguous()
            if log_progress:
                done = len(encoded)
                if done >= next_mark or done == len(unique):
                    logger.info("  Text encoding progress: %d/%d", done, len(unique))
                    while next_mark <= done:
                        next_mark += 10
        return encoded

    def _validate_context(self, context: object, *, source: str) -> Tensor:
        if not isinstance(context, Tensor):
            raise ValueError(f"{source} is missing a tensor context")
        expected = (len(self.selected_layers), self.hidden_width)
        if (
            context.ndim != 3
            or context.shape[0] == 0
            or tuple(context.shape[1:]) != expected
            or not context.is_floating_point()
        ):
            raise ValueError(
                f"{source} context must be a non-empty float (seq, {expected[0]}, {expected[1]}), "
                f"got {tuple(context.shape)} / {context.dtype}"
            )
        return context

    def _context_from_payload(self, payload: Mapping[str, object] | None) -> Tensor | None:
        if payload is None:
            return None
        context = payload.get(_CACHE_TENSOR_KEY)
        try:
            return self._validate_context(context, source="Krea2 text cache")
        except ValueError:
            return None

    @staticmethod
    def _payload(context: Tensor) -> dict[str, Tensor]:
        return {_CACHE_TENSOR_KEY: context}

    def _prepare_caption_sidecars(self) -> None:
        contexts: dict[str, Tensor] = {}
        missing_entries: dict[str, list[TextCacheEntry]] = {}
        for caption, entries in self._caption_entries.items():
            context = None
            for entry in entries:
                cached = self._context_from_payload(self.store.read_caption(entry))
                if cached is None:
                    missing_entries.setdefault(caption, []).append(entry)
                elif context is None:
                    context = cached
            if context is not None:
                contexts[caption] = context

        to_encode = [
            caption for caption in self._caption_entries if caption not in contexts
        ]
        if self._caption_entries:
            if to_encode:
                logger.info(
                    "[text-cache] caption sidecar hit %d/%d, need to encode %d...",
                    len(contexts), len(self._caption_entries), len(to_encode),
                )
            else:
                logger.info(
                    "[text-cache] caption sidecar fully hit (%d entries), skipping encoding",
                    len(self._caption_entries),
                )
        contexts.update(self._encode_in_chunks(to_encode, log_progress=True))
        for caption, entries in missing_entries.items():
            for entry in entries:
                self.store.write_caption(entry, self._payload(contexts[caption]))

    def _prepare_prompt_bundle(self) -> None:
        if not self._prompt_captions:
            return
        if self._cache_root is None:
            raise ValueError("Krea2 prompt cache requires cache_root")
        payloads: dict[str, dict[str, Tensor]] = {}
        missing = []
        for caption in self._prompt_captions:
            context = self._context_from_payload(
                self.store.read_prompt(self._cache_root, caption),
            )
            if context is None:
                missing.append(caption)
            else:
                payloads[caption] = self._payload(context)
        for caption, context in self._encode_in_chunks(missing).items():
            payloads[caption] = self._payload(context)
        if missing:
            self.store.write_prompt_bundle(self._cache_root, payloads)

    def prepare_text_cache(
        self,
        captions: Iterable[str],
        extra_prompts: Iterable[str],
        *,
        cache_entries: Iterable[TextCacheEntry] = (),
        cache_root: str | Path | None = None,
    ) -> None:
        """Populate all known text caches, or retain the TE for online mode."""

        entries = list(cache_entries)
        self._caption_entries = {}
        for entry in entries:
            self._caption_entries.setdefault(entry.caption, []).append(entry)
        caption_list = list(dict.fromkeys(str(caption) for caption in captions))
        prompt_list = [str(prompt) for prompt in extra_prompts]
        prompt_list.extend(
            caption for caption in caption_list if caption not in self._caption_entries
        )
        self._prompt_captions = list(dict.fromkeys(prompt_list))
        self._cache_root = Path(cache_root) if cache_root is not None else None

        if not self.cache_enabled:
            self.ensure_model()
            return
        try:
            self._prepare_caption_sidecars()
            self._prepare_prompt_bundle()
        finally:
            self.release_model()

    def _repair_prompt_bundle(self, required: Sequence[str]) -> dict[str, Tensor]:
        if self._cache_root is None:
            raise ValueError("Krea2 cached mode requires cache_root to encode an unknown prompt")
        self._prompt_captions = list(dict.fromkeys([*self._prompt_captions, *required]))
        contexts: dict[str, Tensor] = {}
        missing = []
        for caption in self._prompt_captions:
            context = self._context_from_payload(
                self.store.read_prompt(self._cache_root, caption),
            )
            if context is None:
                missing.append(caption)
            else:
                contexts[caption] = context
        contexts.update(self._encode_in_chunks(missing))
        if missing:
            self.store.write_prompt_bundle(
                self._cache_root,
                {caption: self._payload(context) for caption, context in contexts.items()},
            )
        return contexts

    def encode_text_for_batch(
        self,
        captions: Sequence[str],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Krea2TextCondition:
        """Return a padded condition; cached misses are repaired transparently."""

        caption_list = [str(caption) for caption in captions]
        if not caption_list:
            raise ValueError("Krea2 text batch cannot be empty")
        if not self.cache_enabled:
            unique = list(dict.fromkeys(caption_list))
            contexts: dict[str, Tensor] = {}
            missing: list[str] = []
            for caption in unique:
                cached = self._online_lru_get(caption)
                if cached is None:
                    missing.append(caption)
                else:
                    contexts[caption] = cached
            for caption, context in zip(missing, self._encode_many(missing)):
                stored = context.detach().to("cpu")
                contexts[caption] = stored
                self._online_lru_put(caption, stored)
            ordered = [contexts[caption] for caption in caption_list]
            return pad_text_conditions(ordered, device=device, dtype=dtype)

        unique = list(dict.fromkeys(caption_list))
        contexts: dict[str, Tensor] = {}
        missing_sidecars: list[str] = []
        missing_prompts: list[str] = []
        for caption in unique:
            entries = self._caption_entries.get(caption)
            if entries:
                context = None
                for entry in entries:
                    context = self._context_from_payload(self.store.read_caption(entry))
                    if context is not None:
                        break
                if context is None:
                    missing_sidecars.append(caption)
                else:
                    contexts[caption] = context
            else:
                missing_prompts.append(caption)

        try:
            repaired = self._encode_in_chunks(missing_sidecars)
            for caption, context in repaired.items():
                contexts[caption] = context
                for entry in self._caption_entries[caption]:
                    self.store.write_caption(entry, self._payload(context))
            if missing_prompts:
                contexts.update(self._repair_prompt_bundle(missing_prompts))
        finally:
            self.release_model()

        ordered = [contexts[caption] for caption in caption_list]
        return pad_text_conditions(ordered, device=device, dtype=dtype)


def load_krea2_text_stack(
    model_path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    compute_dtype: torch.dtype | None = None,
    cache_enabled: bool = True,
) -> Krea2TextStack:
    """Create a lazy Krea2 text stack; only the small tokenizer loads immediately."""

    return Krea2TextStack(
        model_path,
        device=device,
        dtype=dtype,
        compute_dtype=compute_dtype,
        cache_enabled=cache_enabled,
    )
