"""Anima family loader (multi-model PR-2b, moved function-by-function from training/models.py).

load_anima_model / load_text_encoders are family knowledge (two-tier
checkpoint shape inference, llm_adapter missing-weight fallback, Qwen+T5 dual
encoder); VAEWrapper / load_vae stay in training.vae as a cross-family shared
asset (D6).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from training.model_loading import (
    _load_safetensors_state_dict,
    _load_weights_best_effort,
)

logger = logging.getLogger(__name__)

#: Block ownership within checkpoint keys (keys may carry a model./module.
#: prefix etc., which _load_weights_best_effort only strips at load time --
#: this matches by substring, so the prefix doesn't matter)
_BLOCK_KEY_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


@lru_cache(maxsize=8)
def _header_param_counts(checkpoint_path: str) -> tuple[tuple[int, ...], int]:
    """(per-block parameter count, full-model parameter count), counted from the safetensors header (payload not read).

    Cached: the block-swap preflight needs to ask for the ratio of each
    candidate value from 0..N, and without caching that would re-read the
    same header N+1 times (same treatment as krea2's
    ``_swapped_param_counts``). The key is the path string; the two returned
    values are immutable, so the cache itself has zero memory risk.
    """
    from safetensors import safe_open

    per_block: dict[int, int] = {}
    total = 0
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            numel = 1
            for dim in f.get_slice(key).get_shape():
                numel *= dim
            total += numel
            m = _BLOCK_KEY_RE.search(key)
            if m:
                idx = int(m.group(1))
                per_block[idx] = per_block.get(idx, 0) + numel
    if not per_block:
        return (), total
    num_blocks = max(per_block) + 1
    return tuple(per_block.get(i, 0) for i in range(num_blocks)), total


def block_count_from_header(checkpoint_path) -> int:
    """Number of DiT backbone layers in the checkpoint (2B=28 / 14B=36). Returns 0 if it can't be read.

    Anima's layer count is determined by the checkpoint, unlike krea2 which
    has a fixed config -- to search for a recommended value the preflight
    needs to know the upper bound first, and that can only be asked of the
    weight file itself.
    """
    try:
        per_block, _total = _header_param_counts(str(checkpoint_path))
    except Exception:  # noqa: BLE001
        return 0
    return len(per_block)


def swapped_param_ratio_from_header(checkpoint_path, blocks_to_swap: int) -> float:
    """Fraction of total model parameters held by the swapped-out layers, counted as numel from the safetensors header (payload not read).

    krea2 counts meta-model parameters from a fixed config; Anima's layer
    count is determined by the checkpoint (2B=28 layers / 14B=36 layers), so
    the header is the source of truth for the version, and counting
    parameters this way is naturally dtype-independent (the VRAM discount
    must scale by the file's actual size ratio -- see the note on the
    same-named function in the krea2 loader).
    """
    if blocks_to_swap <= 0:
        return 0.0
    per_block, total = _header_param_counts(str(checkpoint_path))
    if not per_block or total <= 0:
        return 0.0
    num_blocks = len(per_block)
    first = max(num_blocks - blocks_to_swap, 0)
    swapped = sum(per_block[i] for i in range(first, num_blocks))
    return swapped / total


def place_model_for_block_swap(model, device, dtype, blocks_to_swap: int) -> int:
    """Placement for a model with swapped-out layers that don't go on the GPU: cast to dtype on CPU, only move the non-swapped part to GPU.

    S9.4 discipline (docs/design/block-swap.md): **must not move the whole
    model to the GPU and then back down** -- that would still hit the same
    peak GPU usage as the full model, defeating the small-card target. The
    swapped-out layers stay on CPU (dtype already cast); pinning is taken
    over in place by ``PinnedBlockSwap._build`` (tensors already on CPU are
    only pinned, not copied again).

    Returns the actual number of swapped layers (clamped to the total layer
    count -- blocks_to_swap is a global setting, and a user might feed a
    value tuned for the 36-layer version into the 28-layer version; anything
    over the limit is treated as swapping out everything).
    """
    import torch

    from training.sysmem import check_pinned_budget

    total = len(model.blocks)
    num_swap = min(int(blocks_to_swap), total)
    first = total - num_swap
    swapped_prefixes = tuple(f"blocks.{i}." for i in range(first, total))

    # pinned-memory budget guard runs first (B6: fail-fast, no GPU / pinned allocation has happened yet)
    elem = torch.empty(0, dtype=dtype).element_size()
    need = sum(
        p.numel() for n, p in model.named_parameters()
        if n.startswith(swapped_prefixes)
    ) * elem
    check_pinned_budget(need, blocks=num_swap)

    model.to(dtype=dtype)  # cast on CPU (built as fp32 -> target dtype)
    target = torch.device(device)
    for name, param in model.named_parameters():
        if not name.startswith(swapped_prefixes):
            param.data = param.data.to(target)
    for name, buf in model.named_buffers():
        if not name.startswith(swapped_prefixes):
            buf.data = buf.data.to(target)
    # public marker: whole-model offload during VAE decode at sampling time
    # must skip this model (a blanket .to() would move the CPU-resident
    # copy back onto the GPU on restore, wasting the swap and hitting peak
    # usage = full model); families/anima/sampling.py branches on this
    model.blocks_to_swap = num_swap
    logger.info(
        "Block swap placement: the last %d/%d layers stay in host memory (%.2f GB), the rest go to the GPU",
        num_swap, total, need / 1024 ** 3,
    )
    return num_swap


def load_anima_model(transformer_path, device, dtype, repo_root, *,
                     flash_attn: bool = True, blocks_to_swap: int = 0):
    """Load the Anima transformer model.

    `flash_attn=False` explicitly disables the flash_attn fast path (passed
    in by the caller when attention_backend=xformers/none), letting the
    caller fully decide the attention implementation -- the PR #17 version's
    default fn(True) forced flash_attn on with no way for the user to turn
    it off, which didn't fully decouple from cfg.attention_backend.
    """
    from safetensors import safe_open

    # repo_root is kept but no longer used (the sister-contract signature is
    # "may add, may not remove or change"): model code ships with the repo
    # and goes through a normal import -- single module identity, exec-load
    # has been retired (multi-model PR-2a), and the attention-backend switch
    # no longer needs cross-module alias broadcasting.
    from modeling.anima import anima_modeling, cosmos_predict2_modeling

    Anima = anima_modeling.Anima

    # global attention-backend switch: set_attention_backend() clears the unselected fast path in one shot
    flash_enabled = False
    for module in (cosmos_predict2_modeling, anima_modeling):
        set_backend = getattr(module, "set_attention_backend", None)
        if set_backend is not None:
            try:
                effective = str(set_backend("flash_attn" if flash_attn else "none"))
                flash_enabled = (effective == "flash_attn") or flash_enabled
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to set attention backend, falling back to SDPA: %s", exc)
                continue
        fn = getattr(module, "set_flash_attn_enabled", None)
        if fn is None:
            continue
        try:
            flash_enabled = bool(fn(flash_attn)) or flash_enabled
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to enable flash_attn, falling back to SDPA: %s", exc)
    if flash_enabled:
        logger.info("flash_attn enabled (training + sampling use the fast path)")
    else:
        logger.info("flash_attn disabled (attention_backend=%s or package not installed)",
                    "flash_attn" if flash_attn else "non-flash")

    # infer config from the checkpoint
    with safe_open(transformer_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith("x_embedder.proj.1.weight"):
                w = f.get_tensor(k)
                break

    in_channels = (w.shape[1] // 4) - 1  # concat_padding_mask=True
    model_channels = w.shape[0]

    if model_channels == 2048:
        num_blocks, num_heads = 28, 16
    elif model_channels == 5120:
        num_blocks, num_heads = 36, 40
    else:
        raise RuntimeError(f"Unknown model_channels={model_channels}")
    # The layer count is taken from the checkpoint: a width of 2048 can be
    # either 28 layers (2B) or 40 layers (2.9B). Hard-coding 28 would make a
    # strict=False load silently drop the extra layers.
    header_blocks = block_count_from_header(transformer_path)
    if header_blocks and header_blocks != num_blocks:
        logger.info("Anima DiT layer count taken from checkpoint: %d (default %d)", header_blocks, num_blocks)
        num_blocks = header_blocks

    config = dict(
        max_img_h=1024, max_img_w=1024, max_frames=128,
        in_channels=in_channels, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=model_channels,
        num_blocks=num_blocks, num_heads=num_heads,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True, adaln_lora_dim=256,
        rope_h_extrapolation_ratio=4.0 if in_channels == 16 else 3.0,
        rope_w_extrapolation_ratio=4.0 if in_channels == 16 else 3.0,
        rope_t_extrapolation_ratio=1.0,
    )

    model = Anima(**config)

    # load the weights
    sd = _load_safetensors_state_dict(Path(transformer_path))
    # The RoPE tables (seq = arange(max length) etc.) are derived from
    # config; some exports saved them into the weights under a different max
    # length (2.9B: seq 256 vs. model 512), so let the model rebuild them itself.
    sd = {
        k: v for k, v in sd.items()
        if not ("pos_embedder." in k and k.rsplit(".", 1)[-1] in ("seq", "dim_spatial_range", "dim_temporal_range"))
    }
    info = _load_weights_best_effort(model, sd, label="Transformer")

    # If the checkpoint has no llm_adapter weights at all, a random init would scramble the cross-attn conditioning, so disabling it is safer
    has_llm_adapter = any("llm_adapter" in k for k in sd.keys())
    if not has_llm_adapter and hasattr(model, "llm_adapter"):
        try:
            model.llm_adapter = None
            logger.warning("Checkpoint has no llm_adapter weights: llm_adapter disabled (falling back to using Qwen embeddings directly)")
        except Exception:
            pass
    if blocks_to_swap > 0:
        place_model_for_block_swap(model, device, dtype, blocks_to_swap)
    else:
        model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)

    logger.info(f"Anima model loaded: {model_channels}ch, {num_blocks} blocks")
    return model


def load_text_encoders(
    qwen_path,
    t5_tokenizer_path,
    device,
    dtype,
    *,
    comfy_qwen: bool = False,
    t5_fast: bool = False,
):
    """Load the text encoders (Qwen + T5)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5TokenizerFast

    # Qwen
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    if comfy_qwen:
        from training.families.anima.comfy_qwen import load_comfy_qwen3_encoder

        qwen_model = load_comfy_qwen3_encoder(qwen_path, device=device, dtype=dtype)
    else:
        qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval().requires_grad_(False)

    # T5 tokenizer
    t5_cls = T5TokenizerFast if t5_fast else T5Tokenizer
    if t5_tokenizer_path and Path(t5_tokenizer_path).exists():
        t5_tokenizer = t5_cls.from_pretrained(t5_tokenizer_path)
    else:
        logger.warning(
            "T5 tokenizer local directory missing (t5_tokenizer_path=%s), "
            "downloading google/t5-v1_1-xxl from Hugging Face",
            t5_tokenizer_path or "not configured",
        )
        try:
            t5_tokenizer = t5_cls.from_pretrained("google/t5-v1_1-xxl")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download the T5 tokenizer (google/t5-v1_1-xxl): {type(e).__name__}: {e}\n"
                f"Please check your network connection and retry; or download the t5_tokenizer model "
                f"from the Studio settings page, and make sure t5_tokenizer_path "
                f"(current value: {t5_tokenizer_path or 'not configured'}) points to that directory."
            ) from e

    logger.info("Text encoders loaded")
    return qwen_model, qwen_tokenizer, t5_tokenizer
