"""AnimaFamily -- the ModelFamily implementation for the Anima family (multi-model PR-2b).

Behavior implementations live in sibling modules in this directory:
loader.py (DiT/TE loading), forward.py (checkpointed forward), preset.py
(LoRA target); sampling and text encoding currently still live in
training.families.anima.sampling / training.families.anima.text_encoding
(moved into this directory in S3, methods are funneled through here).
"""

from __future__ import annotations

from typing import Any, Iterable

from training.families.anima import ANIMA_SPEC
from training.families.anima.preset import ANIMA_PRESET


class AnimaFamily:
    spec = ANIMA_SPEC

    # -- Loading --------------------------------------------------------
    def load_dit(self, path, device, dtype, *,
                 attention_backend: str = "flash_attn", repo_root=None,
                 purpose: str = "train", blocks_to_swap: int = 0):
        # purpose is a quantized-inference knob (krea2 fp8); Anima has no
        # quantized form, so it's accepted and ignored.
        del purpose
        from training.families.anima.loader import load_anima_model
        from training.model_loading import enable_xformers

        model = load_anima_model(
            path, device, dtype, repo_root,
            flash_attn=(attention_backend == "flash_attn"),
            blocks_to_swap=blocks_to_swap,
        )
        if attention_backend == "xformers":
            enable_xformers(model)
        return model

    def swappable_blocks(self, *, checkpoint_path: str | None = None) -> int:
        """Upper bound on the number of swappable layers (= DiT backbone layer count).
        The block-swap preflight recommendation needs this upper bound.

        Anima's layer count is determined by the checkpoint (2B=28 layers /
        14B=36 layers), so ``checkpoint_path`` must be given; if it can't be
        read this returns 0, and the preflight skips the whole check based on
        that (degrading to the old behavior).
        """
        if not checkpoint_path:
            return 0
        try:
            from training.families.anima.loader import block_count_from_header

            return block_count_from_header(checkpoint_path)
        except Exception:  # noqa: BLE001
            return 0

    def swapped_param_ratio(self, blocks_to_swap: int, *,
                            checkpoint_path: str | None = None) -> float:
        """Fraction of total model parameters held by the swapped-out layers
        (used for the VRAM budget discount, same semantics as krea2).

        Anima's layer count is determined by the checkpoint (2B=28 layers /
        14B=36 layers), so it can't count meta parameters from a fixed config
        the way krea2 does -- instead it counts numel from the safetensors
        header, which is naturally version-independent. If no path is given
        or reading fails, this returns 0: the guard degrades to conservative
        (budgets VRAM as if for the full model, never wrongly lets it through).
        """
        if blocks_to_swap <= 0 or not checkpoint_path:
            return 0.0
        try:
            from training.families.anima.loader import swapped_param_ratio_from_header

            return swapped_param_ratio_from_header(checkpoint_path, blocks_to_swap)
        except Exception:  # noqa: BLE001
            return 0.0

    def load_vae(self, path, device, dtype, *, tiling: str = "auto"):
        # Shared implementation across families (D6); kept as a method on
        # family so a 3rd family with a different VAE needs zero migration.
        from training.vae import load_vae

        return load_vae(path, device, dtype, None, tiling=tiling)

    def load_text(self, text_encoder_path, device, dtype, *,
                  t5_tokenizer_path: str = "", comfy_qwen: bool = False,
                  t5_fast: bool = False, purpose: str = "train",
                  cache_enabled: bool = True):
        from training.families.anima.loader import load_text_encoders

        # purpose is accepted and ignored (same as load_dit): Anima's TE
        # backend is decided only by the caller's explicit comfy_qwen flag --
        # the generate call site passing purpose must not override the
        # user-selected text_encoder_backend (default hf).
        return load_text_encoders(
            text_encoder_path, t5_tokenizer_path, device, dtype,
            comfy_qwen=comfy_qwen,
            t5_fast=t5_fast,
        )

    # -- Text conditioning ------------------------------------------------
    def prepare_text_cache(self, captions: Iterable[str],
                           extra_prompts: Iterable[str], *, cache_entries=(),
                           cache_root=None, text=None, device=None,
                           dtype=None) -> None:
        return None  # Anima encodes online every step (spec.text.strategy="online"), no cache

    def encode_text_for_batch(self, text, dit, captions, device, dtype, *,
                              comfy_encoding: bool = True, kv_trim: bool = False,
                              return_t5_attn: bool = False):
        """Online encoding per step (moved down from the loop.py text block, behavior unchanged).

        comfy_encoding=True (default): raw caption goes into Qwen; T5 tokenizes
        the whole string literally, same pipeline as test-generation / training
        preview conditioning. False = the legacy A/B path.
        Returns cross -- opaque to the loop (03 S2.7-4).

        When return_t5_attn=True, returns (cross, t5_attn): NaViT packing needs
        a per-image attention mask to truncate cross-attn padding. navit is an
        Anima capability bit (D5), so this kwarg is family-private and not part
        of the protocol.
        """
        import torch
        import torch.nn.functional as F

        from training.families.anima.text_encoding import (
            _build_qwen_text_from_prompt,
            encode_qwen,
            tokenize_t5_comfy_literal,
            tokenize_t5_weighted,
        )

        qwen_model, qwen_tok, t5_tok = text
        max_len = self.spec.text.max_seq_len
        with torch.no_grad():
            if comfy_encoding:
                qwen_texts = [str(c) for c in captions]
                qwen_emb, _ = encode_qwen(qwen_model, qwen_tok, qwen_texts, device)
                t5_ids, t5_attn, t5_w = tokenize_t5_comfy_literal(t5_tok, captions, max_length=max_len)
            else:
                qwen_texts = [_build_qwen_text_from_prompt(c) for c in captions]
                qwen_emb, _ = encode_qwen(qwen_model, qwen_tok, qwen_texts, device)
                t5_ids, t5_attn, t5_w = tokenize_t5_weighted(t5_tok, captions, max_length=max_len)
            t5_ids = t5_ids.to(device)
            t5_attn = t5_attn.to(device)
            t5_w = t5_w.to(device, dtype=torch.float32)
            # t5_w is multiplied into the LLMAdapter output inside preprocess_text_embeds (ComfyUI parity)
            cross = dit.preprocess_text_embeds(qwen_emb, t5_ids, t5xxl_weights=t5_w)
            if cross.shape[1] < max_len:
                cross = F.pad(cross, (0, 0, 0, max_len - cross.shape[1]))
            if kv_trim:
                # KV trim: truncate padding down to the nearest valid token bucket (64/128/256/512)
                _actual = int(t5_attn.sum(dim=-1).max().item())
                _bucket = max_len
                for _b in (64, 128, 256, max_len):
                    if _b >= _actual:
                        _bucket = _b
                        break
                cross = cross[:, :_bucket, :].contiguous()
        if return_t5_attn:
            return cross, t5_attn
        return cross

    # -- Training forward / sampling --------------------------------------
    def forward_train(self, dit, noisy, t, cond, *, use_checkpoint: bool = False):
        """v_pred = f(noisy, t, cond). pad_mask (concat_padding_mask private
        input) and reshaping t ((B,)->(B,1)) are family-internal concerns (03-3)."""
        import torch

        from training.families.anima.forward import forward_with_optional_checkpoint

        pad_mask = torch.zeros(
            noisy.shape[0], 1, noisy.shape[-2], noisy.shape[-1],
            device=noisy.device, dtype=noisy.dtype,
        )
        return forward_with_optional_checkpoint(
            dit, noisy, t.view(-1, 1), cond, pad_mask,
            use_checkpoint=use_checkpoint,
        )

    def sample_image(self, model, vae, text, prompt, **kwargs):
        from training.families.anima.sampling import sample_image

        # distilled is a knob for distilled-inference families (Krea2
        # Turbo); Anima has no distilled variant, so it's accepted and
        # ignored (callers pass it per the unified protocol, no need to
        # branch by family). Same for vram_policy -- Anima's resident TE
        # orchestration doesn't participate in the VRAM policy yet.
        kwargs.pop("distilled", None)
        kwargs.pop("vram_policy", None)
        return sample_image(model, vae, *text, prompt, **kwargs)

    # -- LoRA output --------------------------------------------------------
    def lora_preset(self) -> dict[str, Any]:
        return ANIMA_PRESET

    def lora_metadata(self) -> dict[str, str]:
        return {
            "model_family": self.spec.family_id,
            "preset": self.spec.lora.preset_name,
        }

    def convert_lora_state_dict(self, sd: dict) -> dict:
        return sd  # kohya format is already the output format (04 S7.1), identity
