"""MOSS behavior adapter. The shared training loop remains rectified flow."""
from . import MOSS_SPEC
from .preset import MOSS_PRESET


class MossFamily:
    spec = MOSS_SPEC

    def load_options(self, args):
        get = args.get if isinstance(args, dict) else lambda k, d: getattr(args, k, d)
        return {
            "dit": {"quantization": get("transformer_quantization", "none"),
                    "sample_flow_shift": get("moss_sample_shift", 5.0),
                    "sample_uncond_skip_layers": get("moss_uncond_skip_layers", [22])},
            "text": {"quantization": get("text_encoder_quantization", "none"),
                     "offload_layers": get("text_encoder_layers_to_offload", 0),
                     "max_length": get("moss_max_token_length", 4096),
                     "tokenizer_path": get("moss_tokenizer_path", "")},
        }

    def load_dit(self, path, device, dtype, *, attention_backend="none", repo_root=None,
                 purpose="train", blocks_to_swap=0, quantization="none",
                 sample_flow_shift=5.0, sample_uncond_skip_layers=(22,)):
        from .loader import load_dit
        model = load_dit(path, device, dtype, quantization=quantization,
                         blocks_to_swap=blocks_to_swap, attention_backend=attention_backend)
        model.sample_flow_shift = sample_flow_shift
        model.sample_uncond_skip_layers = tuple(sample_uncond_skip_layers)
        return model

    def load_vae(self, path, device, dtype, *, tiling="auto"):
        from .vae import load_vae
        return load_vae(path, device, tiling=tiling)

    def load_text(self, text_encoder_path, device, dtype, *, t5_tokenizer_path="",
                  comfy_qwen=False, t5_fast=False, purpose="train", cache_enabled=True,
                  quantization="none", offload_layers=0, max_length=4096, tokenizer_path=""):
        from .text_encoding import MossTextStack
        return MossTextStack(text_encoder_path, device, quantization=quantization,
            offload_layers=offload_layers, max_length=max_length, tokenizer_path=tokenizer_path,
            cache_enabled=cache_enabled)

    def prepare_text_cache(self, captions, extra_prompts, *, cache_entries=(), cache_root=None,
                           text=None, device=None, dtype=None):
        text.prepare_text_cache(captions, extra_prompts, cache_entries=cache_entries, cache_root=cache_root)

    def encode_text_for_batch(self, text, dit, captions, device, dtype, *, comfy_encoding=True, kv_trim=True):
        return text.encode_text_for_batch(captions, device=device, dtype=dtype)

    def forward_train(self, dit, noisy, t, cond, *, use_checkpoint=False):
        from modeling.moss_image.vae import patchify_latents, unpatchify_latents
        if use_checkpoint:
            # Set once (logging on every step would obscure the training progress).
            if not dit.layers[0].gradient_checkpointing:
                dit.enable_gradient_checkpointing()
        else:
            if dit.layers[0].gradient_checkpointing:
                dit.disable_gradient_checkpointing()
        packed = patchify_latents(noisy[:, :, 0])
        pred = dit(packed, t.float() * 1000, cond.context, cond.attention_mask)
        return unpatchify_latents(pred).unsqueeze(2)

    def sample_image(self, *args, **kwargs):
        from .sampling import sample_image
        return sample_image(*args, **kwargs)

    def lora_preset(self):
        return dict(MOSS_PRESET)

    def lora_metadata(self):
        return {"model_family": "moss_image", "ss_base_model_version": "moss_image",
                "modelspec.architecture": "moss_image/lora"}

    def convert_lora_state_dict(self, sd):
        return sd
