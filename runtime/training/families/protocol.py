"""ModelFamily behavior contract (multi-model PR-2b; frozen surface =
docs/design/multi-model/03 Sec 4.1 + 04 Sec 3-C4).

Nine methods: load_dit / load_vae / load_text / prepare_text_cache /
encode_text_for_batch / forward_train / sample_image / lora_preset /
lora_metadata, plus convert_lora_state_dict (identity by default). Evolution
rule: new parameters are always keyword-only with a default; no **kwargs
black holes (03 Sec 4.2).

The shared loop's seven invariants (03 Sec 2.7) are guaranteed by
implementers: latent is always 5D with T==1, t in (0,1), rectified-flow
algebra stays in the loop, v_pred has the same shape, cond is opaque,
autocast stays in the loop, RNG discipline.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

from training.families.spec import ModelSpec


@runtime_checkable
class ModelFamily(Protocol):
    spec: ModelSpec

    # -- Loading (models_phase + all bypass callers, 04 D8') -----------------
    def load_dit(self, path: str, device, dtype, *,
                 attention_backend: str = "flash_attn", repo_root=None,
                 purpose: str = "train") -> Any: ...

    def load_vae(self, path: str, device, dtype, *, tiling: str = "auto") -> Any: ...

    def load_text(self, text_encoder_path: str, device, dtype, *,
                  t5_tokenizer_path: str = "", comfy_qwen: bool = False,
                  t5_fast: bool = False, purpose: str = "train",
                  cache_enabled: bool = True) -> Any: ...

    # -- Text conditioning -----------------------------------------------------
    def prepare_text_cache(self, captions: Iterable[str],
                           extra_prompts: Iterable[str], *, cache_entries=(),
                           cache_root=None, text=None, device=None,
                           dtype=None) -> None: ...

    def encode_text_for_batch(self, text, dit, captions: list[str],
                              device, dtype, *, comfy_encoding: bool = True,
                              kv_trim: bool = True) -> Any: ...

    # -- Training forward / sampling --------------------------------------------
    def forward_train(self, dit, noisy, t, cond, *, use_checkpoint: bool = False): ...

    def sample_image(self, *args, **kwargs): ...

    # -- LoRA artifacts ------------------------------------------------------------
    def lora_preset(self) -> dict[str, Any]: ...

    def lora_metadata(self) -> dict[str, str]: ...

    def convert_lora_state_dict(self, sd: dict) -> dict: ...
