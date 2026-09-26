# NaViT / Patch-n-Pack Block-Diagonal Packed Training (this fork's port)

> Status: **ported** — model kernel + data layer + training-loop wiring + schema. opt-in / default-off,
> when disabled it is byte-for-byte equivalent to before the change (146 local tests all green, including default-path regressions).
> **The block-diagonal packed forward pass has a hard dependency on the xformers varlen kernel and can only be verified on Colab GPU** —
> locally, with no GPU/xformers, the packed forward pass cannot run; the first real training run must be observed on Colab (see §4).
> Ported from upstream v0.18.0, with the leap/sra/tlora exclusivity items this fork doesn't have stripped out.

## 1. What problem this solves

When you have a small dataset + multiple resolutions + want a high batch size, ARB bucketing by exact `(h, w)` means each bucket only has a
few images → can't fill a large batch. **NaViT/Patch-n-Pack (arXiv 2307.06304)** packs multiple heterogeneous images into one sequence,
using block-diagonal attention so each image only attends to its own tokens (self) and its own caption (cross), with each image carrying
its own timestep. This decouples "how many images are processed per step" from "the shape of a single image" — zero padding, and it runs
on the fast xformers varlen kernel.

When combined with `navit_native_resolution`, a single image is floored to a 16px grid at its **native size**, bypassing bucket quantization —
this is exactly where "preserving original fabric/texture detail instead of having it smoothed away by downscaling" comes from.

## 2. How to enable it (config / yaml)

```yaml
cache_latents: true            # required (packing budgets by latent token count, needs pre-encoded cache)
navit_packing: true            # master switch (default false)
navit_token_budget: 16384      # total token budget per pack (sized by VRAM; see table below)
navit_max_images_per_pack: 0   # max images per pack, 0 = unlimited
# Packing strategy
navit_pack_strategy: next_fit  # next_fit (default, sequential greedy) / ffd (windowed FFD, packs tighter)
navit_pack_ffd_window: 256     # FFD window size (0 = global FFD; >0 = windowed FFD + cross-epoch reshuffle)
navit_drop_last: false         # drop the last under-budget pack of each epoch
navit_text_trim_padding: false # pack cross-attn text by each image's actual T5 length, dropping the 512-token padding
# Native resolution (optional, key to preserving texture)
navit_native_resolution: true       # requires navit_packing + cache_latents; default false
navit_native_over_budget: downscale # oversized images: downscale (default, never OOMs) / fail
```

When `navit_packing=true`, `attention_backend` is automatically forced to `xformers` (schema-level coercion).

### VRAM ↔ token_budget reference (conservative starting points with grad_checkpoint=true)

| VRAM | starting token_budget | ≈ (4096 tokens/image) |
|---|---|---|
| 16 GB | 16384 | ~4 images |
| 24 GB | 32768 | ~8 images |
| 48 GB | 65536 | ~16 images |
| 80 GB | 98304 | ~24 images |

**Always observe peak VRAM on the first run before tuning further** — GPU, rank, and base model size all affect it.

## 3. v1 supported scope and gating (schema fail-fast)

**Mutually exclusive (enabling both → error at startup)**: `infonoise_enabled` (I-MMSE recording needs the standard per-sample MSE path;
navit's per-image loss semantics differ). *Upstream also excludes leap/sra/tlora, but this fork doesn't have those features.*

**Prerequisites**: `cache_latents=true`, `navit_token_budget>0`; `navit_native_resolution` requires `navit_packing`.

**Supported**: basic flow-matching (per-image t + noise + per-image loss), `grad_accum`, per-block gradient checkpointing
(`grad_checkpoint`), LoRA/LoKr save/resume, `loss_weighting` (weights computed from per-image t), reg-set down-weighting
(`loss_weight`, applied per image).

## 4. Colab verification (packed forward pass cannot run locally)

Already verified locally: schema validation, native fixed-size sizing, packed sampler/collate, token counting (`tests/test_navit.py`
+ `tests/test_multires_and_local_base.py`). **The block-diagonal forward pass must be verified on a GPU**:

1. Install xformers on Colab (matching the torch version): `pip install xformers`.
2. Take a small dataset, in the config set `cache_latents=true` + `navit_packing=true` +
   `navit_token_budget=16384`, run for a few dozen steps. Observe:
   - No `forward_packed_navit requires xformers` error at startup (= xformers is working).
   - Loss is finite and decreasing; peak VRAM is within expectations.
   - Sample images look normal (compare against training the same dataset without navit — style should match).
3. Then enable `navit_native_resolution=true` to verify native sizing (texture preserved on large images, no OOM).
4. Do a full run together with LoRA/LoKr injection (the key check on the first run).

If there's any doubt about the numerical correctness of the packed forward pass, refer to upstream's `tests/test_navit_packed_objective.py`
(GPU + xformers), which asserts `forward_packed_navit ≡ each image's individual forward pass concatenated` — it can be ported to this fork
and run on Colab.

## 4.5 Combined with block swap (a previously silent wrong-weights bug)

**Symptom**: with `navit_packing=true` + `blocks_to_swap>0`, training appears to run normally (no errors, no NaN,
loss even drops faster than usual), but sample images turn into a mosaic/flat color blocks after 2-5 epochs.

**Root cause**: block swap uses four nn.Module hooks to take over weight fetch/release (`training/block_swap.py::attach`), and those
hooks **only fire on `__call__`**. The packed loop used to call `blk.forward_tokens(...)` directly — none of the hooks fired, so a
swapped-out block kept pointing at whatever weights were left in its two GPU slots from the last standard forward pass (the step-0
baseline sample). Everything stayed on the GPU, so there was no device error: 26 layers silently became duplicates of two layers, and
the same applied to backward (the `attach()` docs already note that missing the backward hook silently computes wrong gradients).
The LoRA ended up fitting a base model that didn't exist, and the generated images naturally fell apart.

**Fix**: the packed loop now calls `blk(..., packed_tokens=True)`, so `Block.forward` dispatches to `forward_tokens`, using the same
hook semantics as the standard checkpoint loop in `training/families/anima/forward.py`. `final_layer` isn't under swap's jurisdiction
(it only manages `blocks.{i}.`), so it's still called directly.

**Regression coverage**: `tests/test_block_swap_anima.py::test_navit_packed_forward_backward_matches_without_swap`
(real model, swap on/off compared value-by-value, requires CUDA + xformers) +
`tests/test_block_packed_dispatch.py` (dispatch and hook-firing + source invariants, CPU-only).

## 5. Implementation map (this fork)

| Layer | File |
|---|---|
| block-diagonal attention op + per-image AdaLN token forward + packed forward | `models/cosmos_predict2_modeling.py` (`torch_attention_op` attn_mask branch, `Block/FinalLayer.forward_tokens`, `patchify_latents_to_tokens`, `_packed_rope_from_grid`, `forward_packed_navit`) |
| core training step | `runtime/training/navit.py` (`navit_packed_forward_and_loss` / `pack_cross_embeddings`) |
| token-budget packing + native fixed-sizing | `runtime/training/dataset.py` (`NavitPackBatchSampler` / `collate_fn_navit_pack` / `plan_native_fit_image` / `dataset_token_counts`) |
| training loop wiring | `runtime/training/loop.py` (navit branch: pack cross → per-image weighting → packed forward) |
| data loading wiring | `runtime/training/phases/dataset.py` (navit_packing → NavitPackBatchSampler + collate; native → ImageDataset native params) |
| config keys + exclusivity validation | `studio/domain/training.py` (`navit_*` / `cache_encode_*` fields + `_validate_navit_exclusive` + `_coerce_navit_attention_backend`) |
