# ADR 0015: MOSS-Image family and generic weight library

Date: 2026-10-23

## Decision

Add `moss_image` through the existing model/spec/assets registries. Vendor the architecture, FLUX.2 VAE converter and ConvRot math from sd-scripts commit `11d8cedd0c67b93a10f8b2d75f43554f6f12c794`, preserving checkpoint paths. Adapt attention, checkpointing, weight loading and runtime integration locally. Do not replace the shared training loop or existing transformer swap engine.

Keep Studio's 5D latent contract using normalized, unpacked 32-channel latents. Pack/unpack 2x2 at the MOSS boundary; normalization belongs to the VAE's packed 128-channel BN. A distinct fingerprint prevents reuse of Anima/Krea2 latent caches. Text caches include weight identity, tokenizer identity, quantization and token-length policy. MOSS task prompts use individual cache files under `.text-cache/moss-prompts/`, preventing DOP prompt sets from materializing a large aggregate in RAM. Other families retain their existing cache layout.

Separate transformer/encoder weight quantization from training precision. Stream safetensors one tensor at a time, freeze bases, validate every key/shape and INT8 metadata, and use LyCORIS bypass with quantized weights. Reuse H2D-only transformer swapping. Frozen Qwen encoder layers use a separate inference-only offloader because their shapes differ.

Replace family-specific Settings cards with three lists using the existing card, row, status, picker and download components. `weight_folders` stores additional search directories; existing `selected` and `selected_te` mappings retain their semantics. Add `selected_vae_by_family` for incompatible VAE formats while preserving the legacy shared `selected_vae` for Anima/Krea2. Existing catalog sections remain available to clients and base-model dropdowns. Discovery reads bounded safetensors headers only and leaves files untouched.

## Alternatives

Running a second trainer subprocess/backend would duplicate optimizer, resume and logging integration. Replacing the application with KohakuMossTrainer would also bring unrelated precision and packing changes. Both are deferred in favor of the existing family contract.

## Consequences

MOSS needs transformers 5.3 or later for Qwen3.5. Unsupported model-specific controls are gated; shared options remain. The existing checkpoint-file-size preflight is hidden for MOSS until it can model shard and quantization memory correctly. Hardware validation remains required before merging to main; only CPU checks and actual weight-header compatibility were performed because the GPU was occupied.

Rollback is a branch switch to main or the published pre-MOSS tag. No schema migration or user dataset/weight edits are required. See [the user guide](../user-guide/moss-image.md).

## Verification

The focused CPU suite passed 176 checks with `CUDA_VISIBLE_DEVICES=-1` and `OMP_NUM_THREADS=2`. Frontend verification passed 42 checks covering the weight library, its Settings integration, schema form and generation snapshots; lint, TypeScript and production build passed.

The seven unrelated Settings tests targeting previously removed credentials/tagging/LLM tabs have baseline failures and were excluded from this focused run. The original Settings component and tests from the parent commit were checked separately to establish that baseline. This is not a claim that the full repository suite passes. GPU training, CUDA quantization kernels and GPU offloading remain pending.
