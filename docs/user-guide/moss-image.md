# MOSS-Image Base

The training family is `moss_image`. Select **MOSS-Image Base** in the existing configuration form. Anima and Krea2 keep their own defaults and options.

## Files and settings

Settings now groups training weights into **Checkpoints**, **VAE**, and **Text Encoders**. Add a folder or browse to it, then select a discovered file. Scanning reads file headers, not tensor data. Additional folders are scanned recursively; unknown formats stay visible but cannot become a default automatically. Official downloads remain available in the same lists. Defaults are remembered per compatible family.

MOSS needs:

- The Base transformer: official BF16 shards, an equivalent single safetensors checkpoint, or a Comfy ConvRot INT8 checkpoint.
- The FLUX.2 VAE, including the smaller decoder supplied with MOSS. The Anima/Krea2 VAE is incompatible. MOSS always runs its VAE in FP32.
- The Qwen3.5-9B text encoder: official shards or compatible ConvRot INT8 weights. This is different from the Qwen3/Qwen3-VL encoders used by other families.
- Tokenizer files from the official MOSS `tokenizer` directory. The built-in text-encoder download includes them. A separate tokenizer folder can be set in advanced options; otherwise the official tokenizer is fetched if absent locally.

Use `transformers>=5.3,<6`. Updating this source tree does not install packages into a running environment. Update dependencies only after stopping active training.

## Memory controls

| Setting | Effect |
|---|---|
| Training Precision | Precision used for training calculations; BF16 is the default. |
| Transformer Quantization | `ConvRot INT8` reduces frozen transformer weight memory. |
| Text Encoder Quantization | Independently converts frozen Qwen3.5 weights to ConvRot INT8. |
| Transformer Layer Offloading | Keeps the selected number of transformer layers in RAM; 0–42 for MOSS. The stored config key remains `blocks_to_swap`. |
| Text Encoder Layer Offloading | Keeps the selected number of Qwen3.5 layers in RAM; 0–32. Applies while encoding captions. |
| Text encoder cache | Precomputes caption features and releases Qwen before loading the transformer. Keep enabled on small GPUs. |

Already quantized ConvRot files are detected automatically, including their scales and rotation metadata. Missing shards, mismatched shapes, unsupported quantization, and incomplete INT8 metadata produce an explicit error. Quantization does not change checkpoint files on disk.

For a 12 GB GPU, a starting point for later testing is BF16 training, both quantization selectors on ConvRot INT8, batch size 1, gradient checkpointing, caption caching, and 32 text-encoder layers offloaded. Adjust transformer offloading and resolution after measuring a real step. This is a starting configuration, not a verified VRAM or speed guarantee. Transformer offloading requires gradient checkpointing. The legacy file-size block-swap preflight is hidden for MOSS because it does not account for dynamic quantization and shards.

## Compatibility

The family provides LoRA/LoKr/LoHa/T-LoRA training, normalized latent caching, caption caching, and Euler sample previews. ConvRot uses adapter bypass computation so frozen INT8 weights are never merged with floating-point adapter weights during training. OrthoLoRA and DoRA are rejected with ConvRot. Unquantized OrthoLoRA remains available.

FP16 training and FP8 checkpoints are not offered for this MOSS implementation. Anima-specific text-encoder training, norm training, edit conditioning, NaViT, SRA, Leap, tag shuffling, and KV-trim controls are hidden or rejected as appropriate. Shared optimizers, learning-rate schedules, losses, augmentation, saving, and resume options remain available. DOP captions without the trigger are pre-cached before transformer loading.

MOSS starts with its logit-normal training distribution (mean -1.09, standard deviation 0.95), Euler/simple sampling, 50 inference steps, CFG 3, sampling shift 5, and layer 22 skipped in the negative CFG branch. Training timestep distribution and sampling shift are separate settings.

## H2D-only offloading

H2D means RAM-to-GPU transfer. The frozen original weights remain in RAM. A layer is copied into a temporary GPU buffer when needed; after use, that buffer can be reused without copying identical weights back into RAM. This saves the reverse transfer. It still requires RAM, PCIe transfers, and enough GPU memory for active buffers and activations.

Studio's existing `PinnedBlockSwap` already uses this principle for the supported transformer families, with two GPU buffers and prefetching. MOSS reuses that implementation. There is no reason to introduce a second global swap engine just to match the new name. The new text encoder offloader handles heterogeneous Qwen layers independently under `no_grad`; it currently uses synchronous transfers without prefetching. It cannot be used for text-encoder training.

The labels **Transformer Layer Offloading** and **Text Encoder Layer Offloading** describe the two controls accurately. They do not imply that their units or implementation are identical to AI Toolkit's controls.

## KohakuMossTrainer

[KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer) is a separate trainer built around LyCORIS. Its small-memory profiles combine weight streaming, activation and gradient offloading, and optimizer state on the CPU. Its 6 GB profile requires offline latent and text caches; this claim does not mean every preparation stage fits in 6 GB. Published measurements also distinguish startup/kernel preparation from training steps.

Its FP16 residual fix, FP8/MXFP8 compute, token packing, and activation/gradient/optimizer offloading are separate features. They have not been transplanted into Studio. Bringing them over would require separate gradient, resume, memory, and speed checks. The current integration follows kohya's MOSS architecture and ConvRot implementation while retaining Studio's training loop and UI.

## Validation and rollback

CPU checks cover a small transformer, checkpointed gradients, BF16/INT8 loading, fused QKV conversion, adapter training/saving, text padding/cache reuse, and family configuration. The actual official file headers were compared without downloading weight tensors: all 463 transformer, 426 text-model, and 251 converted VAE entries matched the expected names and shapes.

Full-size GPU training, CUDA ConvRot kernels, GPU layer streaming, and measured VRAM/speed remain unverified here: the user's GPU was reserved for an active training run. No GPU tests or dependency installation were performed during this change.

Code before this integration is saved on GitHub at tag `backup/before-moss-image-2026-10-22`, commit `c088095730b0d50268295a04d7c45779ff66c120`. Development is isolated on `codex/moss-image-training`; `main` is unchanged. After stopping the application and its training jobs, `git switch main` returns to the previous code. The tag can also be opened/downloaded on GitHub. Git does not back up ignored datasets, model files, secrets, or training outputs. New settings fields are additive; there is no database migration.

Sources: [official model and files](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base), [pinned sd-scripts implementation](https://github.com/kohya-ss/sd-scripts/tree/11d8cedd0c67b93a10f8b2d75f43554f6f12c794), [sd-scripts 0.12.2 release](https://github.com/kohya-ss/sd-scripts/releases/tag/v0.12.2), [KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer).
