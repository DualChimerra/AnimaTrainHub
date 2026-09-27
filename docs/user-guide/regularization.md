# Anima LoRA Training Regularization Analysis Report

**Model**: Anima (an anime-tuned DiT model based on Cosmos)
**Training framework**: AnimaLoraToolkit
**Analysis date**: 2025-02

---

## 1. Model and training framework overview

### 1.1 Model architecture

| Component | Type | Description |
|------|------|------|
| **Backbone** | Cosmos DiT (MiniTrainDIT) | ViT-like Diffusion Transformer, not a U-Net |
| **Text encoding** | Qwen + T5 | Dual encoders, supports weighted tokens |
| **VAE** | AutoencoderKLCosmos | Based on the Qwen-Image VAE |
| **Diffusion formulation** | Flow Matching | Continuous-time flow matching, not DDPM/DDIM |
| **Scheduling** | CONST(flow) / simple | ER-SDE-Solver at inference |

### 1.2 LoRA injection structure

- **Target layers**: `q_proj`, `k_proj`, `v_proj`, `output_proj`, `mlp.layer1`, `mlp.layer2`
- **Injected layers**: approximately 316
- **Type**: standard LoRA or LoKr (LyCORIS)
- **Trainable parameters**: LoRA adapters only, the base Transformer is frozen

### 1.3 Current training configuration

| Item | Current state |
|------|------|
| Loss function | MSE (pred vs target, target = noise - latent) |
| Optimizer | AdamW, only `lr` is configurable, `weight_decay=0` |
| LR schedule | CosineAnnealingWarmRestarts |
| Gradient clipping | Not enabled |
| LoRA dropout | Supported but defaults to 0 |
| Data augmentation | tag_dropout, flip_augment |

---

## 2. How Cosmos/DiT differs from SD, and what that means for regularization

### 2.1 Architectural differences

| Dimension | SD (U-Net) | Cosmos DiT |
|------|------------|------------|
| Backbone | Convolution + local/global attention | Pure Transformer (self-attention + cross-attention) |
| Gradient path | Multi-scale, skip connections | Sequential blocks, longer gradient path |
| Parameter distribution | Convolutions make up a larger share | Entirely Linear layers and attention |
| Prior | General-purpose images | Anime-tuned |

### 2.2 Regularization design considerations

1. **Transformer gradients**: self-attention tends to produce larger gradients, so gradient clipping is more valuable for DiT.
2. **No convolutional inductive bias**: the model relies more on the data, so overfitting risk is relatively higher, making moderate regularization more warranted.
3. **Anime-tuned base**: the base model already carries a style prior; LoRA is a fine-tune, so regularization shouldn't be too strong or it will weaken that tuning.
4. **Flow Matching**: the target is smoother than in DDPM, but gradient explosion and overfitting still need to be guarded against.

---

## 3. Regularization options, one by one

### 3.1 Weight decay (L2)

**Principle**: applies an L2 penalty on parameters within AdamW, suppressing weight magnitude.

**Fit**:
- The LoRA weights are the only trainable part, so directly constraining their norm is reasonable
- DiT's Linear layers are sensitive to weight scale, so L2 helps stabilize training
- There's little public experience with LoRA weight decay in the Cosmos/DiT community, so start small

**Recommendation**:
- Enable it, starting at `0.01`
- Try `0.05` if overfitting is clearly visible
- Lower to `0.001` or disable it if outputs look blurry or underfit

**Implementation complexity**: low (optimizer parameter only)

---

### 3.2 Gradient clipping

**Principle**: bounds the gradient norm (or individual gradient elements) to prevent gradient explosion.

**Fit**:
- Self-attention tends to produce large gradients, so gradient clipping is common practice in DiT
- Mixed precision (bf16) is more numerically sensitive, and clipping helps stability
- Simple to implement and generally doesn't hurt convergence

**Recommendation**:
- Enable it, `clip_grad_norm_(params, max_norm=1.0)`
- Try `0.5` if training is still unstable
- If you've never seen NaN/Inf, `1.0` or slightly higher is fine

**Implementation complexity**: low (one line before `optimizer.step()`)

---

### 3.3 LoRA dropout

**Principle**: randomly drops part of the activations in the LoRA forward pass, effectively a random mask on the LoRA output.

**Fit**:
- Helps generalization with small datasets
- Less commonly used in diffusion LoRA, may slightly weaken fitting ability
- Already implemented (`LoRAInjector(dropout=...)`), just pass a nonzero value

**Recommendation**:
- Try `0.1` for small datasets (<500 images)
- Keep at `0` for medium/large datasets, prioritize weight decay instead
- If enabled, pick either dropout or weight decay as the primary one and weaken the other

**Implementation complexity**: low (interface already exists)

---

### 3.4 LoRA output regularization

**Principle**: applies an L2 penalty to the LoRA adapter's output, limiting how much it can modify the original model's output.

**Fit**:
- Suits scenarios that want to "fine-tune while preserving base model capability"
- The anime-tuned base already has its own style; over-modifying it risks damaging that prior
- Requires changes to the forward pass to obtain the adapter output, so implementation is more involved

**Recommendation**:
- Could be an advanced option, coefficient on the order of `1e-4` ~ `1e-3`
- Validate weight decay first before considering this

**Implementation complexity**: medium (requires changes to the forward pass and loss computation)

---

### 3.5 L1 regularization (sparsification)

**Principle**: applies an L1 penalty to LoRA parameters, pushing some weights toward zero.

**Fit**:
- Not commonly used in diffusion LoRA
- May weaken representational capacity and hurt generation quality
- Limited benefit for character/style fine-tuning, which needs a fair number of effective channels

**Recommendation**: not recommended as a first choice; only try it if you have a clear sparsity requirement.

**Implementation complexity**: low

---

### 3.6 EMA (Exponential Moving Average)

**Principle**: maintains an exponential moving average of the LoRA weights, and uses the EMA weights for inference/saving.

**Fit**:
- Can smooth out training fluctuations and improve generalization
- Fairly common in diffusion model training
- Requires maintaining extra EMA parameters and correctly switching between them at save/inference time

**Recommendation**:
- Could be a future enhancement
- Decay coefficient of `0.999` or `0.9999` is a common choice

**Implementation complexity**: high (needs a new EMA module and save logic)

---

## 4. Recommended implementation priority

| Priority | Option | Expected benefit | Implementation difficulty | Suggested parameter |
|--------|------|----------|----------|----------|
| P0 | Weight decay | Suppress overfitting, stabilize training | Low | 0.01 |
| P0 | Gradient clipping | Prevent gradient explosion, improve stability | Low | max_norm=1.0 |
| P1 | LoRA dropout | Generalization on small datasets | Low | 0.1 (optional) |
| P2 | LoRA output regularization | Limit modification of the base model | Medium | λ=1e-4 |
| P2 | EMA | Smoothing and generalization | High | decay=0.999 |
| P3 | L1 sparsification | Specific sparsity needs | Low | Not recommended |

---

## 5. Cosmos/DiT-specific notes

1. **Don't copy SD configs verbatim**: weight decay and similar parameters from Kohya-style SD LoRA scripts can't be transplanted directly — they need to be validated against Anima specifically.
2. **Anime-tuned base**: regularization shouldn't be too strong, or it risks weakening the existing anime style prior.
3. **Flow Matching**: the target is relatively smooth, but Transformer gradients can still be large, so gradient clipping still has value.
4. **Experimentation order**: start by adding weight decay and gradient clipping, observe loss and sample quality, then consider dropout or output regularization.

---

## 6. Summary

Anima, as an anime-tuned DiT model based on Cosmos, is well-suited for introducing regularization in LoRA training. The following are recommended as priorities:

1. **AdamW weight_decay = 0.01**
2. **Gradient clipping max_norm = 1.0**

Both are simple to implement, low-risk, and help with both overfitting and training stability. Other options can be introduced gradually based on need and experimental results.
