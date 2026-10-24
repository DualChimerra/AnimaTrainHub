# Optimizer selection and starting parameters

Recommended starting lr / weight_decay for each optimizer, plus conversion rules when switching from AdamW. The schema field descriptions only say "what the parameter is" — **tuning advice lives here**.

## Overview

| Optimizer | Recommended starting lr | weight_decay | scheduler | State VRAM vs AdamW fp32 | Best for |
|---|---|---|---|---|---|
| **adamw** | 1e-4 | 0.01 | cosine / cosine_with_warmup | 100% (baseline) | Default baseline, almost no pitfalls |
| **adamw8bit** | same as adamw (1e-4) | same as adamw (0.01) | same as adamw | **≈ 25%** (both moments quantized to int8) | Tight on VRAM; hyperparameters carry over from AdamW with no conversion |
| **lion** | ≈ AdamW lr / 3 (1e-4 → 3e-5) | AdamW wd × 3-10 (0.01 → 0.03-0.1) | cosine / cosine_with_warmup | **≈ 50%** (only exp_avg) | Tight on VRAM but still want a fixed lr |
| **automagic** | **1e-6** (mandatory; UI switches it automatically) | 0 (usually left off) | **none** (internal per-param adaptation) | ≈ 50% (factored 2nd moment + int8 lr_mask) | Don't want to tune lr, don't want Prodigy either |
| **prodigy** | 1.0 (fixed, locked in the UI) | 0.01 | constant or cosine | Slightly more than AdamW (one extra `d` state) | General-purpose adaptive, the most reliable "don't tune lr" option |
| **prodigy_plus_schedulefree** | 1.0 (fixed) | 0.0 | **none** (Schedule-Free handles averaging internally) | A bit more than Prodigy (averaged weights) | Fixes Prodigy's mutation epoch / style-jump issue |
| **soap** | AdamW-scale (1e-4 ~ 3e-4) | 0.01 | cosine / cosine_with_warmup | **> AdamW** (exp_avg + exp_avg_sq + per-axis Shampoo GG/Q for each matrix) | Matrix-shaped adapters (LoRA/LoKr) that want faster convergence |
| **soap_sf** | AdamW-scale (1e-4 ~ 3e-4) | 0.01 | **none** (Schedule-Free averaging) | ≈ soap (z replaces exp_avg) | Want SOAP's speedup plus no lr schedule to tune; **for very short runs (≤ ~100 steps) use soap instead** |

> VRAM note: AdamW8bit (bitsandbytes) is the real VRAM-saving baseline (≈ 25% of AdamW fp32). Lion / Automagic save about half compared to fp32 AdamW, but **don't save more than AdamW8bit**.

## Muon

Select **Muon** under **Optimization → Learning rate and optimizer**. It works with the existing LoRA, LoKr,
LoHa, OrthoLoRA and T-LoRA adapters, including MOSS Image Base. Matrix parameters
use Muon; vector parameters use auxiliary AdamW in the same optimizer.

The default `muon_adjust_lr=match_rms_adamw` allows an AdamW-scale starting
learning rate; it does not guarantee the same best learning rate or result.
Momentum defaults to `0.95`, and internal adjustment iterations to `5`.
The `original` scaling option requires separate learning-rate tuning.
Normal learning-rate schedules and training-state checkpoints are supported.

## Riemann LoRA + Riemannion (experimental)

Select **Muon**, then enable **Use Riemannian** at the bottom of the optimizer card.
This selects the fixed-rank adapter internally; no separate Riemann adapter or
optimizer choice is needed. Config: `optimizer_type: muon`, `use_riemannian: true`.
Old configs with `riemann` / `riemannion` are migrated on load.

This adapter updates the combined LoRA correction. Its learning rate has a
different meaning from AdamW's; there is no established automatic conversion.
The application keeps the entered learning rate. Momentum defaults to `0.9`.
This implementation uses random initialization instead of the paper's LOI
initialization. It is experimental and has no demonstrated advantage for MOSS
style LoRAs in this project. It also adds computation compared with ordinary LoRA.

- Works through the shared adapter path for Anima, Krea 2 and MOSS Image Base.
  MOSS ConvRot INT8 and block swapping keep the frozen base separate from the adapter.
- Saved `.safetensors` files are ordinary LoRA files; generation needs no
  Riemannion optimizer or special loader.
- Full training-state snapshots preserve the moving bases and optimizer state.
  `resume_lora` can load an ordinary LoRA with the same rank and layer sizes;
  use a full state snapshot to continue with optimizer history.
- Alpha, rs-LoRA, network dropout and per-layer rank controls do not apply and are hidden.
  Caption Dropout remains available because it changes the training text, not the adapter.
  DoRA, normalization training, text encoder training and weight EMA are disabled.
  Normal checkpoints and samples remain available.

Adapted from upstream [Muon](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/039b3c445e06a7119d391b3bf703753d9c3008d7)
and [Riemannion](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/commit/42f510faa6e0a74676541393ca44fb506d8c2093).
Research: [Muon](https://kellerjordan.github.io/posts/muon/) and
[LoRA meets Riemannion](https://arxiv.org/abs/2507.12142).

## AdamW8bit — the only VRAM-saving option that needs no re-tuning

The update math is identical to AdamW; what it saves is **state storage**: `exp_avg` / `exp_avg_sq` are quantized to int8 in blocks (8 bytes per parameter → 2 bytes). So `lr` / `betas` / `weight_decay` carry over from AdamW **unchanged, with no conversion needed** — this is its real advantage over Lion (lr needs to be divided by 3) and Automagic (lr must be 1e-6).

How much it saves (at LoRA-training scale):

| Scenario | Trainable params | AdamW fp32 state | AdamW8bit state | Saved |
|---|---|---|---|---|
| Krea 2 LoRA rank 32 (all 264 layers) | 117.3M | ≈ 0.87 GB | ≈ 0.22 GB | **≈ 0.65 GB** |
| Krea 2 LoKr factor 8 / rank 32 | 14.7M | ≈ 0.11 GB | ≈ 0.03 GB | ≈ 0.08 GB |
| Anima, rank 32 | Depends on preset | 8 bytes/param | 2 bytes/param | params × 6 bytes |

On a 12GB card, that 0.65GB is real headroom when running Krea 2 **LoRA**; **LoKr already has only 1/8 the parameter count of LoRA, so optimizer state isn't a pressure point — switching to adamw8bit doesn't buy much there**. On a 24GB card, neither is usually necessary.

**Dependency**: `bitsandbytes` is an optional dependency and isn't installed by default (the Windows wheel doesn't always install cleanly). If you select adamw8bit without it installed, training fails immediately at startup with the install command shown — it won't run halfway through and then crash. To install: `pip install bitsandbytes`.

**Small tensors aren't quantized**: tensors below `min_8bit_size=4096` stay in fp32 (quantization gains too little and the precision loss is relatively larger). LoRA's A/B matrices are far above this threshold, so in practice they're all quantized to 8-bit.

**Resuming**: state is stored/loaded via the standard `state_dict()` / `load_state_dict()`, and the int8 buffers and quantization maps go into the checkpoint together. **But you can't switch optimizers mid-run** — a checkpoint saved with adamw8bit can't be resumed with adamw (and vice versa), since the state structures differ. To switch optimizers, either start over or use `resume_lora` only (without `resume_state`).

## Lion — switching from AdamW

Empirically, per the Lion paper (Chen et al. 2023, [arxiv 2302.06675](https://arxiv.org/abs/2302.06675) §4.3):

> "Lion needs a smaller learning rate than AdamW, e.g. 3-10× smaller, and a larger weight decay, e.g. 3-10× larger, to maintain similar effective weight decay strength."

| AdamW value | Recommended Lion conversion |
|---|---|
| lr = 1e-4 | **lr ≈ 3e-5** (× 1/3) |
| lr = 1e-5 | lr ≈ 3e-6 |
| weight_decay = 0.01 | **weight_decay ≈ 0.03-0.1** (× 3-10) |

**Why**: Lion's update is a fixed-size `sign()` step (`±lr`), unlike AdamW which scales by gradient magnitude. The same lr takes a much bigger step in Lion, so it needs to be lowered. Since the decoupled weight-decay update multiplies by lr, lowering lr means wd needs to go up to maintain the same effective decay strength.

If you plug in AdamW's 1e-4 directly, loss will most likely diverge or stall early in training. AnimaLoraStudio's `create_lion` prints a warning when lr ≥ 1e-4 is detected.

## Automagic — must start at 1e-6

Automagic ([Ostris](https://github.com/ostris/ai-toolkit)) uses per-parameter adaptive lr and needs no scheduler at all. **The `lr` field here is the initial per-parameter learning rate**, not the global step size you'd expect from a normal optimizer.

- Upstream (ostris / tdrussell) both default to `lr=1e-6`
- `[automagic_min_lr, automagic_max_lr]` defaults to `[1e-7, 1e-3]`; each parameter adapts within this range on its own via sign-agreement
- If the starting lr is too high (e.g. AdamW-scale 1e-4), the sign-agreement schedule needs many steps to pull the per-param lr back into the working range — early on this is equivalent to running 100× too hot

**UI switching**: when a user switches from another optimizer to Automagic, the frontend automatically rewrites `learning_rate` to 1e-6 (still manually adjustable). Saving a config / passing a value above 1e-5 directly via CLI triggers a warning from `create_automagic` at training startup, but it isn't force-corrected.

**Known behavior**: `automagic_min_lr` / `automagic_max_lr` / `automagic_lr_bump` are instance globals — **shared across all param groups when there are multiple, not per-group**. The current trainer only has a single group so this doesn't affect it; if LoRA+-style multi-group lr scheduling (B matrix at 16× lr) is introduced in the future, min/max/bump will still be a single value. This matches upstream behavior in both ostris/ai-toolkit and tdrussell/diffusion-pipe.

## Prodigy / PPSF — lr locked to 1.0

The Prodigy family internally estimates its own step size `d`, so **the `lr` field must be 1.0** (the factory forces this). Tuning focuses on:

- `prodigy_d_coef` / `ppsf_d_coef`: overall scaling factor for the estimated `d`. Push toward 2.0+ if underfitting, toward 0.5 if overfitting / on a small dataset.
- PPSF has one extra field beyond Prodigy, `prodigy_steps`: freezes the `d` estimate in the later part of training to avoid jumps; recommended to set it to 1/4 ~ 1/2 of total steps.

PPSF uses Schedule-Free averaging, so `optimizer.eval()` must be called before sample/save and `optimizer.train()` afterward. Studio handles this internally via the `optimizer_eval_mode` context manager; CLI users should refer to `utils/optimizer_utils.py:optimizer_eval_mode`.

## SOAP / SOAP-SF — second-order preconditioning for faster convergence

SOAP (Vyas et al. 2024, [arxiv 2409.11321](https://arxiv.org/abs/2409.11321)) is **Adam running in Shampoo's eigenbasis**: it rotates the gradient into the eigenbasis of the gradient covariance, runs standard Adam there, then rotates back. This converges faster for matrix-shaped parameters (the low-rank factors of LoRA / LoKr); compared to pure Shampoo, it saves compute by refreshing the eigenbasis less often via `soap_precondition_frequency`. **The point is convergence speed**, not better texture/quality on its own — switching to SOAP trades VRAM for speed.

`soap_sf` wraps SOAP in Schedule-Free (Defazio et al. 2024, *The Road Less Scheduled*, [arxiv 2405.15682](https://arxiv.org/abs/2405.15682)): it drops first-moment momentum and replaces the LR schedule with an interpolation between the base sequence z and the Polyak average x, so **`lr_scheduler` must be none** (validated fatally at startup), and sample/save automatically use the averaged x (handled uniformly by `optimizer_eval_mode`, same as PPSF).

**lr**: the SOAP family uses real AdamW-scale lr (**unlike Prodigy, don't put 1.0**). Start LoRA/LoKr at 1e-4 ~ 3e-4.

**The key speedup knob is `soap_max_precond_dim`** (per-axis threshold):

- If an axis's dimension is ≤ the threshold, that axis gets a full-rank second-order preconditioner; if it's above the threshold, that axis degrades to plain Adam.
- Setting it large (e.g. `10000`) lets large feature dimensions also get second-order treatment = **the main source of speedup**; setting it small (e.g. `256`) only preconditions the rank dimension = SOAP-lite, which saves VRAM but loses most of the speedup.
- Combine with `soap_precond_in_state: false` to keep the recomputable GG/Q out of the checkpoint, keeping state small (zero cost when training from scratch without resuming; resuming requires a cold rebuild of the eigenbasis, with a few transitional steps).

**Short-run caveat**: Schedule-Free's Polyak averaging lags badly on very short runs (≤ ~100 steps) — x ends up ≈ the centroid of the trajectory, i.e. underfit. In that regime use plain `soap`, not `soap_sf`; for runs in the thousands of steps, SF behaves normally, and it's best to judge sample quality after roughly step 880.

## Optimizer arguments

**Optimizer arguments** is the last block of the Model tab. Each line is one `name=value` passed to the selected optimizer, as in kohya's `--optimizer_args`:

```
betas=(0.9, 0.99)
eps=1e-6
nesterov=False
```

Values are Python literals; a plain word such as `original` is a string. These lines override the matching optimizer fields on the Training tab. Learning rate and weight decay keep their own fields and are rejected here. An unknown name stops training at startup, so typos do not pass silently.

## Which one to pick

- **No strong preference, want something reliable**: AdamW + cosine_with_warmup, follow the Anima default preset
- **Tight on VRAM but don't want to touch lr**: Lion, divide lr by 3 per the conversion above
- **Don't want to tune lr and don't want to deal with Schedule-Free quirks**: Prodigy
- **Style LoRA worried about mutation epochs**: prodigy_plus_schedulefree
- **Fine-grained per-param adaptation**: Automagic, remember to start at 1e-6
- **Want faster convergence and have the VRAM budget**: soap (with a scheduler) or soap_sf (schedule-free); set `soap_max_precond_dim` large; use soap for short runs
