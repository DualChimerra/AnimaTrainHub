# EMA and DOP — Two switches for "don't wreck the style LoRA"

Both are opt-in / default-off; when disabled the whole pipeline is equivalent to before the change.
They're grouped in one doc because they address the same kind of pain for style LoRAs:
**overfitting and checkpoint selection being a matter of luck.**

---

## 1. EMA — Exponential Moving Average of weights

`ema_enabled` / `ema_decay` / `ema_start_ratio` · implementation: `runtime/training/ema.py`

### What it solves

The weights at any given training step are the result of "wherever the last random batch happened to push the parameters." One epoch drifts one way, the next drifts another way, so "which epoch is best" becomes a coin toss.

EMA maintains an extra smoothed copy during training; after every optimizer.step():

```
shadow ← decay · shadow + (1 - decay) · current weights
```

The normal checkpoint is still saved as usual, and an **additional** `*_ema.safetensors` is saved alongside it. You can try either one.

### How to configure it

| Field | Meaning | Recommendation |
|---|---|---|
| `ema_decay` | smoothing window. 0.999 ≈ the most recent 1000 update steps | total steps 2000-3000 → **0.999**; very few steps → 0.99 |
| `ema_start_ratio` | what percentage of total steps in to start accumulating | **0** (with warmup already included, this is enough); to only average over the plateau → 0.3 |

### Two implementation details

* **The shadow is fp32.** Training weights in bf16 only have 8 mantissa bits; the thousandth-scale increment from decay=0.999 gets rounded straight to 0 in a bf16 accumulator, silently degrading EMA into "an unmoving old copy."
* **The shadow starts from the weights at the moment training begins**, not from the LoRA's zero initialization; combined with the
  `(1+n)/(10+n)` warmup, it nearly tracks the current weights directly in the early steps. Otherwise, after 2760 steps there would still be
  0.999^2760 ≈ 6% of "zero" left in the average, needlessly diluting the LoRA by 6%.

Saving to disk reuses the adapter's own `save()` (via the `ema.applied()` context manager, which temporarily swaps in the shadow), so alpha
rewriting / `ss_*` metadata / family tagging all stay consistent. The shadow is saved together with the training state, so it survives resume.

---

## 2. DOP — Differential Output Preservation

`dop_enabled` / `dop_weight` / `dop_ratio` · implementation: `runtime/training/dop.py`
source: kohya-ss/sd-scripts PR #1710, ostris/ai-toolkit's feature of the same name.

### What it solves

Trained style LoRAs commonly have two problems:

1. Even without the trigger word in the prompt, it still alters the image (the style "leaks" everywhere);
2. It carries over the **content** of the dataset (the same typical character, the same kind of background) into generation results —
   "instead of applying the style on top, it copies the dataset."

The traditional fix is a regularization set: prepare a separate batch of neutral images. DOP doesn't need extra images:

1. Take the **same batch of training images**, strip the trigger word out of the caption;
2. Run a forward pass with the **adapter turned off** → what the base model would draw on its own (no_grad, a constant target);
3. Run the same input with the **adapter turned on** → what it draws now;
4. Add the MSE between the two, times `dop_weight`, into the total loss.

This teaches the LoRA two things simultaneously: **with the trigger word = apply my style; without the trigger word = change nothing.**
The content is identical in both branches, so "copying content" gets no reward on that path.

### How to configure it

| Field | Meaning | Recommendation |
|---|---|---|
| `dop_weight` | weight of the preservation term relative to the main loss (both are MSE, 1.0 = equal weight) | **1.0**; if style still leaks → 2-5; if style isn't learned at all → 0.3-0.5 |
| `dop_ratio` | what fraction of steps it's enabled on | **1.0**; if it's too slow → 0.5 (constraint strength also halves) |

### Prerequisites and cost

* **Requires a non-empty `trigger_word`.** The preservation branch is exactly "with the trigger word removed"; without a trigger word
  the two branches are identical and the constraint degenerates to 0. This fails fast at startup rather than silently degrading.
  The trigger word is entered in the "Trigger word" card at the top of the Train page (written to the version and forced to sync into
  config.yaml); "detect from caption" reads the captions under train/ and finds the word that's present in (almost) every one — usually
  the leading `@handle`. When enqueuing training with DOP enabled but the version has no trigger word: it first falls back to any existing
  value in the yaml, then tries auto-detecting from captions (must be 100% coverage and located at the start, or be an `@handle`); only if
  neither works does it error out at enqueue time.
* **Every enabled step costs two extra forward passes** (one no_grad reference + one with gradients), roughly 2-2.5× the per-step time.
* **Scope (v1)**: standard rectified flow path only. Mutually exclusive at the schema level with LeapAlign (which has its own objective
  function) and NaViT packing (which requires re-packing for per-image cross-attention).
* The adapter must implement `disabled()` (temporarily zeroing the scale). LyCORIS / OrthoLoRA already implement this;
  other adapters will error at startup rather than fail midway through training.

The monitoring dashboard and wandb get an extra `dop_loss` curve: it should decrease first and then flatten. If it stays flat long-term, it means
`dop_weight` is too small (the constraint isn't biting) or the trigger word's position in the caption is inconsistent.

---

## 3. Using both together

They don't conflict, and the recommended combination is: DOP handles "don't let the style leak, don't copy content," EMA handles "checkpoint
selection isn't a gamble." A starter recipe for style LoRAs:

```yaml
timestep_sampling: style_friendly   # concentrate firepower on the style band
style_snr_mean: -6.0
dop_enabled: true                   # don't copy content, don't leak
dop_weight: 1.0
ema_enabled: true                   # checkpoint selection isn't a gamble
ema_decay: 0.999
```

Tests: `tests/test_ema.py` (12 cases), `tests/test_dop.py` (17 cases), both CPU-only.
