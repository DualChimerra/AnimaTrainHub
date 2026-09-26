# Style-Friendly SNR Sampler (timestep sampling for style LoRAs)

> Status: **implemented** (`timestep_sampling: style_friendly`), opt-in / default-off.
> When this mode isn't selected, the whole pipeline is byte-for-byte equivalent to before the change.
> Paper: [Style-Friendly SNR Sampler for Style-Driven Generation](https://arxiv.org/abs/2411.14793)
> (validated on FLUX-dev / SD3.5; both are rectified flow like Anima, so the math carries over directly).

## 1. What problem this solves

A training step first draws a noise level `t`. **Different noise bands teach the model different things**:

| t range | what the model can see at this level | what it learns |
|---|---|---|
| high (≈0.8-0.99) | only large color blobs and light/dark relationships | color, lighting, composition, blocking, overall mood = **style** |
| mid (≈0.4-0.7) | outlines and form taking shape | structure, anatomy, pose |
| low (≈0.05-0.3) | fine lines and texture | brush texture, fur strands, noise grain = **detail / easily overfit region** |

SD3/Anima's default `logit_normal` concentrates firepower in the mid range, undersampling the style band. The paper's observation is:
pushing the log-SNR distribution as a whole toward low values (= high noise) significantly improves style fidelity — **rank 32 with this
sampler beats rank 128 with the SD3 sampler**, i.e. "sampling the right noise band" is more cost-effective than "stacking trainable
parameters."

The practical implication for style LoRAs: when you want "the overall mood to match, without memorizing the dataset's details," this is a
more direct lever than tuning rank / LR / step count.

## 2. Math

This repo's rectified flow convention: `t=0` is the data end, `t=1` is the noise end, `x_t = (1-t)·x0 + t·x1`.

```
SNR = ((1-t)/t)²                  # signal-to-noise power ratio
λ   = log-SNR = 2·ln((1-t)/t)
t   = sigmoid(-λ/2)               # inverse
```

Sampling: `λ ~ N(style_snr_mean, style_snr_sigma²)` → `t = sigmoid(-λ/2)`.

Implementation: `runtime/training/timestep_sampling.py::sample_t_style_friendly`.

## 3. How to enable it

```yaml
timestep_sampling: style_friendly   # new enum value
style_snr_mean: -6.0                # log-SNR mean; the paper's FLUX/SD3.5 recipe
style_snr_sigma: 2.0                # log-SNR standard deviation; paper recommends 2.0-3.0
```

| mean | median t | effect |
|---|---|---|
| -8 | ≈0.982 | extreme style band; structure/detail are barely learned |
| **-6** | **≈0.953** | the paper's recipe, style-prioritized |
| -4 | ≈0.881 | mostly style + a bit of structure |
| -2 | ≈0.731 | roughly equivalent to the current default `logit_normal + shift 3` (≈0.75) |
| 0 | 0.5 | neutral |

`style_snr_sigma` is the window width: smaller = firepower concentrated but narrow coverage (overfitting risk in that band),
larger = wide coverage but back to being diluted. Start with 2.0.

## 3.5 Exact relationship to the current default

The current default `logit_normal + timestep_shift=s` is **exactly a special case of this mode**:

```
u = sigmoid(z), z~N(0,1); the Möbius shift is a translation in log-odds space, logit(t) = z + ln s
λ = 2·ln((1-t)/t) = -2·logit(t)  ⇒  λ ~ N(-2·ln s, 2²)
```

That is, `shift=s` ≡ `style_friendly(mean=-2·ln s, sigma=2)`:

| timestep_shift | equivalent style_snr_mean | median t |
|---|---|---|
| 2.0 | -1.39 | 0.667 |
| 2.5 | -1.83 | 0.714 |
| **3.0 (current default)** | **-2.20** | 0.749 |
| 4.0 | -2.77 | 0.799 |

The paper's style-band recipe is **-6** — nearly 4 log-SNR units lower than the current default. This is the quantified answer to
"how far the default config is from the style band," and also why simply increasing `timestep_shift` can't get you there: reaching
mean=-6 would require shift≈20, far beyond the field's cap of 10 (and sigma on that path is hardcoded to 2, not adjustable).

For comparison (sigma=2.0):

| style_snr_mean | median t | fraction of samples with t>0.9 |
|---|---|---|
| -8 | 0.982 | 96% |
| -6 | 0.953 | 79% |
| -5 | 0.924 | 62% |
| -4 | 0.881 | 42% |
| -3 | 0.817 | 24% |
| -2.2 (= current default) | 0.749 | 13% |

This contract is pinned down by `test_is_a_strict_generalisation_of_logit_normal_shift`.

## 4. Relationship to existing knobs

- **`timestep_shift` doesn't participate.** It's a Möbius offset that acts on the `u` inside logit-normal; this mode's offset comes
  entirely from `style_snr_mean`, and stacking both would be a double offset. The schema hides this field when this mode is selected,
  and `sample_t` doesn't read it either (`test_sample_t_dispatches_and_ignores_timestep_shift`).
- **`timestep_schedule_shift` still stacks** (default 1.0 = identity). It's a global σ-schedule offset applied after sampling, orthogonal
  to this mode; generally keep it at 1.0.
- **`timestep_shift_resolution_aware` still applies independently**: it does a per-image resolution correction based on token count.
  Note that when combined with this mode it's still a multiplicative composition — under native-resolution training, large images get
  pushed further toward the noise end.
- **`loss_weighting=detail_inv_t`** points in the opposite direction from this mode (it emphasizes low-t detail). Using both together is
  like hitting the gas and the brake at once — not recommended.
- **InfoNoise**: when enabled, the adaptive CDF takes over the distribution during the main phase; this mode only serves as the
  warmup-phase baseline (the user's mean/sigma settings are passed through to it).

## 5. Tests

`tests/test_style_friendly_sampler.py` (CPU-only): convention mapping, the paper's default landing point,
mean monotonicity, sigma width, open interval, shift not participating, schedule_shift still stacking,
registry wiring, `mean=0.0` not being swallowed by a falsy default, schema field and show_when.
