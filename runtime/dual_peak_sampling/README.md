# Dual Peak Sampling

Experimental, configurable timestep sampling for flow-matching training.
The reusable package depends only on PyTorch, not on AnimaTrainHub, NumPy,
Pydantic, a model, or a GPU. It does **not** promise better style fidelity.

## Standalone use

This whole directory can become the root of a separate repository unchanged.
From that directory, run `pip install .` (or `pip install -e '.[test]'` for
development), then `python -m pytest`. No package installation is needed inside
AnimaTrainHub: its runtime directory already exposes this package.

```python
import torch
from dual_peak_sampling import DualPeakConfig, sample_timesteps, density

config = DualPeakConfig(peak1_position=0.525, peak2_position=0.85)
generator = torch.Generator(device="cpu").manual_seed(42)
t = sample_timesteps(4096, "cpu", config, generator=generator)
x = torch.linspace(0.001, 0.999, 10000, dtype=torch.float64)
p = density(x, config)  # analytic curve for plotting, NOT loss weights
```

`sample_timesteps` returns float32 `[batch_size]`, on the requested device.
Without an explicit generator it uses the global PyTorch RNG, compatible with
the trainer's seed and RNG checkpoint restoration. CPU and CUDA are supported;
matching seeds do not imply identical results across devices/PyTorch versions.
There is no internal RNG, learned state, or sample history.

## Distribution and agreed preset

Convention: `x_t = (1-t)*data + t*noise`; `t=0` is clean, `t=1` is noise.
On a 1000-step display, `t=0.85` is 850. Do not confuse this flow noise
coefficient with noise-to-signal sigma `t/(1-t)`, or a width in log-SNR units.

| Component | Location | Width in log-SNR | Relative weight |
|---|---|---|---|
| First peak | component mode t=0.525 | 0.55 | 0.15 |
| Second peak | component mode t=0.850 | 1.20 | 0.35 |
| Broad Style-Friendly background | log-SNR mean=-3.2 | 2.20 | 0.45 |
| Uniform coverage | entire interval | — | 0.05 |

For each sample, choose **one** component with normalized weights, then draw
from that component. Do not average component samples: that produces a different
distribution. Increasing one weight reduces the normalized shares of the others;
weights represent probability mass, not independent absolute peak heights.
Weights can be zero, but cannot all be zero. They need not add to one.

For a normal log-SNR component:

```
lambda ~ Normal(mean, width**2)
t = sigmoid(-lambda/2)
mean = -2 * (logit(position) - (width/2)**2 * (2*position - 1))
```

The last equation places the **individual component's density mode** at
`position`, including the change-of-variable factor. Peak widths are restricted
to `(0, 2.8]` so that each peak remains unimodal; the broad background allows
widths up to 6 and is specified by mean, not mode.

The sum has different maxima: with defaults, approximately **540 and 870**.
The broad background contributes over the whole interval, not only the tail.
Expected masses before optional trainer shifts (rounded):

| t interval | Probability |
|---|---|
| 0–0.3 | 2.09% |
| 0.3–0.6 | 21.46% |
| 0.6–0.9 | 58.64% |
| 0.9–1 | 17.80% |

The sampler clamps results to `[1e-4, 1-1e-4]` for numerical safety. `density`
describes the continuous mixture before that clamp, excluding its endpoint atoms.
Arbitrary settings can merge the two peaks; two visible maxima are not guaranteed.

Inspired by the [Anima style experiment](https://note.com/kuon_noise/n/n82be977f167d)
and [Style-Friendly SNR](https://arxiv.org/abs/2411.14793). This is a custom
mixture, not an official implementation or a proven universal style optimum.

## AnimaTrainHub integration

Select **Dual Peak — два пика для стиля** in the timestep sampling section
(advanced settings). YAML/CLI mode: `dual_peak`. All package parameters are
prefixed with `dual_peak_` in the trainer:

```yaml
timestep_sampling: dual_peak
dual_peak_peak1_position: 0.525
dual_peak_peak1_width: 0.55
dual_peak_peak1_weight: 0.15
dual_peak_peak2_position: 0.85
dual_peak_peak2_width: 1.20
dual_peak_peak2_weight: 0.35
dual_peak_background_mean: -3.2
dual_peak_background_width: 2.20
dual_peak_background_weight: 0.45
dual_peak_uniform_weight: 0.05

# Preserve the agreed curve for a controlled comparison:
timestep_schedule_shift: 1.0
timestep_shift_resolution_aware: false
infonoise_enabled: false
leap_enabled: false
loss_weighting: none
```

These are sampler settings, not a complete training configuration. Existing
configs/default sampling are not changed. `timestep_shift` is ignored in this
mode; schedule and resolution shifts still transform the final distribution.
InfoNoise uses this configuration only until its adaptive distribution is ready.
Leap uses it only for ordinary (non-Leap) training steps. Loss weighting changes
the relative training contribution after sampling, not the sampled histogram.

CLI example: `--timestep-sampling dual_peak --dual-peak-peak2-position 0.85`.
Only the small adapter in `training/timestep_samplers/dual_peak.py` knows about
the trainer's prefixed arguments. To extract the package, keep that adapter in
the trainer and replace the bundled directory with the installed dependency.

Before publishing a separate repository, choose a license and the final package
name/version. Nothing is published or uploaded by this integration.
