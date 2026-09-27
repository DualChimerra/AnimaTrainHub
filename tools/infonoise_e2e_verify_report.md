# InfoNoise E2E Verify Report

Script: `tools/infonoise_e2e_verify.py`  ·  generated at: 2026-06-04 16:58:58

## 0. Experiment setup

- **config × mmse_shape × grad_accum × baseline** combination matrix
- total_optsteps = 5000, N_warm = 500, log_every = 100, K = 64, bs = 16, B = 256, N_min = 50, M = 100
- seed = 42, mock noise_std = 0.1
- **paper reference**: CIFAR c = 0.15 (arxiv 2602.18647 §5, Algorithm 1, Eq 87); info window σ ∈ (0.05, 1.5) (Fig 4)
- 96 combinations total

## 1. Suggestion 1 (Gate pivot bug) end-to-end verify

### 1.1 paper_fig4 toy + grad_accum=1 + baseline=logit_normal comparison (core table)

| config | c final | c stable step | mass_low | mass_info_window | mass_high | KL→target | refresh_status |
|---|---|---|---|---|---|---|---|
| **current** | 0.001115 | never | 98.70% | 0.80% | 0.00% | 3.968 | ok |
| **fix_last_above** | 0.1037 | 500 | 19.90% | 63.30% | 0.00% | 0.0341 | ok |
| **fix_paper_c015** | 0.15 | 500 | 15.00% | 73.00% | 0.00% | 0.02865 | ok |
| **oracle** | 0.15 | 0 | 15.70% | 71.40% | 0.00% | 0.02488 | ok |

### 1.2 Robustness check across other mmse shapes (grad_accum=1, baseline=logit_normal)

#### unimodal_log

| config | c final | mass_low | mass_info | mass_high | KL→target |
|---|---|---|---|---|---|
| current | 0.001115 | 56.00% | 29.00% | 0.00% | 2.474 |
| fix_last_above | 1.715 | 0.10% | 80.10% | 0.00% | 1.956 |
| fix_paper_c015 | 0.15 | 1.30% | 96.30% | 0.00% | 0.02438 |
| oracle | 0.15 | 1.00% | 96.30% | 0.00% | 0.01986 |

#### bimodal_log

| config | c final | mass_low | mass_info | mass_high | KL→target |
|---|---|---|---|---|---|
| current | 0.001115 | 55.60% | 23.90% | 0.00% | 1.499 |
| fix_last_above | 0.4698 | 1.30% | 95.70% | 0.00% | 0.4453 |
| fix_paper_c015 | 0.15 | 2.70% | 90.30% | 0.00% | 0.03396 |
| oracle | 0.15 | 2.90% | 88.80% | 0.00% | 0.02973 |

#### monotone_decay

| config | c final | mass_low | mass_info | mass_high | KL→target |
|---|---|---|---|---|---|
| current | 0.001115 | 100.00% | 0.00% | 0.00% | 1.391 |
| fix_last_above | 0.007779 | 99.80% | 0.10% | 0.00% | 0.6339 |
| fix_paper_c015 | 0.15 | 72.80% | 19.10% | 0.00% | 0.07673 |
| oracle | 0.15 | 74.10% | 17.70% | 0.00% | 0.08269 |

### 1.3 X1 interaction effect (impact of grad_accum) - paper_fig4 + logit_normal

X1: N_warm is counted in `_internal_step` units (record count), not optimizer steps; when grad_accum>1,
warmup ends early, causing the sampler to run the gate on an EMA that hasn't converged enough yet. Table
below compares each config's behavior across different grad_accum values.

| config | grad_accum | c final | mass_info_window | mass_low | KL→target |
|---|---|---|---|---|---|
| current | 1 | 0.001115 | 0.80% | 98.70% | 3.968 |
| current | 2 | 0.001115 | 1.10% | 98.50% | 4.021 |
| current | 4 | 0.001115 | 0.60% | 98.50% | 4.032 |
| fix_last_above | 1 | 0.1037 | 63.30% | 19.90% | 0.0341 |
| fix_last_above | 2 | 0.1037 | 61.30% | 22.90% | 0.04011 |
| fix_last_above | 4 | 0.1037 | 63.00% | 20.80% | 0.04351 |
| fix_paper_c015 | 1 | 0.15 | 73.00% | 15.00% | 0.02865 |
| fix_paper_c015 | 2 | 0.15 | 71.20% | 17.70% | 0.02338 |
| fix_paper_c015 | 4 | 0.15 | 73.70% | 14.80% | 0.02691 |
| oracle | 1 | 0.15 | 71.40% | 15.70% | 0.02488 |
| oracle | 2 | 0.15 | 73.10% | 15.60% | 0.02607 |
| oracle | 4 | 0.15 | 71.30% | 15.10% | 0.01885 |

## 2. Baseline mode effect (paper_fig4 + grad_accum=1)

Baseline only affects sampling during warmup and while the CDF isn't ready yet; once adaptive,
InfoNoise's CDF takes over. Different baselines should converge to the same final mass under an ok-config.

| config | baseline | c final | mass_info_window | KL→target |
|---|---|---|---|---|
| current | logit_normal | 0.001115 | 0.80% | 3.968 |
| current | uniform | 0.001115 | 0.80% | 3.968 |
| fix_last_above | logit_normal | 0.1037 | 63.30% | 0.0341 |
| fix_last_above | uniform | 0.1037 | 63.30% | 0.0341 |
| fix_paper_c015 | logit_normal | 0.15 | 73.00% | 0.02865 |
| fix_paper_c015 | uniform | 0.15 | 73.00% | 0.02865 |
| oracle | logit_normal | 0.15 | 71.40% | 0.02488 |
| oracle | uniform | 0.15 | 71.40% | 0.02488 |

## 3. Key findings

### Finding 1: end-to-end reproduction of Suggestion 1 (gate pivot bug)

- Under `current` config, c_pivot final = **0.001115** (paper reports 0.15, off by 0.007434×)
- mass_low_quarter = **98.70%**, mass_info_window = **0.80%**, mass_high_quarter = **0.00%**
- **Verdict**: bug reproduced end-to-end (criterion: mass_low_quarter > 70%)

### Finding 2: effect of the fix_last_above fix

- c_pivot final = **0.1037** (back to paper's order of magnitude ✓)
- mass_info_window = **63.30%** (deviates from paper's 37-57%)
- KL→target = **0.0341**

### Finding 3: fix_paper_c015 vs fix_last_above (which is closer to oracle)

- KL(oracle → target) = 0.02488 (should be near 0; mock sample noise sets the floor)
- KL(fix_paper_c015 → target) = **0.02865**
- KL(fix_last_above → target) = **0.0341**
- Closer to oracle: **fix_paper_c015**

### Finding 4: robustness of fix_last_above across mmse shapes

- **paper_fig4**: c=0.1037, mass_info_window=63.30%, mass_low=19.90%
- **unimodal_log**: c=1.715, mass_info_window=80.10%, mass_low=0.10%
- **bimodal_log**: c=0.4698, mass_info_window=95.70%, mass_low=1.30%
- **monotone_decay**: c=0.007779, mass_info_window=0.10%, mass_low=99.80%

- **Degradation**: on `['monotone_decay']`, mass_info_window < 20%. Reason: when mmse decreases monotonically
  (monotone_decay), the 1/σ³ tail decays in the same direction as mmse, r_norm declines gently over log-σ,
  the "above" region extends down into the low-σ end, and `last_above` still lands in the low-σ range

### Finding 5: X1 interaction effect (impact of grad_accum)

- **current**: mass_info_window ga1: 0.80% → ga2: 1.10% → ga4: 0.60%
- **fix_last_above**: mass_info_window ga1: 63.30% → ga2: 61.30% → ga4: 63.00%

- **Verdict**: fix_last_above still works at grad_accum=4, mass_info_window=63.00% (✓)

- **Note**: this verify uses a log-uniform baseline so every bin fills up, which sidesteps the other half of
  X1 (the real anima logit_normal_shift=3 baseline barely fills bins at low σ → n_count.min()=0 → refresh
  skips forever). That X1 component needs its own separate verify.


## 4. Quantifying deviation from paper §5 reported values

Paper CIFAR report: c ≈ 0.15 (Eq 87, §5), info window occupies 37-57% of sampled mass (Fig 4 B).

| config | mean c (final) | c / 0.15 | mean mass_info | deviation from paper |
|---|---|---|---|---|
| current | 0.001115 | 0.007434 | 0.80% | low by 36.2% |
| fix_last_above | 0.1037 | 0.6913 | 63.30% | high by 6.3% |
| fix_paper_c015 | 0.15 | 1 | 73.00% | high by 16.0% |
| oracle | 0.15 | 1 | 71.40% | high by 14.4% |

## 5. Recommendation: which fix should ship

### Recommendation: default to `fix_paper_c015`, with an escape-hatch field for users to override

Rationale:

1. **`current` reproduces the bug end-to-end**: mass_low_quarter = 98.70% on paper_fig4 (matches the paper's
   Algorithm 1 Eq 87 + §B.6 Θ(σ⁻¹) tail warning) — InfoNoise isn't actually taking effect
2. **fix_paper_c015 is more robust**: fix_last_above degrades on `['monotone_decay']` (mass_info_window < 20%),
   see Finding 4 for why. fix_paper_c015 pins c to the paper's CIFAR value, giving a floor even in the
   worst-case 1/σ³ tail shape
3. **average KL across mmse shapes**: fix_paper_c015=0.04093 vs fix_last_above=0.7673
4. **escape hatch**: add `infonoise_gate_pivot_c: float = 0.15` to the schema (defaults to the paper value);
   users can set it to 0 to fall back to dynamic `fix_last_above`, or set any other value to customize c
5. **doesn't break existing tests**: `tests/test_infonoise.py`'s oracle test only checks CDF monotonicity +
   endpoint values, not the actual value of c; landing the patch won't break tests. Suggest adding
   `test_gate_pivot_not_pinned_to_sigma_min` (asserting c >> σ_min across all 4 mmse profiles) to guard
   against regressions
6. **monotone_decay edge case**: all 4 configs show mass_info < 20% on monotone_decay, because under this
   mmse shape the 1/σ³ tail decays in the same direction as mmse — the gate fix alone can't fix this. This
   falls under P0-4's territory (Jacobian σ³→σ²), not P0-5 (gate pivot). Suggest shipping P0-4 + P0-5 in the
   same PR

## Appendix A: final metrics table for all combinations

| run_id | c_pivot | mass_low | mass_info | mass_high | KL | refresh_status |
|---|---|---|---|---|---|---|
| `current__bimodal_log__ga1__logit_normal` | 0.001115 | 55.60% | 23.90% | 0.00% | 1.499 | ok |
| `current__bimodal_log__ga1__uniform` | 0.001115 | 55.60% | 23.90% | 0.00% | 1.499 | ok |
| `current__bimodal_log__ga2__logit_normal` | 0.001115 | 57.90% | 22.40% | 0.00% | 1.618 | ok |
| `current__bimodal_log__ga2__uniform` | 0.001115 | 57.90% | 22.40% | 0.00% | 1.618 | ok |
| `current__bimodal_log__ga4__logit_normal` | 0.001115 | 59.00% | 23.10% | 0.00% | 1.575 | ok |
| `current__bimodal_log__ga4__uniform` | 0.001115 | 59.00% | 23.10% | 0.00% | 1.575 | ok |
| `current__monotone_decay__ga1__logit_normal` | 0.001115 | 100.00% | 0.00% | 0.00% | 1.391 | ok |
| `current__monotone_decay__ga1__uniform` | 0.001115 | 100.00% | 0.00% | 0.00% | 1.391 | ok |
| `current__monotone_decay__ga2__logit_normal` | 0.001115 | 100.00% | 0.00% | 0.00% | 1.381 | ok |
| `current__monotone_decay__ga2__uniform` | 0.001115 | 100.00% | 0.00% | 0.00% | 1.381 | ok |
| `current__monotone_decay__ga4__logit_normal` | 0.001115 | 100.00% | 0.00% | 0.00% | 1.392 | ok |
| `current__monotone_decay__ga4__uniform` | 0.001115 | 100.00% | 0.00% | 0.00% | 1.392 | ok |
| `current__paper_fig4__ga1__logit_normal` | 0.001115 | 98.70% | 0.80% | 0.00% | 3.968 | ok |
| `current__paper_fig4__ga1__uniform` | 0.001115 | 98.70% | 0.80% | 0.00% | 3.968 | ok |
| `current__paper_fig4__ga2__logit_normal` | 0.001115 | 98.50% | 1.10% | 0.00% | 4.021 | ok |
| `current__paper_fig4__ga2__uniform` | 0.001115 | 98.50% | 1.10% | 0.00% | 4.021 | ok |
| `current__paper_fig4__ga4__logit_normal` | 0.001115 | 98.50% | 0.60% | 0.00% | 4.032 | ok |
| `current__paper_fig4__ga4__uniform` | 0.001115 | 98.50% | 0.60% | 0.00% | 4.032 | ok |
| `current__unimodal_log__ga1__logit_normal` | 0.001115 | 56.00% | 29.00% | 0.00% | 2.474 | ok |
| `current__unimodal_log__ga1__uniform` | 0.001115 | 56.00% | 29.00% | 0.00% | 2.474 | ok |
| `current__unimodal_log__ga2__logit_normal` | 0.001115 | 58.80% | 28.20% | 0.00% | 2.662 | ok |
| `current__unimodal_log__ga2__uniform` | 0.001115 | 58.80% | 28.20% | 0.00% | 2.662 | ok |
| `current__unimodal_log__ga4__logit_normal` | 0.001115 | 59.10% | 28.70% | 0.00% | 2.591 | ok |
| `current__unimodal_log__ga4__uniform` | 0.001115 | 59.10% | 28.70% | 0.00% | 2.591 | ok |
| `fix_last_above__bimodal_log__ga1__logit_normal` | 0.4698 | 1.30% | 95.70% | 0.00% | 0.4453 | ok |
| `fix_last_above__bimodal_log__ga1__uniform` | 0.4698 | 1.30% | 95.70% | 0.00% | 0.4453 | ok |
| `fix_last_above__bimodal_log__ga2__logit_normal` | 0.4698 | 2.20% | 92.90% | 0.00% | 0.4072 | ok |
| `fix_last_above__bimodal_log__ga2__uniform` | 0.4698 | 2.20% | 92.90% | 0.00% | 0.4072 | ok |
| `fix_last_above__bimodal_log__ga4__logit_normal` | 0.4698 | 1.70% | 94.30% | 0.00% | 0.4087 | ok |
| `fix_last_above__bimodal_log__ga4__uniform` | 0.4698 | 1.70% | 94.30% | 0.00% | 0.4087 | ok |
| `fix_last_above__monotone_decay__ga1__logit_normal` | 0.007779 | 99.80% | 0.10% | 0.00% | 0.6339 | ok |
| `fix_last_above__monotone_decay__ga1__uniform` | 0.007779 | 99.80% | 0.10% | 0.00% | 0.6339 | ok |
| `fix_last_above__monotone_decay__ga2__logit_normal` | 0.007779 | 99.50% | 0.20% | 0.00% | 0.6182 | ok |
| `fix_last_above__monotone_decay__ga2__uniform` | 0.007779 | 99.50% | 0.20% | 0.00% | 0.6182 | ok |
| `fix_last_above__monotone_decay__ga4__logit_normal` | 0.007779 | 99.70% | 0.10% | 0.00% | 0.6409 | ok |
| `fix_last_above__monotone_decay__ga4__uniform` | 0.007779 | 99.70% | 0.10% | 0.00% | 0.6409 | ok |
| `fix_last_above__paper_fig4__ga1__logit_normal` | 0.1037 | 19.90% | 63.30% | 0.00% | 0.0341 | ok |
| `fix_last_above__paper_fig4__ga1__uniform` | 0.1037 | 19.90% | 63.30% | 0.00% | 0.0341 | ok |
| `fix_last_above__paper_fig4__ga2__logit_normal` | 0.1037 | 22.90% | 61.30% | 0.00% | 0.04011 | ok |
| `fix_last_above__paper_fig4__ga2__uniform` | 0.1037 | 22.90% | 61.30% | 0.00% | 0.04011 | ok |
| `fix_last_above__paper_fig4__ga4__logit_normal` | 0.1037 | 20.80% | 63.00% | 0.00% | 0.04351 | ok |
| `fix_last_above__paper_fig4__ga4__uniform` | 0.1037 | 20.80% | 63.00% | 0.00% | 0.04351 | ok |
| `fix_last_above__unimodal_log__ga1__logit_normal` | 1.715 | 0.10% | 80.10% | 0.00% | 1.956 | ok |
| `fix_last_above__unimodal_log__ga1__uniform` | 1.715 | 0.10% | 80.10% | 0.00% | 1.956 | ok |
| `fix_last_above__unimodal_log__ga2__logit_normal` | 1.715 | 0.10% | 80.00% | 0.00% | 1.84 | ok |
| `fix_last_above__unimodal_log__ga2__uniform` | 1.715 | 0.10% | 80.00% | 0.00% | 1.84 | ok |
| `fix_last_above__unimodal_log__ga4__logit_normal` | 1.715 | 0.20% | 80.80% | 0.00% | 1.892 | ok |
| `fix_last_above__unimodal_log__ga4__uniform` | 1.715 | 0.20% | 80.80% | 0.00% | 1.892 | ok |
| `fix_paper_c015__bimodal_log__ga1__logit_normal` | 0.15 | 2.70% | 90.30% | 0.00% | 0.03396 | ok |
| `fix_paper_c015__bimodal_log__ga1__uniform` | 0.15 | 2.70% | 90.30% | 0.00% | 0.03396 | ok |
| `fix_paper_c015__bimodal_log__ga2__logit_normal` | 0.15 | 3.70% | 88.30% | 0.00% | 0.02415 | ok |
| `fix_paper_c015__bimodal_log__ga2__uniform` | 0.15 | 3.70% | 88.30% | 0.00% | 0.02415 | ok |
| `fix_paper_c015__bimodal_log__ga4__logit_normal` | 0.15 | 3.50% | 88.60% | 0.00% | 0.02432 | ok |
| `fix_paper_c015__bimodal_log__ga4__uniform` | 0.15 | 3.50% | 88.60% | 0.00% | 0.02432 | ok |
| `fix_paper_c015__monotone_decay__ga1__logit_normal` | 0.15 | 72.80% | 19.10% | 0.00% | 0.07673 | ok |
| `fix_paper_c015__monotone_decay__ga1__uniform` | 0.15 | 72.80% | 19.10% | 0.00% | 0.07673 | ok |
| `fix_paper_c015__monotone_decay__ga2__logit_normal` | 0.15 | 73.20% | 18.80% | 0.00% | 0.07675 | ok |
| `fix_paper_c015__monotone_decay__ga2__uniform` | 0.15 | 73.20% | 18.80% | 0.00% | 0.07675 | ok |
| `fix_paper_c015__monotone_decay__ga4__logit_normal` | 0.15 | 73.90% | 18.40% | 0.00% | 0.07874 | ok |
| `fix_paper_c015__monotone_decay__ga4__uniform` | 0.15 | 73.90% | 18.40% | 0.00% | 0.07874 | ok |
| `fix_paper_c015__paper_fig4__ga1__logit_normal` | 0.15 | 15.00% | 73.00% | 0.00% | 0.02865 | ok |
| `fix_paper_c015__paper_fig4__ga1__uniform` | 0.15 | 15.00% | 73.00% | 0.00% | 0.02865 | ok |
| `fix_paper_c015__paper_fig4__ga2__logit_normal` | 0.15 | 17.70% | 71.20% | 0.00% | 0.02338 | ok |
| `fix_paper_c015__paper_fig4__ga2__uniform` | 0.15 | 17.70% | 71.20% | 0.00% | 0.02338 | ok |
| `fix_paper_c015__paper_fig4__ga4__logit_normal` | 0.15 | 14.80% | 73.70% | 0.00% | 0.02691 | ok |
| `fix_paper_c015__paper_fig4__ga4__uniform` | 0.15 | 14.80% | 73.70% | 0.00% | 0.02691 | ok |
| `fix_paper_c015__unimodal_log__ga1__logit_normal` | 0.15 | 1.30% | 96.30% | 0.00% | 0.02438 | ok |
| `fix_paper_c015__unimodal_log__ga1__uniform` | 0.15 | 1.30% | 96.30% | 0.00% | 0.02438 | ok |
| `fix_paper_c015__unimodal_log__ga2__logit_normal` | 0.15 | 1.80% | 94.60% | 0.00% | 0.02193 | ok |
| `fix_paper_c015__unimodal_log__ga2__uniform` | 0.15 | 1.80% | 94.60% | 0.00% | 0.02193 | ok |
| `fix_paper_c015__unimodal_log__ga4__logit_normal` | 0.15 | 1.20% | 95.90% | 0.00% | 0.02352 | ok |
| `fix_paper_c015__unimodal_log__ga4__uniform` | 0.15 | 1.20% | 95.90% | 0.00% | 0.02352 | ok |
| `oracle__bimodal_log__ga1__logit_normal` | 0.15 | 2.90% | 88.80% | 0.00% | 0.02973 | ok |
| `oracle__bimodal_log__ga1__uniform` | 0.15 | 2.90% | 88.80% | 0.00% | 0.02973 | ok |
| `oracle__bimodal_log__ga2__logit_normal` | 0.15 | 2.60% | 90.50% | 0.00% | 0.03433 | ok |
| `oracle__bimodal_log__ga2__uniform` | 0.15 | 2.60% | 90.50% | 0.00% | 0.03433 | ok |
| `oracle__bimodal_log__ga4__logit_normal` | 0.15 | 2.50% | 90.20% | 0.00% | 0.02371 | ok |
| `oracle__bimodal_log__ga4__uniform` | 0.15 | 2.50% | 90.20% | 0.00% | 0.02371 | ok |
| `oracle__monotone_decay__ga1__logit_normal` | 0.15 | 74.10% | 17.70% | 0.00% | 0.08269 | ok |
| `oracle__monotone_decay__ga1__uniform` | 0.15 | 74.10% | 17.70% | 0.00% | 0.08269 | ok |
| `oracle__monotone_decay__ga2__logit_normal` | 0.15 | 71.30% | 20.10% | 0.00% | 0.07809 | ok |
| `oracle__monotone_decay__ga2__uniform` | 0.15 | 71.30% | 20.10% | 0.00% | 0.07809 | ok |
| `oracle__monotone_decay__ga4__logit_normal` | 0.15 | 73.30% | 18.90% | 0.00% | 0.07919 | ok |
| `oracle__monotone_decay__ga4__uniform` | 0.15 | 73.30% | 18.90% | 0.00% | 0.07919 | ok |
| `oracle__paper_fig4__ga1__logit_normal` | 0.15 | 15.70% | 71.40% | 0.00% | 0.02488 | ok |
| `oracle__paper_fig4__ga1__uniform` | 0.15 | 15.70% | 71.40% | 0.00% | 0.02488 | ok |
| `oracle__paper_fig4__ga2__logit_normal` | 0.15 | 15.60% | 73.10% | 0.00% | 0.02607 | ok |
| `oracle__paper_fig4__ga2__uniform` | 0.15 | 15.60% | 73.10% | 0.00% | 0.02607 | ok |
| `oracle__paper_fig4__ga4__logit_normal` | 0.15 | 15.10% | 71.30% | 0.00% | 0.01885 | ok |
| `oracle__paper_fig4__ga4__uniform` | 0.15 | 15.10% | 71.30% | 0.00% | 0.01885 | ok |
| `oracle__unimodal_log__ga1__logit_normal` | 0.15 | 1.00% | 96.30% | 0.00% | 0.01986 | ok |
| `oracle__unimodal_log__ga1__uniform` | 0.15 | 1.00% | 96.30% | 0.00% | 0.01986 | ok |
| `oracle__unimodal_log__ga2__logit_normal` | 0.15 | 1.00% | 96.50% | 0.00% | 0.02251 | ok |
| `oracle__unimodal_log__ga2__uniform` | 0.15 | 1.00% | 96.50% | 0.00% | 0.02251 | ok |
| `oracle__unimodal_log__ga4__logit_normal` | 0.15 | 1.00% | 96.70% | 0.00% | 0.01998 | ok |
| `oracle__unimodal_log__ga4__uniform` | 0.15 | 1.00% | 96.70% | 0.00% | 0.01998 | ok |
</content>
