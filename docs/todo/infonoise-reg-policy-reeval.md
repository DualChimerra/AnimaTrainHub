# InfoNoise reg-set record policy re-evaluation list

**Created** 2026-06-05
**Trigger** follow-up to PR #216 commit `cfbfd218`'s threshold approach; during review a user found that the `loss_weight >= 0.99` threshold has an internal inconsistency at the `reg_weight=1.0` boundary, so after a three-way algorithmic analysis it was changed to a hard filter on the `is_reg` flag.
**Current status** 🟢 Route 1 (is_reg hard exclude) has been landed in `runtime/training/loop.py:141-152` + `runtime/training/dataset.py`. This document records the rationale for deferring Route 2 (loss_weight soft weighting), and when to re-evaluate.

---

## Current implementation (Route 1: is_reg hard filter)

```python
# runtime/training/loop.py
if "is_reg" in batch:
    _main_mask = ~batch["is_reg"].to(t.device)
    if _main_mask.any():
        ctx.timestep_sampler.record(t.detach()[_main_mask], _raw_mse[_main_mask])
else:
    ctx.timestep_sampler.record(t.detach(), _raw_mse)
```

- `MergedDataset.__getitem__` writes `is_reg=False` for main samples and `is_reg=True` for reg samples
- `collate_fn` / `collate_fn_cached` pass `is_reg` through as a `torch.bool` tensor
- InfoNoise's FIFO/EMA only looks at (t, raw_mse) from main samples; the CDF learns mmse_main(t)
- reg samples still go into the gradient as usual, weighted by `reg_weight` (the block at loop.py:154-156 is untouched)
- for any value of `reg_weight` (0.3 / 0.7 / 1.0 / anything), the masking behavior is the same

## Alternative Route 2 (deferred: loss_weight soft weighting)

Instead of filtering, record into FIFO/EMA weighted by loss_weight:

```python
ctx.timestep_sampler.record(t, raw_mse, weights=batch["loss_weight"])
# InfoNoise's internal bucket aggregation changes to weighted: sum(w*mse) / sum(w)
```

Mathematical property: for the real objective `L = E_main[ℓ] + λ E_reg[ℓ]`, this is a min-variance importance sampling proposal
(Katharopoulos-Fleuret et al.), unbiased and continuous in λ.

## Rationale for choosing Route 1 over Route 2 (2026-06-05, three-agent analysis)

Three independent lenses:

| Lens | Q1 | Confidence |
|---|---|---|
| information theory / I-MMSE | exclude_all | 72% |
| weighted ERM / IS | soft_weight_by_loss_weight | 78% |
| training dynamics / user intent | exclude_all | 78% |

**Majority (2/3) support hard exclude.** All three unanimously rejected the old `loss_weight >= 0.99` approach (A: type error / B: arbitrary θ / C: identity vs. weight mismatch).

**Core point of contention**: what is InfoNoise's job as a tool?
- Route 1 (A+C): "an MMSE estimator for the main distribution," where reg is a means, not an end
- Route 2 (B): "the min-variance IS proposal for the weighted ERM you wrote down"

The real-world grounds for Route 1 winning:
1. The InfoNoise paper's contract (I-MMSE) strictly depends on a single distribution; the mixture generalization is unproven (Lens B admits this as blind spot #1)
2. 99% of AnimaLoraStudio's current user base is single-character / single-style LoRAs + general-purpose image reg; reg is an auxiliary prior, not a co-training target
3. `reg_weight=1.0` in actual community usage semantically means "strong regularization," not "multi-task balanced training"
4. Low engineering cost (1 field + 1 line of masking); Route 2 would require changing InfoNoise sampler's internal FIFO/EMA accumulators

## Re-evaluation trigger conditions

**Re-evaluate whether to switch to Route 2 or a hybrid approach if any of the following are met**:

1. **Data-scale trigger**: the proportion of users whose reg set is "distributionally similar" to their main set rises significantly (e.g. a lot of training configs appear that "use the same booru tag range as reg" or "use the same artist's other works as reg"). Measurement method: sample recent N reg sets from community/wandb submissions, look at the distribution of KL distance between their caption distribution and the main set's.

2. **Multi-main-distribution scenarios emerge**: multi-concept LoRA / training multiple characters simultaneously / training character+style simultaneously become mainstream. At that point the "single main distribution" assumption breaks down, and Route 1's "hard exclude reg" is no longer sufficient either — would need a per-distribution CDF (each distribution maintains its own entropy curve) or Route 2's weighted aggregation.

3. **High proportion of `reg_weight ≥ 0.7`**: community research finds a large number of users setting `reg_weight ≥ 0.7` (close to 1.0), indicating user intent has shifted from "auxiliary prior" to "co-training" — at which point Route 2's λ-continuous semantics fit user intent better.

4. **Progress in the InfoNoise paper / academic literature**: a formal generalization of I-MMSE to mixture distributions appears (e.g. a joint mixture-MMSE + weighted ERM analysis), or a paper connecting Katharopoulos-Fleuret-style IS with diffusion schedule optimization. This would give Route 2 the missing mathematical guarantee.

5. **Route 1 empirically found sub-optimal**: user feedback or a controlled experiment shows that, with reg set + InfoNoise enabled, convergence speed / final quality is clearly worse than a Route 2 simulation (even when the reg set's distribution differs). Would need an A/B experiment design (same config, run both recording strategies and compare against the main task's metric).

## Concrete actions upon re-evaluation

- **Decision: stay with Route 1**: archive this document to `docs/todo/archive/`, add a CHANGELOG entry describing the re-evaluation conclusion + data
- **Decision: switch to Route 2**:
  1. Add a `record(t, mse, weights=None)` interface to the InfoNoise sampler
  2. Change the FIFO bucket to a weighted streaming average (watch numerical stability, don't let low-weight samples get drowned out)
  3. Update the EMA accumulator to be weighted as well
  4. Change loop.py:141 to `ctx.timestep_sampler.record(t, raw_mse, weights=batch["loss_weight"])`
  5. Keep the `is_reg` field in dataset.py (still likely needed for identity labeling in multi-main-distribution scenarios)
  6. Rewrite the reg line in training-tips.md
  7. Add tests: (a) when λ=0, reg samples have zero effect on the schedule; (b) when λ=1, reg is weighted equally with main in recording; (c) numerical stability of the weighted streaming average
- **Decision: hybrid approach** (per-distribution CDF): open a separate ADR

## References / context

- Commit that triggered this discussion: `cfbfd218` (PR #216 follow-up to e7ac3e3)
- 3-agent workflow run: `wf_42e355a5-f01` (2026-06-05)
- full three-way analysis transcript: any archive under `tmp/infonoise/` + memory `infonoise-reg-policy-open.md`
- current implementation: `runtime/training/loop.py:141-152` + `runtime/training/dataset.py` MergedDataset / collate_fn
- related tests: `tests/test_infonoise.py::test_record_accepts_partial_batch_after_reg_mask`
- related memory: `infonoise_reg_policy_open.md`, `reg_dreambooth_alignment.md` (the overall role of reg sets in modern DiT training)

## Recheck cadence

- **Once every six months** (2026-12-05 / 2027-06-05 ...): sample the last 6 months of community/wandb reg-set configs, check whether "general-purpose image reg" is still the dominant pattern
- **Triggered recheck**: any of the 5 trigger conditions above is met
- **If multi-concept LoRA still isn't mainstream after 3 years** (after 2029-06-05): Route 1 can be considered a stable long-term solution, and this document can be downgraded to archive
