# Block swap (per-layer weight swap-in/swap-out) research

> **This fork's port scope (read this before the rest of the doc)**: what's
> been ported from upstream v0.21.0 / v0.21.1 is the **training side**
> (`runtime/training/block_swap.py` + the loader landing pinned memory
> directly + budget guardrails + `TrainingConfig.blocks_to_swap`). **The
> inference side (§4 / §9.6, `runtime/anima_daemon.py`'s image-generation
> path) has not been ported yet** — it depends on the
> `lora_merge_precision` / `chunk_rows` / `keep_backup` merge parameter set
> introduced by upstream #455 ("reduce peak VRAM during generation"), which
> this fork doesn't have. As a result, `GenerateConfig` has no
> `blocks_to_swap`, and generation still loads the full model onto the GPU.
> Sections in this document about the inference side describe upstream's
> state, kept as a reference for a future port.

- Status: **Gate-0 has passed (real measurements in §5.1); the approach
  hasn't been finalized.** The mechanism, cost model, hardware-impact
  assessment, and measured data are locked in; all product and
  implementation choices remain in §7's open questions, to be confirmed
  one by one through the five-step process before any code is written.
- One-line conclusion: **already implemented and verified on real
  hardware. With fp8 + full swap-out (28 layers), the training-step peak is
  8.4GB, minimum requirement ≈10.1GB → comfortable on 16GB, feasible on
  12GB (§9.5c), with a speed cost of about 4%; hardware-wear concerns were
  ruled out by a 1.4TB transfer test (zero PCIe replay increase). The B12
  target has been achieved, pending confirmation on smaller cards.**
- Date: 2026-07-21
- Upstream basis: `docs/design/multi-model/00-decisions.md` D2 (fp8 / block
  swap shelved, K2's floor set at 32GB bf16); `docs/design/multi-model/04-synthesis.md`
  §7 Phase 0 measurement "32GB is trainable but with ≈0 headroom" → this
  promotes "K2-specific small-scale block-swap fallback" from a candidate
  to an **in-plan optional feature (off by default, on when VRAM runs
  out)**. This document expands on that item.
- Reference implementations: kohya-ss/musubi-tuner (`blocks_to_swap`,
  Apache-2.0), ai-toolkit, diffusion-pipe, ComfyUI's `--lowvram` partial
  load.

---

## 1. What it is

Keep part of the DiT's transformer blocks' **weights resident in CPU
memory**, moving them into VRAM only when it's that block's turn to compute
in the forward/backward pass, and releasing them immediately afterward. The
number of swapped-out blocks is the only knob (the musubi ecosystem's
`blocks_to_swap=N`).

The premise relies on a structural fact about DiTs: N structurally identical
blocks stacked serially, with **only one actually participating in
computation at any moment**. Keeping all weights resident in VRAM exists
purely to save transfer time — it's not a requirement of the computation
itself.

The structure of the two families in this repo (probed directly from code,
not estimated):

| Family | Block count | Key dimensions | Single block (bf16) | Per-block hook point |
|---|---|---|---|---|
| Anima | 28 (36 for the large variant) | features set by the ckpt, heads 16 / 40 for the 36-block version | untested (§5.3-2) | [`forward.py:31`](../../runtime/training/families/anima/forward.py) is already a per-block loop |
| Krea2 | 28 | features 6144, heads 48, kvheads 12 (GQA), multiplier 4 | **828 MB** (434.2M params) | `modeling/krea2/krea2_modeling.py`'s `SingleStreamDiT`, `forward` is already a per-block loop |

Krea2's 28-layer structure is completely isomorphic to Anima's — this means
**a single swap mechanism can be shared by both families**, with no need to
fan it out per family (exactly the maintenance debt criticized in
`02-ecosystem-survey.md` §7 regarding SimpleTuner).

## 2. Mechanism and cost model

### 2.1 Mechanism

1. The master copy of the weights lives in CPU **pinned memory** (page-locked).
   Non-pinned pageable memory can't do asynchronous DMA transfer;
   `non_blocking=True` silently degrades to a synchronous copy, halving
   bandwidth and exposing it entirely.
2. An independent CUDA stream handles prefetching: while computing block
   *i*, block *i+1* is moved in the background.
3. Once block *i* finishes computing, its GPU copy is released immediately.
4. The backward pass order is reversed (N→0), requiring reverse-order
   prefetching.

### 2.2 The single success/failure criterion

```
fully hideable  ⟺  T_transfer(block) < T_compute(block)
T_transfer  = bytes_per_block / effective_pcie_bandwidth
```

Exposed time = `max(0, T_transfer − T_compute) × swapped_block_count × passes_per_step`.

This gives two counterintuitive but important corollaries:

- **The higher the resolution and the larger the batch, the more worthwhile
  block swap becomes.** Compute time grows with token count, while transfer
  time stays constant (weight size is fixed). Low resolution + small batch
  is the worst-case scenario.
- **An fp8 base model makes block swap easier to hide**, since transfer
  bytes are halved while compute time isn't halved (this repo's
  `quant_fp8.py` does per-layer dequant followed by bf16 matmul, so compute
  volume is unchanged). Combining the two is a positive synergy.

### 2.3 The extra bonus for LoRA training

The base model's weights are **frozen** → once the GPU copy is used, it can
just be discarded, with **no D2H write-back needed** — only a one-way H2D
transfer. Full fine-tuning would require moving data in both directions. So:

| Scenario | Transfer direction per step | Relative transfer volume |
|---|---|---|
| LoRA training (this repo's only scenario) | forward H2D + backward H2D | 2x |
| Full fine-tuning | forward H2D + backward H2D + D2H write-back | 3x-4x |
| Inference (per step) | H2D | 1x × steps |

This repo only does LoRA → falling into the cheapest tier. LoRA parameters
themselves are tiny, and along with optimizer state, **stay resident on the
GPU and never participate in swap**.

### 2.4 Interaction with gradient checkpointing (an implementation hurdle)

With checkpointing enabled, the backward pass has to **recompute the
forward pass**, at which point that block's weights need to be in place
again. A naive implementation would move the same block 3 times within a
single step (forward once + recompute once + backward once). Musubi's
approach merges the recompute and backward into the same residency window.

This repo's Anima
[`forward_with_optional_checkpoint`](../../runtime/training/families/anima/forward.py)
is already a manually unrolled per-block loop, **which is a natural and the
only hook point** — a prefetch hook can be inserted with no architecture
changes required. On the Krea2 side, it still needs confirming whether
`SingleStreamDiT.forward` has the same unrolled surface.

### 2.5 Orthogonal relationship with existing measures

| Measure | What it cuts | Cost | Current status in this repo |
|---|---|---|---|
| gradient checkpointing | activations | recompute ≈ +30% time | already present, forced on by default for K2 v1 |
| fp8 quantization | weight **byte count** | precision | `quant_fp8.py` already implemented (inference + fp8_base training) |
| model-level offload | **inactive models** (TE/VAE/DiT as a whole) | switching latency | the 3-tier `vram_policy` already implemented |
| **block swap** | weight **residency location** (byte count unchanged) | PCIe time | the subject of this document |

**Fundamental difference from the existing `vram_policy`**: model-level
offload solves "TE + DiT don't fit at the same time," while block swap
solves "**a single DiT alone doesn't fit**." The former can't rescue running
a 20B model on 24GB; the latter can. This is its sole, irreplaceable value
for K2.

## 3. Hardware impact and wear assessment

An area the user explicitly cares about. The conclusion first: **there is
no hardware wear in the sense of "wearing out"; the real risks are all
around system stability and thermals, and both are measurable.**

### 3.1 Parts that don't constitute wear (can proceed with confidence)

- **PCIe link**: a differential-signal electrical link, with no mechanical
  or storage-media wear mechanism. Sustained full-bandwidth use is a normal
  operating condition within spec (datacenter GPUs run like this year-round).
  PHY power draw at full load is on the order of single-digit watts,
  negligible against a GPU's overall 300-450W.
- **VRAM (GDDR6/6X) and system memory (DDR)**: DRAM is capacitor-based
  storage, and reads/writes **produce no fatigue**. NAND flash's
  write-cycle-endurance concept doesn't apply here.
- **The actual mechanisms of semiconductor aging** (electromigration,
  NBTI/HCI) are driven by **temperature and voltage**, and only indirectly
  related to "how many bytes were moved" — the indirect path is "more
  activity → higher power draw → higher temperature." And during a block
  swap wait bubble, the GPU's **average power draw actually drops**.

### 3.2 Parts that constitute real risk (guardrails required)

**① Pinned memory can't be paged out — the same root cause as this repo's
known hang case.**

The whole reason `training/sysmem.py` exists is that mmap file-cache pages
overflowing the working set caused the whole machine to hang while paging
(see the `mmap_working_set_paging_freeze` case). Pinned memory is even more
rigid than that:

- `trim_working_set()` is **completely ineffective** on pinned pages — the
  definition of page-locking is precisely that it cannot be reclaimed.
- `check_load_budget()`'s existing RAM budget counts a weight file's size
  as a "reclaimable mmap peak," but pinned memory is a **permanent
  allocation** — the same byte count carries a different risk level, and
  the existing guardrail's semantics **don't cover** this.
- Windows has a system-level cap on total lockable physical memory;
  allocation failure is a hard error, and there must be a fallback path.

This is this approach's **biggest real risk**, far greater than any
hardware concern.

**② PCIe link errors (correctable error / replay).**

Sustained full-bandwidth DMA can expose marginal quality issues in slot
contact, riser cables, or motherboard trace routing. This manifests as link
replay retransmission → effective bandwidth quietly drops, without raising
an error. NVML exposes `PcieReplayCounter`, and **the probe must sample its
delta** as the link-health criterion. If this machine's link is already
running in a degraded mode (x8 instead of x16, or through a chipset lane
rather than direct CPU attachment), bandwidth would be halved — this must
also be measured up front.

**③ Memory bandwidth contention and thermals.**

Sustained DMA occupies system memory bandwidth, competing with the
dataloader / tagging process. PCH and GPU board-edge temperatures will rise,
but stay within spec. The probe samples temperature and power to confirm
nothing abnormal.

### 3.2b Startup preflight check (`training/block_swap_preflight.py`)

Both ①/② guardrails trigger at **load time**: `check_load_budget` budgets
at the moment weights land on the GPU, and `check_pinned_budget` gates at
the moment a swapped-out layer is about to land in pinned memory. They
prevent incidents, but they don't answer the user's real question — **what
should `blocks_to_swap` actually be set to**.

`blocks_to_swap` defaults to 0, and Krea 2's DiT is 13GB even at fp8: on a
12GB card, "select a model and just start" would inevitably OOM, and that
OOM would happen only after the dataset scan, latent caching, and text
encoding have all already run — the error message is just a bare
`CUDA out of memory`.

The preflight check merges both sides' arithmetic and moves it earlier, to
`models.run` (the 2nd phase, right after bootstrap, before any weights are
loaded), and **gives a recommended value**:

- The VRAM side reuses `check_load_budget`'s arithmetic
  (`file_size × (1 - swap_fraction) + _VRAM_BASE_BYTES`), so the two
  guardrails don't end up in a "preflight says OK, loading rejects it"
  situation;
- The RAM side reuses `pinned_safe_limit()` — sharing the same function with
  `check_pinned_budget`, since writing this logic twice would drift sooner
  or later;
- The recommended value is the smallest swap-out count that **still leaves
  training headroom** (`_RECOMMEND_HEADROOM_BYTES`, wider than
  `_VRAM_BASE_BYTES`, covering LoRA + optimizer state + activations + fp8
  dequant temporary weights). If no such tier exists, it falls back to the
  minimum that "at least fits the weights," clearly flagged in the copy as
  tight — recommending a number that will still OOM without saying so is
  worse than not recommending anything.

Avoiding false rejections is a hard requirement: the toggle
`block_swap_preflight` (in the UI next to `blocks_to_swap`) can turn it off;
if the family has no `block_swap` capability bit, hasn't implemented
`swapped_param_ratio` / `swappable_blocks`, the weight file's size can't be
read, the VRAM query fails, or the preflight itself throws — it silently
lets the run proceed in every case. If the memory query fails, only the
VRAM side is checked — it never rejects the run based on a made-up "locked
memory exceeded" verdict.

### 3.3 Relationship to the WDDM VRAM cliff (a positive benefit)

The 190s hang recorded in PR #281 was a WDDM paging cliff near full load.
Block swap lowers the resident peak, **naturally staying away from the
cliff zone** — a net positive benefit here.

## 4. The inference side

The same logic applies here too, and the ecosystem is even more mature
(ComfyUI's `--lowvram` partial load is essentially this, at module rather
than block granularity). But the arithmetic differs:

- **Cheaper**: no activations, no optimizer state, no backward pass, purely
  one-way H2D.
- **More expensive**: diffusion is an N-step loop, and **every single step
  requires moving the entire model**. Training's one step = 2 passes;
  inference's 30 steps = 30 passes. Total transfer volume is amplified by
  the step count.
- **Higher leverage**: inference VRAM is almost entirely weights
  (activations are negligible at batch=1), so block swap directly decides
  "**whether it can run at all**," rather than "how comfortably it runs."

For this repo's specific situation: on 32GB, K2 Generate's TE+DiT in bf16
resident together exceed VRAM, currently solved by `_should_offload_te` /
`_should_yield_dit`'s model-level yielding. If a higher resolution or
multiple LoRAs are added later, the DiT's own residency becomes the next
bottleneck — at that point, DiT block swap is a natural extension of the
same tool.

## 5. Gate-0 probe and measured results

`tools/block_swap_probe.py`. **Must be run before writing any implementation
code.**

Note that Gate-0's meaning here is **not a falsification threshold**: block
swap is a deterministic trade of "time for VRAM" — there's no such thing as
"not worth doing," only "worth it for whom" (see §8.3 for details). The
probe's job is to **calibrate expectations** and check the hardware. The
only thing that still carries veto power is section F's link-health metric.

| Stage | What it measures | What question it answers |
|---|---|---|
| A link checkup | actual vs. maximum PCIe gen/width, GPU/RAM capacity, replay baseline | whether this machine's link is already degraded |
| B bandwidth matrix | pinned/pageable × H2D/D2H × multiple sizes; pin allocation time | measured effective bandwidth |
| C compute benchmark | forward/backward time of a real `SingleStreamBlock` at real shapes | `T_compute` |
| D hiding criterion | the B/C ratio → the `blocks_to_swap` × (VRAM saved, time added) curve | whether it's worthwhile |
| E end-to-end | wall-clock time for a real dual-stream swap loop vs. a fully-resident loop (**forward-only view**) | whether prefetch can actually hide the transfer |
| F stability | temperature/power/replay delta/available RAM under sustained load | measured data for §3.2's three risks |
| G training view | a full step with checkpoint + reverse-order backward prefetch, interleaved A/B comparison | how much slower training actually is (B10) |

### 5.1 Measured data (2026-07-21, RTX 5090 32GB / PCIe 4.0 x16 / 37GB available RAM)

**A link**: gen4 x16 at full spec (dropping to gen1 while idle to save power
is normal), replay baseline 0.

**B bandwidth**: pinned H2D **26.8 GB/s** (consistent across sizes, saturated
starting at 16MB), pageable 18.5-23 GB/s, **pinned is 1.45x faster**. Pinned
allocation takes **59 ms/GB** — implementation must pre-allocate and reuse,
not allocate per step.

**C scale and compute** (1024², batch 1, bf16, seq_len 4608 = text 512 +
image 4096):

| Quantity | Value |
|---|---|
| single block | 434.2M params / **828 MB** |
| 28-layer total | **22.64 GB** (+ txtfusion/embed/last ≈ 25.8GB total, matching D2's record) |
| forward | 29.1 ms |
| forward+backward | 103.7 ms (backward portion ≈ 74.5 ms) |

**D hiding criterion**: `T_transfer` = 828MB ÷ 26.8GB/s = **30.2 ms**.

| View | Transfer/compute ratio | Verdict |
|---|---|---|
| forward (inference) | **1.04** | marginal, each block exposes 1.1 ms |
| backward portion (training) | **0.40** | fully hidden |

→ In the training view, the **exposed** portion is only 1.1%. But this isn't
the whole cost — see the contention term found in section E below.

**E end-to-end** (forward-only view = the worst case, double buffer +
independent copy stream):

| Resolution | per_tensor | flat (contiguous buffer) |
|---|---|---|
| 1024² (ratio 1.04) | +18.8% / +19.3% | +20.5% / +16.6% |
| 1536² (ratio ~0.42) | +18.5% | **+11.4%** |

Three conclusions:

- **§2.2's resolution corollary is validated** — the more headroom in the
  ratio, the lower the overhead (1024² about 19% → 1536² about 11%).
- There's **an extra cost outside the theoretical model**; after subtracting
  the exposed portion, there's still: about 4.4 ms/block at 1024², about
  8.2 ms/block at 1536². **It grows with activation size, so it isn't a
  "fixed overhead" — it's the copy stream's DMA writes contending with the
  compute kernel for HBM bandwidth** (event sync only accounts for a few
  tens of μs of it). This is the actual source of the gap between the
  theoretical value (3.5%) and the measured one (19%).
- Flattening into a contiguous buffer only clearly wins when the ratio has
  headroom (1536²); at the marginal ratio it's indistinguishable from
  per-tensor copying, or even mixed results (within noise). **"Flattening
  into a contiguous buffer" isn't a cure-all.**

**G training-view end-to-end measurement** (B10; gradient checkpointing +
reverse-order backward prefetch, frozen base model = the LoRA scenario. The
baseline is fully-resident with the same checkpoint semantics):

| Resolution | baseline (jitter) | swap | overhead | extra per block |
|---|---|---|---|---|
| 1024² 1st run | 920.3 ms (±0.6%) | 982.7 ms | **+6.8%** | 7.80 ms |
| 1024² 2nd run | 912.8 ms (±1.0%) | 983.3 ms | **+7.7%** | 8.81 ms |
| 1536² | 1963.0 ms (±0.3%) | 2013.9 ms | **+2.6%** | 8.48 ms |

Three conclusions:

1. **Measured training-view overhead is about +7% at 1024² and +2.6% at
   1536²**, better than the earlier estimate of 9.5%. The baseline's own
   jitter is only ±0.3-1.0%, so the numbers are trustworthy.
2. **The extra per-block time is a roughly constant 8 ms** (7.80 / 8.81 /
   8.48), **independent of resolution**; the percentage dropping at 1536²
   is purely because the baseline compute volume grows (2.25x), diluting it.
   This corrects the "grows with activation size" judgment made from
   §5.1-E's forward-only view — that was an artifact of the forward-only
   view; in the training view, it's a constant. It also means **the overhead
   ratio stays the same when extrapolated to the full 28 layers** (constant
   per block).
3. Each residency window is about 4 ms (8ms ÷ 2 windows), corroborating
   section E's forward-only measurement of 4.4ms/window at 1024².

> **Methodological lesson**: the first version of this section measured
> "baseline first, then swap" sequentially, yielding an absurd **-2.2%**
> negative overhead — the GPU's clock state differed between the two runs
> (the one run first was still cold, not yet boosted). Switching to
> **interleaved A/B** (both paths kept resident simultaneously, timed in
> alternating rounds) produced reproducible numbers. Any measurement result
> of "A is faster than B, but that's physically impossible" should first be
> suspected of an ordering effect.

**F hardware impact** (60s of sustained load, 213 rounds × 8 blocks =
**1704 swap-ins ≈ 1.4 TB transferred**):

| Metric | Result | Verdict |
|---|---|---|
| PCIe replay delta | **0** | zero link retransmissions, §3.2 ② passes |
| GPU temperature | max 70°C / mean 64°C | normal, no thermal stress |
| GPU power | max 497W / mean 484W | same order of magnitude as normal full-load training, no abnormal spikes |
| Available-memory drift | −99 MB | noise-level, no pinned-memory leak |

**→ All of §3's hardware-wear concerns are ruled out**: zero link errors, no
abnormal temperature or power, no memory drift. The only remaining real risk
is still §3.2 ①'s pinned-memory budget (only 6.5GB was pinned this run,
nowhere near the cap).

### 5.2 Capacity conclusions derived from the measured data

K2 DiT's total bf16 size ≈ 25.8GB, of which 22.64GB is the 28 swappable
blocks:

| blocks_to_swap | resident VRAM | measured training overhead (1024² / 1536²) | significance |
|---|---|---|---|
| 0 (current) | 25.8 GB | 0% | 32GB has ≈0 headroom (Phase 0 measurement) |
| 14 | 14.5 GB | ≈ 3.5% / 1.3% | plenty of headroom on 32GB; **24GB becomes feasible** |
| 28 | 3.2 GB | ≈ 7% / 2.6% | **16GB theoretically feasible** |

(The extra per-block time is constant, so overhead scales linearly with
`blocks_to_swap`, independent of the total layer count.)

In other words: block swap pulls K2 training's VRAM floor down from 32GB to
24GB or even lower, while the training-view time cost is in the single-digit
percentage range — this is exactly the second path D2's "the only unlock
for going below 24GB in the future is fp8_scaled" hadn't accounted for at
the time, and it **costs no precision**.

### 5.3 Known limitations of the probe

1. ~~Section E only measures forward pass; the training view hasn't been
   measured end-to-end~~ — **now filled in by section G** (B10). Section G
   measures timing, not numerical correctness: buffer weights get
   overwritten by rotation, making the gradients meaningless; and LoRA
   parameters' compute (resident, not participating in swap) isn't counted —
   negligible but not zero relative to the base model.
2. Only covers krea2. Anima's Block needs a string of pre-built tensors for
   rope/adaln_lora, making it expensive to construct with low payoff (24GB
   is already sufficient) — not included.
3. Behavior when pinned memory approaches the system's limit hasn't been
   measured (§3.2 ①'s actual risk surface).
4. Coexistence with `compile_blocks` hasn't been measured — but per B5,
   there's currently no implementation of it, so it's not a blocker.

## 6. Implementation landing point (if Gate-0 passes)

Per B3, the mechanism must be **family-agnostic from the start**: wrapping
an `nn.ModuleList`, providing a prefetch iterator from outside, without
requiring changes to the model's internal code (reasoning in §7.1).

- Krea2 (first): the block loop in
  `modeling/krea2/krea2_modeling.py` `SingleStreamDiT.forward:513`;
  `01-code-layout.md:189` has already reserved an "fp8/block-swap fallback
  hook" for `families/krea2/loader.py`. **This file has a byte-for-byte
  parity requirement with ComfyUI, so changes must be minimal.**
- Anima (later): swap out one line for an iterator in the block loop in
  [`forward.py:31`](../../runtime/training/families/anima/forward.py).
- Inference side (concurrent with B4): `runtime/anima_daemon.py`'s model
  stack, layered together with the existing `_should_yield_dit` /
  `_should_offload_te` model-level yielding — block swap manages inside a
  single model, `vram_policy` manages between models.
- Guardrails: `training/sysmem.py` needs a new **pinned-specific budget**
  (§3.2 ① / §8.1), and can't reuse `check_load_budget`'s mmap semantics
  (which assumes memory is reclaimable — pinned memory isn't).
- Config: `blocks_to_swap` field metadata defined in one place (enforced on
  both ends via `config_rules.py`, see `config-pipeline-refactor.md`). The
  field name has no unit-suffix issue ("blocks" is itself the unit).
- UI: per B9, gives hints only, no recommended numbers.

## 7. Decision (2026-07-21, user's first round of ruling)

| # | Decision | Notes |
|---|---|---|
| **B2** | The knob = **an integer `blocks_to_swap`**, no `vram_policy` tier | consistent with the musubi/ai-toolkit/diffusion-pipe ecosystem; user control beats automatic estimation |
| **B3** | **Ship K2 first, validate maturity, then add Anima** | conditional on migration cost being manageable, per §7.1's determination: both families' block-loop structures are fully identical, the mechanism is designed family-agnostic from the start, so Anima later = just wiring |
| **B4** | **The inference side is done at the same time** | §4's leverage is higher (it decides "can it run at all" rather than "how fast") |
| **B7** | **fp8 + block swap, one and the same effort** | the clear goal = pulling K2's fp8 training floor down to **12-16GB**; §2.2 already established the positive synergy (fp8 halves transfer bytes without halving compute, giving a wider ratio) |
| **B9** | The upper limit **gives hints only, no precise recommended number** | user's ruling: VRAM, PCIe generation, and DRAM speed differ too much across GPUs, and a precise number would mislead |
| **B5** | The `compile_blocks` conflict — **doesn't currently exist, not designed for** | verified: the main line only has a capability bit (a placeholder entry in `FAMILY_CAPABILITIES` / `KNOWN_CAPABILITIES`'s vocabulary), **no config field, no `torch.compile` code**; the implementation lives on the unmerged `pr257-review` branch. Revisit if #257 merges first |

### 7.1 B3's migration-cost determination (supporting "K2 first, then Anima")

The two families' block-loop structures are **fully isomorphic**:

```
Anima  runtime/training/families/anima/forward.py:31   for block in model.blocks:  x = block(x, ...)
Krea2  modeling/krea2/krea2_modeling.py:513            for block in self.blocks:   h = block(h, ...)
```

As long as the mechanism is abstracted into a family-agnostic component
("wrap an `nn.ModuleList` + provide a prefetch iterator," rather than
hardcoding the krea2 type), adding Anima later = swapping in one line for an
iterator in its loop — a few dozen lines plus tests.

**The one asymmetry that needs to be handled at design time**: both
families' structural definitions live in `modeling/<family>/` (layered by
architecture, per `01-code-layout.md` §2.1: `modeling` = structural
definition → `runtime/training/families` = behavior adaptation →
`studio/services/models/families` = asset manifest, a one-way dependency),
but **the per-block loop itself lives in different places**:

| Family | Where the block loop lives | Why |
|---|---|---|
| Krea2 | `modeling/krea2/krea2_modeling.py:513`, **inside the structural-definition layer**; `use_checkpoint` is a parameter native to the model | this file was written by us following ComfyUI's naming, so the switch can be built in directly |
| Anima | `runtime/training/families/anima/forward.py:31`, **the behavior-adaptation layer**, a hand-unrolled rewrite of the model's internal API forward pass | `modeling/anima/cosmos_predict2_modeling.py` is a ported external Cosmos backbone (2068 lines) whose `forward` provides no checkpoint switch, and needs to stay comparable with upstream — so it's unrolled outside rather than modified in place |

So the component must be able to wrap an `nn.ModuleList` **from outside**
without requiring changes to the model's internal code — this way, the
Krea2 side never has to touch the parity-sensitive `modeling/` files, and
the Anima side never has to duplicate the logic (the latter being exactly
the per-family fan-out that `02-ecosystem-survey.md` §7 criticized
SimpleTuner for).

## 8. Second-round ruling and remaining questions

| # | Decision | Notes |
|---|---|---|
| **B6** | Pinned allocation failure = **raise an error, don't silently degrade** | user's ruling: since the failure timing is deterministic (§8.1: it can only happen at the startup-time allocation), it should raise an explicit error. Lands as a DomainError + actionable copy, matching `check_load_budget`'s two existing guardrails |
| **B8** | The HBM contention term **isn't squeezed further** | see §8.2's three reasons; solved incidentally by B7's fp8 halving the transfer volume |
| **B10** | **Next step = measure the training-view end-to-end first** | user's ruling. **Done** — the probe's section G, measured +7% at 1024² / +2.6% at 1536² (§5.1 G) |
| **B1'** | **Gate-0's threshold carries no veto semantics** | see §8.3 — this isn't a bet on "is it worth doing," it's a deterministic trade of "time for whether it can run at all" |

### 8.3 What the threshold number actually decides (where B1' comes from)

User's question: "What is this threshold number actually deciding — if it
goes over that time, do we just not do it?" — this question exposed that
Gate-0's semantics had been mapped onto the wrong template.

The LPL project's Gate-0 was a **falsification threshold**: a new
algorithm's effectiveness was unknown, and if it didn't meet the bar, it had
no reason to exist and should be parked. Block swap isn't that kind of
thing — it's a **deterministic trade**: take away some time, get back some
VRAM, both sides being measurable known quantities. For different users,
this trade's value is wildly different:

| User's VRAM | off | on (~10% slower) | what the threshold means |
|---|---|---|---|
| 32GB+ | trains fine | pure loss | default **off** |
| 24GB | **can't run at all** | can run | must be on even at 30% slower |
| 12-16GB (the B7 target) | **can't run at all** | can run | same as above |

**For a user who can't run at all, any percentage is better than "can't
run."** So exceeding the threshold shouldn't mean "don't do it."

What the threshold number should actually decide is only two things, and
neither involves a veto:

1. **The default value and auto-suggestions**: if training overhead stably
   stays < 15%, it can be actively suggested when VRAM is insufficient;
   if > 30%, it only takes effect when the user explicitly turns it on,
   never actively recommended.
2. **The strength of the UI's wording** (given B9's "hints only, no
   numbers"): below the threshold, say "will slow down slightly"; above it,
   say "will slow down noticeably."

So §5's earlier wording, "the approach is directly rejected if it doesn't
meet the threshold," has been voided, and is now used only for calibrating
expectations. **The only thing that still carries veto power is section F's
hardware-health metrics** (sustained replay growth = a link problem, which
genuinely warrants stopping).

### 8.1 Timing and recovery for pinned allocation failure (answering B6's prerequisite question)

- **The failure happens at the moment of allocation, not randomly mid-run**
  — `cudaHostAlloc` either gets page-locked memory or immediately returns
  `cudaErrorMemoryAllocation` (PyTorch raises a `RuntimeError`). Once
  allocation succeeds, that memory is locked and owned by this process for
  good — it won't be reclaimed mid-run, and won't "disappear while in use."
- **But the same config can succeed this time and fail next time**, because
  success depends on the system's available physical memory at the moment
  of allocation (another training/tagging process, or even the browser, can
  affect it). This is the same kind of uncertainty `check_load_budget`
  already deals with.
- **This leads to an implementation discipline**: **pre-allocate all pinned
  buffers up front at startup**, don't allocate on demand. This way,
  failure can only happen during the training-startup phase — predictable,
  fail-fast, and able to give a clear error. The cost is startup taking
  roughly `59ms × GB` longer (§5.1 B); at 22.6GB that's about 1.3 seconds,
  acceptable.
- **Automatic recovery**: technically possible (fall back to pageable, 1.45x
  slower, or reduce `blocks_to_swap` and retry). But silently degrading
  would leave the user with a training run that's "mysteriously about half
  as fast" for no apparent reason — conflicting with the
  `feedback_no_silent_magic_protection` discipline. Leaning toward: **an
  explicit DomainError failure + error copy with actionable suggestions**
  (close other memory-hungry applications / reduce `blocks_to_swap`),
  matching `check_load_budget`'s existing two guardrails' copy. Pending the
  user's final call.

### 8.2 Whether to keep squeezing the HBM contention term (answering B8)

**Relative impact** (1024², full 28-layer swap): per-step time goes from
2.90s → 3.18s. A 2000-step training run goes from about 97 minutes to about
106 minutes, **9 minutes more**, in exchange for 22.6GB of VRAM and "a 24GB
card can train K2."

**Recommendation: don't squeeze further**, for three reasons:

1. The physical source of the contention is DMA writes competing with the
   compute kernel for HBM bandwidth — **it's not overhead that scheduling
   can eliminate**. Adding buffers (3+ rotation) addresses jitter, not
   contention, and each buffer costs an extra 828MB — directly eating into
   the benefit. Copy-stream priority affects SM scheduling, but has no
   effect on the DMA engine. The two cheapest tricks are both likely
   ineffective.
2. The only genuinely effective direction is **reducing transfer bytes**,
   and that's exactly what B7 has already decided to do with fp8 — halving
   the transfer volume drops contention proportionally. **Solved
   incidentally, no need to open a separate optimization track.**
3. Asymmetric maintenance cost: overlapping chunked transfers would turn
   swap from "a single `copy_`" into a state machine, and would also need
   to handle interaction with checkpoint recomputation. The complexity jump
   is a step-change, while the payoff ceiling is only a few percentage
   points.

### 8.4 Third-round ruling (moving into implementation)

| # | Decision | Notes |
|---|---|---|
| **B11** | **Default value = 0 (off)**, no auto-suggestion for now | user's ruling: decide the default policy after the real implementation has actually trained a LoRA and there's real-world experience. The threshold number is likewise deferred (B1' already removed its veto semantics, and now even "setting the default" is deferred too) |
| **B12** | Project goal = **fp8 + swap running K2 LoRA training stably on 16GB / 12GB consumer cards** | the user's explicit maximum expectation. This is the acceptance criterion, not just "it runs" — emphasizing **stability** |

## 9. Implementation design

### 9.1 Must swap `param.data` in place, can't use buffer rotation (the key difference between the probe and the implementation)

Probe sections E/G used "2 reserved block instances as a rotating buffer" to
measure timing. **The real implementation can't do this**:

LyCORIS's `apply_to()` ([`utils/lycoris_adapter.py:157`](../../utils/lycoris_adapter.py))
creates LoRA modules that **hold a reference to the original block's Linear
layers** and wrap its forward (in bypass mode, it's
`org_forward(x) + lora_up(lora_down(x))`). If the forward pass runs against
a buffer instance instead, it would **completely bypass LoRA** — training
would silently learn nothing.

So the implementation takes an **in-place swap** approach: the module
object itself never changes; only each parameter's `.data` pointer is
switched.

| | buffer rotation (used by the probe) | in-place `param.data` swap (used by the implementation) |
|---|---|---|
| LoRA compatibility | **broken** (forward bypasses the LoRA module) | compatible (module identity unchanged) |
| fp8 `weight_scale` | **mismatched** (a non-persistent buffer bound to the module, doesn't rotate with the weights) | **automatically correct** (module never changes, scale stays paired) |
| timing characteristics | equivalent to the implementation (same transfer volume, same sync pattern) | — |

The performance numbers measured by the probe remain valid (timing is
equivalent), but **the code shape can't be copied as-is**.

Worth noting in passing: fp8's `weight_scale` is registered by
`patch_fp8_linears` as a **non-persistent buffer** (not part of
`state_dict`). Any design that moves things around based on `state_dict()`
would miss it — the in-place swap naturally avoids this pitfall, but if
anyone ever switches back to a state_dict-based approach, this is the first
thing they'd trip on.

### 9.2 Cut breakdown

1. ✅ **Cut 1 (core)**: `PinnedBlockSwap` in `runtime/training/block_swap.py`
   + unit tests.
2. ✅ **Cut 2 (K2 wiring)**: capability bit + `blocks_to_swap` field
   (default 0) + pinned-budget guardrail + loader/family/phases wiring.
3. 🔄 **Cut 3 (fp8 stacking + 12/16GB acceptance)**: B7/B12. The code side is
   done (fp8 combination validated end-to-end, budget discount, observability),
   **real-hardware acceptance pending the user's 16GB/12GB card**.
4. ✅ **Cut 4 (inference-side wiring)**: the B4 debt has been repaid, see §9.6.

Anima wiring is deferred until after K2 has been validated as mature, per B3.

### 9.5 The VRAM budget must discount the swapped-out portion (the real bug cut 3 fixed)

`check_load_budget` budgets VRAM using the weight file's full size. With
swap on, the DiT doesn't fully load onto the GPU, and **a 16GB card setting
`blocks_to_swap=28` would be falsely rejected by this guardrail as "the
full 25.8GB model doesn't fit,"** while the actual resident amount is only
3.2GB — B12's goal would be blocked by its own guardrail.

`vram_discount_bytes` has been added: it **only discounts the VRAM side**,
the RAM side is computed as before (swapped-out layers still occupy memory,
and it's locked memory, separately gated by `check_pinned_budget`). The
discount amount is reported by the family
(`Krea2Family.estimate_swapped_bytes` → the meta model's parameter count,
no disk reads, no VRAM used): 14 layers → 11.32GB, 28 layers → 22.64GB,
matching §5.1's probe measurements.

### 9.5b First real-hardware run (2026-07-21, 5090 32GB, fp8 base model + `blocks_to_swap=14`)

| Point in time | torch alloc | reserved | whole card |
|---|---|---|---|
| after DiT load | 6.82 GB | 6.92 GB | 8.62 GB |
| after swap attaches (LoRA already injected) | 7.85 GB | 7.96 GB | 9.67 GB |
| **before** sampling (= state after the training step finishes) | 8.9 GB | **16.0-16.4 GB** | **19.4-19.9 GB** |
| **after** sampling (after `empty_cache`) | 8.9 GB | 9.2 GB | 12.7-12.8 GB |

Reading notes:

1. **The peak comes from the training step, not sampling.** The "before
   sampling" reading was taken before `empty_cache`, reflecting the
   allocator state accumulated by the training step; sampling itself is
   cheap (TE has already been released, DiT is already swapped out,
   sampling uses 12.8GB). This contradicts the earlier intuition based on
   the forward-only view (which assumed sampling was the spike).
2. **What decides "is this card enough" is `alloc`, not `reserved`.**
   `alloc` holds steady at 8.9GB throughout, while `reserved` climbs to
   16.4GB — the 7.5GB gap is allocator-reserved segments from multiple
   bucket shapes. On a 32GB card, the allocator happily holds onto extra
   memory; on a smaller card it would reclaim and reuse it, not OOM because
   of this. **The real requirement ≈ alloc 8.9GB + CUDA context ≈ 10.4GB**
   → 16GB has headroom, 12GB needs `blocks_to_swap` pushed higher.
3. The numbers are self-consistent: `pinned 5.66GB` is the real fp8 value
   (half of bf16); after attaching, alloc +1.03GB = 264 LoRA modules + 2 GPU
   slots (0.4GB per fp8 block).
4. **This run caught a directional bug in §9.5's discount** (discounting by
   dtype byte count broke through the guardrail's discount under fp8),
   which has been changed to discount by parameter ratio instead.

### 9.5c Full swap-out measurement (same machine, fp8 base model + `blocks_to_swap=28`) — B12 achieved

| Reading point | 14 layers | **28 layers** |
|---|---|---|
| alloc after DiT load | 6.82 GB | **1.16 GB** |
| alloc before training starts | 7.85 GB | **2.19 GB** |
| alloc at steady state | 8.9 GB | **2.8 GB** |
| **training step peak** | — | **8.4 GB** |
| **sampling-period peak** | — | **7.1 GB** |
| whole card (before sampling) | 19.9 GB | **13.4 GB** |
| pinned | 5.66 GB | **11.32 GB** |

**Capacity verdict** (this machine's CUDA context ≈ 1.7GB, derived from the
static point "whole card 3.98 − reserved 2.27"):

> Minimum requirement ≈ training step peak 8.4GB + context 1.7GB ≈
> **10.1 GB**
> → **comfortable on 16GB (5.9GB to spare), feasible on 12GB (1.9GB to spare)**

The B12 target, extrapolated from the 32GB run, has been achieved, pending
confirmation on smaller cards.

Two side conclusions:

1. **The training step is the bottleneck, not sampling** — this run has
   actual peak data (8.4GB vs. 7.1GB), no longer inferred from a reserved
   snapshot. §9.5b already overturned the "sampling is the spike" intuition,
   and this round confirms it with actual peaks.
2. **The speed cost is nearly invisible**: for the same sampling task, 14
   layers took 52-53s, 28 layers took 53-55s — **doubling the swap-out
   layer count only slowed things by about 4%**, better than the probe's
   training-view +7% (and the probe used bf16, whose transfer volume is
   double fp8's).

Note that `reserved` (10.2GB) is higher than the peak `alloc` (8.4GB): on a
32GB card, the allocator happily holds onto extra memory, while a smaller
card would actively reclaim it — a smaller card's measured value should be
closer to 8.4GB than to 13.4GB.

#### 9.5c-1 Real-hardware whole-card reading (image generation, added 2026-07-21)

Same machine, same config (fp8 + `blocks_to_swap=28`, 1024²), using the
**Task Manager** to check whole-card usage during generation: **8.9-9.3
GB**, of which about **3 GB** belongs to other applications open at the same
time — about **6 GB** belongs to this program.

This differs from the torch-based readings in the table above; both are
kept, don't mix them up:

| View | value during sampling | notes |
|---|---|---|
| `max_memory_allocated` (table above) | 7.1 GB + context 1.7 GB ≈ **8.8 GB** | doesn't miss transient spikes, on the conservative side |
| Task Manager whole-card (this section) | **≈ 6 GB** (after subtracting other apps' baseline) | a sampled reading, might miss millisecond-scale spikes |

The ≈2.8GB gap hasn't been pinned down yet (candidates: Task Manager's
sampling missing a peak, how `reserved` is counted against the whole card,
memory reuse between the baseline apps and this process). **The public-facing
number uses the whole-card reading** (that's what the user actually sees) —
the announcement and README say "generation can run on an 8GB card"; for
scenarios genuinely at the 8GB boundary, the torch peak is the safer number
to rely on. **Testing on smaller cards is still owed** (same as noted at the
end of §9.5c), and will be settled once real 8/12GB cards are tested.

### 9.6 Inference-side wiring (cut 4, done)

B4 ruled that "the inference side is done at the same time," and §6 already
noted a landing point, but §9.2's cut breakdown only split the training side
into three cuts, missing the inference side. The component is
family-agnostic, `attach()` is generic too, and `load_dit` already carries a
`blocks_to_swap` parameter (`purpose="generate"` goes through the same
loader), so the wiring cost is low. **But the inference side's LoRA
semantics differ from the training side, and this must be handled before
wiring**:

| Base model | Inference-side LoRA approach | Relationship with swap |
|---|---|---|
| bf16 | a lycoris hook (adapter), doesn't modify weights | isomorphic to the training side, usable directly |
| fp8 | **merged into the weights** (ComfyUI semantics: dequant → add delta → stochastic rounding write-back) | merging writes to `module.weight`, but the swapped-out layer is at this moment a CPU pinned tensor |

The fp8 path is correct as long as the **order is right**: the loader lands
into CPU pinned memory → `apply_loras` merges and writes into the master
copy → constructing `PinnedBlockSwap` takes over in place → subsequent
swap-ins are then of the merged weights. This requires fp8 to be computable
on CPU — already verified: `pin_memory` / bit-exact H2D / `to(bf16)` dequant
/ `fp32→fp8` write-back all work.

Also worth noting: inference has to move the entire model on every single
step (§4), so 30 steps = 30 passes; the swap object stays resident with the
daemon's model cache across steps and is reused, not rebuilt every step.

**Implementation conclusion**: the daemon was written assuming "the entire
model stays resident on the GPU," which creates three adversarial
interactions the training side doesn't have, all of which have been
handled:

| Interaction | Consequence | Handling |
|---|---|---|
| `_move_runtime_to_device`'s blanket `module.to(device)` | moves even the CPU pinned master copy onto the GPU → swap is wasted, and the instantaneous usage = the full model (OOM on smaller cards) | `move_module_excluding` skips managed tensors. Measured: both timings (right after load / after running a forward pass) result in zero extra VRAM usage |
| fp8 LoRA merge writing to `module.weight` | after a forward pass has run, `.data` points at a GPU slot that gets overwritten by the next layer → the merge is **silently lost** | `apply_loras` starts with `restore_masters()`, so the delta lands on the master copy |
| `unload` not releasing pinned memory | page-locked memory can't be reclaimed by anything except GC → a persistent leak of 11GB+ | `unload` now does `detach()` + clears references |

`blocks_to_swap` is included in `ModelCache`'s identity comparison (which
layers get swapped out is decided at loader time, so changing it requires
reloading the DiT); the VRAM budget uses the same proportional discount as
the training side.

Implementation gotcha: **tensor-identity comparisons must use `data_ptr()`,
not `id()`** — every access to `param.data` returns a new Python wrapper
object, so comparing by `id` would miss every match (caught directly by
tests).

Another easily-missed semantic worth noting: **a swapped-out layer's weights
are only valid within its own forward window**. Once a pass finishes, the
slot its `.data` pointed at has already been overwritten by a later layer
(rel 0 and rel 2 share slot 0). Reading or modifying weights outside that
window requires calling `restore_masters()` first. This is now documented
in the module's docstring + a dedicated test.

### 9.3 Wiring approach: forward hooks (cut 2's implementation conclusion, better than originally planned)

The original plan was "insert a prefetch hook inside the family's forward
loop." What actually got implemented instead is **registering a forward
pre/post hook on each swapped-out block**
(`PinnedBlockSwap.attach()`), which is better:

- **Doesn't touch `modeling/krea2/` at all** — that has a byte-for-byte
  parity requirement with ComfyUI (§7.1).
- ~~**Backward automatically works**: checkpoint recompute triggers the
  forward hook, so reverse-order swap-in automatically works~~ — **this
  claim is wrong, corrected in §9.10**. Backward requires its own hooks to
  retrieve the weights.
- Wiring Anima is therefore also just "construct + attach," without even
  needing to touch its hand-unrolled loop.

### 9.4 Three pitfalls caught during implementation testing (all have regression tests now)

1. **`_rebind` can't iterate `named_parameters()`**: parameters added after
   construction (the case where LoRA builds a submodule inside the block)
   would make `buf[name]` raise a bare KeyError. It can only rebind names
   registered at construction time.
2. **Trainable parameters must not be swapped out**: LoRA parameters are the
   optimizer's target, and moving them away would break training; also
   they're tiny relative to the base model, with no value in swapping them
   out. The component skips anything with `requires_grad`, managing only
   the frozen base weights.
3. **fp8 `weight_scale`'s device**: `patch_fp8_linears` originally made the
   scale follow `module.weight.device`, but a swapped-out layer's weights
   were on CPU at patch time → the scale ended up on CPU, while at forward
   time the weights have already been moved to GPU, causing a device
   mismatch in `weight * scale` (`scale.to()` only changes dtype, not
   device). A `device` parameter has been added, keeping the scale always on
   the compute device (a per-layer scalar, negligible overhead).

One more thing already avoided in the design, but worth noting: **the loader
must load the last N layers directly into CPU pinned memory**, rather than
"loading everything onto the GPU first and then moving it down" — the
latter's peak would still equal the full model's size, and B12's 12/16GB
target wouldn't hold.

### 9.7 Returning pinned memory (a real leak, now fixed)

**Dropping a reference doesn't mean the memory is returned.** Pinned memory
goes through PyTorch's independent host caching allocator, and releasing a
tensor just returns it to that cache pool:

| Operation | System available memory |
|---|---|
| pin 6GB | −8.18 GB |
| `del` + `gc.collect()` | **returns 0.00 GB** |
| `torch._C._host_emptyCache()` | returns 8.02 GB |

Cut 4's first version only did `detach()` + set to None inside `unload()`,
with a comment that even claimed "the pinned master copy is released along
with it" — **that comment is wrong; not a single byte was returned.** Block
swap's master copy can be 11GB+, meaning that after unloading, that memory
is still held long-term, and since page-locked memory can't even be paged
out, other programs can't use it at all.

This is **the host-side version of the same class of problem** as
`_cuda_clearCublasWorkspaces` in the same function: both are residency at
the C++/allocator layer, invisible to Python's GC.

**Cleanup timing: tied to unloading, not after every generation.** When
`blocks_to_swap > 0`, what's inside pinned memory **is the model weights
themselves**, not a temporary buffer:

- **Between generations (the model is still loaded)**: must never be
  cleared — clearing it would be equivalent to unloading the model, and the
  next generation would have to re-read from disk + re-pin (`59ms/GB`,
  11.32GB ≈ 0.7s, plus the disk read).
- **On unload (idle timeout / manual "clear VRAM")**: must be returned along
  with VRAM. Both paths go through `CACHE.unload()`, where this is now
  handled uniformly.

That is: **cleared together with VRAM, never cleared separately.** The
existing "auto-clear on idle / manual clear" semantics were already
naturally correct — they just previously failed to return the host-side
half. Only cleared when block swap was actually used (gated by
`had_block_swap`) — only returning what it allocated itself, without
touching pinned caches elsewhere (e.g. the dataloader's).

The training side doesn't need equivalent handling: training is an
independent child process, and the OS reclaims memory once it exits.

### 9.8 Answers to three "training-side memory" questions

**Q: Is training's memory released?** Previously **no** — training is an
independent child process, relying on OS reclamation at exit. But there's a
tail before exit: `finalize` uploads the final LoRA to wandb (which can take
tens of seconds to minutes), and by that point `eval_training_finished` has
already fired, the supervisor has already queued the post-training
evaluation job, and the evaluation process needs to load a model while this
process is still holding onto 11GB+ of page-locked memory (which can't even
be paged out). On a memory-constrained machine (exactly the kind block swap
is meant to serve), this would tank the evaluation. `finalize` now
proactively releases it.

**Q: Does training's sampling go through swap?** **Yes.**
`sample_image(ctx.model, ...)` takes the exact same model with hooks
attached, and the forward pass triggers swap-ins as usual. Corroborated by
measurement: with 28 layers swapped out, the sampling-period peak is only
7.1GB — if sampling didn't go through swap, the complete fp8 DiT
(≈12.9GB) would need to stay resident, and the peak couldn't be lower than
that.

**Q: Is training's sampling swap released?** Sampling and training **share
the same swap object**, so there's nothing extra to release, and training
still needs to continue, so it **can't** be released anyway. The
`empty_cache()` calls before/after `sample_runner` only reclaim device-side
caches — swap's GPU slots are live references and are unaffected — which is
the correct behavior.

### 9.9 `close()` and `release()` are two different things (a naming trap)

The component has two methods with completely different semantics, and
implementation hit a collision here (a new method overwrote an old one, and
17 tests went red at once):

| Method | Semantics | When to call it |
|---|---|---|
| `release(absolute_index)` | a layer has finished computing, its GPU slot can be overwritten by a later layer | after every layer's forward pass (inside the post-hook) |
| `close()` | letting go entirely: removes hooks + points parameters at empty tensors + discards the master copy and slots | at the end of training / on model unload; **the model is unusable afterward** |

The reason for `close()` rather than "just dropping the model reference":
the pinned master copy is referenced by `param.data`, and more than just
`ctx.model` holds a block — the LyCORIS injector holds `org_module`, the
optimizer holds the parameters, and hook closures might also hold
references. Real-hardware testing confirmed that dropping just the swap
object returns **0 bytes**; only dropping the model afterward returns
anything. Rather than hunting down every holder, it's simpler to have the
component redirect the parameters itself. After `close()`,
`release_pinned_host_cache()` is still needed to actually return the memory
to the OS (§9.7's other layer).

### 9.10 Backward needs its own separate hooks (overturning §9.3's wrong claim)

**Symptom**: a user enabled block swap + PPSF training, and Prodigy's `d`
estimate jumped wildly, the learning rate ran out of control, and training
failed.

**Root cause**: `attach()` originally only hooked forward pre/post, based on
a wrong claim — "with gradient checkpointing on, backward recomputes the
forward pass, which triggers the forward hook, so reverse-order swap-in
automatically works." Neither part of this held up under testing:

1. **Recompute doesn't trigger forward_hook**: measured that under
   checkpoint, the pre-hook fires 2N times while the post-hook only fires N
   times (recompute only goes through pre).
2. **More critically**: after recompute, this block's **backward pass**
   still needs to read the weights, but the forward post-hook has already
   released the slot — the next swap-in directly overwrites the weights
   that backward is in the middle of reading.

The consequence was **silent**: no error, no NaN, just numerically wrong
gradients. Quantitatively (RTX 5090, a real 6144-dim block):

| | noise floor | swap deviation | multiple |
|---|---|---|---|
| before fix · checkpoint | 4.4e-3 | **1.31** | **298x** |
| before fix · no checkpoint | 2.2e-3 | 1.48 | 675x |
| after fix · checkpoint | 4.4e-3 | 4.7e-3 | 1x |
| after fix · no checkpoint | 3.5e-3 | 4.7e-3 | 1.4x |

**PPSF is the alarm, not the culprit**: AdamW with a fixed lr would just
train through with poisoned gradients, producing a LoRA with degraded
quality but no obvious sign of trouble; Prodigy-family optimizers rely on
gradient-consistency estimation to compute `d`, so a messed-up gradient
makes `d` blow up immediately. **Any LoRA trained with AdamW + block swap
should be considered affected too.**

**Fix**: all four hooks are required — forward pre retrieves / post
releases, and **backward pre retrieves again / post releases**. When
checkpoint is on, backward pre usually lands on weights just placed there by
the recompute, with zero extra transfer; when checkpoint is off, it's the
only opportunity to swap the weights back in.

### 9.11 Why this bug slipped past every existing test

The checkpoint-backward test in `test_block_swap.py` **stayed fully green
even while the bug existed**. The reasons:

- Small tensors (16-32 dim), and the `_Tiny` block **has no attention**, so
  computation is too fast for the race window to manifest;
- It compared bit-for-bit with `assert_close` — while at real sizes, SDPA's
  backward pass on CUDA/bf16 **is inherently non-deterministic** (running
  the same weights twice gives gradients that differ by about 5e-3), making
  bit-for-bit comparison unusable at real sizes in the first place.

Two lessons, now baked into `tests/test_block_swap_grad_fidelity.py`:

1. **Verifying concurrency/races must use real sizes.** A green light on a
   small case is false confidence.
2. **In a non-deterministic environment, measure the noise floor before
   building a criterion.** During the first investigation, "gradients
   aren't bit-for-bit equal" was taken directly as evidence, when in fact
   even running without swap twice wouldn't be equal either — a measurement
   without a control group produces both false positives (that time) and
   false negatives (the original unit test) at once. The criterion should
   be "matches the control group's own repeatability, same order of
   magnitude."
### 9.12 Actual pinned lock = weights × 1.47 (a 5080 / 32GB RAM machine crashed for real, now fixed)

**Symptom**: a machine with a 16GB card + 32GB RAM, with `blocks_to_swap=28`
enabled, hit a `CUDA error: out of memory` in `load_krea2_model` at
`tensor.pin_memory()` — at this point not a single DiT layer had loaded onto
the GPU yet, so this isn't a VRAM OOM, it's a `cudaHostAlloc` (host
page-locked memory) failure.

**Two compounding factors**:

1. **Windows WDDM has a hard cap on `cudaHostAlloc` of ≈50% of physical
   memory**, managed by Windows and not adjustable by the driver,
   independent of allocation block size (confirmed across multiple NVIDIA
   forum posts). `check_pinned_budget` only computes based on the available
   memory ratio, and hasn't modeled this line.
2. **PyTorch's host caching allocator rounds every pinned allocation up to a
   power of 2** (`CachingHostAllocator.h`'s `PowerOf2Ceil`), while the
   loader calls `pin_memory()` tensor by tensor. Krea2's sizes happen to be
   particularly unfavorable for this:

   | Tensor | actual | locked |
   |---|---|---|
   | 16384×6144 fp8 | 96 MB | 128 MB |
   | 6144×6144 fp8 | 36 MB | 64 MB |
   | 1536×6144 fp8 | 9 MB | 16 MB |

   | blocks_to_swap | pinned counted by log/guardrail | **actual lock** |
   |---|---|---|
   | 14 | 5.66 GB | 8.31 GB |
   | 18 | 7.28 GB | 10.69 GB |
   | 28 | 11.32 GB | **16.63 GB** |

   §9.5c's `pinned 11.32 GB` is the `numel × element_size` sum, not the
   allocator's real usage; that 5090 machine had a lot of RAM (37.5GB
   available), so 16.6GB fit without exposing the problem. On a 32GB
   machine, the 50% line = 16GB, and 28 layers' 16.63GB would definitely
   blow past it — while the guardrail, computing from 11.32GB, would let it
   through. Anima's 2B-parameter tensor sizes happen to already be powers of
   2 (1.02x), and the 36-layer Anima variant is 1.29x.

**Fix (`PinnedPacker`, `training/block_swap.py`)**: instead of pinning
tensor by tensor, pre-allocate a handful of blocks in one shot using a
binary decomposition of the **exact total size** (8G+2G+1G+256M+64M, each
block exactly a power of 2 → zero rounding by the allocator), best-fitting
each tensor into one of the blocks, 256B-aligned, returning a view onto that
block. Simulated across multiple families/configs (krea2 fp8/bf16 ×
14/18/28, anima 2048/5120 × 8/14/full), the actual lock = weights ×
1.00-1.03; tensors that don't fit fall back to the overflow path
(`pow2_ceil(nbytes)` in its own block = the old behavior, never worse than
before). Both pinning paths were wired up: the krea2 loader landing
directly (the planned byte count is computed exactly from the header by
`_swapped_pinned_bytes`, mirroring the loading rule — fp8 as-is, everything
else by compute dtype), and `PinnedBlockSwap._build`'s non-pre-pinned path
(Anima placement / GPU-resident). The view semantics are transparent to the
existing mechanism: `is_pinned()` holds true, `param.data = view` works as
before, H2D works as usual, and `release_pinned_host_cache` still returns
memory per §9.7.

Along the way, `cudaHostAlloc` failures were translated into actionable copy
(`PinnedAllocationError`: explains it's page-locked memory, not VRAM, how
much needs to be locked, where the Windows cap is, and to reduce
`blocks_to_swap`).

**Not changed**: the guardrail still computes in `numel × element_size`
terms and doesn't model the 50% line; the `pinned x GB` log inside `_build`
is still the numel-based figure — after fixing the packing, both are within
≤3% of the actual lock, an acceptable margin of error; the loader adds one
new log line, "weights X GB packed into N blocks, actually locked Y GB," for
verification on real hardware.
