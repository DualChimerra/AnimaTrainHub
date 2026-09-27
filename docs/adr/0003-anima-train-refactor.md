# 0003 — anima_train.py modularization refactor (plugin boundary + adapter hook protocol)

**Status**: Proposed
**Date**: 2026-05-14
**Decision makers**: @WalkingMeatAxolotl

## Background

`runtime/anima_train.py` has grown to **2901 lines**, with `main()` alone
taking up **793 lines** (L2105-L2897). 53 defs/classes sit flat in a single
module, spanning 10 different responsibilities: CLI / model loading / text
encoding / dataset / sampling scheduling / training-state IO / wandb /
training loop / periodic IO / signal handling.

### Two immediate factors forcing this refactor

1. **PR #49** (saltysalrua + Claude Sonnet 4.6 co-author, 1508 lines) tried to
   stuff in three LoRA variants: T-LoRA, Ortho-Hydra, and OrthoGrad. After a
   three-way review (for/against/PM), the decision was to **only merge the
   stability patches to dev** (PR #55, already merged), with the three
   variants parked on the `experimental/pr49-adapters` parking-lot branch
   (see [memory/pr49-handling]). This operation forced a 9-commit
   cherry-pick split + a surgical split of c5e81c2, and cleaning up dead
   dispatch leftovers scattered across commits. Most of the pain came from
   `main()`'s dispatch logic being tangled together with variant-specific
   implementations.

2. The PPSF integration (PR #46) had already identified two layers of
   refactoring need, which were parked at the time in
   [memory/anima_train_refactor_pending]:
   - **P0**: hoist the optimizer kwargs dispatch in `main()` L2336-2374 up to
     `utils/optimizer_utils.py`
   - **P1**: split `main()` into phases under a `runtime/training/` subpackage

Once PR #55 merged into dev, both of these became doable, since the base is
now stable.

### Current-state inventory (based on dev `ada49a3`)

The `runtime/` directory:

| File | Lines | Responsibility |
|---|---:|---|
| `anima_train.py` | 2901 | Giant mega-script |
| `anima_daemon.py` | 651 | Studio's long-running process (generation / xy-matrix), borrows loading logic via `import anima_train as _T` |
| `anima_generate.py` | 320 | One-off inference CLI, same as above |
| `anima_reg_ai.py` | 252 | AI-captions the regularization set, same as above |
| `train_monitor.py` | 168 | monitor_state.json writer (already independent, no change needed) |

`anima_train.py`'s 53 defs/classes naturally cluster by responsibility:

| Cluster | Lines | Functions/classes | Lines |
|---|---|---|---:|
| bootstrap | 60-156 | `ensure_dependencies` / `load_yaml_config` / `apply_yaml_config` / `_lazy_imports` / `init_progress` | 96 |
| progress + monitoring display | 183-369 | `render_loss_curve` / `render_curve_panel` / `WandBMonitor` / `init_wandb_monitor` | 186 |
| model-loading infrastructure | 370-612 | `forward_with_optional_checkpoint` / `enable_xformers` / `find_diffusion_pipe_root` / `load_module_from_path` / `_strip_prefixes` / `_pick_best_prefix_remap` / `_load_safetensors_state_dict` / `resolve_path_best_effort` / `_load_weights_best_effort` | 242 |
| model-loading entry points | 614-775 | `ensure_models_namespace` / `load_anima_model` / `load_vae` / `load_text_encoders` | 161 |
| text encoding | 777-1071 | `encode_qwen` / `_parse_weighted_tag` / `_build_qwen_text_from_prompt` / `tokenize_t5_weighted` | 294 |
| flow/sigma/sampling schedule | 822-961, 1677-1937 | `_time_snr_shift` / `_flow_sigmas_simple` / `_default_noise_sampler` / `_sample_er_sde_const_x0` / `sample_image` / `sample_t` / `make_noise` / `compute_loss_weight` | 400 |
| training-state IO | 1073-1142 | `save_training_state` / `load_training_state` | 69 |
| dataset | 1144-1675 | `BucketManager` / `ImageDataset` / `RepeatDataset` / `MergedDataset` / `BucketBatchSampler` / `CachedLatentDataset` | 531 |
| collate | 1939-1962 | `collate_fn` / `collate_fn_cached` | 23 |
| CLI | 1963-2103 | `parse_args` / `_try_rich` / `_ask_*` / `_guess_default_paths` / `prompt_for_args` | 140 |
| main() | 2105-2897 | — | 793 |

### External dependencies on the `anima_train` module (contracts that cannot be broken)

Three sister scripts call top-level `anima_train` names via
`import anima_train as _T`:

- `runtime/anima_daemon.py:44, 160-218, 530, 650` — `_T.find_diffusion_pipe_root` / `_T.resolve_path_best_effort` / `_T.load_anima_model` / `_T.enable_xformers` / `_T.load_vae` / `_T.load_text_encoders` / `_T.sample_image`
- `runtime/anima_generate.py:43, 122-191, 344` — the same 7 names
- `runtime/anima_reg_ai.py:43, 232-283` — the same 7 names

Tests import these names directly:

- `tests/test_anima_train_migration.py` — `parse_args` / `apply_yaml_config` (CLI aliasing + YAML merge semantics)
- `tests/test_lycoris_resume.py` — `save_training_state` / `load_training_state` (resuming from a checkpoint)

**Constraint 1**: after the refactor, `anima_train.py` must re-export the
above 9 names at the top level, or every sister script and test breaks.

### Other affected references

- `docs/adr/0002-webui-self-update.md` L17, L388 hardcode
  `runtime/anima_train.py#L2374-L2401` (the Ctrl+C handler) — the line numbers
  will change after the refactor and the link needs updating (or switch to a
  grep-keyword anchor)
- `models/cosmos_predict2_modeling.py:34` comment "cli.py / runtime/anima_train.py
  calls enable once at startup" — comment-level, no impact
- `README.md` / `docs/architecture/studio-pipeline.md` and similar
  documentation that mentions anima_train — no behavioral impact

## Candidate solutions

### Candidate A — keep the status quo

Keep adding variant dispatch inside `main()`.

- Pros: zero effort
- Cons: parking-lot rebases get more painful over time; the next paper-level
  LoRA variant (T-LoRA-style) would again need code stuffed into the middle
  of `main()`; review diffs always span 10 responsibility domains
- Rejected

### Candidate B — split files only (no plugin abstraction)

Move the 53 defs/classes into a `runtime/training/` subpackage by
responsibility, while `main()` stays a single function and dispatch remains
inline.

- Pros: purely mechanical move, zero behavior change; single-PR review
  friendly
- Cons: solves "code is hard to locate" but **not "adding a variant requires
  editing main()"**; adding the next LoRA variant would still require editing
  training orchestration logic
- Partially adopted (as PR-A)

### Candidate C — split files + a full plugin system (including the adapter hook protocol)

Building on B, introduce 4 plugin subpackages (adapter / optimizer /
lr_scheduler / inference_sampler) + give the adapter a Protocol with
extension hooks (per-step, loss additions).

- Pros: adding/removing a variant becomes a local change; the parking lot no
  longer blocks dev's evolution; paper-level adapter variants (per-step
  behavior classes) get a clear extension point
- Cons: abstraction cost of ~3 weeks; introduces a new "convention" (registry
  dicts, Protocol hooks) that future maintainers need to learn
- **Adopted**

## Decision

**Candidate C is adopted**, implemented across three PRs, merged into dev
three times:

| PR | Scope | Behavior change | Estimated net lines |
|---|---|---|---:|
| **PR-A** file move | Move the 53 defs/classes into a `runtime/training/` subpackage by responsibility; `anima_train.py` becomes a thin shim re-exporting the 9 names used by sisters/tests | 0 | -2600 / +2600 (move) |
| **PR-B** main() phase split | Introduce a `TrainingContext` dataclass; split `main()`'s 793 lines into 6 phase functions; extract train_loop | 0 | -700 / +700 (reorganize) |
| **PR-C** plugin-ify + Protocol | 4 plugin subpackages + `AdapterProtocol` with hooks; `main()`'s dispatch disappears entirely | 0 | -200 / +500 (net +300) |

All three PRs **keep training behavior bit-for-bit identical** (the same yaml
produces the same loss curve); locked in place by
`tests/test_anima_train_migration.py` for CLI/YAML behavior, and validated by
end-to-end smoke training to confirm the loop is unchanged.

### Design principles

- **Zero user-facing change**: `studio/schema.py:TrainingConfig` stays a
  single large class with all fields flat; the frontend UI / CLI arguments /
  `config.yaml` field names change zero. Plugins exist only inside
  `runtime/training/`.
- **Zero sister-script / test API change**: `anima_train.py` re-exports all 9
  names used externally, at the top level.
- **Plugins = explicit dicts, not decorators**:
  `BUILDERS = {"lokr": lycoris.build, ...}`. Removing a variant = delete one
  dict line + delete one file. Decorators + side-effect imports look elegant
  but are load-order sensitive, and "where are all the registered entries"
  becomes unfindable.
- **"A new variant introducing new schema fields → its own subfolder"** is a
  hard rule of the plugin-ification. Each adapter / optimizer / lr_scheduler
  / inference_sampler variant carries its own config fields → one variant,
  one file. timestep_sampling / loss_weighting / noise are parameterized pure
  math functions that don't introduce schema fields → one file, multiple
  functions.

## Target layout

```
runtime/
  anima_train.py                 ← ~150-line thin shim: main() orchestration + re-exports
  anima_daemon.py                ← unchanged (still does import anima_train as _T)
  anima_generate.py              ← unchanged
  anima_reg_ai.py                ← unchanged
  train_monitor.py               ← unchanged
  training/
    __init__.py                  ← package init; exposes names used by sister scripts
    context.py                   ← TrainingContext dataclass / StepContext dataclass
    bootstrap.py                 ← deps / yaml / args preprocessing
    cli.py                       ← parse_args / interactive / prompt_for_args
    observability.py             ← WandBMonitor + curve rendering + emit
    model_loading.py             ← prefix inference / safetensors / path resolution (internal utils)
    models.py                    ← load_anima_model / load_vae / load_text_encoders (public)
    text_encoding.py             ← qwen / t5 + tokenize_weighted
    state.py                     ← save/load_training_state (public)
    dataset.py                   ← BucketManager / ImageDataset / Merged / Sampler / Cached / collate
    sampling.py                  ← sigma utils + sample_image (public)
    timestep_sampling.py         ← sample_t modes (one file, multiple fns)
    loss_weighting.py            ← compute_loss_weight schemes
    noise.py                     ← make_noise (offset + pyramid)
    loop.py                      ← train_loop.run(ctx)
    phases/
      __init__.py
      models.py                  ← model + vae + text_encoder loading
      dataset.py                 ← build_datasets + dataloader
      optimizer.py                ← build_optimizer + scheduler + grad_clip
      resume.py                  ← state recovery + signal handler
      finalize.py                ← final save + progress cleanup

    # 4 plugin subpackages
    adapters/
      __init__.py                ← BUILDERS dict + build_adapter(args) + AdapterProtocol
      protocol.py                ← AdapterProtocol + StepContext
      lycoris.py                 ← registers lokr/loha/lora; internally calls utils.lycoris_adapter
    optimizers/
      __init__.py                ← BUILDERS + VALIDATORS dict
      adamw.py
      prodigy.py
      prodigy_plus_schedulefree.py
    schedulers/
      __init__.py                ← BUILDERS dict + build_scheduler(args, optimizer, total_steps)
      cosine.py
      cosine_with_restart.py
    inference_samplers/
      __init__.py                ← BUILDERS dict (only er_sde for now)
      er_sde.py
```

After the refactor, `anima_train.py` looks roughly like this:

```python
# runtime/anima_train.py
"""Thin shim: this module keeps the main() entry point and the
backward-compatible re-exports. The real implementation lives in the
runtime/training/ subpackage."""

from training.cli import parse_args, prompt_for_args
from training.bootstrap import apply_yaml_config, ensure_dependencies, load_yaml_config
from training.model_loading import find_diffusion_pipe_root, resolve_path_best_effort
from training.models import enable_xformers, load_anima_model, load_text_encoders, load_vae
from training.sampling import sample_image
from training.state import load_training_state, save_training_state
from training.context import TrainingContext
from training import loop, phases

def main():
    args = parse_args()
    if args.config:
        args = apply_yaml_config(args, load_yaml_config(args.config))
    if args.interactive or _any_required_missing(args):
        args = prompt_for_args(args)
    ensure_dependencies(auto_install=args.auto_install)

    ctx = TrainingContext.bootstrap(args)
    ctx = phases.models.run(ctx)
    ctx = phases.dataset.run(ctx)
    ctx = phases.optimizer.run(ctx)
    ctx = phases.resume.run(ctx)
    loop.run(ctx)
    phases.finalize.run(ctx)


if __name__ == "__main__":
    main()
```

The re-export block guarantees calls like `anima_daemon.py:_T.load_anima_model`
need zero changes; `main()` moves to phase orchestration.

## Plugin pattern details

### Registry: explicit dict

```python
# training/adapters/__init__.py
from typing import Callable
from .protocol import AdapterProtocol, StepContext
from . import lycoris

BUILDERS: dict[str, Callable[..., AdapterProtocol]] = {
    "lokr": lycoris.build,
    "loha": lycoris.build,
    "lora": lycoris.build,
}

def build_adapter(args) -> AdapterProtocol:
    if args.lora_type not in BUILDERS:
        raise ValueError(
            f"unknown lora_type={args.lora_type!r}; registered: {sorted(BUILDERS)}"
        )
    return BUILDERS[args.lora_type](args)
```

```python
# training/adapters/lycoris.py
from utils.lycoris_adapter import AnimaLycorisAdapter
from .protocol import AdapterProtocol  # actually a Protocol, used only for type hints

def build(args) -> AdapterProtocol:
    return AnimaLycorisAdapter(
        algo=args.lora_type,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        factor=args.lokr_factor,
        dropout=args.lora_dropout,
        rank_dropout=args.lora_rank_dropout,
        module_dropout=args.lora_module_dropout,
        weight_decompose=args.lora_dora,
        rs_lora=args.lora_rs,
    )
```

### AdapterProtocol: required methods + default no-op hooks

```python
# training/adapters/protocol.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
import torch
from torch import nn, Tensor

@dataclass(frozen=True)
class StepContext:
    """Minimal context passed to adapter hooks."""
    global_step: int
    total_steps: Optional[int]
    epoch: int
    sigma_t: Tensor          # shape [B], the sigma for this micro-batch
    args: object             # the Namespace produced by parse_args; read fields as needed

@runtime_checkable
class AdapterProtocol(Protocol):
    # ─── required ───
    def inject(self, model: nn.Module) -> None: ...
    def get_param_groups(self, weight_decay: float) -> list[dict]: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...

    # ─── optional hooks: default no-op; variants override as needed ───
    def on_step_begin(self, ctx: StepContext) -> None:
        """Called before each micro-batch's forward pass.

        T-LoRA / AdaLoRA / B-LoRA use this to adjust rank masks, active
        subsets, column dropping, and other "runtime structural
        adjustments" based on sigma_t / step. Default no-op."""
        return None

    def regularization_loss(self, ctx: StepContext) -> Optional[Tensor]:
        """Returns a regularization term to add to the main loss; None means
        none.

        OFT returns an orthogonality penalty; Ortho-Hydra returns an expert
        balance loss; default None. When train_loop receives None, it does
        nothing extra."""
        return None

    def excludes_weight_decay(self, param_name: str) -> bool:
        """Whether this param should be excluded from weight_decay.

        Replaces the current hardcoded `injector.use_lokr` check. In the
        LoKr implementation: return "w1" in param_name. Default False."""
        return False
```

`AnimaLycorisAdapter` already implements the first 4 required methods; all
hooks default to no-op, so **current lokr/loha/lora behavior is completely
unchanged** — the only change is in `optimizer_setup.py`, replacing the
hardcoded `injector.use_lokr` check with `adapter.excludes_weight_decay(name)`.

### Hook calls inside train_loop

```python
# training/loop.py (excerpt)
for batch_idx, batch in enumerate(dataloader):
    captions, latents = ...
    t = sample_t(bs, device, mode=ts_mode, shift=ts_shift)

    step_ctx = StepContext(
        global_step=global_step,
        total_steps=total_steps,
        epoch=epoch,
        sigma_t=t,
        args=args,
    )

    # ★ adapter hook 1: can adjust runtime structure
    ctx.adapter.on_step_begin(step_ctx)

    noise = make_noise(latents, ...)
    noisy = (1 - t) * latents + t * noise
    target = noise - latents

    with torch.autocast("cuda", dtype=dtype):
        pred = forward_with_optional_checkpoint(model, noisy, t, cross, pad_mask, ...)
        loss_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none")
        # ... loss_weight / loss_weighting / regularization-set weighting ...
        loss = loss_per_sample.mean()

        # ★ adapter hook 2: variants can add a regularization term
        reg = ctx.adapter.regularization_loss(step_ctx)
        if reg is not None:
            loss = loss + reg

    if not torch.isfinite(loss):
        ...

    loss = loss / args.grad_accum
    loss.backward()
    ...
```

The two hook calls + StepContext construction amount to roughly 8 extra
lines, with zero performance impact on the LyCORIS-only path (a Python
function call returning None).

## Real-world landing cases for different variants

### Case 1: DoRA (NVIDIA 2024)

**Situation**: already supported. The current schema has `lora_dora: bool`,
passed to `AnimaLycorisAdapter(weight_decompose=...)`, and LyCORIS internally
takes the DoRA path.

**Plugin perspective**: DoRA doesn't add a schema enum value (it's not a new
`lora_type`), it's a switch on lokr/lora. After plugin-ification, **no
changes are needed at all**.

### Case 2: adding a new pure-weight variant — VeRA (shared A/B, training only d/b vectors)

**Situation**: not implemented, shown as an example. VeRA introduces a new
layer structure.

**Steps**:

1. New file `utils/vera_adapter.py` implementing the actual algorithm
   (subclassing `nn.Module`, implementing `inject` / `get_param_groups` /
   `save` / `load`)
2. New file `training/adapters/vera.py`:
   ```python
   from utils.vera_adapter import AnimaVeRAAdapter
   def build(args):
       return AnimaVeRAAdapter(
           rank=args.lora_rank,
           shared_seed=args.vera_shared_seed,
           d_init=args.vera_d_init,
       )
   ```
3. Add one line to `training/adapters/__init__.py`'s dict: `"vera": vera.build`
4. `studio/schema.py`:
   - Add one more value to `lora_type: Literal[..., "vera"]`
   - Add fields like `vera_shared_seed: int = Field(default=42, _meta(group="LoRA", show_when="lora_type=='vera'"))`

**Zero changes to main() / loop / other phases**. The hook protocol is
untouched (VeRA needs no per-step adjustment).

### Case 3: adding a new per-step variant — T-LoRA (adjusting rank based on sigma_t)

**Situation**: a complete implementation
(`utils/tlora_adapter.py`) exists on the parking lot. How to reclaim it once
dev has this refactor.

**T-LoRA's core algorithm**: allows full rank early in training (learning
more LoRA content), progressively masking out columns based on noise level
to avoid overfitting the low-noise regime. train_loop needs to feed the
current sigma to the adapter every step.

**Steps**:

1. `utils/tlora_adapter.py` already exists on the parking lot; move it to dev
2. New file `training/adapters/tlora.py`:
   ```python
   from pathlib import Path
   from utils.tlora_adapter import AnimaTLoRAAdapter as _Inner
   from .protocol import StepContext

   class TLoRABundle:
       """Wraps utils.AnimaTLoRAAdapter to satisfy AdapterProtocol,
       specifically the on_step_begin hook."""

       def __init__(self, args):
           self._inner = _Inner(
               rank=args.lora_rank,
               alpha=args.lora_alpha,
               min_rank=args.tlora_min_rank,
               alpha_rank_scale=args.tlora_alpha_rank_scale,
               sig_type=args.tlora_sig_type,
           )

       def inject(self, model): self._inner.inject(model)
       def get_param_groups(self, wd): return self._inner.get_param_groups(wd)
       def save(self, path: Path): self._inner.save(path)
       def load(self, path: Path): self._inner.load(path)

       def on_step_begin(self, ctx: StepContext) -> None:
           # T-LoRA's core hook: adjust the mask based on the current sigma
           self._inner.set_sigma(ctx.sigma_t)
           # can also adjust the mask based on step (e.g. full rank during warmup)
           if ctx.total_steps:
               progress = ctx.global_step / ctx.total_steps
               self._inner.set_mask_progress(progress)

   def build(args):
       return TLoRABundle(args)
   ```
3. Add `"tlora": tlora.build` to `training/adapters/__init__.py`
4. Add `Literal[..., "tlora"]` + `tlora_*` fields to `studio/schema.py`

train_loop is unchanged (the `on_step_begin` call site already exists).

**Why the original PR #49 can't just be cherry-picked and instead needs a
rewrite**: the original PR wrote `injector.set_sigma()` inline in train_loop,
coupled to main()'s orchestration. After the refactor, train_loop calls
through `adapter.on_step_begin(ctx)`, T-LoRA implements the hook, and
main() never sees a variant-specific call like set_sigma at all.

### Case 4: a variant that needs to add a regularization term to the main loss — OFT (Orthogonal Fine-Tuning)

**Situation**: not implemented, shown as an example. OFT parameterizes the
LoRA delta with an orthogonal matrix; training needs an orthogonality
penalty added to the loss to prevent numerical drift.

**Steps**:

1. `utils/oft_adapter.py` implements `AnimaOFTAdapter`, including a
   `compute_orth_penalty()` method
2. `training/adapters/oft.py`:
   ```python
   class OFTBundle:
       def __init__(self, args):
           self._inner = AnimaOFTAdapter(rank=args.lora_rank, ...)
           self._penalty_weight = args.oft_orth_penalty

       # ... the 4 required methods delegate to _inner ...

       def regularization_loss(self, ctx: StepContext) -> Optional[Tensor]:
           penalty = self._inner.compute_orth_penalty()
           return penalty * self._penalty_weight

   def build(args): return OFTBundle(args)
   ```
3. One line each in the dict + the schema

train_loop already picks this up at `if reg is not None: loss = loss + reg`.

### Case 5: a mixed-requirement case — Ortho-Hydra (router lr + balance loss + per-step warmup)

**Situation**: a complete implementation exists on the parking lot. It has
the most varied requirements: (1) router params need their own lr scale
(10x default); (2) balance loss needs to be weighted by a warmup ratio; (3)
loading it in the inference ecosystem has compatibility issues (non-standard
LoRA keys) — this is the actual reason ADR 0001 doesn't bring it into the
main repo, and is unrelated to plugin-ification.

**What plugin-ification can cover**:
- router lr: currently on dev, `get_param_groups` returns multiple groups,
  each tagged with an `_is_router_group` flag, and main() pops it out and
  multiplies by the scale. After the refactor, the adapter's
  `get_param_groups(wd)` directly returns multiple groups with the correct
  `lr` field, and main()/the optimizer phase has no variant-specific logic
  left.
- balance loss: uses the `regularization_loss(ctx)` hook, internally
  computing the warmup factor from `ctx.global_step / total_steps`.

**What plugin-ification cannot cover**:
- The output `.safetensors` isn't standard LoRA keys (contains
  `S_q / S_p / Q_basis / P_bases / lambda_layer / router`), so a1111/ComfyUI
  can't load it — this is a structural problem outside the scope of the
  refactor. If an OrthoHydra loader ever appears in the ecosystem, revisit
  the GC decision in [memory/pr49-handling].

### Case 6: a new optimizer — Lion / CAME / Schedule-Free AdamW

**Situation**: optimizer-class variants have a simpler interface than
adapters (no per-step hook), using a plain build function.

**Steps** (using Lion as an example):

1. `pip install lion-pytorch` — one line added to `requirements.txt`
2. New file `training/optimizers/lion.py`:
   ```python
   from lion_pytorch import Lion

   def build(args, params, lr, weight_decay):
       return Lion(
           params, lr=lr, weight_decay=weight_decay,
           betas=(args.lion_beta1, args.lion_beta2),
       )
   ```
3. `training/optimizers/__init__.py`: `BUILDERS["lion"] = lion.build`
4. `studio/schema.py`: add `"lion"` to the `optimizer_type` Literal + add
   `lion_beta1` / `lion_beta2` fields

Zero changes to main() / loop.

**Special case**: if this optimizer needs PPSF-like "switch to averaged
weights during eval" behavior (e.g. Schedule-Free AdamW), it's automatically
covered via the `utils.optimizer_utils.optimizer_eval_mode` context manager,
which detects `hasattr(optimizer, "train")`.

**Validation logic** (e.g. PPSF requires `lr_scheduler=none`):

```python
# training/optimizers/__init__.py
VALIDATORS: dict[str, Callable[[object], None]] = {}

# training/optimizers/prodigy_plus_schedulefree.py
def validate(args):
    if args.lr_scheduler != "none":
        raise SystemExit(
            f"ProdigyPlusScheduleFree requires lr_scheduler=none; "
            f"got lr_scheduler={args.lr_scheduler!r}"
        )

def build(args, params, lr, weight_decay):
    ...

# __init__.py
BUILDERS["prodigy_plus_schedulefree"] = ppsf.build
VALIDATORS["prodigy_plus_schedulefree"] = ppsf.validate
```

```python
# in phases/optimizer.py:
def run(ctx):
    validator = VALIDATORS.get(ctx.args.optimizer_type)
    if validator:
        validator(ctx.args)
    ctx.optimizer = build_optimizer(
        ctx.args, ctx.adapter.get_param_groups(ctx.weight_decay),
        ctx.args.learning_rate, ctx.weight_decay,
    )
    return ctx
```

### Case 7: a new LR Scheduler — warmup_cosine

**Situation**: only `cosine` / `cosine_with_restart` exist now. A version
with warmup needs to be added.

**Steps**:

1. New file `training/schedulers/warmup_cosine.py`:
   ```python
   from torch.optim.lr_scheduler import LambdaLR
   import math

   def build(args, optimizer, total_steps):
       warmup = args.lr_scheduler_warmup_steps
       def lr_lambda(step):
           if step < warmup:
               return step / max(1, warmup)
           p = (step - warmup) / max(1, total_steps - warmup)
           return 0.5 * (1 + math.cos(math.pi * p))
       return LambdaLR(optimizer, lr_lambda=lr_lambda)
   ```
2. `training/schedulers/__init__.py`: `BUILDERS["warmup_cosine"] = warmup_cosine.build`
3. `studio/schema.py`: add `"warmup_cosine"` to the `lr_scheduler` Literal +
   `lr_scheduler_warmup_steps: int = Field(default=500, ...)`

### Case 8: a new Inference Sampler — Euler / DPM++2M

**Situation**: currently `sample_image` internally hardcodes a call to
`_sample_er_sde_const_x0`.

**Steps**:

1. New file `training/inference_samplers/euler.py`:
   ```python
   import torch
   def sample(model, noise, cond, sigmas, *, device, dtype, cfg_scale, ...):
       """Standard Euler; takes noise, a target sigma schedule, and cond as
       input, outputs a latent."""
       x = noise.clone()
       for i in range(len(sigmas) - 1):
           ...
       return x
   ```
2. `training/inference_samplers/__init__.py`: `BUILDERS["euler"] = euler.sample`
3. `sample_image` in `training/sampling.py` changes to:
   ```python
   def sample_image(model, vae, ..., sampler_name="er_sde", scheduler="simple", ...):
       sigmas = build_sigma_schedule(scheduler, steps)
       sampler = inference_samplers.BUILDERS[sampler_name]
       noise = torch.randn(shape, device=device, dtype=dtype)
       latent = sampler(model, noise, cond, sigmas, ...)
       return vae.decode(latent)
   ```
4. `studio/schema.py`: add `"euler"` to the `sample_sampler_name` Literal

Zero changes to anima_train / loop / phases.

### Case 9: removing a variant

The simplest example: the day someone notices `cosine_with_restart` has no
users and needs cleanup.

**Steps** (a 3-line change):

1. `git rm training/schedulers/cosine_with_restart.py`
2. Delete one line from `training/schedulers/__init__.py`:
   `"cosine_with_restart": cosine_with_restart.build`
3. `studio/schema.py`: remove `"cosine_with_restart"` from
   `lr_scheduler: Literal[...]`

No one references it, and there's no dispatch logic to delete. This is what
"good to remove" looks like.

## What stays unchanged (explicit boundary)

- `studio/schema.py` keeps a single `TrainingConfig` class, with all fields
  flat. **Plugins don't split the schema**.
- `studio/argparse_bridge.py` has zero changes (CLI args are still derived
  from the schema fields via Pydantic)
- The frontend UI / `config.yaml` / `studio.sh` / `studio/cli.py` /
  `studio/supervisor.py` have zero changes
- `runtime/anima_daemon.py` / `anima_generate.py` / `anima_reg_ai.py` /
  `train_monitor.py` have zero changes
- Adapter / optimizer implementations under `utils/` (`lycoris_adapter.py` /
  `optimizer_utils.py` / `caption_utils.py`, etc.) have zero changes; the
  plugin subpackages are purely a "dispatch layer" reusing the existing
  implementations
- Zero change in training behavior: the loss curve and final LoRA weights
  produced from the same `config.yaml` should be bit-for-bit identical

## Risks and open questions

### R1 — experimental/pr49-adapters needs 3 rebases over the course of the three PRs

Each time a dev refactor PR merges, the parking-lot branch needs
`git rebase dev`. The location of variant adapter files changes (e.g.
`utils/tlora_adapter.py` stays put, but train_loop's dispatch moves out of
main()), and most conflicts during rebase happen in main()/anima_train.py.

**Mitigation**: after PR-C merges, rewrite the parking-lot
`tlora_adapter.py` / `orthohydra_adapter.py` as bundle files conforming to
the new Protocol, and store them under `training/adapters/`. After that,
the parking lot no longer has anima_train inline code, and rebase pain
disappears.

[memory/pr49-handling] already has a note that "experimental needs a rebase
after the refactor"; [memory/anima_train_refactor_pending]'s status has also
been updated.

### R2 — dev cannot take other training-side changes during the three PRs

PR-A moves ~5000 lines of changes, which will conflict with any
anima_train change. A freeze window is recommended (~1 week for A, then ~3
days for B, then ~3 days for C). Training-related hotfixes will have to wait
during this window.

### R3 — sister-script re-exports might miss a name

`anima_train` has 53 top-level defs/classes. We've only confirmed 9 names
used by sister scripts + tests. Other names (internal utils like
`_strip_prefixes` / `forward_with_optional_checkpoint`) shouldn't
theoretically be depended on externally, but if there's a hidden caller,
PR-A merging would cause an immediate ImportError.

**Mitigation**: before PR-A ships, run
`grep -r "anima_train\." --include="*.py"` across the `tests/` / `studio/` /
`runtime/` directories to search exhaustively.

### R4 — schema and registry staying in sync

Adding a variant touches 2 places (the plugin dict + the schema Literal).
It's easy for a schema enum value to get added without registering the
plugin, or vice versa.

**Mitigation**: add a `_validate_schema_consistency()` function in
`training/adapters/__init__.py`, run once at startup:

```python
def _validate_schema_consistency():
    from studio.schema import TrainingConfig
    schema_options = set(TrainingConfig.model_fields["lora_type"].annotation.__args__)
    registered = set(BUILDERS)
    if schema_options != registered:
        raise RuntimeError(
            f"adapter registration out of sync with schema:\n"
            f"  in schema but not registered: {schema_options - registered}\n"
            f"  registered but not in schema: {registered - schema_options}"
        )
```

Do the same for optimizer / scheduler / inference_sampler.

### R5 — changes to other training-related modules like `cosmos_predict2_modeling.py`

PR-A doesn't touch these; if PR-B or PR-C needs to adjust the model-loading
flow (e.g. changing `load_anima_model`'s signature), it might affect .py
files under `models/`. This needs to be confirmed during PR-B's design.

### R6 — ADR 0002's line-number references

Once PR-A merges, the `runtime/anima_train.py#L2374-L2401` reference in
[docs/adr/0002-webui-self-update.md L17, L388] becomes stale.

**Mitigation**: update ADR 0002's links in the same PR as PR-A, changing
from line numbers to a keyword anchor pointing at
`runtime/training/phases/resume.py:signal_handler` or similar.

## Acceptance criteria

PR-A:
- `tests/test_anima_train_migration.py` fully green
- `tests/test_lycoris_resume.py` fully green
- `tests/test_anima_generate_xy.py` fully green (verifies the sister-script
  API isn't broken)
- `python runtime/anima_train.py --help` output diffs to empty against
  pre-refactor
- End-to-end: train 100 steps on a small dataset; the final `safetensors`
  matches the pre-refactor output bit-for-bit (or the loss curve matches
  within fp error)

PR-B: same as above + `python runtime/anima_train.py --config sample.yaml --max-steps 100` runs successfully

PR-C: same as above + a new `tests/test_plugin_registry.py`:
- registering / looking up / removing a mock plugin works correctly
- the schema-registry consistency check fires
- `AdapterProtocol` runtime_checkable returns True for `AnimaLycorisAdapter`
- mock adapters like T-LoRA / OFT (defined only within the test) can attach
  hooks and get called once by train_loop

## Related

- [docs/adr/0001-lokr-via-lycoris-lora.md](0001-lokr-via-lycoris-lora.md) — explains why the adapter implementation layer is LyCORIS (PR-C doesn't touch it)
- [docs/adr/0002-webui-self-update.md](0002-webui-self-update.md) — the Ctrl+C handler's line-number reference needs updating
- [memory/pr49-handling](../../C:/Users/Mei/.claude/projects/G--AnimaLoraStudio/memory/pr49-handling.md) — the parking-lot policy; the path improves after PR-C
- [memory/anima_train_refactor_pending](../../C:/Users/Mei/.claude/projects/G--AnimaLoraStudio/memory/anima_train_refactor_pending.md) — this ADR is its execution plan
- [memory/feedback_authoring_time_normalize](../../C:/Users/Mei/.claude/projects/G--AnimaLoraStudio/memory/feedback_authoring_time_normalize.md) — a single, unsplit schema class follows the "normalize at authoring time" principle

## Resolved items (aligned before kickoff, starting 2026-05-14)

- **Automatic schema↔registry consistency validation**: PR-C **does add**
  `_validate_schema_consistency()`, run once at startup, for each of the 4
  plugin subpackages (adapter / optimizer / scheduler / inference_sampler).
  Reasoning: forgetting to update the schema or the registry on dev is a
  silent bug, and discovering it only when running training costs more than
  a 5-line startup check.
- **`inference_samplers/` is done this round**: even though only er_sde
  exists right now, still build the subfolder + `BUILDERS` dict +
  `er_sde.py` as a placeholder. Adding euler/dpmpp later is just dropping in
  a file, no changes to sample_image.
- **Shared math utils live close to their use**: small utilities needed
  internally by `noise.py` / `timestep_sampling.py` / `loss_weighting.py`
  stay in their own files; no `training/math_utils.py` is created. The three
  currently share no code — YAGNI.

## Pre-kickoff freeze state (2026-05-14)

- No in-flight training-side changes on dev (the user has already paused
  them); PR-A can start immediately
- Workspace files `_pr49_train.py` / `_tmp_pr49.diff` / `pr49.diff` /
  `pr18_review.md` have been cleaned up (leftovers from the previous
  cherry-pick round)

## Acceptance strategy (confirmed)

- **Merge threshold for each PR into dev**: unit tests fully green +
  `python runtime/anima_train.py --help` diffs to empty
- **After PR-C completes**: the user runs a full LoRA training locally +
  evaluates the results (not just loss, but whether generated image quality
  has drifted)

---

## Follow-up extension log

After ADR 0003's three PRs (PR #56/#57/#58) merged, plugin subpackages
outside the original plan have been added as needed. This section logs those
extensions, to validate whether the ADR's "3-4 local steps to add a variant"
promise holds up in real new-addition scenarios.

### Extension 1: `timestep_samplers/` — PR #63 / PR #66 (2026-05-15)

**Trigger**: PR #63 introduced InfoNoise (an I-MMSE adaptive timestep
sampler); the original implementation hard-wired an `ctx.info_noise` field
onto `TrainingContext`, with loop.py guarding it in three places with
`if ctx.info_noise is not None`. This violates ADR 0003's plugin principle:

- "The next person who wants to add a Min-SNR-aware sampler will copy the
  hard-wire pattern" — exactly the anti-pattern warned about in ADR 0003's
  "Case 1 DoRA" section
- A second adaptive sampler would either need another `ctx.min_snr_sampler`
  field, or be turned into a dispatch chain — **both are bad**

**Structure PR #66 landed**:

```
training/timestep_samplers/
├── protocol.py        ← TimestepSamplerProtocol (1 required + 3 optional hooks, runtime_checkable)
├── baseline.py        ← BaselineTimestepSampler wraps the existing sample_t's 4 modes
├── infonoise.py       ← InfoNoiseScheduler + build
└── __init__.py         ← BUILDERS + build_timestep_sampler
```

Interface design (modeled on `AdapterProtocol`):

```python
class TimestepSamplerProtocol(Protocol):
    def sample(self, bs: int, device) -> Tensor: ...               # required

    # optional hooks; non-adaptive samplers (baseline) default to no-op
    def record(self, t: Tensor, raw_mse: Tensor) -> None: ...      # lets adaptive samplers collect stats
    def maybe_refresh(self, global_step: int) -> None: ...         # lets adaptive samplers periodically update their distribution
    def status(self) -> dict: ...                                  # for wandb monitoring
```

**Caller-side simplification** (the three `if` guards in loop.py disappear):

```python
t = ctx.timestep_sampler.sample(bs, ctx.device)           # unified interface; baseline goes through this too
...
ctx.timestep_sampler.record(t.detach(), _raw_mse)         # no-op for baseline
...
ctx.timestep_sampler.maybe_refresh(ctx.global_step)       # no-op for baseline
```

**Deviation from the original ADR**: this is the 5th plugin subpackage; the
original ADR's target layout only listed 4 (adapters / optimizers /
schedulers / inference_samplers). This is a normal "add a new dimension as
needed" extension, not a violation of the ADR.

**Dispatch strategy differs**: `build_timestep_sampler` uses a **bool-switch
dispatch** (`args.infonoise_enabled`), rather than the `Literal` dispatch
used by the other 4 plugins. This is intentional:

| Dispatch style | When to use | Example |
|---|---|---|
| `Literal[...]` + fully mutually exclusive | The user picks one from a set (mutex), and each option's meaning is clear | `lora_type` / `optimizer_type` / `lr_scheduler` |
| bool switch + priority chain | Enable conditions are orthogonal, and there may eventually be "baseline fallback + adaptive layered on top" | `infonoise_enabled` |

When a second adaptive sampler (e.g. Min-SNR-aware) is added, revisit this:
if it's mutually exclusive with InfoNoise → switch to `Literal`; if it can be
stacked (e.g. Min-SNR-aware as the baseline, with InfoNoise doing importance
sampling on top of it) → keep the bool switch.

**`validate_schema_consistency()` not yet added**: bool dispatch doesn't
depend on a Literal set — the schema field is a single source of truth, and
no bidirectional check is needed. Add it once it switches to Literal
dispatch.

**When to switch to Literal dispatch**: in principle, once there are ≥3
kinds of adaptive samplers. With only baseline + infonoise currently,
abstracting early would violate YAGNI.

### Evaluation: how well the new structure supports future plugin maintenance

Checking PR #66's `timestep_samplers/` against ADR 0003's "3-4 local steps to
add a variant" promise, for **adding a similar one** (e.g. a Min-SNR-aware
sampler):

| Step | Change | File |
|---|---|---|
| 1. Implement | New `MinSnrAwareSampler` class implementing the protocol + a `build()` factory | `timestep_samplers/min_snr_aware.py` (new file) |
| 2. Register | `BUILDERS["min_snr_aware"] = min_snr_aware.build` | `timestep_samplers/__init__.py` (1 line) |
| 3. Dispatch | `if args.min_snr_aware_enabled: return BUILDERS["min_snr_aware"](args, total_steps)` | `timestep_samplers/__init__.py` (2 lines) |
| 4. schema | `min_snr_aware_enabled: bool` + this sampler's own fields | `studio/schema.py` |

`loop.py` / `phases/optimizer.py` / `context.py` / `loop.py` call sites need
**zero changes**.

Promise kept.

### To-do

- New plugin subpackages get added to `phases/bootstrap.run()`'s call to
  `validate_schema_consistency()` only once that subpackage switches to
  Literal dispatch (currently `timestep_samplers/` doesn't call it)
- If more plugin subpackages are added in the future (loss_weighters /
  noise_schedulers, etc.), log them following this section's pattern
- A single PR doesn't require "bit-for-bit identical loss over 100 real-model
  steps" — the cost is too high and fp non-determinism could produce false
  positives. Ground-truth verification relies on the final end-to-end
  training run
