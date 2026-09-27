# `runtime/training/` -- training pipeline package

Full implementation of the training flow kicked off by `anima_train.py`. ADR
0003 split the original 2901-line single file into this subpackage;
**main()** now only holds phase orchestration:

```python
def main():
    args = parse_args()
    ctx = TrainingContext(args=args)
    phases.bootstrap.run(ctx)
    phases.models.run(ctx)
    phases.dataset.run(ctx)
    phases.text_cache.run(ctx)
    phases.models.finish(ctx)
    phases.optimizer.run(ctx)
    phases.resume.run(ctx)
    loop.run(ctx)
    phases.finalize.run(ctx)
```

Full design in [`docs/adr/0003-anima-train-refactor.md`](../../docs/adr/0003-anima-train-refactor.md).

## Directory structure

```
runtime/training/
├── context.py              ← TrainingContext dataclass (emit / get_next_sample_prompt / handle_interrupt methods;
│                              the text stack is opaque to non-family code via ctx.text_stack)
├── loop.py                 ← main training loop: for epoch / for batch / accumulation / forward / loss / periodic IO
│                              (family-agnostic, forward is dispatched via ctx.family, zero `if family == ...` branches)
├── sample_runner.py        ← run_sample(ctx, prompt, path, ...) helper; sampling dispatched via ctx.family.sample_image
│
├── bootstrap.py            ← deps check / yaml loading / progress bar init (called by phases.bootstrap)
├── cli.py                  ← parse_args / interactive helpers
├── observability.py        ← WandBMonitor + loss curve ASCII / Rich rendering
├── model_loading.py        ← prefix inference / safetensors / path resolution / xformers
├── vae.py                  ← VAEWrapper + load_vae (shared across families; formerly named models.py)
├── sysmem.py                ← VRAM / RAM orchestration: working set trim + RAM/GPU load guardrails (whole-card NVML view)
├── state.py                ← save / load_training_state
├── snapshot.py              ← state snapshot helpers for pause/resume (ADR 0006)
├── dataset.py               ← BucketManager + ImageDataset + subclasses + collate + navit sampler/collate
├── text_cache.py            ← varlen text sidecar / task archive .text-cache prompt bundle protocol and atomic safetensors I/O
├── timestep_sampling.py     ← sample_t used by training steps (logit_normal / uniform / mode);
│                              reused by timestep_samplers/baseline.py
├── noise.py                 ← make_noise (offset + pyramid)
├── loss_weighting.py        ← compute_loss_weight (min_snr / cosmap / detail_inv_t);
│                              note: this is a *loss weight* (a per-step Flow Matching scaling factor),
│                              orthogonal to *loss type* (mse / huber, see losses/)
│
├── families/                ← model family registry (architecture-level; adding a 3rd family = pure additions,
│                              one registration line per registry)
│   ├── protocol.py          ← the ModelFamily nine-method contract + get_family / resolve_family (fail-fast on unknown family)
│   ├── spec.py               ← ModelSpec frozen surface: LatentSpec / TextSpec / SamplingDefaults / capability set / reject flags
│   ├── latent_spaces.py      ← cross-family shared latent space facts like WAN21_F8C16 (single instance)
│   ├── anima/                ← Anima family: family / loader / forward / preset / sampling /
│   │                            text_encoding (Qwen 0.6B + T5) / comfy_qwen / navit / leap / sra_align
│   ├── krea2/                ← Krea 2 family: loader (strict bf16/fp8 loading) / preset (krea2_full, all Linear layers) /
│   │                            sampling (FlowMatchEuler + simple schedule) / text_encoding (Qwen3-VL 12-layer varlen) /
│   │                            quant_fp8 (fp8 dequant forward) / lora_fp8_merge (LoRA merge writeback)
│   └── README.md             ← tour of the three homes + steps to add a family
│
├── phases/                  ← the 7 phases of main(); each run(ctx) mutates in place
│   ├── bootstrap.py          ← yaml + interactive + seed + device + wandb + monitor_state writer +
│   │                            calls validate_schema_consistency on each plugin subpackage
│   ├── models.py              ← path resolve + per-family loading (cached_varlen families load in two steps: run loads VAE/TE,
│   │                            finish loads the DiT + LoRA inject after TE is released) + fp8_base sanity checks + RAM guardrails
│   ├── dataset.py             ← build main set + regularization set + dataloader + VAE roundtrip self-check
│   ├── text_cache.py          ← cached_varlen families pre-scan the final caption + call the family's encoding cache
│   ├── optimizer.py           ← build_optimizer + validate + scheduler + total_steps +
│   │                            build_timestep_sampler + build_loss
│   ├── resume.py               ← init_progress + state recovery + SIGINT + sample prompts + baseline
│   └── finalize.py             ← final LoRA save + cleanup progress + final loss curve + wandb finish
│
└── ── 6 plugin subpackages ── (the key to localizing variants)
    ├── adapters/            ← LoRA variants
    │   ├── protocol.py       ← AdapterProtocol + StepContext
    │   ├── lycoris.py         ← build_adapter for lokr/loha/lora (family preset explicitly injected)
    │   ├── ortho.py / tlora.py ← OrthoLoRA / T-LoRA builder
    │   └── __init__.py        ← BUILDERS dict + build_adapter + validate_schema_consistency
    │
    ├── optimizers/          ← adamw / adamw8bit / automagic / came / lion / prodigy / prodigy_plus_schedulefree / soap / soap_sf
    │   └── __init__.py        ← BUILDERS + VALIDATORS + build_optimizer + validate_optimizer
    │
    ├── schedulers/          ← cosine / cosine_with_restart / cosine_with_warmup ("none" is schema-only, no file)
    │   └── __init__.py        ← BUILDERS + build_scheduler
    │
    ├── inference_samplers/  ← er_sde / dpmpp_3m_sde (used by Anima; Krea 2's Euler lives in its own family's sampling.py)
    │   └── __init__.py        ← BUILDERS + build_inference_sampler
    │
    ├── timestep_samplers/   ← training timestep samplers (introduced in PR #66)
    │   ├── protocol.py        ← TimestepSamplerProtocol (optional token_counts batch context)
    │   ├── baseline.py        ← thin wrapper for sample_t's 4 modes (non-adaptive)
    │   ├── infonoise.py       ← InfoNoise I-MMSE adaptive sampler (arxiv 2602.18647)
    │   ├── krea2_shift.py     ← Krea2 dynamic resolution shift (corrected by per-image token count)
    │   └── __init__.py        ← BUILDERS + build_timestep_sampler
    │
    └── losses/              ← training loss types (mse / huber / ...)
        ├── protocol.py        ← LossProtocol (compute(pred, target, t) → Tensor)
        ├── mse.py              ← F.mse_loss wrapper (default)
        ├── huber.py            ← Huber loss with constant/snr/sigma delta schedule
        └── __init__.py         ← BUILDERS + build_loss + validate_schema_consistency
```

Anima-specific training steps (navit / leap / sra_align), text encoding
(text_encoding / comfy_qwen) and inference sampling (sampling) have moved
into `families/anima/` as part of the multi-model rework; these files no
longer exist at the top level.

## Data flow

```
parse_args()                          ┐
        ↓                              │
TrainingContext(args=args)             │
        ↓                              │
phases.bootstrap.run(ctx)              │  fills device / dtype / output_dir / wandb / monitor
        ↓                              │
phases.models.run(ctx)                 │  regular families load the full model stack; cached_varlen families load VAE / text_stack first
        ↓                              ├─ one-time setup
phases.dataset.run(ctx)                │  fills bucket_mgr / dataset / reg_dataset / dataloader
        ↓                              │
phases.text_cache.run(ctx)             │  cached_varlen families write the per-image sidecar; online families no-op
        ↓                              │
phases.models.finish(ctx)              │  loads the deferred DiT + injector once TE is released (no-op for regular families)
        ↓                              │
phases.optimizer.run(ctx)              │  fills optimizer / scheduler / total_steps / trainable_params
        ↓                              │
phases.resume.run(ctx)                 │  fills progress / live / global_step / sample_prompts;
        ↓                              ┘  runs the baseline sample; registers SIGINT
loop.run(ctx)                          ──  for epoch / for batch (reads+writes almost all of ctx.*)
        ↓
phases.finalize.run(ctx)               ──  final save + cleanup
```

**ctx is the single mutable state bundle**; every phase function has the
signature `run(ctx: TrainingContext) -> None` and mutates fields on ctx
in place. No return values, no `ctx = phase.run(ctx)` pattern.

## Adding a variant: 3-4 local steps

### Adding a new LoRA variant (e.g. T-LoRA / OFT / VeRA)

1. **Algorithm implementation**: write `utils/{variant}_adapter.py` implementing the underlying algorithm class
2. **Registry shell**: write `training/adapters/{variant}.py` with `build(args) -> AdapterProtocol`
3. **Register**: add one line to the `BUILDERS` dict in `training/adapters/__init__.py`
4. **Schema**: add a value to `lora_type: Literal[...]` in `studio/schema.py` + add fields specific to that variant (use `_meta(group, show_when=f"lora_type=='{variant}'")`)

**Zero changes** to `main()` / `phases/models.py` / `loop.py`.

If the new variant needs per-step adjustments to internal structure (T-LoRA
adjusting a mask by sigma_t), implement the `on_step_begin(ctx)` hook; if it
needs to add a regularization term to the loss (OFT's orthogonality
penalty), implement `regularization_loss(ctx) -> Tensor`. LyCORIS uses the
default no-op.

### Adding a new optimizer (e.g. Lion / CAME)

1. **Build wrapper**: write `training/optimizers/{name}.py` with `build(args, params, lr, weight_decay) -> Optimizer`
2. **Optional**: if it has startup-time constraints (PPSF requires `lr_scheduler=none`), add `validate(args)`
3. **Register**: add one line to the `BUILDERS` dict in `training/optimizers/__init__.py` (and to `VALIDATORS` if it has a validate)
4. **Schema**: add a value to `optimizer_type: Literal[...]` + fields specific to that variant
5. **Dependencies**: add the package to `requirements.txt` if needed

### Adding a new lr scheduler (e.g. warmup_cosine / one_cycle)

1. `training/schedulers/{name}.py` with `build(args, optimizer, total_steps) -> LRScheduler`
2. Add one line to the `BUILDERS` dict in `training/schedulers/__init__.py`
3. Add a value to `lr_scheduler: Literal[...]` in the schema + fields specific to that variant

### Adding a new inference sampler (e.g. euler / dpmpp2m)

1. `training/inference_samplers/{name}.py` with `sample(denoise_fn, x, sigmas, **kw) -> Tensor`
2. Add one line to the `BUILDERS` dict in `__init__.py`
3. Add a value to `sample_sampler_name: Literal[...]` in the schema, and add it
   to the matching family's whitelist in `FAMILY_SAMPLING` in
   `studio/domain/common.py` (per-family option filtering and cross-family
   value validation are both derived from that table)

### Adding a new timestep sampler (e.g. Min-SNR-aware / P-Loss-aware)

Slightly different from the other plugin patterns: the current registry
dispatches on a **bool switch** rather than a `Literal` enum, because each
adaptive sampler may have different args / enablement conditions.

1. **Implement**: write `training/timestep_samplers/{name}.py` containing:
   - `class {Name}Sampler` implementing `TimestepSamplerProtocol` (`sample` is
     required; `record` / `maybe_refresh` / `status` overridden as needed)
   - a `build(args, total_steps) -> {Name}Sampler` factory
2. **Register**: add one line to the `BUILDERS` dict in `training/timestep_samplers/__init__.py`
3. **Dispatch**: add an if-branch to `build_timestep_sampler` in the same file
   (checked in priority order via `args.{name}_enabled == True`)
4. **Schema**: add `{name}_enabled: bool` to `studio/schema.py` + fields specific to that sampler

A plain sampler requires **zero changes** to `loop.py` / `phases/optimizer.py`
/ `context.py`. If the algorithm needs per-sample latent token counts, have
the sampler declare `requires_token_counts = True`; the shared loop will
inject the generic batch context via `sample(..., token_counts=...)` -- it
must never branch by family name.

If there are ever >= 3 adaptive samplers, consider refactoring to a
`timestep_sampler_kind: Literal["baseline", "infonoise", "min_snr_aware", ...]`
Literal dispatch + `validate_schema_consistency()`, matching adapters /
optimizers; with only 2 today (baseline + infonoise) this abstraction isn't worth it yet.

### Removing a variant

The reverse: delete the file + the dict line + the schema Literal entry.
`validate_schema_consistency()` catches anything missed, at startup.

## AdapterProtocol hooks: which one to use

```python
class AdapterProtocol(Protocol):
    # 4 required
    def inject(self, model) -> None
    def get_param_groups(self, weight_decay) -> list[dict]
    def save(self, path)
    def load(self, path)

    # 3 optional hooks (no-op by default)
    def on_step_begin(self, ctx: StepContext) -> None
    def regularization_loss(self, ctx) -> Optional[Tensor]
    def excludes_weight_decay(self, name) -> bool
```

| Variant type | Which hook | Example |
|---|---|---|
| Pure weight variant (unchanged after structural setup) | None | DoRA / rsLoRA / PiSSA / VeRA / LoRA-FA |
| LoRA+ different lr per submodule | more groups from `get_param_groups` | LoRA+ B matrix at 16x lr |
| Structure adjusted by sigma_t / step | `on_step_begin(ctx)` | T-LoRA / AdaLoRA / B-LoRA |
| Adds a regularization term to the loss | `regularization_loss(ctx)` | OFT / Ortho-Hydra balance loss |
| Excludes params from weight_decay by name | `excludes_weight_decay(name)` | LoKr's w1 |

`StepContext` is a frozen 5-field dataclass: `global_step / total_steps / epoch / sigma_t / args`.

## Relationship to `utils/`

Dependency direction is **one-way**: `training/` → `utils/`, never the reverse.

```
training/adapters/lycoris.py            ← build shell (family preset injected explicitly by phases.models via ctx.family.lora_preset())
        ↓ import
utils/lycoris_adapter.py                ← algorithm implementation layer
        ↓ import
utils/lycoris_patch.py                  ← patch for an upstream lycoris-lora bug
```

- `training/` knows about "args / TrainingContext / phase / registry"
- `utils/` knows about "algorithms / library APIs / framework patches", and
  **knows nothing** about the training pipeline's existence
- The inference path (`studio/services/inference_core`) can also reuse
  `utils/lycoris_adapter` precisely because it isn't tied to a training context
- The DiT layer-selection rule (LoRA preset) is **family knowledge**, living
  in `families/{anima,krea2}/preset.py`, not in utils/

## Schema<->registry consistency

`phases/bootstrap.run()` calls 4 `validate_schema_consistency()` functions
very early on (model families have their own layer: self-consistency
validation at `families/` registration time + fail-fast in `resolve_family`
for unknown families; consistency between the schema's `model_family`
Literal and the studio side's `FAMILY_ASSETS` / capability matrix is locked
down by an identity assertion in `tests/test_model_family_gating.py`):

```python
from training.adapters import validate_schema_consistency as _va
from training.optimizers import validate_schema_consistency as _vo
from training.schedulers import validate_schema_consistency as _vs
from training.losses import validate_schema_consistency as _vl
_va(); _vo(); _vs(); _vl()
```

Logic: take the `Literal[...]` set for `TrainingConfig.{lora_type,
optimizer_type, lr_scheduler, loss_type}` and compare it against the
corresponding `BUILDERS` key set. A mismatch raises, failing fast at
startup instead of after training has been running for a while.

`schedulers/` is a special case: `"none"` is schema-only and not in
BUILDERS (`build_scheduler` returns None explicitly for it);
`SCHEMA_ONLY_OPTIONS = {"none"}` skips validation for it.

`sample_sampler_name` / `sample_scheduler` are `Literal`s constrained per
family (single source of truth is the whitelist in
`studio/domain/common.py:FAMILY_SAMPLING`): for anima, historical values
outside the Literal are silently merged into the family default (the #256
migration contract), while for krea2 an out-of-family value is a hard
error. Krea 2's Euler doesn't go through the `inference_samplers/`
registry; it lives in that family's own `families/krea2/sampling.py`.

`timestep_samplers/` dispatches via a bool switch (`infonoise_enabled`)
rather than a `Literal`, so it has no schema<->registry consistency check
either. Consider switching to `Literal` dispatch once there are >= 3
adaptive samplers.

## Tests

```bash
# Unit tests directly relevant to training/
pytest tests/test_anima_train_migration.py        # CLI / YAML / parse_args contract
pytest tests/test_anima_generate_xy.py            # sister script `_T.X` access pattern
pytest tests/test_plugin_registry.py              # registry trio + Protocol hooks
pytest tests/test_infonoise.py                    # InfoNoise EMA formula + state machine + factory (includes
                                                  # codifying the paper's Algorithm 1 formula, to guard against P0-2-style regressions)
```

`test_plugin_registry.py` anti-regression assertions: `phases/optimizer.py`
should no longer contain the literal `if optimizer_type == "prodigy"`,
`phases/models.py` should no longer contain `AnimaLycorisAdapter(`,
`sampling.py` should no longer contain `if sampler_name == "er_sde"`.

End-to-end verification relies on the user **running a full LoRA training +
evaluation generation** (ADR 0003 acceptance strategy R2); a single PR is
not required to be bit-for-bit identical.

## Sister script contract

`runtime/anima_daemon.py` / `anima_generate.py` / `anima_reg_ai.py` use
`import anima_train as _T` and then `_T.find_diffusion_pipe_root` /
`_T.load_anima_model` / `_T.load_vae` / `_T.load_text_encoders` /
`_T.sample_image` / `_T.enable_xformers` / `_T.resolve_path_best_effort`; the
multi-model rework added `_T.get_family` / `_T.resolve_family` on top (the
choke point for per-family dispatch; the sampling call surface actually
goes through `family.sample_image`).

These 7 names, plus `parse_args` / `apply_yaml_config` /
`save_training_state` / `load_training_state` used by tests, are all
re-exported at the top level of `runtime/anima_train.py`. Don't break this
contract when changing internals of `training/` -- `tests/test_anima_generate_xy.py` will catch it.

## History + further reading

- [ADR 0003](../../docs/adr/0003-anima-train-refactor.md) -- full design doc + 9 landed variant case studies
- [ADR 0001](../../docs/adr/0001-lokr-via-lycoris-lora.md) -- why adapters go through the lycoris-lora pip package
- [ADR 0002](../../docs/adr/0002-webui-self-update.md) -- the Ctrl+C handler now lives at `ctx.handle_interrupt`, called from within `phases/resume.py:run()`
- [`studio/domain/training.py`](../../studio/domain/training.py) -- `TrainingConfig`'s Literal enums + field `_meta(group, show_when, ...)` for the frontend UI (`studio/schema.py` is a compat shim)

PR #56 / #57 / #58 are the execution record of ADR 0003's three cuts; commit history is clean, rollback is precise.
