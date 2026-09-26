# 0001 — LoKr adapter goes through lycoris-lora, not a switch to sd-scripts

**Status**: Accepted
**Date**: 2025
**Decision makers**: repo maintainer

## Background

At the time, the repo had its own hand-rolled `LoRALayer` / `LoKrLayer` / `LoRALinear` / `LoRAInjector` (`anima_train.py:875–1100`), a simplified rewrite of the official LyCORIS LoKr that only covered about 30% of the official feature surface. To keep building out the LoRA toolset, more capability was needed (DoRA / dropout / LoHa / multi-GPU, etc.), and there were three directions:

1. **Keep hand-writing it in-repo** — add a class for every new feature
2. **Adopt the official [`lycoris-lora`](https://github.com/KohakuBlueleaf/LyCORIS) package** — keep the anima_train training loop, monitoring, and Studio backend
3. **Switch entirely to [`kohya-ss/sd-scripts`](https://github.com/kohya-ss/sd-scripts)** — retire anima_train, reuse sd-scripts' training core

Constraints:
- Studio is the core of the product (projects / versions / pipeline / SSE monitoring); switching the training backend must not marginalize Studio into "a web GUI for sd-scripts"
- Backward compatibility with old checkpoints is **not** a hard constraint (users can retrain)
- Maintainer bandwidth is limited; the goal is "pip install and go," avoiding a long-term fork to maintain

## Candidate approaches

### Approach A — Keep hand-writing it

Add DoRA / rs-LoRA / dropout / Conv2d / Tucker on top of the existing homegrown implementation.

**Pros**: full control; no new dependency.
**Cons**: every new feature is its own chunk of work; field names don't align with the community ecosystem (ComfyUI / sd-scripts / kohya-ss GUI); the homegrown implementation already has a latent bug (`_find_factor` silently falls back to 1, degrading LoKr into LoRA with an exploded parameter count).

### Approach B — Adopt the official lycoris-lora package

Replace `LoRAInjector` with `create_lycoris(...) + apply_preset(ANIMA_PRESET)`, keeping the training loop, optimizer, monitoring, checkpoint resume, and all other non-adapter logic as-is.

**Already verified feasible**:
- `LycorisNetwork(module, ...)` accepts any single `nn.Module`, doesn't depend on an SD/SDXL pipeline, and accepts a DiT
- `target_name` accepts a list of current layer names (fnmatch + regex)
- `apply_preset({...})` lets you customize layer selection, bypassing the SD presets baked into the PRESET dict
- Interoperates with standard PyTorch `state_dict`; ComfyUI's LyCORIS loader reads it directly
- Dependency footprint: pure Python, ~200KB, only depends on `einops` / `safetensors` / `torch` (already installed)

**Estimated effort**: 4–5 working days (excluding old checkpoint migration)

### Approach C — Switch to sd-scripts

Replace `anima_train.py` with sd-scripts' `anima_train_network.py`, leveraging the kohya team's ongoing updates.

**Current Studio ↔ training coupling surface** (5 file-level interfaces, no function-level calls):
| Interface | File | Current protocol |
|---|---|---|
| Launch command | `studio/supervisor.py` | `subprocess: python anima_train.py --config X.yaml --monitor-state-file Y.json` |
| Training config | `versions/{label}/config.yaml` | `studio/schema.py:TrainingConfig` fields dumped directly |
| Progress state | `versions/{label}/monitor_state.json` | written by anima_train, Studio polls mtime |
| Training logs | `studio_data/logs/task_*.log` | stdout redirected, Studio `LogTailer` appends line by line |
| Sample images | `versions/{label}/output/samples/*.png` | written by anima_train, Studio proxies over HTTP |

Switching to sd-scripts means redefining the protocol for all 5 of these interfaces.

**Scope of change**:
- 🟡 Medium: full rewrite of `schema.py:TrainingConfig` (80+ field mappings); `argparse_bridge` changed to dual-output schema → TOML; `supervisor` command construction + accelerate config generation; entire test suite rewritten
- 🔴 Large: **progress monitoring has to be rebuilt from scratch** — sd-scripts doesn't write monitor_state.json, it only prints tqdm to stdout. Options are a fragile stdout parser (approach A), forking sd-scripts to add a patch (approach B), or upstreaming a PR (approach C). This is the biggest and most failure-prone chunk of work
- 🔴 Medium: checkpoint-resume mechanism rebuilt (the state.pt concept goes away, replaced by accelerate's save_state directory)
- 🔴 Medium: sample image monitoring becomes directory scanning (no more event push)

**Estimated effort**: 3–5 weeks

**Extra capabilities gained**:
- Multi-GPU (accelerate)
- `--blocks_to_swap` VRAM offload
- Adafactor + fused backward
- `--unsloth_offload_checkpointing`
- Multiple timestep sampling modes (sigma / sigmoid / shift / flux_shift)
- Multiple loss functions (l1 / l2 / huber / smooth_l1)
- Per-module rank/lr (`network_reg_dims`)
- Ongoing updates from the kohya team

**What would be lost**:
- `state.pt`'s clean checkpoint-resume semantics
- The rich monitoring fields actively pushed by `update_monitor()`
- The flexibility of owning the training loop (e.g. adding a custom loss/sampler in the future)
- Studio's product positioning as an "end-to-end pipeline"

## Decision

**Approach B**: adopt the official lycoris-lora package, keep the anima_train training loop.

## Rationale

- **Best cost/benefit**: 4–5 days vs 3–5 weeks. lycoris-lora delivers most of the missing features — LoHa / DoRA / dropout / rs-LoRA — while sd-scripts-only capabilities like multi-GPU aren't currently a blocking need
- **Preserves the product shape**: Studio stays an end-to-end pipeline; the monitoring SSE protocol, checkpoint-resume semantics, and training loop are all untouched. Switching to sd-scripts would downgrade Studio from "product" to "a web GUI for sd-scripts"
- **Reversible**: approach B is a working-branch strategy that can be abandoned any time before merge; if problems surface after merging, `git revert` the PR to fall back to the master implementation. **Approach C is not reversible** — once the schema is fully rewritten there's no going back
- **Ecosystem alignment**: weights produced by lycoris-lora are read directly by ComfyUI / sd-scripts / kohya-ss GUI, giving automatic field alignment
- **Light dependency**: lycoris-lora is pure Python, ~200KB, depends only on einops / safetensors / torch (already installed), no new system dependencies
- **Fixes the `_find_factor` latent bug as a side effect**: the homegrown factor search set `[target, 4, 2, 1]` was too narrow — dimensions that don't divide evenly silently fall back to 1, degrading LoKr into a full-matrix LoRA with an exploded parameter count. lycoris's `factorization()` algorithm is more robust

## Consequences

### Already implemented

- `lycoris-lora>=3.0` added as a dependency
- `utils/lycoris_adapter.py` (`AnimaLycorisAdapter`) wraps lycoris calls, replacing the homegrown implementation
- `utils/lycoris_patch.py` patches a `LokrModule.get_weight` rank_dropout device bug in lycoris-lora 3.4.0 (fixed upstream in v0.5.0, see CHANGELOG)
- Schema exposes `lora_algo` / `lora_dora` / `lora_rs` / `lora_dropout` / `lora_rank_dropout` / `lora_module_dropout` and other fields
- ComfyUI loading verified; saved weight prefix `lora_unet_*` aligned with downstream consumers

### New constraints

- Need to track `lycoris-lora` API changes on upgrade; requirements pin `lycoris-lora>=3.0,<4.0` to prevent a major-version upgrade from landing silently
- Old checkpoints are incompatible; loading a checkpoint from the old branch produces a clear error prompting a retrain

### Not yet done (can still be added independently)

- Conv2d / Tucker support (Anima is pure DiT — the backbone, TE, and LLM adapter are all `nn.Linear`, the VAE is frozen — not currently needed)
- Uncommon algorithms like IA³ / GLoRA / BOFT
- Multi-GPU; if genuinely needed in the future, open a separate ADR to evaluate switching to sd-scripts

## References

- Official source: [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)
- Paper: *Navigating Text-To-Image Customization* (ICLR 2024), LoKr covered in §3.3
- Implementation: `utils/lycoris_adapter.py`, `utils/lycoris_patch.py`
- The `attention_backend` consolidation in v0.5.0 (PR #21) builds on this ADR
