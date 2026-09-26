# Third-Party Notices

This repository includes, adapts, or derives from third-party code and implementation
snippets, as well as several algorithm ports based on public papers/engineering
implementations. Please comply with the relevant licenses and retain the required
copyright and license notices when distributing.

---

## Repository origin

### Moeblack / AnimaLoraToolkit

- **Source**: [`Moeblack/AnimaLoraToolkit`](https://github.com/Moeblack/AnimaLoraToolkit)
- **Relationship**: this repo's fork origin — the core training scripts and the early
  anima_train entry point are derived from this project, with substantial refactoring
  since. The "Upstream & acknowledgments" section of CLAUDE.md / README gives a
  high-level pointer; the codebase no longer keeps per-file attribution.

---

## Models and weights

### circlestone-labs / Anima

- **Source**: [`circlestone-labs/Anima`](https://huggingface.co/circlestone-labs/Anima)
- **Relationship**: the main diffusion model + VAE. **Model weight licensing is
  separate** (includes Non-Commercial and other restrictions) — governed by the
  HuggingFace model card terms; this repo's NOTICES does not restate them.

---

## Code ports / algorithm implementations

### ComfyUI (GPL-3.0)

- **Source**: [`comfyanonymous/ComfyUI`](https://github.com/comfyanonymous/ComfyUI) (now maintained by Comfy-Org)
- **License**: GPL-3.0
- **Files involved**:
  - `models/anima_modeling.py` — implementation structure closely related to ComfyUI's `comfy/ldm/anima/model.py`
  - `runtime/training/inference_samplers/er_sde.py` — `sample_er_sde` + `default_noise_sampler`
    reference ComfyUI's `k_diffusion_sampling` (with the model_patcher dependency removed)
  - `runtime/training/sampling.py` — `_time_snr_shift` / `_flow_sigmas_simple` /
    sample helpers align with ComfyUI's `ModelSamplingDiscreteFlow` + KSampler behavior

> Because it includes/derives from GPL-3.0 code, this project as a whole is released under GPL-3.0 (see `LICENSE`).

### NVIDIA Cosmos (Apache-2.0)

- **Source**: NVIDIA-related implementations (files include an SPDX header)
- **License**: Apache-2.0 (see the file header `SPDX-License-Identifier: Apache-2.0`)
- **Files involved**:
  - `models/cosmos_predict2_modeling.py`
  - `models/anima_modeling_core.py`

This repo additionally provides `LICENSE-APACHE` to distribute the Apache-2.0 license text.

### Alibaba Wan2.1 VAE (please re-verify the upstream license)

- **Source**: the VAE implementation from [`Wan-Video/Wan2.1`](https://github.com/Wan-Video/Wan2.1)
  (corresponds to `wan/modules/vae.py`)
- **Files involved**:
  - `models/wan/vae2_1.py`

This file's header currently only contains a copyright notice (no explicit SPDX). The
upstream repo generally claims Apache-2.0, but we recommend you **re-check the upstream
repo's LICENSE/NOTICE** before open-sourcing, to ensure distribution compliance.

### ostris / ai-toolkit — Automagic optimizer and 8-bit lr_mask (MIT)

- **Source**: [`ostris/ai-toolkit`](https://github.com/ostris/ai-toolkit) — Ostris (Jaret Burkett)
- **License**: MIT — Copyright (c) 2024 Ostris, LLC
- **Files involved**:
  - `utils/optimizer_utils.py`
    - `class Auto8bitTensor` — 8-bit quantized tensor wrapper (per-tensor int8 + scale)
    - `class Automagic` — sign-agreement -> per-parameter `lr_bump` scheduling + Adafactor
      factored 2nd moment + RMS clip
    - `_copy_stochastic` / `_copy_stochastic_bf16` / `_stochastic_grad_accumulation`
      — stochastic-rounding helper functions (the grad-accum hook is disabled by default,
      matching upstream's already-commented-out behavior; see the comment above
      `class Automagic.__init__` for details)
- **Modifications**:
  - the bf16 path uses Kahan compensated summation (`state['shift']`), borrowed from the
    downstream `tdrussell/diffusion-pipe` port of the same name, rather than upstream's
    original stochastic rounding
  - the `paramiter_swapping` feature was not ported

The original MIT license block is included above `class Auto8bitTensor` / `class
Automagic` in `utils/optimizer_utils.py` — do not remove it.

### tdrussell / diffusion-pipe — Automagic bf16 Kahan path

- **Source**: [`tdrussell/diffusion-pipe`](https://github.com/tdrussell/diffusion-pipe)
  `optimizers/automagic.py` (a port of the same name from ostris/ai-toolkit, with a bf16
  Kahan improvement)
- **Relationship**: the bf16 Kahan compensated summation path of this repo's Automagic
  implementation (`state['shift']` accumulation + `p.add_(shift)` +
  `shift.add_(grad.sub_(p))`, the classic Kahan sequence) matches diffusion-pipe; the rest
  of the algorithm core comes from upstream ai-toolkit.

### Lion optimizer (research attribution — self-implemented)

- **Paper**: Chen et al. 2023, *Symbolic Discovery of Optimization Algorithms*,
  [arXiv:2302.06675](https://arxiv.org/abs/2302.06675) (Google Brain)
- **Reference implementations consulted** (for cross-checking only, no code copied directly):
  - [`google/automl/lion`](https://github.com/google/automl/tree/master/lion) (Apache 2.0)
  - [`lucidrains/lion-pytorch`](https://github.com/lucidrains/lion-pytorch) (MIT)
- **Files involved**:
  - `utils/optimizer_utils.py` `class Lion` / `create_lion`

`class Lion` is a self-implementation (~50 lines), rewritten from the paper's Algorithm 1
without directly copying reference code, so no license attribution is required; the paper
citation + reference URLs are noted in the docstring as academic courtesy.

### nikhilvyas / SOAP — SOAP optimizer (MIT)

- **Source**: [`nikhilvyas/SOAP`](https://github.com/nikhilvyas/SOAP) — Nikhil Vyas
- **Paper**: Vyas et al. 2024, *SOAP: Improving and Stabilizing Shampoo using Adam*,
  [arXiv:2409.11321](https://arxiv.org/abs/2409.11321)
- **License**: MIT — Copyright (c) 2024 Nikhil Vyas
- **Files involved**:
  - `utils/soap_optimizer.py` `class SOAP` — Adam-in-Shampoo-eigenbasis update
    (`_project` / `_project_back` / `_orthogonal_matrix(_qr)` / `_update_preconditioner`)
    derived from the official reference implementation
  - `runtime/training/optimizers/soap.py` — registry wiring (this repo's code)
  - `utils/optimizer_utils.py` `create_soap` — factory shim (this repo's code)
- **Modifications**: optimizer state is fixed at fp32 (for numerical stability in bf16
  LoRA/LoKr training); added `precond_in_state=False`, which excludes the recomputable
  GG/Q from the state_dict to keep checkpoints small + allow a cold rebuild on resume.
  The original MIT license block is included at the top of `utils/soap_optimizer.py` —
  do not remove it.

### facebookresearch / schedule-free — Schedule-Free mechanism (Apache-2.0, research attribution)

- **Source**: [`facebookresearch/schedule-free`](https://github.com/facebookresearch/schedule-free)
  `AdamWScheduleFree`
- **Paper**: Defazio et al. 2024, *The Road Less Scheduled*,
  [arXiv:2405.15682](https://arxiv.org/abs/2405.15682)
- **License**: Apache-2.0
- **Files involved**:
  - `utils/soap_optimizer.py` `class SOAPScheduleFree` — wraps a Schedule-Free trajectory
    around the SOAP preconditioner (drops first-moment momentum, z/x Polyak averaging,
    `train()`/`eval()` weight swap). The SF mechanism is base-optimizer agnostic;
    self-implemented from the paper plus the in-place y/z update and train/eval swap of
    the reference `AdamWScheduleFree`, not a direct code copy; the SOAP preconditioning
    part has the MIT attribution noted above
  - `runtime/training/optimizers/soap_sf.py` — registry wiring + lr_scheduler=none
    validation (this repo's code)
  - `utils/optimizer_utils.py` `create_soap_sf` — factory shim (this repo's code)

### InfoNoise timestep sampler (research attribution — self-implemented)

- **Paper**: *Information-Guided Noise Allocation for Efficient Diffusion Training*,
  [arXiv:2602.18647](https://arxiv.org/abs/2602.18647)
- **Files involved**:
  - `runtime/training/timestep_samplers/infonoise.py` `class InfoNoiseScheduler`
- **Relationship**: self-implemented from the paper's Algorithm 1 (the I-MMSE identity +
  log-sigma bin EMA), without consulting any specific reference code. Anima adapts this
  with `sigma = t/(1-t)` within the Flow Matching `t in (0,1)` space, keeping it aligned
  with the paper's sigma-space design.

---

## Data downloaded at runtime

### DominikDoom / a1111-sd-webui-tagcomplete — tag list (MIT)

- **Source**: [`tags/danbooru.csv`](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete/blob/main/tags/danbooru.csv)
- **Relationship**: the default tag list for autocomplete. It is downloaded on first launch
  into `studio_data/tag_dictionary/` and is not shipped in this repository.

---

## Pip dependencies (licensed as distributed with their respective wheels)

The following dependencies are only pulled in via `pip install`, are not copied into this
repo's source, and this NOTICES file does not restate their licenses. They're listed here
to help with auditing the key algorithm sources:

| Package | License (reference) | Purpose |
|---|---|---|
| `lycoris-lora` | Apache-2.0 | LoRA / LoKr / LoHa / DoRA / rs-LoRA adapter backend (wrapped by `utils/lycoris_adapter.py`) |
| `prodigyopt` | MIT | Prodigy optimizer (`utils/optimizer_utils.py` create_prodigy) |
| `prodigy-plus-schedulefree` | MIT | PPSF optimizer (same file, create_prodigy_plus_schedulefree) |
| `transformers` / `diffusers` | Apache-2.0 | Text encoding / inference helpers / scheduler-form reference (cosine_with_warmup is mathematically equivalent to transformers' `get_cosine_with_min_lr_schedule_with_warmup`) |
| `optimum-quanto` | Apache-2.0 | Automagic `QBytesTensor` quantized base-model compatibility path |
| `safetensors` / `bitsandbytes` / `wandb` etc. | Their respective licenses | — |

---

If you want to relicense this project under something more permissive (e.g. MIT), you'll
need to first remove/replace all GPL-3.0-derived parts (the ComfyUI-related ones) and
re-audit the license compatibility of the third-party dependencies.

