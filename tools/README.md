# tools/

One-off scripts, bootstrap helpers, diagnostics, and release / migration tools for the repo
root. All scripts run from the **repo root** (`python tools/xxx.py ...`), and most need the
venv activated first.

> Not here: training / inference code (in `runtime/`), Studio services (in `studio/`),
> subprocesses invoked at production time (in `studio/services/`).

---

## Bootstrap helpers (called by studio.bat / studio.sh, stdlib only)

### `check_requirements_changed.py`
Detects at startup whether `requirements.txt`'s content hash has changed, to decide
whether new dependencies need installing. Uses a content hash rather than mtime, to avoid
false "stale" reads after a `git checkout`.

```
python tools/check_requirements_changed.py             # prints stale / current / missing
python tools/check_requirements_changed.py --update-marker   # writes the new hash after a successful sync
```

### `select_torch_index.py`
On the venv's **first install**, detects the driver version via `nvidia-smi` and prints
the matching PyTorch wheel index URL, so the caller installs torch with the right CUDA
wheel (instead of the PyPI default CPU build).

```
python tools/select_torch_index.py     # detected -> prints the URL; otherwise silently exits 0
```

The driver -> cu wheel mapping is kept in sync both ways with
`studio/services/torch_setup.py:_DRIVER_TO_BEST_CU`.

### `launcher.py` (added in this fork)
Source for the one-click local launcher -- compiled into `AnimaLoraStudio.exe` and just
double-clicked. Does the same thing as `studio.bat` (find the repo -> create/reuse the
venv -> install torch for the GPU -> install requirements -> start
`python -m studio run`), except the user doesn't need to know how to open a terminal
first, and every error path stops and keeps the window open so it can be read.
**Stdlib only**: it has to run before the venv even exists.

```
python tools/launcher.py                 # equivalent to double-clicking the exe
python tools/launcher.py --check         # self-check report (folders / Python / venv / GPU), installs nothing
python tools/launcher.py --reinstall     # delete and rebuild the venv (studio_data/ untouched)
python tools/launcher.py --port 8800     # unrecognized args are passed through as-is to `python -m studio run`
```

### `build_launcher.py` (added in this fork)
Uses PyInstaller to compile the above into a single-file executable (~7 MB, doesn't
include torch / the frontend).

```
pip install pyinstaller
python tools/build_launcher.py           # -> dist/AnimaLoraStudio(.exe)
```

PyInstaller can't cross-compile, so the Windows exe can only be produced on Windows --
CI handles it via `.github/workflows/build-launcher.yml` (builds on both platforms +
`--check` smoke test + ships a release when a tag is pushed).

---

## Model / environment setup

### `download_models.py`
Downloads the models + tokenizers needed for Anima / Krea 2 training (a thin CLI shell;
the logic lives in `studio.services.model_downloader`, shared with the Studio settings
page UI).

```
python tools/download_models.py
python tools/download_models.py --family krea2
python tools/download_models.py --family krea2 --variant turbo
python tools/download_models.py --family krea2 --variant raw_fp8    # official fp8 (same for turbo_fp8)
python tools/download_models.py --variant preview3-base
python tools/download_models.py --no-mirror              # force the official HF source
python tools/download_models.py --modelscope             # use ModelScope instead
python tools/download_models.py --skip-main --skip-vae
python tools/download_models.py --output /data/anima
```

### `install_flash_attn.py`
CLI for installing prebuilt flash_attn wheels (shares wheel-selection logic with the
Settings UI via `studio.services.flash_attention_setup`).

```
python tools/install_flash_attn.py            # auto-selects the best wheel
python tools/install_flash_attn.py --url URL  # manually specify one
python tools/install_flash_attn.py --dry-run  # only lists the environment + candidates, installs nothing
python tools/install_flash_attn.py --force    # reinstall even if already installed
```

Exit codes: 0 success / 1 install failure / 2 environment unsupported.

### `validate_local_models.py`
Verifies offline that local models load correctly (sets `HF_HUB_OFFLINE=1` +
`TRANSFORMERS_OFFLINE=1`, tests the T5 tokenizer and the Qwen tokenizer + model
separately; covers the Anima family only, Krea 2's Qwen3-VL is not covered). Looks for
weights under `tools/models/` by default.

```
python tools/validate_local_models.py
```

---

## Diagnostics / benchmarks

### `diagnose_onnx_gpu.py`
Run this to find the root cause when WD14 tagging shows a "CUDA EP silently fell back to
CPU" warning. Run it inside the same venv as studio, and paste the full stdout into the
PR / issue.

```
python tools/diagnose_onnx_gpu.py
```

### `bench_wd14.py`
WD14 tagging performance diagnostics: staged timing (preprocess / session.run /
postprocess) + EP / preload / model / thread-count self-check. Also answers: how many
times faster is measured GPU throughput vs CPU.

```
python tools/bench_wd14.py [<image_dir>] [--n 10] [--model <hf_id>]
```

If no image directory is given, defaults to scanning `studio_data/projects/*/raw_*` for
the most recent batch of images and takes the first N. Logs go to both stdout and
`bench_wd14.log`.

### `bench_gelbooru.py`
Locates gelbooru download speed bottlenecks in three steps: network / upstream rate
limiting / Studio code.

```
python tools/bench_gelbooru.py
```

Credentials are read automatically from `studio_data/secrets.json`; falls back to the
`GELBOORU_USER_ID` / `GELBOORU_API_KEY` environment variables if missing. Logs go to both
stdout and `bench_gelbooru.log`.

### `infonoise_e2e_verify.py`
End-to-end algorithm verification for InfoNoise: runs `InfoNoiseScheduler` through a pure
numpy mock training loop + a closed-form toy mmse function, comparing 4 pivot
configurations (`current` / `fix_last_above` / `fix_paper_c015` / `oracle`) and printing
paper-aligned metrics (c time series / mass distribution / KL-to-target rho / gate
entropy). **No GPU dependency, doesn't train a real model**; used for: end-to-end repro of
algorithm bugs, regression verification before a fix PR, and comparison against the
values reported in paper §5.

```
python tools/infonoise_e2e_verify.py                                # all 96 combinations (4 config x 4 mmse x 3 ga x 2 baseline), 5-15 minutes
python tools/infonoise_e2e_verify.py --quick                        # CI smoke (<1 minute), 4 core combinations
python tools/infonoise_e2e_verify.py --mmse-shape paper_fig4 \
        --config fix_last_above --grad-accum 1                      # a single combination
python tools/infonoise_e2e_verify.py --out-dir tmp/my_run --no-plots
```

Outputs `<out>/report.md` (comparison table + findings + recommendation) + a `log.csv`
per combination + a 4-panel `plots.png` (c time series / mass distribution evolution /
sampled t histogram / final gate shape). See the script's top-of-file docstring for the
design doc; don't monkey-patch the source code (the script implements fix configs by
dynamically overriding `_refresh`).

---

## Release / schema maintenance

### `bump_version.py`
Validates `tag: release` posts under `docs/announcements/`, syncs version numbers, and
derives `CHANGELOG.md` (ADR 0013). Format is in `docs/announcements/README.md`, writing
guidelines are in `docs/announcements/CONTENT-GUIDE.md`. **Does not create posts** --
that's the maintainer's job when writing the release markdown.

```
python tools/bump_version.py validate          # schema-validates the whole yaml
python tools/bump_version.py bump              # reads the yaml's top version, syncs 3 version files
python tools/bump_version.py bump --version 0.6.1
python tools/bump_version.py render-changelog  # only rewrites CHANGELOG.md, doesn't touch the version number
python tools/bump_version.py verify-versions   # checks __init__.py / package.json / package-lock.json drift (used in CI)
```

---

## One-off migrations

### `preset_toml_to_yaml.py`
Recovers preset files that were "fake yaml, really toml" due to a frontend
`downloadCurrentPreset` bug that shipped between **2026-05-21 and 2026-05-24**. The
current frontend now serves the original yaml directly via the server's
`/api/presets/{name}/download` endpoint; this tool exists only to recover historical
downloads.

```
python tools/preset_toml_to_yaml.py broken.yaml          # outputs broken.fixed.yaml
python tools/preset_toml_to_yaml.py --in-place x.yaml    # overwrites + backs up as .toml-bak
python tools/preset_toml_to_yaml.py --output o.yaml x.yaml
python tools/preset_toml_to_yaml.py --no-validate x.yaml # skip schema validation
```

Handles 3 kinds of corruption: multi-line `{...}` -> inline table, empty-value keys ->
dropped, TOML -> parsed with tomllib + validated against TrainingConfig -> written back
as yaml. Python 3.11+ uses the stdlib `tomllib`; 3.10 and earlier need
`pip install tomli`.

---

## `spike/`

Temporary scripts used to validate ADRs (deleted in a cleanup PR once validated).
Currently contains the ADR 0006 pause/resume signal-chain spike; see its own README at
[`spike/README.md`](spike/README.md).
