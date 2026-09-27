"""Studio business services — split into 11 topic sub-packages since 0.11.0 (see ADR-0008).

Each submodule follows a "pure functions + dataclass" style, called by workers or the
server side. No db dependency; db operations happen at the call site.

## 11 sub-package overview

| Sub-package | Contents |
|---|---|
| `tagging/`     | wd14 / cltagger / llm_tagger / joycaption / caption_format / caption_snapshot / onnx_tagger_base / tagger (factory) |
| `booru/`       | gelbooru / danbooru HTTP API (api.py) + connection pool + token-bucket rate limiting (pool.py) + per-project image download (downloader.py) |
| `reg/`         | reg dataset build main flow (builder.py, 742 lines) + pure analysis/scoring functions (analysis.py, 424 lines) + post-process cluster pruning (postprocess.py) |
| `inference/`   | LoRA metadata + apply (inference_core.py) + long-running daemon (inference_daemon.py) + test-generation cache (generate_cache.py) + upscaling (upscaler.py) |
| `models/`      | model downloader split into 4 files: catalog / paths / sources / downloader (PR-3.8) |
| `preprocess/`  | preprocess main flow (core.py) + duplicate finder (duplicates.py) + manifest (manifest.py) |
| `projects/`    | project CRUD (projects.py) + version (versions.py) + phase state machine (versions_phase.py) + curation (curation.py) + project_jobs (project_jobs.py) |
| `dataset/`     | dataset scan + browse + thumb_cache + task_snapshot + presets_io |
| `presets/`     | preset fork / save-as flow (io.py + __init__.py provide fork_preset_for_version / save_version_config_as_preset) |
| `runtime/`     | runtime install classes: onnxruntime_setup / torch_setup / flash_attention_setup / xformers_setup / pending_install / updater |
| `data_io/`     | train.zip / bundle.zip import-export (train_io.py) |

## shim compatibility (since PR-3)

Some legacy flat paths keep a sys.modules alias shim (e.g. `model_downloader` → the
`models` package) to keep `from studio.services.X import Y` working. The tagger family
keeps no shim — new code imports directly from `studio.services.tagging import X`, and
the factory also loads by sub-package path.
0.11.1+ will remove the remaining shims batch by batch, per sub-package (see the
0.11.0 ADR-0008 follow-ups).
"""
