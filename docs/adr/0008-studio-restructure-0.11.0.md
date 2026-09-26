# 0008 — studio/ 4-layer refactor (0.11.0)

**Status**: Accepted
**Date**: 2026-05-28
**Decision makers**: @WalkingMeatAxolotl

## Decision

Refactor `studio/` from a flat 25k-line single package into a 4-layer architecture. `server.py` goes from 4657 lines down to 51; 130 `@app` decorators are split across 27 router files; everything else — routes / models / business logic / infrastructure — is split into packages by responsibility.

The main purpose of this ADR: **tell future contributors where code should live.**

---

## Current structure (as of 0.11.0)

```
studio/
├── api/                  HTTP surface (FastAPI)
│   ├── app.py            FastAPI instance + middleware + all include_router calls
│   ├── lifespan.py       startup/shutdown: ensure_dirs / db.init_db / supervisor start-stop / SSE
│   ├── main.py           uvicorn entry point (main())
│   ├── middleware.py     _SelectiveGZipMiddleware
│   ├── errors.py         4 HTTPException helpers (_safe_join_or_400 / _preset_err_code / ...)
│   ├── responses.py      EMPTY_STATE constant + _thumb_response
│   ├── static.py         SPAStaticFiles (react-router fallback)
│   ├── deps.py           helpers shared across routers (_supervisor / _resolve_anima_model_paths)
│   ├── schemas/          inline BaseModels extracted per router
│   │   ├── presets.py / models.py / installs.py / system.py / generate.py
│   │   ├── queue.py / curation.py / ingestion.py / projects.py / exports.py / training.py
│   └── routers/          27 router files
│       ├── health.py / presets.py / browse.py / events_sse.py
│       ├── root.py / samples.py / logs.py / data_exports.py / tagger.py
│       ├── jobs.py / secrets.py / models.py / upscalers.py / installs.py / system.py / generate.py
│       ├── queue/        subpackage: lifecycle (12) + io (3) + outputs (5)
│       └── projects/     subpackage: crud (16) + exports (6) + ingestion (14) + curation (12) + training (23)
│                         └── _shared.py (helpers shared within the projects domain)
│
├── services/             business services (no db dependency; db calls happen on the caller's side)
│   ├── booru/            api + pool + downloader
│   ├── tagging/          wd14 / cltagger / llm / joycaption / caption_format / caption_snapshot
│   │                     / onnx_base / base (tagger factory)
│   ├── reg/               builder + analysis + postprocess
│   ├── inference/        core (LoRA apply) + daemon + cache + upscaler
│   ├── models/           catalog + paths + sources + downloader (split 4 ways in PR-3.8)
│   ├── preprocess/       core + duplicates + manifest
│   ├── projects/         projects + versions + jobs + phase + curation
│   ├── dataset/          scan + browse + thumb_cache + tagedit + uploads
│   ├── presets/          io + fork/save flow
│   ├── runtime/          onnxruntime / torch / flash_attention / xformers / pending_install / updater
│   ├── data_io/          train_io (train.zip / bundle.zip import/export)
│   ├── queue_io.py       queue task import/export
│   ├── task_snapshot.py  freezes config when a task starts
│   ├── version_config.py per-version yaml config CRUD
│   ├── release_notes.py  release_notes.yaml parsing
│   └── system_stats.py   CPU/GPU sampling
│
├── domain/               pydantic models (the source of the frontend schema contract)
│   ├── training.py       TrainingConfig (643-line single class; PR-2 decided not to split into mixins)
│   ├── lora.py           LoraEntry
│   ├── xy_matrix.py      XYAxisSpec / XYMatrixSpec
│   ├── generate.py       GenerateConfig
│   ├── reg.py            RegAiConfig
│   ├── migrations.py     historical field name / field value migrations
│   └── common.py         GROUP_ORDER / AttentionBackend
│
├── infrastructure/       paths / DB / config / logging
│   ├── paths.py          REPO_ROOT and other path constants + safe_join / validate_path_component
│   ├── db.py             SQLite connection + tasks table CRUD
│   ├── migrations/       _v2 ~ _v9 schema migrations
│   ├── secrets.py        all secrets.yaml models + load/save + legacy migration
│   ├── event_bus.py      in-process SSE bus
│   ├── log_tail.py       per-task incremental log reads + monitor state polling
│   ├── argparse_bridge.py derives argparse arguments from pydantic models
│   └── llm_presets.py    loads builtin LLM caption presets
│
├── supervisor/           task scheduling daemon thread (used across layers)
│   ├── core.py           Supervisor main class (1100-line single class; PR-4 decided not to split into mixins)
│   ├── slot.py           _Slot dataclass
│   ├── cmd_builder.py    default cmd builder + monitor_state_path
│   ├── finalizer.py      maps task terminal state → version.status
│   └── process.py        _kill_process_tree
│
├── workers/              4 subprocess entry points (used across layers)
│   ├── _base.py          worker_main + reconfigure_console_utf8 shared template
│   ├── download_worker.py / tag_worker.py / preprocess_worker.py / reg_build_worker.py
│
├── api/__init__.py and 3 other docstrings strengthened for directory navigation (see PR-9)
├── server.py             51-line shim (FastAPI app + main + SPA mount + test fixture re-imports)
├── db.py / paths.py / secrets.py  ~10-line shims each (for test fixture monkeypatch compatibility)
├── cli.py                841-line launcher (python -m studio run/dev/build/test) — to be split in a future 0.11.1
├── __init__.py           __version__ + top-level architecture docstring
├── __main__.py           python -m studio entry point
├── llm_presets/*.json    builtin LLM caption preset data
└── web/                  React frontend (Vite build)
```

**Layer dependency direction (strictly one-way, no reverse dependencies allowed)**:

```
api/  →  services/  →  domain/
                ↓
        infrastructure/
```

`supervisor/` and `workers/` are cross-layer consumers (they don't belong to any one of the 4 layers).

---

## Guide for future development

### Adding a new HTTP route

1. Find `api/routers/<domain>.py` or the `api/routers/<domain>/` subpackage; create it if it doesn't exist
2. Extract all inline `BaseModel`s into `api/schemas/<domain>.py`
3. Route handlers don't call db / filesystem directly; go through `services/`
4. Add `app.include_router(<domain>.router)` in `api/app.py`
5. Helpers shared across routers → `api/deps.py`; helpers within a single domain → `api/routers/<domain>/_shared.py`
6. Tests use the FastAPI TestClient; fixture monkeypatches target the **new module paths** (not `server.X`)

**Route ordering constraint**: FastAPI matches paths in declaration order. `/api/queue/export` must be included before `/api/queue/{task_id}` (otherwise "export" gets parsed as a task_id integer and returns 422). Already codified as a comment in `api/app.py`.

### Adding new business functionality

Add a module under `services/<domain>/`. Rules:
- Pure functions + dataclasses, no dependency on fastapi or a db connection (the db connection is passed in by the caller)
- Failures raise a domain-specific Exception (e.g. `CurationError`), which `api/routers` translates into an HTTPException
- Helpers shared across services → extract into `services/<domain>/__init__.py` or a new `services/<domain>/_shared.py`

### Adding a new data model

Add it to `domain/<name>.py` as an independent pydantic BaseModel. If it's a historical field-name / field-value migration, add it to `domain/migrations.py`.

**Note**: `TrainingConfig`'s 643-line single class **should not be split into mixins** (PR-2 decision — it affects pydantic field declaration order, which would break the frontend schema's field ordering). Add new fields directly to the appropriate existing field group in `domain/training.py`.

### Adding new infrastructure

New path constants / db helpers / event types go into `infrastructure/<appropriate module>.py`. **Don't add new files at the top level of studio/**.

### Adding a new subprocess worker

1. Write `studio/workers/<name>_worker.py`: define `run(job_id) -> int`
2. At the bottom: `if __name__ == "__main__": from ._base import worker_main; worker_main(run)`
3. Don't read/write a db connection directly; go through `studio.infrastructure.db.connection_for()`
4. Logging goes only to stdout (the supervisor redirects it to a log file)
5. The supervisor launches it via `python -m studio.workers.<name>_worker --job-id N`

### Testing

- **Don't** patch `server.X` (except for one of the permanent shim attributes: `db / paths / secrets / OUTPUT_DIR / WEB_DIST / STUDIO_DB / USER_PRESETS_DIR / LOGS_DIR / REPO_ROOT`); patch the actual module imported inside the handler directly
- Fixtures reused across routers → `conftest.py`
- Tests involving the supervisor inject `_StubSupervisor` into `app.state.supervisor` (see `tests/test_studio_queue_endpoints.py`)

### The 4 permanent shims (**do not delete**)

- `studio/server.py` (51 lines) — FastAPI `app` + `main` + SPA mount + `HTTPException` + 6 path constants + `db` re-import for test fixtures
- `studio/db.py` / `studio/paths.py` / `studio/secrets.py` (~10 lines each) — heavily used by test fixtures via `monkeypatch.setattr(server.db, ...)`

These 4 are part of the architecture; removing them would mean changing 30+ test files with a high risk of ImportError, which isn't worth the ROI. The other 45 shims left over from PR-3/7 were all removed in PR-9.

### Naming / convention quick reference

| Topic | Convention |
|---|---|
| New router file | `api/routers/<domain>.py`, paired with `api/schemas/<domain>.py` |
| New business service | `services/<domain>/<feature>.py`, exporting functions + dataclasses |
| New pydantic model | `domain/<name>.py` |
| New path / db / config | `infrastructure/<topic>.py` |
| New worker | `studio/workers/<kind>_worker.py` + `worker_main(run)` |
| Inline BaseModel | extract to `api/schemas/`, handler files don't inline them |
| Cross-router helper | `api/deps.py` (genuinely cross-domain) or `api/routers/<domain>/_shared.py` (within a domain) |
| Exception | business layer raises domain-specific (`CurationError`); router layer translates to HTTPException |
| Error codes | `api/errors.py:_preset_err_code` template: message string → HTTP code |

---

## Implementation lessons (accumulated across PR-1..9)

Reference these during code review as needed. Full background is in the git log + PR descriptions.

1. **`sys.modules` alias shim pattern** — when a package shadows the same name, `_sys.modules[__name__] = _real` makes the old path transparently forward to the new one (PR-3)
2. **Cross-submodule calls go through `module.func()`** — `from .sub import func` binds the name into this module's namespace, so patching `sub.func` in a test has no effect; use `from . import sub as _sub; _sub.func()` instead (PR-3.8)
3. **Fixture patch paths must move with the code** — after a handler moves, the old `monkeypatch.setattr(server, X)` can't see the new location; fixtures need patches added at the new location (came up repeatedly in PR-5/6)
4. **Package shims also use a module file + sys.modules** — `studio/migrations.py` is a single-file proxy for the entire `studio.infrastructure.migrations` package, making submodule access transparent (PR-7)
5. **Moving a level deeper needs an extra `.parent`** — `Path(__file__).resolve().parent.parent` needs an added `.parent` after moving into a subdirectory (PR-7)
6. **`del sys.modules + reimport` pollutes a shared app instance** — re-importing server.py causes `@app` decorators to register routes on the same app twice, doubling the route count (PR-6)
7. **Don't split classes with high internal state coupling** — `TrainingConfig` (pydantic ordering) / `Supervisor` (shared self) stayed as single classes; splitting into mixins would raise the cost of future extension (PR-2/4)
8. **Removing shims: fail loud, not lazy fallback** — don't use `__getattr__` lazy fallback (which hides ImportError); update every caller explicitly to the canonical path (PR-9)

---

## Debt still owed (0.11.1+)

| Item | Reason for deferring |
|---|---|
| Split `cli.py` (841 lines) → 7 files | a single-file launcher is still readable; a separate PR isolates the risk |
| Split `secrets.py` (763 lines) → models/store/migrations, 3 files | risk of cross-file circular imports under Pydantic v2; the 3-way split's benefit is mostly visual separation |
| Split `db.py` (188 lines) → connection/tasks/settings, 3 files | same as above |
| ~~Unified exception handler replacing the 4 separate `err_code` helpers~~ | **folded into [ADR-0009](0009-logging-error-system.md)** (dual-write envelope + DomainError system, 0.12.0) |
| Add a deprecation log to the 170-line legacy migration in `secrets.py` | to be done alongside the 3-way split |

---

## References

- 11 PRs: #141 #142 #143 #144 #145 #147 #148 #149 #150 #151 #152 #153
- Related ADR: [#0003 anima_train.py modular refactor](0003-anima-train-refactor.md) (same kind of runtime-side split)
- Key files / index: the three docstrings in `studio/__init__.py` / `studio/services/__init__.py` / `studio/api/routers/__init__.py` serve as top-level navigation
