# AnimaStudio

Web workbench for the training pipeline (scrape -> curate -> tag -> regularize -> train -> sample test). Backend is FastAPI + SQLite, frontend is React + Vite.

## Directory layout (ADR 0008 four-layer architecture)

```
studio/
├── api/               # HTTP surface: FastAPI app + routers + schemas + deps + exception_handlers
├── services/          # business service subpackages: tagging / booru / reg / inference / models (incl. families/ model-family asset registry) /
│                      #   preprocess / projects / dataset / presets / runtime / data_io
├── domain/            # pydantic models: TrainingConfig (incl. model_family / capability gating / config_rules) /
│                      #   LoRA / XY / Generate / RegAi / family_switch + migrations
├── infrastructure/    # paths / database / event bus / secrets / logging / argparse bridge / migrations
├── supervisor/        # task-scheduling daemon thread
├── workers/           # background subprocess entry points (download / tag / reg_build / preprocess)
├── server.py          # compatibility shim, re-exports `app` / `main` (the real entry point is api/app.py / api/main.py)
└── web/               # React + Vite frontend source
    ├── src/
    └── dist/          # npm run build output (backend mounts it at the root path /, ADR 0012)
```

Root-level files like `schema.py` / `secrets.py` / `paths.py` are compatibility shims for
the pre-refactor API; the real implementations live in `domain/training.py`,
`infrastructure/secrets.py`, and `services/models/paths.py` respectively.

Runtime data is written to `studio_data/` at the repo root (SQLite + user presets + task
archives), which is in `.gitignore`.

## Starting the app

### Cross-platform launcher (recommended)

`python -m studio` is the unified entry point, with subprocesses managed by Python (the
same commands work on Windows / macOS / Linux):

```bash
python -m studio              # default = run
python -m studio run          # build the frontend (if missing) + start the backend
python -m studio dev          # frontend + backend dev mode (5173 + 8765 --reload, in parallel)
python -m studio build        # build the frontend only
python -m studio test         # run pytest + vitest
```

dev mode starts both a Vite and a uvicorn subprocess; Ctrl+C kills them together (uses
`CTRL_BREAK_EVENT` on Windows, process-group SIGTERM on POSIX).

### Windows shortcut script

`studio.bat` calls the same Python launcher; just double-click it.

### Calling the backend directly

```bash
python -m studio.server --host 0.0.0.0 --port 8765 [--reload]
```

### Frontend

Dev mode (hot reload):

```bash
cd studio/web
npm install            # first time only
npm run dev            # -> http://127.0.0.1:5173/ (/api, /samples proxied to the backend)
```

Production build (output is mounted by the backend at the root path `/`, ADR 0012):

```bash
cd studio/web
npm run build          # outputs to studio/web/dist/
# no need to restart the backend, just refresh the browser (the server checks for dist/ on startup)
```

## Frontend pages

- **Projects** (`/`) -- project list; inside a project, the sidebar switches between Stepper steps (download / curate / preprocess / tag / tag editor / regularization set / training); training config includes a model family (Anima / Krea 2) switch
- **Queue** (`/queue`) -- unified ledger for all task types: cancel / retry / delete / pause-resume / schedule; sectioned + filtered pagination; live SSE refresh; task detail includes logs / monitoring / output
- **Presets** (`/tools/presets`) -- global training preset pool (auto-saved), two-way fork with version config
- **Test** (`/tools/generate`) -- single image / XY matrix / resident inference daemon; base model / TE selection by family, fp8 / VRAM strategy
- **Monitor** (`/tools/monitor`) -- live training loss / lr / sample images
- **Settings** (`/tools/settings`) -- sectioned by tab; the model download center (sectioned by family + single-select variant) lives under the "Training" tab

For the cross-step architecture (data model / SQLite / SSE / secrets / Tagger abstraction / Preset pool) see
[docs/architecture/studio-pipeline.md](../docs/architecture/studio-pipeline.md).
