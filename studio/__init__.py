"""AnimaStudio — training monitor, config editor, and task queue daemon.

## Top-level architecture (post-0.11.0 restructure, see ADR-0008)

studio/ has 4 layers (dependency direction: top to bottom, **never reversed**):

```
    api/             HTTP surface (FastAPI app + 27 routers + schemas + deps)
      |
    services/        business services (11 subpackages: tagging / booru / reg /
                    inference / models / preprocess / projects / dataset /
                    presets / runtime / data_io)
      |
    domain/          pydantic models (TrainingConfig, 643 lines + LoRA / XY /
                    Generate / RegAi + migrations)
      |
    infrastructure/  path constants / db / event bus / secrets / logging /
                    argparse bridge / migrations
```

`supervisor/` (the task-scheduling daemon thread) and `workers/` (4 subprocess
entry points) cut across layers and don't belong to any one of the 4.

## Entry files

- `server.py` — a 51-line shim that re-exports `app` / `main` for backward
  compatibility with `from studio.server` (the real implementation lives in
  api/app.py / api/main.py)
- `cli.py` — the `python -m studio` launcher (build / run / dev / test subcommands)
- `__main__.py` — the `python -m studio` entry point

## __version__

The single source of truth for the version number across the whole repo:
- the FastAPI app injects it as `app.version`, exposed via `/api/health`
- the frontend Sidebar fetches it from `/api/health` instead of hardcoding it
- `studio/web/package.json`'s version field must be kept in sync manually
- on every release: bump this, add a CHANGELOG.md entry, and sync package.json

Version scheme: MAJOR.MINOR.PATCH (semver, but during the 0.x phase a MINOR
bump is treated as a breaking change).
"""
__version__ = "0.20.2"
