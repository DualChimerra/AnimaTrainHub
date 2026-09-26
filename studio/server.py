"""AnimaStudio FastAPI server -- PR-8 final shim.

server.py used to be a single 4657-line file holding every router, the
lifespan, middleware, and helper functions. PR-5..PR-6.5 moved all the
routes, BaseModels, and helpers into the `studio/api/` subpackage; the
PR-8 final shim is down to 30-odd lines:

- `app` -- the FastAPI instance (the real one lives in `studio/api/app.py`),
  with all routers registered there via `include_router`. External code doing
  `from studio.server import app` gets that same object.
- `main` -- the uvicorn entry point (the real one lives in `studio/api/main.py`).
- `HTTPException` -- kept for `pytest.raises(server.HTTPException)` compat.
- 6 path constants -- test fixtures still expect `monkeypatch.setattr(server, X)`
  to work; re-importing here keeps old fixtures from breaking (the new
  location is also patched -- see the PR-5/6 descriptions).
- SPA mount -- depends on a runtime `WEB_DIST.exists()` check; kept here
  because moving it to `api/app.py` would push that check to module
  import-time (before lifespan), changing behavior.

Entry points:
    uvicorn studio.server:app                # old cli.py / dev mode
    python -m studio                         # cli.py default run
    python -m studio.server [--host ...]     # start the server directly (main())
"""
from __future__ import annotations

# kept for pytest.raises(server.HTTPException) compat
from fastapi import HTTPException  # noqa: F401

from . import db  # noqa: F401  test fixtures: monkeypatch.setattr(server.db, "STUDIO_DB", X)
from .api.app import app
from .api.main import main
from .api.static import SPAStaticFiles
from .paths import (
    LOGS_DIR,          # noqa: F401  test fixture monkeypatch path
    OUTPUT_DIR,        # noqa: F401
    REPO_ROOT,         # noqa: F401
    STUDIO_DB,         # noqa: F401
    USER_PRESETS_DIR,  # noqa: F401
    WEB_DIST,
)


# SPA mount — see module docstring for why this stays in server.py.
# ADR 0012: mounted at root `/` (no more /studio sub-path). Bare `/` is
# handled at request time by root.py's route (returns a JSON hint when dist
# is missing); this mount only covers /assets/* static files and the
# react-router deep-link fallback (returns index.html when SPAStaticFiles
# misses and the path isn't under the api/samples namespace). Registered
# after all routers so explicit /api and /samples routes take priority.
if WEB_DIST.exists():
    app.mount(
        "/",
        SPAStaticFiles(directory=str(WEB_DIST), html=True),
        name="studio",
    )


if __name__ == "__main__":
    main()
