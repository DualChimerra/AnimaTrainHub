"""FastAPI app factory (extracted from server.py in PR-5).

Only creates the `app` instance and wires up middleware + lifespan. **Route
registration does not happen here**:
    - Legacy routes still live in `studio/server.py`, decorated onto this
      instance via `@app.get(...)` (PR-5 commit 1 scope; later commits move
      routers over to api/routers/ in batches)
    - New routes go through `api/routers/<name>.py` + `app.include_router(...)`

As long as `studio.server` is imported at least once, all 130 legacy route
decorators register onto this app — starting via `uvicorn studio.server:app`
or `studio.api.app:app` gets you the same FastAPI instance either way.
"""
from __future__ import annotations

from fastapi import FastAPI

from .. import __version__
from .exception_handlers import register_exception_handlers
from .lifespan import lifespan
from .middleware import _SelectiveGZipMiddleware
from .trace_middleware import TraceIdMiddleware
from ..services.tunnel import TunnelAuthMiddleware
from .routers import (
    browse,
    client_errors,
    data_exports,
    events_sse,
    generate,
    graph,
    health,
    installs,
    logs,
    models,
    models_storage,
    presets,
    root,
    runtime,
    samples,
    secrets as secrets_router,
    soup,
    studio_data,
    tag_dictionary,
    tagger,
    tunnel,
    upscalers,
)
from .routers.projects import crud as projects_crud
from .routers.projects import exports as projects_exports
from .routers.projects import curation as projects_curation
from .routers.projects import eval_metrics as projects_eval_metrics
from .routers.projects import eval_samples as projects_eval_samples
from .routers.projects import ingestion as projects_ingestion
from .routers.projects import training as projects_training
from .routers.queue import io as queue_io_router
from .routers.queue import lifecycle as queue_lifecycle
from .routers.queue import outputs as queue_outputs

app = FastAPI(title="AnimaTrainHub", version=__version__, lifespan=lifespan)
# Middleware registration order: in starlette, a middleware registered later
# sits on the outer layer of the stack. TraceIdMiddleware is registered
# first -> it ends up wrapping around GZip -> trace_id is bound before GZip
# runs, so logger.x calls inside the gzip handler can pick it up too.
app.add_middleware(_SelectiveGZipMiddleware, minimum_size=1000)
app.add_middleware(TraceIdMiddleware)
# Phone access: when no tunnel is running this is a no-op, and local requests
# are never gated — only requests that arrived through Cloudflare are.
app.add_middleware(TunnelAuthMiddleware)
# ADR-0009 PR-2 C2: register 3 exception handlers (DomainError / RequestValidation /
# Exception fallback) — dual-write envelope keeps the detail contract while adding
# a structured error field. HTTPException is not registered, so starlette's default
# handler keeps the shape of the existing ~175 raise sites unchanged.
register_exception_handlers(app)

# Router registration order doesn't matter (FastAPI matches paths exactly;
# include_router order only affects the ordering of catch-alls with
# include_in_schema=False). Sorted by PR / alphabetically for easier review.
# PR-5 commit 2: health / presets / browse / events_sse
app.include_router(health.router)
app.include_router(presets.router)
app.include_router(browse.router)
app.include_router(events_sse.router)
# ADR-0009 PR-3 C1: frontend error reporting (ErrorBoundary / window.onerror / unhandledrejection)
app.include_router(client_errors.router)
# PR-6 commit 1: 5 small routers (root / samples / logs / data_exports / tagger)
# This fork: the announcements banner and self-update (system) were removed
# along with the in-app updater.
app.include_router(root.router)
app.include_router(samples.router)
app.include_router(logs.router)
app.include_router(data_exports.router)
app.include_router(tagger.router)
# PR-6 commit 2: admin router (secrets / models / upscalers). jobs router removed (merged into the R-5 ledger; data jobs now go through /api/queue)
app.include_router(secrets_router.router)
# This fork: Colab / Local run mode (frontend first-screen picker + settings toggle)
app.include_router(runtime.router)
app.include_router(models.router)
app.include_router(upscalers.router)
app.include_router(tag_dictionary.router)
# PR-6 commit 3: installs router (10 routes: wd14/torch/flash-attn/xformers/llm-tagger admin)
app.include_router(installs.router)
# studio_data migration + unified model source management (#446). This fork
# does not include the system router (self-update).
app.include_router(studio_data.router)
app.include_router(models_storage.router)
# PR-6 commit 5: generate router (8 routes: image generation + daemon status + TAEFlux)
app.include_router(generate.router)
app.include_router(soup.router)
app.include_router(graph.router)
app.include_router(tunnel.router)
# PR-6 commit 6: 3 files in the queue subpackage (lifecycle 12 + io 3 + outputs 5 = 20 routes)
# Registration order: io must come before lifecycle (FastAPI matches paths in
# definition order; otherwise the "export" / "import" strings would get
# intercepted by `/api/queue/{task_id}`'s integer parsing and 422)
app.include_router(queue_io_router.router)
app.include_router(queue_lifecycle.router)
app.include_router(queue_outputs.router)
# PR-6.5 commit 1: first cut of the projects/versions CRUD subpackage (16 routes)
app.include_router(projects_crud.router)
# ADR-0011: eval sample runs + manual run trigger (task-scoped)
app.include_router(projects_eval_samples.router)
# ADR-0011: metric result endpoints and concrete metric runner triggers
app.include_router(projects_eval_metrics.router)
# PR-6.5 commit 2: train.zip / bundle.zip / export-bundle / import-bundle (path/upload) /
# import-train (6 routes)
app.include_router(projects_exports.router)
# PR-6.5 commit 3: download/upload + preprocess (14 routes)
app.include_router(projects_ingestion.router)
# PR-6.5 commit 4: files/thumb + curation + duplicates (12 routes)
app.include_router(projects_curation.router)
# PR-6.5 commit 5: tag + captions + reg + reg_ai + version_config + queue training + version_thumb (23 routes)
app.include_router(projects_training.router)
