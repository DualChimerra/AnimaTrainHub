"""HTTP routers — migrated over from studio/server.py in batches starting at PR-5.

Each file = one domain. api/app.py calls `app.include_router` on all of them at once.

## 17 top-level routers + 1 subpackage

| router file | routes | description |
|---|---:|---|
| `health.py`        | 3   | health / system stats / training monitor state |
| `presets.py`       | 13  | preset CRUD + import/export + schema + configs redirect |
| `browse.py`        | 3   | datasets / browse / dataset thumbnail |
| `events_sse.py`    | 1   | /api/events SSE |
| `announcements.py` | 1   | /api/announcements (announcements bar, derived from docs/announcements/) |
| `root.py`          | 3   | / → SPA index; /studio, /studio/{rest} → / compat redirect (ADR 0012) |
| `samples.py`       | 2   | /samples/{filename} + /api/queue/{task_id}/samples (sample image listing) |
| `logs.py`          | 1   | /api/logs/{task_id} |
| `data_exports.py`  | 1   | /api/data-exports |
| `tagger.py`        | 1   | /api/tagger/{name}/check |
| `jobs.py`          | 3   | /api/jobs/{jid} / log / cancel |
| `secrets.py`       | 2   | secrets read/update |
| `models.py`        | 3   | models catalog / path-defaults / download |
| `upscalers.py`     | 2   | upscaler select / custom download |
| `installs.py`      | 10  | wd14 + torch + flash-attn + xformers + llm-tagger admin |
| `system.py`        | 9   | restart / update / rollback / preflight / dev_commits / init_git |
| `generate.py`      | 8   | test generation + daemon control + TAEFlux |
| `queue/`           | 21  | split internally into lifecycle (13) / io (3) / outputs (5) |
| `projects/`        | 71  | split internally into crud (16) / exports (6) / ingestion (14) / curation (12) / training (23) |

**Total**: about 158 routes (+ 5 non-APIRoute: SPA mount + openapi/docs/redoc etc.). The exact route triple set lives in `tests/_snapshots/studio_routes.json` (snapshot test gate).

## Subpackage-internal helpers

- `queue/__init__.py` — subpackage overview
- `projects/_shared.py` — 8 shared helpers for the projects domain (_project_payload /
  _publish_*_state / _version_dir_or_404 / etc.), imported only inside the projects sub-router

## Include-order constraint

queue/'s io must be included before lifecycle (FastAPI matches paths in the order
routes are defined, otherwise `/api/queue/export` / `/api/queue/import` would get
intercepted by `/api/queue/{task_id}`'s integer parsing and 422).
"""
