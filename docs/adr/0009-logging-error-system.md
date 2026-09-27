# 0009 — Unified logging + error system (0.12.0)

**Status**: Accepted
**Date**: 2026-05-28
**Decision makers**: @WalkingMeatAxolotl (three-way agent review: architect / auditor / migration strategist, two rounds each)
**Landed via**: PR #155 (PR-1 backend infrastructure) / PR #156 (PR-2 error system) / PR #TBD (PR-3 frontend + CLI cleanup)

## Decision

Unify logging and error handling across the 5 surfaces (backend Python / API HTTP / frontend React / subprocess workers / CLI) around:

- **stdlib `logging`** + `concurrent-log-handler` (the only new dependency, for cross-process file locking)
- **ContextVar + HTTP header + subprocess env**, three parallel channels propagating `trace_id` (a 26-character ULID)
- A **`DomainError` base class** + 5 subclasses (NotFound / Validation / Conflict / Auth / Forbidden) in `studio/domain/errors.py`
- **3 `exception_handler`s** doing the unified translation (DomainError / RequestValidationError / Exception fallback)
- **Error response envelope via dual-write**: keep the `{"detail": ...}` legacy contract, add a parallel `{"error": {"code", "message", "trace_id"}}` field, migrated gradually over 3 releases
- **Landing in 3 PRs** (backend infrastructure / error system / frontend+CLI), roughly 72h ≈ 2 working weeks

The main purpose of this ADR: **tell future contributors how to log, how to raise errors, and how to hook a new surface into this system.**

It also closes item 4 of ADR-0008 §"Debt still owed" ("unified exception handler replacing the 4 separate err_code helpers").

---

## Background

### Current state (post-0.11.0)

Logging/error-handling state across the 5 surfaces (agent B's audit found 31 issues: P0 × 4 / P1 × 14 / P2 × 13):

| Surface | Key pain points |
|---|---|
| Backend Python | 21 files call `getLogger(__name__)` + 101 `logger.x` call sites, **0 `basicConfig` / FileHandler**  → INFO is entirely filtered out by root's WARNING level; `except Exception: pass` scattered across 13+ files |
| API HTTP | **0 registered `exception_handler`s**; 4 separate `_err_code` helpers relying on matching source-language strings (e.g. matching "not found" text → 404); 380 router try/except sites repeating the same boilerplate |
| Frontend React | `ErrorBoundary` only does `console.error`, never reports anywhere; **0 `window.onerror` / `unhandledrejection` handlers** |
| Subprocess workers | 4 workers use `print()` as logging (no level / timestamp); malformed `__EVENT__:` payloads are dropped silently; `db.tasks.error_msg` only ever stores `"exit code 1"` |
| CLI | 48 `print()` call sites, no verbose-level control |
| **Cross-surface** | **No trace_id / request_id concept at all** — the full chain triggered by a user action ("router → service → supervisor → worker → SSE → toast") has no ID running through it, making it impossible for on-call to join logs together |

### Motivation

- ADR-0008 §"Debt still owed" item 4 already flagged unifying the exception handler, but doing it alone would once again miss the trace_id lifeline
- After the 0.11.0 refactor, the architecture is now clean — a good time to introduce a cross-cutting system
- When users report problems they can only give a task_id, and developers have to cross-reference 4 different logs (jobs/<id>.log / uvicorn stderr / browser devtools / the daemon ring buffer) to reconstruct the time window

---

## Current structure (after this ADR lands)

```
studio/
├── infrastructure/
│   └── logging.py                # ⭐ New. Single file, ~200 LOC.
│                                 # Exposes: setup_logging / bind_trace_id / get_trace_id / new_trace_id
│                                 #       TRACE_HEADER / TRACE_ENV / PROCESS_ENV
│                                 # Contains: JsonLineFormatter / HumanConsoleFormatter / RotatingFileHandler config
│                                 #       contextvars._trace_id_var / _process_var / _job_id_var / _task_id_var
│                                 #       silence list for third-party libraries (asyncio/urllib3/PIL/...)
│                                 #       takes over the uvicorn.access / uvicorn.error handlers
│
├── domain/
│   └── errors.py                 # ⭐ New. DomainError base class + 5 subclasses.
│                                 # No fastapi dependency, pure Python; services can raise it directly
│
├── api/
│   ├── middleware/
│   │   └── trace.py              # ⭐ New. TraceIdMiddleware (pure ASGI)
│   │                             # reads the X-Trace-Id header → ContextVar → writes it to the response header
│   ├── exception_handlers.py     # ⭐ New. register_exception_handlers(app)
│   │                             # registers 3 handlers: DomainError / RequestValidationError / Exception fallback
│   ├── errors.py                 # Reworked. The old 4 _err_code helpers degrade into thin wrappers (removed incrementally across 8 commits in PR-2)
│   ├── routers/
│   │   └── client_errors.py      # ⭐ New. POST /api/client-errors (frontend ErrorBoundary reporting)
│   └── app.py                    # calls setup_logging("webui") + register_exception_handlers + installs TraceIdMiddleware
│
├── supervisor/
│   └── core.py                   # Reworked: _popen injects ANIMA_TRACE_ID + ANIMA_PROCESS_NAME env vars
│                                 #       _finish_slot tails the last 10 lines of jobs/<id>.log and writes them back to db.tasks.error_msg
│                                 #       the dispatcher calls var.set(task.request_trace_id) when spawning
│
├── workers/
│   ├── _base.py                  # Reworked: worker_main() calls setup_logging("worker:<kind>/<job_id>") at the top
│   │                             #       reads trace_id from the ANIMA_TRACE_ID env var and binds the contextvar
│   │                             #       reconfigure_console_utf8 is folded into setup_logging
│   └── *_worker.py               # print → logger (the __EVENT__: IPC lines are left untouched)
│
├── services/
│   └── inference/daemon.py       # Reworked: _read_stderr_loop thread gets a death watchdog (restarts once or marks STOPPED)
│
├── infrastructure/
│   ├── event_bus.py              # Reworked: _safe_put adds logger.warning on QueueFull (no longer silent)
│   └── migrations/
│       └── _vN_request_trace.py  # ⭐ New migration: adds a request_trace_id TEXT column to the tasks table
│
├── cli.py                        # Reworked: adds a _say(msg, level="info") wrapper; all 48 print sites go through _say
└── web/src/
    ├── components/ErrorBoundary.tsx     # Reworked: componentDidCatch reports to /api/client-errors
    ├── main.tsx                          # Reworked: installs window.addEventListener('error'|'unhandledrejection')
    ├── lib/errors/setup.ts               # ⭐ New. Three-way capture + reportClientError
    ├── lib/errors/report.ts              # ⭐ New. POST reporting (silently swallows failures)
    └── api/client.ts                     # Reworked: req() / xhrUpload() / importPreset consolidated to pick up X-Trace-Id
                                          #       toasts display a "trace ab12cd34" suffix
```

### Data flow across the 5 surfaces

```
┌─ Frontend (browser) ─────────────────────────────────────────────┐
│ ErrorBoundary ─┐                                                  │
│ window.error  ─┼──► reportClientError ──► POST /api/client-errors │
│ unhandledrej. ─┘    (attach lastTraceId from atom)               │
└────────────────────────────────┬─────────────────────────────────┘
                                  │ HTTP + X-Trace-Id header (both directions)
┌─────────────────────────────────▼────────────────────────────────┐
│ FastAPI (process="webui")                                        │
│   TraceIdMiddleware: header → ContextVar → response              │
│   exception_handler(DomainError) → JSON {detail:{...}, error:{}} │
│   logger.x ──► JsonLineFormatter ──┐                             │
└────────────────┬───────────────────│─────────────────────────────┘
                 │ supervisor.spawn   │
                 │ env: ANIMA_TRACE_ID=<id from task.request_trace_id>
                 │      ANIMA_PROCESS_NAME=worker:tag/42            │
                 ▼                    ▼
┌──────────────────────────┐    ┌───────────────────────────────────┐
│ Worker subprocess         │    │ logs/studio.log                  │
│  setup_logging("worker..")│    │  {ts, level, process, trace_id,  │
│  ContextVar bind          │    │   logger, msg, exc, extra}       │
│  stdout ──► supervisor    │    │  rotated *.1 ~ *.5 (50MB each)   │
│   redirect log_fp         │    └───────────────────────────────────┘
└──────────────┬────────────┘                    ▲
               │ stdout/stderr                    │ written only to jobs/<id>.log
               ▼                                  │ + the supervisor's own logger also writes to studio.log
       logs/jobs/<task_id>.log ───────────────────┘
       (human-readable, feeds the SSE LogTailer + frontend <pre>)
```

Layer dependencies (strictly one-way):

```
api/middleware/  ────►  api/exception_handlers  ────►  domain/errors
       ▲                         │
       │                         ▼
       │              infrastructure/logging
       │                         ▲
       └──── services/ ──────────┘
              ▲
       supervisor/, workers/, cli/
```

`domain/errors` has no fastapi dependency (pure Exception subclasses), so services can raise it without importing api backward.

---

## Key rulings after the three-way review

| Topic | Candidates | Final choice | Reason for rejection |
|---|---|---|---|
| Logging library | stdlib / loguru / structlog | **stdlib + concurrent-log-handler** | loguru/structlog are hard runtime dependencies; subprocess startup overhead + caplog isn't orthogonal to them; a JSON schema is simple enough to hand-write in 50 lines |
| `trace_id` length/format | uuid4 hex[:16] / ULID 26 | **ULID 26** (with a `bg-{ULID}` prefix for background spawns, keeping length consistent) | uuid4 has no lexicographic sort; bg-uuid8 vs ULID inconsistent lengths make grep painful |
| `trace_id` propagation path | header / env / contextvar / db column | **all four at once** | header only reaches the frontend; env reaches subprocesses; contextvar reaches within a process; the db column lets the supervisor's background dispatcher pick it up (request-time ID = spawn-time ID = worker log ID) |
| Error envelope | new schema / keep contract / dual-write | **dual-write** (see §envelope gradual migration) | 3 frontend parsing sites + 11 test assertions depend on `detail`; a new schema would break immediately; header-only would be inconvenient for user screenshots |
| `DomainError` location | api/ / domain/ | **`domain/errors.py`** | services depending backward on api is an anti-pattern; domain/ is pure Exception subclasses with no fastapi |
| Number of subclasses | 5 / 7 / more | **5 core** (NotFound/Validation/Conflict/Auth/Forbidden) | 7 subclasses landing at once would require touching the base in 7 services files, a 200+-line diff; the rest can grow incrementally |
| Worker logging | single-write / dual-write | **single-write for 0.12.0** (worker stdout → redirected by the supervisor) | cross-process file locking on Windows is complex; the supervisor's `_finish_slot` writing back `error_msg` already resolves pain point 1.6; dual-write can evolve in 0.13.x |
| `infrastructure/logging` | single file / subpackage | **single file, ~200 LOC** | split into a subpackage once it exceeds 400 LOC; at the current size a subpackage is ceremony overhead |
| CLI output | print / logger | **keep print + a `_say()` wrapper** | the CLI's short 5-second lifecycle gives little value to persisted logs; users seeing `[studio] ...` in their terminal is cleaner than logger's default format; rewriting the 7 capsys test files to "loose `in` matching" is cheap |
| Third-party logger noise | silence individually / leave alone | **explicit silence list inside `setup_logging`** | not silencing them means stderr noise increases 10x once root level=INFO, and developers would roll it back within an hour |
| uvicorn access log | keep uvicorn's own / take it over | **take it over as a JSON handler** | leaving the access log in its own style disconnects it from business logs, making the "request→service→worker" join impossible |

---

## Error envelope gradual migration

| Phase | release | Backend | Frontend |
|---|---|---|---|
| Phase 1 | 0.12.0 | dual-write: `{"detail": <legacy>, "error": {"code", "message", "trace_id"}}` | prefers reading `body.error.trace_id` for the toast; falls back to `body.detail` |
| Phase 2 | 0.13.0 | every `raise HTTPException` gets a deprecation log; frontend fully migrated to `body.error.*` | removes the `body.detail` parsing path in `client.ts`, leaving only a single `ApiError` |
| Phase 3 | 0.14.0 | handler drops the `detail` key; 11 sites across 5 test files migrated | — |

**Key point**: Phase 1 writes ~30 extra lines (the handler fills both keys at once), giving the frontend toast trace_id visibility 3 releases early.

---

## Implementation plan (3 PRs)

### PR-1: backend infrastructure (~28h / ~6 commits)

| Commit | Content |
|---|---|
| C1 | add the `concurrent-log-handler` dependency; create an empty skeleton for `infrastructure/logging.py` exposing only `make_studio_log_handler` |
| C2 | complete `infrastructure/logging.py`: full `setup_logging` + JsonLineFormatter + HumanConsoleFormatter + third-party silence list + takes over uvicorn handlers + utf8 reconfigure |
| C3 | 3 call sites — `api/lifespan.py` + `cli.py` + `workers/_base.py` — call `setup_logging` |
| C4 | `infrastructure/logging.py` gains ContextVar + Filter; new `api/middleware/trace.py` with TraceIdMiddleware; installed in `api/app.py` |
| C5 | `supervisor/core.py:_popen` injects the ANIMA_TRACE_ID env var; new migration `_vN_request_trace.py` adds the `tasks.request_trace_id` column; the dispatcher reads it and calls var.set; the API endpoint writes request_trace_id when submitting a task |
| C6 | workers/*.py print → logger; the supervisor's `_finish_slot` tails jobs/<id>.log and writes back error_msg (resolves B-1.6); `_on_task_log` switches malformed events to logger.error + emits an SSE warning_event (resolves B-4.4); `services/inference/daemon.py:_read_stderr_loop` gets a thread watchdog + restart (resolves B-4.5); `event_bus._safe_put` adds logger.warning on QueueFull (resolves B-1.5) |

**Safety net beforehand** (a commit before C1):
- `tests/test_error_response_snapshot.py` — locks the shape of the 20 existing 4xx/5xx endpoints
- `tests/test_log_baseline.py` — validates typical `logger.warning` paths via caplog
- `tests/test_worker_event_protocol.py` — locks the `__EVENT__:` IPC prefix-filtering behavior
- `tests/test_cli_stdout_baseline.py` — locks the existing CLI output keywords via capsys
- `tests/test_logging_import_inertia.py` — verifies that after `import studio.infrastructure.logging`, sys.excepthook is still default and the root handler count is still 0

**Regression net**: the full `pytest -q tests/` + `npm test --silent` suite run on every commit.

**Rollback**: each commit can be reverted independently; after reverting C5, `os.environ.get("ANIMA_TRACE_ID")` on the worker side remains safe; the migration is reverted with a reverse migration (column drop only).

### PR-2: error system (~26h / ~5 commits)

| Commit | Content |
|---|---|
| C1 | create `studio/domain/errors.py` with the DomainError base class + 5 subclasses (NotFound/Validation/Conflict/Auth/Forbidden), no fastapi dependency |
| C2 | create `studio/api/exception_handlers.py` and register it in `app.py`: DomainError handler / RequestValidationError handler / Exception fallback handler; dual-write envelope (`{"detail":<legacy>, "error":{"code","message","trace_id"}}`) |
| C3 | 5 service error classes (PresetError/ProjectError/VersionError/CurationError/TrainIOError) get a DomainError base; the 4 old _err_code helpers degrade into thin wrappers |
| C4 | migrate router try/except batches 1+2 (preset / projects/* domain) — remove the `try: ... except XxxError: raise HTTPException(...)` sandwiches, letting the raise flow directly into the handler |
| C5 | migrate router try/except batches 3+4 (curation / training / queue); delete the 4 old _err_code helper files; remove the remnants in `api/errors.py` |

**Key constraints**:
- The DomainError `message` field is standardized to **English**, and the frontend looks up the localized text via `code` in the i18n table (avoiding the trap from ADR-0008 §cross-cutting issue D of matching source-language strings)
- The old HTTPException path keeps its `{"detail": <string>}` shape unchanged (trace_id falls back to the header only); the handler registration order preserves starlette's default behavior so the existing 175 raise sites aren't broken

**Rollback**: each commit can be reverted independently; the hotfix path (some frontend toast parsing breaks a week after merge) is patching `_error_body` to force a fallback to `{"detail": <string>}` (a 30-minute hotfix).

### PR-3: frontend + CLI cleanup (~18h / ~5 commits)

| Commit | Content |
|---|---|
| C1 | create `studio/api/routers/client_errors.py` with `POST /api/client-errors` (rate-limited to 10/min per IP, writes to an independent `client_errors.jsonl`) |
| C2 | create `web/src/lib/errors/{setup.ts, report.ts}`; `main.tsx` installs `window.error` + `unhandledrejection` listeners |
| C3 | rework `ErrorBoundary.tsx::componentDidCatch` to report; the 3 fetch wrappers in `api/client.ts` consolidated to pick up X-Trace-Id; toasts show a `trace ab12cd34` suffix |
| C4 | add a `_say(msg, level="info")` wrapper to `cli.py`; all 48 print sites go through `_say`; startup messages stay on the print path, diagnostic messages go through logger (enabled by `--verbose`) |
| C5 | rewrite the 7 capsys test files to use loose `in` matching; mark ADR-0009 as Accepted; mark ADR-0008 §"Debt still owed" item 4 as "folded into ADR-0009" |

---

## Out of scope (deferred to 0.13.x+)

| Item | Reason for deferring |
|---|---|
| `infrastructure/logging.py` single file → subpackage | currently ~200 LOC; triggered once it exceeds 400; a pure file-move PR, 2h |
| Worker single-write → dual-write (worker directly emits `studio.log`) | `concurrent-log-handler` stability on Windows + OneDrive + Defender needs verification first; the current single-write + supervisor write-back already resolves pain point 1.6; ops can also aggregate via `grep jobs/*.log studio.log` |
| envelope Phase 2/3 (removing the legacy `detail` key) | needs a cross-release deprecation window, earliest 0.13.0 |
| IndexedDB offline reporting retry | silently swallowing a failed frontend report is a reasonable default; the offline scenario is an enhancement |
| True rotation for `jobs/<id>.log` | a single job is on the order of a few MB, manageable; changing `_finish_slot` to delete .log files older than 7 days (GC rather than rotation) is a separate follow-up |
| CLI cmd_run doesn't take over subprocess stdout (B-5.2) | related to nssm/systemd wrapper behavior, not touched in this batch |
| CLI cmd_test has no junit-xml (B-5.3) | a DX feature, not part of the logging system |
| Unifying the 64 scattered `console.error/warn/log` sites on the frontend (B-3.4) | to be done alongside the i18n epic |
| Mixed source-language/English error messages (full resolution of B cross-cutting issue D) | a separate i18n epic; this ADR only standardizes the DomainError message to English + code-based lookup, with message as a fallback |

---

## Rejected alternatives

### A. 12-PR fine-grained split
Agent C's first-round proposal was 8 → 12 PRs. Rejected because: this isn't a 25k-line refactor (cf. ADR-0008's 12 PRs) — it's adding a new cross-cutting system; splitting into too many PRs mismatches the review/merge cadence, leading to repeated merges → repeated rebases. 3 PRs split by theme, each with a clear mental model, merged as a stack.

### B. Envelope hard cutover to `{"error": {...}}`
Agent A's first-round proposal. Rejected because: 3 sites in the frontend's client.ts + 1 in Presets.tsx + 11 sites across 5 test files depend on `body.detail`; a hard cutover would break toasts immediately; dual-write gives a 3-release gradual migration window instead.

### C. Worker dual-write to `studio.log` + `jobs/<id>.log`
Agent A's first-round proposal. Rejected because: cross-process file locking on Windows is complex (`concurrent-log-handler` hasn't been validated on OneDrive-synced paths + with Defender); ops can also aggregate trace_id via `grep jobs/*.log studio.log`; can evolve independently in 0.13.x.

### D. CLI print → logger everywhere
Agent A's first-round proposal. Rejected because: the CLI has a short 5-second lifecycle, so persisted logs have little value; rewriting the 7 capsys test files to use caplog would take ≈4h; users seeing a blob like `2026-05-28 14:32 [INFO] studio.cli:` in their terminal is uglier than `[studio] ...`.

### E. loguru / structlog
Agent A's first-round consideration. Rejected because: they're hard runtime dependencies that pollute requirements; subprocess startup overhead; caplog isn't orthogonal to them (needs a `loguru-caplog` shim); rotation on Windows has the same locking issues as stdlib; a JSON schema is simple enough to hand-write in 50 lines.

### F. `infrastructure/logging/` as a 6-module subpackage
Agent A's first-round proposal. Rejected because: the current ~200-LOC single file holds up fine; 6 files in parallel actually make review harder; splitting later once it exceeds 400 LOC is a zero-risk pure file move.

---

## Consequences

### Benefits

- **A trace_id runs through the whole cross-process chain** — when a user reports a problem with the toast's trace suffix, `jq 'select(.trace_id=="...")' studio.log` reconstructs the full timeline from webui → supervisor → worker → the point where the error was raised, in one command
- **Unified log format across all 5 surfaces** — JSON lines with 10 fixed fields, searchable across surfaces with jq / grep
- **Unified API error responses** — the DomainError system replaces the 4 string-matching helpers, removing 4 helper files
- **Frontend crashes become observable** — ErrorBoundary + window.onerror + unhandledrejection all report to the backend's `client_errors.jsonl`
- **Database error root causes become visible** — the supervisor tails the job log and writes back to `db.tasks.error_msg`, upgrading the UI's Task list from "exit code 1" to a traceback summary
- **Closes ADR-0008 §"Debt still owed" item 4**

### New constraints

- Writing a new router **must not** hand-write `try/except XxxError: raise HTTPException`; raise a DomainError subclass and let the handler catch it
- New service-level errors **must** subclass DomainError (or a further subclass of it)
- New workers **must** call `setup_logging(process="worker:<kind>/<job_id>")` rather than calling `print` directly
- New fetch calls **must not** call `fetch(url)` directly; go through the `apiClient.req()` wrapper to automatically pick up X-Trace-Id
- The DomainError `message` field is **standardized to English** + the frontend looks it up via `code` in i18n (preventing ADR-0008's cross-cutting issue D of source-language string matching from resurfacing through DomainError)
- When adding a new third-party dependency that ships its own logger, **evaluate whether to add it to the silence list** (see the existing list inside `infrastructure/logging.py`)
- The `studio.server` shim is kept permanently, untouched (see ADR-0008 §permanent 4 shims)

### Debt still owed (0.13.x+)

| Item | Trigger condition |
|---|---|
| `infrastructure/logging.py` single file → subpackage | once the file exceeds 400 LOC (expected when an OpenTelemetry / Sentry adapter is introduced) |
| Worker single-write → dual-write (emit `studio.log`) | once cross-process trace lookups become a real ops pain point (feedback from ops) + `concurrent-log-handler` Windows stability is validated |
| envelope Phase 2 (deprecation log) | once the frontend has fully migrated to `body.error.*` |
| envelope Phase 3 (remove the legacy `detail` key) | one release cycle after Phase 2 |
| `jobs/<id>.log` GC after 7 days | a separate ~10-line PR (not full rotation) |
| Full standardization of source-language/English error messages | once the i18n epic starts |
| Consolidating the 64 scattered frontend `console.*` calls | once the i18n epic starts |
| CLI cmd_run / cmd_test taking over stdout | when the wrapper (nssm / systemd) is reworked |

---

## Implementation lessons

Reference these during code review as needed. Full background is in the git log + PR descriptions.

1. **TraceIdMiddleware must be pure ASGI, not BaseHTTPMiddleware** — in starlette
   0.36+, the latter wraps things with anyio.Stream, and ContextVar hops across
   threads have edge cases on Python 3.10+. Pure ASGI reads receive/send directly,
   keeping the ContextVar stable for the lifetime of the request. (PR-1 C5)

2. **The fallback `Exception` handler runs outside ServerErrorMiddleware, where the contextvar has already reset** —
   FastAPI's `app.add_exception_handler(Exception, ...)` is registered on ServerErrorMiddleware
   (outside TraceIdMiddleware); named exceptions like DomainError are registered on ExceptionMiddleware
   (inside, where the contextvar is still available). The fallback path must read
   `request.scope["state"]["trace_id"]` instead — the contextvar isn't reachable there. Unified via a
   `_trace_id_from(req)` helper that prefers scope state and falls back to the contextvar. (PR-2 C2)

3. **ContextFilter must be attached to the handler, not the root logger** — stdlib's `Logger.filter` is only
   called once, at the top of `Logger.handle`; when a child logger propagates up to root, it does
   **not** call `root.filter`, only `root.handlers[*].emit`. A filter attached to a logger is completely bypassed by
   child-logger records. Attaching it to every handler ensures every record picks up ContextVar
   injection as it passes through. (PR-1 C5)

4. **`ANIMA_LOGGING_NO_BOOTSTRAP` env guard + `ANIMA_LOG_DIR` env isolation** — the business entry points
   (api/lifespan / cli.main / workers/_base.worker_main) install a real
   file handler writing into the repo's `studio_data/logs/` when triggered by tests, polluting
   caplog and disk. The conftest session fixture sets two env vars: business setup_logging
   returns early at the top; ANIMA_LOG_DIR falls back to a tmp_path_factory location. The fixture
   testing setup_logging itself lifts these with `monkeypatch.delenv`. (PR-1 C4)

5. **The db.tasks.request_trace_id column lets the dispatcher recover the request-time trace_id** — when the supervisor's
   background dispatcher tick spawns a worker, it is **not** inside an HTTP request context, so the contextvar
   trace_id is None. If it only fell back to a `bg-{uuid}` marking a background trigger, the trace_id
   shown in the user's toast screenshot (request time) wouldn't match the one in the worker log (spawn time), breaking the chain.
   Fix: the API endpoint calls `get_trace_id()` and writes it to the `tasks.request_trace_id` column when submitting a task;
   the dispatcher reads that column when starting the worker and injects it into the worker's env. (PR-1 C6)

6. **Bulk-replacing print → _say with a regex accidentally hit _say itself** — `replace_all "print(f\"[studio] "
   → "_say(f\""` also replaced the `print(f"[studio] {msg}", file=...)` call
   inside `_say`'s own implementation, causing infinite recursion + wrong kwargs. Fix: `_say`'s internal implementation uses
   string concatenation, `print("[studio] " + str(msg), ...)`, instead of an f-string, so it doesn't match the pattern. (PR-3 C4)

7. **route_snapshot.json must be regenerated after adding a new endpoint** — `test_route_snapshot`
   locks the entire route set by method+path+name+type. When adding `/api/client-errors`,
   delete the snapshot file and rerun to generate a new baseline; commit the new baseline into git.
   (PR-3 C1)

8. **Silently swallowing frontend reporting failures is a hard requirement** — ErrorBoundary is already in a
   catch state, so if the reporting call itself fails and throws → a second crash → ErrorBoundary enters
   an infinite loop with itself. `reportClientError` wraps everything in
   `try/catch` (even the `console.warn`); the fetch uses `keepalive:true` so it still tries to send even
   at the moment the tab closes. (PR-3 C2/C3)

---

## References

- Three-agent two-round review docs: `tmp/log_unify_agent_{a,b,c}_{architect,audit,migration,round2}.md`
- Related ADRs:
  - [#0008 studio/ 4-layer refactor (0.11.0)](0008-studio-restructure-0.11.0.md) — §"Debt still owed" item 4 folded into this ADR
- Key files / index (once landed):
  - `studio/infrastructure/logging.py` — logging system entry point
  - `studio/domain/errors.py` — error system entry point
  - `studio/api/exception_handlers.py` — the 3 handler registration points
  - `studio/api/middleware/trace.py` — trace_id entry point
