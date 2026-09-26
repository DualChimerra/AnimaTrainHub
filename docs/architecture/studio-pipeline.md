# Studio Architecture Overview

Cross-cutting concerns: data model, directory layout, SQLite schema, secrets, Sidebar, SSE events, Tagger abstraction, Preset relationships. For the internal module structure of Studio, see [`studio/README.md`](../../studio/README.md).

---

## 1. Pipeline flow

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
│ Create   │ → │ Download │ → │ Curate   │ → │ Preprocess│ → │ Tagging  │ → │ Reg-set    │ → │ Configure/│
│ project  │   │ data     │   │ data     │   │           │   │          │   │ generation │   │ enqueue  │
│ Project  │   │ Download │   │ Curation │   │ Upscale   │   │ Tagging  │   │ Reg-build  │   │ Train    │
│ incl. v1 │   │ project- │   │ version- │   │ version-  │   │ version- │   │ version-   │   │ version- │
│          │   │ level    │   │ level    │   │ level(opt)│   │ level    │   │ level      │   │ level    │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └────────────┘   └──────────┘
                                                        ↓ can loop (create v2 to re-curate/re-tag)
```

Each version maintains its own `train/` `reg/` `output/` `samples/` `monitor_state.json`; `download/` is shared at the project level and is **never deleted** (it is always the "full source of truth"). Preprocessing (upscale + crop + dedupe) happens on the version-level `train/` directory, and its state lives in `versions/{label}/train/manifest.json` (ADR 0010, supersedes ADR 0004). `projects/{id}/preprocess/` only remains as a read-only fallback for legacy projects, used by `ensure_train_manifest` for migration. See §2 for the physical layout.

---

## 2. Physical directory layout

```
studio_data/
├── secrets.json                          ★ Global service config (gelbooru token, etc.)
│                                         studio_data/ is already .gitignored, so this is naturally safe
├── presets/                              ★ Global preset pool
│   ├── train_baseline.yaml
│   └── proj_42_baseline.yaml             A preset pushed back from a version
├── projects/{id}-{slug}/
│   ├── project.json                      title / stage / active_version_id / ts
│   ├── download/                         Project-level shared, full backup
│   │   ├── 12345.png
│   │   └── 12345.json                    Gelbooru metadata, optional
│   ├── preprocess/                       [legacy project remnant] read-only fallback for ensure_train_manifest only
│   │   └── manifest.json                 v0.8 legacy schema; no longer written by new code
│   └── versions/
│       └── {label}/                      ★ label is user-filled: "baseline" / "high-lr"
│           ├── version.json              config_name / stage / note
│           ├── train/                    ★ Both preprocessing output and state live here (ADR 0010)
│           │   ├── manifest.json         {images:{"folder/file":{origin,mtime,size,processed?}}}
│           │   └── 5_concept/            Kohya-style N_xxx
│           │       ├── 12345.png         Copied from ../../../download/; upscale overwrites in place
│           │       ├── 67890_c0.png      Multi-crop derivative: origin=67890.png
│           │       ├── 67890_c1.png      Same origin, different crop box (multi-crop fan-out)
│           │       ├── 12345.txt         Tagging output
│           │       └── 12345.json        Categorized caption (optional)
│           ├── reg/                      ★ Version-level (regenerated when train changes)
│           │   ├── meta.json             {generated_at, source_version, target_count, source_tags, generation_method}
│           │   └── 1_general/
│           │       ├── reg_001.png
│           │       └── reg_001.txt
│           ├── output/                   Training output
│           │   ├── lora_step500.safetensors
│           │   ├── lora_final.safetensors
│           │   └── state_step1000.pt
│           ├── samples/
│           │   └── step500_p0.png
│           └── monitor_state.json        This version's training loss/lr curves
```

> Since v0.8, project / version deletion is a direct `rmtree` (no recycle bin) — there used to be a soft-delete `_trash/`, but with no restore UI and no periodic cleanup it was effectively a hard delete that silently accumulated orphan directories, so it was removed (see [release notes 0.8.0](../../release_notes.yaml)).

**Slug rule**: title converted to ASCII lowercase + hyphens; on conflict, append `-2`, `-3`, etc.
**id**: auto-increment, combined with the slug to form the directory name `{id}-{slug}`.

---

## 3. SQLite schema

The DB lives at `studio_data/studio.db`. Migrations in `studio/migrations/` are applied sequentially (controlled by `PRAGMA user_version`).

```sql
CREATE TABLE projects (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    slug                TEXT UNIQUE NOT NULL,
    title               TEXT NOT NULL,
    active_version_id   INTEGER REFERENCES versions(id) ON DELETE SET NULL,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    note                TEXT
);
CREATE INDEX idx_projects_slug ON projects(slug);

CREATE TABLE versions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id           INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    label                TEXT NOT NULL,             -- baseline / high-lr / ...
    config_name          TEXT,                       -- references presets/{config_name}.yaml
    -- ADR-0007: status + phase are two orthogonal fields (added in v8; stage removed in v9)
    status               TEXT NOT NULL DEFAULT 'preparing',
                         -- preparing | training | completed | failed | canceled
    phase                TEXT NOT NULL DEFAULT 'curating',
                         -- curating | preprocessing | tagging | editing | regularizing | ready
                         -- only meaningful when status=preparing; preprocessing / regularizing can be skipped
    last_failure_reason  TEXT,                       -- when task=failed, comes from task.error_msg
    trigger_word         TEXT NOT NULL DEFAULT '',
    created_at           REAL NOT NULL,
    output_lora_path     TEXT,                       -- filled in with the main artifact once training completes
    note                 TEXT,
    UNIQUE(project_id, label)
);
CREATE INDEX idx_versions_project ON versions(project_id);

CREATE TABLE project_jobs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id          INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    version_id          INTEGER REFERENCES versions(id) ON DELETE CASCADE,
                        -- NULL = project-level (download)
                        -- non-NULL = version-level (preprocess, tag, reg_build, generate)
    kind                TEXT NOT NULL,             -- download | preprocess | tag | reg_build | generate
    params              TEXT NOT NULL,             -- JSON-serialized input parameters
    status              TEXT NOT NULL,             -- pending | running | done | failed | canceled
    started_at          REAL,
    finished_at         REAL,
    pid                 INTEGER,
    log_path            TEXT,                       -- studio_data/jobs/{job_id}.log
    error_msg           TEXT
);
CREATE INDEX idx_jobs_project ON project_jobs(project_id);
CREATE INDEX idx_jobs_status ON project_jobs(status);

-- tasks table (training tasks), contains project_id / version_id foreign keys
```

**Version status progression rules** (ADR-0007 §11.3-B / §11.5-A):

| Dimension | Who advances it | When |
|---|---|---|
| `status` (5-value enum) | supervisor | task start / terminal state: `pending/running/paused → training`; `done → completed`; `failed → failed`; `canceled → canceled` |
| `phase` (6-value enum, only while preparing) | user clicking "Next" in PhaseHeaderNav | validation passes (see ADR §11.5-B for the completion criteria of each phase) → cursor advances |
| `phase` skip | user clicking "Next" (`preprocessing` / `regularizing` can be skipped) | validation passes (preprocessing has no mandatory checks; regularizing requires no reg job running) → cursor jumps straight to the next phase |
| **No automatic rollback** | — | if the user deletes data, the next "Next" validation fails and prompts the user to redo it (§11.5-C) |

**Projects have no stage**: removed by ADR-0007 PR-5. Project-level dataset status (download file count) is derived by a live UI scan (§6.10).

---

## 4. Global service config `studio_data/secrets.json`

```jsonc
{
  "gelbooru": {
    "user_id": "",
    "api_key": "",
    "save_tags": false,                   // whether to also save the booru's own tags
    "convert_to_png": true,
    "remove_alpha_channel": false
  },
  "huggingface": {
    "token": "",                           // not required for the public WD14 model; fill in for private repos or rate limiting
    "endpoint": ""                         // empty = official HF; can paste a self-hosted mirror URL. As of 0.8.2, hf-mirror is temporarily hidden, see docs/todo/hf-mirror-recheck.md
  },
  "joycaption": {
    "base_url": "http://localhost:8000/v1",
    "model": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    "prompt_template": "Descriptive Caption"
  },
  "wd14": {
    "model_id": "SmilingWolf/wd-vit-tagger-v3",
    "local_dir": null,                    // null = models/wd14/{model_id}/
    "threshold_general": 0.35,
    "threshold_character": 0.85,
    "blacklist_tags": []
  }
}
```

The Pydantic model is in `studio/secrets.py`; operated on via GET / PUT `/api/secrets`; sensitive fields (`token` / `api_key`) display as `"***"` on GET, and the client sends `"***"` on PUT to mean "leave unchanged."

The frontend `/tools/settings` form is split into 7 tabs (Dataset / Tagging / Training / Monitor / Test / Page / System); password fields use `<input type="password">`. The System tab includes the webui self-update version card (see [ADR 0002](../adr/0002-webui-self-update.md)) and service restart.

---

## 5. Sidebar and routing

```
┌──────────────────────┐
│  Anima                │
│  lora studio · 0.12.0 │ ← version number is fetched from /api/health, the single source of truth
├──────────────────────┤
│ ▶ Projects            │ /
│   Queue               │ /queue
├──────────────────────┤
│   Tools               │
│ ──────                │
│   Presets             │ /tools/presets
│   Test (Generate)     │ /tools/generate
│   Monitor             │ /tools/monitor
│   Settings            │ /tools/settings
└──────────────────────┘

After entering a project, the sidebar switches to a Stepper (only while under /projects/:pid/*):

┌──────────────────────┐
│ ← Back to project list │
│ Project: Cosmic Kaguya │
│ Version: [baseline ▾]  │ ← VersionTabs
├──────────────────────┤
│ ① Download   ✓        │ /projects/:pid/download
│ ② Curate     ✓        │ /projects/:pid/v/:vid/curate
│ ③ Preprocess (optional) ✓ │ /projects/:pid/v/:vid/preprocess?tool=…  ← moved to version level in v0.12
│ ④ Tag        ✓        │ /projects/:pid/v/:vid/tag
│ ⑤ Edit tags  ✓        │ /projects/:pid/v/:vid/edit
│ ⑥ Reg set (optional) ●│ /projects/:pid/v/:vid/reg        ← current
│ ⑦ Train      ○        │ /projects/:pid/v/:vid/train
└──────────────────────┘
```

Status symbols: ✓ complete / ● in progress / ○ not started (derived from stage + version.stats).

Inside "③ Preprocess," a query param switches between multiple **tool** sub-pages (no step ordering):

- `?tool=overview` — Overview (gallery + multi-select + undo)
- default / `?tool=upscale` — Upscale (ESRGAN and other spandrel models)
- `?tool=crop` — Crop (manual / smart-cluster pre-fill)
- `?tool=inpaint` — Inpaint (placeholder, not implemented)

See [crop design](../design/preprocess-crop-design.md) §5 / §9 for details.

---

## 6. SSE event directory

Reuses `studio.event_bus.bus`:

| type | fields | trigger |
|---|---|---|
| `task_state_changed` | `task_id`, `status`, `project_id?`, `version_id?` | training task status changes |
| `monitor_state_updated` | `task_id`, `state` | anima_train writes monitor_state.json, full state stuffed into the payload |
| `project_state_changed` | `project_id`, `stage` | project stage advances |
| `version_state_changed` | `project_id`, `version_id`, `stage` | version stage advances |
| `job_state_changed` | `job_id`, `project_id`, `version_id?`, `kind`, `status` | download / tag / reg_build / generate / preprocess job |
| `job_log_appended` | `job_id`, `text`, `seq` | worker writes a log line → incremental push to frontend |
| `generate_progress` | `job_id`, `step`, `total_steps` | inference daemon image generation progress |
| `preprocess_progress` | `job_id`, `project_id`, `idx`, `total`, `name`, `status`, `action?`, `succeeded`, `failed`, `skipped` | preprocess_worker finishes upscaling one image → frontend live-refreshes files / progress / disk usage |
| `crop_progress` | `job_id`, `project_id`, `idx`, `total`, `name`, `status`, `n_out?`, `outputs?`, `succeeded`, `failed`, `skipped` | preprocess_worker finishes cropping one image; worker-side throttled to ≥1Hz (done events are batched, skip/fail/first/last always sent) |
| `system_stats_updated` | `cpu`, `gpu`, `mem`, `vram` | `_StatsThread` pushes the Topbar system resource pills on a 2.5s cycle (v0.6) |

The frontend `useEventStream.ts` shares a single `EventSource`; multiple components subscribing does not create duplicate connections.

### 6.1 Worker → Supervisor event marker convention

When a child-process worker wants to emit a custom typed SSE event (rather than just a plain log line), it writes to stdout:

```
__EVENT__:<event_type>:<json_payload>
```

For example:

```python
print('__EVENT__:preprocess_progress:{"idx":5,"total":73,"status":"done"}', flush=True)
```

After `studio/supervisor.py:_EVENT_MARKER` recognizes this prefix:

1. It parses `event_type` and the JSON payload
2. It **automatically injects** `job_id` / `project_id` / `version_id` / `kind` (the worker doesn't need to know these and can't forge them)
3. `bus.publish`es it as a typed SSE event to the frontend
4. This line does **not** go into `job_log_appended` (the frontend log window does not display this kind of internal marker)

Design trade-offs:
- Lighter than building a dedicated IPC (queue / socket / state file) — reuses the existing stdout → log_tail channel
- More reliable than having the frontend grep log text — explicit schema, stable fields
- On parse failure, the supervisor calls `logger.exception` but does not crash; the marker line is dropped and the main flow is unaffected
- Don't put sensitive information in the payload — any client that can see the SSE stream can read it

Who can use this: any child-process worker under `studio/workers/`. Currently only `preprocess_worker` uses it;
the download / tag / reg_build workers still only go through `job_log_appended`. If finer-grained progress is needed in the future,
follow the same convention to add an event type (and remember to also add a row to the event directory table above).

---

## 7. Tagger abstraction

```python
# studio/services/tagger.py
class TagResult(TypedDict):
    image: Path
    tags: list[str]                       # sorted (descending by probability)
    raw_scores: dict[str, float]          # optional: probability per tag

class Tagger(Protocol):
    name: str                              # "wd14" / "cltagger" / "joycaption"
    requires_service: bool                 # False for local ONNX; True for JoyCaption (vLLM)

    def is_available(self) -> tuple[bool, str]: ...
    def prepare(self) -> None: ...
    def tag(
        self,
        image_paths: list[Path],
        on_progress: Callable[[int, int], None] = None,
    ) -> Iterator[TagResult]: ...
```

ONNX-based taggers (WD14 / CLTagger) inherit from `OnnxTaggerBase`, which automatically gives them thread-pool scheduling, GPU EP fallback, and model resolution (local → auto-download from HF). New ONNX taggers register with the tagger registry, and the UI lists them automatically.

`<name>_overrides` is the standard persistence key convention (e.g. `wd14_overrides` / `cltagger_overrides`); the frontend derives fields by tagger name.

---

## 8. Preset pool relationships

```
                  ┌──── presets/ (global pool) ────┐
                  │  train_baseline.yaml      │
                  │  high-lr.yaml             │
                  │  proj_42_baseline.yaml    │  ← name pushed back from a project
                  └──────────────────────────┘
                         ↑               ↓
                   save_as_preset    from_preset
                         │               │
                  ┌──────┴───────────────┴────────┐
                  │  versions/baseline/           │
                  │    config_name = "..."        │
                  └──────────────────────────────┘
```

| Operation | Flow |
|---|---|
| Create version | user picks "fork from preset" or "start from scratch" |
| Fork preset | copy `presets/{name}.yaml` → auto-rename to `proj_{pid}_{label}.yaml`, write back to `presets/` → version.config_name points to it |
| Edit config | goes through `/api/presets/{name}` PUT; the version shares the referenced yaml |
| Push back to preset | `save_as_preset {target_name}` → copies the yaml, **clearing project-specific fields**: `data_dir` `reg_data_dir` `output_dir` `output_name` `resume_lora` `resume_state` |
| Switch to another preset | `from_preset` overwrites version.config_name; the old `proj_*` file is not deleted and can be cleaned up manually |

The list of "project-specific fields" lives in the `PROJECT_SPECIFIC_FIELDS` constant in `studio/services/version_config.py`.

---

## 9. Test suite

| Type | Tool | Scope |
|---|---|---|
| Backend unit | pytest | `projects.py` `versions.py` `services/*` `secrets.py` |
| Backend integration | pytest + TestClient | full coverage of API endpoints (200/4xx paths) |
| Backend process | pytest + fake cmd_builder | supervisor scheduling of project_jobs |
| Frontend unit | Vitest | pure functions in `lib/*` |
| Frontend component | Vitest + RTL | key interactive components (ImageGrid multi-select, TagEditor, Stepper, PreviewXYGrid) |

Entry point:

```bash
python -m studio test    # pytest + vitest
```

---

## 10. Known constraints

| Item | Description |
|---|---|
| Slug is immutable | title can be changed, but once set the slug is fixed, to avoid directory migrations |
| Windows num_workers=0 | multiprocess spawn is prone to crashing; the dataloader worker is forced to run single-process on Windows |
| Single GPU | the training loop does not implement DDP/FSDP; multiple GPUs would require switching training backends (see [ADR 0001](../adr/0001-lokr-via-lycoris-lora.md)) |
| JoyCaption requires the user to start their own vLLM | Studio does not manage the vLLM process on Windows, it only calls it via base_url |
| Manual disk edits by the user | every time a step is entered, the disk is rescanned as the source of truth (no stale cache is maintained) |

---

## 11. Frontend styling conventions

**No global responsive layout for now** — the layout is desktop/wide-screen first. If an individual page/component gets cramped on narrow screens, a simple one-off fix can be added, but it must follow the conventions below so that a future global responsive pass can upgrade things consistently:

- All media queries are centralized in `studio/web/src/styles/responsive.css`; don't scatter them across components
- The breakpoint is uniformly `max-width: 1280px` (below the mid-size laptop threshold); don't invent new breakpoints
- A one-off fix should only touch padding / sizing / show-hide of labels, **not change the layout structure** (structural rework is left for the future global responsive pass)
- Add a dedicated className to the target element (e.g. `.banner-shell` / `.phase-timeline-label`) and target it in CSS via that className, not via global selectors
