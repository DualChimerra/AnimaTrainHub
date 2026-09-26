# Project / Version / Task lifecycle refactor (discussion draft)

> **Document status**: this is a two-round discussion draft. **The final decisions
> are in [ADR-0007](../adr/0007-project-version-lifecycle-refactor.md)**.
> Kept as discussion history — §1-§10 are round-one decisions (some later
> overturned by §11), §11 is the round-two convergence, §12 is parked follow-up
> work.
> Implement per ADR-0007, not per this document's first 10 sections.
>
> Branch: `feat/project-queue-relation` · Drafted: 2026-05-22 · Round-two
> convergence: 2026-05-23

---

## 1. Background and motivation

The current `Project.stage` / `Version.stage` state machines (9 enum values
each) have structural problems:

- **The project card lies**: after a training run fails / is canceled /
  paused, `projects.stage` never advances, so the badge permanently shows
  "training" with an animated dot — the user comes back thinking it's still
  running.
- **Dead states**: `curating` / `configured` / `regularizing` (at the project
  level) are never advanced by any code path; 3 of the 9 color mappings are
  never seen by the user.
- **Skipping levels while contradicting itself**: uploading a train image
  jumps straight to `tagging`, skipping 5 stages — which shows that stage is
  really "the set of steps that have been done," but the UI displays it as
  linear progress.
- **No source of truth**: `advance_stage()` is an unconditional setter, the
  rules are scattered across 9 call sites, and Project / Version / Task each
  have their own state machine that never reconcile with each other.
- **Data-model mismatch**: Project is an aggregate (1:N -> Version) and
  shouldn't carry "in-progress state" at all.

After two rounds of debate among PM / user / designer, the consensus was:
**the model itself is wrong at the foundation**, and needs a ground-up
rewrite rather than a patch.

---

## 2. Core refactor principles

1. **Project has no process state.** A project is a container — only
   "exists / doesn't exist" (no soft delete, see §3).
2. **Version is the real "training experiment" entity**, carrying the full
   run-state state machine.
3. **Phase (which steps have been done) and Status (current run state) must
   be separated** — this is the root cause that eliminates the "stage
   skipping levels" contradiction.
4. **Task is fact; Version.status is derived + cached.** The supervisor is
   the only thing that advances it; the UI never writes status directly.
5. **The dataset (download + preprocess) belongs at the project level** —
   multiple versions share the same pool of original images (matches how
   local-tool users actually think about it).

---

## 3. Data model (final decisions)

### Project (minimal, a pure container)

```python
{
    id, title, slug, note, created_at, updated_at,
}
```

**Fields removed**: `stage`, `active_version_id`

**On phase**: the Project table **stores no phase field at all**. Dataset
info (whether download/ is non-empty, image count) is computed live by the
UI scanning the filesystem — this is inherently a derived filesystem state,
and storing a bool would be redundant and drift-prone.

**On a "representative" version / focus / pinning**: **not doing this**.
Reasoning: the metric strip already answers the core question — "what's
running / what failed / what's done." If the user wants to see which
version specifically, they click into the version list — **the card doesn't
need to guess who the "representative" is**. Pinning is even more of a
burden (requires manual action, and picking the wrong one is more confusing
than not having one).
- Default version opened on the project detail page: the most recently
  updated one (purely derived, not stored)
- Project card: metric strip + dataset snapshot + latest-artifact entry
  point, **no focus row**

**On soft delete**: this refactor round **does not** add archive/trashed
states. A project only has "exists / physically deleted." Soft delete can
get its own ADR later if needed.

### Version (the core entity)

```python
{
    id, project_id, label, note, created_at, updated_at,

    # Run status (state machine, see §4.2)
    status: "draft" | "training" | "paused"
          | "completed" | "failed" | "canceled",

    # Step-completion markers (5 independent bools, starting from curation)
    # Note: dataset import/preprocess is project-level; the UI checklist
    # dynamically prepends it as "item 0"
    phases: {
        curated:             bool,   # train image set has been selected for this version
        captioned:           bool,   # caption coverage on train images meets the threshold
        regularization_ready: bool,  # a regularization set exists, or the user explicitly opted out
        training_configured: bool,   # a training config exists and validates
        artifact_produced:   bool,   # output/*.safetensors has landed on disk
    },

    # Derived / cached
    output_lora_path: str | None,
    last_task_id:     int | None,
    last_failure_reason: str | None,
}
```

**Field removed**: `stage`

### Task (keeps `project_id` as a redundant field)

```python
{
    id, project_id, version_id, kind: "train" | "reg_ai" | "generate" | ...,
    status: "pending" | "running" | "paused" | "done" | "failed" | "canceled",
    # ... remaining execution fields unchanged
}
```

**`project_id` is kept**, but is now positioned as **redundant with
`versions.project_id`**:
- NOT NULL (after migration, every task must belong to a version)
- Consistency constraint: `task.project_id == version.project_id`, enforced
  both at the application layer and by a DB CHECK
- Benefit: high-frequency queries (queue page / monitor) need zero joins;
  clicking a task straight into its project works directly
- Cost: write paths must maintain the consistency (the supervisor / API must
  explicitly assign it when creating a task)

### project_jobs (one table, scope distinguished by kind)

Not splitting into separate tables. The `project_jobs` table's schema already
supports a nullable `version_id`:

| kind | scope | version_id | effect on completion |
|---|---|---|---|
| `download` | project | NULL | computed live by the UI (no DB phase) |
| `preprocess` | project | NULL | computed live by the UI |
| `tag` | version | NOT NULL | sets `version.phases.captioned = true` |
| `reg_build` | version | NOT NULL | sets `version.phases.regularization_ready = true` |

Each kind shares the same 5 states: `pending -> running -> done | failed |
canceled` (no paused).

> Naming note: the table name `project_jobs` actually also covers
> version-scoped jobs, so `async_jobs` would be more accurate. But a rename
> means a migration, so it's left alone for now.

---

## 4. State-machine definitions

### 4.1 Project lifecycle

Project has no explicit state machine — only the two life events `created ->
physically deleted`. Everything shown on the card is derived from the
Version aggregate + project-level phases.

### 4.2 Version status (6 states)

```
                  [createVersion]
                        |
                        v
                  +-------------+
              +-->|    draft    | <- merges the old draft + ready + preparing
              |   +------+------+    covers every "not training / not done" prep state
              |          | submit train task
              |          v
              |   +-------------+  pause     +-------------+
              |   |  training   | ---------> |   paused    |
              |   |             | <--------- |             |
              |   +------+------+  resume    +-------------+
              |          |             ^ merges the old queued + training
              |          |
              |          |-- task done -----> +------------+
              |          |-- task failed ---> | completed  | * terminal
              |          `-- task canceled -> +------------+
              |                                +------------+
              |                                |   failed   | * terminal
              |                                +------------+
              |                                +------------+
              |                                |  canceled  | * terminal
              |                                +-----+------+
              |                                      |
              `--------------------------------------'
                  user clicks "retrain" -> back to draft, or straight to training
```

**Merge decisions**:
- `draft + ready + preparing` -> **unified into `draft`**
  - Reasoning: "still in prep" is a single mental model for the user; splitting
    it into sub-states adds no value to them
  - Naming note: `draft` literally leans toward "blank," but actually covers
    "already prepared a lot, just hasn't submitted training yet." What the
    user actually perceives comes from the phases checklist, not the status
    label
- `queued + training` -> **unified into `training`**
  - Reasoning: from the user's perspective it's just "running" — waiting for
    a GPU slot vs. actually running feels the same

**Transition constraints**:
- No going from `completed` back to `draft` (no going backward — protects
  historical fact)
- Starting a new training run from `completed / failed / canceled` is
  allowed -> back to `training`
- `resume` only works from `paused`
- Submitting a new task is **forbidden** while in `training` / `paused` (one
  at a time)
- `status` is advanced by the supervisor; **the UI never writes it directly**

### 4.3 Task status (unchanged)

```
[submit] -> pending -> running -> done | failed | canceled
                       |
                       `- pause -> paused - resume -> running
```

Terminal states: `done / failed / canceled`. Every terminal state fires a
transition event at its owning Version.

### 4.4 project_jobs status (one table, split by kind)

```
[create] -> pending -> running -> done | failed | canceled
```

Effect on completion:

| Job kind | scope | version_id | effect on completion |
|---|---|---|---|
| `download` | project | NULL | computed live by the UI (no DB phase) |
| `preprocess` | version | NOT NULL | modifies physical files under `train/`, computed live by the UI (ADR 0010) |
| `tag` | version | NOT NULL | sets `version.phases.captioned = true` |
| `reg_build` | version | NOT NULL | sets `version.phases.regularization_ready = true` |

### 4.5 Phases (independent bools, not a state machine)

**Storage strategy**: **persisted as DB fields** (decision §6.2). Lives only
on the Version table; Project has no phase field.

**Version phases** (5): `curated`, `captioned`, `regularization_ready`,
`training_configured`, `artifact_produced`

Each phase is an independent bool: unordered, repeatable, and **can be
rolled back**. The UI presents them as a capability checklist.

**Forward writes** (setting true):
- Written by the supervisor when the corresponding job completes
- Written by user action (saving config -> `training_configured = true`,
  curating a selection -> `curated = true`)

**Rollback writes** (setting false) — mandated by decision §6.2:
| User/system action | Triggers rollback of |
|---|---|
| Deleting train images below the caption-coverage threshold | `captioned = false` |
| Deleting the training config | `training_configured = false` |
| Clearing the regularization set | `regularization_ready = false` |
| Deleting all train images | `curated = false` |
| Deleting the output safetensors | `artifact_produced = false` |

Rollback logic lives inside each relevant endpoint, submitted as a side
effect.

**UI checklist item 0 (project-level dataset), assembled dynamically**:
- if the project's `download/` is empty -> shows an "Import dataset" CTA
  (jumps to the project page)
- if non-empty -> shows "Dataset ready (N images)"
- This item is **not** a field on Version.phases — it's a UI-side derivation

---

## 5. Relationships

```
Project (id, title, slug, note)
   |  1:N
   v
Version (id, project_id, status, phases, output_lora_path)
   |  1:N
   v
Task (id, project_id, version_id, status)
                       ^
              project_jobs (id, project_id, version_id?, kind, status)
```

- Project -1:N-> Version: multiple training experiments under one project
- Version -1:N-> Task: a version can be retrained multiple times (previous
  attempt failed / re-tuning parameters)
- Project -1:N-> project_jobs: one shared table, scope distinguished by kind
  (version_id nullable)

**On `Task.project_id`**: kept as a redundant field, strongly consistent with
`task.project_id == version.project_id`. High-frequency queries (queue page,
monitor) get zero joins; clicking a task straight into its project works
directly.

---

## 6. Key decisions (confirmed)

### 6.1 Dataset ownership -> project level (unchanged from current)

`download/` and `preprocess/` stay at the **project directory** level;
multiple versions share the same pool of original images.

- Reasoning: in local-training use, "same character, different version,
  same batch of originals" is the 90% case — a separate per-version download
  wastes disk and contradicts the user's mental model.
- UI impact:
  - The project card does **not** directly show dataset status (avoids
    noise)
  - The project detail page gets a new "training set" section, showing
    download/preprocess image counts, status, and entry points
  - Version's capability checklist has **item 0 assembled dynamically**: no
    project dataset -> "Import dataset" CTA; project has a dataset -> "Dataset
    ready"
  - Version.phases starts from `curated` (no dataset phase stored)

### 6.2 Version phases -> persisted storage + mandatory rollback

Phases are stored as DB fields, **not computed live**.

**Write strategy**:
- Forward (set true): on job completion / user action (saving config,
  selecting images, etc.)
- Rollback (set false): deleting images, clearing config, clearing the
  regularization set, etc. must synchronously roll back the corresponding
  phase
- Consistency is maintained explicitly by each endpoint, not by
  "after-the-fact scan-and-repair"

Reasoning:
- Write paths are explicit, making state-machine consistency easy to
  guarantee
- Read performance: a list page might show dozens of projects x multiple
  versions each — scanning the filesystem every time is unacceptable
- The cost is extra work (every rollback path needs code), but each rollback
  is a 1-line SQL UPDATE, manageable

### 6.3 Project stores no phase at all

Dataset status (`download/` non-empty, image count) is computed live by the
UI scanning the filesystem.

- Reasoning: this is a derived fact of filesystem state; storing a bool
  would be redundant and drift-prone
- Implementation: the project detail page's "training set" section reads
  `len(os.listdir(download_dir))`; the list page does **not** show this
  information
- Project table stays completely clean: `id / title / slug / note /
  pinned_version_id / timestamps`

### 6.4 Version status granularity -> 6 states

After merging, drops from 9 states to 6:
- `draft` (merges draft + ready + preparing)
- `training` (merges queued + training)
- `paused`
- `completed`
- `failed`
- `canceled`

### 6.5 Task.project_id kept + consistency constraint

Not dropping the `task.project_id` field, but positioning it as redundant
with `versions.project_id`.

- NOT NULL (after migration, every task must belong to a version)
- Constraint `task.project_id == version.project_id`: enforced at the
  application layer (explicitly assigned when creating a task) + a DB CHECK
  as a second line of defense
- Benefit: high-frequency queries (queue page, monitor) get zero joins;
  clicking a task straight into its project works directly

### 6.6 Project has no soft delete

No `lifecycle: active/archived/trashed` is introduced. A project only has
"exists / physically deleted." Can get its own ADR later if needed.

### 6.7 project_jobs is not split or renamed

`download / preprocess / tag / reg_build` all live in the `project_jobs`
table, scope distinguished by kind (`version_id` nullable).

- Not splitting: splitting is schema perfectionism at the cost of more
  supervisor / event / migration code; one table is already clear enough
- Not renaming to `async_jobs`: a naming nitpick isn't worth a dedicated
  migration; can be done opportunistically if riding along with another
  migration in the future

### 6.8 No "representative version" / focus row / pinning

The project card does **not** show which version is "the" one.

- Reasoning: the metric strip already answers the core question — "what's
  running / what failed / what's done"
- If the user wants to see which one specifically -> click into the project
  detail page's version list; **the card doesn't need to guess a
  representative**
- Manual pinning is a burden on the user, and picking the wrong pin is even
  more confusing
- Default version opened on the project detail page: the most recently
  updated one (purely derived, not stored)

### 6.9 Consistency check: always runs

Every read of a version (list / get / any endpoint) validates
`version.status` against the value derived from the latest task.status.

- Mismatch -> log a warning + correct the field using the task-derived value
  + fire an SSE event so the frontend refreshes
- Performance: if this turns out to be a bottleneck, add a cached
  `status_implied_at` field the supervisor writes in lockstep
- Guiding principle: correctness over performance

### 6.10 No cache for the live dataset scan

The project detail page's "training set" section always does a live
`os.listdir(download_dir)`, **no** mtime cache.

- Reasoning: a listdir over a few hundred files is already millisecond-scale;
  caching would be premature optimization
- The user sees changes to download/ content immediately — more honest UX

### 6.11 "Retrain" semantics and entry point

A version in a terminal state (`completed / failed / canceled`) shows a
clear CTA:
- `completed` -> `[Train again]` (reuses config)
- `failed` -> `[View logs] [Retry with the same config] [Retry with a
  smaller batch (for OOM)]`
- `canceled` -> `[Continue training] [Restart with the same config]`

**Retraining does not create a new Version**: it keeps the same version's
multi-task history; if the user wants a different config, they open a new
version.

Difference from the current behavior: currently there's no "retrain"
concept — retraining means resubmitting a task manually. After the refactor,
"retrain" becomes a first-class action on Version, one click in the UI.

---

## 7. UI derivation rules (project card)

Everything on the project card is fully derived, **no focus row / no
pinning**:

```python
def project_card_view(project):
    versions = list_versions(project.id)
    return {
        # Metric strip (the core info)
        "total":     len(versions),
        "running":   count(v.status in ["training", "paused"]),
        "completed": count(v.status == "completed"),
        "failed":    count(v.status in ["failed", "canceled"]),

        # Dataset snapshot (project-level, scanned live)
        "dataset_image_count": len(os.listdir(download_dir)),

        # Latest-artifact entry point (from the most recently completed version)
        "latest_artifact": latest_completed_version_output(versions),
    }
```

**Default version opened on the project detail page**: the most recently
updated one (purely derived, not stored).

```python
def default_version_for_detail(project):
    versions = list_versions(project.id)
    return max(versions, key=lambda v: v.updated_at) if versions else None
```

Detailed UI design (metric strip / capability checklist / failure
affordances / three-tier notifications) is covered in a separate UI design
doc (yet to be written).

---

## 8. Legacy data migration mapping

```
old projects.stage              -> dropped entirely (Project no longer has stage)
old projects.active_version_id  -> removed (no more "representative version" concept)

old versions.stage -> mapping:
    "created"                  -> status="draft", all phases false
    "downloading"              -> status="draft" (download status now belongs to the project, scanned live by the UI)
    "preprocessing"            -> status="draft"
    "curating"                 -> status="draft", phases.curated=true
    "tagging"                  -> status="draft", phases.captioned=true
    "regularizing"             -> status="draft", phases.regularization_ready=true
    "configured" / "ready"     -> status="draft", phases.training_configured=true
    "training"                 -> look at the latest task:
                                   task.status=running    -> "training"
                                   task.status=paused     -> "paused"
                                   task.status=done       -> "completed"
                                   task.status=failed     -> "failed"
                                   task.status=canceled   -> "canceled"
                                   no task exists         -> "draft"
    "done"                     -> status="completed", phases.artifact_produced=true

old tasks.project_id           -> kept, gains NOT NULL + CHECK constraint
                                 (backfilled via a JOIN from version for NULL rows)

old Project.phase / dataset status -> not migrated (Project has no phase field, computed live by the UI)
```

---

## 9. Refactor scope checklist (for the ADR)

### Backend
- [ ] `studio/projects.py`: remove stage / advance_stage / active_version_id
  (the "representative version" concept is no longer needed)
- [ ] `studio/versions.py`: rewrite the state machine (6 states), the 5
  phases fields, transition constraints, forward + rollback write functions
- [ ] `studio/db.py` (tasks): keep project_id, add NOT NULL + DB CHECK;
  migration backfills NULL rows
- [ ] `studio/project_jobs.py`: don't split the table; make each kind's
  version_id required/optional rule explicit
- [ ] `studio/supervisor.py`: on task completion, advance version.status +
  the relevant phases; failure/cancel/pause must also advance it
- [ ] `studio/server.py`: remove the 9 `projects.advance_stage()` calls + 4
  `versions.advance_stage()` calls, replace with the corresponding
  phases/version-status advancement
- [ ] Add phases-rollback logic to each relevant endpoint (delete images /
  delete config / clear regularization set / delete artifact)
- [ ] New DB migration: drop fields + add fields + backfill data (per §8)
- [ ] Consistency-check function: on reading a version, assert
  `version.status_implied_by_tasks() == version.status`; mismatch -> log +
  correct using the task-derived value + fire SSE

### Frontend
- [ ] Remove the `StageBadge` component (or keep it repurposed for Version
  status)
- [ ] New `MetricStripBadge` component (for the project card)
- [ ] New `PhasesChecklist` component (for the Version detail page, item 0
  assembled dynamically for the dataset)
- [ ] Project detail page gains a "training set" section (live-scans
  download/ preprocess/, shows image counts)
- [ ] `Projects.tsx` card rewrite: metric strip + dataset snapshot +
  artifact entry point (**no focus row**)
- [ ] `Sidebar.tsx` inline stepper becomes a capability checklist
- [ ] Version detail page: shows a "retrain" CTA in terminal states
  (completed/failed/canceled)
- [ ] Failure-state affordance: static icon + red hue + explicit verb copy +
  "View logs" / "Retry with a smaller batch" actions
- [ ] Three-tier notifications (tab red dot / desktop notification / email)
  -- can be its own ADR

### Docs / ADR
- [ ] Once this document's discussion converges, turn it into
  `docs/adr/000X-project-version-lifecycle-refactor.md`
- [ ] A separate UI design doc,
  `docs/design/project-card-redesign.md`
- [ ] Update `AGENTS.md` §3.6 (the stage-advancement rules are all obsolete)
- [ ] Update the state-transition table in
  `architecture/studio-pipeline.md`

---

## 10. Decision log (fully converged as of 2026-05-23)

- [x] **Version status naming**: `draft` (merges old draft + ready +
  preparing)
- [x] **No "focus version"**: the metric strip already answers the core
  question, so a focus row is redundant; the card shows only the metric
  strip + dataset snapshot + latest-artifact entry point
- [x] **No pinned_version**: manual pinning is a burden on the user, and a
  wrong pin is even more confusing
- [x] **No active_version_id field**: the project detail page defaults to
  the "most recently updated" version, purely derived, not stored
- [x] **Phases rollback**: doing it. Each delete/clear action's endpoint
  explicitly runs an UPDATE to roll it back (see §4.5 / §6.2 table)
- [x] **project_jobs not split**: one table, scope distinguished by kind,
  version_id nullable
- [x] **Not renaming to async_jobs**: a naming nitpick isn't worth a
  dedicated migration
- [x] **Consistency check always runs**: every version read validates
  `version.status` against the value derived from the latest task.status;
  mismatch -> log + correct using the task-derived value + fire SSE
- [x] **No cache on the live dataset scan**: `os.listdir` over a few hundred
  files is already millisecond-scale, an mtime cache would be premature
  optimization
- [x] **"Retrain" entry point**: the Version detail page shows a CTA in
  terminal states (completed -> "Train again"; failed -> "View logs / Retry
  with the same config / Retry with a smaller batch"); doesn't create a new
  version, keeps the same version's multi-task history

~~**No open questions remain, ready to move to an ADR.**~~
**^ Overturned by the user's round-two review on 2026-05-23, see §11.**

---

## 11. Round-two open questions (2026-05-23)

> §10's "fully converged" was overturned after the user's round-two review.
> 9 new questions + 2 fact corrections.
> Simple ones marked done; complex ones marked open for discussion one at a
> time.
> **All changes in this round are recorded only in §11**; §1-§10 are left
> untouched for now, to be updated in §3 / §4 / §6 / §8 / §9 once discussion
> converges.

### 11.0 Fact corrections (reconciled against master)

- **§1 was wrong** — "`Project.stage` / `Version.stage`, 9 enum values each"
  is incorrect.
  - `projects.VALID_STAGES` really is 9 states
    (`created/downloading/preprocessing/curating/tagging/regularizing/configured/training/done`)
  - `versions.VALID_STAGES` has **only 6 states**
    (`curating/tagging/regularizing/ready/training/done`) — no `draft` /
    `preparing` / `created`
- **§4.2 / §6.4's baseline for the argument was wrong** — "merge old draft +
  ready + preparing -> draft" was said relative to an **earlier refactor
  proposal**, not master's actual current state.
  - master is really a 6-to-new-6 re-split, not 9-to-6.
  - Suggested actual mapping (pending convergence in §11.3):
    - `curating / tagging / regularizing / ready` -> `draft` (distinguished
      via phases)
    - `training` -> `training`
    - `done` -> `completed`

### 11.1 `active_version_id` kept (done)

Not removed. Its meaning changes to **"last-opened version"** (matches
current behavior), not the PM/UX sense of "pinned" or "representative
version."

**Sections to revisit**:
- §3 Project fields -> add `active_version_id` back
- §6.8 -> change to "not doing pinning / representative version, but keeping
  the lightweight last-opened semantic"
- §8 -> remove the "active_version_id -> removed" line
- §9 -> backend task list removes "delete active_version_id"; frontend
  `Layout.tsx` / `Overview.tsx` / `Sidebar.tsx` keep it

### 11.2 Sidebar stepper mapping and completion criteria (done — landed via §11.5 + §11.8)

**Mapping the 7 steps to the new model**:
- (1) Download / (2) Preprocess -> **project level** (above Version in the
  sidebar, tied to the project rather than the version)
- (3) Curate / (4) Tag / (5) Edit / (6) Regularization set / (7) Train ->
  **version phase** (a 5-value enum cursor, §11.3-B)
- The numbers (1)-(7) are **kept globally** (not renumbered — preserves
  master's existing UX mental model)

**Completion criteria** (derived per §11.5):
- Version phase (3)-(7): `phase_index < cursor_index` counts as "done"
- Project-level (1)(2): derived live from a filesystem scan (§6.10)

**Visual treatment** (finalized per §11.8):
- Completed phase: **the number character turns green** (e.g. the single
  character "(3)" turns green, text stays normal weight)
- Current cursor: that sidebar item's **background is highlighted**
- Current route: that sidebar item's **background is highlighted** (both
  cursor and focus are expressed via background; overlapping when they're
  the same phase is fine)
- Project-level (1)(2) show a file count in parentheses to the right (e.g.
  "(1) Download (80)"), no checkmark for completion

**Conclusion**: no checkmark icon needed — the number itself turning green
is the "done" UI expression.

### 11.3 Version state machine: paused / 6-to-5 merge / multi-task aggregation

§11.0 already noted the merge baseline needs rewriting against master. This
section splits into three sub-questions — A is decided, B is written out, C
is a new open question.

#### 11.3-A version-level `paused` (done — removed)

**Decision** (discussion on 2026-05-23): drop version-level `paused`.
`paused` is not in the version state machine.

**Reasoning**:
- Task-level pause (ADR 0006) already covers short/medium-term pause
  scenarios (cases 1/2)
- "Long-term shelving of a version" (case 3) is a non-need — if you really
  want to shelve something, use `completed` + a note, or cancel and start
  over
- One more state means one more place to maintain consistency (supervisor
  writes + SSE firing + version-vs-task state mismatch races)
- Mental-model consistency: the pause button maps to exactly one action
  (task pause), no hesitation over "should I pause the task or the
  version?"

**UI fallback approach**:
- Pause button -> only fires task pause (doesn't touch version.status)
- Version card / metric strip rendering:
  `status === 'training' && active_task?.status === 'paused'` -> shows
  "Training - paused Nmin" + a pause icon
- No need for `version_state_changed` to fire on task pause/resume

**Final state machine**: 5 states = `draft / training / completed / failed /
canceled`

#### 11.3-B state model: 5-state status + 5-enum phase (done)

**Decision** (discussion on 2026-05-23): split master's single
`versions.stage` field into **two orthogonal fields**.

> **2026-06-04 addendum**: once ADR 0010 pushed preprocessing down to the
> version-level `train/`, the phase sequence gained a `preprocessing` step,
> so the cursor is now a **6-value enum**:
> `curating -> preprocessing -> tagging -> editing -> regularizing -> ready`,
> and the skippable set expanded to `{preprocessing, regularizing}`. See ADR
> 0007 Addendum 1 + migration `_v11`. The rest of this section is kept as
> the historical 5-enum reasoning.

**Orthogonal model diagram**:
```
status (main state machine, 5 states)
   preparing / training / completed / failed / canceled
       |
       `- phase only means anything while status=preparing
              |
              v
          phase (cursor, 5-value enum)
              curating -> tagging -> editing -> regularizing -> ready
                                                ^
                                            only this phase is skippable (§11.5)
```

**status**: answers "what's the run state" (used for the metric strip,
whether a submit is allowed, whether the badge is lying).
**phase**: answers "how far along is prep," mapping to the current sidebar
stepper steps (3)-(7) (the dataset steps (1)(2) belong to the project, §6.1,
not part of version phase).

**Key insight**: phase is an **enum cursor**, **not** 5 independent bools.

-> **overturns the design draft's original §4.5 proposal** (5 independently
togglable phase bools).
-> How to store "the completed set" if partial skipping is allowed / phase
advancement rules / time locks / forced ordering -> all left for **§11.5**.
-> Of the 5 phase values, `editing` is a new value master doesn't have
(master lumps "tagging" and "tag editing" together as `tagging`); it's
broken out as its own value to match sidebar step (5).

**Migration table (master's `versions.stage` -> new status + phase)**:

| master `versions.stage` | new `status` | new `phase` | notes |
|---|---|---|---|
| `curating` | `preparing` | `curating` | |
| `tagging` | `preparing` | `tagging` | even if the user is already editing, a freshly opened new version still shows tagging — not being able to tell the difference is acceptable |
| `regularizing` | `preparing` | `regularizing` | |
| `ready` | `preparing` | `ready` | |
| `training` | look at the latest task | `N/A` | task `done -> completed` / `failed -> failed` / `canceled -> canceled` / `pending/running/paused -> training` |
| `done` | `completed` | `N/A` | |

**Dirty-data fallback** (stage='training' but no task — nearly impossible
but the SQL needs a default): `status=preparing + phase=ready`. Reasoning:
lets the user see "ready to start," and decide the next step themselves.

**status transition constraints** (task terminal states map independently,
the "don't lie" principle):
- task `done` -> version `completed`
- task `failed` -> version `failed`
- task `canceled` -> version `canceled`
- The three branches are independent, not merged into `completed`

**Phase-naming loose end** (not blocking convergence): `ready` literally
means "ready," but semantically it's "the user is configuring training
settings on /train." Whether to rename it to `configuring` is left for a
later review pass over the earlier sections.

#### 11.3-C version.status aggregation with multiple concurrent tasks (done — resolved via §11.7)

**Decision** (discussion on 2026-05-23, unblocked once §11.7 converged):

- active task count <= 1 (guaranteed by the §4.2 queue constraint)
- `version.status` computation:
  - has an active task (pending / running / paused) -> `training`
  - no active task, look at the most recent terminal task ->
    `completed / failed / canceled`
  - never had a task -> `preparing`
- "Look at the active task" settles it in one sentence — no complex
  aggregation logic needed

### 11.4 Clarifying what `project_jobs` means (done)

§3 / §4.4 need a clarifying note added:

> "`project_jobs` is essentially the **unified table for async jobs**:
> download / preprocess / tag / reg_build. It is **not** a training task —
> training tasks go through the `tasks` table. The two tables are parallel,
> distinct execution units with different lifecycles."

Avoids readers mistaking `project_jobs` for "project-level training tasks."

### 11.5 Phase-cursor advancement / invalidation logic

§11.3-B already decided phase is an enum cursor (not 5 bools). This section
splits into sub-questions:
- **11.5-A** advancement model (done)
- **11.5-B** completion / invalidation criteria per phase (done)
- **11.5-C** rollback actions (done)
- **11.5-D** how "the completed set" is stored (done)

#### 11.5-A advancement rules (done)

**Decision** (discussion on 2026-05-23):

**Advancement model**: hybrid
- cursor is one-directional (forward only)
- mandatory phases: `curating / tagging / editing / ready`
- skippable phase: `regularizing`

**Advancement entry point**: **a unified "Previous / Next" button in the
header** (replacing each page's own bottom button, consistent placement is
more visible). **Always clickable** (never disabled — friendlier to new
users; better to give a hint on error than to silently refuse to click).

| Button | focus state | behavior |
|---|---|---|
| Next | `focus < cursor` | focus++ (cursor unchanged, pure navigation) |
| Next | `focus == cursor` (mandatory phase) | validates the completion condition; pass -> cursor++ + focus++; fail -> a toast (e.g. "the train set is empty, please select a train set first"), cursor unchanged |
| Next | `focus == cursor` (skippable phase) | no validation, cursor++ + focus++ (this is effectively a skip) |
| Previous | `focus > 0` | focus-- (cursor unchanged) |
| Previous | `focus == 0` | disabled |

**Sidebar phase items**:
- before cursor + at cursor -> clickable (jumps focus, cursor unchanged)
- **after cursor -> disabled**, greyed out
- Reasoning: allowing multiple steps to be skipped at once would leave
  nowhere to show the hints for each intervening step -> undefined UI
  behavior, not allowed

**Dual UI expression (fully decoupled)**:
| Element | Expresses |
|---|---|
| checkmark | phase is complete (= every phase before the cursor) |
| background highlight | the phase the cursor is currently at |
| which page is showing | focus (determined by the URL) |

The three are independent dimensions — the user navigating to a
before-cursor phase doesn't move the cursor, and doesn't move the checkmark.

**Core concept separation**:
- **cursor** = prep progress (how far it's gotten), stored in the DB
- **focus** = the user's current route (URL), not stored
- **completion checkmark** = derived from being before the cursor, not
  stored
- The "Next" button's advance is the **only** entry point for cursor
  advancement (except for rollback actions, §11.5-C)

#### 11.5-B per-phase completion criteria (done)

**Decision** (discussion on 2026-05-23):

| phase | validation condition | failure message example |
|---|---|---|
| `curating` | `train/ has >= 1 image` (cannot be empty, no warning threshold) | "The training set is empty, please select training images first" |
| `tagging` | caption files at **100% coverage** (every train image has a matching .txt) | "N images still lack captions, please rerun or delete them" |
| `editing` | caption files at **100% coverage** (same as tagging, as a backstop) | Same message. Usually already 100% by the time it arrives from tagging -> passes automatically; only triggers if the user deleted captions |
| `regularizing` | **no related reg job is running / pending** (skippable = doesn't require a non-empty regularization set) | "A regularization job is running, please wait for it to finish" |
| `ready` | the training config file exists + passes schema validation | "Please finish the training configuration" / points at the specific invalid field |

**The `ready` phase + Next's special case**: passing validation means
`status: preparing -> training` + submitting the task. This is the
**interface point** between the §11.5 (phase) and §11.3 (status) state
machines — the cursor doesn't advance further (it's already the last one);
the status transition takes over from here.

**Advantages of the regularizing check design** (vs. a confirm dialog):
- Doesn't force the user into a binary "use / don't use a regularization
  set" decision (many users don't want to be interrupted)
- Only prevents the cursor advancing while a job is still running, avoiding
  state corruption
- Both an empty and a non-empty regularization set are allowed through — the
  user's choice is respected

**Implementation location**: one `check_completion(version_id) -> (ok: bool,
reason: str)` per phase, all living in `studio/versions_phase.py` or a
similar module. `advance_cursor()` calls the check; a failure reason is
handed to the frontend as a toast.

#### 11.5-C rollback actions (done)

**Decision** (discussion on 2026-05-23): **the cursor never actively rolls
back**.

**Reasoning**:
- The user's mental model of "going back" is "freely browsing already-done
  pages" (i.e. focus jumping, already covered in §11.5-A), not the cursor
  retreating
- Destructive actions (deleting images / captions / config, etc.) — **the UI
  should show a confirm prompt before the action** explaining the
  consequences (user education)
- If the user still deletes something after being warned, that's on them —
  the cursor doesn't auto-correct, avoiding a complex rollback state machine
- The next validation-on-Next (per §11.5-B) will naturally fail with a clear
  message -> the user redoes the step

**Summary of 5 cases**:

| # | Triggering action | cursor behavior | Next-time validation |
|---|---|---|---|
| 1 | delete train images (-> 0 or partial) | unchanged | curating/tagging/editing fail on Next (per each phase's own coverage check in §11.5-B) |
| 2 | delete captions (.txt) | unchanged | tagging/editing fail on Next |
| 3 | delete the training config | unchanged | ready fails on Next |
| 4 | clear / delete regularization-set images | unchanged | regularizing still passes on Next (only checks for no running job) |
| 5 | delete the LoRA artifact | status doesn't roll back | status=completed is kept + the UI derives an "artifact missing" label |

**A deliberately accepted UI illusion**: the sidebar checkmarks still derive
from the cursor (everything before it shows as done), even if the user has
since destroyed the underlying data. The cost is that the user only
discovers they need to redo something when they hit Next — but that's the
consequence of their own deliberate destructive choice.

**Accompanying requirement** (add to §9): every delete endpoint must have a
UI-level confirm dialog explaining "deleting this will make phase X's data
inconsistent, and you'll need to redo it to advance."

#### 11.5-D storing "the completed set" / expressing skipped (done)

**Decision** (discussion on 2026-05-23, a direct consequence of §11.5-C's
"cursor never retreats"):

**No extra field is needed** to store "the completed set" or "the skipped
set."

Reasoning:
- Cursor is one-directional and never retreats -> every phase before the
  cursor is considered "passed"
- Checkmark derives directly: `phase_index < cursor_index` == checked
- Skipped vs. done are **equivalent** from the cursor's point of view — both
  are "already passed"

**Should the UI distinguish skipped vs. done** (recommended: derive, don't
store):
- Check whether that phase's "actual data" exists
  - regularizing: is `reg/` non-empty
  - tagging: caption file coverage
- data exists -> "Done"; doesn't exist -> "Done (skipped)"
- The derivation being not 100% accurate doesn't hurt functionality — it's
  just a UI hint

**Final, minimal shape of the model**:
- `version.status` (5-value enum)
- `version.phase` (5-value enum, only meaningful when status=preparing)
- All "fully done / skipped / invalidated" info is **derived**, none of it
  stored

-> **Overturns the design draft's entire §4.5** (the 5-bool phases +
rollback table); §4.5's 5-bool model is fully abandoned.

### 11.6 Dataset UI presentation (done)

**Decision** (discussion on 2026-05-23):

**Project-level dataset**:
- Copy: **"Dataset"** (not "training set")
- Display location:
  - Sidebar top ((1) Download / (2) Preprocess, with counts in parentheses)
  - Project detail page top (a count + a [Manage] link, kept minimal)
- No teaching dialog explaining "the dataset is project-level" (keep it
  simple, express the relationship through layout)

**Version-level dataset**:
- Copy: still "Dataset" (same name is fine — context disambiguates: on the
  version detail page it's clearly the version's own)
- Display location: the project detail page's [Details] tab (a grid layout,
  §11.8-C)
- Contents (per each cell in §11.8-C's grid):
  - **List of folders (repeat_N_concept/) with per-folder file counts + a
    total**
  - **Resolution distribution** (matching the stats widget on the upscale
    page)
  - **Aspect-ratio distribution** (matching the bucket stats on the crop
    page)
  - **Tag distribution** (matching the stats on the tag-editing page)
  - **Regularization-set count**
- **Empty-state placeholders** for unfinished items: each cell gets its own
  empty state, styled to match the associated phase page's existing empty
  state (not a uniform grey-box-with-text look)

**Teaching mechanism (implicit rather than explicit)**:
- The project-level dataset sits at the very top + labeled "project-level"
- The version-level dataset lives inside the [Details] tab
- Keep copy minimal — let the layout hierarchy express the ownership
  relationship

### 11.7 Retraining data handling (done)

**Decision** (discussion on 2026-05-23): multiple tasks under the same
version (model B) + **only freeze config** within a task (simplified
version of model G). Model D (hard links) is abandoned.

**What gets frozen**:
- Yes: `config.yaml` — copied into the task's snapshot directory at task
  creation
- No: captions / images / regularization set — none of these are frozen

Reasoning for abandoning model D (full hard-link freeze):
- Breaks on cross-OS export (a Windows-trained user importing into a Linux
  compute platform would break this use case)
- The preprocess->train copy chain is already an independent problem —
  pulling it in here would let the scope run away (-> §12.1)

**Mental-model separation (the core UX design)**:
- The task detail page gets its own **"training config record" page**
  (read-only yaml / reusing the version-config component in read-only mode)
- **Clicking a task does NOT jump to the version's config editing page**
  (avoids the user mistakenly thinking "what the task shows is what config
  was actually used at the time")
- The page separation expresses it: **config is a historical snapshot;
  captions/images are the version's current state** (every training run
  uses the current data, and the user is expected to understand that
  themselves)

**Retrain entry point**: a button on the task's config-record page
(tentatively named "Retrain with this config"):
- Click -> jumps to the version config editing page + auto-loads that frozen
  config
- The user can edit it (or not) -> starts training normally -> creates a new
  task
- No separate "retrain" button is needed (per the user's earlier Q3: going
  through the normal training flow just creates a new task)

**Relationship to the design draft's §6.11**:
- "Retraining doesn't create a new Version, keeps the same version's
  multi-task history" -> **kept** (yes)
- §6.11's several CTAs ("Train again / Retry with the same config / Retry
  with a smaller batch") -> simplified to a single "Retrain with this
  config" button

**Implementation notes**:
- On task creation (supervisor enqueue): `tasks/{task_id}/snapshot/config.yaml`
  <- copied from the version's current config
- New task detail route: `/projects/:pid/v/:vid/task/:tid/config` (read-only)
- Accompanying requirements (-> §9):
  - backend: task snapshot directory creation + config copy logic
  - frontend: read-only task config page component + "Retrain with this
    config" button's routing + auto-prefill

**Experiment matrix (per the user's earlier Q4)**: not specifically
supported — a user wanting a matrix forks a version manually (§6.11's PP10.1
already has a "full copy from an old version" feature).

### 11.8 UI restructure, layout v3 (done)

**Complete layout decisions** (discussion on 2026-05-23). 5 parts.

#### A. Sidebar layout (after entering a project)

- Top-level tabs (**Projects / Queue / Tools / Settings**) never collapse,
  always visible
- Entering a project **expands** its content under the "Projects" tab
  (expressing ownership)
- Queue / Test etc. top-level tabs remain visible below
- Visual rules per §11.2 (completed phase number turns green / current page
  background highlight)

```
+-------------------------------+
|  Anima . 0.10.0               |
+-------------------------------+
| v Projects                    | <- top-level tab, expands after entering a project
|    Project: Cosmic Kaguya     |
|                                |
|    Overview     <- current page| <- background highlight = active route
|    (1) Download  (80)         | <- project-level phase, count in parens
|    (2) Preprocess (50)        |
|                                |
|    Versions (list, all shown):|
|     * test                    | <- currently active version
|       baseline                |
|       highlr                  |
|                                |
|    (3) Curate                 | <- number turns green when done
|    (4) Tag                    |
|    (5) Edit                   | <- cursor, background highlight
|    (6) Regularization set (skippable) |
|    (7) Train                  |
|                                |
|   Queue                       | <- top-level tab, still visible
|   Test                        |
|   Tools / Settings            |
+-------------------------------+
```

#### B. Phase page header

A top-right advance button (numbered + phase name + direction arrow),
replacing the generic "Previous / Next" copy:

```
+----------------------------------------------------+
| Tag editing        [<- (4) Tag]    [(6) Reg set ->] |
+----------------------------------------------------+
|                                                      |
|   ... phase page content ...                        |
|                                                      |
+----------------------------------------------------+
```

- The button is always clickable (§11.5-A's philosophy — new-user friendly)
- Non-skippable phase: Next fails validation -> toast
- Skippable phase (regularizing): advances with no validation
- First / last phase: the corresponding direction button is disabled

#### C. Project detail page (reached via the sidebar's "Overview")

**Two-layer structure**: top half = project-level info (every Project field
shown, except pipeline — Project has no pipeline to begin with, §6.3);
bottom half = version scope (a dropdown to pick a version + 3 tabs).

```
+-----------------------------------------------------------+
| [Project-level -- doesn't change with the dropdown]        |
| keta                                                       | <- project.title
| slug: keta . created 2026-05-01                            | <- slug + timestamp
| {project.note}                                              | <- project note (hidden if empty)
+-----------------------------------------------------------+
| Dataset: (1) 264 images . (2) 247 images    [Manage]        | <- project-level dataset (live scan)
| 2 versions                                                  | <- version count
+-----------------------------------------------------------+
| [Version picker -- independent of the sidebar's active one] |
| Version: [v1.1 v]                  [* Completed]             | <- dropdown + selected version's status
+-----------------------------------------------------------+
| Tabs: [Details] [Tasks] [Output]                            | <- all version-scoped
+-----------------------------------------------------------+
| [Details tab -- grid layout, not stacked]                   |
|                                                              |
|  +------------------+------------------+                    |
|  | Folders/repeats  | Tag distribution |                    |
|  |  repeat_5_... 30 |  [chart / list]  |                    |
|  |  repeat_2_... 17 |                  |                    |
|  |  Total 47        |                  |                    |
|  +------------------+------------------+                    |
|  | Resolution dist. | Aspect-ratio dist|                    |
|  |  [chart]         |  [chart]         |                    |
|  +------------------+------------------+                    |
|  | Regularization set: 0 images (not generated)              |
|  +------------------------------------------+                |
|                                                              |
|  Unfinished cells -> each gets its own empty state           |
|  (styled after the related phase page)                      |
|                                                              |
+-----------------------------------------------------------+
| [Tasks tab]  = lists tasks for **this version** (not every  |
|                task under the project); table style same as |
|                /queue; click a task -> task detail page      |
+-----------------------------------------------------------+
| [Output tab] = **this version's** LoRA artifact              |
|                output_lora_path + step/epoch checkpoints     |
+-----------------------------------------------------------+
```

**Dropdown independence** (a key decision):
- The overview page has a local-only `selectedVersionId`, initialized to
  `project.active_version_id`
- Changing the dropdown **only affects the data source for the overview
  page's 3 tabs**, **doesn't touch the sidebar's active version and fires no
  API call**
- The sidebar's version list still follows the active version (clicking a
  version in the sidebar = activating it)
- Purpose: lets the user quickly compare two versions' dataset/task/output
  without repeatedly activating one or the other
- The overview page's "project-level" top half (title/slug/dataset/version
  count) **does not change with the dropdown** — it's inherently
  project-level

#### D. Task detail page

5 tabs: **[Overview] [Logs] [Monitor] [Output] [Linked config]** (the last
one is new, per the §11.7 decision).

```
+-----------------------------------------------------------+
| Task #45                                                    |
| Tabs: [Overview] [Logs] [Monitor] [Output] [Linked config]  |
+-----------------------------------------------------------+
| [Linked config tab -- new]                                  |
|                                                              |
|   Read-only yaml / reuses the version-config component in    |
|   read-only mode (clicking a task does NOT jump to the       |
|   version's config editing page -- mental-model separation)  |
|                                                              |
|   +--------------------------------+                        |
|   |  read-only config display...   |                        |
|   |                                |                        |
|   +--------------------------------+                        |
|                                                              |
|   [Apply this config]  <- click jumps to the (7) Train phase |
|                          config page, prefilled              |
|                          -> user edits, then starts training |
|                          normally                             |
|                          -> creates a new task (same version, |
|                          multiple tasks)                      |
+-----------------------------------------------------------+
```

#### E. Project list card (home page /projects)

```
+----------------------------------+
| Cosmic Kaguya     [preparing]    | <- top-right = the active version's status
| test                              | <- active version's name (direct, no prefix copy)
+----------------------------------+
```

- Drops the stage badge (the §1 pain point — the badge that lied in master)
- Top-right status = **the current active version's** status (not a project
  stage — the project has no stage)
- Doesn't show the artifact / doesn't show a timestamp / doesn't write an
  "active version" prefix label
- Everything else about the layout stays as-is (not redesigning the card
  structure as a whole)

### 11.9 Suggested discussion order

Low-coupling items first, UI last:

1. ~~**§11.4 + §11.1** (already decided, just writing it down)~~ done
2. ~~**§11.3** necessity of version-level paused + 6-state rewrite~~ done, A/B/C all decided
3. ~~**§11.5** phase advancement / invalidation logic~~ done, A/B/C/D all decided
4. ~~**§11.7** retraining vs. Version Up disk cost~~ done
5. ~~**§11.6 + §11.8** UI integration~~ done
6. ~~**§11.2** sidebar checklist~~ done

**§11 has fully converged** — next step: go back and update §1-§10 to
reflect §11's decisions, then convert to ADR-0007.

---

## 12. Follow-up issues (independent work, out of this round's scope)

Items that surfaced during §11's discussion but are **not part of this
refactor's scope**, tracked as independent issues.

### 12.1 `preprocess -> train` data-flow redesign

**Current state**:
- `preprocess/` is project-level; `train/` is version-level
- Curating a selection = a `preprocess -> train` physical copy (2x disk
  usage)
- Changes to preprocess don't propagate to images already selected into
  train

**Directions raised during discussion** (undecided):
- Switch to hard links (saves disk, but hard to export cross-OS)
- Make preprocess replace train (version only stores a "which ones were
  selected" manifest)
- Keep the current state (accept the copy cost)

**How it surfaced**: came up while discussing model D (hard links) for task
snapshots in §11.7 — not directly coupled to task data freezing, but touches
the same data-flow model.

**Parking**: an independent issue / a separate ADR, **not part of
ADR-0007's scope**.

### 12.2 Confirm-dialog copy audit for delete endpoints

Once §11.5-C decided the cursor never actively rolls back, every frontend
delete endpoint (delete images / delete captions / delete config) needs its
confirm dialog to explain "this deletion will affect phase X's data
consistency." An audit is needed of which endpoints currently have a confirm
step and whether the copy is adequate.

**Parking**: treated as a sub-item of §9's frontend checklist, to be handled
when that work is picked up.
