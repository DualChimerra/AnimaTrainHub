# 0007 — Project / Version / Task lifecycle refactor

**Status**: Proposed
**Date**: 2026-05-23
**Decision makers**: @WalkingMeatAxolotl

> **Maintenance note**: this ADR carries forward the two-round discussion draft in `docs/design/project-version-task-lifecycle.md`.
> Round-one decisions are in design draft §1–§10; round-two review reversed 9 of them, converging in §11; the follow-ups parked in §12 are out of scope for this ADR.
> Already-accepted round-one decisions are not modified or deleted; later audits/reversals are appended under "## Incremental updates" at the end.

## Background

Master's current `Project.stage` (9 states) / `Version.stage` (6 states) mix two orthogonal concepts — "runtime state" and "preparation stage" — causing structural problems:

- **Project cards lie**: after a training failure / cancellation / pause, `projects.stage` never advances, so the badge shows `training` with an animated dot forever
- **Dead states**: `projects.stage`'s `curating / configured / regularizing` are never advanced to by any code path; 3 of the 9 color mappings are never seen by users
- **Stage skipping**: uploading training images jumps straight to `tagging`, skipping 5 stages — meaning stage is really "the set of steps performed," but the UI presents it as linear progress
- **No source of truth**: `advance_stage()` is an unconditional setter, with rules scattered across 13 call sites; Project / Version / Task have three separate state machines that never reconcile
- **Data model mismatch**: Project is an aggregate (1:N → Version) and shouldn't carry "process state"

For the full pain points and the two-round debate among the three stakeholders (PM / User / Designer), see `docs/design/project-version-task-lifecycle.md`; this ADR carries forward its logical model and focuses on locking in the decisions and the implementation path.

## Candidate approaches

Each item below lists the rejected alternatives, to record why they were rejected. The full debate is in design draft §11.x.

### Project data model

- **A. Add a status field**: rejected. A project is an aggregate (1:N → Version) and shouldn't have process state.
- **B. Keep stage**: rejected. Doesn't address the root cause (stage mixing two kinds of state).
- **C. Minimal — no stage, no process field (adopted)**: all "process" is derived from aggregating child versions + real-time filesystem scans of project-level datasets

### Version state machine granularity

- **A. Single status field, 5 states (draft / training / completed / failed / canceled)**: rejected. Loses "which preparation step am I on" information.
- **B. Keep master's 6 stages + training substates driven by task**: rejected. The stage concept still mixes two layers.
- **C. Orthogonal status + phase fields (adopted)**:
  - `status` (5-value enum) answers "what's the runtime state"
  - `phase` (5-value enum, meaningful only when status=preparing) answers "which preparation step"

### Version-level paused state

- **A. Add version-level paused**: rejected. Task pause (ADR 0006) already covers the short/medium-term scenario; "leaving a version paused long-term" is not a real need.
- **B. Remove it (adopted)**: when task=paused, the UI derives "training · paused for Nmin"

### Phase model

- **A. 5 independent bools, any order** (design draft's original round-one proposal): rejected. Violates the user's linear mental model (filter → tag → edit → reg set → train strongly implies an order).
- **B. Enum cursor, 5 values (adopted)**: `curating → tagging → editing → regularizing → ready`, cursor moves one-way, the completed set is derived directly

### Phase advancement mode

- **A. Strictly one-way**: rejected. Skipping the reg set / going back to edit captions requires a heavyweight "phase rollback" mental model.
- **B. Fully free (click anywhere)**: rejected. The cursor loses its "objective progress" meaning; ✓ becomes just "has the user clicked here."
- **C. Hybrid (adopted)**: cursor is one-way; `regularizing` can be skipped; the header's "next step" button is **always clickable** (never disabled, friendlier to newcomers) with a toast shown on validation failure; **the cursor never rolls back automatically** (if the user deletes data afterward, ✓ shows a false state until the next validation catches it)

### Data freezing on retrain

- **A. Force a Version bump (every retrain = new version + full copy)**: rejected. Doubles disk usage.
- **B. Multiple tasks share one version, no freezing** (master's current behavior): rejected. Old tasks see the current config / current training set → forensic record is distorted.
- **D. Multiple tasks share one version + full data freeze (caption copy + image hardlinks)**: rejected. Hardlinks break across OS export (train on Windows → import on a Linux compute platform breaks); the existing `preprocess→train` copy chain is a separate problem and pulling it in would blow up scope.
- **G. Multiple tasks share one version + only config is frozen (adopted)**:
  - When a task is created, `tasks/{task_id}/snapshot/config.yaml` is copied from the version's current config
  - Captions / images / regularization set are **not** frozen
  - **UI mental-model separation**: the task detail view has an independent "linked config" tab (clicking a task doesn't jump to the version's config edit page), with an "apply this config" button → jumps to phase ⑦ (train) pre-filled, following the normal flow

### Dataset ownership

- **A. Version-level (each version has its own independent training set)**: rejected. Doesn't match the user's mental model (using the same batch of source images for different versions of the same character is 90% of usage), and wastes disk.
- **B. Project-level + real-time scan (adopted)**: `download/` + `preprocess/` stay in the project directory; project-level phase isn't stored as a DB field — the UI derives it in real time via `os.listdir` (§6.10)

### The `active_version_id` field

- **A. Remove it** (design draft's original round-one proposal): rejected. Touches 10+ code sites and degrades UX.
- **B. Keep it, meaning "last opened version" (adopted)**: matches master's current behavior; not "pinned" or "the representative version" in the PM/UX sense

### The `project_jobs` table

- **A. Split into separate tables (one each for download / preprocess / tag / reg_build)**: rejected. Schema purism, and adds a lot of supervisor / event / migration code.
- **B. Single table + kind distinguishes scope (adopted)**: `version_id` can be null — null for project-level jobs, NOT NULL for version-level ones

## Decision

### Final data model

**Project** (minimal):
```
id, title, slug, note, active_version_id, created_at, updated_at
```
- `stage` field removed
- `active_version_id` kept ("last opened" semantics)
- No process/phase fields at all; dataset state is derived by real-time UI scans

**Version** (two orthogonal fields):
```
id, project_id, label, note, trigger_word, created_at, updated_at,
status: preparing | training | completed | failed | canceled,
phase:  curating | tagging | editing | regularizing | ready  (meaningful only when status=preparing),
output_lora_path, last_task_id, last_failure_reason
```
- `stage` field removed
- The completed set / skipped set / rollback state are all **derived**, never stored

**Task** (unchanged + new snapshot):
- Task state machine unchanged (already settled by ADR 0006)
- New: when a task is created, `tasks/{task_id}/snapshot/config.yaml` is copied from the version's current config
- New endpoint: GET `/api/tasks/{tid}/snapshot/config` (read-only)

**project_jobs** (unchanged): single table + kind distinguishes scope

### State machine definition

**Version status transitions** (driven by the supervisor; the UI never writes directly):
- task `done` → `completed`
- task `failed` → `failed`
- task `canceled` → `canceled`
- the three branches are independent (no lying)

**Version status derivation** (when there's no active task):
- has an active task (pending / running / paused) → `training`
- no active task, look at the most recent terminal task → `completed / failed / canceled`
- never had a task → `preparing`

**Phase advancement rules** (details in §11.5-A):
- cursor is one-way (forward only)
- advancement entry point: the "← phase name | phase name →" buttons on the right of the phase page header, always clickable
- required phases (`curating / tagging / editing / ready`) show a toast on validation failure
- skippable phase (`regularizing`) validation = no reg job running/pending, no confirm dialog
- sidebar phase items: everything before and at the cursor is clickable (jump via focus); everything after the cursor is disabled
- the cursor never rolls back automatically (if the user deletes data afterward, ✓ shows a false state until the next "next" validation catches it)

### Completion criteria per phase (§11.5-B)

| phase | validation |
|---|---|
| `curating` | `train/` has ≥ 1 image |
| `tagging` | 100% caption coverage (every training image has a .txt) |
| `editing` | same as tagging (fallback) |
| `regularizing` | no reg job running/pending (skippable) |
| `ready` | config file exists + passes schema validation; passing "next" = `status: preparing → training` + submit task |

### UI Layout v3 (§11.8)

Full ASCII layout in `design §11.8`:
- **A. Sidebar**: top-level tabs for Projects/Queue/Tools/Settings never collapse; expands under "Projects" once inside a project
- **B. Phase page header**: `[← ④ Tagging]  [⑥ Regularization →]` (with number + phase name + arrow)
- **C. Project detail page** (entered by clicking "Overview" in sidebar): 3 tabs `[Details] [Tasks] [Output]`; details tab uses a grid layout; top-right = current version's status
- **D. Task detail page**: 5 tabs `[Overview] [Logs] [Monitor] [Output] [Linked Config]`; the last one is new, with an "apply this config" button
- **E. Project list card**: stage badge removed; top-right = active version's status; no artifacts or timestamps shown

## Rationale

Summary of the core reasons other approaches were rejected:

1. **Adding status to Project** → a project is an aggregate, has no process-state semantics
2. **Merging Version's 6 states** → loses "which step am I on" information
3. **5 independent bool phases** → violates the user's linear mental model, ✓ meaning is unclear
4. **Version-level paused** → already covered by task pause (ADR 0006)
5. **Strictly one-way phases** → poor UX for skip scenarios
6. **Fully free phases** → cursor loses its objective-progress meaning
7. **Task full data freeze / hardlinks** → breaks across-OS export
8. **Forced Version bump on retrain** → doubles disk usage
9. **Version-level datasets** → doesn't match the user's mental model, wastes disk
10. **Removing `active_version_id`** → UX regression with no upside

## Consequences

**Benefits**:
- `status` never lies (task terminal states map independently, the supervisor is the sole driver)
- The model is minimal (status + phase + active_task fully describe the state machine; completed / skipped / stale are all derived)
- Covers 95% of forensic value (config is 100% accurate; caption/image drift is acceptable)
- Cross-OS export works fully (no dependency on hardlinks / COW)
- Phase header button UX is beginner-friendly (always clickable + validation hints beat silent disabling)
- Migration can be incremental (v8 add-only → dual-write → frontend cutover → v9 destructive)

**Constraints / new debt**:
- Large frontend change surface (layout v3 touches all five areas → 3 frontend PRs)
- `_v9` migration breaks the existing `studio/migrations/__init__.py` convention ("never rewrite existing columns backward") — an explicit exception this time
- §11.5-C accepts a "UI illusion": the cursor doesn't roll back → after a user deletes data, ✓ shows a false state until the next "next" validation catches it (a deliberately accepted design trade-off)
- Task snapshot requires matching work: each delete endpoint needs confirm-dialog copy added (→ §12.2)

**Follow-ups parked (out of scope for this ADR)**:
- §12.1 Reworking the `preprocess → train` data-copy chain (hardlinks / a project-level manifest replacing train/) — separate ADR
- §12.2 Audit of confirm-dialog copy for delete endpoints — a sub-item of the §9 frontend rework

## Implementation plan

7 PRs, each independently mergeable to dev without breaking the current state. Full commit breakdown is in the design draft's implementation-plan section; PR merge order:

| PR | Scope | Blocked by |
|---|---|---|
| 1 | ADR + design draft banner + README index | none |
| 2 | `_v8` schema (add-only) + new model fields, readonly | 1 |
| 3 | Backend business logic: phase validation / supervisor-driven status / consistency / task snapshot | 2 |
| 4 | Frontend layout v3 wave 1: sidebar + phase header + project detail page's three tabs | 3 |
| 5 | Frontend task detail [Linked Config] tab + "apply this config" flow | 3, 4 |
| 6 | Frontend project list card | 3 |
| 7 | Cleanup + `_v9` destructive migration (DROP `stage` column) | 2, 3, 4, 5, 6 |

## Addendum 1 — `preprocessing` phase (2026-06-04)

ADR 0010 moves preprocessing down from project scope to version scope (upscale / crop happen in place under `versions/{label}/train/`). The phase model gains a corresponding new step:

### Changes

- **Add `preprocessing` phase**, inserted after `curating`, before `tagging`. Full order:
  `curating → preprocessing → tagging → editing → regularizing → ready`
- **Skippable set expanded to `{preprocessing, regularizing}`**: users don't need to upscale to continue
- **Completion criteria**: `preprocessing` has no mandatory validation (same as `regularizing`) — skipping counts as moving to the next step
- **DB migration `_v11`**: existing versions with `phase ∈ {tagging, editing, regularizing, ready}` and a non-empty training set are backfilled one-time to the `preprocessing` state, avoiding the appearance of older versions suddenly "regressing" to the preprocessing step in the UI

### Rationale

- Preprocessing = "modifying the pixels of the training set," which is a distinct action from "modifying which images are in the training set" (curating) and "writing captions" (tagging); folding it into curating would confuse validation (curating validates "≥1 image" vs. preprocessing validates "everything upscaled" — two half-finished states mixed together)
- Skippability matches real usage: many LoRA workflows go straight to tagging without upscaling

## References

- `docs/design/project-version-task-lifecycle.md` — the full two-round review discussion draft
- Related ADRs: [0006](0006-queue-pause-resume.md) (task pause/resume) / [0004](0004-preprocess-manifest.md) (preprocess manifest)
- Implementation PRs: numbers to be backfilled on merge
