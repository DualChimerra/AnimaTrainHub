# AGENTS.md — Project code quality and collaboration conventions

**Audience**: everyone and everything that touches this repo's code — AI agents
(Claude Code / Codex / Cursor / Copilot / ...), community contributors, maintainers.
**Goal**: make life painless for whoever picks up this code 6 months from now.
Maintainability beats cleverness.

**Entry point**: this doc lives at `docs/AGENTS.md`, reachable from the
"Collaboration conventions" section of [`README.md`](../README.md) and the
"Additional notes for AI agents" section of [`CONTRIBUTING.md`](../CONTRIBUTING.md).

If you want AI tools to auto-load this doc every time, create a (gitignored)
`CLAUDE.md` or `AGENTS.md` at the working-tree root with a single line:
`@docs/AGENTS.md`.

Human contributors should also read [`CONTRIBUTING.md`](../CONTRIBUTING.md) (process /
branching / commits / PRs / releases). This document doesn't repeat that — it only
covers code quality, consistency, maintainability, and cross-tool collaboration
conventions.

**Scope by change size**: the alignment protocol (§1) and pre-PR checklist (§5) in
this document are scaled to the size of the change. Small changes (typos / docs /
one-line fixes / renames / copy edits) don't need the full process; medium/large
changes (new features / new UI / cross-module / schema changes) do. The point of
these conventions is to reduce misunderstandings and rework — when in doubt, ask the
maintainer rather than getting stuck on the letter of the rule.

---

## 0. First principles

1. **Maintainability over cleverness**: prefer a 5-line named function over a
   1-line nested lambda. Someone still has to be able to read the code you write
   today, 6 months from now.
2. **Consistency > personal taste**: this repo already has conventions (naming,
   directory layout, import direction, error-handling patterns, component reuse) —
   follow them. Don't "opportunistically improve" things into a different style.
3. **Grep before you build**: whatever component / helper / fixture / CLI / registry
   you're about to create, there's a 90% chance it already exists. Search first.
   `useProjectCtx` / `SchemaForm` / `PathPicker` / `Toast` / `ImageGrid` /
   `TagEditor` / `OnnxTaggerBase` / `useLocalStorageState` all already exist.
4. **Verify the paper / upstream before touching external algorithms**: for external
   paper implementations like InfoNoise / Prodigy / LyCORIS, don't "correct" them by
   intuition. Check the paper, upstream issues, and the original author's commits
   first, then add a unit test to codify the formula (see `tests/test_infonoise.py`
   for an example).
5. **Data flow: normalize at authoring time**: this project prefers structured
   yaml/schema + tooling validation + derived markdown (see `release_notes.yaml` ->
   `CHANGELOG.md`). **Do not** introduce a design that parses free-form text at
   runtime.
6. **Take small steps**: one PR = one unit of work. Unrelated changes make review
   harder.

---

## 1. Pre-work alignment protocol (guards against scope drift)

**Applies to**: adding features / changing UI-UX flows / cross-module changes /
schema changes.
**Does not apply to**: typos / docs / one-line bug fixes / renames / copy edits /
test-only additions — just go ahead and make these directly.

When it applies, the AI should align with the author first. If alignment fails,
stop — don't "write while asking," and definitely don't "write a version first and
see."

### 1.1 The restatement the AI should proactively produce

Before writing code, answer the following in a short paragraph for the author to
review (so they can immediately catch a misunderstanding):

```
My understanding of this task is:
- What problem it solves: [one sentence]
- Entry point for the user: [page / CLI command / which step]
- User-visible behavior change: [before -> after]
- Files affected: [list, exact paths]
- Does it affect data / schema / SSE: [yes/no + which ones]
- Test coverage approach: [regression / new tests / manual testing]
- Out of scope: [things the author might assume are also included, explicitly excluded]
```

### 1.2 Situations where the AI should proactively ask follow-up questions

- **The author describes a symptom, not a root cause**: "this button doesn't do
  anything when clicked" — is the handler not wired up? A network error? State not
  updating? Locate the cause first, then ask "what do you expect it to do?"
- **The UI/UX description is vague**: "add a button" — where? What style? What
  happens on click? Which existing component on the page does it resemble?
  **Don't guess** — look at the Sidebar/Stepper structure in
  [`architecture/studio-pipeline.md`](architecture/studio-pipeline.md), look at the
  style of existing pages (e.g. `studio/web/src/pages/`), then ask.
- **The code you're about to change was just changed, or its PR hasn't merged yet**:
  check the relevant PR's discussion context first, to avoid overturning a decision
  made just days ago.
- **The author doesn't seem to know about an existing constraint**: e.g. they want
  to add `if optimizer_type == "xxx"` in `loop.py` — this violates the plugin
  boundary (see §3.4); tell the author and offer an alternative.

### 1.3 What not to do

- ❌ Don't assume the author has fully described the requirement — in most cases
  the author gave you 50%; the AI should either ask for the other 50% or explicitly
  flag "I'm assuming X."

---

## 2. Spec-file map (look things up by scenario)

| Scenario | Where to look |
|---|---|
| Opening a PR / branching / writing commit messages | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| Cutting a release / writing release notes / bumping version numbers | [release-notes-spec.md](release-notes-spec.md) + [CONTRIBUTING.md § Release process](../CONTRIBUTING.md) |
| Adding / changing the training stack (LoRA variants / optimizer / scheduler / sampler / loss) | [runtime/training/README.md](../runtime/training/README.md) + [adr/0003-anima-train-refactor.md](adr/0003-anima-train-refactor.md) |
| Adding / changing Studio Web features | [architecture/studio-pipeline.md](architecture/studio-pipeline.md) + [studio/README.md](../studio/README.md) |
| Adding a new dependency / changing default behavior / interfaces / schema | Write an ADR: [adr/README.md](adr/README.md) |
| Changed user-visible behavior | Update the corresponding section under [user-guide/](user-guide/) |
| Looking up historical decisions | [adr/0001-0005](adr/) — **old ADRs are never deleted or rewritten**, only get a status appended |
| Writing / changing tests | [tests/conftest.py](../tests/conftest.py) (`runtime/` is already injected into sys.path) |

---

## 3. Project-specific hard conventions (general knowledge omitted — only what's unique here)

### 3.1 Dependency direction (one-way, reverse is forbidden)

```
models  ->  utils  ->  runtime  ->  studio  ->  tools
```

- `utils/` does not import `runtime/` or `studio/`
- `runtime/training/` -> `utils/`, never the reverse
- `studio/services/inference_core` reuses `utils/lycoris_adapter` precisely because
  the latter isn't tied to a training context

### 3.2 Sister-script contract

`runtime/anima_daemon.py` / `anima_generate.py` / `anima_reg_ai.py` get 7 names via
`import anima_train as _T` (`find_diffusion_pipe_root` / `load_anima_model` /
`load_vae` / `load_text_encoders` / `sample_image` / `enable_xformers` /
`resolve_path_best_effort`).

The top-level re-exports in `runtime/anima_train.py` are a contract: **you may add
to them, but never remove or change their signatures**.
`tests/test_anima_generate_xy.py` will catch violations.

### 3.3 Single source of truth (change it here -> run the sync tool, don't hand-edit derived files)

| Source of truth | Derived from / referenced by | How to sync |
|---|---|---|
| `studio/__init__.py:__version__` | FastAPI `/api/health` -> Sidebar | Sync `studio/web/package.json` + the `README.md` badge + the "## Version" section |
| `release_notes.yaml` | `CHANGELOG.md`, GitHub Release Body | `python tools/bump_version.py bump --version X.Y.Z` |
| `studio/schema.py:TrainingConfig` | `argparse_bridge`-generated argparse, the frontend `SchemaForm` renderer, 4 `validate_schema_consistency()` checks | When changing a field, change it **together with** the Literal enums in the 4 plugin registries (adapters / optimizers / schedulers / losses) — startup will reject a mismatch |
| `studio/secrets.py` schema | `/api/secrets` + the Settings 7-tab form | Change the Pydantic model and the frontend form together |

### 3.4 Training-stack plugin boundary (ADR 0003)

Adding a variant (LoRA type / optimizer / scheduler / sampler / loss) goes through
the plugin registry — **don't touch** `main()` / `phases/` / `loop.py` /
`context.py`. Full steps in
[`runtime/training/README.md`](../runtime/training/README.md).

Regression guard: `test_plugin_registry.py` rejects these literals appearing in the
wrong place:
- `phases/optimizer.py` must not contain `if optimizer_type == "xxx"`
- `phases/models.py` must not contain `AnimaLycorisAdapter(...)`
- `sampling.py` must not contain `if sampler_name == "xxx"`

### 3.5 SSE event convention

To add a new event type:
1. **Register a row first** in the event catalog table in
   [architecture/studio-pipeline.md §6](architecture/studio-pipeline.md)
2. Backend goes through `bus.publish` in `studio/event_bus.py`
3. Worker subprocesses emit typed events by writing `__EVENT__:<type>:<json>` to
   stdout; the supervisor automatically injects `job_id / project_id / version_id /
   kind` — the worker **should not and cannot** forge these fields
4. The frontend uses `useEventStream` (a shared connection) — don't open your own
   `EventSource`

### 3.6 Stage advancement

The backend is authoritative; the frontend is read-only. Lives in
`studio/projects.py:advance_stage()`. The frontend Stepper's highlighted step is
derived from stage + version.stats. **Do not** mutate stage on the frontend, and
**do not** bypass `advance_stage` to update the DB directly.

### 3.7 Tagger abstraction

New ONNX taggers subclass `OnnxTaggerBase` to automatically get thread-pool
scheduling, GPU EP fallback, and model resolution. Register with the tagger
registry and the UI lists it automatically. **Do not** implement parallel ONNX
scheduling logic.

---

## 4. Commit / PR boundaries

### 4.1 One PR = one thing

- Adding a new feature **doesn't** opportunistically refactor something else
- Fixing a bug **doesn't** opportunistically rename things / change style /
  touch unrelated code
- Spot a typo near code you're changing? Leave it for a separate docs/chore PR

### 4.2 Multi-purpose PRs should trigger a warning

If the AI finds that what the author is asking for **actually** consists of ≥ 2
independent units (typical signals: the two groups of changes could be merged
separately, affect different modules, or have different motivations), **warn the
author first**:

```
Note: this task looks like it's actually N things:
1. [description] — affects [files]
2. [description] — affects [files]

Bundling them into one commit will make review / revert / cherry-pick harder.
Consider splitting into N PRs.
Do you want to bundle them anyway? What's the reason?
```

Only bundle them if the author gives a valid reason (must be an atomic change,
otherwise an intermediate state would break, or they're tightly coupled). Record
the reason in the PR description.

---

## 5. Pre-PR self-check

Applies to: medium/large changes. Small changes (typos / docs / renames /
test-only additions) skip this section.

**Always check**:

- The PR does exactly one thing; if it bundles more, the reason is in the description
- The change matches what was aligned on before starting — no scope creep
- No unrelated "opportunistic" changes
- Commit message / PR title follows Conventional Commits format
- Bug fixes include a regression test; features include test coverage

**Triggered by change type**:

- User-visible behavior change -> add an entry to `release_notes.yaml` (summary <= 80 characters, plain text)
- Changed user-perceptible behavior -> update the corresponding `docs/user-guide/` section
- Changed a single source of truth (version / schema / secrets / SSE event) -> sync all derived places
- Changed the plugin registry / a sister script / the Stepper -> check the §8 pitfalls list to confirm there's no regression
- UI change -> include screenshots in the PR description
- Triggers an ADR condition (§7) -> a new ADR has been drafted

CI already covers this — the AI doesn't need to re-run these manually: local tsc /
lint / pytest / vitest / plugin-registry startup validation.

If you find a violation, list the relevant items and confirm with the author — don't
silently let it pass.

---

## 6. Hard limits and strong recommendations

### 6.1 Hard constraints

- ❌ Never push directly to `master`
- ❌ Never `--no-verify` / skip tests / disable lint — fix the root cause of a failure
- ❌ Never connect directly to SQLite from a layer outside the backend (only go
  through `studio/db.py`)

### 6.2 Strongly discouraged (reviewers will ask you to split / rework)

- Opportunistic refactors inside a bug-fix PR
- Changing unrelated code / style / naming for "prettiness"
- Mocking things that an existing `tests/conftest.py` fixture already covers
- Hardcoding version numbers / paths / schema field names (see §3.3 Single source of truth)

### 6.3 Design principles

- Don't introduce a data flow that "parses free-form text at runtime" (use yaml/schema + tooling validation)
- Don't "correct" an external paper implementation by intuition (verify the paper / upstream first)
- Don't add abstraction for imagined future needs (abstract only after you see three similar pieces of code)

---

## 7. Architecture Decision Record (ADR) trigger conditions

**Must** write a new ADR for:
- Adding / swapping a core dependency (torch backend / training framework / DB / model library)
- Changing the meaning of an existing column in the SQLite schema (not adding a column — changing one)
- Changing the meaning of a field in the secrets.json schema
- Changing default behavior (e.g. the default sampler / attention backend)
- Removing a released endpoint / CLI flag / schema field
- A major cross-module refactor (like ADR 0003)

**Does not need** an ADR: bug fixes / new features that are opt-in and default-off
/ renaming internal functions / new components / tests / docs.

Template in [adr/README.md](adr/README.md). **The body of an Accepted ADR is never
changed** — only append `Superseded by #N` / `Deprecated` to its status line.

---

## 8. Pitfalls checklist (easy for AI / new contributors to trip on)

| Pitfall | Consequence | Defense |
|---|---|---|
| Changing the Stepper steps only in `components/ProjectStepper.tsx` | No visible UI change (that's dead code) — the actual place to change is the inline version in `Sidebar.tsx` | grep for the step strings to find every render site |
| Hardcoding a version number in `Sidebar.tsx` | Goes out of sync with `/api/health` | Always fetch from `/api/health` |
| Adding `if optimizer_type == "xxx"` in `runtime/training/loop.py` | Breaks the plugin boundary; `test_plugin_registry.py` rejects it | Add plugins through the `BUILDERS` dict |
| Changing one of the 7 re-exported names in `runtime/anima_train.py` | Breaks sister scripts | Top-level re-exports can be added to, never removed |
| Writing `summary: "**bold**"` in `release_notes.yaml` | `bump_version.py validate` rejects it | `summary` is plain text; put emphasis in `detail` |
| Sneaking a feature into a hotfix PR | Ties an unreleased dev feature to the hotfix | Hotfixes strictly fix exactly 1 bug |
| Reimplementing ONNX scheduling for WD14 / CLTagger | `OnnxTaggerBase` already provides this abstraction | Subclass the base class |
| Adding an SSE event type without registering it in the event catalog | Frontend and backend won't know the event exists | Follow the §3.5 process — register it in the architecture/studio-pipeline.md §6 table first |

---

## 9. The AI's role in collaboration

When helping a contributor change this project, an AI agent has three
responsibilities:

1. **Point out project-specific constraints**: single sources of truth, dependency
   direction, plugin boundaries, sister-script contracts, stage advancement, etc.
   These are documented project knowledge that a contributor may not have seen —
   the AI should proactively cite the relevant §3 section before starting work.
2. **Offer an alternative when a request violates a constraint**: if a contributor
   wants to add a literal `if optimizer_type == "lion"` branch, the AI explains the
   §3.4 plugin boundary and offers the `BUILDERS`-dict approach instead of silently
   complying or flatly refusing.
3. **Run through the §5 checklist before submitting a PR**: hand the check results
   to the contributor. The final decision to merge / not merge / redo belongs to the
   contributor / maintainer — the AI doesn't make that call for them.

---

## 10. Maintaining this document

- The AI / contributors keep hitting the same pitfall -> add it to the §8 pitfalls list
- Team conventions change -> update this doc + sync CONTRIBUTING / ADR / spec
- A new category of spec file is added -> add a row to the §2 file map
- Adding support for a new agent tool (e.g. Cursor's `.cursor/rules/`) -> put a stub
  at its conventional path pointing to this doc, **don't** copy the content (it will
  drift)
- Don't let this document balloon into a second CONTRIBUTING — its value is in
  **pointing the way + project-specific conventions**, not duplicated content
- Things any general-purpose AI tool already knows (type hints / pathlib / f-strings
  / TS strict / hooks rules, etc.) **should not** be written in here — that's just
  noise
