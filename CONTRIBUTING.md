# Contributing Guide

Intended audience:
- **Contributor** — adding a new feature / fixing a bug and want to open a PR
- **Maintainer** — deciding on merges / releases
- **AI agent** (Claude Code or similar) — helping with either of the roles above

---

## 30-second overview

```
contributor cuts a feature branch from dev
    ↓ pushes multiple commits (noise is OK, this is dev work)
PR into dev: squash merge (noise gets squashed into 1 PR-title commit)
    ↓ dev accumulates changes over time
maintainer batches up a set of features → bumps version → PR dev → master: merge commit (keeps per-feature granularity)
    ↓
tags v0.X.0 → GitHub Release
```

**Why this setup**: noise commits on a feature branch have no lasting value; on master, every commit = one complete feature; release boundaries are marked by git tags, not by squashing.

---

## Branching strategy

| Branch | Role | Who can push |
|---|---|---|
| `master` | release branch, always stable | **Nobody pushes directly**; only accepts release PRs from dev |
| `dev` | long-lived integration branch | By default only accepts contributor PRs (squash merge); **exception**: maintainers can push small chores directly (docs / typo fixes / config / `.gitignore`, etc.) |
| `feat/<topic>` `fix/<topic>` `refactor/<topic>` `docs/<topic>` `chore/<topic>` | contributor's temporary branches | Push it yourself, cut from `dev` |

Cutting a branch:

```bash
git checkout dev
git pull origin dev
git checkout -b feat/cool-thing
```

---

## Commit message convention

Format follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
type(scope): subject

Optional multi-line body explaining why (not what — the code is the what).

Optional footer (e.g. BREAKING CHANGE / Co-Authored-By).
```

**type**: `feat` / `fix` / `refactor` / `docs` / `chore` / `test` / `perf` / `style`

**scope** (optional): module name, e.g. `train` / `ui` / `web` / `tagger` / `setup` / `release`

Examples:

```
feat(train): add a three-way choice field for attention_backend
fix(ui): fix misaligned rendering of the SchemaForm path field
refactor: reorganize repo layout — runtime/ now holds the anima_* runtime core
docs: restructure docs/ into user-guide / architecture / adr
chore(release): v0.5.0 — versioning system + bump 0.4 → 0.5.0
```

**The PR title** = the commit message that lands on dev after the final squash, so **write the PR title following the format above**.

---

## PR process (Contributor)

### 1. Before you start

- Confirm the issue / the work you want to do is aligned with the maintainer (to avoid doing work that won't be accepted)
- Run the existing test suite and make sure it passes: `python -m studio test`

### 2. Development

- Commit freely on your own `feat/<topic>` branch (noise like "tweak again" or "fix lint" is fine — it gets squashed away at the end)
- **Keep it small**: one PR = one complete unit of work — don't cram 3 unrelated things into a single PR
- If you change functionality, add or update the corresponding tests

### 3. Self-check before submitting

```bash
python -m studio test          # backend pytest + frontend vitest must all pass
cd studio/web
npm run lint                   # frontend lint
npx tsc --noEmit               # frontend typecheck
```

### 4. Open the PR

- **Base branch is `dev`** (not master)
- **PR title uses Conventional Commits format** — this becomes the commit message after squashing
- **PR description** should include at least:
  - Background / motivation (why)
  - Summary of changes (what)
  - How it was tested (manual testing / automated test coverage)
  - Screenshots (for UI changes)

### 5. Review

- The maintainer reviews it and may ask for changes
- No need to open a new PR after making changes — just keep pushing to the same branch
- CI must pass fully (if configured)

### 6. Merge

- The maintainer uses **Squash and merge** (GitHub's default)
- Your noise commits get squashed into a single PR-title commit on dev
- Your feature branch can be deleted

---

## Release process (Maintainer)

### 1. Decide the version number

Once dev has accumulated a set of changes, **look at the most significant change** to decide which part of the version to bump:

| Changes on dev | Bump | Example |
|---|---|---|
| Only bug fixes / internal refactors / docs / config | **PATCH** | v0.5.0 → v0.5.1 |
| Contains **any** new feature / schema change / API change / behavior change | **MINOR** | v0.5.0 → v0.6.0 |

> During the 0.x phase, SemVer is applied more strictly: MINOR is treated as "possibly not backward-compatible"; PATCH **must never touch** existing config / API / behavior.
>
> Having few features in a cycle is normal — most releases are PATCH; bump MINOR once a feat has accumulated.

**Mapping to the `kind` field in release_notes.yaml**: if this cycle's entries are all `fixed` / `improved` /
`removed` / `deprecated` → PATCH; if any is `added` / `changed` → MINOR; a pure
`security` hotfix → PATCH (an urgent patch stays PATCH even if it includes a `changed` entry,
so users feel safe upgrading).

Pre-releases use a `-rc1` / `-beta1` suffix (v0.6.0-rc1).

### 2. Prepare the release on `dev`

```bash
git checkout dev
git pull origin dev
```

**Write release notes (structured YAML, agent-friendly)**

The source of truth is [`release_notes.yaml`](release_notes.yaml); `CHANGELOG.md`
is derived from the YAML by tooling — **do not hand-edit** CHANGELOG (it will be overwritten on the next render).

Following [`docs/release-notes-spec.md`](docs/release-notes-spec.md), insert a new version
block at the top of the YAML, with one entry per user-facing PR (pure chore / docs / internal
refactor PRs are skipped):

```yaml
- version: "0.X.0"
  date: "YYYY-MM-DD"
  summary: "one-line overview"
  entries:
    - kind: added            # added/changed/improved/fixed/removed/deprecated/security
      summary: "one user-facing line, ≤ 80 chars, ending with the PR number (#NN)"
      pr_refs: [NN]
      detail: |              # optional, multi-line markdown
        multi-line markdown detail
```

Pull all PRs merged during this cycle (example command):

```bash
gh pr list --state merged --base dev \
  --search 'merged:>=<date-of-last-release>' \
  --json number,title,body,labels,author,mergedAt --limit 100
```

See release-notes-spec.md for detailed do/don't guidance, `kind` classification rules, and good vs. bad examples.

**`bump_version.py` syncs the version number and rewrites CHANGELOG.md in one step**

```bash
python tools/bump_version.py validate                 # validate the yaml schema first
python tools/bump_version.py bump --version 0.X.0     # sync the version number + rewrite CHANGELOG.md
```

The tool automatically updates:

1. `studio/__init__.py` — `__version__ = "0.X.0"`
2. `studio/web/package.json` — `"version": "0.X.0"`
3. `CHANGELOG.md` — derived from the YAML

`bump` automatically runs `validate` first; schema errors (kind not in the allowed list / summary over 80 chars /
version ordering wrong / pr_refs not an int / etc.) will cause it to reject outright.

**One place still needs a manual edit**: the shields.io badge URL at the top of `README.md` and the
"current version **0.X.0**" line in the "## Version" section (README styling is a strong personal preference,
so the tool leaves it alone for now).

> The frontend Sidebar fetches its version number from `/api/health` — **do not hardcode it in Sidebar.tsx**.

Run a grep as a safety net to catch anything missed (architecture doc diagrams, ADR references, etc.):

```bash
grep -rn "<old version number, e.g. 0.5.0>" --include="*.md" --include="*.json" \
  --include="*.py" --exclude-dir=node_modules --exclude-dir=venv
```

Leave anything in the output that's a "historical record" untouched (past CHANGELOG sections, ADRs referencing
the facts of a previous release, historical entries in release_notes.yaml); update anything that's "currently
displayed" to the new version (README badge / doc diagrams / any sentence describing the "current version").

**Review the full contents of README + docs** (not just the version number — whether the content itself is still accurate):

The version number is a "pointer" — even with the pointer correct, the content can still drift. A release is
the window to realign the content with the actual current functionality. For anything changed this cycle, double-check
the corresponding descriptions / examples / screenshots / data models / field names / UI wording in the docs.

- **Read through README.md end to end**: project overview / feature list / screenshots / the 7-step pipeline /
  Studio Web workbench description / quick start / system prerequisites / project structure / acknowledgments —
  check each section against this cycle's PRs to see if it's still accurate (a feature was removed / renamed /
  behavior changed / a major new feature isn't mentioned)
- **Read through docs/**:
  - `docs/user-guide/` — user-facing (tag format / training tips / caption format, etc.) — if the
    schema / defaults / UI flow changed this cycle, sync the corresponding sections
  - `docs/architecture/` — developer-facing architecture overview — check whether **version numbers / module
    paths / field names in the diagrams** have drifted against the new code (grep can't catch structural drift)
  - `docs/adr/` — historical decision records, **leave as-is, do not edit** (an ADR is a point-in-time snapshot;
    referencing "the version at the time" is part of the historical record). Unless an ADR was implemented this
    cycle, in which case you can flip that ADR's status at the top from Proposed → Accepted
- **Screenshots / GIFs**: for a major UI change this cycle (e.g. Settings / the training page / the auto-update
  panel), screenshots in README / docs may be stale — retake them if needed

If you find drift, **fix it in this same release commit** — don't put it off to next time. Docs tend to drift
from the version number more than you'd expect.

Make one commit:

```bash
git add release_notes.yaml CHANGELOG.md \
        studio/__init__.py studio/web/package.json README.md
git commit -m "chore(release): v0.X.0"
git push origin dev
```

**The release commit body is empty / minimal by default.** A release commit is a mechanical version bump
+ render — it's not where the actual fix happened. The root cause / fix for a bug already lives in the original
fix commit's message + PR description; CHANGELOG.md / the GitHub Release body carry the user-facing description.
Repeating that information in the release commit body just blurs the layering of git log archeology
(commit messages are for engineers, release notes are for users — each has its own job). Example: the v0.9.1 /
v0.10.0 release commit bodies are both empty.

### 3. Open the Release PR: dev → master

- **Base = master, compare = dev**
- **Title**: `feat: v0.X.0 — <theme>`
- **Description**: copy the corresponding CHANGELOG section (so the full set of changes is visible right on the PR page)

### 4. Merge

- Choose **Create a merge commit** (**do not squash**) — this preserves the per-feature commit granularity from dev
- The merge commit message defaults to `Merge pull request #N from .../dev`; keep the extended description brief, e.g. `Version: v0.X.0 — <theme>`

### 5. Tag the release

```bash
git checkout master
git pull origin master
git tag -a v0.X.0 -m "v0.X.0 — <theme>"
git push origin v0.X.0
```

### 6. Publish the GitHub Release

- Go to `https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/releases`
- Find the tag you just pushed → **Create release from tag**
- **Title**: `vX.Y.Z` (plain version number, **without** the theme — every past release title in this repo
  follows this style: v0.10.0 / v0.9.1 / v0.8.3 ... The theme belongs in the tag message, see §5)
- **Body**: copy the corresponding section from [`CHANGELOG.md`](CHANGELOG.md)
- Check **Set as the latest release**
- Publish

---

---

## Hotfix fast path (bypassing dev)

**Only use this when you can't wait for the next release** — production is on fire / a security issue / dev has a feature that shouldn't ship yet but a bug still needs an immediate fix. Ordinary bug fixes go through the normal dev flow and **do not count as a hotfix**.

```bash
# 1. Cut from master (not from dev)
git checkout master && git pull origin master
git checkout -b hotfix/<topic>

# 2. Fix + test + commit + open a PR
#    - Base is master (not dev)
#    - Title: fix(scope): ... describing the urgent bug
#    - Description explains clearly why it can't wait for the next release

# 3. Squash and merge into master

# 4. Follow steps 2 / 5 / 6 of the release process above:
#    - bump PATCH (__init__.py + package.json + CHANGELOG.md + README.md)
#    - commit the version bump directly on master (also a hotfix exception)
#    - tag v0.X.Y + publish the GitHub Release

# 5. Must sync back to dev! Otherwise dev is missing this fix and it will regress on the next release
git checkout dev
git merge master
git push origin dev
```

Step 5 is the critical one — it's easy to forget. After a hotfix, dev must contain the fix, or the next release (dev → master) PR will show this bug coming back.

---

## Versioning rules

Follows [SemVer](https://semver.org/): `MAJOR.MINOR.PATCH`

| Phase | Rule |
|---|---|
| **0.x** (current) | MINOR is treated as a breaking upgrade (schema changes / API changes / major refactors all go in MINOR); PATCH is reserved for hotfixes |
| **1.0+** | PATCH = bug fixes / internal refactors; MINOR = backward-compatible new features; MAJOR = breaking changes |

Single source of truth: `studio/__init__.py:__version__`. The FastAPI app version is derived from this (exposed via `/api/health`), and the frontend Sidebar fetches `/api/health` to get it.

---

## Documentation

| Where | Content |
|---|---|
| [README.md](README.md) | Project overview + quick start + project structure |
| [CHANGELOG.md](CHANGELOG.md) | Version history (Keep a Changelog format) |
| [docs/README.md](docs/README.md) | Documentation entry point |
| [docs/user-guide/](docs/user-guide/) | User-facing: tag format / training tips / regex sets / caption format |
| [docs/architecture/](docs/architecture/) | Developer-facing: cross-module architecture overview |
| [docs/adr/](docs/adr/) | Architecture Decision Records (ADR) |

**Changed a feature?** Update the corresponding docs at the same time.

**Architectural "we chose X over Y" decisions**: write a new ADR (`docs/adr/NNNN-title.md`); see [docs/adr/README.md](docs/adr/README.md) for the template. Example: [ADR 0001](docs/adr/0001-lokr-via-lycoris-lora.md) decides that LoKr goes through lycoris-lora instead of switching to sd-scripts.

---

## Repository layout conventions

```
models/      Model implementations + weight storage
utils/       lycoris_adapter / optimizer / shared training utilities
runtime/     Anima runtime core (separate process; launched as a subprocess by Studio / can also run as a standalone CLI)
studio/      Web backend + frontend
tools/       User-facing CLI + setup helpers (includes dev bench)
docs/        Three sections: user-guide / architecture / adr
tests/       Backend pytest + frontend vitest (frontend tests live in studio/web/src/**/*.test.ts*)
```

Dependencies flow in one direction: `models → utils → runtime → studio → tools`. Do not import in reverse.

---

## Additional notes for AI agents

**AI agents must read** [`docs/AGENTS.md`](docs/AGENTS.md) **before taking on a task in this repo** — the full convention for code quality, consistency, maintainability, and AI collaboration (including the pre-work alignment protocol, the single-source-of-truth checklist, a pitfalls list, and the PR self-check). This section is the condensed quick reference.

If you're Claude Code / Cursor / a similar agent helping a maintainer or contributor:

**Do**:

- One PR = one complete unit of work — don't mix unrelated changes together
- When changing the version, **all four locations must stay in sync**: `studio/__init__.py` + `studio/web/package.json` + a new section at the top of `CHANGELOG.md` + `README.md` (the top badge + the current-version line in the "## Version" section)
- Don't refactor unrelated code while fixing a bug (unless the maintainer explicitly asks for it)
- Test coverage: bug fix → add a regression test; feat → add a test for the new functionality
- Conventional Commits' `type(scope):` prefix must be in English
- Final commit message / PR description formatting follows whatever the maintainer tells you

**Don't**:

- Never push directly to `master`, under any circumstances
- Don't create git tags / GitHub Releases — that's the maintainer's call; only do it once authorized
- Don't squash / rebase / force-push a branch that's already been pushed to origin (unless the maintainer explicitly asks for it)
- Don't bypass tests (`--no-verify` / skipping tests / disabling lint) — fix the root cause first
- Don't change unrelated code / style / naming just to make it "nicer" (unrelated changes make review harder)

**Prefer reusing existing code**:

- Before writing a new component, grep the repo to see if one already exists (`useProjectCtx` / `Toast` / `PathPicker` / `SchemaForm`, etc. already exist)
- Before writing a new CLI, check `tools/` for overlap
- Before writing new tests, check the fixtures in `tests/conftest.py`
