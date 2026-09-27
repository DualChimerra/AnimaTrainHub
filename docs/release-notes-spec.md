# Release Notes Writing Spec (for AI agents)

This document is for whoever writes release notes — human, Claude, or any LLM.
After reading it you should be able to, **without asking any follow-up
questions**: (1) know which file to change, (2) know where to pull content from,
(3) know what structure/style/tone to use, (4) know which tool to run to validate.

> **Core principle — the three surfaces share the same text, all of it user-facing**
>
> The `summary` + `detail` text in `release_notes.yaml` is rendered to three
> user-visible surfaces at once:
>
> 1. **Studio Web UI** Settings → System → the version card: summary shown as one
>    line, detail opens in a modal on click
> 2. **`CHANGELOG.md`** reachable from the repo root and the GitHub repo homepage;
>    `render-changelog` derives it from the yaml, both summary and detail go in
> 3. **GitHub Release body** (one page per tag under `/releases`); the maintainer
>    manually copies the matching section from CHANGELOG
>
> All three surfaces share the same yaml, so **both summary and detail must be
> written from the user's point of view**. The old rule of thumb "summary is for
> users, detail can use developer jargon freely" is wrong — detail is equally
> visible on the two user-facing surfaces CHANGELOG and the Release body.
> Technical references (implementation paths, internal symbol names, refactor
> rationale, design trade-offs, alternatives considered) belong in the **commit
> message + PR description**, not in the yaml.
>
> A 4-layer division of "archaeology":
>
> | Layer | Audience | Content |
> |---|---|---|
> | Commit message | git blame / engineers | root cause / fix for an atomic change |
> | PR description | reviewers / historical digging | design trade-offs, alternatives, scope |
> | CHANGELOG / Release body | users upgrading | "what changed when I upgrade" |
> | ADR | architecture-decision archaeology | "why this path and not another" |

## 1. Single source of truth: `release_notes.yaml`

**Always edit `release_notes.yaml`; never hand-edit `CHANGELOG.md`.**

`CHANGELOG.md` is **derived** from the yaml by
`tools/bump_version.py render-changelog`; any manual edit to it gets overwritten
on the next `render-changelog`. The GitHub release page reads `CHANGELOG.md`, so
the yaml is the source of truth.

`studio/__init__.py:__version__` and `studio/web/package.json:version` are also
rewritten together by `bump_version.py bump --version X.Y.Z` — don't edit them by
hand.

## 2. Data model (yaml schema)

```yaml
- version: "0.6.1"              # str, semver; aligned with the git tag / __version__
  date: "2026-05-13"            # str, ISO YYYY-MM-DD
  summary: "..."                # str?, optional. One-sentence overview of the whole
                                 # version (used in the CHANGELOG's top paragraph);
                                 # can be omitted, but the agent usually writes one
  entries:                      # list, >= 1 entry
    - kind: added               # enum, required, see §4
      summary: "..."            # str, required, <= 80 chars, plain text, user-facing
      pr_refs: [18, 34]         # list[int], optional; associated PR numbers
      detail: |                 # str, optional; markdown allowed; multi-line
        Detail block, can span multiple paragraphs.
```

**Version ordering**: the yaml is a list, with **the newest version first**
(at the top). `bump_version.py bump` automatically prepends the new block to the
head of the list.

## 3. Workflow

When an agent is asked to write release notes, follow this order:

### Step 1: Find the last release commit / tag

```bash
# The newest version currently in the yaml (the agent's previous release marker)
tail -n +1 release_notes.yaml | grep -m1 '^- version:'

# The matching tag in the repo (if one was cut)
git describe --tags --abbrev=0 2>/dev/null
```

If the yaml says `0.6.0` but the repo has no `v0.6.0` tag, use **the last commit
recorded at that time as the boundary** (usually the `chore(release): 0.6.0`
commit; find it with git log):

```bash
git log --oneline --grep='chore(release): 0.6.0' -1
```

### Step 2: Pull all PRs merged since the last release

```bash
# PR list (base is usually dev or master, pick whichever matches your release flow)
gh pr list --state merged --base dev --limit 100 \
  --search 'merged:>=2026-05-12' \
  --json number,title,body,labels,author,mergedAt
```

The date in `--search` = the date of the last release; `--base` matches the
branch your releases are cut from (default `dev`). If your releases are cut from
`master`, change the base to `master`.

Supplementary material (PR descriptions are sometimes too sparse):

```bash
# All commits associated with that PR
gh pr view <num> --json commits --jq '.commits[].messageHeadline'

# The full commit picture across the whole version range
git log <last_release_sha>..HEAD --pretty='%h %s'
```

### Step 3: For each PR, decide the kind and write summary + detail

**By default, write one entry per PR.** Exceptions:

- **Pure chore PRs** (dependency bumps / formatting / internal refactors with zero
  effect on user-facing behavior): **skip, no entry**
- **Pure docs PRs** (README / comments): **skip, no entry**
- **Large PRs spanning multiple areas**: can be split into multiple entries, each
  covering part of the PR; fill in the same PR number for all of them in `pr_refs`
- **A chain of PRs on the same topic** (a main feature PR + follow-up fix PRs):
  can be merged into one entry, with `pr_refs` listing all related PRs (agent's
  judgment call). E.g. `[18, 34, 35]`: main feature + P0 fix + a redo

### Step 4: Append to the yaml

Prepend the new version block to the top of the list; order entries within it by
importance (what users care about most first), **not** by PR chronology.

### Step 5: Validate + derive

```bash
python tools/bump_version.py validate
python tools/bump_version.py bump --version 0.6.1 --date 2026-05-13
```

`bump` will: (1) re-run validate, (2) update `studio/__init__.py:__version__` +
`studio/web/package.json`, (3) call `render-changelog` to rewrite `CHANGELOG.md`,
(4) print a suggested commit. It will **not** auto-commit or tag — that's a human
task.

## 4. `kind` categories (standard + our extension)

Based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), plus `improved`.

| `kind` | Meaning | When to use |
| --- | --- | --- |
| `added` | A brand-new feature / endpoint / UI / file | The user can "discover" something new |
| `changed` | Behavior / interface / default / UI flow **changed** | What used to work one way now works another way |
| `improved` | An existing feature got **better** in performance / UX / copy / error messages | Surface behavior unchanged, but it feels better |
| `fixed` | A bug fix | It used to be broken, now it's fixed |
| `removed` | A feature / endpoint / field / file was removed | The user can "feel" something is missing |
| `deprecated` | Marked for future removal, but still usable | Usually a heads-up that it will be `removed` in a version or two |
| `security` | CVE / auth / credential-leak class fixes | **Highest priority**, always written up separately |

**Judgment calls for ambiguous cases**:

- "Rewrote the UI but the user's workflow didn't change" → `changed` (the user can
  perceive the UI changed)
- "Rewrote the backend service internals, frontend API unchanged" → usually
  **no entry** (agent skips this PR)
- "Used to be slow, now 5x faster after optimization" → `improved`
- "Used to error occasionally, now fixed" → `fixed`
- "Added a new config option, default matches the old behavior" → `added` (default
  behavior unchanged, but the user can discover it)
- "Changed a feature's default from off to on" → `changed` (a default-value
  change is a behavior change)

## 5. `summary` writing rules

**Purpose**: shown as one line in the UI's release notes panel. A user should
know what this version did at a glance.

### Do

- **<= 80 characters** (including parentheses and the PR number). `bump_version.py
  validate` rejects longer ones
- **User-facing tone**: written from the user's point of view, not the developer's
- **Present tense / imperative**: "Fix X", "Add Y", "Improve Z" (not "We did X" or
  "X has been implemented")
- **End with `(#N)`**: link the PR; multiple PRs use `(#N, #M)`, up to 3
- **Specific, not generic**: "Fix Danbooru 403" is better than "fix some bugs"
- **Avoid technical jargon** (unless the user genuinely needs to know it): "Add GPU
  usage to training monitor" is better than "`_StatsThread` pushes an SSE event"
- **Consistent user-facing voice across all three surfaces**: after writing it,
  imagine this entry appearing simultaneously on the Studio version card,
  CHANGELOG, and the GitHub Release body — all read by users upgrading, not by
  reviewers or people doing git blame. Technical references belong in the
  commit/PR, not here

### Don't

- Markdown formatting (`**bold**` / `code` / links) — summary is plain text
- Wrapping the title in quotes: not `Add "LLM tagger"`, just `Add LLM tagger`
- A PR number that isn't at the end: not `#18 LLM tagger`, use `LLM tagger (#18)`
- Multiple sentences / sentence chains: not `Add X. Fix Y.` — split into two
  entries
- Vague words: "improved experience" / "various improvements" / "misc fixes" —
  what experience, specifically?

### Examples

| Bad | Good |
| --- | --- |
| `**LLM tagger + WandB monitoring**(#18)` | `LLM tagger (second tagger backend) + WandB training monitoring (#18)` |
| `Fixed some UI bugs` | `Fix Settings System tab not refreshing on switch (#52)` |
| `Refactor services/booru_api` | (no entry — purely internal) |
| `Improved experience (#33)` | `Queue output downloads use direct links + batch + sort (#33)` |
| `**onnxruntime-gpu silently falls back to CPU**(#29 Windows / #30 Linux)` | `onnxruntime-gpu silently falls back to CPU on Win/Linux (#29, #30)` |

## 6. `detail` writing rules

**Purpose**: shown when a user clicks into an entry wanting to know "specifically
what changed"; also used to fill in the secondary information when CHANGELOG.md
is derived from the yaml.

### Do

- **Markdown allowed**: lists, code fences, inline code, links, bold
- **Write the _why_, not just the _what_**: summary already covers what changed;
  detail covers why it changed / its impact / its edge cases
- **Include specific paths / commands / config keys**: so users can locate things
- **Use bullets for multiple sub-points**: `-` at the top level

### Don't

- Repeat what the summary already said
- Paste the commit message in verbatim — detail is also a user-facing surface
  (shown on both CHANGELOG and the Release body)
- Write implementation details instead of user-perceivable ones (e.g. "`_StatsThread`
  uses nvidia-ml-py instead of pynvml" → the user doesn't care, skip it)
- Write "implementation paths" / "internal class names" / "refactor rationale" /
  "alternatives considered" / "design trade-offs" — these belong in the commit
  message or PR description, not the yaml. For example, "`useEffect` deps use the
  `[task]` object reference, so React's shallow clone triggers a re-fetch" is a
  classic violation — users don't know what `useEffect` is
- Single-line detail (if it fits on one line, put it in summary instead)

### Example

```yaml
- kind: added
  summary: "Training monitor adds a Topbar system-resource pill (CPU / GPU / MEM / VRAM) (#37, #42)"
  pr_refs: [37, 42]
  detail: |
    - The Topbar always shows 4 equal-width pills (min-w 96px); pulled from
      `nvidia-ml-py`, replacing the unmaintained `pynvml`
    - The backend `_StatsThread` pushes to the frontend every 2.5s via the SSE
      `system_stats_updated` event
    - The Monitor view switched to an incremental protocol (stepwise deltas
      instead of a per-second snapshot); for a 10k-step training run, payload
      size drops from O(N) to O(1)
    - Cold start defaults to `max_points=0` (no downsampling); the frontend cap
      went from 5000 to 50000 to match the backend's `train_monitor`
```

## 7. Full end-to-end example

**Scenario**: the last release, `0.6.0`, was on 2026-05-12; now preparing to ship
`0.6.1`.

### Steps 1-2: gather

```bash
$ gh pr list --state merged --base dev \
    --search 'merged:>=2026-05-12' \
    --json number,title,body --limit 50

[
  {"number": 51, "title": "feat: webui self-update (ADR 0002)", "body": "..."},
  {"number": 52, "title": "feat(version-section): dual-channel upgrade panel", "body": "..."},
  {"number": 53, "title": "chore: bump nvidia-ml-py to 12.0.3", "body": "..."}
]
```

The agent's reasoning:
- #51 → a big feature, tied to ADR 0002, should get an entry (`added` or
  `changed`? an in-webui visual upgrade is a new capability → `added`)
- #52 → UI redesign, the old version was already clickable but the styling/state
  machine changed → `changed`
- #53 → pure chore dep bump, **skip**

### Step 3: write entries

```yaml
- version: "0.6.1"
  date: "2026-05-13"
  summary: "One-click in-webui upgrade + redesigned system settings version panel"
  entries:
    - kind: added
      summary: "One-click upgrade + restart + rollback inside the webui (ADR 0002) (#51)"
      pr_refs: [51]
      detail: |
        Studio no longer needs a CLI `git pull` + restart — just click Update on
        the Settings → System → version card:

        - `git fetch` + `git reset --hard origin/master` happens during cli.py
          startup (avoids the issue of the server process holding a lock on
          native modules)
        - A `tmp/restart` flag + the studio.sh/bat wrapper loop trigger the
          restart
        - Training/tagging jobs in progress reject the update (a precondition
          check returns 422)
        - On failure it automatically stays on the current version;
          `.last_version` records the previous commit to support one-click
          rollback
        - PR-D adds installer self-check (cli.py / studio.sh / studio.bat sha256
          changes → exit 42 → the wrapper re-execs itself) + a dev-channel
          toggle
        - See [`docs/adr/0002-webui-self-update.md`](docs/adr/0002-webui-self-update.md)
          for details
    - kind: changed
      summary: "Settings System → version card gets a dual-channel layout + inline preview (#52)"
      pr_refs: [52]
      detail: |
        - master / dev shown side by side; the current channel is highlighted
          ("you are here")
        - The master card shows the release tag (v0.6.0) prominently, no longer
          exposing the commit hash
        - The dev card shows a commit timeline; any commit is clickable to
          switch to (not just HEAD)
        - Actions no longer go through a modal dialog: a single click opens an
          inline preview panel (with release notes + pre-flight checks +
          cancel/confirm)
        - The dev-channel toggle moved below the card (demoted); it's forced on
          and locked while already on the dev channel
```

### Steps 4-5: validate + bump

```bash
$ python tools/bump_version.py validate
✓ 0.6.1 (2026-05-13) 2 entries
✓ 0.6.0 (2026-05-12) 5 entries
✓ ... (older versions)
validate ok

$ python tools/bump_version.py bump --version 0.6.1 --date 2026-05-13
[bump] studio/__init__.py: 0.6.0 → 0.6.1
[bump] studio/web/package.json: 0.6.0 → 0.6.1
[bump] CHANGELOG.md re-rendered from release_notes.yaml
[bump] git diff --stat:
   studio/__init__.py        | 2 +-
   studio/web/package.json   | 2 +-
   CHANGELOG.md              | 38 ++++++++++++++++++++
   release_notes.yaml        | 24 ++++++++++++

next: review changes, then:
   git add -A && git commit -m 'chore(release): 0.6.1'
   git tag v0.6.1
   git push --tags
```

## 8. Validation rules (run by `bump_version.py validate`)

Hard errors that block the yaml from going live:

- `version` isn't a valid semver (`X.Y.Z` or `X.Y.Z-suffix`)
- `version` is duplicated (the same version number appears twice)
- `version` isn't monotonically decreasing (the list should have the newest on top)
- `date` isn't ISO `YYYY-MM-DD`
- `entries` is an empty list
- `kind` isn't in the allowlist (added/changed/improved/fixed/removed/deprecated/security)
- `summary` is missing / empty / longer than 80 characters
- `summary` contains markdown characters (`*`, `` ` ``, `[`) — hints to use detail
  instead of summary
- `pr_refs` isn't a list[int], or a single int exceeds 9999

Soft checks that only warn (CI doesn't reject):

- `summary` < 10 characters (possibly written too tersely)
- `detail` < 20 characters (in this case, consider folding the detail content into
  summary instead)
- A single `version` block has >= 10 entries (possibly not well organized)
- A single PR appears in `pr_refs` across more than 3 entries (possibly split too
  finely)

## 9. Common anti-patterns

| Bad | Why | Good |
| --- | --- | --- |
| `summary: "deps"` | A single label word — the user can't tell what changed | `summary: "Upgrade nvidia-ml-py 0.6.0 → 12.0.3, replacing the unmaintained pynvml"` |
| `summary: "**bold**"` | summary doesn't allow markdown | Remove the bold; put emphasis in detail instead |
| `pr_refs: ["18", "34"]` | Must be int, not str | `pr_refs: [18, 34]` |
| `kind: refactor` | Not in the allowlist; refactors are usually imperceptible to users | Skip this PR, or use `changed` / `improved` |
| `summary: "Add LLM tagger (#18). Fix Danbooru 403 (#41)."` | Two independent changes crammed into one line | Split into two entries |
| `detail: "useEffect deps use the [task] object reference, React's shallow clone triggers a re-fetch..."` | detail is a user-facing surface (CHANGELOG / Release body) — it shouldn't expose React internals | `detail: "The linked config tab was re-fetching every 2 seconds during a training run, causing browser lag..."` |
| `detail: "ADR 0003 PR-A moved utils/ to the root, so Path(__file__).parent.parent resolves one level too few..."` | Implementation-path detail belongs in the commit message / PR | `detail: "A path-resolution bug introduced in v0.10.0 caused training to fail to find the JSON caption tool file..."` |
| Committing the yaml right after writing it | Didn't run `bump_version.py validate` first | Validate first — CI will reject errors anyway |

## 10. Maintaining this document

If a rule stops making sense, or the agent keeps getting something wrong in
practice, or team conventions change → update this document instead of working
around it. After updating, keep `bump_version.py`'s validate logic in sync so
"what the doc says" and "what the tool rejects" stay consistent.
