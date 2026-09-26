# 0005 — Update channel (master / dev) as a user view preference, decoupled from git worktree state

**Status**: Accepted
**Date**: 2026-05-16
**Decision makers**: @WalkingMeatAxolotl

## Background

After ADR 0002 (webui self-update) landed, the version panel in Settings → System had two long-standing latent issues, which surfaced right after the v0.8.0 release:

### Symptoms

After the user cut the v0.8.0 release (dev → master + PR merge + tag), before running `git pull` on local master, opening the version panel showed:

| UI element | Displayed | User's interpretation |
|---|---|---|
| master card "You are here" | I'm on master | branch name = master |
| master card `v0.8.0` | I have v0.8.0 installed | `__version__` string |
| master card "↑2 commits behind" | I'm behind | git `rev-list --count` |
| dev card "HEAD f6f202b" | dev's tip is this | remote |
| dev card list "● current" | this is what I have installed | commit hash comparison |
| "Switch to dev (f6f202b)" button | I can switch | onDev=false |
| `v0.8.0 → v0.8.0` arrow | same version before and after upgrade? |  |

Clicking "Switch to dev (f6f202b)" → preview appears → confirm → restart → the UI **doesn't change**, still shows "master · You are here".

### Two root causes

1. **The UI used git vocabulary instead of product language.**
   - `commits_ahead` came straight from the backend's `git rev-list --count`, and the frontend displayed it verbatim as "↑N commits behind."
   - But from the user's point of view, "behind" has exactly one meaningful axis: the version number. When version v0.8.0 == v0.8.0, there is no "behind" at all. The `__version__` string and the commit hash are two independent axes, and mixing them in the UI produced the contradiction of `v0.8.0 → v0.8.0` alongside "2 commits behind."

2. **The "channel" concept was tied directly to git worktree state.**
   - "Which channel am I on" was determined by `git rev-parse --abbrev-ref HEAD` (the branch name)
   - "Switch channel" was implemented as `git reset --hard <ref>` (which doesn't change the branch name)
   - The two mechanisms happened to work together in most cases: dev is usually ahead of master, so after a reset the commit hash changes and the UI appears to "move."
   - **The exception is right after a release**: local commit `f6f202b` already equals origin/dev's HEAD (the release commit itself was made on dev), so switching to dev is a no-op, the branch name doesn't change either → the UI is permanently stuck on "master · You are here."

Deeper still: the UI rendered the master card and the dev card **side by side on the same screen**, turning two channels that should be mutually exclusive into two cards that both look "live" at once, leaving users stuck in the contradictory question of "which channel am I actually on."

## Decision

**The channel is a user view preference, not git worktree state:**

1. **The user preference is persisted as `system.update_channel`** (`"stable"` / `"dev"`), stored in `secrets.json`. **Toggling it triggers no git operation at all** — it's a pure UI view switch.
2. **The actual "switch to dev HEAD" / "update to vX.Y.Z" / "rollback" actions are separate buttons**, decoupled from the channel preference — a user can "subscribe to the dev channel to follow development" while still "running the stable build."
3. **Only the card for the currently selected channel is shown on screen** (no more master + dev side by side).
4. **The backend introduces an "what's actually installed" classification, `installed_kind`** (`stable` / `dev` / `custom`), inferred by comparing commit hashes, **replacing the frontend's reliance on the `branch` field for product-level judgments**.
5. **Frontend copy uses only version-number / status language**, with no `commits` / `sha` / `branch` or other git vocabulary. "N commits behind" becomes "New stable version vX.Y.Z available" / "Up to date"; on the dev channel it becomes "N new updates."

### Backend API shape

`VersionInfo` (`/api/system/version`):

```python
@dataclass
class VersionInfo:
    version: str
    commit: str
    commit_short: str
    commit_time_iso: str
    branch: str              # for debugging only, frontend no longer makes product decisions from it
    tag: Optional[str]
    is_dirty: bool
    # ---- ADR 0005 ----
    installed_kind: str      # "stable" / "dev" / "custom"
    installed_label: str     # "v0.8.0" / "dev @ f6f202b · 2026-05-16" / "custom (feat/foo @ a1b2c3d)"
    stable_version: Optional[str]  # "vX.Y.Z", only set when stable
```

Classification rules (in priority order):

1. HEAD matches a `vX.Y.Z` release tag → `stable`
2. `__version__` matches some vX.Y.Z tag and the current commit's **tree matches** the tag commit's tree → `stable` (covers the case right after a release, where the release commit lives on dev and the tag sits on the merge commit)
3. commit == `origin/dev` HEAD → `dev`
4. else → `custom` (feature branch / detached)

`UpdateCheckResult` (`/api/system/update_check`):

```python
@dataclass
class UpdateCheckResult:
    channel: str
    current_commit: str
    latest_commit: str
    commits_ahead: int       # kept internally for debugging
    has_update: bool         # compatibility field = (state == "update_available")
    latest_tag: Optional[str]
    checked_at: float
    # ---- ADR 0005 ----
    state: str               # "up_to_date" / "update_available" / "ahead" / "detached"
    installed_version: Optional[str]
    latest_version: Optional[str]
    behind_count: int        # the "N updates" number the frontend uses
    error: Optional[str]
```

`state` inference:
- **master channel**: version number takes priority (`installed_version == latest_version` → up_to_date); falls back to commit comparison when there's no version number (custom / dev)
- **dev channel**: direct commit hash comparison + ahead/behind count

### Frontend

- `formatMasterStateText(check)` / `formatDevStateText(check)`: pure functions turning state + check data into user-readable copy
- `shouldShowMasterUpdateButton(check)`: state=update_available and latest_version is present
- `isDevSwitchButtonDisabled(check)`: state=up_to_date → disabled
- The toggle's visual switch goes through `<button role="radio">`, writes to `secrets.json` but **does not** call `/api/system/update`

### Migration

The old `system.show_dev_channel`: kept as a pydantic field for compatibility; `_migrate_legacy_schema` does a one-time mapping of `show_dev_channel=true → update_channel="dev"`, and the frontend writes both fields on PATCH to stay compatible with older-version rollback.

## Consequences

### Positive

- Right after a release, the user's version panel correctly shows "Up to date, v0.8.0" + "matches dev HEAD," with no more contradiction between "2 commits behind" and "switching to dev did nothing"
- UI copy is entirely free of git vocabulary, friendlier to non-git users
- Decoupling "what's installed" from "which channel is subscribed to" means future flexible combinations — like "running stable but watching dev previews" — won't require another architecture change

### Negative / to be evaluated

- `installed_kind=stable` detection relies on a `git diff --quiet` tree comparison, adding 1-2 extra git calls to every `current_version()` call. Shortly after a release, if local tags haven't fully been fetched yet, this can misclassify as custom, self-correcting after the next `check_update` fetches the tag
- The backend's `commits_ahead` / `has_update` fields are kept for compatibility but no longer read by the frontend — any external script reading those fields still works, and can migrate gradually

## Out of scope

- "Switch to this commit" (clicking an older commit in the dev list to jump to it) is kept, but as a secondary action inside the expanded dev-channel area
- Automatic checks + the Topbar red dot keep ADR 0002's "only watch master" decision; this isn't changed by the channel preference
- Fully translating the remaining git vocabulary elsewhere on the Settings page (e.g. wandb sync status under secrets) is out of scope for this ADR
