# 0002 — In-webui self-update (flag + shell wrapper loop)

**Status**: Proposed
**Date**: 2026-05-12
**Decision makers**: @WalkingMeatAxolotl

## Background

Currently users must upgrade via the CLI: `git pull` + restart `studio.sh` / `studio.bat`. Pain points:

- Less technical users get stuck on the git operations
- Across minor versions, `requirements.txt` changes require a manual `pip install`; it's documented in the CHANGELOG but not surfaced clearly
- A quirk specific to this kind of long-running-task tool: upgrading mid-training loses progress, and without pause-resume users won't dare click update

Discussion order:

1. Pause + resume training ([existing Ctrl+C mechanism](../../runtime/anima_train.py), see `signal_handler` / `signal.SIGINT`; after PR-A it moves to `runtime/training/phases/resume.py`. Still needs the supervisor to send the right signal + UI wiring)
2. One-click webui update (this ADR)

Together these two form a complete "painless upgrade + long-task safety net" story.

## Current-state inventory

The repo already implicitly has a "startup bootstrap" layer — it just doesn't loop or accept external triggers:

- `studio.sh` / `studio.bat` shell wrappers (one-time venv / system-level setup, single call to `python -m studio`)
- `studio/cli.py:cmd_run` Python launcher: `_ensure_python_deps` / `_web_dist_is_stale` triggers a frontend rebuild / `_apply_pending_install` / `_check_torch_cuda` / `_bootstrap_onnxruntime`, finally `subprocess.call(server)` which blocks
- `studio/services/pending_install.py` implements "pip operations deferred to the next startup" (designed for torch reinstalls)
- `cli.py:_web_dist_is_stale` already implements stale detection via git HEAD comparison + mtime comparison
- `studio.sh:135-148` already implements `requirements.txt` sha256 marker comparison → incremental `pip install -r`

## Industry research

Full detail in the conversation log (2026-05-12). Three main schools of thought:

| School | Examples | Restart mechanism | Pros/cons |
| --- | --- | --- | --- |
| A. flag + shell wrapper loop | A1111, SwarmUI | The service writes `tmp/restart` + `os._exit(0)`; an outer wrapper detects this and loops back to the entry point | Most stable, consistent across platforms; requires being launched via the wrapper |
| B. `os.execv` in-process restart | ComfyUI-Manager, SD.Next | Python calls `os.execv(sys.executable, ['python'] + sys.argv)` on itself to transform in place | Doesn't depend on a wrapper; on Linux+CUDA, VRAM isn't released (issue #576); on Windows, paths containing spaces fail to locate sys.executable |
| C. External launcher management | InvokeAI Launcher, StabilityMatrix, Pinokio | A desktop app (Electron / Avalonia / etc) spawns/kills the subprocess | Cleanest, zero self-update code in Python; requires building a separate desktop app |

**Industry consensus**:

- No one hot-swaps code or native modules (torch / onnxruntime lock their files after dlopen; can't even be reinstalled on Windows while loaded)
- Dependency updates always go through two steps: pull → restart → entry comparison → pip again
- Almost nobody protects running tasks — pressing update just force-kills them
- ComfyUI-Manager's `install-scripts.txt` (LAZY scripts) and our own `pending_install.py` are the same pattern
- oobabooga's `update_wizard` checks the installer's own hash after pulling, and aborts if it changed, prompting the user to rerun the wizard — guards against a "half-updated launcher," worth borrowing

## Decision

Adopt **school A (flag + shell wrapper loop) as the primary approach, with school B (execv) as a fallback**. The differentiator —
**hard constraint: all tasks must be paused / done / canceled before an update is allowed**.

### Supporting decisions

| Dimension | Decision |
| --- | --- |
| Scope | **git-clone deployments only**; future PyPI / Docker paths get their own update path, in a separate ADR |
| Automatic check | **On by default**; triggered **asynchronously** at startup (doesn't block cold start), result written to `studio_data/.update_cache` (TTL 24h); subsequent startups within 24h reuse the cache. Fails silently |
| Check target | **Always checks master only** (even if the toggle is on / the local branch is currently dev) |
| Update notification | **Update badge in the Topbar** (small red dot), shown only when a new master version is available; clicking jumps to Settings → System section |
| Channel | **Master only by default**; an "Advanced" toggle in Settings, "Show dev channel updates," unlocks a "manually check dev" + "update to dev" button pair; automatic-check behavior is unchanged |
| Telemetry | **None reported**. On failure, the UI offers a "copy `.update_log` into a GitHub Issue template" button instead |

## Rationale

- The repo's `studio.sh`/`studio.bat` + `cli.py` + `server.py` three-layer structure already aligns well with A1111; minimal changes needed
- The existing `pending_install` / `_STALE` / `_web_dist_is_stale` machinery was already designed for "startup bootstrap"; it can simply be looped
- Approach B's VRAM-not-released / spaces-in-path bugs are especially fatal for a training tool (a VRAM leak means an OOM on the next run); kept as a fallback for developers
- Approach C requires building a desktop app, beyond this project's current scope
- Task protection is an industry blind spot; the cost of losing a training run is far higher than losing an image generation, so this must be built

## Architecture

### Process layer

```
studio.sh / studio.bat                    ← OS-level setup (one-time)
    while true; do                        ← new loop
        python -m studio                  ← cli.py's main loop (mostly unchanged)
        rc=$?
        if [[ ! -f tmp/restart ]]; then break; fi
        rm tmp/restart
        if [[ $rc -eq 42 ]]; then         ← exit code for "installer itself changed"
            echo "[studio] launcher updated, re-exec wrapper"
            exec "$0" "$@"                ← the wrapper itself re-execs too
        fi
    done
    exit $rc
```

`studio.bat` gets the equivalent treatment (goto-based).

### Flag protocol

| File | Meaning | Writer → reader |
| --- | --- | --- |
| `tmp/restart` | needs a restart | server → wrapper |
| `studio_data/.update_pending` | git pull needed at startup; contents are the target ref (defaults to `origin/master`) | server → cli.py |
| `studio_data/.last_version` | the previous HEAD (for rollback) | cli.py (written before every pull) |
| `studio_data/.update_log` | log of the most recent update (pull / pip / errors) | cli.py |

Why the two-layer flag split: `tmp/restart` only handles "restart"; `.update_pending` handles "what to do after restarting." `tmp/` is git-ignored and consumed on restart; `studio_data/` is persistent and keeps history.

### Server endpoints

```
GET  /api/system/version              current HEAD / tag / commit time / dirty state / current_branch
GET  /api/system/update_check?channel=master|dev
                                       git fetch + comparison, returns {has_update, latest_tag, latest_sha,
                                       commits_ahead, changelog_md}
                                       defaults to channel=master; channel=dev only accepted
                                       when settings.show_dev_channel=true
                                       (the frontend doesn't even send the query when the toggle is off)
POST /api/system/update               body: {target: "origin/master" | "origin/dev" | "<sha>"}
                                       checks has_running_task() first → writes both flags → triggers graceful shutdown
POST /api/system/restart              writes only tmp/restart (no pull)
POST /api/system/rollback             re-runs update using .last_version's contents as target
GET  /api/system/update_log           tails .update_log
```

Every write endpoint first calls `supervisor.has_running_task()`; if true, returns 422 + the list of currently running tasks.

**Automatic check path**: an async thread at cli.py startup → `updater.check(channel="master")` → result written to `studio_data/.update_cache` (with a `checked_at` timestamp). On the next startup, if the cache hasn't expired (mtime < 24h), the fetch is skipped. "Manually re-check" on the Settings page overrides the cache.

**Dev channel path**: when the toggle is off, the UI exposes no dev entry point at all; once the toggle is on, Settings gets a "manually check dev" button (not triggered automatically, to avoid flooding developers with signals). Clicking it does a single `update_check?channel=dev`, and the result is **not written** to the cache (so it doesn't affect master's update_available state). The Topbar badge is **driven only by the master cache** — dev never lights up the badge.

### Cli.py changes

`cmd_run` main flow:

```python
def cmd_run(args):
    while True:
        rc = _ensure_python_deps()
        if rc != 0: return rc

        # New: check for a pending update, git pull first, then bootstrap
        _apply_update_pending()             # reads .update_pending, git pulls, writes .last_version, clears the flag

        # Existing: build / pending_install / torch / onnx
        ...

        # New: detect installer self-changes (borrowed from oobabooga)
        if _installer_changed_since_start():
            print("[studio] launcher code has been updated, please restart via the wrapper")
            return 42                       # special exit code, the wrapper sees this and execs itself

        rc = subprocess.call([sys.executable, "-m", "studio.server", ...])

        # Existing: after the blocking call ends, decide whether to restart
        flag = REPO_ROOT / "tmp" / "restart"
        if not flag.exists():
            return rc
        flag.unlink()
        # loop continue
```

`_installer_changed_since_start()`: at process startup, records the sha256 of `cli.py` + `studio.sh` + `studio.bat`; recomputes after the server exits, and returns true if changed. **Must exit outside the wrapper loop** (exit 42), otherwise it's still running the old wrapper / old cli.py.

### Updater module

`studio/services/updater.py`:

```python
def check() -> UpdateInfo: ...
    # git fetch origin
    # git rev-list HEAD..origin/master --count
    # pull the latest tag + changelog section

def request_update(target: str = "origin/master") -> None:
    # 1. precondition: no running task
    # 2. precondition: git status --porcelain is clean
    # 3. write .update_pending = target
    # 4. write tmp/restart
    # 5. trigger uvicorn graceful shutdown (server.should_exit = True)

def apply_pending() -> ApplyResult:                # called at cli.py startup
    if not has_pending(): return
    target = read_pending()
    write(LAST_VERSION, current_head())

    rc = subprocess.call(["git", "pull", "--ff-only", "origin", target])
    if rc != 0: return failed("git pull failed")

    if requirements_sha_changed():
        new_native = diff_requirements_for_native_packages()  # torch / onnx
        for pkg in new_native:
            pending_install.queue(pkg)                        # reuses the existing mechanism
        # pure python deps installed directly
        subprocess.call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    if web_package_json_changed():
        subprocess.call(["npm", "install"], cwd=WEB_DIR)

    clear_pending()
    return ok()

def rollback() -> None:
    target = read(LAST_VERSION)
    request_update(target)                          # goes through the same update flow
```

`apply_pending` is called at the top of `cli.py:cmd_run`'s main flow, **before** every bootstrap step other than `_ensure_python_deps` (because the new version may have changed the deps list).

### UI

Settings → new "System" section (or a standalone tab, fitting with the 6-tab structure after #36's reorganization).

**Default state** (toggle off, what the vast majority of users see):

```
┌─────────────────────────────────────────────────────┐
│ Version                                              │
│   Current  v0.6.0  (commit 879378e · 2 days ago)     │
│   Latest   v0.6.1  (commit abc1234 · 3h ago)         │
│   ↑ 5 commits ahead                                  │
│                                                       │
│   [View CHANGELOG]  [Update now]  [Roll back to v0.6.0] │
│                                                       │
│   Auto-check for updates  ☑   Every 24 hours          │
│                                                       │
│   ─────────────────────────────                       │
│   ▸ Advanced settings                                 │
└─────────────────────────────────────────────────────┘
```

**With the toggle on** (expand "Advanced settings" → check "Show dev channel updates"):

```
┌─────────────────────────────────────────────────────┐
│ Version                                              │
│   Current  v0.6.0  (commit 879378e · 2 days ago)     │
│   Latest   v0.6.1  (commit abc1234 · 3h ago)         │
│   ↑ 5 commits ahead                                  │
│                                                       │
│   [View CHANGELOG]  [Update now (stable)]            │
│   [Roll back to v0.6.0]                              │
│                                                       │
│   Auto-check for updates  ☑   Every 24 hours (stable channel only) │
│                                                       │
│   ─────────────────────────────                       │
│   ▾ Advanced settings                                 │
│   Show dev channel updates  ☑                         │
│                                                       │
│   Dev channel (for development / testing)             │
│   [Manually check dev]                                 │
│   ─ shown after checking ─                             │
│   dev  commit ef56789                                  │
│   ↑ 12 commits ahead of master (2h ago)                │
│   [View dev diff]  [Update to dev]                      │
│   ⚠️ dev is unreleased and untested — may crash or break training │
└─────────────────────────────────────────────────────┘
```

**"Update now" click flow** (master channel):

1. Frontend checks `/api/queue/running` — if a task is running, the button is disabled + a tooltip shows the current task
2. A modal appears: "This will close and restart studio. Expect 1-3 minutes. The webui will be unavailable during this time."
3. POST `/api/system/update` `{target: "origin/master"}`, the server writes the flags and responds 200
4. The frontend shows an "updating" status card: stages (git pull → pip → restart) + auto-polls `/api/health`
5. Once the server is back up, reconnect → toast "Updated to v0.6.1" + auto-refresh the page
6. On failure: pull and display `/api/system/update_log`, plus a prompt "please restart via the shell wrapper / contact the developer"

**"Update to dev" click flow**:

1. Same running-task check
2. The modal adds an extra red warning: "This is the dev branch and may be unstable. Rolling back requires a separate action."
3. POST `/api/system/update` `{target: "origin/dev"}`, the rest of the flow is the same as above
4. On success, toast "Updated to dev (commit ef56789)" — no tag shown (dev doesn't have one)

**Topbar badge**:

- Shows a small red dot only when the master cache has `has_update=true`
- Dev never lights up the badge even with new commits (avoids continuously pestering developers)
- Clicking the badge jumps directly to Settings → System section

### Reusing pending install

When a new `requirements.txt` changes a native module (torch / onnxruntime), it is **not** installed at update time (the old venv process still has it imported) — it's queued into `pending_install` instead. It runs at the next startup's `_apply_pending_install` stage — by then the new venv process has just started, with no already-loaded torch to conflict with.

This logic already exists ([`studio/services/pending_install.py`](../../studio/services/pending_install.py)) — no new code needed.

## State machine

```
Idle ──[check]──> HasUpdate ──[update click]──> CheckPrecondition
                                                    │
                              has running task ─────┤ → 422 "pause all tasks and try again"
                                                    │
                              local dirty ──────────┤ → 422 "local changes not committed"
                                                    │
                                              all clean
                                                    ↓
                                          WriteFlag + ServerShutdown
                                                    ↓
                                      CliLoop._apply_update_pending
                                                    ↓
                                            git pull, write .last_version
                                                    ↓
                                  requirements changed?──── yes ──> pip install / pending_install.queue
                                                    │
                                                    ↓
                                  installer changed?──── yes ──> exit 42 → wrapper exec self
                                                    │
                                                    no
                                                    ↓
                                       subprocess.call(server) → Idle
```

## Failure modes

| Failure | Impact | Mitigation |
| --- | --- | --- |
| `git pull` conflict (local uncommitted changes) | Update doesn't happen, server restarts | Rejected at the precondition-check stage; UI shows the list of dirty files |
| `git pull` network failure | Same as above | Recorded in `.update_log`, shown in the UI |
| `pip install` failure | Server may fail to start | Auto-rollback via `.last_version` + a UI prompt |
| Native module locked | .pyd in use while reinstalling into the venv | Deferred to a fresh process via `pending_install`; if the new version's entry point detects a required package is missing, it fails fast |
| User starts directly with `python -m studio` | Restart doesn't work (`tmp/restart` gets created but there's no wrapper loop) | cli.py checks whether `parent_pid` is studio.sh / studio.bat; otherwise the update endpoint returns 400 telling the user to use the wrapper |
| Update lands on a broken version (server won't start) | webui stays permanently black-screened | The wrapper detects N consecutive server-startup failures and aborts + prints a command-line hint for the user to run `git reset --hard <last_version>` |
| A training task is force-killed mid-run | LoRA progress lost | Enforced by the precondition (requires the "pause training" feature to already be in place) |
| `os.execv` path contains spaces (Windows) | The execv fallback path doesn't work | cli.py detects a space-containing path at startup and disables execv, requiring the wrapper instead |

## Implementation steps (split by PR)

1. **PR-A: basic restart pipeline** (no git pull)
   - `tmp/restart` flag + `cli.py` loop + `studio.sh`/`studio.bat` wrapper loop changes
   - `/api/system/restart` endpoint + Settings UI "restart server" button
   - Verify: the webui can restart the server, both the shell-wrapper and direct-python paths work
   - Estimated effort ~1 day

2. **PR-B: main update path**
   - `updater.py` + `/api/system/update_check` / `/update` endpoints
   - `apply_pending` hook at cli.py startup
   - Settings UI version section
   - Estimated effort ~1.5 days

3. **PR-C: rollback + logging + failure handling**
   - `.last_version` rollback + `/api/system/rollback`
   - `update_log` tail + UI failure prompts
   - Estimated effort ~1 day

4. **PR-D: installer self-check**
   - sha256 comparison for cli.py / studio.sh / studio.bat
   - Exit code 42 protocol + wrapper exec-self path
   - Estimated effort ~half a day

5. **PR-E: cross-platform QA**
   - Windows + Linux + macOS (if applicable) wrapper loop / process exit / flag file races
   - Spaces in path / non-ASCII path / abnormal exit / continuous-failure fallback
   - Estimated effort ~1 day

**Total ~5 days**. Prerequisite: **the pause-training feature must land first** (otherwise PR-B's precondition check is meaningless).

## Settled details

Design choices confirmed after discussion (expanded notes matching the "supporting decisions" table):

1. **Automatic check (merges original Q1 + Q6)**: on by default + triggered asynchronously + 24h cache. Cache is written to `studio_data/.update_cache`, TTL judged by mtime. Fails silently (corporate networks / unstable GitHub access shouldn't block startup). Settings offers "manually re-check" to override the cache
2. **Notification**: Topbar gets an update badge (small red dot), clicking jumps to Settings → System section. Only master triggers the badge
3. **Dev channel**: hidden by default under Settings' Advanced section; once the toggle is on, it unlocks two buttons — "manually check dev" + "update to dev." Automatic checking still only fetches master, to avoid developers being pestered by frequent commits. The rollback protocol **must support commit hashes** (dev has no tags)
4. **Telemetry**: not implemented. On failure, the UI offers a "copy update_log into a GitHub Issue template" button instead
5. **Scope**: git-clone users only. PyPI / Docker paths are out of scope for this ADR, to be covered by a future ADR

## Consequences

**Benefits**:

- Users no longer need the CLI to upgrade
- Combined with the pause feature, forms a complete "painless upgrade" story — a differentiator
- Reuses the existing bootstrap / pending_install / stale_check machinery, maximizing code reuse

**New constraints**:

- Must launch via `studio.sh` / `studio.bat` to get the full update flow (already the documented recommended path, but now becomes a hard requirement)
- Automatic git pull means the master commit effectively becomes the user's production code, raising the bar for regression testing (recommend introducing a staging step in the future: a full e2e run internally before merging dev → master)
- The training-pause feature becomes a hard prerequisite dependency

**Future debt**:

- The update path for major versions (0.x → 1.0 involving db schema migrations) needs its own design, not covered by this ADR
- In a multi-user scenario (if server sharing is built later), update-triggering permissions need an ACL
- If the venv itself is broken when an update rollback is attempted (a half-completed pending_install uninstall), the rollback path doesn't work — in a truly broken state, the user has to fall back to `./studio.sh --reinstall` from the shell; the UI can't recover it

## References

- Industry research: [A1111 restart.py](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/restart.py) / [ComfyUI-Manager manager_server.py](https://github.com/Comfy-Org/ComfyUI-Manager/blob/main/glob/manager_server.py) / [oobabooga one_click.py](https://github.com/oobabooga/text-generation-webui/blob/main/one_click.py) / [SwarmUI AdminAPI.cs](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/src/WebAPI/AdminAPI.cs)
- Existing repo mechanisms: [`studio/cli.py:cmd_run`](../../studio/cli.py) / [`studio/services/pending_install.py`](../../studio/services/pending_install.py) / [`studio.sh:135-148`](../../studio.sh) / [`runtime/anima_train.py`](../../runtime/anima_train.py)'s `signal_handler` (grep `signal.SIGINT`); after the ADR 0003 refactor this moves to `runtime/training/phases/resume.py`
- Pause-training prerequisite feature: tracked in a separate ADR / PR
