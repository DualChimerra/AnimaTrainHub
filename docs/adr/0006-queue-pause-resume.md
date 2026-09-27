# 0006 — Queue task pause / resume + queue hold / release scheduling

**Status**: Accepted (PR-1 #97 / PR-2 #98 / PR-3 #99 / PR-4 #100 all merged into dev; PR-5 removes the feature flag, enabled by default) + Addendum 1 2026-05-19 (dev training-stack audit + pause-semantics reversal, see "Incremental updates" at the end)
**Date**: 2026-05-18 (initial) / 2026-05-19 (Addendum 1)
**Decision makers**: @WalkingMeatAxolotl

> **Maintenance convention**: this ADR evolves across multiple rounds of
> discussion. The already-accepted initial decisions (candidate solutions /
> decision / rationale / consequences / out of scope / references) are
> **never modified or deleted**; when a later audit finds something, reverses
> a decision, or patches it, a new `### YYYY-MM-DD — Addendum N: Title`
> subsection is appended to the "## Incremental updates" section at the end
> of the file, stating which section of the original it affects.

## Background

Today the only way for the queue system to stop training is "cancel" — the
supervisor sends a hard termination signal (Windows `CTRL_BREAK_EVENT` /
POSIX `SIGTERM`), the child process exits, state isn't saved, and re-running
must start from step 0.

The CLI side actually already has a full save/resume chain:

- `runtime/training/context.py:109` `handle_interrupt` saves the state +
  LoRA + finishes wandb;
- `runtime/training/phases/resume.py:82` binds it to SIGINT via
  `signal.signal(SIGINT, ctx.handle_interrupt)`;
- `runtime/training/phases/resume.py:60-79` implements `--resume-state`,
  loading state + restoring monitor history.

But this chain **can only be triggered by manually pressing Ctrl+C in the
console** — the `CTRL_BREAK_EVENT` / `SIGTERM` that the supervisor's cancel
sends never hits the SIGINT handler, bypassing handle_interrupt entirely.

`ResumeFieldPicker` lets the user manually pick a `.pt` to resume from when
**creating a new task**, but that's a separate task: a new task_id, a new
log, and loss / monitoring history disconnected.

`save_state_every` / `save_state_every_epochs` periodically write a `.pt`,
but the path doesn't include the task_id
(`<output_dir>/training_state_step{N}.pt`), so multiple tasks running under
the same version will overwrite each other. This is a latent bug, and
implementing pause/resume would amplify it into actual data loss.

**User pain points (by frequency)**:

- Wanting to free up the GPU mid-training for something else (generate /
  another LoRA) → currently the only option is to cancel and lose progress;
- Shutting down / going temporarily offline → same as above;
- Loss looks wrong partway through and the user wants to pause to analyze
  before deciding what to do → same as above.

The detailed state machine, user cases, file storage, and UI flow were
discussed across three rounds in
`docs/design/queue-pause-resume-design.md` (PM / end-user / designer
three-way review); this ADR inherits that document's logic model, focusing
on locking in decisions and code-level direction.

## Candidate solutions

### A: signal channel, reusing the handle_interrupt chain (adopted)

The supervisor sends a signal to the child process, triggering the existing
handle_interrupt. POSIX uses SIGINT; Windows uses `CTRL_BREAK_EVENT` + the
child registers an extra SIGBREAK handler.

- Pros: reuses an existing save chain; signals are a standard cross-process
  notification mechanism; small change footprint.
- Cons: on Windows, a `CREATE_NEW_PROCESS_GROUP` child can't receive
  `CTRL_C_EVENT`, only `CTRL_BREAK_EVENT`; Python maps it to SIGBREAK rather
  than SIGINT, requiring the child to register it separately. Needs a spike
  to validate the whole chain.

### B: sentinel file / named-pipe IPC

The supervisor writes a sentinel file; the child opens a watcher thread that
polls periodically, and calls handle_interrupt on its own once it sees the
file.

- Pros: fully consistent cross-platform behavior, independent of signal
  semantics.
- Cons: adds a new IPC channel; the watcher thread adds CPU overhead + a
  trigger delay (the poll interval); sentinel cleanup / leftovers become a
  new problem; unnecessary if candidate A works.

Kept as a fallback if candidate A's spike fails.

### C: build a full RPC layer from scratch (gRPC / WebSocket)

Too heavyweight, over-engineered. Rejected.

### D: fake pause — after cancel, automatically resume from the most recent save_state_every checkpoint

- Pros: zero code cost.
- Cons: strongly depends on `save_state_every` (default 0, most users don't
  enable it); the resume point is imprecise (the most recent periodic save
  could be hundreds of steps behind); the UI would falsely claim "paused
  successfully" when it's actually a cancel — long-term debt.

Rejected.

## Decision

**Candidate A is adopted**: a signal channel + reusing handle_interrupt. The
specific decisions are summarized below; detailed reasoning is in the
corresponding sections of the design document (references marked §N point to
the design doc).

1. **Add a new task status, `paused`** (non-terminal, non-live). No
   intermediate `pausing` / `resuming` states are introduced. (design §1-2)
2. **Add a queue-hold switch**: a single bool in the db kv store, preserved
   across server restarts. It is not part of the task state machine.
   (design §3.2)
3. **Terminology**: tasks are **paused / resumed** (pause / resume); the
   queue is **held / released** (hold / release). Deliberately different
   verbs are used to avoid ambiguity. (design §3)
4. **State file path**: `<output_dir>/state/task_<TID>/`, with pause files
   given a `pause_` prefix to distinguish them from periodic saves. This also
   fixes today's latent bug in passing. (design §5.1, §5.3)
5. **Config snapshot**: on pause, write `pause_step_<N>.config.json` to disk,
   serializing all args / dataset / sample parameters currently in actual use
   by training. Resume strictly uses the snapshot, never reading the task
   table / version config / an external yaml. (design §5.7, §8.5)
6. **Pause-file lifecycle**: managed automatically along with the paused
   status; deleted together in three cases — successful resume, full cancel,
   or task deletion. At any moment, a task has at most 1 pause-file pair.
   (design §5.5)
7. **UI pause-progress modal**: clicking pause immediately locks the screen
   with a modal that guides the user through the whole process
   (saving → success/timeout/failure); a 30s timeout doesn't silently
   downgrade to cancel — the user gets a three-way choice instead. (design §4.3)
8. **Hold confirmation modal**: if a running task is detected, ask an
   additional question — "also pause it?" — with a radio button linked to
   the primary button's label; there's no combined "pause everything"
   button. (design §4.4)
9. **Hold status is shown as a banner, not a task chip** — the banner is a UI
   element, not part of the task state machine. (design §4.1)
10. **Premature-pause guard**: the UI's `is_pausable` signal controls the
    button's visibility; the API rejects it too, as defense in depth.
    (design §8.1)
11. **No automatic state-save on server crash** — instead, guide users to
    enable `save_state_every`. (design §9)

### Backend code direction

#### `runtime/training/context.py`

Changes to `TrainingContext.handle_interrupt`:

```python
def handle_interrupt(self, sig, frame) -> None:
    if self.interrupted:
        sys.exit(1)
    self.interrupted = True

    state_path = _build_pause_state_path(self.output_dir, self.task_id, self.global_step)
    config_path = state_path.with_suffix(".config.json")

    _write_config_snapshot(config_path, self.args, self.sample_prompts)
    save_training_state(state_path, ...)  # existing call
    self.injector.save(...)
    self.wandb_monitor.finish()

    self._emit_event("pause_state", {
        "state_path": str(state_path),
        "config_path": str(config_path),
        "step": self.global_step,
    })
    sys.exit(0)
```

A new field, `TrainingContext.task_id: Optional[int]`, is read from the
`LORA_TASK_ID` env var at startup (injected by the supervisor when spawning).

`_build_pause_state_path` / `_write_config_snapshot` / `_emit_event` become
module-level helper functions, landing in `runtime/training/state.py` or a
new file `runtime/training/snapshot.py`.

Config snapshot contents (a candidate list; the final result is whatever
gets serialized at implementation time):

- All `args.*`: lr, optimizer, optimizer_args, scheduler, batch_size,
  grad_accum, max_train_steps, num_epochs, noise schedule, loss weighting,
  network_dim, network_alpha, dropout, rank, ...
- dataset_config / resolution / caption_extension / shuffle / repeat
- output_dir / output_name / sample_prompts / sample_every / save_every_n_steps
- Key model paths (base, vae, text encoder) — stores the path, not a hash
- random seed
- **Not stored**: wandb run id (already finished), monitor live state
  (already dumped inside the .pt)

#### `runtime/training/loop.py`

The periodic save's write path is likewise changed to a per-task
subdirectory, keeping the `step_<N>.pt` naming (no pause prefix — it's
distinguished from pause files by the naming convention alone). This is
fixing the latent bug in passing, shipped cleanly as its own PR first (see
"Suggested PR split").

#### `runtime/training/phases/resume.py`

```python
def run(ctx: TrainingContext) -> None:
    # ... existing logic ...
    signal.signal(signal.SIGINT, ctx.handle_interrupt)
    if os.name == "nt":
        signal.signal(signal.SIGBREAK, ctx.handle_interrupt)  # new

    # emit before entering train_loop:
    ctx._emit_event("train_loop_started", {})

    # emit after load_training_state succeeds (after the existing load_training_state call):
    if args.resume_state:
        # ... existing load ...
        ctx._emit_event("resume_state_loaded", {"path": args.resume_state})
```

#### `studio/supervisor.py`

New fields on the `_Slot` dataclass:

```python
@dataclass
class _Slot:
    # ... existing fields ...
    pause_pending: bool = False
    pause_state_path: Optional[Path] = None
    pause_config_path: Optional[Path] = None
    pause_step: Optional[int] = None
    train_loop_started: bool = False
```

New methods:

- `pause(task_id) -> bool`: at the same level as `cancel(task_id)`
- `_signal_pause_async(slot)`: shaped like `_signal_terminate_async`, sends
  the pause signal, **does not force-kill on timeout** (the modal decides
  what happens next)
- `_send_pause_signal(proc)`: `CTRL_BREAK_EVENT` on Windows,
  `os.kill(pid, SIGINT)` on POSIX

`_finish_slot`'s three-way branch (replacing `supervisor.py:972-977`):

```python
if slot.pause_pending and slot.pause_state_path:
    status = "paused"
elif slot.cancel_pending:
    status = "canceled"
elif rc == 0:
    status = "done"
else:
    status = "failed"
```

The `paused` branch additionally sets `paused_state_path` /
`paused_config_path` / `paused_step` / `paused_at` when writing to the db.

`_on_line` recognizes the new events `pause_state` / `train_loop_started` /
`resume_state_loaded`, updating slot fields:

- `pause_state` → sets `pause_state_path` + `pause_config_path` + `pause_step`
- `train_loop_started` → sets `train_loop_started=True`
- `resume_state_loaded` → marks the old pause-file pair for cleanup (cleaned
  up in `_on_finish` or a separate thread)

The existing logic that marks `status='running'` tasks as failed on startup
reload must explicitly skip `paused`.

#### Cancel and pause colliding on Windows signals

Today, cancel sends `CTRL_BREAK_EVENT` on Windows, and pause needs to send
the same signal → the child process can't distinguish intent.

**Decision**: on Windows, cancel **no longer sends a soft signal** — it goes
straight to `taskkill /T /F` to hard-kill the process tree. Reasoning:
cancel's semantics are inherently a hard interrupt, and "graceful first, then
force-kill" is meaningless on Windows (the 30s grace period almost always
ends in a force-kill anyway). `CTRL_BREAK_EVENT` is reserved exclusively for
pause.

POSIX cancel keeps sending SIGTERM (force-kill after grace), while pause
sends SIGINT — no collision.

#### `studio/db.py`

Migration adds columns:

- `paused_state_path TEXT NULL`
- `paused_config_path TEXT NULL`
- `paused_step INTEGER NULL`
- `paused_at REAL NULL`

```python
VALID_STATUSES = {"pending", "running", "done", "failed", "canceled", "paused"}
TERMINAL_STATUSES = {"done", "failed", "canceled"}  # paused not added
```

`next_pending` is unchanged (it naturally skips paused).

The hold switch is stored using kv storage:

```python
def get_queue_held(conn) -> bool: ...
def set_queue_held(conn, held: bool) -> None: ...
```

Choose either a new `app_settings(key TEXT PRIMARY KEY, value TEXT)` table
or an existing kv table.

#### New `studio/server.py` endpoints

```
POST /api/queue/{task_id}/pause   → supervisor.pause(task_id)
POST /api/queue/{task_id}/resume  → see below
POST /api/queue/hold              → db.set_queue_held(True)
POST /api/queue/release           → db.set_queue_held(False)
GET  /api/queue/hold              → {"held": bool, "pending_waiting": N}
```

`/api/queue/{id}/pause` checks `is_pausable` (looking at the supervisor
slot's `train_loop_started`); returns 409 if not ready.

`/api/queue/{id}/resume` flow:

1. Read the task's `paused_state_path` + `paused_config_path`.
2. Verify the files exist (returns 409 if not, guiding the user toward
   starting a new task via ResumeFieldPicker).
3. Change the task's status from paused back to pending.
4. On the next dispatch round, cmd_builder detects `paused_state_path` /
   `paused_config_path`:
   - Builds args from the `paused_config_path` snapshot;
   - The only override: `--resume-state <paused_state_path>`;
   - Injects env `LORA_TASK_ID=<task_id>` (to keep the state subdirectory
     consistent).

`cancel_task` is enhanced: allows paused → canceled directly by updating the
db + clearing the pause-file pair (the process has already exited, no signal
needs to be sent).

The supervisor's main dispatch loop checks:

```python
if db.get_queue_held(conn):
    continue  # skip this dispatch round; already-running tasks are unaffected
```

### Frontend code direction

#### `studio/web/src/types.ts` + API client

```ts
type TaskStatus = 'pending' | 'running' | 'done' | 'failed' | 'canceled' | 'paused'
const TERMINAL: TaskStatus[] = ['done', 'failed', 'canceled']  // paused not added
```

Additions to `studio/web/src/api/client.ts`:

```ts
pauseTask: (id: number) => req(`/api/queue/${id}/pause`, { method: 'POST' }),
resumeTask: (id: number) => req(`/api/queue/${id}/resume`, { method: 'POST' }),
holdQueue: () => req(`/api/queue/hold`, { method: 'POST' }),
releaseQueue: () => req(`/api/queue/release`, { method: 'POST' }),
getQueueHold: () => req<QueueHoldState>(`/api/queue/hold`),
```

The monitor SSE protocol gains an `is_pausable: boolean` field, derived by
the supervisor from `slot.train_loop_started`.

#### `studio/web/src/pages/Queue.tsx` / `QueueDetail.tsx`

- Top banner (shown only when `held=true`, sticky);
- Top actions: pause / cancel / hold queue / release queue;
- Pause button: hidden (not disabled) when `!isPausable`;
- Inline on a paused row: resume / cancel-permanently buttons;
- Info attached to a paused row: "paused at step N …";
- Pending rows show a "waiting for the queue to resume" note when `held=true`.

#### New components

- `PauseProgressModal.tsx`: the pause-progress modal (saving / timeout /
  success / failure states), subscribing to the task's SSE event stream.
- `HoldQueueModal.tsx`: the hold confirmation modal (case A: no running task
  + case B: has a running task, with a linked radio button).

#### i18n

New keys (English terms use hold / release):

- `queue.pause` / `queue.resume` / `queue.holdQueue` / `queue.releaseQueue`
- `queue.pauseProgress.*` (modal state copy)
- `queue.holdModal.*` (case A / B copy)
- `status.paused`
- etc.

### Required spike (the first thing after merging this ADR)

Windows-side validation:

1. Whether the supervisor's `proc.send_signal(signal.CTRL_BREAK_EVENT)`
   reaches a `CREATE_NEW_PROCESS_GROUP` child process group;
2. Whether the child process's Python
   `signal.signal(signal.SIGBREAK, handler)` catches it;
3. Whether the handler can fully complete save_training_state + writing the
   snapshot before `sys.exit(0)`;
4. Whether the supervisor correctly reads the `__EVENT__:pause_state` line
   from the child's stdout before marking it paused via `_finish_slot`.

If the spike fails → fall back to candidate B (sentinel-file IPC); this ADR
is revised in a second decision phase.

### Suggested PR split

1. **PR-0 spike**: spike script + report only, doesn't touch the main line.
2. **PR-1 latent-bug prefix fix**: `runtime/training/loop.py`'s periodic save
   path gets a per-task subdirectory. Shipped independently, a clean base.
3. **PR-2 backend skeleton**: context / supervisor / db migration, new API
   endpoints, all with unit tests. Feature flag `enable_pause_resume`
   defaults off.
4. **PR-3 resume path + cmd_builder**: end-to-end integration test (pause at
   step N → resume → verify global_step picks up from N+1 + loss is
   continuous).
5. **PR-4 frontend UI**: banner / buttons / two modals / i18n.
6. **PR-5 docs + changelog + gradual rollout of the feature flag**.

Each PR is independently revertible, for fine-grained rollback.

## Rationale

**Why candidate B was rejected**: the signal mechanism is more standard, and
once the spike works there's no need to add an IPC channel. B is kept as a
fallback in case the spike fails.

**Why candidate D was rejected**: it strongly depends on `save_state_every`
(default 0), which most users don't enable; the resume point is imprecise;
and the UI would falsely report success — accruing debt.

**Why the state file path needs a per-task subdirectory**: today,
`save_state_every` already overwrites across multiple tasks under the same
version. This isn't a problem introduced by pause/resume — it's a latent
bug that this feature fixes in passing. Splitting it out as an independent
PR-1 isolates the regression risk.

**Why a config snapshot instead of reusing the task config fields**: task
config fields are frozen at creation time, but some parameters come from a
version / preset / an external yaml, and what's stored is a reference path,
not an inline value. If the user edits the version config afterward,
resolving by path again would fetch the new value. Writing a snapshot to
disk means expanding all references into concrete inline values, decoupling
it at the root from "the user's current config."

**Why the hold status isn't part of the task state machine**: queue hold is
a dispatcher-level property, unrelated to any individual task's status. A
task's state doesn't change across a hold boundary (a running task keeps
running, a paused task stays paused). Making it part of the state machine
would mean turning 5 states into 6, plus a full set of transition rules —
unnecessary.

**Why the pause-in-progress uses a full-screen modal rather than a button +
toast**: during a pause, the user has no opportunity to "change their mind
about pausing" (the signal has already been sent), so a forced screen lock
prevents a stray click from losing progress. The 30s timeout gives the user
a choice of [wait 30 more seconds] / [force-cancel and keep progress
saved-so-far] / [terminate the task], rather than silently downgrading to
cancel — the intent behind clicking pause is to preserve progress, so a
silent cancel would startle the user.

**Why cancel on Windows switches directly to taskkill /T /F**: cancel's
semantics are inherently a hard interrupt, and the 30s grace period almost
always ends in a force-kill anyway. `CTRL_BREAK_EVENT` is reserved for pause
to keep signal intent unambiguous. POSIX doesn't have this problem (SIGINT
vs. SIGTERM naturally separate the two).

## Consequences

### Positive

- Existing cancel semantics are unchanged, so users' existing habits are
  unaffected.
- Users can resume interrupted progress — "cancel" no longer means "all
  progress lost."
- Paused tasks automatically survive a server restart, enabling a
  "shut down, then start back up" workflow.
- Fixes the `save_state_every` per-task-subdirectory latent bug in passing.
- The config-snapshot design fully decouples a paused task from any config
  edits the user makes afterward.
- The independent queue-hold switch enables workflows like "don't run
  overnight" or "maintenance window."

### Negative / to be evaluated

- Whether the Windows signal chain works depends on the spike; candidate B
  is a fallback but requires another design round.
- The pause-file pair (.pt + .config.json) adds some disk usage, but it's
  cleaned up automatically along with the task's lifecycle.
- Supervisor's `_finish_slot` branch goes from binary to three-way; the
  regression risk relies on unit-test coverage.
- Cancel on Windows switching to `taskkill /T /F` skips the soft-signal grace
  phase; existing cancel behavior is essentially unchanged for the user, but
  logging / telemetry that depends on the grace phase would need migrating.
- The completeness of the snapshot's serialization list needs coverage in
  the PR-3 integration tests — missing even one field could cause resume
  behavior to drift.

### Future debt (explicitly out of scope for this ADR)

- Continuing the same wandb run id on resume (resume currently starts a new
  run, not reusing run_id)
- Bulk pause / resume
- Scheduled automatic resume of a held queue
- Reminders for a task paused more than X days
- Recommending `save_state_every` be enabled on the first training run in the UI
- Success metrics / gradual-rollout telemetry
- Editing config after pausing, then resuming (never supported — a fork is
  required instead)

## Out of scope

- Automatically saving state when the server actively stops / crashes /
  loses power (the coverage is uncontrollable; instead guide users to enable
  `save_state_every` periodic checkpoints)
- A combined "pause all" button (covered already by the extra question in
  the hold modal)
- Keeping the paused status after a hard kill (state isn't trustworthy after
  a hard kill, so it must be marked canceled)
- Automatically cleaning up periodic save files (these are disaster-recovery
  points the user opted into; the user manages them)
- Pausing generate / download / tag tasks (they run fast enough that this
  would be pointless)

## References

- Design document (three review rounds): `docs/design/queue-pause-resume-design.md`
- Existing code touch points:
  - `runtime/training/context.py:109` `handle_interrupt`
  - `runtime/training/phases/resume.py:82` SIGINT registration
  - `runtime/training/loop.py:271` `save_state_every` writes to disk (where the latent bug lives)
  - `studio/supervisor.py:952` `_finish_slot` status branching
  - `studio/supervisor.py:1081` `_send_terminate_signal`
  - `studio/db.py:36` `VALID_STATUSES`
  - `studio/server.py:2893` the existing `cancel_task` endpoint
  - `studio/web/src/pages/Queue.tsx:159` the existing cancel-only comment
- memory: `memory/queue_pause_resume_via_sigint.md` (a record of earlier decisions)

---

## Incremental updates

### 2026-05-19 — Addendum 1: deep audit of dev's training stack + pause-semantics reversal (Pause-as-Cancel + Epoch Auto-Backup)

**Scope of impact**: affects the initial ADR's "decision items 4/5/6",
"backend code direction / `handle_interrupt` pseudocode", and "out of scope
/ automatically saving state on server-initiated stop" sections. Once this
Addendum lands, the initial ADR's **mid-epoch save path is entirely
retired**, and the pause signal no longer triggers any save to disk.

**Trigger**: PR-1 through PR-5 had already merged into dev, and just before
the user was about to start using it, they raised a question — "with all
these special parameters added (InfoNoise / Prodigy / PPSF / Cosine LR /
loss_weighting / pyramid noise / ...), can the state saved by clicking pause
actually resume perfectly?" A deep audit of the dev training stack, run in
parallel with 3 algorithm-expert sub-agents doing adversarial review, found
that the initial ADR's implicit assumption — "the CLI side already has a
complete save/resume chain, so a supervisor signal trigger is fine" — has
**7 concrete risks** against dev's current code, 2 of which are **dev-only
discoveries** (not covered by the local draft in the worktree).

#### Round 1: inventory of dev's training-stack components

Every currently active component under `runtime/training/` was reviewed,
tagging whether each one carries internal state:

| Component | File | Internal state |
|---|---|---|
| baseline timestep sampler | `timestep_samplers/baseline.py` | none (pure function wrapper) |
| **InfoNoise** | `timestep_samplers/infonoise.py:29-80` | **9 fields**: `_fifo` (K×B floats) / `_mse_ema` / `_n_count` / `_cdf_values` / `_internal_step` + 4 metadata counters |
| make_noise / pyramid / noise_offset | `noise.py` | none (resampled every step, torch RNG) |
| timestep_sampling (6 modes + Möbius shift) | `timestep_sampling.py` | none (pure functions) |
| loss_weighting (min_snr / detail_inv_t / cosmap) | `loss_weighting.py` | none (pure functions) |
| ER-SDE-3 inference sampler | `inference_samplers/er_sde.py` | inference only, unrelated to resume |
| AdamW | `optimizers/adamw.py` | torch builtin, state_dict complete |
| Prodigy | `optimizers/prodigy.py` | `d / d_max / d_numerator / s` are in state_dict |
| **PPSF (ProdigyPlusScheduleFree)** | `optimizers/prodigy_plus_schedulefree.py` | three sets of weights x/y/z; **switching train/eval does an in-place lerp on p.data** (see `utils/optimizer_utils.py:466-491`) |
| CosineAnnealingLR | `schedulers/cosine.py` | `last_epoch / _last_lr` are in state_dict |
| CosineAnnealingWarmRestarts | `schedulers/cosine_with_restart.py` | `T_cur / T_i` are in state_dict |
| lycoris injector | `adapters/lycoris.py` | LoRA weights are in `injector.state_dict()` |

**dev's current `save_training_state`** (`runtime/training/state.py:22-39`)
serializes: the LoRA injector + optimizer.state_dict + epoch + global_step +
loss_history + rng_state (torch + cuda + random) + monitor_state +
scheduler.state_dict (if present). **It serializes neither the numpy rng nor
any timestep_sampler internal state at all.**

#### Round 2: three-way adversarial algorithm review

Three sub-agents were run in parallel, each auditing dev's current
pause/resume path from a different angle:

| Angle | Key finding | Recommended boundary |
|---|---|---|
| timestep / noise | None of InfoNoise's 9 state fields are serialized (dev's current state) → after resume, `_internal_step=0` re-runs the entire N_warm (default 5000 steps) | step boundary (assuming InfoNoise adds a state_dict) |
| optimizer | PPSF resume is missing `.train()` + the grad_accum boundary isn't guarded + Prodigy's d drifts mid-epoch | accum boundary (defer the handler to the next global_step) |
| scheduler / loss / data | BucketBatchSampler causes 5% double-training + cosine restart's T_cur drifts + `current_epoch` is ambiguous | **epoch boundary** (other approaches would need ≥8 more fields to get right) |

#### Round 3: the seven-item bug list (including 2 dev-exclusive findings)

| # | Bug | Severity | dev-exclusive? |
|---|---|---|---|
| 1 | InfoNoise's 9 state fields aren't serialized | 🔴 | No (already identified in the worktree draft) |
| 2 | **PPSF doesn't explicitly call `.train()` after resume**, so `p.data` stays at averaged x but PPSF internally thinks it's = y → the first step's gradient direction is off | 🔴 | **Yes** |
| 3 | `handle_interrupt` fires mid-way through a grad_accum cycle, leaving partial backward gradients on `p.grad` that never enter the optimizer state | 🔴 | No |
| 4 | BucketBatchSampler's progress isn't saved; resume restarts from the beginning of the epoch, retraining the first half of the epoch — for short LoRA runs, this 5% double-training skews Prodigy's d estimate | 🔴 | No |
| 5 | **`current_epoch` semantics are ambiguous**: the mid-epoch path saves `epoch` (`context.py:175`), while the epoch-end path saves `epoch+1` (`loop.py:297,342`); `loop.py:341-344`'s periodic epoch save has the same off-by-one | 🟡 | **Yes** |
| 6 | `CosineAnnealingWarmRestarts`'s `T_cur` drifts by 5% on every pause/resume | 🟡 | No |
| 7 | `epoch_loss_sum` / wandb `train/loss_epoch` accumulation gets confused (after a mid-epoch resume) | 🟡 | No |

#### Round 4: the user proposes a new design — "pause = cancel + automatic epoch backup"

> "Change the save logic — instead of clicking pause saving the current step's
> state, automatically back up once per epoch, with each new epoch
> overwriting the old one. Pausing directly pauses the task without producing
> new state; resuming starts from the previously saved epoch state,
> discarding the current epoch's progress."

This design **fully decouples** the pause path from the save path. Pause
degrades to "a cancel with a wandb-finish tidy-up"; save responsibility
falls entirely to a periodic backup at the training loop's own epoch
boundary. Bugs #3 / #4 / #6 / #7 **disappear naturally**; #1 / #2 / #5 still
need explicit fixes.

This new design also matches the product semantics of "pause = release the
GPU immediately" — the whole point of clicking pause is to free up the GPU
for something else, not to wait for the current step to finish saving.

#### Final decision (approach Δ)

**Pause-as-Cancel + Epoch Auto-Backup is adopted**. Specific decisions:

1. **Add a new automatic backup, `auto_epoch_state.pt`**: at the end of
   every epoch, overwrite-write `<state_dir>/auto_epoch_state.pt` plus a
   matching `auto_epoch_state.config.json`. **No args gate** (this is a
   system-level guarantee, not a user toggle). **Does not also save the
   LoRA `.safetensors`** — it's purely for resume, unrelated to pause; a
   user who wants a LoRA every epoch should turn on `save_every=1`
   themselves.
2. **`handle_interrupt` is drastically simplified**: the calls to
   `save_training_state` / `injector.save(interrupted_*)` /
   `write_config_snapshot` are removed; it keeps
   `wandb_monitor.finish()` + `emit pause_state(state_path=<the most recent auto_epoch_state.pt>)` +
   `sys.exit(0)`. A repeated SIGINT still forces exit via `sys.exit(1)`.
3. **Pausing within the first epoch → cancel**: `handle_interrupt` checks
   ctx's `last_auto_epoch_state_path` field; if it's None (the first epoch
   hasn't finished), it emits `pause_state(state_path=None)`, and the
   supervisor routes this to the cancel branch. On the UI side,
   `is_pausable=false` hides the button entirely (keeping the initial ADR's
   §8.1 rule in effect; if the user can't see the button, they can't click
   it, and no explanatory copy is needed).
4. **`save_state_every` (step) and `save_state_every_epochs` (user-initiated
   epoch) behavior is completely unchanged**. They remain user-opted-in
   disaster-recovery points, coexisting alongside the auto backup — three
   files coexist under `<state_dir>/task_<TID>/`:
   - `auto_epoch_state.pt` (system-enforced, single overwritten file)
   - `training_state_epoch{N}.pt` (the user's per-epoch backups, multiple archived copies)
   - `training_state_step{N}.pt` (the user's per-step backups, multiple archived copies)
5. **PauseProgressModal gains a confirm sub-modal**: clicking the pause
   button first pops a confirm dialog, with **uniform copy containing no
   dynamic fields**:
   ```
   Title: Pause training?

   Some experimental parameters (e.g. the InfoNoise adaptive
   sampler, Prodigy-family adaptive optimizers, cosine LR
   scheduling) may cause quality fluctuations after a
   pause/resume cycle.

   Resuming will continue from the end of the previous
   epoch; progress in the current epoch will be discarded.

   [Cancel]  [Confirm pause]
   ```
   After confirming, it proceeds into the original PauseProgressModal's
   saving / success / failure state machine (as designed in the initial ADR
   §4.3).
6. **InfoNoise serialization is mandatory (PR-A)**: add optional
   `state_dict()` / `load_state_dict()` hooks to
   `runtime/training/timestep_samplers/{protocol,baseline,infonoise}.py`;
   add a `timestep_sampler=` keyword to `state.py`'s `save_training_state` /
   `load_training_state`; an empty dict writes no key; a K/B mismatch only
   warns, never blocks.
7. **PPSF resume `.train()` fix (PR-C2)**: needs a spike first (1-2 hours) to
   decide between approaches:
   - **Approach X**: split the save protocol — the `lora_state_dict` inside
     `state.pt` stores y (not saved within `optimizer_eval_mode`); the
     `.safetensors` file continues to store the averaged x (for the user to
     download and use). The current "same `with optimizer_eval_mode` block
     wrapping two saves" in loop.py / context.py needs to be split apart.
   - **Approach Y**: after loading, use the Schedule-Free lerp formula
     `y = (x - β·z) / (1-β)` to back out y and rewrite `p.data`, reading β
     from the PPSF source.
   - Decide between X and Y once the spike is done, then ship.
8. **`loop.py:341-344` off-by-one fixed in passing (PR-C1)**: change
   `save_training_state(..., epoch, ...)` to `... ctx.current_epoch ...`
   (which is already `epoch+1`). This is a latent bug independently
   identified in the worktree draft, fixed together with this Addendum.

#### Disposition matrix for the three-way audit's seven bugs

| # | Disposition |
|---|---|
| 1 InfoNoise 9 state fields | 🔴 explicitly fixed in PR-A |
| 2 PPSF .train() | 🔴 fixed in PR-C2, pending the Spike-PPSF conclusion |
| 3 grad_accum boundary | ✅ disappears naturally (end-of-epoch always coincides with a completed accum cycle + a step boundary) |
| 4 dataloader double-training | ✅ disappears naturally (re-shuffling via set_epoch at an epoch boundary is expected behavior) |
| 5 current_epoch ambiguity + off-by-one | 🟡 fixed in PR-C1 (a one-line keyword change) |
| 6 CosineWarmRestarts T_cur | ✅ disappears naturally |
| 7 epoch_loss_sum confusion | ✅ disappears naturally |

#### Code-level direction

**Loop** (`runtime/training/loop.py`):
- Insert a "forced auto epoch backup" block at the end of the epoch
  (L296-348, no args gate) + set ctx's `last_auto_epoch_state_path` /
  `last_auto_epoch_config_path` + emit an `auto_epoch_backup_written` event
  (so the supervisor can upgrade `is_pausable`)
- Wrap it in `with optimizer_eval_mode(...):` (for PPSF's averaged x); if
  Spike-PPSF selects approach X, this needs adjusting
- Fix the off-by-one in the existing `save_state_every_epochs` call at
  L341-344 in passing (`epoch` → `ctx.current_epoch`)

**Context** (`runtime/training/context.py:126-188`):
```python
def handle_interrupt(self, sig, frame) -> None:
    if self.interrupted:
        sys.exit(1)
    self.interrupted = True
    self.wandb_monitor.finish()
    emit_event("pause_state", {
        "state_path": str(self.last_auto_epoch_state_path)
                      if self.last_auto_epoch_state_path else None,
        "config_path": str(self.last_auto_epoch_config_path)
                       if self.last_auto_epoch_config_path else None,
        "step": self.global_step,
    })
    sys.exit(0)
```
New fields: `last_auto_epoch_state_path: Optional[Path] = None` /
`last_auto_epoch_config_path: Optional[Path] = None`.

**State** (`runtime/training/state.py`):
- `save_training_state` gains a `timestep_sampler=` keyword, calling
  `state_dict()` to serialize it; an empty dict writes no key
- `load_training_state` calls `load_state_dict()`; a K/B mismatch only warns,
  never raises
- The end of `load_training_state` adds PPSF handling per the Spike-PPSF
  conclusion

**Supervisor** (`studio/supervisor.py`):
- `_on_line` recognizes the new event `auto_epoch_backup_written` → sets
  `slot.last_auto_epoch_state_path`
- `_on_line` receiving `pause_state(state_path=None)` → routes to the cancel
  branch
- `is_pausable` SSE field upgraded to:
  `train_loop_started AND last_auto_epoch_state_path is not None`

**UI** (`studio/web/src/pages/Queue.tsx` / new component `PauseConfirmModal.tsx`):
- The pause button stays fully hidden when `is_pausable=false` (unchanged)
- Clicking pause → PauseConfirmModal (uniform copy, no dynamic fields like
  epoch N) → confirm → call the pause API → the original PauseProgressModal

#### PR split

| PR | Contents | Dependency |
|---|---|---|
| PR-A | InfoNoise + the sampler protocol's `state_dict` / `load_state_dict` + `state.py` gaining the `timestep_sampler=` argument + 14 unit tests | none |
| PR-B | `loop.py` gains auto_epoch_backup + the off-by-one fix + ctx fields + event emission | none |
| PR-C1 | Unify `current_epoch` semantics + `loop.py:341-344` off-by-one fix (if not already bundled into PR-B) | none |
| **Spike-PPSF** | 1-2 hours running a short training + resume + comparing loss curves, to confirm approach X vs. Y | doesn't block the main line |
| PR-C2 | PPSF resume fix | Spike-PPSF |
| PR-D | supervisor's new `_on_line` events + `is_pausable` upgrade + cancel routing | PR-B |
| PR-E | UI: PauseConfirmModal + copy + i18n | PR-D |

PR-A / PR-B have no hard dependency on each other and can run in parallel;
PR-C1 can be merged into PR-B; PR-C2 ships separately; PR-D comes after PR-B;
PR-E comes after PR-D.

#### Test plan for the rollout

- `test_auto_epoch_backup_overwrites_in_place` — only one
  auto_epoch_state.pt remains after 3 epochs
- `test_handle_interrupt_no_save_only_emit` — save_training_state isn't
  called after SIGINT fires
- `test_pause_before_first_epoch_marks_canceled` — SIGINT mid-way through the
  first epoch, supervisor marks it canceled
- `test_resume_from_auto_epoch_no_double_training` — 3 epochs + half an
  epoch + SIGINT + resume → the optimizer step count ==
  3 × steps_per_epoch, and epoch 4 is redone
- `test_ppsf_train_mode_after_load` — PPSF's state matches after loading
  (the exact assertion depends on the spike's conclusion)
- `test_current_epoch_off_by_one_fixed` — the `epoch` field inside
  auto_epoch_state.pt == `ctx.current_epoch`
- 14 unit tests cherry-picked from the worktree draft: timestep sampler
  resume bit-exactness / RNG / K/B mismatch warning / deque maxlen /
  corrupted sampler_state warning doesn't block / a roundtrip integration
  test

#### Reasons for rejecting other approaches

- **Approach A (mid-epoch dataloader skip)**: already audited by the
  three-way adversarial review in the worktree draft; Prodigy's `d` is
  monotonically non-decreasing + `safeguard_warmup`'s reverse constraint +
  cosine LR not being kept in sync — three biases stacking in the same
  direction; 5% double-training is 5x beyond academia's usual safe zone
  (N/S < 1%).
- **Approach B (deferred interrupt, waiting until the current epoch ends
  before exiting)**: violates the product semantics of "pause = release the
  GPU immediately" — a user freeing up the GPU for something else can't wait
  several minutes to tens of minutes.
- **Approach C (the worktree draft's original approach: epoch boundary, but
  save still triggered by SIGINT)**: dev would still need to fix the 2
  dev-exclusive bugs #2 and #5; compared to approach Δ, approach C keeps the
  SIGINT handler holding save responsibility, whereas Δ reduces SIGINT to a
  pure signal notification, making the handler minimal and the path far
  easier to verify.

#### Remaining follow-ups (explicitly out of scope for this Addendum)

- The leftover micro-batch at the end when `ep_size % grad_accum != 0`
  (an independent latent bug, unrelated to pause/resume)
- DataLoader skip-K mid-epoch resume (never doing this — approach A was
  already rejected)
- The numpy RNG slot (dev's training path makes no numpy.random RNG calls;
  kept as a future defensive measure)
- Serializing `speed_ema` / `sample_prompt_idx` (monitoring metrics / sample
  rotation; losing them has no algorithmic impact)
- Continuing the same wandb run id (never doing this — resume always starts
  a new run)

#### New references

- The three-way adversarial sub-agent reports (a one-time conversation
  artifact, not archived)
- `utils/optimizer_utils.py:466-491` — the actual semantics of the
  `optimizer_eval_mode` context manager
- `runtime/training/timestep_samplers/infonoise.py:29-80` — InfoNoiseScheduler's
  9 internal state fields
- `runtime/training/optimizers/prodigy_plus_schedulefree.py` — the PPSF factory
- The local draft in worktree-090validation (partially stale content; per
  the user's decision, it is not being merged)
