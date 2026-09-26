# Queue pause / resume training — logic design draft

> A working design document laying out only the **logic model**: the state
> machine, button semantics, file storage, and user scenarios. It doesn't
> cover code implementation details. See the follow-up ADR / PR descriptions
> for the implementation workflow.

## 0. Purpose and non-goals

**Purpose**: let the user, from the UI, "pause → release the GPU → later
continue from the same progress" for a task that is **currently training**,
instead of losing all progress the way today's "cancel" does.

**Non-goals**:
- Not replacing today's "cancel" — cancel remains terminal, releases the GPU
  immediately, and cannot be resumed.
- Not changing the existing `--resume-state` / `ResumeFieldPicker` manual
  resume-from-.pt path — that path continues to exist as a separate entry
  point from the new "pause / resume" feature (see §6).
- Doesn't cover non-training tasks like generate / download / tag. These
  tasks run quickly, so pausing them wouldn't be meaningful.

---

## 1. Task status (including the new paused status)

### Current status set

| Status | Category | Description |
|------|------|------|
| `pending` | live | waiting in the queue to be scheduled |
| `running` | live | currently executing |
| `done` | terminal | completed normally |
| `failed` | terminal | exited abnormally |
| `canceled` | terminal | user canceled it (hard interrupt) |

### New status

| Status | Category | Description |
|------|------|------|
| `paused` | **non-terminal, non-live** | the user deliberately paused it; state has been saved and it can be resumed |

**Key properties**:
- `paused` is **not** in `TERMINAL_STATUSES` — it's not a terminal state and
  can be resumed.
- `paused` **doesn't** occupy a scheduling slot — when the dispatcher sees a
  queue with only paused tasks and no pending ones, it won't automatically
  pick up a paused task; the user must explicitly resume it.
- `paused` **doesn't** hold the GPU — the child process has already exited,
  resources are fully released, and other pending tasks can be scheduled
  normally.

### Intermediate states not introduced

- **No** `pausing` ("in the process of pausing") state is introduced: there's
  a window of a few seconds between the user clicking pause and the child
  process exiting, during which the task is still `running`; the UI shows
  this via an "action in progress" flag like `pause_pending`, which never
  enters the db state.
- **No** `resuming` state is introduced: after the user clicks resume, the
  task's status goes straight back to `pending`, and the dispatcher naturally
  picks it up. The UI distinguishes "native pending" from "pending with a
  resume attached" using the `resume_state` field, with no extra status
  needed.

---

## 2. State transition diagram

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
   new ─► pending ──┴─► running ──┬─► done                     │
              │          │  ▲     │                            │
              │          │  │     ├─► failed                   │
              │          │  │     │                            │
              │          │  │     ├─► canceled                 │
              │          │  │     │                            │
              │          │  │     └─► paused ─► (resume) ──────┘
              │          │  │           │
              │          │  │           ├─► (cancel) ─► canceled
              │          │  │           │
              │          ▼  │           ▼
              └──► canceled │       canceled
                   (cancel  │       (server restart → stays paused)
                    pending)│
                            │
                  (cancel running)
```

Transition notes:
- `running → paused`: the user clicks pause → a pause signal is sent → the
  child process triggers `handle_interrupt` → saves state → the process
  exits → the supervisor writes status=paused.
- `paused → pending` (carrying resume_state): the user clicks resume → the
  original task config is reused, with `resume_state` injected pointing at
  the saved .pt → the status is written back to pending → the dispatcher
  reschedules it.
- `paused → canceled`: the user, while paused, decides to give up entirely →
  the db status is changed directly; the process already exited long ago, so
  no signal is needed.
- `paused → paused` (across a server restart): the process has already
  exited, and the status lives only in the db, so restarting the server
  doesn't affect it.

---

## 3. Two kinds of pause semantics (terminology: pause vs. hold)

The user mentioned both "global pause" and "task-level pause" — these are
**two independent features** with different targets, lifecycles, and state
storage. To avoid ambiguity, **deliberately different verbs are used**:

| Term | Target | Verb | Opposite verb |
|------|---------|------|---------|
| **Task pause** | a single running task | pause | resume |
| **Queue hold** | the whole dispatcher (a switch for "should new pending tasks be picked up") | hold | release |

English equivalents (in case i18n is done later): task **pause** / **resume**
vs. queue **hold** / **release** (or suspend / resume scheduling, a common
term in the HPC world). The localized terms settled on the standard computing
term for "suspend" for hold, and a "resume scheduling" phrasing for release —
adding the word "scheduling" specifically to disambiguate it from the
task-level "resume."

### 3.1 Task pause

**Target**: a training task that is currently running.

**Effect**:
- A pause signal is sent to that task's child process → it goes through
  `handle_interrupt` → saves state → exits.
- The task's status changes from `running` → `paused`.
- The GPU is released. **If the queue isn't held**, the supervisor's main
  loop sees the TRAIN slot empty and automatically schedules the next
  pending task.
- The paused task only gets picked up again once the user clicks "resume"
  later.

### 3.2 Queue hold

**Target**: the supervisor's dispatcher, a single bool switch.

**State storage**: persisted in the db (suggested to live in `app_settings`
or a kv table, a single record), **preserved across server restarts**.
Reasoning: holding is an explicit user decision, and restarting the server
shouldn't automatically release the hold — otherwise the queue would
suddenly start running by itself after a maintenance restart, which the user
wouldn't expect.

**Effect**:
- The dispatcher stops pulling new tasks from the `pending` queue.
- **Doesn't affect** the task currently running — it keeps running to
  completion naturally, ending as done / failed.
- **Doesn't affect** the task-level pause / resume API — the two work
  independently (see §3.3 for details).
- The UI always shows the current hold status (a top banner / a toggle
  button's status color).
- After the user releases the hold, the dispatcher resumes normal
  scheduling, pulling tasks per the existing `next_pending` priority +
  creation-time ordering.

**Pending tasks while held**: nothing happens — this counts as neither an
error nor an anomaly; the UI shows a "waiting for the queue to resume" note.

### 3.3 How the two interact

**Orthogonal relationship**: the two switches can be combined freely, and
all 4 quadrants are valid:

| Queue state | running task | behavior |
|---------|------------|------|
| not held + has a running task | when it finishes → automatically pulls the next pending task |
| not held + no running task | the dispatcher immediately pulls the next pending task (if any) |
| held + has a running task | when it finishes → does **not** pull the next pending task |
| held + no running task | the dispatcher idles, waiting to be released |

**Task operations while held**:
- Task pause: allowed. Pausing a running task while held → the task enters
  paused, and the TRAIN slot sits empty with no dispatch.
- Task resume: allowed. The task goes from paused back to pending, but
  **won't actually be picked up** — it's just enqueued into pending, waiting
  for the hold to be released. In the UI, the task shows as pending, with a
  small "waiting for the queue to resume" note next to it.
- Task cancel / delete: allowed, unrelated to the hold status.

**Why "hold + resume" is allowed to have "deferred start" semantics**:
- Simplifies the model: resume just means "mark this task as schedulable
  again"; scheduling itself is controlled by the hold status.
- Counter-argument: if the resume button were disabled while held, the user
  would have to "release the hold first → resume → hold again" just to queue
  something up — a clunky workflow.
- But the UI must clearly show "this task is queued, waiting for the queue
  to resume," or the user would be confused about "why didn't it start."

### 3.4 User cases for when to use hold

See §7 Case F / G for details. In short: scenarios where the user wants "the
queue to slow itself down" — e.g. temporary observation, not running
overnight, a maintenance window.

---

## 4. Button / entry-point placement

### 4.1 The Queue list page

**Top banner** (visualizing the hold status, **banner only, no task chip**):
- Queue not held: **no banner shown at all** — avoiding visual noise.
- Queue held: a sticky top banner, "Queue is held; pending tasks won't start
  automatically [Release]."
- Note: the banner is a UI element, **not** part of the task state machine.
  The task status badge only reflects the task's own status (pending /
  running / paused / done / failed / canceled) and never represents the
  queue-hold state.

**Top actions**:

| Button | Appears when | Behavior |
|------|---------|------|
| Pause current | there's a running task **and it has entered train_loop** (see §8.1) | task-level pause |
| Cancel current | there's a running task | (existing) task-level cancel, hard interrupt |
| Hold queue | shown when the queue isn't held | opens a confirmation modal (see §4.3) |
| Release | shown when the queue is held | releases the hold directly, no confirmation needed |

**Not doing**: there's no combined "pause everything" button. Hold + pausing
the running task is accomplished in one go via the §4.3 modal.

**Per row**:

| Inline button | Appears when | Behavior |
|---------|---------|------|
| Resume | task.status = paused | status goes back to pending, injecting resume_state |
| Cancel permanently | task.status = paused | paused → canceled, deletes the pause files |

Copy for a paused task: shown inline as "paused at step N on YYYY-MM-DD
HH:MM."
While held, every pending row (including those that just came from a
resume) shows a small "waiting for the queue to resume" note.

### 4.2 The QueueDetail page

Matches the Queue page, with buttons placed on the detail card:
- running + has entered train_loop: pause / cancel
- running + still starting up: only cancel is shown (the pause button is hidden)
- paused: resume / cancel permanently + shows "paused at step N …"
- other statuses: unchanged from today

### 4.3 The pause-task progress modal (covers the whole flow)

Clicking the "Pause current" button **immediately pops a modal** covering the
whole pause flow: sending the signal → saving → complete / timeout /
failure. Other actions are **locked** while the modal is open, to prevent a
stray click (e.g. clicking cancel and losing all progress).

**State machine** (self-managed within the modal):

```
[Pause clicked]
   ↓
State 1: saving
  Shows: spinner + "Saving training state… (Ns elapsed)"
  Buttons: none (or a disabled "waiting" one), blocking any other action
   ↓ child process emits __EVENT__:pause_state (success)
   ↓ OR 30s timeout
   ↓ OR child process exits abnormally
   ├─→ State 2A: saved successfully
   │     Shows: ✓ "Paused at step N — can be resumed anytime"
   │     Button: [OK] → closes the modal + a toast
   │
   ├─→ State 2B: timed out (still hasn't received the event after >30s)
   │     Shows: ⚠️ "Saving is taking longer than expected (Ns elapsed)"
   │     Buttons: [wait 30 more seconds] [force-cancel (keep the pause file)] [terminate the task (discard progress)]
   │     - Wait: continues waiting for another 30s round
   │     - Force-cancel: sends a hard terminate signal; if the pause file
   │       has already been written to disk, it's still marked paused,
   │       otherwise it falls back to canceled
   │     - Terminate the task: sends a hard terminate + marks it canceled
   │
   └─→ State 2C: child process exited abnormally (rc != 0)
         Shows: ✗ "An error occurred while saving (exit code N)"
         Buttons: [view logs] [terminate the task]
         The task is marked failed
```

**Key design points**:
1. **The modal locks the screen the moment it opens** — during a pause,
   there's no opportunity for the user to "change their mind about
   pausing," since the signal has already been sent, so changing their mind
   would be meaningless anyway.
2. **The 30s threshold lets the user decide what happens next**, rather than
   "silently downgrading to cancel." The three options let the user judge
   based on disk / IO conditions themselves.
3. **The modal shows elapsed seconds** — transparent feedback, so the user
   doesn't think the program has hung.
4. **The child process's emitted `__EVENT__:pause_state` is the only signal
   that triggers "success"** — rc=0 alone isn't enough (rc can be unreliable
   due to Windows wrapper rewriting).

### 4.4 The queue-hold confirmation modal

Clicking the "Hold queue" button shows a different modal depending on the
current running state:

**Case A: no running task**
```
Hold the queue?
Once held, new pending tasks won't start automatically.
[Confirm hold] [Cancel]
```

**Case B: has a running task**
```
Hold the queue?
Once held, new pending tasks won't start automatically.

Currently running: task #42 "my_anime_v1".
What should happen to task #42?

  ○ Let task #42 finish running (default)
  ○ Also pause task #42 (saves progress; can be resumed either after the hold is released, or individually)

[Confirm hold, let task #42 finish]  [Cancel]
```

The primary button's label **updates in sync** with the radio selection
("Let task #42 finish" / "Also pause task #42"), so the user isn't left
unsure what they just confirmed. The specific task name / id **appears
directly** in the options, rather than using "it."

**Case B's 3 outcomes**:
- Cancel → nothing changes.
- Confirm + "Let task #42 finish" → only writes queue_held=true.
- Confirm + "Also pause task #42" → writes queue_held=true + calls the
  task-level pause API (triggering the §4.3 pause-progress modal).

**Release button**: takes effect immediately, no confirmation needed
(lifting a restriction is a low-risk operation).

### 4.5 Unchanged entry points

- The `ResumeFieldPicker` on the new-task page (starting a new task from any
  .pt) is **unchanged**. See §6 path B.

---

## 5. State file storage and naming

### 5.1 File location and naming (per-task subdirectory + pause prefix + config snapshot)

**Write paths**:

| Source | Path |
|------|------|
| `handle_interrupt` (pause-triggered, state) | `<output_dir>/state/task_<TID>/pause_step_<N>.pt` |
| `handle_interrupt` (pause-triggered, config snapshot) | `<output_dir>/state/task_<TID>/pause_step_<N>.config.json` |
| `save_state_every` / `save_state_every_epochs` (periodic) | `<output_dir>/state/task_<TID>/step_<N>.pt` |

Three changes:
1. **Adding a per-task subdirectory**: the reasoning is in §5.3 / §5.4 — the
   same version might run multiple tasks over time, and state files must be
   isolated by task_id, otherwise they'll overwrite each other (today's
   latent bug).
2. **Pause files get a `pause_` prefix**: to distinguish "a pause anchor"
   from "a periodic checkpoint." The two have different lifecycles (see §5.5
   / §5.6), so the naming must be able to tell them apart.
3. **A config snapshot is written to disk at the same time as pause**:
   freezes all training parameters currently in use into a JSON file, with
   the same name as the .pt (with a `.config.json` suffix). Resume **runs
   from the snapshot**, never reading the task table / version config /
   external preset. See §5.7 for details.

LoRA output (`.safetensors`) and the samples directory are **unchanged** —
only state-related files are moved.

### 5.2 Association with the task

A new column is added to the db's task table: `paused_state_path` (an
absolute path), recording the .pt from the most recent pause.

- Every pause overwrites this field (it always points to the "latest
  resumable point").
- It is **not** cleared on paused → pending (resume).
- It's kept as a historical record once the task reaches a terminal state
  (done / failed / canceled).

### 5.3 Isolating multiple tasks under the same version (key scenario 1)

**Scenario**: the user runs task #42 under version V (which gets paused),
then starts task #43 under the same V.

| Task | State path | Config snapshot path |
|------|-----------|---------------------|
| #42 | `<output_dir>/state/task_42/pause_step_1000.pt` | `…/state/task_42/pause_step_1000.config.json` |
| #43 | `<output_dir>/state/task_43/pause_step_500.pt`  | `…/state/task_43/pause_step_500.config.json` |

The two tasks are **completely isolated**, not interfering with each other;
either paused task can be resumed independently, and neither task's config
snapshot overwrites the other's.

**Today's implementation has a bug here**: with no task_id subdirectory,
both task #42 and #43 write `step_500.pt` / `step_1000.pt` into the same
directory — whichever runs later overwrites the earlier one. This feature
must fix that bug, or pause/resume can't be relied upon.

### 5.4 Retraining after a config change (still the same version) (key scenario 2)

**Scenario**: task #42 runs for 1000 steps and gets paused. The user changes
the lr / dataset / optimizer configuration (whether by editing the version
config, editing a preset, or editing an external yaml), and submits task #43
(same version).

- A config change = a **new task** (a different task_id), not "resuming
  #42." The db records two independent task rows.
- Task #42 and #43 each write to their own state subdirectory (see §5.3),
  and their state doesn't affect each other.
- **Task #42's config snapshot has already been written to disk in
  `pause_step_1000.config.json`** — at pause time, all training parameters
  in use at that moment were frozen.
- After the user **edits the version / preset / external config file, task
  #42's snapshot is unaffected** — resuming #42 strictly uses the snapshot,
  fully decoupled from "what the latest config looks like."
- The user **cannot** resume task #42 with the edited config — see §8.5
  (resuming doesn't allow editing config; the UI disables the resume button
  and directs the user toward fork when it detects the task's config field
  or snapshot differs from the current effective config).
- If the user wants to "reuse the old state but run with a new config,"
  they use ResumeFieldPicker to start a brand-new task (§6 path B), manually
  pointing the .pt path at task #42's state file — this is an explicit fork.

**Core principle**: state ownership is locked to a task_id; config ownership
is locked to a snapshot. Config changes never retroactively pollute anything
about an old task.

### 5.5 Pause-file lifecycle (auto-deleted after a successful resume)

**Core rule**: pause files (the `.pt` + the matching `.config.json` snapshot
pair) are "pause anchors," valid only while the task is in the paused
status. Once the task leaves the paused status, **both files are
automatically deleted together** — at any moment, a task has at most 1 pause
file pair.

| Event | Handling of the pause file pair (.pt + .config.json) |
|------|---------------|
| `running → paused` | writes a new pause file pair, `paused_state_path` points to the .pt |
| `paused → pending` (resume, path A) | deletes the file pair after the child process **successfully loads** the state, clears `paused_state_path` |
| `paused → canceled` (fully canceled) | silently deleted (the task is abandoned, the files are meaningless) |
| deleting a paused task | deleted together (see §5.8) |

**Why the deletion happens "after a successful load" rather than "when the
user clicks resume"**:
- After the user clicks resume, the task's status changes to pending first,
  and it might take anywhere from seconds to minutes for the dispatcher to
  schedule it (other tasks might be running ahead of it).
- Once the child process is spawned, `load_training_state` could fail (a
  corrupted .pt / version mismatch / OOM). If the files were deleted at
  click-time, a failed load would leave the user with no way to retry.
- The safe timing: once the CLI side's `load_training_state` returns
  successfully, it emits `__EVENT__:resume_state_loaded:{"path":"..."}`, and
  the supervisor deletes the files only after receiving that event. On
  failure, the task → failed, the pause files are kept, and the user can
  choose to resume again or debug via ResumeFieldPicker.

**A reverse scenario the user was worried about** ("what if I want to keep
it"):
- Wanting to roll back to the old pause-time state → once resumed, training
  has already overwritten the in-memory state, and that .pt is already a
  stale copy. To actually roll back, the current resumed task would need to
  be canceled first, but by then the new pause file hasn't been generated
  yet and the old one is already deleted — the conclusion is: **rolling back
  isn't a feature of this feature**; to roll back, use §6 path B (manual
  fork) + time it right (use ResumeFieldPicker to start a fork task before
  resuming).
- Wanting to fork a new branch → should be done **before** resuming (use
  ResumeFieldPicker to start a new task, keeping the paused task around in
  parallel). Forking while the pause files still exist is a legitimate
  operation. Once resume is clicked, the intent is "keep running the same
  task," and the old state is no longer needed.

### 5.6 Periodic save file lifecycle (kept, managed by the user)

Completely independent from pause files:
- Written by `save_state_every` / `save_state_every_epochs`, these are
  disaster-recovery points the user opted into.
- **Not automatically cleaned up** — if the user turned this on, they want
  these checkpoints kept.
- Accumulate in the same `state/task_<TID>/` subdirectory, named
  `step_<N>.pt` (no `pause_` prefix).
- Distinguished from pause files by filename, so they're never accidentally
  deleted by the §5.5 "delete pause files" rule.
- Kept as historical checkpoints after the task is done / failed, and can be
  selected by ResumeFieldPicker to start a new task.
- Cleaned up manually by the user; or a future "bulk-clean after task is
  done" toggle could be added.

### 5.7 Config snapshot design (the other half of resume's anchor)

**Motivation**: resuming from the .pt file alone isn't enough —
load_training_state restores the optimizer + step + lr scheduler, etc., but
training parameters (dataset path, lr, optimizer type, batch size, loss
weighting, noise schedule, sampling config, etc.) come from external args /
config. These parameters **might get changed by the user** between pause and
resume (editing the version config, changing a preset, editing an external
yaml).

If resume read "the current effective config" again, a paused task's
training would be affected by whatever the user edited afterward, making
behavior unpredictable.

**Approach**: when pause triggers, `handle_interrupt` writes the `.pt` and,
alongside it, serializes **all training parameters actually in use by the
current process** into JSON, written to `pause_step_<N>.config.json`. Resume
strictly builds args from this snapshot, never reading the task table's
config field, the version config, or an external preset.

**Snapshot contents** (a candidate list; the final result is whatever gets
serialized from args at implementation time):
- All `args.*`: lr, optimizer, optimizer_args, scheduler, batch_size,
  grad_accum, max_train_steps, num_epochs, noise schedule, loss weighting,
  network_dim, network_alpha, rank, alpha, dropout, conv_dim, conv_alpha, …
- Dataset config: dataset_config / resolution / caption_extension / shuffle
  / repeat / class_tokens
- output_dir, output_name, sample_prompts, sample_every,
  save_every_n_steps, save_state_every…
- Key paths: base model, vae, text encoder (stores the path, not a hash)
- random seed
- **Not stored**: wandb run id (already finished), monitor live state
  (already dumped inside the .pt)

**Resume flow**:
1. UI clicks resume → the API reads task.paused_state_path → derives the
   adjacent `.config.json`.
2. Builds new args: based on snapshot.config.json, **only overriding**
   `--resume-state <pt_path>`, everything else uses the snapshot.
3. cmd_builder assembles the command → spawns the child process → the CLI
   runs with the snapshot's config + loads state from the .pt.

**db fields**: besides the already-planned `paused_state_path`, add
`paused_config_path` (an absolute path, overwritten together with state). It
could also be conventioned that "the snapshot path is always = the state
path's name with .config.json" and the db only stores one field — the
latter is simpler.

**The user edits config, then clicks resume**:
- The snapshot might now differ from the user's current config content —
  that's expected, the snapshot is the correct one.
- Suggested UI copy: "This task's config was frozen at pause time; resuming
  will use the config from that moment. To train with the new config,
  create a new task via ResumeFieldPicker."
- §8.5 covers the disabling / guidance rules further.

**The snapshot shares a lifecycle with the .pt**: see §5.5 / §5.8 — when the
pause files are deleted, the snapshot is deleted along with them (same
folder, same prefix).

### 5.8 Deleting a paused task (or a task in any status)

- Delete the db record.
- Handling of the `state/task_<TID>/` subdirectory:
  - **Pause files (.pt + .config.json)**: deleted along with the task (no
    downside scenario, as already argued in §5.5).
  - **Periodic save files**: the UI pops "also delete N checkpoint files?",
    defaulting to **keep** (the user might want to reuse them via
    ResumeFieldPicker).
- Per-subdirectory granularity makes bulk cleanup simple:
  `rm -rf state/task_<TID>/` works fine, without accidentally touching
  another task.

---

## 6. Two coexisting resume paths

| Entry point | File selection | Task status | Purpose |
|------|----------|----------|------|
| **A. Resume button** (a paused task) | automatically uses `paused_state_path` | the original task revives (paused → pending) | main path: coming back to keep running after a pause |
| **B. ResumeFieldPicker** (starting a new task) | the user manually picks any .pt | starts a new pending task | existing: cross-task resume / restarting from a done task's mid checkpoint / disaster recovery |

**The two aren't mutually exclusive or conflicting**:
- A only appears for a paused task, continuing a task's own internal
  lifecycle.
- B is always available, picking a .pt from the filesystem to start a new
  task.
- The same .pt file could theoretically be reused via path B repeatedly to
  start new tasks; paused_state_path is just a convenience pointer for path A.

**Why they aren't merged into one**: A preserves the original task's
identity (same row, same task_id, loss history / monitoring history
continues), while B is a new task (new id, new log). These are different
semantics, and forcing them together would confuse users about "why did my
task_id change when I clicked resume."

---

## 7. User Case list

### Case A — temporarily freeing up the GPU for something else
The user has trained 1000 steps and temporarily wants to run a generate on
the GPU.
→ task-level pause → the generate task runs (the queue's next item, or a
new one inserted) → once done with the GPU → clicks "resume."

### Case B — shutting down / going offline overnight
The user wants to shut their computer down at night, but wants to continue
tomorrow.
→ task-level pause (saves state) → shut down → the next day, power on,
start the server → the task's status is still paused → click "resume."

### Case C — deciding after observing the loss
1000 steps in, the loss looks off, and the user wants to analyze the sample
images offline before deciding.
→ pause → look at samples / the loss curve → decide to continue OR cancel
permanently (paused → canceled).

### Case D — adjusting scheduling strategy
There are 5 tasks in the queue; the second one is halfway done when the
user realizes the third one is more urgent.
→ pause the second → reorder the queue (or just let the third one queue up)
→ the dispatcher automatically runs the third → once it's done, click
"resume" on the second.

### Case E — the server needs to restart (user-initiated)
The user needs to restart the server (updating code / changing config).
→ task-level pause the currently running task (the user does this step
**deliberately**) → wait for the child process to exit → restart the server
→ resume.

**Key convention**: the server **does not** automatically pause a running
task on shutdown. The reasoning is detailed in §9 — there are too many
abnormal server-exit scenarios (crash / power loss / an OS kill), so "state
is always saved" can't be promised, and attempting it would give users a
false sense of security. The user should pause deliberately when they mean
to.

### Case F — letting the queue naturally stop tonight
"I don't want the queue to keep starting new tasks tonight, but it's fine
for the current one to finish."
→ hold the queue (a modal pops, case B, choose "let it finish") → the
current task finishes naturally → pending items in the queue won't start
automatically.
→ release the hold in the morning.

### Case G — stopping everything for maintenance
"I need to do system maintenance, stop everything."
→ hold the queue (a modal pops, case B, choose "also pause it") → neither
schedules anything new, nor keeps the current progress running unsaved.
→ after maintenance, release the hold + the user manually resumes each
paused task they want to resume.

### Case H — resuming an old task while held
"The queue is held, but I want a previously paused task A to get in line so
it's first to run once I release the hold tomorrow."
→ click "resume" on task A's row → task A's status goes paused → pending →
the UI shows "waiting for the queue to resume."
→ release the hold tomorrow → the dispatcher pulls tasks by priority +
creation-time ordering, and task A enters running in its turn.
→ (if the user wants it to run **first**, they'd need to reorder the
queue too — that's existing functionality, not part of this feature.)

### Case I — disaster / abnormal recovery (out of scope for this feature)
Not part of this feature; covered by the existing `save_state_every` +
`ResumeFieldPicker`:
process dies unexpectedly / OOM / BSOD / power loss → the task gets marked
failed → the user picks the most recent periodic .pt via ResumeFieldPicker
to start a new task.

**Prerequisite**: the user must have **deliberately enabled**
`save_state_every` in the training config (default 0, i.e. no intermediate
checkpoints are written). This feature doesn't replace that safety net; see
§9.

---

## 8. Boundaries / undefined behavior

### 8.1 Pausing too early (before global_step=0)
The user wants to pause while still in the dataset-loading / model-loading
stage, before entering train_loop, with `global_step=0` — any saved state
would be meaningless.

**Strategy** (two layers of defense, UI as primary):
1. **UI layer (primary)**: the pause button is **only shown once the task
   has entered train_loop**. During the startup stage, only the cancel
   button is shown.
   - Implementation: the CLI emits `__EVENT__:train_loop_started:{}` when it
     enters train_loop; the supervisor caches this on the slot and exposes
     `is_pausable: bool` through the task API / SSE.
   - The UI's `useMonitorProgress(taskId)` already subscribes to state, so
     it just needs one more field.
2. **API layer (defense)**: when the pause API is called directly, the
   server checks the `is_pausable` flag and returns 409 + explanatory copy
   if not ready. This covers cases where the UI display is delayed or the
   user calls the API directly.

Not doing "automatically deferring the pause until the loop starts" —
that would be over-engineering something simple; if the user can't see the
button, they already know it can't be paused right now.

### 8.2 The pause-in-progress modal (replaces "silently downgrading on timeout")
After the child process receives the pause signal, `handle_interrupt`
spends anywhere from a few seconds to over ten seconds saving state + the
config snapshot. The whole process is **covered by a modal on the UI side
the entire time**, as detailed in §4.3:

- The modal locks the screen the moment it opens; the user can't do
  anything else during this time (to avoid a stray click on cancel losing
  progress).
- On a 30s timeout, it does **not** automatically downgrade — instead, the
  modal lets the user choose from: [wait 30 more seconds] / [force-cancel
  and keep saved progress] / [terminate the task and discard progress].
- If the child process exits abnormally (rc != 0): the modal enters a
  failure state, guiding the user to check the logs.
- On success: the modal closes + a toast reads "Paused at step N."

**Underlying rule kept in place**:
- A task is only marked `paused` once the child process emits
  `__EVENT__:pause_state` and the files have been fully written to disk.
- If the user clicks "force-cancel and keep saved progress" in the modal: if
  the pause files have already been written to disk (both the snapshot and
  the .pt exist) → it's marked paused; otherwise it falls back to canceled.
- If the user clicks "terminate the task": a hard terminate is sent + it's
  marked canceled, with no pause files kept.

### 8.3 A task still running when the server restarts / exits abnormally
Current behavior: `supervisor.stop()` synchronously sends a hard terminate
signal, and state isn't saved; if the process is killed externally / the
power is lost, the task stays in the running status, and after a restart
the orphan scan marks it failed.

**This feature doesn't change that** — it's a deliberate choice not to
"automatically pause on server stop."
The reasoning is in §9: the coverage would be too narrow (only a
user-initiated stop could hook into this — crash / power loss / an OOM kill
would have no chance to run it at all), and doing it anyway would mislead
users. Instead, guide users to proactively configure periodic checkpoints
via `save_state_every`.

### 8.4 A paused task surviving a server restart
The state lives entirely in the db, and restarting the server doesn't touch
it. After a restart, the task is still paused, and the user can resume it
normally.
The orphan-cleanup logic only scans tasks that were "still running at
restart time" — paused tasks are unaffected.

### 8.5 Config changes (backstopped by the config snapshot)
**The snapshot already froze the config at the moment of pausing, in
`pause_step_<N>.config.json`** (see §5.7 for details), so if the user edits
the version config / preset / external yaml after pausing, it **won't
pollute** the paused task's resume.

- The resume button always runs from the snapshot, completely independent
  of "what the user's current effective config looks like."
- The UI shows an info note on the paused row / detail page: "This task's
  config was frozen at pause time; resuming will use the config from that
  moment. To train with a new config, create a new task via
  ResumeFieldPicker."
- No "edit config before resuming" entry point is exposed — changing config
  is only possible via §6 path B, explicitly forking a new task.

**Why the user isn't allowed to resume directly with a new config**:
- Optimizer state is tightly coupled to the old lr / old betas; switching
  optimizer type would just crash.
- Scheduler state is tightly coupled to the old warmup / total_steps.
- If the dataset changes, the loss history becomes meaningless.
- Supporting it anyway would create a pile of "sort-of-resume,
  sort-of-not" edge cases. Forking a new task is a clean boundary.

### 8.6 The wandb run
`handle_interrupt` already calls `finish()` on the original run; resume
starts a new run (not reusing run_id).
**This trade-off is accepted**, and documented as such. Continuing the same
run in the future would need a separately stored run_id — not done this
round.

### 8.7 Whether to clean the state subdirectory when deleting a paused task
- The pause file pair (.pt + .config.json): **deleted along with it** (no
  downside scenario, as already argued in §5.5 / §5.8).
- Periodic save files: the UI pops a confirmation, defaulting to **keep**.

### 8.8 Repeated pause / resume doesn't accumulate pause files
Every pause writes a new file pair (a different step), but **only 1 pair is
ever alive at a time** — it's deleted on a successful resume, and a new
pair is written on the next pause.
Periodic save files accumulate independently, with unchanged rules (§5.6).

---

## 9. Explicitly not doing

| Item | Reason |
|----|------|
| Editing config while paused | too semantically complex, easy to break resume |
| Moving the LoRA output directory while paused | output_dir's path is hardcoded inside the state file |
| Continuing the same wandb run id | a separate feature, needs an extra stored field |
| Automatically cleaning up periodic save files | user-opted-in checkpoints, managed by the user |
| Pausing generate / download / tag tasks | these tasks run quickly, so it wouldn't be meaningful |
| Marking it paused after a hard kill of the child process | state isn't trustworthy after a hard kill, must be marked canceled |
| **Automatically saving state on server stop / crash / power loss** | coverage is uncontrollable (only a user-initiated stop could hook into this — crash / power loss / an OOM kill / BSOD all have no chance), and doing it anyway would give users a false sense of protection; guiding users to enable `save_state_every` in the training config is the actual right answer |
| **Forcibly disabling task resume while held** | too clunky for the UI to keep toggling; "resume then queue up waiting for release" reads more naturally (§3.3) |
| **A combined "pause everything" button** | redundant for the UI; the hold modal already asks the extra question "also pause the running task?" (§4.3) |

---

## 10. Decision log

### Round 1 (the initial 5 open questions)

1. ~~**Whether queue-level pause is in scope for this round**~~ **decided**
   (§3.2 / §3.3 / §4 / §7 Case F-H):
   included in this round. Named **hold / release** to distinguish it from
   "pause." Status is a single db-persisted bool (survives a server
   restart), the task state machine is untouched. Hold is **orthogonal**
   to task-level pause / resume, and all 4 quadrants are valid.
2. ~~**A combined "pause everything" button**~~ **decided** (§4.4 / §9):
   not building a separate button. The queue-hold confirmation modal
   detects a running task and asks an extra question — "also pause it?" —
   covering both intents in one action.
3. ~~**Automatically saving state on server stop / crash**~~ **decided**
   (§8.3 / §9):
   not doing it. The coverage is too narrow (can only hook into a
   user-initiated stop; crash / power loss / BSOD are all missed), and
   promising "state is always saved" when it can't be guaranteed would
   mislead users. Instead, guide users to enable periodic checkpoints via
   `save_state_every`.
4. ~~**Handling of the .pt file when deleting a paused task**~~ **decided**
   (§5.5 / §5.8):
   the pause file pair (.pt + .config.json) is deleted along with the task
   (whenever it's resumed / canceled / the task is deleted) — a task has at
   most 1 pause file pair at any moment.
   Periodic save files: the UI confirms with a popup, defaulting to keep.
5. ~~**Pausing too early (train_loop hasn't started)**~~ **decided** (§8.1):
   the UI's pause button is **not shown** before the task enters train_loop
   (gated by the `__EVENT__:train_loop_started` event). The API adds a
   defense-in-depth rejection.

### Round 2 (incorporating the three-way review)

6. ~~**The term "freeze" is counterintuitive**~~ **decided** (§3 / throughout):
   renamed to hold / release scheduling, in English hold / release (or
   suspend / resume scheduling). Reasoning: "hold" is the standard computing
   term for suspend, and adding "scheduling" to the release term disambiguates
   it from the task-level "resume."
7. ~~**"Continue training" naming is asymmetric + collides with
   ResumeFieldPicker**~~ **decided**: the inline button is unified to
   **Resume**.
8. ~~**§8.2's pause timeout silently downgrading to cancel is startling to
   the user**~~ **decided** (§4.3 / §8.2):
   redesigned as a "pause-progress modal" covering the whole flow — the
   moment pause is clicked, the screen locks with a modal showing progress,
   and a 30s timeout offers the user a choice from [wait 30 more seconds] /
   [force-cancel] / [terminate the task].
9. ~~**Decoupling via a config snapshot**~~ **decided** (§5.1 / §5.7 / §8.5):
   `pause_step_<N>.config.json` is written to disk at the same time as
   pause, freezing all parameters currently in use by training. Resume
   strictly uses the snapshot, fully decoupled from any later edits the
   user makes to the version / preset / external yaml.
10. ~~**The modal copy's "it" is ambiguous**~~ **decided** (§4.4):
    every mention of a running task uses the specific `#{id} "{name}"`
    directly, with the radio button and primary button copy linked together.
11. ~~**Should the hold status use a banner or a chip**~~ **decided** (§4.1):
    only a top sticky banner is used, **not a task chip** — the banner is
    a UI element, not part of the task state machine. Nothing is shown when
    not held, to avoid noise.

### Round 3 (other review feedback, not addressed in this document, left for the ADR / UI spec / v2)

PM perspective:
- A success-metrics section (goes in the ADR)
- Feature flag / gradual rollout strategy (goes in the ADR)
- Splitting the per-task subdirectory change into its own upfront PR (goes
  in the PR breakdown plan)
- Changing `save_state_every`'s default away from 0 (evaluated in a separate PR)

User perspective:
- Showing step + loss on the paused row (v2)
- Bulk pause / resume (v2)
- Scheduled auto-release of a held queue (v2)
- A reminder for a task paused more than X days (v2)
- Recommending `save_state_every` be enabled in the UI on first training run (a separate PR)

Designer perspective:
- Status badge colors / banner / timestamp format (a separate UI spec document)
- a11y details (UI spec)
- Pause / cancel button colors + ordering (UI spec)
- Using a purple chip instead of small text for "waiting for the queue to resume" (UI spec)

---

## 11. Next steps

Once this document's direction is settled, it can be split into an ADR +
implementation PRs:
- The ADR lands at `docs/adr/000X-queue-pause-resume.md`, inheriting this
  document's logic model.
- The implementation PR breakdown follows the workflow discussed previously
  (spike → backend skeleton → API → cmd → UI → docs).
- Before implementation, run the Windows signal spike first, confirming the
  `CTRL_BREAK_EVENT` → `SIGBREAK` → handle_interrupt chain works.
