# Queue Pause Signal Chain Spike Report

**Date**: 2026-05-18
**ADR**: [0006-queue-pause-resume](../adr/0006-queue-pause-resume.md)
**Spike script**: [`tools/spike/`](../../tools/spike/)

## Conclusion

**ADR §Candidate A (signal channel + reusing `handle_interrupt`) is viable** and moves to PR-1 on the main line.
Candidate B (sentinel-file IPC) goes back to the parking lot.

## What was run

On Windows 11, a mock training child process was spawned and the full signal
chain was run end to end, following the ADR §backend code direction:

```
parent (supervisor mock)             child (training mock)
─────────────────────────            ─────────────────────────
Popen(CREATE_NEW_PROCESS_GROUP)
                          ──spawn──▶ signal.signal(SIGBREAK, h)
                                     signal.signal(SIGINT, h)
                                     emit __EVENT__:train_loop_started
                                     fake step loop (step += 1 / 0.5s)
sleep 3s
proc.send_signal(CTRL_BREAK_EVENT)
                          ──CTRL_BREAK──▶
                                     handler triggered (sig=21)
                                     fake_save_training_state(.pt)
                                     fake_write_config_snapshot(.json)
                                     emit __EVENT__:pause_state
                                     sys.exit(0)
stdout reader:
  parse __EVENT__:pause_state ◀─────
proc.wait() → rc=0
validate 6 checks
```

## Results of the 6 checks (both consecutive runs passed all of them)

| # | Check | Result |
|---|------|------|
| 1 | `CTRL_BREAK_EVENT` reaches the `CREATE_NEW_PROCESS_GROUP` child process group | PASS |
| 2 | `signal.signal(SIGBREAK, handler)` catches the signal | PASS (handler received sig=21) |
| 3 | Handler fully completes save + emit + `sys.exit(0)` | PASS (rc=0) |
| 4 | Parent reads `__EVENT__:pause_state` and parses the payload | PASS |
| 5 (extra) | A pause `.pt` + `.config.json` pair is written to disk | PASS |
| 6 (extra) | The `__EVENT__:train_loop_started` event can serve as the `is_pausable` signal | PASS |

**Key number**: signal-sent to child-process-exit took **0.62 s** (including a
0.5 s fake IO sleep). Real training state is tens to hundreds of MB, so disk IO
dominates in practice — but this spike validates the signal path, not IO
throughput.

## Key findings

### Python Windows signal mapping

`CTRL_BREAK_EVENT` (OS level) maps to `SIGBREAK` (Python signal constant value
`21`). **Not SIGINT** — the Python Windows docs explicitly state that only
`CTRL_C_EVENT` maps to SIGINT, but a `CREATE_NEW_PROCESS_GROUP` child process
group never receives `CTRL_C_EVENT`, only `CTRL_BREAK_EVENT`. So the ADR's
requirement that `runtime/training/phases/resume.py` additionally register a
SIGBREAK handler on Windows is mandatory, not an optional optimization.

### subprocess.Popen + stdout=PIPE goes through line buffering

For the parent to receive events in real time via `for raw in proc.stdout:`,
both sides are required:
- Child side: the `PYTHONUNBUFFERED=1` environment variable **and**
  `print(..., flush=True)` as a double safeguard.
- Parent side: `bufsize=0` to disable the parent's own readahead buffering.

Setting only one side delays events until several KB of stdout accumulate —
in the first version of the spike, without `bufsize=0`, event lines didn't
appear until the child process exited, which nearly got misread as "the event
was never emitted."

### Parent stdout's own encoding

A minor pitfall unrelated to the main project: on Windows, the shell's default
codepage (cp936 / cp932) raises `UnicodeEncodeError` when encoding non-ASCII
`print` output. `env["PYTHONIOENCODING"] = "utf-8"` only takes effect for the
**child process** — the parent itself needs
`sys.stdout.reconfigure(encoding="utf-8")`. The real supervisor writes to
`stdout=log_fp` (a file) rather than a console, so it never hits this.

## Impact on the implementation

### Adopted directly

- On Windows, ADR §`runtime/training/phases/resume.py` **must** additionally
  register `signal.SIGBREAK` — it cannot rely on SIGINT alone.
- On Windows, ADR §`studio/supervisor.py`'s `_send_pause_signal` sends
  `CTRL_BREAK_EVENT`, and on POSIX it sends `SIGINT` — consistent with the
  spike.
- After ADR §`runtime/training/context.py`'s `handle_interrupt` finishes
  saving, emitting `__EVENT__:pause_state` is required — the parent cannot
  determine "paused" from `rc=0` alone; it must observe the event line (`rc`
  is unreliable under the Windows wrapper's rewriting scenario).
- On Windows, ADR §Cancel switches to `taskkill /T /F`: the signal channel is
  **genuinely** reserved exclusively for pause, and this decision is now
  implemented.

### Fallback candidate B is not needed

Candidate B (sentinel-file IPC) was kept as a fallback in the three-way review
in case the spike failed. Since the spike passed, candidate B does not go into
PR-1 through PR-5; it stays in the ADR §Candidate solutions section as a
historical record. It can be revisited if future Windows behavior changes or a
new platform breaks this approach.

### PR-3 integration test coverage is still needed

The spike used a **fake** save (500ms sleep + tens of KB of files) to validate
the flow. In the real case, tens to hundreds of MB of optimizer state, a
wandb finish, and a monitor flush mean the whole handler running for 3-10
seconds would not be unusual. Whether the ADR §4.3 pause-in-progress modal's
30s timeout threshold is reasonable, and edge cases like slow disks / a full
SSD / antivirus file locks, must be covered by the PR-3 end-to-end integration
tests — they are out of scope for this spike.

## Reproducing

```bash
git checkout chore/queue-pause-signal-spike
python tools/spike/pause_signal_parent.py
```

Expected: all 6 checks PASS, plus `Conclusion: all passed — ADR candidate A is
viable`, and `rc=0`.

## Spike script lifecycle

Once PR-0 merges into dev, the script stays as executable evidence that
"candidate A was once validated." A cleanup PR removes `tools/spike/` when
either of the following happens:

- The full ADR 0006 PR set (PR-1 through PR-5) has merged and been validated
  in production.
- Or this report's conclusion is reversed (candidate A turns out unstable on
  the real signal chain).

This report is kept independently of the script — the script shows "how it was
validated," while the report shows "that it was validated, and what the
conclusion was."
