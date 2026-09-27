# Architecture Decision Records

Records architectural-level "**we chose X over Y**" decisions and the reasoning behind them. Code tells you "what," but not "why not the other way" — that's the gap ADRs fill.

## Index

| # | Title | Status | Date |
|---|---|---|---|
| 0001 | [LoKr adapter via lycoris-lora instead of switching to sd-scripts](0001-lokr-via-lycoris-lora.md) | Accepted | 2025 |
| 0002 | [In-webui self-update (flag + shell wrapper loop)](0002-webui-self-update.md) | Proposed | 2026-05-12 |
| 0003 | [anima_train.py modular refactor (plugin boundaries + adapter hook protocol)](0003-anima-train-refactor.md) | Proposed | 2026-05-14 |
| 0004 | [Replace "dual bucket + per-image sidecar" preprocessing state with a single manifest](0004-preprocess-manifest.md) | Superseded by #0010 | 2026-05-15 |
| 0005 | [Update channel as a user view preference, decoupled from git worktree state](0005-update-channel-as-preference.md) | Accepted | 2026-05-16 |
| 0006 | [Queue task pause/resume + queue suspend/resume scheduling](0006-queue-pause-resume.md) | Accepted | 2026-05-18 |
| 0007 | [Project/Version/Task lifecycle refactor](0007-project-version-lifecycle-refactor.md) | Proposed | 2026-05-23 |
| 0008 | [studio/ 4-layer refactor (0.11.0)](0008-studio-restructure-0.11.0.md) | Accepted | 2026-05-28 |
| 0009 | [Unified logging + error system (0.12.0)](0009-logging-error-system.md) | Accepted | 2026-05-28 |
| 0010 | [Move preprocess scope from project-level download down to version-level train](0010-preprocess-train-scope.md) | Accepted | 2026-06-03 |
| 0014 | [LyCORIS 4 fused kernels and Windows Triton](0014-lycoris4-fused-kernels.md) | Accepted | 2026-09-20 |

## Status values

- **Proposed** — drafted, undecided
- **Accepted** — adopted and implemented
- **Superseded by #N** — replaced by a newer ADR (original text kept, not deleted)
- **Deprecated** — no longer applicable (reason noted, original text kept)

## Writing a new ADR

```markdown
# NNNN — Short title (starts with a verb)

**Status**: Proposed | Accepted | Superseded by #N | Deprecated
**Date**: YYYY-MM-DD
**Decision makers**: @handle / team

## Background

The problem, constraints, and external factors at the time. Written so a future reader can understand it without needing the original context.

## Candidate approaches

Briefly list every approach discussed, including the ones not chosen. Give pros and cons for each.

## Decision

Which one was chosen, and what was done.

## Rationale

Why this choice was made. Focus on **the specific reasons other approaches were rejected** — that's the core value of an ADR.

## Consequences

Benefits gained after implementation, new constraints introduced, technical debt that may need to be repaid later.

## References

Links to relevant PRs, commits, external resources.
```

File naming rule: `NNNN-kebab-case-title.md`, with the number a four-digit incrementing sequence.
