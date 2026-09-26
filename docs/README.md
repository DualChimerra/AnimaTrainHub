# Documentation

Docs are split into five categories, each for a different use case:

| Directory | Audience | Update cadence |
|---|---|---|
| [`user-guide/`](user-guide/) | Users, community contributors | Updated whenever behavior changes |
| [`architecture/`](architecture/) | Developers changing code or debugging | Updated on architecture changes |
| [`adr/`](adr/) | Anyone who wants to know "why is it this way" | New entries added for new decisions; old ones are never rewritten, only get a status appended |
| [`design/`](design/) | Anyone who wants to see how an ADR's discussion played out | Updated continuously during the design phase; frozen as a reference once the ADR lands |
| [`todo/`](todo/) | Maintainers; notes on "can't do this now, revisit later" | Revisited or archived once the trigger condition is met |

> **Not here**: version changes are in the root [`CHANGELOG.md`](../CHANGELOG.md); Studio's internal module structure is in [`studio/README.md`](../studio/README.md).

---

## User guide

| Doc | Contents |
|---|---|
| [tagging-guide.md](user-guide/tagging-guide.md) | Anima tag format, best practices, tag ordering |
| [training-tips.md](user-guide/training-tips.md) | Training parameters, VRAM configuration matrix, over/underfitting troubleshooting, ComfyUI usage |
| [regularization.md](user-guide/regularization.md) | How regularization-set generation works (tag-distribution greedy search + AR clustering) |
| [caption-format.md](user-guide/caption-format.md) | JSON caption format + category shuffling |
| [custom-models.md](user-guide/custom-models.md) | Using your own base model / VAE / text encoder weights, and how local base models work |

## Architecture

| Doc | Contents |
|---|---|
| [studio-pipeline.md](architecture/studio-pipeline.md) | Cross-cutting architecture overview: data model, directory layout, SQLite schema, secrets, SSE events, tagger abstraction, preset pool |

## Architecture Decision Records (ADR)

Historical decision records. They document "why we chose X over Y" — once landed, they're history and **are never deleted**; they're kept so that if we ever want to revisit the choice, we know what the original trade-offs were.

| ADR | Status | Contents |
|---|---|---|
| [0001-lokr-via-lycoris-lora.md](adr/0001-lokr-via-lycoris-lora.md) | Accepted (2025) | Switched LoKr to the official lycoris-lora library instead of moving to sd-scripts |

See [adr/README.md](adr/README.md) for details.

---

## Local drafts

`docs/_local/` is listed in `.gitignore`. Use it for scratch notes, unfinished design writeups, or temporary TODOs within the repo without polluting commits.
