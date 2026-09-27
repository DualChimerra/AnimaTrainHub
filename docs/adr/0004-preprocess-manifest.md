# 0004 — Replace preprocessing state's "dual bucket + per-image sidecar" with a single manifest

**Status**: Superseded by [0010](0010-preprocess-train-scope.md) (2026-06-04 — preprocessing has moved down to version-level train scope; this ADR's project-level `preprocess/` + `preprocess/manifest.json` write path has been removed, keeping only a read-only fallback for 0010 §`ensure_train_manifest` to migrate old projects)
**Date**: 2026-05-15
**Decision makers**: @WalkingMeatAxolotl

## Background

PR #69 added a "preprocess" stage to AnimaLoraStudio (an upscaler + a smart-skip
strategy). Once shipped, it produced two UX / architecture problems:

### 1. UX: disk layout translated verbatim into the UI

The current pipeline is two parallel directory levels on disk:

```
projects/{id}-{slug}/
  download/      original images
  preprocess/    output PNGs + a {name}.preprocess.json sidecar next to each image
```

The frontend `Preprocess.tsx` maps 1:1 onto disk: the "unprocessed grid"
corresponds to images in `download/` with no output, and the "processed grid"
corresponds to images with output in `preprocess/`. The user **has to open the
preprocess tab to "discover" how many images need processing**, and the "two
grids" visual model on the page implies "two sets of images," which doesn't
match the user's mental model of "my dataset = one set of images."

Downstream seams expose this problem further. `studio/curation.py:98-109`'s
`_active_left_dir`:

```python
def _active_left_dir(pdir):
    pre = pdir / "preprocess"
    if pre.exists() and any(f.is_file() ... for f in pre.iterdir()):
        return pre, "preprocess"
    return pdir / "download", "download"
```

"if preprocess/ is non-empty, use preprocess/, otherwise use download/" is
**all-or-nothing**. It means users cannot mix — for example, "of these 100
images, I only want to upscale the 20 small ones, and use the original 80 as
is" isn't possible (once preprocess/ has any output, the entire left-side
batch switches to preprocess/, and those 80 images end up filtered out for
lacking a source there).

The `left_source` field leaks into the frontend API (asserted by
`tests/test_curation_endpoints.py:78`), and the frontend uses it to branch the
thumbnail URL builder — yet another place exposing disk layout to the client.

### 2. Architecture: state scattered across N sidecars

`studio/services/upscaler.py:353` writes a `{name}.png.preprocess.json` after
processing each image:

```json
{"source": "...", "model": "...", "scale": 4, "src_size": [W,H], "dst_size": [W,H],
 "action": "upscale", "elapsed_seconds": 12.3, ...}
```

`studio/preprocess.py:list_processed` reads each sidecar individually when
listing images and assembles a response from them. This has several problems:

- **No "user decision" state**: a sidecar is only written once a worker
  finishes, but a missing sidecar does not mean "the user chose to skip
  it" — it could simply never have run, or run and failed and been deleted.
  There's no way to express "I **deliberately** don't want this image
  processed, use the original."
- **Bulk operations are expensive**: listing 1000 images means 1000 stats and
  1000 JSON parses.
- **No atomicity**: a single image's state is fine, but "reset this project's
  preprocessing state" requires 1000 unlinks, and a ctrl-c mid-way leaves a
  half-clean directory.
- **"Restore" logic is scattered**: it has to delete both the PNG and the
  sidecar; missing one leaves the state inconsistent.

### 3. The target mental model the user proposed

In the user's own words:

> When a version is created, make a copy directly in the preprocess folder;
> unmodified images can use JSON to represent their placeholder state;
> downstream filtering pulls the actual file from download, and modified
> images get a copy created, with filtering pulling from preprocess. In the
> user's eyes there's no "preprocess," just one folder.

What's wanted is **one set of images + a per-image processing state**, with
the disk layout invisible to the user.

Constraints added during the design discussion:

- preprocess/ stays **project-level** (not pushed down to version-level) —
  reason: upscaling hundreds of images is genuinely expensive, and reusing
  preprocessing results across versions fits the real workflow far better
  than "preprocess independently per version" (users mostly iterate on
  dataset composition + training hyperparameters, and preprocessing
  decisions are usually settled once).
- **One manifest file** records all state, **not** N sidecars.
- **No "deleted" state** — unchecking in the filtering stage is enough; the
  preprocess page doesn't introduce a "delete image" semantic, to avoid
  confusing "delete the original vs. delete the output."
- **No version field** — YAGNI; add it the day a schema migration is actually
  needed.
- **Implicit original**: an image not recorded in the manifest defaults to
  using the original; the manifest only stores non-default decisions.

## Candidate solutions

### Candidate A — keep the status quo

Keep using `_active_left_dir`'s dual bucket + sidecar approach.

- Pros: zero effort
- Cons: none of the UX/architecture problems above are addressed; there's
  nowhere to put a future "user skipped this image" semantic
- **Rejected**

### Candidate B — push preprocess down to version-level + the same N sidecars

`versions/{label}/preprocess/`, independent per version.

- Pros: enables comparisons like "v1 unprocessed / v2 fully upscaled"; keeps
  filtering/tagging/training consistent at the version level throughout
- Cons: reusing preprocessing results across versions becomes **impossible**
  — the same batch of images, once processed for v1, would have to be
  reprocessed for v2, hundreds of upscales taking tens of minutes each time.
  This is the most expensive part of the preprocessing stage
- **Rejected** (explicitly vetoed by the user: "don't move it to version
  level, or it'll have to be reprocessed every time")

### Candidate C — project-level single manifest + implicit original (**chosen**)

- A single `projects/{id}/preprocess/manifest.json` records **non-default
  decisions**
- "Not recorded in the manifest" = use the `download/` original (implicit
  original)
- "Manifest has `kind: processed`" = use `preprocess/{name}.png` (an actual
  copy)
- A single resolver function, called by every downstream consumer
  (thumbnail / curation / tagging / training materialize)
- Pros: single-grid UX + status badges; downstream seams collapse from "dual
  bucket fallback" to "a resolver lookup"; per-image state becomes
  expressible; bulk listing is a single JSON read
- Cost: needs the supervisor to serialize manifest writes (to prevent
  concurrency issues); needs to migrate old sidecars
- Full implementation details in the next section

## Decision

**Candidate C** is chosen.

### Manifest schema

`projects/{id}/preprocess/manifest.json`:

```json
{
  "images": {
    "bar.png": {
      "kind": "processed",
      "source": "bar.jpg",
      "model": "RealESRGAN_x4",
      "scale": 4,
      "action": "upscale",
      "target_area": 1048576,
      "src_size": [512, 512],
      "dst_size": [2048, 2048],
      "elapsed_seconds": 12.3,
      "mtime": 1731000000
    }
  }
}
```

- key = output file name (always `.png`)
- `value.kind` currently only has one value, `"processed"`; it can be
  extended later (e.g. `"cropped"`, `"masked"`)
- Implicit originals get no entry
- **No `version` field**: on the day the schema is extended, old files
  without the field are treated as v0

### Single-point resolver

`studio/services/preprocess_manifest.py:resolve(project, name)` → `Path | None`:

```python
def resolve(project_dir, name):
    m = load_manifest(project_dir)
    entry = m["images"].get(name)
    if entry is None:
        return project_dir / "download" / name   # implicit original
    if entry["kind"] == "processed":
        return project_dir / "preprocess" / name  # copy
    raise ValueError(f"unknown kind: {entry['kind']}")
```

Every image-reading entry point (thumbnail / curation left side /
copy_to_train) goes through this single function.
**Remove `_active_left_dir` / `list_left_source` / `left_source` from the API
response**.

### Concurrent writes: single-consumer supervisor

All manifest mutations go through
`studio/services/preprocess_manifest.py:_with_lock` (an in-process
`threading.Lock` for serialization + atomic tmp+rename writes to disk). Both
write sources use this:

- Preprocess worker finishes an image → `add_processed(project_id, name, meta)`
- User clicks "restore" → `restore(project_id, name)` (deletes the entry +
  deletes the PNG)

Reasoning for an in-process lock rather than a file lock: the server is a
single process, and the CLI never bypasses it to write the manifest (the CLI
has no preprocessing workflow). If cross-process access is needed in the
future, upgrade to `portalocker`.

### "Not listed" semantics

`list_pending` / `list_processed` are changed to read the manifest and diff
against the download dir:

- `download/foo.png` exists + no manifest entry → "unprocessed" (pending)
- `download/foo.png` exists + manifest has an entry → "processed", thumbnail
  reads from preprocess/foo.png
- `download/foo.png` doesn't exist + manifest has an entry → orphan (output
  exists but the source was deleted), UI marks it as orphan
- `download/foo.png` doesn't exist + no manifest entry → the image doesn't
  exist, not returned

### Migration

Old projects have `*.preprocess.json` sidecars (written under preprocess/ per
the `SIDECAR_SUFFIX` convention in `studio/preprocess.py:48`). On first access
to that project's preprocess data:

1. Check whether `manifest.json` exists; if so, skip migration
2. Scan `preprocess/*.preprocess.json` → aggregate them into manifest.json
3. Old sidecar files are **kept, not deleted** (defensive rollback + zero
   deletion risk); new code no longer reads them

Migration is idempotent: if the manifest already exists, it just returns
without retrying.

### UI model

`Preprocess.tsx` changes from "dual grid" to "single grid + status badge":

```
N total · Unprocessed X · Processed Y                [All] [Unprocessed] [Processed]

┌─[img]─┐ ┌─[img]─┐ ┌─[img]─┐
│ ⊘ Pending│ │ ✓ 4x │ │ ⊘ Pending│
└───────┘ └───────┘ └───────┘

┌─ X pending · [model ▾] [tile 256] [Start preprocessing] ─┐
```

The "restore" button lives in the preview/edit overlay of a processed image;
it also supports multi-select bulk restore.

### No special handling for externally deleted download files

If a user deletes `download/foo.png` in the OS file manager:

- No manifest entry → resolver returns None → downstream skips it, no error
- Manifest has an entry → preprocess/foo.png still exists, downstream reads
  it as usual; UI marks it as orphan

**No active reconciliation is performed** — this is user action, and we don't
do "auto cleanup," to avoid eating state the user didn't intend to remove.

## Rationale

**Why a single manifest instead of keeping N sidecars plus an overview file**:
DRY — state has one single source of truth. Keeping both means every update
has to keep the two in sync, which inevitably drifts.

**Why project-level instead of pushed down to version-level**: driven by the
user's real workflow — "preprocess once, reuse across multiple versions with
different dataset compositions + hyperparameters" is the common case, while
"different preprocessing per version" is rare. If the latter is ever truly
needed, a version-override field can be added (per-version branches nested in
the manifest), but it's not pre-invested in v1.

**Why implicit original instead of an explicit placeholder for everything**:

- Saves manifest size (changing 50 out of 1000 images → 50 entries)
- "No decision" and "decided to use the original" naturally merge into one
  state — there's no semantic difference to the user
- "Newly added images automatically count as unprocessed" needs no extra
  backfill code

**Why no "deleted" state**: the user explicitly said not to include it in v1.
"Deleting an image on the preprocess page" would confuse users about "which
directory is this deleting from." Deleting images belongs to the filtering
stage: uncheck it, and it doesn't go into train/.

**Why no version field**: YAGNI. Schema changes are rare; when one is truly
needed, a field can be added and old files treated as "missing field = v0."
Adding `"version": 1` preemptively is just ceremony without solving a real
problem.

**Why a serializing supervisor instead of a file lock**: the server is a
single process with no cross-process writer. File locks exist to handle
multiple processes / an external CLI writing simultaneously — that scenario
doesn't exist here. Falling back to `threading.Lock` is 90% simpler.

## Consequences

### Benefits

- The disk abstraction disappears from the user's view: the UI shows only
  "one set of images + status badges"
- Downstream seams (`curation.py` / `server.py` thumbnails / `copy_to_train`)
  collapse to a single `resolver()` call
- Per-image state now has an explicit schema; adding "crop / mask / mark as
  skipped" later is just a new manifest kind
- Bulk listing drops from O(N) stat+JSON-parse to O(1) manifest read
- "Reset this project's preprocessing" becomes atomic (rm preprocess/ +
  rewrite an empty manifest)

### Costs / new constraints

- **Every manifest write must go through `_with_lock`** — any bypass write
  will lose updates. Code review needs to enforce this
- **The migration entry point must be correct**: it triggers on first access
  to that project's preprocess data; if any code path bypasses that entry
  point and reads the manifest directly, it needs its own migration call
  added
- **Old sidecars are kept but ignored**: disk usage increases slightly
  (~500B per image), which is acceptable. A future major version could clean
  them up
- **Larger test surface**: needs unit tests for manifest schema / atomic
  writes / migration / concurrent writes

### Debt taken on / future extensions

- If version-level preprocessing overrides are ever truly needed (the user
  changes their mind), nest a `version_overrides:
  {v1: {bar.png: {kind: "skip"}}}` layer in the manifest, with the
  project-level entries remaining the baseline — no storage redesign needed
- If an explicit placeholder for "the user deliberately skipped
  preprocessing" (distinguishing "not yet decided" from "decided to skip")
  is ever truly needed, add `kind: "skip"` — but it's not introduced in v1
- If cross-process writes ever become necessary (e.g. an independent
  preprocessing daemon), upgrade from `threading.Lock` to
  `portalocker.Lock` — the surrounding logic stays the same

## References

- The PR that triggered this discussion: [#69 feat(preprocess): preprocessing stage (upscaler)](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/pull/69)
- Affected code: `studio/curation.py` `_active_left_dir`, `studio/preprocess.py` `list_processed`,
  `studio/services/upscaler.py:353` sidecar writes, `studio/web/src/pages/project/steps/Preprocess.tsx` dual grid
- Design discussion: this session, 2026-05-15

---

## Addendum 1 — introducing the crop stage + simplifying the manifest schema (2026-05-21)

**Trigger**: a PR introduced a second preprocessing stage, "crop" (one image
can be cropped multiple times, producing fan-out derivatives like
`X.png → X_c0.png / X_c1.png`), and the original schema's "output name →
processing-step metadata" model was no longer sufficient: derivatives aren't
in a direct 1:1 relationship with a single download original, and clearer
origin tracing is needed. See
[`docs/design/preprocess-crop-design.md`](../design/preprocess-crop-design.md)
for details.

During the discussion, the user explicitly called out over-engineering: rect
coordinates / target AR / action classification / VRAM estimation are all
"process information" that becomes stale the moment it's written to disk, and
shouldn't be persisted. The final choice was a minimal schema to start with;
old fields are read for compatibility but dropped on write, and deprecated
gradually over the following minor releases.

### Schema evolution (v0 → v1, read-compat dual support)

**v1 (new writes)**: entries keep only three fields.

```json
{
  "images": {
    "X.png":     { "origin": "X.png",  "mtime": 1731000000, "size": 1234567 },
    "Y_c0.png":  { "origin": "Y.png",  "mtime": ...,         "size": ... },
    "Y_c1.png":  { "origin": "Y.png",  "mtime": ...,         "size": ... }
  }
}
```

- `origin` = the original image name under download/. Multiple **multi-crop
  derivative** entries can share the same origin
- The `kind` field is **no longer written**. An entry existing means
  "processed" (in v0, `kind` only ever had one value, `processed`, so it was
  effectively dead already; if a real "future state" is ever needed, add a
  `state` field — don't revive `kind`)
- The old `source / model / scale / action / target_area / src_size /
  dst_size / elapsed_seconds` fields are **no longer written at all** — they
  are process information and should be dropped once written to disk

**v0 (old reads)**: read-compat is kept for at least 3 minor releases.

- Missing `origin` → falls back to the `source` field (same meaning, just
  renamed)
- Missing `source` → falls back to the entry key (= the 1:1 same-name case)
- If a `kind` field is present and is not `"processed"`, it's still treated
  per the v0 rule as a future extension, and the resolver returns None
- Old model/scale/... fields, if present, are still passed through to the
  frontend when read (the sidebar can still display them), but any
  subsequent mutation rewrites the entry as v1 schema, dropping the old
  fields

The old sidecar (`*.preprocess.json`) migration logic (`ensure_manifest`)
stays unchanged — old formats already aggregated into manifest entries remain
covered by read-compat.

### New API

Three additions to `studio/services/preprocess_manifest.py`:

| Function | Purpose |
|---|---|
| `entry_origin(entry, fallback)` | Extracts origin from an entry, handling v0/v1 schema compatibility |
| `resolve_origin(project_dir, download_name)` | Reverse resolve: given download/X, returns every derivative in preprocess/ whose origin matches (multi-crop is one-to-many); falls back to download if there's no match |
| `replace_with_crops(project_dir, source_name, outputs)` | Atomic replace: deletes source_name itself plus every entry whose origin points to source_name, and writes N new entries in one batch. Called by the crop worker during fan-out |

The `resolve(name)` interface itself is unchanged.

### Downstream consumer changes

- `studio/preprocess.py:list_pending` — compares by origin set (no longer by
  stem), handling the "processed" determination for multi-crop fan-out (if
  any entry's origin == download/X.jpg, it doesn't count as pending)
- `studio/preprocess.py:summary` — same origin-set determination
- `studio/curation.py:copy_to_train` — uses `resolve_origin()` to fan-out copy
  multiple crop derivatives into train/{folder}, with filenames matching
  preprocess's actual output names (including the `_c0`/`_c1` suffixes), and
  metadata matched by stem
- `studio/server.py` thumbnail endpoint — likewise uses `resolve_origin()`,
  and for multi-crop shows the first preprocess derivative

### File naming convention (fan-out)

- N=1 crop → overwrites `preprocess/{stem}.png` (same-name overwrite, origin
  unchanged)
- N>1 crops → writes `preprocess/{stem}_c0.png` / `{stem}_c1.png` ..., and
  deletes the original `{stem}.png`
- Cropping `{stem}_c0.png` again → `{stem}_c0_c0.png` / `{stem}_c0_c1.png`
  (chained suffixes; origin always points back to the very first download
  original)

### Restore semantics unchanged

`restore(names)` still deletes the entry + the corresponding preprocess/
file. Multi-crop derivative entries can each be restored independently
(deleting `Y_c0.png` doesn't affect `Y_c1.png`). Reverting a whole image to
its download original = deleting every entry derived from that image's
origin.

### Stages don't enforce an ordering

"Upscale → crop → upscale → crop" is a valid chain: each stage overwrites the
current state of `preprocess/`, with no partial undo. The `kind: cropped` /
`kind: masked` classification assumed in the original ADR's §"Future
extensions" section is **not adopted** — all output entries are the same
type (`origin + mtime + size`); stage information exists only while a worker
is running and never enters the manifest.

### Alignment with training ARB buckets

The crop clustering's target AR is internally snapped to training buckets
(`runtime/training/dataset.py:BucketManager`, mirrored on the frontend in TS
by `studio/web/src/lib/trainBuckets.ts`), but the UI label still displays a
pretty AR. This is unrelated to the manifest schema — it's purely frontend
logic, noted here only in passing. See
[`preprocess-crop-design.md §7`](../design/preprocess-crop-design.md) for
details.

### Anti-drift measure

The backend Python `BucketManager` and the frontend TS `trainBuckets.ts` must
keep their algorithm + default parameters in sync. Both sides cross-reference
each other in a comment at the top of the file, plus a TS unit test asserting
the bucket count is exactly 37 (the Python output is also 37 under the
default parameters). Changing one side without the other fails CI.

### Retirement timeline for old-schema read support

The v1 `origin` schema becomes observable once dev merges to master after
running clean on dev. Read-compat for v0:

- For 3 minor releases after the v1 schema is introduced, v0 fields are still
  read (as a fallback for old entries written during the 0.7-0.9 era)
- An old user's first access to preprocess data triggers the
  `ensure_manifest` migration; after that, any mutation rewrites old entries
  as v1
- In a later release (around 1.0+), the `source` fallback branch in
  `entry_origin()` and `_scan_legacy_sidecars()` can be removed

No explicit deprecation timeline is written into the code, to avoid pressuring
users with a "must upgrade" feeling. When the day comes to clean it up, decide
based on actual usage data.
