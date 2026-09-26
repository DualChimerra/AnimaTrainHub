# 0010 — Move preprocess scope from project-level download down to version-level train

**Status**: Accepted (PR-1/2/3/4/5 all merged; wrapped up 2026-06-04)
**Date**: 2026-06-03
**Decision makers**: @WalkingMeatAxolotl
**Supersedes**: [ADR 0004 — Replace "dual bucket + per-image sidecar" preprocessing state with a single manifest](0004-preprocess-manifest.md) (including Addendum 1)

## Background

ADR 0004 fixed preprocessing state as **a single project-level manifest + dual-bucket resolver + implicit original**, with a workflow of:

```
download/  ──(preprocess the whole set)──>  preprocess/  ──(curate selects)──>  versions/{label}/train/
   ↑ project-level                    ↑ project-level                          ↑ version-level
```

"Reusing preprocessing results across versions" was handled directly by the project-level `preprocess/` directory — this was the core argument for choosing project-level scope in ADR 0004 §74-83.

Use by beta users revealed **four real pain points**:

1. **Wasted upfront time**: a booru scrape brings in hundreds-to-thousands of images, of which only a small fraction ends up used; the user has to act on every image in the whole set (waiting for upscales, skipping through crops one by one), even though most of them will never enter train
2. **Meaningless statistics**: resolution distribution / aspect-ratio distribution are computed over the entire `download/` set, not the final train set — they don't inform training decisions
3. **Smart clustering breaks down**: the goal of clustering is unified ARB buckets, but clustering happens over the entire download set, and downstream curation filters it further, **scrambling the buckets again**
4. **Scope mismatch**: in the user's mental model, "the images I want to train on" is the train set; preprocessing operating on the full download set doesn't match that model

The two factual assumptions behind ADR 0004's project-level scope have also changed:

- After **ADR 0007** landed, `create_version(fork_from_version_id=...)` (`studio/services/projects/versions.py:407-498`) implements a **whole-tree fork** via `_copytree("train")`. train/, including already-upscaled outputs, naturally carries over to the child version on fork — **no project-level cache is needed** to get cross-version reuse
- User research: **the vast majority of new versions are copies of the previous version with minor tweaks**; the worst case of "completely redo preprocessing in v2" almost never happens

The optimization ADR 0004 targeted (cross-version reuse) is now covered by ADR 0007's fork mechanism under these new facts, and the scope mismatch has instead become the dominant cost.

## Candidate approaches

### A — Keep ADR 0004's current state

- Pros: 0 effort
- Cons: none of the four pain points are addressed; statistics / clustering built on the wrong set is a **correctness bug**, not just a UX shortcoming
- **Rejected**

### B — Compromise with a UI filter (add a "show train only" filter to the Preprocess page)

Keep the project-level disk layout, and just add a UI filter so the user's view focuses on the train set. Cost ~2.5 days, doesn't touch any ADR.

Checking each pain point against this fix:

| Pain point | Does the filter solve it? |
|---|---|
| Wasted upfront time (acting on images that won't be used) | **No** — the filter is applied after the fact, the time is already spent |
| Statistics meaningless because based on the full download set | **No** — the statistics source is unchanged |
| Broken clustering (unified ARB buckets scrambled again downstream) | **No** — the clustering source is unchanged |
| Only a small fraction of booru material is used | Partially (hidden visually, doesn't change the data flow) |

**0/4 real fixes**. The filter addresses "visual clutter" — a symptom, not the root cause. **Rejected**.

### C — Move preprocess scope down to the version-level train/ (**adopted**)

New workflow:

```
download/  ──(curate selects)──>  versions/{label}/train/  ──(process train)──>  tag / train
   ↑ project-level source-of-truth     ↑ version-level (preprocessing and training co-located)
```

Preprocessing happens **after** curation, applying only to the final training set. Cross-version reuse is covered by fork's whole-tree copy.

Details in §Decision.

## Decision

**Approach C** was chosen.

### Disk layout

```
projects/{id}-{slug}/
  download/                       # project-level, still the source-of-truth (unchanged)
    X.jpg, Y.jpg, ...
  preprocess/                     # old directory, kept as-is (fallback rebuild source; not actively deleted)
    manifest.json                 # old v0/v1 schema, read-only
    *.png                         # old outputs, read-only
  versions/{label}/
    train/
      manifest.json               # new, per-version (v2 schema)
      X.png + X.txt               # training bytes + caption (caption is not part of the manifest)
      Y_c0.png + Y_c0.txt         # multi-crop derivatives
      Y_c1.png + Y_c1.txt
    reg/ samples/ output/ config.yaml
```

**Key invariant**: the project-level `preprocess/` directory is **never actively deleted** (see §Invariants).

### Manifest schema v2

`versions/{label}/train/manifest.json`:

```json
{
  "version": 2,
  "images": {
    "X.png":    { "origin": "X.jpg", "mtime": 1731000000, "size": 1234567 },
    "Y_c0.png": { "origin": "Y.jpg", "mtime": 1731000000, "size": 800000 },
    "Y_c1.png": { "origin": "Y.jpg", "mtime": 1731000000, "size": 850000 }
  }
}
```

**Field semantics**:

- `version` (int): schema version, currently fixed at `2`
- `images` (map): key = the **POSIX-style relative path** within train, `"{N_label}/{image}"`
  (LoRA's repeat-folder structure: `train/1_data/X.png` → entry key `"1_data/X.png"`),
  value = the entry
- `entry.origin` (string): the **flat filename** of the corresponding source image under `download/`
  (download/ has no sub-folder structure), used for reverse lookup during restore
- `entry.mtime` / `entry.size` (number): metadata of the output file, used to detect external modification

**Why a relative path instead of a flat name**: LoRA's training dataset config treats `train/{N_label}/`
as a repeat folder (N = repeat count); the same image can appear in multiple folders (rare but legal).
A flat entry key can't express uniqueness across folders or resolve same-name collisions. The POSIX form with `/` is consistent across platforms.

Images placed directly at the `train/` root are **ignored** (LoRA training only reads inside sub-folders); the validator
`_validate_rel_name` enforces the two-segment `folder/image` format, preventing path traversal.

**Status field** (2026-06-04 fixup):

- `processed: true`: written once a worker completes upscale / crop. The frontend uses this field to render the
  "processed" badge. An entry missing this field (curation copied the original image / an old entry) is treated as unprocessed.
- Old entries (from manifests written before the fixup, after an upscale pass) missing `processed`
  are treated as unprocessed; rerunning preprocess upgrades them to the new field.

**Why `processed` is stored explicitly** (a revision of the original "infer from field differences" principle):

The original design planned to infer "processed" from "final path segment's extension differs from origin's,"
but after the fixup, **upscale doesn't change the extension** (it overwrites in place, preserving the source extension; see §worker behavior),
so an extension mismatch is no longer a reliable signal. Other alternatives (size diff vs. download, mtime diff)
are also unreliable or expensive. An explicit bool field is the simplest, most accurate choice. It remains the only status field,
still in keeping with the "minimal manifest" spirit — it doesn't store model/scale/action or other process information,
only "has this been processed" — the one piece of semantics a user actually sees in the UI.

**Fields not stored** (explicitly rejected, consistent with ADR 0004 Addendum 1):

- `kind` (old field: `processed` / `cropped` / `masked`, overlaps with `processed:bool`)
- `source` (old field name, replaced by `origin`)
- `model / scale / action / target_area / src_size / dst_size / elapsed_seconds` (process information, should be discarded once written)
- `state` (a multi-value enum) — `processed:bool` as a two-state field is enough; multi-state is derived by the UI by combining `processed`
  + `duplicate_removed` + `orphan`
- `caption_mtime / tags / caption` (caption belongs to the tagging stage, its own independent lifecycle)
- `kind: duplicate_removed` (manual dedup review state — this ADR moves dedup scope down to the train set, reviewed independently per version, with state not shared across versions; see §dedup scope)

### Worker behavior: extension preserved (2026-06-04 fixup)

Worker upscale / crop **doesn't change the filename or extension** — `X.jpg` stays
`X.jpg` after upscaling, `X.png` stays `X.png`. Reasons:

- Changing the extension (`X.jpg → X.png`) would break the LoRA dataset config's extension-based glob matching
- Changing the stem would break the caption correspondence (captions are matched by stem: `X.txt`)

The upscaler saves according to the source extension:

- `.jpg` / `.jpeg` → JPEG quality=95
- `.webp` → WebP quality=95 method=6
- everything else (PNG / BMP / GIF etc.) → uncompressed PNG

In-place overwrite goes through a tmp file + atomic rename to prevent the training framework from reading a half-written file. JPEG re-encoding does introduce a small quality loss —
this trade-off is accepted in exchange for caption / dataset-config compatibility.

### Fallback rebuild mechanism: `ensure_train_manifest`

Old projects already have preprocessing outputs sitting in train/ (copied during curation); the only thing missing is the train ↔ download origin relationship. `ensure_train_manifest(project_dir, version_label)` is called defensively at every manifest read/write entry point, and is idempotent.

**Rebuild rules** (in priority order):

1. The target `versions/{label}/train/manifest.json` already exists → return immediately (an O(1) stat, zero overhead on the hot path)
2. Doesn't exist + the old `projects/{id}/preprocess/manifest.json` exists → recursively scan
   `train/`'s first-level sub-folders to collect the set of relative image paths (image extensions only; images placed
   directly at the root are ignored); match by filename (the last segment of the relative path) against old flat entry names → rebuild the v2
   schema, using the relative path as the entry key
3. The old manifest doesn't exist / is corrupted (not a dict / invalid JSON) → write an empty v2 manifest (`{"version": 2, "images": {}}`)
4. An old entry marked `kind: duplicate_removed` → **skipped** (manual dedup review state doesn't migrate across models; users redo dedup within train scope under the new model)
5. Same-named images across different sub-folders (rare but legal, e.g. `1_data/X.png` + `5_extra/X.png`) →
   each gets its own independent entry, both inheriting the same old origin

Concurrency: serialized via `threading.Lock` + atomic `tmp+rename` writes + double-check to prevent races.

**Why a fallback instead of an explicit migration script**: the physical bytes in the user's train/ are already outputs copied over during preprocessing — **deleting the project-level `preprocess/` doesn't affect train/'s contents**. The only thing lost is the origin lookup, and a 30-line lazy rebuild is enough. This saves ~1 person-day and zero perceived user impact compared to an explicit script + UI dialog, with zero failure-rollback cost (a failed rebuild just retries on the next call).

### Fate of the resolver

ADR 0004's `resolve(name) → Path` resolver was the central abstraction designed to eliminate the "dual-bucket fallback (download/ vs preprocess/)". **Under the new model, train/ is self-contained** — thumbnail / curation / tagging / training materialization all read directly from `train/{name}`, with no ambiguity.

| Function | Fate |
|---|---|
| `resolve(project_dir, name)` | **removed** (the dual-bucket concept is gone) |
| `resolve_origin(project_dir, download_name)` | **removed** (no caller needed reverse resolve) |
| `list_pending()` | **removed** (the binary "unprocessed / processed" concept is gone) |
| `ensure_manifest()` + `_scan_legacy_sidecars()` | **removed** (no longer migrates old sidecars; the old manifest is read once by `ensure_train_manifest`) |
| `entry_origin()` | **kept** (needed for fallback reading of old entries / reverse lookup during restore) |
| `add_processed / restore / mark_duplicate_removed / clear_all / replace_with_crops` | **kept, with a `version_label` parameter added** |

### Restore semantics

`restore(project_dir, version_label, name)`:

1. Calls `ensure_train_manifest` first
2. Looks up `manifest.images[name].origin`
3. Copies `download/{origin}` over `train/{name}`
4. Updates the manifest entry's mtime/size

**When `download/{origin}` is missing**: an explicit failure + a UI prompt. The specific image is listed with three options:

- **Drag in a replacement** — pick a local file via a picker to overwrite `download/{origin}`, then retry restore
- **Keep the processed version** — ignore the failure, leave train/{name} unchanged
- **Remove from train/** — delete train/{name} + delete the manifest entry (destructive, requires confirmation)

Carries forward ADR 0004 §215-219's "no active reconciliation when download/ is deleted externally" principle: **no hidden backup bytes** (no per-version `.backup/`, which would violate the "download is the sole backup" invariant). Restore failure is an intentional accepted trade-off — an explicit UI failure beats a silent false restore.

### Cross-version reuse: fork's whole-tree copy

Handled through ADR 0007's existing fork mechanism:

- `create_version(fork_from_version_id=...)` calls `_copytree("train")`, recursively copying the train subtree
- `train/manifest.json` **comes along automatically with the copy** (it's within the recursive copy target, zero code changes)
- After forking, `ensure_train_manifest(new_label)` is called once as a safety net (in case the source manifest was corrupted and needs rebuilding)
- v2 starts out with train/ identical to v1 → the preprocessing phase auto-skips (all outputs inherited)

Cost: forking copies several GB of train/ (including upscale outputs). **Accepted by the user** — "v3 usually only changes training parameters, not train" is the most common case, so the disk cost amortizes well; for the rare case of a full preprocessing redo, users explicitly trigger a "redo stage" button.

### Phase state machine change (companion to the ADR 0007 amendment)

`VersionPhase.ORDER` gains `preprocessing`:

```python
# Before (ADR 0007 §132):
ORDER = (curating, tagging, editing, regularizing, ready)         # 5 phases
SKIPPABLE = {regularizing}

# After:
ORDER = (curating, preprocessing, tagging, editing, regularizing, ready)   # 6 phases
SKIPPABLE = {preprocessing, regularizing}
```

`check_preprocessing(conn, version_id)` follows the same pattern as `check_regularizing`:

- Validation = no preprocess job pending / running
- Doesn't hard-require "has any image been processed" — skippable, matching the `regularizing` mental model
- The UI adds "(optional)" after the phase name, exactly the same pattern as `regularizing`'s existing `nav.reg = "Regularization set (optional)"`

### Two-layer migration mechanism

**Layer 1 — `_v11_preprocessing_phase` DB migration** (implicit add-only, same pattern as `_v8_version_status_phase.py`):

Runs after lifecycle PR-5 (`_v9 destructive`). Backfill rules:

| Existing phase | train/ state | New phase |
|---|---|---|
| `curating` | empty | `curating` (unchanged) |
| `curating` | non-empty | **`preprocessing`** |
| Others (tagging / editing / regularizing / ready) | any | unchanged |

Rationale: per the user, "preprocessing's images have already been copied into the existing train" — a non-empty train/ means curating has effectively already been completed. Zero perceived impact for users.

**Layer 2 — the `ensure_train_manifest` implicit fallback** (see §Fallback rebuild mechanism):

The DB migration handles the phase field; the fallback handles the manifest file. The two layers are decoupled.

### Dedup / blur / clustering scope

All moved down to the **train set** as well:

- `studio/services/preprocess/duplicates.py:_resolve_download_sources` → `_resolve_train_sources(version_label)`, scope changed to `versions/{label}/train/`
- **No module split** (during the R1/R2 design phase, "moving dedup up to the ingestion stage" was considered as a hard prerequisite, and ultimately rejected) — once moved down to train, dedup exists to clean up the train set itself, which is a different mental model from "cleaning the project-level material pool"; splitting the module would only add complexity
- Manual dedup review state is now **independent per version** under the new model (carried along with the train manifest on fork, not shared across versions)

### What's left unchanged

- `download/` remains the project-level source-of-truth; the semantics of copying from download into train during curation are unchanged
- The `services/preprocess/` module (worker / upscale / crop / blur / dedup core logic) is kept — this ADR changes **where state is stored**, not the computation's responsibilities
- ADR 0007 §70's decision that "dataset ownership = project-level" is **not overturned** — the train set's source-of-truth is still the project-level download pool; this ADR only co-locates **preprocessing outputs** with train

## Invariants (must-read before future modifications)

The following constraints are hard constraints of this ADR's data model. Future refactors in related areas **must preserve them** or explicitly overturn them and write a new ADR:

1. **`download/` is the sole persistent backup of the original images** — it exists forever unless the user deletes it externally; never invent a `.backup/` or shadow directory duplicating download bytes
2. **The old `projects/{id}/preprocess/` directory is never actively deleted** — it serves as the fallback rebuild source + a backup of old data; the user decides when to clean it up; only consider adding a guided user cleanup in a later minor release
3. **The manifest only stores "current-state reverse lookup relationships," never "process information"** — anything outside the schema's three fields `{origin, mtime, size}` counts as process information (kind / model / scale / action / state / op chains / rect / target_area / etc.). Process information should be discarded once written
4. **Status fields are kept minimal** — `processed: bool` is the only allowed status bool (added by the 2026-06-04
   fixup, see §Manifest schema v2); adding another status field first requires demonstrating that field-difference inference
   or the existing fields are insufficient
5. **train/ is self-contained** — every downstream consumer (thumbnail / curation / tagging / training materialization) reads directly from `train/{name}`; **none** go through a dual-bucket fallback. The manifest is used for reverse origin lookup (restore / derivation relationships), not for "path resolution"
6. **caption is decoupled from the manifest** — captions (`.txt`) belong to the tagging stage, and are not part of the manifest; preprocessing edits to images don't touch captions
7. **Cross-version reuse goes through fork's whole-tree copy**, not a project-level cache — this is handled by ADR 0007's existing mechanism, not reinvented here
8. **Restore failure must be explicitly visible** — when `download/{origin}` is missing, restore **must fail explicitly** and give the user three options; **silently succeeding / faking a restore is forbidden**
9. **DB migration is decoupled from manifest fallback** — `_v11` changes the phase field, `ensure_train_manifest` rebuilds the manifest file; the two mechanisms are independent and each idempotent
10. **An in-process `threading.Lock` is sufficient** — the server is single-process, with no cross-process writer. If cross-process writes emerge in the future (e.g. an independent preprocessing daemon), upgrade to `portalocker.Lock` without changing the outer logic

## Rationale

**Why ADR 0004's project-level scope was overturned**:

It's not that the original reasoning was wrong at the time — two **factual assumptions** changed:

1. The fork mechanism landed by ADR 0007 achieves "cross-version reuse" without needing a project-level cache — the core argument of ADR 0004 §74-83 is now covered by fork
2. The actual user workflow is "v2 copied from v1 with tweaks" far more often than "v2 redone from scratch" — the worst case of redoing an upscale pass that takes tens of minutes almost never happens

ADR 0004 §227's "cross-version reuse" argument no longer supports project-level scope under the new facts; instead the scope mismatch (preprocessing spanning inside and outside of train) has become the dominant cost.

**Why approach B (UI filter) was not chosen**:

A filter only changes the view — the upfront time is already spent / the statistics source is unchanged / the clustering source is unchanged. **0/4 real fixes**. Treating a filter as a solution is mistaking a visual symptom for the root cause.

**Why a fallback instead of an explicit migration**:

The physical bytes in train/ are already outputs copied over during preprocessing (which already happened during curation) — the only thing lost is the origin lookup. A 30-line lazy rebuild is enough, with zero perceived user impact, saving ~1 day of effort and zero failure-rollback cost compared to an explicit script + UI dialog.

**Why the schema is kept minimal**:

Carries forward ADR 0004 Addendum 1's principle: state is inferred implicitly from field differences, and process information should be discarded once written. This principle has already been validated through two rounds of iteration in 0004; this ADR inherits it without introducing new rules.

**Why add a phase instead of only touching the Sidebar UI**:

Adding a phase makes the state machine more strictly conform to ADR 0007's design principles; the precondition is that it's purely an implicit DB migration with zero perceived user impact — which is satisfied (`_v8_version_status_phase.py` already proved add-only migrations are a clean pattern). `check_preprocessing` follows the exact same pattern as `check_regularizing`, at zero design cost.

**Why ADR 0007 §70 ("dataset ownership = project-level") isn't overturned**:

§70 rejected "a per-version independent train set (the dataset itself being version-level)" — under this ADR the train set's source-of-truth is still the project-level `download/` pool (the curating phase copies from download into train); what this ADR moves down is **preprocessing outputs**, not dataset ownership. ADR 0007 just needs an amendment to sharpen the wording.

## Consequences

### Benefits

- All 4 user pain points are genuinely fixed: upfront time / statistics / clustering / scope mismatch, all resolved
- Aligned with user mental models: preprocessing is "fine-tuning the images I'm going to train on"
- ADR 0007's fork mechanism naturally covers cross-version reuse without reinventing a cache
- The manifest module is substantially slimmed down (25 functions cut roughly in half + a much simpler schema)
- Zero-perception upgrade for old projects (implicit fallback)
- Dedup / clustering metrics finally become meaningful for training

### Costs / new constraints

- `_copytree("train")` copying several GB of train + manifest during fork is a real disk cost. Already accepted by users ("v3 usually only changes parameters" being the most frequent case)
- `restore` **genuinely fails** when download is missing — intentional, consistent with ADR 0004's principle; an explicit UI failure beats a silent false restore
- The old `projects/{id}/preprocess/` directory occupies disk space long-term — the next minor release notes will remind users they can manually clean it up, not enforced
- 11 preprocess API endpoint URLs change from pid → (pid, vid), a **breaking change** (acceptable given beta expectations + frontend and backend switching in the same PR, no redirect compatibility period)
- The `_v11_preprocessing_phase` migration is the second time the phase column is touched after ADR 0007's `_v9 destructive` — it must strictly run after _v9

### Debt still owed / future extension

- Cleaning up the old `preprocess/` directory: the next minor release notes will remind users they can delete it; after a few releases, consider adding a branch in `ensure_train_manifest` that returns an empty manifest directly if the old manifest doesn't exist (once most old projects have already been lazily rebuilt, the old fallback path becomes dead code that can be cleaned up)
- If a high-frequency need emerges in the future for "redo a specific stage after copying v1 into v2": currently the user explicitly clicks a "redo upscale" button inside v2 (manual); a fork-time dialog prompt could be added later if needed, but this isn't pre-invested in now
- If manual dedup review state really needs to be shared across versions in the future: this ADR explicitly skips `duplicate_removed` entries from the old manifest during fallback. Reintroducing this would need a new schema field + an ADR

## References

- Superseded ADR: [ADR 0004](0004-preprocess-manifest.md) (including Addendum 1)
- Related ADR: [ADR 0007](0007-project-version-lifecycle-refactor.md) (gains Addendum 1: the `preprocessing` phase)
- Unchanged ADRs: [ADR 0008](0008-studio-restructure-0.11.0.md) (module boundaries) / [ADR 0009](0009-logging-error-system.md) (logging system)
- Implementation detail (file:line change list + 4-PR breakdown + test cases + risk list): [`docs/design/preprocess-train-scope-plan.md`](../design/preprocess-train-scope-plan.md)
- Key code files:
  - `studio/services/preprocess/manifest.py` (schema + `ensure_train_manifest` + `entry_origin` / `restore`)
  - `studio/services/projects/versions.py:41-58` (`VersionPhase.ORDER` / `SKIPPABLE`)
  - `studio/services/projects/versions.py:407-498` (`create_version` fork flow)
  - `studio/services/projects/phase.py` (`check_preprocessing`)
  - `studio/infrastructure/migrations/_v11_preprocessing_phase.py` (DB migration)
