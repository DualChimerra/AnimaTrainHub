# Preprocess scope relocation — implementation plan

**Status**: Plan (finalized before implementation)
**Date**: 2026-06-03
**Authors**: @WalkingMeatAxolotl (lead) + Claude (Opus 4.7) (primary reviewer)
**Companion ADR**: [ADR 0010](../adr/0010-preprocess-train-scope.md) supersedes ADR 0004 + Addendum 1
(Note: ADR 0009 is already taken by "unified logging + error system," so this proposal is 0010)

---

## 0. One-sentence goal

Narrow the preprocess stage's scope from the **project-level full `download/` set**
down to the **per-version, already-selected `versions/{label}/train/` set** —
stats / clustering / dedup all follow along. This fixes the user pain point:
"preprocessing images that won't end up used is wasted time, and stats/clustering
based on the full download set are meaningless for training."

---

## 1. Design decisions at a glance (final outcome of each discussion point)

Each row is tagged with its source (user's own words / round-3 consensus / lead
reviewer's recommendation).

| # | Topic | Decision | Source |
|---|---|---|---|
| D1 | Data-flow direction | **`download → curate (into train/) → preprocess (operates on train/) → tag → train`** — preprocessing moves to **after** curation/selection | User vote, 2026-06-03 |
| D2 | Where preprocess output lands | **Inside `versions/{label}/train/`**, co-located with training bytes/captions | Round-3 three-way consensus |
| D3 | Project-level `preprocess/` directory | **Kept as-is, never actively deleted** — serves as a fallback rebuild source + legacy backup | User feedback, round 2, 2026-06-03 |
| D4 | Manifest file name | **`train/manifest.json`** (not hidden, no leading dot) | User feedback, round 2, 2026-06-03: "shouldn't be hidden" |
| D5 | Manifest schema | **v2, minimal: `{origin, mtime, size}`** — no ops/rect/action/scale/model or other process metadata | MEMORY `feedback_preprocess_data_model_simple` + round-3 three-way consensus |
| D6 | Cross-version reuse mechanism | **Via fork's whole-tree copy** (`versions.py:_copytree("train")` already supports this; manifest.json now rides along automatically during the recursive copy). **No** project-level cache | User: "a new version is never created from an empty template — it's always copied from the previous version" |
| D7 | `restore` semantics | **Copy `download/{entry.origin}` back over `train/{name}`**; fails with an explicit UI message if the download original is missing | User: "download is the single source of truth — once it's deleted, there's no way to restore" |
| D8 | Restore fallback design | **Not building** a per-version `.backup/` copy of the bytes (would violate the "download is the only backup" principle) | User accepts this limitation |
| D9 | Dedup / blur-detection scope | **Entirely relocated to the train set**, no module split (`duplicates.py:_resolve_download_sources` → `_resolve_train_sources`) | User: "stats based on the full download set are meaningless for training, clustering breaks down" |
| D10 | Stats metrics (resolution distribution, aspect ratio) | **Scope changes to train/**, moved to the Preprocess page's Overview sub-page; removed from the Download / Curation pages | User's own words |
| D11 | Smart clustering / ARB buckets | **Scope changes to train/** — since nothing downstream filters afterward, buckets stay stable | User: "clustering exists to unify ARB buckets, but filtering after unification scrambled the buckets again" |
| D12 | Legacy-project compatibility strategy | **Implicit lazy fallback**: an `ensure_train_manifest()` function that, when `train/manifest.json` is missing, implicitly rebuilds it by reverse-matching the old `preprocess/manifest.json` against the actual files in `train/`; **zero user-visible impact** | User feedback, round 2, 2026-06-03 (details in §3.2) |
| D13 | Migration trigger points | **Every manifest-read entry point defensively calls `ensure_train_manifest`** (idempotent, near-zero cost) + also called once from `create_version(fork_from=...)` | User: "every manifest [read path]" |
| D14 | Migration target version | **No migration needed** (train/ already holds the processed images; deleting preprocess/ doesn't affect train/) | User: "no need for a migration target version" |
| D15 | Add a `preprocessing` phase to ADR 0007 | **Yes** (`VersionPhase.ORDER` gets a new value), on the condition that it's only an implicit, user-invisible DB migration | User: "adding it fits our state-machine design more strictly" |
| D16 | Can the preprocessing phase be skipped | **Yes** (`SKIPPABLE` gets `PREPROCESSING` added), same pattern as `regularizing` | User: "skippable, same as regularization" |
| D17 | UI copy | Sidebar label `"Preprocess (optional)"` — exactly the same pattern as the existing `nav.reg = "Regularization set (optional)"` | User: "the name can have (optional) appended" + current `i18n/locales/en.json:13` |
| D18 | Sidebar stepper order | `download → curate → preprocess → tag → edit → reg → train` (preprocess moves from its original `idx=②` (project scope) to version scope, positioned after curate) | Round-3 impl §2.4 + a necessary consequence of entering the phase state machine |
| D19 | `_v11` DB migration backfill | phase=curating + train/ **non-empty** → advance to preprocessing; phase=curating + train/ empty → stays curating; other phases untouched | User's own words: "the preprocessing images have already been copied into the existing train" → a non-empty train/ implies curating has already passed |
| D20 | API endpoint URL redesign | 11 preprocess endpoints change from `pid` to `(pid, vid)` | Round-3 impl §1.3 |
| D21 | Backward compatibility for old API URLs | **No compatibility redirect** (frontend PR switches over in lockstep; beta mindset) | Beta + user accepts the break |
| D22 | Legacy sidecar `*.preprocess.json` | **Dropped entirely** (`manifest.py:_scan_legacy_sidecars` + `ensure_manifest` removed) | Beta, no dead read paths left behind |
| D23 | Dead-code compatibility shim | **Not kept** (violates the `tagger_config_two_surfaces` lesson: two inconsistent surfaces are a testing nightmare) | Round-3 UX §5.3 + impl §3.1 |
| D24 | Feature flag | **Not added** | Beta + one-shot migration + keeping old files as backup is sufficient |
| D25 | ADR revision path | New **ADR 0010** supersedes 0004; ADR 0004's header gets `Status: Superseded by ADR 0010`; ADR 0007 gets an amendment adding the PREPROCESSING phase; ADR 0008 / 0009 untouched | Round-3 arch §5 (numbering correction: 0009 is already taken by the unified logging/error system) |
| D26 | Multi-crop fan-out | **Kept** (derived images `Y_c0.png` / `Y_c1.png` still live in train/ + share an origin; origin reverse-lookup still holds) | Round-3 arch §3.2 |
| D27 | Caption (.txt) placement | The manifest **does not record captions** (captions are owned by the tagging stage — a separate lifecycle from preprocess origin, shouldn't be mixed into the manifest) | Round-3 arch §6.3 |
| D28 | Fate of `copy_to_train` | **Drastically simplified**: removes the "preprocess-derived vs. original download image" dual branch (`curation.py:210-275`), becomes a plain download → train copy | Round-3 impl §2.1 |
| D29 | PR slicing | **4 PRs**: PR-1 ADR 0010 + fallback (0.5d) / PR-2 backend manifest+core+worker (3.0d) / PR-3 _v11 migration + phase + API (1.5d) / PR-4 frontend + Sidebar (2.5d) | Round-3 impl §7 |
| D30 | PR sequencing | PR-1 can start immediately, independent; PR-2/3/4 need the lifecycle PR chain + `feat/preprocess-multitool` merged to dev first | User: "I've looked at all the new PRs, they don't conflict functionally with what's current" |

---

## 2. Data model

### 2.1 On-disk layout

```
projects/{id}-{slug}/
  project.json
  download/                        # project-level source of truth (unchanged)
    X.jpg, Y.jpg, ...
  preprocess/                      # legacy directory, kept, never deleted (D3)
    manifest.json                  # old v1 schema, read-only, serves as a fallback rebuild source
    *.png                          # legacy output, read-only
  versions/{label}/
    version.json                   # unchanged from ADR 0007's current state
    train/
      manifest.json                # new, per-version (D4)
      X.png                        # a train image (may be post-upscale output, may be the original)
      X.txt                        # caption (not in the manifest, D27)
      Y_c0.png + Y_c0.txt          # multi-crop derivatives
      Y_c1.png + Y_c1.txt
    reg/
    samples/
    output/
```

### 2.2 Manifest v2 schema

```json
{
  "version": 2,
  "images": {
    "1_data/X.png":    { "origin": "X.jpg", "mtime": 1731000000, "size": 1234567 },
    "1_data/Y_c0.png": { "origin": "Y.jpg", "mtime": 1731000000, "size": 800000 },
    "1_data/Y_c1.png": { "origin": "Y.jpg", "mtime": 1731000000, "size": 850000 }
  }
}
```

**Field semantics**:

- `version` (int): schema version, currently fixed at `2`. Future schema changes
  branch on this.
- `images` (map): key = **POSIX-relative path** `"{N_label}/{image}"` (the LoRA
  repeat folder: `train/1_data/X.png` → entry key `"1_data/X.png"`), value = entry.
- `entry.origin` (string): the flat filename of the corresponding original image
  under `download/` (download/ has no sub-folder structure). Used for restore
  reverse-lookup.
- `entry.mtime` (number): the output file's mtime (Unix timestamp, seconds). A
  mismatch can be used to infer "the user modified this externally."
- `entry.size` (number): the output file's size in bytes.

**Why a relative path instead of a flat name**: the LoRA dataset config treats
`train/{N_label}/` as a repeat folder, so the same image name can legitimately
(if rarely) appear in multiple folders. Images placed directly at the `train/`
root are ignored; `core.py:_validate_rel_name` enforces the two-segment
`folder/image` shape to prevent path traversal.

**Processing state is inferred from field differences, not stored explicitly**:

- `name="X.png"` + `origin="X.jpg"` → was processed (extension changed, which
  necessarily went through transcode / upscale / crop)
- `name="X.jpg"` + `origin="X.jpg"` and `mtime`/`size` match the actual
  train/X.jpg file → untouched, as-is
- `name="X.jpg"` + `origin="X.jpg"` but `mtime`/`size` don't match → modified
  externally
- There is no explicit "was it processed" bool. This carries forward the "don't
  store process metadata" principle from ADR 0004 Addendum 1.

### 2.3 Fields deliberately NOT stored

Consistent with ADR 0004 Addendum 1. **Not stored**:

- `kind` (the old v0 field: `processed` / `cropped` / `masked`) — already rejected
- `source` (the old v0 field name) — superseded by `origin`
- `model` / `scale` / `action` / `target_area` / `src_size` / `dst_size` /
  `elapsed_seconds` — process metadata that should be discarded once written
- `state` (a new enum like `"upscale-done"`) — inferred from field differences,
  never stored explicitly
- `caption_mtime` / `tags` — captions are owned by the tagging stage (D27)

---

## 3. Data flow and lifecycle

### 3.1 Full pipeline

```
[Create project]
   │
   ▼
1. Download page (project scope)
   │   Import images into download/ (booru scrape / drag-in / upload)
   │   Shows: total original count / total disk usage
   │   Does NOT show: resolution distribution / aspect-ratio distribution /
   │   ARB buckets (these move to Preprocess)
   ▼
2. Create / select a version
   │
   ▼
3. ① Curate page (version scope, phase=curating)
   │   Pick images from the download/ pool — "selected" = copy
   │   download/X.jpg → train/X.jpg
   │   ※ copy_to_train simplified: a plain 1:1 file copy + creates a manifest
   │      entry { origin: "X.jpg" }
   │   Completion check: train/ ≥ 1 image (unchanged)
   │   Advance button → phase: curating → preprocessing
   ▼
4. ② Preprocess page (version scope, phase=preprocessing, **optional**)
   │   ┌─ Overview sub-page: stats / ARB bucket preview computed from train/ (D10/D11)
   │   ├─ Upscale sub-page: batch RealESRGAN, replacing training's default upscaling
   │   │   ※ worker reads train/X.jpg → writes train/X.png + updates the manifest entry
   │   ├─ Crop sub-page: multi-crop fan-out (X.png → X_c0.png / X_c1.png)
   │   │   ※ worker removes the manifest entry for the original X.png + writes
   │   │      X_c0.png / X_c1.png sharing its origin
   │   └─ Dedup/Blur sub-page: scoped to train/
   │   "Next" button is always clickable (D16, skippable)
   │   Skip advance: phase: preprocessing → tagging (as long as there's no
   │   concurrent preprocess job)
   ▼
5. ③ Tag page (version scope, phase=tagging)
   │   Tag train/ (unchanged)
   ▼
6. ④ Edit page / ⑤ Reg / ⑥ Train (unchanged)
```

### 3.2 `ensure_train_manifest` implicit fallback rebuild (D12 / D13)

**Target function** (new): `studio/services/preprocess/manifest.py:ensure_train_manifest()`

**Triggered**: defensively, by every entry point that needs to read the train
manifest (D13), idempotently. Specifically:

| Call site | File:line | When triggered |
|---|---|---|
| `load_manifest(project_dir, version_label)` | `manifest.py` entry point | Before any read |
| `restore(project_dir, version_label, names)` | `manifest.py` | Before restoring |
| `add_processed / mark_duplicate_removed / replace_with_crops` | `manifest.py` | Before any write |
| Thumbnail endpoint | `api/routers/projects/curation.py:117-186` | Before listing images |
| `list_train_images` | `services/preprocess/core.py` | Before listing images |
| `create_version(fork_from_version_id=...)` | `services/projects/versions.py:407-498` | Called proactively once, on fork (ensures v2/v3 have a manifest from the start, avoiding lazy inconsistency) |

**Rebuild rules** (pseudocode):

```python
def ensure_train_manifest(project_dir: Path, version_label: str) -> Path:
    """If versions/{label}/train/manifest.json is missing, implicitly rebuild
    it from the old project-level preprocess manifest. Zero user-visible
    impact, idempotent.

    Rules (D12):
      1. Target already exists -> return its path directly (O(1) stat check)
      2. Target missing + old preprocess/manifest.json missing -> write an
         empty manifest and return
      3. Target missing + old manifest exists -> rebuild by matching train/'s
         actual filenames against the old entries' origin
    """
    train_dir = project_dir / "versions" / version_label / "train"
    target = train_dir / "manifest.json"
    if target.exists():
        return target

    train_dir.mkdir(parents=True, exist_ok=True)
    new_data = {"version": 2, "images": {}}

    legacy_manifest = project_dir / "preprocess" / "manifest.json"
    if legacy_manifest.exists():
        try:
            legacy = json.loads(legacy_manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            legacy = {"images": {}}

        train_files = {
            f.name for f in train_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        }

        for name, entry in legacy.get("images", {}).items():
            # Only rebuild entries for images that actually exist in train/
            if name not in train_files:
                continue
            new_data["images"][name] = {
                "origin": entry.get("origin") or entry.get("source") or name,
                "mtime": entry.get("mtime", 0),
                "size": entry.get("size", 0),
            }

    # Atomic write: tmp + rename
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new_data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
    return target
```

**Correctness argument**:

- **An image exists in train/ but the old manifest never recorded it** (the user
  dragged it into train/ manually) → no entry is written. Afterward, such images
  are treated as `origin = name` (restore looks it up under `download/{name}`;
  fails if not found) — **this is the intended fallback**, consistent with D7.
- **An image is in the old manifest but not in train/** (the user didn't select
  it during curation) → no entry is written. Correct — under the new model this
  image shouldn't be in the train scope at all.
- **Multi-crop derivatives** (`Y_c0.png` / `Y_c1.png` sharing origin `Y.jpg`) →
  the old manifest has two entries; each is checked against train/ and written
  in independently.

### 3.3 `restore(name)` semantics (D7 / D8)

```python
def restore(project_dir, version_label, name):
    """Restore train/{name} to a fresh copy of download/{entry.origin}."""
    ensure_train_manifest(project_dir, version_label)
    manifest = load_manifest(project_dir, version_label)
    entry = manifest["images"].get(name)
    if entry is None:
        raise RestoreError(f"No manifest entry for {name}")
    origin = entry["origin"]
    src = project_dir / "download" / origin
    if not src.exists():
        raise RestoreError(
            f"Original download/{origin} not found — cannot restore. "
            "Likely deleted from disk."
        )
    dst = project_dir / "versions" / version_label / "train" / name
    shutil.copy2(src, dst)
    # Update the manifest entry: origin unchanged, but mtime/size sync to the
    # download file's values
    manifest["images"][name] = {
        "origin": origin,
        "mtime": int(src.stat().st_mtime),
        "size": src.stat().st_size,
    }
    save_manifest(project_dir, version_label, manifest)
```

**Failure UX** (D8): the frontend receives `RestoreError` → shows a toast "N
images could not be restored: the originals have been deleted from download/,"
listing the specific origin filenames, with three options
[drag in a replacement / keep the processed version / remove from train/].

### 3.4 Fork-version behavior (D6)

The existing `create_version(fork_from_version_id=...)`
(`studio/services/projects/versions.py:407-498`) calls `_copytree("train")` to
recursively copy the train subtree.

**New approach**:

- `train/manifest.json`, being part of that recursive copy target, **rides
  along automatically** (0 code changes needed)
- Also call `ensure_train_manifest(project_dir, new_version_label)`
  (defensive — in case the source manifest was corrupt, it gets rebuilt)
- v2 starts out with a train/ identical to v1, so all preprocess output is
  inherited → the preprocessing phase auto-skips (the sidebar shows it as
  already complete)

### 3.5 Phase-advance behavior

Following the ADR 0007 §11.5-A pattern:

- Current phase = curating → validate train/ has ≥ 1 image → advance to
  preprocessing
- Current phase = preprocessing → validate no preprocess job is
  pending/running → advance (skippable) to tagging (doesn't require having
  processed any image)
- Current phase = tagging → validate caption coverage is 100% → advance to
  editing
- The rest is unchanged

---

## 4. Phase state-machine changes

### 4.1 `VersionPhase` enum (`studio/services/projects/versions.py:41-58`)

**Before**:

```python
class VersionPhase:
    CURATING = "curating"
    TAGGING = "tagging"
    EDITING = "editing"
    REGULARIZING = "regularizing"
    READY = "ready"
    ORDER = (CURATING, TAGGING, EDITING, REGULARIZING, READY)
    VALUES = frozenset(ORDER)
    SKIPPABLE = frozenset({REGULARIZING})
```

**After**:

```python
class VersionPhase:
    CURATING = "curating"
    PREPROCESSING = "preprocessing"    # <- new
    TAGGING = "tagging"
    EDITING = "editing"
    REGULARIZING = "regularizing"
    READY = "ready"
    ORDER = (CURATING, PREPROCESSING, TAGGING, EDITING, REGULARIZING, READY)  # 6 total
    VALUES = frozenset(ORDER)
    SKIPPABLE = frozenset({PREPROCESSING, REGULARIZING})  # PREPROCESSING added
```

### 4.2 `check_preprocessing` (`studio/services/projects/phase.py`)

New function, same pattern as `check_regularizing`:

```python
def check_preprocessing(conn: sqlite3.Connection, version_id: int) -> CheckResult:
    """No preprocess job may be pending/running (D16; skippable, doesn't
    strictly require completeness)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM project_jobs "
        "WHERE version_id = ? AND kind = 'preprocess' "
        "  AND status IN ('pending', 'running')",
        (version_id,),
    ).fetchone()
    if int(row[0]) > 0:
        return CheckResult(False, "A preprocess task is running, please wait for it to finish")
    return CheckResult(True)
```

The `check_phase` dispatcher gets a new branch:
`if phase == P.PREPROCESSING: return check_preprocessing(conn, version_id)`.

### 4.3 `_v11_preprocessing_phase` migration

File: `studio/infrastructure/migrations/_v11_preprocessing_phase.py`

**Backfill rule** (D19):

| Existing phase | train/ state | New phase |
|---|---|---|
| `curating` | empty | `curating` (unchanged) |
| `curating` | non-empty | **`preprocessing`** (per the user: train/ already holds processed images) |
| `tagging` | * | `tagging` (unchanged) |
| `editing` | * | `editing` (unchanged) |
| `regularizing` | * | `regularizing` (unchanged) |
| `ready` | * | `ready` (unchanged) |

**Pseudocode**:

```python
"""v10 -> v11: ADR-0009 adds the preprocessing phase.

VersionPhase grows from 5 to 6 values, with preprocessing inserted between
curating and tagging.

Backfill strategy (all silent, same add-only pattern as _v8):
- phase=curating + train/ non-empty -> advance to preprocessing
- phase=curating + train/ empty -> stays curating
- other phases untouched
"""
from __future__ import annotations
import sqlite3
from pathlib import Path


def migrate(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT v.id, v.label, p.id AS project_id, p.dir_path "
        "FROM versions v JOIN projects p ON v.project_id = p.id "
        "WHERE v.phase = 'curating'"
    ).fetchall()
    for vid, label, _, project_dir in rows:
        train_dir = Path(project_dir) / "versions" / label / "train"
        if not train_dir.exists():
            continue
        has_files = any(
            f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            for f in train_dir.iterdir()
        )
        if has_files:
            conn.execute(
                "UPDATE versions SET phase = 'preprocessing' WHERE id = ?",
                (vid,),
            )
    conn.commit()
```

**Test cases** (`tests/test_migration_v11.py`):

- v=curating + train/ empty → unchanged
- v=curating + train/ has a .png → preprocessing
- v=curating + train/ has a .txt but no image → unchanged
- v=tagging + any train/ → tagging (unchanged)
- v=ready + any train/ → ready (unchanged)

### 4.4 Frontend phase enum sync

In `studio/web/src/api/client.ts`, the `PHASE_ORDER` / `PHASE_SKIPPABLE` /
`VersionPhase` type:

```typescript
export type VersionPhase =
  | 'curating'
  | 'preprocessing'   // new
  | 'tagging'
  | 'editing'
  | 'regularizing'
  | 'ready'

export const PHASE_ORDER: VersionPhase[] = [
  'curating',
  'preprocessing',    // new
  'tagging',
  'editing',
  'regularizing',
  'ready',
]

export const PHASE_SKIPPABLE: VersionPhase[] = ['preprocessing', 'regularizing']  // preprocessing added
```

---

## 5. Module change checklist

### 5.1 Backend: removals

| File:line | Function / block | Reason for removal |
|---|---|---|
| `manifest.py:132-152` | `resolve()` dual-bucket fallback | Under the new model, train/ is self-contained, no fallback needed |
| `manifest.py:155-180` | `resolve_origin()` (reverse-lookup download -> derivative) | train/ is no longer a derivation source, reverse-lookup is meaningless |
| `manifest.py:443-460` | `ensure_manifest()` legacy sidecar migration + `_scan_legacy_sidecars` | D22: legacy sidecars no longer supported |
| `core.py:106-151` | `list_pending()` | The "unprocessed / processed" binary concept goes away — only physical files in train/ matter |
| `curation.py:109-163` | multi-crop fan-out row expansion in `list_download()` | download is a read-only snapshot, no derivatives shown there |

### 5.2 Backend: modified (gain a `version_label` parameter)

| File:line | Current signature | New signature | Notes |
|---|---|---|---|
| `manifest.py:70-71` | `manifest_path(project_dir)` | `manifest_path(project_dir, version_label)` → `versions/{label}/train/manifest.json` | Path changes |
| `manifest.py:62` | global `_LOCK = threading.Lock()` | `_LOCKS: dict[str, Lock]` (keyed by `(pid, vid)`) | Prevents cross-version lock contention |
| `manifest.py:194-410` | `add_processed / restore / mark_duplicate_removed / replace_with_crops / clear_all / all_processed / duplicate_removed_origins` | all gain a `version_label` parameter | |
| `manifest.py` | new `ensure_train_manifest(project_dir, version_label) → Path` | see §3.2 |
| `core.py:154-219` | `list_processed(project)` | `list_train_images(project, version_label)`: scans train/'s physical files + joins in manifest entry metadata | |
| `core.py:260-300` | `resolve_targets(project, ...)` | gains `version_label`, lists names from train_dir | |
| `core.py:353-466` | `list_crop_workspace / list_duplicate_removed_workspace` | gains `version_label` | |
| `duplicates.py:430-451` | `_resolve_download_sources` | `_resolve_train_sources(version_label)`, scoped to train/ | D9 |
| `curation.py:210-275` | `copy_to_train` dual branch | single branch: `shutil.copy2(download_dir / name, train_dir / name)` + writes manifest entry `{origin: name}` | D28 |
| `preprocess_worker.py:189-216` | `add_processed` writes to `preprocess/{name}.png` | writes to `train/{name}.png` + removes the old `train/{origin}` file (if any) | |
| `preprocess_worker.py:241-265` | `_resolve_crop_source` source path | source = `train/{folder}/{name}` | |
| `preprocess_worker.py:351-399` | crop writes to `preprocess/{name}.png` | writes to `train/{name}.png` | |

### 5.3 Backend: additions

- `studio/infrastructure/migrations/_v11_preprocessing_phase.py` (§4.3)
- `studio/services/projects/phase.py:check_preprocessing` (§4.2)
- `studio/services/preprocess/manifest.py:ensure_train_manifest` (§3.2)
- Registration of _v11 in `studio/infrastructure/migrations/__init__.py`

### 5.4 Frontend: changes

| File:line | Change |
|---|---|
| `studio/web/src/api/client.ts` | `VersionPhase` gains `'preprocessing'`; `PHASE_ORDER` gains it; `PHASE_SKIPPABLE` gains it |
| `studio/web/src/components/Sidebar.tsx:11-29` | `STEP_KEY_TO_PHASE` gains `preprocess: 'preprocessing'`; `PHASE_TO_STEP_KEY` gains `preprocessing: 'preprocess'` |
| `Sidebar.tsx:266-274` | `STEPS` order changes: `preprocess` moves from `idx=''` (project scope) to `idx: '2'` (version scope), positioned after curate; scope becomes `'version'` |
| `Sidebar.tsx:266-274` | old idx: curate '1' / tag '2' / edit '3' / reg '4' / train '5' → new: curate '1' / preprocess '2' / tag '3' / edit '4' / reg '5' / train '6' |
| `Sidebar.tsx:472` | `projectScopeStep` regex `/^\/projects\/[^/]+\/(download|preprocess)$/` → `/^\/projects\/[^/]+\/download$/` (preprocess is no longer project scope) |
| `studio/web/src/i18n/locales/en.json:15` | `"preprocess": "Preprocess Tools"` → `"preprocess": "Preprocess (optional)"` (D17, same pattern as `reg: "Regularization set (optional)"`) |
| `studio/web/src/i18n/locales/en.json:149` | `"preprocess": "② Preprocess"` → removed (no longer project scope) or redirected to the version step's idx |
| corresponding spot in `studio/web/src/i18n/locales/ru.json` | `"preprocess": "Preprocess (optional)"` |
| `studio/web/src/pages/project/steps/Preprocess.tsx` (905 lines) | data source switches to train/; removes the pending/processed dual concept; becomes a single "train view" grid + status badges |
| `PreprocessOverview.tsx` (316 lines) | stats metrics rescoped to train/, adds an ARB bucket distribution card |
| `PreprocessCrop.tsx` (982 lines) | all source paths switch to train/ |
| `PreprocessDuplicates.tsx` (778 lines) | scoped to train/ |
| `PreprocessHub.tsx` (31 lines) | adds `useActiveVersion` to get the vid, passes it down to sub-routes |
| Curation.tsx | adds a "Send selection to preprocess ->" button at the top (minor, can go into a PR-4 follow-up) |

### 5.5 i18n copy changes summary

`en.json`:
```diff
-  "preprocess": "Preprocess Tools",
+  "preprocess": "Preprocess (optional)",
```

`ru.json` gets the equivalent change.

`en.json:149-153` (Sidebar idx-numbered section), current state:

```
"download": "① Dataset",
"preprocess": "② Preprocess",
"curate": "③ Curate",
...
```

changes to:

```
"download": "① Dataset",
"curate": "② Curate",
"preprocess": "③ Preprocess (optional)",
"tag": "④ Tag",
"edit": "⑤ Tag editing",
"reg": "⑥ Regularization set (optional)",
"train": "⑦ Train",
```

---

## 6. API endpoint changes

11 endpoint paths change from `pid` to `(pid, vid)` (D20 / D21: **no redirect
compat layer** — backend PR-3 and frontend PR-4 switch together):

| Old URL | New URL | File:line |
|---|---|---|
| `POST /api/projects/{pid}/preprocess/start` | `POST /api/projects/{pid}/versions/{vid}/preprocess/start` | `ingestion.py:222` |
| `GET /api/projects/{pid}/preprocess/status` | `GET /api/projects/{pid}/versions/{vid}/preprocess/status` | `ingestion.py:275` |
| `GET /api/projects/{pid}/preprocess/files` | `GET /api/projects/{pid}/versions/{vid}/preprocess/files` | `ingestion.py:301` |
| `GET /api/projects/{pid}/preprocess/duplicates/removed` | `GET /api/projects/{pid}/versions/{vid}/preprocess/duplicates/removed` | `ingestion.py:315` |
| `GET /api/projects/{pid}/preprocess/crop/workspace` | `GET /api/projects/{pid}/versions/{vid}/preprocess/crop/workspace` | `ingestion.py:331` |
| `POST /api/projects/{pid}/preprocess/crop` | `POST /api/projects/{pid}/versions/{vid}/preprocess/crop` | `ingestion.py:348` |
| `POST /api/projects/{pid}/preprocess/files/reset` | `POST /api/projects/{pid}/versions/{vid}/preprocess/files/reset` | `ingestion.py:380` |
| `POST /api/projects/{pid}/preprocess/files/restore` | `POST /api/projects/{pid}/versions/{vid}/preprocess/files/restore` | `ingestion.py:397` |
| `GET /api/projects/{pid}/preprocess/thumb` | `GET /api/projects/{pid}/versions/{vid}/preprocess/thumb` | `ingestion.py:421` |
| `POST /api/projects/{pid}/preprocess/duplicates/scan` | `POST /api/projects/{pid}/versions/{vid}/preprocess/duplicates/scan` | `curation.py:271` |
| `POST /api/projects/{pid}/preprocess/duplicates/apply` | `POST /api/projects/{pid}/versions/{vid}/preprocess/duplicates/apply` | `curation.py:342` |

**Also**: the thumb endpoint's (`curation.py:117-186`) `bucket` param semantics:

- `bucket=download` kept (used by the curation page)
- `bucket=preprocess` removed (project-level preprocess no longer exists)
- new `bucket=train` (thumbnails fetched from train/)

---

## 7. UI changes

### 7.1 Sidebar stepper order (D18)

```
Current: ① Dataset -> ② Preprocess (project) -> ③ Curate -> ④ Tag -> ⑤ Edit -> ⑥ Reg (optional) -> ⑦ Train

New:     ① Dataset -> ② Curate -> ③ Preprocess (optional, version) -> ④ Tag -> ⑤ Edit -> ⑥ Reg (optional) -> ⑦ Train
```

Phase-cursor mapping: preprocess step idx '③' -> version phase `preprocessing`
(version scope).

The optional step's visual treatment matches reg exactly (the i18n label adds
"(optional)," no other special styling).

### 7.2 Preprocess page grid

Data source = the entirety of `versions/{label}/train/` + its co-located
manifest.json.

Each image card:

```
+--------------+
|  [thumbnail] |   X.png  (origin: X.jpg)
|              |   2048x2048 . 4MB
|   [check] upscale |
|   [check] crop    |
+--------------+
```

Badge logic (based on entry field differences, the inferred-state approach from
§2.2):

- `name.suffix != origin.suffix` -> shows `[check] upscale` (extension change =
  processed)
- `"_c" in name` -> shows a `[check] crop` derivative marker
- `mtime`/`size` mismatch against the physical train/{name} file -> shows
  `[warn] modified externally`

Top stats bar (D10 / D11): `N images . avg WxH . ARB bucket distribution bar
chart`.

### 7.3 Error-recovery UX (D8 detailed)

`restore` failure toast (frontend, the restore button's failure handling in
`PreprocessOverview.tsx`):

```
[x] 3 images could not be restored
-------------------------
Originals deleted from download/:
- X.jpg
- Y.jpg
- Z.jpg

[Drag in a replacement] [Keep the processed version] [Remove from training set]
```

The three options' semantics:

- **Drag in a replacement**: opens a file picker, the user selects a local
  image to overwrite `download/{origin}`, then retries restore
- **Keep the processed version**: ignores the failure, train/{name} stays as
  is (manifest entry unchanged)
- **Remove from training set**: deletes train/{name} + the manifest entry
  (irreversible, requires a second confirmation dialog)

---

## 8. Test plan

### 8.1 Unit-test focus (PR-1 + PR-2)

| Test file | Contents |
|---|---|
| `tests/test_train_manifest_fallback.py` (**new**) | 8 cases for `ensure_train_manifest`: target already exists / old manifest missing / old manifest corrupt / multi-crop derivative matching / some train/ images have no entry / repeated calls are idempotent / concurrency safety / `_LOCKS` per-version isolation |
| `tests/test_preprocess_manifest.py` | remove ~60% of old cases (v0/v1 read-compat / `resolve` / `resolve_origin` / `_scan_legacy_sidecars`); modify ~20% to add the `version_label` parameter; add ~20% new cases for the v2 schema |
| `tests/test_preprocess.py` | remove `list_pending` tests (~80 lines); modify `list_train_images` to add vid; keep worker behavior tests |
| `tests/test_preprocess_crop_worker.py` | change source-path constants to train/; keep multi-crop fan-out behavior tests |
| `tests/test_preprocess_endpoints.py` | change URL pattern + add the vid path param; test the thumb endpoint's bucket=train |
| `tests/test_curation.py` | remove `test_copy_to_train_uses_processed_bytes_when_available:158-168` (the dual branch is merged); remove `test_curation_view_expands_multi_crop_derivatives:183-204` (fan-out row expansion removed) |
| `tests/test_curation_endpoints.py` | URL pattern changes |
| `tests/test_migration_v11.py` (**new**) | 5 backfill cases for _v11 (listed at the end of §4.3) |
| `tests/test_phase.py` | adds `check_preprocessing` tests: job pending -> fails; no job -> passes; tests that ORDER includes preprocessing; tests that SKIPPABLE includes preprocessing |

### 8.2 Integration tests

- e2e: create project -> import via download -> create version -> curate a
  selection -> preprocess upscale -> tag -> train (confirm phase advances
  correctly)
- fork: v1 runs fully to ready -> fork v2 -> confirm v2's train/manifest.json
  is auto-copied + phase follows
- legacy-project compatibility: spin up a fixture project (with an old
  `preprocess/manifest.json` + images already in train/) -> start the server
  -> first access -> confirm `ensure_train_manifest` implicitly rebuilds
  successfully

### 8.3 Frontend tests

vitest currently has no Preprocess unit tests (confirmed by grep). PR-4 does
not require adding vitest coverage — relying on e2e instead.

---

## 9. PR slicing + sequencing (D29 / D30)

### 9.1 PR breakdown

| # | PR title | Scope | Effort | Blocking dependency |
|---|---|---|---|---|
| **PR-1** | `feat(preprocess): add ensure_train_manifest fallback + ADR 0010 draft` | ADR 0010 draft (supersedes 0004) + `manifest.py:ensure_train_manifest` + `tests/test_train_manifest_fallback.py` | **0.5d** | none (independent) |
| **PR-2** | `refactor(preprocess): move manifest scope from project to version train/` | `manifest.py` slimmed down (removals §5.1 + changes §5.2) + `core.py` / `duplicates.py` / `preprocess_worker.py` / `curation.py:copy_to_train` changes + test updates | **3.0d** | the lifecycle PR chain + `feat/preprocess-multitool` merged to dev; PR-1 merged to dev |
| **PR-3** | `feat(lifecycle): add preprocessing phase + _v11 migration + endpoint vid routing` | `VersionPhase.ORDER` gains PREPROCESSING + `SKIPPABLE` gains it + `check_preprocessing` + `_v11_preprocessing_phase.py` + 11 API endpoints gain the vid path | **1.5d** | PR-2 merged to dev |
| **PR-4** | `feat(ui): preprocess as version phase + sidebar reorder + train-scope grid` | frontend Preprocess hub + 4 sub-pages wired to vid context + Sidebar STEPS reorder + i18n "(optional)" + Preprocess.tsx data contract | **2.5d** | PR-3 merged to dev |
| | i18n review / a11y check (parallel with PR-4) | | +0.5d | |
| | **Total** | | **8.0d** | |

### 9.2 Sequencing diagram

```
[Now]
  |-- PR-1 (0.5d) -- can start immediately
  |
  `-- waiting on the in-flight chain to merge to dev:
       feat/lifecycle-v8-schema (lifecycle PR-2)
       `-- feat/lifecycle-business-logic (lifecycle PR-3/4)
            `-- feat/lifecycle-v9-destructive (lifecycle PR-5)
                 `-- feat/lifecycle-frontend-v3 (lifecycle PR-6)
                      `-- feat/preprocess-multitool
                           `-- fix/preprocess-stage-and-crop-thumb
                                |
                                `-- PR-2 (3.0d)
                                     `-- PR-3 (1.5d)
                                          `-- PR-4 (2.5d)
                                               -> MERGE
```

### 9.3 Parallelization opportunities

- PR-1 is 100% parallel with the lifecycle PR chain (no conflicts)
- PR-2 / PR-3 / PR-4 are sequential (schema change -> migration + API -> frontend)
- While backend PR-2 is in review, PR-3's _v11 migration + check_preprocessing
  code can be **drafted** in parallel (doesn't depend on PR-2 being merged —
  just needs to be developed on a branch based on PR-2's commits)

---

## 10. ADR coordination

### 10.1 Drafting ADR 0010 (within PR-1's scope)

**Key sections** (see the standalone ADR file for full detail; this plan's
§10 only lists the outline):

- Status: Proposed
- Supersedes: ADR 0004 (including Addendum 1)
- **Numbering note**: 0009 is skipped (already taken by "unified logging +
  error system," see `docs/adr/README.md`)
- Background: four user pain points (lead time / meaningless stats /
  clustering breakdown / booru-material characteristics)
- Candidate approaches: A. keep current state / B. UI filter / **C. this
  proposal**
- Decision: approach C (preprocess scope -> train set + per-version manifest
  + fallback rebuild)
- Data model (§2 content)
- Compatibility strategy (§3.2 ensure_train_manifest)
- Relationship to ADR 0007: amendment adding the PREPROCESSING phase
- Relationship to ADR 0008: module boundaries unchanged
- Consequences / new tech debt

### 10.2 ADR 0004 changes (only once ADR 0010 is accepted)

**While ADR 0010 is Proposed, 0004's status stays unchanged** — both the
README index and the ADR 0004 file itself remain as-is, with only a README
hint added: "will become Superseded once ADR 0010 is accepted." The following
changes happen once ADR 0010 is accepted (when PR-2 or PR-3 merges to dev):

The file's status line changes to:

```markdown
**Status**: Superseded by [ADR 0010](0010-preprocess-train-scope.md) (YYYY-MM-DD)
```

The body is not deleted (kept as a historical decision record), but gets a
banner added:

> **Note**: ADR 0010 relocates preprocess scope from the project level down
> to the version-level train/. This ADR's schema / resolver / project-level
> manifest design is **deprecated**; legacy projects are implicitly rebuilt
> into the new model via `ensure_train_manifest()`. This ADR is kept as a
> historical decision record.

### 10.3 ADR 0007 amendment

`docs/adr/0007-project-version-lifecycle-refactor.md` gains an addendum:

```markdown
## Addendum 1 -- add the `preprocessing` phase (2026-06-03, companion to ADR 0010)

ADR 0010 relocates preprocess scope down to the version-level train/, adding
it as a new optional phase in the cursor.

VersionPhase.ORDER becomes 6 values:

```
curating -> preprocessing -> tagging -> editing -> regularizing -> ready
```

VersionPhase.SKIPPABLE gains `preprocessing` (same mindset as `regularizing`).

Phase check `check_preprocessing`: passes as long as no preprocess job is
pending/running (doesn't require having processed any image).

§70 "dataset ownership decided at the version level" is not being overturned
-- the train set's source of truth is still the project-level `download/`
pool (the curating phase copies from download into train). This amendment
only makes preprocess output ride along with train at the version level; it
does not reverse the dataset-ownership decision.

See ADR 0010 for details.
```

### 10.4 ADR 0008 / 0009 unchanged

`services/preprocess/` as a module still exists (the core upscale / crop /
blur / dedup worker logic is unchanged) — only the state storage location
moves from project level to version/train/. Module boundaries are unchanged.

---

## 11. Risk list

| Risk | Category | Impact | Mitigation |
|---|---|---|---|
| `ensure_train_manifest` rebuild logic is wrong (misses images / wrong origin) | Medium | User restores the wrong image | 8 unit-test cases + integration tests against a real legacy-project fixture |
| _v11 migration wrongly advances phase=curating + empty train/ to preprocessing | Low | UX: user enters the version and finds curating skipped | strict `any(train_dir.iterdir())` check + unit tests |
| Old preprocess/ directory occupies disk space long-term | Low | User storage pressure | 0.13.0 release notes mention it can be deleted manually; not actively deleted |
| Concurrent multi-version manifest writes deadlock | Low | Manifest corruption | `_LOCKS: dict[(pid, vid), Lock]` per-version isolation + atomic tmp+rename writes |
| Multi-crop fan-out cross-version sync issue after fork | Low | v2 crop changes shouldn't affect v1 | test case: crop in v1, then fork v2, changing crop in v2 doesn't write back to v1 |
| The 11 renamed endpoint URLs 404 for old frontend / third-party callers | Medium | User updates to a new version + browser cache of old JS -> errors | beta mindset + frontend switches in the same PR; frontend cache busted via hash |
| A one-off i18n copy change gets missed somewhere | Low | Inconsistent copy | grep all i18n keys and batch-fix |
| Ordering issue between ADR 0007 PR-7's destructive `_v9` and _v11 | Low | _v11 assumes the phase column exists | _v11 must run after _v9; enforced by `migrations/__init__.py` ordering |

---

## 12. Open questions (already decided / notes for the implementer)

### 12.1 Already decided (D1-D30 in full; this plan is ground truth)

### 12.2 Small decisions for the implementer to keep in mind during the PRs

1. **`_LOCKS` implementation detail**: `WeakValueDictionary[(pid, vid), Lock]`
   or a plain dict? The former is GC-friendly, the latter simpler.
   **Lead reviewer recommends a plain dict** (version count << 1000, memory
   pressure negligible).
2. **Should `ensure_train_manifest` calls short-circuit**: the first call
   already ensures it; should subsequent calls cache the stat result?
   **Recommend not caching** (an O(1) stat is negligible cost, and caching
   introduces invalidation problems instead).
3. **Manifest copying on fork**: `_copytree("train")` already recursively
   copies it; should an **explicit** `ensure_train_manifest(new_label)` call
   also be made after fork as a fallback (in case the source manifest was
   corrupt)? **Recommend calling it explicitly** (defensive, near-zero cost).
4. **i18n idx numbering**: the current `en.json:149-153` embeds `① ② ③ ④ ⑤`
   numbering directly in the i18n values. If the implementer decides to
   switch to "label without a number + Sidebar renders the number
   separately," that's an independent small refactor, **out of scope for
   this plan**. Keep the inline numbering as-is for now.

### 12.3 Coordination with other in-flight work

- `feat/preprocess-multitool` has already split Preprocess into a hub + 4
  sub-page structure -> PR-4 reuses it fully, no redesign needed
- `feat/lifecycle-frontend-v3` PR-6 lands the phase header -> PR-4 adding a
  preprocessing step to the phase header should be trivial (same pattern as
  reg)
- `feat/lifecycle-v9-destructive` removes the `stage` field -> must merge to
  dev before _v11 (_v11 assumes the stage column is already gone and only
  touches the phase column)

---

## 13. References

### Key ADRs / code
- `docs/adr/0004-preprocess-manifest.md` (with Addendum 1) — pending supersede
- `docs/adr/0007-project-version-lifecycle-refactor.md` — gains Addendum 1 (PREPROCESSING phase)
- `docs/adr/0010-preprocess-train-scope.md` — pending draft (PR-1 scope)
- `studio/services/preprocess/manifest.py` — main change target (slimmed down + `ensure_train_manifest` added)
- `studio/services/preprocess/core.py:106-219` — `list_pending` removed; `list_processed` changed
- `studio/services/preprocess/duplicates.py:430-451` — scope changed to train/
- `studio/services/dataset/curation.py:210-275` — `copy_to_train` drastically simplified
- `studio/workers/preprocess_worker.py:189-216,241-265,351-399` — write paths changed to train/
- `studio/services/projects/versions.py:41-58` — `VersionPhase.ORDER` / `SKIPPABLE` gain a new value
- `studio/services/projects/versions.py:407-498` — `create_version` fork flow (adds `ensure_train_manifest` fallback)
- `studio/services/projects/phase.py` — adds `check_preprocessing`
- `studio/infrastructure/migrations/_v11_preprocessing_phase.py` — new file
- `studio/api/routers/projects/ingestion.py:217-441` — 11 endpoint URLs changed
- `studio/api/routers/projects/curation.py:117-186,271,342` — thumb endpoint bucket=train + 2 duplicates endpoint URLs
- `studio/web/src/components/Sidebar.tsx:11-29,266-274,472` — STEP order / phase mapping / project-scope regex
- `studio/web/src/api/client.ts` — `VersionPhase` / `PHASE_ORDER` / `PHASE_SKIPPABLE`
- `studio/web/src/i18n/locales/{ru,en}.json` — "(optional)" copy + idx numbering
- `studio/web/src/pages/project/steps/Preprocess*.tsx` — data contract + 4 sub-pages
