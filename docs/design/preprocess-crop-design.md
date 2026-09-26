# Preprocess - Crop — feature design

> A working design document laying out the **logic model + user scenarios +
> data contract**. See the follow-up PRs for implementation details.
>
> Several rounds of UI iteration happened after this landed; see
> [§9 Addendum 1](#addendum-1--ui-evolution-2026-05-21).
>
> **2026-06-04 status update**: [ADR 0010](../adr/0010-preprocess-train-scope.md)
> moved preprocessing down to version-level `versions/{label}/train/` as a
> whole. Every `preprocess/` reference in this document should now be read as
> `versions/{label}/train/{folder}/`, with the manifest living at
> `versions/{label}/train/manifest.json` (entry keys are POSIX relative paths
> like `"folder/file"`). The crop logic model (multi-crop fan-out / naming
> rules / restore collapsing to origin) is unchanged — only the scope has
> narrowed. "Restore" semantics changed to "copy from `download/{origin}`
> over `train/{folder}/{origin}` + clear siblings" (see ADR 0010's §Restore
> semantics for details).

## 0. Purpose and non-goals

**Purpose**: add a "crop" stage after preprocessing's "upscale" step, letting
the user cut up the `preprocess/` working set either manually or via smart
clustering.

**Non-goals**:
- No third working directory is introduced. `download/` remains the sole
  backup, and `preprocess/` remains the sole working set.
- No stage ordering is enforced (upscale → crop → upscale → crop are all
  valid).
- No partial undo — restore always goes back to download.
- No persisting crop process information (rect coordinates, target AR,
  cluster id) — once written to disk, process info is dropped.

---

## 1. Data model

### Folders

| Directory | Role | Mutable? |
|---|---|---|
| `download/` | The sole backup of original images | **never touched** |
| `preprocess/` | The current working set, overwritten in place by each stage | mutable |

### File naming

- Default 1:1: `preprocess/X.png` corresponds to `download/X.png`
- Multi-crop derivatives: `preprocess/X_c0.png`, `preprocess/X_c1.png`, both
  with origin pointing to `download/X.png`
- Repeated cropping: cropping `X_c0.png` again into multiple pieces →
  `X_c0_c0.png` / `X_c0_c1.png` (origin is still `X.png`)

### Manifest schema (new version)

```jsonc
{
  "images": {
    "X.png":     { "origin": "X.png",  "mtime": 1731000000, "size": 1234567 },
    "Y_c0.png":  { "origin": "Y.png",  "mtime": ...,        "size": ...     },
    "Y_c1.png":  { "origin": "Y.png",  "mtime": ...,        "size": ...     }
  }
}
```

Only `{origin, mtime, size}` is recorded. `kind / model / scale / action /
target_area / src_size / dst_size / elapsed_seconds` are **not** recorded
(these are all "process information" that gets dropped once written to
disk).

### Old-schema compatibility

ADR 0004's `{kind, model, scale, action, target_area, src_size, dst_size,
elapsed_seconds, source}` fields are deprecated (ADR 0010 PR-5 removed the
reader compatibility code):
- A missing `origin` falls back to the entry key itself; the `source` field
  is **no longer** read
- Only `kind == "duplicate_removed"` still means anything, as a tombstone;
  other `kind` values no longer branch anywhere

### resolve(download_name)

Downstream (curation / thumbnail / copy_to_train):

- If an entry exists in the manifest with origin = `download_name` → returns
  those preprocess/ files (possibly several)
- Otherwise → returns `download/{download_name}`

### Restore

Deletes the preprocess files + every entry with `origin == download_name`.
Downstream falls back to the download original. No single-stage undo is
performed.

---

## 2. User cases

| # | Scenario | Mode | Output |
|---|---|---|---|
| U1 | A full-body image, keeping both a headshot and the full body | manual / free AR / multiple boxes | `X_c0.png` (headshot) + `X_c1.png` (full body) |
| U2 | Training buckets unified to 1:1 / 2:3 | manual / locked AR / single box | `X.png` (overwritten) |
| U3 | Dataset ARs are all over the place, wants to bucket them | smart clustering → fine-tune | each image gets `X.png` (overwritten) |
| U4 | Custom ratio 5:7 | manual / custom W:H | same as U2 |
| U5 | An image passes through untouched | draw no box | preprocess file stays as-is / after restore, uses download |
| U6 | Back to the original | restore | deletes the preprocess entry + file |

---

## 3. Features

> **Version note**: the v1 design used a segmented tab for "manual /
> clustering"; after shipping, this iterated away from tabs — cropping is a
> single concept, and "smart clustering" is an **optional pre-fill tool**
> rather than a separate mode. See §9 Addendum 1 for details.

### Core cropping capability

**AR dropdown**: `Free (unlocked)` / `1:1` / `4:3` / `3:2` / `16:9` / `3:4` /
`2:3` / `9:16` / `4:5` / `Custom…`

- Free → drag to draw a box of any AR
- Locked → a new box follows the AR; resize handles scale proportionally;
  moving doesn't affect AR
- Custom → pops two numeric inputs, W and H
- One image, multiple crops: N boxes → N outputs

**Canvas interaction**: 8 handles (4 corners + 4 edges) + a rule-of-thirds
grid + dimmed area outside the box + a live pixel-size / AR readout.

**Right-side rect list**: thumbnail + editable label + output pixel size +
copy/delete (a header icon appears while selected).

**Filter chips**: All / To crop / Cropped (filtered by the in-session
`cropsByImage` state).

**Primary actions**: `Crop current image` / `▶ Crop all (N)`.

### Smart clustering (optional pre-fill)

An independent section below the OperationPanel, **collapsed by default**;
clicking `▸ Smart clustering` expands it.

**Parameters**: `max_crop ∈ [0, 0.30]` (maximum allowed cropped-area
fraction), `k_min ∈ [1, 10]`, `k_max ∈ [2, 15]`.

**Algorithm (frontend JS)**:

1. Compute AR = w/h for every image in `preprocess/`
2. 1-D k-means over `[k_min, k_max]`, picking k via the elbow method
3. Snap each cluster center → to the training bucket grid (see §7 ARB alignment)
4. The displayed label uses the nearest common pretty AR; the rect is
   computed from the training bucket AR
5. Each cluster member is center-cropped to the largest rectangle that fits
   the target AR
6. `max_crop` constraint: if the cropped-away area fraction exceeds
   max_crop, no box is added (the user can handle it manually)

**Result**: written into cropsByImage (each image gets 1 box marked with a
✦ as coming from a cluster). After clustering, the user can freely tweak /
delete / add boxes using the same main canvas.

**Primary action**: `▶ Start clustering` — a button local to the section,
**not** a submit-to-disk action. Submission still goes through the outer
`Crop all` button.

---

## 4. Backend contract

### New endpoint (all moved down to version scope after ADR 0010)

```
POST /api/projects/:id/versions/:vid/preprocess/crop
body: {
  crops: {
    "1_data/IMG_2741.png": [
      { x: 0.10, y: 0.05, w: 0.55, h: 0.45, label: "headshot" },
      { x: 0.12, y: 0.42, w: 0.72, h: 0.55, label: "full body" }
    ],
    "1_data/IMG_2742.png": [ { x: 0.25, y: 0.12, w: 0.50, h: 0.78, label: "" } ]
  }
} → Job
```

Helper endpoints:

- `GET /api/projects/:id/versions/:vid/preprocess/crop/workspace` — lists
  every `train/{folder}/{image}` plus pixel dimensions and a processed
  marker, used by the frontend crop page's filmstrip
- `POST /api/projects/:id/versions/:vid/preprocess/files/reset` — called by
  the overview tab's "undo all," clears the train manifest (**does not
  touch** physical files under train/, see ADR 0010's §`train_clear_all`
  decision)

### Worker logic

For each source name:

1. Resolve the source path → preprocess/source or download/source
2. Open with PIL
3. For each rect: `crop()` → write `preprocess/{stem}_c{n}.png` (when n>1)
   or `preprocess/{stem}.png` (when n=1, overwriting)
4. When n>1, delete the original `preprocess/{stem}.png` (if it exists)
5. Add N entries to the manifest, with origin pointing to the source
   download name

### SSE events

- `crop_progress`: pushed once per completed image, but the worker side
  **throttles to ≥ 1Hz** (the first/last item, skips, and failures are always
  sent; other "done" events are only emitted at least 1s apart). This avoids
  flooding the event stream — a 264-image dataset would otherwise fire
  ~500 events.
- `job_state_changed`: state change

---

## 5. Frontend structure

> **Version note**: the v1 design used a `/preprocess/crop` sub-route + a
> horizontal filmstrip + stage pills with an embedded OperationPanel. After
> shipping, this changed to a `?tool=crop` query string + a shared toolbar +
> a vertical filmstrip. See §9 Addendum 1 for details.

### Routing

All preprocessing tools share a single route,
`/projects/:pid/preprocess`, plus a `?tool=` query:

- `/preprocess` (default) / `?tool=upscale` → the upscale tool
- `?tool=overview` → overview (multi-select + undo)
- `?tool=crop` → the crop tool
- `?tool=inpaint` → a placeholder (not implemented)

Dispatched by `PreprocessHub.tsx`. Switching the query doesn't unmount the
parent route, so tool switching is smoother, and the sidebar's `/preprocess`
match is never interrupted.

### Entry point

An independent **toolbar** at the top of the page
(`PreprocessToolsBar.tsx`) with three or four pills; the leftmost is
"Overview." Each pill is a tool; clicking it becomes a
`<Link to="?tool=...">`. **There's no completion ✓ badge** — a tool isn't a
pipeline node.

### Page layout (shared frame)

```
StepShell (title / subtitle)
└─ grid 1fr / 260px
   ├─ left
   │  ├─ PreprocessToolsBar  [Overview][Upscale][Crop][Inpaint]
   │  ├─ OperationPanel (tool-specific config)
   │  │  ├─ AR dropdown + primary action button
   │  │  └─ Smart clustering section (collapsed by default)
   │  ├─ PreprocessJobStrip (shown only while a job is running / has logs)
   │  └─ WorkArea (crop)
   │     ├─ filter chips · current-image meta · clear this image
   │     └─ grid: filmstrip 220px / canvas 1fr / rect list 260px
   └─ right RightRail (crop progress / estimated output / AR distribution / disk usage)
```

Inside WorkArea, three columns: the filmstrip is arranged vertically (a
3-col grid of square cover thumbs) / the canvas container measures and
adapts / the rect list shows ⎘ ✕ icons in its header when a rect is
selected.

### Overview tab

An independent page, `PreprocessOverview.tsx`: a grid of every image in the
preprocess workspace + a click-to-preview modal + ctrl/shift multi-select +
`Undo selected` + `↶ Undo all`. Undo logic moved here from the upscale page —
no tool handles undo on its own anymore, unifying the UX mental model.

---

## 6. Implementation breakdown

| Step | Effort | Notes |
|---|---|---|
| 1 | M | Backend endpoint + worker + manifest read compatibility / new write schema |
| 2 | M | Frontend: CropPage container + routing + OperationPanel |
| 3 | L | Frontend: FreeCropEditor canvas + gestures + AR-lock |
| 4 | M | Frontend: rect list + filmstrip + filter chips + RightRail |
| 5 | S | Frontend: clustering JS (k-means + elbow + max_crop constraint) |
| 6 | S | Upscale page stage pill converted to a link + i18n additions |
| 7 | S | Tests (pytest for the crop endpoint + manifest, vitest for the editor + k-means) |

---

## 7. ARB bucket alignment (keeping crops consistent with training buckets)

### 7.1 The problem

During training, `runtime/training/dataset.py:BucketManager` derives ~30
`(w, h)` buckets from `(base_reso=1024, step=64, area_tol=0.10,
max_ar=2.0)`; each image is assigned to the nearest bucket by **absolute AR
distance** and resized to that bucket's dimensions. If clustered cropping
picks a pretty AR like "4:3 = 1.333" as its target, the cropped image would
then get resized a second time by the trainer to (1152, 896) = 1.286 or
(1216, 832) = 1.461 — introducing extra distortion.

Cropping's clustered target AR should **exactly match the bucket the trainer
actually lands on**, so the trainer never has to resize the image a second
time.

### 7.2 UX principle (users don't need to know about ARB internals)

The underlying ARB buckets ("1024×1024," "1216×832," bucket count, area
band, step) are **never exposed to the user**:

- **Users unfamiliar with ARB**: the defaults just work; the labels they see
  are all familiar ratios like `1:1` / `4:3` / `3:2` / `16:9`, used as
  normal
- **Users with a little knowledge**: they know 4:3 is a landscape ratio, and
  use it as normal
- **Power users**: if they want the underlying details, they can read the
  source at `runtime/training/dataset.py` themselves — the UI doesn't
  surface it for them

Also: manual mode supports the "crop out the bad part" use case (free AR +
drag), with no forced alignment to training buckets.

### 7.3 Implementation

**Internal** (not exposed):
- The frontend's `studio/web/src/lib/trainBuckets.ts` ports the Python
  `BucketManager` algorithm to TS 1:1
- Default parameters are hardcoded as `base_reso=1024, min_reso=512,
  max_reso=2048, step=64, area_tolerance=0.10, max_ar_ratio=2.0` — 100%
  matching the backend defaults
- `generateBuckets()` generates the bucket grid; `snapToBucket(aspect,
  buckets)` snaps by absolute AR distance

**Integration points**:
- Clustering target AR: cluster center → `snapToBucket()` → the training
  bucket (w, h), with the rect computed from that ratio
- Cluster card label display: uses the "nearest pretty AR" to the training
  bucket AR as the display label (e.g. `cluster 3:2`), while the internal
  rect strictly follows the training bucket's ratio

**Not integrated**:
- Histogram (`arBucket`): unchanged, still snaps to 11 pretty ARs, sorted by
  aspect
- Manual mode's AR dropdown: unchanged (`1:1` / `4:3` / ... /
  `Custom W:H`), UX takes priority

### 7.4 Anti-drift

Backend `runtime/training/dataset.py:BucketManager` and frontend
`lib/trainBuckets.ts` are two independent implementations of the same
algorithm, and the biggest risk is changing one side and forgetting the
other.

- Both files carry a comment at the top cross-referencing each other,
  stating "changing the algorithm / default parameters → must be committed
  on both sides together"
- These two files are flagged as linked files during review
- A cross-language sync test could be added later (optional, not done for now)

### 7.5 Where base_reso comes from

**Hardcoded to 1024**. Reasoning:
- Covers the default SDXL / Flux / Anima scenario (90%+ of users)
- At the preprocess stage, the user doesn't yet need to think about training
  resolution
- For the minority of SD1.5 users, even if the bucket prediction is slightly
  off, the trainer will re-bucket using its real parameters anyway — at
  worst one extra light resize, with no impact on training
- Adding a UI control would be equivalent to exposing the ARB concept,
  violating the §7.2 principle

Making base_reso adjustable can be a follow-up (if it turns out to be a real
per-project pain point).

---

## 8. Not doing

- **Rects aren't persisted**: process information is dropped once written to
  disk; redrawing starts from scratch
- **No stage-ordering constraint**: none (upscale ↔ crop in any order, any
  number of times)
- **No partial undo**: none (only a full restore to download is possible)
- **No multiple manifests / directories**: none (still a single manifest +
  a single preprocess/)
- **No backend-side clustering**: none (it's frontend JS)
- **No keeping old upscale output as a crop backup**: none (each stage
  overwrites)
- **No exposing ARB internals (base_reso / step / bucket count / bucket
  (w,h)) to the user**: none (see §7.2, UX principle)
- **No per-project adjustable base_reso**: none (hardcoded to 1024, see §7.5)
- **No replacing manual mode's AR dropdown with training buckets**: none
  (keeps pretty ARs, UX takes priority)

---

## 9. Addendum 1 — UI evolution (2026-05-21)

Several rounds of UI iteration happened after the design shipped; this logs
the main decisions that deviated from the original draft:

### 9.1 Removing the "manual / smart clustering" segmented tabs

v1 made these two mutually exclusive segmented tabs. User feedback was that
"they aren't mutually exclusive" — boxes generated by clustering can be
edited manually, so it's really the same cropping capability underneath.

What shipped:
- Removed the mode tabs
- The core cropping capability (AR dropdown + canvas + primary action) is
  always visible
- "Smart clustering" was demoted to an independent section below the
  OperationPanel, collapsed by default as `▸ Smart clustering`; expanding it
  shows sliders + a `Start clustering` button
- State persists: once clustering finishes, a `✓ k=N` badge appears at the
  top of the section

### 9.2 URL changed from `/preprocess/crop` to `/preprocess?tool=crop`

The sub-path model had two problems: (1) the sidebar's `/preprocess` path
match was broken by the `/crop` suffix, losing the highlight; (2) switching
tools unmounted the parent route, losing all state.

What shipped:
- A single route, `/projects/:pid/preprocess`
- Dispatched by the query string `?tool=overview|upscale|crop|inpaint`
- A new `PreprocessHub.tsx` dispatcher
- Switching tools no longer unmounts the parent route

### 9.3 "Stage" renamed to "tool," removing the ✓ completed state

Stage pills implied pipeline ordering, but upscale / crop / inpaint aren't
stages at all — they're tools usable in any order, any number of times.

What shipped:
- Copy changed from "stage" to "tool"
- A shared `PreprocessToolsBar.tsx` component sits at the top of every tool
  page
- Pills have no completion badge

### 9.4 Overview tab — a unified undo entry point

The upscale page originally had a "restore N images" button. But other
tools like crop also need undo, and adding it separately to each tool would
be redundant.

What shipped:
- A new `PreprocessOverview.tsx` overview page
- A `[Overview]` pill on the left of the toolbar
- An ImageGrid + shift/ctrl multi-select + a single-image preview modal
- "Undo selected N" + "↶ Undo all" + a confirm modal
- The upscale page's restore controls were removed, keeping the image grid
- A new backend `POST /preprocess/files/reset` route added, routing to
  `preprocess_manifest.clear_all()`

### 9.5 Filmstrip changed from a horizontal bottom strip to a vertical left column

264 images laid out horizontally would squeeze down to a 5px-wide strip,
effectively invisible. Changed to a vertical 3-column grid of square cover
thumbnails, freeing up more vertical space for the canvas.

CSS note: hanging `aspect-ratio: 1` directly on a `<button>` inside a grid
collapses (Chromium/WebKit's anonymous flow-root affects `::before`
padding-top). Wrapping it in a div using the padding-top trick is what makes
it stable.

### 9.6 Canvas sizing adapts to its container

A fixed maxWidth/maxHeight either wastes space or overflows depending on the
viewport. Changed to a ResizeObserver measuring the parent container, with
maxWidth/maxHeight falling back to just an upper bound.

### 9.7 Two pitfalls with AR-lock resizing

- **Collapsing to the full image**: clamping w/h independently when
  exceeding the canvas breaks AR (a rect locked to 1:1 on a 2:3 source image
  turns into the full image = 2:3). Fix: scale by the anchor corner
  proportionally, always preserving AR.
- **A sticky feel**: dragging back after overshooting requires first
  absorbing the accumulated dxN/dyN before it moves. Fix: re-anchor every
  frame, so delta is always the increment from the previous frame to now.

### 9.8 Multi-crop thumbnail addressing

`bucket=download` + `resolve_origin` taking `[0]` always lands on the same
thumbnail once multiple derivatives share an origin after multi-crop.

What shipped:
- The thumb endpoint gained a `bucket=preprocess` option, addressing
  directly by the preprocess filename
- Fallback: when `bucket=download` can't find the file and the name is a
  manifest entry key, it also falls through to preprocess/
- The crop page / overview page / upscale page all address by "processed
  goes through the preprocess bucket + im.name, unprocessed goes through
  download"

### 9.9 SSE throttling

Crop is faster than upscale (300-700ms per image), so 264 images produce
~500 events, flooding the EventSource.

What shipped: `emit_throttled(force=...)` inside the worker: "done" events
are spaced ≥1s apart; the first/last, skips, and failures are always sent.

### 9.10 Pixel distribution + training-bucket alignment

The right-panel stats used to be just the crop's AR distribution. The
upscale page had a pixel-area histogram (6 bins) ported over, aligned with
sd-scripts' ARB training-bucket semantics (see §7 ARB alignment).

The upscale page's filter chips also changed from `All / Unprocessed /
Processed` to `All + pixel bins` (matching the sidebar histogram), since
"unprocessed / processed" is meaningless for upscaling from a UX standpoint.

### 9.11 JobStrip doesn't persist logs

After a page refresh, the status endpoint still returns the historical job +
log_tail, resulting in a pointless empty JobStrip sitting on the page.

What shipped:
- Logs are no longer initialized from `status.log_tail`; logs only
  accumulate from this session's SSE stream
- JobStrip's render condition gained `(isLive || logs.length > 0)`; the
  whole block is hidden when there's no active job and no session-local log
