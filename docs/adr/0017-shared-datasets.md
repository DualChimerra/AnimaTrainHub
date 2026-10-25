# 0017 — Make datasets shared, with training caches inside the dataset folder

**Status**: Proposed
**Date**: 2026-10-25

## Background

Images lived in each project's `download/`. Training the same images in a new
project meant uploading them again, and the VAE latent / caption caches were
sidecars next to each version's `train/` copies, so every new version re-encoded
everything.

## Decision

- A shared dataset is a `projects` row with `kind='dataset'`, stored in
  `studio_data/datasets/{id}-{slug}/` (`download/` + `cache/`). Every download,
  upload, file and thumbnail endpoint already takes a project id, so they work for
  datasets unchanged; `projects.project_dir()` resolves the datasets folder.
  `/api/projects` hides datasets unless `include_datasets=true`; the library has
  its own routes under `/api/library/datasets`.
- `version_datasets` (migration v22) links a version to any number of datasets. A
  new version inherits the links of its fork source or the active version.
- The curation pool is the project's own `download/` plus the linked datasets.
  Dataset images are named by a **pool key** `@{dataset_id}/{file}`, which becomes
  the train manifest `origin` (validation keeps a small `origins.json`). Bare names
  still mean the project's own `download/`, so old projects read exactly as before.
  Two datasets with the same file name get distinct copies (`001-d{id}.png`).
- Caches are content-addressed inside the dataset: at spawn the supervisor writes
  `train/.cache_routes.json` (train image → dataset `cache/`), and the trainer
  (`runtime/training/dataset_cache.py`) stores
  `cache/latents/<fingerprint>-v<layout>/<sha>_<w>x<h>.npz` and
  `cache/text/<te fingerprint>-v<format>/<caption key>.safetensors`. A different
  model, resolution or text encoder writes new files next to the old ones.
  Images with masks or edit reference frames, reg images and CLI runs keep the
  classic sidecars.

## Alternatives considered

- **Hardlink dataset images into each project's `download/`**: no code changes
  downstream, but deletions and later uploads would not propagate, and the
  curation page could not show which dataset an image came from.
- **Copy the dataset into the project**: the duplicated uploads/disk use this
  change exists to remove.
- **Key caches by train path**: every version would still re-encode; keying by
  image bytes lets identical copies share one entry and gives cropped / upscaled
  copies their own.

## Consequences

- Old projects keep working with no migration of files; their `download/` shows
  as a "Project images" source, and "Make a shared dataset" hardlinks (or copies)
  those images into a new dataset.
- Train copies stay per version (crop, upscale and tags are version-specific).
- Deleting a dataset removes its folder and cache; versions keep their train
  copies, only restoring an original from it stops working ("no_origin").
- The train-bundle "latent cache" export does not include shared-cache entries.
