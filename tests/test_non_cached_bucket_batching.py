"""The non-cached path (cache_latents=False) also batches by ARB bucket.

Regression: the non-cached path used to use a plain DataLoader(batch_size=N) without bucket
grouping -> a batch could mix images with different aspect ratios (different bucket = different H x W), and collate_fn's torch.stack would raise a RuntimeError.
Fix: ImageDataset pre-scans each image's size to build bucket_for_index; the non-cached path now uses BucketBatchSampler.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _write(d: Path, name: str, size) -> None:
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(d / f"{name}.png")
    (d / f"{name}.txt").write_text("1girl", encoding="utf-8")


def test_image_dataset_builds_bucket_for_index(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from runtime.training.dataset import BucketManager, ImageDataset
    _write(tmp_path / "1_data", "a", (1024, 1024))
    _write(tmp_path / "1_data", "b", (1536, 1024))  # different AR -> different bucket
    ds = ImageDataset(tmp_path, 1024, BucketManager(1024), prefer_json=False)
    assert len(ds.bucket_for_index) == len(ds.samples)
    assert all(b is not None for b in ds.bucket_for_index)
    assert len(set(ds.bucket_for_index)) >= 2  # the two ARs land in different buckets


def test_no_bucket_mgr_yields_all_none(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from runtime.training.dataset import ImageDataset
    _write(tmp_path / "1_data", "a", (1024, 1024))
    ds = ImageDataset(tmp_path, 1024, bucket_mgr=None, prefer_json=False)
    assert ds.bucket_for_index == [None]  # no bucketing -> sampler falls back to plain batching


def test_non_cached_batches_are_shape_uniform(tmp_path: Path) -> None:
    # Core regression: non-cached + multiple aspect ratios + bs>1 must iterate without crashing, with uniform size within each batch.
    pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    from runtime.training.dataset import (
        BucketBatchSampler,
        BucketManager,
        ImageDataset,
        collate_fn,
    )

    # 4 aspect ratios x 2 images each -> 4 buckets, each with exactly 2 images for bs=2
    ars = [(1024, 1024), (1536, 1024), (1024, 1536), (1280, 1024)]
    for i, size in enumerate(ars * 2):
        _write(tmp_path / "1_data", f"img{i}", size)

    ds = ImageDataset(tmp_path, 1024, BucketManager(1024), prefer_json=False)
    sampler = BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=True, seed=0)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn)

    n = 0
    for batch in dl:
        px = batch["pixel_values"]   # a successful stack alone proves uniform size within the batch
        assert px.ndim == 4
        n += 1
    assert n > 0
