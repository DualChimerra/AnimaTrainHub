"""compute_navit_pack_estimate: the data source behind the steps/epoch estimate in NaViT packing mode.

Packing goes through the real NavitPackBatchSampler (not an algorithm copy); the scan
rules share their source with compute_bucket_histogram. Regression background: the
training page's step estimate of "samples / batch_size" is distorted under navit
(batch_size doesn't participate in batching) -- a 1536px dataset estimated 2520 but
actually ran 5040.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _img(d: Path, names, size=(64, 64), caption=True) -> None:
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.new("RGB", size).save(d / f"{n}.png")
        if caption:
            (d / f"{n}.txt").write_text("1girl", encoding="utf-8")


def test_native_one_image_per_pack(tmp_path: Path) -> None:
    # This bug's scenario: single-image token > budget/2 -> no two images ever fit together -> pack count = sample count
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "1_data", ["a", "b", "c"])  # 64x64 -> 4x4 = 16 tokens
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=24,
    )
    assert out["samples"] == 3
    assert out["packs_per_epoch"] == 3
    assert out["token_min"] == out["token_max"] == 16


def test_native_packs_fill_budget(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "1_data", ["a", "b", "c"])  # 16 tokens each
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=48,
    )
    assert out["packs_per_epoch"] == 1
    assert out["avg_images_per_pack"] == 3.0
    assert out["sizes"] == [{"w": 64, "h": 64, "count": 3}]


def test_repeat_expands_samples(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "5_data", ["a", "b"])
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=16,
    )
    assert out["samples"] == 10  # 2 images x repeat 5
    assert out["packs_per_epoch"] == 10  # budget exactly fits one image -> 1 per pack


def test_arb_bucket_tokens_without_native(tmp_path: Path) -> None:
    # Non-native: tokens follow the ARB bucket size (w//16)*(h//16), the same convention used to derive the latent shape
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "1_data", ["a", "b"], size=(1024, 1024))
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=False, token_budget=8192,
    )
    # 1024x1024 bucket -> 64x64 = 4096 tokens; 8192 budget fits 2 images
    assert out["token_min"] == out["token_max"] == 4096
    assert out["samples"] == 2
    assert out["packs_per_epoch"] == 1
    assert out["sizes"] == []  # non-native doesn't emit a size histogram (bucket histogram already has it)


def test_native_downscale_over_budget(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "1_data", ["big"], size=(256, 256))  # 16x16 = 256 tokens
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=64,
    )
    assert out["downscaled"] == 1
    assert out["token_max"] <= 64
    assert out["packs_per_epoch"] == 1


def test_uncaptioned_images_skipped(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "1_data", ["a"])
    _img(tmp_path / "1_data", ["b"], caption=False)
    out = compute_navit_pack_estimate(
        [tmp_path], [1024], native_resolution=True, token_budget=16,
    )
    assert out["samples"] == 1


def test_reg_dir_joins_the_pool(tmp_path: Path) -> None:
    # reg set and main are joined into the same packing pool (MergedDataset semantics)
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    _img(tmp_path / "train" / "1_data", ["a"])
    _img(tmp_path / "reg" / "1_prior", ["r1", "r2"])
    out = compute_navit_pack_estimate(
        [tmp_path / "train", tmp_path / "reg"], [1024],
        native_resolution=True, token_budget=16,
    )
    assert out["samples"] == 3
    assert out["packs_per_epoch"] == 3


def test_empty_dataset(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_navit_pack_estimate
    out = compute_navit_pack_estimate(
        [tmp_path / "nope"], [1024], native_resolution=True, token_budget=16,
    )
    assert out["packs_per_epoch"] == 0
    assert out["samples"] == 0
