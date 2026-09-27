"""Unit tests for NaViT native resolution sizing (navit_native_resolution).

Covers:
  1) plan_native_fit_image: floor-align to 16px, zero padding, token count = grid product.
  2) Over-budget downscale: scale proportionally to <= token budget, <= RoPE per-side cap,
     still 16-aligned.
  3) over_budget=fail: raises when over the limit.
  4) **Wiring test (critical)**: ImageDataset(native_resolution=True) actually sizes via
     native floor-16, producing a different size than the ARB bucket path -- this is exactly
     what was missing from the previous PR and caused "the argument is empty".
  5) config: TrainingConfig defaults navit_native_resolution=False, can be enabled, and
     native requires navit_packing.

CPU-only, no GPU / no VAE: runs fine in CI (Linux, no GPU).
"""
from __future__ import annotations

import pytest

from training.dataset import (
    BucketManager,
    ImageDataset,
    NativeFitImagePlan,
    plan_native_fit_image,
)


# A set of heterogeneous native sizes (including some not multiples of 16)
_SIZES = [(1000, 1500), (1536, 512), (777, 777), (2048, 768), (640, 1664)]


# --------------------------------------------------------- plan: floor invariants
def test_floor_alignment_invariants():
    for w, h in _SIZES:
        plan = plan_native_fit_image(w, h, align=16)  # no budget limit
        assert isinstance(plan, NativeFitImagePlan)
        assert plan.width % 16 == 0 and plan.height % 16 == 0
        assert plan.width <= w and plan.height <= h          # floor only crops, never upscales
        assert w - plan.width < 16 and h - plan.height < 16  # each side crops off < 16px remainder
        assert plan.token_count == (plan.width // 16) * (plan.height // 16)
        assert plan.was_downscaled is False


# --------------------------------------------------------- plan: over-budget downscale
def test_over_budget_downscale_fits_and_aligned():
    # 1000x1500 floor -> 992x1488 = 62x93 = 5766 tokens; budget 1024 -> must downscale
    budget = 1024
    plan = plan_native_fit_image(1000, 1500, max_tokens=budget, over_budget="downscale")
    assert plan.was_downscaled is True
    assert plan.width % 16 == 0 and plan.height % 16 == 0
    assert plan.token_count <= budget
    # aspect ratio is roughly preserved (scale + floor deviation stays within one align unit)
    assert plan.height > plan.width  # portrait stays portrait


def test_over_budget_downscale_extreme_aspect_within_budget():
    # extreme aspect ratio: when one axis gets clamped by max(1,.), the other should shrink back
    budget = 64
    plan = plan_native_fit_image(4096, 128, max_tokens=budget, over_budget="downscale")
    assert plan.token_count <= budget
    assert plan.width % 16 == 0 and plan.height % 16 == 0


def test_rope_side_cap_downscales():
    # extremely elongated image over the RoPE per-side cap: should be clamped to <= max_side_tokens
    max_side = 32
    plan = plan_native_fit_image(
        8000, 512, max_tokens=0, max_side_tokens=max_side, over_budget="downscale"
    )
    assert plan.token_w <= max_side and plan.token_h <= max_side


def test_over_budget_fail_raises():
    with pytest.raises(ValueError):
        plan_native_fit_image(4096, 4096, max_tokens=1024, over_budget="fail")


# --------------------------------------------------------- wiring test (critical)
def _make_dataset_dir(tmp_path, sizes):
    """Create a few PNGs with .txt captions in tmp_path, return the directory path."""
    from PIL import Image
    for i, (w, h) in enumerate(sizes):
        Image.new("RGB", (w, h), (i * 7 % 256, 0, 0)).save(tmp_path / f"img{i}.png")
        (tmp_path / f"img{i}.txt").write_text("1girl, solo", encoding="utf-8")
    return str(tmp_path)


def test_dataset_native_sizing_bypasses_buckets(tmp_path):
    """When native_resolution=True, sizing really goes through native floor-16 and differs
    from the ARB bucket path."""
    data_dir = _make_dataset_dir(tmp_path, [(1000, 1500), (777, 777)])

    native = ImageDataset(data_dir, resolution=1024, bucket_mgr=None,
                          native_resolution=True, native_token_budget=1_000_000)
    # ask the sizing directly: native floor-16
    assert native._target_size_for(1000, 1500) == (992, 1488)   # 1000//16*16, 1500//16*16
    assert native._target_size_for(777, 777) == (768, 768)
    # the pre-scanned bucket_for_index should also be native sizes (not None, 16-aligned, <= source)
    for size in native.bucket_for_index:
        assert size is not None
        tw, th = size
        assert tw % 16 == 0 and th % 16 == 0

    # control: the ARB bucket path (native off) gives bucket sizes for the same images, differing from native
    bucketed = ImageDataset(data_dir, resolution=1024,
                            bucket_mgr=BucketManager(1024, aspect_ratio_limit=2.0))
    assert bucketed.native_resolution is False
    assert bucketed._target_size_for(1000, 1500) != native._target_size_for(1000, 1500)


def test_dataset_native_downscale_wiring(tmp_path):
    """native + small token budget: ImageDataset sizing really downscales an over-budget image to fit."""
    data_dir = _make_dataset_dir(tmp_path, [(2048, 2048)])  # floor -> 128x128 token=16384
    budget = 4096
    ds = ImageDataset(data_dir, resolution=1024, bucket_mgr=None,
                      native_resolution=True, native_token_budget=budget,
                      native_over_budget="downscale")
    tw, th = ds._target_size_for(2048, 2048)
    assert (tw // 16) * (th // 16) <= budget


def test_dataset_native_collapses_multi_resolution_fanout(tmp_path):
    """native + a multi-resolution list: fan-out collapses to a single copy.

    Without collapsing, the same image would be duplicated per resolution tier (implicitly
    x N per epoch), and CachedLatentDataset would encode one r{reso}.npz per
    (image, target_reso) even though the latent content is identical.
    """
    data_dir = _make_dataset_dir(tmp_path, [(1000, 1500), (777, 777)])

    ds = ImageDataset(data_dir, resolution=1024, bucket_mgr=None,
                      resolutions=[1024, 512],
                      native_resolution=True, native_token_budget=1_000_000)
    # exactly one sample per image (not duplicated per resolution tier)
    assert len(ds.samples) == 2
    # all samples share the same target_reso -> the cache layer won't split npz per tier (_multi_reso is empty)
    assert len({s.get("target_reso") for s in ds.samples}) == 1

    # control: with native off, multi-resolution fan-out behavior is unchanged (regression guard)
    bucketed = ImageDataset(data_dir, resolution=1024,
                            bucket_mgr=BucketManager(1024, aspect_ratio_limit=2.0),
                            resolutions=[1024, 512])
    assert len(bucketed.samples) == 4


# --------------------------------------------------------- config switch
def test_config_default_off():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.navit_native_resolution is False
    assert cfg.navit_native_over_budget == "downscale"


def test_config_can_enable_with_packing():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig(
        navit_packing=True, cache_latents=True, navit_token_budget=16384,
        navit_native_resolution=True,
    )
    assert cfg.navit_native_resolution is True


def test_config_native_requires_packing():
    from studio.domain import TrainingConfig
    with pytest.raises(ValueError):
        TrainingConfig(navit_native_resolution=True)  # navit_packing defaults to False -> rejected
