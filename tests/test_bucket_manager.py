"""BucketManager ARB bucket generation unit tests.

Locks down two things:
1. The default params (base=1024, R=2.0) = 37 buckets, mirroring the frontend
   `trainBuckets.ts` (`trainBuckets.test.ts` also asserts 37).
2. Once min/max are derived from (base, R), a small base no longer
   degenerates -- base=512 used to only generate a single 512x512 square
   bucket because min_reso=512 was hardcoded (ARB broken), see
   `docs/design/multi-resolution-training-design.md` §6.3.
"""
from __future__ import annotations

from runtime.training.dataset import BucketManager


def test_default_base_yields_37_buckets() -> None:
    m = BucketManager()  # base=1024, R=2.0
    assert len(m.buckets) == 37
    # min/max are now derived from (base, R) rather than hardcoded 512 / 2048
    assert (m.min_reso, m.max_reso) == (640, 1536)


def test_default_contains_canonical_anchors() -> None:
    m = BucketManager()
    for wh in [(1024, 1024), (1152, 896), (1216, 832), (896, 1152), (832, 1216)]:
        assert wh in m.buckets


def test_all_buckets_within_area_band_and_ar_cap() -> None:
    m = BucketManager()
    base_area = 1024 * 1024
    for w, h in m.buckets:
        assert abs(w * h - base_area) / base_area <= 0.1 + 1e-9
        assert max(w / h, h / w) <= 2.0 + 1e-9
        assert w % 64 == 0 and h % 64 == 0


def test_small_base_keeps_aspect_ratio_variety() -> None:
    # regression: with min_reso=512 hardcoded, base=512 could only generate
    # 512x512 (the +-10% area band can't fit any non-square bucket), so the
    # whole dataset got squashed to square and ARB was broken. Deriving
    # min/max restores the variety.
    m = BucketManager(512)
    assert len(m.buckets) > 1
    assert any(w != h for w, h in m.buckets)
    assert any(w > h for w, h in m.buckets)  # has landscape buckets
    assert any(h > w for w, h in m.buckets)  # has portrait buckets


def test_aspect_ratio_limit_widens_bucket_set() -> None:
    narrow = BucketManager(1024, aspect_ratio_limit=2.0)
    wide = BucketManager(1024, aspect_ratio_limit=3.0)
    narrow_max = max(max(w / h, h / w) for w, h in narrow.buckets)
    wide_max = max(max(w / h, h / w) for w, h in wide.buckets)
    assert narrow_max <= 2.0 + 1e-9
    assert 2.0 < wide_max <= 3.0 + 1e-9
    assert len(wide.buckets) > len(narrow.buckets)


def test_explicit_min_max_override_is_respected() -> None:
    # explicitly passing min/max skips derivation (relied on by cache tests: forces a single square bucket).
    m = BucketManager(256, min_reso=256, max_reso=256, step=64)
    assert m.buckets == [(256, 256)]


def test_get_bucket_snaps_by_aspect_ratio_then_area() -> None:
    m = BucketManager()
    # extremely wide -> the widest bucket (AR~=2.0); the image's absolute size doesn't affect bucket selection.
    bw, bh = m.get_bucket(5000, 1000)
    assert abs(bw / bh - 2.0) < 0.1
    # same AR, different absolute size -> same bucket
    assert m.get_bucket(800, 600) == m.get_bucket(1600, 1200)


def test_get_bucket_ties_prefer_base_area_square() -> None:
    # base=1536 generates several 1:1 buckets (1472^2/1536^2/1600^2 all fall within the +-10% area band).
    # bucket selection can't just take the first, smaller square bucket in generation order; when AR ties, it should fall back to the bucket closest to base^2.
    m = BucketManager(1536, aspect_ratio_limit=2.0)
    assert (1472, 1472) in m.buckets
    assert (1536, 1536) in m.buckets
    assert (1600, 1600) in m.buckets
    assert m.get_bucket(2048, 2048) == (1536, 1536)
