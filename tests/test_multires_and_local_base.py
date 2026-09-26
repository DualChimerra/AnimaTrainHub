from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


from training.dataset import BucketManager  # noqa: E402


def test_bucket_defaults_yield_37():
    derived = BucketManager(1024, aspect_ratio_limit=2.0)
    explicit = BucketManager(1024, min_reso=512, max_reso=2048, step=64,
                             aspect_ratio_limit=2.0)
    assert derived.buckets == explicit.buckets
    assert len(derived.buckets) == 37


def test_bucket_explicit_knobs_honored():
    fine = BucketManager(1024, min_reso=512, max_reso=2048, step=64,
                         aspect_ratio_limit=2.0)
    coarse = BucketManager(1024, min_reso=512, max_reso=2048, step=128,
                           aspect_ratio_limit=2.0)
    assert len(coarse.buckets) < len(fine.buckets)


def test_bucket_max_ar_configurable():
    narrow = BucketManager(1024, aspect_ratio_limit=1.0)
    wide = BucketManager(1024, aspect_ratio_limit=3.0)
    assert len(narrow.buckets) < len(wide.buckets)
    for w, h in wide.buckets:
        assert max(w / h, h / w) <= 3.0 + 1e-9


def test_bucket_schema_fields_and_validator():
    from studio.domain.training import TrainingConfig

    c = TrainingConfig()
    assert c.bucket_min_reso is None
    assert c.bucket_max_reso is None
    assert c.bucket_step is None
    assert c.aspect_ratio_limit == 2.0

    c2 = TrainingConfig(bucket_min_reso=512, bucket_max_reso=2048, bucket_step=64)
    assert (c2.bucket_min_reso, c2.bucket_max_reso, c2.bucket_step) == (512, 2048, 64)

    c3 = TrainingConfig.model_validate({"bucket_max_ar": 3.0})
    assert c3.aspect_ratio_limit == 3.0
    c4 = TrainingConfig.model_validate({"bucket_max_ar": 3.0, "aspect_ratio_limit": 2.5})
    assert c4.aspect_ratio_limit == 2.5
