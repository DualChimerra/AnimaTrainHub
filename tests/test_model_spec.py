
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from training.families import SPECS, get_spec
from training.families.anima import ANIMA_SPEC
from training.families.spec import (
    LoraOutputSpec,
    ModelSpec,
    TextSpec,
    validate_spec,
)



def test_anima_spec_registered():
    assert get_spec("anima") is ANIMA_SPEC
    assert SPECS["anima"].family_id == "anima"


def test_unknown_family_raises_with_registered_list():
    with pytest.raises(ValueError, match="anima"):
        get_spec("no-such-family")


def test_anima_latent_constants():
    lat = ANIMA_SPEC.latent
    assert lat.fingerprint == "wan21-f8c16"
    assert lat.channels == 16
    assert lat.spatial_stride == 8
    assert lat.patch_spatial == 2
    assert lat.align_px == 16
    assert not lat.temporal
    assert len(lat.rgb_factors) == lat.channels
    assert all(len(r) == 3 for r in lat.rgb_factors)
    assert len(lat.rgb_bias) == 3


def test_spec_matches_vae_wrapper_constants():
    from training.vae import VAEWrapper

    assert VAEWrapper._UPSAMPLE == ANIMA_SPEC.latent.spatial_stride


def _spec_with(text_strategy, caps):
    return ModelSpec(
        family_id="x",
        display_name="X",
        objective="rectified_flow",
        latent=ANIMA_SPEC.latent,
        text=TextSpec(strategy=text_strategy, max_seq_len=512, fingerprint="f"),
        sampling=ANIMA_SPEC.sampling,
        capabilities=frozenset(caps),
        lora=LoraOutputSpec(prefix="lora_unet", preset_name="x"),
    )


def test_cached_varlen_excludes_caption_tag_ops():
    with pytest.raises(ValueError):
        validate_spec(_spec_with("cached_varlen", {"caption_tag_ops"}))
    validate_spec(_spec_with("cached_varlen", {"masked_loss", "text_cache"}))


def test_cached_varlen_requires_capability_and_text_fingerprint():
    with pytest.raises(ValueError, match="text_cache"):
        validate_spec(_spec_with("cached_varlen", {"masked_loss"}))
    spec = _spec_with("cached_varlen", {"masked_loss", "text_cache"})
    spec = replace(spec, text=replace(spec.text, fingerprint=""))
    with pytest.raises(ValueError, match="TE fingerprint"):
        validate_spec(spec)


def test_unknown_capability_rejected():
    with pytest.raises(ValueError):
        validate_spec(_spec_with("online", {"warp_drive"}))



def _bare_cached_dataset():
    from training.dataset import CachedLatentDataset

    ds = CachedLatentDataset.__new__(CachedLatentDataset)
    ds.np = np
    ds.flip_augment = False
    ds.load_masks = False
    ds.base_image_dataset = None
    ds._expected_bucket_size = lambda img_path, target_reso=None: None
    ds.latent_spec = ANIMA_SPEC.latent
    return ds


def _write_pair(tmp_path, name, **npz_kwargs):
    img = tmp_path / f"{name}.png"
    img.write_bytes(b"fake")
    npz = tmp_path / f"{name}.npz"
    np.savez(npz, latent=np.zeros((16, 1, 4, 4), dtype=np.float32), **npz_kwargs)
    past = npz.stat().st_mtime - 3600
    os.utime(img, (past, past))
    return img, npz


def test_legacy_cache_without_fingerprint_is_grandfathered(tmp_path):
    ds = _bare_cached_dataset()
    img, npz = _write_pair(tmp_path, "legacy", bucket_w=64, bucket_h=64)
    assert ds._is_cache_valid(img, npz) is True
    assert npz.exists()


def test_cache_with_matching_fingerprint_is_valid(tmp_path):
    from training.dataset import LATENT_CACHE_LAYOUT_VERSION

    ds = _bare_cached_dataset()
    img, npz = _write_pair(
        tmp_path, "ok",
        latent_fingerprint=ANIMA_SPEC.latent.fingerprint,
        layout_version=LATENT_CACHE_LAYOUT_VERSION,
    )
    assert ds._is_cache_valid(img, npz) is True


def test_cache_with_wrong_fingerprint_is_deleted(tmp_path):
    ds = _bare_cached_dataset()
    img, npz = _write_pair(
        tmp_path, "alien",
        latent_fingerprint="sdxl-f8c4",
        layout_version=1,
    )
    assert ds._is_cache_valid(img, npz) is False
    assert not npz.exists()


def test_cache_with_unknown_layout_version_is_deleted(tmp_path):
    ds = _bare_cached_dataset()
    img, npz = _write_pair(
        tmp_path, "future",
        latent_fingerprint=ANIMA_SPEC.latent.fingerprint,
        layout_version=999,
    )
    assert ds._is_cache_valid(img, npz) is False
    assert not npz.exists()
