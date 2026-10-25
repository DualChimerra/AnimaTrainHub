"""Runtime side of the shared dataset cache: routes file -> content-addressed
latent / text cache paths inside the dataset folder."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


def _routes(data_dir: Path, mapping: dict[str, Path]) -> None:
    (data_dir / ".cache_routes.json").write_text(
        json.dumps({"version": 1, "routes": {k: str(v) for k, v in mapping.items()}}),
        encoding="utf-8",
    )


def _image(path: Path, color=(200, 120, 40)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 384), color=color).save(path)
    path.with_suffix(".txt").write_text("1girl", encoding="utf-8")
    return path


def test_router_paths_are_content_addressed(tmp_path: Path) -> None:
    from runtime.training.dataset_cache import SharedCacheRouter

    cache = tmp_path / "datasets" / "3-foxes" / "cache"
    v1 = tmp_path / "v1" / "train"
    v2 = tmp_path / "v2" / "train"
    a = _image(v1 / "1_data" / "a.png")
    b = v2 / "5_data" / "renamed.png"
    b.parent.mkdir(parents=True)
    shutil.copy2(a, b)
    _routes(v1, {"1_data/a.png": cache})
    _routes(v2, {"5_data/renamed.png": cache})

    r1 = SharedCacheRouter.load(v1)
    r2 = SharedCacheRouter.load(v2)
    p1 = r1.latent_path(a, fingerprint="wan21-f8c16", layout_version=1, bucket=(512, 384))
    p2 = r2.latent_path(b, fingerprint="wan21-f8c16", layout_version=1, bucket=(512, 384))
    assert p1 == p2  # same bytes, different version / folder / name -> one entry
    assert p1.parent == cache / "latents" / "wan21-f8c16-v1"
    assert p1.name.endswith("_512x384.npz")
    # another model or bucket lands next to it, never on top of it
    assert r1.latent_path(a, fingerprint="flux2-f8c32-patch2-bn-v1", layout_version=1, bucket=(512, 384)).parent != p1.parent
    assert r1.latent_path(a, fingerprint="wan21-f8c16", layout_version=1, bucket=(768, 576)) != p1

    t = r1.text_path(a, cache_key="f" * 64, text_fingerprint="qwen3-vl-4b", format_version=1)
    assert t == cache / "text" / "qwen3-vl-4b-v1" / f"{'f' * 40}.safetensors"

    # unrouted image / missing file -> sidecar behaviour
    assert r1.latent_path(_image(v1 / "1_data" / "other.png"), fingerprint="x", layout_version=1, bucket=(1, 1)) is None
    assert SharedCacheRouter.load(tmp_path / "nowhere") is None

    r1.touch_group(p1.parent, {"kind": "latents", "family": "anima"})
    meta = json.loads((p1.parent / "group.json").read_text(encoding="utf-8"))
    assert meta["family"] == "anima" and meta["last_used_at"] >= meta["created_at"]


def test_cached_latent_dataset_uses_shared_npz(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import numpy as np

    from runtime.training.dataset import BucketManager, CachedLatentDataset, ImageDataset
    from runtime.training.dataset_cache import SharedCacheRouter

    data_dir = tmp_path / "train"
    img = _image(data_dir / "1_data" / "a.png")
    cache = tmp_path / "cache"
    _routes(data_dir, {"1_data/a.png": cache})

    bucket_mgr = BucketManager(512)
    dataset = ImageDataset(data_dir, 512, bucket_mgr)
    cached = object.__new__(CachedLatentDataset)
    cached.base_dataset = dataset
    cached.base_image_dataset = dataset
    cached.np = np
    cached.samples = dataset.samples
    cached.cache_dir = None
    cached.bucket_for_index = []
    cached.shared_cache = SharedCacheRouter.load(data_dir)
    cached._shared_npz = {}
    cached._shared_set = set()

    npz = cached._get_npz_path(img)
    assert npz.is_relative_to(cache / "latents")
    w, h = cached._expected_bucket_size(img)
    assert npz.name.endswith(f"_{w}x{h}.npz")

    cached._save_npz(npz, latent=np.zeros((16, 1, h // 8, w // 8), dtype=np.float32), bucket_w=w, bucket_h=h)
    assert not list(npz.parent.glob("*.tmp.npz"))
    # A freshly copied (newer mtime) identical image still hits the shared entry.
    img.touch()
    assert cached._is_cache_valid(img, npz) is True
