"""Edit LoRA training (source -> target pairs, cosmos temporal reference). CPU only."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from training.dataset import (  # noqa: E402
    BucketManager,
    CachedLatentDataset,
    ImageDataset,
    collate_fn,
    collate_fn_cached,
)


def _pair(root: Path, name: str, *, target=(255, 0, 0), source=(0, 0, 255), size=(96, 64),
          source_size=None, sub="1_data"):
    from PIL import Image

    tgt = root / "train" / sub / f"{name}.png"
    src = root / "control" / sub / f"{name}.jpg"
    tgt.parent.mkdir(parents=True, exist_ok=True)
    src.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, target)
    img.paste((0, 255, 0), (0, 0, size[0] // 4, size[1]))  # left marker survives only if flips agree
    img.save(tgt)
    src_img = Image.new("RGB", source_size or size, source)
    src_img.paste((0, 255, 0), (0, 0, (source_size or size)[0] // 4, (source_size or size)[1]))
    src_img.save(src, quality=100)
    tgt.with_suffix(".txt").write_text(f"Make it red. {name}", encoding="utf-8")
    return tgt, src


def _dataset(root: Path, **kw):
    return ImageDataset(root / "train", 64, BucketManager(64, step=16), prefer_json=False,
                        shuffle_caption=False, control_dir=root / "control", **kw)


def test_pairs_are_aligned_resized_and_flipped_together(tmp_path):
    _pair(tmp_path, "a", source_size=(192, 128))  # same ratio, larger source
    ds = _dataset(tmp_path)
    item = ds.get_with_flip(0, flip=False)
    flipped = ds.get_with_flip(0, flip=True)
    assert item["control_pixel_values"].shape == (1, *item["pixel_values"].shape)
    # The green marker is on the left of both images, and on the right of both after a flip.
    for value in (item, flipped):
        target_green = value["pixel_values"][1].mean(0) > 0.5
        source_green = value["control_pixel_values"][0, 1].mean(0) > 0.5
        assert torch.equal(target_green, source_green)
    assert bool(flipped["pixel_values"][1, :, -1].mean() > 0.5)
    batch = collate_fn([item, flipped])
    assert batch["control_pixel_values"].shape == (2, 1, *batch["pixel_values"].shape[1:])


def test_missing_source_fails_with_the_file_name(tmp_path):
    from PIL import Image

    _pair(tmp_path, "a")
    lonely = tmp_path / "train" / "1_data" / "lonely.png"
    Image.new("RGB", (64, 64)).save(lonely)
    lonely.with_suffix(".txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="lonely.png"):
        _dataset(tmp_path)


def test_flat_control_folder_is_accepted(tmp_path):
    tgt, src = _pair(tmp_path, "a")
    src.rename(tmp_path / "control" / src.name)
    ds = _dataset(tmp_path)
    assert ds.samples[0]["control"] == tmp_path / "control" / src.name


class _FakeVAE:
    """Pixel average-pool by 8 into 16 channels; deterministic and cheap."""

    def encode(self, pixels):
        b, c, t, h, w = pixels.shape
        pooled = torch.nn.functional.avg_pool2d(pixels.flatten(1, 2).float(), 8)
        return pooled.view(b, c, t, h // 8, w // 8).repeat(1, 6, 1, 1, 1)[:, :16]


def test_cache_stores_source_latents_and_invalidates_on_source_edit(tmp_path):
    _pair(tmp_path, "a")
    ds = _dataset(tmp_path, flip_augment=True)
    cached = CachedLatentDataset(ds, _FakeVAE(), "cpu", torch.float32)
    item = cached[0]
    assert item["control_latent"].shape == item["latent"].shape
    batch = collate_fn_cached([item])
    assert batch["control_latents"].shape == batch["latents"].shape
    npz = ds.samples[0]["image"].with_suffix(".npz")
    sample = ds.samples[0]
    assert cached._is_cache_valid(sample["image"], npz, sample.get("target_reso"))

    # A changed source must re-encode, even though the target is untouched.
    later = time.time() + 5
    os.utime(sample["control"], (later, later))
    assert not cached._is_cache_valid(sample["image"], npz, sample.get("target_reso"))


def _tiny_anima():
    from modeling.anima.anima_modeling import Anima

    torch.manual_seed(0)
    return Anima(
        max_img_h=64, max_img_w=64, max_frames=4, in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
        model_channels=64, num_blocks=1, num_heads=2, mlp_ratio=2.0,
        crossattn_emb_channels=32, pos_emb_cls="rope3d", use_adaln_lora=True, adaln_lora_dim=16,
    ).eval()


def test_reference_frame_conditions_the_target_frame():
    from training.families.anima.family import AnimaFamily

    model = _tiny_anima()
    family = AnimaFamily()
    noisy = torch.randn(1, 16, 1, 8, 8)
    t = torch.full((1,), 0.5)
    cross = torch.randn(1, 4, 32)
    with torch.no_grad():
        out_a = family.forward_train(model, torch.cat([noisy, torch.randn(1, 16, 1, 8, 8)], 2), t, cross)
        out_b = family.forward_train(model, torch.cat([noisy, torch.randn(1, 16, 1, 8, 8)], 2), t, cross)
        plain = family.forward_train(model, noisy, t, cross)
    assert out_a.shape == (1, 16, 2, 8, 8)
    assert plain.shape == (1, 16, 1, 8, 8)
    # Different sources change the target frame's prediction: the reference is attended to.
    assert not torch.allclose(out_a[:, :, :1], out_b[:, :, :1])


def test_reference_fit_and_size():
    from PIL import Image
    from training.families.anima.sampling import fit_reference_image, reference_size

    image = Image.new("RGB", (832, 1216))
    w, h = reference_size(image, 1024 * 1024, 16)
    assert w % 16 == 0 and h % 16 == 0 and h > w
    assert abs(w * h - 1024 * 1024) / (1024 * 1024) < 0.05
    assert fit_reference_image(image, 512, 512).size == (512, 512)


def test_schema_gates_incompatible_paths():
    from pydantic import ValidationError
    from studio.schema import TrainingConfig
    from studio.services.presets.io import _tolerant_validate

    assert not TrainingConfig().edit_mode
    TrainingConfig(edit_mode=True, edit_control_dir="x")
    for flag in ("navit_packing", "dop_enabled", "sra_enabled", "leap_enabled"):
        with pytest.raises(ValidationError):
            TrainingConfig(edit_mode=True, **{flag: True})
    with pytest.raises(ValidationError):
        TrainingConfig(edit_mode=True, model_family="krea2")
    # A preset with DOP applied to an edit version keeps edit_mode and drops DOP.
    cfg, _dropped, fixed = _tolerant_validate(
        {"model_family": "anima", "edit_mode": True, "dop_enabled": True})
    assert cfg.edit_mode and not cfg.dop_enabled and "dop_enabled" in fixed


def test_two_inputs_become_two_reference_frames(tmp_path):
    from PIL import Image

    _pair(tmp_path, "a")
    second = tmp_path / "control2" / "1_data" / "a.png"
    second.parent.mkdir(parents=True)
    Image.new("RGB", (96, 64), (0, 200, 0)).save(second)
    ds = ImageDataset(tmp_path / "train", 64, BucketManager(64, step=16), prefer_json=False,
                      shuffle_caption=False, control_dir=[tmp_path / "control", tmp_path / "control2"])
    item = ds.get_with_flip(0, flip=False)
    assert item["control_pixel_values"].shape[0] == 2
    cached = CachedLatentDataset(ds, _FakeVAE(), "cpu", torch.float32)
    latent = cached[0]
    assert latent["control_latent"].shape[1] == 2 and latent["latent"].shape[1] == 1
    batch = collate_fn_cached([latent])
    assert batch["control_latents"].shape[2] == 2
    # Removing the second input invalidates the cache entry.
    second.unlink()
    with pytest.raises(ValueError, match="a.png"):
        ImageDataset(tmp_path / "train", 64, BucketManager(64, step=16), prefer_json=False,
                     control_dir=[tmp_path / "control", tmp_path / "control2"])


def test_model_accepts_two_reference_frames():
    from training.families.anima.family import AnimaFamily

    model = _tiny_anima()
    noisy = torch.randn(1, 16, 1, 8, 8)
    with torch.no_grad():
        out = AnimaFamily().forward_train(
            model, torch.cat([noisy, torch.randn(1, 16, 2, 8, 8)], 2), torch.full((1,), 0.5), torch.randn(1, 4, 32))
    assert out.shape == (1, 16, 3, 8, 8)


def test_edit_previews_redo_first_examples(tmp_path):
    from types import SimpleNamespace
    from training.phases.resume import _edit_preview_examples

    for name in ("a", "b", "c", "d"):
        _pair(tmp_path, name)
    ds = _dataset(tmp_path)
    picks = _edit_preview_examples(SimpleNamespace(base_dataset=ds), limit=3)
    assert [p[0].split(". ")[-1] for p in picks] == ["a", "b", "c"]
    assert all(len(refs) == 1 and refs[0].parent.parent.name == "control" for _, refs in picks)
