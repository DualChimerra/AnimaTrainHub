"""Quick sanity check for the training mask data pipeline (B1 data plane -> B2 transform-math dry-run).

Takes the masks saved from the paint page (train/{folder}/{stem}.mask sidecar next to the image,
grayscale PNG content) through the same geometric pipeline the trainer will run, to verify that
the "mask -> training" data plane is wired correctly:

  1. mask existence + size validation (trainer's fail-safe semantics: a mismatch means that
     image is treated as having no mask)
  2. bucket assignment + resize-cover + center-crop (mirrors the geometry computation in
     ImageDataset.get_with_flip; masks use NEAREST to avoid grayscale interpolation contamination)
  3. area downsampling to latent /8 resolution (PIL BOX = block averaging, matching the B2 decision)
  4. prints each image's latent weight-map shape + mean / coverage stats

Note: the trainer's masked loss (B2) is not implemented yet; this script only verifies the data
pipeline and transform math. Loss-level effectiveness will be A/B tested once B2 lands
(design doc §9 decision 6).

Usage:
  ./venv/Scripts/python.exe tools/mask_check.py <train_dir> [--reso 1024] [--save-vis DIR]

  train_dir  = projects/{id}-{slug}/versions/{label}/train
  --reso     bucket base resolution (default 1024, matches the training config's resolution)
  --save-vis save visual-inspection images to a directory: {stem}.overlay.png (original image +
             red mask overlay) / {stem}.latent.png (latent weight map, NEAREST-upscaled back to
             bucket size)

Exit code: 0 = all passed; 1 = size mismatch / read failure exists (the trainer will downgrade
these to no-mask).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Non-ASCII output crashes under Windows console cp932; force utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from runtime.training.dataset import BucketManager  # noqa: E402

VAE_DOWNSAMPLE = 8
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MASK_SUFFIX = ".mask"


def mask_path_for(train_dir: Path, folder: str, filename: str) -> Path:
    return train_dir / folder / f"{Path(filename).stem}{MASK_SUFFIX}"


def bucket_geometry(
    w: int, h: int, mgr: BucketManager,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Mirrors ImageDataset.get_with_flip: bucket -> resize-cover -> center-crop.

    Returns (bucket_size, resize_size, crop_lt).
    """
    tw, th = mgr.get_bucket(w, h)
    scale = max(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return (tw, th), (nw, nh), (left, top)


def transform_mask(
    mask: Image.Image, resize_size: tuple[int, int],
    crop_lt: tuple[int, int], bucket: tuple[int, int],
) -> Image.Image:
    """Apply the same geometric transform to the mask as the image (NEAREST)."""
    nw, nh = resize_size
    left, top = crop_lt
    tw, th = bucket
    m = mask.resize((nw, nh), Image.NEAREST)
    return m.crop((left, top, left + tw, top + th))


def downsample_to_latent(mask: Image.Image) -> Image.Image:
    """Area (BOX = block averaging) downsample to /8 latent resolution (B2 §9 decision 4)."""
    lw = max(1, mask.width // VAE_DOWNSAMPLE)
    lh = max(1, mask.height // VAE_DOWNSAMPLE)
    return mask.resize((lw, lh), Image.BOX)


def mean_gray(img: Image.Image) -> float:
    hist = img.histogram()
    total = sum(hist)
    if total == 0:
        return 0.0
    return sum(v * n for v, n in enumerate(hist)) / total / 255.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("train_dir", type=Path)
    ap.add_argument("--reso", type=int, default=1024)
    ap.add_argument("--save-vis", type=Path, default=None)
    args = ap.parse_args()

    train_dir: Path = args.train_dir
    if not train_dir.is_dir():
        print(f"[error] train_dir does not exist: {train_dir}")
        return 1

    mgr = BucketManager(args.reso)
    vis_dir: Path | None = args.save_vis
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_masked = 0
    bad: list[str] = []

    for sub in sorted(train_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            n_images += 1
            rel = f"{sub.name}/{f.name}"
            try:
                with Image.open(f) as im:
                    w, h = im.size
            except (OSError, ValueError) as exc:
                bad.append(f"{rel}: image unreadable ({exc})")
                continue

            mp = mask_path_for(train_dir, sub.name, f.name)
            if not mp.is_file():
                print(f"  {rel:<40} {w}x{h}  mask=none (whole image trains normally)")
                continue

            try:
                with Image.open(mp) as raw:
                    raw.load()
                    mask = raw.convert("L") if raw.mode != "L" else raw.copy()
            except (OSError, ValueError) as exc:
                bad.append(f"{rel}: mask unreadable ({exc})")
                continue

            if mask.size != (w, h):
                bad.append(
                    f"{rel}: mask size {mask.size[0]}x{mask.size[1]} != image {w}x{h}"
                    " (trainer will downgrade this to no-mask)"
                )
                continue

            n_masked += 1
            coverage = 1.0 - mean_gray(mask)
            bucket, resize_size, crop_lt = bucket_geometry(w, h, mgr)
            m_bucket = transform_mask(mask, resize_size, crop_lt, bucket)
            m_latent = downsample_to_latent(m_bucket)
            latent_weight = mean_gray(m_latent)
            print(
                f"  {rel:<40} {w}x{h}  mask=yes  coverage {coverage * 100:5.1f}%  "
                f"bucket {bucket[0]}x{bucket[1]} -> latent {m_latent.width}x{m_latent.height}  "
                f"latent weight mean {latent_weight:.3f}"
            )

            if vis_dir:
                stem = Path(f.name).stem
                with Image.open(f) as im:
                    base = im.convert("RGB")
                red = Image.new("RGB", base.size, (255, 45, 45))
                # alpha = degree of "not learned" (255 - gray) times display intensity
                alpha = mask.point(lambda v: int((255 - v) * 0.45))
                overlay = base.copy()
                overlay.paste(red, (0, 0), alpha)
                overlay.save(vis_dir / f"{stem}.overlay.png")
                m_latent.resize(bucket, Image.NEAREST).save(
                    vis_dir / f"{stem}.latent.png"
                )

    print()
    print(f"[summary] {n_images} images, {n_masked} of which have a valid mask")
    if bad:
        print(f"[warn] {len(bad)} anomalies (the trainer will downgrade these to no-mask instead of crashing):")
        for line in bad:
            print(f"  ⚠ {line}")
        return 1
    print("[ok] mask data pipeline fully passed: size alignment -> bucket transform -> latent /8 area downsample")
    return 0


if __name__ == "__main__":
    sys.exit(main())
