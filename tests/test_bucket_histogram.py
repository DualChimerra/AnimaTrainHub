"""compute_bucket_histogram: the backend uses a real BucketManager to compute the
training set's bucket distribution (the data source for the bucket preview).

Reuses runtime's BucketManager + _parse_folder_meta; the scan rules mirror
ImageDataset._scan (recursive rglob + loose images at the root + only images that
have a caption are counted), keeping this in lockstep with the actual training buckets.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _sq(d: Path, names, size=(1024, 1024), caption=True) -> None:
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.new("RGB", size).save(d / f"{n}.png")
        if caption:
            (d / f"{n}.txt").write_text("1girl", encoding="utf-8")


def test_single_resolution_repeat_counts(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_bucket_histogram
    _sq(tmp_path / "5_data", ["a", "b"])
    out = compute_bucket_histogram(tmp_path, [1024], 2.0)
    assert len(out) == 1 and out[0]["reso"] == 1024
    assert sum(b["count"] for b in out[0]["buckets"]) == 10  # 2 images x repeat 5
    sq = next(b for b in out[0]["buckets"] if b["w"] == 1024 and b["h"] == 1024)
    assert sq["count"] == 10


def test_px_folder_override_resolution(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_bucket_histogram
    _sq(tmp_path / "512px_2_hi", ["a"])
    out = compute_bucket_histogram(tmp_path, [1024], 2.0)
    assert [g["reso"] for g in out] == [512]  # px override -> lands in the 512 bucket group, not 1024
    assert sum(b["count"] for b in out[0]["buckets"]) == 2  # 1 image x repeat 2


def test_resolution_list_fans_out(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_bucket_histogram
    _sq(tmp_path / "data", ["a"])
    out = compute_bucket_histogram(tmp_path, [512, 768, 1024], 2.0)
    assert sorted(g["reso"] for g in out) == [512, 768, 1024]
    for g in out:
        assert sum(b["count"] for b in g["buckets"]) == 1  # 1 image per group


def test_empty_train_dir(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_bucket_histogram
    assert compute_bucket_histogram(tmp_path / "nope", [1024], 2.0) == []


def test_only_captioned_images_counted(tmp_path: Path) -> None:
    # mirrors the trainer: images without a caption are discarded and not counted in the histogram (otherwise the preview != actual training)
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_bucket_histogram
    _sq(tmp_path / "1_data", ["a", "b"])               # has a caption
    _sq(tmp_path / "1_data", ["c"], caption=False)     # no caption
    out = compute_bucket_histogram(tmp_path, [1024], 2.0)
    assert sum(b["count"] for b in out[0]["buckets"]) == 2  # c is not counted


def test_dataset_importable_without_runtime_on_sys_path() -> None:
    """studio server's sys.path only has the repo root (no runtime/) -- dataset.py
    and its import chain must be importable under the `runtime.training.*` name,
    otherwise the bucket-distribution endpoint 500s. conftest injects runtime/ into
    sys.path, so an in-process test can't catch this; a clean subprocess is required.
    Regression: the `from training.*` absolute import introduced by multi-model PR-1
    (#405) once broke this chain (fix: the families subtree switched to relative imports)."""
    pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parent.parent
    code = (
        "import sys; "
        "sys.path[:] = [p for p in sys.path if 'runtime' not in p.lower()]; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "from runtime.training.dataset import BucketManager, ImageDataset; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "ok" in proc.stdout


def test_root_and_nested_images_counted(tmp_path: Path) -> None:
    # 根目录散图（repeat=1）+ 子文件夹深层图（rglob）都要计，跟 _scan 一致。
    pytest.importorskip("torch")
    from studio.services.projects.versions import compute_bucket_histogram
    _sq(tmp_path, ["root"])                              # 根目录散图
    _sq(tmp_path / "2_data" / "nested", ["deep"])        # 子文件夹深层
    out = compute_bucket_histogram(tmp_path, [1024], 2.0)
    total = sum(b["count"] for g in out for b in g["buckets"])
    assert total == 1 + 2  # root×1 + deep×repeat2
