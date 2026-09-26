"""Local upload service: accept_one / accept_many."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from studio.services.dataset import uploads


def _png_bytes(size: tuple[int, int] = (4, 4), color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _jpg_bytes(size: tuple[int, int] = (4, 4), color: tuple[int, int, int] = (0, 128, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _rgba_png_bytes(size: tuple[int, int] = (4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, (10, 20, 30, 100)).save(buf, "PNG")
    return buf.getvalue()


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_accept_single_jpg(tmp_path: Path) -> None:
    out = uploads.accept_one("photo.jpg", io.BytesIO(b"\xff\xd8jpgdata"), tmp_path)
    assert out.added == ["photo.jpg"]
    assert out.skipped == []
    assert (tmp_path / "photo.jpg").read_bytes() == b"\xff\xd8jpgdata"


def test_accept_png_uppercase_ext(tmp_path: Path) -> None:
    out = uploads.accept_one("a.PNG", io.BytesIO(b"png"), tmp_path)
    assert out.added == ["a.PNG"]
    assert (tmp_path / "a.PNG").exists()


def test_reject_unsupported_format(tmp_path: Path) -> None:
    out = uploads.accept_one("note.bin", io.BytesIO(b"hi"), tmp_path)
    assert out.added == []
    assert len(out.skipped) == 1
    assert out.skipped[0]["name"] == "note.bin"
    assert "格式不支持" in out.skipped[0]["reason"]


def test_lone_caption_txt_skipped(tmp_path: Path) -> None:
    """A lone uploaded .txt (no matching image) is skipped and reported, not written to disk."""
    out = uploads.accept_one("note.txt", io.BytesIO(b"hi"), tmp_path)
    assert out.added == []
    assert len(out.skipped) == 1
    assert out.skipped[0]["name"] == "note.txt"
    assert "无对应图片" in out.skipped[0]["reason"]
    assert not (tmp_path / "note.txt").exists()


def test_accepts_extended_image_formats(tmp_path: Path) -> None:
    """PP10: the upload whitelist matches the pipeline-wide IMAGE_EXTS; webp/bmp/gif are also accepted."""
    for fname, payload in [
        ("a.webp", b"WEBP"),
        ("b.bmp", b"BMP"),
        ("c.gif", b"GIF"),
    ]:
        out = uploads.accept_one(fname, io.BytesIO(payload), tmp_path)
        assert out.added == [fname], f"{fname} should be accepted"
        assert (tmp_path / fname).read_bytes() == payload


def test_skip_existing_does_not_overwrite(tmp_path: Path) -> None:
    (tmp_path / "p.png").write_bytes(b"old")
    out = uploads.accept_one("p.png", io.BytesIO(b"new"), tmp_path)
    assert out.added == []
    assert out.skipped[0]["reason"] == "已存在，跳过"
    assert (tmp_path / "p.png").read_bytes() == b"old"


def test_zip_extracts_jpg_png(tmp_path: Path) -> None:
    blob = _zip_bytes(
        {
            "a.jpg": b"AA",
            "sub/b.png": b"BB",
            "ignored.txt": b"X",
        }
    )
    out = uploads.accept_one("pack.zip", io.BytesIO(blob), tmp_path)
    assert sorted(out.added) == ["a.jpg", "b.png"]
    # txt is skipped; subdirectories are flattened
    assert any("ignored.txt" in s["name"] for s in out.skipped)
    assert (tmp_path / "a.jpg").read_bytes() == b"AA"
    assert (tmp_path / "b.png").read_bytes() == b"BB"
    # should not create a sub/ subdirectory
    assert not (tmp_path / "sub").exists()


def test_zip_skip_dup_in_zip(tmp_path: Path) -> None:
    (tmp_path / "x.png").write_bytes(b"existing")
    blob = _zip_bytes({"x.png": b"new"})
    out = uploads.accept_one("p.zip", io.BytesIO(blob), tmp_path)
    assert out.added == []
    assert out.skipped[0]["reason"] == "已存在，跳过"
    assert (tmp_path / "x.png").read_bytes() == b"existing"


def test_corrupt_zip_skipped(tmp_path: Path) -> None:
    out = uploads.accept_one(
        "broken.zip", io.BytesIO(b"not-a-real-zip"), tmp_path
    )
    assert out.added == []
    assert out.skipped[0]["reason"] == "zip 损坏"


def test_zip_path_traversal_flattened(tmp_path: Path) -> None:
    """When a zip entry contains a ../ or absolute path segment, only the basename is used — it never escapes dest_dir."""
    blob = _zip_bytes(
        {
            "../escape.jpg": b"E",
            "/abs/p.png": b"P",
        }
    )
    out = uploads.accept_one("evil.zip", io.BytesIO(blob), tmp_path)
    assert sorted(out.added) == ["escape.jpg", "p.png"]
    assert (tmp_path / "escape.jpg").exists()
    assert (tmp_path / "p.png").exists()
    # should not write outside tmp_path
    assert not (tmp_path.parent / "escape.jpg").exists()


def test_accept_many_aggregates(tmp_path: Path) -> None:
    files = [
        ("a.jpg", io.BytesIO(b"A")),
        ("b.txt", io.BytesIO(b"B")),
        ("c.png", io.BytesIO(b"C")),
    ]
    out = uploads.accept_many(files, tmp_path)
    assert sorted(out.added) == ["a.jpg", "c.png"]
    assert len(out.skipped) == 1
    assert out.skipped[0]["name"] == "b.txt"


def test_empty_filename_skipped(tmp_path: Path) -> None:
    out = uploads.accept_one("", io.BytesIO(b"x"), tmp_path)
    assert out.added == []
    assert out.skipped[0]["reason"] == "文件名为空"


def test_dest_dir_created(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "download"
    out = uploads.accept_one("x.jpg", io.BytesIO(b"x"), target)
    assert out.added == ["x.jpg"]
    assert (target / "x.jpg").exists()


# ---------------------------------------------------------------------------
# convert_to_png mode: shares the gelbooru.convert_to_png setting with booru downloads
# ---------------------------------------------------------------------------


def test_convert_jpg_renamed_to_png(tmp_path: Path) -> None:
    out = uploads.accept_one(
        "photo.jpg", io.BytesIO(_jpg_bytes()), tmp_path,
        convert_to_png=True,
    )
    assert out.added == ["photo.png"]
    assert (tmp_path / "photo.png").exists()
    # re-encoded into a valid PNG
    with Image.open(tmp_path / "photo.png") as im:
        assert im.format == "PNG"


def test_convert_same_stem_collision_gets_suffix(tmp_path: Path) -> None:
    """1.png + 1.jpg uploaded together: the second one collides after conversion -> gets a _1 suffix, avoiding a shared caption."""
    files = [
        ("1.png", io.BytesIO(_png_bytes(color=(0, 0, 0)))),
        ("1.jpg", io.BytesIO(_jpg_bytes(color=(255, 255, 255)))),
    ]
    out = uploads.accept_many(files, tmp_path, convert_to_png=True)
    assert sorted(out.added) == ["1.png", "1_1.png"]
    assert out.skipped == []
    assert (tmp_path / "1.png").exists()
    assert (tmp_path / "1_1.png").exists()


def test_convert_collision_in_zip_gets_suffix(tmp_path: Path) -> None:
    blob = _zip_bytes(
        {
            "a.png": _png_bytes(color=(10, 10, 10)),
            "a.jpg": _jpg_bytes(color=(200, 200, 200)),
        }
    )
    out = uploads.accept_one(
        "pack.zip", io.BytesIO(blob), tmp_path,
        convert_to_png=True,
    )
    assert sorted(out.added) == ["a.png", "a_1.png"]
    assert (tmp_path / "a.png").exists()
    assert (tmp_path / "a_1.png").exists()


def test_convert_collision_against_existing_file(tmp_path: Path) -> None:
    """The target dir already has a.png; a newly uploaded a.jpg lands as a_1.png in convert mode (not skipped)."""
    (tmp_path / "a.png").write_bytes(_png_bytes())
    out = uploads.accept_one(
        "a.jpg", io.BytesIO(_jpg_bytes()), tmp_path,
        convert_to_png=True,
    )
    assert out.added == ["a_1.png"]
    assert (tmp_path / "a_1.png").exists()


def test_convert_corrupt_image_skipped(tmp_path: Path) -> None:
    out = uploads.accept_one(
        "broken.jpg", io.BytesIO(b"not-a-real-image"), tmp_path,
        convert_to_png=True,
    )
    assert out.added == []
    assert len(out.skipped) == 1
    assert "图片损坏" in out.skipped[0]["reason"]


def test_convert_remove_alpha_channel_flattens(tmp_path: Path) -> None:
    out = uploads.accept_one(
        "rgba.png", io.BytesIO(_rgba_png_bytes()), tmp_path,
        convert_to_png=True,
        remove_alpha_channel=True,
    )
    assert out.added == ["rgba.png"]
    with Image.open(tmp_path / "rgba.png") as im:
        assert im.mode == "RGB"  # alpha has been flattened onto a white background


def test_convert_keeps_alpha_when_flag_off(tmp_path: Path) -> None:
    out = uploads.accept_one(
        "rgba.png", io.BytesIO(_rgba_png_bytes()), tmp_path,
        convert_to_png=True,
        remove_alpha_channel=False,
    )
    assert out.added == ["rgba.png"]
    with Image.open(tmp_path / "rgba.png") as im:
        assert im.mode == "RGBA"


def test_convert_off_preserves_raw_bytes(tmp_path: Path) -> None:
    """convert_to_png=False (default) keeps copying with the original extension, skipping when the target already exists."""
    raw = b"\xff\xd8not-decoded"
    out = uploads.accept_one("photo.jpg", io.BytesIO(raw), tmp_path)
    assert out.added == ["photo.jpg"]
    assert (tmp_path / "photo.jpg").read_bytes() == raw


# ---------------------------------------------------------------------------
# caption pairing (kohya_ss / sd-scripts style .txt sidecar)
# ---------------------------------------------------------------------------


def test_zip_pairs_txt_caption_with_image(tmp_path: Path) -> None:
    """zip contains png + a same-stem .txt -> the caption is written alongside the image."""
    blob = _zip_bytes({"a.png": b"AA", "a.txt": b"1girl, solo"})
    out = uploads.accept_one("pack.zip", io.BytesIO(blob), tmp_path)
    assert sorted(out.added) == ["a.png", "a.txt"]
    assert out.skipped == []
    assert (tmp_path / "a.png").read_bytes() == b"AA"
    assert (tmp_path / "a.txt").read_bytes() == b"1girl, solo"


def test_zip_orphan_txt_skipped(tmp_path: Path) -> None:
    """zip has an image but the .txt stem doesn't match -> the caption is skipped, not written to disk."""
    blob = _zip_bytes({"a.png": b"AA", "other.txt": b"x"})
    out = uploads.accept_one("pack.zip", io.BytesIO(blob), tmp_path)
    assert out.added == ["a.png"]
    assert any(
        "other.txt" in s["name"] and "无对应图片" in s["reason"]
        for s in out.skipped
    )
    assert not (tmp_path / "other.txt").exists()


def test_accept_many_pairs_loose_png_and_txt(tmp_path: Path) -> None:
    """png + txt dropped in the same batch (not in the same zip) are also paired by stem."""
    files = [
        ("1.png", io.BytesIO(b"P")),
        ("1.txt", io.BytesIO(b"tag-a, tag-b")),
    ]
    out = uploads.accept_many(files, tmp_path)
    assert sorted(out.added) == ["1.png", "1.txt"]
    assert out.skipped == []
    assert (tmp_path / "1.txt").read_bytes() == b"tag-a, tag-b"


def test_caption_follows_png_conversion_stem(tmp_path: Path) -> None:
    """When convert_to_png turns jpg into png, the caption follows the stem it was written under."""
    files = [
        ("photo.jpg", io.BytesIO(_jpg_bytes())),
        ("photo.txt", io.BytesIO(b"masterpiece")),
    ]
    out = uploads.accept_many(files, tmp_path, convert_to_png=True)
    assert sorted(out.added) == ["photo.png", "photo.txt"]
    assert (tmp_path / "photo.png").exists()
    assert (tmp_path / "photo.txt").read_bytes() == b"masterpiece"


def test_ingest_paths_streams_zip_and_pairs_caption(tmp_path: Path) -> None:
    """ingest_paths (the streaming entry point used by the worker): zip's png+txt -> paired and written to disk."""
    src = tmp_path / "src"
    src.mkdir()
    blob = _zip_bytes({"a.png": b"AA", "a.txt": b"tag1, tag2", "b.jpg": b"BB"})
    (src / "pack.zip").write_bytes(blob)
    dest = tmp_path / "download"
    out = uploads.ingest_paths([src / "pack.zip"], dest)
    assert sorted(out.added) == ["a.png", "a.txt", "b.jpg"]
    assert (dest / "a.txt").read_bytes() == b"tag1, tag2"
    assert (dest / "a.png").read_bytes() == b"AA"


def test_ingest_paths_pairs_loose_files_across_sources(tmp_path: Path) -> None:
    """Loose file paths: 1.png + 1.txt from different sources are still paired by stem."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "1.png").write_bytes(b"P")
    (src / "1.txt").write_bytes(b"caption")
    (src / "lonely.txt").write_bytes(b"orphan")
    dest = tmp_path / "download"
    out = uploads.ingest_paths(
        [src / "1.png", src / "1.txt", src / "lonely.txt"], dest
    )
    assert sorted(out.added) == ["1.png", "1.txt"]
    assert (dest / "1.txt").read_bytes() == b"caption"
    assert any("lonely.txt" in s["name"] for s in out.skipped)
    assert not (dest / "lonely.txt").exists()


def test_ingest_paths_progress_callback(tmp_path: Path) -> None:
    """on_progress receives per-item lines plus a summary line (the worker relays stdout to the frontend)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.png").write_bytes(b"A")
    lines: list[str] = []
    uploads.ingest_paths([src / "a.png"], tmp_path / "d", on_progress=lines.append)
    assert any(l.startswith("[add]") for l in lines)
    assert any(l.startswith("[summary]") for l in lines)


def test_caption_follows_suffixed_collision(tmp_path: Path) -> None:
    """1.png + 1.jpg + 1.txt (convert mode): the caption goes to whichever image was written first; the suffixed copy doesn't take it."""
    files = [
        ("1.png", io.BytesIO(_png_bytes(color=(0, 0, 0)))),
        ("1.jpg", io.BytesIO(_jpg_bytes(color=(255, 255, 255)))),
        ("1.txt", io.BytesIO(b"caption-for-first")),
    ]
    out = uploads.accept_many(files, tmp_path, convert_to_png=True)
    assert sorted(out.added) == ["1.png", "1.txt", "1_1.png"]
    # caption lands on the first image (1.png); the suffixed 1_1.png has no caption
    assert (tmp_path / "1.txt").read_bytes() == b"caption-for-first"
    assert not (tmp_path / "1_1.txt").exists()
