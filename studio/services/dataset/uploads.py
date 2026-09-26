"""Local upload: single image / zip archive (auto-extracted and flattened) / same-named .txt caption, saved to download/.

The user uploads files via the browser's file picker / drag-and-drop -> the
server parses them -> writes into the project's `download/` directory,
sharing the same "full backup" as booru downloads.

Constraints:
- Accepts single images in any format from IMAGE_EXTS (png / jpg / jpeg /
  webp / bmp / gif); the same whitelist is used to extract files from zip
  archives.
- Subdirectory structure inside a zip gets flattened (basename only).
- A corrupt zip is skipped as a whole and reported, without affecting other
  files.

Caption pairing (kohya_ss / sd-scripts style):
- A `.txt` sharing the image's stem is treated as that image's caption /
  tags (`1.png` <-> `1.txt`);
- A `.txt` appearing in a single upload (inside a zip, or dragged in the same
  batch of multiple files) is paired to the matching image by stem, and
  saved alongside it into download/. Later, curation carries the same-named
  .txt into train/ too, so the tagging stage sees these captions right away
  (caption coverage is satisfied immediately);
- An orphan `.txt` with no matching image is skipped and reported;
- When convert_to_png renames the image / adds a suffix (see below), the
  caption follows the **actual stem after saving**, so the caption for
  `1.jpg` -> `1.png` lands on `1.txt` sharing `1.png`'s stem, and a
  suffixed `1_1.png` won't fight `1.png` for the same caption.

`convert_to_png` mode (shares the `gelbooru.convert_to_png` setting with
booru downloads):
- All images are decoded via PIL and uniformly re-encoded to .png, keeping
  the filename stem but changing the suffix to .png;
- A same-stem conflict (including the case of `1.jpg` + `1.png` both
  uploaded at once) gets a `_1`/`_2` suffix on save instead, so the
  `1.txt` caption isn't shared between two different images;
- `remove_alpha_channel=True` flattens alpha onto a white background,
  matching booru downloads;
- A PIL decode failure is reported as skipped with reason "corrupt image".

`convert_to_png=False` (default) keeps the historical behavior: copied with
the original extension, skipped if the target already exists.
"""
from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Optional

from PIL import Image

from ..booru.api import flatten_alpha, has_alpha
from .scan import IMAGE_EXTS

# Since PP10, reuses the same whitelist across the whole pipeline: upload / download / curation / training all share it.
ALLOWED_IMAGE_EXTS = IMAGE_EXTS
ZIP_EXT = ".zip"
# kohya_ss / sd-scripts style caption: a .txt sharing the image's stem.
# curation's _META_EXTS is also (.txt, .json), so a .txt landing in
# download/ gets carried into train/ along with its image.
CAPTION_EXT = ".txt"


@dataclass
class UploadResult:
    """Summary result of a single upload call.

    ``added`` includes both saved image names and successfully paired caption (`*.txt`) names.
    """

    added: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, list]:
        return {"added": self.added, "skipped": self.skipped}

    def merge(self, other: "UploadResult") -> None:
        self.added.extend(other.added)
        self.skipped.extend(other.skipped)


@dataclass
class _Entry:
    """A leaf file after unpacking (image or caption); read fully into memory before processing."""

    report: str  # user-facing name (a zip entry is prefixed with `pack.zip:`)
    base: str    # basename to save under
    data: bytes


def _safe_basename(name: str) -> str:
    """Strip nested subdirectories inside a zip / Windows backslashes, keeping only the basename."""
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def _is_image_ext(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_IMAGE_EXTS


def _is_caption_ext(name: str) -> bool:
    return Path(name).suffix.lower() == CAPTION_EXT


class _SeekableStream:
    """Patches in the three IOBase predicate methods a stream may be missing (seekable/readable/writable); everything else passes through.

    On Python < 3.11, `SpooledTemporaryFile` (the actual type behind FastAPI's
    `UploadFile.file`) only explicitly forwards read/seek/tell to the
    underlying `_file`, missing these three predicate methods (only fixed in
    3.11 when it started inheriting io.IOBase). `zipfile.ZipFile`'s
    constructor and `infolist()` only use seek/read, so those work fine; but
    `zf.open()` goes through `_SharedFile` to read `fileobj.seekable`, which
    raises AttributeError - so "opening the zip" succeeds, and it only blows
    up once it starts reading an inner image.

    The underlying `_file` (a BytesIO while still spooled, or a real temp
    file after rollover) already supports seek, so a zero-copy wrapper is
    enough, preserving the point of streaming uploads (never reading the
    whole archive into memory).
    """

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def __getattr__(self, name: str):  # seek/read/tell/close etc. pass through to the underlying stream
        return getattr(self._stream, name)


def _ensure_seekable(stream: BinaryIO) -> BinaryIO:
    """zipfile needs `fileobj.seekable()`; older SpooledTemporaryFile doesn't have it -> wrap it.

    A real file / BytesIO / a 3.11+ SpooledTemporaryFile that already implements it is returned as-is, unwrapped.
    """
    probe = getattr(stream, "seekable", None)
    if callable(probe):
        try:
            probe()
            return stream
        except Exception:  # noqa: BLE001 - any exception is treated as "predicate unusable" and wrapped
            pass
    return _SeekableStream(stream)


def _unique_target(dest_dir: Path, name: str) -> Path:
    """When the target already exists, append `_1` / `_2` ... until there's no conflict.

    Used in convert_to_png mode: the user's intent is "this image should come
    in too", so a filename collision keeps the second image under a
    suffixed name instead of dropping it. Captions are paired by the actual
    on-disk stem, so a suffixed `1_1.png` gets its own independent
    `1_1.txt`, no longer sharing one with `1.png`.
    """
    target = dest_dir / name
    if not target.exists():
        return target
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        cand = dest_dir / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def _write_image_entry(
    src_name: str,
    src_stream: BinaryIO,
    dest_dir: Path,
    *,
    convert_to_png: bool,
    remove_alpha_channel: bool,
    report_name: str,
    result: UploadResult,
) -> Optional[Path]:
    """Save a single image. Returns the actual saved path; ``None`` if skipped / corrupt."""
    if not convert_to_png:
        target = dest_dir / src_name
        if target.exists():
            result.skipped.append(
                {"name": report_name, "reason": "already exists, skipped"}
            )
            return None
        with target.open("wb") as fh:
            shutil.copyfileobj(src_stream, fh)
        result.added.append(src_name)
        return target

    raw = src_stream.read()
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001 - PIL raises various types, treat them all as corrupt
        result.skipped.append(
            {"name": report_name, "reason": f"corrupt image: {exc}"}
        )
        return None
    target = _unique_target(dest_dir, Path(src_name).stem + ".png")
    if remove_alpha_channel and has_alpha(img):
        img = flatten_alpha(img)
    out = (
        img.convert("RGBA")
        if has_alpha(img) and not remove_alpha_channel
        else img.convert("RGB")
    )
    out.save(target, "PNG", optimize=True)
    result.added.append(target.name)
    return target


def _expand_input(
    src_name: str, src_stream: BinaryIO
) -> tuple[list[_Entry], list[dict[str, str]]]:
    """Expand a single uploaded item into leaf entries (images / captions) + top-level skipped items.

    - a single image / single .txt -> one entry
    - a zip -> unpack all images + .txt (flattened; inner entries also go into the entry list)
    - anything else / no extension / corrupt zip -> no entries produced, recorded as skipped
    """
    base = _safe_basename(src_name or "")
    if not base:
        return [], [{"name": src_name, "reason": "empty filename"}]

    suffix = Path(base).suffix.lower()

    if suffix in ALLOWED_IMAGE_EXTS or suffix == CAPTION_EXT:
        return [_Entry(base, base, src_stream.read())], []

    if suffix == ZIP_EXT:
        entries: list[_Entry] = []
        skipped: list[dict[str, str]] = []
        try:
            with zipfile.ZipFile(_ensure_seekable(src_stream)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    inner_base = _safe_basename(info.filename)
                    label = f"{base}:{info.filename}"
                    if not inner_base:
                        continue
                    inner_suffix = Path(inner_base).suffix.lower()
                    if inner_suffix in ALLOWED_IMAGE_EXTS or inner_suffix == CAPTION_EXT:
                        entries.append(_Entry(label, inner_base, zf.read(info)))
                    else:
                        skipped.append({"name": label, "reason": "unsupported format"})
        except zipfile.BadZipFile:
            skipped.append({"name": base, "reason": "corrupt zip"})
        return entries, skipped

    allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTS)) + ", .txt, .zip"
    return [], [{"name": base, "reason": f"unsupported format (only {allowed})"}]


def _accept_entries(
    entries: Iterable[_Entry],
    dest_dir: Path,
    *,
    convert_to_png: bool,
    remove_alpha_channel: bool,
) -> UploadResult:
    """Save a batch of already-expanded entries, pairing captions to images by stem.

    Pairing rule: a caption is named after the **actual saved stem of its
    image** (`X.png` -> `X.txt`). When several images share a stem, the
    caption goes to the first one saved successfully (FIFO); a suffixed
    duplicate doesn't claim it. An orphan caption with no matching image is
    skipped and reported.
    """
    result = UploadResult()
    dest_dir.mkdir(parents=True, exist_ok=True)

    items = list(entries)
    images = [e for e in items if _is_image_ext(e.base)]
    captions = [e for e in items if _is_caption_ext(e.base)]

    # queue captions by their uploaded stem, to be claimed FIFO by images sharing that stem.
    cap_by_stem: dict[str, list[_Entry]] = {}
    for cap in captions:
        cap_by_stem.setdefault(Path(cap.base).stem, []).append(cap)

    for img in images:
        final = _write_image_entry(
            img.base, BytesIO(img.data), dest_dir,
            convert_to_png=convert_to_png,
            remove_alpha_channel=remove_alpha_channel,
            report_name=img.report,
            result=result,
        )
        if final is None:
            continue
        queue = cap_by_stem.get(Path(img.base).stem)
        if queue:
            cap = queue.pop(0)
            cap_path = dest_dir / (final.stem + CAPTION_EXT)
            cap_path.write_bytes(cap.data)
            result.added.append(cap_path.name)

    # captions with no matching image (orphan .txt / stem doesn't match) -> skipped and reported.
    for queue in cap_by_stem.values():
        for cap in queue:
            result.skipped.append(
                {"name": cap.report, "reason": "no matching image, caption ignored"}
            )

    return result


def accept_one(
    src_name: str,
    src_stream: BinaryIO,
    dest_dir: Path,
    *,
    convert_to_png: bool = False,
    remove_alpha_channel: bool = False,
) -> UploadResult:
    """处理单个上传文件。

    - IMAGE_EXTS 内任一格式 → 落盘（按 convert_to_png 决定是否重编码 PNG）
    - `.txt` → 当 caption；单独上传无对应图片时跳过
    - zip → 解压所有图片 + .txt（拍平、按 stem 配对 caption）
    - 其他 / 没扩展名 → 拒绝
    """
    entries, skipped = _expand_input(src_name, src_stream)
    result = _accept_entries(
        entries, dest_dir,
        convert_to_png=convert_to_png,
        remove_alpha_channel=remove_alpha_channel,
    )
    result.skipped = skipped + result.skipped
    return result


def accept_many(
    files: Iterable[tuple[str, BinaryIO]],
    dest_dir: Path,
    *,
    convert_to_png: bool = False,
    remove_alpha_channel: bool = False,
) -> UploadResult:
    """批量处理；先把每个输入展开成叶子 entry 汇到一起，再统一落盘 + 配对。

    汇总后再配对的好处：同批拖拽的 `1.png` + `1.txt`（不在同一个 zip 里）也能
    按 stem 配上 caption，而不只是 zip 内部。
    """
    all_entries: list[_Entry] = []
    pre_skipped: list[dict[str, str]] = []
    for name, stream in files:
        entries, skipped = _expand_input(name, stream)
        all_entries.extend(entries)
        pre_skipped.extend(skipped)

    result = _accept_entries(
        all_entries, dest_dir,
        convert_to_png=convert_to_png,
        remove_alpha_channel=remove_alpha_channel,
    )
    result.skipped = pre_skipped + result.skipped
    return result


def _collect_captions_from_zip(
    src: Path, cap_by_stem: dict[str, list[tuple[str, bytes]]]
) -> None:
    """zip 内所有 .txt 读进 caption 表（caption 体积小，全量入内存无压力）。"""
    try:
        with zipfile.ZipFile(src) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner = _safe_basename(info.filename)
                if inner and _is_caption_ext(inner):
                    cap_by_stem.setdefault(Path(inner).stem, []).append(
                        (f"{src.name}:{info.filename}", zf.read(info))
                    )
    except zipfile.BadZipFile:
        pass  # 坏 zip 在 pass-2 落 skipped


def ingest_paths(
    sources: Iterable[Path],
    dest_dir: Path,
    *,
    convert_to_png: bool = False,
    remove_alpha_channel: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> UploadResult:
    """从磁盘上的文件路径批量落盘（图片 / zip / .txt caption），按 stem 配对 caption。

    与 ``accept_many`` 同语义，但**流式**处理：图片逐张读（zip entry 逐个 open），
    只有体积极小的 caption 全量入内存。适合后台 worker 处理大 zip——不会把整包
    解压进 RAM（``accept_many`` 会），从而避免 OOM / 长时间卡顿。

    `on_progress(line)` 可选：worker 把它接到 stdout，前端读 log_tail 看进度。
    """
    result = UploadResult()
    dest_dir.mkdir(parents=True, exist_ok=True)

    def log(line: str) -> None:
        if on_progress is not None:
            on_progress(line)

    srcs = list(sources)

    # pass 1：收集所有来源（散文件 + zip 内）的 caption，按 stem 入 FIFO 队列。
    cap_by_stem: dict[str, list[tuple[str, bytes]]] = {}
    for src in srcs:
        suffix = src.suffix.lower()
        if suffix == CAPTION_EXT:
            cap_by_stem.setdefault(src.stem, []).append((src.name, src.read_bytes()))
        elif suffix == ZIP_EXT:
            _collect_captions_from_zip(src, cap_by_stem)

    def place_image(base: str, stream: BinaryIO, report: str) -> None:
        final = _write_image_entry(
            base, stream, dest_dir,
            convert_to_png=convert_to_png,
            remove_alpha_channel=remove_alpha_channel,
            report_name=report,
            result=result,
        )
        if final is None:
            return
        queue = cap_by_stem.get(Path(base).stem)
        if queue:
            _rep, data = queue.pop(0)
            cap_path = dest_dir / (final.stem + CAPTION_EXT)
            cap_path.write_bytes(data)
            result.added.append(cap_path.name)
        log(f"[add] {final.name}")

    # pass 2：图片流式落盘 + 配 caption；非图非 caption / 坏 zip 记 skipped。
    for src in srcs:
        suffix = src.suffix.lower()
        if suffix in ALLOWED_IMAGE_EXTS:
            with src.open("rb") as fh:
                place_image(src.name, fh, src.name)
        elif suffix == CAPTION_EXT:
            continue  # 已在 pass-1 入表，由对应图片领取
        elif suffix == ZIP_EXT:
            try:
                with zipfile.ZipFile(src) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        inner = _safe_basename(info.filename)
                        label = f"{src.name}:{info.filename}"
                        if not inner:
                            continue
                        inner_suffix = Path(inner).suffix.lower()
                        if inner_suffix in ALLOWED_IMAGE_EXTS:
                            with zf.open(info) as entry:
                                place_image(inner, entry, label)
                        elif inner_suffix == CAPTION_EXT:
                            continue
                        else:
                            result.skipped.append(
                                {"name": label, "reason": "格式不支持"}
                            )
            except zipfile.BadZipFile:
                result.skipped.append({"name": src.name, "reason": "zip 损坏"})
        else:
            allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTS)) + ", .txt, .zip"
            result.skipped.append(
                {"name": src.name, "reason": f"格式不支持（仅 {allowed}）"}
            )

    # 没配上图的 caption → 跳过并报告。
    for queue in cap_by_stem.values():
        for rep, _data in queue:
            result.skipped.append(
                {"name": rep, "reason": "无对应图片，已忽略 caption"}
            )

    log(f"[summary] added={len(result.added)} skipped={len(result.skipped)}")
    return result
