"""WD14 ONNX tagging (PP4).

Model resolution order:
    1. if models/wd14/{model_id}/ exists -> use the local copy
    2. otherwise, pull it via huggingface_hub.snapshot_download to models/wd14/{model_id}/

Dependencies: onnxruntime (CPU by default; for GPU, the user installs onnxruntime-gpu themselves) +
huggingface_hub + Pillow + numpy.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ... import secrets
from .. import models as model_downloader
from .onnx_base import OnnxTaggerBase


def _blacklist_key(tag: str) -> str:
    """Normalization key for blacklist matching: insensitive to underscore/space, case, and leading/trailing whitespace."""
    return tag.replace("_", " ").strip().lower()


class WD14Tagger(OnnxTaggerBase):
    name = "wd14"

    def __init__(self, overrides: dict | None = None) -> None:
        """`overrides` are a temporary override for this tagging run only (in-memory effect only).

        Merged from the same-named fields on `secrets.WD14Config` (`threshold_general` /
        `threshold_character` / `model_id` / `blacklist_tags`);
        a field set to None falls back to the global settings, and none of this touches the secrets.json file.
        """
        super().__init__()
        self._overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
        self._tags: list[str] = []
        self._tag_categories: list[int] = []  # 0=general, 4=character, 9=rating
        self._input_size: int = 448  # default; overwritten during prepare

    # -------------------- config --------------------

    def _cfg(self) -> "secrets.WD14Config":
        """Merges global secrets + this run's overrides into the effective config for this run."""
        base = secrets.load().wd14.model_dump()
        for k, v in self._overrides.items():
            if k in base:
                base[k] = v
        return secrets.WD14Config(**base)

    def _get_batch_size_cfg(self) -> int:
        return int(self._cfg().batch_size or 1)

    # -------------------- model resolution --------------------

    def _local_model_dir_status(self) -> tuple[Path, bool]:
        cfg = self._cfg()
        d = model_downloader.wd14_target_dir(model_downloader.models_root(), cfg.model_id)
        ok = (d / "model.onnx").exists() and (d / "selected_tags.csv").exists()
        return d, ok

    def _resolve_model_dir(self) -> Path:
        cfg = self._cfg()
        default = model_downloader.wd14_target_dir(model_downloader.models_root(), cfg.model_id)
        if (default / "model.onnx").exists() and (default / "selected_tags.csv").exists():
            return default
        return self._download_model(cfg.model_id, default)

    def _download_model(self, model_id: str, target: Path) -> Path:
        from huggingface_hub import snapshot_download
        token = secrets.load().huggingface.token or None
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            allow_patterns=["model.onnx", "selected_tags.csv"],
            token=token,
        )
        return target

    # -------------------- protocol --------------------

    def is_available(self) -> tuple[bool, str]:
        try:
            d, ok = self._local_model_dir_status()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        if ok:
            return True, f"Model: {d.name}"
        return False, f"Model needs downloading: {d.name}"

    def prepare(self) -> None:
        if self._session is not None:
            return
        model_dir = self._resolve_model_dir()
        self._create_session(model_dir / "model.onnx")
        # Input: usually [N, H, W, C]; H==W; a dynamic symbol falls back to the default 448
        assert self._session is not None
        ish = self._session.get_inputs()[0].shape
        for dim in ish[1:]:
            if isinstance(dim, int) and dim > 0:
                self._input_size = dim
                break

        with open(model_dir / "selected_tags.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # SmilingWolf models use underscores; the UI is used to spaces
                self._tags.append(row["name"].replace("_", " "))
                self._tag_categories.append(int(row.get("category", 0)))

    def known_tags(self) -> list[str]:
        """The model's vocabulary (in space-lowercased form); usable after `prepare()`.
        eval tag-recall uses it to filter prompt tags "WD14 recognizes".
        """
        return list(self._tags)

    # -------------------- inference --------------------

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        """A single image -> [H, W, 3] BGR float32. When batch inferencing, the caller is responsible for stacking into [N, H, W, 3]."""
        size = self._input_size
        img = ImageOps.exif_transpose(img) or img
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Scale proportionally to size so the long edge == size, then pad to a square with white
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
        arr = np.asarray(canvas, dtype=np.float32)
        # WD14 was trained with BGR
        arr = arr[..., ::-1]
        return arr

    def _postprocess_one(
        self, scores: np.ndarray
    ) -> tuple[list[str], dict[str, float]]:
        """A single image's probability vector -> (sorted_tags, raw_scores_dict)."""
        cfg = self._cfg()
        out: list[tuple[str, float]] = []
        # Blacklist matching normalization: insensitive to underscore/space, case, and leading/trailing whitespace
        # (cat girl / cat_girl / Cat_Girl are equivalent). self._tags is already in space-lowercased form.
        blacklist = {_blacklist_key(b) for b in cfg.blacklist_tags}
        for i, p in enumerate(scores):
            if i >= len(self._tags):
                break
            tag, cat = self._tags[i], self._tag_categories[i]
            if _blacklist_key(tag) in blacklist:
                continue
            # category: 9=rating (doesn't participate in thresholding, discarded); 4=character; everything else follows general
            if cat == 9:
                continue
            thr = cfg.threshold_character if cat == 4 else cfg.threshold_general
            p_f = float(p)
            if p_f >= thr:
                out.append((tag, p_f))
        out.sort(key=lambda x: -x[1])
        return [t for t, _ in out], dict(out)
