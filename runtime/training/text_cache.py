"""Shared storage infrastructure for model-family text-conditioning caches (multi-model Phase 2).

This module only handles the cache protocol and disk I/O; it knows nothing
about tokenizers / text encoders and doesn't import any family -- encoding
behavior is still owned by ``ModelFamily.prepare_text_cache``.

Protocol (docs/design/multi-model/04-synthesis.md D19):

- Image caption caches sit next to the image, as a ``<full image name>.text.safetensors`` sidecar;
- the key is derived from the final caption, the TE fingerprint, and the format version;
- tensors keep their variable token length, not padded to 512;
- sample / negative prompts are aggregated into a single safetensors file
  under the task profile root (``tasks/<id>/.text-cache/``) -- not inside the
  dataset's train/, to avoid being mistaken for a concept folder during
  dataset scanning (D19 revision);
- writes use a sibling tmp file + ``os.replace``, so an interrupted run never
  leaves a half-written file.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional


TEXT_CACHE_FORMAT_VERSION = 1
_SIDECAR_SUFFIX = ".text.safetensors"
_PROMPT_CACHE_DIR = ".text-cache"
_PROMPT_CACHE_PREFIX = "prompts"
_TENSOR_SEPARATOR = "::"


@dataclass(frozen=True)
class TextCacheEntry:
    """An image's final caption and the location of its sidecar."""

    image_path: Path
    caption: str
    cache_path: Path

    @classmethod
    def for_image(cls, image_path, caption: str) -> "TextCacheEntry":
        image = Path(image_path)
        return cls(
            image_path=image,
            caption=str(caption),
            cache_path=caption_sidecar_path(image),
        )


def caption_sha256(caption: str) -> str:
    """Hash of the final caption content (an independent invalidation component, D3/D19)."""

    return hashlib.sha256(str(caption).encode("utf-8")).hexdigest()


def text_cache_key(
    caption: str,
    text_fingerprint: str,
    *,
    format_version: int = TEXT_CACHE_FORMAT_VERSION,
) -> str:
    """Return a stable content key derived from caption + TE fingerprint + format version."""

    payload = (
        f"text-cache-v{int(format_version)}\0{text_fingerprint}\0{caption}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def caption_sidecar_path(image_path) -> Path:
    """The image caption sidecar path; keeps the original extension to avoid ``a.jpg``/``a.png`` collisions."""

    image = Path(image_path)
    return image.with_name(image.name + _SIDECAR_SUFFIX)


def prompt_cache_path(root, text_fingerprint: str) -> Path:
    """Path to the aggregated sample/negative prompt cache (isolated by TE fingerprint).

    ``root`` is decided by the caller -- training passes the task profile root
    (``tasks/<id>/``), and the aggregate file lands under ``<root>/.text-cache/``.
    """

    fp_short = hashlib.sha256(str(text_fingerprint).encode("utf-8")).hexdigest()[:12]
    return (
        Path(root)
        / _PROMPT_CACHE_DIR
        / f"{_PROMPT_CACHE_PREFIX}.{fp_short}.safetensors"
    )


def _normalise_tensors(tensors: Mapping[str, object]) -> dict:
    import torch

    if not tensors:
        raise ValueError("text cache needs at least one tensor")
    out = {}
    for name, value in tensors.items():
        key = str(name)
        if not key or _TENSOR_SEPARATOR in key:
            raise ValueError(f"invalid text cache tensor name: {name!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"text cache value must be a torch.Tensor: {key}")
        out[key] = value.detach().cpu().contiguous()
    return out


def _atomic_save(
    tensors: Mapping[str, object], metadata: Mapping[str, str], path: Path,
) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    # The same dataset may be pre-cached concurrently by two training tasks:
    # the tmp name includes pid/thread, and os.replace still resolves the
    # final file to whichever complete version wins the race, without either
    # one truncating the other's temp file.
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        save_file(dict(tensors), str(tmp_path), metadata=dict(metadata))
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _load(path: Path) -> tuple[dict, dict[str, str]]:
    from safetensors import safe_open

    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors, metadata


class TextCacheStore:
    """Reads/writes sidecars / prompt bundles for one bound TE fingerprint."""

    def __init__(
        self,
        text_fingerprint: str,
        *,
        format_version: int = TEXT_CACHE_FORMAT_VERSION,
    ) -> None:
        fingerprint = str(text_fingerprint).strip()
        if not fingerprint:
            raise ValueError("text_fingerprint must not be empty")
        self.text_fingerprint = fingerprint
        self.format_version = int(format_version)

    def key(self, caption: str) -> str:
        return text_cache_key(
            caption,
            self.text_fingerprint,
            format_version=self.format_version,
        )

    def write_caption(self, entry: TextCacheEntry, tensors: Mapping[str, object]) -> None:
        payload = _normalise_tensors(tensors)
        metadata = {
            "cache_kind": "caption_sidecar",
            "format_version": str(self.format_version),
            "text_fingerprint": self.text_fingerprint,
            "caption_sha256": caption_sha256(entry.caption),
            "cache_key": self.key(entry.caption),
        }
        _atomic_save(payload, metadata, entry.cache_path)

    def read_caption(self, entry: TextCacheEntry) -> Optional[dict]:
        """Returns CPU tensors on a hit; returns ``None`` if missing, corrupt, or any fingerprint mismatches."""

        if not entry.cache_path.is_file():
            return None
        try:
            tensors, metadata = _load(entry.cache_path)
        except Exception:
            return None
        expected = {
            "cache_kind": "caption_sidecar",
            "format_version": str(self.format_version),
            "text_fingerprint": self.text_fingerprint,
            "caption_sha256": caption_sha256(entry.caption),
            "cache_key": self.key(entry.caption),
        }
        if any(metadata.get(k) != v for k, v in expected.items()):
            return None
        return tensors or None

    def get_or_encode_caption(
        self,
        entry: TextCacheEntry,
        encoder: Callable[[str], Mapping[str, object]],
    ) -> tuple[dict, bool]:
        """Reads the sidecar; on a miss, encodes and atomically overwrites it. Returns ``(tensors, was_hit)``."""

        cached = self.read_caption(entry)
        if cached is not None:
            return cached, True
        encoded = _normalise_tensors(encoder(entry.caption))
        self.write_caption(entry, encoded)
        return encoded, False

    def write_prompt_bundle(
        self,
        root,
        encoded: Mapping[str, Mapping[str, object]],
    ) -> Path:
        """Atomically overwrite the aggregated sample/negative prompt cache; tensors may each have a different length."""

        payload = {}
        metadata = {
            "cache_kind": "prompt_bundle",
            "format_version": str(self.format_version),
            "text_fingerprint": self.text_fingerprint,
        }
        for caption, tensors in encoded.items():
            digest = self.key(str(caption))
            normalised = _normalise_tensors(tensors)
            metadata[f"entry.{digest}"] = caption_sha256(str(caption))
            for name, tensor in normalised.items():
                payload[f"{digest}{_TENSOR_SEPARATOR}{name}"] = tensor
        path = prompt_cache_path(root, self.text_fingerprint)
        if payload:
            _atomic_save(payload, metadata, path)
        else:
            path.unlink(missing_ok=True)
        return path

    def get_or_encode_prompts(
        self,
        root,
        captions,
        encoder: Callable[[str], Mapping[str, object]],
    ) -> tuple[dict[str, dict], int]:
        """Batch-reads the aggregate cache and encodes misses; returns ``(caption -> tensors, hit_count)``."""

        unique = list(dict.fromkeys(str(caption) for caption in captions))
        encoded: dict[str, dict] = {}
        hit_count = 0
        dirty = False
        for caption in unique:
            cached = self.read_prompt(root, caption)
            if cached is not None:
                encoded[caption] = cached
                hit_count += 1
                continue
            encoded[caption] = _normalise_tensors(encoder(caption))
            dirty = True
        if dirty:
            self.write_prompt_bundle(root, encoded)
        return encoded, hit_count

    def read_prompt(self, root, caption: str) -> Optional[dict]:
        path = prompt_cache_path(root, self.text_fingerprint)
        if not path.is_file():
            return None
        try:
            tensors, metadata = _load(path)
        except Exception:
            return None
        digest = self.key(caption)
        if (
            metadata.get("cache_kind") != "prompt_bundle"
            or metadata.get("format_version") != str(self.format_version)
            or metadata.get("text_fingerprint") != self.text_fingerprint
            or metadata.get(f"entry.{digest}") != caption_sha256(caption)
        ):
            return None
        prefix = f"{digest}{_TENSOR_SEPARATOR}"
        found = {
            key[len(prefix):]: value
            for key, value in tensors.items()
            if key.startswith(prefix)
        }
        return found or None
