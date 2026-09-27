"""Tagger abstraction + factory (PP4).

Each tagger is its own class exposing the same interface via the `Tagger` protocol:
    name / requires_service / is_available / prepare / tag

The worker gets a name, calls `get_tagger(name)` for an instance, runs `prepare()`, then streams
`tag()` for results. All real I/O (onnx inference / HTTP calls to vLLM) lives in the subclasses;
this module only defines the protocol and factory, to keep mocking easy for tests.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Iterator, Protocol, TypedDict, runtime_checkable

ProgressFn = Callable[[int, int], None]  # (done, total)


class TagResult(TypedDict, total=False):
    image: Path
    tags: list[str]                # sorted (descending probability)
    caption: str                   # optional: rendered full caption text
    caption_json: dict             # optional: structured JSON caption
    raw_scores: dict[str, float]   # optional: per-tag probability
    error: str                     # set on failure


@runtime_checkable
class Tagger(Protocol):
    name: str
    requires_service: bool

    def is_available(self) -> tuple[bool, str]:
        """Quick check whether this tagger can run. Returns (ok, status description). Used by the frontend status bar."""

    def prepare(self) -> None:
        """Expensive init (e.g. WD14 loading ONNX; JoyCaption calling /v1/models).
        Called once by the worker."""

    def tag(
        self,
        image_paths: list[Path],
        on_progress: ProgressFn = lambda d, t: None,
    ) -> Iterator[TagResult]:
        """Stream: yield one TagResult per image; on failure, result carries an 'error' field."""


# tagger name -> (module path, class name, whether it takes an overrides argument)
# Add a new tagger by adding a line here; keep server.py's TagJobRequest in sync by adding a `<name>_overrides`
# field (if per-job overrides are supported). The worker reads it generically via `params.get(f"{name}_overrides")`.
_TAGGER_SPEC: dict[str, tuple[str, str, bool]] = {
    "wd14": ("studio.services.tagging.wd14", "WD14Tagger", True),
    "cltagger": ("studio.services.tagging.cltagger", "CLTagger", True),
    "joycaption": ("studio.services.tagging.joycaption", "JoyCaptionTagger", False),
    "llm": ("studio.services.tagging.llm", "LLMTagger", True),
}


def get_tagger(name: str, overrides: dict | None = None) -> Tagger:
    """Factory: lazily import the implementation from `_TAGGER_SPEC` and instantiate it.

    `overrides` is currently only consumed by local ONNX taggers -- per-job tagging overrides that don't
    touch global secrets.json; taggers that don't accept overrides (e.g. joycaption) ignore it.
    """
    spec = _TAGGER_SPEC.get(name)
    if spec is None:
        raise ValueError(f"unknown tagger: {name!r}")
    module_path, cls_name, takes_overrides = spec
    klass = getattr(importlib.import_module(module_path), cls_name)
    return klass(overrides=overrides) if takes_overrides else klass()


VALID_TAGGER_NAMES: tuple[str, ...] = tuple(_TAGGER_SPEC.keys())
