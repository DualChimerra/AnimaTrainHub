"""JoyCaption backward-compat shim.

JoyCaption has been merged into the LLM tagger's builtin preset. This wrapper only exists as a fallback for old callers
(`get_tagger("joycaption")`) -- new code should call `get_tagger("llm",
overrides={"current_preset": "joycaption"})` directly, or omit overrides to let the global
`current_preset` decide.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import requests

from .llm import LLMTagger
from .base import ProgressFn, TagResult


class JoyCaptionTagger:
    name = "joycaption"
    requires_service = True

    def __init__(self, *, session: Optional[requests.Session] = None) -> None:
        self._session = session or requests.Session()

    def _llm(self) -> LLMTagger:
        # Force-switch to the joycaption preset (its fields are determined jointly by the builtin defaults + any
        # overrides the user has changed in Settings).
        return LLMTagger(overrides={"current_preset": "joycaption"}, session=self._session)

    def is_available(self) -> tuple[bool, str]:
        return self._llm().is_available()

    def prepare(self) -> None:
        self._llm().prepare()

    def tag(
        self,
        image_paths: list[Path],
        on_progress: ProgressFn = lambda d, t: None,
    ) -> Iterator[TagResult]:
        yield from self._llm().tag(image_paths, on_progress=on_progress)
