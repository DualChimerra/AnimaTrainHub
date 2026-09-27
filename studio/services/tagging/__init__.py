"""Tagger family -- PR-3 flattened out of services/ into this subpackage.

File mapping:
  - base.py            was tagger.py (Protocol + factory + VALID_TAGGER_NAMES)
  - caption_format.py  was caption_format.py
  - caption_snapshot.py was caption_snapshot.py
  - onnx_base.py       was onnx_tagger_base.py
  - wd14.py            was wd14_tagger.py
  - cltagger.py        was cltagger_tagger.py
  - llm.py             was llm_tagger.py
  - joycaption.py      was joycaption_tagger.py

Re-exports the main public names for use as `from studio.services.tagging import X`.
The old import paths (`studio.services.tagger` etc.) stay compatible via shim files at the same level.
"""
from .base import VALID_TAGGER_NAMES, ProgressFn, TagResult, Tagger, get_tagger
from .caption_format import (
    caption_json_to_tags,
    caption_json_to_text,
    normalize_caption_json,
    split_tags,
    standard_to_documented_full,
)
from .caption_snapshot import (
    SnapshotError,
    create_snapshot,
    delete_snapshot,
    list_snapshots,
    restore_snapshot,
    snapshot_root,
)
from .cltagger import CLTagger
from .joycaption import JoyCaptionTagger
from .llm import LLMTagger, fetch_openai_compatible_models, test_openai_compatible_connection
from .onnx_base import OnnxTaggerBase, safe_dir_name, silenced_fd_stderr
from .wd14 import WD14Tagger

__all__ = [
    "CLTagger",
    "JoyCaptionTagger",
    "LLMTagger",
    "OnnxTaggerBase",
    "ProgressFn",
    "SnapshotError",
    "TagResult",
    "Tagger",
    "VALID_TAGGER_NAMES",
    "WD14Tagger",
    "caption_json_to_tags",
    "caption_json_to_text",
    "create_snapshot",
    "delete_snapshot",
    "fetch_openai_compatible_models",
    "get_tagger",
    "list_snapshots",
    "normalize_caption_json",
    "restore_snapshot",
    "safe_dir_name",
    "silenced_fd_stderr",
    "snapshot_root",
    "split_tags",
    "standard_to_documented_full",
    "test_openai_compatible_connection",
]
