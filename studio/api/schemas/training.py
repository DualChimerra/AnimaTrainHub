"""tag / captions / reg / version_config request BaseModels (extracted from server.py in PR-6.5 commit 5)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class Wd14Overrides(BaseModel):
    """Tagging page's "this-job override" of the wd14 settings -- only effective within the
    worker process, never written back to secrets.json."""
    threshold_general: Optional[float] = None
    threshold_character: Optional[float] = None
    model_id: Optional[str] = None
    blacklist_tags: Optional[list[str]] = None


class CLTaggerOverrides(BaseModel):
    """Tagging page's "this-job override" of the CLTagger settings -- only effective within
    the worker process."""
    threshold_general: Optional[float] = None
    threshold_character: Optional[float] = None
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    tag_mapping_path: Optional[str] = None
    add_copyright_tag: Optional[bool] = None
    add_artist_tag: Optional[bool] = None
    add_meta_tag: Optional[bool] = None
    add_model_tag: Optional[bool] = None
    add_rating_tag: Optional[bool] = None
    add_quality_tag: Optional[bool] = None
    blacklist_tags: Optional[list[str]] = None


class LLMTaggerOverrides(BaseModel):
    """Tagging page's "this-job override" of the LLM tagger settings -- only effective within
    the worker process.

    - `current_preset`: switches the active preset id
    - other fields: override the same-named field on the active preset
    - `api_key` cannot be overridden (to keep it out of task params/logs)
    """
    current_preset: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    prompt: Optional[str] = None
    output_format: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    max_retries: Optional[int] = None
    concurrency: Optional[int] = None
    requests_per_second: Optional[float] = None
    max_requests_per_minute: Optional[int] = None
    max_side: Optional[int] = None
    jpeg_quality: Optional[int] = None
    max_image_mb: Optional[float] = None


class TagJobRequest(BaseModel):
    tagger: str = "wd14"
    # The on-disk format follows the output, no longer chosen by the request (old clients
    # sending output_format are ignored): an LLM json preset produces structured caption_json
    # -> .json; everything else (local tagger tag list / LLM text preset) -> .txt. An existing
    # .json is still updated as .json.
    # Strategy when a caption file already exists: "overwrite" (default) | "skip" (keep the
    # existing file) | "append" (tag-level merge + dedupe, written back in the original format).
    on_existing: str = "overwrite"
    wd14_overrides: Optional[Wd14Overrides] = None
    cltagger_overrides: Optional[CLTaggerOverrides] = None
    llm_overrides: Optional[LLMTaggerOverrides] = None
    # Trigger word; empty string / None = disabled. Prepended as the first tag to the caption
    # during tagging; also persisted to version.trigger_word and read back from the private
    # yaml during the later train phase.
    trigger_word: Optional[str] = None
    # Tagging scope: "all" (default, all train folders + validation) | "validation" (only the
    # held-out validation set) | a specific train subfolder name (e.g. "1_data", only that one).
    # Manually added validation images have no caption by default; this is how they get included
    # in tagging.
    scope: str = "all"


class CaptionEdit(BaseModel):
    tags: list[str]


class CommitItem(BaseModel):
    folder: str
    name: str
    tags: list[str]


class CommitRequest(BaseModel):
    items: list[CommitItem]


class BatchOp(BaseModel):
    op: str                                   # add|remove|replace|dedupe|stats
    scope: dict[str, Any]                     # {kind, folder?, names?}
    tags: Optional[list[str]] = None          # add/remove
    old: Optional[str] = None                 # replace
    new: Optional[str] = None                 # replace
    position: Optional[str] = "back"          # add: front|back
    top: int = 50                             # stats


class RegBuildRequest(BaseModel):
    excluded_tags: list[str] = []
    auto_tag: bool = True
    # A3: tagger choice for reg-set auto-tagging. Default wd14 (backward compatible); the UI
    # currently exposes wd14/cltagger. LLM / JoyCaption will land in a separate PR -- reg sets
    # are large, so slow/expensive taggers don't fit the default path.
    auto_tag_kind: str = "wd14"
    api_source: str = "gelbooru"
    # Incremental by default (owner decision 2026-05-30): with reg sets, users usually want to
    # keep what's there and only fill the gap, not wipe out yesterday's hard-won images when
    # starting a new generation. Full mode goes through the worker's `clear_reg_dir`, which
    # wipes reg/ entirely (including .deleted_ids.json).
    incremental: bool = True
    # A4: after build, automatically run dedup + top-up loop if short (up to N rounds).
    # On by default (owner decision 2026-05-30). The manual button (RegPreview "auto dedupe")
    # is kept separately.
    auto_dedup: bool = True
    # B1 (PR-2): build mode
    # - mirror: mirrors the train subfolder structure (5_concept/, 1_general/, ...), each
    #   subfolder pulled independently based on its train image count (old behavior).
    #   target_count is ignored in this mode.
    # - flat: all images go into a single `1_data/` bucket; target_count sets the count
    #   (None = total train count).
    # Default is flat (owner decision 2026-05-30); switching modes requires the reg set to
    # already be empty (enforced by the UI).
    build_mode: str = "flat"
    # B1: target image count in flat mode; None = total train image count. Ignored in mirror mode.
    target_count: Optional[int] = None
    # Note: the source (booru / AI prior) is decided by the frontend SourcePicker, which picks
    # the endpoint (/reg/build vs /reg/generate-prior) and doesn't go into this schema.
    # RegMeta.generation_method is written separately on the ai path.
    # PP5.5 advanced config (defaults match the source script; only applies in booru mode)
    skip_similar: bool = True
    aspect_ratio_filter_enabled: bool = False
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    postprocess_method: str = "smart"  # smart | stretch | crop
    postprocess_max_crop_ratio: float = 0.1


class RegDeleteFilesRequest(BaseModel):
    """Bulk-delete the given images from the reg set (including the matching .txt caption).

    `relative_paths` is a list of paths relative to reg/, spanning subfolders is fine.
    The backend appends the deleted booru ID (= filename stem) to `reg/.deleted_ids.json`,
    so an incremental build later excludes it from search results and it won't come back.
    """
    relative_paths: list[str]


class RegRenameFolderRequest(BaseModel):
    """Rename a subfolder under reg/ (changes the Kohya repeat prefix, e.g. 2_data -> 1_data)."""
    name: str
    new_name: str


class RegAiRequest(BaseModel):
    """Prior-generation request -- no lora_configs, prior generation runs without LoRA."""
    excluded_tags: list[str] = []
    # Base model to use for this prior generation (official variant key or a registered local
    # custom path); None -> use Settings' selected_anima. Only swaps the transformer weights.
    base_model: Optional[str] = None
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg_scale: float = 4.0
    # None = unspecified -> the endpoint resolves the default from the version's model family
    # (anima er_sde/simple, krea2 euler/simple). An explicit value is strictly validated against
    # the family's whitelist.
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    seed: int = 0
    incremental: bool = False
    repeat: int = 1  # reg subfolder repeat prefix (N_label); 1 = DreamBooth standard
    mixed_precision: str = "bf16"


class FromPresetRequest(BaseModel):
    name: str  # global preset name


class SaveAsPresetRequest(BaseModel):
    name: str
    overwrite: bool = False
