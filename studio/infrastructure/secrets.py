"""Global service credentials + config — centrally stored in studio_data/secrets.json.

`studio_data/` is already in .gitignore, so this file can hold real tokens / api keys.
Sensitive fields are returned externally as "***" via `to_masked_dict()`; when the
frontend PUTs back "***" it means "keep unchanged", handled by `update()`'s deep-merge.
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, computed_field, model_validator

from .paths import STUDIO_DATA

SECRETS_FILE = STUDIO_DATA / "secrets.json"
MASK = "***"
# Dotted-path + `*` wildcard support: `llm_tagger.presets.*.api_key` walks every dict in the list.
SENSITIVE_FIELDS: tuple[str, ...] = (
    "gelbooru.api_key",
    "danbooru.api_key",
    "huggingface.token",
    "wandb.presets.*.api_key",
    "llm_tagger.presets.*.api_key",
    "modelscope.token",
    "remote_access.access_key",
    "remote_access.ngrok_authtoken",
)


class GelbooruConfig(BaseModel):
    user_id: str = ""
    api_key: str = ""


class DanbooruConfig(BaseModel):
    """Danbooru HTTP Basic auth: username + api_key.

    Mandatory binding as of PR #38 (anonymous no longer allowed):
    - After Danbooru put up Cloudflare, an anonymous UA is no longer reliable (CF can tighten at any time)
    - Requiring an account lets our UA carry (by username), so CF blocking anonymous traffic won't take us down with it
    - Danbooru rate-limits by account tier (standard 2 req/s, higher than anonymous)
    """
    username: str = ""
    api_key: str = ""
    # Account type determines the multi-tag search cap (free=2 / gold=6 / platinum=12)
    account_type: str = "free"


class HuggingFaceConfig(BaseModel):
    token: str = ""
    # PR-S3: HF model download endpoint. `""` uses the huggingface_hub default (direct to huggingface.co).
    # 0.8.2 hotfix: switched the default back from `hf-mirror.com` to `""` (official HF). hf-mirror
    # currently triggers `FileMetadataError` (commit_hash None) on every huggingface_hub version;
    # users in mainland China should use ModelScope or their own reverse proxy; the hf-mirror preset
    # is hidden from the UI for now, but the endpoint field still accepts any URL (users can paste
    # one manually). See the recheck list at docs/todo/hf-mirror-recheck.md.
    # Custom URLs are also supported (tencent / sjtug / self-hosted reverse proxy, etc).
    # Since huggingface_hub>=0.20, both hf_hub_download / snapshot_download accept an `endpoint=`
    # kwarg, which we pass per-call rather than relying on the HF_ENDPOINT env var (the env var is
    # only read at module import time, so changing it at runtime has no effect).
    endpoint: str = ""


class WandBPresetConfig(BaseModel):
    """A WandB account + upload-policy preset (mirrors LLMPresetConfig's preset pattern).

    As of 0.18, WandB config is preset-based: the top-level WandBConfig only keeps
    enabled + the current preset pointer; the account (api_key/entity/base_url) and
    upload policy all live in the preset, so the whole set can be swapped at once.
    """
    id: str = "default"
    label: str = "Default"
    api_key: str = ""
    project: str = "AnimaLoraStudio"
    entity: str = ""
    base_url: str = ""
    mode: str = "online"
    # On by default — when wandb is enabled, sample images upload along with it, saving users an
    # extra toggle each time. Turn this off in Settings for private IP / NSFW datasets; with it
    # off, only metrics upload, no images leave the machine.
    log_samples: bool = True
    # Downscaled to this max side in pixels before upload; source images are often 2K+, and 512 is
    # plenty for browsing in the wandb panel, saving bandwidth.
    sample_max_side: int = 512
    # Step throttling: when >0, only uploads on `global_step % N == 0`, avoiding GB-scale image
    # uploads over a long run. 0 = no extra throttling (uploads at whatever sample frequency the
    # training loop already uses); baseline / epoch boundaries always upload.
    sample_every_n_steps: int = 0
    # Artifact upload: model / training-state checkpoints uploaded to wandb Artifacts for cloud
    # management and version tracking. policy = "all" keeps every version, "last" keeps only the
    # newest (deletes the old version after uploading the new one).
    upload_model: bool = False
    upload_model_policy: str = "last"
    upload_state_manual: bool = False
    upload_state_manual_policy: str = "last"
    upload_state_auto: bool = False
    upload_state_auto_policy: str = "last"

    @model_validator(mode="after")
    def _normalize_values(self) -> "WandBPresetConfig":
        self.id = "".join(
            ch if ch.isalnum() or ch in ("_", "-") else "_"
            for ch in str(self.id or "").strip()
        ).strip("_") or "default"
        self.label = str(self.label or self.id).strip()
        if self.mode not in {"online", "offline", "disabled"}:
            self.mode = "online"
        self.sample_max_side = max(64, int(self.sample_max_side or 512))
        self.sample_every_n_steps = max(0, int(self.sample_every_n_steps or 0))
        _valid_policies = {"all", "last"}
        if self.upload_model_policy not in _valid_policies:
            self.upload_model_policy = "last"
        if self.upload_state_manual_policy not in _valid_policies:
            self.upload_state_manual_policy = "last"
        if self.upload_state_auto_policy not in _valid_policies:
            self.upload_state_auto_policy = "last"
        return self


class WandBConfig(BaseModel):
    """Global WandB: the top level only keeps the overall switch + the current preset pointer; all other fields live in the preset.

    The old flat schema (enabled + flat fields) is wrapped by _migrate_legacy_schema into
    a single preset with id="default". The training process reads the `active` preset and
    injects it as WANDB_* env vars via the supervisor — secrets are never written to any yaml.
    """
    enabled: bool = False
    current_preset: str = "default"
    presets: list[WandBPresetConfig] = Field(
        default_factory=lambda: [WandBPresetConfig()]
    )

    @model_validator(mode="after")
    def _normalize_values(self) -> "WandBConfig":
        # Dedupe ids while preserving order + guarantee at least one preset + make current point to an existing id
        merged: list[WandBPresetConfig] = []
        seen: set[str] = set()
        for preset in self.presets:
            if preset.id and preset.id not in seen:
                merged.append(preset)
                seen.add(preset.id)
        if not merged:
            merged = [WandBPresetConfig()]
        self.presets = merged
        if self.current_preset not in {p.id for p in self.presets}:
            self.current_preset = self.presets[0].id
        return self

    @property
    def active(self) -> WandBPresetConfig:
        """The currently selected preset; the validator guarantees at least one exists."""
        for preset in self.presets:
            if preset.id == self.current_preset:
                return preset
        return self.presets[0]


class ModelScopeConfig(BaseModel):
    token: str = ""
    # ModelScope (modelscope.cn) download token. Public models can download without it; needed for
    # private models or when rate-limited.
    # Requires `pip install modelscope` first; downloads prefer the matching repo from
    # MODELSCOPE_REPO_MAP, falling back to HuggingFace for unmapped models.


class EvalMetricModelsConfig(BaseModel):
    """LoRA eval metric model defaults.

    Metric API callers may still pass `model_name` explicitly. Empty request
    values fall back to these defaults so server-local ModelScope/HF cache paths
    do not need to be repeated for every metric run.
    """
    clip_model_name: str = "openai/clip-vit-base-patch32"
    dino_model_name: str = "facebook/dinov2-small"
    ccip_model_name: str = "ccip-caformer-24-randaug-pruned"
    # Which evaluation metrics are enabled (Settings checkboxes, see eval_registry). Only checked
    # metrics are computed; the existing three metrics stay on by default, the anime-domain
    # metrics (ccip_i / tag_recall) default off and need the user to enable them.
    enabled_metrics: list[str] = Field(
        default_factory=lambda: ["clip_t", "clip_i", "dino_i"]
    )
    # Baseline comparison: after training, evaluation additionally generates a set of
    # pure-base-model (lora_scale=0) images with the same prompt/seed, and each metric reports
    # Δ = checkpoint − baseline (solving the "absolute values are hard to interpret" problem).
    # Runs once per task. Evaluation always runs after training (inline / checkpoint-trigger has
    # been removed); whether it runs is decided by each version's eval_validation_enabled
    # training-config field.
    eval_baseline_enabled: bool = True


class DownloadConfig(BaseModel):
    """Global download preferences (shared across channels)."""
    # Global exclude tags: automatically appends -tag1 -tag2 to searches (same syntax on gelbooru / danbooru)
    exclude_tags: list[str] = Field(default_factory=list)
    # PP9 — Booru API pool rate limiting (shared by downloader + reg_builder)
    parallel_workers: int = 4
    api_rate_per_sec: float = 2.0
    cdn_rate_per_sec: float = 5.0
    # Downloaded-image ingestion processing (used to live under gelbooru, actually shared by all
    # booru downloads / reg / local uploads):
    save_tags: bool = False
    convert_to_png: bool = True
    # New installs default to true: in training, a 4-channel PNG lets the VAE learn the
    # transparent region as noise; most users need alpha stripped. Existing secrets.json with this
    # explicitly set to false is unaffected.
    remove_alpha_channel: bool = True


class RegConfig(BaseModel):
    """Regularization-set generation preferences (global defaults)."""
    # Global default exclude tags: when the regularization-set generation page opens a build with
    # no local selection yet, this list seeds the initial exclusion. It's only a starting point —
    # users can still add/remove tags per-build after entering the page, and it doesn't affect an
    # existing build's local record. The frontend normalizes to booru form (underscores) per this
    # page's convention before storing into the excluded set.
    default_excluded_tags: list[str] = Field(default_factory=list)


LLM_MESSAGE_ROLES: tuple[str, ...] = ("system", "user", "assistant")
LLM_MESSAGE_TYPES: tuple[str, ...] = ("text", "image")


class LLMMessage(BaseModel):
    """A single message inside an LLM payload.

    type=text: a plain text message, requires a role (system/user/assistant); content is the prompt text
    type=image: an image placeholder item — the backend inserts the current image here during tagging
        - the content field is ignored
        - role is fixed to "user" (both OpenAI / Anthropic put images on the user side)
        - each preset must have exactly one type=image item (enforced by the validator)
    """
    type: str = "text"
    role: str = "user"
    content: str = ""

    @model_validator(mode="after")
    def _normalize(self) -> "LLMMessage":
        if self.type not in LLM_MESSAGE_TYPES:
            self.type = "text"
        if self.type == "image":
            self.role = "user"
            self.content = ""
        else:
            if self.role not in LLM_MESSAGE_ROLES:
                self.role = "user"
        return self


def _default_messages_for(prompt: str) -> list["LLMMessage"]:
    """One-line migration of the old prompt field → [{system, prompt}, {image}]."""
    msgs: list[LLMMessage] = []
    if prompt:
        msgs.append(LLMMessage(type="text", role="system", content=prompt))
    msgs.append(LLMMessage(type="image"))
    return msgs


class LLMPresetConfig(BaseModel):
    """Full LLM tagger preset: each preset carries a complete endpoint + messages + generation-parameter set.

    messages is an OpenAI chat-completions-style message sequence, plus a special type=image item
    marking where the image should be inserted. During tagging, the backend expands messages in
    order into the API payload.

    builtin: bool only flags whether the id comes from the builtin list (used by the UI to show
    "reset to default") — it doesn't lock the fields; editing any field of a builtin preset still persists.
    """
    id: str
    label: str = ""
    builtin: bool = False
    # Endpoint identity
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    model_ids: list[str] = Field(default_factory=list)
    endpoint: str = "chat_completions"  # chat_completions | responses
    # Prompt message sequence (including the image position)
    messages: list[LLMMessage] = Field(default_factory=lambda: _default_messages_for(""))
    output_format: str = "json"  # json | text
    # Assist tagging: pre-tag images with a local ONNX tagger before LLM calls,
    # then inject tags into {{tags}} placeholders in text messages.
    assist_tagger: str = ""  # "" | wd14 | cltagger
    # Generation parameters
    temperature: float = 0.2
    max_tokens: int = 700
    # Image handling
    max_side: int = 1280
    jpeg_quality: int = 85
    max_image_mb: float = 5.0
    # Retry / timeout
    timeout: int = 60
    max_retries: int = 3
    # Request pool / throttling
    concurrency: int = 1
    requests_per_second: float = 0.0
    max_requests_per_minute: int = 0

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_prompt(cls, data: Any) -> Any:
        """Backward compat for the old schema's prompt: str → messages list."""
        if not isinstance(data, dict):
            return data
        if "messages" in data and data["messages"]:
            return data
        legacy_prompt = str(data.pop("prompt", "") or "").strip()
        data["messages"] = [m.model_dump() if isinstance(m, LLMMessage) else m
                            for m in _default_messages_for(legacy_prompt)]
        return data

    @model_validator(mode="after")
    def _normalize_values(self) -> "LLMPresetConfig":
        self.id = "".join(
            ch if ch.isalnum() or ch in ("_", "-") else "_"
            for ch in str(self.id or "").strip()
        ).strip("_")
        self.label = str(self.label or self.id).strip()
        if self.endpoint not in {"chat_completions", "responses"}:
            self.endpoint = "chat_completions"
        if self.output_format not in {"json", "text"}:
            self.output_format = "json"
        if self.assist_tagger not in {"", "wd14", "cltagger"}:
            self.assist_tagger = ""
        self.temperature = max(0.0, min(float(self.temperature), 2.0))
        self.max_tokens = max(64, int(self.max_tokens or 700))
        self.timeout = max(5, int(self.timeout or 60))
        self.max_retries = max(1, int(self.max_retries or 3))
        self.concurrency = max(1, min(8, int(self.concurrency or 1)))
        self.requests_per_second = max(
            0.0,
            min(60.0, float(self.requests_per_second or 0.0)),
        )
        self.max_requests_per_minute = max(
            0,
            min(3600, int(self.max_requests_per_minute or 0)),
        )
        self.max_side = max(64, int(self.max_side or 1280))
        self.jpeg_quality = max(1, min(100, int(self.jpeg_quality or 85)))
        self.max_image_mb = max(0.1, float(self.max_image_mb or 5.0))
        # The currently selected model always appears at the head of the candidate list (same as WD14Config)
        if self.model and self.model not in self.model_ids:
            self.model_ids = [self.model, *self.model_ids]
        seen: set[str] = set()
        clean: list[str] = []
        for mid in self.model_ids:
            text = str(mid or "").strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            clean.append(text)
        self.model_ids = clean
        # messages fallback: must have exactly one type=image item; append one if missing
        if not self.messages:
            self.messages = _default_messages_for("")
        else:
            has_image = any(m.type == "image" for m in self.messages)
            if not has_image:
                self.messages = [*self.messages, LLMMessage(type="image")]
            else:
                # multiple images → keep only the first
                kept: list[LLMMessage] = []
                seen_image = False
                for m in self.messages:
                    if m.type == "image":
                        if seen_image:
                            continue
                        seen_image = True
                    kept.append(m)
                self.messages = kept
        return self


def _default_llm_presets() -> list[LLMPresetConfig]:
    from .llm_presets import builtin_llm_presets

    return [LLMPresetConfig(**item) for item in builtin_llm_presets()]


class LLMTaggerConfig(BaseModel):
    """Top-level LLM tagger config: only keeps "the currently selected preset id" + "the preset list".

    All endpoint / prompt / generation parameters live in LLMPresetConfig.
    """
    current_preset: str = "style_json"
    presets: list[LLMPresetConfig] = Field(default_factory=_default_llm_presets)

    @model_validator(mode="after")
    def _normalize_values(self) -> "LLMTaggerConfig":
        from .llm_presets import BUILTIN_PRESET_ORDER, builtin_llm_presets

        builtin_defaults = {item["id"]: item for item in builtin_llm_presets()}
        user_by_id = {p.id: p for p in self.presets if p.id}

        merged: list[LLMPresetConfig] = []
        seen_ids: set[str] = set()
        # 1) Order by builtin order: user-modified presets override the builtin default; missing ones are refilled from the default
        for bid in BUILTIN_PRESET_ORDER:
            if bid in user_by_id:
                preset = user_by_id[bid]
                preset.builtin = True
                merged.append(preset)
                seen_ids.add(bid)
            elif bid in builtin_defaults:
                preset = LLMPresetConfig(**builtin_defaults[bid])
                preset.builtin = True
                merged.append(preset)
                seen_ids.add(bid)
        # 2) Append user-defined presets (id not in the builtin list)
        for preset in self.presets:
            if preset.id and preset.id not in seen_ids:
                preset.builtin = False
                merged.append(preset)
                seen_ids.add(preset.id)
        if not merged:
            merged = _default_llm_presets()
        self.presets = merged
        preset_ids = {p.id for p in self.presets}
        if self.current_preset not in preset_ids:
            self.current_preset = self.presets[0].id
        return self

    @property
    def active(self) -> LLMPresetConfig:
        """The currently selected preset; the validator guarantees at least one exists."""
        for preset in self.presets:
            if preset.id == self.current_preset:
                return preset
        return self.presets[0]


# Default WD14 candidate models; users can add/remove them under "Settings → WD14 → candidate
# models". The currently selected `model_id` is always normalized into `model_ids` (see the
# WD14Config validator).
DEFAULT_WD14_MODELS: tuple[str, ...] = (
    "SmilingWolf/wd-eva02-large-tagger-v3",
    "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-vit-large-tagger-v3",
    "SmilingWolf/wd-v1-4-convnext-tagger-v2",
)


class WD14Config(BaseModel):
    model_id: str = "SmilingWolf/wd-eva02-large-tagger-v3"
    model_ids: list[str] = Field(
        default_factory=lambda: list(DEFAULT_WD14_MODELS)
    )
    threshold_general: float = 0.35
    threshold_character: float = 0.85
    blacklist_tags: list[str] = Field(default_factory=list)
    # PP8 — batch inference size; used when the GPU EP is active, falls back to 1 automatically on CPU
    batch_size: int = 8

    @model_validator(mode="after")
    def _ensure_model_ids_invariant(self) -> "WD14Config":
        """Guarantees `model_id ∈ model_ids` and that the candidate list is never empty.

        - Empty list (including an old secrets.json missing this field that then got explicitly
          cleared) → refilled with the default 4 entries.
        - The currently selected model_id not in the list → prepended to the list (so users can
          run a one-off model while the dropdown still always shows the current value).
        Side effect: if a user wants to "remove the currently selected" model from the candidates,
        they must switch to a different model_id on the tagging / settings page first, then delete;
        the frontend enforces this order.
        """
        if not self.model_ids:
            self.model_ids = list(DEFAULT_WD14_MODELS)
        if self.model_id and self.model_id not in self.model_ids:
            self.model_ids = [self.model_id, *self.model_ids]
        return self


class CLTaggerConfig(BaseModel):
    model_id: str = "cella110n/cl_tagger"
    model_path: str = "cl_tagger_1_02/model.onnx"
    tag_mapping_path: str = "cl_tagger_1_02/tag_mapping.json"
    threshold_general: float = 0.35
    threshold_character: float = 0.6
    # The CLTagger model outputs 8 categories: General / Character go through threshold filtering,
    # the other 6 are gated by bool switches. General / Character / Copyright are checked on by
    # default — the standard caption shape for LoRA training; Artist / Meta / Model / Rating /
    # Quality default off, to avoid polluting captions (artist names and meta info like "highres",
    # "best quality", "explicit").
    add_copyright_tag: bool = True
    add_artist_tag: bool = False
    add_meta_tag: bool = False
    add_model_tag: bool = False
    add_rating_tag: bool = False
    add_quality_tag: bool = False
    blacklist_tags: list[str] = Field(default_factory=list)
    # Same as WD14: real batching only kicks in with a CUDA EP, falls back to 1 automatically on CPU.
    batch_size: int = 8


class QueueConfig(BaseModel):
    """Queue scheduling policy (the R-1 resource-tier model, docs/design/queue-resource-model-0.17.md).

    Work items fall into three tiers: exclusive (training / regularization AI / image generation /
    evaluation generation — base-model-scale VRAM, only 1 runs system-wide at a time, never in
    parallel), light (tagging / upscaling / regularization build / evaluation metrics — a few
    hundred MB, small models), io (downloads, always allowed through).

    - `light_tasks_during_train`: whether light-tier tasks are allowed to run in parallel while an
      exclusive task is running. On by default — light tasks only load small models. The exclusive
      tier is unaffected by this switch (the old `allow_gpu_during_train` switch let even
      evaluation generation through, an OOM hazard, and has been retired; its semantics changed,
      so the old value isn't migrated).
    """
    light_tasks_during_train: bool = True


class TrainingSecretsConfig(BaseModel):
    """Global training-side behavior switches (Settings → Training).

    - `ram_guard`: memory/VRAM headroom protection for training / AI priors (regularization
      generation). Same semantics as `generate.ram_guard`: before loading a large model, budgets
      system RAM and free GPU VRAM against the weight file's actual size, aborting with an
      actionable error if either is insufficient. **Off by default** (per upstream v0.23.1: the
      file-size-based estimate is conservative, producing a high false-rejection rate on
      well-provisioned machines, and a false rejection leaves the user with no way out); when off,
      insufficient resources let loading proceed, which may trigger system-wide paging stutter.
      Injected into the training subprocess via the ``LORA_RAM_GUARD`` environment variable
      (supervisor `_popen`). Block swap's pinned-memory guardrail **is not affected by this
      switch** — pinned memory can't be paged out, filling it hard-stalls the whole machine, and it
      has its own way out (lower blocks_to_swap).

    Upstream calls this model `TrainingConfig`; this fork renames it `TrainingSecretsConfig`:
    `studio.domain.training.TrainingConfig` (the 643-line training-parameter schema) is one of the
    most commonly imported names in the whole repo, and two same-named pydantic models would
    invite wrong imports.
    The key in secrets.json is still `training`, matching upstream's shape.
    """
    ram_guard: bool = False


class ModelsConfig(BaseModel):
    """Global model configuration (PP7).

    - `root`: the root directory where models are stored. `None/""` → falls back to
      `REPO_ROOT/models/` (default). Cloud / large-capacity data disks can point this at an
      absolute path, e.g. `D:/anima-models` or `/data/anima`. All training models (Anima / VAE /
      Qwen3 / T5 tokenizer / WD14) share this one root directory.
    - `selected_anima`: the current default main model. Can be an official variant key (`1.0`,
      etc.) **or** an absolute path to a local `.safetensors` file registered in
      `custom_anima_paths`. When Studio creates a new version, this field determines the absolute
      path written into `transformer_path` in the yaml; existing versions are left untouched (to
      preserve training reproducibility).
    - `custom`: per-family storage of local main-model weights the user registered via the
      PathPicker. `custom_anima_paths` is kept as a read/write compat surface for older clients.
      Only registers the path — nothing is downloaded or copied; if an entry becomes invalid,
      resolution automatically falls back to the official variant.
    - `selected_vae`: the current default VAE. Empty string = the official `qwen_image_vae`
      location; otherwise a user-registered absolute path to a local `.safetensors` file. VAE is a
      family-agnostic shared asset (see `models/paths.qwen_image_vae_target`), so this is a single
      value rather than per-family.
    - `selected_upscaler`: the default preprocessing upscaler. Can be a preset label (e.g.
      "4x-AnimeSharp") or a custom/uploaded filename (e.g. "my-anime-model.pth"). Empty
      string/None → falls back to DEFAULT_UPSCALER.
    - `auto_sync_paths`: when forking a preset into a version, whether to automatically overwrite
      the preset's 4 model fields (transformer / vae / text_encoder / t5_tokenizer) with the global
      model paths. ON (default) → for most users: the 4 fields are never touched, forking always
      uses the Settings global values; the 4 fields are disabled in the project page / preset page
      UI.
      OFF → for independent-model users: forking respects the preset's own values, the 4 fields are
      editable + have a picker.
    """
    root: Optional[str] = None
    # Per-family selected main model (multi-model PR-4): family_id → variant key or custom path.
    # The old selected_anima key is migrated by the before-validator (a settings PUT's merged dict
    # can carry both keys — the inbound selected_anima takes priority, overriding the old selected
    # merged in).
    selected: dict[str, str] = Field(default_factory=lambda: {"anima": "1.0"})
    # Per-family selected text encoder: an official variant key (krea2: "bf16"|"fp8", missing=bf16)
    # **or** an absolute path to a user-registered local text-encoder directory (custom CLIP /
    # Qwen encoder). Determines the text_encoder_path default when a new training version is
    # created + the test-generation TE default; existing versions' configs are untouched (training
    # reproducibility, same convention as selected). A local path that becomes invalid (deleted /
    # moved) automatically falls back to the official directory.
    selected_te: dict[str, str] = Field(default_factory=dict)
    # Selected VAE: empty string = the official qwen_image_vae location, otherwise an absolute
    # local .safetensors path (one of domain "vae"'s candidates). Also falls back to the official
    # location when invalid.
    selected_vae: str = ""
    # Per-family local main-model paths. The old custom_anima_paths key is migrated by the
    # validator; the computed_field keeps a read surface for older clients.
    custom: dict[str, list[str]] = Field(default_factory=dict)
    selected_upscaler: str = "4x-AnimeSharp"
    auto_sync_paths: bool = True

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_model_fields(cls, data):
        """Migrates the old Anima keys while preserving the new per-family structure."""
        if isinstance(data, dict) and (
            "selected_anima" in data or "custom_anima_paths" in data
        ):
            data = dict(data)
            if "selected_anima" in data:
                legacy_selected = data.pop("selected_anima")
                selected = dict(data.get("selected") or {})
                if legacy_selected:
                    selected["anima"] = str(legacy_selected)
                data["selected"] = selected
            if "custom_anima_paths" in data:
                legacy_custom = data.pop("custom_anima_paths")
                custom = dict(data.get("custom") or {})
                custom["anima"] = list(legacy_custom or [])
                data["custom"] = custom
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def selected_anima(self) -> str:
        """Read-compat surface (frontend settings reads + dump echoes to disk); writes should go through selected."""
        return self.selected.get("anima") or "1.0"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def custom_anima_paths(self) -> list[str]:
        """Backward-compat Anima local-model list for old clients; writes should go through custom."""
        return list(self.custom.get("anima") or [])


class GenerateConfig(BaseModel):
    """Test-generation daemon behavior (PR Phase 2).

    - `preview_every_n_steps`: intermediate-step preview throttling. 0=off; >0 → the daemon
      pushes a 256px JPEG preview to the frontend every N steps via TAEFlux decode. Requires the
      TAEFlux model to already be downloaded (via the Settings entry point or
      POST /api/generate/taeflux/install).
    - `attention_backend`: attention backend selection. `'auto'` (default) → uses whatever is
      installed (priority flash_attn > xformers > none/SDPA); an explicit value (flash_attn/
      xformers/none) forces it — useful for debugging or comparisons.
    - `idle_timeout_minutes`: the daemon auto-unloads the model to free VRAM after N minutes idle.
      0 = off, the model stays resident until manually cleared. The timer only runs while the
      daemon is idle + a model is loaded; it's cancelled on entering busy / after unload.
    - `vae_precision`: test-generation VAE decode precision. `'bf16'` (default) matches ComfyUI's
      auto VAE dtype on modern GPUs; `'fp32'` is full-precision decode (higher peak VRAM — the
      daemon temporarily offloads DiT/Qwen before decode to free VRAM).
    - `save_test_images`: toggles automatic disk-saving of test generations. Off by default; when
      on, the frontend calls /api/generate/save after each generation to save the image to
      studio_data/test/<date>/{single,xy}/image_N.png (N = the current folder's highest existing
      number + 1). Compare mode never saves to disk.
    - `vram_policy`: test-generation VRAM strategy (applies to krea2). `'auto'` (default) decides
      whether the text encoder and DiT yield to each other based on free VRAM; `'save_vram'` forces
      sequential loading (lowest peak, a few extra seconds moving things per image);
      `'performance'` keeps everything resident in VRAM (highest peak, zero moving).
    - `ram_guard`: memory/VRAM headroom protection. Before loading a large model, budgets system
      RAM and free GPU VRAM against the weight file's actual size, aborting with an actionable
      error if either is insufficient (when on, the VRAM check can catch multi-process VRAM
      stacking). **Off by default** (per upstream v0.23.1: the file-size-based estimate is
      conservative, producing a high false-rejection rate); when off, insufficient resources let
      loading proceed, which may trigger system-wide paging stutter.
    - `task_timeout_minutes`: a fallback timeout for generation tasks. If a task hasn't finished N
      minutes after starting → the daemon process is force-killed (a hung task can't be cancelled
      at the protocol level, only killed at the process level; the next task restarts it
      automatically). 0 (default) = off.
    """
    preview_every_n_steps: int = 3
    attention_backend: str = "auto"
    vae_precision: str = "bf16"
    idle_timeout_minutes: int = 10
    save_test_images: bool = False
    vram_policy: str = "auto"
    ram_guard: bool = False
    task_timeout_minutes: int = 0


class SystemConfig(BaseModel):
    """System-level preferences (ADR 0002 / 0005).

    - `update_channel`: which update track the user is subscribed to. "stable" (default) = only
      see stable-release update prompts; "dev" = see the dev channel (recent commit timeline, can
      switch to dev HEAD). This is a **user view preference**, decoupled from the git working
      tree's state — toggling it triggers no git operations; actually "switching to dev HEAD" /
      "updating to vX.Y.Z" are separate buttons.
    - `show_dev_channel`: deprecated, one-time migrated to `update_channel` by
      `_migrate_legacy_schema` (true → "dev", false → "stable"); kept so pydantic doesn't error
      reading an old secrets.json; don't use it in new code.
    - `enable_automagic_v2`: an experimental feature flag. Automagic v2 (fused backward) hasn't
      been officially released, so the UI hides the automagic_variant field by default
      (/api/schema dynamically marks it hidden). The Settings page **deliberately doesn't render**
      this toggle — it can only be enabled by hand-editing secrets.json; the CLI/yaml path is
      unaffected (validation still catches incompatible combinations like grad_accum/fp16).
    """
    update_channel: str = "stable"  # "stable" / "dev"
    show_dev_channel: bool = False  # deprecated, migration source only
    enable_automagic_v2: bool = False  # experimental: file-level switch, not exposed in the UI
    # One-time migration sentinel for "ram_guard default flipped to off" (_migrate_legacy_schema
    # step 10). The test is **the key missing on disk** (only an old disk written before this
    # version lacks this key) → discard the old on-disk generate/training ram_guard values so all
    # users land on the new default (off); new code always writes this disk with the key present,
    # so an explicitly-enabled value is never discarded. Default True: a fresh install has no old
    # value to migrate, so "migration already done" holds trivially.
    ram_guard_default_off: bool = True


class RuntimeConfig(BaseModel):
    """Runtime mode (Colab / Local) — see `infrastructure/runtime_mode.py`.

    - `mode`: `""` (not chosen yet, the frontend pops the picker on app entry) / `"local"` /
      `"colab"`. An invalid value is reset to `""` by the validator — better to ask again than to
      silently run under the wrong mode.
    - `asked`: whether the user has already been through the choice flow once. It's necessarily
      True whenever `mode` has a value; it's kept as a separate field for a possible future "skip
      once, ask again next time," and for now is just a read-only marker.

    The `ALS_RUNTIME_MODE` environment variable overrides this field and is never persisted
    (injected by the Colab notebook).
    """
    mode: str = ""
    asked: bool = False

    @model_validator(mode="after")
    def _normalize_values(self) -> "RuntimeConfig":
        text = str(self.mode or "").strip().lower()
        # Deliberately not importing runtime_mode here: secrets is reverse-imported by
        # runtime_mode.stored(); an in-function import can break the cycle but a module-level one can't.
        # There are only two possible values, so inline them directly.
        self.mode = text if text in ("local", "colab") else ""
        if self.mode:
            self.asked = True
        return self


class RemoteAccessConfig(BaseModel):
    """Reaching the studio from outside (phone) — see services/tunnel.py.

    - `provider`: "cloudflare" (quick tunnel, new random address every start),
      "tailscale" (Tailscale Funnel, permanent https://<pc>.<tailnet>.ts.net) or
      "ngrok" (permanent with the free static domain).
    - `autostart`: open the link as soon as the studio starts.
    - `access_key`: the `?k=` key. Kept across restarts so a permanent address
      stays a permanent *link*; rotated only on request.
    """
    provider: str = "cloudflare"
    autostart: bool = False
    access_key: str = ""
    ngrok_authtoken: str = ""
    ngrok_domain: str = ""

    @model_validator(mode="after")
    def _normalize_values(self) -> "RemoteAccessConfig":
        p = str(self.provider or "").strip().lower()
        self.provider = p if p in ("cloudflare", "tailscale", "ngrok") else "cloudflare"
        d = str(self.ngrok_domain or "").strip()
        for prefix in ("https://", "http://"):
            if d.lower().startswith(prefix):
                d = d[len(prefix):]
        self.ngrok_domain = d.strip("/")
        return self


class ProxyConfig(BaseModel):
    """Global HTTP/HTTPS proxy configuration."""
    enabled: bool = False
    http_proxy: str = ""  # e.g.: http://127.0.0.1:7890
    https_proxy: str = ""
    no_proxy: str = ""    # exceptions, e.g. localhost,127.0.0.1


# Per-type download-source selection key (dual-source types). Types fixed to HF (cltagger / t5 /
# taeflux) aren't in this list — routing forces HF for them. training = the whole pre-training
# group: Anima main + VAE + qwen3 + t5.
DOWNLOAD_SOURCE_TYPES: tuple[str, ...] = ("training", "wd14", "upscaler")
DOWNLOAD_SOURCE_VALUES: tuple[str, ...] = ("huggingface", "modelscope")


# ---------------------------------------------------------------------------
# Unified model-source candidates (docs/design/model-source-unification.md)
# ---------------------------------------------------------------------------


class SourceCandidate(BaseModel):
    """A model-source candidate the user added.

    - kind="download": `repo` (an HF/MS repo id) + single-file assets also need `filename`
      (upscaler / main model); directory-type assets (wd14 / eval / cltagger) only have repo.
    - kind="local": `path` an absolute local path (file or directory). Never deletes the file;
      removing it only takes it off the candidate list.
    - `extra`: domain-specific keys (cltagger: model_path / tag_mapping_path, the two file paths
      relative to the repo root).
    """
    kind: Literal["download", "local"]
    repo: str = ""
    filename: str = ""
    path: str = ""
    extra: dict[str, str] = Field(default_factory=dict)

    def identity(self) -> tuple[str, str, str]:
        """Dedup identity key: download=(repo, filename), local=(path,)."""
        if self.kind == "download":
            return ("download", self.repo, self.filename)
        return ("local", self.path, "")


# Repo-type domains (selected value = repo id or an absolute local path). These domains
# participate in the shared invariant "selected value not in builtins and not in candidates →
# auto-add a candidate"; upscaler (filename semantics + disk-scan fallback) and the main-model
# families (the families registry lives at the services layer, whose fallback resolution is
# already solid) aren't in this list — their candidates are maintained entirely by their own
# endpoints.
MODEL_SOURCE_REPO_DOMAINS: tuple[str, ...] = (
    "wd14", "cltagger", "eval_clip", "eval_dino", "eval_ccip",
)

#: Candidate domain for the family-agnostic VAE weights (selected value lands in models.selected_vae).
VAE_DOMAIN = "vae"
#: Suffix for per-family text-encoder candidate domains: `anima_te` / `krea2_te` (selected value
#: lands in models.selected_te[family], sharing the same field as the official variant key).
TE_DOMAIN_SUFFIX = "_te"


def te_domain(family: str) -> str:
    """Family id → text-encoder candidate domain (`krea2` → `krea2_te`)."""
    return f"{family}{TE_DOMAIN_SUFFIX}"


def te_domain_family(domain: str) -> str:
    """`krea2_te` → `krea2`; returns an empty string for a non-TE domain."""
    if domain.endswith(TE_DOMAIN_SUFFIX) and len(domain) > len(TE_DOMAIN_SUFFIX):
        return domain[: -len(TE_DOMAIN_SUFFIX)]
    return ""


def is_weight_asset_domain(domain: str) -> bool:
    """VAE / text-encoder domain? — their local candidates don't go into the models.custom compat
    surface (that field only holds "per-family local main models", see ModelsConfig.custom)."""
    return domain == VAE_DOMAIN or bool(te_domain_family(domain))


def is_abs_path(value: str) -> bool:
    """Cross-platform absolute-path check (Windows drive letter / UNC / posix root); a repo id
    like `owner/name` is relative → False."""
    return (
        PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
    )


class Secrets(BaseModel):
    gelbooru: GelbooruConfig = Field(default_factory=GelbooruConfig)
    danbooru: DanbooruConfig = Field(default_factory=DanbooruConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    reg: RegConfig = Field(default_factory=RegConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    modelscope: ModelScopeConfig = Field(default_factory=ModelScopeConfig)
    eval_metrics: EvalMetricModelsConfig = Field(
        default_factory=EvalMetricModelsConfig
    )
    # The old global download source (retired to a "migration seed"). No UI toggle anymore; new
    # models each pick a source per type in download_sources. Kept only for compat with old
    # secrets.json: on load, its value seeds any download_sources type not yet set, so old users
    # (especially those in mainland China who had set modelscope) don't silently fall back to HF.
    download_source: str = "huggingface"
    # Per-type download-source selection: {"training"|"wd14"|"upscaler": "huggingface"|"modelscope"}.
    # If the selected source is missing a variant/file, the downloader automatically falls back to the other source.
    download_sources: dict[str, str] = Field(default_factory=dict)
    # JoyCaptionConfig has been folded into llm_tagger's joycaption builtin preset;
    # any leftover joycaption field in secrets.json is migrated then dropped by _migrate_legacy_schema.
    llm_tagger: LLMTaggerConfig = Field(default_factory=LLMTaggerConfig)
    wd14: WD14Config = Field(default_factory=WD14Config)
    cltagger: CLTaggerConfig = Field(default_factory=CLTaggerConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    training: TrainingSecretsConfig = Field(default_factory=TrainingSecretsConfig)
    # This fork: the in-app updater was removed, but SystemConfig is kept (the enable_automagic_v2
    # feature flag + compat for old secrets.json's update_channel field).
    system: SystemConfig = Field(default_factory=SystemConfig)
    # This fork: persisted choice of Colab / Local runtime mode (infrastructure/runtime_mode.py).
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    remote_access: RemoteAccessConfig = Field(default_factory=RemoteAccessConfig)
    # Unified model-source candidates: domain → the candidate list the user added. Domain
    # whitelist validation lives at the API layer (the families registry lives at the services
    # layer). Builtin presets aren't stored here — the full candidate set = code-builtin + this
    # field. The currently selected value is still written to each domain's own field
    # (wd14.model_id / eval_metrics.*_model_name / models.selected, etc — see the two-surface
    # compat contract in docs/design/model-source-unification.md §3).
    model_sources: dict[str, list[SourceCandidate]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _sync_legacy_source_fields(cls, data: Any) -> Any:
        """Full sync of the old candidate fields (wd14.model_ids / models.custom) → model_sources.

        When the inbound dict **carries these keys** (an old on-disk file / an old client's PUT),
        they take precedence and rebuild the corresponding kind's candidate set — preserving the
        old client's add/remove semantics; `update()` already strips these rebuild keys from the
        merge base, so when the new UI only writes model_sources, it won't be overwritten by a
        stale old value. The other half (model_sources → rebuilding the old fields for writing to
        disk) is in the after-validator.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        raw_sources = data.get("model_sources")
        sources: dict[str, list[dict[str, Any]]] = {}
        if isinstance(raw_sources, dict):
            for domain, cands in raw_sources.items():
                if isinstance(cands, list):
                    sources[str(domain)] = [
                        dict(c) for c in cands if isinstance(c, dict)
                    ]

        def _sync(domain: str, kind: str, wanted: list[dict[str, Any]]) -> None:
            cur = sources.get(domain, [])
            kept = [c for c in cur if c.get("kind") != kind]
            sources[domain] = wanted + kept

        wd14_raw = data.get("wd14")
        if isinstance(wd14_raw, dict) and isinstance(wd14_raw.get("model_ids"), list):
            wanted = [
                {"kind": "download", "repo": str(m)}
                for m in wd14_raw["model_ids"]
                if str(m).strip() and str(m) not in DEFAULT_WD14_MODELS
            ]
            _sync("wd14", "download", wanted)

        models_raw = data.get("models")
        if isinstance(models_raw, dict):
            custom = models_raw.get("custom")
            if not isinstance(custom, dict) and isinstance(
                models_raw.get("custom_anima_paths"), list
            ):
                custom = {"anima": models_raw["custom_anima_paths"]}
            if isinstance(custom, dict):
                for family, paths in custom.items():
                    if not isinstance(paths, list):
                        continue
                    wanted = [
                        {"kind": "local", "path": str(p)}
                        for p in paths if str(p).strip()
                    ]
                    _sync(str(family), "local", wanted)

        data["model_sources"] = sources
        return data

    @model_validator(mode="after")
    def _model_sources_invariants(self) -> "Secrets":
        """The shared invariant + compat-surface rebuild for writing to disk.

        1. If a repo-type domain's currently selected value is neither a builtin preset nor in the
           candidates → automatically add a candidate (a generalization of WD14's existing
           "model_id is always visible" invariant; the user must switch away before removing it,
           the frontend enforces this order).
        2. Rebuild the compat surfaces: wd14.model_ids = builtins + download candidates (readable
           on rollback); models.custom = each family's local candidate paths. Both are stripped
           from the merge base in `update()` — the single source of truth is model_sources.
        """
        cltagger_official = str(
            CLTaggerConfig.model_fields["model_id"].default
        )
        selected_by_domain: dict[str, tuple[str, tuple[str, ...]]] = {
            "wd14": (self.wd14.model_id, DEFAULT_WD14_MODELS),
            "cltagger": (self.cltagger.model_id, (cltagger_official,)),
            "eval_clip": (
                self.eval_metrics.clip_model_name,
                (str(EvalMetricModelsConfig.model_fields["clip_model_name"].default),),
            ),
            "eval_dino": (
                self.eval_metrics.dino_model_name,
                (str(EvalMetricModelsConfig.model_fields["dino_model_name"].default),),
            ),
            "eval_ccip": (
                self.eval_metrics.ccip_model_name,
                (str(EvalMetricModelsConfig.model_fields["ccip_model_name"].default),),
            ),
        }
        for domain in MODEL_SOURCE_REPO_DOMAINS:
            sel, builtins = selected_by_domain[domain]
            sel = (sel or "").strip()
            if not sel or sel in builtins:
                continue
            cands = self.model_sources.setdefault(domain, [])
            if any(
                (c.kind == "download" and c.repo == sel)
                or (c.kind == "local" and c.path == sel)
                for c in cands
            ):
                continue
            if is_abs_path(sel):
                cands.append(SourceCandidate(kind="local", path=sel))
            elif domain == "cltagger":
                # fork repo migration carries over the current pair of relative file paths (mirror override retired, D4)
                cands.append(SourceCandidate(
                    kind="download", repo=sel,
                    extra={
                        "model_path": self.cltagger.model_path,
                        "tag_mapping_path": self.cltagger.tag_mapping_path,
                    },
                ))
            else:
                cands.append(SourceCandidate(kind="download", repo=sel))

        # Rebuilding the compat surfaces (written to disk for older versions to read; the
        # runtime-read selected-value fields aren't in this list)
        wd14_downloads = [
            c.repo for c in self.model_sources.get("wd14", [])
            if c.kind == "download" and c.repo
        ]
        self.wd14.model_ids = list(DEFAULT_WD14_MODELS) + [
            m for m in wd14_downloads if m not in DEFAULT_WD14_MODELS
        ]
        custom: dict[str, list[str]] = {}
        for domain, cands in self.model_sources.items():
            if (
                domain in MODEL_SOURCE_REPO_DOMAINS
                or domain == "upscaler"
                or is_weight_asset_domain(domain)
            ):
                continue
            paths = [c.path for c in cands if c.kind == "local" and c.path]
            if paths or domain in self.models.custom:
                custom[domain] = paths
        self.models.custom = custom
        return self

    @model_validator(mode="after")
    def _seed_and_normalize_download_sources(self) -> "Secrets":
        """Migration seed + normalization: the old global download_source → per-type download_sources.

        Types not yet set inherit from the old global value (so existing users don't lose their
        source preference); invalid values fall back to huggingface. Every load does a setdefault
        (idempotent): once a user has explicitly chosen a source for a type, it's never overwritten.
        """
        legacy = str(self.download_source or "").strip().lower()
        if legacy not in DOWNLOAD_SOURCE_VALUES:
            legacy = "huggingface"
        for key in DOWNLOAD_SOURCE_TYPES:
            self.download_sources.setdefault(key, legacy)
        for key, val in list(self.download_sources.items()):
            if str(val).strip().lower() not in DOWNLOAD_SOURCE_VALUES:
                self.download_sources[key] = "huggingface"
            else:
                self.download_sources[key] = str(val).strip().lower()
        return self


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load() -> Secrets:
    """Reads secrets.json; returns a default instance if missing or corrupted (never raises)."""
    if not SECRETS_FILE.exists():
        return Secrets()
    try:
        raw = json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
        raw = _migrate_legacy_schema(raw) if isinstance(raw, dict) else raw
        return Secrets.model_validate(raw)
    except Exception:
        # A corrupted file shouldn't block Studio from starting; fall back to defaults
        return Secrets()


def save(s: Secrets) -> None:
    SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SECRETS_FILE.write_text(s.model_dump_json(indent=2), encoding="utf-8")


def get(path: str) -> Any:
    """Get a value by dotted path, e.g. `get('wd14.threshold_general')`."""
    cur: Any = load()
    for seg in path.split("."):
        cur = getattr(cur, seg)
    return cur


def update(partial: dict[str, Any]) -> Secrets:
    """Deep-merges `partial` into the current persisted value; returns the new Secrets and saves it to disk.

    - A leaf value of MASK ("***") in `partial` means "keep the original value unchanged".
    - llm_tagger.presets is list[dict], matched and deep-merged by preset.id, so when the frontend
      PUTs the whole list, a single preset's api_key=MASK still keeps its original value.
    - Fields not mentioned keep their old value.
    """
    current_dict = load().model_dump()
    # Strip the read-compat computed keys of models (selected_anima / custom_anima_paths): they
    # aren't storage fields, and leaving them in the merge base would let them, as "inbound legacy
    # keys", get overwritten via _migrate_legacy_model_fields by the newly-written selected/custom
    # from partial. A genuinely inbound legacy key (from an old client) is in partial and wins as usual.
    models_base = current_dict.get("models")
    if isinstance(models_base, dict):
        models_base.pop("selected_anima", None)
        models_base.pop("custom_anima_paths", None)
        # model_sources compat-rebuild key (same reasoning as above): leaving it in the merge base
        # would let _sync_legacy_source_fields overwrite the newly-written model_sources from
        # partial with a stale value.
        models_base.pop("custom", None)
    wd14_base = current_dict.get("wd14")
    if isinstance(wd14_base, dict):
        wd14_base.pop("model_ids", None)
    merged = _deep_merge(current_dict, partial)
    new = Secrets.model_validate(merged)
    save(new)
    return new


def to_masked_dict(s: Secrets) -> dict[str, Any]:
    """Returned by GET /api/secrets; non-empty sensitive fields are replaced with MASK.

    SENSITIVE_FIELDS supports a `*` wildcard (for list-of-dict cases like
    llm_tagger.presets.*.api_key).
    """
    d = s.model_dump()
    for path in SENSITIVE_FIELDS:
        _apply_mask(d, path.split("."))
    return d


def _apply_mask(node: Any, segs: list[str]) -> None:
    if not segs:
        return
    head, *rest = segs
    if head == "*":
        if isinstance(node, list):
            for item in node:
                _apply_mask(item, rest)
        return
    if not isinstance(node, dict):
        return
    if not rest:
        if node.get(head):
            node[head] = MASK
        return
    _apply_mask(node.get(head), rest)


# ---------------------------------------------------------------------------
# WandB preset import/export (0.18 preset-ification)
# ---------------------------------------------------------------------------


def get_wandb_preset(preset_id: str) -> Optional["WandBPresetConfig"]:
    """Gets a preset by id (**including the real api_key**, bypassing the mask) — for the explicit export endpoint only."""
    for preset in load().wandb.presets:
        if preset.id == preset_id:
            return preset
    return None


def import_wandb_preset(
    data: Any, fallback_label: str = ""
) -> tuple[Secrets, "WandBPresetConfig"]:
    """Imports a wandb preset: an id collision auto-appends a suffix, and the imported preset becomes the current selection.

    - Compatible with the old frontend JSON export format ``{kind, version, preset: {...}}`` (auto-unwrapped)
    - An ``api_key == MASK`` sentinel (from an old client's export) is treated as empty; a backup
      file with a real key is restored as-is
    - Raises a pydantic ValidationError on invalid values, which the caller turns into a 400
    """
    if not isinstance(data, dict):
        raise ValueError("preset data must be a mapping")
    payload = dict(data)
    inner = payload.get("preset")
    if isinstance(inner, dict):
        payload = dict(inner)
    if str(payload.get("api_key") or "") == MASK:
        payload["api_key"] = ""

    label = str(payload.get("label") or fallback_label or "imported").strip() or "imported"
    slug = "".join(
        ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label
    ).strip("_") or "imported"

    s = load()
    used = {p.id for p in s.wandb.presets}
    pid, idx = slug, 1
    while pid in used:
        idx += 1
        pid = f"{slug}_{idx}"

    preset = WandBPresetConfig(**{**payload, "id": pid, "label": label})
    s.wandb.presets.append(preset)
    s.wandb.current_preset = preset.id
    new = Secrets.model_validate(s.model_dump())  # re-run the validator (dedup / fallback safety net)
    save(new)
    return new, preset


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _migrate_legacy_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """One-time migration from the old schema → the new schema.

    Migration targets (PR #18 schema → preset-unified schema):
    1. Top-level LLMTaggerConfig.base_url / api_key / model / endpoint / temperature /
       max_tokens / max_side / jpeg_quality / max_image_mb / timeout / max_retries
       sink down into each preset
    2. prompt_presets[{id,label,prompt,builtin,output_format}] are upgraded into full presets
       (inheriting the top-level endpoint + generation-parameter fields)
    3. prompt_preset = "custom" + a non-empty custom_prompt → creates a `user_custom` preset
    4. JoyCaptionConfig.base_url / model / prompt_template → written into the joycaption preset
       (base_url/model overwrite directly; a non-default prompt_template creates `user_joycaption`)
    5. Deletes the raw["joycaption"] field
    6. system.show_dev_channel=true → system.update_channel="dev" (ADR 0005)

    Idempotent: a new-schema payload (llm_tagger containing current_preset / presets) is returned as-is.
    """
    # 6. One-time migration of the system channel preference (done first regardless of which llm_tagger path runs below)
    sys_raw = raw.get("system")
    if isinstance(sys_raw, dict):
        # New field already explicitly set → don't overwrite (idempotent)
        if "update_channel" not in sys_raw and sys_raw.get("show_dev_channel") is True:
            sys_raw["update_channel"] = "dev"

    # 10. ram_guard default flipped to off (upstream v0.23.1): since save() always writes the whole
    #     file, "the user explicitly enabled it" and "the old default true was passively written" are
    #     indistinguishable (same precedent as step 8), so old disk files have their ram_guard value
    #     discarded once, returning all users to the new default (off). The test is the sentinel
    #     **key being absent** — not the value: any disk written by code after this version always
    #     carries this key, and its value is then the source of truth; testing "value is false" would
    #     also discard a newly-installed user's first explicit enable. The sentinel is only written to
    #     disk on the next save(); discarding repeatedly before that save is idempotent (it's still
    #     discarding the old disk value).
    sys_raw_rg = raw.setdefault("system", {})
    if isinstance(sys_raw_rg, dict) and "ram_guard_default_off" not in sys_raw_rg:
        for section in ("generate", "training"):
            sec_raw = raw.get(section)
            if isinstance(sec_raw, dict):
                sec_raw.pop("ram_guard", None)
        sys_raw_rg["ram_guard_default_off"] = True

    # 8. R-1 resource tiers (0.17): queue.allow_gpu_during_train is retired. Its semantics changed
    #    (the old switch let even base-model-scale tasks like eval_samples through, an OOM hazard;
    #    the new light_tasks_during_train switch only covers the light tier and defaults on), and
    #    since save() always writes the whole file, "explicitly false" and "default false" are
    #    indistinguishable — so the old value isn't migrated, just dropped.
    q_raw = raw.get("queue")
    if isinstance(q_raw, dict):
        q_raw.pop("allow_gpu_during_train", None)

    # 9. WandB preset-ification (0.18): the old flat wandb {enabled, api_key, project, ...} →
    #    {enabled, current_preset, presets: [{id: "default", ...}]}. enabled stays at the top level
    #    (the overall switch doesn't change with the preset), everything else sinks as a whole into
    #    the id="default" preset. Idempotent: skips immediately if a presets key already exists.
    wb_raw = raw.get("wandb")
    if isinstance(wb_raw, dict) and "presets" not in wb_raw:
        enabled = bool(wb_raw.pop("enabled", False))
        preset = {**wb_raw, "id": "default", "label": "Default"}
        raw["wandb"] = {
            "enabled": enabled,
            "current_preset": "default",
            "presets": [preset],
        }

    # 7. gelbooru's image-ingestion settings moved to the global download.* (these three were
    #    already shared by all booru downloads / reg / local uploads, and didn't belong under
    #    gelbooru). Only moves fields the download side hasn't explicitly set — idempotent.
    gel_raw = raw.get("gelbooru")
    if isinstance(gel_raw, dict):
        dl_raw = raw.setdefault("download", {})
        if isinstance(dl_raw, dict):
            for k in ("save_tags", "convert_to_png", "remove_alpha_channel"):
                if k in gel_raw and k not in dl_raw:
                    dl_raw[k] = gel_raw[k]
                gel_raw.pop(k, None)

    llm_old = raw.get("llm_tagger")
    if not isinstance(llm_old, dict):
        # No llm_tagger field: probably an even older secrets.json; leave it to pydantic's defaults
        raw.pop("joycaption", None)
        return raw

    # Already the new schema: just clean up any leftover joycaption field and return
    if "presets" in llm_old or "current_preset" in llm_old:
        raw.pop("joycaption", None)
        return raw

    # Old top-level fields (PR #18 schema)
    def _get(key: str, default: Any) -> Any:
        val = llm_old.get(key)
        return default if val is None else val

    old_base_url = _get("base_url", "")
    old_api_key = _get("api_key", "")
    old_model = _get("model", "")
    old_model_ids = list(_get("model_ids", []) or [])
    old_endpoint = _get("endpoint", "chat_completions")
    old_temperature = _get("temperature", 0.2)
    old_max_tokens = _get("max_tokens", 700)
    old_timeout = _get("timeout", 60)
    old_max_retries = _get("max_retries", 3)
    old_concurrency = _get("concurrency", 1)
    old_requests_per_second = _get("requests_per_second", 0.0)
    old_max_requests_per_minute = _get("max_requests_per_minute", 0)
    old_max_side = _get("max_side", 1280)
    old_jpeg_quality = _get("jpeg_quality", 85)
    old_max_image_mb = _get("max_image_mb", 5.0)
    old_custom_prompt = str(_get("custom_prompt", "")).strip()
    old_prompt_preset = _get("prompt_preset", "style_json")
    old_prompt_presets = list(_get("prompt_presets", []) or [])

    from .llm_presets import builtin_llm_presets  # local import to avoid a cycle

    builtin_defaults = {item["id"]: item for item in builtin_llm_presets()}

    def _endpoint_fields() -> dict[str, Any]:
        return {
            "base_url": old_base_url,
            "api_key": old_api_key,
            "model": old_model,
            "model_ids": list(old_model_ids),
            "endpoint": old_endpoint,
            "temperature": old_temperature,
            "max_tokens": old_max_tokens,
            "max_side": old_max_side,
            "jpeg_quality": old_jpeg_quality,
            "max_image_mb": old_max_image_mb,
            "timeout": old_timeout,
            "max_retries": old_max_retries,
            "concurrency": old_concurrency,
            "requests_per_second": old_requests_per_second,
            "max_requests_per_minute": old_max_requests_per_minute,
        }

    new_presets: list[dict[str, Any]] = []
    for p in old_prompt_presets:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip()
        if not pid:
            continue
        base_default = builtin_defaults.get(pid, {})
        merged = {
            **_endpoint_fields(),
            "id": pid,
            "label": p.get("label") or base_default.get("label") or pid,
            "builtin": pid in builtin_defaults,
            "prompt": p.get("prompt") or base_default.get("prompt", ""),
            "output_format": p.get("output_format") or base_default.get("output_format", "json"),
        }
        # The joycaption builtin uses its own recommended temperature/max_tokens (if the user never touched the old top-level values)
        if pid in builtin_defaults and old_temperature == 0.2 and old_max_tokens == 700:
            merged["temperature"] = base_default.get("temperature", old_temperature)
            merged["max_tokens"] = base_default.get("max_tokens", old_max_tokens)
        new_presets.append(merged)

    current = str(old_prompt_preset or "").strip() or "style_json"
    if current == "custom" and old_custom_prompt:
        new_presets.append({
            **_endpoint_fields(),
            "id": "user_custom",
            "label": "Custom",
            "builtin": False,
            "prompt": old_custom_prompt,
            "output_format": "json",
        })
        current = "user_custom"

    # JoyCaption card merge ----
    joycap = raw.get("joycaption") if isinstance(raw.get("joycaption"), dict) else {}
    joy_base_url = str(joycap.get("base_url", "") or "").strip()
    joy_model = str(joycap.get("model", "") or "").strip()
    joy_prompt = str(joycap.get("prompt_template", "") or "").strip()

    joycap_default_base = "http://localhost:8000/v1"
    joycap_default_model = "fancyfeast/llama-joycaption-beta-one-hf-llava"
    joycap_default_prompt = "Descriptive Caption"

    if joy_base_url or joy_model:
        # Write into the joycaption preset (create one if old prompt_presets didn't have joycaption)
        joy_preset = next((p for p in new_presets if p["id"] == "joycaption"), None)
        if joy_preset is None:
            joy_default = builtin_defaults.get("joycaption", {})
            joy_preset = {**_endpoint_fields(), **joy_default, "id": "joycaption", "builtin": True}
            new_presets.append(joy_preset)
        if joy_base_url and joy_base_url != joycap_default_base:
            joy_preset["base_url"] = joy_base_url
        if joy_model and joy_model != joycap_default_model:
            joy_preset["model"] = joy_model
            if joy_model not in joy_preset.get("model_ids", []):
                joy_preset["model_ids"] = [joy_model, *joy_preset.get("model_ids", [])]
    if joy_prompt and joy_prompt != joycap_default_prompt:
        # User edited the joycaption prompt_template → create a user custom preset, keeping this prompt
        new_presets.append({
            "base_url": joy_base_url or joycap_default_base,
            "api_key": "",
            "model": joy_model or joycap_default_model,
            "model_ids": [joy_model] if joy_model else [],
            "endpoint": "chat_completions",
            "temperature": 0.6,
            "max_tokens": 300,
            "max_side": 1280,
            "jpeg_quality": 85,
            "max_image_mb": 5.0,
            "timeout": 60,
            "max_retries": 3,
            "concurrency": 1,
            "requests_per_second": 0.0,
            "max_requests_per_minute": 0,
            "id": "user_joycaption",
            "label": "JoyCaption (custom prompt)",
            "builtin": False,
            "prompt": joy_prompt,
            "output_format": "text",
        })

    raw["llm_tagger"] = {
        "current_preset": current,
        "presets": new_presets,
    }
    raw.pop("joycaption", None)
    return raw


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Merges patch into base: nested dicts are merged recursively; a leaf value of MASK is dropped.

    When a list[dict] has an id field (e.g. llm_tagger.presets), it's deep-merged by id: presets in
    base that patch didn't touch are kept; presets present in patch are deep-merged with the
    same-id entry in base.
    """
    out = dict(base)
    for key, val in patch.items():
        if (
            isinstance(val, list)
            and isinstance(out.get(key), list)
            and val
            and all(isinstance(x, dict) and "id" in x for x in val)
            and all(isinstance(x, dict) and "id" in x for x in out[key])
        ):
            base_by_id = {x["id"]: x for x in out[key]}
            merged_list: list[Any] = []
            seen: set[str] = set()
            for px in val:
                bx = base_by_id.get(px["id"], {})
                merged_list.append(_deep_merge(bx, px))
                seen.add(px["id"])
            out[key] = merged_list
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        elif val == MASK:
            # keep the old value
            continue
        else:
            out[key] = val
    return out


def has_danbooru_credentials() -> bool:
    """Frontend / endpoint check for whether Danbooru auth is already configured."""
    d = load().danbooru
    return bool(d.username and d.api_key)


def has_gelbooru_credentials() -> bool:
    """Convenience: frontend / endpoint check for whether Gelbooru is already configured."""
    g = load().gelbooru
    return bool(g.user_id and g.api_key)


def has_credentials_for(api_source: str) -> bool:
    """Per-download-channel "can it run" check (both sources require binding, no anon):
    - gelbooru: requires user_id + api_key (the API mandates it)
    - danbooru: requires username + api_key (mandatory as of PR #38, after CF tightened up)
    """
    if api_source == "gelbooru":
        return has_gelbooru_credentials()
    if api_source == "danbooru":
        return has_danbooru_credentials()
    return False
