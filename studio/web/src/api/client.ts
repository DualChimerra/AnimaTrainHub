// Thin wrapper around interaction with the FastAPI daemon.
// In dev it's forwarded by the Vite proxy to 127.0.0.1:8765; in production it's same-origin with the API.

// ADR-0009 PR-3 C3: write the X-Trace-Id returned by the backend to an atom, so ErrorBoundary /
// window.onerror reports can attach it (letting developers join "the last API failure before the
// frontend crashed" with "the toast the user actually saw" in the server log).
import { setLastApiTraceId } from '../lib/errors/report'
import i18n from '../i18n'

export interface HealthResponse {
  status: string
  version: string
}

export interface GpuStats {
  index: number
  name: string
  util_pct: number
  vram_used_gb: number
  vram_total_gb: number
  temp_c: number | null
}

export interface SystemStats {
  cpu_pct: number
  ram_used_gb: number
  ram_total_gb: number
  /** null = NVML unavailable (no NVIDIA / driver missing); [] = NVML available but 0 cards. Neither shows a GPU pill. */
  gpu: GpuStats[] | null
}

export interface SchemaProperty {
  type?: string | string[]
  default?: unknown
  description?: string
  enum?: unknown[]
  minimum?: number
  maximum?: number
  exclusiveMinimum?: number
  exclusiveMaximum?: number
  group?: string
  control?: string
  cli_alias?: string
  show_when?: string
  /** Option-level show_when (multi-model P4-2): enum value -> expression (same syntax as show_when),
   * options evaluating to false are hidden from the dropdown. Unlisted options are always visible; the
   * currently selected value stays visible even if gated (the form reflects config as-is, the backend
   * rejects out-of-range values on validation). */
  option_show_when?: Record<string, string>
  /** When this expression is true, the field is disabled in the UI (value auto-falls back to default via SchemaForm).
   * Expression syntax matches show_when: `key==value` / `key!=value`.
   * Example: lr_scheduler is disabled when optimizer_type=prodigy_plus_schedulefree. */
  disable_when?: string
  /** Option-level disable value (knife 2 / R2 v2, D4): enum value -> when the expression is true, that
   * option is greyed out and unselectable (not hidden -- the user can see why it's unselectable via
   * the disable_hint title). The backend's _enforce_disable_rules consumes the same declaration for validation. */
  option_disable_when?: Record<string, string>
  /** Value to write back when disable_when triggers; falls back to default if unset. */
  disable_value?: unknown
  /** Hint badge text shown when disable_when triggers. */
  disable_hint?: string
  /** Conditional description text: replaces description when the alt_description_when expression is true. */
  alt_description?: string
  /** Condition expression that triggers alt_description, same syntax as show_when. */
  alt_description_when?: string
  /** Advanced-mode-only field, hidden in simple mode. */
  advanced?: boolean
  /** A field the backend marked hidden=True: the value still passes through / saves with ConfigData, but
   * SchemaForm doesn't render it. Used as a fallback for "this field is meaningless to the current user
   * group but the schema must keep it". */
  hidden?: boolean
  anyOf?: Array<{ type?: string }>
  items?: SchemaProperty
}

export interface JsonSchema {
  properties: Record<string, SchemaProperty>
  required?: string[]
}

export interface SchemaResponse {
  schema: JsonSchema
  groups: Array<{ key: string; label: string; default_collapsed?: boolean }>
}

export interface PresetSummary {
  name: string
  path: string
  updated_at: number
}

/** Before PP0 this was called ConfigSummary -- keeping the alias around for a while so external code doesn't break. */
export type ConfigSummary = PresetSummary

export type ConfigData = Record<string, unknown>

// ---- secrets (settings) ---------------------------------------------------

export interface GelbooruConfig {
  user_id: string
  api_key: string
}

export interface DanbooruConfig {
  username: string
  api_key: string
  account_type: 'free' | 'gold' | 'platinum'
}

export interface DownloadGlobalConfig {
  exclude_tags: string[]
  /** PP9 -- Booru concurrency pool: worker count. */
  parallel_workers: number
  /** PP9 -- API host (gelbooru.com / danbooru.donmai.us) rate limit. */
  api_rate_per_sec: number
  /** PP9 -- CDN host (img*.gelbooru.com / cdn.donmai.us) rate limit. */
  cdn_rate_per_sec: number
  /** Image ingest processing (shared by booru downloads / reg / local uploads). */
  save_tags: boolean
  convert_to_png: boolean
  remove_alpha_channel: boolean
}

export interface RegConfig {
  /** Global default excluded tags for reg-set generation; seeds a new build when it has no local selection. */
  default_excluded_tags: string[]
}

export interface HuggingFaceConfig {
  token: string
  /** PR-S3 -- HF model download endpoint.
   *  `""` -> huggingface_hub default (direct to huggingface.co); recommended for overseas users
   *  `"https://hf-mirror.com"` -> mainland China default (this project's primary market)
   *  any other URL -> custom reverse proxy / self-hosted mirror */
  endpoint: string
}

/** A WandB account + upload-policy preset (0.18 preset-ified, aligned with the LLMPreset pattern). */
export interface WandBPreset {
  id: string
  label: string
  api_key: string
  project: string
  entity: string
  base_url: string
  mode: 'online' | 'offline' | 'disabled'
  /** Whether to upload training sample images to wandb.ai, on by default; turn off for private / NSFW datasets. */
  log_samples: boolean
  /** Longest side (px) to downscale to before upload */
  sample_max_side: number
  /** Step throttling: when >0, only upload on global_step % N == 0; 0 = no extra throttling */
  sample_every_n_steps: number
  /** Upload the model artifact to wandb */
  upload_model: boolean
  /** Model artifact retention policy: all=every version / last=latest only */
  upload_model_policy: 'all' | 'last'
  /** Upload manually saved training-state artifacts */
  upload_state_manual: boolean
  /** Manual state artifact retention policy */
  upload_state_manual_policy: 'all' | 'last'
  /** Upload auto-saved training-state artifacts */
  upload_state_auto: boolean
  /** Auto state artifact retention policy */
  upload_state_auto_policy: 'all' | 'last'
}

/** Global WandB: only the master switch + preset selector live at the top level; everything else is in the preset. */
export interface WandBConfig {
  enabled: boolean
  current_preset: string
  presets: WandBPreset[]
}

export interface ModelScopeConfig {
  /** ModelScope community token. Optional for public models; required for private models / rate limiting. */
  token: string
}

export interface EvalMetricModelsConfig {
  /** CLIP-T / CLIP-I default model name or local dir. */
  clip_model_name: string
  /** DINO-I default model name or local dir. */
  dino_model_name: string
  /** CCIP (anime character identity) default ONNX variant name. */
  ccip_model_name: string
  /** Which eval metrics are enabled (Settings checkboxes); eval only computes checked ones. */
  enabled_metrics: string[]
  /** Post-training eval also runs a pure-base-model (scale=0) control; each metric reports delta = checkpoint - baseline. */
  eval_baseline_enabled: boolean
}

/** Eval metric registry entry (catalog.eval_metric_catalog): used for the Settings checkbox list. */
export interface EvalMetricCatalogItem {
  key: string
  label: string
  runner: string
  models: string[]
  default: boolean
  desc: string
  note: string
}

export interface EvalMetricSpec {
  key: string
  label: string
  question: string
  requires: string[]
  higher_is_better: boolean
}

export interface EvalMetricState {
  key: string
  label?: string
  status: 'not_run' | 'pending' | 'running' | 'done' | 'failed' | 'unavailable' | string
  value: number | null
  reason?: string
  question?: string
  requires?: string[]
  higher_is_better?: boolean
  count?: number
  model_name?: string
  job_id?: number
}

export interface EvalMetricResult {
  schema_version: number
  has_metrics: boolean
  status: string
  run_id: string
  project_id?: number
  project_slug?: string
  version_id?: number
  version_label?: string
  created_at?: number | null
  updated_at?: number | null
  manifest_digest?: string
  checkpoint?: {
    kind?: string
    label?: string
    path?: string
    value?: number
    mtime?: number
  }
  metrics: Record<string, unknown>
  metric_states: Record<string, EvalMetricState>
  summary?: Record<string, number>
  /** Pure-base-model (lora_scale=0) control run; not shown as a checkpoint, only used to compute delta. */
  baseline?: boolean
  /** Net gain per metric relative to baseline: delta = checkpoint value - baseline value. */
  delta?: Record<string, number>
  /** Per-metric baseline values (for reference). */
  baseline_metrics?: Record<string, number>
  /** Status of the sample-generation stage (eval_samples run.json) plus a per-image summary
   *  {total, pending, running, done, failed}. Sample generation is the most time-consuming part
   *  of evaluation; used to show a "generating done/total" sub-progress. */
  sample_run?: {
    run_id: string
    path?: string
    status: string
    summary: Record<string, number>
    created_at?: number | null
    updated_at?: number | null
  }
}

/** A post-training / manual evaluation job (inline in-training evaluation has no job). Related to a
 *  checkpoint row via run_id, used to fetch the raw log (including errors). */
export interface EvalJobInfo {
  id: number
  kind: 'eval_samples' | 'eval_clip' | 'eval_dino' | string
  status: string
  run_id?: string | null
  checkpoint_path?: string | null
}

export interface EvalMetricsListResponse {
  metric_specs: EvalMetricSpec[]
  cache: {
    embeddings_dir: string
    entries: Array<{ key: string; path: string; file_count: number; size_bytes: number }>
  }
  results: EvalMetricResult[]
}

/** A single item in the preset messages sequence.
 *  - type='text': plain text, must specify role; content is the prompt text
 *  - type='image': image placeholder item, the backend fills in the current image while tagging;
 *    the UI can't edit content but can drag it to reorder
 */
export interface LLMMessage {
  type: 'text' | 'image'
  role: 'system' | 'user' | 'assistant'
  content: string
}

/** A single LLM tagger preset = a full set of endpoint + messages + generation params.
 *  builtin only flags that the id is in the built-in list (used to show "Reset to default" in the
 *  UI); it doesn't lock the fields.
 */
export interface LLMPreset {
  id: string
  label: string
  builtin: boolean
  base_url: string
  api_key: string
  model: string
  model_ids: string[]
  endpoint: 'chat_completions' | 'responses'
  messages: LLMMessage[]
  output_format: 'json' | 'text'
  /** Local ONNX tagger used to pre-tag images and inject {{tags}} into messages ('' = off). */
  assist_tagger: string
  temperature: number
  max_tokens: number
  max_side: number
  jpeg_quality: number
  max_image_mb: number
  timeout: number
  max_retries: number
  concurrency: number
  requests_per_second: number
  max_requests_per_minute: number
}

export interface LLMTaggerConfig {
  current_preset: string
  presets: LLMPreset[]
}

export interface LLMConnectionTestResult {
  ok: boolean
  endpoint: LLMPreset['endpoint']
  endpoint_url: string
  model: string
  elapsed_ms: number
  status_code: number | null
  response_preview: string
  error: string
  request_shape: string
}

export interface WD14Config {
  model_id: string
  /** Candidate model list; the user maintains it under "Settings -> WD14", model_id must be in this list. */
  model_ids: string[]
  threshold_general: number
  threshold_character: number
  blacklist_tags: string[]
  /** PP8 -- batch inference size; forced to 1 on the CPU EP. */
  batch_size: number
}

export interface CLTaggerConfig {
  model_id: string
  model_path: string
  tag_mapping_path: string
  threshold_general: number
  threshold_character: number
  add_copyright_tag: boolean
  add_artist_tag: boolean
  add_meta_tag: boolean
  add_model_tag: boolean
  add_rating_tag: boolean
  add_quality_tag: boolean
  blacklist_tags: string[]
  batch_size: number
}

/** PR-S2 -- PyTorch install status + driver detection + recommended cu tag. */
export type TorchCuTag = 'cu128' | 'cu126' | 'cu124' | 'cu118' | 'cpu'
export interface TorchStatus {
  installed: boolean
  version: string | null              // "2.5.0+cu128"
  cuda_build: TorchCuTag | null       // parsed from the +suffix
  cuda_available: boolean             // torch.cuda.is_available()
  device_name: string | null          // "NVIDIA GeForce RTX 5090"
  cuda_detect: {
    available: boolean
    driver_version: string | null
    gpu_name: string | null
  }
  recommended_cu_tag: TorchCuTag      // recommended based on driver version
  /** CPU wheel installed but an NVIDIA GPU is present -> mis-installed, UI shows a red "reinstall CUDA build" hint. */
  is_cpu_with_gpu: boolean
  /** CUDA wheel installed but cuda.is_available()=False -> driver / WSL issue, pip can't fix it. */
  is_cuda_build_unavailable: boolean
}
/** torch reinstall is always deferred: the server writes a marker, and pip runs on the next launcher
 *  start. This avoids a deadlock on Windows where the torch .pyd is already loaded by the server
 *  process and pip can't replace it. */
export interface TorchReinstallResult {
  pending: true                       // always true, tells the UI to show the "please restart" branch
  target: string                      // what the user passed ("auto" etc.)
  tag: TorchCuTag                     // actually selected (auto already resolved by the server)
  message: string                     // human-readable status message, shown directly in the UI
}

/** PR-7b -- Flash Attention install status + environment detection + candidate GitHub wheels. */
export interface FlashAttnEnv {
  python_tag: string                 // cp311
  cuda_tag: string | null            // cu128 / null = no nvidia-smi and no torch
  cuda_ver: string | null            // 12.8 (bound at PyTorch build time, flash_attn ABI follows it)
  /** Highest CUDA supported by the driver, as reported by nvidia-smi; may differ from cuda_ver.
   * Shown to the user for troubleshooting: "driver supports cu130, PyTorch is cu128, install the cu128 wheel". */
  driver_cuda_ver: string | null
  torch_tag: string | null           // torch2.5
  torch_ver: string | null
  /** 'cu128' / 'cu130' = CUDA build of torch; 'cpu' = CPU build (can't install flash_attn);
   *  null = torch not installed / detection failed. UI uses 'cpu' to trigger a "reinstall CUDA build first" hint. */
  torch_cuda_build: string | null
  platform: 'linux_x86_64' | 'win_amd64' | null
}
export interface FlashAttnCandidate {
  url: string
  name: string                       // flash_attn-2.8.3+cu128torch2.5-cp311-cp311-win_amd64.whl
  notes: string[]                    // compatibility notes (CUDA major version mismatch / Python incompatible)
  usable: boolean                    // false = Python ABI mismatch, UI greys it out but still allows forcing install
}
export interface FlashAttnStatus {
  installed: boolean
  version: string | null
  env: FlashAttnEnv
  candidates: FlashAttnCandidate[]   // sorted by score descending, max 20
  fetch_error: string | null         // GitHub API rate-limited / network error
}
export interface FlashAttnInstallResult {
  installed: boolean
  version: string | null
  url: string
  stdout_tail: string                // last 40 lines of pip output
  restart_required: boolean
}

/** onnxruntime install status + nvidia-smi detection + platform id (used by the frontend to disable
 *  buttons per platform). */
export interface WD14Runtime {
  installed: 'onnxruntime' | 'onnxruntime-gpu' | 'onnxruntime-directml' | null
  version: string | null
  providers: string[]
  cuda_available: boolean
  /** DirectML EP available (true on Windows with onnxruntime-directml installed). */
  directml_available: boolean
  /** Backend sys.platform: 'win32' / 'linux' / 'darwin' etc. The Settings UI uses this to disable
   *  buttons unavailable on the current platform (DirectML is Windows-only; GPU + nvidia-* wheel is
   *  best on Linux). */
  platform: string
  /** The installed package (dist-info) doesn't match the .pyd already imported by the current process
   *  -> Studio needs a restart. */
  restart_required: boolean
  /** PP9.5 -- the actual dlopen error reported when creating an InferenceSession (e.g. missing
   *  libcurand.so.10); non-null means it already auto-fell-back to the CPU EP, and the UI should
   *  prompt the user to install CUDA libraries or switch to DirectML. */
  cuda_load_error: string | null
  /** torch's CUDA major version (the anchor for the onnxruntime-gpu build): 12 / 13 / null.
   *  The installed ORT build must match the same major version, or import-time dlopen hangs
   *  (cu128 torch -> 12). */
  torch_cuda_major?: number | null
  /** The installed ORT's CUDA major version doesn't match torch's (e.g. cu13 installed but torch is cu12). */
  ort_cuda_major_mismatch?: boolean
  /** PP9.5 -- result of preloading torch's bundled CUDA .so files (applied=true only on Linux). */
  preload?: {
    applied: boolean
    platform_skip: boolean
    preloaded: string[]
    errors: [string, string][]
    candidates: number
  } | null
  cuda_detect: {
    available: boolean
    driver_version: string | null
    gpu_name: string | null
  }
}

export interface WD14InstallResult extends WD14Runtime {
  target: string
  installed_pkg: string | null
  installed_version: string | null
  stdout_tail: string
  /** PP9.6 -- for the GPU path, reports the installed nvidia-*-cu12 wheels alongside it; null for the
   *  CPU path or non-Linux. Presence of an `error` field means onnxruntime-gpu installed fine but the
   *  CUDA wheels failed to install (non-fatal). */
  cuda_runtime: {
    installed: string[]
    skipped: string[]
    platform_skip: boolean
    stdout?: string
    error?: string
  } | null
}

export const DEFAULT_WD14_MODELS: readonly string[] = [
  'SmilingWolf/wd-eva02-large-tagger-v3',
  'SmilingWolf/wd-vit-tagger-v3',
  'SmilingWolf/wd-vit-large-tagger-v3',
  'SmilingWolf/wd-v1-4-convnext-tagger-v2',
]

export interface ModelsConfig {
  /** Whether forking a preset to a version automatically overrides the 4 model fields with the
   * global model paths.
   * ON (default): the common case, the 4 fields are disabled in the UI; a fork always uses the
   * Settings global paths.
   * OFF: for users with independent models, a fork respects the preset value, the 4 fields are
   * editable + have a picker. */
  auto_sync_paths: boolean
  /** Root directory for training models; null/empty -> falls back to REPO_ROOT/models/ (change this on cloud machines) */
  root: string | null
  /** The current default base model: either an official variant key (1.0 / preview3-base / ...) or
   * a local .safetensors path from custom_anima_paths.
   * Studio expands it to an absolute path written to yaml.transformer_path when creating a new
   * version; existing versions are left untouched (to keep training reproducible). */
  selected_anima: string
  /** Default base model saved per model family: a variant key or a registered local path. */
  selected: Record<string, string>
  /** Text encoder selected per model family: an official variant (krea2: "bf16"|"fp8", missing = bf16)
   * or an absolute path to a user-registered local encoder directory. Determines the
   * text_encoder_path default for new training versions + the default TE for test generation. */
  selected_te?: Record<string, string>
  /** Selected VAE: empty string = the official qwen_image_vae location, otherwise a local
   * .safetensors absolute path (family-agnostic, both families share one selection). */
  selected_vae?: string
  /** User-registered local custom base models (.safetensors absolute paths). Used for fine-tune
   * training / test generation on fine-tuned weights; only registers the path, doesn't download or copy. */
  custom_anima_paths: string[]
  /** Default preprocessing upscaler: a preset label ("4x-AnimeSharp" etc.) or a custom filename
   * ("my-anime.pth"). Used by the Preprocess page and the worker to resolve the weights path. */
  selected_upscaler: string
}

export interface QueueConfig {
  /** R-1 resource tier: whether the exclusive tier (train/regularization AI/generate/eval generate)
   *  lets the light tier (tagging/upscale/regularization build/eval metrics, small models) run
   *  concurrently while it's active. Default true. The exclusive tier never runs in parallel with
   *  itself regardless of this switch. */
  light_tasks_during_train: boolean
}

/** Phase 2 commit 14 -- test-generation daemon behavior. */
export interface GenerateSecretsConfig {
  /** TAEFlux intermediate-step preview throttle. 0=off; >0 -> the daemon pushes a 256px JPEG every N steps.
   * The daemon silently falls back when the model is missing (no preview doesn't block generation). */
  preview_every_n_steps: number
  /** Default attention backend (design decision: the user configures it once, not on every
   * generation). Auto-injected when enqueuing from the Generate page; switched from the Settings
   * training tab. */
  attention_backend: AttentionBackend
  /** VAE decode precision for test generation. bf16 (default) matches ComfyUI's auto VAE dtype on
   * modern GPUs; fp32 is full precision (the daemon temporarily offloads DiT/Qwen to free VRAM before decode). */
  vae_precision: 'bf16' | 'fp32'
  /** Auto-unload the model to free VRAM after the test-generation daemon idles for N minutes. 0 =
   * off, the model stays resident until "Clear VRAM" is clicked manually. The timer only runs while
   * idle + model loaded. */
  idle_timeout_minutes: number
  /** Generation task timeout fallback: force-kill the daemon process if it hasn't finished after N
   * minutes (normal cancel doesn't work in a hung state). 0 (default) = disabled. */
  task_timeout_minutes: number
  /** Test-generation VRAM strategy (applies to krea2). auto = decide whether the text encoder and
   * DiT yield based on free VRAM; save_vram = force sequential (lowest peak, a few extra seconds of
   * transfer per image); performance = keep everything resident (highest peak, zero transfer). */
  vram_policy: 'auto' | 'save_vram' | 'performance'
  /** Memory/VRAM headroom guard: before loading a large model, budget RAM and free VRAM against the
   * weight file size, aborting with an error if insufficient. **Off by default** (the estimate is
   * conservative, causing a high false-reject rate on well-provisioned machines); when off,
   * insufficient resources let loading proceed anyway, which may trigger system-wide paging stalls. */
  ram_guard: boolean
  /** When on, each generation is automatically saved to disk at studio_data/test/<date>/{single,xy}/image_N.png.
   * Off by default; compare mode never saves to disk. */
  save_test_images: boolean
}

/** Global training-side behavior switches (Settings -> Training). */
export interface TrainingSecretsConfig {
  /** Memory/VRAM headroom guard for training / AI regularization priors. Same semantics as
   * `generate.ram_guard`, off by default. Block-swap's pinned-memory guard rail **is not affected by
   * this switch** (locked memory can't be paged out, the fix is to lower blocks_to_swap). */
  ram_guard: boolean
}

export interface ProxyConfig {
    enabled: boolean;
    http_proxy: string;
    https_proxy: string;
    no_proxy: string;
}

/** Runtime mode (this fork). `''` = the user hasn't chosen yet -> a chooser pops up on first screen. */
export type RuntimeMode = 'local' | 'colab'

export interface RuntimeConfig {
  /** `''` / `'local'` / `'colab'`. Empty string means not chosen. */
  mode: RuntimeMode | ''
  /** Whether the user has already gone through the chooser flow once (must be true when mode is non-empty). */
  asked: boolean
}

/** Payload of GET/PUT /api/runtime. */
export interface RuntimeInfo {
  /** The effective user choice (env override takes priority); `''` = not chosen yet. */
  mode: RuntimeMode | ''
  /** The choice persisted in secrets (excluding env override). */
  stored: RuntimeMode | ''
  /** Backend detection result, only used to preselect, never overrides the user's decision. */
  detected: RuntimeMode
  /** Fallback for "need a value right now": mode || detected. */
  effective: RuntimeMode
  /** Value of ALS_RUNTIME_MODE (unset = `''`). */
  env_override: RuntimeMode | ''
  /** true = an env var pins the mode, the UI neither pops a dialog nor allows changing it. */
  locked: boolean
  /** Detection signals, visible when the settings section is expanded (lets the user check why a
   *  mode was inferred). */
  signals: Record<string, boolean>
  modes: RuntimeMode[]
  environment: {
    platform: string
    python: string
    studio_data: string
    studio_data_env: string
    disk_total: number | null
    disk_free: number | null
    gpu: string
  }
}

/** Autocomplete tag list — meta. kind=default: downloaded automatically on
 *  first start; kind=user: uploaded by hand. */
export interface TagDictionaryMeta {
  source_name: string
  source_url: string
  entry_count: number
  downloaded_at: number
  kind: 'default' | 'user'
}

export interface TagDictionaryMetaResponse {
  loaded: boolean
  meta: TagDictionaryMeta | null
}

export interface TagDictionaryPayload {
  tags: string[]
  meta: TagDictionaryMeta
}

export interface Secrets {
  gelbooru: GelbooruConfig
  danbooru: DanbooruConfig
  download: DownloadGlobalConfig
  reg: RegConfig
  huggingface: HuggingFaceConfig
  wandb: WandBConfig
  modelscope: ModelScopeConfig
  eval_metrics: EvalMetricModelsConfig
  /** Legacy global download source (retired to a migration seed, no UI). New models each choose per-type in download_sources. */
  download_source: string
  /** Download source per type: {training|wd14|upscaler: 'huggingface'|'modelscope'}. Types pinned to HF aren't included. */
  download_sources: Record<string, string>
  // JoyCaption has been merged into llm_tagger's builtin preset
  llm_tagger: LLMTaggerConfig
  wd14: WD14Config
  cltagger: CLTaggerConfig
  models: ModelsConfig
  queue: QueueConfig
  generate: GenerateSecretsConfig
  training: TrainingSecretsConfig
  /** This fork: persisted choice of Colab / Local runtime mode. */
  runtime: RuntimeConfig
  proxy: ProxyConfig
}

/** Body of PUT /api/secrets: a nested partial dict; MASK ("***") means "keep unchanged". */
export type SecretsPatch = Partial<{
  [K in keyof Secrets]: Partial<Secrets[K]>
}>

// ---- models management (PP7) ---------------------------------------------

export interface ModelFileStatus {
  exists: boolean
  size: number
  mtime: number
}

/** Official variant of a family's base model (multi-model P4-5 unified shape; anima has no purpose/repo split). */
export interface FamilyMainVariantInfo extends ModelFileStatus {
  variant: string
  is_latest: boolean
  target_path: string
  /** 'preset' = a downloadable official variant; 'custom' = a local checkpoint candidate. */
  kind?: 'preset' | 'custom'
  is_current?: boolean
  /** Variant-level repo (krea2: separate HF repos for Raw/Turbo); anima uses the section repo. */
  repo?: string
  /** Purpose declaration (krea2: raw=training / turbo=inference). */
  purpose?: 'training' | 'inference'
  size_estimate?: number
}

/** A user-registered local custom base model (.safetensors already present in the PathPicker's disk picker). */
export interface CustomModelInfo extends ModelFileStatus {
  /** Registered absolute path (also the value written to selected_anima when chosen). */
  path: string
  /** Filename, for list display. */
  name: string
}

/** Unified shape of a family main-model catalog section (anima_main / krea2_main share this shape, P4-5). */
export interface FamilyMainCatalog {
  id: string
  name: string
  description: string
  repo: string
  variants: FamilyMainVariantInfo[]
  /** List of locally registered custom base models. */
  custom: CustomModelInfo[]
  /** Currently selected base model: a variant key or a custom path. */
  selected: string
  latest: string
  /** License display (krea2 community license; anima has none). */
  license?: string
  license_url?: string
}

export interface AnimaVaeCatalog extends ModelFileStatus {
  id: 'anima_vae'
  name: string
  description: string
  repo: string
  target_path: string
}

export interface ModelDirCatalog {
  id: 'qwen3' | 't5_tokenizer' | 'krea2_text_encoder' | 'krea2_text_encoder_fp8'
  name: string
  description: string
  repo: string
  target_dir: string
  /** krea2_text_encoder only: the selected TE ('bf16' | 'fp8' | local directory absolute path). */
  selected?: string
  files: Array<{ name: string; exists: boolean; size: number; mtime: number }>
}

export interface WD14VariantInfo {
  model_id: string
  is_current: boolean
  target_path: string
  exists: boolean
  size: number
  files: Array<{ name: string; exists: boolean; size: number; mtime: number }>
}

export interface WD14Catalog {
  id: 'wd14'
  name: string
  description: string
  repo: string
  current_model_id: string
  variants: WD14VariantInfo[]
}

export interface CLTaggerVariantInfo {
  label: string
  model_id: string
  model_path: string
  tag_mapping_path: string
  description?: string
  is_current: boolean
  target_path?: string
  version_dir?: string
  exists: boolean
  size: number
  files: Array<{ name: string; exists: boolean; size: number; mtime: number }>
}

export interface CLTaggerCatalog {
  id: 'cltagger'
  name: string
  description: string
  repo: string
  target_dir: string
  current_model_path: string
  current_tag_mapping_path: string
  variants: CLTaggerVariantInfo[]
}

export interface EvalVariantInfo {
  kind: 'clip' | 'dino'
  model_id: string
  target_path: string
  exists: boolean
  size: number
  /** Estimated size before download (bytes); 0 for an unknown model_id. */
  size_estimate: number
}

export interface EvalMetricsCatalog {
  id: 'eval_metrics'
  name: string
  description: string
  variants: EvalVariantInfo[]
}

export interface ModelDownloadStatus {
  key: string
  status: 'pending' | 'running' | 'done' | 'failed'
  started_at: number
  finished_at: number | null
  message: string
  log_tail: string[]
}

/** A unified model-source candidate row (catalog.model_sources[domain], capability bits assembled
 *  by the backend). docs/design/model-source-unification.md sec.6. */
export interface ModelSourceRow {
  kind: 'preset' | 'download' | 'local' | 'scanned'
  /** The user candidate's raw storage record (identity key for DELETE); null for preset / scanned rows. */
  candidate: ModelSourceCandidate | null
  /** The value written into that domain's selected-value field (repo id / absolute path / filename). */
  value: string
  label: string
  /** Row subtitle (upscaler description / repo source for custom candidates, etc.). */
  description: string
  /** model_id for POST /api/models/download; null for local candidates (not downloadable). */
  download_id: string | null
  /** variant param passed to the download trigger (default = value; for main-model/upscaler candidates, the in-repo file path). */
  download_variant: string | null
  /** status key in catalog.downloads; null for local candidates. */
  status_key: string | null
  exists: boolean
  size: number
  files?: Array<{ name: string; exists: boolean; size: number; mtime: number }> | null
  size_estimate: number
  is_current: boolean
  /** Built-in presets can't be removed (protects the defaults). */
  removable: boolean
  /** Local candidates never delete the on-disk file from the UI. */
  deletable: boolean
  extra: Record<string, string>
}

/** Candidate description for POST/DELETE /api/model-sources/{domain}. */
export interface ModelSourceCandidate {
  kind: 'download' | 'local'
  repo?: string
  filename?: string
  path?: string
  extra?: Record<string, string>
}

export interface UpscalerVariant {
  label: string
  filename: string
  kind: 'preset' | 'custom'
  hf_repo: string | null
  ms_repo: string | null
  size_mb: number | null
  description: string
  target_path: string
  is_current: boolean
  exists: boolean
  size: number
  mtime: number
  /** @deprecated kept for old builds, new code uses hf_repo/ms_repo */
  repo?: string
}
export interface UpscalersCatalog {
  id: 'upscalers'
  name: string
  description: string
  default: string
  /** Currently selected upscaler (from secrets.models.selected_upscaler, falls back to default) */
  current: string
  target_dir: string
  variants: UpscalerVariant[]
}

export interface FamilySwitchChange {
  field: string
  from: unknown
  to: unknown
}

export interface FamilySwitchResponse {
  config: ConfigData
  changes: FamilySwitchChange[]
}

export interface ModelsCatalog {
  models_root: string
  anima_main: FamilyMainCatalog
  anima_vae: AnimaVaeCatalog
  qwen3: ModelDirCatalog
  t5_tokenizer: ModelDirCatalog
  krea2_main: FamilyMainCatalog
  krea2_text_encoder: ModelDirCatalog
  krea2_text_encoder_fp8: ModelDirCatalog
  wd14: WD14Catalog
  cltagger: CLTaggerCatalog
  eval_metrics?: EvalMetricsCatalog
  /** Eval metric registry (Settings checkbox list). */
  eval_metric_catalog?: EvalMetricCatalogItem[]
  upscalers?: UpscalersCatalog
  /** Unified source candidate rows (consumed by the generic candidate card; key = domain: wd14 / eval_clip / ...). */
  model_sources?: Record<string, ModelSourceRow[]>
  /** Download source options per type: current = currently selected, available = selectable sources (length 1 = fixed single source). */
  download_source_options: Record<string, { current: string; available: string[] }>
  downloads: Record<string, ModelDownloadStatus>
}

// ---- projects / versions (PP1) -------------------------------------------

// ADR-0007 PR-5: the old ProjectStage / VersionStage are gone (their DB columns also dropped by v9's destructive migration).
// Replaced by VersionStatus + VersionPhase.

/** ADR-0007 sec.11.3-B new model: version runtime status state machine (5 enum values). */
export type VersionStatus =
  | 'preparing'
  | 'training'
  | 'completed'
  | 'failed'
  | 'canceled'

/** ADR-0007 sec.11.3-B new model: version preparation cursor (only meaningful when status=preparing).
 *  In PHASE_ORDER order: curating -> preprocessing -> editing ->
 *  regularizing -> ready (the auto-tagging step has been removed). */
export type VersionPhase =
  | 'curating'
  | 'preprocessing'
  | 'editing'
  | 'regularizing'
  | 'ready'

export const PHASE_ORDER: VersionPhase[] = [
  'curating', 'preprocessing', 'editing', 'regularizing', 'ready',
]

export const PHASE_SKIPPABLE: VersionPhase[] = ['preprocessing', 'regularizing']

/** ADR-0007 sec.11.5-A: advance / skip phase endpoint response. */
export interface PhaseAdvanceResult {
  advanced: boolean
  ok: boolean
  reason: string
  new_phase: VersionPhase | null
  version: Version | null
}

export interface VersionStats {
  train_image_count: number
  tagged_image_count: number
  train_folders: Array<{ name: string; image_count: number }>
  validation_image_count: number
  validation_tagged_count: number
  reg_image_count: number
  reg_meta_exists: boolean
  has_output: boolean
}

export interface Version {
  id: number
  project_id: number
  label: string
  config_name: string | null
  /** ADR-0007 sec.11.3-B: main runtime status state machine (5 enum values). */
  status: VersionStatus
  /** ADR-0007 sec.11.3-B: phase cursor, only meaningful when status=preparing. */
  phase: VersionPhase
  last_failure_reason: string | null
  created_at: number
  output_lora_path: string | null
  note: string | null
  /** Trigger word; written by Step 4 (Tagging), prepended to each caption while tagging; empty string = disabled. */
  trigger_word: string
  stats?: VersionStats
}

export interface ProjectSummary {
  id: number
  slug: string
  title: string
  active_version_id: number | null
  /** ADR-0007 sec.11.8-E: project card's top-right status badge / card shows the version name (enriched by the list endpoint). */
  active_version_label: string | null
  active_version_status: VersionStatus | null
  /** v12: phase cursor while preparing (badge shows "Preparing - Tagging"); null when there's no active version. */
  active_version_phase: VersionPhase | null
  created_at: number
  updated_at: number
  /** v12: non-null = archived (soft-hidden). The list endpoint returns both archived/active; splitting happens on the frontend. */
  archived_at: number | null
  note: string | null
  download_image_count?: number
  preprocess_image_count?: number
}

export interface ProjectDetail extends ProjectSummary {
  versions: Version[]
  download_image_count: number
  preprocess_image_count: number
}

// ---- jobs (PP2) -----------------------------------------------------------

export type JobStatus = 'pending' | 'running' | 'done' | 'failed' | 'canceled'
export type JobKind =
  | 'download' | 'preprocess' | 'tag' | 'reg_build'
  | 'eval_samples' | 'eval_clip' | 'eval_dino' | 'eval_tag' | 'eval_ccip'
  | 'upload'

export interface Job {
  id: number
  project_id: number
  version_id: number | null
  kind: JobKind
  params: string
  params_decoded?: Record<string, unknown> | null
  status: JobStatus
  /** v16 -- enqueue time; NULL for old jobs (enqueue time wasn't recorded then, UI shows -). */
  created_at?: number | null
  started_at: number | null
  finished_at: number | null
  pid: number | null
  log_path: string | null
  error_msg: string | null
}

export interface DownloadFile {
  name: string
  size: number
  /** File mtime (unix seconds); absent from servers older than the 2026 redesign. */
  mtime?: number
  has_meta: boolean
}

export interface UploadResult {
  added: string[]
  skipped: { name: string; reason: string }[]
}

export interface DataExportItem {
  filename: string
  path: string
  size: number
  mtime: number
}

export interface BundleImportResult {
  project: ProjectDetail
  version: Version
  stats: {
    train_image_count: number
    train_tagged_count: number
    reg_image_count: number
    preset_count: number
  }
}

// ---- preprocess (ADR 0010 train scope) -----------------------------------

/** An item in the crop page's workspace (train scope, rel path form): name + pixel size + processed flag. */
export interface CropWorkspaceItem {
  name: string
  /** Original filename under download/ (origin); downstream restoration uses this name. */
  source: string
  w: number
  h: number
  mtime: number
  size: number
  processed: boolean
  /** mtime of the training mask sidecar; null when there's no mask. Doubles as the badge condition + cache-buster. */
  mask_mtime: number | null
}

/** Inpaint save result: output is always .png, name changes if the source wasn't png (X.jpg -> X.png). */
export interface InpaintSaveResult {
  name: string
  origin: string
  mtime: number
  size: number
  w: number
  h: number
}

/** An item on the overview page's "Removed" tab: an entry marked by dedup review. The physical image still lives at download/{source}. */
export interface DuplicateRemovedItem {
  /** The manifest entry's key (usually == source). Passed under this name for restore. */
  name: string
  /** Original filename under download/ (origin). Thumbnails are fetched by source + bucket=download. */
  source: string
  /** Pixel size -- null when the origin file doesn't exist. */
  w: number | null
  h: number | null
  mtime: number
  size: number
}

// ---- ADR 0010 train-scope types -----------------------------------------

/** ADR 0010 train scope: lists all images under versions/{label}/train/ plus manifest metadata.
 *  Replaces the old `{processed, pending}` dual-list concept -- under the new model, train/ IS the
 *  "training set grid", and state is inferred implicitly from field differences (see ADR 0010
 *  sec.Manifest schema v2 + backend `_is_processed`: extension change / `_cN` suffix / train size != download size). */
export interface TrainImage {
  /** POSIX rel path "{N_label}/{image}" (e.g. "1_data/X.png"). */
  name: string
  mtime: number
  size: number
  /** Read from the image header via PIL; null if corrupt / physically missing. */
  w: number | null
  h: number | null
  /** Original filename under download/ (no sub-folder structure); restore lookup uses this. */
  origin: string | null
  /** @deprecated compatibility field; the backend keeps both fields equal. */
  source: string | null
  /** download/{origin} is physically missing (restore lands it as no_origin). */
  orphan: boolean
  /** Manual dedup review mark. The UI distinguishes "included in training" vs "skipped by review". */
  duplicate_removed: boolean
  /** ADR 0010 state inference (backend `_is_processed`): a train file that was upscaled / cropped /
   *  transcoded -> true; an as-is copy made during curate -> false. The UI uses this to draw the
   *  "processed" badge. */
  processed: boolean
  /** Old-schema passthrough fields (always null for new entries; frontend tolerates this). */
  model: string | null
  scale: number | null
  action: string | null
  target_area: number | null
  src_size: [number, number] | null
  dst_size: [number, number] | null
  elapsed_seconds: number | null
}

/** ADR 0010 sec.Restore semantics: restore returns three groups: succeeded / no manifest entry /
 *  missing from download. `no_origin` feeds the UI's three options [drag in a replacement / keep /
 *  remove]. */
export interface TrainRestoreResult {
  restored: string[]
  missing: string[]
  no_origin: string[]
}

// ---- curation (PP3) -------------------------------------------------------

/**
 * An item in the Curation list: filename + on-disk mtime (unix seconds).
 * mtime supports sorting "by download time"; the backend makes no sort guarantee (other than a
 * stable name-lexicographic order), sorting is decided by the frontend per user preference.
 */
export interface CurationItem {
  name: string
  mtime: number
  /** ADR 0010 fixup (2026-06-04): train-side items carry the original download filename (looked up
   *  via the train manifest entry.origin; old projects with no manifest -> falls back to name
   *  itself). Curation's right-side thumb uses the `download` bucket + this origin, showing
   *  **the pre-processing look** -- avoiding multi-crop fan-out / dedup / upscale byte changes
   *  making the curation page thumbnail "shift position". View processed results in Preprocess
   *  Overview instead. Missing/meaningless for left-side items (download candidates). */
  origin?: string
}

export interface CurationView {
  left: CurationItem[] // download - train - validation
  right: Record<string, CurationItem[]> // folder -> items
  download_total: number
  train_total: number
  folders: string[]
}

/** A single image in the held-out validation set: a flat list (no folder concept), but carries a
 *  physical `folder` for thumbnail addressing (needed by the version thumb's validation bucket) and precise deletion. */
export interface ValidationItem {
  name: string
  mtime: number
  folder: string
}

export interface CurationValidationView {
  left: CurationItem[] // download - train - validation (shares the candidate pool with the training set)
  right: ValidationItem[] // full flat validation list
  download_total: number
  val_total: number
}

export interface CopyResult {
  copied: string[]
  skipped: string[]
  missing: string[]
}

/** Dedup scan request body. The algorithm internally has a batch of threshold/perf parameters, but
 *  they've all been baked into backend constants; the UI only exposes these two:
 *   - match_scope: whether to check only full-image duplicates, or also scene-variant diffs/crops (crop detection only turns on with 'both')
 *   - sensitivity: looseness/strictness of the variant/crop verdict (drives the backend's variant_score + crop_score) */
export interface DuplicateScanOptions {
  match_scope: 'strict' | 'both'
  sensitivity: 'loose' | 'standard' | 'strict'
}

export interface DuplicateMetrics {
  score: number
  match_type: 'keep' | 'strict-duplicate' | 'same-scene-variant' | 'linked-indirectly' | string
  structure_diff: number
  phash_diff: number
  soft_phash_diff: number
  dhash_diff: number
  ahash_diff: number
  edge_diff: number
  color_diff: number
  tile_median: number
  tile_mean: number
  tile_close_ratio: number
  gray_diff: number
  gray_close_ratio: number
  aspect_delta: number
  note: string
}

export interface DuplicateItem {
  name: string
  keep: boolean
  width: number
  height: number
  filesize_kb: number
  metrics: DuplicateMetrics | null
}

export interface DuplicateGroup {
  group_id: number
  keep: string
  items: DuplicateItem[]
  best: DuplicateMetrics | null
}

export interface DuplicateScanResult {
  target: 'preprocess' | 'download'
  match_scope: DuplicateScanOptions['match_scope']
  total_images: number
  readable_images: number
  group_count: number
  candidate_count: number
  crop_relation_count: number
  elapsed_seconds: number
  stats: {
    total_pairs: number
    aspect_skipped_pairs: number
    prefiltered_pairs: number
    compared_pairs: number
  }
  groups: DuplicateGroup[]
}

export interface DuplicateApplyResult {
  removed: string[]
  missing: string[]
  skipped: string[]
}

// ---- captions (PP4) -------------------------------------------------------

export type TaggerName = 'wd14' | 'cltagger' | 'joycaption' | 'llm'

export interface TaggerStatus {
  name: TaggerName
  ok: boolean
  msg: string
  requires_service: boolean
}

export interface CaptionPreview {
  name: string
  folder: string
  tag_count: number
  tags_preview: string[]
  has_caption: boolean
}

/** Caption list item returned when full=1; includes full tags + format. */
export interface CaptionEntry extends CaptionPreview {
  tags: string[]
  format: 'txt' | 'json' | 'none'
}

export interface CommitItem {
  folder: string
  name: string
  tags: string[]
}

export interface CommitResult {
  snapshot: CaptionSnapshot
  written: number
  skipped: string[]
}

export interface CaptionFull {
  name: string
  tags: string[]
  format: 'txt' | 'json' | 'none'
}

export type BatchScope =
  | { kind: 'all' }
  | { kind: 'folder'; name: string }
  | { kind: 'files'; items: Array<{ folder: string; name: string }> }

export interface BatchOpRequest {
  op: 'add' | 'remove' | 'replace' | 'dedupe' | 'stats'
  scope: BatchScope
  tags?: string[]
  old?: string
  new?: string
  position?: 'front' | 'back'
  top?: number
}

export interface BatchOpResult {
  op: string
  affected?: number
  items?: Array<[string, number]>
}

export interface CaptionSnapshot {
  id: string
  created_at: number
  size: number
  file_count: number
}

// PP5 ----------------------------------------------------------------

export interface RegMeta {
  generated_at: number
  based_on_version: string
  api_source: string
  target_count: number
  actual_count: number
  source_tags: string[]
  excluded_tags: string[]
  blacklist_tags: string[]
  failed_tags: string[]
  train_tag_distribution: Record<string, number>
  auto_tagged: boolean
  /** A3 -- name of the tagger that actually ran auto_tag ("wd14" / "cltagger" / ...);
   * null = didn't run / old meta lacks this field. auto_tagged=true with this field null is
   * treated as old-version data (unknown tagger). */
  auto_tag_kind?: string | null
  /** B1 (PR-2) -- the build_mode at the time this reg set was generated; old meta lacking this
   * field -> backend defaults to 'mirror'. The frontend's mode-switch interception checks this
   * first; only falls back to inferring from reg.files path prefixes. */
  build_mode?: string
  incremental_runs: number
  // PP5.5 -- postprocess summary (postprocessed_at is null when it hasn't run or K couldn't be found)
  postprocessed_at: number | null
  postprocess_clusters: number | null
  postprocess_method: string | null
  postprocess_max_crop_ratio: number | null
  // "scrape" = pulled from booru, "ai_base" = generated from the base model prior; defaults to "scrape" when absent (old meta compat)
  generation_method?: 'scrape' | 'ai_base'
}

export interface RegStatus {
  exists: boolean
  meta: RegMeta | null
  image_count: number
  files: string[]
}

export interface RegTagCount {
  tag: string
  count: number
}

// PP6.2 -- Train config (private to the version, independent of the global preset pool)
export interface VersionConfigResponse {
  has_config: boolean
  config: ConfigData | null
  /** Project-specific fields force-overridden by the server (the frontend form should disable these) */
  project_specific_fields: string[]
  /** Project prefill values the backend will inject when forking a preset (project path + global
   * model path + reg detection). Used by the new-preset preview form to show "the value you'll get
   * after saving". Returned regardless of has_config -- creating a new preset can be clicked while
   * the version already has a config (overwriting the current preset), so this hint is independent
   * of has_config's state. */
  project_specific_defaults?: ConfigData
  dropped_fields?: string[]
  defaulted_fields?: string[]
}

/** Training set ARB bucket distribution (computed by the backend's real BucketManager). count = effective sample count (including repeat x fan-out). */
/** Pre-run estimate: does it fit, and how long will it take.
 *
 *  Both halves are independently optional. `speed` is null until this project
 *  has produced at least one measured run; `memory` is null when no verdict can
 *  be reached (no CUDA, unreadable checkpoint, family without block swap). The
 *  UI renders nothing for a null half rather than inventing a number. */
export interface TrainEstimate {
  speed: { it_per_s: number; task_id: number; task_name: string } | null
  memory: {
    ok: boolean
    vram_need_bytes: number | null
    ram_need_bytes: number | null
    free_vram_bytes: number | null
    avail_ram_bytes: number | null
    recommended_blocks_to_swap: number | null
    blocks_to_swap: number
    total_blocks: number
  } | null
}

export interface BucketDistribution {
  resolutions: number[]
  aspect_ratio_limit: number
  /** Authoritative effective sample count (train + reg), counted the way the
   *  trainer counts: images without a caption are excluded. The frontend used
   *  to scan folders itself and included uncaptioned files, so the pre-run step
   *  count came out higher than the number the trainer then printed. */
  effective_samples?: number
  train_samples?: number
  reg_samples?: number
  groups: Array<{
    reso: number
    buckets: Array<{ w: number; h: number; count: number }>
  }>
  /** NaViT packing estimate (only present when config.navit_packing is set). packs_per_epoch =
   *  numerator of optimizer steps/epoch (simulated by the backend's real NavitPackBatchSampler,
   *  exact for epoch-0). sizes is non-empty only in native mode = native-size histogram (ARB
   *  buckets don't exist in this mode). */
  navit?: {
    packs_per_epoch: number
    samples: number
    avg_images_per_pack: number
    token_min: number
    token_max: number
    token_budget: number
    strategy: string
    native: boolean
    downscaled: number
    sizes: Array<{ w: number; h: number; count: number }>
  } | null
}

export interface RegBuildRequest {
  excluded_tags?: string[]
  auto_tag?: boolean
  /** A3 -- tagger used for auto-tag. The current UI only exposes wd14 / cltagger;
   * backend 422 validation is likewise restricted to these two. */
  auto_tag_kind?: 'wd14' | 'cltagger'
  api_source?: 'gelbooru' | 'danbooru'
  /** Default true (incremental) -- user's decision: avoid wiping out yesterday's hard-won pulled
   * images when starting a new generation. false = full: the worker clears reg/ (including
   * .deleted_ids.json) up front. */
  incremental?: boolean
  /** A4 v2 -- after building, the worker automatically runs dedup + an incremental top-up loop when
   * short, up to 3 rounds, before resolution clustering. Default true. */
  auto_dedup?: boolean
  /** B1 (PR-2) -- build mode:
   * - mirror: mirrors the train subfolders (5_concept/, 1_general/ ...), target_count is ignored
   * - flat: all images go into a single 1_data/ bucket, target_count determines the total image count (null = train's total count)
   * Default flat; switching requires the reg set to already be empty (frontend intercepts this). */
  build_mode?: 'mirror' | 'flat'
  /** B1 (PR-2) -- target image count in flat mode; null = use train's total image count. */
  target_count?: number | null
  // PP5.5 advanced
  skip_similar?: boolean
  aspect_ratio_filter_enabled?: boolean
  min_aspect_ratio?: number
  max_aspect_ratio?: number
  postprocess_method?: 'smart' | 'stretch' | 'crop'
  postprocess_max_crop_ratio?: number
}

/** Attention backend, one of three -- replaces the original xformers/flash_attn dual bool. */
/** secrets.generate.attention_backend: 'auto' = use whatever's installed (default);
 *  an explicit value (flash_attn/xformers/none) forces it. GenerateRequest also accepts this type
 *  as a per-request override (the frontend no longer sends it; the server auto-reads from secrets + resolves auto). */
export type AttentionBackend = 'auto' | 'none' | 'xformers' | 'flash_attn'

/** PR-9 -- prior generation (base model generates a reg set in reverse, no LoRA). */
export interface RegAiRequest {
  excluded_tags?: string[]
  /** Base model temporarily selected for this prior generation (an official variant key or a local
   *  custom path); omitted -> the server uses Settings' selected_anima. */
  base_model?: string
  negative_prompt?: string
  width?: number
  height?: number
  steps?: number
  cfg_scale?: number
  sampler_name?: string
  scheduler?: string
  seed?: number
  incremental?: boolean
  repeat?: number
  mixed_precision?: string
}

/** PR-9 -- test generation (a standalone tool page, multi-LoRA + multi-prompt). */
export interface LoraEntry {
  path: string
  scale: number
  /** Project / version binding from the picker; absent for external files */
  project_id?: number | null
  version_id?: number | null
  /** Placeholder state only: when resolving fails on history backfill, keeps the original basename
   *  (e.g. "my-lora.safetensors"), letting SidebarLoras render a warning placeholder card
   *  prompting the user to reselect. Ignored once `path` is non-empty; entries with path='' on
   *  submit get skipped by `.filter(l => l.path.trim())`, not sent to the daemon. */
  name?: string | null
}

/** XY matrix: loop the whole grid within a single task, the frontend lays it out as a (yi, xi) grid.
 *  When xy_matrix is set, the backend forces prompts to a single entry + count=1 (to avoid a
 *  combinatorial explosion).
 *  v1 doesn't support a lora_path axis (missing an unhook interface, left for v2). */
export type XYAxisType =
  | 'lora_scale'
  | 'steps'
  | 'cfg_scale'
  | 'lora_ckpt'  // different step/epoch checkpoints of the same LoRA (for finding the overfit inflection point)

export interface XYAxisSpec {
  axis: XYAxisType
  /** Value type derived from axis: steps -> int; lora_scale/cfg_scale -> number; lora_ckpt -> string (path) */
  values: Array<number | string>
  /** Required when axis=lora_scale / lora_ckpt -- which lora_configs entry it's bound to */
  lora_index?: number | null
}

export interface XYMatrixSpec {
  x: XYAxisSpec
  y?: XYAxisSpec | null
}

export interface GenerateRequest {
  prompts: string[]
  /** Model family the base model belongs to (multi-model P4-4); omitted = anima. */
  model_family?: 'anima' | 'krea2'
  /** Base model temporarily selected for this generation (an official variant key or a local
   *  custom path); omitted -> the server uses that family's selected from Settings. */
  base_model?: string
  /** Text encoder variant for this generation (applies to krea2): omitted = follow the download
   *  center's selected TE (selected_te); an explicit bf16/fp8 temporarily overrides it (symmetric with base_model). */
  text_encoder?: 'bf16' | 'fp8'
  negative_prompt?: string
  width?: number
  height?: number
  steps?: number
  cfg_scale?: number
  sampler_name?: string
  scheduler?: string
  count?: number
  seed?: number
  lora_configs?: LoraEntry[]
  mixed_precision?: string
  attention_backend?: AttentionBackend
  /** When set, prompts is limited to a single entry + count=1 (schema validation) */
  xy_matrix?: XYMatrixSpec | null
  /** A GenerateParamsSnapshot dict built by the frontend; the server doesn't interpret its
   *  structure, passes it through to the daemon -> stuffed into the encrypted cache payload header
   *  on image_done. Returned by /api/generate/cache/index for backfilling as CacheEntry.params. */
  params_snapshot?: Record<string, unknown> | null
}

/** GET /api/generate/cache/index -- index of the current session's encrypted on-disk cache.
 *  The server-side SessionCache aggregates it by task_id; the frontend converts it to CacheEntry. */
export interface CacheGenerateHistoryEntry {
  /** "cache:<task_id>" */
  id: string
  taskId: number
  mode: 'single' | 'xy'
  /** Unix timestamp ms */
  createdAt: number
  /** All filenames from this task (sorted by filename for XY) */
  filenames: string[]
  /** GenerateParamsSnapshot dict */
  params: Record<string, unknown>
  /** Present only for mode=xy; lists each image's xy position, used by PreviewXYGrid to rebuild the grid */
  samples?: Array<{
    filename: string
    xy: { xi: number; yi: number; xv: string | number; yv: string | number | null }
  }>
}

/** An on-disk test-image history entry (GET /api/generate/disk-history).
 *  params is a GenerateParamsSnapshot (the frontend interprets it via paramsSnapshot.ts's types);
 *  typed as unknown here so api/client.ts doesn't depend on the pages layer's types. */
export interface DiskGenerateHistoryEntry {
  /** Stable ID: "disk:<date>:<mode>:image_<N>"; the frontend dedupes by this */
  id: string
  /** YYYY-MM-DD */
  date: string
  mode: 'single' | 'xy'
  filename: string
  /** Server-side absolute path, used to dedupe against the IDB entry.diskPath */
  path: string
  /** /api/generate/disk-image/<date>/<mode>/<filename> */
  url: string
  /** Unix timestamp (written by the sidecar, or falls back to the file's mtime) */
  created_at: number
  schema_version: number
  /** The params object inside the sidecar (the frontend interprets it as GenerateParamsSnapshot) */
  params: Record<string, unknown>
}

/** A training_state_step*.pt found under a version's output/ (used for resuming training). */
export interface StateCkpt {
  /** global_step count */
  step: number
  /** Display text: "step 2476" */
  label: string
  /** Absolute path */
  path: string
  /** File mtime timestamp */
  mtime: number
}

/** Project-level ckpt list grouped by version (used by the resume_state / resume_lora picker). */
export interface VersionCkptGroup<T> {
  version_id: number
  /** Version label, e.g. "baseline" / "high-lr" */
  label: string
  items: T[]
}

/** A LoRA checkpoint file found under a version's output/ (GET .../lora_ckpts). */
export interface LoraCkpt {
  /** 'final' / 'step' / 'epoch' / 'other' */
  kind: 'final' | 'step' | 'epoch' | 'other'
  /** step / epoch count; 0 for final / other */
  value: number
  /** Display text: 'final' / 'step 2476' / 'epoch 5' / filename */
  label: string
  /** Absolute path */
  path: string
  /** File mtime timestamp */
  mtime: number
}


// -- Checkpoint soup (merge several adapters) --------------------------------

/** A file in the soup directory (an uploaded ingredient, or a merged result). */
export interface SoupFile {
  name: string
  /** Absolute path -- both merge / generate reference it by path */
  path: string
  size: number
  mtime: number
}

/** An adapter's "fingerprint": used to judge whether it can be averaged before merging. */
export interface SoupSourceInfo {
  name: string
  path: string
  size: number
  tensor_count: number
  dtypes: string[]
  algo: string | null
  rank: number | null
  alpha: number | null
  factor: unknown
  family: string | null
  module: string | null
}

/** POST /api/soup/inspect -- when ok=false, errors explains why it can't be merged. */
export interface SoupCompatibility {
  ok: boolean
  errors: string[]
  warnings: string[]
  items: SoupSourceInfo[]
}

export interface SoupMergeResult extends SoupFile {
  warnings: string[]
  sources: { name: string; weight: number; effective: number }[]
}


// -- Trigger word detection --------------------------------------------------
export interface TriggerCandidate {
  word: string
  count: number
  /** share of captions containing it (0–1) */
  coverage: number
  /** share of captions where it is the first chunk (0–1) */
  first: number
}
export interface TriggerDetectResult {
  total: number
  suggested: string | null
  candidates: TriggerCandidate[]
  current: string
}

// -- Remote access (phone access over a quick tunnel) ------------------------

/** GET /api/tunnel -- `url` already carries ?k=<key>, requests without the key get a 401. */
export interface TunnelState {
  running: boolean
  /** Full shareable link (includes the access key); null when not enabled */
  url: string | null
  port: number | null
  started_at: number | null
  error: string | null
  binary: string
  installed: boolean
  /** Whether the current platform has an official precompiled binary (if not -> manual install only) */
  can_install: boolean
  log: string[]
  /** provider the running tunnel was started with */
  running_provider?: TunnelProvider | null
  /** configured provider — what "start" will use */
  provider: TunnelProvider
  /** open the link as soon as the studio starts */
  autostart: boolean
  ngrok_domain: string
  has_ngrok_token: boolean
  /** the configured provider gives the same address every time */
  permanent: boolean
  providers: {
    cloudflare: { installed: boolean; can_install: boolean }
    tailscale: { installed: boolean; can_install: boolean; download_url: string }
    ngrok: { installed: boolean; can_install: boolean; token_url: string; domains_url: string }
  }
}

export type TunnelProvider = 'cloudflare' | 'tailscale' | 'ngrok'

/** Phase 2 commit 14 -- TAEFlux model status (GET /api/generate/taeflux/status). */
export interface TaeFluxStatus {
  available: boolean
  dir: string
  files: string[]
}

/** Phase 2 -- current inference daemon status (GET /api/generate/daemon/status). */
export interface DaemonStatus {
  state: 'stopped' | 'starting' | 'idle' | 'busy' | 'unloading'
  model_loaded: boolean
  busy: boolean
  alive: boolean
}

/** xformers install status / install result (simplified version, compare with FlashAttnStatus). */
export interface XformersStatus {
  installed: boolean
  version: string | null
}

export interface XformersInstallResult {
  installed: boolean
  version: string | null
  stdout_tail: string
  restart_required: boolean
}

export type TaskStatus =
  'pending' | 'running' | 'done' | 'failed' | 'canceled' | 'paused' | 'scheduled'

/** Valid values of tasks.task_type. The R-3 ledger merge folded in nine kinds of data-job kind.
 *  Tiers: exclusive = train/reg_ai/generate/eval_samples; light = everything else; io = download. */
export type TaskType =
  | 'train' | 'reg_ai' | 'generate'
  | 'download' | 'preprocess' | 'tag' | 'reg_build'
  | 'eval_samples' | 'eval_clip' | 'eval_dino' | 'eval_tag' | 'eval_ccip'

/** R-5 tier view param: GPU view = exclusive, data view = data (light+io). */
export type QueueResourceClass = 'exclusive' | 'data'

/** Terminal task statuses -- the UI generally disables action buttons (cancel / pause etc.) on these.
 *  `paused` is **not** terminal -- it can be revived via resume. */
export const TERMINAL_TASK_STATUSES: ReadonlyArray<TaskStatus> = [
  'done', 'failed', 'canceled',
]

export interface Task {
  id: number
  name: string
  config_name: string
  /** 0.17 P-D -- backend-authoritative task type (added in the _v5 migration, values train/reg_ai/generate).
   *  Old rows get it via an automatic `NOT NULL DEFAULT 'train'` ALTER backfill; optional here only for
   *  compatibility with test mocks that omit the field -- it always has a value at runtime. */
  task_type?: TaskType
  status: TaskStatus
  priority: number
  created_at: number
  started_at: number | null
  finished_at: number | null
  pid: number | null
  exit_code: number | null
  output_dir: string | null
  error_msg: string | null
  /** Added in PP1; null on old tasks. */
  project_id?: number | null
  /** Added in PP1; null on old tasks. */
  version_id?: number | null
  /** PP6.3 -- per-version private config path (null on old tasks, falls back to _configs_dir). */
  config_path?: string | null
  /** PP6.1 -- per-task monitor state.json path. */
  monitor_state_path?: string | null
  /** ADR 0006 PR-2 -- paused task's .pt file path (pause_step_<N>.pt). */
  paused_state_path?: string | null
  /** ADR 0006 PR-2 -- paused task's config snapshot path (pause_step_<N>.config.json). */
  paused_config_path?: string | null
  /** ADR 0006 PR-2 -- global_step at pause time (shown by the UI as "paused at step N"). */
  paused_step?: number | null
  /** ADR 0006 PR-2 -- pause time (unix seconds). */
  paused_at?: number | null
  /** 0.17 P-B -- scheduled start time (unix seconds). Set when status='scheduled'; kept as a record
   *  after promotion to pending. Always null for non-scheduled tasks. */
  scheduled_at?: number | null
  /** R-2/_v17 -- kind-specific params JSON for data-job tasks; always null for train/reg_ai. */
  params?: string | null
  /** Decoded alongside the backend read path (same convention as the old jobs DAO). */
  params_decoded?: Record<string, unknown> | null
  /** ADR 0006 PR-4 -- is_pausable signal (SS8.1): tells the UI whether to show the pause
   *  button. Enriched by the server while the supervisor is alive; defaults to false when idle. */
  is_pausable?: boolean
  /** ADR 0006 Addendum 2 -- path of the most recent end-of-epoch auto backup .pt file
   *  (auto_epoch_state.pt, a single file that gets overwritten). Resume point for failed/canceled tasks. */
  last_state_path?: string | null
  /** ADR 0006 Addendum 2 -- config snapshot path that pairs with the auto backup. */
  last_config_path?: string | null
  /** ADR 0006 Addendum 2 -- backup point epoch (shown by the UI as "resume from epoch N"). */
  last_state_epoch?: number | null
  /** ADR 0006 Addendum 2 -- backup point global_step. */
  last_state_step?: number | null
  /** ADR 0006 Addendum 2 -- is_resumable signal: status in paused/failed/canceled
   *  and the resume-point file exists on disk. Tells the UI whether to show the "resume training" button. */
  is_resumable?: boolean
  /** _v20 -- user-written task note (written via right-click on the queue page / edited on the detail page). Empty = null. */
  note?: string | null
}

/** One row of GET /api/queue/{id}/samples -- a training sample image (found by scanning disk, includes finished tasks). */
export interface TaskSample {
  filename: string
  /** File mtime (unix seconds); the list is sorted ascending by this = the training timeline. */
  mtime: number
  size: number
  /** Parsed from the filename; only present for `epoch_N_*.png`. */
  epoch: number | null
  /** Parsed from the filename; only present for `step_N_*.png`. */
  step: number | null
}

/** 0.17 P-E -- paginated response for /api/queue?group=history. */
export interface QueueHistoryPage {
  items: Task[]
  total: number
  page: number
  page_size: number
}

/** ADR 0006 PR-2 -- response of GET /api/queue/hold. When `held=true` the UI shows a sticky
 *  banner at the top; `pending_waiting` is the current pending-queue length (for the banner text). */
export interface QueueHoldState {
  held: boolean
  pending_waiting: number
}

export interface LogResponse {
  task_id: number
  content: string
  size: number
}

/** /api/state — per-task monitor state written by the training process */
export interface MonitorState {
  task_id?: number
  project_id?: number
  project_slug?: string
  version_id?: number
  version_label?: string
  step?: number
  total_steps?: number
  epoch?: number
  total_epochs?: number
  speed?: number          // it/s
  start_time?: number     // unix seconds
  losses?: Array<{ step: number; loss: number }>
  lr_history?: Array<{ step: number; lr: number }>
  optimizer_metrics_history?: Array<{
    step: number
    lr?: number
    actual_lr?: number
    base_lr?: number
    effective_lr?: number
    d?: number
    d_min?: number
    d_max?: number
    actual_lr_min?: number
    actual_lr_max?: number
  }>
  samples?: Array<{
    path: string
    step?: number
    /** Carries cell metadata in XY mode (generate tasks only; empty for training tasks). */
    xy?: { xi: number; yi: number; xv: number | string; yv: number | string | null }
  }>
  config?: Record<string, string | number | boolean>
  vram_used_gb?: number
  vram_total_gb?: number
}

export interface TaskOutputFile {
  name: string
  path: string
  size: number
  mtime: number
  kind: 'lora' | 'training_state' | 'pause_state' | 'auto_epoch_state' | 'other'
  is_lora: boolean
}

export interface TaskOutputs {
  task_id: number
  output_dir: string | null
  exists: boolean
  /** True only for loopback requests; always false in the cloud. The frontend uses this to control the "open folder" button's visibility. */
  supports_open_folder: boolean
  files: TaskOutputFile[]
  /** "{slug}-{label}", used as the zip filename prefix for bundle downloads (matches train.zip's naming style).
   * Old tasks not bound to a project / version -> null, callers fall back to task_{id}. */
  archive_basename: string | null
}

export interface DatasetFolder {
  name: string
  label: string
  repeat: number
  image_count: number
  caption_types: { json: number; txt: number; none: number }
  samples: string[]
  path: string
}

export interface DatasetScan {
  root: string
  exists: boolean
  folders: DatasetFolder[]
  total_images?: number
  weighted_steps_per_epoch?: number
}

export interface ImportResult {
  imported_count: number
  task_ids: number[]
  renamed: Record<string, string>
}

/**
 * API error: besides `message` (a string suitable for a toast directly), also keeps `status` and
 * `detail` (when the FastAPI side does `raise HTTPException(status, detail=dict(...))`,
 * detail is a structured object; callers can branch on `e.detail.error`).
 *
 * Uses Error rather than a custom class because many existing callsites do the generic
 * `catch (e) { toast(String(e)) }`; keeping `Error.prototype.toString()` behavior doesn't break them.
 * New callsites that need structured handling cast explicitly: `(e as ApiError).detail`.
 *
 * ADR-0009 PR-3 C3: added the `traceId` field -- the backend dual-write envelope's
 * `body.error.trace_id` or the X-Trace-Id response header. Shown as a "trace ab12cd34"
 * suffix in the toast so the user can screenshot it for the dev; also attached when ErrorBoundary
 * reports, tying it to the frontend's last failure before the crash.
 */
export type ApiError = Error & {
  status?: number
  /** ADR-0009 Phase 2: the backend's body.error.code (a semantic error code); the frontend looks up errors.* i18n by it. */
  code?: string
  detail?: unknown
  traceId?: string
}

/**
 * ADR-0009 Phase 2 unified error parsing: shared by every fetch / XHR failure path so toast copy
 * stays consistent and localizable.
 *
 * Prefers `body.error`: looks up `errors.<code>` i18n via `error.code` (interpolating `error.details`,
 * falling back to the English `error.message` when the key is missing). `body.detail` is a fallback --
 * structured detail (e.g. a 409 conflict's config/suggested_name) still gets attached to `err.detail`
 * for callers; only used for copy directly when there's no error envelope (RequestValidationError 422 list / very old paths).
 */
export function makeApiError(
  status: number,
  statusText: string,
  body: unknown,
  headerTraceId?: string | null,
): ApiError {
  let message = `${status} ${statusText}`
  let code: string | undefined
  let detail: unknown = null
  let traceId: string | undefined
  const b = body as {
    detail?: unknown
    error?: { code?: unknown; message?: unknown; trace_id?: unknown; details?: unknown }
  } | null | undefined
  const err = b?.error
  if (err && typeof err === 'object') {
    code = typeof err.code === 'string' ? err.code : undefined
    const enMsg =
      typeof err.message === 'string' && err.message ? err.message : message
    const params =
      err.details && typeof err.details === 'object'
        ? (err.details as Record<string, unknown>)
        : {}
    message = code ? i18n.t(`errors.${code}`, { ...params, defaultValue: enMsg }) : enMsg
    if (typeof err.trace_id === 'string') traceId = err.trace_id
     // Structured data now lives on error.details (e.g. a 409 conflict's config/suggested_name,
     // running_tasks list); callers read it via err.detail.
    if (err.details && typeof err.details === 'object') detail = err.details
  }
  if (b && b.detail !== undefined) {
    if (!err) {
      if (typeof b.detail === 'string') {
        message = b.detail
      } else if (b.detail && typeof b.detail === 'object') {
        detail = b.detail
        const dm = (b.detail as { message?: unknown }).message
        if (typeof dm === 'string') message = dm
      }
    } else if (detail === null && b.detail && typeof b.detail === 'object') {
      detail = b.detail
    }
  }
  if (!traceId && headerTraceId) traceId = headerTraceId
  if (traceId) setLastApiTraceId(traceId)
  const e = new Error(message) as ApiError
  e.status = status
  e.code = code
  e.detail = detail
  e.traceId = traceId
  return e
}

/**
 * ADR-0009 PR-3 C3: format the last 8 chars of ApiError.traceId into the toast suffix.
 *
 * When a user reports an issue they screenshot the toast; the dev takes these 8 chars and runs
 * `jq 'select(.trace_id | endswith("..."))' studio.log` to reconstruct the full chain.
 *
 * Call pattern (opt-in by callsites, not enforced -- doesn't break existing toast(e.message,'error') calls):
 *     toast(`${e.message}${formatErrorTraceSuffix(e)}`, 'error')
 */
export function formatErrorTraceSuffix(err: unknown): string {
  const traceId = (err as ApiError | undefined)?.traceId
  if (!traceId) return ''
  return `  ·  trace ${traceId.slice(-8)}`
}

async function req<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const resp = await fetch(path, {
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
    },
    ...init,
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => null)
    throw makeApiError(resp.status, resp.statusText, body, resp.headers.get('X-Trace-Id'))
  }
  if (resp.status === 204) return undefined as T
  return (await resp.json()) as T
}

/** Upload progress event -- mirrors XMLHttpRequestEventTarget#progress fields one to one. */
export interface UploadProgressEvent {
  loaded: number
  total: number
  /** False when total === 0 (server didn't return Content-Length or used chunked encoding); ETA can't be computed. */
  lengthComputable: boolean
}

/**
 * XHR-based multipart upload; fetch() has no request-body progress event, so
 * upload progress must go through XHR. Error format matches `req` (ApiError + parsed detail).
 */
async function xhrUpload<T>(
  url: string,
  body: FormData,
  onProgress?: (e: UploadProgressEvent) => void,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', url, true)
    xhr.responseType = 'text'
    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        onProgress({
          loaded: e.loaded,
          total: e.total,
          lengthComputable: e.lengthComputable,
        })
      }
    }
    xhr.onload = () => {
      const text = xhr.responseText
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(text ? (JSON.parse(text) as T) : (undefined as T))
        } catch {
          reject(new Error('invalid JSON response'))
        }
        return
      }
      let parsed: unknown = null
      try {
        parsed = JSON.parse(text)
      } catch {
        /* body isn't JSON: makeApiError falls back to statusText */
      }
      reject(
        makeApiError(xhr.status, xhr.statusText, parsed, xhr.getResponseHeader('X-Trace-Id')),
      )
    }
    xhr.onerror = () => reject(new Error('network error'))
    xhr.send(body)
  })
}

/** studio_data storage location: current/default + full scan (used by the migration confirmation modal). */
export interface StudioDataScanEntry {
  name: string
  is_dir: boolean
  files: number
  bytes: number
}

export interface StudioDataInfo {
  current: string
  default: string
  is_custom: boolean
  /** Null when the request has scan=false (Settings page shows the path only, skipping the disk scan) */
  scan: {
    total_files: number
    total_bytes: number
    entries: StudioDataScanEntry[]
  } | null
}

/** Migration status snapshot (used when reopening the modal / as a fallback for missed SSE events; live progress
 *  goes through the `studio_data_migrate_progress` / `_done` events). */
export interface StudioDataMigrateStatus {
  state: 'idle' | 'running' | 'done' | 'error'
  target: string
  total_files: number
  total_bytes: number
  done_files: number
  done_bytes: number
  current_file: string
  error: string
}

/** Models-root storage location: same shape as studio_data (reused by the migration confirmation modal). */
export interface ModelsRootInfo {
  current: string
  default: string
  is_custom: boolean
  /** Null when the request has scan=false (Settings page shows the path only, skipping the disk scan) */
  scan: {
    total_files: number
    total_bytes: number
    entries: StudioDataScanEntry[]
  } | null
}

/** Models-root migration status snapshot (live progress goes through SSE `models_root_migrate_progress` / `_done`). */
export interface ModelsRootMigrateStatus {
  state: 'idle' | 'running' | 'done' | 'error'
  target: string
  total_files: number
  total_bytes: number
  done_files: number
  done_bytes: number
  current_file: string
  error: string
}

export const api = {
  health: () => req<HealthResponse>('/api/health'),
  systemStats: () => req<SystemStats>('/api/system/stats'),
  state: () => req<Record<string, unknown>>('/api/state'),

  schema: () => req<SchemaResponse>('/api/schema'),

  // Presets (PP0+) -----------------------------------------------------
  listPresets: () =>
    req<{ items: PresetSummary[] }>('/api/presets').then((r) => r.items),
  getPreset: (name: string) => req<ConfigData>(`/api/presets/${name}`),
  getPresetWithWarnings: (name: string) =>
    req<{ config: ConfigData; dropped_fields: string[]; defaulted_fields: string[] }>(
      `/api/presets/${name}?warnings=true`,
    ),
  savePreset: (name: string, data: ConfigData) =>
    req<{ name: string; path: string }>(`/api/presets/${name}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  deletePreset: (name: string) =>
    req<{ deleted: string }>(`/api/presets/${name}`, { method: 'DELETE' }),
  duplicatePreset: (src: string, newName: string) =>
    req<{ name: string; path: string }>(`/api/presets/${src}/duplicate`, {
      method: 'POST',
      body: JSON.stringify({ new_name: newName }),
    }),
  exportPresetToDataExports: (name: string, config: ConfigData) =>
    req<DataExportItem>(`/api/presets/${encodeURIComponent(name)}/export`, {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),
  /** End-to-end yaml file download link; the server FileResponse already sets Content-Disposition.
   *  <a href={...} download> is enough to trigger it, no fetch needed. */
  presetDownloadUrl: (name: string) =>
    `/api/presets/${encodeURIComponent(name)}/download`,
  importPresetFromPath: (path: string) =>
    req<{ name: string; path: string }>('/api/presets/import-from-path', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  /** End-to-end file upload: hands a .yaml/.yml/.json file to the backend to parse + schema-validate + persist,
   *  returns {name, path}. The frontend just does refreshList + setSelected(name) with the returned name.
   *
   *  On conflict (a preset with the same name exists) -> throws ApiError(status=409), err.detail =
   *  {message, config, suggested_name}; the call site pops an ImportConflictDialog based on that
   *  so the user can choose overwrite / save-as, then goes through PUT /api/presets/{name}.
   *  Bypasses req()'s JSON header so the browser adds the multipart boundary itself. */
  importPreset: async (file: File): Promise<{ name: string; path: string }> => {
    const fd = new FormData()
    fd.append('file', file, file.name)
    const resp = await fetch('/api/presets/import', { method: 'POST', body: fd })
    if (!resp.ok) {
      const body = await resp.json().catch(() => null)
      throw makeApiError(resp.status, resp.statusText, body, resp.headers.get('X-Trace-Id'))
    }
    return (await resp.json()) as { name: string; path: string }
  },

  /** WandB preset yaml download link (**contains the real api_key**, an explicit server-side export endpoint).
   *  <a href={...} download> is enough to trigger it, no fetch needed. */
  wandbPresetExportUrl: (id: string) =>
    `/api/secrets/wandb/presets/${encodeURIComponent(id)}/export`,
  /** Upload a yaml/json to import a wandb preset; returns the new preset id + the latest masked secrets. */
  importWandbPreset: async (
    file: File,
  ): Promise<{ id: string; label: string; secrets: Secrets }> => {
    const fd = new FormData()
    fd.append('file', file, file.name)
    const resp = await fetch('/api/secrets/wandb/presets/import', { method: 'POST', body: fd })
    if (!resp.ok) {
      const body = await resp.json().catch(() => null)
      throw makeApiError(resp.status, resp.statusText, body, resp.headers.get('X-Trace-Id'))
    }
    return (await resp.json()) as { id: string; label: string; secrets: Secrets }
  },

  // Compat aliases: before PP0 these were called listConfigs / getConfig / ... Keeping them around for a while.
  listConfigs: () =>
    req<{ items: PresetSummary[] }>('/api/presets').then((r) => r.items),
  getConfig: (name: string) => req<ConfigData>(`/api/presets/${name}`),
  saveConfig: (name: string, data: ConfigData) =>
    req<{ name: string; path: string }>(`/api/presets/${name}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  deleteConfig: (name: string) =>
    req<{ deleted: string }>(`/api/presets/${name}`, { method: 'DELETE' }),
  duplicateConfig: (src: string, newName: string) =>
    req<{ name: string; path: string }>(`/api/presets/${src}/duplicate`, {
      method: 'POST',
      body: JSON.stringify({ new_name: newName }),
    }),

  // Secrets ------------------------------------------------------------
  getSecrets: () => req<Secrets>('/api/secrets'),

  // Runtime mode (Colab / Local) ---------------------------------------
  /** Fetched once on first load: mode='' -> pops the mode-selection dialog. */
  getRuntime: () => req<RuntimeInfo>('/api/runtime'),
  /** Persists the user's choice. Returns 409 (runtime.mode_locked) when the env var pins the mode. */
  setRuntimeMode: (mode: RuntimeMode) =>
    req<RuntimeInfo>('/api/runtime', {
      method: 'PUT',
      body: JSON.stringify({ mode }),
    }),

  // Tag dictionary -----------------------------------------------------
  /** Current dictionary meta + whether it's loaded. Settings UI pings on startup to decide whether to show "not initialized" or the details. */
  getTagDictionaryMeta: () =>
    req<TagDictionaryMetaResponse>('/api/tag-dictionary/meta'),
  /** Full dict JSON (~600KB gzip). store.ts fetches it once at startup and caches it in memory. */
  getTagDictionaryData: () =>
    req<TagDictionaryPayload>('/api/tag-dictionary/data'),
  /** Upload a csv/txt to replace the current dictionary. Returns the new meta. */
  uploadTagDictionary: async (file: File): Promise<TagDictionaryMetaResponse> => {
    const fd = new FormData()
    fd.append('file', file, file.name)
    const resp = await fetch('/api/tag-dictionary/upload', { method: 'POST', body: fd })
    if (!resp.ok) {
      const body = await resp.json().catch(() => null)
      throw makeApiError(resp.status, resp.statusText, body, resp.headers.get('X-Trace-Id'))
    }
    return (await resp.json()) as TagDictionaryMetaResponse
  },
  /** Re-fetch the default dictionary from GitHub (used both when the first fetch failed and when the user wants to reset). */
  resetTagDictionary: () =>
    req<TagDictionaryMetaResponse>('/api/tag-dictionary/reset', { method: 'POST' }),

  // Models management (PP7) ------------------------------------------------
  getModelsCatalog: () => req<ModelsCatalog>('/api/models/catalog'),
  /** Absolute paths for the 4 model fields as currently computed by Settings. Used by the presets page's reset / new. */
  getModelPathDefaults: () => req<Record<string, string>>('/api/models/path-defaults'),
  /** YAML preview (R4): serializes the current form config through the same path used when saving to disk,
   * returning the yaml text. Pure computation, doesn't write to disk; tolerant-repair semantics match saving. */
  previewConfigYaml: (config: ConfigData) =>
    req<{ yaml: string }>('/api/schema/preview-yaml', {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),
  /** Preview computation for switching model families in a training config (multi-model P4-3). Pure computation,
   * doesn't write to disk: returns the recomputed paths + the full config after resetting family-flavor fields,
   * plus a changelog, and goes through the normal save flow once the frontend confirms. */
  switchModelFamily: (target: string, config: ConfigData) =>
    req<FamilySwitchResponse>('/api/models/family-switch', {
      method: 'POST',
      body: JSON.stringify({ target, config }),
    }),
  startModelDownload: (body: { model_id: string; variant?: string }) =>
    req<{ key: string; status: string }>('/api/models/download', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  /** Delete a downloaded asset (the reverse of downloading: delete then re-download). Path is
   *  resolved server-side; 409 while downloading / file in use. Returns the catalog after deletion. */
  deleteModelAsset: (model_id: string, variant?: string) =>
    req<ModelsCatalog>(
      `/api/models/asset?model_id=${encodeURIComponent(model_id)}`
      + (variant ? `&variant=${encodeURIComponent(variant)}` : ''),
      { method: 'DELETE' },
    ),
  /** Add a unified source candidate (download-type / local file), returns the new catalog. */
  addModelSource: (domain: string, cand: ModelSourceCandidate) =>
    req<ModelsCatalog>(`/api/model-sources/${domain}`, {
      method: 'POST',
      body: JSON.stringify(cand),
    }),
  /** Remove a candidate (doesn't touch disk; the server falls back to the default when the current selection is removed). */
  removeModelSource: (domain: string, cand: ModelSourceCandidate) =>
    req<ModelsCatalog>(`/api/model-sources/${domain}`, {
      method: 'DELETE',
      body: JSON.stringify(cand),
    }),
  selectUpscaler: (label: string) =>
    req<{ selected: string }>('/api/upscalers/select', {
      method: 'POST',
      body: JSON.stringify({ label }),
    }),
  refreshLLMModels: (body: {
    preset_id?: string
    base_url?: string
    api_key?: string
    timeout?: number
  }) =>
    req<{ items: string[]; preset_id: string; secrets: Secrets }>('/api/llm-tagger/models/refresh', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  testLLMConnection: (
    body:
      & { preset_id?: string }
      & Partial<Pick<LLMPreset, 'base_url' | 'api_key' | 'model' | 'endpoint' | 'timeout' | 'max_tokens' | 'temperature'>>,
  ) =>
    req<LLMConnectionTestResult>('/api/llm-tagger/test', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updateSecrets: (patch: SecretsPatch) =>
    req<Secrets>('/api/secrets', {
      method: 'PUT',
      body: JSON.stringify(patch),
    }),

  // Projects / Versions (PP1) -------------------------------------------
  listProjects: () =>
    req<{ items: ProjectSummary[] }>('/api/projects').then((r) => r.items),
  getProject: (pid: number) =>
    req<ProjectDetail>(`/api/projects/${pid}`),
  createProject: (body: {
    title: string
    slug?: string
    note?: string
    initial_version_label?: string
  }) =>
    req<ProjectDetail>('/api/projects', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updateProject: (
    pid: number,
    body: Partial<{
      title: string
      note: string
      active_version_id: number | null
    }>
  ) =>
    req<ProjectDetail>(`/api/projects/${pid}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteProject: (pid: number) =>
    req<{ deleted: number }>(`/api/projects/${pid}`, { method: 'DELETE' }),
  /** Archive (soft-hide, reversible): folders / versions / tasks are all left as-is. */
  archiveProject: (pid: number) =>
    req<ProjectDetail>(`/api/projects/${pid}/archive`, { method: 'POST' }),
  unarchiveProject: (pid: number) =>
    req<ProjectDetail>(`/api/projects/${pid}/unarchive`, { method: 'POST' }),

  listVersions: (pid: number) =>
    req<{ items: Version[] }>(`/api/projects/${pid}/versions`).then(
      (r) => r.items
    ),
  getVersion: (pid: number, vid: number) =>
    req<Version>(`/api/projects/${pid}/versions/${vid}`),
  createVersion: (
    pid: number,
    body: {
      label: string
      fork_from_version_id?: number
      note?: string
    }
  ) =>
    req<Version>(`/api/projects/${pid}/versions`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  /** Trigger word candidates read back from the captions in train/ (DOP needs one). */
  detectTrigger: (pid: number, vid: number) =>
    req<TriggerDetectResult>(`/api/projects/${pid}/versions/${vid}/trigger-detect`),
  updateVersion: (
    pid: number,
    vid: number,
    body: Partial<{
      note: string
      status: VersionStatus
      phase: VersionPhase
      config_name: string | null
      trigger_word: string
    }>
  ) =>
    req<Version>(`/api/projects/${pid}/versions/${vid}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deleteVersion: (pid: number, vid: number) =>
    req<{ deleted: number }>(`/api/projects/${pid}/versions/${vid}`, {
      method: 'DELETE',
    }),
  activateVersion: (pid: number, vid: number) =>
    req<{ active_version_id: number }>(
      `/api/projects/${pid}/versions/${vid}/activate`,
      { method: 'POST' }
    ),

  // Phase cursor advance / skip (ADR-0007 SS11.5-A) --------------------------
  advanceVersionPhase: (pid: number, vid: number) =>
    req<PhaseAdvanceResult>(
      `/api/projects/${pid}/versions/${vid}/advance-phase`,
      { method: 'POST' }
    ),

  skipVersionPhase: (pid: number, vid: number) =>
    req<PhaseAdvanceResult>(
      `/api/projects/${pid}/versions/${vid}/skip-phase`,
      { method: 'POST' }
    ),

  // Task config snapshot (ADR-0007 §11.7) --------------------------------
  getTaskSnapshotConfig: (taskId: number) =>
    req<{ yaml: string; config: Record<string, unknown> }>(
      `/api/queue/${taskId}/snapshot/config`
    ),

  // Download / jobs (PP2) ------------------------------------------------
  estimateDownload: (
    pid: number,
    body: { tag: string; api_source?: 'gelbooru' | 'danbooru' }
  ) =>
    req<{
      tag: string
      api_source: 'gelbooru' | 'danbooru'
      exclude_tags: string[]
      effective_query: string
      count: number
    }>(`/api/projects/${pid}/download/estimate`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  startDownload: (
    pid: number,
    body: { tag: string; count: number; api_source?: 'gelbooru' | 'danbooru' }
  ) =>
    req<Job>(`/api/projects/${pid}/download`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getDownloadStatus: (pid: number) =>
    req<{ job: Job | null; log_tail: string }>(
      `/api/projects/${pid}/download/status`
    ),
  /**
   * Local upload: single image / zip bundle / same-name .txt caption. Goes through XHR to get upload progress events
   * (fetch has no request-body progress).
   *
   * Backend is async: returns an `upload` job right after the bytes hit disk (to avoid a big zip's synchronous processing
   * triggering Cloudflare 524). Actual unzip / transcode runs on a background worker; the caller polls
   * `getUploadStatus` for the job's terminal state to get added/skipped.
   */
  uploadProjectFiles: (
    pid: number,
    files: File[],
    onProgress?: (e: UploadProgressEvent) => void,
  ): Promise<Job> => {
    const fd = new FormData()
    for (const f of files) fd.append('files', f, f.name)
    return xhrUpload<Job>(`/api/projects/${pid}/upload`, fd, onProgress)
  },
  uploadProjectFileFromPath: (pid: number, path: string) =>
    req<Job>(`/api/projects/${pid}/upload-from-path`, {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  /** Upload job status polling: result is null before the job reaches a terminal state; contains added/skipped once done. */
  getUploadStatus: (pid: number) =>
    req<{ job: Job | null; log_tail: string; result: UploadResult | null }>(
      `/api/projects/${pid}/upload/status`
    ),
  listFiles: (pid: number, bucket = 'download') =>
    req<{ items: DownloadFile[]; count: number }>(
      `/api/projects/${pid}/files?bucket=${encodeURIComponent(bucket)}`
    ),
  /** Delete the given image + same-name metadata (.booru.txt/.txt/.json) from the project's download/. */
  deleteProjectFiles: (pid: number, names: string[]) =>
    req<{ deleted: string[]; missing: string[] }>(
      `/api/projects/${pid}/files/delete`,
      {
        method: 'POST',
        body: JSON.stringify({ names }),
      }
    ),
  /** `v`: file mtime (unix s), used only as a browser-side cache-buster. **The server ignores** this param
   *  (the backend cache key is still computed from src+mtime+size); the point is to change the URL after an
   *  in-place overwrite (crop / upscale writing to the same name) so the browser doesn't hit the memory image
   *  cache and reuse the old decoded pixels. `Cache-Control: no-cache` forces disk-cache revalidation,
   *  but CSS `background-image`'s in-memory decoded image isn't bound by that and can only be invalidated
   *  via URL uniqueness -- see the PreprocessCrop bug fix. */
  projectThumbUrl: (
    pid: number,
    name: string,
    bucket = 'download',
    size = 256,
    v?: number,
    /** raw=true (only valid for bucket=download): skips resolve_origin, forces the raw bytes of download/{name}.
     *  Used by the "compare preview" left pane -- must not be hijacked by preprocess derivatives. */
    raw?: boolean,
  ) =>
    `/api/projects/${pid}/thumb?bucket=${encodeURIComponent(bucket)}&name=${encodeURIComponent(name)}&size=${size}`
    + (v ? `&v=${v}` : '')
    + (raw ? '&raw=1' : ''),

  // ---- ADR 0010 train-scope endpoints -----------------------------------
  // Added in PR-3; the frontend switched to this set in PR-4; the old ones (`/preprocess/*` without vid) will be removed in PR-5.
  startPreprocessTrain: (
    pid: number,
    vid: number,
    body: {
      mode: 'all' | 'selected' | 'all_force'
      names?: string[]
      model?: string
      tile_size?: number
      tile_pad?: number
      device?: 'auto' | 'cuda' | 'cpu'
      target_area?: number | null
    },
  ) =>
    req<Job>(`/api/projects/${pid}/versions/${vid}/preprocess/start`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getPreprocessStatusTrain: (pid: number, vid: number) =>
    req<{
      job: Job | null
      log_tail: string
      summary: { image_count: number }
    }>(`/api/projects/${pid}/versions/${vid}/preprocess/status`),
  listPreprocessFilesTrain: (pid: number, vid: number) =>
    req<{
      images: TrainImage[]
      summary: { image_count: number }
    }>(`/api/projects/${pid}/versions/${vid}/preprocess/files`),
  /** ADR 0010 SSRestore: copies download/{entry.origin} back over train/{name};
   *  returns a `no_origin` list when the download is missing (UI offers three options [drag in a replacement / keep / remove]). */
  restorePreprocessFilesTrain: (pid: number, vid: number, names: string[]) =>
    req<TrainRestoreResult>(
      `/api/projects/${pid}/versions/${vid}/preprocess/files/restore`,
      { method: 'POST', body: JSON.stringify({ names }) },
    ),
  /** Only clears the train manifest, **does not touch** the physical files in train/ (train is the training data itself). */
  resetPreprocessFilesTrain: (pid: number, vid: number) =>
    req<{ ok: boolean }>(
      `/api/projects/${pid}/versions/${vid}/preprocess/files/reset`,
      { method: 'POST' },
    ),
  listCropWorkspaceTrain: (pid: number, vid: number) =>
    req<{ images: CropWorkspaceItem[] }>(
      `/api/projects/${pid}/versions/${vid}/preprocess/crop/workspace`,
    ),
  listPreprocessDuplicatesRemovedTrain: (pid: number, vid: number) =>
    req<{ images: DuplicateRemovedItem[] }>(
      `/api/projects/${pid}/versions/${vid}/preprocess/duplicates/removed`,
    ),
  startPreprocessCropTrain: (
    pid: number,
    vid: number,
    crops: Record<string, { x: number; y: number; w: number; h: number; label?: string }[]>,
  ) =>
    req<Job>(`/api/projects/${pid}/versions/${vid}/preprocess/crop`, {
      method: 'POST',
      body: JSON.stringify({ crops }),
    }),
  /** Paint-over whole-image save (synchronous, no job): exports the canvas as a PNG over train/{name}.
   *  multipart bypasses req()'s JSON header so the browser adds the boundary itself. */
  saveInpaintTrain: async (
    pid: number,
    vid: number,
    name: string,
    blob: Blob,
  ): Promise<InpaintSaveResult> => {
    const fd = new FormData()
    fd.append('name', name)
    fd.append('file', blob, 'inpaint.png')
    const resp = await fetch(
      `/api/projects/${pid}/versions/${vid}/preprocess/inpaint/save`,
      { method: 'POST', body: fd },
    )
    if (!resp.ok) {
      const body = await resp.json().catch(() => null)
      throw makeApiError(resp.status, resp.statusText, body, resp.headers.get('X-Trace-Id'))
    }
    return (await resp.json()) as InpaintSaveResult
  },
  /** Training mask file URL (grayscale PNG, same size as the source image). No mask -> 404. */
  maskUrl: (pid: number, vid: number, name: string) =>
    `/api/projects/${pid}/versions/${vid}/preprocess/mask?name=${encodeURIComponent(name)}`,
  /** Write the training mask (grayscale PNG exported from the frontend mask layer). */
  saveMaskTrain: async (
    pid: number,
    vid: number,
    name: string,
    blob: Blob,
  ): Promise<{ name: string; mtime: number; size: number }> => {
    const fd = new FormData()
    fd.append('name', name)
    fd.append('file', blob, 'mask.png')
    const resp = await fetch(
      `/api/projects/${pid}/versions/${vid}/preprocess/mask`,
      { method: 'PUT', body: fd },
    )
    if (!resp.ok) {
      const body = await resp.json().catch(() => null)
      throw makeApiError(resp.status, resp.statusText, body, resp.headers.get('X-Trace-Id'))
    }
    return (await resp.json()) as { name: string; mtime: number; size: number }
  },
  /** Delete the training mask (= restore full-image normal learning for that image). */
  deleteMaskTrain: (pid: number, vid: number, name: string) =>
    req<{ deleted: boolean }>(
      `/api/projects/${pid}/versions/${vid}/preprocess/mask?name=${encodeURIComponent(name)}`,
      { method: 'DELETE' },
    ),

  // R-5 ledger merge: /api/jobs* was removed; jobs and tasks share the same source /api/queue (a single ID space).
  // getJob / cancelJob keep their names for the step pages (Download/Tagging/Reg/Preprocess);
  // internally they point at /api/queue now; kind is derived from task_type.
  getJob: (jid: number) =>
    req<Task & { kind?: JobKind }>(`/api/queue/${jid}`).then(
      (t) => ({ ...t, kind: (t.task_type ?? 'train') as JobKind }) as unknown as Job,
    ),
  cancelJob: (jid: number) =>
    req<{ task_id: number; canceled: boolean }>(`/api/queue/${jid}/cancel`, {
      method: 'POST',
    }),
  getLatestVersionJob: (
    pid: number,
    vid: number,
    kind: 'download' | 'reg_build',
  ) =>
    req<{ job: Job | null; log: string }>(
      `/api/projects/${pid}/versions/${vid}/jobs/latest?kind=${kind}`,
    ),

  // Tagger readiness check (used by reg-assist tagging; the auto-tagging step was removed in this fork) ------------
  checkTagger: (name: TaggerName) =>
    req<TaggerStatus>(`/api/tagger/${name}/check`),
  listCaptions: (pid: number, vid: number, folder?: string) => {
    const qs = folder ? `?folder=${encodeURIComponent(folder)}` : ''
    return req<{ folder: string | null; items: CaptionPreview[] }>(
      `/api/projects/${pid}/versions/${vid}/captions${qs}`
    )
  },
  listCaptionsFull: (pid: number, vid: number) =>
    req<{ folder: null; items: CaptionEntry[] }>(
      `/api/projects/${pid}/versions/${vid}/captions?full=1`
    ),
  commitCaptions: (pid: number, vid: number, items: CommitItem[]) =>
    req<CommitResult>(
      `/api/projects/${pid}/versions/${vid}/captions/commit`,
      { method: 'POST', body: JSON.stringify({ items }) }
    ),
  getCaption: (pid: number, vid: number, folder: string, filename: string) =>
    req<CaptionFull>(
      `/api/projects/${pid}/versions/${vid}/captions/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}`
    ),
  putCaption: (
    pid: number,
    vid: number,
    folder: string,
    filename: string,
    tags: string[]
  ) =>
    req<CaptionFull>(
      `/api/projects/${pid}/versions/${vid}/captions/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}`,
      { method: 'PUT', body: JSON.stringify({ tags }) }
    ),
  batchTag: (pid: number, vid: number, body: BatchOpRequest) =>
    req<BatchOpResult>(
      `/api/projects/${pid}/versions/${vid}/captions/batch`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  createCaptionSnapshot: (pid: number, vid: number) =>
    req<CaptionSnapshot>(
      `/api/projects/${pid}/versions/${vid}/captions/snapshot`,
      { method: 'POST' }
    ),
  listCaptionSnapshots: (pid: number, vid: number) =>
    req<{ items: CaptionSnapshot[] }>(
      `/api/projects/${pid}/versions/${vid}/captions/snapshots`
    ).then((r) => r.items),
  restoreCaptionSnapshot: (pid: number, vid: number, sid: string) =>
    req<{ id: string; written: number; removed_old: number; skipped: string[] }>(
      `/api/projects/${pid}/versions/${vid}/captions/snapshots/${sid}/restore`,
      { method: 'POST' }
    ),
  deleteCaptionSnapshot: (pid: number, vid: number, sid: string) =>
    req<{ deleted: string }>(
      `/api/projects/${pid}/versions/${vid}/captions/snapshots/${sid}`,
      { method: 'DELETE' }
    ),

  // Regularization (PP5) ------------------------------------------------
  getRegStatus: (pid: number, vid: number) =>
    req<RegStatus>(`/api/projects/${pid}/versions/${vid}/reg`),
  previewRegTags: (pid: number, vid: number, top = 20) =>
    req<{ items: RegTagCount[] }>(
      `/api/projects/${pid}/versions/${vid}/reg/preview-tags?top=${top}`
    ).then((r) => r.items),
  startRegBuild: (pid: number, vid: number, body: RegBuildRequest) =>
    req<Job>(`/api/projects/${pid}/versions/${vid}/reg/build`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  deleteReg: (pid: number, vid: number) =>
    req<{ deleted: boolean; reason?: string }>(
      `/api/projects/${pid}/versions/${vid}/reg`,
      { method: 'DELETE' }
    ),
  /** A1 -- bulk-delete the given images from the reg set (plus same-name .txt).
   * `relative_paths` is a list of paths relative to reg/, across subfolders is fine.
   * The backend automatically appends the deleted booru IDs to reg/.deleted_ids.json,
   * excluding them automatically on the next incremental build. */
  deleteRegFiles: (pid: number, vid: number, relative_paths: string[]) =>
    req<{ deleted: string[]; count: number }>(
      `/api/projects/${pid}/versions/${vid}/reg/delete-files`,
      { method: 'POST', body: JSON.stringify({ relative_paths }) }
    ),
  /** A4 -- sweeps the reg set with preprocess dedup's default params and auto-deletes each group's suggested-delete items
   * (no review panel popup, "recommended for deletion" is deleted right away; the reg set's quality bar is lower than train's).
   * Returns synchronously -- can be slow on a large set; the frontend must disable the button + show a spinner. */
  dedupPurgeReg: (pid: number, vid: number) =>
    req<{ scanned: number; groups: number; deleted: string[]; count: number }>(
      `/api/projects/${pid}/versions/${vid}/reg/dedup-purge`,
      { method: 'POST' }
    ),
  getRegCaption: (pid: number, vid: number, path: string) =>
    req<{ path: string; tags: string[] }>(
      `/api/projects/${pid}/versions/${vid}/reg/caption?path=${encodeURIComponent(path)}`
    ),
  /** PR-9 -- start a prior-generation task (the base model inverts a control image for each train image). */
  enqueueRegPrior: (pid: number, vid: number, body: RegAiRequest) =>
    req<Task>(`/api/projects/${pid}/versions/${vid}/reg/generate-prior`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  /** Replay the most recent prior-generation task + log, used to restore the log after switching pages / refreshing. */
  getLatestRegPriorTask: (pid: number, vid: number) =>
    req<{ task: Task | null; log: string }>(
      `/api/projects/${pid}/versions/${vid}/reg/generate-prior/latest`,
    ),
  /** Query the prior-generation task status. */
  getRegPriorTask: (pid: number, vid: number, taskId: number) =>
    req<Task>(`/api/projects/${pid}/versions/${vid}/reg/generate-prior/${taskId}`),

  /** Rename a reg/ subfolder (changes the Kohya repeat prefix, e.g. 2_data -> 1_data). */
  renameRegFolder: (pid: number, vid: number, name: string, newName: string) =>
    req<{ path: string }>(`/api/projects/${pid}/versions/${vid}/reg/folder`, {
      method: 'POST',
      body: JSON.stringify({ name, new_name: newName }),
    }),

  /** List all LoRA ckpt files under a version's output/ (used by the XY ckpt axis + single-image mode's ckpt switcher). */
  listVersionLoraCkpts: (pid: number, vid: number) =>
    req<{ items: LoraCkpt[] }>(`/api/projects/${pid}/versions/${vid}/lora_ckpts`)
      .then((r) => r.items),

  /** List state.pt for all of a project's versions, grouped by version (used by the Train page's resume_state picker). */
  listProjectStateCkpts: (pid: number) =>
    req<{ groups: VersionCkptGroup<StateCkpt>[] }>(`/api/projects/${pid}/state_ckpts`)
      .then((r) => r.groups),

  /** ── Remote access ──────────────────────────────────────────────────── */

  getTunnel: () => req<TunnelState>('/api/tunnel'),
  startTunnel: () => req<TunnelState>('/api/tunnel/start', { method: 'POST' }),
  stopTunnel: () => req<TunnelState>('/api/tunnel/stop', { method: 'POST' }),
  installTunnel: (provider: TunnelProvider = 'cloudflare') =>
    req<TunnelState>('/api/tunnel/install', { method: 'POST', body: JSON.stringify({ provider }) }),
  configureTunnel: (body: Partial<{ provider: TunnelProvider; autostart: boolean; ngrok_domain: string; ngrok_authtoken: string }>) =>
    req<TunnelState>('/api/tunnel/config', { method: 'PUT', body: JSON.stringify(body) }),
  rotateTunnelKey: () => req<TunnelState>('/api/tunnel/rotate-key', { method: 'POST' }),

  /** ── Checkpoint soup ────────────────────────────────────────────────── */

  /** Uploaded raw material + already-merged results (in-project ckpt goes through listProjectLoraCkpts). */
  listSoupSources: () =>
    req<{ uploads: SoupFile[]; outputs: SoupFile[] }>('/api/soup/sources'),

  /** Upload your own .safetensors; a parse failure on the backend returns 422 (catches truncated files immediately). */
  uploadSoupSource: async (file: File): Promise<SoupFile> => {
    const fd = new FormData()
    fd.append('file', file, file.name)
    const resp = await fetch('/api/soup/upload', { method: 'POST', body: fd })
    if (!resp.ok) {
      const body = await resp.json().catch(() => null)
      throw makeApiError(resp.status, resp.statusText, body, resp.headers.get('X-Trace-Id'))
    }
    return (await resp.json()) as SoupFile
  },

  deleteSoupUpload: (name: string) =>
    req<{ ok: boolean }>(`/api/soup/uploads/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  deleteSoupOutput: (name: string) =>
    req<{ ok: boolean }>(`/api/soup/outputs/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  soupDownloadUrl: (name: string) =>
    `/api/soup/outputs/${encodeURIComponent(name)}/download`,

  /** Pre-merge compatibility check (whether rank / target modules / algo match). */
  inspectSoupSources: (paths: string[]) =>
    req<SoupCompatibility>('/api/soup/inspect', {
      method: 'POST', body: JSON.stringify({ paths }),
    }),

  mergeSoup: (body: {
    inputs: { path: string; weight: number }[]
    name: string
    method: 'average' | 'sum'
    overwrite?: boolean
  }) => req<SoupMergeResult>('/api/soup/merge', { method: 'POST', body: JSON.stringify(body) }),

  /** List LoRA ckpt for all of a project's versions, grouped by version (used by the Train page's resume_lora picker). */
  listProjectLoraCkpts: (pid: number) =>
    req<{ groups: VersionCkptGroup<LoraCkpt>[] }>(`/api/projects/${pid}/lora_ckpts`)
      .then((r) => r.groups),

  /** PR-9 -- start a test-generation task. As of Phase 2: images go through the server's in-memory cache and are lost on page close. */
  enqueueGenerate: (body: GenerateRequest) =>
    req<Task>('/api/generate', { method: 'POST', body: JSON.stringify(body) }),
  /** On-disk history: scans studio_data/test/&lt;date&gt;/{single,xy}/image_N.json sidecars,
   *  returns sorted by created_at desc; used by the history rail to look back at test images persisted across sessions.
   *  Note: the path uses the disk/ subprefix to avoid /api/generate/{task_id}'s single-segment catch-all. */
  listDiskGenerateHistory: (limit = 500) =>
    req<{ entries: DiskGenerateHistoryEntry[] }>(`/api/generate/disk/history?limit=${limit}`),
  /** Current session's encrypted disk-cache history (the only source when save_test_images=false).
   *  Entries disappear after a server restart / 30s SSE disconnect + LRU; refreshing / switching routes both re-fetch this. */
  listCacheGenerateHistory: () =>
    req<{ entries: CacheGenerateHistoryEntry[] }>('/api/generate/cache/index'),
  /** Query the test task's status. */
  getGenerateTask: (id: number) => req<Task>(`/api/generate/${id}`),
  /** URL of a single test-generation image (fetched while the task is running or just finished; 404 after a 30s client disconnect + LRU). */
  generateSampleUrl: (taskId: number, filename: string) =>
    `/api/generate/${taskId}/sample/${encodeURIComponent(filename)}`,
  /** Phase 2 -- daemon status query (used by the frontend's DaemonControls). */
  getDaemonStatus: () => req<DaemonStatus>('/api/generate/daemon/status'),
  /** Phase 2 -- manually unload the daemon model (409 if busy). */
  unloadDaemon: () => req<{ ok: boolean; noop?: boolean }>(
    '/api/generate/daemon/unload', { method: 'POST' }
  ),
  /** daemon stderr ring buffer. Returns only the increment when since_seq>0. */
  getDaemonLogs: (sinceSeq = 0, limit = 2000) =>
    req<{ entries: Array<{ ts: number; seq: number; line: string }>; next_seq: number }>(
      `/api/generate/daemon/logs?since_seq=${sinceSeq}&limit=${limit}`,
    ),
  /** Phase 2 commit 14 -- TAEFlux status. */
  getTaeFluxStatus: () => req<TaeFluxStatus>('/api/generate/taeflux/status'),
  /** Phase 2 commit 14 -- synchronously download TAEFlux (~1.6MB, sub-second). No-op if it already exists. */
  installTaeFlux: () => req<{ ok: boolean; noop?: boolean }>(
    '/api/generate/taeflux/install', { method: 'POST' }
  ),

  // Train config (PP6.2) -------------------------------------------------
  getVersionConfig: (pid: number, vid: number) =>
    req<VersionConfigResponse>(`/api/projects/${pid}/versions/${vid}/config`),
  getBucketDistribution: (pid: number, vid: number) =>
    req<BucketDistribution>(
      `/api/projects/${pid}/versions/${vid}/bucket-distribution`
    ),
  getTrainEstimate: (pid: number, vid: number) =>
    req<TrainEstimate>(`/api/projects/${pid}/versions/${vid}/train-estimate`),
  putVersionConfig: (pid: number, vid: number, data: ConfigData) =>
    req<{ has_config: true; config: ConfigData }>(
      `/api/projects/${pid}/versions/${vid}/config`,
      { method: 'PUT', body: JSON.stringify(data) }
    ),
  forkPresetForVersion: (pid: number, vid: number, name: string) =>
    req<{
      has_config: true
      config: ConfigData
      from_preset: string
      dropped_fields: string[]
      defaulted_fields: string[]
    }>(
      `/api/projects/${pid}/versions/${vid}/config/from_preset`,
      { method: 'POST', body: JSON.stringify({ name }) }
    ),
  saveVersionConfigAsPreset: (
    pid: number,
    vid: number,
    name: string,
    overwrite = false
  ) =>
    req<{ saved_preset: string; config: ConfigData }>(
      `/api/projects/${pid}/versions/${vid}/config/save_as_preset`,
      { method: 'POST', body: JSON.stringify({ name, overwrite }) }
    ),
  /** 0.17 P-B -- when scheduledAt (unix seconds) is given, creates the task as scheduled; once due,
   *  the supervisor promotes it to pending (doesn't enqueue immediately -- unlike the old behavior). */
  enqueueVersionTraining: (pid: number, vid: number, opts?: { scheduledAt?: number }) =>
    req<Task>(
      `/api/projects/${pid}/versions/${vid}/queue`,
      {
        method: 'POST',
        ...(opts?.scheduledAt != null
          ? { body: JSON.stringify({ scheduled_at: opts.scheduledAt }) }
          : {}),
      }
    ),

  // Curation (PP3) -------------------------------------------------------
  getCuration: (pid: number, vid: number) =>
    req<CurationView>(`/api/projects/${pid}/versions/${vid}/curation`),
  copyToTrain: (
    pid: number,
    vid: number,
    body: { files: string[]; dest_folder: string }
  ) =>
    req<CopyResult>(`/api/projects/${pid}/versions/${vid}/curation/copy`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  removeFromTrain: (
    pid: number,
    vid: number,
    body: { folder: string; files: string[] }
  ) =>
    req<{ removed: string[]; missing: string[] }>(
      `/api/projects/${pid}/versions/${vid}/curation/remove`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  folderOp: (
    pid: number,
    vid: number,
    body: { op: 'create' | 'rename' | 'delete'; name: string; new_name?: string }
  ) =>
    req<Record<string, unknown>>(
      `/api/projects/${pid}/versions/${vid}/curation/folder`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  // Validation set (held-out) is maintained manually -- symmetric with train curation, the right column is flat with no folders
  getCurationValidation: (pid: number, vid: number) =>
    req<CurationValidationView>(
      `/api/projects/${pid}/versions/${vid}/curation/validation`
    ),
  copyToValidation: (pid: number, vid: number, body: { files: string[] }) =>
    req<CopyResult>(
      `/api/projects/${pid}/versions/${vid}/curation/validation/copy`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  removeFromValidation: (
    pid: number,
    vid: number,
    body: { items: { folder: string; name: string }[] }
  ) =>
    req<{ removed: string[]; missing: string[] }>(
      `/api/projects/${pid}/versions/${vid}/curation/validation/remove`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  // ADR 0010 train scope duplicates
  scanDuplicatesTrain: (
    pid: number,
    vid: number,
    body: DuplicateScanOptions,
  ) =>
    req<DuplicateScanResult>(
      `/api/projects/${pid}/versions/${vid}/preprocess/duplicates/scan`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  applyDuplicateActionTrain: (
    pid: number,
    vid: number,
    body: { names: string[] },
  ) =>
    req<DuplicateApplyResult>(
      `/api/projects/${pid}/versions/${vid}/preprocess/duplicates/apply`,
      { method: 'POST', body: JSON.stringify(body) }
    ),
  versionThumbUrl: (
    pid: number,
    vid: number,
    bucket: 'train' | 'reg' | 'samples' | 'validation',
    name: string,
    folder?: string,
    size: number = 256
  ) => {
    const qs = new URLSearchParams({ bucket, name, size: String(size) })
    if (folder) qs.set('folder', folder)
    return `/api/projects/${pid}/versions/${vid}/thumb?${qs.toString()}`
  },

  // Queue --------------------------------------------------------------
  listQueue: (status?: TaskStatus, opts?: { includeGenerate?: boolean }) => {
    const params: string[] = []
    if (status) params.push(`status=${status}`)
    // /api/queue hides generate (test-generation) tasks by default so the list doesn't mix them with train slots;
    // an explicit toggle shows generate tasks (e.g. Overview's "view output").
    if (opts?.includeGenerate) params.push('include_generate=true')
    const qs = params.length ? `?${params.join('&')}` : ''
    return req<{ items: Task[] }>(`/api/queue${qs}`).then((r) => r.items)
  },
  // 0.17 P-A/P-C -- data source for the queue page's sections. live = in progress + waiting (running/paused/pending),
  // not paginated; q searches name/config_name.
  // Not paginated; q searches name/config_name; type filters by task_type (0.17 P-F).
  listQueueLive: (q?: string, type?: TaskType, resourceClass?: QueueResourceClass) => {
    const params = new URLSearchParams({ group: 'live' })
    if (q) params.set('q', q)
    if (type) params.set('types', type)
    if (resourceClass) params.set('resource_class', resourceClass)
    return req<{ items: Task[] }>(`/api/queue?${params}`).then((r) => r.items)
  },
  // 0.17 P-E -- history = finished (done/failed/canceled), paginated on the backend. status passes a terminal state
  // as a sub-filter, q searches name/config_name, type filters by task_type (P-F). Returns
  // { items, total, page, page_size }.
  listQueueHistory: (opts: {
    page: number; pageSize: number; q?: string; status?: TaskStatus;
    type?: TaskType; resourceClass?: QueueResourceClass
  }) => {
    const params = new URLSearchParams({
      group: 'history',
      page: String(opts.page),
      page_size: String(opts.pageSize),
    })
    if (opts.q) params.set('q', opts.q)
    if (opts.status) params.set('status', opts.status)
    if (opts.type) params.set('types', opts.type)
    if (opts.resourceClass) params.set('resource_class', opts.resourceClass)
    return req<QueueHistoryPage>(`/api/queue?${params}`)
  },
  getTask: (id: number) => req<Task>(`/api/queue/${id}`),
  enqueue: (payload: { config_name: string; name?: string; priority?: number }) =>
    req<Task>('/api/queue', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  cancelTask: (id: number) =>
    req<{ task_id: number; canceled: boolean }>(`/api/queue/${id}/cancel`, {
      method: 'POST',
    }),
  /** _v20 -- write / clear a task's note. Empty string -> clears it (stored as NULL server-side). Editable in any status. */
  setTaskNote: (id: number, note: string) =>
    req<Task>(`/api/queue/${id}/note`, {
      method: 'PUT',
      body: JSON.stringify({ note }),
    }),
  /** A task's training sample-image listing (found by scanning disk, also available for finished tasks). Used by the queue page's inline sample strip. */
  listTaskSamples: (id: number) =>
    req<{ items: TaskSample[]; total: number }>(`/api/queue/${id}/samples`),
  /** 0.17 P-B -- manually bump a scheduled task: converts it to pending immediately so it joins scheduling. 409 if not scheduled. */
  startTaskNow: (id: number) =>
    req<{ task_id: number; status: string }>(`/api/queue/${id}/start_now`, {
      method: 'POST',
    }),
  retryTask: (id: number) =>
    req<Task>(`/api/queue/${id}/retry`, { method: 'POST' }),
  /** ADR 0006 -- pause a running task. The task is still running when this returns; subscribe to the SSE
   *  task_state_changed event to see status flip to paused. Throws 409 if the state is wrong (not running / train_loop
   *  hasn't started yet). */
  pauseTask: (id: number) =>
    req<{ task_id: number; pause_pending: boolean }>(
      `/api/queue/${id}/pause`,
      { method: 'POST' },
    ),
  /** ADR 0006 PR-3 + Addendum 2 -- resume a paused / failed / canceled task
   *  (continues training from the most recent end-of-epoch auto backup). Returns 409 if the resume-point file is missing,
   *  which routes to ResumeFieldPicker to start a new task. done tasks can't be resumed (use retry instead). */
  resumeTask: (id: number) =>
    req<{ task_id: number; status: string }>(
      `/api/queue/${id}/resume`,
      { method: 'POST' },
    ),
  /** ADR 0006 PR-2 -- query the queue's hold state + the pending count waiting to resume scheduling. */
  getQueueHold: () => req<QueueHoldState>('/api/queue/hold'),
  /** Hold the queue: the dispatcher stops pulling new tasks; already-running ones are unaffected. */
  holdQueue: () =>
    req<{ held: boolean }>('/api/queue/hold', { method: 'POST' }),
  /** Resume scheduling: the dispatcher pulls pending tasks by priority again. */
  releaseQueue: () =>
    req<{ held: boolean }>('/api/queue/release', { method: 'POST' }),
  deleteTask: (id: number) =>
    req<{ deleted: number }>(`/api/queue/${id}`, { method: 'DELETE' }),
  /** List every file in a task's output directory (with size/mtime/whether it's a lora).
   * `supports_open_folder` is true only when the request comes from loopback, false in the cloud. */
  getTaskOutputs: (id: number) =>
    req<TaskOutputs>(`/api/queue/${id}/outputs`),
  /** Direct download link for a single output file, no request needed. <a href={...} download> is enough. */
  taskOutputDownloadUrl: (id: number, path: string) =>
    `/api/queue/${id}/output/${path.split('/').map(encodeURIComponent).join('/')}`,
  /** Direct download link for a zip of the output directory.
   * No files -> the whole thing; pass an array of relative paths -> only those (validated against a backend whitelist).
   * Trigger with <a href download>, the browser handles the download bar natively; once the backend finishes writing the
   * zip it publishes task_outputs_zip_ready / task_outputs_zip_failed events for the frontend to clear the loading state. */
  taskOutputsZipUrl: (id: number, files?: ReadonlyArray<string>) => {
    if (!files || files.length === 0) return `/api/queue/${id}/outputs.zip`
    const q = files.map((n) => encodeURIComponent(n)).join(',')
    return `/api/queue/${id}/outputs.zip?files=${q}`
  },
  exportTaskOutputs: (id: number, files?: ReadonlyArray<string>) =>
    req<DataExportItem>(`/api/queue/${id}/export-outputs`, {
      method: 'POST',
      body: JSON.stringify({ files: files && files.length > 0 ? Array.from(files) : null }),
    }),
  /** Delete selected files under the output directory (batch). relative_paths are relative to output/.
   *  If any doesn't exist -> the backend rejects the whole batch with 404; the frontend toasts the error and the caller should refresh itself. */
  deleteTaskOutputs: (id: number, files: ReadonlyArray<string>) =>
    req<{ deleted: string[] }>(`/api/queue/${id}/outputs`, {
      method: 'DELETE',
      body: JSON.stringify({ files: Array.from(files) }),
    }),

  // PP8 -- WD14 runtime / GPU package install ------------------------------------------
  /** Current onnxruntime status: package name / version / providers / nvidia-smi detection result. */
  getWD14Runtime: () => req<WD14Runtime>('/api/wd14/runtime'),
  /** Switch onnxruntime (synchronous pip, takes a few minutes; UI must show loading). */
  installWD14Runtime: (target: 'auto' | 'gpu' | 'cpu' | 'directml') =>
    req<WD14InstallResult>('/api/wd14/install', {
      method: 'POST',
      body: JSON.stringify({ target }),
    }),

  // PR-S2 -- PyTorch runtime / one-click reinstall ---------------------------------------
  /** Current torch status: version / CUDA build / cuda.is_available / driver detection / recommended cu tag. */
  getTorchStatus: () => req<TorchStatus>('/api/torch/status'),
  /** Uninstall and reinstall torch + torchvision; synchronous pip, can take 5-30 minutes, UI must show loading.
   *  Studio must be restarted after install (C extensions can't be hot-swapped). */
  reinstallTorch: (target: 'auto' | TorchCuTag) =>
    req<TorchReinstallResult>('/api/torch/reinstall', {
      method: 'POST',
      body: JSON.stringify({ target }),
    }),

  // PR-7b -- Flash Attention runtime / wheel install ----------------------------
  /** Current flash_attn status + environment detection + candidate GitHub wheels (top 20).
   *  When fetch_error is non-null, candidates=[] and the UI should prompt the user to use a manual URL instead. */
  getFlashAttnStatus: () => req<FlashAttnStatus>('/api/flash-attention/status'),
  /** Install a flash_attn wheel; url=null lets the service auto-match one.
   *  Synchronous pip install (remote wheel ~150MB), can take a few minutes; the UI button must show loading.
   *  Studio must be restarted to switch after install (C extensions can't be hot-swapped). */
  installFlashAttn: (url: string | null) =>
    req<FlashAttnInstallResult>('/api/flash-attention/install', {
      method: 'POST',
      body: JSON.stringify({ url }),
    }),

  // xformers runtime (used when attention_backend=xformers) -----------------------
  /** xformers install status. Simpler than flash_attn: xformers installs directly from PyPI,
   *  no complex selection logic over a list of candidate GitHub wheels. */
  getXformersStatus: () => req<XformersStatus>('/api/xformers/status'),
  /** pip install xformers --index-url <torch-cu-index>. Synchronous pip, takes a few minutes.
   *  On failure the backend passes the tail of stderr through to message; most failures mean the upstream wheel doesn't cover
   *  the current torch+cu combination. Studio must be restarted after install (C extensions can't be hot-swapped). */
  installXformers: () =>
    req<XformersInstallResult>('/api/xformers/install', { method: 'POST' }),

  // PP7 -- training-set export / import -----------------------------------------------
  /** Direct download link for the current version's train/ packed as a zip. <a href download> is enough to trigger it,
   * the backend publishes version_train_zip_ready/_failed SSE for the frontend to clear the "packing..." state. */
  versionTrainZipUrl: (pid: number, vid: number) =>
    `/api/projects/${pid}/versions/${vid}/train.zip`,

  /** Direct download link for the current version's bundle.zip. <a href download> triggers the browser download. */
  versionBundleZipUrl: (
    pid: number,
    vid: number,
    opts: {
      train?: boolean
      trainCaptions?: boolean
      reg?: boolean
      regCaptions?: boolean
      includeConfig?: boolean
      trainLatentCache?: boolean
      regLatentCache?: boolean
      trainMasks?: boolean
    },
  ): string => {
    const p = new URLSearchParams()
    p.set('train', opts.train !== false ? '1' : '0')
    p.set('train_captions', opts.trainCaptions !== false ? '1' : '0')
    p.set('reg', opts.reg ? '1' : '0')
    p.set('reg_captions', opts.regCaptions ? '1' : '0')
    p.set('include_config', opts.includeConfig ? '1' : '0')
    p.set('train_latent_cache', opts.trainLatentCache ? '1' : '0')
    p.set('reg_latent_cache', opts.regLatentCache ? '1' : '0')
    p.set('train_masks', opts.trainMasks ? '1' : '0')
    return `/api/projects/${pid}/versions/${vid}/bundle.zip?${p.toString()}`
  },
  exportBundleToDataExports: (
    pid: number,
    vid: number,
    opts: {
      train?: boolean
      trainCaptions?: boolean
      reg?: boolean
      regCaptions?: boolean
      includeConfig?: boolean
      trainLatentCache?: boolean
      regLatentCache?: boolean
      trainMasks?: boolean
    },
  ) =>
    req<DataExportItem>(`/api/projects/${pid}/versions/${vid}/export-bundle`, {
      method: 'POST',
      body: JSON.stringify({
        train: opts.train !== false,
        train_captions: opts.trainCaptions !== false,
        reg: opts.reg === true,
        reg_captions: opts.regCaptions === true,
        include_config: opts.includeConfig === true,
        train_latent_cache: opts.trainLatentCache === true,
        reg_latent_cache: opts.regLatentCache === true,
        train_masks: opts.trainMasks === true,
      }),
    }),
  listDataExports: () => req<DataExportItem[]>('/api/data-exports'),

  /** Import a bundle from a zip path chosen in PathPicker (supports both v1/v2) -> creates a new project + v1. */
  importBundleFromPath: (path: string) =>
    req<BundleImportResult>('/api/projects/import-bundle', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  importBundleFromDataExports: (filename: string) =>
    req<BundleImportResult>('/api/projects/import-bundle', {
      method: 'POST',
      body: JSON.stringify({ filename }),
    }),
  importBundleUpload: (
    file: File,
    onProgress?: (e: UploadProgressEvent) => void,
  ): Promise<BundleImportResult> => {
    const fd = new FormData()
    fd.append('file', file, file.name)
    return xhrUpload<BundleImportResult>('/api/projects/import-bundle/upload', fd, onProgress)
  },
  /** Upload a training-set zip -> creates a new project + v1, returns the new project. */
  importTrainProject: (
    file: File,
    onProgress?: (e: UploadProgressEvent) => void,
  ): Promise<{
    project: ProjectDetail
    version: Version
    stats: { image_count: number; tagged_count: number; untagged_count: number; concepts: string[] }
  }> => {
    const fd = new FormData()
    fd.append('file', file)
    return xhrUpload('/api/projects/import-train', fd, onProgress)
  },
  /** Open the output directory in the server host's OS file manager (loopback only). */
  openTaskFolder: (id: number) =>
    req<{ opened: string }>(`/api/queue/${id}/open-folder`, {
      method: 'POST',
    }),
  reorderQueue: (orderedIds: number[]) =>
    req<{ reordered: number }>('/api/queue/reorder', {
      method: 'POST',
      body: JSON.stringify({ ordered_ids: orderedIds }),
    }),
  getLog: (id: number) => req<LogResponse>(`/api/logs/${id}`),
  /** Fetches the full history by default (max_points=0, server skips downsampling); pass a specific number
   *  for a downsampled preview. Cold start is a one-off HTTP request, and even for a long training run (10k+ steps)
   *  it's only ~500KB payload -- not worth trading visual fidelity to save network. */
  getMonitorState: (taskId: number, maxPoints?: number) =>
    req<MonitorState>(
      `/api/state?task_id=${taskId}` +
      (maxPoints != null ? `&max_points=${maxPoints}` : '') +
      `&_=${Date.now()}`,
    ),
  sampleImageUrl: (filename: string, taskId: number, w?: number) =>
    `/samples/${filename}?task_id=${taskId}${w ? `&w=${w}` : ''}`,
  listEvalMetrics: (pid: number, vid: number, taskId?: number) =>
    req<EvalMetricsListResponse>(
      `/api/projects/${pid}/versions/${vid}/eval/metrics?` +
      (taskId ? `task_id=${taskId}&` : '') +
      `_=${Date.now()}`,
    ),
  /** List a task's post-training/manual eval jobs (joins checkpoint rows by run_id + fetches the raw log). */
  listTaskEvalJobs: (pid: number, vid: number, taskId: number) =>
    req<{ jobs: EvalJobInfo[] }>(
      `/api/projects/${pid}/versions/${vid}/eval/jobs?task_id=${taskId}`,
    ),
  /** Manually evaluate a completed task's selected checkpoint (task-scoped, bypasses the auto-eval toggle). */
  runTaskEval: (
    pid: number,
    vid: number,
    body: { task_id: number; checkpoints: string[] },
  ) =>
    req<{ queued: number }>(
      `/api/projects/${pid}/versions/${vid}/eval/run`,
      { method: 'POST', body: JSON.stringify(body) },
    ),

  // Datasets -----------------------------------------------------------
  listDatasets: (path?: string) => {
    const qs = path ? `?path=${encodeURIComponent(path)}` : ''
    return req<DatasetScan>(`/api/datasets${qs}`)
  },
  thumbnailUrl: (folder: string, name: string) =>
    `/api/datasets/thumbnail?folder=${encodeURIComponent(folder)}&name=${encodeURIComponent(name)}`,

  // Browse -------------------------------------------------------------
  browse: (path?: string) => {
    const qs = path ? `?path=${encodeURIComponent(path)}` : ''
    return req<BrowseResult>(`/api/browse${qs}`)
  },

  // studio_data storage location -------------------------------------------------
  // withScan=true includes a full scan, can take seconds on a large directory -- the caller should show a loading state.
  getStudioDataInfo: (withScan = true) =>
    req<StudioDataInfo>(`/api/studio-data/info?scan=${withScan}`),
  // 422 = invalid target / a running task exists; 409 = a migration is already running.
  startStudioDataMigrate: (target: string) =>
    req<{ ok: boolean }>('/api/studio-data/migrate', {
      method: 'POST',
      body: JSON.stringify({ target }),
    }),
  getStudioDataMigrateStatus: () =>
    req<StudioDataMigrateStatus>('/api/studio-data/migrate_status'),

  // Models-root storage location (mirrors studio_data, but the migration takes effect immediately, no restart needed) ----------
  getModelsRootInfo: (withScan = true) =>
    req<ModelsRootInfo>(`/api/models-root/info?scan=${withScan}`),
  // 422 = invalid target / a running task exists; 409 code models_root.migration_busy = a
  // migration is already running; 409 code models_root.target_conflict = the target already has models data (detail carries
  // existing_files/existing_bytes/same_name_files, the modal pops [skip/overwrite/cancel] and resends with
  // onConflict).
  startModelsRootMigrate: (target: string, onConflict?: 'skip' | 'overwrite') =>
    req<{ ok: boolean }>('/api/models-root/migrate', {
      method: 'POST',
      body: JSON.stringify({ target, on_conflict: onConflict ?? null }),
    }),
  getModelsRootMigrateStatus: () =>
    req<ModelsRootMigrateStatus>('/api/models-root/migrate_status'),

}

// This fork: the System update/announcements API types were removed along with the in-app updater.

export interface BrowseEntry {
  name: string
  type: 'dir' | 'file'
}

export interface BrowseResult {
  path: string
  parent: string | null
  entries: BrowseEntry[]
  /** If a file path is passed in, the backend falls back to the parent directory and puts the filename here for the picker to highlight. */
  selected?: string | null
}
