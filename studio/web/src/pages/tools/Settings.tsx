import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import { playSound, useSoundEnabled } from '../../lib/sound'
import type { TFunction } from 'i18next'
import { Trans, useTranslation } from 'react-i18next'
import { getStoredLangWithDefault, setStoredLang } from '../../i18n'
import {
  api,
  DEFAULT_WD14_MODELS,
  type LLMPreset,
  type FlashAttnStatus,
  type XformersStatus,
  type ModelDownloadStatus,
  type ModelsCatalog,
  type Secrets,
  type SecretsPatch,
  type TorchCuTag,
  type TorchStatus,
  type TunnelProvider,
  type TunnelState,
} from '../../api/client'
import { useDialog } from '../../components/Dialog'
import {
  AddLocalModelButton,
  LocalModelRows,
} from '../../components/LocalModelSources'
import FieldLabel from '../../components/ds/FieldLabel'
import { useToast } from '../../components/Toast'
import { useRuntimeModeOptional } from '../../lib/RuntimeMode'
import { useSettingsData } from '../../lib/SettingsData'
import { useSettingsDrawer } from '../../lib/SettingsDrawer'
import HelpTip from '../../components/ds/HelpTip'

const MASK = '***'

type Section =
  | 'gelbooru'
  | 'danbooru'
  | 'download'
  | 'huggingface'
  | 'wandb'
  | 'modelscope'
  | 'llm_tagger'
  | 'wd14'
  | 'cltagger'
  | 'models'
  | 'queue'
  | 'generate'
  | 'training'
  | 'proxy'

// Settings now has only the "training" config group (tagging / test / appearance /
// system tabs have been removed). The section index for the right-hand sticky nav.
/** Languages offered in Settings. Labels stay in their own language on
 *  purpose — that is how you find yours when the UI is in one you cannot read. */
const LANGUAGES: { code: string; label: string }[] = [
  { code: 'en', label: 'English' },
  { code: 'ru', label: 'Русский' },
]

const TRAINING_SECTIONS: { id: string; labelKey: string }[] = [
  { id: 'appearance', labelKey: 'settings.appearance' },
  { id: 'runtime-mode', labelKey: 'runtimeMode.current' },
  { id: 'remote-access', labelKey: 'remote.title' },
  { id: 'download-source', labelKey: 'settings.modelSource' },
  { id: 'training-runtime', labelKey: 'settings.trainingRuntime' },
  { id: 'pytorch', labelKey: 'settings.torch' },
  { id: 'flash-attn', labelKey: 'settings.flashAttn' },
  { id: 'xformers', labelKey: 'settings.xformers' },
  { id: 'models', labelKey: 'settings.trainingModels' },
]

// Fallback preset: only acts as a placeholder when GET /api/secrets fails; the real
// prompt comes from the backend's builtin json file. Hitting this fallback and then
// PUTting it back won't clobber the builtin (the backend validator backfills the
// builtin defaults again).
function _makeFallbackPreset(id: string, label: string, output_format: 'json' | 'text', extra: Partial<LLMPreset> = {}): LLMPreset {
  return {
    id,
    label,
    builtin: true,
    base_url: '',
    api_key: '',
    model: '',
    model_ids: [],
    endpoint: 'chat_completions',
    messages: [
      { type: 'text', role: 'system', content: '' },
      { type: 'image', role: 'user', content: '' },
    ],
    output_format,
    temperature: 0.2,
    max_tokens: 700,
    max_side: 1280,
    jpeg_quality: 85,
    max_image_mb: 5,
    timeout: 60,
    max_retries: 3,
    concurrency: 1,
    requests_per_second: 0,
    max_requests_per_minute: 0,
    assist_tagger: '',
    ...extra,
  }
}

const DEFAULT_LLM_PRESETS: LLMPreset[] = [
  _makeFallbackPreset('style_json', 'Style LoRA JSON', 'json'),
  _makeFallbackPreset('general_json', 'General LoRA JSON', 'json'),
  _makeFallbackPreset('txt_tags', 'TXT tag list', 'json'),
  _makeFallbackPreset('joycaption', 'JoyCaption (vLLM local)', 'text', {
    base_url: 'http://localhost:8000/v1',
    model: 'fancyfeast/llama-joycaption-beta-one-hf-llava',
    temperature: 0.6,
    max_tokens: 300,
  }),
]

const DEFAULT_WANDB_PRESET = {
  id: 'default',
  label: 'Default',
  api_key: '',
  project: 'AnimaLoraStudio',
  entity: '',
  base_url: '',
  mode: 'online' as const,
  log_samples: true,
  sample_max_side: 1216,
  sample_every_n_steps: 0,
  upload_model: false,
  upload_model_policy: 'last' as const,
  upload_state_manual: false,
  upload_state_manual_policy: 'last' as const,
  upload_state_auto: false,
  upload_state_auto_policy: 'last' as const,
}

const EMPTY: Secrets = {
  gelbooru: { user_id: '', api_key: '' },
  danbooru: { username: '', api_key: '', account_type: 'free' },
  download: {
    exclude_tags: [],
    parallel_workers: 4,
    api_rate_per_sec: 2,
    cdn_rate_per_sec: 5,
    save_tags: false,
    convert_to_png: true,
    remove_alpha_channel: true,
  },
  reg: { default_excluded_tags: [] },
  huggingface: { token: '', endpoint: '' },
  wandb: {
    enabled: false,
    current_preset: 'default',
    presets: [DEFAULT_WANDB_PRESET],
  },
  modelscope: { token: '' },
  eval_metrics: {
    clip_model_name: 'openai/clip-vit-base-patch32',
    dino_model_name: 'facebook/dinov2-small',
    ccip_model_name: 'ccip-caformer-24-randaug-pruned',
    enabled_metrics: [],
    eval_baseline_enabled: true,
  },
  download_source: 'huggingface',
  download_sources: {},
  llm_tagger: {
    current_preset: 'style_json',
    presets: [...DEFAULT_LLM_PRESETS],
  },
  wd14: {
    model_id: 'SmilingWolf/wd-eva02-large-tagger-v3',
    model_ids: [...DEFAULT_WD14_MODELS],
    threshold_general: 0.35,
    threshold_character: 0.85,
    blacklist_tags: [],
    batch_size: 8,
  },
  cltagger: {
    model_id: 'cella110n/cl_tagger',
    model_path: 'cl_tagger_1_02/model.onnx',
    tag_mapping_path: 'cl_tagger_1_02/tag_mapping.json',
    threshold_general: 0.35,
    threshold_character: 0.6,
    add_copyright_tag: true,
    add_artist_tag: false,
    add_meta_tag: false,
    add_model_tag: false,
    add_rating_tag: false,
    add_quality_tag: false,
    blacklist_tags: [],
    batch_size: 8,
  },
  models: {
    root: null,
    selected_anima: '1.0',
    selected: { anima: '1.0' },
    selected_te: {},
    custom_anima_paths: [],
    selected_upscaler: '4x-AnimeSharp',
    auto_sync_paths: true,
  },
  queue: { light_tasks_during_train: true },
  generate: {
    preview_every_n_steps: 3,
    attention_backend: 'auto',
    vae_precision: 'bf16',
    idle_timeout_minutes: 10,
    task_timeout_minutes: 0,
    vram_policy: 'auto',
    ram_guard: false,
    save_test_images: false,
  },
  training: { ram_guard: false },
  runtime: { mode: '', asked: false },
  proxy: {
    enabled: false,
    http_proxy: '',
    https_proxy: '',
    no_proxy: '',
  }
}

/** Keep the settings screen compatible with older secrets payloads.
 *
 * New sections are added over time, while an already running backend or a
 * test fixture can still return the previous shape.  The UI must not crash
 * during that short mismatch.  Secrets are only two levels deep, so a
 * shallow merge per section is sufficient and preserves server values.
 */
function withSecretsDefaults(value: Secrets): Secrets {
  const merged = { ...EMPTY, ...value } as Record<string, unknown>
  for (const key of Object.keys(EMPTY) as (keyof Secrets)[]) {
    const fallback = EMPTY[key]
    const actual = value[key]
    if (fallback && actual
      && typeof fallback === 'object' && !Array.isArray(fallback)
      && typeof actual === 'object' && !Array.isArray(actual)) {
      merged[key as string] = { ...fallback, ...actual }
    }
  }
  return merged as unknown as Secrets
}

const textInputClass = 'ds-inp'

const MODEL_DESCRIPTION_KEYS: Record<string, string> = {
  anima_main: 'settings.modelDescriptions.animaMain',
  anima_vae: 'settings.modelDescriptions.animaVae',
  qwen3: 'settings.modelDescriptions.qwen3',
  t5_tokenizer: 'settings.modelDescriptions.t5Tokenizer',
  wd14: 'settings.modelDescriptions.wd14',
  cltagger: 'settings.modelDescriptions.cltagger',
}

// The backend sends English catalog names; localize the main models on the frontend.
const MODEL_NAME_KEYS: Record<string, string> = {
  anima_main: 'settings.modelNames.animaMain',
  krea2_main: 'settings.modelNames.krea2Main',
}

function translatedCatalogText(keys: Record<string, string>, id: string, fallback: string | undefined, t: TFunction): string {
  const key = keys[id]
  return key ? t(key, { defaultValue: fallback ?? '' }) : (fallback ?? '')
}

export default function SettingsPage() {
  const { t } = useTranslation()
  // Shared data layer (SettingsDataProvider): secrets / catalog / SSE / downloadBusy
  // all live at the root level, so this component mounting/unmounting (drawer
  // open/close) no longer triggers a refetch. The `server` alias is kept so the
  // large block of form code below needs minimal changes.
  const {
    secrets: rawServer,
    secretsError,
    setSecrets: setServer,
    catalog,
    catalogError,
    reloadCatalog,
    downloadBusy,
    startDownload,
  } = useSettingsData()
  const server = useMemo(
    () => (rawServer ? withSecretsDefaults(rawServer) : null),
    [rawServer],
  )
  const [draft, setDraft] = useState<Secrets>(EMPTY)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const { toast } = useToast()
  const drawer = useSettingsDrawer()
  // For the right-hand section index: the sticky nav's IntersectionObserver root + scroll container
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  // Sync draft from secrets the first time it arrives; afterward, server changes
  // (after save) no longer overwrite draft, to avoid wiping the user's unsaved edits
  // (save itself calls setDraft(next)).
  const draftInitRef = useRef(false)
  useEffect(() => {
    if (server && !draftInitRef.current) {
      setDraft(server)
      draftInitRef.current = true
    }
  }, [server])
  // When the data layer's secrets fetch fails, surface the error into this component's error state, reusing the bottom error bar.
  useEffect(() => { if (secretsError) setError(secretsError) }, [secretsError])

  const dirty = useMemo(
    () => server !== null && JSON.stringify(server) !== JSON.stringify(draft),
    [server, draft]
  )

  // Before closing the drawer, this ref is used to ask "is it dirty"; the ref
  // refreshes on every render, but the registered function only mounts once, avoiding effect churn.
  const dirtyRef = useRef(false)
  dirtyRef.current = dirty
  useEffect(() => {
    drawer.registerDirtyGuard(() => dirtyRef.current)
    return () => drawer.registerDirtyGuard(null)
  }, [drawer])

  // When the drawer opens with open({ section }), jump to that section (replaces the
  // old ?section= URL param). sectionRequest carries a nonce, so opening the same
  // section again still retriggers the effect.
  const drawerSectionReq = drawer.sectionRequest
  useEffect(() => {
    if (!drawerSectionReq) return
    const section = drawerSectionReq.section
    const t1 = setTimeout(() => {
      const el = document.getElementById(section)
      el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 50)
    return () => clearTimeout(t1)
  }, [drawerSectionReq])

  const update = <S extends Section, K extends keyof Secrets[S]>(
    section: S,
    key: K,
    value: Secrets[S][K]
  ) => {
    setDraft((prev) => ({
      ...prev,
      [section]: { ...prev[section], [key]: value },
    }))
  }

  /** Update a top-level, non-object field on Secrets (e.g. download_source). */
  const updateTop = <K extends keyof Secrets>(key: K, value: Secrets[K]) => {
    setDraft((prev) => ({ ...prev, [key]: value }))
  }

  const save = async () => {
    if (!server) return
    const patch = buildPatch(draft, server)
    setSaving(true)
    setError(null)
    try {
      const next = await api.updateSecrets(patch)
      setServer(next)
      setDraft(next)
      // After the candidate model_ids change, the wd14 variants in the catalog need a refresh
      void reloadCatalog()
      toast(t('settings.saved'), 'success')
    } catch (e) {
      setError(String(e))
      toast(t('settings.saveFailed'), 'error')
    } finally {
      setSaving(false)
    }
  }

  if (error && !server) {
    return (
      <div className="text-err font-mono text-sm p-4 bg-err-soft rounded-md">
        {error}
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div style={{ height: 52, flex: 'none', display: 'flex', alignItems: 'center', gap: 10, padding: '0 18px', borderBottom: '1px solid var(--line)' }}>
        <span style={{ fontSize: 14, fontWeight: 600, letterSpacing: '-.02em' }}>{t('settings.title')}</span>
        <span className={`ds-badge ${dirty ? 'ds-warn' : 'ds-ok'}`} style={{ marginLeft: 'auto' }}>
          {saving ? t('common.saving') : dirty ? t('settings.unsavedBadge') : t('settings.savedBadge')}
        </span>
        {drawer.isOpen && (
          <button
            type="button"
            onClick={() => void drawer.close()}
            title={t('settings.drawerClose')}
            aria-label={t('settings.drawerClose')}
            className="ds-kebab"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>
          </button>
        )}
      </div>

      <div className="ds-set-grid">
        <div className="ds-set-rail">
          <div className="ds-cap" style={{ margin: '0 0 9px 9px' }}>{t('settings.sectionsCap')}<HelpTip>{t('settings.railNote')}</HelpTip></div>
          <SectionIndex sections={TRAINING_SECTIONS} scrollContainer={scrollContainerRef} />
        </div>

      <div ref={scrollContainerRef} className="ds-set-scroll">
      <div className="ds-set-cards">

      {error && <div className="ds-note ds-err ds-mono">{error}</div>}

      <AppearanceSection />

      <RuntimeModeSection />

      <RemoteAccessSection />

      <SettingsSection id="download-source" title={t('settings.modelSource')} tip={t('settings.downloadSourceHelp')}>
        <div className="ds-fgrid">
          <SetCell label={t('settings.downloadSource')}>
            <DownloadSourceSelect
              value={draft.download_source}
              onChange={(v) => updateTop('download_source', v)}
            />
          </SetCell>

          {/* HF/ModelScope tokens both stay in secrets (not lost when switching
           * sources); only the current source's credentials are shown. */}
          {draft.download_source === 'huggingface' ? (
            <>
              <SetCell label={t('settings.tokenLabel')} tip={t('settings.hfTokenHelp')}>
                <SensitiveInput
                  value={draft.huggingface.token}
                  serverValue={server?.huggingface.token ?? ''}
                  onChange={(v) => update('huggingface', 'token', v)}
                />
              </SetCell>
              <SetCell label={t('settings.endpointLabel')} tip={t('settings.hfEndpointHelp')}>
                <HFEndpointSelect
                  value={draft.huggingface.endpoint}
                  onChange={(v) => update('huggingface', 'endpoint', v)}
                />
              </SetCell>
            </>
          ) : (
            <SetCell
              label={t('settings.tokenLabel')}
              tip={<>
                <p>{t('settings.modelscopeTokenHelp')}</p>
                <p><Trans i18nKey="settings.modelscopeInstallHelp" components={{ code: <code /> }} /></p>
              </>}
            >
              <SensitiveInput
                value={draft.modelscope.token}
                serverValue={server?.modelscope.token ?? ''}
                onChange={(v) => update('modelscope', 'token', v)}
              />
            </SetCell>
          )}
        </div>
      </SettingsSection>

      <SettingsSection id="training-runtime" title={t('settings.trainingRuntime')}>
        <div className="ds-switches">
          <SwitchRow
            label={t('settings.lightTasksDuringTrain')}
            tip={t('settings.lightTasksDuringTrainHint')}
            value={draft.queue.light_tasks_during_train}
            onChange={(v) => update('queue', 'light_tasks_during_train', v)}
          />
          <SwitchRow
            label={t('settings.trainingRamGuard')}
            tip={<>
              <p>{t('settings.trainingRamGuardHelp')}</p>
              <p>{t('settings.trainingRamGuardDefaultHelp')}</p>
            </>}
            value={draft.training.ram_guard}
            onChange={(v) => update('training', 'ram_guard', v)}
          />
        </div>
      </SettingsSection>

      <PyTorchSection />

      <FlashAttentionSection />

      <XformersSection />

      <ModelsSection
        catalog={catalog}
        busy={downloadBusy}
        start={startDownload}
        reloadCatalog={reloadCatalog}
        catalogError={catalogError}
        t={t}
      />

    </div>
      </div>
      </div>

      <div className="ds-set-foot">
        {dirty && (
          <button type="button" className="ds-btn-primary" onClick={save} disabled={saving}>
            {saving ? t('common.saving') : t('common.save')}
          </button>
        )}
        {drawer.isOpen && (
          <button type="button" className="ds-btn-dark" onClick={() => void drawer.close()}>{t('settings.done')}</button>
        )}
      </div>
    </div>
  )
}

// ── Runtime mode (Colab / Local) ───────────────────────────────────────────

/** Runtime mode switch (Colab / Local). After the first-screen `RuntimeModeGate`
 *  asks once, this is the only place left to change it.
 *
 *  Deliberately **doesn't** go through the draft/save flow: the mode isn't part of
 *  the training config, it has its own endpoint (`PUT /api/runtime`), and after
 *  changing it we need to show "restart for the bound address to take effect" --
 *  there'd be nowhere to hang that notice if it were folded into the batch Save. */
function RuntimeModeSection() {
  const { t } = useTranslation()
  const runtime = useRuntimeModeOptional()
  const { toast } = useToast()
  const [busy, setBusy] = useState(false)

  const info = runtime?.info
  const mode = runtime?.mode ?? 'local'
  if (!runtime || !info) return null

  const apply = async (next: 'local' | 'colab') => {
    if (next === mode || busy) return
    setBusy(true)
    try {
      await runtime.setMode(next)
      toast(t('runtimeMode.switched', { mode: t(`runtimeMode.${next}.name`) }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <SettingsSection
      id="runtime-mode"
      title={t('runtimeMode.current')}
      tip={<span style={{ wordBreak: 'break-all' }}>studio_data: {info.environment.studio_data}</span>}
    >
      <div className="ds-set-pad">
      <div className="ds-optcards ds-compact">
        {info.modes.map((m) => (
          <button
            key={m}
            type="button"
            disabled={busy || info.locked}
            onClick={() => void apply(m)}
            className={`ds-optcard${mode === m ? ' ds-is-on' : ''}`}
            aria-pressed={mode === m}
          >
            <span className="ds-optcard-ico">
              {m === 'local'
                ? <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></svg>
                : <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M17.5 19a4.5 4.5 0 1 0-1.4-8.8A6 6 0 0 0 4.5 12 3.5 3.5 0 0 0 6 19z" /></svg>}
            </span>
            <span className="ds-optcard-txt">
              <span className="ds-optcard-name">
                {t(`runtimeMode.${m}.name`)}
                {mode === m && <span className="ds-badge ds-ok">{t('settings.modeActive')}</span>}
              </span>
              <span className="ds-optcard-desc">{t(`runtimeMode.${m}.summary`)}</span>
            </span>
            <span className={`ds-radio${mode === m ? ' ds-on' : ''}`} />
          </button>
        ))}
      </div>
      {info.locked && (
        <div className="ds-note ds-warn">{t('runtimeMode.lockedBy', { env: 'ALS_RUNTIME_MODE' })}</div>
      )}
      </div>
    </SettingsSection>
  )
}


// ── Remote access (reach the studio from a phone) ──────────────────────────

/** Provider errors carry the page to open (e.g. Tailscale's "allow Funnel"
 *  consent link); make those clickable instead of making the user copy them. */
function linkify(text: string): React.ReactNode[] {
  return text.split(/(https?:\/\/[^\s)]+)/g).map((part, i) =>
    /^https?:\/\//.test(part)
      ? <a key={i} href={part} target="_blank" rel="noreferrer" className="underline break-all">{part}</a>
      : part,
  )
}

const TUNNEL_PROVIDERS: TunnelProvider[] = ['tailscale', 'ngrok', 'cloudflare']

/** Public link to the studio, with a choice of how it is made.
 *
 *  Tailscale Funnel and ngrok (with its free static domain) give a permanent
 *  address; Cloudflare's quick tunnel needs no account but changes on every
 *  start. The access key is persistent too, so with a permanent provider the
 *  whole link stays the same and can live in the phone's bookmarks. "Open on
 *  startup" brings the link up as soon as the studio starts.
 *
 *  Deliberately blunt about what the link is: a public address onto a machine
 *  whose API can read files and start jobs. */
function RemoteAccessSection() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const [state, setState] = useState<TunnelState | null>(null)
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState(false)
  const [domain, setDomain] = useState('')
  const [token, setToken] = useState('')

  useEffect(() => {
    api.getTunnel().then((s) => {
      setState(s)
      setDomain(s.ngrok_domain ?? '')
    }).catch(() => setState(null))
  }, [])

  const run = async (fn: () => Promise<TunnelState>) => {
    setBusy(true)
    try {
      const s = await fn()
      setState(s)
      return s
    } catch (e) {
      toast(String((e as Error).message || e), 'error')
      // The server keeps the reason; refresh so it shows under the button.
      api.getTunnel().then(setState).catch(() => {})
      return null
    } finally {
      setBusy(false)
    }
  }

  const copy = async () => {
    if (!state?.url) return
    try {
      await navigator.clipboard.writeText(state.url)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch {
      toast(t('remote.copyFailed'), 'error')
    }
  }

  if (!state) return null

  const provider = state.provider ?? 'cloudflare'
  const info = state.providers?.[provider]
  const installed = info?.installed ?? state.installed
  const ngrokReady = provider !== 'ngrok' || state.has_ngrok_token
  const runningOther = state.running && state.running_provider && state.running_provider !== provider

  const saveNgrok = () => run(() => api.configureTunnel({
    ngrok_domain: domain.trim(),
    ...(token.trim() ? { ngrok_authtoken: token.trim() } : {}),
  })).then((s) => { if (s) { setToken(''); toast(t('remote.ngrokSaved'), 'success') } })

  const rotate = async () => {
    if (!(await confirm(t('remote.rotateConfirm'), { tone: 'danger', okText: t('remote.rotate') }))) return
    await run(api.rotateTunnelKey)
  }

  return (
    <SettingsSection id="remote-access" title={t('remote.title')} tip={t('remote.blurb')}>
      <div className="ds-set-pad">
        <div className="ds-optcards ds-compact ds-set-provs">
          {TUNNEL_PROVIDERS.map((p) => {
            const active = provider === p
            return (
              <label
                key={p}
                className={`ds-optcard${active ? ' ds-is-on' : ''}${state.running ? ' ds-is-locked' : ''}`}
              >
                <input
                  type="radio" name="tunnel-provider" className="sr-only"
                  checked={active} disabled={busy || state.running}
                  onChange={() => void run(() => api.configureTunnel({ provider: p }))}
                />
                <span className="ds-optcard-txt">
                  <span className="ds-optcard-name">{t(`remote.provider.${p}.name`)}</span>
                  <span className="ds-optcard-desc">{t(`remote.provider.${p}.hint`)}</span>
                </span>
                <span className={`ds-radio${active ? ' ds-on' : ''}`} />
              </label>
            )
          })}
        </div>

        {/* provider-specific setup */}
        {provider === 'tailscale' && !installed && (
          <div className="ds-note ds-warn">
            <span>
              {t('remote.tailscaleMissing')}{' '}
              <a href={state.providers.tailscale.download_url} target="_blank" rel="noreferrer" className="underline">
                tailscale.com/download
              </a>
            </span>
          </div>
        )}

        {provider !== 'tailscale' && !installed && (
          <div className="ds-note ds-warn ds-set-note-row">
            <span style={{ flex: 1 }}>
              {t(provider === 'ngrok' ? 'remote.ngrokNotInstalled' : 'remote.notInstalled')}
              {!info?.can_install && <HelpTip>{t('remote.installManually')}</HelpTip>}
            </span>
            <button
              type="button" className="ds-ctl ds-sm" disabled={busy || !info?.can_install}
              onClick={() => void run(() => api.installTunnel(provider))}
            >
              {busy ? t('remote.installing') : t(provider === 'ngrok' ? 'remote.installNgrok' : 'remote.install')}
            </button>
          </div>
        )}
      </div>

      {provider === 'ngrok' && (
        <div className="ds-fgrid ds-set-grid-tight">
          <SetCell
            label={t('remote.ngrokToken')}
            extra={<a href={state.providers.ngrok.token_url} target="_blank" rel="noreferrer" className="ds-set-link">{t('remote.whereToGet')}</a>}
          >
            <input
              className="ds-inp ds-mono" type="password" autoComplete="off"
              placeholder={state.has_ngrok_token ? t('remote.ngrokTokenSaved') : t('remote.ngrokTokenPlaceholder')}
              value={token} onChange={(e) => setToken(e.target.value)} disabled={state.running}
            />
          </SetCell>
          <SetCell
            label={t('remote.ngrokDomain')}
            tip={t('remote.ngrokHint')}
            extra={<a href={state.providers.ngrok.domains_url} target="_blank" rel="noreferrer" className="ds-set-link">{t('remote.whereToGet')}</a>}
          >
            <div className="ds-set-inline">
              <input
                className="ds-inp ds-mono" placeholder="calm-otter-123.ngrok-free.app"
                value={domain} onChange={(e) => setDomain(e.target.value)} disabled={state.running}
                autoCapitalize="off" autoCorrect="off" spellCheck={false}
              />
              <button
                type="button" className="ds-ctl"
                disabled={busy || state.running || (!token.trim() && domain.trim() === (state.ngrok_domain ?? ''))}
                onClick={() => void saveNgrok()}
              >
                {t('common.save')}
              </button>
            </div>
          </SetCell>
        </div>
      )}

      {/* link / start / stop */}
      <div className="ds-set-pad ds-set-bar">
        {state.running && state.url ? (
          <>
            <div className="ds-set-inline">
              <code className="ds-set-url">{state.url}</code>
              <button type="button" className="ds-ctl" onClick={() => void copy()}>
                {copied ? t('remote.copied') : t('remote.copy')}
              </button>
              <button type="button" className="ds-ctl" disabled={busy} onClick={() => void run(api.stopTunnel)}>
                {t('remote.stop')}
              </button>
              <button type="button" className="ds-ctl ds-ghost" disabled={busy} onClick={() => void rotate()}>
                {t('remote.rotate')}
              </button>
            </div>
            {runningOther && <div className="ds-note ds-info">{t('remote.restartToApply')}</div>}
            <div className="ds-note ds-warn">{t('remote.shareWarning')}</div>
          </>
        ) : (
          <>
            <div className="ds-set-inline">
              <button
                type="button" className="ds-btn-primary"
                disabled={busy || !installed || !ngrokReady}
                onClick={() => void run(api.startTunnel)}
              >
                {busy ? t('remote.starting') : t('remote.start')}
              </button>
              {provider === 'tailscale' && installed && <HelpTip>{t('remote.tailscaleSteps')}</HelpTip>}
            </div>
            {state.error && <div className="ds-note ds-err" style={{ wordBreak: 'break-word' }}><span>{linkify(state.error)}</span></div>}
          </>
        )}
      </div>

      <div className="ds-switches ds-after">
        <SwitchRow
          label={t('remote.autostart')}
          tip={state.permanent ? t('remote.autostartHintPermanent') : t('remote.autostartHintQuick')}
          value={state.autostart}
          disabled={busy}
          onChange={(v) => void run(() => api.configureTunnel({ autostart: v }))}
        />
      </div>
    </SettingsSection>
  )
}

// ── Section / Field ────────────────────────────────────────────────────────

/** Language picker. A pure client-side preference kept in localStorage, so this
 *  section never touches the API.
 *
 *  Density deliberately has no switch here: it was removed by design decision
 *  (see lib/theme.ts) and the app is pinned to the compact scale. */
function AppearanceSection() {
  const { t, i18n } = useTranslation()
  const [lang, setLang] = useState(() => getStoredLangWithDefault())
  const [soundOn, setSoundOn] = useSoundEnabled()

  const applyLang = (next: string) => {
    setLang(next)
    setStoredLang(next)
    void i18n.changeLanguage(next)
  }

  return (
    <SettingsSection id="appearance" title={t('settings.appearance')}>
      <div className="ds-fgrid">
      <SetCell label={t('settings.language')} tip={t('settings.languageDesc')}>
        <div className="ds-seg" style={{ alignSelf: 'flex-start' }}>
          {LANGUAGES.map((option) => (
            <button
              key={option.code}
              type="button"
              onClick={() => applyLang(option.code)}
              aria-pressed={lang === option.code}
              className={`ds-seg-item${lang === option.code ? ' ds-is-active' : ''}`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </SetCell>
      </div>
      <div className="ds-switches ds-after">
        <SwitchRow
          label={t('settings.sounds')}
          tip={t('settings.soundsDesc')}
          value={soundOn}
          onChange={(v) => {
            setSoundOn(v)
            if (v) playSound('success')
          }}
        />
      </div>
    </SettingsSection>
  )
}

/** One settings card, the same grey card the training config uses. The body
 *  is up to the caller: a `.ds-fgrid` of cells, a `.ds-switches` list, or a
 *  padded block. Longer explanations go under the "?" next to the title. */
function SettingsSection({
  id, title, tip, extra, children,
}: {
  id?: string
  title: string
  tip?: React.ReactNode
  /** Right side of the header (status badge and the like). */
  extra?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <section id={id} className="ds-fgroup ds-set-card">
      <div className="ds-fgroup-head">
        <h2 className="ds-fgroup-title">{title}</h2>
        {tip && <HelpTip>{tip}</HelpTip>}
        {extra && <span className="ds-set-extra">{extra}</span>}
      </div>
      {children}
    </section>
  )
}

/** Same as SettingsSection, folded by default unless something needs attention. */
function CollapsibleSection({
  id, title, tip, extra, open, children,
}: {
  id: string
  title: string
  tip?: React.ReactNode
  extra?: React.ReactNode
  open: boolean
  children: React.ReactNode
}) {
  return (
    <details id={id} open={open} className="ds-fgroup ds-set-card ds-set-fold">
      <summary className="ds-fgroup-head">
        <svg className="ds-set-chev" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m9 6 6 6-6 6" /></svg>
        <h2 className="ds-fgroup-title">{title}</h2>
        {tip && <HelpTip>{tip}</HelpTip>}
        {extra && <span className="ds-set-extra">{extra}</span>}
      </summary>
      {children}
    </details>
  )
}

/** A labelled cell of a `.ds-fgrid`, same shape as the training config cells. */
function SetCell({ label, tip, extra, full, children }: {
  label: string
  tip?: React.ReactNode
  /** Small trailing item in the label row (a link, a badge). */
  extra?: React.ReactNode
  full?: boolean
  children: React.ReactNode
}) {
  return (
    <div className={`ds-fcell${full ? ' ds-full' : ''}`}>
      <div className="ds-fcell-head">
        <FieldLabel label={label} tip={tip} />
        {extra}
      </div>
      <div className="ds-fcell-ctl">{children}</div>
    </div>
  )
}

/** One line of a `.ds-switches` list: switch, then the name. */
function SwitchRow({ label, tip, value, onChange, disabled }: {
  label: string
  tip?: React.ReactNode
  value: boolean
  onChange: (v: boolean) => void
  disabled?: boolean
}) {
  return (
    <div className="ds-switchrow">
      <label className={`ds-switchrow-hit${disabled ? ' ds-is-disabled' : ''}`}>
        <span className={`ds-switch ds-sm${value ? ' ds-on' : ''}`}>
          <input
            type="checkbox"
            role="switch"
            className="sr-only"
            checked={value}
            disabled={disabled}
            onChange={(e) => onChange(e.target.checked)}
            aria-label={label}
          />
          <i />
        </span>
        <span className="ds-switchrow-name"><FieldLabel label={label} tip={tip} /></span>
      </label>
    </div>
  )
}

/**
 * The right-hand sticky section index. Tracks the currently visible section within
 * scrollContainer's viewport using an IntersectionObserver, and provides smooth-scroll on click.
 *
 * rootMargin is tuned to -20% top / -70% bottom: this concentrates the "currently
 * visible" judgment toward the upper part of the viewport, so the highlight follows
 * scrolling more naturally (the user's eye tends to sit around the top third of the viewport).
 */
function SectionIndex({
  sections,
  scrollContainer,
}: {
  sections: { id: string; labelKey: string }[]
  scrollContainer: RefObject<HTMLDivElement>
}) {
  const { t } = useTranslation()
  const [active, setActive] = useState<string>(sections[0]?.id ?? '')

  useEffect(() => {
    // Reset active to the first item after switching tabs
    setActive(sections[0]?.id ?? '')
  }, [sections])

  useEffect(() => {
    const root = scrollContainer.current
    if (!root || sections.length === 0) return
    // jsdom (vitest environment) has no IntersectionObserver; skip outright in non-browser environments.
    if (typeof IntersectionObserver === 'undefined') return
    const observers: IntersectionObserver[] = []
    // Collect (id, top) to pick the topmost currently visible section on intersect
    const visible = new Set<string>()
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) visible.add(e.target.id)
          else visible.delete(e.target.id)
        }
        // Take the first visible one in sections order as active
        const next = sections.find((s) => visible.has(s.id))
        if (next) setActive(next.id)
      },
      { root, rootMargin: '-20% 0px -70% 0px', threshold: 0 },
    )
    sections.forEach((s) => {
      const el = document.getElementById(s.id)
      if (el) obs.observe(el)
    })
    observers.push(obs)
    return () => observers.forEach((o) => o.disconnect())
  }, [sections, scrollContainer])

  const onJump = (id: string) => {
    const el = document.getElementById(id)
    if (!el) return
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setActive(id)
  }

  return (
    <nav className="ds-railnav" aria-label={t('settings.sectionsCap')}>
      {sections.map((s) => (
        <a
          key={s.id}
          href={`#${s.id}`}
          onClick={(e) => { e.preventDefault(); onJump(s.id) }}
          className={active === s.id ? 'ds-is-active' : undefined}
          aria-current={active === s.id ? 'true' : undefined}
        >
          {t(s.labelKey)}
        </a>
      ))}
    </nav>
  )
}

function SensitiveInput({ value, serverValue, onChange }: {
  value: string; serverValue: string; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const [localValue, setLocalValue] = useState(value)

  useEffect(() => {
    setLocalValue(value)
  }, [value])

  const masked = localValue === MASK

  const handleBlur = () => {
    if (localValue !== value) {
      onChange(localValue)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.currentTarget.blur()
    }
  }

  return (
    <input
      type="password"
      value={masked ? '' : localValue}
      placeholder={serverValue === MASK ? t('settings.sensitiveSavedPlaceholder') : ''}
      onChange={(e) => setLocalValue(e.target.value || MASK)}
      onBlur={handleBlur}
      onKeyDown={handleKeyDown}
      autoComplete="new-password"
      data-lpignore="true"
      data-1p-ignore
      data-form-type="other"
      className={textInputClass}
    />
  )
}


// ── HFEndpointSelect ────────────────────────────────────────────────────────
//
// HF model download endpoint selector: preset + custom URL input.
// 0.8.2 hotfix: the hf-mirror.com preset is temporarily hidden (after a server-side
// redirect change, it fails on every huggingface_hub version, see
// docs/todo/hf-mirror-recheck.md). The endpoint field itself still accepts any URL,
// so users can paste hf-mirror / sjtug / a Tencent mirror / a self-hosted proxy via
// "Custom URL". Add the preset back once it's revived.

const HF_ENDPOINT_PRESETS: { value: string; label: string; hintKey: string }[] = [
  { value: '', label: 'huggingface.co', hintKey: 'settings.hfOfficialHint' },
  { value: '__custom__', label: 'Custom URL...', hintKey: 'settings.hfCustomHint' },
]

function HFEndpointSelect({ value, onChange }: {
  value: string; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const isPreset = HF_ENDPOINT_PRESETS.some(p => p.value !== '__custom__' && p.value === value)
  const [mode, setMode] = useState<'preset' | 'custom'>(isPreset ? 'preset' : 'custom')
  const selectedPreset = isPreset
    ? value
    : (mode === 'custom' ? '__custom__' : '')

  return (
    <div className="flex flex-col gap-1.5">
      <select
        value={selectedPreset}
        onChange={(e) => {
          const v = e.target.value
          if (v === '__custom__') {
            setMode('custom')
            // Don't clear the current value, let the user type below
          } else {
            setMode('preset')
            onChange(v)
          }
        }}
        className={textInputClass}
      >
        {HF_ENDPOINT_PRESETS.map(p => (
          <option key={p.value} value={p.value}>
            {p.label}{p.hintKey ? ` — ${t(p.hintKey)}` : ''}
          </option>
        ))}
      </select>
      {mode === 'custom' && (
        <input
          type="text"
          value={value && !isPreset ? value : ''}
          placeholder="https://your-mirror.example.com"
          onChange={(e) => onChange(e.target.value.trim())}
          className={textInputClass}
        />
      )}
    </div>
  )
}

// ── DownloadSourceSelect ────────────────────────────────────────────────────

function DownloadSourceSelect({ value, onChange }: {
  value: string; onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={textInputClass}
    >
      <option value="huggingface">{t('settings.downloadSourceHuggingface')}</option>
      <option value="modelscope">{t('settings.downloadSourceModelscope')}</option>
    </select>
  )
}

// Top-level non-object fields (string / number / bool), compared directly and put into the patch.
const TOP_LEVEL_SCALARS: (keyof Secrets)[] = ['download_source']

function buildPatch(draft: Secrets, server: Secrets): SecretsPatch {
  const out: Record<string, unknown> = {}
  for (const key of Object.keys(draft) as (keyof Secrets)[]) {
    if (TOP_LEVEL_SCALARS.includes(key)) {
      if (draft[key] !== server[key]) out[key] = draft[key]
      continue
    }
    const sub: Record<string, unknown> = {}
    const d = draft[key] as unknown as Record<string, unknown>
    const s = server[key] as unknown as Record<string, unknown>
    for (const k of Object.keys(d)) {
      const dv = d[k]
      const sv = s[k]
      if (dv === MASK) continue
      if (JSON.stringify(dv) !== JSON.stringify(sv)) sub[k] = dv
    }
    if (Object.keys(sub).length) out[key] = sub
  }
  return out as SecretsPatch
}

// ── Models Section ─────────────────────────────────────────────────────────

// A local base model's "working mode" = which model family it's attached to: the
// family determines the training config's defaults (sampler / timestep / caption
// capability bits, see the capability matrix in domain/common.py) and which VAE /
// text encoder it resolves against. The domain name matches the keys in catalog.model_sources.
const FAMILY_DOMAIN_OPTIONS = [
  { value: 'anima', label: 'Anima' },
  { value: 'krea2', label: 'Krea 2' },
]

/** Absolute path -> last path segment (file/dir name), for showing "which weights got selected" in a toast. */
function basename(p: string): string {
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'))
  return i >= 0 ? p.slice(i + 1) : p
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function ModelsSection({ catalog, busy, start, reloadCatalog, catalogError, t }: {
  catalog: ModelsCatalog | null
  busy: Set<string>
  start: (model_id: string, variant?: string) => Promise<void>
  reloadCatalog: () => Promise<unknown>
  catalogError: string | null
  t: TFunction
}) {
  const { toast } = useToast()
  const [rootDraft, setRootDraft] = useState<string>('')
  const [serverRoot, setServerRoot] = useState<string | null>(null)
  const [savingRoot, setSavingRoot] = useState(false)
  const [selectedAnima, setSelectedAnima] = useState<string>('1.0')
  const [selectedKrea2, setSelectedKrea2] = useState<string>('raw')
  const [selectedKrea2Te, setSelectedKrea2Te] = useState<string>('bf16')
  // VAE is a shared asset independent of family -> a single selected value (''=the
  // official location); Anima's text encoder has only one official directory, so its
  // selected value is likewise '' (official) or a local directory's absolute path.
  const [selectedVae, setSelectedVae] = useState<string>('')
  const [selectedAnimaTe, setSelectedAnimaTe] = useState<string>('')
  const [autoSyncPaths, setAutoSyncPaths] = useState<boolean>(true)
  const [savingAutoSync, setSavingAutoSync] = useState(false)
  const [secretsLoaded, setSecretsLoaded] = useState(false)

  // One-time fetch of secrets to get models.root + selected (per family) +
  // auto_sync_paths (these go through their own PUT, not the SettingsPage's global
  // dirty flow). The catalog is injected by the parent.
  useEffect(() => {
    void api.getSecrets().then((sec) => {
      setServerRoot(sec.models?.root ?? null)
      setSelectedAnima(sec.models?.selected?.anima ?? sec.models?.selected_anima ?? '1.0')
      setSelectedKrea2(sec.models?.selected?.krea2 ?? 'raw')
      setSelectedKrea2Te(sec.models?.selected_te?.krea2 ?? 'bf16')
      setSelectedVae(sec.models?.selected_vae ?? '')
      setSelectedAnimaTe(sec.models?.selected_te?.anima ?? '')
      setAutoSyncPaths(sec.models?.auto_sync_paths ?? true)
      setSecretsLoaded(true)
    }).catch(() => { setSecretsLoaded(true) })
  }, [])

  // Once both secrets + catalog are in, prefill the input with the "saved value" or
  // the "actual default absolute path". Use prev !== '' as the "already
  // initialized / user has edited it" flag, to avoid overwriting user input.
  useEffect(() => {
    if (!secretsLoaded || !catalog) return
    setRootDraft((prev) => (prev !== '' ? prev : (serverRoot ?? catalog.models_root ?? '')))
  }, [secretsLoaded, catalog, serverRoot])

  const pickAnima = async (variant: string) => {
    if (variant === selectedAnima) return
    setSelectedAnima(variant)
    try {
      await api.updateSecrets({ models: { selected_anima: variant, selected: { anima: variant } } })
      toast(t('settings.mainModelSelected', { name: variant }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
      void reloadCatalog()
    }
  }

  const pickKrea2 = async (variant: string) => {
    if (variant === selectedKrea2) return
    setSelectedKrea2(variant)
    try {
      await api.updateSecrets({ models: { selected: { krea2: variant } } })
      toast(t('settings.mainModelSelected', { name: variant }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
      void reloadCatalog()
    }
  }

  const pickKrea2Te = async (variant: string) => {
    if (variant === selectedKrea2Te) return
    setSelectedKrea2Te(variant)
    try {
      await api.updateSecrets({ models: { selected_te: { krea2: variant } } })
      toast(t('settings.teSelected', { name: variant }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
      void reloadCatalog()
    }
  }

  // Writing back the VAE / text encoder selection: same shape as pickAnima /
  // pickKrea2Te (optimistic update + rollback on failure = refetch catalog + secrets).
  const pickVae = async (value: string) => {
    if (value === selectedVae) return
    const prev = selectedVae
    setSelectedVae(value)
    try {
      await api.updateSecrets({ models: { selected_vae: value } })
      toast(t('settings.vaeSelected', {
        name: value ? basename(value) : t('settings.officialWeights'),
      }), 'success')
      await reloadCatalog()
    } catch (e) {
      setSelectedVae(prev)
      toast(String(e), 'error')
      void reloadCatalog()
    }
  }

  const pickAnimaTe = async (value: string) => {
    if (value === selectedAnimaTe) return
    const prev = selectedAnimaTe
    setSelectedAnimaTe(value)
    try {
      await api.updateSecrets({ models: { selected_te: { anima: value } } })
      toast(t('settings.teSelected', {
        name: value ? basename(value) : t('settings.officialWeights'),
      }), 'success')
      await reloadCatalog()
    } catch (e) {
      setSelectedAnimaTe(prev)
      toast(String(e), 'error')
      void reloadCatalog()
    }
  }

  // After registering/unregistering a local candidate or changing its mode: both
  // catalog and secrets may change (the server falls back an invalidated selection to
  // the default), so both need refetching together, or the UI would look stale.
  const reloadSources = async () => {
    await reloadCatalog()
    try {
      const sec = await api.getSecrets()
      setSelectedAnima(sec.models?.selected?.anima ?? sec.models?.selected_anima ?? '1.0')
      setSelectedKrea2(sec.models?.selected?.krea2 ?? 'raw')
      setSelectedKrea2Te(sec.models?.selected_te?.krea2 ?? 'bf16')
      setSelectedVae(sec.models?.selected_vae ?? '')
      setSelectedAnimaTe(sec.models?.selected_te?.anima ?? '')
    } catch {
      // A failed secrets read shouldn't make a just-succeeded registration look like
      // a failure: the catalog has already refreshed, and it'll reconcile again next time the page loads.
    }
  }

  const sourceRows = (domain: string) => catalog?.model_sources?.[domain] ?? []

  // After changing a local base model's "working mode", reselect it in the new
  // family (LocalModelRows only knows about domain; the entry point that writes back
  // the selection dispatches per family).
  const selectMainInDomain = async (domain: string, value: string) => {
    if (domain === 'krea2') await pickKrea2(value)
    else await pickAnima(value)
  }

  const saveRoot = async () => {
    const v = rootDraft.trim()
    setSavingRoot(true)
    try {
      await api.updateSecrets({ models: { root: v ? v : null } })
      toast(v ? t('settings.modelRootSaved', { path: v }) : t('settings.modelRootDefault'), 'success')
      setServerRoot(v ? v : null)
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setSavingRoot(false)
    }
  }

  const saveAutoSync = async (next: boolean) => {
    setSavingAutoSync(true)
    const prev = autoSyncPaths
    setAutoSyncPaths(next)
    try {
      await api.updateSecrets({ models: { auto_sync_paths: next } })
      toast(next ? t('settings.autoSyncPathsOn') : t('settings.autoSyncPathsOff'), 'success')
    } catch (e) {
      setAutoSyncPaths(prev)
      toast(String(e), 'error')
    } finally {
      setSavingAutoSync(false)
    }
  }

  const rootDirty = rootDraft.trim() !== (serverRoot ?? catalog?.models_root ?? '')
  const error = catalogError

  return (
    <>
      <SettingsSection id="models" title={t('settings.trainingModels')}>
        <div className="ds-fgrid">
          <SetCell label={t('settings.modelsRoot')} full>
            <div className="ds-set-inline">
              <input
                type="text"
                value={rootDraft}
                onChange={(e) => setRootDraft(e.target.value)}
                className={`${textInputClass} ds-mono`}
              />
              <button onClick={saveRoot} disabled={!rootDirty || savingRoot} className="ds-ctl"
                title={rootDirty ? t('settings.savePathConfig') : t('settings.notModified')}>
                {savingRoot ? t('common.saving') : t('common.save')}
              </button>
              {rootDirty && (
                <button onClick={() => setRootDraft(serverRoot ?? (catalog?.models_root ?? ''))} disabled={savingRoot}
                  className="ds-iconbtn"
                  aria-label={t('common.reset')}
                  title={t('common.reset')}
                >↺</button>
              )}
            </div>
          </SetCell>
        </div>
        <div className="ds-switches ds-after">
          <SwitchRow
            label={t('settings.autoSyncPathsLabel')}
            tip={t('settings.autoSyncPathsHelp')}
            value={autoSyncPaths}
            disabled={savingAutoSync}
            onChange={(v) => void saveAutoSync(v)}
          />
        </div>
      </SettingsSection>

      {error && <div className="ds-note ds-err ds-mono">{error}</div>}
      {!catalog ? (
        <div className="ds-set-empty">{t('settings.loadingModelCatalog')}</div>
      ) : (
        <>
          {/* Anima base model */}
          <ModelGroupCard
            title={translatedCatalogText(MODEL_NAME_KEYS, 'anima_main', catalog.anima_main.name, t)}
            helpTooltip={
              <>
                <p><Trans i18nKey="settings.repoHelp" values={{ desc: translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'anima_main', catalog.anima_main.description, t), repo: catalog.anima_main.repo }} components={{ code: <code /> }} /></p>
                <p><Trans i18nKey="settings.defaultTransformerHelp" components={{ strong: <strong /> }} /></p>
              </>
            }
            add={<AddLocalModelButton domain="anima" shape="file" initialPath={catalog.models_root} onChanged={reloadSources} />}
          >
            {catalog.anima_main.variants.map((v) => {
              const key = `anima_main:${v.variant}`
              const dl = catalog.downloads[key]
              const isSel = v.variant === selectedAnima
              const canSelect = v.exists && dl?.status !== 'running'
              return (
                <li key={v.variant} className={`model-row ds-model-row${isSel ? ' ds-is-on' : ''}`}>
                  <input type="radio" name="anima_variant" checked={isSel} disabled={!canSelect}
                    onChange={() => void pickAnima(v.variant)}
                    title={canSelect ? t('settings.selectDefaultMainModel') : v.exists ? t('settings.downloadInProgress') : t('settings.downloadRequiredFirst')}
                  />
                  <code className="ds-model-name" title={v.variant}>{v.variant}</code>
                  <ModelStatusBadge exists={v.exists} size={v.size} status={dl?.status} />
                  {v.kind === 'custom'
                    ? <span className="ds-badge ds-mute" title={v.target_path}>{t('settings.localBaseModel')}</span>
                    : <DownloadButton exists={v.exists} status={dl?.status} busy={busy.has(key)} onClick={() => void start('anima_main', v.variant)} />}
                </li>
              )
            })}
            {/* Own weights: a local .safetensors, in the same radio group as the official variants */}
            <LocalModelRows
              domain="anima"
              rows={sourceRows('anima')}
              radioName="anima_variant"
              onSelect={(value) => void pickAnima(value)}
              onChanged={reloadSources}
              familyOptions={FAMILY_DOMAIN_OPTIONS}
              selectInDomain={selectMainInDomain}
            />
          </ModelGroupCard>

          {/* VAE (shared by both families, can be swapped for your own local weights) */}
          <ModelGroupCard
            title={catalog.anima_vae.name}
            helpTooltip={<>
              <p>{translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'anima_vae', catalog.anima_vae.description, t)}</p>
              <p>{t('settings.vaeHelp')}</p>
            </>}
            add={<AddLocalModelButton domain="vae" shape="file" initialPath={catalog.models_root} onChanged={reloadSources} />}
          >
            <li className={`model-row ds-model-row${selectedVae === '' ? ' ds-is-on' : ''}`}>
              <input type="radio" name="vae_source" checked={selectedVae === ''}
                onChange={() => void pickVae('')}
                title={t('settings.selectDefaultVae')}
              />
              <code className="ds-model-name" title={catalog.anima_vae.repo}>{catalog.anima_vae.repo}</code>
              <ModelStatusBadge exists={catalog.anima_vae.exists} size={catalog.anima_vae.size} status={catalog.downloads.anima_vae?.status} />
              <DownloadButton exists={catalog.anima_vae.exists} status={catalog.downloads.anima_vae?.status} busy={busy.has('anima_vae')} onClick={() => void start('anima_vae')} />
            </li>
            <LocalModelRows
              domain="vae"
              rows={sourceRows('vae')}
              radioName="vae_source"
              onSelect={(value) => void pickVae(value)}
              onChanged={reloadSources}
            />
          </ModelGroupCard>

          {/* Anima text encoder: the official Qwen3 directory + a user-registered local encoder (radio) */}
          <ModelGroupCard
            title={catalog.qwen3.name}
            helpTooltip={<>
              <p>{translatedCatalogText(MODEL_DESCRIPTION_KEYS, 'qwen3', catalog.qwen3.description, t)}</p>
              <p>{t('settings.animaTeHelp')}</p>
            </>}
            add={<AddLocalModelButton domain="anima_te" shape="dir" initialPath={catalog.qwen3.target_dir} onChanged={reloadSources} />}
          >
            {(() => {
              const m = catalog.qwen3
              const dl = catalog.downloads.qwen3
              const allExist = m.files.every((f) => f.exists)
              const totalSize = m.files.reduce((sum, f) => sum + f.size, 0)
              return (
                <li className={`model-row ds-model-row${selectedAnimaTe === '' ? ' ds-is-on' : ''}`}>
                  <input type="radio" name="anima_te" checked={selectedAnimaTe === ''}
                    onChange={() => void pickAnimaTe('')}
                    title={t('settings.selectDefaultTe')}
                  />
                  <code className="ds-model-name" title={m.repo}>{m.repo}</code>
                  <ModelStatusBadge exists={allExist} size={totalSize} status={dl?.status} fileCount={m.files.length} existsCount={m.files.filter((f) => f.exists).length} />
                  <DownloadButton exists={allExist} status={dl?.status} busy={busy.has('qwen3')} onClick={() => void start('qwen3')} />
                </li>
              )
            })()}
            <LocalModelRows
              domain="anima_te"
              rows={sourceRows('anima_te')}
              radioName="anima_te"
              onSelect={(value) => void pickAnimaTe(value)}
              onChanged={reloadSources}
            />
          </ModelGroupCard>

          {/* T5 tokenizer (Anima-specific, no custom entry point -- it's just a tokenizer file) */}
          {(['t5_tokenizer'] as const).map((id) => {
            const m = catalog[id]
            const dl = catalog.downloads[id]
            const allExist = m.files.every((f) => f.exists)
            const totalSize = m.files.reduce((s, f) => s + f.size, 0)
            return (
              <ModelGroupCard key={id} title={m.name} helpTooltip={translatedCatalogText(MODEL_DESCRIPTION_KEYS, id, m.description, t)}>
                <li className="ds-model-row">
                  <code className="ds-model-name" title={m.repo}>{m.repo}</code>
                  <ModelStatusBadge exists={allExist} size={totalSize} status={dl?.status} fileCount={m.files.length} existsCount={m.files.filter((f) => f.exists).length} />
                  <DownloadButton exists={allExist} status={dl?.status} busy={busy.has(id)} onClick={() => void start(id)} />
                </li>
              </ModelGroupCard>
            )
          })}

          {/* Krea 2 base model (the second model family; shares qwen_image_vae with Anima for VAE) */}
          {catalog.krea2_main && (
            <ModelGroupCard
              title={translatedCatalogText(MODEL_NAME_KEYS, 'krea2_main', catalog.krea2_main.name, t)}
              helpTooltip={t('settings.krea2MainHelp')}
              add={<AddLocalModelButton domain="krea2" shape="file" initialPath={catalog.models_root} onChanged={reloadSources} />}
            >
              {catalog.krea2_main.variants.map((v) => {
                const key = `krea2_main:${v.variant}`
                const dl = catalog.downloads[key]
                const isSel = v.variant === selectedKrea2
                const canSelect = v.exists && dl?.status !== 'running'
                return (
                  <li key={v.variant} className={`model-row ds-model-row${isSel ? ' ds-is-on' : ''}`}>
                    <input type="radio" name="krea2_variant" checked={isSel} disabled={!canSelect}
                      onChange={() => void pickKrea2(v.variant)}
                      title={canSelect ? t('settings.selectDefaultMainModel') : v.exists ? t('settings.downloadInProgress') : t('settings.downloadRequiredFirst')}
                    />
                    <code className="ds-model-name" title={v.variant}>{v.variant}</code>
                    {v.purpose && (
                      <span className="ds-badge ds-mute">
                        {v.purpose === 'training' ? t('settings.purposeTraining') : t('settings.purposeInference')}
                      </span>
                    )}
                    <ModelStatusBadge exists={v.exists} size={v.size} status={dl?.status} />
                    <DownloadButton exists={v.exists} status={dl?.status} busy={busy.has(key)} onClick={() => void start('krea2_main', v.variant)} />
                  </li>
                )
              })}
              <LocalModelRows
                domain="krea2"
                rows={sourceRows('krea2')}
                radioName="krea2_variant"
                onSelect={(value) => void pickKrea2(value)}
                onChanged={reloadSources}
                familyOptions={FAMILY_DOMAIN_OPTIONS}
                selectInDomain={selectMainInDomain}
              />
            </ModelGroupCard>
          )}

          {/* Krea 2 text encoder Qwen3-VL: bf16 directory version + the official fp8 single-file version (radio) */}
          {catalog.krea2_text_encoder && (
            <ModelGroupCard
              title={catalog.krea2_text_encoder.name}
              helpTooltip={t('settings.krea2TeHelp')}
              add={<AddLocalModelButton domain="krea2_te" shape="dir" initialPath={catalog.krea2_text_encoder?.target_dir ?? catalog.models_root} onChanged={reloadSources} />}
            >
              {([['bf16', catalog.krea2_text_encoder], ['fp8', catalog.krea2_text_encoder_fp8]] as const).map(([teKey, m]) => {
                if (!m) return null
                const dlKey = teKey === 'bf16' ? 'krea2_text_encoder' : 'krea2_text_encoder_fp8'
                const dl = catalog.downloads[dlKey]
                const allExist = m.files.every((f) => f.exists)
                const totalSize = m.files.reduce((s, f) => s + f.size, 0)
                const isSel = teKey === selectedKrea2Te
                const canSelect = allExist && dl?.status !== 'running'
                return (
                  <li key={teKey} className={`model-row ds-model-row${isSel ? ' ds-is-on' : ''}`}>
                    <input type="radio" name="krea2_te" checked={isSel} disabled={!canSelect}
                      onChange={() => void pickKrea2Te(teKey)}
                      title={canSelect ? t('settings.selectDefaultTe') : allExist ? t('settings.downloadInProgress') : t('settings.downloadRequiredFirst')}
                    />
                    <code className="ds-model-name">{teKey}</code>
                    <ModelStatusBadge exists={allExist} size={totalSize} status={dl?.status} fileCount={m.files.length} existsCount={m.files.filter((f) => f.exists).length} />
                    <DownloadButton exists={allExist} status={dl?.status} busy={busy.has(dlKey)} onClick={() => void start(dlKey)} />
                  </li>
                )
              })}
              {/* Custom text encoder: a local transformers directory (containing config.json) */}
              <LocalModelRows
                domain="krea2_te"
                rows={sourceRows('krea2_te')}
                radioName="krea2_te"
                onSelect={(value) => void pickKrea2Te(value)}
                onChanged={reloadSources}
              />
            </ModelGroupCard>
          )}

          {/* Download log */}
          {Object.values(catalog.downloads).filter((d) => d.status === 'running' || d.status === 'failed').length > 0 && (
            <SettingsSection title={t('settings.downloadLogs', { n: Object.values(catalog.downloads).filter((d) => d.status === 'running' || d.status === 'failed').length })}>
              <div className="ds-set-pad">
                {Object.values(catalog.downloads).map((d) => (
                  <div key={d.key} className="ds-set-box">
                    <div className="ds-set-inline">
                      <code className="ds-mono">{d.key}</code>
                      <ModelStatusBadge exists={d.status === 'done'} size={0} status={d.status} />
                      {d.message && <span className="ds-set-errline">{d.message}</span>}
                    </div>
                    <pre className="ds-set-log">{d.log_tail.join('\n') || t('settings.emptyLog')}</pre>
                  </div>
                ))}
              </div>
            </SettingsSection>
          )}
        </>
      )}
    </>
  )
}

function ModelGroupCard({
  title, helpTooltip, add, children,
}: {
  title: string
  helpTooltip?: React.ReactNode
  /** "Pick a file / folder" entry for own weights, under the list. */
  add?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <SettingsSection title={title} tip={helpTooltip}>
      <div className="ds-set-pad">
        <ul className="ds-set-list">{children}</ul>
        {add}
      </div>
    </SettingsSection>
  )
}

function ModelStatusBadge({ exists, size, status, fileCount, existsCount }: {
  exists: boolean; size: number; status?: ModelDownloadStatus['status']; fileCount?: number; existsCount?: number
}) {
  const { t } = useTranslation()
  if (status === 'running') {
    return <StatusLabel bg="bg-warn-soft" fg="text-warn" text={t('settings.downloadInProgress')} pulse />
  }
  if (status === 'failed') {
    return <StatusLabel bg="bg-err-soft" fg="text-err" text={t('status.failed')} />
  }
  if (exists) {
    return <StatusLabel bg="bg-ok-soft" fg="text-ok" text={`✓ ${fmtBytes(size)}${fileCount !== undefined ? ` (${existsCount}/${fileCount})` : ''}`} />
  }
  if (fileCount !== undefined && existsCount! > 0) {
    return <StatusLabel bg="bg-warn-soft" fg="text-warn" text={t('settings.partialFiles', { exists: existsCount, total: fileCount })} />
  }
  return <StatusLabel bg="bg-overlay" fg="text-fg-tertiary" text={t('settings.notDownloaded')} />
}

function StatusLabel({ fg, text, pulse }: { bg: string; fg: string; text: string; pulse?: boolean }) {
  const tone = fg === 'text-ok' ? 'ds-ok' : fg === 'text-err' ? 'ds-err' : fg === 'text-warn' ? 'ds-warn' : 'ds-mute'
  return (
    <span className={`ds-badge ds-mono ${tone}`}
      style={pulse ? { animation: 'pulse 1.5s infinite' } : undefined}
    >{text}</span>
  )
}

/** Inline "name value" facts about an installed package or the environment. */
function KV({ items }: { items: [string, string | null | undefined][] }) {
  return (
    <div className="ds-set-kv">
      {items.filter(([, v]) => v).map(([k, v]) => (
        <span key={k}><span className="ds-set-kv-k">{k}</span><code>{v}</code></span>
      ))}
    </div>
  )
}

function RefreshButton({ onClick, disabled }: { onClick: () => void; disabled?: boolean }) {
  const { t } = useTranslation()
  return (
    <button type="button" onClick={onClick} disabled={disabled} className="ds-iconbtn"
      title={t('settings.refreshStatus')} aria-label={t('settings.refreshStatus')}>
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7" /></svg>
    </button>
  )
}

function DownloadButton({ exists, status, busy, onClick }: {
  exists: boolean; status?: ModelDownloadStatus['status']; busy: boolean; onClick: () => void
}) {
  const { t } = useTranslation()
  const running = status === 'running' || busy
  if (running) {
    return <button disabled className="ds-ctl ds-ghost ds-sm">…</button>
  }
  return (
    <button onClick={onClick} className={exists ? 'ds-ctl ds-ghost ds-sm' : 'ds-btn-primary ds-sm'}
      title={exists ? t('settings.redownloadTitle') : t('common.download')}>
      {exists ? t('settings.redownload') : t('settings.downloadAction')}
    </button>
  )
}

// ── PyTorch Section (training tab) ─────────────────────────────────────────
//
// The "one-click fix" entry point for users with an existing venv. On startup PR-4
// warns "GPU detected but torch is the CPU build" and gives a pip command; this
// turns that command into UI, so regular users don't need to open a terminal.
//
// Three states:
// - cuda_available=True               -> checkmark, everything OK (collapsed by default; offers a "switch CUDA version" advanced option)
// - is_cpu_with_gpu=True               -> red mis-installed warning + a prominent "reinstall as CUDA" primary button
// - is_cuda_build_unavailable=True     -> yellow driver warning (pip can't fix it, links to docs)

function PyTorchSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [status, setStatus] = useState<TorchStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const s = await api.getTorchStatus()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const reinstall = async (target: 'auto' | TorchCuTag) => {
    const tag = target === 'auto' ? status?.recommended_cu_tag ?? '?' : target
    // Register -> user Ctrl+C restarts -> the launcher process runs pip. On Windows,
    // torch.pyd is locked by the current server process, so it can't be replaced
    // directly; it has to be deferred to the launcher.
    if (!(await dialog.confirm(
      t('settings.confirmRegisterTorch', { tag }),
      { tone: 'warn', okText: t('settings.registerRequest') },
    ))) return
    setBusy(true)
    try {
      const result = await api.reinstallTorch(target)
      // The backend already wrote the marker; the server process hasn't actually installed it, prompt the user to restart
      toast(result.message, 'success')
    } catch (e) {
      toast(t('settings.registerFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const hasIssue = !!error || (status && (status.is_cpu_with_gpu || status.is_cuda_build_unavailable || !status.installed))
  const statusOk = status?.cuda_available && !error
  const statusLabel = error
    ? t('settings.loadFailedShort')
    : !status
      ? t('settings.loadingStatus')
      : !status.installed
        ? t('settings.notInstalledShort')
        : status.is_cpu_with_gpu
          ? t('settings.cpuBuildMisinstalled')
          : !status.cuda_available && status.cuda_build !== 'cpu'
            ? t('settings.cudaUnavailableDriver')
            : status.cuda_available
              ? `CUDA ✓ ${status.cuda_build}`
              : `CPU ${status.cuda_build}`

  return (
    <CollapsibleSection
      id="pytorch"
      title="PyTorch"
      tip={t('settings.trainingCoreDependency')}
      open={!!hasIssue}
      extra={<span className={`ds-badge ds-mono ${statusOk ? 'ds-ok' : status?.is_cpu_with_gpu ? 'ds-err' : 'ds-warn'}`}>{statusLabel}</span>}
    >
      <div className="ds-set-pad">
        {error && <div className="ds-note ds-err ds-mono">{error}</div>}
        {!error && !status && <div className="ds-set-empty">{t('settings.loadingStatus')}</div>}

        {status && (<>
          <KV items={[
            ['torch', status.version ?? t('settings.notInstalledParen')],
            ['build', status.cuda_build],
            ['GPU', status.cuda_available ? status.device_name : null],
            [t('settings.driverLabel'), status.cuda_detect.driver_version ?? t('settings.notDetected')],
            [t('settings.systemGpu'), !status.cuda_available ? status.cuda_detect.gpu_name : null],
          ]} />

          {/* Mis-installed: CPU torch + a GPU present */}
          {status.is_cpu_with_gpu && (
            <div className="ds-note ds-err">
              <span>
                <Trans
                  i18nKey="settings.torchCpuWithGpuWarning"
                  values={{ tag: status.recommended_cu_tag }}
                  components={{ code: <code className="font-mono" /> }}
                />
              </span>
            </div>
          )}

          {/* CUDA build present but unusable at runtime: driver / WSL issue */}
          {status.is_cuda_build_unavailable && (
            <div className="ds-note ds-warn">
              <span>
                <Trans
                  i18nKey="settings.torchCudaUnavailableWarning"
                  components={{ code: <code className="font-mono" /> }}
                />
              </span>
            </div>
          )}

          <div className="ds-set-inline">
            <button
              onClick={() => void reinstall('auto')}
              disabled={busy || !status.cuda_detect.available}
              className={status.is_cpu_with_gpu ? 'ds-btn-primary ds-sm' : 'ds-ctl ds-sm'}
              title={status.cuda_detect.available
                ? t('settings.autoSelect', { tag: status.recommended_cu_tag })
                : t('settings.noNvidiaDriverCannotCuda')}
            >
              {busy ? t('settings.installing') : status.is_cpu_with_gpu
                ? t('settings.reinstallCudaBuild', { tag: status.recommended_cu_tag })
                : t('settings.reinstallAuto', { tag: status.recommended_cu_tag })}
            </button>
            <RefreshButton onClick={() => void refresh()} disabled={busy} />
            <button type="button" onClick={() => setAdvancedOpen(!advancedOpen)}
              className="ds-ctl ds-ghost ds-sm" style={{ marginLeft: 'auto' }}>
              {t('settings.advancedManualCuda')}
            </button>
          </div>

          {/* Manually pick a version */}
          {advancedOpen && (
            <div className="ds-set-inline ds-set-sub">
              {(['cu128', 'cu126', 'cu124', 'cu118', 'cpu'] as const).map((tag) => (
                <button
                  key={tag}
                  onClick={() => void reinstall(tag)}
                  disabled={busy}
                  className={`ds-ctl ds-sm ds-mono${status.cuda_build === tag ? ' ds-is-on' : ''}`}
                  title={
                    tag === 'cpu'
                      ? t('settings.installCpuBuildHint')
                      : t('settings.installCudaBuildHint', { tag })
                  }
                >
                  {tag}
                </button>
              ))}
              <HelpTip>{t('settings.manualCudaHint')}</HelpTip>
            </div>
          )}
        </>)}
      </div>
    </CollapsibleSection>
  )
}

// ── Flash Attention Section (training tab) ─────────────────────────────────
//
// An optional training speed-up. Once flash_attn is installed, startup automatically
// calls set_flash_attn_enabled(True). This component gives the UI a one-click way to
// install a wheel, reusing PR-7a's service: status + GitHub candidates + install.
//
// Design notes:
// - install is a synchronous pip call (a few minutes), guarded with confirm() + a busy state to prevent double-clicks
// - A wheel with a mismatched Python ABI (usable=false) is grayed out, but keeps a
//   "force install" button (in rare cases the user may be running within the ABI-compatible subset)
// - When the GitHub API is rate-limited, candidates=[] + fetch_error, falling back to a manual URL input

function FlashAttentionSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [status, setStatus] = useState<FlashAttnStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [candidatesOpen, setCandidatesOpen] = useState(false)
  const [manualUrl, setManualUrl] = useState('')
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const s = await api.getFlashAttnStatus()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const install = async (url: string | null) => {
    const msg = url ? t('settings.confirmInstallFlashUrl') : t('settings.confirmInstallFlashAuto')
    if (!(await dialog.confirm(msg, { tone: 'warn', okText: t('settings.startInstall') }))) return
    setBusy(true)
    try {
      const result = await api.installFlashAttn(url)
      toast(t('settings.flashAttnInstalled', { version: result.version ?? '?' }), 'success')
      await refresh()
    } catch (e) {
      toast(t('settings.installFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const env = status?.env
  const candidates = status?.candidates ?? []
  const fetchError = status?.fetch_error ?? null
  const usable = candidates.filter((c) => c.usable)
  const bestCandidate = usable[0] ?? null
  const hasIssue = !!error || (status && !status.installed)
  const canAutoInstall = !!env?.torch_tag && !!env?.platform && usable.length > 0

  const statusLabel = error
    ? t('settings.loadFailedShort')
    : !status
      ? t('settings.loadingStatus')
      : status.installed
        ? t('settings.installedVersion', { version: status.version ?? '?' })
        : t('settings.notInstalledShort')
  const statusOk = status?.installed && !error

  return (
    <CollapsibleSection
      id="flash-attn"
      title="Flash Attention"
      tip={t('settings.trainingAccelerationOptional')}
      open={!!hasIssue}
      extra={<span className={`ds-badge ds-mono ${statusOk ? 'ds-ok' : 'ds-warn'}`}>{statusLabel}</span>}
    >
      <div className="ds-set-pad">
        {error && <div className="ds-note ds-err ds-mono">{error}</div>}
        {!error && !status && <div className="ds-set-empty">{t('settings.loadingStatus')}</div>}

        {status && env && (<>
          <KV items={[
            ['flash_attn', status.installed ? `v${status.version ?? '?'}` : t('settings.notInstalledParen')],
            ['Python', env.python_tag],
            ['CUDA', env.cuda_tag ?? t('settings.notDetected')],
            ['PyTorch', env.torch_tag ?? t('settings.notDetected')],
            [t('settings.platform'), env.platform ?? t('settings.unsupported')],
          ]} />

          {/* GitHub API failure */}
          {fetchError && (
            <div className="ds-note ds-err">
              <span>
                {t('settings.githubApiFailed')}
                <code className="block mt-0.5 break-all">{fetchError}</code>
              </span>
            </div>
          )}

          {/* No matching wheel */}
          {!canAutoInstall && !fetchError && env.platform && env.torch_tag && (
            <div className="ds-note ds-warn">{t('settings.noWheelForPython', { python: env.python_tag })}</div>
          )}

          <div className="ds-set-inline">
            <button
              onClick={() => void install(null)}
              disabled={busy || !canAutoInstall}
              className="ds-ctl ds-sm"
              title={canAutoInstall
                ? t('settings.autoSelect', { tag: bestCandidate?.name ?? '' })
                : t('settings.noWheelManual')}
            >
              {busy ? t('settings.installing') : status.installed ? t('settings.reinstallAutoMatch') : t('settings.autoMatchInstall')}
            </button>
            <RefreshButton onClick={() => void refresh()} disabled={busy} />
            <button type="button" onClick={() => setCandidatesOpen(!candidatesOpen)}
              className="ds-ctl ds-ghost ds-sm" style={{ marginLeft: 'auto' }}>
              {t('settings.candidateWheels', { n: usable.length })}
            </button>
          </div>

          {/* Candidate list + manual URL */}
          {candidatesOpen && (
            <div className="ds-set-sub">
              {candidates.length === 0 ? (
                <div className="ds-set-empty">{t('settings.wheelQueryFailed')}</div>
              ) : (
                <ul className="ds-set-list">
                  {candidates.map((c) => (
                    <li key={c.url} className="ds-model-row" style={c.usable ? undefined : { opacity: 0.55 }}>
                      <span className="ds-model-name" style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <code className="ds-mono" style={{ fontSize: 11, wordBreak: 'break-all' }}>{c.name}</code>
                        {c.notes.map((n, i) => (
                          <span key={i} className="ds-set-warnline">{n}</span>
                        ))}
                      </span>
                      <button
                        onClick={() => void install(c.url)}
                        disabled={busy}
                        className={c.usable ? 'ds-btn-primary ds-sm' : 'ds-ctl ds-sm'}
                        title={c.usable ? t('settings.installWheel') : t('settings.wheelAbiIncompatible')}
                      >
                        {c.usable ? t('settings.installAction') : t('settings.forceInstall')}
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              <div className="ds-fcell">
                <div className="ds-fcell-head"><span className="ds-label">{t('settings.manualUrl')}</span></div>
                <div className="ds-set-inline">
                  <input
                    type="text"
                    value={manualUrl}
                    onChange={(e) => setManualUrl(e.target.value)}
                    placeholder="https://github.com/.../flash_attn-...whl"
                    className={`${textInputClass} ds-mono`}
                  />
                  <button
                    onClick={() => { if (manualUrl.trim()) void install(manualUrl.trim()) }}
                    disabled={busy || !manualUrl.trim()}
                    className="ds-ctl"
                  >{t('settings.install')}</button>
                </div>
              </div>
            </div>
          )}
        </>)}
      </div>
    </CollapsibleSection>
  )
}

// ── xformers Section (training tab) ────────────────────────────────────────
//
// A simplified attention speed-up (an alternative to flash_attn). xformers installs
// directly from PyPI, no GitHub candidate wheel list like flash_attn needs. On failure, stderr is shown so the user can debug it.

function XformersSection() {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [status, setStatus] = useState<XformersStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const { toast } = useToast()

  const refresh = useCallback(async () => {
    try {
      const s = await api.getXformersStatus()
      setStatus(s)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const install = async () => {
    if (
      !(await dialog.confirm(
        t('settings.confirmInstallXformers'),
        { tone: 'warn', okText: t('settings.startInstall') },
      ))
    ) return
    setBusy(true)
    try {
      const r = await api.installXformers()
      toast(t('settings.xformersInstalled', { version: r.version ?? '?' }), 'success')
      await refresh()
    } catch (e) {
      toast(t('settings.installFailed', { error: String(e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const statusLabel = error
    ? t('settings.loadFailedShort')
    : !status
      ? t('settings.loadingStatus')
      : status.installed
        ? t('settings.installedVersion', { version: status.version ?? '?' })
        : t('settings.notInstalledShort')
  const statusOk = status?.installed && !error
  const hasIssue = !!error

  return (
    <CollapsibleSection
      id="xformers"
      title="xformers"
      tip={<>
        <p>{t('settings.xformersSubtitle')}</p>
        <p><Trans i18nKey="settings.xformersHelp1" components={{ strong: <strong />, code: <code /> }} /></p>
        <p>{t('settings.xformersHelp2')}</p>
        <p>{t('settings.xformersHelp3')}</p>
      </>}
      open={!!hasIssue}
      extra={<span className={`ds-badge ds-mono ${statusOk ? 'ds-ok' : 'ds-warn'}`}>{statusLabel}</span>}
    >
      <div className="ds-set-pad">
        {error && <div className="ds-note ds-err ds-mono">{error}</div>}
        {!error && !status && <div className="ds-set-empty">{t('settings.loadingStatus')}</div>}

        {status && (<>
          <KV items={[['xformers', status.installed ? `v${status.version ?? '?'}` : t('settings.notInstalledParen')]]} />
          <div className="ds-set-inline">
            <button
              onClick={() => void install()}
              disabled={busy}
              className="ds-ctl ds-sm"
            >
              {busy
                ? t('settings.installing')
                : status.installed
                  ? t('settings.reinstallAutoMatchPlain')
                  : t('settings.installAutoMatchPlain')}
            </button>
            <RefreshButton onClick={() => void refresh()} disabled={busy} />
          </div>
        </>)}
      </div>
    </CollapsibleSection>
  )
}
