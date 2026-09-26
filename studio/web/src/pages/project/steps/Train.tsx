import type { TFunction } from 'i18next'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate, useOutletContext } from 'react-router-dom'
import {
  api,
  type ApiError,
  type BucketDistribution,
  type ConfigData,
  type PresetSummary,
  type ProjectDetail,
  type RegStatus,
  type SchemaResponse,
  type SystemStats as SystemStatsData,
  type Task,
  type TrainEstimate,
  type Version,
  type VersionConfigResponse,
} from '../../../api/client'
import { parseFolderMeta } from '../../../lib/folderMeta'
import { useLocalStorageState } from '../../../lib/useLocalStorageState'
import ConfigSkeleton from '../../../components/ConfigSkeleton'
import ConfigYamlPanel from '../../../components/ConfigYamlPanel'
import { useDialog } from '../../../components/Dialog'
import SchemaForm, { sameValue } from '../../../components/SchemaForm'
import StepShell from '../../../components/StepShell'
import JobLogBar from '../../../components/ds/JobLogBar'
import KebabMenu from '../../../components/ds/KebabMenu'
import { evalShowWhen, schemaFieldLabel, schemaGroupLabel } from '../../../lib/schema'
import { useEventStream } from '../../../lib/useEventStream'
import { useMonitorProgress } from '../../../lib/useMonitorProgress'
import type { SaveStatus } from '../../../lib/SettingsData'
import { useToast } from '../../../components/Toast'
import { useSettingsDrawer } from '../../../lib/SettingsDrawer'
import {
  PRESET_NAME_RE,
  defaultsFromSchema,
  generateUniquePresetName,
} from '../../../lib/preset-helpers'
import FamilySwitchDialog from '../../../components/FamilySwitchDialog'
import TriggerWordCard from '../../../components/TriggerWordCard'

// Global model fields come from global settings and are read-only per version
const GLOBAL_MODEL_FIELDS = [
  'transformer_path',
  'vae_path',
  'text_encoder_path',
  't5_tokenizer_path',
]

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

export default function TrainPage() {
  const { t, i18n } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm, prompt } = useDialog()
  const navigate = useNavigate()
  const settingsDrawer = useSettingsDrawer()

  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [presets, setPresets] = useState<PresetSummary[]>([])
  const [configResp, setConfigResp] = useState<VersionConfigResponse | null>(null)
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [reg, setReg] = useState<RegStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [autoSyncPaths, setAutoSyncPaths] = useState<boolean>(true)
  const [droppedFields, setDroppedFields] = useState<string[]>([])
  const [defaultedFields, setDefaultedFields] = useState<string[]>([])

  /** The config JSON snapshot as last saved to disk; the baseline for dirty checks. */
  const savedJsonRef = useRef<string | null>(null)
  /** Synchronous mirror of the current config. React setState is queued and only
   * flushes after the event handler finishes; onEnqueue / cleanup-on-unmount need to
   * read the latest value immediately, they can't wait for a React commit. Every
   * setConfig call goes through the setConfigSync wrapper: it writes the ref
   * synchronously and the state asynchronously. */
  const configRef = useRef<ConfigData | null>(null)
  /** The currently in-flight save promise, to dedupe overlapping save requests. */
  const inFlightSaveRef = useRef<Promise<void> | null>(null)
  /** The pending debounce setTimeout id; onEnqueue needs to cancel it. */
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // 0.8.2 hotfix: removed presetBaselineRef + the "customized" label logic. After the
  // fork, the version yaml is decoupled from the global preset, so this is now
  // acknowledged as "project-specific config". The "customized" label was
  // misleading (the 4 global model fields get absolute paths injected on fork, which
  // always diffs against the global preset's relative paths -> it would always show as customized).

  // Preset picker (dropdown mode, matching the Presets page)
  const [pickerOpen, setPickerOpen] = useState(false)
  const [pickerSearch, setPickerSearch] = useState('')
  // 0.17 P-B -- scheduled training popover (two entry points: delay N hours / pick an absolute time, D7).
  const [scheduleOpen, setScheduleOpen] = useState(false)
  const [scheduleTime, setScheduleTime] = useState('')
  const pickerAnchorRef = useRef<HTMLButtonElement | null>(null)
  const pickerPopRef = useRef<HTMLDivElement | null>(null)

  // "New preset" = one-click create + apply: clicking the "+ New preset" card
  // directly generates a preset named `<slug>_<label>`, writes it to the global pool,
  // and forks it onto the current version. No intermediate form pops up, to avoid the
  // user clicking + and thinking "it's created" when nothing was actually saved yet
  // (state isn't persisted, coming back to the page it'd be empty again).

  /** Wraps setConfig: first writes configRef synchronously (bypassing React's state
   * flush delay), then calls setConfig to trigger a React render. Every entry point
   * that wants to change config -- SchemaForm.onChange or anything else -- must go
   * through this, never call setConfig directly. */
  const setConfigSync = useCallback((v: ConfigData | null) => {
    configRef.current = v
    setConfig(v)
  }, [])

  /** The pending family-switch target (renders FamilySwitchDialog when non-null). */
  const [familySwitchTarget, setFamilySwitchTarget] = useState<string | null>(null)

  /** Header auto-save indicator (the same SaveIndicator as the Settings page). */
  const [saveStatus, setSaveStatus] = useState<SaveStatus>({ state: 'idle' })

  /** Config as loaded for this version: the form marks fields that differ
   *  from it, and the recent-changes list reverts back to it. */
  const [baseline, setBaseline] = useState<ConfigData | null>(null)
  const [changes, setChanges] = useState<ChangeEntry[]>([])
  const [tab, setTab] = useLocalStorageState<TrainTabId>('train.formTab', 'model')

  /** Fold an edit into the session change log: one row per field, keeping its
   *  first "from"; a field edited back to where it started drops out. */
  const recordChanges = useCallback((prev: ConfigData | null, next: ConfigData) => {
    if (!prev) return
    const at = Date.now()
    const edits: ChangeEntry[] = []
    for (const k of new Set([...Object.keys(prev), ...Object.keys(next)])) {
      if (!sameValue(prev[k], next[k])) edits.push({ field: k, from: prev[k], to: next[k], at })
    }
    if (edits.length === 0) return
    setChanges((cur) => {
      const map = new Map(cur.map((e) => [e.field, e]))
      for (const e of edits) {
        const old = map.get(e.field)
        if (!old) map.set(e.field, e)
        else if (sameValue(old.from, e.to)) map.delete(e.field)
        else map.set(e.field, { ...old, to: e.to, at })
      }
      return Array.from(map.values()).sort((a, b) => b.at - a.at)
    })
  }, [])

  /** SchemaForm.onChange entry point: intercepts model_family changes and routes
   * them through the switch action (P4-3). Switching family isn't a plain field edit
   * -- it pops a structured confirmation dialog (the backend recomputes paths + resets
   * family-flavored fields); canceling leaves the old value untouched. Every other
   * field change passes straight through to setConfigSync. */
  const onFormChange = useCallback((v: ConfigData) => {
    const prev = configRef.current
    const prevFamily = String(prev?.model_family ?? 'anima')
    const nextFamily = String(v.model_family ?? 'anima')
    if (prev && nextFamily !== prevFamily) {
      setFamilySwitchTarget(nextFamily)
      return
    }
    recordChanges(prev, v)
    setConfigSync(v)
  }, [setConfigSync, recordChanges])

  /** Edits that bypass the family check (family switch result, path reset). */
  const applyEdit = useCallback((v: ConfigData) => {
    recordChanges(configRef.current, v)
    setConfigSync(v)
  }, [setConfigSync, recordChanges])

  const revertChange = useCallback((e: ChangeEntry) => {
    const cur = configRef.current
    if (!cur) return
    onFormChange({ ...cur, [e.field]: e.from })
  }, [onFormChange])

  const vid = activeVersion?.id ?? null

  const applyPresetWarnings = useCallback((r: { dropped_fields?: string[]; defaulted_fields?: string[] }) => {
    setDroppedFields(r.dropped_fields ?? [])
    setDefaultedFields(r.defaulted_fields ?? [])
  }, [])

  const refreshConfig = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.getVersionConfig(project.id, vid)
      setConfigResp(r)
      setConfigSync(r.config)
      setBaseline(r.config)
      setChanges([])
      savedJsonRef.current = JSON.stringify(r.config)
      // Old-config compatibility (e.g. InfoNoise's mutual exclusion getting
      // auto-disabled by the backend) is written into r.defaulted_fields by the
      // backend, and rendered as the top banner. dropped_fields covers old fields
      // dropped as the schema evolved.
      applyPresetWarnings(r)
    } catch (e) {
      toast(t('train.loadConfigFailed', { error: e }), 'error')
    }
  }, [project.id, vid, toast, setConfigSync, t, applyPresetWarnings])

  useEffect(() => {
    api.schema().then(setSchema).catch((e) => toast(t('train.loadSchemaFailed', { error: e }), 'error'))
    api.listPresets().then(setPresets).catch(() => setPresets([]))
    api.getSecrets().then((s) => setAutoSyncPaths(s.models?.auto_sync_paths ?? true)).catch(() => {})
  }, [toast, t])

  useEffect(() => {
    setDroppedFields([])
    setDefaultedFields([])
    void refreshConfig()
  }, [refreshConfig])

  // Fetch reg status to display the "training set + regularization" distribution
  useEffect(() => {
    if (!vid) return
    api.getRegStatus(project.id, vid).then(setReg).catch(() => setReg(null))
  }, [project.id, vid])

  const stats = useTrainStats(project.id, activeVersion, reg, config)


  // auto_sync_paths ON (default / most users): the 4 model path fields are disabled,
  // and on fork the backend automatically overrides them with the global Settings
  // values; OFF (users with independent models): the fields are editable + get a
  // reset button, and on fork the preset's absolute paths are respected.
  const disabledFields = autoSyncPaths ? GLOBAL_MODEL_FIELDS : []
  const disabledHints = useMemo(() => {
    const h: Record<string, React.ReactNode> = {}
    if (autoSyncPaths) {
      const node = (
        <>
          {t('train.globalAutoLockedPrefix')} ·{' '}
          <button
            type="button"
            onClick={() => settingsDrawer.open({ section: 'models' })}
            className="bg-transparent border-none p-0 underline text-warn hover:opacity-80 cursor-pointer"
          >
            {t('train.globalAutoLockedLink')}
          </button>
        </>
      )
      for (const f of GLOBAL_MODEL_FIELDS) h[f] = node
    }
    return h
  }, [t, autoSyncPaths, settingsDrawer])
  // Project-specific fields (data_dir / reg_data_dir / output_dir, etc.): prefilled
  // by the project but not locked, tagged with an "Auto - project setting" badge so
  // the user knows it's prefilled, not from the preset. When the toggle is OFF, the
  // global model fields also get a "Defaults from Settings - editable" hint.
  const autoHints = useMemo(() => {
    const h: Record<string, string> = {}
    for (const f of configResp?.project_specific_fields ?? []) {
      if (!GLOBAL_MODEL_FIELDS.includes(f)) h[f] = t('train.projectAutoHint')
    }
    if (!autoSyncPaths) {
      for (const f of GLOBAL_MODEL_FIELDS) h[f] = t('train.globalAutoEditableHint')
    }
    return h
  }, [configResp?.project_specific_fields, t, autoSyncPaths])

  // When the toggle is OFF, add a "reset to global default" button to the 4 model
  // fields. The value comes from configResp.project_specific_defaults (the backend
  // already computed the absolute paths using default_paths_for_new_version).
  const makeResetSuffixes = useCallback(
    (formValues: ConfigData | null, setForm: (v: ConfigData) => void): Record<string, React.ReactNode> => {
      if (autoSyncPaths) return {}
      const psd = configResp?.project_specific_defaults
      if (!psd || !formValues) return {}
      const out: Record<string, React.ReactNode> = {}
      for (const f of GLOBAL_MODEL_FIELDS) {
        const dv = psd[f]
        if (typeof dv !== 'string' || !dv) continue
        out[f] = (
          <button
            type="button"
            onClick={() => setForm({ ...formValues, [f]: dv })}
            className="ds-ctl ds-ghost"
            style={{ height: 24 }}
            title={t('train.resetToGlobalDefaultTitle')}
          >
            {t('train.resetToGlobalDefault')}
          </button>
        )
      }
      return out
    },
    [autoSyncPaths, configResp?.project_specific_defaults, t],
  )

  /** Persist cfg to disk. Serialization guarantee: if a previous save is still in
   * flight, wait for it before deciding whether to save again; this way, multiple
   * setConfig + debounce calls never lose any of their content.
   *
   * Watch out for the race: the user may edit config again while we're awaiting --
   * in that case we must not overwrite React state with the server's normalized
   * result (it would wipe out what they're currently typing). We tell the two apart
   * by reference-comparing configRef.current === cfg:
   *   - equal -> the user hasn't touched it, safe to sync the server's normalized result to the UI
   *   - not equal -> the user has new content, only update the savedJson baseline,
   *     leave UI state alone; the useEffect debounce will naturally trigger the next
   *     save round to converge on the new content */
  const persistConfig = useCallback(async (cfg: ConfigData, force = false): Promise<void> => {
    while (inFlightSaveRef.current) {
      await inFlightSaveRef.current
    }
    // force: PUT even if the content hasn't changed ("clean up old fields" rewrites
    // the yaml -- a stale key on disk isn't in the GET-normalized result, so a JSON
    // diff wouldn't show any difference).
    if (!force && JSON.stringify(cfg) === savedJsonRef.current) return
    const p = (async () => {
      setSaveStatus({ state: 'saving' })
      try {
        const r = await api.putVersionConfig(project.id, vid!, cfg)
        setConfigResp((prev) => prev ? { ...prev, has_config: true, config: r.config } : prev)
        // Use the server-normalized r.config as the baseline, so the next dirty diff doesn't false-positive.
        savedJsonRef.current = JSON.stringify(r.config)
        if (configRef.current === cfg) {
          configRef.current = r.config
          setConfig(r.config)
        }
        // PUT rewrites the whole yaml (tolerant validate + prune), so there are no
        // more stale fields / invalid values on disk -- the compat banner's info is
        // now stale, clear it.
        applyPresetWarnings({})
        setSaveStatus({ state: 'saved', at: Date.now() })
      } catch (e) {
        setSaveStatus({ state: 'error', error: String(e) })
        throw e
      }
    })()
    inFlightSaveRef.current = p
    try { await p } finally { inFlightSaveRef.current = null }
  }, [project.id, vid, applyPresetWarnings])

  // ── auto-save ─────────────────────────────────────────────────────────
  // config changes -> persist after 600ms with no further changes. If it changes
  // again in the meantime -> cleanup clears the timeout and resets it.
  useEffect(() => {
    if (!config) return
    if (JSON.stringify(config) === savedJsonRef.current) return
    debounceTimerRef.current = setTimeout(() => {
      debounceTimerRef.current = null
      void persistConfig(config).catch((e) => toast(t('train.saveFailed', { error: e }), 'error'))
    }, 600)
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
        debounceTimerRef.current = null
      }
    }
  }, [config, persistConfig, toast, t])

  // On unmount (navigating away) if there's still a dirty state that hasn't been
  // saved -> fire-and-forget the PUT. Once a fetch is initiated the browser keeps
  // sending it, no need to await. Swallow errors so cleanup doesn't throw.
  useEffect(() => {
    return () => {
      const cur = configRef.current
      if (!cur || !vid) return
      if (JSON.stringify(cur) === savedJsonRef.current) return
      void api.putVersionConfig(project.id, vid, cur).catch(() => {})
    }
  }, [project.id, vid])

  const filteredPresets = useMemo(
    () => presets.filter((p) => !pickerSearch || p.name.toLowerCase().includes(pickerSearch.toLowerCase())),
    [presets, pickerSearch],
  )

  // Preview card view (data / YAML), remembered across visits.
  const [previewTab, setPreviewTab] = useLocalStorageState<'stats' | 'config'>('train.previewTab', 'stats')

  // Close the popover: click outside / Esc
  useEffect(() => {
    if (!pickerOpen) return
    const onDocClick = (e: MouseEvent) => {
      const target = e.target as Node
      if (pickerPopRef.current?.contains(target) || pickerAnchorRef.current?.contains(target)) return
      setPickerOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setPickerOpen(false) }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [pickerOpen])

  if (!activeVersion || !vid) {
    return <p className="text-fg-tertiary p-6">{t('train.noVersion')}</p>
  }

  const onForkPreset = async (name: string) => {
    if (!name) return
    if (configResp?.has_config) {
      const ok = await confirm(
        t('train.confirmReset', { name }),
        { tone: 'warn', okText: t('train.resetOkText') },
      )
      if (!ok) return
    }
    setBusy(true)
    try {
      const r = await api.forkPresetForVersion(project.id, vid, name)
      applyPresetWarnings(r)
      // refreshConfig refreshes this page's config state; reload refreshes the
      // parent's activeVersion, so the main form fields sync to show the new preset's content.
      await Promise.all([refreshConfig(), reload()])
      toast(t('train.resetSuccess', { name }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const onSaveAsPreset = async () => {
    const name = await prompt(t('train.promptPresetName'), {
      placeholder: 'my-preset',
      validate: (v) => {
        const trimmed = v.trim()
        if (!trimmed) return t('train.nameEmpty')
        if (!PRESET_NAME_RE.test(trimmed)) return t('train.nameInvalid')
        return null
      },
    })
    if (!name) return
    const trimmed = name.trim()
    setBusy(true)
    try {
      await api.saveVersionConfigAsPreset(project.id, vid, trimmed, false)
      const list = await api.listPresets()
      setPresets(list)
      toast(t('train.savedAsPreset', { name: trimmed }), 'success')
    } catch (e) {
      const msg = String(e)
      // Match on the error code, not on the message: the message is localized
      // (and was never Chinese to begin with), so a text match silently turned
      // the overwrite prompt into a plain error toast.
      if ((e as ApiError).code === 'preset.exists') {
        const overwrite = await confirm(t('train.alreadyExists', { name: trimmed }), {
          tone: 'danger',
          okText: t('train.overwriteOkText'),
        })
        if (overwrite) {
          try {
            await api.saveVersionConfigAsPreset(project.id, vid, trimmed, true)
            const list = await api.listPresets()
            setPresets(list)
            toast(t('train.overwritePreset', { name: trimmed }), 'success')
          } catch (e2) {
            toast(String(e2), 'error')
          }
        }
      } else {
        toast(msg, 'error')
      }
    } finally {
      setBusy(false)
    }
  }

  /** Default preset name = `<slug>_<label>`; falls back to `<slug>_v<id>` when the
   * label contains invalid characters. Editable by the user in the form input. */
  const defaultPresetName = (): string => {
    if (!activeVersion) return project.slug
    const candidate = `${project.slug}_${activeVersion.label}`
    if (PRESET_NAME_RE.test(candidate)) return candidate
    return `${project.slug}_v${activeVersion.id}`
  }

  /** One-click new preset + apply it to the current version.
   *
   * Steps:
   *   1. If the version already has a config -> pop a confirmation to overwrite (same as onForkPreset)
   *   2. Fetch the latest project_specific_defaults (a common user path is running
   *      the reg build only after the fork, so the cached configResp's reg status is stale)
   *   3. config = schema defaults + project path prefill (only when autoSyncPaths is on)
   *   4. Auto name = `<slug>_<label>` (PRESET_NAME_RE-compatible), with a _1 _2 suffix on collision
   *   5. savePreset -> forkPresetForVersion -> refresh all three states
   */
  const startCreatePreset = async () => {
    setPickerOpen(false)
    if (!vid || !schema) return

    if (configResp?.has_config) {
      const ok = await confirm(
        t('train.confirmReset', { name: t('train.newPresetAction') }),
        { tone: 'warn', okText: t('train.resetOkText') },
      )
      if (!ok) return
    }

    setBusy(true)
    try {
      const fresh = await api.getVersionConfig(project.id, vid).catch(() => null)
      const psd =
        fresh?.project_specific_defaults
        ?? configResp?.project_specific_defaults
        ?? {}

      // The global preset pool doesn't carry project-specific fields (dataset path /
      // output name, etc.): schema defaults are enough, and on fork the backend
      // injects the project prefill into the version's private config.
      // The 4 model fields: use the absolute paths computed from the current
      // Settings when autoSyncPaths is ON, keep the schema defaults when OFF (the
      // independent-model user scenario). Kept aligned with the cleanup logic in
      // services/presets.py: save_version_config_as_preset.
      const cleaned: ConfigData = { ...defaultsFromSchema(schema) }
      if (autoSyncPaths) {
        for (const f of GLOBAL_MODEL_FIELDS) {
          if (typeof psd[f] === 'string' && psd[f]) cleaned[f] = psd[f]
        }
      }

      const name = generateUniquePresetName(defaultPresetName(), presets)
      await api.savePreset(name, cleaned)
      const r = await api.forkPresetForVersion(project.id, vid, name)
      applyPresetWarnings(r)

      const list = await api.listPresets()
      setPresets(list)
      // refreshConfig refreshes this page's config state; reload refreshes the parent's
      // activeVersion -- only then will the main form fields show the new preset's content.
      await Promise.all([refreshConfig(), reload()])
      toast(t('train.createdPreset', { name }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const onEnqueue = async (scheduledAt?: number) => {
    if (!configResp?.has_config) {
      toast(t('train.noPresetError'), 'error')
      return
    }
    setBusy(true)
    try {
      // 1. Kill any pending debounced save; otherwise it could fire after enqueue, causing the
      //    worker to read a stale config when it starts up.
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
        debounceTimerRef.current = null
      }
      // 2. Wait for any in-flight save to finish (the one the debounce just fired).
      if (inFlightSaveRef.current) await inFlightSaveRef.current
      // 3. Diff once more using configRef (not the config closure). This covers the case of "user
      //    finishes typing a value in an input, then clicks start training without blurring
      //    first": input.onBlur (commit) queues setConfig synchronously but React hasn't flushed
      //    yet, so the config closure is stale, while configRef was already updated synchronously in setConfigSync.
      const cur = configRef.current
      if (cur && JSON.stringify(cur) !== savedJsonRef.current) {
        await persistConfig(cur)
      }
      const task = await api.enqueueVersionTraining(
        project.id, vid, scheduledAt != null ? { scheduledAt } : undefined,
      )
      if (scheduledAt != null) {
        toast(t('train.scheduledNav', {
          id: task.id,
          time: new Date(scheduledAt * 1000).toLocaleString(i18n.language, { hour12: false }),
        }), 'success')
      } else {
        toast(t('train.enqueuedNav', { id: task.id }), 'success')
      }
      setScheduleOpen(false)
      void reload()
      navigate('/queue')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  // datetime-local's value format (local timezone, minute precision).
  const toLocalInputValue = (d: Date) => {
    const pad = (n: number) => String(n).padStart(2, '0')
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
      + `T${pad(d.getHours())}:${pad(d.getMinutes())}`
  }

  const onScheduleAbsolute = () => {
    if (!scheduleTime) return
    const ts = new Date(scheduleTime).getTime() / 1000
    if (!Number.isFinite(ts) || ts <= Date.now() / 1000 + 30) {
      toast(t('train.schedulePast'), 'error')
      return
    }
    void onEnqueue(ts)
  }

  // ── form tabs, filters and the config map ───────────────────────────────
  const props = schema?.schema.properties ?? {}
  const disabledSet = new Set(disabledFields)
  const isLocked = (name: string) => {
    const p = props[name]
    return disabledSet.has(name) || (!!p?.disable_when && !!config && evalShowWhen(p.disable_when, config))
  }
  const isChanged = (name: string) => !!baseline && !!config && name in baseline && !sameValue(baseline[name], config[name])
  const isNonDefault = (name: string) => !!config && props[name] != null && !sameValue(props[name].default, config[name])
  /** Fields of a group as the form would show them right now. */
  const groupFields = (key: string): string[] =>
    Object.entries(props)
      .filter(([, p]) => !p.hidden && (p.group ?? 'misc') === key
        && (!config || evalShowWhen(p.show_when, config)))
      .map(([n]) => n)
  const tabGroups = (id: TrainTabId): string[] => {
    const own = TRAIN_TABS.find((x) => x.id === id)!.groups as readonly string[]
    if (id !== 'system') return [...own]
    // groups a newer schema adds land on the last tab
    const known = new Set(TRAIN_TABS.flatMap((x) => x.groups as readonly string[]))
    return [...own, ...(schema?.groups ?? []).map((g) => g.key).filter((k) => !known.has(k))]
  }
  const tabOf = (groupKey: string): TrainTabId =>
    TRAIN_TABS.find((x) => (x.groups as readonly string[]).includes(groupKey))?.id ?? 'system'
  const tabFieldNames = (id: TrainTabId) => tabGroups(id).flatMap(groupFields)
  const currentFields = tabFieldNames(tab)
  const totalFields = TRAIN_TABS.reduce((s, x) => s + tabFieldNames(x.id).length, 0)

  const resetTab = async () => {
    const cur = configRef.current
    if (!cur || !schema) return
    const keep = new Set([...GLOBAL_MODEL_FIELDS, ...(configResp?.project_specific_fields ?? [])])
    const next: ConfigData = { ...cur }
    let n = 0
    for (const name of currentFields) {
      if (keep.has(name) || isLocked(name)) continue
      const def = props[name]?.default
      if (def === undefined || sameValue(def, cur[name])) continue
      next[name] = def
      n++
    }
    if (n === 0) {
      toast(t('train.resetTabNothing'), 'success')
      return
    }
    const ok = await confirm(t('train.resetTabConfirm', { count: n, tab: t(`train.tab_${tab}`) }), { tone: 'warn', okText: t('train.resetTabOk') })
    if (!ok) return
    onFormChange(next)
  }

  const openGroup = (key: string) => {
    setTab(tabOf(key))
    requestAnimationFrame(() => {
      document.getElementById(`schema-group-${key}`)?.scrollIntoView({ block: 'start', behavior: 'smooth' })
    })
  }

  const modelName = (() => {
    const p = String(config?.transformer_path ?? '')
    const base = p.split(/[\\/]/).pop() ?? ''
    return base.replace(/\.(safetensors|ckpt|pt|bin|gguf)$/i, '') || String(config?.model_family ?? '—')
  })()
  const subtitle = config
    ? <><code>{modelName}</code>{' · '}{[`${LORA_TYPE_LABEL[String(config.lora_type ?? 'lora')] ?? String(config.lora_type)} r${String(config.lora_rank ?? '—')}`, t('train.nParams', { count: totalFields })].join(' · ')}</>
    : t('steps.train.subtitle')

  const triggerUnderDop = !!config?.dop_enabled && groupFields('loss').includes('dop_enabled')
  const triggerRow = config ? (
    <TriggerWordCard
      projectId={project.id}
      version={activeVersion}
      dopEnabled={Boolean(config.dop_enabled)}
      onSaved={reload}
    />
  ) : null

  const saveBadge = saveStatus.state === 'saving'
    ? <span className="ds-badge ds-mute">{t('train.saving')}</span>
    : saveStatus.state === 'saved'
      ? <span className="ds-badge ds-ok">{t('train.savedAt', { time: new Date(saveStatus.at).toLocaleTimeString(i18n.language, { hour: '2-digit', minute: '2-digit' }) })}</span>
      : saveStatus.state === 'error'
        ? <span className="ds-badge ds-err" title={saveStatus.error}>{t('train.saveError')}</span>
        : null

  return (
    <StepShell
      idx={6}
      mobilePageScroll
      eyebrow={t('steps.eyebrowStep', { n: 5, label: activeVersion.label })}
      title={t('steps.train.title')}
      subtitle={subtitle}
      actions={
        <>
          {/* 0.17 P-B -- scheduled training: delay N hours / a specific time, creates a scheduled task. */}
          <button
            type="button"
            onClick={() => setScheduleOpen(true)}
            disabled={busy || !configResp?.has_config}
            className="ds-ctl"
            title={t('train.scheduleHint')}
            data-testid="train-schedule-btn"
          >
            {t('train.scheduleBtn')}
          </button>
          <button
            type="button"
            onClick={() => void onEnqueue()}
            disabled={busy || !configResp?.has_config}
            className="ds-btn-primary"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z" /></svg>
            {t('train.startTrainBtn')}
          </button>
          {scheduleOpen && (
            <ScheduleDialog
              busy={busy}
              scheduleTime={scheduleTime}
              minTime={toLocalInputValue(new Date())}
              onTimeChange={setScheduleTime}
              onDelay={(h) => void onEnqueue(Date.now() / 1000 + h * 3600)}
              onConfirm={onScheduleAbsolute}
              onClose={() => setScheduleOpen(false)}
            />
          )}
        </>
      }
      footer={<TrainLogBar project={project} vid={vid} />}
    >
      <div className="ds-train-scroll">
        {/* config bar: which config this version trains with, save state, swap */}
        <div className="ds-cfgbar" style={{ position: 'relative' }}>
          <span className="ds-sect-icon" style={{ width: 34, height: 34, borderRadius: 10, background: 'var(--green-soft)', color: 'var(--green-text)' }}>{TrainIcon.sliders}</span>
          <span style={{ flex: 1, minWidth: 0 }}>
            <span className="ds-cfg-name">
              {configResp?.has_config
                ? t('train.scopedConfigLabel', { title: project.title, label: activeVersion.label })
                : t('train.notConfiguredLabel')}
            </span>
            <span className="ds-cfg-sub">{configResp?.has_config ? t('train.cfgSub') : t('train.noConfigHint')}</span>
          </span>
          {saveBadge}
          <button
            ref={pickerAnchorRef}
            type="button"
            className="ds-ctl"
            style={{ height: 34 }}
            onClick={() => { setPickerOpen((v) => !v); setPickerSearch('') }}
            disabled={busy}
            aria-expanded={pickerOpen}
            title={configResp?.has_config ? t('train.pickerTitleConfigured') : t('train.pickerTitleEmpty')}
          >
            {configResp?.has_config ? t('train.changeConfig') : t('train.pickConfig')}
          </button>
          <KebabMenu
            label={t('train.configActions')}
            trigger="icon"
            items={[
              { label: t('train.saveAsPreset'), onSelect: () => void onSaveAsPreset(), disabled: busy || !configResp?.has_config },
              { label: t('train.newPresetAction'), onSelect: () => void startCreatePreset(), disabled: busy },
            ]}
          />
          {pickerOpen && (
            <div
              ref={pickerPopRef}
              role="dialog"
              aria-label={t('train.presetLabel')}
              className="ds-card"
              style={{ position: 'absolute', top: 'calc(100% + 6px)', right: 14, width: 480, maxWidth: 'calc(100% - 28px)', maxHeight: 480, display: 'flex', flexDirection: 'column', zIndex: 50, boxShadow: '0 18px 40px -18px rgba(20,22,20,.35)' }}
            >
              <div style={{ padding: 10, borderBottom: '1px solid var(--line)' }}>
                <span className="ds-search">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" /></svg>
                  <input
                    autoFocus
                    className="ds-inp"
                    placeholder={t('train.filterPresets')}
                    value={pickerSearch}
                    onChange={(e) => setPickerSearch(e.target.value)}
                  />
                </span>
              </div>
              <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: 10, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, alignContent: 'start' }}>
                {/* + New preset is always the first cell (consistent with the Presets page). Hidden
                    when pickerSearch is non-empty -- the user is searching for an existing one; creating new is a separate intent. */}
                {!pickerSearch && (
                  <button
                    type="button"
                    onClick={() => void startCreatePreset()}
                    disabled={busy}
                    className="ds-optcard"
                    style={{ borderStyle: 'dashed', boxShadow: 'none', color: 'var(--green-text)', fontWeight: 600, fontSize: 12.5 }}
                  >
                    {t('train.newPreset')}
                  </button>
                )}
                {filteredPresets.map((p) => (
                  <button
                    key={p.name}
                    type="button"
                    onClick={() => { setPickerOpen(false); void onForkPreset(p.name) }}
                    disabled={busy}
                    className="ds-optcard"
                    style={{ flexDirection: 'column', gap: 3 }}
                  >
                    <span className="ds-mono" style={{ fontSize: 12, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', width: '100%' }}>{p.name}</span>
                    <span className="ds-kpi-meta">{t('train.readPresetParams')}</span>
                  </button>
                ))}
                {presets.length > 0 && filteredPresets.length === 0 && (
                  <div className="ds-muted" style={{ gridColumn: '1 / -1', textAlign: 'center', padding: 16, fontSize: 12.5 }}>
                    {t('train.noMatch', { search: pickerSearch })}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {(droppedFields.length > 0 || defaultedFields.length > 0) && (
          <div className="ds-note ds-warn" style={{ alignItems: 'center' }}>
            <span style={{ flex: 1 }}>
              <b style={{ fontWeight: 600 }}>{t('presets.compatNoticeTitle')}</b>
              {droppedFields.length > 0 && (
                <span style={{ display: 'block' }}>{t('presets.droppedFieldsBody')} <span className="ds-mono" style={{ fontSize: 11 }}>{droppedFields.join(', ')}</span></span>
              )}
              {defaultedFields.length > 0 && (
                <span style={{ display: 'block' }}>{t('presets.defaultedFieldsBody')} <span className="ds-mono" style={{ fontSize: 11 }}>{defaultedFields.join(', ')}</span></span>
              )}
            </span>
            <button
              type="button"
              className="ds-ctl"
              style={{ height: 26, flex: 'none' }}
              title={t('presets.cleanLegacyTitle')}
              onClick={() => {
                const cur = configRef.current
                if (!cur) return
                void persistConfig(cur, true)
                  .then(() => toast(t('presets.cleanLegacyDone'), 'success'))
                  .catch((e) => toast(t('train.saveFailed', { error: e }), 'error'))
              }}
            >
              {t('presets.cleanLegacyBtn')}
            </button>
          </div>
        )}

        <TrainKpis stats={stats} />
        <StepBudget stats={stats} activeVersion={activeVersion} />

        {configResp === null || !schema ? (
          <ConfigSkeleton label={t('train.loadingConfig')} />
        ) : !configResp.has_config || !config ? (
          <div className="ds-empty">
            <span style={{ fontWeight: 500, color: 'var(--ink-2)' }}>{t('train.notConfiguredLabel')}</span>
            <span style={{ fontSize: 11.5 }}>{t('train.noConfigHint')}</span>
            <button type="button" className="ds-btn-primary" style={{ marginTop: 6 }} onClick={() => setPickerOpen(true)} disabled={busy}>{t('train.pickConfig')}</button>
          </div>
        ) : (
          <>
            <div className="ds-formsplit">
              <div className="ds-card" style={{ minWidth: 0 }}>
                <div className="ds-tabs" role="tablist">
                  {TRAIN_TABS.map((x) => (
                    <button
                      key={x.id}
                      type="button"
                      role="tab"
                      aria-selected={tab === x.id}
                      className={`ds-tab${tab === x.id ? ' ds-is-active' : ''}`}
                      onClick={() => setTab(x.id)}
                    >
                      {t(`train.tab_${x.id}`)}<span className="ds-badge ds-mute">{tabFieldNames(x.id).length}</span>
                    </button>
                  ))}
                  <span style={{ flex: 1 }} />
                  <button type="button" className="ds-ctl ds-sm" style={{ alignSelf: 'center' }} onClick={() => void resetTab()} title={t('train.resetTab')}>
                    {TrainIcon.reset}{t('train.resetTabShort')}
                  </button>
                </div>
                {/* Trigger word: under DOP (gated row) when DOP is on and its field is
                    shown; otherwise at the top of the data tab, next to captions. */}
                {tab === 'data' && !triggerUnderDop && triggerRow}
                <SchemaForm
                  schema={schema}
                  values={config}
                  onChange={onFormChange}
                  disabledFields={disabledFields}
                  disabledHints={disabledHints}
                  autoHints={autoHints}
                  fieldSuffixes={makeResetSuffixes(config, applyEdit)}
                  groupKeys={tabGroups(tab)}
                  baseline={baseline}
                  afterField={triggerUnderDop ? { dop_enabled: triggerRow } : undefined}
                />
                {familySwitchTarget && (
                  <FamilySwitchDialog
                    target={familySwitchTarget}
                    config={config}
                    onApply={(switched) => {
                      applyEdit(switched)
                      setFamilySwitchTarget(null)
                    }}
                    onCancel={() => setFamilySwitchTarget(null)}
                  />
                )}
              </div>

              <div className="ds-card ds-train-preview">
                <div className="ds-card-head ds-pad">
                  <div><div className="ds-card-title">{t('train.previewTitle')}</div><div className="ds-card-sub">{t('train.previewSub')}</div></div>
                  <div className="ds-card-tools">
                    <div className="ds-seg" role="group">
                      <button type="button" className={`ds-seg-item${previewTab === 'stats' ? ' ds-is-active' : ''}`} onClick={() => setPreviewTab('stats')} aria-pressed={previewTab === 'stats'}>{t('train.previewTabStats')}</button>
                      <button type="button" className={`ds-seg-item${previewTab === 'config' ? ' ds-is-active' : ''}`} onClick={() => setPreviewTab('config')} aria-pressed={previewTab === 'config'}>{t('train.previewTabYaml')}</button>
                    </div>
                  </div>
                </div>
                {previewTab === 'stats' ? (
                  <PreviewData stats={stats} projectId={project.id} vid={vid} maskedLoss={config.masked_loss === true} />
                ) : (
                  <div style={{ padding: '2px 17px 16px', display: 'flex', flexDirection: 'column', minHeight: 420, maxHeight: 'calc(100vh - 220px)' }}>
                    <ConfigYamlPanel config={config} fileLabel="config.yaml" className="flex-1 flex flex-col min-h-0" />
                  </div>
                )}
              </div>
            </div>

            <ConfigMap
              schema={schema}
              groupFields={groupFields}
              isChanged={isChanged}
              isNonDefault={isNonDefault}
              isLocked={isLocked}
              openKey={tab}
              tabOf={tabOf}
              onOpen={openGroup}
            />

            <RecentChanges
              changes={changes}
              schema={schema}
              onRevert={revertChange}
            />
          </>
        )}
      </div>
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// Pieces of the page (mockup Train / TrainModel / TrainData / TrainOptim /
// TrainSystem / TrainOutput)
// ---------------------------------------------------------------------------

type TrainTabId = 'model' | 'data' | 'optim' | 'system'

/** Which schema groups each tab of the form shows. */
/** How the network type reads in the page subtitle (the mockup writes "LoKr r32"). */
const LORA_TYPE_LABEL: Record<string, string> = { lora: 'LoRA', lokr: 'LoKr', loha: 'LoHa', ortho: 'OrthoLoRA', tlora: 'T-LoRA' }

const TRAIN_TABS = [
  { id: 'model', groups: ['model', 'lora'] },
  { id: 'data', groups: ['dataset', 'caption'] },
  { id: 'optim', groups: ['training', 'timestep_sampling', 'loss', 'noise_augmentation'] },
  { id: 'system', groups: ['system', 'output', 'sample', 'eval_validation', 'monitor'] },
] as const satisfies ReadonlyArray<{ id: TrainTabId; groups: readonly string[] }>

interface ChangeEntry {
  field: string
  from: unknown
  to: unknown
  at: number
}

const TrainIcon = {
  sliders: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round"><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" /><circle cx="16" cy="6" r="2" /><circle cx="10" cy="12" r="2" /><circle cx="18" cy="18" r="2" /></svg>
  ),
  reset: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v5h5" /></svg>
  ),
  samples: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round"><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></svg>
  ),
  steps: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round"><path d="M4 18h4v-4h4v-4h4V6h4" /></svg>
  ),
  vram: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round"><rect x="3" y="7" width="18" height="10" rx="2" /><path d="M7 7V4M11 7V4M15 7V4M19 20v-3M5 20v-3" /></svg>
  ),
  gpu: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round"><path d="M3 12h4l3-7 4 14 3-7h4" /></svg>
  ),
}

// Donut / legend colours of the epoch composition (train folders, then reg).
const COMPOSITION_COLORS = ['#a3db52', '#cfe3ad', '#e3ecd4', '#bcd98e', '#8fc43f', '#dbe8c4']
const REG_COLOR = '#d4d5d1'

function formatNum(n: number, lang: string): string {
  return new Intl.NumberFormat(lang).format(n)
}

/** Scheduling dialog: a delay of 1–8 hours or an exact time. */
function ScheduleDialog({ busy, scheduleTime, minTime, onTimeChange, onDelay, onConfirm, onClose }: {
  busy: boolean
  scheduleTime: string
  minTime: string
  onTimeChange: (v: string) => void
  onDelay: (hours: number) => void
  onConfirm: () => void
  onClose: () => void
}) {
  const { t } = useTranslation()
  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-40 flex items-center justify-center"
      style={{ background: 'rgba(20,22,20,.18)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
      data-testid="train-schedule-modal"
    >
      <div className="ds-card" style={{ width: '90%', maxWidth: 440, padding: 20, display: 'flex', flexDirection: 'column', gap: 16, boxShadow: '0 24px 60px -24px rgba(20,22,20,.4)' }}>
        <div className="ds-card-title" style={{ fontSize: 16 }}>{t('train.scheduleBtn')}</div>
        <div>
          <div className="ds-cap" style={{ marginBottom: 8 }}>{t('train.scheduleDelaySection')}</div>
          <div style={{ display: 'flex', gap: 6 }}>
            {[1, 2, 4, 8].map((h) => (
              <button key={h} type="button" onClick={() => onDelay(h)} disabled={busy} className="ds-ctl" style={{ flex: 1 }} data-testid={`train-schedule-delay-${h}h`}>
                +{h}h
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="ds-cap" style={{ marginBottom: 8 }}>{t('train.scheduleAbsoluteSection')}</div>
          <input
            type="datetime-local"
            className="ds-inp"
            value={scheduleTime}
            min={minTime}
            onChange={(e) => onTimeChange(e.target.value)}
            data-testid="train-schedule-time"
          />
        </div>
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button type="button" onClick={onClose} className="ds-ctl ds-ghost">{t('common.cancel')}</button>
          <button type="button" onClick={onConfirm} disabled={busy || !scheduleTime} className="ds-btn-primary" data-testid="train-schedule-confirm">
            {t('train.scheduleConfirm')}
          </button>
        </div>
      </div>
    </div>
  )
}

/** Normalizes config.resolution to number[] (schema is list[int]; also falls back for old configs / scalar values). */
function configResolutions(config: ConfigData | null): number[] {
  const r = config?.resolution as unknown
  if (Array.isArray(r)) return r.length ? (r as number[]) : [1024]
  if (typeof r === 'number') return [r]
  return [1024]
}

/** A folder's effective sample count = repeat × image count × resolution-tier count (a px folder is fixed at 1 tier; otherwise follows the config list). */
function folderEffective(name: string, imageCount: number, resoCount: number): number {
  const { reso, repeat } = parseFolderMeta(name)
  return repeat * imageCount * (reso ? 1 : resoCount)
}

/** reg.files look like `5_concept/12345.png` -- counts are aggregated by the first path segment (folder). */
function aggregateRegFolders(files: string[]): Array<{ name: string; image_count: number }> {
  const m = new Map<string, number>()
  for (const f of files) {
    const idx = f.indexOf('/')
    if (idx < 0) continue
    const folder = f.slice(0, idx)
    m.set(folder, (m.get(folder) ?? 0) + 1)
  }
  return Array.from(m.entries())
    .map(([name, image_count]) => ({ name, image_count }))
    .sort((a, b) => a.name.localeCompare(b.name))
}

interface TrainStats {
  trainFolders: Array<{ name: string; image_count: number }>
  regFolders: Array<{ name: string; image_count: number }>
  resoCount: number
  trainEffective: number
  regEffective: number
  shownEffective: number
  bs: number
  ga: number
  epochs: number
  maxSteps: number
  navitOn: boolean
  navitEst: BucketDistribution['navit'] | null
  stepsPerEpoch: number | null
  naturalTotal: number | null
  finalTotal: number | null
  maxStepsTruncates: boolean
  dist: BucketDistribution | null
  estimate: TrainEstimate | null
  reg: RegStatus | null
}

/** Dataset numbers the trainer will see: effective samples (repeats ×
 *  resolutions, train + reg), steps per epoch and in total, plus the backend's
 *  bucket distribution and runtime estimate. */
function useTrainStats(projectId: number, activeVersion: Version | null, reg: RegStatus | null, config: ConfigData | null): TrainStats {
  const trainFolders = useMemo(() => activeVersion?.stats?.train_folders ?? [], [activeVersion])
  const regFolders = useMemo(
    () => (reg && reg.exists ? aggregateRegFolders(reg.files) : []),
    [reg]
  )
  const resoCount = configResolutions(config).length
  const trainEffective = trainFolders.reduce((s, f) => s + folderEffective(f.name, f.image_count, resoCount), 0)
  const regEffective = regFolders.reduce((s, f) => s + folderEffective(f.name, f.image_count, resoCount), 0)
  const totalEffective = trainEffective + regEffective

  // Bucket distribution + NaViT packing estimate (computed backend-side with the real BucketManager / NavitPackBatchSampler).
  const vid = activeVersion?.id ?? 0
  const navitOn = config?.navit_packing === true
  const [dist, setDist] = useState<BucketDistribution | null>(null)
  const distSig = JSON.stringify([
    config?.resolution,
    config?.aspect_ratio_limit,
    // Folder list (including px prefix / repeat / image count) -- a rename adding a px prefix
    // must also trigger a refetch, so it can't just look at totals
    activeVersion?.stats?.train_folders,
    // Inputs to the navit packing estimate -- a change in any of them can change the pack count
    config?.navit_packing,
    config?.navit_native_resolution,
    config?.navit_token_budget,
    config?.navit_max_images_per_pack,
    config?.navit_pack_strategy,
    config?.navit_pack_ffd_window,
    config?.navit_drop_last,
    config?.navit_native_over_budget,
    config?.seed,
    reg?.exists,
    reg && reg.exists ? reg.files.length : 0,
  ])
  useEffect(() => {
    if (!projectId || !vid) return
    let cancelled = false
    api.getBucketDistribution(projectId, vid)
      .then((d) => { if (!cancelled) setDist(d) })
      .catch(() => { if (!cancelled) setDist(null) })
    return () => { cancelled = true }
  }, [projectId, vid, distSig])

  // Runtime estimate (VRAM verdict + measured it/s). Rides the same signature as
  // the distribution so it refreshes when the config that feeds it changes.
  const [estimate, setEstimate] = useState<TrainEstimate | null>(null)
  const estSig = JSON.stringify([distSig, config?.blocks_to_swap, config?.batch_size, config?.lora_rank, config?.mixed_precision])
  useEffect(() => {
    if (!projectId || !vid) return
    let cancelled = false
    api.getTrainEstimate(projectId, vid)
      .then((d) => { if (!cancelled) setEstimate(d) })
      .catch(() => { if (!cancelled) setEstimate(null) })
    return () => { cancelled = true }
  }, [projectId, vid, estSig])

  // Estimated optimizer steps per epoch (same semantics as sd-scripts' max_train_steps).
  // - Regular path: samples ÷ (batch × ga). Doesn't account for AR-bucketing loss (a bucket's
  //   last batch may be under-filled); error is < 5% for a dataset with uniform AR.
  // - navit_packing: batch_size doesn't participate in batching (NavitPackBatchSampler packs by
  //   token budget; one step = one pack) -- steps/epoch = ceil(pack count ÷ ga), where the pack
  //   count comes from the backend's real packing simulation; no estimate is shown before the
  //   simulation result is in hand (prefer no number over a wrong one).
  const bs = Number(config?.batch_size) || 1
  const ga = Number(config?.grad_accum) || 1
  const epochs = Number(config?.epochs) || 0
  const maxSteps = Number(config?.max_steps) || 0
  const navitEst = navitOn ? (dist?.navit ?? null) : null
  // Prefer the backend's count over the frontend folder scan: it is the number
  // the trainer will actually use (uncaptioned images excluded, repeats and
  // resolution fan-out applied, reg set included).
  const authoritativeSamples = dist?.effective_samples ?? null
  const stepsPerEpoch = navitOn
    ? (navitEst && navitEst.packs_per_epoch > 0
        ? Math.ceil(navitEst.packs_per_epoch / ga)
        : null)
    : ((authoritativeSamples ?? totalEffective) > 0
        ? Math.ceil((authoritativeSamples ?? totalEffective) / (bs * ga))
        : null)
  const naturalTotal = stepsPerEpoch !== null && epochs > 0 ? stepsPerEpoch * epochs : null
  const finalTotal = naturalTotal !== null && maxSteps > 0 ? Math.min(maxSteps, naturalTotal) : naturalTotal
  const maxStepsTruncates = maxSteps > 0 && naturalTotal !== null && maxSteps < naturalTotal
  // Under navit, the effective sample count defers to the real packing simulation (natively
  // collapses multi-resolution fan-out and includes reg); the frontend's folderEffective resoCount fan-out would over-count in this mode
  const shownEffective = navitEst && navitEst.samples > 0
    ? navitEst.samples
    : (authoritativeSamples ?? totalEffective)

  return {
    trainFolders, regFolders, resoCount, trainEffective, regEffective, shownEffective,
    bs, ga, epochs, maxSteps, navitOn, navitEst, stepsPerEpoch, naturalTotal, finalTotal,
    maxStepsTruncates, dist, estimate, reg,
  }
}

/** Seconds → a short human duration. Deliberately coarse: an estimate that
 *  prints "2h 14m 07s" pretends to a precision it does not have. */
function formatDuration(seconds: number, t: TFunction): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return '—'
  const h = Math.floor(seconds / 3600)
  const m = Math.round((seconds % 3600) / 60)
  if (h > 0) return m > 0 ? t('train.durHM', { h, m }) : t('train.durH', { h })
  if (m > 0) return t('train.durM', { m })
  return t('train.durS', { s: Math.round(seconds) })
}

function gib(bytes: number | null): number | null {
  return bytes == null ? null : bytes / 1024 ** 3
}

function formatGiB(bytes: number | null): string {
  const v = gib(bytes)
  return v == null ? '—' : `${v.toFixed(1)} GB`
}

/** Live GPU / CPU load for the fourth KPI (same feed as the top bar). */
function useSystemLoad(): SystemStatsData | null {
  const [stats, setStats] = useState<SystemStatsData | null>(null)
  useEffect(() => {
    let cancelled = false
    api.systemStats().then((s) => { if (!cancelled) setStats(s) }).catch(() => {})
    return () => { cancelled = true }
  }, [])
  useEventStream((evt) => {
    if (evt.type !== 'system_stats_updated') return
    const payload = evt.payload as SystemStatsData | undefined
    if (payload) setStats(payload)
  })
  return stats
}

function TrainKpis({ stats }: { stats: TrainStats }) {
  const { t, i18n } = useTranslation()
  const sys = useSystemLoad()
  const lang = i18n.language
  const { shownEffective, trainEffective, regEffective, stepsPerEpoch, finalTotal, estimate } = stats
  const mem = estimate?.memory ?? null
  const vNeed = gib(mem?.vram_need_bytes ?? null)
  const vFree = gib(mem?.free_vram_bytes ?? null)
  const gpu = sys?.gpu && sys.gpu.length > 0 ? sys.gpu[0] : null
  const trainShare = shownEffective > 0 ? Math.min(100, (trainEffective / Math.max(1, trainEffective + regEffective)) * 100) : 0
  return (
    <div className="ds-kpis ds-tight ds-train-kpis">
      <div className="ds-card ds-kpi">
        <div className="ds-kpi-top">
          <span className="ds-kpi-icon">{TrainIcon.samples}</span>
          <span style={{ marginLeft: 'auto' }} className="ds-kpi-meta">
            {regEffective > 0 ? t('train.kpiSamplesSplit', { train: formatNum(trainEffective, lang), reg: formatNum(regEffective, lang) }) : t('train.kpiSamplesTrainOnly')}
          </span>
        </div>
        <div className="ds-kpi-val" style={{ marginTop: 16 }}>{shownEffective > 0 ? formatNum(shownEffective, lang) : '—'}</div>
        <div className="ds-kpi-label">{t('train.kpiSamples')}</div>
        <span className="ds-meter" style={{ marginTop: 9 }}><i style={{ width: `${trainShare}%` }} /></span>
      </div>

      <div className="ds-card ds-kpi">
        <div className="ds-kpi-top">
          <span className="ds-kpi-icon">{TrainIcon.steps}</span>
          <span style={{ marginLeft: 'auto' }} className="ds-kpi-meta">
            {stepsPerEpoch != null ? t('train.kpiStepsPerEpoch', { n: formatNum(stepsPerEpoch, lang) }) : ''}
          </span>
        </div>
        <div className="ds-kpi-val" style={{ marginTop: 16 }}>
          {finalTotal != null ? formatNum(finalTotal, lang) : '—'}<small>{t('train.stepsUnit')}</small>
        </div>
        <div className="ds-kpi-label">
          {estimate?.speed && finalTotal != null
            ? t('train.kpiStepsTime', { time: formatDuration(finalTotal / estimate.speed.it_per_s, t), speed: estimate.speed.it_per_s.toFixed(2) })
            : t('train.kpiStepsNoSpeed')}
        </div>
      </div>

      <div className="ds-card ds-kpi">
        <div className="ds-kpi-top">
          <span className="ds-kpi-icon">{TrainIcon.vram}</span>
          {mem && (
            <span style={{ marginLeft: 'auto' }}>
              <span className={`ds-badge ${mem.ok ? 'ds-ok' : 'ds-warn'}`}>{mem.ok ? t('train.memoryFits') : t('train.memoryTight')}</span>
            </span>
          )}
        </div>
        <div className="ds-kpi-val" style={{ marginTop: 16 }}>
          {vNeed != null ? vNeed.toFixed(1) : '—'}{vFree != null && <small>/ {vFree.toFixed(1)} GB</small>}
        </div>
        <div className="ds-kpi-label">
          {mem ? t('train.kpiVram', { swap: mem.blocks_to_swap, total: mem.total_blocks }) : t('train.kpiVramNone')}
        </div>
        {vNeed != null && vFree != null && vFree > 0 && (
          <span className="ds-meter" style={{ marginTop: 9 }}><i className={mem?.ok ? undefined : 'ds-warn'} style={{ width: `${Math.min(100, (vNeed / vFree) * 100)}%` }} /></span>
        )}
      </div>

      <div className="ds-card ds-kpi">
        <div className="ds-kpi-top">
          <span className="ds-kpi-icon">{TrainIcon.gpu}</span>
          <span style={{ marginLeft: 'auto' }} className="ds-kpi-meta">
            {sys ? [t('train.kpiCpu', { n: Math.round(sys.cpu_pct) }), gpu?.temp_c != null ? `${gpu.temp_c} °C` : null].filter(Boolean).join(' · ') : ''}
          </span>
        </div>
        <div className="ds-kpi-val" style={{ marginTop: 16 }}>
          {gpu ? Math.round(gpu.util_pct) : sys ? Math.round(sys.cpu_pct) : '—'}<small>{gpu ? t('train.pctGpu') : t('train.pctCpu')}</small>
        </div>
        <div className="ds-kpi-label" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {gpu ? t('train.kpiLoadGpu', { name: gpu.name }) : t('train.kpiLoad')}
        </div>
        {(gpu || sys) && (
          <span className="ds-meter" style={{ marginTop: 9 }}><i style={{ width: `${Math.min(100, gpu ? gpu.util_pct : sys!.cpu_pct)}%` }} /></span>
        )}
      </div>
    </div>
  )
}

/** "Step budget": from files on disk to optimiser steps, the way the trainer
 *  counts them. */
function StepBudget({ stats, activeVersion }: { stats: TrainStats; activeVersion: Version }) {
  const { t, i18n } = useTranslation()
  const lang = i18n.language
  const trainImages = stats.trainFolders.reduce((s, f) => s + f.image_count, 0)
  const tagged = activeVersion.stats?.tagged_image_count ?? null
  const regImages = stats.regFolders.reduce((s, f) => s + f.image_count, 0)
  const repeats = Array.from(new Set(stats.trainFolders.map((f) => parseFolderMeta(f.name).repeat))).sort((a, b) => b - a)
  const rows: Array<{ label: string; n: number | null; drop: React.ReactNode; tone?: 'hatch' | 'green' | 'soft'; dropTone?: 'red' | 'amber' | 'mute' }> = [
    { label: t('train.budgetFiles'), n: trainImages, drop: null, tone: 'hatch' },
  ]
  if (tagged != null) {
    const missing = Math.max(0, trainImages - tagged)
    rows.push({
      label: t('train.budgetCaptioned'),
      n: tagged,
      drop: missing > 0 ? t('train.budgetUncaptioned', { n: missing }) : t('train.budgetAllCaptioned'),
      dropTone: missing > 0 ? 'red' : 'mute',
    })
  }
  if (regImages > 0) {
    rows.push({ label: t('train.budgetReg'), n: regImages, drop: t('train.budgetRegNote'), dropTone: 'mute' })
  }
  rows.push({
    label: t('train.budgetSamples'),
    n: stats.shownEffective || null,
    drop: stats.navitOn
      ? t('train.budgetNavitSamples')
      : [repeats.length ? repeats.map((r) => `×${r}`).join(' · ') + ' ' + t('train.budgetByFolders') : null,
        stats.resoCount > 1 ? t('train.budgetResos', { n: stats.resoCount }) : null,
        regImages > 0 ? t('train.budgetPlusReg') : null].filter(Boolean).join(' · '),
    tone: 'soft',
    dropTone: 'mute',
  })
  rows.push({
    label: t('train.budgetSteps'),
    n: stats.stepsPerEpoch,
    drop: stats.navitOn
      ? (stats.navitEst ? t('train.budgetNavitSteps', { packs: formatNum(stats.navitEst.packs_per_epoch, lang), ga: stats.ga }) : t('train.navitEstimating'))
      : t('train.budgetStepsNote', { bs: stats.bs, ga: stats.ga, epochs: stats.epochs, total: stats.finalTotal != null ? formatNum(stats.finalTotal, lang) : '—' })
        + (stats.maxStepsTruncates ? ` · ${t('train.maxStepsLabel', { n: stats.maxSteps })}` : ''),
    tone: 'green',
    dropTone: 'mute',
  })
  const max = Math.max(1, ...rows.map((r) => r.n ?? 0))
  return (
    <div className="ds-card">
      <div className="ds-card-head">
        <div><div className="ds-card-title">{t('train.budgetTitle')}</div><div className="ds-card-sub">{t('train.budgetSub')}</div></div>
      </div>
      <div style={{ padding: '12px 17px 16px' }}>
        {rows.map((r) => (
          <div key={r.label} className="ds-frow">
            <span className="ds-frow-label" style={{ width: 186 }}>{r.label}</span>
            <span className="ds-frow-pct">{r.n != null && trainImages > 0 ? `${Math.round((r.n / trainImages) * 100)}%` : ''}</span>
            <span className="ds-frow-n">{r.n != null ? formatNum(r.n, lang) : '—'}</span>
            <span className="ds-frow-track">
              <span
                className={`ds-bar${r.tone === 'hatch' ? ' ds-hatch' : r.tone === 'green' ? ' ds-green' : ''}`}
                style={{ width: `${Math.max(1, ((r.n ?? 0) / max) * 62)}%`, ...(r.tone === 'soft' ? { background: 'var(--green-line)' } : null) }}
              />
              {r.drop && <span className={`ds-drop ds-${r.dropTone ?? 'mute'}`}>{r.drop}</span>}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

/** Right card, "Data" view: epoch composition, step counts, memory, buckets. */
function PreviewData({ stats, projectId, vid, maskedLoss }: { stats: TrainStats; projectId: number; vid: number; maskedLoss: boolean }) {
  const { t, i18n } = useTranslation()
  const lang = i18n.language
  const segments = [
    ...stats.trainFolders.map((f, i) => ({
      key: `t-${f.name}`, label: f.name, color: COMPOSITION_COLORS[i % COMPOSITION_COLORS.length],
      value: folderEffective(f.name, f.image_count, stats.resoCount),
    })),
    ...stats.regFolders.map((f) => ({
      key: `r-${f.name}`, label: `reg / ${f.name}`, color: REG_COLOR,
      value: folderEffective(f.name, f.image_count, stats.resoCount),
    })),
  ].filter((s) => s.value > 0)
  const sum = segments.reduce((s, x) => s + x.value, 0)
  const trainImages = stats.trainFolders.reduce((s, f) => s + f.image_count, 0)
  let offset = 25
  const mem = stats.estimate?.memory ?? null
  const vNeed = gib(mem?.vram_need_bytes ?? null)
  const vFree = gib(mem?.free_vram_bytes ?? null)

  return (
    <div style={{ padding: '2px 17px 16px', display: 'flex', flexDirection: 'column', gap: 15 }}>
      <div>
        <div className="ds-cap" style={{ marginBottom: 10 }}>{t('train.compositionTitle', { n: formatNum(stats.shownEffective, lang) })}</div>
        {segments.length === 0 ? (
          <div className="ds-muted" style={{ fontSize: 12 }}>{t('train.noTrainImages')}</div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <svg width="96" height="96" viewBox="0 0 42 42" aria-hidden="true" style={{ flex: 'none' }}>
              <circle cx="21" cy="21" r="15.9155" fill="none" stroke="var(--sunken)" strokeWidth="5" />
              {segments.map((s) => {
                const pct = (s.value / sum) * 100
                const el = (
                  <circle key={s.key} cx="21" cy="21" r="15.9155" fill="none" stroke={s.color} strokeWidth="5"
                    strokeDasharray={`${pct} ${100 - pct}`} strokeDashoffset={offset} />
                )
                offset -= pct
                return el
              })}
              <text x="21" y="20.6" textAnchor="middle" fontSize="7" fontWeight="600" fill="var(--ink)">{trainImages}</text>
              <text x="21" y="25.4" textAnchor="middle" fontSize="3.4" fill="var(--ink-3)">{t('train.trainImagesShort')}</text>
            </svg>
            <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
              {segments.map((s) => (
                <div key={s.key} className="ds-kv">
                  <span className="ds-k ds-legend" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}><s style={{ background: s.color }} />{s.label}</span>
                  <span className="ds-v">{formatNum(s.value, lang)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div style={{ borderTop: '1px solid var(--line)', paddingTop: 13, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div><div className="ds-cap">{t('train.statStepsEpoch')}</div><div className="ds-stat-v">{stats.stepsPerEpoch != null ? formatNum(stats.stepsPerEpoch, lang) : '—'}</div></div>
        <div><div className="ds-cap">{stats.maxStepsTruncates ? t('train.maxStepsLabel', { n: stats.maxSteps }) : t('train.totalSteps')}</div><div className="ds-stat-v" style={{ color: 'var(--green-text)' }}>{stats.finalTotal != null ? formatNum(stats.finalTotal, lang) : '—'}</div></div>
        <div><div className="ds-cap">{t('train.statEpochs')}</div><div className="ds-stat-v">{stats.epochs || '—'}</div></div>
        <div><div className="ds-cap">{t('train.estDuration')}</div><div className="ds-stat-v">{stats.estimate?.speed && stats.finalTotal != null ? `≈ ${formatDuration(stats.finalTotal / stats.estimate.speed.it_per_s, t)}` : '—'}</div></div>
      </div>

      {mem && (
        <div style={{ borderTop: '1px solid var(--line)', paddingTop: 13 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 9 }}>
            <span className="ds-cap" style={{ flex: 1 }}>{t('train.memoryTitle')}</span>
            <span className={`ds-badge ${mem.ok ? 'ds-ok' : 'ds-warn'}`}>{mem.ok ? t('train.memoryFits') : t('train.memoryTight')}</span>
          </div>
          {vNeed != null && vFree != null && vFree > 0 && (
            <div style={{ display: 'flex', height: 12, borderRadius: 4, overflow: 'hidden', background: 'var(--sunken)' }}>
              <span style={{ width: `${Math.min(100, (vNeed / vFree) * 100)}%`, background: mem.ok ? 'var(--green-600)' : '#d3a53a' }} title={t('train.vramNeeded')} />
            </div>
          )}
          <div className="ds-kv" style={{ marginTop: 8 }}><span className="ds-k">{t('train.vramNeeded')}</span><span className="ds-v">{formatGiB(mem.vram_need_bytes)}</span></div>
          <div className="ds-kv"><span className="ds-k">{t('train.vramFree')}</span><span className="ds-v">{formatGiB(mem.free_vram_bytes)}</span></div>
          <div className="ds-kv"><span className="ds-k">{t('train.ramNeeded')}</span><span className="ds-v">{formatGiB(mem.ram_need_bytes)}{mem.avail_ram_bytes != null ? ` / ${formatGiB(mem.avail_ram_bytes)}` : ''}</span></div>
          <div className="ds-kv"><span className="ds-k">{t('train.blocksSwapped')}</span><span className="ds-v">{mem.blocks_to_swap} / {mem.total_blocks}</span></div>
          {!mem.ok && mem.recommended_blocks_to_swap != null && (
            <div className="ds-kv"><span className="ds-k">{t('train.blocksRecommended')}</span><span className="ds-v ds-acc">{mem.recommended_blocks_to_swap}</span></div>
          )}
        </div>
      )}

      <BucketBars dist={stats.dist} />

      <MaskedLossHint projectId={projectId} vid={vid} maskedLoss={maskedLoss} />
    </div>
  )
}

/** Resolution buckets as bars (top six, the rest folded into "…"). With
 *  NaViT-native there are no buckets, so the native size histogram is shown. */
function BucketBars({ dist }: { dist: BucketDistribution | null }) {
  const { t } = useTranslation()
  if (!dist) return null
  const native = !!dist.navit?.native
  const all = native
    ? dist.navit!.sizes
    : dist.groups.flatMap((g) => g.buckets)
  if (all.length === 0) return null
  const sorted = [...all].sort((a, b) => b.count - a.count)
  const top = sorted.slice(0, 6)
  const restCount = sorted.slice(6).reduce((s, x) => s + x.count, 0)
  const max = Math.max(1, ...top.map((b) => b.count), restCount)
  const label = (b: { w: number; h: number }) => (b.w === b.h ? `${b.w}²` : `${b.w}×${b.h}`)
  return (
    <div style={{ borderTop: '1px solid var(--line)', paddingTop: 13 }}>
      <div className="ds-cap" style={{ marginBottom: 10 }}>{native ? t('train.navitDistTitle') : t('train.bucketDistTitle')}</div>
      <div className="ds-barset">
        {top.map((b) => <i key={`${b.w}x${b.h}`} style={{ height: `${(b.count / max) * 100}%` }} title={`${b.w}×${b.h} · ${b.count}`} />)}
        {restCount > 0 && <i className="ds-mute" style={{ height: `${(restCount / max) * 100}%` }} title={t('train.bucketRest', { n: restCount })} />}
      </div>
      <div className="ds-axis">
        {top.map((b) => <span key={`${b.w}x${b.h}`}>{label(b)}</span>)}
        {restCount > 0 && <span>…</span>}
      </div>
      <div className="ds-kpi-meta" style={{ marginTop: 8 }}>
        {native ? t('train.navitDistHint') : dist.navit ? t('train.navitBucketHint') : t('train.bucketDistHint')}
        {native && dist.navit!.downscaled > 0 && <> {t('train.navitDistDownscaled', { n: dist.navit!.downscaled })}</>}
      </div>
    </div>
  )
}

/** Hint shown when the train set has masks but masked_loss is off (decision D7: hint only, don't auto-enable it). */
function MaskedLossHint({
  projectId, vid, maskedLoss,
}: {
  projectId: number
  vid: number
  maskedLoss: boolean
}) {
  const { t } = useTranslation()
  const [maskCount, setMaskCount] = useState(0)

  useEffect(() => {
    if (!projectId || !vid) return
    let cancelled = false
    api.listCropWorkspaceTrain(projectId, vid)
      .then((r) => {
        if (!cancelled) {
          setMaskCount(r.images.filter((im) => im.mask_mtime != null).length)
        }
      })
      .catch(() => { if (!cancelled) setMaskCount(0) })
    return () => { cancelled = true }
  }, [projectId, vid])

  if (maskCount === 0 || maskedLoss) return null
  return <div className="ds-note ds-info" style={{ fontSize: 11.5 }}>{t('train.maskedLossHint', { n: maskCount })}</div>
}

/** "Config map": every schema group with its field counts and state; a click
 *  opens the tab that holds it. */
function ConfigMap({ schema, groupFields, isChanged, isNonDefault, isLocked, openKey, tabOf, onOpen }: {
  schema: SchemaResponse
  groupFields: (key: string) => string[]
  isChanged: (name: string) => boolean
  isNonDefault: (name: string) => boolean
  isLocked: (name: string) => boolean
  openKey: TrainTabId
  tabOf: (key: string) => TrainTabId
  onOpen: (key: string) => void
}) {
  const { t } = useTranslation()
  const rows = schema.groups
    .map((g) => {
      const fields = groupFields(g.key)
      return {
        key: g.key,
        label: schemaGroupLabel(g.key, g.label, t),
        n: fields.length,
        changed: fields.filter(isChanged).length,
        nonDefault: fields.filter(isNonDefault).length,
        locked: fields.filter(isLocked).length,
      }
    })
    .filter((r) => r.n > 0)
  const total = rows.reduce((s, r) => s + r.n, 0)
  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div><div className="ds-card-title">{t('train.mapTitle')}</div><div className="ds-card-sub">{t('train.mapSub', { groups: rows.length, fields: total })}</div></div>
      </div>
      <table className="ds-tbl">
        <thead>
          <tr>
            <th>{t('train.mapSection')}</th>
            <th>{t('train.mapFields')}</th>
            <th>{t('train.mapChanged')}</th>
            <th>{t('train.mapNonDefault')}</th>
            <th>{t('train.mapStatus')}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const open = tabOf(r.key) === openKey
            return (
              <tr key={r.key} onClick={() => onOpen(r.key)} style={{ cursor: 'pointer', background: open ? 'var(--green-soft)' : undefined }}>
                <td>
                  <span className="ds-cell-main">
                    <span className="ds-sect-icon" style={open ? { background: '#fff' } : undefined}>{TrainIcon.sliders}</span>
                    <span style={{ fontWeight: open ? 600 : undefined }}>{r.label}<span className="ds-cell-key">{r.key}</span></span>
                  </span>
                </td>
                <td className="ds-num">{r.n}</td>
                <td className={`ds-num${r.changed ? '' : ' ds-muted'}`}>{r.changed || '—'}</td>
                <td className={`ds-num${r.nonDefault ? '' : ' ds-muted'}`}>{r.nonDefault || '—'}</td>
                <td>
                  {r.locked === r.n
                    ? <span className="ds-badge ds-err">{t('train.mapLocked')}</span>
                    : r.locked > 0
                      ? <span className="ds-badge ds-warn">{t('train.mapSomeLocked', { n: r.locked })}</span>
                      : open
                        ? <span className="ds-badge ds-ok">{t('train.mapOpen')}</span>
                        : <span className="ds-badge ds-ok">{t('train.mapActive')}</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function formatVal(v: unknown): string {
  if (v === null || v === undefined || v === '') return '—'
  if (Array.isArray(v)) return v.join(', ')
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

/** Edits of this session, newest first, each with a one-click revert. */
function RecentChanges({ changes, schema, onRevert }: {
  changes: ChangeEntry[]
  schema: SchemaResponse
  onRevert: (e: ChangeEntry) => void
}) {
  const { t, i18n } = useTranslation()
  const [, setTick] = useState(0)
  useEffect(() => {
    const id = window.setInterval(() => setTick((n) => n + 1), 30000)
    return () => window.clearInterval(id)
  }, [])
  const rel = new Intl.RelativeTimeFormat(i18n.language, { numeric: 'auto' })
  const ago = (at: number) => {
    const s = Math.round((Date.now() - at) / 1000)
    if (s < 60) return rel.format(0, 'minute')
    if (s < 3600) return rel.format(-Math.floor(s / 60), 'minute')
    return rel.format(-Math.floor(s / 3600), 'hour')
  }
  const groupLabel = (field: string) => {
    const g = schema.schema.properties[field]?.group ?? 'misc'
    const def = schema.groups.find((x) => x.key === g)
    return schemaGroupLabel(g, def?.label ?? g, t)
  }
  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div><div className="ds-card-title">{t('train.recentTitle')}</div><div className="ds-card-sub">{t('train.recentSub')}</div></div>
      </div>
      {changes.length === 0 ? (
        <div className="ds-muted" style={{ padding: '0 17px 16px', fontSize: 12.5 }}>{t('train.recentEmpty')}</div>
      ) : (
        <table className="ds-tbl">
          <thead>
            <tr><th>{t('train.recentField')}</th><th>{t('train.mapSection')}</th><th>{t('train.recentWhen')}</th><th>{t('train.recentValue')}</th><th style={{ width: 40 }} /></tr>
          </thead>
          <tbody>
            {changes.slice(0, 12).map((e) => (
              <tr key={e.field}>
                <td><span>{schemaFieldLabel(e.field, t)}<span className="ds-cell-key">{e.field}</span></span></td>
                <td className="ds-muted">{groupLabel(e.field)}</td>
                <td className="ds-muted">{ago(e.at)}</td>
                <td>
                  <span className="ds-diff">
                    <span className="ds-from">{formatVal(e.from)}</span><span className="ds-muted">→</span><span className="ds-to">{formatVal(e.to)}</span>
                  </span>
                </td>
                <td>
                  <button type="button" className="ds-kebab" onClick={() => onRevert(e)} aria-label={t('train.recentRevert')} title={t('train.recentRevert')}>{TrainIcon.reset}</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

/** Bottom strip: the version's latest training run (live epoch / loss while
 *  it runs), with its log tail. Hidden until the version has a run. */
function TrainLogBar({ project, vid }: { project: ProjectDetail; vid: number }) {
  const { t } = useTranslation()
  const [task, setTask] = useState<Task | null>(null)
  const [log, setLog] = useState<string[]>([])
  const load = useCallback(() => {
    let cancelled = false
    void api.listQueue()
      .then((items) => {
        if (cancelled) return
        const mine = items
          .filter((tk) => tk.project_id === project.id && tk.version_id === vid && (tk.task_type ?? 'train') === 'train')
          .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
        setTask(mine[0] ?? null)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [project.id, vid])
  useEffect(() => load(), [load])
  useEventStream((evt) => {
    if (evt.type === 'task_state_changed' || evt.type === 'train_loop_started') load()
  })
  const running = task?.status === 'running'
  const { state: monitor } = useMonitorProgress(running ? task!.id : null)
  const taskId = task?.id ?? null
  useEffect(() => {
    if (taskId == null) { setLog([]); return }
    let cancelled = false
    const pull = () => {
      api.getLog(taskId).then((r) => { if (!cancelled) setLog(r.content ? r.content.split('\n').slice(-400) : []) }).catch(() => {})
    }
    pull()
    if (!running) return () => { cancelled = true }
    const id = window.setInterval(pull, 10000)
    return () => { cancelled = true; window.clearInterval(id) }
  }, [taskId, running])
  if (!task) return null
  const lastLoss = monitor?.losses?.length ? monitor.losses[monitor.losses.length - 1].loss : null
  const detail = running
    ? [
        `#${task.id} train`,
        monitor?.epoch != null && monitor.total_epochs ? t('train.logbarEpoch', { e: monitor.epoch, total: monitor.total_epochs }) : null,
        lastLoss != null ? `loss ${lastLoss.toFixed(4)}` : null,
      ].filter(Boolean).join(' · ')
    : `#${task.id} train · ${t(`train.runStatus_${task.status}`, { defaultValue: task.status })}`
  const pct = running && monitor?.step != null && monitor.total_steps ? (monitor.step / monitor.total_steps) * 100 : null
  return <JobLogBar title={t('train.logbarTitle')} running={running} detail={detail} pct={pct} log={log} />
}
