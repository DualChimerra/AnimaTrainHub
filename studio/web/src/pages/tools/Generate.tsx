import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  TERMINAL_TASK_STATUSES,
  type GenerateRequest,
  type LoraEntry,
  type Task,
  type XYMatrixSpec,
} from '../../api/client'
import BaseModelSelect, { useBaseModelOptions, useKrea2TeOptions } from '../../components/BaseModelSelect'
import FieldLabel from '../../components/ds/FieldLabel'
import KebabMenu, { type KebabItem } from '../../components/ds/KebabMenu'
import PageHead from '../../components/ds/PageHead'
import { useToast } from '../../components/Toast'
import { schemaEnumLabel } from '../../lib/schema'
import { useEventStream } from '../../lib/useEventStream'
import { useMonitorProgress } from '../../lib/useMonitorProgress'
import { useLocalStorageState } from '../../lib/useLocalStorageState'
import { useQueueFormat } from '../../lib/queueFormat'
import AspectChips, { aspectFromDimensions, type AspectName } from './generate/AspectChips'
import DaemonControls from './generate/DaemonControls'
import DaemonLogDrawer from './generate/DaemonLogDrawer'
import GenerateProgressBar, { type GenerateProgress, type GeneratePhase } from './generate/GenerateProgress'
import NumField from './generate/NumField'
import PreviewCompare from './generate/PreviewCompare'
import ZoomableImage from '../../components/ZoomableImage'
import PreviewHistoryRail, { type TimelineItem } from './generate/PreviewHistoryRail'
import PromptFromDatasetPicker, { type DatasetPick } from './generate/PromptFromDatasetPicker'
import {
  PARAMS_SNAPSHOT_VERSION, applySnapshot, loraBasename, resolveLoraFromCkpts,
  transformAxisRawForSnapshot,
  type GenerateParamsSnapshot, type SnapshotLora,
} from './generate/paramsSnapshot'
import { saveSingleSamples, saveXYMatrix } from './generate/saveTestImages'
import { useGenerateHistory } from './generate/useGenerateHistory'
import {
  entryImageUrl,
  entryParams,
  entryTaskId,
  type HistoryEntry,
} from './generate/entryAdapter'
import PreviewXYGrid from './generate/PreviewXYGrid'
import PromptList from './generate/PromptList'
import NegPromptInput from './generate/NegPromptInput'
import SampleGallery from './generate/SampleGallery'
import SidebarLoras from './generate/SidebarLoras'
import SidebarXYAxes from './generate/SidebarXYAxes'
import StatusBadge from './generate/StatusBadge'
import ViewModeTabs, { type ViewMode } from './generate/ViewModeTabs'
import {
  DEFAULT_NEG, DEFAULT_SAMPLER, DEFAULT_SCHEDULER,
  DISTILLED_GENERATE_DEFAULTS, FAMILY_GENERATE_DEFAULTS,
  SAMPLER_OPTIONS_BY_FAMILY, SCHEDULER_OPTIONS_BY_FAMILY,
  type GenerateFamily, type SamplerName, type SchedulerName,
} from './generate/types'
import { useLoraCatalog } from './generate/useLoraCatalog'
import { buildXYMatrix, cellCount, parseAxisValues, type XYAxisDraft } from './generate/xy'

const GENERATE_PREFS_KEY = 'studio:generate:params:v1'

const DEFAULT_GENERATE_PREFS = {
  mode: 'single' as ViewMode,
  modelFamily: 'anima' as GenerateFamily,
  prompts: ['newest, safe, 1girl, masterpiece, best quality'],
  negPrompt: DEFAULT_NEG,
  aspect: '1:1' as AspectName,
  width: 1024,
  height: 1024,
  steps: 25,
  cfgScale: 4.0,
  samplerName: DEFAULT_SAMPLER as SamplerName,
  scheduler: DEFAULT_SCHEDULER as SchedulerName,
  seed: 0,
  // single / xy LoRA lists are fully independent (product decision 2026-05-29):
  // switching mode doesn't affect the other.
  // compare is a sub-view of xy and shares xyLoras with it.
  singleLoras: [] as LoraEntry[],
  xyLoras: [] as LoraEntry[],
  xDraft: { axis: 'steps', raw: '20, 25, 30', loraIndex: null } as XYAxisDraft,
  yDraft: null as XYAxisDraft | null,
  datasetPick: null as DatasetPick | null,
  // Explicit base model / TE overrides are also persisted (user feedback: resetting
  // to the global default every time you switch pages was annoying).
  // null = follow the Settings page's selected / selected_te (still the default).
  baseModel: null as string | null,
  textEncoder: null as 'bf16' | 'fp8' | null,
}

type GeneratePrefs = typeof DEFAULT_GENERATE_PREFS

/** Normalize / migrate persisted prefs (readPersisted doesn't merge in the default,
 *  so we have to backfill it ourselves):
 *  - Old versions only had a shared `loras` (single/xy shared it, which was exactly
 *    the bug being fixed) -> split into a copy each for singleLoras/xyLoras, so the
 *    migration doesn't lose any selected LoRA; after migration the two are independent.
 *  - Backfill missing fields (old shape / fields added in a later version).
 *  - Clamp xDraft/yDraft.loraIndex into xyLoras' valid range (the xy axis loraIndex
 *    points into xyLoras; out of range would make submit throw axisLoraMissing).
 */
function normalizePrefs(p: GeneratePrefs): GeneratePrefs {
  const anyP = p as Partial<GeneratePrefs> & { loras?: LoraEntry[]; count?: number }
  const legacy = Array.isArray(anyP.loras) ? anyP.loras : []
  const singleLoras = Array.isArray(anyP.singleLoras) ? anyP.singleLoras : legacy
  const xyLoras = Array.isArray(anyP.xyLoras) ? anyP.xyLoras : legacy
  const clampIdx = (d: XYAxisDraft | null): XYAxisDraft | null => {
    if (!d || d.loraIndex == null || d.loraIndex < xyLoras.length) return d
    return { ...d, loraIndex: xyLoras.length > 0 ? 0 : null }
  }
  const { loras: _legacy, count: _count, ...rest } = anyP  // count is now transient; drop the old persisted value
  const merged = {
    ...DEFAULT_GENERATE_PREFS,
    ...rest,
    singleLoras,
    xyLoras,
    xDraft: clampIdx(rest.xDraft ?? DEFAULT_GENERATE_PREFS.xDraft) ?? DEFAULT_GENERATE_PREFS.xDraft,
    yDraft: clampIdx(rest.yDraft ?? null),
  }
  // Family/sampler consistency (multi-model P4-4): when old prefs have no modelFamily,
  // or the persisted sampler doesn't match the current family's whitelist (a
  // cross-family value gets a 422 from the backend), fall back to the family default (first item).
  const family: GenerateFamily =
    merged.modelFamily === 'krea2' ? 'krea2' : 'anima'
  const samplers = SAMPLER_OPTIONS_BY_FAMILY[family] as readonly string[]
  const schedulers = SCHEDULER_OPTIONS_BY_FAMILY[family] as readonly string[]
  return {
    ...merged,
    modelFamily: family,
    samplerName: (samplers.includes(merged.samplerName)
      ? merged.samplerName : samplers[0]) as SamplerName,
    scheduler: (schedulers.includes(merged.scheduler)
      ? merged.scheduler : schedulers[0]) as SchedulerName,
    textEncoder: (merged.textEncoder === 'bf16' || merged.textEncoder === 'fp8')
      ? merged.textEncoder : null,
  }
}

export default function GeneratePage() {
  const { t } = useTranslation()
  const { toast } = useToast()

  const [rawPrefs, setRawPrefs] = useLocalStorageState(GENERATE_PREFS_KEY, DEFAULT_GENERATE_PREFS)
  const prefs = useMemo(() => normalizePrefs(rawPrefs), [rawPrefs])
  // Every setPrefs update normalizes prev first (migrates old shape + clamps), so the
  // updater always receives the new shape (with singleLoras/xyLoras, no legacy loras).
  const setPrefs = useCallback(
    (next: GeneratePrefs | ((p: GeneratePrefs) => GeneratePrefs)) =>
      setRawPrefs((prev) => {
        const norm = normalizePrefs(prev)
        return typeof next === 'function' ? next(norm) : next
      }),
    [setRawPrefs],
  )
  // One-time migration of the old shape (shared loras) into storage, so it doesn't
  // linger as a stale field; after this, reads see the clean singleLoras/xyLoras shape.
  useEffect(() => {
    const raw = rawPrefs as Partial<GeneratePrefs> & { loras?: unknown }
    if ('loras' in raw || !('singleLoras' in raw) || !('xyLoras' in raw)) {
      setRawPrefs(normalizePrefs(rawPrefs))
    }
    // Runs once on mount only: the migration is idempotent, no need to rerun on later rawPrefs changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const { mode, modelFamily, prompts, negPrompt, aspect, width, height, steps, cfgScale, samplerName, scheduler, seed, xDraft, yDraft, datasetPick } = prefs
  // LoRA lists are fully independent per mode: single uses singleLoras, xy (including
  // the compare sub-view) uses xyLoras. Reads/writes route by the current mode, so
  // switching mode doesn't affect the other.
  const loras = mode === 'single' ? prefs.singleLoras : prefs.xyLoras
  const setLoras = (loras: LoraEntry[]) =>
    setPrefs((p) => (p.mode === 'single' ? { ...p, singleLoras: loras } : { ...p, xyLoras: loras }))
  const setMode = (mode: ViewMode) => setPrefs((p) => ({ ...p, mode }))
  const setPrompts = (prompts: string[]) => setPrefs((p) => ({ ...p, prompts }))
  const setNegPrompt = (negPrompt: string) => setPrefs((p) => ({ ...p, negPrompt }))
  const setAspect = (aspect: AspectName) => setPrefs((p) => ({ ...p, aspect }))
  const setWidth = (width: number) => setPrefs((p) => ({ ...p, width }))
  const setHeight = (height: number) => setPrefs((p) => ({ ...p, height }))
  const setSteps = (steps: number) => setPrefs((p) => ({ ...p, steps }))
  const setCfgScale = (cfgScale: number) => setPrefs((p) => ({ ...p, cfgScale }))
  const setSamplerName = (samplerName: SamplerName) => setPrefs((p) => ({ ...p, samplerName }))
  const setScheduler = (scheduler: SchedulerName) => setPrefs((p) => ({ ...p, scheduler }))
  const setSeed = (seed: number) => setPrefs((p) => ({ ...p, seed }))
  /** Switch model family: sampler/scheduler/steps/cfg fall back to the target family's
   *  defaults (a cross-family value gets a 422 from the backend); the base model's
   *  temporary override is cleared (the variant key is family-specific). */
  const setModelFamily = (family: GenerateFamily) => {
    setBaseModel(null)
    setTextEncoder(null)
    setPrefs((p) => ({
      ...p,
      modelFamily: family,
      samplerName: SAMPLER_OPTIONS_BY_FAMILY[family][0] as SamplerName,
      scheduler: SCHEDULER_OPTIONS_BY_FAMILY[family][0] as SchedulerName,
      steps: FAMILY_GENERATE_DEFAULTS[family].steps,
      cfgScale: FAMILY_GENERATE_DEFAULTS[family].cfgScale,
    }))
  }
  // 0.17 P-I: batch size (number of tasks enqueued per click) is a **transient** UI
  // value -- it doesn't go into prefs, isn't persisted, and isn't backfilled when
  // clicking a history item (if the user sets 2, it stays 2); refreshing the page resets it to 1.
  const [batchSize, setBatchSize] = useState(1)

  // LoRA prefill via URL query (?lora=<path>&projectId=N&versionId=N)
  // When arriving here via the Overview StatusBanner's "Load in test" CTA, the URL
  // explicitly expresses "test this LoRA" intent -> it lands in the single mode list
  // (replacing it with [urlLora]) and switches to single; the xy list is independent
  // and unaffected (the xy axis loraIndex is already clamped into xyLoras by normalizePrefs).
  // Uses history.replaceState to clear the query so a refresh doesn't retrigger it.
  useEffect(() => {
    const sp = new URLSearchParams(window.location.search)
    const lora = sp.get('lora')
    if (!lora) return
    const projectId = sp.get('projectId')
    const versionId = sp.get('versionId')
    setPrefs((p) => {
      const newLoras: LoraEntry[] = [{
        path: lora,
        scale: 1.0,
        project_id: projectId ? Number(projectId) : null,
        version_id: versionId ? Number(versionId) : null,
      }]
      return { ...p, mode: 'single', singleLoras: newLoras }
    })
    const url = new URL(window.location.href)
    url.searchParams.delete('lora')
    url.searchParams.delete('projectId')
    url.searchParams.delete('versionId')
    window.history.replaceState({}, '', url.toString())
  }, [setPrefs])
  // Test generation omits attention_backend here; the server applies the
  // Comfy-style runtime and reads the configured generate backend there.

  const setXDraft = (xDraft: XYAxisDraft) => setPrefs((p) => ({ ...p, xDraft }))
  const setYDraft = (yDraft: XYAxisDraft | null) => setPrefs((p) => ({ ...p, yDraft }))
  const setDatasetPick = (datasetPick: DatasetPick | null) => setPrefs((p) => ({ ...p, datasetPick }))

  // Compare view: indices of the 2 selected samples (collected from PreviewXYGrid cell clicks)
  const [selectedIndices, setSelectedIndices] = useState<number[]>([])

  // submitting: HTTP enqueue in flight (a brief window before currentTask comes back)
  // busy is derived from currentTask.status, to avoid the UI getting stuck relying on
  // setBusy(false) to clear it -- with a plain useState we used to hit cases where a
  // missed SSE event / race left busy=true stuck, the button disabled, with no way to
  // retry or cancel (cancelable=false when status=failed)
  const [submitting, setSubmitting] = useState(false)
  // 0.17 P-I: currentTask = **the display target** (what the daemon is running / the
  // most recent one), no longer "the last submitted one". Submitting only enqueues;
  // the display follows whatever is running (refreshLiveGenerates).
  const [currentTask, setCurrentTask] = useState<Task | null>(null)
  // 0.17 P-I: running + pending generates submitted in this session (including our
  // own), driving the "N queued" list + running detection. Comes from
  // listQueueLive(undefined,'generate').
  const [liveGenerates, setLiveGenerates] = useState<Task[]>([])
  const prevGenIdsRef = useRef<Set<number>>(new Set())
  // #1: each task's "run state" frozen (XY axes + full params snapshot), stored at
  // dispatch time. The active result grid / compare view / ingestion read this
  // instead of the live prefs, so editing the sidebar after a task starts doesn't
  // retroactively change an already-dispatched result.
  // 0.17 P-I: single value -> stored in a Map keyed by taskId, each task reads its own.
  const runsRef = useRef<Map<number, {
    xDraft: XYAxisDraft
    yDraft: XYAxisDraft | null
    snapshot: GenerateParamsSnapshot
  }>>(new Map())
  // Base model / TE chosen for this generation (null = follow the Settings page's
  // selected / selected_te). Explicit overrides are persisted in prefs (user
  // feedback: a transient design that resets on every page switch was annoying).
  const baseModel = prefs.baseModel
  const setBaseModel = (v: string | null) => setPrefs((p) => ({ ...p, baseModel: v }))
  const textEncoder = prefs.textEncoder
  const setTextEncoder = (v: 'bf16' | 'fp8' | null) =>
    setPrefs((p) => ({ ...p, textEncoder: v }))
  const teOptions = useKrea2TeOptions()
  const effectiveTe = textEncoder ?? teOptions.selected
  // Don't send a TE override when the Settings page has a local encoder directory
  // selected (the request only accepts official variant keys); let the server
  // resolve the custom directory itself from selected_te.
  const teOverride = (
    modelFamily === 'krea2' && effectiveTe !== 'custom' ? effectiveTe : undefined
  )
  // Base model options for the current family (with purpose metadata) -- selecting a
  // distilled inference variant (Krea2 Turbo) applies the 8-step / no-CFG defaults
  // (still editable afterward, A1 doesn't restrict it)
  const { options: baseModelOptions } = useBaseModelOptions(modelFamily)
  const onBaseModelChange = (v: string) => {
    setBaseModel(v)
    const picked = baseModelOptions.find((o) => o.value === v)
    if (picked?.purpose === 'inference') {
      setPrefs((p) => ({
        ...p,
        steps: DISTILLED_GENERATE_DEFAULTS.steps,
        cfgScale: DISTILLED_GENERATE_DEFAULTS.cfgScale,
      }))
    }
  }
  // Monitoring goes through the useMonitorProgress hook (PR #37's incremental
  // protocol): when currentTask changes, the hook automatically refetches a snapshot
  // and subscribes to merge SSE deltas; this component only uses the samples field,
  // the rest aren't needed for this generation page's use case.
  const { state: monitorState } = useMonitorProgress(currentTask?.id ?? null)
  // commit 14: intermediate-step preview (only meaningful in single mode; less useful for multiple XY/compare cells)
  const [previewStep, setPreviewStep] = useState<{ step: number; total: number; dataUrl: string } | null>(null)
  // Generation progress (aggregated from image_started + preview_step)
  const [progress, setProgress] = useState<GenerateProgress>({
    phase: null, batchIdx: null, batchTotal: null, currentStep: null, totalSteps: null,
  })
  const [datasetPickerOpen, setDatasetPickerOpen] = useState(false)
  const [logOpen, setLogOpen] = useState(false)
  // While a GPU task like training / reg-ai / tagging is running, generation is
  // disabled to avoid VRAM contention (the driver grabbing the 3D / copy engine
  // stalls image rendering, or can even OOM the training process). listQueue
  // excludes generate tasks themselves by default, so generating doesn't self-lock.
  const [activeBlockingTask, setActiveBlockingTask] = useState<Task | null>(null)
  // commit 16: image history rail. Clicking a history item swaps the main preview to that item's cover.
  const history = useGenerateHistory()
  // 0.17 P-I: useGenerateHistory returns a new object on every render
  // (refresh/refreshCache aren't memoized). Use a ref to read the latest, keeping
  // ingestGenerateTask/refreshLiveGenerates deps stable, so the mount effect doesn't
  // rerun endlessly because their identity changes every render (a fetch storm).
  const historyRef = useRef(history)
  historyRef.current = history
  const [historyOverride, setHistoryOverride] = useState<HistoryEntry | null>(null)
  const taskIdRef = useRef<number | null>(null)
  taskIdRef.current = currentTask?.id ?? null
  const currentTaskRef = useRef<Task | null>(null)
  currentTaskRef.current = currentTask
  // 0.17 P-I: taskIds already ingested (dedup, replaces the old lastSnapshotRef).
  const ingestedRef = useRef<Set<number>>(new Set())

  // Clear the XY selection when switching to single (it's tied to XY results, meaningless in single mode)
  useEffect(() => {
    if (mode === 'single') setSelectedIndices([])
  }, [mode])

  // Selecting 2 -> auto-switches to compare; toggles an already-selected item; once 2 are selected, a new pick replaces the oldest
  const handleCellClick = (idx: number) => {
    setSelectedIndices((prev) => {
      if (prev.includes(idx)) return prev.filter((i) => i !== idx)
      if (prev.length >= 2) return [prev[1], idx]
      const next = [...prev, idx]
      // Selecting 2 automatically enters the compare sub-view inside xy (doesn't switch the top-level mode)
      // Current mode is already 'xy' (cell click only fires in xy mode), so no need to call setMode
      return next
    })
  }

  // Inside xy mode, switch to the compare sub-view when selectedIndices has 2 entries
  const showCompareView = mode === 'xy' && selectedIndices.length === 2

  const catalog = useLoraCatalog()
  // Use useMemo to keep a stable reference: when monitorState doesn't change, samples'
  // reference doesn't change either, avoiding the useEffect below rerunning
  // unnecessarily because samples is a dependency
  const samples = useMemo(() => monitorState?.samples ?? [], [monitorState])
  const samplesRef = useRef(samples)
  samplesRef.current = samples

  // #1: the active result grid uses the "axes frozen at dispatch time" rather than
  // the live xDraft/yDraft. When the displayed task has a frozen run (runsRef), use
  // the frozen values (so editing the sidebar after a task starts doesn't leak into
  // the right-hand side); otherwise fall back to live. runsRef is a ref, but
  // currentTask changing triggers a re-render -> this recomputes along with it, reactive enough.
  const frozenRun = currentTask ? runsRef.current.get(currentTask.id) ?? null : null
  const gridXDraft = frozenRun ? frozenRun.xDraft : xDraft
  const gridYDraft = frozenRun ? frozenRun.yDraft : yDraft

  // 0.17 P-I: the unified generate timeline = live queue (pending/running) union done
  // history (scanned from cache/disk), deduped by taskId (for the running->done
  // transition window). Live entries always sort to the top (latest submitted), done
  // below. Feeds the right-hand rail.
  // If the backend ever gets a dedicated endpoint for this, only this derivation
  // needs to change (the rest of the frontend stays untouched).
  const timelineItems = useMemo<TimelineItem[]>(() => {
    const doneIds = new Set(
      history.entries.map(entryTaskId).filter((x): x is number => x != null),
    )
    const done: TimelineItem[] = [...history.entries]
      .sort((a, b) => b.createdAt - a.createdAt)
      .map((entry) => ({ kind: 'done', entry }))
    const live: TimelineItem[] = [...liveGenerates]
      .filter((task) => !doneIds.has(task.id))
      .sort((a, b) => b.created_at - a.created_at)
      .map((task) => ({
        kind: 'live',
        task,
        mode: runsRef.current.get(task.id)?.snapshot.mode ?? 'single',
      }))
    return [...live, ...done]
  }, [liveGenerates, history.entries])

  // In XY mode, the button shows "Generate N x M = K images"
  const xyCellCount = useMemo(() => {
    if (mode !== 'xy') return 0
    try {
      const xLen = parseAxisValues(xDraft.axis, xDraft.raw).length
      const yLen = yDraft ? parseAxisValues(yDraft.axis, yDraft.raw).length : null
      return cellCount(xLen, yLen)
    } catch {
      return 0
    }
  }, [mode, xDraft, yDraft])

  const refreshBlockingTask = useCallback(async () => {
    try {
      const running = await api.listQueue('running')
      setActiveBlockingTask(running.length > 0 ? running[0] : null)
    } catch {
      // Don't block generation if fetching the queue fails -- a conservative fix, better to let it through than to falsely lock it.
    }
  }, [])

  // 0.17 P-I: ingest a given generate. **Each done task is ingested on its own,
  // decoupled from "which one is currently displayed"** (with multiple tasks,
  // currentTask follows whatever's running, it won't linger on each done one).
  // temp (default, save_test_images=off): the server already wrote the image +
  //   params into the encrypted cache on image_done -> just refreshCache to pull the new index.
  // disk (on): use that task's frozen run (runsRef) + samples to save to disk.
  //   samplesOverride: pass directly when the displayed task already has live
  //   samples, saving a getMonitorState call.
  const ingestGenerateTask = useCallback(async (taskId: number, samplesOverride?: typeof samples) => {
    if (ingestedRef.current.has(taskId)) return
    const sec = await api.getSecrets().catch(() => null)
    const saveToDisk = !!sec?.generate?.save_test_images
    if (!saveToDisk) {
      ingestedRef.current.add(taskId)
      await historyRef.current.refreshCache()
      return
    }
    const runSnap = runsRef.current.get(taskId)
    const snapMode = runSnap?.snapshot.mode
    if (snapMode !== 'single' && snapMode !== 'xy') return  // compare / missing run -> can't rebuild, don't mark (leave it for a later retry)
    let s = samplesOverride ?? []
    if (s.length === 0) {
      const st = await api.getMonitorState(taskId).catch(() => null)
      s = (st?.samples as typeof samples | undefined) ?? []
    }
    if (s.length === 0) return
    ingestedRef.current.add(taskId)
    const params = runSnap!.snapshot
    const filenames = s.map((x) => x.path.split(/[\\/]/).pop() ?? '').filter(Boolean)
    if (snapMode === 'single') {
      await saveSingleSamples(taskId, filenames, params)
    } else {
      const xd = runSnap!.xDraft
      const yd = runSnap!.yDraft
      const xValues = xd.raw.split(',').map((v) => v.trim()).filter(Boolean)
      const yValues = yd ? yd.raw.split(',').map((v) => v.trim()).filter(Boolean) : [null as string | null]
      const xySamples = s
        .filter((x): x is typeof x & { xy: NonNullable<typeof x.xy> } => x.xy != null)
        .map((x) => ({ path: x.path, xy: { xi: x.xy.xi, yi: x.xy.yi } }))
      await saveXYMatrix({
        samples: xySamples,
        taskId,
        xAxis: xd.axis as Parameters<typeof saveXYMatrix>[0]['xAxis'],
        yAxis: (yd?.axis ?? null) as Parameters<typeof saveXYMatrix>[0]['yAxis'],
        xValues,
        yValues,
      }, params)
    }
    await historyRef.current.refresh()
  }, [])

  // 0.17 P-I: fetch running+pending generates of this type (listQueueLive's type
  // param), driving the queued list + the display following running + ingesting each
  // item that just left the list (done/failed/canceled) on its own.
  const refreshLiveGenerates = useCallback(async () => {
    let items: Task[]
    try { items = await api.listQueueLive(undefined, 'generate') } catch { return }
    setLiveGenerates(items)
    const newIds = new Set(items.map((t) => t.id))
    // finished = was in live last time, isn't now = just finished/canceled.
    const finished = [...prevGenIdsRef.current].filter((id) => !newIds.has(id))
    prevGenIdsRef.current = newIds
    const cur = currentTaskRef.current
    const running = items.find((t) => t.status === 'running') ?? null
    if (running) {
      // The display follows whatever's currently running
      if (!cur || cur.id !== running.id) setCurrentTask(running)
    } else if (cur && finished.includes(cur.id)) {
      // No running task and the currently displayed one just finished -> fetch its
      // final frozen status badge (image samples are already on disk/cache)
      void api.getGenerateTask(cur.id).then(setCurrentTask).catch(() => {})
    }
    // Ingest each just-finished one individually (use live samples for the displayed one, saving a getMonitorState call)
    for (const id of finished) {
      void ingestGenerateTask(id, id === cur?.id ? samplesRef.current : undefined)
    }
  }, [ingestGenerateTask])

  useEffect(() => {
    void refreshBlockingTask()
    void refreshLiveGenerates()
  }, [refreshBlockingTask, refreshLiveGenerates])

  // SSE: task_state_changed triggers a task refresh; monitor_state_updated pushes the sample list.
  useEventStream((evt) => {
    if (evt.type === 'task_state_changed') {
      void refreshBlockingTask()
      // 0.17 P-I: display state + queued list + per-item ingestion are all driven by refreshLiveGenerates.
      void refreshLiveGenerates()
    }
    const tid = taskIdRef.current
    if (tid == null) return
    if (evt.type === 'task_state_changed' && evt.task_id === tid) {
      // Advancing currentTask is left to refreshLiveGenerates; here we only clear progress when the displayed task reaches a terminal state.
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        setProgress({ phase: null, batchIdx: null, batchTotal: null, currentStep: null, totalSteps: null })
      }
    } else if (
      evt.type === 'generate_phase'
      && String(evt.task_id) === String(tid)
    ) {
      // Phase advances (load/clip/sample/vae) -> the progress bar covers non-sampling phases too
      const name = typeof evt.name === 'string' ? (evt.name as GeneratePhase) : null
      setProgress((p) => ({ ...p, phase: name }))
    } else if (
      evt.type === 'generate_preview_step'
      && String(evt.task_id) === String(tid)
    ) {
      const step = Number(evt.step) || 0
      const total = Number(evt.total) || 0
      // Progress always updates
      setProgress((p) => ({ ...p, currentStep: step, totalSteps: total }))
      // image_b64 is optional (absent when preview isn't enabled in settings)
      if (typeof evt.image_b64 === 'string') {
        setPreviewStep({
          step, total,
          dataUrl: `data:image/jpeg;base64,${evt.image_b64}`,
        })
      }
    } else if (
      evt.type === 'generate_image_started'
      && String(evt.task_id) === String(tid)
    ) {
      // A new batch starts -> reset step progress, update the batch count (phase is driven by the following generate_phase events)
      setProgress({
        phase: null,
        batchIdx: typeof evt.batch_idx === 'number' ? evt.batch_idx : null,
        batchTotal: typeof evt.batch_total === 'number' ? evt.batch_total : null,
        currentStep: 0,
        totalSteps: typeof evt.total_steps === 'number' ? evt.total_steps : null,
      })
    }
  })

  // Clear the intermediate preview on task switch / completion / mode switch (the final image takes over)
  useEffect(() => {
    setPreviewStep(null)
  }, [currentTask?.id, mode, samples.length])

  // 0.17 P-I: **no longer** auto-clears the override when currentTask.id changes.
  // With multiple tasks, currentTask automatically follows running, so clearing the
  // override here would kick a done item the user is reviewing back to the live
  // view. Instead, only clear it on explicit user actions: clicking a running
  // timeline item (rail onSelect) -> clear; or switching mode (below) -> clear.
  // On mode switch, only clear an override that "belongs to a different mode":
  // manually switching mode still clears it (the rail buckets by mode, and
  // override.mode always equals the old mode != the new one -> clear); but for a
  // ?task= deep link into a task of a different mode, handleHistorySelect aligns
  // mode to entry.mode, so override.mode === new mode -> keep it.
  useEffect(() => {
    setHistoryOverride((cur) => (cur && cur.mode !== mode ? null : cur))
  }, [mode])


  const handleHistorySelect = (entry: HistoryEntry) => {
    setHistoryOverride(entry)  // Switch the image first (sync); sidebar backfill follows asynchronously as ckpts resolve
    // applySnapshot is the single entry point for all "apply snapshot" logic
    // (decision #8 / Step 3); now async: LoRA resolution fetches the version's ckpts
    // on demand (lazy cascade), not relying on the full mount-time list. An old entry
    // missing params falls through to the catch (accessing snap.loras etc. errors ->
    // no backfill, just the image switch).
    void (async () => {
    let applied
    try {
      const projects = await catalog.loadProjects()
      const projIds = new Set(projects.map((p) => p.id))
      applied = await applySnapshot(
        entry.params,
        async (snap) => {
          if (snap.project_id == null || snap.version_id == null) {
            return resolveLoraFromCkpts(snap, [])
          }
          const ckpts = await catalog
            .fetchCkpts(snap.project_id, snap.version_id)
            .catch(() => [])
          return resolveLoraFromCkpts(snap, ckpts)
        },
        (pid) => projIds.has(pid),
      )
    } catch {
      return
    }
    if (applied.unresolvedLoraCount > 0) {
      toast(t('generate.historyLorasMissing', { n: applied.unresolvedLoraCount }), 'info')
    }
    // When datasetPick is non-empty, auto-expand the picker so the user sees the
    // selected row + tags text (the picker is closed by default, and without
    // expanding it prompts[0] is often "" -- a common case where the user relies
    // entirely on dataset tags as the prompt -- which on the surface looks like
    // "nothing got backfilled"). The fallback path already pours the tags into
    // prompts[0] with datasetPick=null, so checking `applied` here is enough.
    if (applied.datasetPick) {
      setDatasetPickerOpen(true)
    }
    // The base model isn't in prefs (it's separate ephemeral state) -> backfill it separately.
    setBaseModel(applied.baseModel)
    setPrefs((prev) => {
      const base: GeneratePrefs = {
        ...prev,
        mode: applied.mode,
        modelFamily: applied.modelFamily,
        prompts: applied.prompts.length > 0 ? applied.prompts : prev.prompts,
        negPrompt: applied.negPrompt,
        width: applied.width,
        height: applied.height,
        aspect: aspectFromDimensions(applied.width, applied.height),
        steps: applied.steps,
        cfgScale: applied.cfgScale,
        samplerName: applied.samplerName,
        scheduler: applied.scheduler,
        seed: applied.seed,
        datasetPick: applied.datasetPick,
        // 0.17 P-I: batch size is transient, clicking a history item does **not** backfill it (the user's set value stays as is).
      }
      if (applied.mode === 'single') {
        return { ...base, singleLoras: applied.loras }
      }
      return {
        ...base,
        xyLoras: applied.loras,
        xDraft: applied.xDraft ?? prev.xDraft,
        yDraft: applied.yDraft ?? null,
      }
    })
    })()
  }

  // 0.17 P-H deep-link review: Queue Detail's "View generate result" -> /tools/generate?task=<id>.
  // A Task doesn't carry mode/params -- only a generate history entry does -- so once
  // history loads we match by task_id and go through the existing historyOverride
  // review path (handleHistorySelect aligns mode + backfills the sidebar).
  const deepLinkTaskId = useMemo(() => {
    const v = new URLSearchParams(window.location.search).get('task')
    const n = v ? Number(v) : NaN
    return Number.isFinite(n) ? n : null
  }, [])
  const deepLinkConsumedRef = useRef(false)
  useEffect(() => {
    if (deepLinkTaskId == null || deepLinkConsumedRef.current || history.loading) return
    deepLinkConsumedRef.current = true
    // Clear the query so a refresh doesn't retrigger it (same pattern as ?lora=)
    const url = new URL(window.location.href)
    url.searchParams.delete('task')
    window.history.replaceState({}, '', url.toString())
    const entry = history.entries.find((e) => entryTaskId(e) === deepLinkTaskId)
    if (entry) handleHistorySelect(entry)
    // No image source left (cache evicted within the same session / disk save wasn't on) = physically can't review it, so show a fallback toast.
    else toast(t('generate.taskResultUnavailable', { id: deepLinkTaskId }), 'info')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deepLinkTaskId, history.loading, history.entries])

  const handleGenerate = async () => {
    const datasetSuffix = datasetPick && datasetPick.tags.length > 0
      ? datasetPick.tags.join(', ')
      : ''
    if (!prompts.some((p) => p.trim()) && !datasetSuffix) {
      toast(t('generate.promptOrDatasetRequired'), 'error')
      return
    }

    let xy_matrix: XYMatrixSpec | null = null
    // single: base LoRA = send all of singleLoras. xy: only send the anchors
    // referenced by an axis (see buildXYMatrix -- xyLoras accumulates orphaned
    // anchors left behind by switching project/version in the picker or deleting an
    // axis; sending the whole bucket would stack orphans onto every cell, which was
    // the root cause of the recurring "an unselected LoRA snuck in" bug).
    let loraConfigs: LoraEntry[] = loras.filter((l) => l.path.trim())
    if (mode === 'xy') {
      // The schema forces a single prompt + count=1
      if (prompts.filter((p) => p.trim()).length > 1) {
        toast(t('generate.xySinglePromptOnly'), 'error')
        return
      }
      try {
        const built = buildXYMatrix(xDraft, yDraft, loras)
        xy_matrix = built.xy_matrix
        loraConfigs = built.loraConfigs
      } catch (e) {
        toast(typeof e === 'string' ? e : String(e), 'error')
        return
      }
    }

    // 0.17 P-I: submitting only enqueues, it **doesn't clear or hijack the display**
    // -- the display follows whatever's currently running, and the new submission
    // goes to the back of the queue (the daemon runs them one by one). The old
    // setCurrentTask(null)/setRun(null)/clearing selection & progress, which
    // interrupted the image currently being generated, has been removed.
    setSubmitting(true)
    try {
      // Concatenation order: hand-written positive prompt first, dataset tags after (matches the product's agreed convention)
      const baseTrimmed = prompts.map((p) => p.trim()).filter((p) => p)
      const mergedPrompts = datasetSuffix
        ? (baseTrimmed.length > 0
            ? baseTrimmed.map((p) => `${p}, ${datasetSuffix}`)
            : [datasetSuffix])
        : baseTrimmed
      // Send the snapshot to the server along with dispatch: on image_done it's
      // stuffed into the encrypted cache payload header (save=false) and returned on
      // list_index for backfilling. The save=true (disk) branch still builds its own
      // via saveSingleSamples/saveXYMatrix; the fields on both sides are kept aligned.
      const snapshotLoras: SnapshotLora[] = loras.map((l) => ({
        name: loraBasename(l.path),
        scale: l.scale,
        project_id: l.project_id ?? null,
        version_id: l.version_id ?? null,
      }))
      const baseSnapshot: GenerateParamsSnapshot = {
        schema_version: PARAMS_SNAPSHOT_VERSION,
        mode,
        model_family: modelFamily,
        prompts,
        negative_prompt: negPrompt,
        width, height, steps,
        cfg_scale: cfgScale,
        sampler_name: samplerName,
        scheduler,
        count: 1,  // 0.17 P-I: each task produces 1 image; batch is split into multiple tasks (loop below)
        seed,
        base_model: baseModel,
        text_encoder: teOverride,
        loras: snapshotLoras,
        xy_draft: mode === 'xy'
          ? {
              x: transformAxisRawForSnapshot(xDraft),
              y: yDraft ? transformAxisRawForSnapshot(yDraft) : null,
            }
          : null,
        dataset_pick: datasetPick,
      }
      // 0.17 P-I: count now = **batch size** (number of tasks enqueued per click).
      // single is split into `batch` tasks (each producing 1 image, seeds incrementing
      // to distinguish them) -> queued one by one in the right-hand timeline; xy is one
      // matrix per submission (batch is ignored).
      const batch = mode === 'xy' ? 1 : Math.max(1, batchSize)
      let firstId: number | null = null
      for (let i = 0; i < batch; i++) {
        const taskSeed = seed + i
        const snap: GenerateParamsSnapshot = { ...baseSnapshot, seed: taskSeed }
        const body: GenerateRequest = {
          prompts: mergedPrompts,
          model_family: modelFamily,
          base_model: baseModel ?? undefined,
          text_encoder: teOverride,
          negative_prompt: negPrompt,
          width, height, steps,
          count: 1,
          seed: taskSeed,
          cfg_scale: cfgScale,
          sampler_name: samplerName,
          scheduler,
          lora_configs: loraConfigs,
          // attention_backend omitted: the server applies the Comfy-style runtime and reads the generate backend itself.
          xy_matrix,
          params_snapshot: snap as unknown as Record<string, unknown>,
        }
        const task = await api.enqueueGenerate(body)
        // #1 + P-I: each task's frozen run state is stored in the Map (xDraft/yDraft
        // are shallow-copied plain objects, isolated from later edits; each snapshot
        // carries its own seed). Display/ingestion each read by taskId.
        runsRef.current.set(task.id, {
          xDraft: { ...xDraft }, yDraft: yDraft ? { ...yDraft } : null, snapshot: snap,
        })
        if (firstId === null) {
          firstId = task.id
          // Clicking "Start generating" = the user clearly wants to see this
          // generation -> return to the live view: clear the history override
          // currently being reviewed (otherwise the result area stays on the old
          // image and doesn't show the newly queued/running one, especially with XY --
          // generation is slow, and users often click generate while still in the
          // review state). P-I removed the effect that auto-cleared the override
          // when currentTask.id changed (with multiple tasks, that would kick a done
          // item being reviewed back to live); this now clears it only on an explicit
          // user submission, covering both cases.
          setHistoryOverride(null)
          // First generation (nothing currently displayed): optimistically set it to
          // the first task, so the user immediately sees "queued/started" instead of a blank screen.
          if (!currentTaskRef.current || TERMINAL_TASK_STATUSES.includes(currentTaskRef.current.status)) {
            setCurrentTask(task)
          }
        }
      }
      void refreshLiveGenerates()
      toast(
        batch > 1
          ? t('generate.batchEnqueued', { n: batch })
          : t('generate.taskEnqueued', { id: firstId ?? 0 }),
        'success',
      )
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setSubmitting(false)
    }
  }

  const handleCancel = async () => {
    if (!currentTask) return
    try {
      await api.cancelTask(currentTask.id)
      toast(t('generate.cancelRequested', { id: currentTask.id }), 'info')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // 0.17 P-I: cancel a single queued generate (the x on a live timeline item).
  const cancelQueued = async (id: number) => {
    try {
      await api.cancelTask(id)
      toast(t('generate.cancelRequested', { id }), 'info')
      void refreshLiveGenerates()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // 0.17 P-I: clear the queue -- cancel all pending generates (leave the running one alone).
  const pendingGenerateIds = useMemo(
    () => liveGenerates.filter((t) => t.status === 'pending').map((t) => t.id),
    [liveGenerates],
  )
  const clearQueue = async () => {
    if (pendingGenerateIds.length === 0) return
    await Promise.allSettled(pendingGenerateIds.map((id) => api.cancelTask(id)))
    toast(t('generate.queueCleared', { n: pendingGenerateIds.length }), 'info')
    void refreshLiveGenerates()
  }

  const cancelable = currentTask
    && (currentTask.status === 'pending' || currentTask.status === 'running')

  // busy derivation: HTTP enqueue in flight OR the task is still pending/running.
  // Any terminal status (done/failed/canceled) is always busy=false, so the button is
  // immediately clickable to retry.
  const busy: boolean = submitting || Boolean(cancelable)

  // 0.17 P-I: the button is now clickable even while a generation is in progress
  // (submitting a new task to the queue), so the label only shows "Generating" during
  // this submission's HTTP window (submitting), and shows the action label otherwise.
  const generateLabel = submitting
    ? t('generate.generating')
    : mode === 'xy' && xyCellCount > 0
      ? t('generate.startGenerateCount', { n: xyCellCount })
      : t('generate.startGenerate')

  const fmt = useQueueFormat()
  const fmtRun = (sec: number) => (sec < 60 ? t('generate.secShort', { n: Math.max(0, sec).toFixed(1) }) : fmt.dur(sec))

  // What the result card describes: the reviewed history entry, else the run
  // frozen at dispatch for the displayed task, else the live sidebar values.
  const shownParams: GenerateParamsSnapshot | null = historyOverride
    ? entryParams(historyOverride)
    : frozenRun?.snapshot ?? null
  const shown = {
    loras: shownParams ? shownParams.loras.map((l) => ({ name: l.name, scale: l.scale })) : loras.filter((l) => l.path).map((l) => ({ name: loraBasename(l.path), scale: l.scale })),
    seed: shownParams?.seed ?? seed,
    steps: shownParams?.steps ?? steps,
    cfg: shownParams?.cfg_scale ?? cfgScale,
    sampler: shownParams?.sampler_name ?? samplerName,
    scheduler: shownParams?.scheduler ?? scheduler,
  }
  const firstLora = shown.loras[0]
  const resultSub = [
    firstLora
      ? `${firstLora.name.replace(/\.safetensors$/i, '')} · ${t('generate.weightShort', { w: firstLora.scale.toFixed(2) })}`
      : t('generate.noLora'),
    shown.loras.length > 1 ? `+${shown.loras.length - 1}` : null,
    `seed ${shown.seed}`,
  ].filter(Boolean).join(' · ')
  const paramsLine = `${t('generate.stepsN', { count: shown.steps })} · guidance ${shown.cfg.toFixed(1)} · ${schemaEnumLabel('sample_sampler_name', shown.sampler, t)} / ${schemaEnumLabel('sample_scheduler', shown.scheduler, t)}`
  const footerStatus = historyOverride || !currentTask || busy
    ? null
    : currentTask.status === 'done'
      ? {
          text: currentTask.started_at && currentTask.finished_at
            ? t('generate.doneIn', { time: fmtRun(currentTask.finished_at - currentTask.started_at) })
            : t('status.done'),
          tone: 'ok' as const,
        }
      : currentTask.status === 'failed'
        ? { text: currentTask.error_msg || t('status.failed'), tone: 'err' as const }
        : currentTask.status === 'canceled'
          ? { text: t('status.canceled') }
          : null

  const resultMenu: KebabItem[] = [
    { label: t('generate.cancelCurrentTitle'), onSelect: () => void handleCancel(), disabled: !cancelable },
    {
      label: pendingGenerateIds.length > 0 ? t('generate.clearQueue', { n: pendingGenerateIds.length }) : t('generate.clearQueueEmpty'),
      onSelect: () => void clearQueue(),
      disabled: pendingGenerateIds.length === 0,
    },
  ]

  // Sampler and scheduler are picked together, as "euler / simple".
  const samplerCombos = SAMPLER_OPTIONS_BY_FAMILY[modelFamily].flatMap((s) =>
    SCHEDULER_OPTIONS_BY_FAMILY[modelFamily].map((sc) => ({ s, sc })))

  return (
    <div className="fade-in" style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <PageHead
        eyebrow={t('generate.eyebrow')}
        title={t('generate.title')}
        subtitle={t('generate.subtitle')}
        tools={
          <>
            <ViewModeTabs mode={mode} onModeChange={setMode} />
            {/* 0.17 P-I: batch size (number of tasks enqueued per click); xy is one matrix per submission, not applicable. */}
            {mode !== 'xy' && (
              <input
                type="number"
                className="ds-inp ds-mono"
                style={{ width: 58, textAlign: 'center' }}
                min={1} max={32}
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                title={t('generate.batchSizeTitle')}
                aria-label={t('generate.batchSizeTitle')}
              />
            )}
            {/* R-5: not hard-disabled while a GPU task is running -- the backend's admission control guarantees mutual exclusion, submitting just enqueues. */}
            <button
              type="button"
              className="ds-btn-primary"
              onClick={handleGenerate}
              disabled={submitting}
              title={activeBlockingTask ? t('generate.queuedBehindActiveTask', { id: activeBlockingTask.id }) : undefined}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden><path d="M8 5v14l11-7z" /></svg>
              {generateLabel}
            </button>
          </>
        }
      />

      <div className="ds-scroll ds-tight" style={{ minHeight: 0 }}>
        <div className="ds-gen-grid">

          {/* Settings */}
          <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
            <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 13, minHeight: 0, overflowY: 'auto', scrollbarGutter: 'stable' }}>

              <div>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: 7 }}>
                  <span className="ds-cap" style={{ flex: 1 }}>{t('generate.positive')}</span>
                  {!datasetPickerOpen && (
                    <button
                      type="button"
                      className="ds-ctl-note"
                      style={{ color: 'var(--green-text)' }}
                      onClick={() => setDatasetPickerOpen(true)}
                      title={t('generate.pickFromDatasetTitle')}
                    >
                      {t('generate.pickFromDataset')}
                    </button>
                  )}
                </div>
                {datasetPickerOpen && (
                  <div style={{ marginBottom: 8 }}>
                    <PromptFromDatasetPicker
                      value={datasetPick}
                      onChange={setDatasetPick}
                      onClose={() => {
                        setDatasetPick(null)
                        setDatasetPickerOpen(false)
                      }}
                    />
                  </div>
                )}
                <PromptList prompts={prompts} onChange={setPrompts} modelFamily={modelFamily} />
              </div>

              <div>
                <div className="ds-cap" style={{ marginBottom: 7 }}>{t('generate.negative')}</div>
                <NegPromptInput value={negPrompt} onChange={setNegPrompt} modelFamily={modelFamily} />
              </div>

              {/* single → LoRA slots; xy → axes (LoRA picking is part of the axes) */}
              <div>
                {mode === 'single' ? (
                  <>
                    <div className="ds-cap" style={{ marginBottom: 8 }}>
                      <FieldLabel label="LoRA" tip={t('generate.loraHint')} />
                    </div>
                    <SidebarLoras loras={loras} onChange={setLoras} catalog={catalog} />
                  </>
                ) : (
                  <SidebarXYAxes
                    xDraft={xDraft}
                    yDraft={yDraft}
                    onXChange={setXDraft}
                    onYChange={setYDraft}
                    loras={loras}
                    onLorasChange={setLoras}
                    catalog={catalog}
                  />
                )}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                <NumField label={t('generate.width')} value={width} onChange={(v) => { setWidth(v); setAspect(aspectFromDimensions(v, height)) }} min={256} max={4096} step={64} />
                <NumField label={t('generate.height')} value={height} onChange={(v) => { setHeight(v); setAspect(aspectFromDimensions(width, v)) }} min={256} max={4096} step={64} />
                <NumField label={t('generate.steps')} value={steps} onChange={setSteps} min={1} max={150} />
                <NumField label={t('generate.guidance')} value={cfgScale} onChange={setCfgScale} min={0} max={20} step={0.5} />
                <NumField
                  label={t('generate.seed')}
                  tip={t('generate.seedHint')}
                  value={seed}
                  onChange={setSeed}
                  min={0}
                  onRandom={() => setSeed(1 + Math.floor(Math.random() * 2 ** 31))}
                />
                <div style={{ minWidth: 0 }}>
                  <div className="ds-cap" style={{ marginBottom: 6 }}>{t('generate.sampler')}</div>
                  {/* Copy shares the schema.enums.* mapping with the training config page; options are filtered by the family whitelist */}
                  <select
                    className="ds-inp"
                    value={`${samplerName}|${scheduler}`}
                    onChange={(e) => {
                      const [s, sc] = e.target.value.split('|')
                      setSamplerName(s as SamplerName)
                      setScheduler(sc as SchedulerName)
                    }}
                    aria-label={t('generate.sampler')}
                  >
                    {samplerCombos.map(({ s, sc }) => (
                      <option key={`${s}|${sc}`} value={`${s}|${sc}`}>
                        {schemaEnumLabel('sample_sampler_name', s, t)} / {schemaEnumLabel('sample_scheduler', sc, t)}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="ds-ctl-note" style={{ marginTop: -4, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ flex: 1 }}>{t('generate.sizeNote', { mp: ((width * height) / 1e6).toFixed(2) })}</span>
                <button
                  type="button"
                  className="ds-ctl-note"
                  style={{ color: 'var(--green-text)' }}
                  onClick={() => {
                    const newW = height, newH = width
                    setWidth(newW); setHeight(newH)
                    setAspect(aspectFromDimensions(newW, newH))
                  }}
                  title={t('generate.swapSizeTitle')}
                >
                  {t('generate.swapSize')}
                </button>
              </div>

              <div>
                <div className="ds-cap" style={{ marginBottom: 7 }}>{t('generate.aspect')}</div>
                <AspectChips
                  aspect={aspect}
                  onPick={(a, w, h) => {
                    setAspect(a)
                    if (w && h) { setWidth(w); setHeight(h) }
                  }}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, borderTop: '1px solid var(--line)', paddingTop: 13 }}>
                <div className="ds-cap">{t('generate.modelSection')}</div>
                <select
                  className="ds-inp"
                  value={modelFamily}
                  onChange={(e) => setModelFamily(e.target.value as GenerateFamily)}
                  aria-label={t('generate.modelFamily')}
                  title={t('generate.modelFamily')}
                >
                  <option value="anima">{schemaEnumLabel('model_family', 'anima', t)}</option>
                  <option value="krea2">{schemaEnumLabel('model_family', 'krea2', t)}</option>
                </select>
                <div>
                  <div className="ds-cap" style={{ marginBottom: 6 }}>
                    <FieldLabel label={t('generate.baseModel')} tip={t('generate.baseModelHint')} />
                  </div>
                  <BaseModelSelect
                    value={baseModel}
                    onChange={onBaseModelChange}
                    family={modelFamily}
                    className="ds-inp"
                    ariaLabel={t('generate.baseModel')}
                  />
                </div>
                {modelFamily === 'krea2' && (
                  <div>
                    <div className="ds-cap" style={{ marginBottom: 6 }}>{t('generate.textEncoder')}</div>
                    <select
                      className="ds-inp"
                      value={effectiveTe}
                      onChange={(e) => setTextEncoder(
                        // "Custom" = no override, follows the Settings page's selected_te
                        e.target.value === 'custom'
                          ? null
                          : e.target.value as 'bf16' | 'fp8',
                      )}
                      aria-label={t('generate.textEncoder')}
                    >
                      {effectiveTe === 'custom' && (
                        <option value="custom">{t('generate.textEncoderCustom')}</option>
                      )}
                      <option value="bf16">{t('generate.textEncoderBf16')}</option>
                      <option value="fp8" disabled={!teOptions.fp8Ready}>
                        {teOptions.fp8Ready
                          ? t('generate.textEncoderFp8')
                          : t('generate.textEncoderFp8NotDownloaded')}
                      </option>
                    </select>
                  </div>
                )}
              </div>

            </div>
          </div>

          {/* Result */}
          <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
            <div className="ds-card-head ds-pad">
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="ds-card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  {t('generate.results')}
                  {currentTask && !historyOverride && (
                    <>
                      <span className="ds-mono ds-muted" style={{ fontSize: 11.5, fontWeight: 400 }}>#{currentTask.id}</span>
                      <StatusBadge status={currentTask.status} />
                    </>
                  )}
                </div>
                <div className="ds-card-sub ds-mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{resultSub}</div>
              </div>
              <div className="ds-card-tools">
                <KebabMenu trigger="icon" label={t('generate.moreActions')} items={resultMenu} />
              </div>
            </div>
            <div className="ds-card-body" style={{ paddingTop: 2, flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              {historyOverride ? (
                <div className="flex-1 min-h-0 flex flex-col gap-2">
                  {historyOverride.mode === 'xy' && historyOverride.xyMeta ? (
                    /* XY review (shared by cache / disk): per-cell info is complete ->
                       PreviewXYGrid. For cache, taskId is the real task id (GridCell
                       fallback goes through the cache URL); for disk, the server
                       already provides imageUrl, taskId uses a -1 sentinel (never
                       used). For disk, compositeUrl is also passed -> exporting PNG
                       downloads the file directly instead of re-composing. */
                    <PreviewXYGrid
                      samples={historyOverride.xyMeta.samples.map((s) => ({
                        path: s.path,
                        xy: {
                          xi: s.xy.xi, yi: s.xy.yi,
                          xv: s.xy.xv as never, yv: s.xy.yv as never,
                        },
                        imageUrl: s.imageUrl,
                      }))}
                      taskId={historyOverride.source === 'cache' ? historyOverride.taskId : -1}
                      xDraft={{
                        axis: historyOverride.xyMeta.xAxis as never,
                        raw: historyOverride.xyMeta.xValues.join(', '),
                        loraIndex: null,
                      }}
                      yDraft={historyOverride.xyMeta.yAxis ? {
                        axis: historyOverride.xyMeta.yAxis as never,
                        raw: (historyOverride.xyMeta.yValues as string[]).filter(Boolean).join(', '),
                        loraIndex: null,
                      } : null}
                      onCellClick={undefined /* History review doesn't allow selecting a cell to enter compare */}
                      selectedIndices={[]}
                      compositeUrl={historyOverride.source === 'disk' ? historyOverride.imageUrl : undefined}
                    />
                  ) : (
                    /* DiskEntry single / legacy XY (no xyMeta) / CacheEntry single -> single image view */
                    <div className="flex-1 min-h-0 w-full">
                      <ZoomableImage
                        key={historyOverride.id}
                        src={entryImageUrl(historyOverride, 0)}
                        alt=""
                      />
                    </div>
                  )}
                  {historyOverride.source === 'disk' && historyOverride.xyMeta && (
                    <div className="ds-cell-key" style={{ flex: 'none' }}>
                      {historyOverride.folder ?? (historyOverride.filename ?? '').replace(/\.png$/i, '')}
                    </div>
                  )}
                </div>
              ) : !currentTask ? (
                <div className="ds-thumb" style={{ flex: 1, minHeight: 200, fontSize: 12 }}>
                  {t('generate.emptyHint')}
                </div>
              ) : mode === 'xy' && showCompareView ? (
                /* Sub-view inside xy: switches to compare when 2 are selected (doesn't switch the top-level mode) */
                <PreviewCompare
                  samples={samples}
                  taskId={currentTask.id}
                  selectedIndices={selectedIndices as [number, number]}
                  xDraft={gridXDraft}
                  yDraft={gridYDraft}
                  onBack={() => setSelectedIndices([])}
                />
              ) : mode === 'xy' ? (
                <PreviewXYGrid
                  samples={samples}
                  taskId={currentTask.id}
                  xDraft={gridXDraft}
                  yDraft={gridYDraft}
                  onCellClick={handleCellClick}
                  selectedIndices={selectedIndices}
                />
              ) : samples.length === 0 && previewStep ? (
                <div className="flex-1 min-h-0 flex flex-col items-center gap-2">
                  <div className="flex-1 min-h-0 w-full flex items-center justify-center">
                    {/* The intermediate-step preview is a low-res latent2rgb image: fills the result area (object-contain scales it up while preserving aspect ratio) */}
                    <img
                      src={previewStep.dataUrl}
                      alt={`step ${previewStep.step}/${previewStep.total}`}
                      style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: 10 }}
                    />
                  </div>
                  <div className="ds-cell-key" style={{ flex: 'none' }}>
                    {t('generate.previewStep', { step: previewStep.step, total: previewStep.total })}
                  </div>
                </div>
              ) : samples.length === 0 ? (
                <div className="ds-thumb" style={{ flex: 1, minHeight: 200, fontSize: 12 }}>
                  {busy ? t('generate.waitingImages') : t('generate.finishedNoImages')}
                </div>
              ) : (
                <SampleGallery samples={samples} taskId={currentTask.id} />
              )}
            </div>
            <GenerateProgressBar busy={busy} progress={progress} status={footerStatus} params={paramsLine} />
          </div>

          {/* History: live queue + done results, bucketed by the current mode */}
          <PreviewHistoryRail
            items={timelineItems}
            mode={mode}
            selectedId={historyOverride?.id ?? null}
            onSelect={(it) => {
              if (it.kind === 'done') handleHistorySelect(it.entry)
              // running item: clear the override to return to the live view (currentTask already follows running).
              else if (it.task.status === 'running') setHistoryOverride(null)
              // pending item: no content, not selectable (can only be canceled).
            }}
            onCancel={cancelQueued}
            onRefresh={history.refresh}
            loading={history.loading}
          />
        </div>
      </div>

      <DaemonControls queued={pendingGenerateIds.length} logOpen={logOpen} onToggleLog={() => setLogOpen((v) => !v)} />

      {/* Daemon log drawer (fixed positioning + translateY; fully invisible and takes no layout space when hidden) */}
      <DaemonLogDrawer open={logOpen} onClose={() => setLogOpen(false)} />
    </div>
  )
}

