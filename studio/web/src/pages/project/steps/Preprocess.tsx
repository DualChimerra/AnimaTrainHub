import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type Job,
  type ProjectDetail,
  type UpscalerVariant,
  type Version,
} from '../../../api/client'
import { parseFolderMeta } from '../../../lib/folderMeta'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import PreprocessToolsBar, { PreprocessCard, PreprocessHeadTools } from '../../../components/preprocess/PreprocessToolsBar'
import StepShell from '../../../components/StepShell'
import { PX_BINS, pxBinFor, type PxBinId } from '../../../lib/pixelBins'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

interface Status {
  job: Job | null
  log_tail: string
  summary: { image_count: number }
}

interface FilesView {
  /** ADR 0010: all images in the train set (replaces the old pending+processed binary). */
  images: import('../../../api/client').TrainImage[]
  summary: Status['summary']
}

/** Single-image view: a status-tagged list derived from the train manifest (ADR 0010).
 *
 *  ADR 0010: from the user's perspective there is one set of images; "unprocessed / processed" is
 *  a badge on the image rather than a grouping. Status is inferred from a diff in entry fields
 *  (rel path's last segment ≠ origin → processed; same → pending/unchanged).
 */
interface ImageRow {
  /** train rel path "1_data/X.png" (used as the selection key + manifest entry key). */
  name: string
  /** the final filename segment of name (used for restore/crop actions + the thumb URL). */
  filename: string
  /** the folder segment of name (used for the thumb URL). */
  folder: string
  status: 'pending' | 'processed'
  processed?: import('../../../api/client').TrainImage
  size: number
  w: number | null
  h: number | null
  mtime: number
}

/** Pixel-area histogram bins — shared by the sidebar histogram + grid filter chips +
 *  the Overview detail tab. Definition/logic lives in lib/pixelBins.ts. */
type FilterMode = 'all' | PxBinId

const FALLBACK_MODEL = '4x-AnimeSharp'
const TILE_OPTIONS = [128, 192, 256, 384, 512] as const
type Device = 'auto' | 'cuda' | 'cpu'

// Target resolution presets -- areas commonly used by LoRA training buckets.
// value=null is "smart off" mode: take the model's raw 4x output directly (the old path, costly on disk).
const DEFAULT_TARGET_EDGE = 1024

export default function PreprocessPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const vid = activeVersion?.id ?? 0

  const [files, setFiles] = useState<FilesView | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [tileSize, setTileSize] = useState<number>(256)
  const [device, setDevice] = useState<Device>('auto')
  // targetEdge: edge length in pixels, squared gives the area; null = smart off; 0 = custom in progress
  const [targetEdge, setTargetEdge] = useState<number | null>(DEFAULT_TARGET_EDGE)
  const [customEdge, setCustomEdge] = useState<string>(String(DEFAULT_TARGET_EDGE))
  const [filter, setFilter] = useState<FilterMode>('all')
  const [folderFilter, setFolderFilter] = useState<string>('all')
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [selAnchor, setSelAnchor] = useState<string | null>(null)
  // Large preview: index refers into visibleRows[] (the currently visible ImageRow list under filter)
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)

  // Model weight readiness (fetched once from the catalog; updated by manual refresh or SSE after download completes)
  const [allUpscalers, setAllUpscalers] = useState<UpscalerVariant[]>([])
  // Currently selected upscaler label. Starts as the fallback; refreshUpscaler overwrites it from catalog.upscalers.current
  const [selectedModel, setSelectedModel] = useState<string>(FALLBACK_MODEL)
  const [downloadingModel, setDownloadingModel] = useState(false)
  const upscaler = useMemo<UpscalerVariant | null>(
    () => allUpscalers.find((x) => x.label === selectedModel) ?? null,
    [allUpscalers, selectedModel],
  )

  const refreshFiles = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.listPreprocessFilesTrain(project.id, vid)
      setFiles(r)
    } catch {
      /* ignore */
    }
  }, [project.id, vid])

  const refreshStatus = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.getPreprocessStatusTrain(project.id, vid)
      setStatus(r)
      // Replay (issue #251): restore logs from log_tail on page entry / SSE reconnect;
      // don't overwrite when it's the same job and we already have local SSE accumulation (tail is only 50 lines, shorter than local).
      const rid = r.job?.id ?? null
      setLogs((prev) =>
        rid !== null && rid === jobIdRef.current && prev.length > 0
          ? prev
          : r.log_tail
            ? r.log_tail.split('\n')
            : [],
      )
    } catch {
      /* ignore */
    }
  }, [project.id, vid])

  const refreshUpscaler = useCallback(async () => {
    try {
      const cat = await api.getModelsCatalog()
      const variants = cat.upscalers?.variants ?? []
      setAllUpscalers(variants)
      const current = cat.upscalers?.current
      setSelectedModel(current || FALLBACK_MODEL)
    } catch {
      /* ignore */
    }
  }, [])

  const changeSelectedModel = useCallback(async (label: string) => {
    setSelectedModel(label)
    try {
      await api.selectUpscaler(label)
    } catch (e) {
      toast(String(e), 'error')
      void refreshUpscaler()
    }
  }, [refreshUpscaler, toast])

  useEffect(() => {
    void refreshFiles()
    void refreshStatus()
    void refreshUpscaler()
  }, [refreshFiles, refreshStatus, refreshUpscaler])

  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = status?.job?.id ?? null
  useEventStream((evt) => {
    const jid = jobIdRef.current
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'preprocess_progress' && jid && evt.job_id === jid) {
      void refreshFiles()
    } else if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      void refreshStatus()
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        void refreshFiles()
        void reload()
      }
    } else if (evt.type === 'project_state_changed' && evt.project_id === project.id) {
      void refreshFiles()
    } else if (evt.type === 'model_download_changed') {
      void refreshUpscaler()
    }
  }, { onOpen: () => void refreshStatus() })

  const job = status?.job ?? null
  const isLive = job?.status === 'running' || job?.status === 'pending'
  const summary = files?.summary ?? status?.summary ?? { image_count: 0 }
  const modelReady = !!upscaler?.exists

  // ADR 0010: TrainImage[] → ImageRow[]. processed is inferred by the backend's `_is_processed`
  // (extension change / _cN suffix / train size != download size); the frontend doesn't compute it itself.
  const rows = useMemo<ImageRow[]>(() => {
    if (!files) return []
    const out: ImageRow[] = []
    for (const img of files.images) {
      if (img.duplicate_removed) continue // soft-deleted, doesn't enter the grid
      const lastSlash = img.name.lastIndexOf('/')
      const folder = lastSlash >= 0 ? img.name.slice(0, lastSlash) : ''
      const filename = lastSlash >= 0 ? img.name.slice(lastSlash + 1) : img.name
      out.push({
        name: img.name,
        filename,
        folder,
        status: img.processed ? 'processed' : 'pending',
        processed: img.processed ? img : undefined,
        size: img.size,
        w: img.w, h: img.h,
        mtime: img.mtime,
      })
    }
    out.sort((a, b) => a.name.localeCompare(b.name))
    return out
  }, [files])

  // Per-bin counts — drives both filter chip labels and the "hide empty bins"
  // logic so users only see chips for bins they actually have images in.
  const binCounts = useMemo(() => {
    const m = new Map<PxBinId, number>()
    for (const r of rows) {
      const id = pxBinFor(r.w, r.h)
      if (id) m.set(id, (m.get(id) ?? 0) + 1)
    }
    return m
  }, [rows])

  // Dataset subfolder list (multi-resolution: upscale one folder at a time; the target resolution can follow the px prefix).
  const folders = useMemo(
    () => Array.from(new Set(rows.map((r) => r.folder).filter(Boolean))).sort(),
    [rows],
  )
  // When a folder is selected, the "upscale all" scope = every image in that folder (ignoring the pixel-bin filter);
  // when 'all', this is null → falls back to global 'all' mode.
  const folderScopedNames = useMemo(
    () =>
      folderFilter === 'all'
        ? null
        : rows.filter((r) => r.folder === folderFilter).map((r) => r.name),
    [rows, folderFilter],
  )

  const visibleRows = useMemo(
    () =>
      rows.filter((r) => {
        if (folderFilter !== 'all' && r.folder !== folderFilter) return false
        if (filter === 'all') return true
        return pxBinFor(r.w, r.h) === filter
      }),
    [rows, filter, folderFilter],
  )

  // Selecting a folder with a px prefix → the target resolution automatically follows that folder (e.g. 1024px_xxx → 1024).
  useEffect(() => {
    if (folderFilter === 'all') return
    const reso = parseFolderMeta(folderFilter).reso
    if (reso !== null) {
      setTargetEdge(reso)
      setCustomEdge(String(reso))
    }
  }, [folderFilter])
  // ADR 0010: grid key = rel path (manifest entry key), unique across sub-folders.
  const visibleNames = useMemo(
    () => visibleRows.map((r) => r.name),
    [visibleRows],
  )

  const gridItems = useMemo(
    () =>
      visibleRows.map((r) => ({
        name: r.name,
        // train bucket thumb: folder + filename
        thumbUrl: api.versionThumbUrl(
          project.id, vid, 'train', r.filename, r.folder, 256,
        ) + `&_=${r.mtime}`,
        // ADR 0010: processed entries show an action badge (carried over from the old schema passthrough)
        meta: r.status === 'processed' ? (r.processed?.action ?? undefined) : undefined,
        badge: r.status === 'processed' ? (r.processed?.scale ? `${r.processed.scale}× ✓` : '✓') : undefined,
      })),
    [visibleRows, project.id, vid],
  )

  // ADR 0010: the selected names passed to start_job_train use rel path directly
  // (resolve_targets_train accepts rel path, matching the manifest entry key).
  const selectedTargets = useMemo(() => {
    const names: string[] = []
    for (const k of sel) names.push(k)
    return { count: names.length, names }
  }, [sel])

  // ----- actions ---------------------------------------------------------------
  const downloadModel = async () => {
    if (downloadingModel) return
    if (upscaler?.kind === 'custom') {
      toast(t('preprocess.customModelGoSettings'), 'error')
      return
    }
    setDownloadingModel(true)
    try {
      await api.startModelDownload({ model_id: 'upscaler', variant: selectedModel })
      toast(t('preprocess.downloadingModel', { model: selectedModel }), 'success')
      setTimeout(() => void refreshUpscaler(), 1500)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setDownloadingModel(false)
    }
  }

  const startPreprocess = async (
    mode: 'all' | 'selected' | 'all_force',
    names?: string[],
  ) => {
    if (!modelReady) {
      toast(t('preprocess.needModelFirst', { model: selectedModel }), 'error')
      return
    }
    let target_area: number | null = null
    if (targetEdge === null) {
      target_area = null
    } else if (targetEdge === 0) {
      const n = Number(customEdge)
      if (!Number.isFinite(n) || n < 256 || n > 4096) {
        toast(t('preprocess.customEdgeRange'), 'error')
        return
      }
      target_area = Math.round(n) * Math.round(n)
    } else {
      target_area = targetEdge * targetEdge
    }
    setBusy(true)
    try {
      const j = await api.startPreprocessTrain(project.id, vid, {
        mode,
        names,
        model: selectedModel,
        tile_size: tileSize,
        device,
        target_area,
      })
      setLogs([])
      setStatus((prev) => ({
        job: j,
        log_tail: '',
        summary: prev?.summary ?? summary,
      }))
      toast(t('preprocess.started', { id: j.id }), 'success')
      setSel(new Set())
      setSelAnchor(null)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const cancel = async () => {
    if (!job) return
    try {
      await api.cancelJob(job.id)
      toast(t('preprocess.canceled'), 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // The undo (restore) flow has moved to the "Overview" tab (PreprocessOverview), as a single
  // cross-tool entry point, so each tool page doesn't need its own undo button.

  // ADR 0010: do the vid guard after the hooks (hook order must not be interrupted by an early return)
  if (!activeVersion) {
    return (
      <div className="p-6 text-fg-secondary">
        {t('projectStepper.selectVersion')}
      </div>
    )
  }

  // Upscale operation count: scoped to the folder filter when one is active, otherwise everything
  const upscaleTotal = folderScopedNames ? folderScopedNames.length : rows.length
  const upscaleBusy = busy || isLive

  const processedCount = rows.filter((r) => r.status === 'processed').length
  const processedBytes = rows.reduce((s, r) => s + (r.status === 'processed' ? r.size : 0), 0)
  const onFilter = (f: FilterMode) => {
    setFilter(f)
    setSel(new Set())
    setSelAnchor(null)
    setPreviewIdx(null)
  }
  const onFolder = (f: string) => {
    setFolderFilter(f)
    setSel(new Set())
    setSelAnchor(null)
    setPreviewIdx(null)
  }
  const nonEmptyBins = PX_BINS.filter((b) => (binCounts.get(b.id) ?? 0) > 0)

  return (
    <StepShell
      idx={2}
      mobilePageScroll
      eyebrow={t('ppFrame.eyebrow')}
      title={t('ppFrame.title')}
      subtitle={t('ppFrame.subtitle')}
      actions={<PreprocessHeadTools projectId={project.id} versionId={vid} />}
      belowHeader={<PreprocessToolsBar current="upscale" projectId={project.id} versionId={vid} />}
      logSources={[
        job && {
          key: 'preprocess',
          label: t('ppFrame.logLabel'),
          status: job.status,
          lines: logs,
          startedAt: job.started_at,
          finishedAt: job.finished_at,
          onCancel: () => void cancel(),
        },
      ]}
    >
      <PreprocessCard current="upscale" projectId={project.id} versionId={vid}>
        <div className="ds-pp-split">
          <OperationPanel
            tileSize={tileSize}
            setTileSize={setTileSize}
            device={device}
            setDevice={setDevice}
            targetEdge={targetEdge}
            setTargetEdge={setTargetEdge}
            customEdge={customEdge}
            setCustomEdge={setCustomEdge}
            modelReady={modelReady}
            downloadingModel={downloadingModel}
            onDownloadModel={() => void downloadModel()}
            upscaler={upscaler}
            allUpscalers={allUpscalers}
            selectedModel={selectedModel}
            onSelectedModelChange={(label) => void changeSelectedModel(label)}
            busy={upscaleBusy}
            processedCount={processedCount}
            processedBytes={processedBytes}
            runLabel={t('ppFrame.runUpscale', { n: upscaleTotal })}
            onRun={() => void (folderScopedNames ? startPreprocess('selected', folderScopedNames) : startPreprocess('all'))}
            runDisabled={upscaleBusy || !modelReady || upscaleTotal === 0}
            selectedCount={selectedTargets.count}
            onRunSelected={() => void startPreprocess('selected', selectedTargets.names)}
          />

          <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, minWidth: 0 }}>
            <div style={{ padding: '12px 17px', borderBottom: '1px solid var(--line)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <FilterChip active={filter === 'all'} onClick={() => onFilter('all')} label={t('preprocess.filterAll')} n={summary.image_count} />
              {nonEmptyBins.map((b) => (
                <FilterChip key={b.id} active={filter === b.id} onClick={() => onFilter(b.id)} label={b.label} n={binCounts.get(b.id) ?? 0} />
              ))}
              {folders.length > 1 && (
                <select className="ds-inp" style={{ width: 'auto', height: 27 }} value={folderFilter} onChange={(e) => onFolder(e.target.value)} aria-label={t('preprocess.folderFilter')}>
                  <option value="all">{t('preprocess.folderAll')}</option>
                  {folders.map((f) => <option key={f} value={f}>{f}</option>)}
                </select>
              )}
              <span style={{ flex: 1 }} />
              {isLive && job && (
                <>
                  <span className="ds-badge ds-ok"><span className="ds-dot ds-dot-run" />{t('ppFrame.jobBadge', { id: job.id, n: processedCount, total: rows.length })}</span>
                  <span className="ds-meter" style={{ width: 120 }}><i style={{ width: `${rows.length ? Math.round((processedCount / rows.length) * 100) : 0}%` }} /></span>
                </>
              )}
              {sel.size > 0 ? (
                <button type="button" className="ds-ctl ds-ghost" style={{ height: 27 }} onClick={() => { setSel(new Set()); setSelAnchor(null) }}>{t('ppFrame.clearN', { n: sel.size })}</button>
              ) : (
                <button type="button" className="ds-ctl ds-ghost" style={{ height: 27 }} onClick={() => setSel(new Set(visibleNames))} disabled={gridItems.length === 0}>{t('common.selectAll')}</button>
              )}
            </div>
            <div style={{ padding: '14px 17px', flex: 1, minHeight: 0 }}>
              <ImageGrid
                items={gridItems}
                selected={sel}
                onSelect={(name, e) => {
                  const r = applySelection(sel, name, e, visibleNames, selAnchor)
                  setSel(r.next)
                  setSelAnchor(r.anchor)
                }}
                onActivate={(name) => { const i = visibleNames.indexOf(name); if (i >= 0) setPreviewIdx(i) }}
                onPreview={(name) => { const i = visibleNames.indexOf(name); if (i >= 0) setPreviewIdx(i) }}
                clickMode="activate"
                columnsClass="grid-cols-[repeat(auto-fill,minmax(92px,1fr))]"
                ariaLabel="preprocess-grid"
                emptyHint={t('preprocess.emptyForBin')}
              />
            </div>
          </div>
        </div>
      </PreprocessCard>

      {previewIdx !== null && visibleRows[previewIdx] && (
        <ImagePreviewModal
          src={api.versionThumbUrl(
            project.id, vid, 'train',
            visibleRows[previewIdx].filename,
            visibleRows[previewIdx].folder, 1600,
          ) + `&_=${visibleRows[previewIdx].mtime}`}
          caption={`${visibleRows[previewIdx].name} · ${
            visibleRows[previewIdx].status === 'processed' ? t('ppFrame.previewProcessed') : t('ppFrame.previewPending')
          }`}
          index={previewIdx}
          total={visibleRows.length}
          hasPrev={previewIdx > 0}
          hasNext={previewIdx < visibleRows.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() => previewIdx > 0 && setPreviewIdx(previewIdx - 1)}
          onNext={() => previewIdx < visibleRows.length - 1 && setPreviewIdx(previewIdx + 1)}
        />
      )}
    </StepShell>
  )
}

function FilterChip({ active, onClick, label, n }: { active: boolean; onClick: () => void; label: string; n: number }) {
  return (
    <button type="button" className={`ds-chip${active ? ' ds-is-active' : ''}`} style={{ paddingRight: 6 }} onClick={onClick} aria-pressed={active}>
      {label} <b>{n.toLocaleString()}</b>
    </button>
  )
}

// ---------------------------------------------------------------------------
// settings column
// ---------------------------------------------------------------------------

interface OperationPanelProps {
  tileSize: number
  setTileSize: (n: number) => void
  device: Device
  setDevice: (d: Device) => void
  targetEdge: number | null
  setTargetEdge: (n: number | null) => void
  customEdge: string
  setCustomEdge: (s: string) => void
  modelReady: boolean
  downloadingModel: boolean
  onDownloadModel: () => void
  upscaler: UpscalerVariant | null
  allUpscalers: UpscalerVariant[]
  selectedModel: string
  onSelectedModelChange: (label: string) => void
  busy: boolean
  processedCount: number
  processedBytes: number
  runLabel: string
  onRun: () => void
  runDisabled: boolean
  selectedCount: number
  onRunSelected: () => void
}

function OperationPanel({
  tileSize, setTileSize, device, setDevice, targetEdge, setTargetEdge, customEdge, setCustomEdge,
  modelReady, downloadingModel, onDownloadModel, upscaler, allUpscalers, selectedModel, onSelectedModelChange,
  busy, processedCount, processedBytes, runLabel, onRun, runDisabled, selectedCount, onRunSelected,
}: OperationPanelProps) {
  const { t } = useTranslation()
  const estVramMB = Math.round((tileSize * tileSize * 16 * 2 * 7) / (1024 * 1024))

  const DEVICE_OPTIONS: { value: Device; label: string }[] = [
    { value: 'auto', label: t('preprocess.deviceAuto') },
    { value: 'cuda', label: 'CUDA' },
    { value: 'cpu', label: 'CPU' },
  ]
  const TARGETS: Array<{ key: string; label: string; edge: number | null }> = [
    { key: '768', label: '768', edge: 768 },
    { key: '1024', label: '1024', edge: 1024 },
    { key: '1536', label: '1536', edge: 1536 },
    { key: '2048', label: '2048', edge: 2048 },
    { key: 'custom', label: t('ppFrame.targetCustom'), edge: 0 },
    { key: 'off', label: t('ppFrame.targetOff'), edge: null },
  ]

  const targetHint = targetEdge === null
    ? t('ppFrame.targetHintOff')
    : targetEdge === 0
      ? t('preprocess.targetHintCustom', { edge: customEdge })
      : targetEdge === DEFAULT_TARGET_EDGE
        ? t('ppFrame.targetHintRecommended', { edge: targetEdge })
        : t('preprocess.targetHintEdge', { edge: targetEdge, mpx: (targetEdge * targetEdge / 1e6).toFixed(2) })

  const fmtMB = (b: number) => (b >= 1024 ** 3 ? `${(b / 1024 ** 3).toFixed(2)} ${t('overview.unitGB')}` : `${Math.round(b / 1024 ** 2)} ${t('overview.unitMB')}`)
  const modelNote = upscaler?.kind === 'custom'
    ? t('preprocess.customModelLocal')
    : upscaler?.exists
      ? t('ppFrame.modelReadySize', { mb: upscaler.size_mb ?? '?' })
      : t('ppFrame.modelMissing')

  return (
    <div className="ds-pp-side">
      {!modelReady && (
        <div className="ds-note ds-warn" style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 11.5 }}>
          <span>
            <b>{t('preprocess.needDownload')}</b>{' '}
            {upscaler?.kind === 'custom'
              ? t('preprocess.customModelLocal')
              : `${upscaler?.hf_repo ?? upscaler?.ms_repo ?? '—'} · ~${upscaler?.size_mb ?? 64} MB`}
          </span>
          <button type="button" className="ds-btn-primary" style={{ height: 28, alignSelf: 'flex-start' }} onClick={onDownloadModel} disabled={downloadingModel || upscaler?.kind === 'custom'}>
            {downloadingModel ? t('preprocess.modelDownloading') : t('preprocess.downloadModel', { model: selectedModel })}
          </button>
        </div>
      )}

      <div>
        <div className="ds-cap" style={{ marginBottom: 8 }}>{t('ppFrame.fieldModel')}</div>
        <select className="ds-inp ds-mono" value={selectedModel} onChange={(e) => onSelectedModelChange(e.target.value)} disabled={busy}>
          {allUpscalers.map((v) => (
            <option key={v.label} value={v.label}>
              {v.label}{!v.exists ? t('preprocess.notDownloaded') : ''}{v.kind === 'custom' ? t('preprocess.customModel') : ''}
            </option>
          ))}
          {allUpscalers.length === 0 && <option value={selectedModel}>{selectedModel}</option>}
        </select>
        <div className="ds-ctl-note" style={{ marginTop: 6 }}>{modelNote}</div>
      </div>

      <div>
        <div className="ds-cap" style={{ marginBottom: 8 }}>{t('ppFrame.fieldDevice')}</div>
        <select className="ds-inp" value={device} onChange={(e) => setDevice(e.target.value as Device)} disabled={busy}>
          {DEVICE_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </div>

      <div>
        <div className="ds-cap" style={{ marginBottom: 8 }}>{t('ppFrame.fieldTarget')}</div>
        <div className="ds-seg" style={{ display: 'flex', flexWrap: 'wrap' }} role="group" aria-label={t('ppFrame.fieldTarget')}>
          {TARGETS.map((p) => {
            const on = p.edge === null ? targetEdge === null : p.edge === 0 ? targetEdge === 0 : targetEdge === p.edge
            return (
              <button key={p.key} type="button" className={`ds-seg-item${on ? ' ds-is-active' : ''}`} style={{ flex: 1, padding: '0 6px' }} aria-pressed={on} disabled={busy} onClick={() => setTargetEdge(p.edge)}>
                {p.label}
              </button>
            )
          })}
        </div>
        {targetEdge === 0 && (
          <span className="ds-stepper" style={{ width: 120, marginTop: 8 }}>
            <input type="number" min={256} max={4096} step={64} value={customEdge} onChange={(e) => setCustomEdge(e.target.value)} disabled={busy} placeholder={t('preprocess.edgePlaceholder')} aria-label={t('ppFrame.targetCustom')} />
          </span>
        )}
        <div className="ds-ctl-note" style={{ marginTop: 6 }}>{targetHint}</div>
        {targetEdge === null && <div className="ds-note ds-warn" style={{ marginTop: 10, fontSize: 11.5 }}>{t('ppFrame.targetOffWarn')}</div>}
      </div>

      <div>
        <div className="ds-cap" style={{ marginBottom: 8 }}>{t('ppFrame.fieldTile')}</div>
        <div className="ds-seg" style={{ display: 'flex' }} role="group" aria-label={t('ppFrame.fieldTile')}>
          {TILE_OPTIONS.map((n) => (
            <button key={n} type="button" className={`ds-seg-item${tileSize === n ? ' ds-is-active' : ''}`} style={{ flex: 1, padding: '0 4px' }} aria-pressed={tileSize === n} disabled={busy} onClick={() => setTileSize(n)}>
              {n}
            </button>
          ))}
        </div>
        <div className="ds-kv" style={{ marginTop: 6 }}><span className="ds-k">{t('preprocess.vramEst')}</span><span className="ds-v">~{estVramMB} {t('overview.unitMB')}</span></div>
        {processedCount > 0 && (
          <div className="ds-kv"><span className="ds-k">{t('ppFrame.diskProcessed', { n: processedCount })}</span><span className="ds-v">{fmtMB(processedBytes)}</span></div>
        )}
      </div>

      <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
        {selectedCount > 0 && (
          <button type="button" className="ds-ctl" style={{ justifyContent: 'center', height: 34 }} onClick={onRunSelected} disabled={busy || !modelReady}>
            {t('ppFrame.runSelected', { n: selectedCount })}
          </button>
        )}
        <button type="button" className="ds-btn-primary" style={{ justifyContent: 'center', height: 34 }} onClick={onRun} disabled={runDisabled}>
          {runLabel}
        </button>
      </div>
    </div>
  )
}
