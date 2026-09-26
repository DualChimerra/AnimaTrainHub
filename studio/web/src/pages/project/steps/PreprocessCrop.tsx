import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type CropWorkspaceItem,
  type Job,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import Filmstrip from '../../../components/preprocess/Filmstrip'
import FreeCropEditor, { type CropRect } from '../../../components/preprocess/FreeCropEditor'
import PreprocessToolsBar, { PreprocessCard, PreprocessHeadTools } from '../../../components/preprocess/PreprocessToolsBar'
import StepShell from '../../../components/StepShell'
import BarHistogram from '../../../components/BarHistogram'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'
import { arBucket, arLabel } from '../../../lib/aspectRatio'
import { clusterByAspectRatio } from '../../../lib/cropClustering'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

/** Aspect-ratio choices for the crop canvas. `free` = no lock; `custom` opens W:H fields. */
interface ArOption {
  id: string
  label: string
  w: number | null
  h: number | null
}
const AR_OPTIONS: ArOption[] = [
  { id: 'free',  label: 'Free (unlocked)', w: null, h: null },
  { id: '1:1',   label: '1:1 Square',    w: 1,  h: 1 },
  { id: '4:3',   label: '4:3 Landscape',      w: 4,  h: 3 },
  { id: '3:2',   label: '3:2 Landscape',      w: 3,  h: 2 },
  { id: '16:9',  label: '16:9 Widescreen',   w: 16, h: 9 },
  { id: '3:4',   label: '3:4 Portrait',      w: 3,  h: 4 },
  { id: '2:3',   label: '2:3 Portrait',      w: 2,  h: 3 },
  { id: '9:16',  label: '9:16 Phone',   w: 9,  h: 16 },
  { id: '4:5',   label: '4:5 Portrait',      w: 4,  h: 5 },
  { id: 'custom', label: 'Custom…',    w: null, h: null },
]

type Filter = 'all' | 'pending' | 'cropped'

interface AutoParams {
  maxCropFraction: number
  kMin: number
  kMax: number
}

/** Reset / clear is "uncropped" (pending), having any rect is "cropped". Status quo:
 *  the page bypasses upscale-vs-pending distinction — every workspace image is a
 *  candidate; whether it has crops drawn is the only filter dimension. */
function genRectId(): string {
  return 'c' + Math.random().toString(36).slice(2, 9)
}

export default function PreprocessCropPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const vid = activeVersion?.id ?? 0

  // ────── Workspace data ──────
  const [images, setImages] = useState<CropWorkspaceItem[]>([])
  const [loading, setLoading] = useState(true)

  const refreshWorkspace = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.listCropWorkspaceTrain(project.id, vid)
      setImages(r.images)
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [project.id, vid])

  useEffect(() => { void refreshWorkspace() }, [refreshWorkspace])

  // ────── Cropping state ──────
  // AR lock + cluster params live side-by-side; clustering is just a feature
  // inside the same crop flow, not a separate "mode" — once cluster runs it
  // pre-fills rects that the user can manually tweak via the same canvas.
  const [arSel, setArSel] = useState<string>('free')
  const [customAR, setCustomAR] = useState<{ w: number; h: number }>({ w: 5, h: 7 })
  const [autoParams, setAutoParams] = useState<AutoParams>({
    maxCropFraction: 0.10, kMin: 3, kMax: 6,
  })
  const [lastClusterK, setLastClusterK] = useState<number | null>(null)
  // Auto-cluster parameters now live in the modal popped up by the header button (no longer taking up space in the crop-params area).
  const [clusterModalOpen, setClusterModalOpen] = useState(false)

  // ────── Editor state ──────
  const [activeName, setActiveName] = useState<string | null>(null)
  const [cropsByImage, setCropsByImage] = useState<Record<string, CropRect[]>>({})
  const [selectedRectId, setSelectedRectId] = useState<string | null>(null)
  const [filter, setFilter] = useState<Filter>('all')

  // Keep activeName in sync with the available images. Multi-crop fan-out
  // renames a source file (X.png → X_c0.png / X_c1.png), so after a crop job
  // the previous activeName disappears from the workspace; without this
  // fallback the editor would render blank until the user navigates manually.
  useEffect(() => {
    if (images.length === 0) return
    if (!activeName || !images.find((im) => im.name === activeName)) {
      setActiveName(images[0].name)
    }
  }, [images, activeName])

  // ────── Job tracking ──────
  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = job?.id ?? null

  // Replay (issue #251): crop and upscale share kind=preprocess and go through the same status
  // endpoint, restoring the most recent preprocess job + log_tail; not overwritten when it's the same job and local SSE accumulation already exists.
  const refreshJobStatus = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.getPreprocessStatusTrain(project.id, vid)
      const rid = r.job?.id ?? null
      setJob(r.job)
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

  useEffect(() => { void refreshJobStatus() }, [refreshJobStatus])

  useEventStream((evt) => {
    const jid = jobIdRef.current
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      // mirror status change in our job object so the log drawer renders the new badge
      setJob((prev) =>
        prev ? { ...prev, status: evt.status as Job['status'] } : prev,
      )
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        void refreshWorkspace()
        void reload()
        // Clear in-memory crops only on success — keep on failure for retry
        if (evt.status === 'done') setCropsByImage({})
      }
    } else if (evt.type === 'crop_progress' && jid && evt.job_id === jid) {
      // Backend throttles crop_progress to ≥1Hz; safe to refresh per event here
      if (evt.status === 'done') void refreshWorkspace()
    }
  }, { onOpen: () => void refreshJobStatus() })

  const cancelJob = useCallback(async () => {
    if (!job) return
    try {
      await api.cancelJob(job.id)
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [job, toast])

  // ────── Derived ──────
  const arLock = useMemo<{ w: number; h: number } | null>(() => {
    if (arSel === 'free') return null
    if (arSel === 'custom') {
      const w = Math.max(1, customAR.w)
      const h = Math.max(1, customAR.h)
      return { w, h }
    }
    const o = AR_OPTIONS.find((x) => x.id === arSel)
    return o && o.w && o.h ? { w: o.w, h: o.h } : null
  }, [arSel, customAR])

  const totalRects = useMemo(
    () => Object.values(cropsByImage).reduce((s, arr) => s + arr.length, 0),
    [cropsByImage],
  )
  const configuredImages = useMemo(
    () => Object.entries(cropsByImage).filter(([, arr]) => arr.length > 0).length,
    [cropsByImage],
  )

  const activeImage = useMemo(
    () => images.find((im) => im.name === activeName) ?? null,
    [images, activeName],
  )
  const activeCrops = activeName ? (cropsByImage[activeName] ?? []) : []

  const counts = useMemo(() => {
    let pending = 0, cropped = 0
    for (const im of images) {
      const n = (cropsByImage[im.name] ?? []).length
      if (n === 0) pending++; else cropped++
    }
    return { all: images.length, pending, cropped }
  }, [images, cropsByImage])

  const filteredImages = useMemo(() => {
    return images.filter((im) => {
      const n = (cropsByImage[im.name] ?? []).length
      if (filter === 'pending') return n === 0
      if (filter === 'cropped') return n > 0
      return true
    })
  }, [images, filter, cropsByImage])

  // ────── Mutations ──────
  const updateRect = useCallback((id: string, newRect: CropRect) => {
    if (!activeName) return
    setCropsByImage((prev) => ({
      ...prev,
      [activeName]: (prev[activeName] ?? []).map((c) => c.id === id ? newRect : c),
    }))
  }, [activeName])

  const createRect = useCallback((r: Omit<CropRect, 'id' | 'label'>) => {
    if (!activeName) return
    const newId = genRectId()
    setCropsByImage((prev) => {
      const existing = prev[activeName] ?? []
      const newRect: CropRect = {
        id: newId,
        x: r.x, y: r.y, w: r.w, h: r.h,
        label: `${t('preprocessCrop.rectDefaultLabel')} ${existing.length + 1}`,
      }
      return { ...prev, [activeName]: [...existing, newRect] }
    })
    setSelectedRectId(newId)
  }, [activeName, t])

  const deleteRect = useCallback((id: string) => {
    if (!activeName) return
    setCropsByImage((prev) => ({
      ...prev,
      [activeName]: (prev[activeName] ?? []).filter((c) => c.id !== id),
    }))
    setSelectedRectId(null)
  }, [activeName])

  const duplicateRect = useCallback((id: string) => {
    if (!activeName) return
    const newId = genRectId()
    setCropsByImage((prev) => {
      const existing = prev[activeName] ?? []
      const src = existing.find((c) => c.id === id)
      if (!src) return prev
      return {
        ...prev,
        [activeName]: [
          ...existing,
          {
            ...src,
            id: newId,
            x: Math.min(1 - src.w, src.x + 0.04),
            y: Math.min(1 - src.h, src.y + 0.04),
            label: src.label + ' copy',
          },
        ],
      }
    })
    setSelectedRectId(newId)
  }, [activeName])

  const clearActive = useCallback(() => {
    if (!activeName) return
    setCropsByImage((prev) => ({ ...prev, [activeName]: [] }))
    setSelectedRectId(null)
  }, [activeName])

  // ────── Clustering (optional prefill action) ──────
  const runClustering = useCallback(() => {
    if (images.length === 0) return
    const summary = clusterByAspectRatio(
      images.map((im) => ({ id: im.name, w: im.w, h: im.h })),
      {
        maxCropFraction: autoParams.maxCropFraction,
        kMin: autoParams.kMin,
        kMax: autoParams.kMax,
      },
    )
    const newCrops: Record<string, CropRect[]> = {}
    for (const a of summary.assignments) {
      if (a.skipped) continue
      newCrops[a.id] = [{
        id: 'cl_' + a.id,
        x: a.rect.x, y: a.rect.y, w: a.rect.w, h: a.rect.h,
        label: `Cluster ${a.targetAr.w}:${a.targetAr.h}`,
        fromCluster: true,
      }]
    }
    setCropsByImage(newCrops)
    setSelectedRectId(null)
    setLastClusterK(summary.kUsed)
    const n = summary.assignments.filter((a) => !a.skipped).length
    toast(t('preprocessCrop.clusterToast', { k: summary.kUsed, n }), 'success')
    setClusterModalOpen(false)
  }, [images, autoParams, toast, t])

  // ────── Submit crop job ──────
  const submitCrop = useCallback(async (onlySelected = false) => {
    const payload: Record<string, { x: number; y: number; w: number; h: number; label?: string }[]> = {}
    const entries = Object.entries(cropsByImage)
    for (const [name, rects] of entries) {
      if (rects.length === 0) continue
      if (onlySelected && name !== activeName) continue
      payload[name] = rects.map((r) => ({
        x: r.x, y: r.y, w: r.w, h: r.h,
        label: r.label || undefined,
      }))
    }
    if (Object.keys(payload).length === 0) {
      toast(t('preprocessCrop.toastNoCrops'), 'error')
      return
    }
    setBusy(true)
    try {
      const j = await api.startPreprocessCropTrain(project.id, vid, payload)
      setJob(j)
      setLogs([])
      toast(t('preprocessCrop.toastStarted', { id: j.id }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }, [cropsByImage, activeName, project.id, vid, toast, t])

  // ────── Render ──────
  // ADR 0010: do the vid guard after the hooks
  if (!activeVersion) {
    return (
      <div className="p-6 text-fg-secondary">
        {t('projectStepper.selectVersion')}
      </div>
    )
  }

  return (
    <StepShell
      idx={2}
      mobilePageScroll
      eyebrow={t('ppFrame.eyebrow')}
      title={t('ppFrame.title')}
      subtitle={t('ppFrame.subtitle')}
      actions={<PreprocessHeadTools projectId={project.id} versionId={vid} />}
      belowHeader={<PreprocessToolsBar current="crop" projectId={project.id} versionId={vid} />}
      logSources={[
        job && {
          key: 'preprocess',
          label: t('ppFrame.logLabel'),
          status: job.status,
          lines: logs,
          startedAt: job.started_at,
          finishedAt: job.finished_at,
          onCancel: () => void cancelJob(),
        },
      ]}
    >
      <PreprocessCard current="crop" projectId={project.id} versionId={vid}>
        <div className="ds-pp-toolbar" style={{ justifyContent: 'flex-start' }}>
          <div className="ds-pills" role="group">
            {(['all', 'pending', 'cropped'] as const).map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setFilter(k)}
                className={`ds-pill${filter === k ? ' ds-is-active' : ''}`}
                aria-pressed={filter === k}
              >
                {t(`preprocessCrop.filter.${k}`)} <span className="ds-count">{counts[k]}</span>
              </button>
            ))}
          </div>
          {activeImage && (
            <span className="ds-crop-imginfo" title={activeImage.name}>
              <b>{activeImage.name}</b> · {activeImage.w}×{activeImage.h} · {arLabel(activeImage.w, activeImage.h)}
            </span>
          )}
          <span style={{ flex: 1 }} />
          <button
            type="button"
            onClick={clearActive}
            disabled={!activeName || (cropsByImage[activeName] ?? []).length === 0}
            className="ds-ctl"
          >{t('preprocessCrop.clearActive')}</button>
          <button
            type="button"
            onClick={() => setClusterModalOpen(true)}
            disabled={busy || images.length === 0}
            className="ds-ctl"
          >
            {t('preprocessCrop.autoCluster')}
          </button>
          <button
            type="button"
            onClick={() => void submitCrop(false)}
            disabled={busy || totalRects === 0}
            className="ds-ctl"
          >
            {t('preprocessCrop.cropAll', { n: totalRects })}
          </button>
          <button
            type="button"
            onClick={() => void submitCrop(true)}
            disabled={busy || activeCrops.length === 0}
            className="ds-btn-primary"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z" /></svg>
            <span>{t('preprocessCrop.cropActive')}</span>
          </button>
        </div>

        {loading && (
          <div className="ds-pp-body"><p className="ds-muted" style={{ fontSize: 12.5, margin: 0 }}>{t('preprocessCrop.loading')}</p></div>
        )}
        {!loading && images.length === 0 && (
          <div className="ds-pp-body">
            <div className="ds-empty">
              {t('preprocessCrop.emptyWorkspace')}
              <Link to={`/projects/${project.id}/v/${vid}/preprocess?tool=upscale`} style={{ color: 'var(--green-text)', fontWeight: 500 }}>
                {t('preprocessCrop.goToUpscale')}
              </Link>
            </div>
          </div>
        )}

        {activeImage && (
          /* filmstrip / canvas / frames + summary. With 264+ image datasets a
             bottom filmstrip gets squeezed to a hairline; a vertical column
             scrolls cleanly and leaves the canvas the full height. */
          <div className="ds-crop-grid pp-editor-grid">
            {/* Always render the filmstrip column — an empty filter keeps the
                grid in place; the empty state lives inside Filmstrip. */}
            <Filmstrip
              items={filteredImages}
              activeName={activeName}
              onSelect={(name) => {
                setActiveName(name)
                setSelectedRectId(null)
              }}
              thumbUrl={(im) => {
                const i = im.name.lastIndexOf('/')
                const folder = i >= 0 ? im.name.slice(0, i) : ''
                const filename = i >= 0 ? im.name.slice(i + 1) : im.name
                return api.versionThumbUrl(
                  project.id, vid, 'train', filename, folder, 256,
                ) + `&_=${im.mtime}`
              }}
              emptyHint={t(`preprocessCrop.filmstripEmpty.${filter}`)}
              renderOverlay={(im) => {
                const crops = cropsByImage[im.name] ?? []
                if (crops.length === 0) return null
                return (
                  <>
                    {crops.map((c, i) => (
                      <div
                        key={c.id}
                        className={'fs-overlay ' + (crops.length > 1 ? 'is-multi' : '')}
                        style={{
                          left: `${c.x * 100}%`,
                          top: `${c.y * 100}%`,
                          width: `${c.w * 100}%`,
                          height: `${c.h * 100}%`,
                        }}
                        aria-label={`crop ${i + 1}`}
                      />
                    ))}
                    {crops.length > 1 && <span className="fs-badge">×{crops.length}</span>}
                  </>
                )
              }}
            />

            <div className="ds-crop-canvas">
              <FreeCropEditor
                image={{
                  id: activeImage.name,
                  name: activeImage.name,
                  w: activeImage.w,
                  h: activeImage.h,
                  thumbUrl: (() => {
                    const i = activeImage.name.lastIndexOf('/')
                    const folder = i >= 0 ? activeImage.name.slice(0, i) : ''
                    const filename = i >= 0 ? activeImage.name.slice(i + 1) : activeImage.name
                    return api.versionThumbUrl(
                      project.id, vid, 'train', filename, folder, 1024,
                    ) + `&_=${activeImage.mtime}`
                  })(),
                }}
                crops={activeCrops}
                selectedId={selectedRectId}
                arLock={arLock}
                onSelect={setSelectedRectId}
                onChange={updateRect}
                onCreate={createRect}
              />
            </div>

            <div className="ds-crop-side">
              <RectListPanel
                activeImage={activeImage}
                crops={activeCrops}
                selectedId={selectedRectId}
                arLock={arLock}
                arSel={arSel}
                setArSel={setArSel}
                customAR={customAR}
                setCustomAR={setCustomAR}
                busy={busy}
                onSelect={setSelectedRectId}
                onLabelChange={(id, label) => {
                  const r = activeCrops.find((c) => c.id === id)
                  if (r) updateRect(id, { ...r, label })
                }}
                onDelete={deleteRect}
                onDuplicate={duplicateRect}
              />
              <RightRail
                totalRects={totalRects}
                configuredImages={configuredImages}
                totalImages={images.length}
                lastClusterK={lastClusterK}
                cropsByImage={cropsByImage}
                images={images}
              />
            </div>
          </div>
        )}

        {clusterModalOpen && (
          <ClusterModal
            autoParams={autoParams}
            setAutoParams={setAutoParams}
            lastClusterK={lastClusterK}
            busy={busy}
            totalImages={images.length}
            onRun={runClustering}
            onClose={() => setClusterModalOpen(false)}
          />
        )}
      </PreprocessCard>
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// Auto-cluster modal -- popped up by the header button. Pre-fills crop boxes for all images by aspect ratio (can be fine-tuned manually afterward).
// Parameters use numeric input boxes (previously a drag slider).
// ---------------------------------------------------------------------------

function ClusterModal({
  autoParams, setAutoParams, lastClusterK, busy, totalImages, onRun, onClose,
}: {
  autoParams: AutoParams
  setAutoParams: (v: AutoParams) => void
  lastClusterK: number | null
  busy: boolean
  totalImages: number
  onRun: () => void
  onClose: () => void
}) {
  const { t } = useTranslation()
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="cluster-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]"
      onClick={onClose}
    >
      <div
        className="w-[90%] max-w-[460px] flex flex-col gap-5 p-6 bg-elevated border border-subtle rounded-2xl shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="cluster-modal-title" className="m-0 text-lg font-semibold text-fg-primary">
          {t('preprocessCrop.autoCluster')}
        </h2>
        <p className="m-0 text-sm text-fg-secondary leading-relaxed">
          {t('preprocessCrop.clusterModalDesc')}
        </p>
        <div className="grid grid-cols-3 gap-3">
          <ClusterField
            label={t('preprocessCrop.clusterMaxCrop')}
            value={autoParams.maxCropFraction}
            min={0} max={0.3} step={0.01}
            onChange={(v) => setAutoParams({ ...autoParams, maxCropFraction: Math.max(0, Math.min(0.3, v)) })}
          />
          <ClusterField
            label={t('preprocessCrop.clusterKMin')}
            value={autoParams.kMin}
            min={1} max={10} step={1}
            onChange={(v) => setAutoParams({ ...autoParams, kMin: Math.min(v, autoParams.kMax) })}
          />
          <ClusterField
            label={t('preprocessCrop.clusterKMax')}
            value={autoParams.kMax}
            min={2} max={15} step={1}
            onChange={(v) => setAutoParams({ ...autoParams, kMax: Math.max(v, autoParams.kMin) })}
          />
        </div>
        {lastClusterK !== null && (
          <p className="m-0 text-xs text-ok font-mono">
            ✓ {t('preprocessCrop.clusterDone')} {t('preprocessCrop.clusterUsed', { k: lastClusterK })}
          </p>
        )}
        <div className="flex justify-end gap-2 pt-1">
          <button type="button" onClick={onClose} className="btn btn-secondary">
            {t('common.cancel')}
          </button>
          <button
            type="button"
            onClick={onRun}
            disabled={busy || totalImages === 0}
            className="btn btn-primary"
          >
            ▶ {t('preprocessCrop.runCluster')}
          </button>
        </div>
      </div>
    </div>
  )
}

function ClusterField({
  label, value, min, max, step, onChange,
}: {
  label: string
  value: number
  min: number; max: number; step: number
  onChange: (v: number) => void
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-fg-tertiary">{label}</span>
      <input
        type="number"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="input input-mono text-sm"
        style={{ padding: '4px 8px' }}
      />
    </label>
  )
}

// ---------------------------------------------------------------------------
// Rect list panel (right side of editor)
// ---------------------------------------------------------------------------

function RectListPanel({
  activeImage,
  crops,
  selectedId,
  arLock,
  arSel,
  setArSel,
  customAR,
  setCustomAR,
  busy,
  onSelect,
  onLabelChange,
  onDelete,
  onDuplicate,
}: {
  activeImage: CropWorkspaceItem
  crops: CropRect[]
  selectedId: string | null
  arLock: { w: number; h: number } | null
  arSel: string
  setArSel: (s: string) => void
  customAR: { w: number; h: number }
  setCustomAR: (v: { w: number; h: number }) => void
  busy: boolean
  onSelect: (id: string) => void
  onLabelChange: (id: string, label: string) => void
  onDelete: (id: string) => void
  onDuplicate: (id: string) => void
}) {
  const { t } = useTranslation()
  return (
    <section className="ds-crop-panel">
      <header className="ds-crop-panel-head">
        <span className="ds-crop-panel-title">{t('preprocessCrop.rectListTitle')}<span className="ds-samples-count">{crops.length}</span></span>
        <span className="ds-kpi-meta" style={{ marginLeft: 'auto' }}>
          {arLock ? t('preprocessCrop.arLockedTo', { ar: `${arLock.w}:${arLock.h}` }) : t('preprocessCrop.arUnlocked')}
        </span>
      </header>
      <div className="ds-crop-rects">
        {crops.length === 0 && (
          <div className="ds-empty" style={{ padding: '18px 12px' }}>
            <span style={{ color: 'var(--ink-2)', fontWeight: 500 }}>{t('preprocessCrop.emptyHintLine1')}</span>
            <span style={{ fontSize: 11.5 }}>{t('preprocessCrop.emptyHintLine2')}</span>
          </div>
        )}
        {crops.map((c, i) => {
          const outW = Math.round(c.w * activeImage.w)
          const outH = Math.round(c.h * activeImage.h)
          const isSel = c.id === selectedId
          return (
            <div
              key={c.id}
              className={`ds-crop-rect${isSel ? ' ds-is-on' : ''}`}
              onClick={() => onSelect(c.id)}
            >
              <span className="ds-crop-rect-ar" style={{ aspectRatio: `${outW}/${outH}` }}>{arLabel(outW, outH)}</span>
              <span style={{ minWidth: 0, flex: 1 }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span className="ds-kpi-meta">#{i + 1}</span>
                  <input
                    value={c.label}
                    onChange={(e) => onLabelChange(c.id, e.target.value)}
                    onClick={(e) => e.stopPropagation()}
                    className="ds-crop-rect-label"
                  />
                </span>
                <span className="ds-kpi-meta">{outW}×{outH} px</span>
              </span>
              <button type="button" className="ds-kebab" onClick={(e) => { e.stopPropagation(); onDuplicate(c.id) }} title={t('preprocessCrop.duplicate')} aria-label={t('preprocessCrop.duplicate')}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"><rect x="8" y="8" width="12" height="12" rx="2.5" /><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" /></svg>
              </button>
              <button type="button" className="ds-kebab ds-danger" onClick={(e) => { e.stopPropagation(); onDelete(c.id) }} title={t('preprocessCrop.delete')} aria-label={t('preprocessCrop.delete')}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>
              </button>
            </div>
          )
        })}
      </div>

      {/* Aspect lock: applies to frames drawn on the canvas. */}
      <footer className="ds-crop-panel-foot">
        <label className="ds-crop-ar">
          <span>{t('preprocessCrop.aspectRatio')}</span>
          <select
            value={arSel}
            onChange={(e) => setArSel(e.target.value)}
            disabled={busy}
            className="ds-inp"
          >
            {AR_OPTIONS.map((o) => (
              <option key={o.id} value={o.id}>{o.label}</option>
            ))}
          </select>
        </label>
        {arSel === 'custom' && (
          <label className="ds-crop-ar">
            <span>W : H</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <input
                type="number" min={1} max={64}
                value={customAR.w}
                onChange={(e) => setCustomAR({ ...customAR, w: Number(e.target.value) || 1 })}
                className="ds-inp"
                style={{ width: 64 }}
              />
              <span className="ds-muted">:</span>
              <input
                type="number" min={1} max={64}
                value={customAR.h}
                onChange={(e) => setCustomAR({ ...customAR, h: Number(e.target.value) || 1 })}
                className="ds-inp"
                style={{ width: 64 }}
              />
            </span>
          </label>
        )}
      </footer>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Right rail
// ---------------------------------------------------------------------------

function RightRail({
  totalRects,
  configuredImages,
  totalImages,
  lastClusterK,
  cropsByImage,
  images,
}: {
  totalRects: number
  configuredImages: number
  totalImages: number
  lastClusterK: number | null
  cropsByImage: Record<string, CropRect[]>
  images: CropWorkspaceItem[]
}) {
  const { t } = useTranslation()
  const pct = totalImages > 0 ? Math.round((configuredImages / totalImages) * 100) : 0

  // AR histogram covers the whole dataset, always. Per image: if it has
  // crops, count each crop's AR; otherwise count the source AR. A single
  // crop on the current image must NOT collapse the histogram to a single
  // entry — the user is mid-edit and still needs to see how the rest of
  // their 264 images distribute.
  //
  // `crops` mode kicks in once ALL images have at least one crop (the user
  // has fully configured the dataset and now wants to see post-crop AR).
  const arHist = useMemo(() => {
    const m = new Map<string, { n: number; sortKey: number }>()
    const allCovered = images.length > 0
      && images.every((im) => (cropsByImage[im.name] ?? []).length > 0)
    for (const im of images) {
      const rects = cropsByImage[im.name] ?? []
      if (rects.length > 0) {
        for (const r of rects) {
          const v = (r.w * im.w) / (r.h * im.h)
          const { label, sortKey } = arBucket(v)
          const prev = m.get(label)
          m.set(label, { n: (prev?.n ?? 0) + 1, sortKey })
        }
      } else {
        const v = im.w / im.h
        const { label, sortKey } = arBucket(v)
        const prev = m.get(label)
        m.set(label, { n: (prev?.n ?? 0) + 1, sortKey })
      }
    }
    const bins = Array.from(m.entries())
      .map(([label, { n, sortKey }]) => ({ label, n, sortKey }))
      .sort((a, b) => b.sortKey - a.sortKey) // wide first, tall last
    return { bins, fromSource: !allCovered }
  }, [cropsByImage, images])

  return (
    <section className="ds-crop-panel">
      <header className="ds-crop-panel-head">
        <span className="ds-crop-panel-title">{t('preprocessCrop.rrProgress')}</span>
        <span className="ds-kpi-meta" style={{ marginLeft: 'auto' }}>{pct}%</span>
      </header>
      <div style={{ padding: '0 14px 12px' }}>
        <span className="ds-meter ds-thin" style={{ margin: '2px 0 8px' }}><i style={{ width: `${pct}%` }} /></span>
        <StatRow label={t('preprocessCrop.rrWorkspace')} value={totalImages} />
        <StatRow label={t('preprocessCrop.rrConfigured')} value={configuredImages} accent={configuredImages > 0 ? 'ok' : undefined} />
        <StatRow label={t('preprocessCrop.rrPending')} value={totalImages - configuredImages} accent={totalImages - configuredImages > 0 ? 'warn' : undefined} />
        <StatRow label={t('preprocessCrop.rrOutputFiles')} value={totalRects} />
        {lastClusterK !== null && (
          <StatRow label={t('preprocessCrop.rrSource')} value={`Cluster k=${lastClusterK}`} accent="ok" />
        )}
        <p className="ds-kpi-meta" style={{ margin: '8px 0 0' }}>{t('preprocessCrop.rrNote')}</p>
      </div>
      <header className="ds-crop-panel-head" style={{ borderTop: '1px solid var(--line)' }}>
        <span className="ds-crop-panel-title">{t('preprocessCrop.rrArDist')}</span>
        <span className="ds-kpi-meta" style={{ marginLeft: 'auto' }}>
          {arHist.fromSource ? t('preprocessCrop.rrFromSource') : t('preprocessCrop.rrFromCrops')}
        </span>
      </header>
      <div style={{ padding: '0 14px 12px' }}>
        <BarHistogram bins={arHist.bins} />
      </div>
    </section>
  )
}

function StatRow({ label, value, accent }: { label: string; value: string | number; accent?: 'ok' | 'warn' | 'err' }) {
  const cls =
    accent === 'ok' ? 'text-ok' :
    accent === 'warn' ? 'text-warn' :
    accent === 'err' ? 'text-err' :
    'text-fg-primary'
  return (
    <div className="ds-kv">
      <span className="ds-k">{label}</span>
      <span className={`ds-v ${cls}`}>{value}</span>
    </div>
  )
}
