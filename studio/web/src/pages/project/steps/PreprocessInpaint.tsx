import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type CropWorkspaceItem,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import Filmstrip from '../../../components/preprocess/Filmstrip'
import InpaintCanvas, {
  renderInpaintedBlob,
  renderMaskBlob,
  type InpaintCanvasHandle,
  type InpaintMode,
  type InpaintStroke,
} from '../../../components/preprocess/InpaintCanvas'
import PreprocessToolsBar, { PreprocessCard, PreprocessHeadTools } from '../../../components/preprocess/PreprocessToolsBar'
import StepShell from '../../../components/StepShell'
import { useToast } from '../../../components/Toast'
import { useLocalStorageState } from '../../../lib/useLocalStorageState'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

type Filter = 'all' | 'pending' | 'edited'

/** 统一编辑历史条目：涂抹与 mask 笔画共用一条时间线。 */
interface HistoryEntry {
  kind: InpaintMode
  stroke: InpaintStroke
}

interface BrushState {
  color: string
  size: number
  hardness: number
}

const DEFAULT_BRUSH: BrushState = { color: '#ffffff', size: 24, hardness: 1 }

function splitRel(name: string): { folder: string; filename: string } {
  const i = name.lastIndexOf('/')
  return {
    folder: i >= 0 ? name.slice(0, i) : '',
    filename: i >= 0 ? name.slice(i + 1) : name,
  }
}

/** 状态模型对齐裁剪页：双数据面双桶（strokesByImage / maskStrokesByImage），
 *  随便切图改动都留在内存，保存按当前模式分发（§9 决策 2）。只有活动图挂
 *  真实 canvas，「保存全部」对非活动图走离屏重放。 */
export default function PreprocessInpaintPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const vid = activeVersion?.id ?? 0

  // ────── Workspace data（复用 crop workspace：name + w/h + mtime + mask_mtime）──────
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

  // ────── Editor state ──────
  // 统一编辑历史：涂抹与 mask 笔画混合入同一时间线 —— 模式只是笔刷，
  // dirty / undo / 保存都跨模式共用，切模式不改变页面状态语义。
  const [mode, setMode] = useState<InpaintMode>('paint')
  // 画笔 / 橡皮跨模式共用：涂抹橡皮擦未保存笔画，遮罩橡皮擦 mask
  const [erase, setErase] = useState(false)
  const [activeName, setActiveName] = useState<string | null>(null)
  const [historyByImage, setHistoryByImage] = useState<Record<string, HistoryEntry[]>>({})
  const [redoByImage, setRedoByImage] = useState<Record<string, HistoryEntry[]>>({})
  const [filter, setFilter] = useState<Filter>('all')
  const [busy, setBusy] = useState(false)

  const [brush, setBrush] = useLocalStorageState<BrushState>(
    'studio:inpaint:brush', DEFAULT_BRUSH,
  )
  const [recentColors, setRecentColors] = useLocalStorageState<string[]>(
    'studio:inpaint:recent_colors', [],
  )

  const canvasRef = useRef<InpaintCanvasHandle | null>(null)

  useEffect(() => {
    if (images.length === 0) return
    if (!activeName || !images.find((im) => im.name === activeName)) {
      setActiveName(images[0].name)
    }
  }, [images, activeName])

  // ────── Derived ──────
  const activeImage = useMemo(
    () => images.find((im) => im.name === activeName) ?? null,
    [images, activeName],
  )
  const activeHistory = useMemo(
    () => activeName ? (historyByImage[activeName] ?? []) : [],
    [activeName, historyByImage],
  )
  const activeRedo = activeName ? (redoByImage[activeName] ?? []) : []
  const activePaintStrokes = useMemo(
    () => activeHistory.filter((h) => h.kind === 'paint').map((h) => h.stroke),
    [activeHistory],
  )
  const activeMaskStrokes = useMemo(
    () => activeHistory.filter((h) => h.kind === 'mask').map((h) => h.stroke),
    [activeHistory],
  )

  /** dirty 图集合 = 任一数据面有未保存笔画（保存全部 / filter / 计数共用）。 */
  const editedNames = useMemo(
    () => Object.entries(historyByImage)
      .filter(([, h]) => h.length > 0)
      .map(([n]) => n),
    [historyByImage],
  )

  const counts = useMemo(() => {
    const edited = images.filter((im) => (historyByImage[im.name] ?? []).length > 0).length
    return { all: images.length, pending: images.length - edited, edited }
  }, [images, historyByImage])

  const filteredImages = useMemo(() => images.filter((im) => {
    const n = (historyByImage[im.name] ?? []).length
    if (filter === 'pending') return n === 0
    if (filter === 'edited') return n > 0
    return true
  }), [images, filter, historyByImage])

  const rawUrl = useCallback((im: CropWorkspaceItem) => {
    const { folder, filename } = splitRel(im.name)
    return api.versionThumbUrl(project.id, vid, 'train', filename, folder, 0)
      + `&_=${im.mtime}`
  }, [project.id, vid])

  const maskBaseUrlFor = useCallback((im: CropWorkspaceItem): string | null => {
    if (im.mask_mtime == null) return null
    return api.maskUrl(project.id, vid, im.name) + `&_=${im.mask_mtime}`
  }, [project.id, vid])

  // ────── Stroke mutations（统一时间线，undo/redo 跨模式）──────
  const pushRecentColor = useCallback((hex: string) => {
    setRecentColors((prev) => [hex, ...prev.filter((c) => c !== hex)].slice(0, 8))
  }, [setRecentColors])

  const pushEntry = useCallback((entry: HistoryEntry) => {
    if (!activeName) return
    setHistoryByImage((prev) => ({
      ...prev,
      [activeName]: [...(prev[activeName] ?? []), entry],
    }))
    setRedoByImage((prev) => ({ ...prev, [activeName]: [] }))
  }, [activeName])

  const onStrokeEnd = useCallback((s: InpaintStroke) => {
    pushEntry({ kind: 'paint', stroke: s })
    pushRecentColor(s.color)
  }, [pushEntry, pushRecentColor])

  const onMaskStrokeEnd = useCallback((s: InpaintStroke) => {
    pushEntry({ kind: 'mask', stroke: s })
  }, [pushEntry])

  const undo = useCallback(() => {
    if (!activeName) return
    setHistoryByImage((prev) => {
      const cur = prev[activeName] ?? []
      if (cur.length === 0) return prev
      const last = cur[cur.length - 1]
      setRedoByImage((r) => ({
        ...r,
        [activeName]: [...(r[activeName] ?? []), last],
      }))
      return { ...prev, [activeName]: cur.slice(0, -1) }
    })
  }, [activeName])

  const redo = useCallback(() => {
    if (!activeName) return
    setRedoByImage((prev) => {
      const cur = prev[activeName] ?? []
      if (cur.length === 0) return prev
      const last = cur[cur.length - 1]
      setHistoryByImage((h) => ({
        ...h,
        [activeName]: [...(h[activeName] ?? []), last],
      }))
      return { ...prev, [activeName]: cur.slice(0, -1) }
    })
  }, [activeName])

  const clearActive = useCallback(() => {
    if (!activeName) return
    setHistoryByImage((prev) => ({ ...prev, [activeName]: [] }))
    setRedoByImage((prev) => ({ ...prev, [activeName]: [] }))
  }, [activeName])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== 'z') return
      const el = e.target as HTMLElement | null
      if (
        el &&
        (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
      ) {
        return
      }
      e.preventDefault()
      if (e.shiftKey) redo()
      else undo()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redo])

  const onPickColor = useCallback((hex: string) => {
    setBrush((prev) => ({ ...prev, color: hex }))
    pushRecentColor(hex)
  }, [setBrush, pushRecentColor])

  // ────── Save（保存 = 该图全部未保存改动，两个数据面一次写完）──────
  /** 保存成功后从历史滤掉对应数据面的 entries（redo 时间线随之作废）。 */
  const clearSavedKind = useCallback((name: string, kind: InpaintMode) => {
    setHistoryByImage((prev) => ({
      ...prev,
      [name]: (prev[name] ?? []).filter((h) => h.kind !== kind),
    }))
    setRedoByImage((prev) => ({ ...prev, [name]: [] }))
  }, [])

  /** 单图两面保存。涂抹先行 —— 产物可能改名（X.jpg→X.png），mask 的 PUT
   *  必须用新 name（旧源文件已删，服务端按 name 校验源图存在）。
   *  返回保存后的 name（无涂抹改动时原样）。 */
  const saveImageBoth = useCallback(async (
    im: CropWorkspaceItem,
    paintStrokes: InpaintStroke[],
    maskStrokes: InpaintStroke[],
    exporters?: {
      paint: () => Promise<Blob | null>
      mask: () => Promise<{ blob: Blob; coverage: number } | null>
    },
  ): Promise<string> => {
    let name = im.name
    if (paintStrokes.length > 0) {
      const blob = exporters
        ? await exporters.paint()
        : await renderInpaintedBlob(rawUrl(im), im.w, im.h, paintStrokes)
      if (!blob) throw new Error('canvas not ready')
      const res = await api.saveInpaintTrain(project.id, vid, name, blob)
      clearSavedKind(im.name, 'paint')
      name = res.name
    }
    if (maskStrokes.length > 0) {
      const res = exporters
        ? await exporters.mask()
        : await renderMaskBlob(maskBaseUrlFor(im), im.w, im.h, maskStrokes)
      if (res === null) {
        if (im.mask_mtime != null) await api.deleteMaskTrain(project.id, vid, name)
      } else {
        await api.saveMaskTrain(project.id, vid, name, res.blob)
      }
      clearSavedKind(im.name, 'mask')
    }
    return name
  }, [project.id, vid, rawUrl, maskBaseUrlFor, clearSavedKind])

  const saveActive = useCallback(async () => {
    if (!activeName || !activeImage) return
    if (activeHistory.length === 0) return
    setBusy(true)
    try {
      // 活动图用挂载中的 canvas 导出（所见即所得），非活动图才走离屏重放
      const newName = await saveImageBoth(
        activeImage, activePaintStrokes, activeMaskStrokes,
        {
          paint: () => canvasRef.current?.exportBlob() ?? Promise.resolve(null),
          mask: async () => {
            const r = await canvasRef.current?.exportMaskBlob()
            return r ?? null
          },
        },
      )
      toast(t('preprocessInpaint.toastSaved', { name: newName }), 'success')
      await refreshWorkspace()
      if (newName !== activeName) setActiveName(newName)
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }, [
    activeName, activeImage, activeHistory.length,
    activePaintStrokes, activeMaskStrokes,
    saveImageBoth, refreshWorkspace, reload, toast, t,
  ])

  const saveAll = useCallback(async () => {
    const dirty = editedNames
    if (dirty.length === 0) return
    setBusy(true)
    let ok = 0
    const failed: string[] = []
    try {
      for (const name of dirty) {
        const im = images.find((i) => i.name === name)
        const hist = historyByImage[name] ?? []
        if (!im || hist.length === 0) continue
        try {
          await saveImageBoth(
            im,
            hist.filter((h) => h.kind === 'paint').map((h) => h.stroke),
            hist.filter((h) => h.kind === 'mask').map((h) => h.stroke),
          )
          ok++
        } catch {
          failed.push(name)
        }
      }
      toast(
        failed.length > 0
          ? t('preprocessInpaint.toastSavedAllPartial', { ok, failed: failed.length })
          : t('preprocessInpaint.toastSavedAll', { n: ok }),
        failed.length > 0 ? 'error' : 'success',
      )
      await refreshWorkspace()
      void reload()
    } finally {
      setBusy(false)
    }
  }, [
    editedNames, images, historyByImage, saveImageBoth,
    refreshWorkspace, reload, toast, t,
  ])

  // ────── Render ──────
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
      belowHeader={<PreprocessToolsBar current="inpaint" projectId={project.id} versionId={vid} />}
    >
      <PreprocessCard current="inpaint" projectId={project.id} versionId={vid}>
        <div className="ds-pp-toolbar" style={{ justifyContent: 'flex-start' }}>
          <div className="ds-pills" role="group">
            {(['all', 'pending', 'edited'] as const).map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setFilter(k)}
                className={`ds-pill${filter === k ? ' ds-is-active' : ''}`}
                aria-pressed={filter === k}
              >
                {t(`preprocessInpaint.filter.${k}`)} <span className="ds-count">{counts[k]}</span>
              </button>
            ))}
          </div>
          {activeImage && (
            <span className="ds-crop-imginfo" title={activeImage.name}>
              <b>{activeImage.name}</b> · {activeImage.w}×{activeImage.h}
            </span>
          )}
          <span style={{ flex: 1 }} />
          <span className="ds-actgroup">
            <button type="button" className="ds-ico" onClick={undo} disabled={!activeName || activeHistory.length === 0} title={`${t('preprocessInpaint.undo')} · Ctrl+Z`} aria-label={t('preprocessInpaint.undo')}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 14 4 9l5-5" /><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" /></svg>
            </button>
            <button type="button" className="ds-ico" onClick={redo} disabled={!activeName || activeRedo.length === 0} title={`${t('preprocessInpaint.redo')} · Ctrl+Shift+Z`} aria-label={t('preprocessInpaint.redo')}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15 14 5-5-5-5" /><path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13" /></svg>
            </button>
          </span>
          <button
            type="button"
            onClick={clearActive}
            disabled={!activeName || activeHistory.length === 0}
            className="ds-ctl"
          >{t('preprocessInpaint.clearActive')}</button>
          {/* 保存 = 两个数据面的全部未保存改动；文案 / 可用性不随模式变 */}
          <button
            type="button"
            onClick={() => void saveAll()}
            disabled={busy || editedNames.length === 0}
            className="ds-ctl"
          >
            {t('preprocessInpaint.saveAll', { n: editedNames.length })}
          </button>
          <button
            type="button"
            onClick={() => void saveActive()}
            disabled={busy || activeHistory.length === 0}
            className="ds-btn-primary"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M17 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7l-4-4zm-5 16a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm3-10H5V5h10v4z" />
            </svg>
            <span>{t('preprocessInpaint.saveActive')}</span>
          </button>
        </div>

        {loading && (
          <div className="ds-pp-body"><p className="ds-muted" style={{ fontSize: 12.5, margin: 0 }}>{t('preprocessInpaint.loading')}</p></div>
        )}
        {!loading && images.length === 0 && (
          <div className="ds-pp-body">
            <div className="ds-empty">
              {t('preprocessInpaint.emptyWorkspace')}
              <Link to={`/projects/${project.id}/v/${vid}/preprocess`} style={{ color: 'var(--green-text)', fontWeight: 500 }}>
                {t('preprocessInpaint.goToOverview')}
              </Link>
            </div>
          </div>
        )}

        {activeImage && (
          <div className="ds-crop-grid pp-editor-grid">
            <Filmstrip
              items={filteredImages}
              activeName={activeName}
              onSelect={setActiveName}
              thumbUrl={(im) => {
                const { folder, filename } = splitRel(im.name)
                return api.versionThumbUrl(
                  project.id, vid, 'train', filename, folder, 256,
                ) + `&_=${im.mtime}`
              }}
              emptyHint={t(`preprocessInpaint.filmstripEmpty.${filter}`)}
              renderOverlay={(im) => {
                const hist = historyByImage[im.name] ?? []
                const hasPaint = hist.some((h) => h.kind === 'paint')
                const hasMask = im.mask_mtime != null
                  || hist.some((h) => h.kind === 'mask')
                if (!hasPaint && !hasMask) return null
                return (
                  <span className="fs-badge">
                    {hasPaint ? '✎' : ''}{hasMask ? 'M' : ''}
                  </span>
                )
              }}
            />

            <div className="ds-crop-canvas">
              <InpaintCanvas
                key={activeImage.name}
                ref={canvasRef}
                imageUrl={rawUrl(activeImage)}
                imageW={activeImage.w}
                imageH={activeImage.h}
                mode={mode}
                strokes={activePaintStrokes}
                maskStrokes={activeMaskStrokes}
                maskBaseUrl={maskBaseUrlFor(activeImage)}
                brush={brush}
                erase={erase}
                onStrokeEnd={onStrokeEnd}
                onMaskStrokeEnd={onMaskStrokeEnd}
                onPickColor={onPickColor}
              />
            </div>

            <div className="ds-crop-side">
              <ToolPanel
                mode={mode}
                setMode={setMode}
                erase={erase}
                setErase={setErase}
                brush={brush}
                setBrush={setBrush}
                recentColors={recentColors}
              />
            </div>
          </div>
        )}
      </PreprocessCard>
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// Tool panel（right side）
// ---------------------------------------------------------------------------

function ToolPanel({
  mode,
  setMode,
  erase,
  setErase,
  brush,
  setBrush,
  recentColors,
}: {
  mode: InpaintMode
  setMode: (m: InpaintMode) => void
  erase: boolean
  setErase: (v: boolean) => void
  brush: BrushState
  setBrush: (v: BrushState | ((prev: BrushState) => BrushState)) => void
  recentColors: string[]
}) {
  const { t } = useTranslation()
  const [recentOpen, setRecentOpen] = useState(false)
  const hardness = Math.round(brush.hardness * 100)
  return (
    <section className="ds-crop-panel">
      <header className="ds-crop-panel-head">
        <span className="ds-crop-panel-title">{t('preprocessInpaint.panelTitle')}</span>
      </header>
      <div className="ds-ip-body">
        <div className="ds-ip-row">
          <span className="ds-ip-label">{t('preprocessInpaint.modeLabel')}</span>
          <div className="ds-seg" role="radiogroup" aria-label={t('preprocessInpaint.modeLabel')}>
            {(['paint', 'mask'] as const).map((m) => (
              <button key={m} type="button" role="radio" aria-checked={mode === m} className={`ds-seg-item${mode === m ? ' ds-is-active' : ''}`} onClick={() => setMode(m)}>
                {t(`preprocessInpaint.mode.${m}`)}
              </button>
            ))}
          </div>
        </div>
        <div className="ds-ip-row">
          <span className="ds-ip-label">{t('preprocessInpaint.toolLabel')}</span>
          <div className="ds-seg" role="radiogroup" aria-label={t('preprocessInpaint.toolLabel')}>
            <button type="button" role="radio" aria-checked={!erase} className={`ds-seg-item${!erase ? ' ds-is-active' : ''}`} onClick={() => setErase(false)}>{t('preprocessInpaint.toolBrush')}</button>
            <button type="button" role="radio" aria-checked={erase} className={`ds-seg-item${erase ? ' ds-is-active' : ''}`} onClick={() => setErase(true)}>{t('preprocessInpaint.toolEraser')}</button>
          </div>
        </div>

        {mode === 'paint' && (
          <div className="ds-ip-row">
            <span className="ds-ip-label">{t('preprocessInpaint.brushColor')}</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <label className="ds-ip-swatch" style={{ background: brush.color }} title={t('preprocessInpaint.colorWheel')}>
                <input
                  type="color"
                  value={brush.color}
                  onChange={(e) => setBrush((p) => ({ ...p, color: e.target.value }))}
                  aria-label={t('preprocessInpaint.colorWheel')}
                />
              </label>
              <span className="ds-kpi-meta" style={{ width: 58 }}>{brush.color}</span>
              <button
                type="button"
                onClick={() => setRecentOpen((v) => !v)}
                disabled={recentColors.length === 0}
                className={`ds-iconbtn${recentOpen ? ' ds-is-open' : ''}`}
                style={{ width: 30, height: 30 }}
                title={t('preprocessInpaint.recentColors')}
                aria-label={t('preprocessInpaint.recentColors')}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /><path d="M12 7v5l3 2" /></svg>
              </button>
            </span>
          </div>
        )}
        {mode === 'paint' && recentOpen && recentColors.length > 0 && (
          <div className="ds-ip-recent">
            {recentColors.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => {
                  setBrush((p) => ({ ...p, color: c }))
                  setRecentOpen(false)
                }}
                className={`ds-ip-dot${c === brush.color ? ' ds-is-on' : ''}`}
                style={{ backgroundColor: c }}
                title={c}
                aria-label={c}
              />
            ))}
          </div>
        )}

        <div className="ds-ip-slider">
          <span className="ds-ip-label">{t('preprocessInpaint.brushSize')}</span>
          <input
            type="range"
            min={1} max={400} step={1}
            value={brush.size}
            onChange={(e) => setBrush((p) => ({ ...p, size: Number(e.target.value) }))}
            className="ds-range"
            style={{ '--p': `${((brush.size - 1) / 399) * 100}%` } as React.CSSProperties}
            aria-label={t('preprocessInpaint.brushSize')}
          />
          <input
            type="number"
            min={1} max={400}
            value={brush.size}
            onChange={(e) => setBrush((p) => ({
              ...p, size: Math.max(1, Math.min(400, Number(e.target.value) || 1)),
            }))}
            className="ds-inp"
            style={{ width: 62, height: 28 }}
          />
        </div>
        <div className="ds-ip-slider">
          <span className="ds-ip-label">{t('preprocessInpaint.brushHardness')}</span>
          <input
            type="range"
            min={0} max={100} step={5}
            value={hardness}
            onChange={(e) => setBrush((p) => ({ ...p, hardness: Number(e.target.value) / 100 }))}
            className="ds-range"
            style={{ '--p': `${hardness}%` } as React.CSSProperties}
            aria-label={t('preprocessInpaint.brushHardness')}
          />
          <input
            type="number"
            min={0} max={100} step={5}
            value={hardness}
            onChange={(e) => setBrush((p) => ({
              ...p,
              hardness: Math.max(0, Math.min(100, Number(e.target.value) || 0)) / 100,
            }))}
            className="ds-inp"
            style={{ width: 62, height: 28 }}
          />
        </div>
      </div>
    </section>
  )
}
