import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api } from '../../../api/client'
import { exportXYMatrix } from './exportXY'
import FullscreenViewer from './FullscreenViewer'
import { axisLabel, formatAxisValue, type XYAxisDraft } from './xy'

/** PreviewXYGrid's local sample type.
 *
 *  `MonitorState['samples']` is structurally assignable to it (the active-task path, where
 *  `xy` is optional -- non-XY-mode samples reuse the same streaming buffer);
 *  the disk-history playback path additionally supplies `imageUrl` (a cell URL already
 *  URL-encoded by the server). GridCell and FullscreenViewer both prefer `imageUrl`,
 *  falling back to `api.generateSampleUrl(taskId, filename)` otherwise. */
export interface XYSample {
  path: string
  step?: number
  xy?: {
    xi: number
    yi: number
    xv?: string | number
    yv?: string | number | null
  }
  /** Given by the server for disk-served entries; left empty for cache / active-task */
  imageUrl?: string
}

// zoom = a single cell's physical width (px). Fixed column width -> wheel-zoom takes visible
// effect immediately; when the total column width exceeds the container it scrolls horizontally
// (already backed by overflow:auto).
// MIN = ZOOM_DEFAULT (100%): by product decision, cells can't go below 100% (smaller cells
// aren't legible enough to be useful); MAX is dynamic = container width (so a single cell can
// fill the whole screen at most); DEFAULT = 200px (100%).
const ZOOM_DEFAULT = 200
const ZOOM_MIN = ZOOM_DEFAULT
const ZOOM_STEP = 24

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

/** XY-mode preview grid: arranges monitorState.samples[].xy into an N x M CSS grid.
 *
 * - No longer caps the column count (previously colsForX truncated to 5, so a 12-cell matrix
 *   only showed 10)
 * - 2px gap (product decision)
 * - cells are 1:1 aspect, filled with object-cover
 * - column template `60px repeat(xLen, minmax(MIN, 1fr))` keeps every cell in a row the same
 *   width, at least MIN wide, splitting any extra space evenly; the whole grid scrolls
 *   horizontally once it exceeds the container width
 */
export default function PreviewXYGrid({
  samples, taskId, xDraft, yDraft, onCellClick, selectedIndices,
  compositeUrl,
}: {
  samples: XYSample[]
  taskId: number
  xDraft: XYAxisDraft
  yDraft: XYAxisDraft | null
  onCellClick?: (sampleIdx: number) => void
  selectedIndices?: number[]
  /** Disk-history playback only: pass the composite full-image URL so the export-PNG button can
   *  just anchor-download it directly, bypassing composeXYMatrix (the disk entry's cache is
   *  long gone by then, so re-composing would fail). Omitted in active-task mode -> falls back
   *  to exportXYMatrix. */
  compositeUrl?: string
}) {
  const { t } = useTranslation()
  const [cellW, setCellW] = useState(ZOOM_DEFAULT)
  const [maxW, setMaxW] = useState(ZOOM_DEFAULT * 6) // Fallback value before the container has mounted
  const [fullscreenIdx, setFullscreenIdx] = useState<number | null>(null)
  const [exporting, setExporting] = useState(false)
  const [exportMsg, setExportMsg] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const handleExport = async () => {
    if (exporting) return
    setExporting(true)
    setExportMsg(null)
    try {
      // Disk history playback: the composite is already on disk, download it directly (don't
      // call composeXYMatrix -- per-cell URLs fetched across tasks would 404 against the cache,
      // and the server-side composite is already a pixel-equivalent result)
      if (compositeUrl) {
        const a = document.createElement('a')
        a.href = compositeUrl
        a.download = `xy_matrix_${taskId}_${compositeUrl.split('/').pop() ?? 'plot.png'}`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
      } else {
        const exportSamples = samples
          .filter((s): s is XYSample & { xy: NonNullable<XYSample['xy']> } => s.xy != null)
          .map((s) => ({ path: s.path, xy: { xi: s.xy.xi, yi: s.xy.yi } }))
        await exportXYMatrix({
          samples: exportSamples,
          taskId,
          xAxis: xDraft.axis,
          yAxis: yDraft?.axis ?? null,
          xValues,
          yValues,
        })
      }
      setExportMsg(t('generate.exportDownloaded'))
      setTimeout(() => setExportMsg(null), 3000)
    } catch (e) {
      setExportMsg(t('generate.exportFailed', { error: e instanceof Error ? e.message : String(e) }))
    } finally {
      setExporting(false)
    }
  }
  // Pan state. movedRef lets a mouseup after "dragging" skip the cell click (intercepted at the capture phase)
  const dragRef = useRef<{ startX: number; startY: number; sX: number; sY: number } | null>(null)
  const movedRef = useRef(false)

  // ZOOM_MAX is dynamic = container width (one image fills the screen); ResizeObserver tracks window / sidebar changes
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const update = () => {
      const w = Math.max(ZOOM_DEFAULT, el.clientWidth)
      setMaxW(w)
      setCellW((prev) => Math.min(prev, w))
    }
    update()
    // jsdom has no ResizeObserver; fall back to a window resize listener in the test environment
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(update)
      ro.observe(el)
      return () => ro.disconnect()
    }
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])

  // The wheel listener must be native + passive=false for preventDefault to work
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      // shift+wheel lets the browser scroll horizontally natively, instead of zooming
      if (e.shiftKey) return
      e.preventDefault()
      const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP
      setCellW((prev) => clamp(prev + delta, ZOOM_MIN, maxW))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [maxW])

  const onMouseDown: React.MouseEventHandler<HTMLDivElement> = (e) => {
    if (e.button !== 0) return
    // Every area (including cell buttons) initiates panning; plain click now requires
    // Ctrl+click to select a cell, so a plain click yields to the drag gesture.
    // preventDefault stops the browser's native image drag.
    e.preventDefault()
    if (!scrollRef.current) return
    dragRef.current = {
      startX: e.clientX, startY: e.clientY,
      sX: scrollRef.current.scrollLeft,
      sY: scrollRef.current.scrollTop,
    }
    movedRef.current = false
  }

  const onMouseMove: React.MouseEventHandler<HTMLDivElement> = (e) => {
    const d = dragRef.current
    if (!d || !scrollRef.current) return
    const dx = e.clientX - d.startX
    const dy = e.clientY - d.startY
    if (Math.abs(dx) > 4 || Math.abs(dy) > 4) movedRef.current = true
    scrollRef.current.scrollLeft = d.sX - dx
    scrollRef.current.scrollTop = d.sY - dy
  }

  const onMouseUp = () => {
    dragRef.current = null
  }

  // Intercept click at the capture phase: a mouseup after a drag must not let the cell button fire a click
  const onClickCapture: React.MouseEventHandler<HTMLDivElement> = (e) => {
    if (movedRef.current) {
      e.stopPropagation()
      e.preventDefault()
      movedRef.current = false
    }
  }

  const xValues = useMemo(
    () => xDraft.raw.split(',').map((s) => s.trim()).filter(Boolean),
    [xDraft.raw],
  )
  const yValues = useMemo(
    () => yDraft ? yDraft.raw.split(',').map((s) => s.trim()).filter(Boolean) : [null],
    [yDraft],
  )
  const xLen = xValues.length
  const yLen = yValues.length

  const cellIndex = useMemo(() => {
    const m = new Map<string, number>()
    samples.forEach((s, idx) => {
      if (s.xy) m.set(`${s.xy.yi}_${s.xy.xi}`, idx)
    })
    return m
  }, [samples])

  const fullscreenNeighbors = useMemo(() => {
    if (fullscreenIdx == null) return null
    const sample = samples[fullscreenIdx]
    if (!sample?.xy) return null
    const at = (dx: number, dy: number): number | null =>
      cellIndex.get(`${sample.xy!.yi + dy}_${sample.xy!.xi + dx}`) ?? null
    return {
      left: at(-1, 0),
      right: at(1, 0),
      up: at(0, -1),
      down: at(0, 1),
    }
  }, [cellIndex, fullscreenIdx, samples])

  const selSet = new Set(selectedIndices ?? [])

  // Grid columns: fixed cellW (adjusted by zoom); with a yDraft there's an extra axis-label
  // column on the left. Using `${cellW}px` instead of minmax(MIN, 1fr) matters -- the latter
  // splits space evenly by 1fr once the container is wide enough, hiding the zoom effect;
  // a fixed column width makes wheel-zoom take visible effect immediately.
  const labelColW = yDraft ? 60 : 0
  const gridCols = yDraft
    ? `${labelColW}px repeat(${xLen}, ${cellW}px)`
    : `repeat(${xLen}, ${cellW}px)`

  return (
    <div className="flex flex-col gap-2 flex-1 min-h-0">
      <div className="flex items-center justify-between shrink-0">
        <span className="caption">
          {t('generate.xyGridCount', { x: xLen, y: yLen, n: xLen * yLen, axis: yDraft ? ` × ${yLen}` : '' })}
          {samples.length < xLen * yLen && samples.length > 0 && (
            <span className="text-fg-tertiary"> {t('generate.generatedCount', { n: samples.length })}</span>
          )}
        </span>
        <div className="flex items-center gap-2 text-2xs text-fg-tertiary font-mono">
          <span>{t('generate.xyGridHelp')}</span>
          <button
            onClick={() => setCellW(ZOOM_DEFAULT)}
            className="btn btn-ghost text-xs"
            title={t('generate.resetZoom')}
          >
            {Math.round((cellW / ZOOM_DEFAULT) * 100)}%
          </button>
          <button
            onClick={() => void handleExport()}
            disabled={exporting || samples.length === 0}
            className="btn btn-secondary text-xs"
            title={t('generate.exportPngTitle')}
          >
            {exporting ? t('generate.exporting') : t('generate.exportPng')}
          </button>
          {exportMsg && (
            <span className={`text-2xs ${exportMsg.startsWith(t('generate.exportFailedPrefix')) ? 'text-err' : 'text-ok'}`}>
              {exportMsg}
            </span>
          )}
        </div>
      </div>

      {/* Grid has built-in horizontal scroll (when too many X columns overflow the container) + wheel-zoom + drag-pan */}
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-auto"
        style={{ cursor: dragRef.current ? 'grabbing' : 'grab' }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onClickCapture={onClickCapture}
      >
        <div style={{ display: 'grid', gridTemplateColumns: gridCols, gap: 2 }}>
          {/* Header row: blank top-left corner (only when there's a yDraft) + X labels */}
          {yDraft && <div />}
          {xValues.map((xv, xi) => (
            <div
              key={`h-${xi}`}
              className="text-2xs text-fg-tertiary font-mono text-center truncate"
              style={{ padding: '4px 2px' }}
              title={xv}
            >
              {formatAxisValue(xDraft.axis, xv)}
            </div>
          ))}

          {/* Data rows */}
          {yValues.map((yv, yi) => (
            <Row
              key={`y-${yi}`}
              yi={yi} yv={yv}
              xValues={xValues}
              xDraft={xDraft}
              yDraft={yDraft}
              cellIndex={cellIndex}
              samples={samples}
              taskId={taskId}
              selSet={selSet}
              onCellClick={onCellClick}
              onCellDoubleClick={(idx) => setFullscreenIdx(idx)}
            />
          ))}
        </div>
      </div>

      {fullscreenIdx != null && samples[fullscreenIdx] && (() => {
        const s = samples[fullscreenIdx]
        const fn = s.path.split(/[\\/]/).pop() ?? ''
        const captionParts: string[] = []
        if (s.xy) {
          captionParts.push(`${axisLabel(xDraft.axis)}=${formatAxisValue(xDraft.axis, String(s.xy.xv ?? ''))}`)
          if (yDraft && s.xy.yv != null) {
            captionParts.push(`${axisLabel(yDraft.axis)}=${formatAxisValue(yDraft.axis, String(s.xy.yv))}`)
          }
        }
        return (
          <FullscreenViewer
            src={s.imageUrl ?? api.generateSampleUrl(taskId, fn)}
            alt={fn}
            caption={captionParts.join(', ')}
            index={fullscreenIdx}
            total={samples.length}
            onClose={() => setFullscreenIdx(null)}
            hasPrev={fullscreenNeighbors?.left != null}
            hasNext={fullscreenNeighbors?.right != null}
            hasUp={fullscreenNeighbors?.up != null}
            hasDown={fullscreenNeighbors?.down != null}
            onPrev={() => {
              if (fullscreenNeighbors?.left != null) setFullscreenIdx(fullscreenNeighbors.left)
            }}
            onNext={() => {
              if (fullscreenNeighbors?.right != null) setFullscreenIdx(fullscreenNeighbors.right)
            }}
            onUp={() => {
              if (fullscreenNeighbors?.up != null) setFullscreenIdx(fullscreenNeighbors.up)
            }}
            onDown={() => {
              if (fullscreenNeighbors?.down != null) setFullscreenIdx(fullscreenNeighbors.down)
            }}
            // shortcutHint is omitted -> FullscreenViewer builds it dynamically from hasX
            // (invalid directions aren't shown for a single row / single column / corner cell)
          />
        )
      })()}
    </div>
  )
}

function Row({
  yi, yv, xValues, xDraft, yDraft, cellIndex, samples, taskId, selSet,
  onCellClick, onCellDoubleClick,
}: {
  yi: number
  yv: string | null
  xValues: string[]
  xDraft: XYAxisDraft
  yDraft: XYAxisDraft | null
  cellIndex: Map<string, number>
  samples: XYSample[]
  taskId: number
  selSet: Set<number>
  onCellClick?: (sampleIdx: number) => void
  onCellDoubleClick?: (sampleIdx: number) => void
}) {
  const { t } = useTranslation()
  return (
    <>
      {yDraft && (
        <div
          className="text-2xs text-fg-tertiary font-mono text-right truncate self-center"
          style={{ paddingRight: 4 }}
          title={yv ?? ''}
        >
          {yv != null ? formatAxisValue(yDraft.axis, yv) : ''}
        </div>
      )}
      {xValues.map((xv, xi) => {
        const idx = cellIndex.get(`${yi}_${xi}`)
        const sample = idx != null ? samples[idx] : null
        const filename = sample ? sample.path.split(/[\\/]/).pop() ?? null : null
        const isSel = idx != null && selSet.has(idx)
        const tooltip = t('generate.xyCellTooltip', {
          label: !yDraft
            ? `${axisLabel(xDraft.axis)}=${formatAxisValue(xDraft.axis, xv)}`
            : `${axisLabel(xDraft.axis)}=${formatAxisValue(xDraft.axis, xv)}, ${axisLabel(yDraft.axis)}=${formatAxisValue(yDraft.axis, yv ?? '')}`,
        })
        return (
          <GridCell
            key={`c-${yi}-${xi}`}
            taskId={taskId}
            filename={filename}
            imageUrl={sample?.imageUrl}
            sampleIdx={idx ?? null}
            isSelected={isSel}
            tooltip={tooltip}
            onClick={onCellClick}
            onDoubleClick={onCellDoubleClick}
          />
        )
      })}
    </>
  )
}

function GridCell({
  taskId, filename, imageUrl, sampleIdx, isSelected, tooltip, onClick, onDoubleClick,
}: {
  taskId: number
  filename: string | null
  /** URL given by the server for disk-served entries; left empty for cache / active task, falling back to api.generateSampleUrl */
  imageUrl?: string
  sampleIdx: number | null
  isSelected: boolean
  tooltip: string
  onClick?: (idx: number) => void
  onDoubleClick?: (idx: number) => void
}) {
  const { t } = useTranslation()
  const [errored, setErrored] = useState(false)

  // src prefers imageUrl (the URL served by the server for disk history), else falls back to the cache URL
  const src = imageUrl ?? (filename ? api.generateSampleUrl(taskId, filename) : null)

  // Reset errored when task/src changes (e.g. clicking into history playback), so the img
  // retries loading. Otherwise a stale errored=true would persist and a new src would still show "...".
  useEffect(() => {
    setErrored(false)
  }, [taskId, src])

  // Placeholder (no sample / cache miss): minHeight keeps the grid row from collapsing
  if (!src || errored) {
    return (
      <div
        className="grid place-items-center rounded-sm border border-subtle bg-sunken text-fg-tertiary text-2xs"
        style={{ minHeight: 80 }}
      >
        {errored ? t('generate.originalReleased') : '…'}
      </div>
    )
  }
  return (
    <button
      onClick={(e) => {
        // A plain click yields to pan (drag scenario); only Ctrl/Cmd+click selects a cell
        if (sampleIdx == null) return
        if (e.ctrlKey || e.metaKey) onClick?.(sampleIdx)
      }}
      onDoubleClick={() => sampleIdx != null && onDoubleClick?.(sampleIdx)}
      className={`block p-0 overflow-hidden rounded-sm border-2 bg-sunken ${
        isSelected ? 'border-accent' : 'border-transparent hover:border-dim'
      }`}
      title={tooltip}
      style={{ minHeight: 80 }}
    >
      {/* Keying on src forces React to remount the img when src changes, avoiding a
          leftover failed browser cache entry or onError state from the previous image */}
      <img
        key={src}
        src={src}
        className="block w-full h-auto pointer-events-none"
        alt={filename ?? ''}
        loading="lazy"
        draggable={false}
        onError={() => setErrored(true)}
        onLoad={() => setErrored(false)}
      />
    </button>
  )
}
