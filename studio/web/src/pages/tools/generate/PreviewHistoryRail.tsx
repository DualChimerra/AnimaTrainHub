/** Right-hand vertical generation timeline (0.17 P-I: upgraded from "list done history only"
 * to a unified timeline).
 *
 * Two kinds of item:
 *  - live (pending / running, from the queue): a placeholder card. pending shows a gray
 *    "queued" badge + a x to cancel; running shows a pulsing border and "generating", and
 *    clicking it returns to the live view.
 *  - done (from a cache/disk scan): a thumbnail; clicking it opens that result (onSelect(entry)).
 * live items always sit on top (most recently submitted), done items flow down by createdAt --
 * naturally forming one timeline.
 *
 * - Filtered by the current mode (single / xy / compare); a live item's mode is decided by the
 *   parent component from runsRef.
 * - Fixed-height windowed virtual scrolling (uniform item size/stride): only the visible range +
 *   overscan is rendered; an outer total*stride placeholder keeps the scrollbar sized correctly,
 *   and each item is absolutely positioned at idx*stride.
 * - The [Refresh] button at the top re-fetches disk history (for manual sync across tabs / after
 *   external disk changes).
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { Task } from '../../../api/client'
import { entryBadge, entryDisplayLabel, entryThumbUrl, type HistoryEntry } from './entryAdapter'

// item is a 84px square + 6px gap = 90px stride (0.17: enlarged 1.5x from 56 -- the old size
// was too small, the x was easy to mis-tap, and the inner text was hard to read)
// Square thumbs filling the 168px history column (13px padding) + 8px gap.
const ITEM_SIZE = 142
const ITEM_STRIDE = 150
const OVERSCAN = 4
const VIEWPORT_FALLBACK = 2000

/** One unified timeline item: in progress (a queue task) or complete (a history entry). */
export type TimelineItem =
  | { kind: 'live'; task: Task; mode: 'single' | 'xy' | 'compare' }
  | { kind: 'done'; entry: HistoryEntry }

function itemMode(it: TimelineItem): 'single' | 'xy' | 'compare' {
  return it.kind === 'live' ? it.mode : it.entry.mode
}
function itemKey(it: TimelineItem): string {
  return it.kind === 'live' ? `live:${it.task.id}` : it.entry.id
}

interface Props {
  items: TimelineItem[]
  mode: 'single' | 'xy' | 'compare'
  /** Clicking a done item reopens it; clicking a running item returns to the live view. pending items don't fire this (they can only be canceled). */
  onSelect: (it: TimelineItem) => void
  /** Cancel a pending/running generate. */
  onCancel: (taskId: number) => void
  onRefresh?: () => Promise<void>
  loading?: boolean
  /** Entry id being reviewed; null = the live view (the running item is "now"). */
  selectedId?: string | null
}

export default function PreviewHistoryRail({
  items, mode, onSelect, onCancel, onRefresh, loading, selectedId = null,
}: Props) {
  const { t } = useTranslation()
  const list = useMemo(() => items.filter((it) => itemMode(it) === mode), [items, mode])

  const scrollRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportH, setViewportH] = useState(VIEWPORT_FALLBACK)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const update = () => setViewportH(el.clientHeight || VIEWPORT_FALLBACK)
    update()
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(update)
      ro.observe(el)
      return () => ro.disconnect()
    }
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])

  const total = list.length
  const start = Math.max(0, Math.floor(scrollTop / ITEM_STRIDE) - OVERSCAN)
  const end = Math.min(total, Math.ceil((scrollTop + viewportH) / ITEM_STRIDE) + OVERSCAN)
  const slice = list.slice(start, end)

  return (
    <div className="ds-card generate-history-rail" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div style={{ padding: '13px 13px 9px', display: 'flex', alignItems: 'center', gap: 6 }}>
        <span className="ds-cap" style={{ flex: 1 }}>{t('generate.history')}</span>
        {onRefresh && (
          <button
            type="button"
            className="ds-kebab"
            onClick={() => void onRefresh()}
            disabled={loading}
            title={t('generate.refreshHistoryTitle')}
            aria-label={t('generate.refreshHistory')}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 12a9 9 0 1 1-3-6.7L21 8" /><path d="M21 3v5h-5" /></svg>
          </button>
        )}
      </div>
      {total === 0 ? (
        <div className="ds-cell-key" style={{ padding: '0 13px 13px', textAlign: 'center' }}>{t('generate.noHistory')}</div>
      ) : (
        <div
          ref={scrollRef}
          style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '0 13px 13px' }}
          onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
        >
          <div style={{ height: total * ITEM_STRIDE - (ITEM_STRIDE - ITEM_SIZE), position: 'relative' }}>
            {slice.map((it, i) => {
              const idx = start + i
              const sel = it.kind === 'done'
                ? it.entry.id === selectedId
                : selectedId == null && it.task.status === 'running'
              return (
                <div
                  key={itemKey(it)}
                  style={{ position: 'absolute', top: idx * ITEM_STRIDE, left: 0, right: 0 }}
                >
                  {it.kind === 'done' ? (
                    <HistoryItem entry={it.entry} selected={sel} onSelect={() => onSelect(it)} />
                  ) : (
                    <LiveItem task={it.task} selected={sel} onSelect={() => onSelect(it)} onCancel={() => onCancel(it.task.id)} />
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

/** Thumb caption: "XY 3×2" for matrices, otherwise seed and the first LoRA weight. */
function entryCaption(entry: HistoryEntry): string {
  const badge = entryBadge(entry)
  if (badge) return badge
  const p = entry.params as Partial<HistoryEntry['params']> | undefined
  if (!p || p.seed == null) return entryDisplayLabel(entry)
  const w = p.loras?.[0]?.scale
  return w != null ? `seed ${p.seed}, w ${w.toFixed(2)}` : `seed ${p.seed}`
}

function HistoryItem({ entry, selected, onSelect }: { entry: HistoryEntry; selected: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      className={`ds-thumb${selected ? ' ds-sel' : ''}`}
      style={{ width: '100%', height: ITEM_SIZE, padding: 0, cursor: 'pointer' }}
      onClick={onSelect}
      title={`${entryDisplayLabel(entry)}, ${new Date(entry.createdAt).toLocaleString()}`}
    >
      <img
        src={entryThumbUrl(entry)}
        alt=""
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
        loading="lazy"
      />
      <span className="ds-cap">{entryCaption(entry)}</span>
    </button>
  )
}

/** In-progress item: pending shows a gray placeholder + x; running shows a pulsing border, clicking it returns to the live view. */
function LiveItem({ task, selected, onSelect, onCancel }: { task: Task; selected: boolean; onSelect: () => void; onCancel: () => void }) {
  const { t } = useTranslation()
  const running = task.status === 'running'
  return (
    <div
      className={`ds-thumb${selected ? ' ds-sel' : ''}`}
      style={{ width: '100%', height: ITEM_SIZE, cursor: running ? 'pointer' : undefined, outline: running ? undefined : '1px dashed var(--line-3)', outlineOffset: -1 }}
      onClick={running ? onSelect : undefined}
      title={`#${task.id} ${running ? t('status.running') : t('status.queued')}`}
    >
      {running
        ? <span className="ds-dot ds-dot-run" style={{ transform: 'scale(1.4)' }} />
        : <span>{t('status.queued')}</span>}
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); onCancel() }}
        className="ds-pin"
        style={{ cursor: 'pointer' }}
        title={t('common.cancel')}
        aria-label={t('common.cancel')}
        data-testid={`timeline-cancel-${task.id}`}
      >✕</button>
      <span className="ds-cap">{running ? t('generate.historyNow') : `#${task.id}`}</span>
    </div>
  )
}
