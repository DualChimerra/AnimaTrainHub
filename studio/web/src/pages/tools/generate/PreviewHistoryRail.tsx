/** 右侧竖排出图时间线（0.17 P-I：从「只列 done 历史」升级成统一时间线）。
 *
 * item 两种：
 *  - live（pending / running，来自队列）：占位卡。pending 灰底「排队中」+ ✕；running
 *    脉冲边框「生成中」，点击回到实时视图。
 *  - done（来自 cache/disk 扫盘）：缩略图，点击回看（onSelect(entry)）。
 * live 恒在最上（最新提交的），done 按 createdAt 往下——天然一条时间线。
 *
 * - 按当前 mode 过滤（single / xy / compare）；live item 的 mode 由父组件按 runsRef 定。
 * - 定高窗口化虚拟滚动（item 同尺寸，uniform stride）：只渲染可见区间 + overscan，
 *   外层 total×stride 占位撑滚动条，item 绝对定位到 idx×stride。
 * - 顶部 [刷新] 重拉 disk-history（多 tab / 外部改盘后手动同步）。
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { Task } from '../../../api/client'
import { entryBadge, entryDisplayLabel, entryThumbUrl, type HistoryEntry } from './entryAdapter'

// item 84px 方形 + 6px 间距 = 90px stride（0.17：从 56 放大 1.5× —— 原来太小、✕ 易误触、
// 内部小字看不清）
// Square thumbs filling the 168px history column (13px padding) + 8px gap.
const ITEM_SIZE = 142
const ITEM_STRIDE = 150
const OVERSCAN = 4
const VIEWPORT_FALLBACK = 2000

/** 统一时间线一项：进行中（队列 task）或已完成（历史 entry）。 */
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
  /** 点 done 项回看 / 点 running 项回到实时视图。pending 项不触发（只可取消）。 */
  onSelect: (it: TimelineItem) => void
  /** 取消某条 pending/running 的 generate。 */
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
  return w != null ? `seed ${p.seed} · w ${w.toFixed(2)}` : `seed ${p.seed}`
}

function HistoryItem({ entry, selected, onSelect }: { entry: HistoryEntry; selected: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      className={`ds-thumb${selected ? ' ds-sel' : ''}`}
      style={{ width: '100%', height: ITEM_SIZE, padding: 0, cursor: 'pointer' }}
      onClick={onSelect}
      title={`${entryDisplayLabel(entry)} · ${new Date(entry.createdAt).toLocaleString()}`}
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

/** 进行中项：pending 灰占位 + ✕；running 脉冲边框，点击回到实时视图。 */
function LiveItem({ task, selected, onSelect, onCancel }: { task: Task; selected: boolean; onSelect: () => void; onCancel: () => void }) {
  const { t } = useTranslation()
  const running = task.status === 'running'
  return (
    <div
      className={`ds-thumb${selected ? ' ds-sel' : ''}`}
      style={{ width: '100%', height: ITEM_SIZE, cursor: running ? 'pointer' : undefined, outline: running ? undefined : '1px dashed var(--line-3)', outlineOffset: -1 }}
      onClick={running ? onSelect : undefined}
      title={`#${task.id} · ${running ? t('status.running') : t('status.queued')}`}
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
