/**
 * TaskSampleStrip —— 队列列表行内联的采样图带。
 *
 * 队列页每行（有 monitor_state_path 的训练任务）底下铺一条横向缩略图带，
 * 不用点进任务详情就能翻这次训练出的图；点任一张开灯箱（ImagePreviewModal，
 * ← / → 在本任务的采样序列里前后切）。
 *
 * 取数走 `GET /api/queue/{id}/samples`（扫盘，已结束的任务也有图），而不是
 * monitor state —— 后者只有 running task 推、且 cap 50。
 *
 * 两个「别把队列页拖垮」的约束：
 * - **懒加载**：IntersectionObserver，行滚进视口才发请求；队列几百行时只有
 *   屏幕上那几行真的打后端。
 * - **live 防抖**：running task 订阅 monitor_progress，只在 delta 里真有
 *   appended_samples 时才安排一次 1.5s 后的重拉（训练每几十步出一张图，
 *   没必要跟着每条 delta 抖）。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { api, type TaskSample } from '../api/client'
import { useEventStream } from '../lib/useEventStream'
import ImagePreviewModal from './ImagePreviewModal'

const THUMB_PX = 72
/** 缩略图请求宽度：2× 显示尺寸，HiDPI 屏不糊。 */
const THUMB_REQ = 160

function marks(s: TaskSample): string {
  const parts: string[] = []
  if (s.epoch != null) parts.push(`ep ${s.epoch.toLocaleString()}`)
  if (s.step != null) parts.push(`step ${s.step.toLocaleString()}`)
  return parts.join(' · ')
}

function shortMarks(s: TaskSample): string {
  const parts: string[] = []
  if (s.epoch != null) parts.push(`ep${s.epoch}`)
  if (s.step != null) parts.push(String(s.step))
  return parts.join('·')
}

export default function TaskSampleStrip({ taskId, live = false }: {
  taskId: number
  /** running task → 订阅 monitor_progress，出新图自动追加。 */
  live?: boolean
}) {
  const { t } = useTranslation()
  const [items, setItems] = useState<TaskSample[] | null>(null)
  const [zoomIdx, setZoomIdx] = useState<number | null>(null)
  const hostRef = useRef<HTMLDivElement | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const [inView, setInView] = useState(false)
  const refetchTimer = useRef<number | null>(null)
  // 用户自己往回滚看早期图时别把他拽回末尾；只在「本来就贴着右端」时跟随。
  const followRef = useRef(true)

  const load = useCallback(async () => {
    try {
      const r = await api.listTaskSamples(taskId)
      setItems(r.items)
    } catch {
      setItems([])  // 网络错 / task 没了 → 当作没图，队列页不弹错
    }
  }, [taskId])

  // 懒加载：滚进视口（提前 200px）才拉一次。
  useEffect(() => {
    const el = hostRef.current
    if (!el) return
    if (typeof IntersectionObserver === 'undefined') { setInView(true); return }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setInView(true)
          io.disconnect()
        }
      },
      { rootMargin: '200px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  useEffect(() => {
    if (!inView) return
    void load()
  }, [inView, load])

  useEventStream((evt) => {
    if (!live || !inView) return
    if (evt.type !== 'monitor_progress') return
    if (String(evt.task_id) !== String(taskId)) return
    const delta = evt.delta as { appended_samples?: unknown[] } | undefined
    if (!delta?.appended_samples?.length) return
    if (refetchTimer.current) return
    refetchTimer.current = window.setTimeout(() => {
      refetchTimer.current = null
      void load()
    }, 1500)
  })

  useEffect(() => () => {
    if (refetchTimer.current) window.clearTimeout(refetchTimer.current)
  }, [])

  // 新图追加后跟到末尾（仅在用户没往回滚时）。
  const count = items?.length ?? 0
  useEffect(() => {
    const el = scrollRef.current
    if (!el || !followRef.current) return
    el.scrollLeft = el.scrollWidth
  }, [count])

  if (items !== null && items.length === 0) return null

  return (
    <div
      ref={hostRef}
      className="border-t border-subtle px-[22px] py-2.5"
      // 行本身是「点进详情」的按钮，采样条的点击不该顺带导航。
      onClick={(e) => e.stopPropagation()}
    >
      {items === null ? (
        <div className="flex gap-1.5">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="rounded-sm bg-overlay opacity-40 shrink-0"
              style={{ width: THUMB_PX, height: THUMB_PX }}
            />
          ))}
        </div>
      ) : (
        <>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-xs text-fg-tertiary font-mono uppercase tracking-wider">
              {t('queue.samples')}
            </span>
            <span className="text-xs text-fg-tertiary font-mono">
              {t('queue.samplesCount', { n: items.length })}
            </span>
          </div>
          <div
            ref={scrollRef}
            onScroll={(e) => {
              const el = e.currentTarget
              followRef.current =
                el.scrollWidth - el.scrollLeft - el.clientWidth < 24
            }}
            className="flex gap-1.5 overflow-x-auto pb-1"
            style={{ scrollbarWidth: 'thin' }}
            data-testid={`sample-strip-${taskId}`}
          >
            {items.map((s, i) => {
              const caption = shortMarks(s)
              return (
                <button
                  key={s.filename}
                  type="button"
                  onClick={() => setZoomIdx(i)}
                  title={marks(s) || s.filename}
                  className="shrink-0 flex flex-col items-center gap-0.5 p-0 bg-transparent border-none cursor-zoom-in"
                >
                  <div
                    className="rounded-sm overflow-hidden border border-subtle hover:border-accent transition-colors bg-sunken"
                    style={{ width: THUMB_PX, height: THUMB_PX }}
                  >
                    <img
                      src={api.sampleImageUrl(s.filename, taskId, THUMB_REQ)}
                      alt=""
                      loading="lazy"
                      className="w-full h-full object-cover block"
                    />
                  </div>
                  {caption && (
                    <span className="text-[10px] font-mono leading-tight text-fg-tertiary">
                      {caption}
                    </span>
                  )}
                </button>
              )
            })}
          </div>
        </>
      )}

      {zoomIdx !== null && items && items[zoomIdx] && (
        <ImagePreviewModal
          src={api.sampleImageUrl(items[zoomIdx].filename, taskId)}
          caption={[marks(items[zoomIdx]), items[zoomIdx].filename]
            .filter(Boolean).join(' · ')}
          index={zoomIdx}
          total={items.length}
          hasPrev={zoomIdx > 0}
          hasNext={zoomIdx < items.length - 1}
          onClose={() => setZoomIdx(null)}
          onPrev={() => setZoomIdx((i) => Math.max(0, (i ?? 0) - 1))}
          onNext={() => setZoomIdx((i) => Math.min(items.length - 1, (i ?? 0) + 1))}
        />
      )}
    </div>
  )
}
