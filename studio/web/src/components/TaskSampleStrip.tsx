/**
 * TaskSampleStrip -- inline sample-image strip on a queue list row.
 *
 * Below each queue-page row for a training task with a monitor_state_path, a horizontal thumbnail strip lets you
 * flip through this run's sample images without opening the task detail; clicking one opens a lightbox
 * (ImagePreviewModal, left/right cycles through this task's sample sequence).
 *
 * Data comes from `GET /api/queue/{id}/samples` (a disk scan, so finished tasks have images too), not from
 * monitor state -- the latter is only pushed for running tasks and capped at 50.
 *
 * Two constraints to keep the queue page from bogging down:
 * - **Lazy loading**: an IntersectionObserver only fires the request once a row scrolls into view; with hundreds
 *   of queue rows, only the ones on screen actually hit the backend.
 * - **Live debouncing**: a running task subscribes to monitor_progress, and only schedules a refetch 1.5s later
 *   when the delta actually contains appended_samples (training emits an image every few dozen steps,
 *   no need to jitter on every single delta).
 */
import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { api, type TaskSample } from '../api/client'
import { useEventStream } from '../lib/useEventStream'
import ImagePreviewModal from './ImagePreviewModal'

const THUMB_PX = 84
/** Requested thumbnail width: 2x the display size, so it doesn't look blurry on HiDPI screens. */
const THUMB_REQ = 192

function marks(s: TaskSample): string {
  const parts: string[] = []
  if (s.epoch != null) parts.push(`ep ${s.epoch.toLocaleString()}`)
  if (s.step != null) parts.push(`step ${s.step.toLocaleString()}`)
  return parts.join(' · ')
}

function shortMarks(s: TaskSample): string {
  const parts: string[] = []
  if (s.epoch != null) parts.push(`ep ${s.epoch}`)
  if (s.step != null) parts.push(s.step.toLocaleString())
  return parts.join(' · ')
}

// Memoized: the queue page re-renders on every monitor_progress delta of the running task;
// the strip (and its open lightbox) only depends on taskId / live.
export default memo(TaskSampleStrip)

function TaskSampleStrip({ taskId, live = false }: {
  taskId: number
  /** running task -> subscribes to monitor_progress, auto-appends new images. */
  live?: boolean
}) {
  const { t } = useTranslation()
  const [items, setItems] = useState<TaskSample[] | null>(null)
  const [zoomIdx, setZoomIdx] = useState<number | null>(null)
  const hostRef = useRef<HTMLDivElement | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const [inView, setInView] = useState(false)
  const refetchTimer = useRef<number | null>(null)
  // Don't drag the user back to the end while they're scrolling back to look at earlier images; only follow when they were already stuck to the right edge.
  const followRef = useRef(true)

  const load = useCallback(async () => {
    try {
      const r = await api.listTaskSamples(taskId)
      // Oldest first, so the strip reads left → right as training went on.
      setItems([...r.items].sort((x, y) =>
        (x.epoch ?? -1) - (y.epoch ?? -1) || (x.step ?? -1) - (y.step ?? -1)))
    } catch {
      setItems([])  // Network error / task gone -> treat as no images, don't pop an error on the queue page
    }
  }, [taskId])

  // Lazy load: only fetches once it scrolls into view (200px ahead of time).
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

  // Follow to the end once a new image is appended (only when the user hasn't scrolled back).
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
      className="ds-samples"
      // The row itself is a "go to detail" button; a click on the sample strip shouldn't also navigate.
      onClick={(e) => e.stopPropagation()}
    >
      <div className="ds-samples-head">
        <span className="ds-cap">{t('queue.samples')}</span>
        {items && <span className="ds-samples-count">{items.length}</span>}
      </div>
      {items === null ? (
        <div className="ds-samples-row">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="ds-sample-thumb ds-skel" style={{ width: THUMB_PX, height: THUMB_PX }} />
          ))}
        </div>
      ) : (
        <div
          ref={scrollRef}
          onScroll={(e) => {
            const el = e.currentTarget
            followRef.current =
              el.scrollWidth - el.scrollLeft - el.clientWidth < 24
          }}
          className="ds-samples-row"
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
                className="ds-sample"
              >
                <span className="ds-sample-thumb" style={{ width: THUMB_PX, height: THUMB_PX }}>
                  <img
                    src={api.sampleImageUrl(s.filename, taskId, THUMB_REQ)}
                    alt=""
                    loading="lazy"
                    decoding="async"
                  />
                </span>
                {caption && <span className="ds-sample-cap">{caption}</span>}
              </button>
            )
          })}
        </div>
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
          preload={[items[zoomIdx - 1], items[zoomIdx + 1]]
            .filter((x): x is TaskSample => !!x)
            .map((x) => api.sampleImageUrl(x.filename, taskId))}
        />
      )}
    </div>
  )
}
