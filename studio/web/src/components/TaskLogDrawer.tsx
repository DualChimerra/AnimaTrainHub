import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

export type LogSourceStatus =
  | 'pending'
  | 'running'
  | 'done'
  | 'failed'
  | 'canceled'
  | 'paused'
  | 'scheduled'

/** A replayable task log stream (works for job / task / frontend-synthesized logs alike). */
export interface LogSource {
  key: string
  label: string
  status: LogSourceStatus
  lines: string[]
  /** Epoch in seconds; when multiple sources are all in a terminal state, this picks the "most recent" one, and is also used for the elapsed-time display on the bar. */
  startedAt?: number | null
  finishedAt?: number | null
  /** Default = not cancelable (e.g. a frontend-synthesized log for a dedup scan). */
  onCancel?: () => void
  /** Shows a retry button on the header's right side when failed; default = not retryable. */
  onRetry?: () => void
}

const STATUS_BADGE: Record<LogSourceStatus, string> = {
  pending: 'ds-badge ds-mute',
  running: 'ds-badge ds-ok',
  done: 'ds-badge ds-mute',
  failed: 'ds-badge ds-err',
  canceled: 'ds-badge ds-mute',
  paused: 'ds-badge ds-warn',
  // scheduled jobs have not run yet and have no log; same look as pending.
  scheduled: 'ds-badge ds-mute',
}

const isLiveStatus = (s: LogSourceStatus) => s === 'pending' || s === 'running'

/** Picking one source to display among several (per issue #251): a live one wins, otherwise the most recently started.
 *  An old task's artifacts have already been superseded by the new task, so historical logs don't get multiple entry points. */
function pickActive(sources: LogSource[]): LogSource | null {
  if (sources.length === 0) return null
  const live = sources.find((s) => isLiveStatus(s.status))
  if (live) return live
  return [...sources].sort((a, b) => (b.startedAt ?? 0) - (a.startedAt ?? 0))[0]
}

/**
 * Task log drawer -- the app-wide unified task progress/log UI (issue #251).
 *
 * Shape: a page-level footer. Collapsed, a single full-width header sticks to the bottom of the page (status badge +
 * last log line + elapsed time); clicking it raises the log panel from below the header (a 200ms height animation,
 * an overlay that doesn't squeeze the page layout), with the header sitting on top of the panel as a divider from the content area.
 *
 * Open/close state machine (manual toggling always takes effect):
 * - Entering live (including a replay scenario that's already running on mount) -> auto-expands
 * - Task finishing does **not** auto-collapse -- the user needs to look back at the result/error; only switching
 *   pages (component unmount) or a manual click collapses it
 * - Mounted already in a terminal state (historical replay) -> collapsed by default
 *
 * Mounted uniformly by StepShell (the `logSources` prop); pages only declare the source.
 */
export default function TaskLogDrawer({
  sources,
}: {
  sources: Array<LogSource | null | undefined | false>
}) {
  const { t } = useTranslation()
  const list = sources.filter((s): s is LogSource => !!s)
  const active = pickActive(list)

  const [expanded, setExpanded] = useState(false)
  const preRef = useRef<HTMLPreElement>(null)
  const prevRef = useRef<{ key: string; status: LogSourceStatus } | null>(null)

  const activeKey = active?.key ?? null
  const activeStatus = active?.status ?? null
  useEffect(() => {
    if (!activeKey || !activeStatus) return
    const prev = prevRef.current
    const wasLive = prev?.key === activeKey && isLiveStatus(prev.status)
    if (isLiveStatus(activeStatus) && !wasLive) setExpanded(true)
    prevRef.current = { key: activeKey, status: activeStatus }
  }, [activeKey, activeStatus])

  // Ticks every 1s while live to refresh the elapsed-time display (same as the old JobProgress)
  const live = !!activeStatus && isLiveStatus(activeStatus)
  const [, setTick] = useState(0)
  useEffect(() => {
    if (!live) return
    const id = window.setInterval(() => setTick((n) => n + 1), 1000)
    return () => window.clearInterval(id)
  }, [live])

  // Scrolls to follow the log to the bottom while expanded
  const lineCount = active?.lines.length ?? 0
  useEffect(() => {
    if (expanded && preRef.current) {
      preRef.current.scrollTop = preRef.current.scrollHeight
    }
  }, [expanded, lineCount])

  if (!active) return null

  const elapsed = active.startedAt
    ? (active.finishedAt ?? Date.now() / 1000) - active.startedAt
    : null
  const lastLine = active.lines[active.lines.length - 1] ?? ''

  return (
    <>
      {/* Placeholder: same height as the footer header, so page content isn't hidden behind the bottom-anchored header */}
      <div className="shrink-0" style={{ height: 34 }} aria-hidden />
      {/* The footer drawer itself: stuck to the page bottom, full width, no rounding or margin; anchored bottom,
          when the body height animates 0 <-> 40vh the header rides up with the drawer, acting as the divider between content and log */}
      <div
        className={`absolute bottom-0 inset-x-0 z-30 flex flex-col ${expanded ? 'shadow-2xl' : ''}`}
      >
        <div
          role="button"
          tabIndex={0}
          aria-expanded={expanded}
          onClick={() => setExpanded((v) => !v)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              setExpanded((v) => !v)
            }
          }}
          className="ds-logbar cursor-pointer select-none"
          style={{ padding: '0 20px' }}
        >
          <span
            className={`ds-dot${live ? ' ds-dot-run' : ''}`}
            style={live ? undefined : { background: active.status === 'failed' ? 'var(--err)' : 'var(--line-3)' }}
          />
          <span className="shrink-0" style={{ color: 'var(--ink-2)' }}>{active.label}</span>
          <span className={STATUS_BADGE[active.status]}>{t(`status.${active.status}`, { defaultValue: active.status })}</span>
          {elapsed != null && elapsed > 0 && (
            <span className="ds-mono shrink-0">{Math.round(elapsed)}s</span>
          )}
          <span className="ds-mono truncate flex-1 min-w-0">{lastLine}</span>
          {live && active.onCancel && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                active.onCancel?.()
              }}
              className="ds-ctl ds-ghost"
              style={{ height: 24, color: 'var(--err)' }}
            >
              {t('common.cancel')}
            </button>
          )}
          {active.status === 'failed' && active.onRetry && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                active.onRetry?.()
              }}
              className="ds-ctl ds-ghost"
              style={{ height: 24 }}
            >
              {t('common.retry')}
            </button>
          )}
          <span className="ds-mono shrink-0" aria-hidden>{expanded ? t('logDrawer.collapse') : t('logDrawer.expand')} {expanded ? '↓' : '↑'}</span>
        </div>
        <div
          data-testid="log-drawer-body"
          className="overflow-hidden transition-[height] duration-200 ease-out"
          style={{ height: expanded ? '40vh' : '0px' }}
        >
          <pre
            ref={preRef}
            className="m-0 h-full py-2 text-[11px] leading-relaxed font-mono overflow-y-auto whitespace-pre-wrap break-words"
            style={{ padding: '10px 20px', background: 'var(--panel)', color: 'var(--ink-2)' }}
          >
            {active.lines.length === 0
              ? t('jobProgress.waitingLogs')
              : active.lines.slice(-1000).join('\n')}
          </pre>
        </div>
      </div>
    </>
  )
}
