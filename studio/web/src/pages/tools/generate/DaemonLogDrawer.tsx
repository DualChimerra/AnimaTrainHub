import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api } from '../../../api/client'
import { useEventStream } from '../../../lib/useEventStream'
import { useToast } from '../../../components/Toast'

interface LogEntry {
  ts: number
  seq: number
  line: string
}

/** Drawer for the daemon's stderr ring buffer.
 *
 * - Slides up 40vh from the bottom, high z-index, covers the Generate page surface below it without taking up layout space
 * - translateY(100%) when hidden, fully invisible (not just visibility:hidden -- the whole drawer leaves the viewport)
 * - Fetches history via GET /api/generate/daemon/logs on first open, then increments via SSE daemon_log_line
 * - Closing and reopening: only shows the history + whatever's arrived since; nothing is lost (ring buffer maxlen=2000)
 */
export default function DaemonLogDrawer({
  open, onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [entries, setEntries] = useState<LogEntry[]>([])
  const [autoScroll, setAutoScroll] = useState(true)
  const seqRef = useRef(0)
  const scrollRef = useRef<HTMLDivElement | null>(null)

  // Fetches history on open; doesn't clear on close (stays visible immediately next time it opens)
  useEffect(() => {
    if (!open) return
    let cancelled = false
    void api.getDaemonLogs(seqRef.current).then((r) => {
      if (cancelled) return
      if (r.entries.length > 0) {
        setEntries((prev) => [...prev, ...r.entries])
        seqRef.current = r.next_seq
      } else if (seqRef.current === 0) {
        seqRef.current = r.next_seq
      }
    }).catch(() => { /* non-blocking */ })
    return () => { cancelled = true }
  }, [open])

  // SSE increments
  useEventStream(useCallback((evt) => {
    if (evt.type !== 'daemon_log_line') return
    const seq = typeof evt.seq === 'number' ? evt.seq : seqRef.current
    if (seq < seqRef.current) return  // ignore stale events
    seqRef.current = seq + 1
    setEntries((prev) => {
      const next = [...prev, {
        ts: Number(evt.ts) || Date.now() / 1000,
        seq,
        line: String(evt.line ?? ''),
      }]
      // Guard memory: the client also caps it at 2000 lines
      return next.length > 2000 ? next.slice(-2000) : next
    })
  }, []))

  // Auto-scroll to the bottom
  useEffect(() => {
    if (!autoScroll || !open) return
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [entries, autoScroll, open])

  const handleClear = () => {
    setEntries([])
  }

  const handleCopy = async () => {
    const text = entries.map((e) => e.line).join('\n')
    try {
      await navigator.clipboard.writeText(text)
      toast(t('generate.logDrawerCopied'), 'success')
    } catch {
      toast(t('queueDetail.copyFailed'), 'error')
    }
  }

  return (
    <div
      aria-hidden={!open}
      style={{
        position: 'fixed',
        left: 0, right: 0, bottom: 0,
        height: '40vh',
        background: 'var(--bg-elevated)',
        borderTop: '1px solid var(--border-default)',
        boxShadow: open ? '0 -8px 32px rgba(0,0,0,0.4)' : 'none',
        transform: open ? 'translateY(0)' : 'translateY(100%)',
        transition: 'transform 220ms ease',
        zIndex: 60,
        display: 'flex',
        flexDirection: 'column',
        pointerEvents: open ? 'auto' : 'none',
      }}
    >
      <header
        style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '8px 16px',
          borderBottom: '1px solid var(--border-subtle)',
          flexShrink: 0,
        }}
      >
        <span style={{ fontSize: 'var(--t-sm)', fontWeight: 600, color: 'var(--fg-primary)' }}>
          {t('generate.logDrawerTitle')}
        </span>
        <span style={{ fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)', fontFamily: 'var(--font-mono)' }}>
          {entries.length}
        </span>
        <span style={{ flex: 1 }} />
        <label className="flex items-center gap-1 text-xs text-fg-secondary cursor-pointer">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
          />
          {t('generate.logDrawerAutoScroll')}
        </label>
        <button className="btn btn-ghost btn-sm" onClick={handleCopy} disabled={entries.length === 0}>
          {t('generate.logDrawerCopy')}
        </button>
        <button className="btn btn-ghost btn-sm" onClick={handleClear} disabled={entries.length === 0}>
          {t('generate.logDrawerClear')}
        </button>
        <button className="btn btn-ghost btn-sm" onClick={onClose}>
          {t('generate.logDrawerClose')}
        </button>
      </header>
      <div
        ref={scrollRef}
        style={{
          flex: 1, minHeight: 0,
          overflowY: 'auto',
          padding: '8px 16px',
          fontFamily: 'var(--font-mono)',
          fontSize: 'var(--t-xs)',
          lineHeight: 1.5,
          background: 'var(--bg-sunken)',
        }}
      >
        {entries.length === 0 ? (
          <div style={{ color: 'var(--fg-tertiary)', fontStyle: 'italic' }}>
            {t('generate.logDrawerEmpty')}
          </div>
        ) : (
          entries.map((e) => (
            <div key={e.seq} style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', color: 'var(--fg-primary)' }}>
              {e.line}
            </div>
          ))
        )}
      </div>
    </div>
  )
}
