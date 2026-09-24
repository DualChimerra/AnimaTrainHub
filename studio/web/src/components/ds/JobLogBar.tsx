import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'

/** The mockup's bottom status strip (.logbar) for background data jobs:
 *  a live dot, the job and its progress, and a "log ↑" toggle that opens the
 *  job's log tail above the strip. Sticks to the bottom of the page scroller. */
export default function JobLogBar({ title, running, detail, pct, log }: {
  /** Left label, e.g. "Задачи данных". */
  title: string
  running: boolean
  /** "#418 booru scrape · 262 / 400"; null shows the idle text. */
  detail: string | null
  /** 0–100 for the meter; null hides it. */
  pct?: number | null
  log: string[]
}) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const paneRef = useRef<HTMLPreElement | null>(null)

  useEffect(() => {
    if (open && paneRef.current) paneRef.current.scrollTop = paneRef.current.scrollHeight
  }, [open, log.length])

  return (
    <div style={{ position: 'sticky', bottom: 0, zIndex: 5 }}>
      {open && (
        <pre
          ref={paneRef}
          className="ds-mono"
          style={{
            margin: 0, maxHeight: 220, overflow: 'auto', padding: '10px 20px',
            background: 'var(--panel)', borderTop: '1px solid var(--line)',
            fontSize: 11, lineHeight: 1.55, color: 'var(--ink-2)', whiteSpace: 'pre-wrap',
          }}
        >
          {log.length > 0 ? log.join('\n') : t('dataset.logEmpty')}
        </pre>
      )}
      <div className="ds-logbar">
        <span className={`ds-dot${running ? ' ds-dot-run' : ''}`} style={running ? undefined : { background: 'var(--line-3)' }} />
        <span>{title}</span>
        <span className="ds-mono" style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {detail ?? t('dataset.logIdle')}
        </span>
        {pct != null && <span className="ds-meter" style={{ width: 160, flex: 'none' }}><i style={{ width: `${pct}%` }} /></span>}
        <button
          type="button"
          className="ds-mono"
          style={{ marginLeft: 'auto', color: 'inherit' }}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {t('dataset.logToggle')} {open ? '↓' : '↑'}
        </button>
      </div>
    </div>
  )
}
