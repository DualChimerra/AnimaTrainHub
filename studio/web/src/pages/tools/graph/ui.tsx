/** Small shared pieces of the Graph UI. */
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'
import { formatNumber, optionOf, scientific } from './model'
import type { Param, RunStatus, RunValue } from './types'
import { graphApi } from './api'

// ── icons ───────────────────────────────────────────────────────────────────
const svg = (d: ReactNode, size = 14, w = 1.8) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={w}
    strokeLinecap="round" strokeLinejoin="round" aria-hidden>{d}</svg>
)
export const Ico = {
  plus: svg(<path d="M12 5v14M5 12h14" />),
  close: svg(<path d="M18 6 6 18M6 6l12 12" />),
  search: svg(<><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></>),
  star: svg(<path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z" />),
  starFill: <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden><path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z" /></svg>,
  heart: svg(<path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z" />),
  heartFill: <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden><path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z" /></svg>,
  image: svg(<><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="9" cy="9" r="1.6" /><path d="m21 15-5-5L5 21" /></>),
  copy: svg(<><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h10" /></>),
  compare: svg(<><rect x="3" y="4" width="7" height="16" rx="1.5" /><rect x="14" y="4" width="7" height="16" rx="1.5" /></>),
  sliders: svg(<><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" /><circle cx="16" cy="6" r="2" /><circle cx="10" cy="12" r="2" /><circle cx="18" cy="18" r="2" /></>),
  queue: svg(<><path d="M4 6h16M4 12h10M4 18h16" /><circle cx="18" cy="12" r="2" fill="currentColor" /></>),
  chev: svg(<path d="m9 6 6 6-6 6" />, 12, 2.2),
  chevD: svg(<path d="m6 9 6 6 6-6" />, 12, 2.2),
  up: svg(<path d="m6 15 6-6 6 6" />, 12, 2.2),
  down: svg(<path d="m6 9 6 6 6-6" />, 12, 2.2),
  download: svg(<path d="M12 4v12m0 0-4-4m4 4 4-4M4 20h16" />),
  play: svg(<path d="M7 5v14l11-7z" />),
  target: svg(<><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="3" /></>),
  grid: svg(<><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>),
  table: svg(<><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 10h18M9 10v10" /></>),
  matrix: svg(<><path d="M4 4v16h16" /><rect x="8" y="12" width="3" height="4" /><rect x="13" y="8" width="3" height="8" /><rect x="18" y="5" width="2" height="11" /></>),
  web: svg(<><circle cx="6" cy="6" r="2.2" /><circle cx="18" cy="7" r="2.2" /><circle cx="12" cy="18" r="2.2" /><circle cx="19" cy="17" r="1.6" /><path d="M8 7l8 0M7 8l4 8M17 9l-4 7M14 18h3.5" /></>),
  link: svg(<><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1" /><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" /></>),
  trash: svg(<><path d="M4 7h16M10 11v6M14 11v6" /><path d="M6 7l1 13h10l1-13M9 7V4h6v3" /></>),
  eye: svg(<><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></>),
  eyeOff: svg(<><path d="M3 3l18 18M10.6 5.1A10 10 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-3 3.9M6.5 6.6C3.8 8.4 2 12 2 12s3.5 7 10 7a9.6 9.6 0 0 0 4.4-1" /></>),
  archive: svg(<><rect x="3" y="4" width="18" height="4" rx="1" /><path d="M5 8v11h14V8M10 12h4" /></>),
  filter: svg(<path d="M3 5h18l-7 8v6l-4-2v-4z" />),
}

// ── value labels ────────────────────────────────────────────────────────────

/** Human label for a parameter value; archived options still resolve. */
export function useValueLabel() {
  const { t } = useTranslation()
  return useCallback((p: Param | undefined, v: RunValue | undefined): string => {
    if (v === undefined) return t('graph.unset')
    if (!p) return String(v)
    if (typeof v === 'number') return formatNumber(v)
    if (p.type === 'text') return v
    const o = optionOf(p, v)
    if (!o) return String(v)
    if (p.type === 'bool') {
      if (o.label === 'Yes' || (!o.label && o.id === 'yes')) return t('graph.yes')
      if (o.label === 'No' || (!o.label && o.id === 'no')) return t('graph.no')
    }
    return o.label || o.id
  }, [t])
}

export function NumberTitle({ v }: { v: RunValue | undefined }) {
  if (typeof v !== 'number') return null
  const s = scientific(v)
  return s ? <span className="gr-sci">{s}</span> : null
}

// ── status / rating ─────────────────────────────────────────────────────────

const STATUS_TONE: Record<RunStatus, string> = {
  planned: 'ds-mute', training: 'ds-info', done: 'ds-ok', stopped: 'ds-warn',
}

export function StatusBadge({ status }: { status: RunStatus }) {
  const { t } = useTranslation()
  return (
    <span className={`ds-badge ${STATUS_TONE[status]}`}>
      {status === 'training' && <span className="gr-pulse" aria-hidden />}
      {t(`graph.status_${status}`)}
    </span>
  )
}

/** 1–3 stars; clicking the current value clears it. */
export function Stars({ value, onChange, size = 14 }: {
  value: number | null
  onChange?: (v: number | null) => void
  size?: number
}) {
  const { t } = useTranslation()
  return (
    <span className="gr-stars" role={onChange ? 'radiogroup' : undefined} aria-label={t('graph.rating')}>
      {[1, 2, 3].map((n) => {
        const on = (value ?? 0) >= n
        const icon = <span style={{ width: size, height: size, display: 'inline-grid' }}>{on ? Ico.starFill : Ico.star}</span>
        return onChange ? (
          <button
            key={n} type="button" className={`gr-star${on ? ' gr-on' : ''}`}
            onClick={(e) => { e.stopPropagation(); onChange(value === n ? null : n) }}
            aria-label={t('graph.rateN', { n })} title={t(`graph.rating_${n}`)}
          >{icon}</button>
        ) : (
          <span key={n} className={`gr-star${on ? ' gr-on' : ''}`}>{icon}</span>
        )
      })}
    </span>
  )
}

// ── thumbnails ──────────────────────────────────────────────────────────────

/** Every preview in the Graph uses this one size: the server makes it once
 *  (right when the image is added) and the browser caches it, so cards,
 *  table rows, the drawer and the web all reuse the same file. */
export const THUMB_PX = 384

/** Lazy thumbnail. */
export function Thumb({ id, alt, className, style, contain }: {
  id: number; alt?: string; className?: string; style?: React.CSSProperties; contain?: boolean
}) {
  const [failed, setFailed] = useState(false)
  const px = THUMB_PX
  if (failed) return <div className={`gr-thumb-missing ${className ?? ''}`} style={style}>{Ico.image}</div>
  return (
    <img
      src={graphApi.imageUrl(id, px)} alt={alt ?? ''} loading="lazy" decoding="async" draggable={false}
      className={className} style={{ objectFit: contain ? 'contain' : 'cover', ...style }}
      onError={() => setFailed(true)}
    />
  )
}

// ── modal shell ─────────────────────────────────────────────────────────────

export function Modal({ title, sub, onClose, children, foot, width = 560, wide }: {
  title: ReactNode
  sub?: ReactNode
  onClose: () => void
  children: ReactNode
  foot?: ReactNode
  width?: number
  wide?: boolean
}) {
  const { t } = useTranslation()
  const closeRef = useRef(onClose)
  closeRef.current = onClose
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); closeRef.current() } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  return createPortal(
    <div className="gr-modal-back ds-backdrop-anim" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div
        className="gr-modal ds-dialog-anim" role="dialog" aria-modal="true"
        style={{ width: wide ? 'min(1180px, calc(100vw - 32px))' : `min(${width}px, calc(100vw - 32px))` }}
      >
        <div className="gr-modal-head">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="ds-modal-title">{title}</div>
            {sub && <div className="ds-card-sub">{sub}</div>}
          </div>
          <button type="button" className="ds-kebab" onClick={onClose} aria-label={t('common.close')}>{Ico.close}</button>
        </div>
        <div className="gr-modal-body">{children}</div>
        {foot && <div className="gr-modal-foot">{foot}</div>}
      </div>
    </div>,
    document.body,
  )
}

/** Pill toggle used for filter values and pickers. */
export function Chip({ on, onClick, children, count, muted, title }: {
  on: boolean; onClick: () => void; children: ReactNode; count?: number; muted?: boolean; title?: string
}) {
  return (
    <button
      type="button" onClick={onClick} title={title} aria-pressed={on}
      className={`gr-chip${on ? ' gr-on' : ''}${muted ? ' gr-muted' : ''}`}
    >
      <span className="gr-chip-l">{children}</span>
      {count !== undefined && <b>{count}</b>}
    </button>
  )
}

export function Switch({ on, onChange, label }: { on: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button type="button" className="gr-switch" onClick={() => onChange(!on)} aria-pressed={on}>
      <span className={`ds-switch${on ? ' ds-on' : ''}`}><i /></span>
      <span>{label}</span>
    </button>
  )
}

export function fmtDate(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString(undefined, { day: '2-digit', month: 'short' })
}
