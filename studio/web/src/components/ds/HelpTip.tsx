import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'

/** The one way the app explains something: a small "?" next to a name or a
 *  title. Hover or focus shows the bubble, a click pins it (touch screens).
 *  The bubble is portalled to <body> so card overflow never clips it. */
export default function HelpTip({ children, label }: { children: ReactNode; label?: string }) {
  const { t } = useTranslation()
  const id = useId()
  const anchor = useRef<HTMLButtonElement | null>(null)
  const bubble = useRef<HTMLDivElement | null>(null)
  const [hover, setHover] = useState(false)
  const [pinned, setPinned] = useState(false)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)
  const open = hover || pinned

  useLayoutEffect(() => {
    if (!open || !anchor.current || !bubble.current) return
    const a = anchor.current.getBoundingClientRect()
    const b = bubble.current.getBoundingClientRect()
    const vw = window.innerWidth
    const vh = window.innerHeight
    const left = Math.max(8, Math.min(a.left - 10, vw - b.width - 8))
    const top = a.bottom + 6 + b.height <= vh - 8 ? a.bottom + 6 : Math.max(8, a.top - 6 - b.height)
    setPos({ left, top })
  }, [open])

  useEffect(() => {
    if (!pinned) return
    const onDown = (e: MouseEvent) => {
      const n = e.target as Node
      if (anchor.current?.contains(n) || bubble.current?.contains(n)) return
      setPinned(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setPinned(false) }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [pinned])

  return (
    <>
      <button
        ref={anchor}
        type="button"
        className={`ds-help${open ? ' ds-is-open' : ''}`}
        aria-label={label ?? t('infoButton.label')}
        aria-describedby={open ? id : undefined}
        aria-expanded={pinned}
        onMouseEnter={() => { setPos(null); setHover(true) }}
        onMouseLeave={() => setHover(false)}
        onFocus={() => { setPos(null); setHover(true) }}
        onBlur={() => setHover(false)}
        onClick={(e) => {
          // inside a <label> or a clickable row the "?" must not act on it
          e.preventDefault()
          e.stopPropagation()
          setPinned((v) => !v)
        }}
      >
        ?
      </button>
      {open && createPortal(
        <div
          ref={bubble}
          id={id}
          role="tooltip"
          className="ds-tipbox"
          style={pos ? { left: pos.left, top: pos.top, pointerEvents: pinned ? 'auto' : 'none' } : { left: 0, top: 0, visibility: 'hidden' }}
        >
          {children}
        </div>,
        document.body,
      )}
    </>
  )
}
