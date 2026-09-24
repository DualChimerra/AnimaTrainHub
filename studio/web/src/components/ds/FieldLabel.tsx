import { useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

/** Setting name whose explanation opens on hover (or keyboard focus) instead of
 *  taking a line under the row. The bubble is portalled to <body> with fixed
 *  positioning so scroll containers and card overflow never clip it. */
export default function FieldLabel({ label, tip }: { label: string; tip?: ReactNode }) {
  const id = useId()
  const anchor = useRef<HTMLSpanElement | null>(null)
  const bubble = useRef<HTMLDivElement | null>(null)
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)

  useLayoutEffect(() => {
    if (!open || !anchor.current || !bubble.current) return
    const a = anchor.current.getBoundingClientRect()
    const b = bubble.current.getBoundingClientRect()
    const vw = window.innerWidth
    const vh = window.innerHeight
    const left = Math.max(8, Math.min(a.left, vw - b.width - 8))
    // Below the name when it fits, otherwise above it.
    const top = a.bottom + 6 + b.height <= vh - 8 ? a.bottom + 6 : Math.max(8, a.top - 6 - b.height)
    setPos({ left, top })
  }, [open])

  if (!tip) return <span className="ds-label">{label}</span>

  const show = () => { setPos(null); setOpen(true) }
  const hide = () => setOpen(false)
  return (
    <>
      <span
        ref={anchor}
        className="ds-label ds-has-tip"
        tabIndex={0}
        aria-describedby={open ? id : undefined}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        onKeyDown={(e) => { if (e.key === 'Escape') hide() }}
      >
        {label}
      </span>
      {open && createPortal(
        <div
          ref={bubble}
          id={id}
          role="tooltip"
          className="ds-tipbox"
          style={pos ? { left: pos.left, top: pos.top } : { left: 0, top: 0, visibility: 'hidden' }}
        >
          {tip}
        </div>,
        document.body,
      )}
    </>
  )
}
