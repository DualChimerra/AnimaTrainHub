import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { playSound } from '../../lib/sound'

/** Where the surface hangs: under an element, or at a pointer position
 *  (context menus). */
export type PopoverAnchor = HTMLElement | { x: number; y: number }

/**
 * One floating surface for every menu, dropdown and context menu, so they all
 * look, animate and dismiss the same way.
 *
 * Portalled into <body> with fixed positioning (cards with overflow:hidden
 * never clip it), placed under the anchor and flipped above it when there is
 * no room below. Closes on an outside press, Escape, window resize, or a
 * scroll outside the surface.
 */
export default function Popover({
  anchor, onClose, children, align = 'start', matchWidth = false, minWidth, maxHeight = 340,
  role = 'menu', ariaLabel, className = '', style, silent = false,
}: {
  anchor: PopoverAnchor
  onClose: () => void
  children: ReactNode
  /** Edge of the anchor the surface lines up with. */
  align?: 'start' | 'end'
  /** At least as wide as the anchor element. */
  matchWidth?: boolean
  minWidth?: number
  maxHeight?: number
  role?: string
  ariaLabel?: string
  className?: string
  style?: CSSProperties
  /** Skip the "open" cue (dropdowns play their own on pick). */
  silent?: boolean
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [pos, setPos] = useState<{ left: number; top: number; minWidth?: number; origin: string } | null>(null)
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const m = el.getBoundingClientRect()
    const vw = window.innerWidth
    const vh = window.innerHeight
    const r = anchor instanceof HTMLElement
      ? anchor.getBoundingClientRect()
      : { left: anchor.x, right: anchor.x, top: anchor.y, bottom: anchor.y, width: 0 }
    const width = Math.max(m.width, matchWidth ? r.width : 0, minWidth ?? 0)
    let left = align === 'end' ? r.right - width : r.left
    left = Math.max(8, Math.min(left, vw - width - 8))
    const gap = anchor instanceof HTMLElement ? 5 : 2
    const below = r.bottom + gap
    const fitsBelow = below + m.height <= vh - 8
    const top = fitsBelow ? below : Math.max(8, r.top - gap - m.height)
    setPos({
      left,
      top,
      minWidth: matchWidth ? r.width : undefined,
      origin: `${align === 'end' ? 'right' : 'left'} ${fitsBelow ? 'top' : 'bottom'}`,
    })
    // Position once per open; the surface closes on scroll/resize anyway.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!silent) playSound('open')
    const anchorEl = anchor instanceof HTMLElement ? anchor : null
    const onDown = (e: MouseEvent) => {
      const n = e.target as Node
      if (ref.current?.contains(n) || anchorEl?.contains(n)) return
      onCloseRef.current()
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); onCloseRef.current() } }
    const onScroll = (e: Event) => { if (!ref.current?.contains(e.target as Node)) onCloseRef.current() }
    const onResize = () => onCloseRef.current()
    // Deferred so the press that opened the surface doesn't close it again.
    const id = window.setTimeout(() => document.addEventListener('mousedown', onDown, true), 0)
    document.addEventListener('keydown', onKey, true)
    window.addEventListener('scroll', onScroll, true)
    window.addEventListener('resize', onResize)
    return () => {
      window.clearTimeout(id)
      document.removeEventListener('mousedown', onDown, true)
      document.removeEventListener('keydown', onKey, true)
      window.removeEventListener('scroll', onScroll, true)
      window.removeEventListener('resize', onResize)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return createPortal(
    <div
      ref={ref}
      role={role}
      aria-label={ariaLabel}
      className={`ds-pop ${pos ? 'ds-pop-in' : ''} ${className}`}
      style={{
        position: 'fixed',
        zIndex: 90,
        maxHeight,
        minWidth: pos?.minWidth ?? minWidth,
        left: pos?.left ?? 0,
        top: pos?.top ?? 0,
        visibility: pos ? 'visible' : 'hidden',
        transformOrigin: pos?.origin,
        ...style,
      }}
      onMouseDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      onContextMenu={(e) => e.preventDefault()}
    >
      {children}
    </div>,
    document.body,
  )
}

// ── menu rows ──────────────────────────────────────────────────────────────

export interface MenuItem {
  label: ReactNode
  onSelect: () => void
  icon?: ReactNode
  /** Right-aligned hint (shortcut, count). */
  hint?: ReactNode
  tone?: 'err'
  disabled?: boolean
  /** Draws a divider above this row. */
  divider?: boolean
  checked?: boolean
  /** Two-line row (title + subtitle). */
  tall?: boolean
}

export function MenuItems({ items, onClose }: { items: MenuItem[]; onClose: () => void }) {
  return (
    <>
      {items.map((it, i) => (
        <div key={i}>
          {it.divider && i > 0 && <div className="ds-menu-sep" role="separator" />}
          <button
            type="button"
            role="menuitem"
            disabled={it.disabled}
            className={`ds-menu-item${it.tall ? ' ds-tall' : ''}${it.tone === 'err' ? ' ds-is-err' : ''}${it.checked ? ' ds-is-on' : ''}`}
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              onClose()
              playSound('select')
              it.onSelect()
            }}
          >
            {it.icon && <span className="ds-menu-ico">{it.icon}</span>}
            <span className="ds-menu-label">{it.label}</span>
            {it.hint != null && <span className="ds-menu-hint">{it.hint}</span>}
            {it.checked && <span className="ds-menu-check">{CheckIcon}</span>}
          </button>
        </div>
      ))}
    </>
  )
}

export const CheckIcon = (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="m5 12.5 4.5 4.5L19 7" />
  </svg>
)

/** Right-click menu state for a list: `open(e, payload)` in onContextMenu,
 *  render `<Popover anchor={at}>` while `menu` is set. */
export function useContextMenu<T>() {
  const [menu, setMenu] = useState<{ at: { x: number; y: number }; payload: T } | null>(null)
  return {
    menu,
    open: (e: React.MouseEvent, payload: T) => {
      e.preventDefault()
      e.stopPropagation()
      setMenu({ at: { x: e.clientX, y: e.clientY }, payload })
    },
    close: () => setMenu(null),
  }
}
