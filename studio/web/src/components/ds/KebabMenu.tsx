import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

export interface KebabItem {
  label: string
  onSelect: () => void
  tone?: 'err'
  disabled?: boolean
}

const DotsIcon = (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
    <circle cx="12" cy="5" r="1.6" /><circle cx="12" cy="12" r="1.6" /><circle cx="12" cy="19" r="1.6" />
  </svg>
)

const TRIGGER_CLASS = {
  kebab: 'ds-kebab hover:bg-sunken hover:text-fg-primary transition-colors',
  icon: 'ds-iconbtn hover:text-fg-primary transition-colors',
  group: 'ds-ico hover:text-fg-primary transition-colors',
} as const

/** The mockup's ⋯ button (.kebab) with a small action menu. Closes on an
 *  outside click, Escape, scroll, or after picking an item. Clicks never bubble
 *  to a surrounding clickable card or row. The menu is portalled with fixed
 *  positioning so cards with overflow:hidden never clip it. */
export default function KebabMenu({ label, items, className = '', trigger = 'kebab' }: {
  /** aria-label for the trigger, e.g. "Действия проекта". */
  label: string
  items: KebabItem[]
  className?: string
  /** 'icon': the bordered 32px .iconbtn of a page head; 'group': the last
   *  segment of an .actgroup; default: the bare ⋯. */
  trigger?: keyof typeof TRIGGER_CLASS
}) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLSpanElement | null>(null)
  const menuRef = useRef<HTMLDivElement | null>(null)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      const n = e.target as Node
      if (!rootRef.current?.contains(n) && !menuRef.current?.contains(n)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    const onScroll = (e: Event) => { if (!menuRef.current?.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    window.addEventListener('scroll', onScroll, true)
    window.addEventListener('resize', onScroll)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
      window.removeEventListener('scroll', onScroll, true)
      window.removeEventListener('resize', onScroll)
    }
  }, [open])

  // Right-aligned under the trigger; above it when there is no room below.
  useLayoutEffect(() => {
    if (!open || !rootRef.current || !menuRef.current) return
    const a = rootRef.current.getBoundingClientRect()
    const m = menuRef.current.getBoundingClientRect()
    const left = Math.max(8, Math.min(a.right - m.width, window.innerWidth - m.width - 8))
    const top = a.bottom + 4 + m.height <= window.innerHeight - 8 ? a.bottom + 4 : Math.max(8, a.top - 4 - m.height)
    setPos({ left, top })
  }, [open])

  return (
    <span ref={rootRef} className={`relative inline-flex ${className}`} onClick={(e) => e.stopPropagation()}>
      <button
        type="button"
        className={TRIGGER_CLASS[trigger]}
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={(e) => { e.preventDefault(); e.stopPropagation(); setPos(null); setOpen((v) => !v) }}
      >
        {DotsIcon}
      </button>
      {open && createPortal(
        <div
          ref={menuRef}
          role="menu"
          className="fixed z-[80] py-1 rounded-[10px] border border-dim bg-elevated shadow-lg"
          style={{ minWidth: 180, ...(pos ?? { left: 0, top: 0, visibility: 'hidden' as const }) }}
          onClick={(e) => e.stopPropagation()}
        >
          {items.map((it) => (
            <button
              key={it.label}
              type="button"
              role="menuitem"
              disabled={it.disabled}
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); setOpen(false); it.onSelect() }}
              className={`block w-full text-left whitespace-nowrap px-3 py-1.5 text-[12.5px] hover:bg-sunken disabled:opacity-40 disabled:cursor-default ${it.tone === 'err' ? 'text-err' : 'text-fg-primary'}`}
            >
              {it.label}
            </button>
          ))}
        </div>,
        document.body,
      )}
    </span>
  )
}
