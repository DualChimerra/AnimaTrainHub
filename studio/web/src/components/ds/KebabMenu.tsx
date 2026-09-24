import { useEffect, useRef, useState } from 'react'

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

/** The mockup's ⋯ button (.kebab) with a small action menu. Closes on an
 *  outside click, Escape, or after picking an item. Clicks never bubble to a
 *  surrounding clickable card or row. */
export default function KebabMenu({ label, items, className = '', trigger = 'kebab' }: {
  /** aria-label for the trigger, e.g. "Действия проекта". */
  label: string
  items: KebabItem[]
  className?: string
  /** 'icon': the bordered 32px .iconbtn of a page head instead of the bare ⋯. */
  trigger?: 'kebab' | 'icon'
}) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLSpanElement | null>(null)

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <span ref={rootRef} className={`relative inline-flex ${className}`} onClick={(e) => e.stopPropagation()}>
      <button
        type="button"
        className={trigger === 'icon' ? 'ds-iconbtn hover:text-fg-primary transition-colors' : 'ds-kebab hover:bg-sunken hover:text-fg-primary transition-colors'}
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={(e) => { e.preventDefault(); e.stopPropagation(); setOpen((v) => !v) }}
      >
        {DotsIcon}
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full mt-1 z-50 py-1 rounded-[10px] border border-dim bg-elevated shadow-lg"
          // inline: the mockup's `.ds-card div { min-width: 0 }` outranks a utility class
          style={{ minWidth: 180 }}
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
        </div>
      )}
    </span>
  )
}
