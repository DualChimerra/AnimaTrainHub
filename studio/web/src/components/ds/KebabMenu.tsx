import { useRef, useState, type ReactNode } from 'react'
import Popover, { MenuItems } from './Popover'

export interface KebabItem {
  label: string
  onSelect: () => void
  tone?: 'err'
  disabled?: boolean
  icon?: ReactNode
  /** Draws a divider above this row. */
  divider?: boolean
}

const DotsIcon = (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
    <circle cx="12" cy="5" r="1.6" /><circle cx="12" cy="12" r="1.6" /><circle cx="12" cy="19" r="1.6" />
  </svg>
)

const TRIGGER_CLASS = {
  kebab: 'ds-kebab',
  icon: 'ds-iconbtn',
  group: 'ds-ico',
} as const

/** The mockup's ⋯ button (.kebab) with a small action menu (shared Popover:
 *  closes on an outside click, Escape, scroll, or after picking an item).
 *  Clicks never bubble to a surrounding clickable card or row. */
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
  const btnRef = useRef<HTMLButtonElement | null>(null)

  return (
    <span className={`relative inline-flex ${className}`} onClick={(e) => e.stopPropagation()}>
      <button
        ref={btnRef}
        type="button"
        className={`${TRIGGER_CLASS[trigger]}${open ? ' ds-is-open' : ''}`}
        aria-label={label}
        title={label}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={(e) => { e.preventDefault(); e.stopPropagation(); setOpen((v) => !v) }}
      >
        {DotsIcon}
      </button>
      {open && btnRef.current && (
        <Popover anchor={btnRef.current} align="end" minWidth={190} onClose={() => setOpen(false)} ariaLabel={label}>
          <MenuItems items={items} onClose={() => setOpen(false)} />
        </Popover>
      )}
    </span>
  )
}
