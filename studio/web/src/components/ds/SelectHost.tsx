import { useCallback, useEffect, useRef, useState } from 'react'
import { playSound } from '../../lib/sound'
import Popover, { CheckIcon } from './Popover'

/**
 * Styled dropdown lists for every native <select> in the app.
 *
 * The OS popup of a <select> ignores the design system (square, system font,
 * blue highlight). Rather than swap 30+ call sites for a custom widget, this
 * host intercepts the moment a select would open — a press on it, or
 * Space / Enter / Alt+↓ / F4 while it has focus — and shows the options in the
 * app's own Popover instead. Picking one writes the native value and fires a
 * real `change` event, so React's onChange, forms, tests (which drive the
 * native element) and keyboard stepping with ↑/↓ all keep working unchanged.
 *
 * Opt out per element with `data-native-select`.
 */

interface Opt {
  value: string
  label: string
  disabled: boolean
  title?: string
  group?: string
}

function readOptions(sel: HTMLSelectElement): Opt[] {
  const out: Opt[] = []
  for (const child of Array.from(sel.children)) {
    if (child instanceof HTMLOptGroupElement) {
      for (const o of Array.from(child.children)) {
        if (o instanceof HTMLOptionElement && !o.hidden) {
          out.push({ value: o.value, label: o.text, disabled: o.disabled || child.disabled, title: o.title || undefined, group: child.label })
        }
      }
    } else if (child instanceof HTMLOptionElement && !child.hidden) {
      out.push({ value: child.value, label: child.text, disabled: child.disabled, title: child.title || undefined })
    }
  }
  return out
}

const valueSetter = typeof window !== 'undefined'
  ? Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set
  : undefined

function commit(sel: HTMLSelectElement, value: string) {
  if (sel.value === value) return
  if (valueSetter) valueSetter.call(sel, value)
  else sel.value = value
  sel.dispatchEvent(new Event('input', { bubbles: true }))
  sel.dispatchEvent(new Event('change', { bubbles: true }))
}

function eligible(el: EventTarget | null): el is HTMLSelectElement {
  return el instanceof HTMLSelectElement
    && !el.multiple
    && el.size <= 1
    && !el.disabled
    && !el.hasAttribute('data-native-select')
}

export default function SelectHost() {
  const [sel, setSel] = useState<HTMLSelectElement | null>(null)
  const [opts, setOpts] = useState<Opt[]>([])
  const [active, setActive] = useState(0)
  const [openId, setOpenId] = useState(0)
  const listRef = useRef<HTMLDivElement | null>(null)
  const typed = useRef({ q: '', at: 0 })

  const open = useCallback((el: HTMLSelectElement) => {
    const o = readOptions(el)
    setOpts(o)
    setActive(Math.max(0, o.findIndex((x) => x.value === el.value)))
    setSel(el)
    setOpenId((n) => n + 1)
    el.classList.add('ds-select-open')
  }, [])

  const close = useCallback(() => {
    setSel((cur) => {
      cur?.classList.remove('ds-select-open')
      return null
    })
  }, [])

  // Intercept the native popup.
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (e.button !== 0 || !eligible(e.target)) return
      e.preventDefault()
      const el = e.target
      el.focus()
      if (sel === el) close()
      else open(el)
    }
    const onKey = (e: KeyboardEvent) => {
      if (sel || !eligible(e.target)) return
      const opens = e.key === ' ' || e.key === 'Enter' || e.key === 'F4' || (e.altKey && (e.key === 'ArrowDown' || e.key === 'ArrowUp'))
      if (!opens) return
      e.preventDefault()
      open(e.target)
    }
    document.addEventListener('mousedown', onDown, true)
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('mousedown', onDown, true)
      document.removeEventListener('keydown', onKey, true)
    }
  }, [sel, open, close])

  const pick = useCallback((i: number) => {
    const o = opts[i]
    if (!sel || !o || o.disabled) return
    commit(sel, o.value)
    playSound('select')
    close()
    sel.focus()
  }, [opts, sel, close])

  // Keyboard inside the open list (focus stays on the <select>).
  useEffect(() => {
    if (!sel) return
    const step = (from: number, d: number) => {
      let i = from
      for (let n = 0; n < opts.length; n++) {
        i = (i + d + opts.length) % opts.length
        if (!opts[i].disabled) return i
      }
      return from
    }
    const onKey = (e: KeyboardEvent) => {
      const k = e.key
      if (k === 'ArrowDown' || k === 'ArrowUp') { e.preventDefault(); setActive((a) => step(a, k === 'ArrowDown' ? 1 : -1)) }
      else if (k === 'Home') { e.preventDefault(); setActive(step(-1, 1)) }
      else if (k === 'End') { e.preventDefault(); setActive(step(opts.length, -1)) }
      else if (k === 'Enter' || k === ' ') { e.preventDefault(); pick(active) }
      else if (k === 'Tab') close()
      else if (k.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
        // Type-ahead: jump to the first option starting with what was typed.
        const now = performance.now()
        typed.current.q = (now - typed.current.at > 700 ? '' : typed.current.q) + k.toLowerCase()
        typed.current.at = now
        const i = opts.findIndex((o) => !o.disabled && o.label.toLowerCase().startsWith(typed.current.q))
        if (i >= 0) setActive(i)
      }
    }
    document.addEventListener('keydown', onKey, true)
    return () => document.removeEventListener('keydown', onKey, true)
  }, [sel, opts, active, pick, close])

  // Keep the highlighted row in view.
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-i="${active}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [active, sel])

  if (!sel || !sel.isConnected) return null
  const current = sel.value
  let lastGroup: string | undefined

  return (
    <Popover key={openId} anchor={sel} matchWidth onClose={close} role="listbox" ariaLabel={sel.getAttribute('aria-label') ?? undefined} maxHeight={320} silent>
      <div ref={listRef}>
        {opts.map((o, i) => {
          const header = o.group && o.group !== lastGroup ? o.group : null
          lastGroup = o.group
          return (
            <div key={`${o.group ?? ''}:${o.value}:${i}`}>
              {header && <div className="ds-menu-group">{header}</div>}
              <div
                role="option"
                data-i={i}
                aria-selected={o.value === current}
                aria-disabled={o.disabled || undefined}
                title={o.title}
                className={`ds-menu-item${i === active ? ' ds-is-hover' : ''}${o.value === current ? ' ds-is-on' : ''}${o.disabled ? ' ds-is-disabled' : ''}`}
                onMouseEnter={() => { if (!o.disabled) setActive(i) }}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(i)}
              >
                <span className="ds-menu-label">{o.label || '—'}</span>
                {o.value === current && <span className="ds-menu-check">{CheckIcon}</span>}
              </div>
            </div>
          )
        })}
      </div>
    </Popover>
  )
}
