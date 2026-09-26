import { useEffect, useState, type RefObject } from 'react'
import { useTranslation } from 'react-i18next'
import { schemaGroupLabel } from '../lib/schema'

/** Right-hand anchor nav for SchemaForm's sections.
 *
 * The active section is determined by a scroll listener: it finds the last
 * (in DOM order) section that is above the point 50px below the scroll
 * container's viewport top. This is steadier than an IntersectionObserver +
 * rootMargin "active band", which flickers between sections when more than
 * one sits inside the band at once. */
export default function SchemaSectionIndex({
  groups,
  scrollContainer,
}: {
  groups: Array<{ key: string; label: string }>
  scrollContainer: RefObject<HTMLElement | null>
}) {
  const { t } = useTranslation()
  const [active, setActive] = useState<string>(groups[0]?.key ?? '')

  useEffect(() => {
    setActive(groups[0]?.key ?? '')
  }, [groups])

  useEffect(() => {
    const root = scrollContainer.current
    if (!root || groups.length === 0) return

    const compute = () => {
      const rootTop = root.getBoundingClientRect().top
      // Use 50px below the container top as the decision line -- the last
      // section above that line is active. 50 is an empirical value: too
      // small (e.g. 0) switches too late, only going active once the section
      // header has already scrolled into view; too large activates a section
      // while it's still far off.
      const probe = rootTop + 50
      let next = groups[0].key
      for (const g of groups) {
        const el = document.getElementById(`schema-group-${g.key}`)
        if (!el) continue
        if (el.getBoundingClientRect().top <= probe) {
          next = g.key
        } else {
          break
        }
      }
      setActive((prev) => (prev === next ? prev : next))
    }

    compute()
    root.addEventListener('scroll', compute, { passive: true })
    // Collapsing / expanding a section changes its height without firing a
    // scroll event, so ResizeObserver is the fallback that catches it.
    let ro: ResizeObserver | null = null
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(compute)
      ro.observe(root)
    }
    return () => {
      root.removeEventListener('scroll', compute)
      ro?.disconnect()
    }
  }, [groups, scrollContainer])

  const onJump = (key: string) => {
    const el = document.getElementById(`schema-group-${key}`)
    if (!el) return
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setActive(key)
  }

  if (groups.length === 0) return null

  return (
    <nav className="flex flex-col gap-0.5">
      <div className="caption mb-2 px-2">{t('settings.pageIndex')}</div>
      {groups.map((g) => (
        <button
          key={g.key}
          onClick={() => onJump(g.key)}
          className={`text-left text-xs px-2 h-7 rounded-md transition-colors border-none ${
            active === g.key
              ? 'text-fg-primary font-medium bg-overlay'
              : 'bg-transparent text-fg-tertiary hover:text-fg-primary hover:bg-overlay'
          }`}
        >
          {schemaGroupLabel(g.key, g.label, t)}
        </button>
      ))}
    </nav>
  )
}
