import { useEffect, useState } from 'react'

/** Breakpoint below which the shell switches to the phone layout.
 *
 *  768px rather than Tailwind's 640px: the sidebar plus a content column only
 *  stops being cramped past roughly 800px, and small tablets in portrait are
 *  much happier with the drawer too. */
export const MOBILE_BREAKPOINT = 768

/** Subscribe to a CSS media query.
 *
 *  SSR/JSDOM safe: `matchMedia` is missing in some test environments, so the
 *  hook degrades to `false` (desktop layout) instead of throwing — that keeps
 *  every existing component test rendering the layout it was written against. */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false
    return window.matchMedia(query).matches
  })

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    const mql = window.matchMedia(query)
    const onChange = (e: MediaQueryListEvent) => setMatches(e.matches)
    setMatches(mql.matches)
    // Safari < 14 only has the deprecated addListener/removeListener pair.
    if (typeof mql.addEventListener === 'function') {
      mql.addEventListener('change', onChange)
      return () => mql.removeEventListener('change', onChange)
    }
    mql.addListener(onChange)
    return () => mql.removeListener(onChange)
  }, [query])

  return matches
}

/** True on phone-sized viewports. */
export function useIsMobile(): boolean {
  return useMediaQuery(`(max-width: ${MOBILE_BREAKPOINT}px)`)
}
