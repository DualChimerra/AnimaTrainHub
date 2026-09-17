import { describe, it, expect, vi, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useIsMobile, useMediaQuery, MOBILE_BREAKPOINT } from './useMediaQuery'

type Listener = (e: MediaQueryListEvent) => void

/** Installs a controllable `matchMedia`, returns a setter that flips the match
 *  state and notifies listeners the way a real viewport resize would. */
function installMatchMedia(initial: boolean) {
  const listeners = new Set<Listener>()
  let matches = initial
  const mql = {
    get matches() { return matches },
    media: '',
    addEventListener: (_: string, cb: Listener) => { listeners.add(cb) },
    removeEventListener: (_: string, cb: Listener) => { listeners.delete(cb) },
    addListener: (cb: Listener) => { listeners.add(cb) },
    removeListener: (cb: Listener) => { listeners.delete(cb) },
    dispatchEvent: () => true,
    onchange: null,
  }
  Object.defineProperty(window, 'matchMedia', {
    writable: true, configurable: true,
    value: vi.fn().mockImplementation((q: string) => ({ ...mql, media: q })),
  })
  return (next: boolean) => {
    matches = next
    listeners.forEach((cb) => cb({ matches: next } as MediaQueryListEvent))
  }
}

afterEach(() => {
  // @ts-expect-error — deliberately removing the stub between tests
  delete window.matchMedia
})

describe('useMediaQuery', () => {
  it('reports the initial match', () => {
    installMatchMedia(true)
    const { result } = renderHook(() => useMediaQuery('(max-width: 768px)'))
    expect(result.current).toBe(true)
  })

  it('updates when the viewport crosses the breakpoint', () => {
    const setMatches = installMatchMedia(false)
    const { result } = renderHook(() => useIsMobile())
    expect(result.current).toBe(false)
    act(() => setMatches(true))
    expect(result.current).toBe(true)
  })

  it('falls back to desktop when matchMedia is unavailable', () => {
    // JSDOM without the stub — the hook must not throw, and every existing
    // component test keeps rendering the desktop layout it was written for.
    expect(typeof window.matchMedia).toBe('undefined')
    const { result } = renderHook(() => useIsMobile())
    expect(result.current).toBe(false)
  })

  it('queries the documented breakpoint', () => {
    installMatchMedia(false)
    renderHook(() => useIsMobile())
    expect(window.matchMedia).toHaveBeenCalledWith(`(max-width: ${MOBILE_BREAKPOINT}px)`)
  })
})
