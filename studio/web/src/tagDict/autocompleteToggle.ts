/** Global toggle: whether tag inputs show the autocomplete popup. On by default.
 *
 * Implemented via localStorage + module-level subscribers, same style as showToggle.ts (raw KV + try-catch).
 * Every autocomplete entry point (useTagSuggest) subscribes to this toggle, so turning it off in Settings applies site-wide.
 */
import { useSyncExternalStore } from 'react'

const STORAGE_KEY = 'studio.tag.autocomplete'

const listeners = new Set<() => void>()

function compute(): boolean {
  // Never set = on (defaults to enabled; only an explicit '0' turns it off)
  try { return localStorage.getItem(STORAGE_KEY) !== '0' } catch { return true }
}

function subscribe(l: () => void): () => void {
  listeners.add(l)
  return () => { listeners.delete(l) }
}

/** For a React component to subscribe: returns [enabled, setEnabled]. */
export function useTagAutocompleteEnabled(): [boolean, (next: boolean) => void] {
  const value = useSyncExternalStore(subscribe, compute, compute)
  const setter = (next: boolean) => {
    try { localStorage.setItem(STORAGE_KEY, next ? '1' : '0') } catch { /* ignore */ }
    listeners.forEach((l) => l())
  }
  return [value, setter]
}
