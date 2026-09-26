/**
 * useLocalStorageState -- generic localStorage-backed persisted state hook.
 *
 * The project used to have 4+ places hand-rolling localStorage reads/writes
 * (preset-helpers / Settings / Regularization / Curation /
 * PromptFromDatasetPicker); the differing signatures made it hard to later
 * add cross-tab sync / SSR guards uniformly. This hook is the single entry
 * point now.
 *
 * Behavior:
 *   - reads the initial value from localStorage on mount, falls back to defaultValue if missing
 *   - setValue(v) writes back to localStorage immediately
 *   - listens for the 'storage' event for cross-tab sync (modeled on useAdvancedMode)
 *   - SSR-safe: silently uses the default when typeof window === 'undefined'
 *   - serializes with JSON.stringify / JSON.parse; falls back to default on parse failure
 *
 * Naming convention: keys use a `studio:scope:field` prefix (modeled on
 * useAdvancedMode's `studio:advanced_mode`), to avoid clashing with other
 * web apps / old versions.
 */
import { useCallback, useEffect, useState } from 'react'

export function useLocalStorageState<T>(
  key: string,
  defaultValue: T,
): [T, (v: T | ((prev: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => readPersisted(key, defaultValue))

  useEffect(() => {
    if (typeof window === 'undefined') return
    const handler = (e: StorageEvent) => {
      if (e.key !== key) return
      if (e.newValue === null) {
        setValue(defaultValue)
        return
      }
      try {
        setValue(JSON.parse(e.newValue) as T)
      } catch {
        // Another tab wrote a non-JSON value (external script?) -> ignore, this tab keeps its current value
      }
    }
    window.addEventListener('storage', handler)
    return () => window.removeEventListener('storage', handler)
  }, [key, defaultValue])

  const update = useCallback(
    (next: T | ((prev: T) => T)) => {
      setValue((prev) => {
        const resolved =
          typeof next === 'function' ? (next as (p: T) => T)(prev) : next
        if (typeof window !== 'undefined') {
          try {
            window.localStorage.setItem(key, JSON.stringify(resolved))
          } catch {
            // quota exceeded / private mode -> silent; state is still valid in memory
          }
        }
        return resolved
      })
    },
    [key],
  )

  return [value, update]
}

function readPersisted<T>(key: string, defaultValue: T): T {
  if (typeof window === 'undefined') return defaultValue
  try {
    const raw = window.localStorage.getItem(key)
    if (raw === null) return defaultValue
    return JSON.parse(raw) as T
  } catch {
    return defaultValue
  }
}
