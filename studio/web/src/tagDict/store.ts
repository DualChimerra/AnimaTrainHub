/** Autocomplete tag list — module-level singleton + useSyncExternalStore hook.
 *
 * The list (~100k tags) is fetched once on first use and kept in memory; the
 * browser HTTP cache handles repeat loads.
 *
 * Public API:
 *   - useTagDict()  subscribe from a component; the first mount triggers loadDict()
 *   - reloadDict()  force a refresh after an upload / reset
 */
import { useEffect, useSyncExternalStore } from 'react'

import type { TagDictMeta, TagDictPayload, TagDictStatus } from './types'

interface State {
  status: TagDictStatus
  /** Tags in file order (popularity order for the default source). */
  tagKeys: string[]
  /** tagKeys without spaces / underscores, same indices. Computed once on
   *  load so a keystroke never regex-replaces the whole list. */
  compactedKeys: string[]
  meta: TagDictMeta | null
  error: string | null
}

let state: State = {
  status: 'idle',
  tagKeys: [],
  compactedKeys: [],
  meta: null,
  error: null,
}

const listeners = new Set<() => void>()
let inFlight: Promise<void> | null = null

function setState(next: Partial<State>): void {
  state = { ...state, ...next }
  listeners.forEach((l) => l())
}

function subscribe(l: () => void): () => void {
  listeners.add(l)
  return () => { listeners.delete(l) }
}

async function fetchDict(): Promise<void> {
  setState({ status: 'loading', error: null })
  try {
    const resp = await fetch('/api/tag-dictionary/data')
    if (resp.status === 404) {
      setState({ status: 'empty', tagKeys: [], compactedKeys: [], meta: null })
      return
    }
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const payload = (await resp.json()) as TagDictPayload
    const tagKeys = Array.isArray(payload.tags) ? payload.tags : []
    setState({
      status: 'ready',
      tagKeys,
      compactedKeys: tagKeys.map((t) => t.replace(/[\s_]/g, '')),
      meta: payload.meta || null,
      error: null,
    })
  } catch (err) {
    setState({
      status: 'error',
      error: err instanceof Error ? err.message : String(err),
    })
  }
}

/** First load / forced refresh; idempotent while a load is in flight. */
export function loadDict(force = false): Promise<void> {
  if (!force && (state.status === 'ready' || state.status === 'loading')) {
    return inFlight ?? Promise.resolve()
  }
  inFlight = fetchDict().finally(() => { inFlight = null })
  return inFlight
}

/** After an upload / reset: force a refresh. */
export function reloadDict(): Promise<void> {
  return loadDict(true)
}

/** Subscribe from React. The first mount triggers the load (idempotent). */
export function useTagDict(): State {
  const snapshot = useSyncExternalStore(subscribe, () => state, () => state)
  useEffect(() => {
    if (state.status === 'idle') void loadDict()
  }, [])
  return snapshot
}

/** Tests only: inject state directly, bypassing the network. */
export function __setStateForTest(next: Partial<State>): void {
  setState(next)
}

/** Internal read for suggest.ts; not for application code. */
export function _getInternalState(): State {
  return state
}
