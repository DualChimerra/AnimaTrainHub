import { useEffect, useRef } from 'react'

export interface StudioEvent {
  type: string
  task_id?: number
  status?: string
  [key: string]: unknown
}

interface Options {
  /** Called whenever a connection succeeds (including every reconnect).
   * EventSource reconnects automatically, and events are lost while
   * disconnected; this gives consumers a hook to cold-fetch and catch up.
   * Also fires on the very first onopen; consumers wanting to distinguish
   * "first connect vs. reconnect" should count with their own ref. */
  onOpen?: () => void
}

type Listener = (evt: StudioEvent) => void
type OpenListener = () => void

// ── shared EventSource ──────────────────────────────────────────────────────
// The whole app opens a single /api/events long-lived connection, shared by
// every useEventStream caller. The early implementation had each caller open
// its own, and ~16 hook call sites plus StrictMode's double-mount blew
// straight through the browser's HTTP/1.1 "6 connections per origin" limit,
// leaving ordinary fetch calls permanently unable to get a socket (outputs
// would hang forever, and even reloading the page couldn't load anything).
const _listeners = new Set<Listener>()
const _openListeners = new Set<OpenListener>()
let _es: EventSource | null = null

function _ensureOpen(): void {
  if (_es || typeof EventSource === 'undefined') return
  const es = new EventSource('/api/events')
  es.onopen = () => {
    for (const cb of _openListeners) {
      try { cb() } catch { /* one subscriber's callback throwing shouldn't affect the others */ }
    }
  }
  es.onmessage = (e) => {
    let evt: StudioEvent
    try { evt = JSON.parse(e.data) as StudioEvent } catch { return }
    for (const cb of _listeners) {
      try { cb(evt) } catch { /* same as above */ }
    }
  }
  es.onerror = () => {
    // EventSource reconnects automatically; this is just a hook, it doesn't close proactively
  }
  _es = es
}

function _maybeClose(): void {
  if (_es && _listeners.size === 0 && _openListeners.size === 0) {
    _es.close()
    _es = null
  }
}

/**
 * Subscribes to the /api/events SSE stream. The callback fires once per event.
 * Reconnects automatically on disconnect (this is just EventSource's built-in behavior).
 *
 * Multiple components share the same underlying EventSource, so they don't eat into the browser's per-origin connection quota.
 */
export function useEventStream(
  onEvent: (evt: StudioEvent) => void,
  options?: Options,
): void {
  // Store the closure in a ref so useEffect only binds once on mount, while the handler always reads the latest value
  const onEventRef = useRef(onEvent)
  onEventRef.current = onEvent
  const onOpenRef = useRef(options?.onOpen)
  onOpenRef.current = options?.onOpen

  useEffect(() => {
    // jsdom / SSR / old browsers have no EventSource -- skip connecting SSE so components can still mount in test environments
    if (typeof EventSource === 'undefined') return
    const handler: Listener = (evt) => onEventRef.current(evt)
    const openHandler: OpenListener = () => onOpenRef.current?.()
    _listeners.add(handler)
    _openListeners.add(openHandler)
    _ensureOpen()
    // When the shared connection is already open, a new subscriber still needs an onOpen call so it can cold-fetch
    if (_es && _es.readyState === EventSource.OPEN) {
      try { onOpenRef.current?.() } catch { /* ignore */ }
    }
    return () => {
      _listeners.delete(handler)
      _openListeners.delete(openHandler)
      _maybeClose()
    }
  }, [])
}
