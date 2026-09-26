/**
 * Frontend error reporting (ADR-0009 §5 / PR-3 C2).
 *
 * Three callers:
 *   - window.addEventListener('error')               sync script / resource load errors
 *   - window.addEventListener('unhandledrejection')   an uncaught Promise.reject
 *   - ErrorBoundary.componentDidCatch                 React render / lifecycle throws
 *
 * All of them go to `POST /api/client-errors` -> backend logger.error -> studio.log.
 *
 * Hard constraints:
 *   - **silent swallow on fail** -- a failed report must never throw again
 *     (to avoid cascading into an ErrorBoundary infinite loop)
 *   - uses keepalive: tries to send even as the tab is closing
 *   - never blocks the main flow (async fire-and-forget)
 */

export type ClientErrorKind =
  | 'react.boundary'
  | 'window.error'
  | 'unhandledrejection'
  | 'manual'

export interface ClientErrorReport {
  kind: ClientErrorKind
  message: string
  stack?: string
  componentStack?: string  // react.boundary only
  source?: string          // window.error script URL
  line?: number
  col?: number
  /** Callers do **not** need to fill this in; report() injects location.href / userAgent etc. automatically. */
}

interface InternalReportBody extends ClientErrorReport {
  url: string
  user_agent: string
  client_ts: string
  app_version: string
  build_hash?: string
  trace_id_last_4xx?: string
}

// Set by client.ts on a 4xx/5xx; read back when report() sends. Lets
// developers join "the error shown in the user's toast" with "the last API
// failure before the frontend crashed" in the server log.
let _lastApiTraceId: string | undefined

export function setLastApiTraceId(traceId: string | undefined): void {
  _lastApiTraceId = traceId
}

export function getLastApiTraceId(): string | undefined {
  return _lastApiTraceId
}

/**
 * Sends the report. Fire-and-forget -- doesn't return a Promise (so a caller can't accidentally block on await).
 *
 * Failures are swallowed silently (logged to console.warn so it isn't completely invisible).
 */
export function reportClientError(input: ClientErrorReport): void {
  // Guard against errors that occur before listeners are registered: check globalThis
  if (typeof globalThis === 'undefined' || typeof fetch === 'undefined') return

  let buildHash: string | undefined
  let appVersion = '0.0.0'
  try {
    // Injected by Vite at build time; falls back silently if missing
    const env = (import.meta as ImportMeta & { env?: Record<string, string> }).env
    buildHash = env?.VITE_BUILD_HASH
    appVersion = env?.VITE_APP_VERSION || appVersion
  } catch {
    // import.meta unavailable (very rare); fall back
  }

  const body: InternalReportBody = {
    ...input,
    url: globalThis.location?.href || '',
    user_agent: globalThis.navigator?.userAgent || '',
    client_ts: new Date().toISOString(),
    app_version: appVersion,
    build_hash: buildHash,
    trace_id_last_4xx: _lastApiTraceId,
  }

  // fire-and-forget; keepalive tries to send even as the tab closes
  try {
    void fetch('/api/client-errors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      keepalive: true,
    }).catch(() => {
      // silently swallow -- a failed report must never throw into ErrorBoundary
    })
  } catch {
    // swallow even if fetch itself throws (extremely rare)
    try {
      console.warn('[reportClientError] failed silently')
    } catch {
      // even console.warn is unusable at this point; give up
    }
  }
}
