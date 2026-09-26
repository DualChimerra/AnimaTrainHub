/**
 * Installs global JS error listeners (ADR-0009 §5.1 / PR-3 C2).
 *
 * Call installGlobalErrorHandlers() once at startup in main.tsx. It installs
 * two listeners:
 *   - window.addEventListener('error', ...)              sync scripts / resource loads
 *   - window.addEventListener('unhandledrejection', ...)  uncaught Promises
 *
 * It does **not** proxy console.error wholesale (too noisy in dev with React
 * warnings / devtools messages). It only catches errors outside React;
 * React's own errors go through ErrorBoundary separately (PR-3 C3).
 *
 * Safe to call more than once -- repeat calls are a no-op (_installed sentinel).
 */
import { reportClientError } from './report'

let _installed = false

export function installGlobalErrorHandlers(): void {
  if (_installed) return
  _installed = true

  if (typeof globalThis === 'undefined' || typeof globalThis.addEventListener !== 'function') {
    return
  }

  globalThis.addEventListener('error', (ev: ErrorEvent) => {
    // ev.error can be null (cross-origin script error / resource 404)
    const err = ev.error as Error | null
    reportClientError({
      kind: 'window.error',
      message: err?.message || ev.message || '(unknown error)',
      stack: err?.stack,
      source: ev.filename,
      line: ev.lineno,
      col: ev.colno,
    })
  })

  globalThis.addEventListener('unhandledrejection', (ev: PromiseRejectionEvent) => {
    const reason = ev.reason as unknown
    let message: string
    let stack: string | undefined
    if (reason instanceof Error) {
      message = reason.message
      stack = reason.stack
    } else if (typeof reason === 'string') {
      message = reason
    } else {
      try {
        message = JSON.stringify(reason)
      } catch {
        message = String(reason)
      }
    }
    reportClientError({
      kind: 'unhandledrejection',
      message,
      stack,
    })
  })
}

/** Test hook -- clears the sentinel so unit tests can reinstall repeatedly. */
export function _resetInstalledForTests(): void {
  _installed = false
}
