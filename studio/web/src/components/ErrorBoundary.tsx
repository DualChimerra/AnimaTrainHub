import { Component, type ErrorInfo, type ReactNode } from 'react'
import i18n from '../i18n'
import { getLastApiTraceId, reportClientError } from '../lib/errors/report'

interface State { error: Error | null }

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', error, info)
    // ADR-0009 PR-3 C3: report to /api/client-errors -> studio.log studio.client logger.
    // Silent swallow on fail (avoids blowing up a second time while ErrorBoundary is already in catch state).
    try {
      reportClientError({
        kind: 'react.boundary',
        message: error.message,
        stack: error.stack,
        componentStack: info.componentStack ?? undefined,
      })
    } catch {
      // Swallow failures from the reporting call itself too
    }
  }

  render() {
    if (this.state.error) {
      // ADR-0009 PR-3 C3: shows the last 8 characters of the trace_id from the
      // most recent API 4xx/5xx (ErrorBoundary can't get traceId itself -- falls
      // back to the frontend's lastApiTraceId). The user can screenshot it for
      // developers: grepping this string locates the whole trace chain.
      const traceId = getLastApiTraceId()
      const traceSuffix = traceId ? traceId.slice(-8) : null
      return (
        <div className="min-h-screen flex items-center justify-center p-8 bg-canvas">
          <div className="card max-w-[560px] w-full p-6">
            <h1 className="text-err font-semibold text-lg mb-2">{i18n.t('errorBoundary.title')}</h1>
            <pre className="text-sm text-fg-secondary whitespace-pre-wrap break-all">
              {this.state.error.message}
            </pre>
            {traceSuffix && (
              <div className="text-xs text-fg-tertiary mt-2 font-mono">
                trace {traceSuffix}
              </div>
            )}
            <button
              className="btn btn-primary btn-sm mt-4"
              onClick={() => window.location.reload()}
            >
              {i18n.t('errorBoundary.reload')}
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
