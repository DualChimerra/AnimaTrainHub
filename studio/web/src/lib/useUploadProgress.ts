/**
 * useUploadProgress -- browser upload progress state machine.
 *
 * Protocol:
 *   1. the caller calls `start(totalBytes)` before the request, so the
 *      progress bar shows 0% immediately
 *   2. the caller passes `onProgress` through to `api.xxx(..., onProgress)`
 *      -- the XHR.upload progress event callback (fetch has no request-body
 *      progress)
 *   3. automatically switches to 'processing' when loaded === total (the
 *      server is still unzipping / writing to disk with no events to report,
 *      so the UI shows a spinner to distinguish "upload done, waiting on
 *      the server")
 *   4. once the request resolves / rejects, the caller calls `finish()` /
 *      `fail(e)` / `reset()`
 *
 * Speed is computed over a 1s sliding window to keep the number from
 * jumping around too much; ETA = (total - loaded) / speed.
 */
import { useCallback, useRef, useState } from 'react'

export type UploadPhase = 'idle' | 'uploading' | 'processing' | 'done' | 'error'

export interface UploadProgressState {
  phase: UploadPhase
  loaded: number
  total: number
  /** bytes/sec, 1s moving average; 0 while processing/idle/done */
  speedBps: number
  /** seconds; null = can't be computed (speed=0 or total=0) */
  etaSec: number | null
  error: string | null
}

const INITIAL: UploadProgressState = {
  phase: 'idle',
  loaded: 0,
  total: 0,
  speedBps: 0,
  etaSec: null,
  error: null,
}

export interface UseUploadProgress {
  state: UploadProgressState
  /** Call before the request, so the UI shows 0% / total immediately */
  start: (totalBytes: number) => void
  /** Pass through to api.xxx as the onProgress callback */
  onProgress: (e: { loaded: number; total: number; lengthComputable: boolean }) => void
  finish: () => void
  fail: (e: unknown) => void
  /** Resets back to idle, for closing a dialog / hiding a panel */
  reset: () => void
}

interface Sample {
  t: number
  loaded: number
}

const SPEED_WINDOW_MS = 1500

export function useUploadProgress(): UseUploadProgress {
  const [state, setState] = useState<UploadProgressState>(INITIAL)
  const samplesRef = useRef<Sample[]>([])

  const start = useCallback((totalBytes: number) => {
    samplesRef.current = [{ t: performance.now(), loaded: 0 }]
    setState({
      phase: 'uploading',
      loaded: 0,
      total: totalBytes,
      speedBps: 0,
      etaSec: null,
      error: null,
    })
  }, [])

  const onProgress = useCallback((e: { loaded: number; total: number; lengthComputable: boolean }) => {
    const now = performance.now()
    const samples = samplesRef.current
    samples.push({ t: now, loaded: e.loaded })
    const cutoff = now - SPEED_WINDOW_MS
    while (samples.length > 1 && samples[0].t < cutoff) samples.shift()
    const first = samples[0]
    const dt = (now - first.t) / 1000
    const speed = dt > 0 ? Math.max(0, (e.loaded - first.loaded) / dt) : 0
    const total = e.lengthComputable && e.total > 0 ? e.total : 0
    const remaining = total > 0 ? Math.max(0, total - e.loaded) : 0
    const eta = speed > 0 && remaining > 0 ? remaining / speed : null
    const isComplete = total > 0 && e.loaded >= total
    setState({
      phase: isComplete ? 'processing' : 'uploading',
      loaded: e.loaded,
      total,
      speedBps: isComplete ? 0 : speed,
      etaSec: isComplete ? null : eta,
      error: null,
    })
  }, [])

  const finish = useCallback(() => {
    samplesRef.current = []
    setState((s) => ({
      ...s,
      phase: 'done',
      speedBps: 0,
      etaSec: null,
      loaded: s.total > 0 ? s.total : s.loaded,
    }))
  }, [])

  const fail = useCallback((e: unknown) => {
    samplesRef.current = []
    setState((s) => ({
      ...s,
      phase: 'error',
      speedBps: 0,
      etaSec: null,
      error: e instanceof Error ? e.message : String(e),
    }))
  }, [])

  const reset = useCallback(() => {
    samplesRef.current = []
    setState(INITIAL)
  }, [])

  return { state, start, onProgress, finish, fail, reset }
}

// ── formatting helpers (exported separately for unit testing / reuse elsewhere) ──

export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${units[i]}`
}

export function formatSpeed(bps: number): string {
  if (!Number.isFinite(bps) || bps <= 0) return '—'
  return `${formatBytes(bps)}/s`
}

export function formatEta(sec: number | null): string {
  if (sec == null || !Number.isFinite(sec) || sec < 0) return '—'
  if (sec < 1) return '<1s'
  if (sec < 60) return `${Math.ceil(sec)}s`
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  if (m < 60) return `${m}m${String(s).padStart(2, '0')}s`
  const h = Math.floor(m / 60)
  const mm = m % 60
  return `${h}h${String(mm).padStart(2, '0')}m`
}
