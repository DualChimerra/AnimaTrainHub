/**
 * useMonitorProgress -- shared monitor state subscription hook (PR #37 delta protocol).
 *
 * Protocol:
 *   - on mount + SSE reconnect: GET /api/state?task_id=X&max_points=1000 fetches a downsampled snapshot
 *   - after that, SSE monitor_progress pushes deltas, and this hook merges
 *     appended_losses/lr/samples into state; scalar fields (step/speed/...)
 *     are replaced each time
 *   - dedup: appended_losses/lr are filtered by already-known step; samples
 *     are filtered by (step, path) -- this prevents duplicate points when a
 *     reconnect's snapshot overlaps with the poller's delta boundary
 *   - cap: losses/lr_history each capped at 5000; samples capped at 50 (same as the backend cap)
 *
 * When taskId is null the hook is fully idle (used by the Queue/Topbar's
 * cross-task views, which don't subscribe when nothing is running). Changing
 * taskId clears state and refetches the snapshot.
 */
import { useCallback, useSyncExternalStore } from 'react'
import { api, type MonitorState } from '../api/client'
import { subscribeEventStream } from './useEventStream'

interface MonitorProgressDelta {
  step?: number
  total_steps?: number
  epoch?: number
  total_epochs?: number
  speed?: number
  start_time?: number | null
  appended_losses?: Array<{ step: number; loss: number; time?: number }>
  appended_lr?: Array<{ step: number; lr: number }>
  appended_optimizer_metrics?: NonNullable<MonitorState['optimizer_metrics_history']>
  appended_samples?: NonNullable<MonitorState['samples']>
  config?: Record<string, string | number | boolean>
}

// Matches the trailing-trim cap built into the backend's
// train_monitor.update_monitor (runtime/train_monitor.py:108-116). The
// earlier 5000 cap was based on the server defaulting to a 1500-point
// downsample; after switching to full snapshots, a cold start can already
// have >=10k points, so 5000 would immediately slice off the early data.
// 50000 matches the backend, with disk as the ultimate backstop for the
// cap; the frontend chart downsamples to 600 internally for rendering, so
// this doesn't hurt perf.
const MAX_LOSSES = 50000
const MAX_LR = 50000
const MAX_SAMPLES = 50

function mergeDelta(prev: MonitorState | null, delta: MonitorProgressDelta): MonitorState {
  const base: MonitorState = prev ?? {}

  // dedup cursors -- inferred from prev's tail, to avoid overlap between a reconnect's snapshot and the next delta
  const losses = base.losses ?? []
  const lrHistory = base.lr_history ?? []
  const optimizerMetricsHistory = base.optimizer_metrics_history ?? []
  const samples = base.samples ?? []
  const lastLossStep = losses.length ? losses[losses.length - 1].step : -1
  const lastLrStep = lrHistory.length ? lrHistory[lrHistory.length - 1].step : -1
  const lastOptimizerMetricsStep = optimizerMetricsHistory.length
    ? optimizerMetricsHistory[optimizerMetricsHistory.length - 1].step
    : -1
  const knownSamples = new Set(samples.map((s) => `${s.step ?? ''}|${s.path}`))

  const newLosses = (delta.appended_losses ?? []).filter((l) => l.step > lastLossStep)
  const newLr = (delta.appended_lr ?? []).filter((l) => l.step > lastLrStep)
  const newOptimizerMetrics = (delta.appended_optimizer_metrics ?? [])
    .filter((l) => l.step > lastOptimizerMetricsStep)
  const newSamples = (delta.appended_samples ?? []).filter(
    (s) => !knownSamples.has(`${s.step ?? ''}|${s.path}`),
  )

  const mergedLosses = newLosses.length ? [...losses, ...newLosses] : losses
  const mergedLr = newLr.length ? [...lrHistory, ...newLr] : lrHistory
  const mergedOptimizerMetrics = newOptimizerMetrics.length
    ? [...optimizerMetricsHistory, ...newOptimizerMetrics]
    : optimizerMetricsHistory
  const mergedSamples = newSamples.length ? [...samples, ...newSamples] : samples

  return {
    ...base,
    step: delta.step ?? base.step,
    total_steps: delta.total_steps ?? base.total_steps,
    epoch: delta.epoch ?? base.epoch,
    total_epochs: delta.total_epochs ?? base.total_epochs,
    speed: delta.speed ?? base.speed,
    start_time: delta.start_time ?? base.start_time,
    losses: mergedLosses.length > MAX_LOSSES ? mergedLosses.slice(-MAX_LOSSES) : mergedLosses,
    lr_history: mergedLr.length > MAX_LR ? mergedLr.slice(-MAX_LR) : mergedLr,
    optimizer_metrics_history: mergedOptimizerMetrics.length > MAX_LR
      ? mergedOptimizerMetrics.slice(-MAX_LR)
      : mergedOptimizerMetrics,
    samples: mergedSamples.length > MAX_SAMPLES ? mergedSamples.slice(-MAX_SAMPLES) : mergedSamples,
    config: delta.config ?? base.config,
  }
}

export interface MonitorProgress {
  state: MonitorState | null
  connected: boolean
  /** Proactively refetches a snapshot (consumers usually don't need this; the hook calls it internally). */
  refetch: () => Promise<void>
}

// ── shared per-task store ───────────────────────────────────────────────────
// Topbar, Sidebar, the queue page, the task page and its dashboard all watch the
// same running task at once. Each used to fetch its own full snapshot (tens of
// thousands of loss points, several MB of JSON) and merge every delta into its own
// copy; now there is one entry per task: one snapshot fetch, one merge per delta,
// and every subscriber reads the same object. Topbar / Sidebar stay mounted across
// pages, so the entry survives navigation while the task runs.

interface Snap { state: MonitorState | null; connected: boolean }
interface Entry {
  snap: Snap
  lastUpdate: number
  subs: Set<() => void>
  inflight: Promise<void> | null
  unsubscribe: () => void
}

const _entries = new Map<number, Entry>()
const IDLE: Snap = { state: null, connected: false }

function _set(e: Entry, next: Partial<Snap>): void {
  e.snap = { ...e.snap, ...next }
  for (const cb of e.subs) cb()
}

function _refetch(tid: number): Promise<void> {
  const e = _entries.get(tid)
  if (!e) return Promise.resolve()
  if (e.inflight) return e.inflight
  e.inflight = (async () => {
    try {
      const s = await api.getMonitorState(tid)
      e.lastUpdate = Date.now()
      _set(e, { state: s, connected: true })
    } catch {
      if (Date.now() - e.lastUpdate > 5000) _set(e, { connected: false })
    } finally {
      e.inflight = null
    }
  })()
  return e.inflight
}

function _acquire(tid: number): Entry {
  const existing = _entries.get(tid)
  if (existing) return existing
  const entry: Entry = { snap: IDLE, lastUpdate: 0, subs: new Set(), inflight: null, unsubscribe: () => {} }
  _entries.set(tid, entry)
  entry.unsubscribe = subscribeEventStream((evt) => {
    if (evt.type !== 'monitor_progress') return
    if (String(evt.task_id) !== String(tid)) return
    const delta = evt.delta as MonitorProgressDelta | undefined
    if (!delta) return
    entry.lastUpdate = Date.now()
    _set(entry, { state: mergeDelta(entry.snap.state, delta), connected: true })
  }, () => { void _refetch(tid) })
  void _refetch(tid)
  return entry
}

function _release(tid: number): void {
  const e = _entries.get(tid)
  if (!e || e.subs.size > 0) return
  e.unsubscribe()
  _entries.delete(tid)
}

export function useMonitorProgress(taskId: number | null): MonitorProgress {
  const subscribe = useCallback((cb: () => void) => {
    if (taskId == null) return () => {}
    const e = _acquire(taskId)
    e.subs.add(cb)
    return () => {
      e.subs.delete(cb)
      _release(taskId)
    }
  }, [taskId])
  const getSnap = useCallback(
    () => (taskId == null ? IDLE : _entries.get(taskId)?.snap ?? IDLE),
    [taskId],
  )
  const snap = useSyncExternalStore(subscribe, getSnap, getSnap)
  const refetch = useCallback(
    () => (taskId == null ? Promise.resolve() : _refetch(taskId)),
    [taskId],
  )
  return { state: snap.state, connected: snap.connected, refetch }
}

// exposed for tests
export { mergeDelta as _mergeDeltaForTest }
