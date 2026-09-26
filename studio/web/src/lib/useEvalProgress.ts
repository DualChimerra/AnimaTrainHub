import { useEffect, useRef, useState } from 'react'
import { api, type EvalMetricResult, type Task } from '../api/client'
import { useEventStream } from './useEventStream'

// Visibility for after-training eval: the moment the training process exits,
// the task is marked done, while eval runs afterward as separate jobs (per
// checkpoint: eval_samples -> eval_clip -> eval_dino) that don't show up in
// the queue list, so the user has no visibility into them. This assembles an
// "evaluating done/total" out of listEvalMetrics' per-checkpoint status,
// starting to watch immediately on the eval_auto_after_training_queued
// event, and can also pick up mid-flight metric state after a page refresh.
// Purely frontend, no new backend events needed.

const ACTIVE_STATUS = new Set(['pending', 'running'])
const METRIC_KEYS = ['clip_t', 'clip_i', 'dino_i'] as const
const POLL_MS = 4000

export interface EvalProgress {
  /** Whether any checkpoint is still generating samples / computing metrics */
  active: boolean
  /** Number of checkpoints that finished evaluating */
  done: number
  /** Total checkpoint count for this evaluation */
  total: number
}

/** Aggregates "evaluating done/total" from listEvalMetrics' results.
 *  A checkpoint is "evaluating" if either its run-level status or any of its
 *  metric statuses is pending/running (run pending/running covers the sample
 *  generation phase, metric status covers the metric computation phase). */
export function evalProgressFromResults(results: EvalMetricResult[]): EvalProgress {
  let active = false
  let done = 0
  for (const r of results) {
    const runActive = ACTIVE_STATUS.has(r.status)
    const metricActive = METRIC_KEYS.some((k) => {
      const s = r.metric_states?.[k]?.status
      return s != null && ACTIVE_STATUS.has(s)
    })
    if (runActive || metricActive) active = true
    else done += 1
  }
  return { active, done, total: results.length }
}

/** Evaluation progress for a single task (used by the QueueDetail header).
 *  Fetches once when `enabled` (as a refresh fallback), polls every 4s while
 *  evaluating, and self-stops once evaluation ends (keeps the final
 *  progress; the caller stops showing the badge once active=false). With no
 *  evaluation running, it just fetches once and stops polling. The
 *  after-training queued event triggers watching to start immediately. */
export function useTaskEvalProgress(
  pid: number | null | undefined,
  vid: number | null | undefined,
  taskId: number | null | undefined,
  enabled: boolean,
): EvalProgress | null {
  const [progress, setProgress] = useState<EvalProgress | null>(null)
  const timer = useRef<number | null>(null)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    const clear = () => {
      if (timer.current != null) { window.clearTimeout(timer.current); timer.current = null }
    }
    const poll = async () => {
      if (!enabled || !pid || !vid || !taskId) return
      let p: EvalProgress | null = null
      try {
        const r = await api.listEvalMetrics(pid, vid, taskId)
        p = evalProgressFromResults(r.results ?? [])
      } catch {
        // Eval progress is auxiliary info; a failed fetch shouldn't be disruptive -- retry on the next tick (if still watching)
      }
      if (!mounted.current) return
      if (p) setProgress(p)
      clear()
      if (p?.active) timer.current = window.setTimeout(() => void poll(), POLL_MS)
    }
    if (enabled) void poll()
    else setProgress(null)
    return () => { mounted.current = false; clear() }
  }, [enabled, pid, vid, taskId])

  // After-training eval queued -> start watching immediately (the first fetch might race ahead of the enqueue and miss it)
  useEventStream((evt) => {
    if (evt.type !== 'eval_auto_after_training_queued') return
    if (evt.task_id !== taskId || !enabled || !pid || !vid || !taskId) return
    void (async () => {
      try {
        const r = await api.listEvalMetrics(pid, vid, taskId)
        if (mounted.current) setProgress(evalProgressFromResults(r.results ?? []))
      } catch { /* ignore */ }
    })()
  })

  return progress
}

/** For the Queue list: an eval-progress Map across multiple tasks. The watch
 *  set is built from the eval_auto_after_training_queued event (immediate)
 *  plus a fallback probe of the "most recently completed done task" whenever
 *  tasks change (so it can pick back up mid-eval after a page refresh). Only
 *  polls the tasks in the watch set (usually 0-1), removing one once its
 *  eval ends. */
export function useEvaluatingTasks(tasks: Task[]): Map<number, EvalProgress> {
  const [progress, setProgress] = useState<Map<number, EvalProgress>>(new Map())
  const watch = useRef<Set<number>>(new Set())
  const meta = useRef<Map<number, { pid: number; vid: number }>>(new Map())

  // id -> pid/vid (eval tasks are always training tasks bound to a project/version)
  useEffect(() => {
    const m = new Map<number, { pid: number; vid: number }>()
    for (const t of tasks) {
      if (t.project_id != null && t.version_id != null) {
        m.set(t.id, { pid: t.project_id, vid: t.version_id })
      }
    }
    meta.current = m
  }, [tasks])

  useEventStream((evt) => {
    if (evt.type === 'eval_auto_after_training_queued' && typeof evt.task_id === 'number') {
      watch.current.add(evt.task_id)
    }
  })

  // Fallback: probe the most recently completed done task to check whether it's still evaluating (covers a missed event on refresh)
  useEffect(() => {
    const done = tasks.filter(
      (t) => t.status === 'done' && t.project_id != null && t.version_id != null,
    )
    if (done.length === 0) return
    const latest = done.reduce((a, b) => ((b.finished_at ?? 0) > (a.finished_at ?? 0) ? b : a))
    let cancelled = false
    void (async () => {
      try {
        const r = await api.listEvalMetrics(latest.project_id!, latest.version_id!, latest.id)
        if (cancelled) return
        const p = evalProgressFromResults(r.results ?? [])
        if (p.active) {
          watch.current.add(latest.id)
          setProgress((prev) => new Map(prev).set(latest.id, p))
        }
      } catch { /* ignore */ }
    })()
    return () => { cancelled = true }
  }, [tasks])

  useEffect(() => {
    let alive = true
    const poll = async () => {
      const ids = Array.from(watch.current)
      if (ids.length === 0) return
      await Promise.all(ids.map(async (id) => {
        const m = meta.current.get(id)
        if (!m) { watch.current.delete(id); return }
        try {
          const r = await api.listEvalMetrics(m.pid, m.vid, id)
          const p = evalProgressFromResults(r.results ?? [])
          if (!alive) return
          setProgress((prev) => new Map(prev).set(id, p))
          if (!p.active && p.total > 0) watch.current.delete(id)  // evaluation ended, stop watching
        } catch { /* ignore */ }
      }))
    }
    const interval = window.setInterval(() => void poll(), POLL_MS)
    void poll()
    return () => { alive = false; window.clearInterval(interval) }
  }, [])

  return progress
}
