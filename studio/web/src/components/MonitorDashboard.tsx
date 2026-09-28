/**
 * MonitorDashboard — native React training monitor
 * Replaces the monitor_smooth.html iframe.
 * Data source: GET /api/state?task_id=N fetches a downsampled snapshot + SSE monitor_progress
 * goes through the useMonitorProgress hook for delta merging (PR #37's incremental protocol).
 */
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api, type EvalJobInfo, type EvalMetricResult, type EvalMetricState, type LoraCkpt, type MonitorState } from '../api/client'
import { evalProgressFromResults } from '../lib/useEvalProgress'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import { useTranslation } from 'react-i18next'
import { InfoButton } from './InfoButton'
import ImagePreviewModal from './ImagePreviewModal'

// ── helpers ────────────────────────────────────────────────────────────────

function fmtSec(sec: number): string {
  if (!sec || sec < 0) return '--'
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = Math.floor(sec % 60)
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`
  if (m > 0) return `${m}m ${String(s).padStart(2, '0')}s`
  return `${s}s`
}

function calcEMA(data: number[], alpha = 0.02): number[] {
  if (!data.length) return []
  const out = [data[0]]
  for (let i = 1; i < data.length; i++) out.push(alpha * data[i] + (1 - alpha) * out[i - 1])
  return out
}

function downsample<T>(arr: T[], n: number): T[] {
  if (arr.length <= n) return arr
  return Array.from({ length: n }, (_, i) => arr[Math.round((i * (arr.length - 1)) / (n - 1))])
}

// ── StatCard ───────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, tone }: {
  label: string
  value: string
  sub?: string
  tone?: 'accent' | 'ok' | 'warn'
}) {
  const colorCls = tone === 'accent' ? 'text-accent' : tone === 'ok' ? 'text-ok' : tone === 'warn' ? 'text-warn' : 'text-fg-primary'
  return (
    <div className="card px-[14px] py-3 min-w-0 monitor-stat">
      <div className="caption mb-1.5">
        {label}
      </div>
      <div className={`text-xl font-bold font-mono tabular-nums tracking-[-0.02em] leading-[1.05] ${colorCls}`}>
        {value}
      </div>
      {sub && (
        <div className="text-xs text-fg-tertiary font-mono mt-1">
          {sub}
        </div>
      )}
    </div>
  )
}

// ── SmoothControl ──────────────────────────────────────────────────────────
// EMA slider; alpha = 1 means "no smoothing" (SeriesChart skips EMA internally based on this).

function SmoothControl({ alpha, setAlpha, min, max, step }: {
  alpha: number
  setAlpha: (v: number) => void
  min: number
  max: number
  step: number
}) {
  return (
    <label className="flex items-center gap-1 cursor-pointer text-xs text-fg-tertiary">
      smooth
      <input
        type="range" min={min} max={max} step={step} value={alpha}
        onChange={(e) => setAlpha(parseFloat(e.target.value))}
        style={{ width: 60, accentColor: 'var(--accent)' }}
      />
      <span className="font-mono w-[4ch] text-right">
        {alpha >= 0.999 ? 'off' : alpha.toFixed(alpha < 0.1 ? 3 : 2)}
      </span>
    </label>
  )
}

// ── SeriesChart (pure SVG) ─────────────────────────────────────────────────
// Generic step-by-value line chart: raw + EMA-smoothed dual lines + xy axis ticks.
// Shared by loss / lr / d: pass rawColor/smoothColor for custom coloring, pass yFormat to control
// the y-axis number format (scientific notation vs. fixed point).

function SeriesChart({ data, rawColor, smoothColor, fillColor, emaAlpha, yFormat, height, minHeight, axes = true, refLine }: {
  data: Array<{ step: number; value: number }>
  rawColor: string
  smoothColor: string
  fillColor?: string
  emaAlpha: number
  yFormat: (v: number) => string
  /** Fixed pixel height (used for secondary charts, e.g. d value) */
  height?: number
  /** Minimum pixel height in flex mode; stretches with the parent height when the viewport is tall enough (used for the main charts, e.g. loss / lr) */
  minHeight?: number
  /** Whether to draw axes + tick labels + gridlines; false degrades to a plain sparkline suited to short charts (d value) */
  axes?: boolean
  /** Optional horizontal reference line (eval: the pure-base-model baseline value); the y range accounts for it. */
  refLine?: number
}) {
  // ResizeObserver measures the real pixel size, the viewBox uses that real size -> the SVG renders 1:1,
  // so text/line widths aren't distorted by preserveAspectRatio's non-uniform scaling.
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const [size, setSize] = useState<{ w: number; h: number } | null>(null)

  useLayoutEffect(() => {
    const el = wrapperRef.current
    if (!el) return
    const measure = (w: number, h: number) => {
      if (w <= 0 || h <= 0) return
      setSize((prev) =>
        prev && Math.abs(prev.w - w) < 1 && Math.abs(prev.h - h) < 1 ? prev : { w, h },
      )
    }
    const rect = el.getBoundingClientRect()
    measure(rect.width, rect.height)
    const ro = new ResizeObserver(([entry]) => {
      const { width, height: h } = entry.contentRect
      measure(width, h)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const wrapperStyle: React.CSSProperties = height != null
    ? { height, width: '100%' }
    : { flex: 1, minHeight: minHeight ?? 0, width: '100%' }

  return (
    <div ref={wrapperRef} style={wrapperStyle}>
      {!data.length ? (
        <div className="grid place-items-center text-fg-tertiary text-sm h-full">
          Waiting for data…
        </div>
      ) : size ? (
        <ChartSvg
          data={data}
          W={size.w}
          H={size.h}
          rawColor={rawColor}
          smoothColor={smoothColor}
          fillColor={fillColor}
          emaAlpha={emaAlpha}
          yFormat={yFormat}
          axes={axes}
          refLine={refLine}
        />
      ) : null}
    </div>
  )
}

function ChartSvg({ data, W, H, rawColor, smoothColor, fillColor, emaAlpha, yFormat, axes, refLine }: {
  data: Array<{ step: number; value: number }>
  W: number
  H: number
  rawColor: string
  smoothColor: string
  fillColor?: string
  emaAlpha: number
  yFormat: (v: number) => string
  axes: boolean
  refLine?: number
}) {
  const pts = downsample(data, 600)
  const raw = pts.map((p) => p.value)
  // alpha = 1 -> skip EMA, pure raw (avoids redundant visual double-lines)
  const smooth = emaAlpha >= 0.999 ? raw : calcEMA(raw, emaAlpha)
  const steps = pts.map((p) => p.step)

  // Sparkline mode (axes=false) drops the left-side space reserved for y labels so the path fills the width;
  // with axes, PX reserves 48px for y ticks (13pt char width ~7-8px x "0.0796"'s 6 chars ~= 46);
  // PY reserves 18px for x ticks (13pt char height ~14, plus 4px breathing room).
  const PX = axes ? 48 : 0
  const PY = axes ? 18 : 2
  const RX = axes ? 8 : 0  // right-side padding
  // y range is computed from smooth (degrades to raw when there's no smoothing) -- raw spikes above the top get
  // clipped, and that's intentional: it trades that for the smooth signal filling the height so the trend stays readable. Same behavior as the old LossChart.
  const refVals = emaAlpha >= 0.999 ? raw : smooth
  const hasRef = typeof refLine === 'number' && Number.isFinite(refLine)
  // The y range accounts for the baseline reference line so it's always visible in the chart (whether the curve is above or below the base line at a glance).
  const minV = Math.min(...refVals, ...(hasRef ? [refLine as number] : []))
  const maxV = Math.max(...refVals, ...(hasRef ? [refLine as number] : []))
  const range = maxV - minV || Math.max(Math.abs(maxV), 1e-9) * 1e-3 || 1e-9
  const x = (i: number) => PX + (i / Math.max(1, pts.length - 1)) * (W - PX - RX)
  const y = (v: number) => PY + (1 - (v - minV) / range) * (H - PY - PY)

  const smoothPath = smooth.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')
  const areaPath = fillColor
    ? smoothPath + ` L${x(smooth.length - 1).toFixed(1)},${H - PY} L${PX},${H - PY}Z`
    : null
  const rawPath = raw.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')

  const yTicks = [minV, (minV + maxV) / 2, maxV].map((v) => ({
    v, y: y(v), label: yFormat(v),
  }))
  // With few points (eval only has a handful of checkpoints), rounding 5 quantiles can collide on the same index (e.g. 3 points ->
  // 0,1,1,2,2), causing labels like "20 20 40 40" to overlap at the same x. Dedupe by index.
  const xTicks = [...new Set(
    [0, 0.25, 0.5, 0.75, 1].map((t) => Math.round(t * Math.max(1, pts.length - 1))),
  )].map((i) => ({ i, x: x(i), label: String(steps[i] ?? '') }))

  const lastY = y(smooth[smooth.length - 1])
  const showSmoothLayer = emaAlpha < 0.999

  // viewBox matches the real size 1:1, skipping preserveAspectRatio="none"'s non-uniform scaling --
  // text/line widths render at real pixel size, no longer distorted by the parent's aspect ratio.
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: '100%', display: 'block' }}>
      {axes && (
        <>
          {/* axis lines */}
          <line x1={PX} y1={PY} x2={PX} y2={H - PY} stroke="var(--border-subtle)" />
          <line x1={PX} y1={H - PY} x2={W - RX} y2={H - PY} stroke="var(--border-subtle)" />
          {/* grid */}
          {[0.25, 0.5, 0.75].map((t) => (
            <line key={t} x1={PX} y1={PY + t * (H - 2 * PY)} x2={W - RX} y2={PY + t * (H - 2 * PY)}
              stroke="var(--border-subtle)" strokeDasharray="3 3" />
          ))}
        </>
      )}
      {/* area (smooth fill, optional) */}
      {areaPath && <path d={areaPath} fill={fillColor} opacity="0.5" />}
      {/* raw -- dimmed when smooth is shown, the main line when there's no smoothing */}
      <path
        d={rawPath}
        stroke={showSmoothLayer ? rawColor : smoothColor}
        strokeWidth={showSmoothLayer ? 1 : 2}
        strokeOpacity={showSmoothLayer ? 0.45 : 1}
        fill="none"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {/* smooth */}
      {showSmoothLayer && (
        <path d={smoothPath} stroke={smoothColor} strokeWidth="2" fill="none" strokeLinejoin="round" strokeLinecap="round" />
      )}
      {/* last point */}
      <circle cx={x(smooth.length - 1)} cy={lastY} r="4" fill={smoothColor} stroke="var(--bg-surface)" strokeWidth="2" />
      {/* Baseline reference line (pure base model) + a "base" label: curve above it = better than base, below = worse than base */}
      {hasRef && (
        <>
          <line
            x1={PX} y1={y(refLine as number)} x2={W - RX} y2={y(refLine as number)}
            stroke="var(--fg-tertiary)" strokeWidth="1" strokeDasharray="4 3" opacity="0.75"
          />
          <text
            x={W - RX - 1} y={y(refLine as number) - 3} fontSize="10"
            fill="var(--fg-tertiary)" fontFamily="var(--font-mono)" textAnchor="end"
          >base</text>
        </>
      )}
      {axes && (
        <>
          {/* y axis labels -- y offset +4.5 = fontSize/2 + a small baseline tweak to vertically center the text on the tick */}
          {yTicks.map(({ v, y: yt, label }) => (
            <text key={v} x={PX - 4} y={yt + 4.5} fontSize="13" fill="var(--fg-tertiary)"
              fontFamily="var(--font-mono)" textAnchor="end">{label}</text>
          ))}
          {/* x axis labels -- the first/last ticks sit right at the SVG edge, a middle anchor would clip half the text;
              switch to start/end anchors to push the text inward; the middle ticks keep a middle anchor. */}
          {xTicks.map(({ i: idx, x: xt, label }, i, arr) => {
            const anchor = i === 0 ? 'start' : i === arr.length - 1 ? 'end' : 'middle'
            return (
              <text key={idx} x={xt} y={H - 3} fontSize="13" fill="var(--fg-tertiary)"
                fontFamily="var(--font-mono)" textAnchor={anchor}>{label}</text>
            )
          })}
        </>
      )}
    </svg>
  )
}

// ── EvalMetricsPanel ──────────────────────────────────────────────────────

export const EVAL_METRIC_KEYS = ['clip_t', 'clip_i', 'dino_i', 'ccip_i', 'tag_recall'] as const
export type EvalMetricKey = typeof EVAL_METRIC_KEYS[number]
// Core metrics are always shown; new anime-domain metrics are off by default and only show a card/column once they've been computed (status != not_run).
export const CORE_METRIC_KEYS = new Set<EvalMetricKey>(['clip_t', 'clip_i', 'dino_i'])

const EVAL_LABELS: Record<EvalMetricKey, string> = {
  clip_t: 'CLIP-T',
  clip_i: 'CLIP-I',
  dino_i: 'DINO-I',
  ccip_i: 'CCIP-I',
  tag_recall: 'Tag-Recall',
}

// One line color per metric (to tell them apart side by side). High contrast and distinguishable on a dark background.
const EVAL_COLORS: Record<EvalMetricKey, string> = {
  clip_t: '#3fb950',
  clip_i: '#58a6ff',
  dino_i: '#bc8cff',
  ccip_i: '#f778ba',
  tag_recall: '#e3b341',
}


export function checkpointSortValue(result: EvalMetricResult, index: number): number {
  const value = result.checkpoint?.value
  if (typeof value === 'number') return value
  return result.updated_at ?? result.created_at ?? index
}

export function checkpointLabel(result: EvalMetricResult): string {
  return result.checkpoint?.label
    || result.checkpoint?.path?.split(/[\\/]/).pop()
    || result.run_id
}

export function metricState(result: EvalMetricResult, key: EvalMetricKey): EvalMetricState | undefined {
  return result.metric_states?.[key]
}

export function metricValue(result: EvalMetricResult, key: EvalMetricKey): number | null {
  const stateValue = metricState(result, key)?.value
  if (typeof stateValue === 'number' && Number.isFinite(stateValue)) return stateValue
  const raw = result.metrics?.[key]
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : null
}

function formatEvalValue(value: number | null, state?: EvalMetricState): string {
  if (value != null) return value.toFixed(4)
  const status = state?.status || 'not_run'
  if (status === 'pending') return 'pending'
  if (status === 'running') return 'running'
  if (status === 'failed') return 'failed'
  if (status === 'unavailable') return 'n/a'
  return '--'
}

function stateTone(state?: EvalMetricState, value?: number | null): 'ok' | 'warn' | 'err' | 'muted' {
  if (state?.status === 'failed') return 'err'
  if (state?.status === 'pending' || state?.status === 'running') return 'warn'
  if (value != null || state?.status === 'done') return 'ok'
  return 'muted'
}

function toneClass(tone: 'ok' | 'warn' | 'err' | 'muted'): string {
  if (tone === 'ok') return 'text-ok'
  if (tone === 'warn') return 'text-warn'
  if (tone === 'err') return 'text-err'
  return 'text-fg-tertiary'
}

// Unified progress copy + tone for a checkpoint row (whether triggered inline/post-training/manually, always
// derived from the same run.json/metrics.json state: sample generation first (sample_run.summary done/total), then metrics.
export function evalRowStatus(result: EvalMetricResult): {
  /** i18n key, or null when the raw backend status is the best label we have. */
  key: string | null
  params?: Record<string, unknown>
  text?: string
  tone: 'ok' | 'warn' | 'err' | 'muted'
} {
  const s = result.sample_run?.summary
  const total = s?.total ?? 0
  const sampleDone = (s?.done ?? 0) + (s?.failed ?? 0)
  const samplingActive = total > 0 && sampleDone < total &&
    (result.status === 'running' || result.status === 'pending' || (s?.running ?? 0) > 0 || (s?.pending ?? 0) > 0)
  if (samplingActive) {
    return { key: 'monitor.rowSampling', params: { done: s?.done ?? 0, total }, tone: 'warn' }
  }
  const metricActive = EVAL_METRIC_KEYS.some((k) => {
    const st = metricState(result, k)?.status
    return st === 'pending' || st === 'running'
  })
  if (metricActive) return { key: 'monitor.rowMetrics', tone: 'warn' }
  if (result.status === 'failed') return { key: 'monitor.rowFailed', tone: 'err' }
  if (result.status === 'done') return { key: 'monitor.rowDone', tone: 'ok' }
  return { key: null, text: result.status, tone: 'muted' }
}

export function EvalMetricsPanel({ state, connected, taskId }: {
  state: MonitorState | null
  connected: boolean
  taskId?: number
}) {
  const { t } = useTranslation()
  const pid = state?.project_id
  const vid = state?.version_id
  const [payload, setPayload] = useState<Awaited<ReturnType<typeof api.listEvalMetrics>> | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Post-training/manual eval jobs (used to fetch the raw log in the unified log area).
  const [evalJobs, setEvalJobs] = useState<EvalJobInfo[]>([])

  const load = useCallback(async (quiet = false) => {
    if (!pid || !vid) return
    if (!quiet) setLoading(true)
    try {
      const next = await api.listEvalMetrics(pid, vid, taskId)
      setPayload(next)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (!quiet) setLoading(false)
    }
    if (taskId) {
      try {
        const ej = await api.listTaskEvalJobs(pid, vid, taskId)
        setEvalJobs(ej.jobs)
      } catch {
        // The log association is auxiliary info; a failed fetch shouldn't be disruptive
      }
    }
  }, [pid, vid, taskId])

  useEffect(() => {
    setPayload(null)
    setError(null)
    if (!pid || !vid) return
    void load()
  }, [pid, vid, load])

  const results = useMemo(() => {
    return [...(payload?.results ?? [])]
      .sort((a, b) => checkpointSortValue(a, 0) - checkpointSortValue(b, 0))
  }, [payload?.results])

  // The baseline run (pure-base-model control) isn't shown as a checkpoint -- it's only used to compute delta
  // for each checkpoint (the backend already attaches it as result.delta). Each "run evaluation" auto-clears the
  // previous round, so this always only has this run's results, one per checkpoint, no cross-round dedup needed.
  const displayResults = useMemo(
    () => results.filter((r) => !r.baseline),
    [results],
  )

  const hasActiveMetric = useMemo(() => {
    return results.some((result) =>
      EVAL_METRIC_KEYS.some((key) => {
        const status = metricState(result, key)?.status
        return status === 'pending' || status === 'running'
      }),
    )
  }, [results])

  // There may still be an eval job running (including the "sample generation" stage -- while metric status is still
  // not_run, hasActiveMetric can't catch it). When re-running eval on an already-done task, this keeps polling going so the new run's progress/log refreshes.
  const hasActiveJob = useMemo(
    () => evalJobs.some((j) => j.status === 'pending' || j.status === 'running'),
    [evalJobs],
  )

  // Core metrics are always shown; new anime-domain metrics are off by default and only shown once computed (status != not_run), to avoid empty cards.
  const displayKeys = useMemo<EvalMetricKey[]>(
    () => EVAL_METRIC_KEYS.filter((k) =>
      CORE_METRIC_KEYS.has(k) ||
      displayResults.some((r) => {
        const s = metricState(r, k)?.status
        return s != null && s !== 'not_run'
      }),
    ),
    [displayResults],
  )

  // Post-training eval progress: reuses the existing results to aggregate "evaluating done/total" for the panel header
  const evalAgg = useMemo(() => evalProgressFromResults(results), [results])

  useEffect(() => {
    if (!pid || !vid) return
    if (!connected && !hasActiveMetric && !hasActiveJob) return
    const id = window.setInterval(() => void load(true), 5000)
    return () => window.clearInterval(id)
  }, [connected, hasActiveMetric, hasActiveJob, load, pid, vid])

  // -- Manual eval: pick a checkpoint -> POST /eval/run (task-scoped) --------------------------
  const [pickerOpen, setPickerOpen] = useState(false)
  const [ckpts, setCkpts] = useState<LoraCkpt[]>([])
  const [ckptsLoading, setCkptsLoading] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [running, setRunning] = useState(false)
  const [runMsg, setRunMsg] = useState<string | null>(null)

  const loadCkpts = useCallback(async () => {
    if (!pid || !vid) return
    setCkptsLoading(true)
    try {
      const items = await api.listVersionLoraCkpts(pid, vid)
      setCkpts(items)
    } catch (err) {
      setRunMsg(err instanceof Error ? err.message : String(err))
    } finally {
      setCkptsLoading(false)
    }
  }, [pid, vid])

  const togglePicker = useCallback(() => {
    setPickerOpen((open) => {
      const next = !open
      if (next && ckpts.length === 0) void loadCkpts()
      return next
    })
  }, [ckpts.length, loadCkpts])

  const toggleCkpt = useCallback((path: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }, [])

  const runEval = useCallback(async () => {
    if (!pid || !vid || !taskId || selected.size === 0) return
    setRunning(true)
    setRunMsg(null)
    try {
      const r = await api.runTaskEval(pid, vid, {
        task_id: taskId,
        checkpoints: [...selected],
      })
      setRunMsg(t('monitor.queuedEvals', { count: r.queued }))
      setSelected(new Set())
      setPickerOpen(false)
      void load(true)
    } catch (err) {
      setRunMsg(err instanceof Error ? err.message : String(err))
    } finally {
      setRunning(false)
    }
  }, [pid, vid, taskId, selected, load, t])

  const latestByKey = useMemo(() => {
    const out: Partial<Record<EvalMetricKey, { result: EvalMetricResult; value: number | null; state?: EvalMetricState }>> = {}
    for (const key of EVAL_METRIC_KEYS) {
      for (let i = displayResults.length - 1; i >= 0; i--) {
        const result = displayResults[i]
        const state = metricState(result, key)
        const value = metricValue(result, key)
        if (value != null || state?.status) {
          out[key] = { result, value, state }
          break
        }
      }
    }
    return out
  }, [displayResults])

  const seriesByKey = useMemo(() => {
    const out = Object.fromEntries(
      EVAL_METRIC_KEYS.map((k) => [k, [] as Array<{ x: number; value: number }>]),
    ) as Record<EvalMetricKey, Array<{ x: number; value: number }>>
    displayResults.forEach((result, index) => {
      const x = checkpointSortValue(result, index)
      for (const key of EVAL_METRIC_KEYS) {
        const value = metricValue(result, key)
        if (value != null) out[key].push({ x, value })
      }
    })
    return out
  }, [displayResults])

  // Pure-base-model baseline value for each metric (drawn as a horizontal reference line on the chart). The backend
  // attaches the same baseline_metrics to every non-baseline result, so any one of them can be used.
  const baselineByKey = useMemo(() => {
    const out: Partial<Record<EvalMetricKey, number>> = {}
    const bm = displayResults.find(
      (r) => r.baseline_metrics && Object.keys(r.baseline_metrics).length,
    )?.baseline_metrics
    if (bm) {
      for (const key of EVAL_METRIC_KEYS) {
        const v = bm[key]
        if (typeof v === 'number' && Number.isFinite(v)) out[key] = v
      }
    }
    return out
  }, [displayResults])

  if (!pid || !vid) {
    return (
      <div className="card px-4 py-3 text-sm text-fg-tertiary">
        {t('monitor.noVersionBound')}
      </div>
    )
  }

  return (
    <div className="card p-4 flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <div className="min-w-0">
          <div className="text-sm font-semibold">{t('monitor.metrics')}</div>
          <div className="text-xs text-fg-tertiary font-mono truncate">
            {state?.project_slug ?? `project ${pid}`}, {state?.version_label ?? `version ${vid}`}
          </div>
        </div>
        <span className="flex-1" />
        {evalAgg.active && (
          <span
            className="badge badge-accent text-xs"
            title={t('monitor.autoEvalRunningHint')}
          >
            <span className="dot dot-running" />
            {t('monitor.evaluating', { done: evalAgg.done, total: evalAgg.total })}
          </span>
        )}
        {loading && <span className="text-xs text-fg-tertiary">{t('common.loading')}</span>}
        {taskId != null && (
          <button
            type="button"
            onClick={togglePicker}
            className={`btn btn-sm ${pickerOpen ? 'btn-primary' : 'btn-secondary'}`}
          >
            {t('monitor.runEval')}
          </button>
        )}
        <button
          type="button"
          onClick={() => void load()}
          className="btn btn-secondary btn-sm"
        >
          {t('common.refresh')}
        </button>
      </div>

      {taskId != null && pickerOpen && (
        <div className="rounded-md border border-subtle bg-overlay px-3 py-2.5 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold">{t('monitor.pickCkpts')}</span>
            <span className="text-[11px] text-fg-tertiary">
              {t('monitor.pickCkptsHint')}
            </span>
            <span className="flex-1" />
            {ckpts.length > 0 && (
              <button
                type="button"
                className="text-[11px] text-fg-tertiary hover:text-fg underline"
                onClick={() =>
                  setSelected((prev) =>
                    prev.size === ckpts.length ? new Set() : new Set(ckpts.map((c) => c.path)),
                  )
                }
              >
                {selected.size === ckpts.length ? t('monitor.clearAll') : t('monitor.selectAll')}
              </button>
            )}
            <button
              type="button"
              onClick={() => setPickerOpen(false)}
              className="btn btn-ghost btn-sm text-fg-tertiary px-1.5"
              title={t('common.close')}
              aria-label={t('monitor.closeCkptPicker')}
            >
              ×
            </button>
          </div>
          {ckptsLoading ? (
            <div className="text-xs text-fg-tertiary py-1">{t('monitor.loadingCkpts')}</div>
          ) : ckpts.length === 0 ? (
            <div className="text-xs text-fg-tertiary py-1">{t('monitor.noCkpts')}</div>
          ) : (
            // Chip grid (same style as the test page's LoRA picker): auto-fill equal-width columns, +/checkmark markers,
            // accent-soft highlight for the selected state.
            <div
              className="grid gap-1.5 overflow-y-auto"
              style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', maxHeight: 200, padding: 2 }}
            >
              {ckpts.map((c) => {
                const isPicked = selected.has(c.path)
                return (
                  <button
                    key={c.path}
                    type="button"
                    onClick={() => toggleCkpt(c.path)}
                    className="font-mono flex items-center gap-1 min-w-0"
                    style={{
                      fontSize: 11,
                      padding: '4px 8px',
                      borderRadius: 'var(--r-md)',
                      border: isPicked ? '1px solid transparent' : '1px solid var(--border-subtle)',
                      background: isPicked ? 'var(--accent-soft)' : 'var(--bg-sunken)',
                      color: isPicked ? 'var(--accent)' : 'var(--fg-secondary)',
                      cursor: 'pointer',
                    }}
                    title={c.path}
                  >
                    <span className="shrink-0">{isPicked ? '✓' : '+'}</span>
                    <span className="truncate flex-1 text-left">{c.label}</span>
                  </button>
                )
              })}
            </div>
          )}
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={running || selected.size === 0}
              onClick={() => void runEval()}
              className="btn btn-primary btn-sm"
            >
              {running ? t('monitor.queueing') : `${t('monitor.runEval')}${selected.size ? ` (${selected.size})` : ''}`}
            </button>
            {runMsg && <span className="text-[11px] text-fg-tertiary">{runMsg}</span>}
          </div>
        </div>
      )}

      {error ? (
        <div className="rounded-md border border-err bg-err-soft px-3 py-2 text-sm text-err">
          {t('monitor.evalLoadFailed', { error })}
        </div>
      ) : results.length === 0 ? (
        <div className="rounded-md border border-dashed border-subtle px-3 py-3 text-sm text-fg-tertiary">
          {t('monitor.noEvalResults')}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
            {displayKeys.map((key) => {
              const latest = latestByKey[key]
              const tone = stateTone(latest?.state, latest?.value)
              const series = seriesByKey[key].map((p) => ({ step: p.x, value: p.value }))
              return (
                <div key={key} className="card p-4 flex flex-col min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1 shrink-0">
                    <span className="inline-flex items-center gap-1.5 text-sm font-semibold">
                      {EVAL_LABELS[key]}
                      <InfoButton ariaLabel={t('monitor.metricHelpFor', { metric: EVAL_LABELS[key] })}>
                        <p>{t(`monitor.evalDesc.${key}`)}</p>
                      </InfoButton>
                    </span>
                    <span className={`text-xs font-mono ${toneClass(tone)}`}>
                      {latest?.state?.status ?? 'not_run'}
                    </span>
                  </div>
                  <div className="flex items-baseline gap-2 shrink-0 mb-1.5">
                    <span className={`text-2xl font-semibold font-mono tabular-nums ${toneClass(tone)}`}>
                      {formatEvalValue(latest?.value ?? null, latest?.state)}
                    </span>
                    {(() => {
                      const d = latest?.result.delta?.[key]
                      return d != null ? (
                        <span
                          className={`text-xs font-mono tabular-nums shrink-0 ${d >= 0 ? 'text-ok' : 'text-err'}`}
                          title={t('monitor.deltaHint')}
                        >
                          {d >= 0 ? '+' : ''}{d.toFixed(4)}
                        </span>
                      ) : null
                    })()}
                    <span className="text-[11px] text-fg-tertiary truncate">
                      {latest ? checkpointLabel(latest.result) : t('monitor.waitingMetrics')}
                    </span>
                  </div>
                  <SeriesChart
                    data={series}
                    rawColor={EVAL_COLORS[key]}
                    smoothColor={EVAL_COLORS[key]}
                    emaAlpha={1}
                    yFormat={(v) => v.toFixed(4)}
                    height={132}
                    refLine={baselineByKey[key]}
                  />
                </div>
              )
            })}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="text-fg-tertiary">
                <tr className="border-b border-subtle">
                  <th className="text-left font-medium py-1.5 pr-3">checkpoint</th>
                  {displayKeys.map((key) => (
                    <th key={key} className="text-right font-medium py-1.5 px-2">
                      {EVAL_LABELS[key]}
                    </th>
                  ))}
                  <th className="text-right font-medium py-1.5 pl-3">{t('monitor.statusCol')}</th>
                </tr>
              </thead>
              <tbody>
                {displayResults.slice(-8).reverse().map((result) => {
                  const rowStatus = evalRowStatus(result)
                  return (
                    <tr key={result.run_id} className="border-b border-subtle last:border-0">
                      <td className="py-1.5 pr-3 max-w-[220px] truncate font-mono" title={checkpointLabel(result)}>
                        {checkpointLabel(result)}
                      </td>
                      {displayKeys.map((key) => {
                        const state = metricState(result, key)
                        const value = metricValue(result, key)
                        const tone = stateTone(state, value)
                        const d = result.delta?.[key]
                        return (
                          <td key={key} className={`py-1.5 px-2 text-right font-mono tabular-nums ${toneClass(tone)}`}>
                            {formatEvalValue(value, state)}
                            {d != null && value != null && (
                              <span
                                className={`ml-1 text-[10px] ${d >= 0 ? 'text-ok' : 'text-err'}`}
                                title={t('monitor.deltaHintShort')}
                              >
                                {d >= 0 ? '+' : ''}{d.toFixed(4)}
                              </span>
                            )}
                          </td>
                        )
                      })}
                      <td className="py-1.5 pl-3 text-right font-mono">
                        <span className={toneClass(rowStatus.tone)}>
                          {rowStatus.key ? t(rowStatus.key, rowStatus.params) : rowStatus.text}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}

// -- SampleViewer (single image + prev/next) --------------------------------

// The monitor records the global_step at the moment each image was triggered, so step can always be shown; epoch
// is only available for images sampled by epoch (filename epoch_N.png), parsed from the filename. When both are
// present -> badge/title "always show step, append ep if present", so the user doesn't have to open the filename to know the epoch.
function sampleMarks(s: { path: string; step?: number }): { step: number | null; epoch: number | null } {
  const fn = s.path.split(/[\\/]/).pop() ?? s.path
  const ep = /^epoch_(\d+)/i.exec(fn)
  const st = /^step_(\d+)/i.exec(fn)
  return {
    epoch: ep ? Number(ep[1]) : null,
    step: st ? Number(st[1]) : (s.step != null ? s.step : null),
  }
}

const SampleViewer = memo(function SampleViewer({ samples, taskId }: {
  samples: Array<{ path: string; step?: number }>
  taskId: number
}) {
  // Laid out in the array's original order (newest last, matching the training timeline). Multiple prompts at the
  // same step just show up as adjacent items with the same step, repeated indices visually convey "same step, different prompt".
  const list = samples
  const [active, setActive] = useState(list.length - 1)
  const [zoomOpen, setZoomOpen] = useState(false)
  const stripRef = useRef<HTMLDivElement | null>(null)

  // When images first appear / a new sample arrives, only follow the new end if the user's current selection was
  // at "the last item or beyond" (i.e. following the end); doesn't interrupt the user while looking back at earlier images.
  const prevLenRef = useRef(0)
  useEffect(() => {
    if (list.length === 0) {
      setActive(0)
      prevLenRef.current = 0
      return
    }
    if (active >= prevLenRef.current - 1) {
      setActive(list.length - 1)
    }
    prevLenRef.current = list.length
  }, [list.length, active])

  // When active changes, scroll the strip to the matching thumbnail (horizontal only, doesn't affect the outer layout)
  useEffect(() => {
    const strip = stripRef.current
    if (!strip) return
    const target = strip.children[active] as HTMLElement | undefined
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' })
    }
  }, [active])

  if (!list.length) return (
    <div className="grid place-items-center h-[300px] text-fg-tertiary text-sm">
      Waiting for samples…
    </div>
  )

  const cur = list[active]
  const filename = cur.path.split(/[\\/]/).pop() ?? cur.path
  const fullUrl = api.sampleImageUrl(filename, taskId)
  const curM = sampleMarks(cur)
  const markText = [
    curM.epoch != null ? `ep ${curM.epoch.toLocaleString()}` : null,
    curM.step != null ? `step ${curM.step.toLocaleString()}` : null,
  ].filter(Boolean).join(', ')

  return (
    <div className="flex flex-col gap-2.5 w-full flex-1">
      {/* Top thumbnail strip -- scrolls horizontally, laid out in the array's original order */}
      <div
        ref={stripRef}
        className="flex gap-1.5 overflow-x-auto pb-1 shrink-0"
        style={{ scrollbarWidth: 'thin' }}
      >
        {list.map((s, i) => {
          const fn = s.path.split(/[\\/]/).pop() ?? s.path
          const thumbUrl = api.sampleImageUrl(fn, taskId, 128)
          const isActive = i === active
          const m = sampleMarks(s)
          const thumbTitle = [
            m.epoch != null ? `ep ${m.epoch}` : null,
            m.step != null ? `step ${m.step}` : null,
          ].filter(Boolean).join(', ') || fn
          // The badge sits on the line below the thumbnail, not overlaid on the 64px thumbnail (would be unreadable).
          const thumbCaption = [
            m.epoch != null ? `ep${m.epoch}` : null,
            m.step != null ? `${m.step}` : null,
          ].filter(Boolean).join(', ')
          return (
            <button
              key={`${fn}-${i}`}
              onClick={() => setActive(i)}
              className="shrink-0 flex flex-col items-center gap-0.5 p-0 bg-transparent border-none cursor-pointer"
              title={thumbTitle}
            >
              <div
                className={[
                  'rounded-sm overflow-hidden border transition-colors bg-sunken',
                  isActive ? 'border-accent ring-2 ring-accent-soft' : 'border-subtle hover:border-bold',
                ].join(' ')}
                style={{ width: 64, height: 64 }}
              >
                <img
                  src={thumbUrl}
                  alt=""
                  loading="lazy"
                  className="w-full h-full object-cover block"
                />
              </div>
              {thumbCaption && (
                <span className={`text-[10px] font-mono leading-tight text-center ${isActive ? 'text-fg-primary' : 'text-fg-tertiary'}`}>
                  {thumbCaption}
                </span>
              )}
            </button>
          )
        })}
      </div>

      {/* Main image -- currently selected.
          The img uses absolute inset-0 to escape the flow, so the sample image's native resolution (1024x*)
          doesn't push the parent container's min-content; letterboxing is handled by object-contain.
          minHeight 220 is the floor (barely enough for the letterbox look); flex-1 fills the rest when the parent row is tall enough. */}
      <div
        className="bg-sunken rounded-sm overflow-hidden relative flex-1 min-h-0 monitor-sample-main"
        style={{ minHeight: 220 }}
      >
        <img
          key={fullUrl}
          src={fullUrl}
          alt="sample preview"
          decoding="async"
          onClick={() => setZoomOpen(true)}
          className="absolute inset-0 w-full h-full object-contain cursor-zoom-in"
        />
        {(curM.epoch != null || curM.step != null) && (
          <div className="absolute bottom-2.5 left-1/2 -translate-x-1/2 border border-subtle rounded-sm px-2.5 py-0.5 text-xs font-mono text-fg-secondary bg-surface/85">
            {curM.epoch != null && (
              <>ep <strong className="text-accent">{curM.epoch.toLocaleString()}</strong>{curM.step != null && ', '}</>
            )}
            {curM.step != null && (
              <>step <strong className="text-accent">{curM.step.toLocaleString()}</strong></>
            )}
            <span className="text-fg-tertiary ml-2">{active + 1} / {list.length}</span>
          </div>
        )}
      </div>

      {/* Click the main image to zoom (mirrors the download page's ImagePreviewModal); left/right cycle through the sample sequence */}
      {zoomOpen && (
        <ImagePreviewModal
          src={fullUrl}
          caption={[markText, filename].filter(Boolean).join(', ')}
          index={active}
          total={list.length}
          hasPrev={active > 0}
          hasNext={active < list.length - 1}
          onClose={() => setZoomOpen(false)}
          onPrev={() => setActive((i) => Math.max(0, i - 1))}
          onNext={() => setActive((i) => Math.min(list.length - 1, i + 1))}
          preload={[list[active - 1], list[active + 1]]
            .filter((x): x is { path: string; step?: number } => !!x)
            .map((x) => api.sampleImageUrl(x.path.split(/[\\/]/).pop() ?? x.path, taskId))}
        />
      )}
    </div>
  )
})

// ── Main Component ─────────────────────────────────────────────────────────

export default function MonitorDashboard({ taskId }: { taskId: number }) {
  const { state, connected } = useMonitorProgress(taskId)
  const [emaAlpha, setEmaAlpha] = useState(0.02)
  // LR / d don't get EMA by default (the data is already an EMA-derived quantity); the slider only smooths once pulled below 1
  const [lrAlpha, setLrAlpha] = useState(1)
  const [dAlpha] = useState(1)

  // Derived stats
  const losses = useMemo(() => state?.losses ?? [], [state?.losses])
  const lrHistory = useMemo(() => state?.lr_history ?? [], [state?.lr_history])
  const optimizerMetricsHistory = useMemo(
    () => state?.optimizer_metrics_history ?? [],
    [state?.optimizer_metrics_history],
  )
  const samples = useMemo(() => state?.samples ?? [], [state?.samples])
  const step = state?.step ?? 0
  const totalSteps = state?.total_steps ?? 0
  const speed = state?.speed ?? 0
  const eta = speed > 0 && totalSteps > step ? fmtSec((totalSteps - step) / speed) : '--'
  const progress = totalSteps > 0 ? Math.min(100, (step / totalSteps) * 100) : 0
  const elapsed = state?.start_time ? fmtSec(Date.now() / 1000 - state.start_time) : '--'

  // Recent loss vs previous (windowed comparison)
  const lossInfo = useMemo(() => {
    if (!losses.length) return null
    const WINDOW = Math.min(50, Math.floor(losses.length / 3)) || losses.length
    const raw = losses.map((l) => l.loss)
    const recent = raw.slice(-WINDOW)
    const prev = raw.length > WINDOW ? raw.slice(-WINDOW * 2, -WINDOW) : null
    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length
    if (!prev || prev.length === 0) return { val: recentAvg, delta: null }
    const prevAvg = prev.reduce((a, b) => a + b, 0) / prev.length
    return { val: recentAvg, delta: recentAvg - prevAvg }
  }, [losses])

  // Average loss (raw)
  const avgLoss = useMemo(() => {
    if (!losses.length) return null
    const raw = losses.map((l) => l.loss)
    return raw.reduce((a, b) => a + b, 0) / raw.length
  }, [losses])

  // Current LR
  const lastLr = lrHistory.length ? lrHistory[lrHistory.length - 1].lr : null
  const lastOptimizerMetrics = optimizerMetricsHistory.length
    ? optimizerMetricsHistory[optimizerMetricsHistory.length - 1]
    : null
  const lastD = lastOptimizerMetrics?.d ?? null
  const fmtLr = (v: number | null) => {
    if (v === null) return '--'
    if (v < 0.0001) return v.toExponential(1)
    return v.toFixed(5).replace(/0+$/, '').replace(/\.$/, '')
  }
  const fmtMetric = (v: number | null) => {
    if (v === null) return '--'
    if (Math.abs(v) < 0.0001 || Math.abs(v) >= 10000) return v.toExponential(2)
    return v.toFixed(5).replace(/0+$/, '').replace(/\.$/, '')
  }

  const vram = state?.vram_used_gb
  const vramTotal = state?.vram_total_gb
  const vramTone = vram && vramTotal ? (vram / vramTotal > 0.85 ? 'warn' : 'ok') as 'ok' | 'warn' : undefined

  // Full raw series (no more slice(-60)) -- SeriesChart downsamples evenly to 600 points internally when rendering.
  // Memoized: the arrays run up to 50k points and the dashboard re-renders on every SSE delta, so the
  // charts only rebuild when their own series actually changed.
  const lossSeries = useMemo(() => losses.map((l) => ({ step: l.step, value: l.loss })), [losses])
  const lrSeries = useMemo(() => lrHistory.map((l) => ({ step: l.step, value: l.lr })), [lrHistory])
  const dSeries = useMemo(() => optimizerMetricsHistory
    .filter((m) => typeof m.d === 'number')
    .map((m) => ({ step: m.step, value: m.d as number })), [optimizerMetricsHistory])

  if (!state && !connected) {
    return (
      <div className="grid place-items-center h-[200px] text-fg-tertiary text-sm">
        Waiting for training data…
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3.5 p-4 h-full overflow-y-auto m-page-scroll m-p-sm">
      {/* Connection status + progress */}
      <div className="flex items-center gap-2.5 text-xs text-fg-tertiary font-mono shrink-0 m-wrap monitor-status-row">
        <span className={`w-[7px] h-[7px] rounded-full inline-block shrink-0 ${connected ? 'bg-ok animate-pulse' : 'bg-err'}`} />
        {connected ? 'Live' : 'Disconnected'}
        {totalSteps > 0 && (
          <>
            <span>{step.toLocaleString()} / {totalSteps.toLocaleString()} steps</span>
            <span>{progress.toFixed(1)}%</span>
            <div className="flex-1 h-1 bg-overlay rounded overflow-hidden monitor-progress">
              <div
                className="h-full bg-accent rounded"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span>Elapsed {elapsed}</span>
            {eta !== '--' && (
              <>
                <span>ETA {eta}</span>
              </>
            )}
          </>
        )}
        <span className="flex-1 m-hide" />
        <a
          href={`/tools/monitor?task=${taskId}`}
          target="_blank"
          rel="noopener"
          className="text-fg-tertiary no-underline hover:text-fg-primary transition-colors m-hide"
        >
          Standalone monitor ↗
        </a>
      </div>

      {/* 6 stat cards */}
      <div className="grid grid-cols-6 gap-2.5 m-grid-3 m-gap-sm">
        <StatCard label="step" value={step ? step.toLocaleString() : '--'}
          sub={totalSteps ? `of ${totalSteps.toLocaleString()}` : undefined} tone="accent" />
        <StatCard
          label="loss"
          value={lossInfo ? lossInfo.val.toFixed(4) : '--'}
          sub={lossInfo?.delta != null
            ? `recent avg, ${lossInfo.delta > 0 ? '↑' : '↓'}${Math.abs(lossInfo.delta).toFixed(4)}`
            : losses.length > 0 ? 'recent avg' : 'awaiting'}
          tone={lossInfo?.delta != null ? (lossInfo.delta < 0 ? 'ok' : 'warn') : undefined}
        />
        <StatCard label="avg loss" value={avgLoss != null ? avgLoss.toFixed(4) : '--'}
          sub={losses.length ? `${losses.length} pts raw mean` : 'awaiting'} />
        <StatCard label="lr" value={fmtLr(lastLr)}
          sub={lastD != null ? `actual, d ${fmtMetric(lastD)}` : lrHistory.length ? 'learning rate' : undefined} />
        <StatCard
          label={vram ? 'vram' : 'speed'}
          value={vram ? `${vram.toFixed(1)} GB` : speed ? `${speed.toFixed(2)} it/s` : '--'}
          sub={vramTotal ? `of ${vramTotal.toFixed(0)} GB, ${((vram! / vramTotal) * 100).toFixed(0)}%` : undefined}
          tone={vramTone}
        />
        <StatCard label="eta" value={eta} sub={speed ? `${speed.toFixed(2)} it/s` : undefined} />
      </div>

      {/* Left: sample image (vertical) / Right: loss -> LR
          gridTemplateRows: '1fr' -> the row fills via flex-1, avoiding the default auto row leaving blank space on large screens;
          the right card's minHeight forms a floor, flex-1 expands evenly once the row height exceeds 3*min+gap;
          when the total min exceeds the viewport, the outer overflow-y-auto scrolls it */}
      <div
        className="grid grid-cols-1 lg:grid-cols-[1.5fr_1fr] gap-3.5 flex-1 m-grid-1 m-page-scroll"
        style={{ gridTemplateRows: '1fr' }}
      >
        {/* Left: sample image */}
        <div className="card p-0 overflow-hidden flex flex-col min-h-0">
          <div className="px-3.5 py-2.5 border-b border-subtle flex items-center justify-between shrink-0">
            <span className="text-sm font-semibold">Samples</span>
            <span className="text-xs text-fg-tertiary font-mono">{samples.length} imgs</span>
          </div>
          <div className="flex-1 p-3 flex flex-col min-h-0">
            <SampleViewer samples={samples} taskId={taskId} />
          </div>
        </div>

            {/* Right: loss / lr / d three cards (d optional), flex-1 splits height evenly but clamped to [140, 300].
            Same structure per card: a single-line header + a chart filling flex-1. LR no longer carries any d info
            (avoids the old problem where the d-block, as a shrink-0 dead-weight cost inside LR, pushed up LR card's min).
            minHeight 140 = readable floor (a smaller chart is hard to read and would trigger the outer scrollbar instead of shrinking further);
            maxHeight 300 = prevents cards from being stretched to an unbalanced height on 4K/large screens (leaves the remaining space for the left column's samples). */}
        <div className="flex flex-col gap-3.5 min-h-0">
          <div className="card p-4 flex-1 flex flex-col monitor-chart-card" style={{ minHeight: 140, maxHeight: 300 }}>
            <div className="flex items-center justify-between mb-2 shrink-0">
              <span className="text-sm font-semibold">loss</span>
              <SmoothControl alpha={emaAlpha} setAlpha={setEmaAlpha} min={0.001} max={0.3} step={0.001} />
            </div>
            <SeriesChart
              data={lossSeries}
              rawColor="color-mix(in srgb, var(--fg-secondary) 35%, transparent)"
              smoothColor="var(--accent)"
              fillColor="var(--accent-soft)"
              emaAlpha={emaAlpha}
              yFormat={(v) => v.toFixed(4)}
              minHeight={60}
            />
          </div>

          <div className="card p-4 flex-1 flex flex-col monitor-chart-card" style={{ minHeight: 140, maxHeight: 300 }}>
            <div className="flex items-center justify-between mb-2 shrink-0">
              <span className="text-sm font-semibold">learning rate</span>
              <SmoothControl alpha={lrAlpha} setAlpha={setLrAlpha} min={0.005} max={1} step={0.005} />
            </div>
            <SeriesChart
              data={lrSeries}
              rawColor="color-mix(in srgb, var(--warn) 35%, transparent)"
              smoothColor="var(--warn)"
              emaAlpha={lrAlpha}
              yFormat={fmtLr}
              minHeight={60}
            />
          </div>

          {/* d value (Prodigy / D-Adaptation etc.) -- only when the optimizer reports it. */}
          {dSeries.length > 0 && (
            <div className="card p-4 flex-1 flex flex-col monitor-chart-card" style={{ minHeight: 140, maxHeight: 300 }}>
              <div className="flex items-center justify-between mb-2 shrink-0">
                <span className="text-sm font-semibold">d</span>
              </div>
              <SeriesChart
                data={dSeries}
                rawColor="color-mix(in srgb, var(--accent) 30%, transparent)"
                smoothColor="var(--accent)"
                emaAlpha={dAlpha}
                yFormat={fmtMetric}
                minHeight={60}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
