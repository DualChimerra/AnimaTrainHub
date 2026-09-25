import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import {
  api,
  type EvalMetricResult,
  type EvalMetricsListResponse,
  type LoraCkpt,
  type Task,
  type TaskSample,
} from '../../api/client'
import FieldLabel from '../../components/ds/FieldLabel'
import KebabMenu from '../../components/ds/KebabMenu'
import PageHead from '../../components/ds/PageHead'
import {
  EVAL_METRIC_KEYS, checkpointLabel, checkpointSortValue, evalRowStatus, metricState, metricValue,
  type EvalMetricKey,
} from '../../components/MonitorDashboard'
import { useToast } from '../../components/Toast'
import { evalProgressFromResults } from '../../lib/useEvalProgress'

const METRIC_CODE: Record<EvalMetricKey, string> = {
  clip_t: 'CLIP-T', clip_i: 'CLIP-I', dino_i: 'DINO-I', ccip_i: 'CCIP-I', tag_recall: 'TAG RECALL',
}
/** Columns of the checkpoints table (the mockup shows the three core metrics). */
const TABLE_KEYS: EvalMetricKey[] = ['clip_t', 'clip_i', 'dino_i']
/** The best checkpoint is picked by DINO-I (does the LoRA really learn the style). */
const BEST_KEY: EvalMetricKey = 'dino_i'

const fmt3 = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(3))
const fmtDelta = (d: number) => `${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(3)}`

export default function MonitorPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const [tasks, setTasks] = useState<Task[]>([])
  // `?task=N` deep link pins the page to that task (bookmarks / links).
  const initialTaskId = useMemo<number | null>(() => {
    if (typeof window === 'undefined') return null
    const raw = new URLSearchParams(window.location.search).get('task')
    const n = raw === null ? NaN : Number(raw)
    return Number.isFinite(n) && n > 0 ? n : null
  }, [])
  const [taskId, setTaskId] = useState<number | null>(initialTaskId)

  useEffect(() => {
    api.listQueue().then(setTasks).catch(() => setTasks([]))
  }, [])

  const defaultTaskId = useMemo<number | null>(() => {
    const running = tasks.find((x) => x.status === 'running')
    if (running) return running.id
    const ended = [...tasks]
      .filter((x) => x.finished_at)
      .sort((a, b) => (b.finished_at ?? 0) - (a.finished_at ?? 0))[0]
    return ended?.id ?? null
  }, [tasks])

  useEffect(() => {
    if (taskId === null && defaultTaskId !== null) setTaskId(defaultTaskId)
  }, [defaultTaskId, taskId])

  const task = tasks.find((x) => x.id === taskId) ?? null
  const pid = task?.project_id ?? null
  const vid = task?.version_id ?? null

  // ── eval results ──
  const [payload, setPayload] = useState<EvalMetricsListResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const load = useCallback(async () => {
    if (!pid || !vid || !taskId) return
    try {
      setPayload(await api.listEvalMetrics(pid, vid, taskId))
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [pid, vid, taskId])
  useEffect(() => { setPayload(null); setError(null); void load() }, [load])

  const results = useMemo(
    () => [...(payload?.results ?? [])]
      .filter((r) => !r.baseline)
      .sort((a, b) => checkpointSortValue(a, 0) - checkpointSortValue(b, 0)),
    [payload?.results],
  )
  const progress = useMemo(() => evalProgressFromResults(results), [results])
  const live = progress.active || task?.status === 'running'
  useEffect(() => {
    if (!live) return
    const id = window.setInterval(() => void load(), 5000)
    return () => window.clearInterval(id)
  }, [live, load])

  const best = useMemo(() => {
    let out: { result: EvalMetricResult; index: number; value: number } | null = null
    results.forEach((r, index) => {
      const v = metricValue(r, BEST_KEY)
      if (v != null && (!out || v > out.value)) out = { result: r, index, value: v }
    })
    return out as { result: EvalMetricResult; index: number; value: number } | null
  }, [results])
  // KPI cards describe the best checkpoint; before any DINO-I, the latest measured one.
  const kpiSource = best?.result
    ?? [...results].reverse().find((r) => EVAL_METRIC_KEYS.some((k) => metricValue(r, k) != null))
    ?? null

  // ── checkpoint selection + manual eval ──
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [pickerOpen, setPickerOpen] = useState(false)
  const [ckpts, setCkpts] = useState<LoraCkpt[] | null>(null)
  const [running, setRunning] = useState(false)
  useEffect(() => { setSelected(new Set()); setPickerOpen(false); setCkpts(null) }, [taskId])
  const toggle = (path: string) => setSelected((prev) => {
    const next = new Set(prev)
    if (next.has(path)) next.delete(path); else next.add(path)
    return next
  })
  const openPicker = () => {
    setPickerOpen((v) => !v)
    if (ckpts === null && pid && vid) {
      api.listVersionLoraCkpts(pid, vid).then(setCkpts).catch((e) => { setCkpts([]); toast(String(e), 'error') })
    }
  }
  const runEval = async () => {
    if (!pid || !vid || !taskId || selected.size === 0) return
    setRunning(true)
    try {
      const r = await api.runTaskEval(pid, vid, { task_id: taskId, checkpoints: [...selected] })
      toast(t('monitor.queuedEvals', { count: r.queued }), 'success')
      setSelected(new Set())
      setPickerOpen(false)
      void load()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setRunning(false)
    }
  }

  const [view, setView] = useState<'table' | 'chart'>('table')
  const statusText = (s: string) => t(`status.${s === 'pending' ? 'queued' : s}`)

  return (
    <div className="fade-in" style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <PageHead
        eyebrow={t('monitor.eyebrow')}
        title={t('monitor.title')}
        subtitle={t('monitor.subtitle')}
        tools={
          <>
            <select
              className="ds-inp"
              style={{ width: 260 }}
              value={taskId ?? ''}
              onChange={(e) => setTaskId(e.target.value === '' ? null : Number(e.target.value))}
              aria-label={t('monitor.taskLabel')}
            >
              {taskId === null && <option value="">{t('monitor.latestHint')}</option>}
              {tasks.map((x) => (
                <option key={x.id} value={x.id}>#{x.id} · {x.name} · {statusText(x.status)}</option>
              ))}
            </select>
            <button type="button" className={`ds-ctl${pickerOpen ? ' ds-is-on' : ''}`} onClick={openPicker} disabled={!pid || !vid}>
              {t('monitor.pickCkptsBtn')}
            </button>
            <button type="button" className="ds-btn-primary" onClick={() => void runEval()} disabled={running || selected.size === 0}>
              {running ? t('monitor.queueing') : selected.size ? `${t('monitor.runEval')} (${selected.size})` : t('monitor.runEval')}
            </button>
            <KebabMenu
              trigger="icon"
              label={t('monitor.moreActions')}
              items={[
                { label: t('monitor.openLive'), onSelect: () => taskId && navigate(`/queue/${taskId}#monitor`), disabled: !taskId },
                { label: t('common.refresh'), onSelect: () => void load(), disabled: !pid },
              ]}
            />
          </>
        }
      />

      <div className="ds-scroll" style={{ minHeight: 0, overflowY: 'auto' }}>
        {!task ? (
          <div className="ds-card"><div className="ds-empty">{tasks.length ? t('monitor.pickTask') : t('monitor.noTasks')}</div></div>
        ) : !pid || !vid ? (
          <div className="ds-card"><div className="ds-empty">{t('monitor.noVersionBound')}</div></div>
        ) : (
          <>
            {progress.active && (
              <div className="ds-note ds-info">
                <span>{t('monitor.progressNote')} <b>{progress.done} / {progress.total}</b>.</span>
              </div>
            )}
            {error && <div className="ds-note ds-warn ds-mono">{t('monitor.evalLoadFailed', { error })}</div>}

            {pickerOpen && (
              <div className="ds-card">
                <div className="ds-card-head ds-pad">
                  <div style={{ flex: 1 }}>
                    <div className="ds-card-title">{t('monitor.pickCkpts')}</div>
                    <div className="ds-card-sub">{t('monitor.pickCkptsHint')}</div>
                  </div>
                  {ckpts && ckpts.length > 0 && (
                    <div className="ds-card-tools">
                      <button
                        type="button"
                        className="ds-ctl ds-ghost"
                        onClick={() => setSelected((prev) => (prev.size === ckpts.length ? new Set() : new Set(ckpts.map((c) => c.path))))}
                      >
                        {selected.size === ckpts.length ? t('monitor.clearAll') : t('monitor.selectAll')}
                      </button>
                    </div>
                  )}
                </div>
                <div className="ds-card-body" style={{ paddingTop: 0, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {ckpts === null ? (
                    <span className="ds-cell-key">{t('monitor.loadingCkpts')}</span>
                  ) : ckpts.length === 0 ? (
                    <span className="ds-cell-key">{t('monitor.noCkpts')}</span>
                  ) : ckpts.map((c) => (
                    <button
                      key={c.path}
                      type="button"
                      className={`ds-chip${selected.has(c.path) ? ' ds-is-active' : ''}`}
                      aria-pressed={selected.has(c.path)}
                      onClick={() => toggle(c.path)}
                      title={c.path}
                    >
                      <span className="ds-mono">{c.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="ds-kpis" style={{ gridTemplateColumns: 'repeat(5,minmax(0,1fr))' }}>
              {EVAL_METRIC_KEYS.map((k) => {
                const v = kpiSource ? metricValue(kpiSource, k) : null
                const st = kpiSource ? metricState(kpiSource, k)?.status : undefined
                const d = kpiSource?.delta?.[k]
                return (
                  <div key={k} className="ds-card ds-kpi" style={{ padding: '13px 15px 14px' }}>
                    <div className="ds-kpi-top">
                      <span className="ds-cap"><FieldLabel label={METRIC_CODE[k]} tip={t(`monitor.evalDesc.${k}`)} /></span>
                      <span style={{ marginLeft: 'auto' }}>
                        {d != null && v != null
                          ? <span className={`ds-delta ${d >= 0 ? 'ds-up' : 'ds-down'}`}>{fmtDelta(d)}</span>
                          : v == null && <span className="ds-kpi-meta">{st === 'unavailable' ? t('monitor.kpiNa') : st === 'pending' || st === 'running' ? t('monitor.kpiRunning') : t('monitor.kpiNotRun')}</span>}
                      </span>
                    </div>
                    <div className="ds-kpi-val" style={{ marginTop: 12, fontSize: 23 }}>{fmt3(v)}</div>
                    <div className="ds-kpi-label">{t(`monitor.kpi.${k}`)}</div>
                  </div>
                )
              })}
            </div>

            <div className="ds-monitor-grid">
              <div className="ds-card">
                <div className="ds-card-head ds-pad">
                  <div style={{ flex: 1 }}>
                    <div className="ds-card-title">{t('monitor.ckptsTitle')}</div>
                    <div className="ds-card-sub">{t('monitor.ckptsSub', { count: results.length })}</div>
                  </div>
                  <div className="ds-card-tools">
                    <div className="ds-seg">
                      <button type="button" className={`ds-seg-item${view === 'table' ? ' ds-is-active' : ''}`} onClick={() => setView('table')}>{t('monitor.viewTable')}</button>
                      <button type="button" className={`ds-seg-item${view === 'chart' ? ' ds-is-active' : ''}`} onClick={() => setView('chart')}>{t('monitor.viewChart')}</button>
                    </div>
                  </div>
                </div>
                {results.length === 0 ? (
                  <div className="ds-empty">{t('monitor.noEvalResults')}</div>
                ) : view === 'table' ? (
                  <table className="ds-tbl">
                    <thead>
                      <tr>
                        <th style={{ width: 32 }} />
                        <th>{t('monitor.ckptCol')}</th>
                        {TABLE_KEYS.map((k) => <th key={k}>{METRIC_CODE[k]}</th>)}
                        <th>{t('monitor.deltaCol')}</th>
                        <th>{t('monitor.statusCol')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((r) => {
                        const isBest = best?.result === r
                        const path = r.checkpoint?.path
                        const on = !!path && selected.has(path)
                        const rs = evalRowStatus(r)
                        const d = r.delta?.[BEST_KEY]
                        return (
                          <tr key={r.run_id} style={isBest ? { background: 'var(--green-soft)' } : undefined}>
                            <td>
                              {path && (
                                <button
                                  type="button"
                                  role="checkbox"
                                  aria-checked={on}
                                  aria-label={checkpointLabel(r)}
                                  className={`ds-cbox${on ? ' ds-on' : ''}`}
                                  onClick={() => toggle(path)}
                                >
                                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round"><path d="m5 12 5 5 9-10" /></svg>
                                </button>
                              )}
                            </td>
                            <td>
                              <span className="ds-mono" style={isBest ? { fontWeight: 600 } : undefined}>{checkpointLabel(r)}</span>
                              {isBest && <> <span className="ds-badge ds-ok">{t('monitor.bestBadge')}</span></>}
                            </td>
                            {TABLE_KEYS.map((k) => {
                              const v = metricValue(r, k)
                              return <td key={k} className={`ds-num${v == null ? ' ds-muted' : ''}`} style={isBest && k === BEST_KEY ? { fontWeight: 600 } : undefined}>{fmt3(v)}</td>
                            })}
                            <td className={`ds-num${isBest ? '' : ' ds-muted'}`} style={isBest ? { color: 'var(--green-text)', fontWeight: 600 } : undefined}>
                              {d != null ? fmtDelta(d) : '—'}
                            </td>
                            <td>
                              <span className={`ds-badge ${rs.tone === 'warn' ? 'ds-warn' : rs.tone === 'err' ? 'ds-err' : isBest ? 'ds-ok' : 'ds-mute'}`}>
                                {rs.key ? t(rs.key, rs.params) : rs.text}
                              </span>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                ) : (
                  <div className="ds-card-body" style={{ paddingTop: 0, display: 'flex', flexDirection: 'column', gap: 14 }}>
                    {EVAL_METRIC_KEYS.filter((k) => results.some((r) => metricValue(r, k) != null)).map((k) => (
                      <MetricLine key={k} label={METRIC_CODE[k]} results={results} metric={k} bestIndex={k === BEST_KEY ? best?.index ?? null : null} />
                    ))}
                  </div>
                )}
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div className="ds-card">
                  <div className="ds-card-head ds-pad">
                    <div>
                      <div className="ds-card-title">{t('monitor.dynamicsTitle')}</div>
                      <div className="ds-card-sub">
                        {best ? t('monitor.dynamicsSub', { ckpt: checkpointLabel(best.result) }) : t('monitor.waitingMetrics')}
                      </div>
                    </div>
                  </div>
                  <div className="ds-card-body" style={{ paddingTop: 4 }}>
                    <MetricLine results={results} metric={BEST_KEY} bestIndex={best?.index ?? null} height={130} />
                    <OvercookNote results={results} best={best} />
                  </div>
                </div>

                {best && task && <BestSamples task={task} result={best.result} />}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

/** One metric over the checkpoints; the best point is ringed. */
function MetricLine({ label, results, metric, bestIndex, height = 90 }: {
  label?: string
  results: EvalMetricResult[]
  metric: EvalMetricKey
  bestIndex: number | null
  height?: number
}) {
  const pts = results
    .map((r, i) => ({ i, v: metricValue(r, metric), label: checkpointLabel(r) }))
    .filter((p): p is { i: number; v: number; label: string } => p.v != null)
  const W = 330
  const H = height
  if (pts.length === 0) return null
  const vs = pts.map((p) => p.v)
  const lo = Math.min(...vs)
  const hi = Math.max(...vs)
  const pad = (hi - lo) * 0.15 || 0.01
  const n = Math.max(1, results.length - 1)
  const x = (i: number) => 10 + (i / n) * (W - 20)
  const y = (v: number) => H - 14 - ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (H - 28)
  const path = pts.map((p, k) => `${k ? 'L' : 'M'}${x(p.i).toFixed(1)} ${y(p.v).toFixed(1)}`).join(' ')
  const bestPt = bestIndex != null ? pts.find((p) => p.i === bestIndex) : undefined
  const grid = [0.2, 0.45, 0.7, 0.95].map((f) => `M0 ${(H * f).toFixed(0)} L${W} ${(H * f).toFixed(0)}`).join(' ')
  return (
    <div>
      {label && <div className="ds-cap" style={{ marginBottom: 4 }}>{label}</div>}
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: '100%', height: H, display: 'block' }} aria-hidden="true">
        <path d={grid} stroke="#ececea" strokeWidth="1" />
        <path d={path} fill="none" stroke="#6e8f42" strokeWidth="1.8" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
        {pts.map((p) => <circle key={p.i} cx={x(p.i)} cy={y(p.v)} r="2.2" fill="#6e8f42" />)}
        {bestPt && (
          <>
            <circle cx={x(bestPt.i)} cy={y(bestPt.v)} r="3.6" fill="#6e8f42" />
            <circle cx={x(bestPt.i)} cy={y(bestPt.v)} r="7" fill="none" stroke="#a3db52" strokeWidth="1.4" />
          </>
        )}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10.5, color: 'var(--ink-3)', fontFamily: 'var(--mono)', marginTop: 5 }}>
        {results.map((r) => <span key={r.run_id}>{r.checkpoint?.value ?? checkpointLabel(r)}</span>)}
      </div>
    </div>
  )
}

/** Warns when the best checkpoint is not the last measured one and the metric
 *  falls after it — the usual sign of overtraining. */
function OvercookNote({ results, best }: { results: EvalMetricResult[]; best: { index: number; value: number; result: EvalMetricResult } | null }) {
  const { t } = useTranslation()
  if (!best) return null
  const after = results.slice(best.index + 1).map((r) => metricValue(r, BEST_KEY)).filter((v): v is number => v != null)
  if (after.length === 0 || after[after.length - 1] >= best.value) return null
  return (
    <div className="ds-note ds-warn" style={{ marginTop: 12 }}>
      <span>{t('monitor.overcookNote', { ckpt: checkpointLabel(best.result) })}</span>
    </div>
  )
}

/** Training samples saved at the best checkpoint's epoch / step. */
function BestSamples({ task, result }: { task: Task; result: EvalMetricResult }) {
  const { t } = useTranslation()
  const [samples, setSamples] = useState<TaskSample[] | null>(null)
  useEffect(() => {
    let alive = true
    api.listTaskSamples(task.id).then((r) => { if (alive) setSamples(r.items) }).catch(() => { if (alive) setSamples([]) })
    return () => { alive = false }
  }, [task.id])
  const value = result.checkpoint?.value
  const byEpoch = result.checkpoint?.kind !== 'step'
  const picked = (samples ?? []).filter((s) => value != null && (byEpoch ? s.epoch === value : s.step === value)).slice(0, 3)
  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div>
          <div className="ds-card-title">{t('monitor.samplesTitle', { ckpt: checkpointLabel(result) })}</div>
          <div className="ds-card-sub">{t('monitor.samplesSub')}</div>
        </div>
      </div>
      <div className="ds-card-body" style={{ paddingTop: 2 }}>
        {samples === null ? (
          <div className="ds-cell-key">{t('common.loading')}</div>
        ) : picked.length === 0 ? (
          <div className="ds-cell-key">{t('monitor.samplesNone')}</div>
        ) : (
          <div className="ds-grid-thumbs" style={{ gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
            {picked.map((s) => (
              <span key={s.filename} className="ds-thumb" style={{ aspectRatio: '1' }}>
                <img src={api.sampleImageUrl(s.filename, task.id, 240)} alt="" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} loading="lazy" />
                <span className="ds-cap">{s.step != null ? `step ${s.step}` : s.filename}</span>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
