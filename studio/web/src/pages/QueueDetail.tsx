import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import {
  api,
  type EvalMetricsListResponse,
  type MonitorState,
  type Task,
  type TaskOutputs,
  type TaskSample,
} from '../api/client'
import { PauseProgressModal } from '../components/PauseProgressModal'
import ImagePreviewModal from '../components/ImagePreviewModal'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import KebabMenu, { type KebabItem } from '../components/ds/KebabMenu'
import PageHead from '../components/ds/PageHead'
import { useEventStream } from '../lib/useEventStream'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import { useQueueFormat, type QueueFormat } from '../lib/queueFormat'
import MonitorDashboard, {
  CORE_METRIC_KEYS, EVAL_METRIC_KEYS, checkpointLabel, checkpointSortValue, metricState, metricValue,
  type EvalMetricKey,
} from '../components/MonitorDashboard'
import { fmtParamValue, jobJumpPath, paramLabel } from './queue/jobUtils'

type Tab = 'overview' | 'log' | 'monitor' | 'eval' | 'outputs' | 'snapshot'
const ALL_TABS: readonly Tab[] = ['overview', 'log', 'monitor', 'eval', 'outputs', 'snapshot']
const TERMINAL = new Set(['done', 'failed', 'canceled'])

/** Tabs a task type has: training gets everything; test renders, reg AI and
 *  data jobs only have a log and their properties. */
function tabsFor(task: Task | null): Tab[] {
  const kind = task?.task_type ?? 'train'
  return kind === 'train' ? [...ALL_TABS] : ['overview', 'log']
}

function tabFromHash(hash: string): Tab | null {
  const v = hash.replace(/^#/, '')
  return (ALL_TABS as readonly string[]).includes(v) ? (v as Tab) : null
}

/** EMA over a series (the loss curve is noisy per step). */
function ema(values: number[], alpha = 0.02): number[] {
  const out: number[] = []
  let cur = values[0] ?? 0
  for (const v of values) { cur = alpha * v + (1 - alpha) * cur; out.push(cur) }
  return out
}

function tail(path: string | null | undefined, parts = 2): string {
  if (!path) return '—'
  const seg = path.split(/[\\/]/).filter(Boolean)
  return seg.length > parts ? `…/${seg.slice(-parts).join('/')}` : path
}

// ── Page ────────────────────────────────────────────────────────────────────
export default function QueueDetailPage() {
  const { id } = useParams<{ id: string }>()
  const taskId = Number(id)
  const { t } = useTranslation()
  const fmt = useQueueFormat()
  const navigate = useNavigate()
  const location = useLocation()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [task, setTask] = useState<Task | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [tab, setTab] = useState<Tab>(() => (typeof window === 'undefined' ? 'overview' : tabFromHash(window.location.hash) ?? 'overview'))
  const [pauseModalOpen, setPauseModalOpen] = useState(false)
  const [outputs, setOutputs] = useState<TaskOutputs | null>(null)

  // tab → hash 写回（点 tab 按钮时同步 URL，replaceState 不触发 router 重渲）
  useEffect(() => {
    if (typeof window === 'undefined') return
    const h = `#${tab}`
    if (window.location.hash !== h) window.history.replaceState(null, '', h)
  }, [tab])

  // hash → tab（在本页 navigate 到同一 task 换 hash 时切 tab，如「查看输出」）。
  useEffect(() => {
    const v = tabFromHash(location.hash)
    if (v) setTab((prev) => (prev === v ? prev : v))
  }, [location.hash])

  const reload = useCallback(async () => {
    if (!Number.isFinite(taskId)) return
    try { setTask(await api.getTask(taskId)); setError(null) }
    catch (e) { setError(String(e)) }
  }, [taskId])

  const reloadOutputs = useCallback(() => {
    if (!Number.isFinite(taskId)) return
    api.getTaskOutputs(taskId).then(setOutputs).catch(() => setOutputs(null))
  }, [taskId])

  useEffect(() => { void reload() }, [reload])

  const kind = task?.task_type ?? 'train'
  const isTrain = kind === 'train'
  useEffect(() => { if (task && isTrain) reloadOutputs() }, [task?.id, task?.status, isTrain, reloadOutputs]) // eslint-disable-line react-hooks/exhaustive-deps

  // ADR 0006 — auto_epoch_backup_written flips is_pausable (the first epoch
  // backup landed), so it reloads like a state change does.
  useEventStream((evt) => {
    if (evt.task_id !== taskId) return
    if (evt.type === 'task_state_changed' || evt.type === 'auto_epoch_backup_written') void reload()
  })

  useEffect(() => {
    if (task?.status !== 'running') return
    const tick = window.setInterval(() => setTask((x) => (x ? { ...x } : x)), 5000)
    return () => window.clearInterval(tick)
  }, [task?.status])

  const { state: monitor } = useMonitorProgress(isTrain && Number.isFinite(taskId) ? taskId : null)

  const tabs = tabsFor(task)
  // A deep link to a tab this task type does not have falls back to the overview.
  const activeTab: Tab = tabs.includes(tab) ? tab : 'overview'

  if (!Number.isFinite(taskId)) return <div className="ds-scroll"><div className="ds-note ds-warn">{t('queueDetail.invalidId')}</div></div>

  const status = task?.status
  const isTerminal = !!status && TERMINAL.has(status)
  const now = Date.now() / 1000

  const cancel = async () => {
    if (!task) return
    const ok = await confirm(t('queue.cancelRunningConfirm', { id: task.id }), { tone: 'warn', okText: t('queueDetail.cancelTask') })
    if (!ok) return
    setBusy(true)
    try { await api.cancelTask(task.id); toast(t('queueDetail.cancelSent'), 'success'); void reload() }
    catch (e) { toast(String(e), 'error') }
    finally { setBusy(false) }
  }

  const retry = async () => {
    if (!task) return
    setBusy(true)
    try {
      const next = await api.retryTask(task.id)
      toast(t('queueDetail.retryQueued', { id: next.id }), 'success')
      navigate(`/queue/${next.id}`)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const remove = async () => {
    if (!task) return
    const ok = await confirm(
      `${t('queueDetail.deleteDesc')} #${task.id} ${task.name}. ${t('queueDetail.deleteNote')}`,
      { title: t('queueDetail.deleteTitle'), tone: 'danger', okText: t('common.delete') },
    )
    if (!ok) return
    setBusy(true)
    try { await api.deleteTask(task.id); toast(t('queueDetail.deleted'), 'success'); navigate('/queue') }
    catch (e) { toast(String(e), 'error'); setBusy(false) }
  }

  // ADR 0006 PR-4: 暂停 / 恢复。
  const pauseRunning = async () => {
    if (!task) return
    setPauseModalOpen(true)
    try {
      await api.pauseTask(task.id)
      toast(t('queue.pauseSent'), 'success')
    } catch (e) {
      toast(t('queue.pauseFailed', { reason: String(e) }), 'error')
      setPauseModalOpen(false)
    }
  }

  const resume = async () => {
    if (!task) return
    setBusy(true)
    try {
      await api.resumeTask(task.id)
      toast(t('queue.resumeSent', { id: task.id }), 'success')
      void reload()
    } catch (e) {
      const msg = String(e)
      toast(msg.toLowerCase().includes('missing') ? t('queue.resumeFailedMissing') : t('queue.resumeFailed', { reason: msg }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const startNow = async () => {
    if (!task) return
    try { await api.startTaskNow(task.id); toast(t('queue.startNowSent', { id: task.id }), 'success'); void reload() }
    catch (e) { toast(String(e), 'error') }
  }

  // Render the newest LoRA of this task in the generator.
  const openRender = () => {
    const lora = outputs?.files.filter((f) => f.is_lora).sort((a, b) => b.mtime - a.mtime)[0]
    if (!task || !lora || !outputs?.output_dir) { toast(t('queue.noLora'), 'error'); return }
    const sp = new URLSearchParams({ lora: `${outputs.output_dir.replace(/[\\/]+$/, '')}/${lora.path}` })
    if (task.project_id) sp.set('projectId', String(task.project_id))
    if (task.version_id) sp.set('versionId', String(task.version_id))
    navigate(`/tools/generate?${sp.toString()}`)
  }

  const subtitle = (() => {
    if (!task) return null
    const bits: React.ReactNode[] = [<code key="c" className="ds-mono">{task.config_name}</code>]
    if (status === 'running' && task.started_at) {
      bits.push(t('queueDetail.startedAtShort', { time: fmt.clock(task.started_at) }), t('queueDetail.runningFor', { time: fmt.dur(now - task.started_at) }))
      if (task.pid) bits.push(`PID ${task.pid}`)
    } else if (status === 'pending' || status === 'scheduled') {
      bits.push(t('queueDetail.queuedAtShort', { time: fmt.clock(task.created_at) }))
      if (task.scheduled_at) bits.push(t('queue.scheduledAt', { time: fmt.clock(task.scheduled_at) }))
    } else if (status === 'paused') {
      bits.push(t('queue.pausedAtStep', { step: task.paused_step ?? 0, time: task.paused_at ? fmt.clock(task.paused_at) : '—' }))
    } else if (task.finished_at) {
      bits.push(t('queueDetail.finishedAtShort', { time: fmt.clock(task.finished_at) }))
      if (task.started_at) bits.push(t('queueDetail.lasted', { time: fmt.dur(task.finished_at - task.started_at) }))
    }
    return bits.map((b, i) => <span key={i}>{i > 0 && ' · '}{b}</span>)
  })()

  const primary = (() => {
    if (!task) return null
    if (status === 'running' && task.is_pausable) {
      return <button type="button" className="ds-ctl" onClick={() => void pauseRunning()} disabled={busy || pauseModalOpen} title={t('queue.pauseHint')} data-testid="detail-pause-btn">{t('queue.pause')}</button>
    }
    if (status === 'paused' || (isTerminal && task.is_resumable)) {
      return <button type="button" className="ds-ctl" onClick={() => void resume()} disabled={busy} title={t('queue.resumeHint')} data-testid="detail-resume-btn">{t('queue.resume')}</button>
    }
    if (status === 'scheduled') return <button type="button" className="ds-ctl" onClick={() => void startNow()}>{t('queue.startNow')}</button>
    if (isTerminal) return <button type="button" className="ds-ctl" onClick={() => void retry()} disabled={busy}>{t('queue.retry')}</button>
    return null
  })()

  const jump = task ? jobJumpPath(task) : null
  const deepLink = (() => {
    if (!task) return null
    if (kind === 'generate') {
      return <Link className="ds-ctl" to={`/tools/generate?task=${task.id}`} data-testid="detail-view-generate">{t('queueDetail.viewInGenerate')}</Link>
    }
    if ((kind === 'reg_ai') && task.project_id && task.version_id) {
      return <Link className="ds-ctl" to={`/projects/${task.project_id}/v/${task.version_id}/reg`} data-testid="detail-view-reg">{t('queueDetail.viewInReg')}</Link>
    }
    if (jump) return <Link className="ds-ctl" to={jump}>{t('queue.jobs.jump')}</Link>
    return null
  })()

  const menuItems: KebabItem[] = task ? [
    ...(task.project_id && task.version_id && isTrain
      ? [{ label: t('queue.openConfig'), onSelect: () => navigate(`/projects/${task.project_id}/v/${task.version_id}/train`) }]
      : []),
    ...(isTerminal && task.is_resumable ? [{ label: t('queue.retry'), onSelect: () => void retry() }] : []),
    ...(!isTerminal ? [{ label: t('queueDetail.cancelTask'), tone: 'err' as const, onSelect: () => void cancel(), disabled: busy }] : []),
    ...(isTerminal ? [{ label: t('queueDetail.deleteRecord'), tone: 'err' as const, onSelect: () => void remove(), disabled: busy }] : []),
  ] : []

  const loraCount = outputs?.files.filter((f) => f.is_lora).length ?? 0
  const tabLabel: Record<Tab, string> = {
    overview: t('queueDetail.tabOverview'),
    log: t('queueDetail.tabLogs'),
    monitor: t('queueDetail.tabMonitor'),
    eval: t('queueDetail.tabEval'),
    outputs: t('queueDetail.tabOutputs'),
    snapshot: t('queueDetail.tabSnapshot'),
  }

  return (
    <div className="fade-in">
      <PageHead
        eyebrow={t('queueDetail.eyebrow', { kind: String(kind).toUpperCase() })}
        title={task ? `#${task.id} · ${task.name}` : `#${taskId}`}
        subtitle={subtitle}
        tools={
          <>
            <Link className="ds-ctl" to="/queue">{t('queueDetail.backToQueue')}</Link>
            {deepLink}
            {primary}
            {menuItems.length > 0 && <KebabMenu trigger="icon" label={t('queue.moreActions')} items={menuItems} />}
          </>
        }
      />

      <div className="ds-scroll">
        {error && <div className="ds-note ds-warn font-mono">{error}</div>}

        <div className="ds-card">
          <div className="ds-tabs" role="tablist">
            {tabs.map((key) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={activeTab === key}
                className={`ds-tab${activeTab === key ? ' ds-is-active' : ''}`}
                onClick={() => setTab(key)}
              >
                {tabLabel[key]}
                {key === 'outputs' && loraCount > 0 && <span className="ds-badge ds-mute">{loraCount}</span>}
              </button>
            ))}
          </div>
          {task && activeTab === 'eval'
            ? <EvalStats task={task} />
            : task && <TaskStats task={task} monitor={isTrain ? monitor : null} fmt={fmt} />}
        </div>

        {!task ? (
          <div className="ds-empty">{t('common.loading')}</div>
        ) : activeTab === 'overview' ? (
          <OverviewTab
            task={task}
            monitor={isTrain ? monitor : null}
            outputs={isTrain ? outputs : null}
            fmt={fmt}
            onSaved={setTask}
            onRender={openRender}
            onOpenLog={() => setTab('log')}
          />
        ) : activeTab === 'log' ? (
          <LogTab taskId={taskId} />
        ) : activeTab === 'monitor' ? (
          <div className="ds-card" style={{ overflow: 'hidden' }}><MonitorDashboard taskId={taskId} /></div>
        ) : activeTab === 'eval' ? (
          <EvalTab task={task} />
        ) : activeTab === 'outputs' ? (
          <OutputsTab taskId={taskId} onChanged={reloadOutputs} />
        ) : (
          <SnapshotConfigTab task={task} />
        )}
      </div>

      {/* ADR §4.3 暂停过程 modal — 跟 Queue.tsx 同组件，UI 锁屏让用户看进度。 */}
      {pauseModalOpen && task && (
        <PauseProgressModal taskId={task.id} taskName={task.name} onClose={() => setPauseModalOpen(false)} />
      )}
    </div>
  )
}

// ── Stat strips under the tabs ──────────────────────────────────────────────

function StatCell({ label, value, unit, sub, accent, children }: {
  label: string
  value: React.ReactNode
  unit?: string
  sub?: React.ReactNode
  accent?: boolean
  children?: React.ReactNode
}) {
  return (
    <div>
      <div className="ds-cap">{label}</div>
      <div className="ds-stat-v" style={accent ? { color: 'var(--green-text)' } : undefined}>
        {value}{unit && <span style={{ fontSize: 12, fontWeight: 400, color: 'var(--ink-3)' }}> {unit}</span>}
      </div>
      {children}
      {sub != null && <div className="ds-cell-key" style={{ marginTop: 6 }}>{sub}</div>}
    </div>
  )
}

function TaskStats({ task, monitor, fmt }: { task: Task; monitor: MonitorState | null; fmt: QueueFormat }) {
  const { t } = useTranslation()
  const now = Date.now() / 1000
  const s = task.status

  if (!monitor || !monitor.total_steps) {
    // Non-training tasks (and training ones without a monitor file yet).
    const statusLabel = t(`status.${s === 'pending' ? 'queued' : s}`)
    return (
      <div className="ds-statgrid" style={{ gridTemplateColumns: 'repeat(4,minmax(0,1fr))' }}>
        <StatCell label={t('common.status')} value={statusLabel} accent={s === 'running'} />
        <StatCell label={t('queueDetail.duration')} value={task.started_at ? fmt.dur((task.finished_at ?? now) - task.started_at) : '—'} />
        <StatCell label={t('queueDetail.startedAt')} value={task.started_at ? fmt.clock(task.started_at) : '—'} sub={t('queueDetail.queuedAtShort', { time: fmt.clock(task.created_at) })} />
        <StatCell label={t('queueDetail.exitCode')} value={task.exit_code ?? '—'} sub={task.pid ? `PID ${task.pid}` : undefined} />
      </div>
    )
  }

  const step = monitor.step ?? 0
  const total = monitor.total_steps
  const pct = Math.min(100, (step / total) * 100)
  const losses = monitor.losses ?? []
  const smooth = ema(losses.map((l) => l.loss))
  const lastEma = smooth.length ? smooth[smooth.length - 1] : null
  let delta: number | null = null
  if (losses.length > 1) {
    const target = losses[losses.length - 1].step - 200
    let i = losses.length - 1
    while (i > 0 && losses[i].step > target) i--
    if (i < losses.length - 1 && smooth[i]) delta = ((smooth[smooth.length - 1] - smooth[i]) / smooth[i]) * 100
  }
  const remaining = s === 'running' && monitor.speed && monitor.speed > 0 ? (total - step) / monitor.speed : null

  return (
    <div className="ds-statgrid" style={{ gridTemplateColumns: 'repeat(4,minmax(0,1fr))' }}>
      <StatCell label={t('queueDetail.progress')} value={step.toLocaleString()} unit={`/ ${total.toLocaleString()}`}>
        <span className="ds-meter" style={{ marginTop: 9 }}><i style={{ width: `${pct}%` }} /></span>
      </StatCell>
      <StatCell
        label={t('queueDetail.speed')}
        value={monitor.speed ? monitor.speed.toFixed(2) : '—'}
        unit="it/s"
        sub={monitor.epoch != null && monitor.total_epochs ? t('queue.epochOf', { n: monitor.epoch, total: monitor.total_epochs }) : undefined}
      />
      <StatCell
        label={t('queueDetail.lossEma')}
        value={lastEma != null ? lastEma.toFixed(4) : '—'}
        accent
        sub={delta != null ? t('queueDetail.lossDelta', { pct: `${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(0)}` }) : undefined}
      />
      {remaining != null ? (
        <StatCell label={t('queueDetail.remaining')} value={`~${fmt.dur(remaining)}`} sub={t('queueDetail.finishAt', { time: fmt.clock(now + remaining) })} />
      ) : (
        <StatCell
          label={t('queueDetail.duration')}
          value={task.started_at ? fmt.dur((task.finished_at ?? now) - task.started_at) : '—'}
          sub={task.finished_at ? t('queueDetail.finishedAtShort', { time: fmt.clock(task.finished_at) }) : t(`status.${s === 'pending' ? 'queued' : s}`)}
        />
      )}
    </div>
  )
}

// ── Overview ────────────────────────────────────────────────────────────────

function OverviewTab({ task, monitor, outputs, fmt, onSaved, onRender, onOpenLog }: {
  task: Task
  monitor: MonitorState | null
  outputs: TaskOutputs | null
  fmt: QueueFormat
  onSaved: (t: Task) => void
  onRender: () => void
  onOpenLog: () => void
}) {
  const isTrain = (task.task_type ?? 'train') === 'train'
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 380px', gap: 14, alignItems: 'start' }} className="ds-qd-grid">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minWidth: 0 }}>
        {isTrain && <CurvesCard task={task} monitor={monitor} />}
        {!isTrain && task.params_decoded && Object.keys(task.params_decoded).length > 0 && <ParamsCard params={task.params_decoded} />}
        <LogCard taskId={task.id} onOpenLog={onOpenLog} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minWidth: 0 }}>
        {isTrain && <SamplesCard taskId={task.id} live={task.status === 'running'} onRender={onRender} canRender={!!outputs?.files.some((f) => f.is_lora)} />}
        <PropsCard task={task} fmt={fmt} onSaved={onSaved} />
        {isTrain && outputs && <OutputsCard taskId={task.id} outputs={outputs} fmt={fmt} />}
      </div>
    </div>
  )
}

/** Loss (EMA, filled) and learning rate (dashed, own scale) over the whole run. */
function CurvesCard({ task, monitor }: { task: Task; monitor: MonitorState | null }) {
  const { t } = useTranslation()
  const losses = useMemo(() => monitor?.losses ?? [], [monitor?.losses])
  const lrs = useMemo(() => monitor?.lr_history ?? [], [monitor?.lr_history])
  const total = monitor?.total_steps ?? 0
  const step = monitor?.step ?? 0
  const W = 860; const H = 210; const top = 10; const bottom = 190
  const maxX = Math.max(total, losses.length ? losses[losses.length - 1].step : 0, 1)

  const paths = useMemo(() => {
    if (losses.length < 2) return null
    const smooth = ema(losses.map((l) => l.loss))
    const every = Math.max(1, Math.floor(losses.length / 400))
    const idx: number[] = []
    for (let i = 0; i < losses.length; i += every) idx.push(i)
    if (idx[idx.length - 1] !== losses.length - 1) idx.push(losses.length - 1)
    const vals = idx.map((i) => smooth[i])
    const lo = Math.min(...vals); const hi = Math.max(...vals)
    const span = hi - lo || 1
    const x = (s: number) => (s / maxX) * W
    const y = (v: number) => bottom - ((v - lo) / span) * (bottom - top - 20)
    const line = idx.map((i, k) => `${k ? 'L' : 'M'}${x(losses[i].step).toFixed(1)} ${y(smooth[i]).toFixed(1)}`).join(' ')
    const lastX = x(losses[idx[idx.length - 1]].step).toFixed(1)
    const fill = `${line} L${lastX} ${H - 10} L${x(losses[idx[0]].step).toFixed(1)} ${H - 10} Z`
    let lr: string | null = null
    if (lrs.length > 1) {
      const le = Math.max(1, Math.floor(lrs.length / 300))
      const pts = lrs.filter((_, i) => i % le === 0 || i === lrs.length - 1)
      const llo = Math.min(...pts.map((p) => p.lr)); const lhi = Math.max(...pts.map((p) => p.lr))
      const lspan = lhi - llo || 1
      lr = pts.map((p, k) => `${k ? 'L' : 'M'}${x(p.step).toFixed(1)} ${(bottom - 4 - ((p.lr - llo) / lspan) * 60).toFixed(1)}`).join(' ')
    }
    return { line, fill, lr, lastX, lastY: y(smooth[smooth.length - 1]) }
  }, [losses, lrs, maxX])

  const running = task.status === 'running'
  const q = (f: number) => Math.round(maxX * f).toLocaleString()
  const ticks = [
    '0', q(0.25),
    running && step ? t('queueDetail.nowStep', { step: step.toLocaleString() }) : q(0.5),
    q(0.75), maxX.toLocaleString(),
  ]

  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div style={{ flex: 1 }}>
          <div className="ds-card-title">{t('queueDetail.curvesTitle')}</div>
          <div className="ds-card-sub">{t('queueDetail.curvesSub', { total: maxX.toLocaleString() })}</div>
        </div>
        <div className="ds-card-tools" style={{ gap: 12 }}>
          <span className="ds-legend"><s style={{ background: '#6e8f42' }} />loss</span>
          <span className="ds-legend"><s style={{ background: '#d9c27a' }} />lr</span>
        </div>
      </div>
      <div className="ds-card-body" style={{ paddingTop: 6 }}>
        {paths ? (
          <>
            <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: '100%', height: 210, display: 'block' }} aria-hidden="true">
              <defs>
                <linearGradient id="qd-loss" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6e8f42" stopOpacity=".18" /><stop offset="100%" stopColor="#6e8f42" stopOpacity="0" />
                </linearGradient>
              </defs>
              <path d={`M0 190 L${W} 190 M0 145 L${W} 145 M0 100 L${W} 100 M0 55 L${W} 55 M0 10 L${W} 10`} stroke="var(--line)" strokeWidth="1" />
              <path d={paths.fill} fill="url(#qd-loss)" />
              <path d={paths.line} fill="none" stroke="#6e8f42" strokeWidth="1.8" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
              {paths.lr && <path d={paths.lr} fill="none" stroke="#d9c27a" strokeWidth="1.4" strokeDasharray="4 3" vectorEffect="non-scaling-stroke" />}
              {running && <path d={`M${paths.lastX} 0 L${paths.lastX} ${H}`} stroke="var(--line-3)" strokeWidth="1" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />}
            </svg>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10.5, color: 'var(--ink-3)', fontFamily: 'var(--mono)', marginTop: 6 }}>
              {ticks.map((x, i) => <span key={i}>{x}</span>)}
            </div>
          </>
        ) : (
          <div className="ds-empty">{t('queueDetail.noCurves')}</div>
        )}
      </div>
    </div>
  )
}

/** The task's log; appended live from task_log_appended events. */
function useTaskLog(taskId: number) {
  const [content, setContent] = useState('')
  const [error, setError] = useState<string | null>(null)
  const contentRef = useRef('')
  const setBoth = useCallback((s: string) => { contentRef.current = s; setContent(s) }, [])

  const refresh = useCallback(async () => {
    try { const log = await api.getLog(taskId); setBoth(log.content); setError(null) }
    catch (e) { setError(String(e)) }
  }, [taskId, setBoth])

  useEffect(() => { setBoth(''); void refresh() }, [taskId, refresh, setBoth])

  useEventStream((evt) => {
    if (evt.task_id !== taskId) return
    if (evt.type === 'task_log_appended') {
      const text = typeof evt.text === 'string' ? evt.text : ''
      const prev = contentRef.current
      const sep = prev && !prev.endsWith('\n') ? '\n' : ''
      setBoth(prev + sep + text + '\n')
    } else if (evt.type === 'task_state_changed') {
      void refresh()
    }
  })
  return { content, error, refresh }
}

function lineTone(line: string): string | undefined {
  if (/traceback|error|exception|failed/i.test(line)) return 'ds-e'
  if (/warn/i.test(line)) return 'ds-w'
  return undefined
}

function ConsoleLines({ text, limit }: { text: string; limit?: number }) {
  const lines = text.split('\n').filter((l, i, a) => l.length > 0 || i < a.length - 1)
  const shown = limit ? lines.slice(-limit) : lines
  return (
    <>
      {shown.map((l, i) => <div key={i} className={lineTone(l)} style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{l || ' '}</div>)}
    </>
  )
}

function LogCard({ taskId, onOpenLog }: { taskId: number; onOpenLog: () => void }) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { content, error } = useTaskLog(taskId)
  const [follow, setFollow] = useState(true)
  const boxRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    if (follow && boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight
  }, [content, follow])
  const copy = async () => {
    try { await navigator.clipboard.writeText(content); toast(t('queueDetail.logCopied'), 'success') }
    catch { toast(t('queueDetail.copyFailed'), 'error') }
  }
  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div style={{ flex: 1 }}>
          <div className="ds-card-title">{t('queueDetail.tabLogs')}</div>
          <div className="ds-card-sub">{t('queueDetail.logSub')}</div>
        </div>
        <div className="ds-card-tools">
          <label className={`ds-switch${follow ? ' ds-on' : ''}`} title={t('queueDetail.autoScroll')} style={{ cursor: 'pointer' }}>
            <input type="checkbox" className="sr-only" checked={follow} onChange={(e) => setFollow(e.target.checked)} aria-label={t('queueDetail.autoScroll')} />
            <i />
          </label>
          <KebabMenu
            trigger="icon"
            label={t('queue.moreActions')}
            items={[
              { label: t('queueDetail.copyLog'), onSelect: () => void copy(), disabled: !content },
              { label: t('queueDetail.openFullLog'), onSelect: onOpenLog },
            ]}
          />
        </div>
      </div>
      <div className="ds-card-body" style={{ paddingTop: 2 }}>
        {error && <div className="ds-note ds-warn font-mono" style={{ marginBottom: 8 }}>{error}</div>}
        <div ref={boxRef} className="ds-console" style={{ maxHeight: 220, overflowY: 'auto' }}>
          {content ? <ConsoleLines text={content} limit={200} /> : <span className="ds-t">{t('queueDetail.noLogs')}</span>}
        </div>
      </div>
    </div>
  )
}

function SamplesCard({ taskId, live, onRender, canRender }: { taskId: number; live: boolean; onRender: () => void; canRender: boolean }) {
  const { t } = useTranslation()
  const [items, setItems] = useState<TaskSample[] | null>(null)
  const [zoom, setZoom] = useState<number | null>(null)
  const load = useCallback(() => {
    api.listTaskSamples(taskId).then((r) => setItems(r.items)).catch(() => setItems([]))
  }, [taskId])
  useEffect(() => { load() }, [load])
  // New samples land every few epochs while training runs.
  useEffect(() => {
    if (!live) return
    const id = window.setInterval(load, 30_000)
    return () => window.clearInterval(id)
  }, [live, load])
  const shown = (items ?? []).slice(-6)
  const mark = (s: TaskSample) => (s.epoch != null ? t('queueDetail.epochN', { n: s.epoch }) : s.step != null ? t('queue.stepN', { n: s.step }) : s.filename)
  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div style={{ flex: 1 }}>
          <div className="ds-card-title">{t('queue.samples')}</div>
          <div className="ds-card-sub">{items == null ? t('common.loading') : t('queue.samplesCount', { n: items.length })}</div>
        </div>
        <div className="ds-card-tools">
          <button type="button" className="ds-ctl ds-ghost" onClick={onRender} disabled={!canRender}>{t('queue.jumpGenerate')}</button>
        </div>
      </div>
      <div className="ds-card-body" style={{ paddingTop: 2 }}>
        {shown.length === 0 ? (
          <div className="ds-empty">{items == null ? t('common.loading') : t('queueDetail.noSamples')}</div>
        ) : (
          <div className="ds-grid-thumbs" style={{ gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
            {shown.map((s, i) => (
              <button key={s.filename} type="button" className="ds-thumb" style={{ aspectRatio: '1' }} onClick={() => setZoom((items?.length ?? 0) - shown.length + i)}>
                <img src={api.sampleImageUrl(s.filename, taskId, 240)} alt="" loading="lazy" style={{ width: '100%', height: '100%', objectFit: 'cover', position: 'absolute', inset: 0 }} />
                <span className="ds-cap">{mark(s)}</span>
              </button>
            ))}
          </div>
        )}
      </div>
      {zoom != null && items && items[zoom] && (
        <ImagePreviewModal
          src={api.sampleImageUrl(items[zoom].filename, taskId)}
          caption={`${mark(items[zoom])} · ${items[zoom].filename}`}
          index={zoom}
          total={items.length}
          hasPrev={zoom > 0}
          hasNext={zoom < items.length - 1}
          onClose={() => setZoom(null)}
          onPrev={() => setZoom((z) => (z != null && z > 0 ? z - 1 : z))}
          onNext={() => setZoom((z) => (z != null && z < items.length - 1 ? z + 1 : z))}
        />
      )}
    </div>
  )
}

function Kv({ k, v, title, mono = true }: { k: string; v: React.ReactNode; title?: string; mono?: boolean }) {
  return (
    <div className="ds-kv">
      <span className="ds-k">{k}</span>
      <span className="ds-v" title={title} style={{ textAlign: 'right', overflowWrap: 'anywhere', ...(mono ? {} : { fontFamily: 'inherit' }) }}>{v}</span>
    </div>
  )
}

function PropsCard({ task, fmt, onSaved }: { task: Task; fmt: QueueFormat; onSaved: (t: Task) => void }) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(task.note ?? '')
  const [saving, setSaving] = useState(false)

  const save = async () => {
    setSaving(true)
    try {
      const updated = await api.setTaskNote(task.id, draft)
      onSaved(updated)
      toast(updated.note ? t('queue.noteSaved') : t('queue.noteCleared'), 'success')
      setEditing(false)
    } catch (e) {
      toast(t('queue.noteFailed', { reason: String(e) }), 'error')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div style={{ flex: 1 }}>
          <div className="ds-card-title">{t('queueDetail.propsTitle')}</div>
          <div className="ds-card-sub">{t('queueDetail.propsSub')}</div>
        </div>
      </div>
      <div className="ds-card-body" style={{ paddingTop: 0 }}>
        <Kv k={t('queueDetail.idSource')} v={`${task.id} · ${task.task_type ?? 'train'} ${task.config_name}`} />
        {task.project_id != null && task.version_id != null && (
          <Kv k={t('queueDetail.source')} v={<Link to={`/projects/${task.project_id}?version=${task.version_id}`} style={{ color: 'var(--green-text)' }}>{t('queueDetail.sourceLink', { projectId: task.project_id, versionId: task.version_id })}</Link>} />
        )}
        <Kv k={t('queueDetail.priority')} v={task.priority} />
        <Kv k={t('queueDetail.enqueuedAt')} v={fmt.stamp(task.created_at)} />
        {task.scheduled_at && <Kv k={t('queueDetail.scheduledAt')} v={fmt.stamp(task.scheduled_at)} />}
        <Kv k={t('queueDetail.startedAt')} v={fmt.stamp(task.started_at)} />
        {task.finished_at && <Kv k={t('queueDetail.finishedAt')} v={fmt.stamp(task.finished_at)} />}
        <Kv k="PID" v={task.pid ?? '—'} />
        {task.config_path && <Kv k={t('queueDetail.configPath')} v={tail(task.config_path)} title={task.config_path} />}
        {task.monitor_state_path && <Kv k={t('queueDetail.monitorFile')} v={tail(task.monitor_state_path)} title={task.monitor_state_path} />}
        <Kv k={t('queueDetail.exitCode')} v={task.exit_code ?? '—'} />
        {task.error_msg && (
          <div className="ds-note ds-warn font-mono" style={{ marginTop: 8, fontSize: 11, overflowWrap: 'anywhere' }}>{task.error_msg}</div>
        )}

        {/* 备注（_v20 tasks.note）—— 队列页右键写的那句话，这里完整显示 + 可改。 */}
        <div style={{ borderTop: '1px solid var(--line)', marginTop: 10, paddingTop: 10 }}>
          {editing ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <textarea
                autoFocus
                value={draft}
                maxLength={500}
                rows={3}
                placeholder={t('queue.notePlaceholder')}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') { e.preventDefault(); setEditing(false) }
                  // Ctrl/Cmd+Enter 保存 —— 纯 Enter 留给换行。
                  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); void save() }
                }}
                className="ds-inp"
                style={{ height: 'auto', padding: '7px 9px', resize: 'vertical' }}
                data-testid="task-note-input"
              />
              <div style={{ display: 'flex', gap: 6 }}>
                <button type="button" className="ds-btn-primary" style={{ height: 26 }} onClick={() => void save()} disabled={saving}>
                  {saving ? t('common.saving') : t('common.save')}
                </button>
                <button type="button" className="ds-ctl ds-ghost" style={{ height: 26 }} onClick={() => setEditing(false)} disabled={saving}>{t('common.cancel')}</button>
              </div>
            </div>
          ) : task.note ? (
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, fontSize: 12 }} data-testid="task-note">
              <span style={{ flex: 1, minWidth: 0, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', color: 'var(--ink-2)' }}>{task.note}</span>
              <button type="button" className="ds-ctl ds-ghost" style={{ height: 24, fontSize: 11 }} onClick={() => { setDraft(task.note ?? ''); setEditing(true) }}>{t('queue.noteEdit')}</button>
            </div>
          ) : (
            <button type="button" className="ds-ctl ds-ghost" style={{ height: 24, fontSize: 11 }} title={t('queue.noteHint')} onClick={() => { setDraft(''); setEditing(true) }} data-testid="task-note-add">
              + {t('queue.noteAdd')}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function ParamsCard({ params }: { params: Record<string, unknown> }) {
  const { t } = useTranslation()
  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad"><div><div className="ds-card-title">{t('queue.jobs.paramsSection')}</div></div></div>
      <div className="ds-card-body" style={{ paddingTop: 0 }}>
        {Object.entries(params).map(([k, v]) => <Kv key={k} k={paramLabel(k, t)} v={fmtParamValue(v, t)} />)}
      </div>
    </div>
  )
}

function OutputsCard({ taskId, outputs, fmt }: { taskId: number; outputs: TaskOutputs; fmt: QueueFormat }) {
  const { t } = useTranslation()
  const loras = outputs.files.filter((f) => f.is_lora).sort((a, b) => b.mtime - a.mtime)
  const size = loras.reduce((n, f) => n + f.size, 0)
  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div style={{ flex: 1 }}>
          <div className="ds-card-title">{t('queueDetail.tabOutputs')}</div>
          <div className="ds-card-sub">{t('queueDetail.outputsSub', { count: loras.length, size: fmt.bytes(size) })}</div>
        </div>
      </div>
      <div className="ds-card-body" style={{ paddingTop: 0 }}>
        {loras.length === 0 ? (
          <div className="ds-empty">{outputs.exists ? t('queueDetail.noCheckpoints') : t('queueDetail.dirNotExist')}</div>
        ) : (
          <>
            {loras.slice(0, 3).map((f) => (
              <div key={f.path} className="ds-kv">
                <span className="ds-k ds-mono" style={{ fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={f.path}>{f.name}</span>
                <span className="ds-v">{fmt.bytes(f.size)}</span>
              </div>
            ))}
            <a
              className="ds-ctl"
              style={{ width: '100%', justifyContent: 'center', marginTop: 10 }}
              href={api.taskOutputsZipUrl(taskId, loras.map((f) => f.path))}
              download={`${outputs.archive_basename ?? `task_${taskId}`}_checkpoints.zip`}
            >
              {t('queueDetail.downloadCheckpoints')}
            </a>
          </>
        )}
      </div>
    </div>
  )
}

// ── Logs tab ────────────────────────────────────────────────────────────────

function LogTab({ taskId }: { taskId: number }) {
  const { t } = useTranslation()
  const { content, error, refresh } = useTaskLog(taskId)
  const [autoScroll, setAutoScroll] = useState(true)
  const boxRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (autoScroll && boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight
  }, [content, autoScroll])
  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div style={{ flex: 1 }}>
          <div className="ds-card-title">{t('queueDetail.tabLogs')}</div>
          <div className="ds-card-sub">{t('queueDetail.logSub')}</div>
        </div>
        <div className="ds-card-tools" style={{ gap: 10 }}>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--ink-3)', cursor: 'pointer' }}>
            <span className={`ds-switch${autoScroll ? ' ds-on' : ''}`}>
              <input type="checkbox" className="sr-only" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} />
              <i />
            </span>
            {t('queueDetail.autoScroll')}
          </label>
          <button type="button" className="ds-ctl ds-ghost" onClick={() => void refresh()}>{t('common.refresh')}</button>
        </div>
      </div>
      <div className="ds-card-body" style={{ paddingTop: 2 }}>
        {error && <div className="ds-note ds-warn font-mono" style={{ marginBottom: 8 }}>{error}</div>}
        <div ref={boxRef} className="ds-console" style={{ height: 'calc(100vh - 360px)', minHeight: 320, overflowY: 'auto' }}>
          {content ? <ConsoleLines text={content} /> : <span className="ds-t">{t('queueDetail.noLogs')}</span>}
        </div>
      </div>
    </div>
  )
}

// ── Metrics tab (post-training eval) ────────────────────────────────────────

const METRIC_NAME_KEY: Record<EvalMetricKey, string> = {
  clip_t: 'queueDetail.metric.clip_t',
  clip_i: 'queueDetail.metric.clip_i',
  dino_i: 'queueDetail.metric.dino_i',
  ccip_i: 'queueDetail.metric.ccip_i',
  tag_recall: 'queueDetail.metric.tag_recall',
}
const METRIC_CODE: Record<EvalMetricKey, string> = {
  clip_t: 'CLIP-T', clip_i: 'CLIP-I', dino_i: 'DINO-I', ccip_i: 'CCIP-I', tag_recall: 'tag recall',
}

function useEvalResults(task: Task) {
  const [payload, setPayload] = useState<EvalMetricsListResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const pid = task.project_id; const vid = task.version_id
  const load = useCallback(async () => {
    if (!pid || !vid) return
    try { setPayload(await api.listEvalMetrics(pid, vid, task.id)); setError(null) }
    catch (e) { setError(String(e)) }
  }, [pid, vid, task.id])
  useEffect(() => { void load() }, [load])
  const all = useMemo(() => [...(payload?.results ?? [])].sort((a, b) => checkpointSortValue(a, 0) - checkpointSortValue(b, 0)), [payload?.results])
  const results = useMemo(() => all.filter((r) => !r.baseline), [all])
  const busy = results.some((r) => EVAL_METRIC_KEYS.some((k) => ['pending', 'running'].includes(metricState(r, k)?.status ?? '')))
  useEffect(() => {
    if (!busy) return
    const id = window.setInterval(() => void load(), 5000)
    return () => window.clearInterval(id)
  }, [busy, load])
  const keys = EVAL_METRIC_KEYS.filter((k) => CORE_METRIC_KEYS.has(k) || results.some((r) => {
    const s = metricState(r, k)?.status
    return s != null && s !== 'not_run'
  }))
  return { payload, results, keys, error, linked: !!(pid && vid) }
}

function EvalStats({ task }: { task: Task }) {
  const { t } = useTranslation()
  const { results, keys } = useEvalResults(task)
  const measured = results.filter((r) => keys.some((k) => metricValue(r, k) != null))
  const images = results.map((r) => r.sample_run?.summary?.total ?? 0).reduce((a, b) => Math.max(a, b), 0)
  const best = results.reduce<{ label: string; v: number; d?: number } | null>((acc, r) => {
    const v = metricValue(r, 'clip_i')
    if (v == null || (acc && acc.v >= v)) return acc
    return { label: checkpointLabel(r), v, d: r.delta?.clip_i }
  }, null)
  const last = results[results.length - 1]
  const doneKeys = last ? keys.filter((k) => metricValue(last, k) != null) : []
  const pendingKeys = last ? keys.filter((k) => ['pending', 'running'].includes(metricState(last, k)?.status ?? '')) : []
  return (
    <div className="ds-statgrid" style={{ gridTemplateColumns: 'repeat(4,minmax(0,1fr))' }}>
      <StatCell label={t('queueDetail.evalChecked')} value={measured.length} unit={`/ ${results.length}`}>
        <span className="ds-meter" style={{ marginTop: 9 }}><i style={{ width: `${results.length ? (measured.length / results.length) * 100 : 0}%` }} /></span>
      </StatCell>
      <StatCell label={t('queueDetail.evalImages')} value={images || '—'} unit={images ? t('queueDetail.imagesUnit') : undefined} sub={t('queueDetail.evalImagesSub')} />
      <StatCell
        label={t('queueDetail.evalBest')}
        value={best ? best.label : '—'}
        accent={!!best}
        sub={best?.d != null ? t('queueDetail.evalVsBase', { d: `${best.d >= 0 ? '+' : '−'}${Math.abs(best.d).toFixed(3)}` }) : undefined}
      />
      <StatCell
        label={t('queueDetail.evalScore')}
        value={last ? `${doneKeys.length} / ${keys.length}` : '—'}
        sub={pendingKeys.length ? t('queueDetail.evalStillRunning', { list: pendingKeys.map((k) => METRIC_CODE[k]).join(', ') }) : undefined}
      />
    </div>
  )
}

function EvalTab({ task }: { task: Task }) {
  const { t } = useTranslation()
  const { payload, results, keys, error, linked } = useEvalResults(task)
  const cols = results.slice(-4)
  const last = results[results.length - 1]

  // CLIP-I across checkpoints (measured points only; no forecast).
  const series = results.map((r) => metricValue(r, 'clip_i')).map((v, i) => ({ v, label: checkpointLabel(results[i]) })).filter((p) => p.v != null) as Array<{ v: number; label: string }>

  if (!linked) return <div className="ds-empty">{t('queueDetail.noProjectAssoc')}</div>

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 380px', gap: 14, alignItems: 'start' }} className="ds-qd-grid">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minWidth: 0 }}>
        {error && <div className="ds-note ds-warn font-mono">{error}</div>}
        <div className="ds-card">
          <div className="ds-card-head ds-pad">
            <div style={{ flex: 1 }}>
              <div className="ds-card-title">{t('queueDetail.evalTableTitle')}</div>
              <div className="ds-card-sub">{t('queueDetail.evalTableSub')}</div>
            </div>
          </div>
          {payload == null ? (
            <div className="ds-card-body"><div className="ds-empty">{t('common.loading')}</div></div>
          ) : results.length === 0 ? (
            <div className="ds-card-body"><div className="ds-empty">{t('queueDetail.evalEmpty')}</div></div>
          ) : (
            <table className="ds-tbl">
              <thead>
                <tr>
                  <th>{t('queueDetail.metricCol')}</th>
                  {cols.map((r) => <th key={r.run_id}>{checkpointLabel(r)}</th>)}
                  <th style={{ width: 150 }}>{t('queueDetail.deltaCol')}</th>
                </tr>
              </thead>
              <tbody>
                {keys.map((k) => {
                  const vals = cols.map((r) => metricValue(r, k))
                  const bestV = Math.max(...vals.filter((v): v is number => v != null))
                  const d = last?.delta?.[k]
                  return (
                    <tr key={k}>
                      <td><span className="ds-cell-main"><span>{t(METRIC_NAME_KEY[k])}<span className="ds-cell-key">{METRIC_CODE[k]}</span></span></span></td>
                      {cols.map((r, i) => {
                        const v = vals[i]
                        const st = metricState(r, k)?.status
                        return (
                          <td key={r.run_id} className="ds-num" style={v != null && v === bestV && cols.length > 1 ? { color: 'var(--green-text)', fontWeight: 600 } : undefined}>
                            {v != null ? v.toFixed(3) : st === 'running' || st === 'pending' ? '…' : '—'}
                          </td>
                        )
                      })}
                      <td>
                        {d != null ? (
                          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span className="ds-meter" style={{ width: 90 }}><i style={{ width: `${Math.min(100, Math.abs(d) * 400)}%`, ...(d < 0 ? { background: 'var(--red-text)' } : {}) }} /></span>
                            <span className="ds-mono" style={{ fontSize: 11 }}>{d >= 0 ? '+' : '−'}{Math.abs(d).toFixed(3)}</span>
                          </span>
                        ) : <span className="ds-muted">—</span>}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>

        {series.length > 1 && (
          <div className="ds-card">
            <div className="ds-card-head ds-pad">
              <div style={{ flex: 1 }}>
                <div className="ds-card-title">{t('queueDetail.clipiTitle')}</div>
                <div className="ds-card-sub">{t('queueDetail.clipiSub')}</div>
              </div>
            </div>
            <div className="ds-card-body" style={{ paddingTop: 6 }}>
              {(() => {
                const W = 860; const H = 170
                const lo = Math.min(...series.map((p) => p.v)); const hi = Math.max(...series.map((p) => p.v))
                const span = hi - lo || 1
                const x = (i: number) => 40 + (i / (series.length - 1)) * (W - 80)
                const y = (v: number) => 140 - ((v - lo) / span) * 110
                return (
                  <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: '100%', height: 170, display: 'block' }} aria-hidden="true">
                    <path d={`M0 150 L${W} 150 M0 110 L${W} 110 M0 70 L${W} 70 M0 30 L${W} 30`} stroke="var(--line)" strokeWidth="1" />
                    <path d={series.map((p, i) => `${i ? 'L' : 'M'}${x(i)} ${y(p.v)}`).join(' ')} fill="none" stroke="#6e8f42" strokeWidth="2.2" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
                    {series.map((p, i) => <circle key={i} cx={x(i)} cy={y(p.v)} r="4" fill="#6e8f42" />)}
                  </svg>
                )
              })()}
              <div className="ds-axis">{series.map((p, i) => <span key={i}>{p.label}</span>)}</div>
            </div>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minWidth: 0 }}>
        <div className="ds-card">
          <div className="ds-card-head ds-pad">
            <div><div className="ds-card-title">{t('queueDetail.howToRead')}</div><div className="ds-card-sub">{t('queueDetail.howToReadSub')}</div></div>
          </div>
          <div className="ds-card-body" style={{ paddingTop: 0 }}>
            {EVAL_METRIC_KEYS.map((k) => (
              <div key={k} className="ds-kv"><span className="ds-k ds-mono">{METRIC_CODE[k]}</span><span className="ds-v" style={{ fontFamily: 'inherit' }}>{t(`queueDetail.metricHint.${k}`)}</span></div>
            ))}
            <div className="ds-note ds-info" style={{ marginTop: 11 }}>
              <span>{t('queueDetail.evalWhereToEnable')}</span>
            </div>
          </div>
        </div>
        {last && (
          <div className="ds-card">
            <div className="ds-card-head ds-pad"><div><div className="ds-card-title">{t('queueDetail.evalRunTitle')}</div><div className="ds-card-sub">{t('queueDetail.evalRunSub')}</div></div></div>
            <div className="ds-card-body" style={{ paddingTop: 0 }}>
              <Kv k={t('queueDetail.idSource')} v={`${task.id} · ${last.version_label ?? task.config_name}`} />
              <Kv k={t('queueDetail.evalRunId')} v={last.run_id} />
              {last.sample_run?.summary?.total != null && <Kv k={t('queueDetail.evalImages')} v={last.sample_run.summary.total} />}
              {last.checkpoint?.path && <Kv k={t('queueDetail.checkpoint')} v={tail(last.checkpoint.path, 1)} title={last.checkpoint.path} />}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Outputs tab ─────────────────────────────────────────────────────────────

type SortKey = 'name' | 'size' | 'mtime'
const STATE_KINDS = new Set(['training_state', 'pause_state', 'auto_epoch_state'])

export function OutputsTab({ taskId, onChanged }: { taskId: number; onChanged?: () => void }) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const fmt = useQueueFormat()
  const [data, setData] = useState<TaskOutputs | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [zipping, setZipping] = useState(false)
  const [exportingOutputs, setExportingOutputs] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [refreshKey, setRefreshKey] = useState(0)
  const [selectMode, setSelectMode] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(() => new Set())
  const [downloadDialog, setDownloadDialog] = useState<null | { destination: 'download' | 'data_exports' }>(null)
  const [sortKey, setSortKey] = useState<SortKey>('mtime')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  useEffect(() => {
    let alive = true
    void api.getTaskOutputs(taskId).then((r) => alive && setData(r)).catch((e) => alive && setError(String(e)))
    return () => { alive = false }
  }, [taskId, refreshKey])

  const refresh = () => {
    setRefreshKey((k) => k + 1)
    onChanged?.()
  }

  // The zip is built by the backend while the browser downloads it; it reports
  // task_outputs_zip_ready / _failed over SSE. A 60 s fallback unlocks the button
  // if the event never arrives.
  useEventStream((evt) => {
    if (evt.task_id !== taskId) return
    if (evt.type === 'task_outputs_zip_ready') {
      setZipping(false)
    } else if (evt.type === 'task_outputs_zip_failed') {
      setZipping(false)
      toast(t('queueDetail.compressionFailed', { error: typeof evt.error === 'string' ? evt.error : '?' }), 'error')
    }
  })

  useEffect(() => {
    if (!zipping) return
    const tid = window.setTimeout(() => {
      setZipping(false)
      toast(t('queueDetail.compressionTimeout'), 'info')
    }, 60_000)
    return () => window.clearTimeout(tid)
  }, [zipping, toast, t])

  const sortedFiles = useMemo(() => {
    if (!data) return []
    const sign = sortDir === 'asc' ? 1 : -1
    return [...data.files].sort((a, b) => {
      // numeric: ep_002 sorts before ep_010
      if (sortKey === 'name') return (a.path || a.name).localeCompare(b.path || b.name, undefined, { numeric: true }) * sign
      if (sortKey === 'size') return (a.size - b.size) * sign
      return (a.mtime - b.mtime) * sign
    })
  }, [data, sortKey, sortDir])
  const stateFiles = useMemo(() => sortedFiles.filter((f) => STATE_KINDS.has(f.kind ?? '')), [sortedFiles])
  const regularFiles = useMemo(() => sortedFiles.filter((f) => !STATE_KINDS.has(f.kind ?? '')), [sortedFiles])

  const onHeaderClick = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(key === 'name' ? 'asc' : 'desc')
    }
  }
  const sortArrow = (key: SortKey) => (sortKey === key ? (sortDir === 'asc' ? ' ↑' : ' ↓') : '')

  // Drop selected paths that disappeared after a refresh.
  useEffect(() => {
    if (selected.size === 0) return
    const names = new Set(sortedFiles.map((f) => f.path))
    let dropped = false
    const next = new Set<string>()
    for (const n of selected) {
      if (names.has(n)) next.add(n); else dropped = true
    }
    if (dropped) setSelected(next)
  }, [sortedFiles, selected])

  const selectedSize = useMemo(() => {
    let total = 0
    for (const f of sortedFiles) if (selected.has(f.path)) total += f.size
    return total
  }, [sortedFiles, selected])
  const totalSize = useMemo(() => sortedFiles.reduce((s, f) => s + f.size, 0), [sortedFiles])

  const allSelected = sortedFiles.length > 0 && selected.size === sortedFiles.length
  const noneSelected = selected.size === 0
  const partialSelected = !allSelected && !noneSelected

  const toggleSelectAll = () => {
    setSelected(allSelected ? new Set() : new Set(sortedFiles.map((f) => f.path)))
  }
  const toggleOne = (name: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name); else next.add(name)
      return next
    })
  }
  const toggleSelectMode = () => {
    setSelectMode((m) => {
      if (m) setSelected(new Set())
      return !m
    })
  }

  const openFolder = async () => {
    setBusy(true)
    try { const r = await api.openTaskFolder(taskId); toast(t('queueDetail.folderOpened', { path: r.opened }), 'success') }
    catch (e) { toast(String(e), 'error') }
    finally { setBusy(false) }
  }

  const outputSelection = () => {
    const partial = selectMode && selected.size > 0
    if (selectMode && !partial) return null
    return partial ? Array.from(selected) : undefined
  }

  const handleDownloadZip = () => {
    if (zipping) return
    const files = outputSelection()
    if (selectMode && files === null) return
    setZipping(true)
    // The backend's Content-Disposition name wins; `download` is only a fallback.
    const baseName = data?.archive_basename ?? `task_${taskId}`
    const zipName = files ? `${baseName}_outputs_selected.zip` : `${baseName}_outputs.zip`
    const a = document.createElement('a')
    a.href = api.taskOutputsZipUrl(taskId, files ?? undefined)
    a.download = zipName
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const handleExportOutputs = async () => {
    if (exportingOutputs) return
    const files = outputSelection()
    if (selectMode && files === null) return
    setExportingOutputs(true)
    try {
      const result = await api.exportTaskOutputs(taskId, files ?? undefined)
      toast(t('queueDetail.exportedToDataExports', { filename: result.filename, path: result.path }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setExportingOutputs(false)
    }
  }

  const copyPath = async () => {
    if (!data?.output_dir) return
    try { await navigator.clipboard.writeText(data.output_dir); toast(t('queueDetail.pathCopied'), 'success') }
    catch { toast(t('queueDetail.copyFailed'), 'error') }
  }

  const handleDelete = async () => {
    if (deleting) return
    const files = Array.from(selected)
    if (files.length === 0) return
    const ok = await confirm(t('queueDetail.deleteConfirmTitle'), { tone: 'danger' })
    if (!ok) return
    setDeleting(true)
    try {
      const r = await api.deleteTaskOutputs(taskId, files)
      toast(t('queueDetail.deletedFiles', { n: r.deleted.length }), 'success')
      setSelected(new Set())
      refresh()
    } catch (e) {
      toast(t('queueDetail.deleteFailed', { error: String(e) }), 'error')
    } finally {
      setDeleting(false)
    }
  }

  const sortHead = (key: SortKey, label: string, align?: 'right') => (
    <th style={align ? { textAlign: 'right' } : undefined}>
      <button type="button" className="ds-sortbtn" onClick={() => onHeaderClick(key)}>{label}{sortArrow(key)}</button>
    </th>
  )

  const fileTable = (files: typeof sortedFiles) => (
    <table className="ds-tbl">
      <thead>
        <tr>
          {selectMode && (
            <th style={{ width: 32 }}>
              <input
                type="checkbox"
                checked={allSelected}
                ref={(el) => { if (el) el.indeterminate = partialSelected }}
                onChange={toggleSelectAll}
                style={{ accentColor: 'var(--ink)', cursor: 'pointer' }}
                aria-label={t('common.selectAll')}
              />
            </th>
          )}
          {sortHead('name', t('common.file'))}
          {sortHead('size', t('common.size'), 'right')}
          {sortHead('mtime', t('queueDetail.modifiedTime'), 'right')}
          {!selectMode && <th />}
        </tr>
      </thead>
      <tbody>
        {files.map((f) => {
          const isSel = selected.has(f.path)
          return (
            <tr
              key={f.path}
              className="outputs-row"
              onClick={selectMode ? () => toggleOne(f.path) : undefined}
              style={selectMode ? { cursor: 'pointer', background: isSel ? 'var(--green-soft)' : undefined } : undefined}
            >
              {selectMode && (
                <td>
                  <input
                    type="checkbox"
                    checked={isSel}
                    onChange={() => toggleOne(f.path)}
                    onClick={(e) => e.stopPropagation()}
                    style={{ accentColor: 'var(--ink)', cursor: 'pointer' }}
                    aria-label={`${t('common.select')} ${f.path || f.name}`}
                  />
                </td>
              )}
              <td>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                  <span className="ds-mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.path || f.name}</span>
                  {f.is_lora && <span className="ds-badge ds-ok">LoRA</span>}
                  {f.kind === 'training_state' && <span className="ds-badge ds-warn">State</span>}
                  {f.kind === 'pause_state' && <span className="ds-badge ds-warn">Pause</span>}
                  {f.kind === 'auto_epoch_state' && <span className="ds-badge ds-warn">Auto</span>}
                </div>
              </td>
              <td className="ds-num ds-mono" style={{ textAlign: 'right', color: 'var(--ink-3)' }}>{fmt.bytes(f.size)}</td>
              <td className="ds-num ds-mono" style={{ textAlign: 'right', color: 'var(--ink-3)' }}>{fmt.stamp(f.mtime)}</td>
              {!selectMode && (
                <td style={{ textAlign: 'right' }}>
                  <a className="ds-ctl" href={api.taskOutputDownloadUrl(taskId, f.path)} download={f.name}>{t('queueDetail.downloadFile')}</a>
                </td>
              )}
            </tr>
          )
        })}
      </tbody>
    </table>
  )

  const hasFiles = !!data?.exists && data.files.length > 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div className="ds-card">
        <div className="ds-card-head ds-pad">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="ds-card-title">{t('queueDetail.tabOutputs')}</div>
            <div className="ds-card-sub ds-mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={data?.output_dir ?? undefined}>
              {data?.output_dir || (data ? t('queueDetail.noProjectAssoc') : t('common.loading'))}
              {hasFiles && ` · ${t('queueDetail.filesCount', { count: sortedFiles.length })} · ${fmt.bytes(totalSize)}`}
            </div>
          </div>
          {data?.output_dir && (
            <div className="ds-card-tools" style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              <button type="button" className="ds-ctl" onClick={copyPath}>{t('queueDetail.copyPath')}</button>
              {data.supports_open_folder ? (
                <button type="button" className="ds-ctl" onClick={openFolder} disabled={busy || !data.exists}>{t('queueDetail.openFolder')}</button>
              ) : (
                <span className="ds-badge ds-mute">{t('common.remote')}</span>
              )}
              <button type="button" className="ds-ctl" onClick={refresh}>{t('common.refresh')}</button>
              {hasFiles && (
                <>
                  <button type="button" className={`ds-ctl${selectMode ? ' ds-is-on' : ''}`} onClick={toggleSelectMode}>
                    {selectMode ? t('queueDetail.exitBatchMode') : t('queueDetail.batchMode')}
                  </button>
                  {selectMode && (
                    <button type="button" className="ds-btn-danger" onClick={handleDelete} disabled={deleting || noneSelected}>
                      {t('queueDetail.deleteSelected', { n: selected.size })}
                    </button>
                  )}
                  <button
                    type="button"
                    className="ds-btn-primary"
                    onClick={() => setDownloadDialog({ destination: 'download' })}
                    disabled={zipping || exportingOutputs || deleting || (selectMode && noneSelected)}
                  >
                    {zipping
                      ? t('queueDetail.compressing')
                      : exportingOutputs
                        ? t('queueDetail.exportingOutputs')
                        : selectMode
                          ? (noneSelected ? t('queueDetail.downloadSelectedEmpty') : t('queueDetail.downloadSelected', { n: selected.size, size: fmt.bytes(selectedSize) }))
                          : t('queueDetail.downloadAll')}
                  </button>
                </>
              )}
            </div>
          )}
        </div>
        {error ? (
          <div className="ds-note ds-warn ds-mono" style={{ margin: '0 17px 16px' }}>{error}</div>
        ) : !data ? null : !data.output_dir ? null : !data.exists ? (
          <div className="ds-empty">{t('queueDetail.dirNotExist')}</div>
        ) : sortedFiles.length === 0 ? (
          <div className="ds-empty">{t('queueDetail.dirEmpty')}</div>
        ) : regularFiles.length > 0 ? (
          <div style={{ borderTop: '1px solid var(--line)' }}>{fileTable(regularFiles)}</div>
        ) : null}
      </div>

      {data?.exists && stateFiles.length > 0 && (
        <div className="ds-card">
          <div className="ds-card-head ds-pad">
            <div>
              <div className="ds-card-title">{t('queueDetail.stateTitle')}</div>
              <div className="ds-card-sub">{t('queueDetail.stateSub')}</div>
            </div>
          </div>
          <div style={{ borderTop: '1px solid var(--line)' }}>{fileTable(stateFiles)}</div>
        </div>
      )}

      {downloadDialog && (
        <OutputsDownloadDialog
          destination={downloadDialog.destination}
          onDestinationChange={(d) => setDownloadDialog({ destination: d })}
          busy={zipping || exportingOutputs}
          onCancel={() => setDownloadDialog(null)}
          onConfirm={() => {
            const dest = downloadDialog.destination
            setDownloadDialog(null)
            if (dest === 'download') handleDownloadZip()
            else void handleExportOutputs()
          }}
        />
      )}
    </div>
  )
}

function OutputsDownloadDialog({
  destination, onDestinationChange, busy, onConfirm, onCancel,
}: {
  destination: 'download' | 'data_exports'
  onDestinationChange: (d: 'download' | 'data_exports') => void
  busy: boolean
  onConfirm: () => void
  onCancel: () => void
}) {
  const { t } = useTranslation()
  const option = (value: 'download' | 'data_exports', label: string) => (
    <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', fontSize: 13 }}>
      <input
        type="radio"
        name="outputs-download-destination"
        checked={destination === value}
        onChange={() => onDestinationChange(value)}
        disabled={busy}
        style={{ accentColor: 'var(--green-600)' }}
      />
      <span>{label}</span>
    </label>
  )
  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(20,22,20,.18)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onCancel() }}
    >
      <div className="ds-card" style={{ width: '100%', maxWidth: 420, boxShadow: '0 24px 48px -24px rgba(20,22,20,.3)' }}>
        <div className="ds-card-head ds-pad">
          <div>
            <div className="ds-card-title">{t('queueDetail.downloadDialogTitle')}</div>
            <div className="ds-card-sub">{t('queueDetail.downloadDialogHint')}</div>
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: '4px 17px 16px' }}>
          {option('download', t('queueDetail.downloadDestinationLocal'))}
          {option('data_exports', t('queueDetail.downloadDestinationDataExports'))}
        </div>
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', padding: '12px 17px', borderTop: '1px solid var(--line)' }}>
          <button type="button" className="ds-ctl" onClick={onCancel} disabled={busy}>{t('common.cancel')}</button>
          <button type="button" className="ds-btn-primary" onClick={onConfirm} disabled={busy}>{busy ? '…' : t('common.confirm')}</button>
        </div>
      </div>
    </div>
  )
}

// ── Snapshot config tab (ADR-0007 §11.7) ────────────────────────────────────

/** Highlights a YAML dump line by line: keys, strings, comments. */
function YamlLines({ text }: { text: string }) {
  return (
    <>
      {text.split('\n').map((line, i) => {
        const comment = line.match(/^(\s*)(#.*)$/)
        if (comment) return <div key={i}>{comment[1]}<span className="ds-c">{comment[2]}</span></div>
        const kv = line.match(/^(\s*-?\s*)([^:#'"\s][^:]*?):(\s+(.*))?$/)
        if (!kv) return <div key={i}>{line || ' '}</div>
        const value = kv[4] ?? ''
        const isString = /^['"]/.test(value) || (value !== '' && !/^(-?[\d.e+-]+|true|false|null|~|\[.*\]|\{.*\})$/i.test(value))
        return (
          <div key={i}>
            {kv[1]}<span className="ds-k">{kv[2]}</span>:{kv[3] != null && ' '}
            {value && <span className={isString ? 'ds-s' : 'ds-v'}>{value}</span>}
          </div>
        )
      })}
    </>
  )
}

/** The training config frozen when the task started, read-only, plus
 *  "apply this config": PUT it into the version and open the train step, where
 *  starting training creates a new task. */
export function SnapshotConfigTab({ task }: { task: Task | null }) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const [data, setData] = useState<{ yaml: string; config: Record<string, unknown> } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [applying, setApplying] = useState(false)
  const [confirmApply, setConfirmApply] = useState(false)

  // The parent re-clones the task every 2 s for the elapsed timer; the snapshot
  // is immutable, so fetch it only when the id changes or the task starts
  // (that is when the snapshot is written).
  const taskId = task?.id ?? null
  const startedAt = task?.started_at ?? null
  useEffect(() => {
    if (taskId == null) { setLoading(false); return }
    let cancelled = false
    setLoading(true)
    void api.getTaskSnapshotConfig(taskId)
      .then((r) => { if (!cancelled) { setData(r); setError(null) } })
      .catch((e) => { if (!cancelled) setError(String(e)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [taskId, startedAt])

  const apply = async () => {
    if (!task || !data) return
    const pid = task.project_id, vid = task.version_id
    if (!pid || !vid) {
      toast(t('snapshot.noVersionLink'), 'error')
      return
    }
    setApplying(true)
    try {
      await api.putVersionConfig(pid, vid, data.config as Parameters<typeof api.putVersionConfig>[2])
      setConfirmApply(false)
      navigate(`/projects/${pid}/v/${vid}/train`)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setApplying(false)
    }
  }

  const canApply = !!(task?.project_id && task?.version_id)

  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div style={{ flex: 1 }}>
          <div className="ds-card-title">{t('snapshot.title')}</div>
          <div className="ds-card-sub">{t('snapshot.subtitle')}</div>
        </div>
        {data && (
          <div className="ds-card-tools">
            <button
              type="button"
              className="ds-btn-primary"
              onClick={() => setConfirmApply(true)}
              disabled={!canApply || applying || confirmApply}
              title={canApply ? undefined : t('snapshot.noVersionLink')}
            >
              {t('snapshot.applyBtn')}
            </button>
          </div>
        )}
      </div>

      {confirmApply && (
        <div className="ds-note ds-warn" style={{ margin: '0 17px 14px', display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <b>{t('snapshot.applyConfirmTitle')}</b>
            <div>{t('snapshot.applyConfirmDesc')}</div>
          </div>
          <button type="button" className="ds-ctl" onClick={() => setConfirmApply(false)} disabled={applying}>{t('common.cancel')}</button>
          <button type="button" className="ds-btn-primary" onClick={() => void apply()} disabled={applying}>
            {applying ? '…' : t('snapshot.applyBtn')}
          </button>
        </div>
      )}

      <div style={{ borderTop: '1px solid var(--line)' }}>
        {loading ? (
          <div className="ds-empty">{t('common.loading')}</div>
        ) : error ? (
          <div style={{ padding: '14px 17px 16px' }}>
            <div className="ds-note ds-warn">
              <div>
                <b>{t('snapshot.empty')}</b>
                <div>{t('snapshot.notFoundHint')}</div>
              </div>
            </div>
            <div className="ds-cell-key ds-mono" style={{ marginTop: 8 }}>{error}</div>
          </div>
        ) : !data ? (
          <div className="ds-empty">{t('snapshot.empty')}</div>
        ) : (
          <pre className="ds-yaml" style={{ overflow: 'auto', maxHeight: 'calc(100vh - 320px)' }}>
            <YamlLines text={data.yaml} />
          </pre>
        )}
      </div>
    </div>
  )
}
