import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import {
  api, type MonitorState, type QueueHistoryPage, type QueueHoldState, type Task,
} from '../api/client'
import { HoldQueueModal, type HoldDecision } from '../components/HoldQueueModal'
import { PauseConfirmModal } from '../components/PauseConfirmModal'
import { PauseProgressModal } from '../components/PauseProgressModal'
import TaskSampleStrip from '../components/TaskSampleStrip'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import KebabMenu, { type KebabItem } from '../components/ds/KebabMenu'
import PageHead from '../components/ds/PageHead'
import { useEventStream } from '../lib/useEventStream'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import { buildTrainingForecast } from '../lib/queueEstimates'
import { jobJumpPath } from './queue/jobUtils'

/** 备注输入上限 —— 与后端 _MAX_NOTE_LEN 对齐（超了后端截断，这里先拦住）。 */
const MAX_NOTE_LEN = 500
/** Finished tasks shown before "show more" (the list can hold hundreds). */
const HISTORY_STEP = 30
/** Data tasks: latest finished ones listed under the live ones. */
const DATA_HISTORY = 8

type ListFilter = 'active' | 'all' | 'done'
const TERMINAL = new Set(['done', 'failed', 'canceled'])

const Icon = {
  drag: (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <path d="M8 6h.01M8 12h.01M8 18h.01M16 6h.01M16 12h.01M16 18h.01" />
    </svg>
  ),
  done: (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#a3db52" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="m5 13 4 4L19 7" />
    </svg>
  ),
  failed: (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--red-text)" strokeWidth="2.4" strokeLinecap="round" aria-hidden="true">
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  ),
  pause: (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="7" y="5" width="3.6" height="14" rx="1" /><rect x="13.4" y="5" width="3.6" height="14" rx="1" />
    </svg>
  ),
  note: (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#878e89" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none' }} aria-hidden="true">
      <path d="M5 4h14v11l-5 5H5z" /><path d="M19 15h-5v5" />
    </svg>
  ),
}

/** Clock time, with the date when it is not today; durations and "ago" in the UI language. */
function useQueueFormat() {
  const { t, i18n } = useTranslation()
  const lang = i18n.language
  return useMemo(() => {
    const clock = (ts: number) => {
      const d = new Date(ts * 1000)
      const today = d.toDateString() === new Date().toDateString()
      return new Intl.DateTimeFormat(lang, today
        ? { hour: '2-digit', minute: '2-digit' }
        : { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }).format(d)
    }
    const dur = (sec: number) => {
      const s = Math.max(0, Math.round(sec))
      if (s < 60) return t('queue.dur.s', { n: s })
      const m = Math.round(s / 60)
      if (m < 60) return t('queue.dur.m', { n: m })
      return t('queue.dur.hm', { h: Math.floor(m / 60), m: String(m % 60).padStart(2, '0') })
    }
    const rtf = new Intl.RelativeTimeFormat(lang, { numeric: 'auto' })
    const ago = (ts: number) => {
      const s = Math.max(0, Date.now() / 1000 - ts)
      if (s < 60) return rtf.format(0, 'second')
      if (s < 3600) return rtf.format(-Math.floor(s / 60), 'minute')
      if (s < 86400) return rtf.format(-Math.floor(s / 3600), 'hour')
      return rtf.format(-Math.floor(s / 86400), 'day')
    }
    return { clock, dur, ago }
  }, [t, lang])
}

export default function QueuePage() {
  const { t } = useTranslation()
  const fmt = useQueueFormat()
  const [tasks, setTasks] = useState<Task[]>([])
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const reloadTimer = useRef<number | null>(null)
  const { toast } = useToast()
  const { confirm, prompt } = useDialog()
  const navigate = useNavigate()
  // 右键菜单（备注 / 采样条）：null = 关闭；否则记录锚点坐标 + 目标 task。
  const [menu, setMenu] = useState<{ x: number; y: number; task: Task } | null>(null)
  // Sample strips open per task on request (the mockup cards carry none by default).
  const [stripsShown, setStripsShown] = useState<ReadonlySet<number>>(new Set())
  // null = not chosen yet: active tasks when there are any, otherwise everything.
  const [pickedFilter, setFilter] = useState<ListFilter | null>(null)
  const [historyLimit, setHistoryLimit] = useState(HISTORY_STEP)
  const [dragId, setDragId] = useState<number | null>(null)
  const [overId, setOverId] = useState<number | null>(null)

  // ADR 0006：队列挂起状态，banner + holdModal 用。
  const [holdState, setHoldState] = useState<QueueHoldState | null>(null)
  const [holdModalOpen, setHoldModalOpen] = useState(false)
  const [pausingTaskId, setPausingTaskId] = useState<number | null>(null)
  // ADR 0006 Addendum 1 §UI：确认 modal 先于 PauseProgressModal。
  const [pauseConfirmTaskId, setPauseConfirmTaskId] = useState<number | null>(null)

  // Data tasks (download / preprocess / tagging / reg build / eval): same task
  // table, resource_class=data.
  const [dataLive, setDataLive] = useState<Task[]>([])
  const [dataHistory, setDataHistory] = useState<QueueHistoryPage | null>(null)
  const [projectTitles, setProjectTitles] = useState<Record<number, string>>({})

  const reloadHold = useCallback(async () => {
    try {
      setHoldState(await api.getQueueHold())
    } catch {
      // 网络错 / 启动期 supervisor 未就绪 → 静默；下一轮 SSE 触发重试。
      setHoldState(null)
    }
  }, [])

  const reloadData = useCallback(async () => {
    try {
      const [live, hist] = await Promise.all([
        api.listQueueLive(undefined, undefined, 'data'),
        api.listQueueHistory({ page: 1, pageSize: DATA_HISTORY, resourceClass: 'data' }),
      ])
      setDataLive(live); setDataHistory(hist)
    } catch {
      // The training list carries the page; a failed data fetch just leaves the table empty.
    }
  }, [])

  const reload = useCallback(async () => {
    try { setTasks(await api.listQueue()); setError(null) }
    catch (e) { setError(String(e)) }
    finally { setLoaded(true) }
    void reloadData()
  }, [reloadData])

  useEventStream(
    (evt) => {
      // ADR 0006 PR-4 — train_loop_started 不改 task.status 但要让 UI 看到
      // is_pausable=true（解锁暂停按钮）；queue_hold_changed 要刷 banner。
      if (
        evt.type === 'task_state_changed' ||
        evt.type === 'job_state_changed' ||
        evt.type === 'train_loop_started' ||
        evt.type === 'queue_hold_changed'
      ) {
        if (evt.type === 'queue_hold_changed') void reloadHold()
        if (reloadTimer.current) return
        reloadTimer.current = window.setTimeout(() => {
          reloadTimer.current = null; void reload()
        }, 100)
      }
    },
    { onOpen: () => { void reload(); void reloadHold() } },
  )

  useEffect(() => { void reload(); void reloadHold() }, [reload, reloadHold])
  useEffect(() => {
    api.listProjects()
      .then((items) => setProjectTitles(Object.fromEntries(items.map((p) => [p.id, p.title]))))
      .catch(() => {})
  }, [])
  // Clock tick while something runs: remaining times and "ago" labels move on
  // without any API call.
  useEffect(() => {
    if (!tasks.some((x) => x.status === 'running') && !dataLive.some((x) => x.status === 'running')) return
    const tick = window.setInterval(() => setTasks((ts) => [...ts]), 15_000)
    return () => window.clearInterval(tick)
  }, [tasks, dataLive])

  const runningTask = useMemo(() => tasks.find((x) => x.status === 'running') ?? null, [tasks])
  const { state: monitor } = useMonitorProgress(runningTask?.id ?? null)

  const forecast = useMemo(() => buildTrainingForecast(tasks, monitor), [tasks, monitor])
  const pendingOrder = useMemo(() => forecast.filter((f) => f.task.status === 'pending').map((f) => f.task), [forecast])
  const waitingCount = pendingOrder.length
  const queueFinish = forecast.length ? forecast[forecast.length - 1].finishesAt : null
  const startOf = useMemo(() => {
    const m = new Map<number, number | null>()
    let prev: number | null = Date.now() / 1000
    for (const f of forecast) { m.set(f.task.id, prev); prev = f.finishesAt }
    return m
  }, [forecast])
  const finishOf = useMemo(() => new Map(forecast.map((f) => [f.task.id, f.finishesAt])), [forecast])

  // Card order: the forecast (running, then waiting by priority), scheduled by
  // time, paused, then finished newest first.
  const ordered = useMemo(() => {
    const live = forecast.map((f) => f.task)
    const liveIds = new Set(live.map((x) => x.id))
    const scheduled = tasks.filter((x) => x.status === 'scheduled')
      .sort((a, b) => (a.scheduled_at ?? 0) - (b.scheduled_at ?? 0))
    const paused = tasks.filter((x) => x.status === 'paused').sort((a, b) => b.id - a.id)
    const rest = tasks.filter((x) => !liveIds.has(x.id) && x.status !== 'scheduled' && x.status !== 'paused' && !TERMINAL.has(x.status))
    const finished = tasks.filter((x) => TERMINAL.has(x.status))
      .sort((a, b) => (b.finished_at ?? b.created_at) - (a.finished_at ?? a.created_at) || b.id - a.id)
    return { active: [...live, ...scheduled, ...paused, ...rest], finished }
  }, [forecast, tasks])

  const filter: ListFilter = pickedFilter ?? (ordered.active.length ? 'active' : 'all')
  const visible = filter === 'active'
    ? ordered.active
    : filter === 'done' ? ordered.finished.slice(0, historyLimit)
    : [...ordered.active, ...ordered.finished.slice(0, historyLimit)]
  const hiddenHistory = filter === 'active' ? 0 : Math.max(0, ordered.finished.length - historyLimit)

  // ── pause / resume / cancel ─────────────────────────────────────────────
  const confirmPause = async () => {
    const taskId = pauseConfirmTaskId
    if (taskId === null) return
    setPauseConfirmTaskId(null)
    await pauseTask(taskId)
  }
  // hold-and-pause 走的快速路径（HoldQueueModal 内已 confirmed，跳过 PauseConfirmModal）
  const pauseTask = async (taskId: number) => {
    setPausingTaskId(taskId)
    try {
      await api.pauseTask(taskId)
      toast(t('queue.pauseSent'), 'success')
    } catch (e) {
      toast(t('queue.pauseFailed', { reason: String(e) }), 'error')
      setPausingTaskId(null)
    }
  }

  const resumeTask = async (task: Task) => {
    try {
      await api.resumeTask(task.id)
      toast(t('queue.resumeSent', { id: task.id }), 'success')
      await reload()
    } catch (e) {
      const msg = String(e)
      toast(msg.toLowerCase().includes('missing') ? t('queue.resumeFailedMissing') : t('queue.resumeFailed', { reason: msg }), 'error')
    }
  }

  const retryTask = async (task: Task) => {
    try {
      const next = await api.retryTask(task.id)
      toast(t('queue.retrySent', { id: next.id }), 'success')
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const cancelTask = async (task: Task) => {
    const message = task.status === 'running'
      ? t('queue.cancelRunningConfirm', { id: task.id })
      : task.status === 'paused'
        ? t('queue.cancelPausedConfirm', { id: task.id })
        : task.status === 'scheduled'
          ? t('queue.cancelScheduledConfirm', { id: task.id })
          : t('queue.cancelPendingConfirm', { id: task.id })
    const ok = await confirm(message, { tone: 'warn', okText: t('queue.cancelPaused') })
    if (!ok) return
    setBusy(true)
    try {
      await api.cancelTask(task.id)
      toast(t('queueDetail.cancelSent'), 'success')
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const deleteTask = async (task: Task) => {
    const ok = await confirm(t('queue.deleteConfirm', { id: task.id }), { tone: 'danger', okText: t('queue.delete') })
    if (!ok) return
    try {
      await api.deleteTask(task.id)
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const startNow = async (task: Task) => {
    const ok = await confirm(t('queue.startNowConfirm', { id: task.id }), { okText: t('queue.startNow') })
    if (!ok) return
    try {
      await api.startTaskNow(task.id)
      toast(t('queue.startNowSent', { id: task.id }), 'success')
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // ── priority: drag between waiting cards, or move one to the front ──────
  const applyOrder = async (ids: number[]) => {
    try {
      await api.reorderQueue(ids)
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }
  const moveFirst = (task: Task) => {
    void applyOrder([task.id, ...pendingOrder.filter((x) => x.id !== task.id).map((x) => x.id)])
  }
  const dropOn = (targetId: number) => {
    const from = dragId
    setDragId(null); setOverId(null)
    if (from == null || from === targetId) return
    const ids = pendingOrder.map((x) => x.id)
    const a = ids.indexOf(from); const b = ids.indexOf(targetId)
    if (a < 0 || b < 0) return
    ids.splice(a, 1); ids.splice(b, 0, from)
    void applyOrder(ids)
  }

  // ── 备注（_v20 tasks.note）───────────────────────────────────────────────
  const editNote = async (task: Task) => {
    const next = await prompt(t('queue.notePrompt', { id: task.id }), {
      title: task.note ? t('queue.noteEdit') : t('queue.noteAdd'),
      defaultValue: task.note ?? '',
      placeholder: t('queue.notePlaceholder'),
      validate: (v) => (v.length > MAX_NOTE_LEN ? t('queue.noteTooLong', { max: MAX_NOTE_LEN }) : null),
      okText: t('common.save'),
    })
    if (next === null) return
    await saveNote(task, next)
  }

  const saveNote = async (task: Task, note: string) => {
    try {
      const updated = await api.setTaskNote(task.id, note)
      setTasks((ts) => ts.map((x) => (x.id === task.id ? { ...x, note: updated.note } : x)))
      toast(updated.note ? t('queue.noteSaved') : t('queue.noteCleared'), 'success')
    } catch (e) {
      toast(t('queue.noteFailed', { reason: String(e) }), 'error')
    }
  }

  const toggleStrip = (taskId: number) => {
    setStripsShown((prev) => {
      const next = new Set(prev)
      if (next.has(taskId)) next.delete(taskId)
      else next.add(taskId)
      return next
    })
  }

  // Render the finished LoRA: the newest .safetensors in the task's output folder.
  const openRender = async (task: Task) => {
    try {
      const out = await api.getTaskOutputs(task.id)
      const lora = out.files.filter((f) => f.is_lora).sort((a, b) => b.mtime - a.mtime)[0]
      if (!lora || !out.output_dir) { toast(t('queue.noLora'), 'error'); return }
      const sp = new URLSearchParams({ lora: `${out.output_dir.replace(/[\\/]+$/, '')}/${lora.path}` })
      if (task.project_id) sp.set('projectId', String(task.project_id))
      if (task.version_id) sp.set('versionId', String(task.version_id))
      navigate(`/tools/generate?${sp.toString()}`)
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // ADR §4.4 hold 队列：弹 confirmation modal，根据 modal 内决策调 hold + 可选 pause
  const onHoldConfirm = async (decision: HoldDecision) => {
    setHoldModalOpen(false)
    try {
      await api.holdQueue()
      toast(t('queue.holdSet'), 'success')
    } catch (e) {
      toast(String(e), 'error')
      return
    }
    if (decision.kind === 'hold-and-pause') await pauseTask(decision.taskId)
    await reloadHold()
    await reload()
  }

  const releaseQueue = async () => {
    try {
      await api.releaseQueue()
      toast(t('queue.holdReleased'), 'success')
      await reloadHold()
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const noteItems = (task: Task): KebabItem[] => [
    { label: task.note ? t('queue.noteEdit') : t('queue.noteAdd'), onSelect: () => void editNote(task) },
    ...(task.note ? [{ label: t('queue.noteRemove'), tone: 'err' as const, onSelect: () => void saveNote(task, '') }] : []),
  ]
  const stripItem = (task: Task): KebabItem[] => (task.monitor_state_path
    ? [{ label: stripsShown.has(task.id) ? t('queue.samplesHide') : t('queue.samplesShow'), onSelect: () => toggleStrip(task.id) }]
    : [])

  const cardMenu = (task: Task): KebabItem[] => {
    const terminal = TERMINAL.has(task.status)
    return [
      { label: t('queue.taskDetailTooltip'), onSelect: () => navigate(`/queue/${task.id}`) },
      ...(task.project_id && task.version_id
        ? [{ label: t('queue.openConfig'), onSelect: () => navigate(`/projects/${task.project_id}/v/${task.version_id}/train`) }]
        : []),
      ...noteItems(task),
      ...stripItem(task),
      ...(task.status === 'running' && task.is_pausable
        ? [{ label: t('queue.pause'), onSelect: () => setPauseConfirmTaskId(task.id), disabled: pausingTaskId !== null }]
        : []),
      ...(terminal && task.is_resumable ? [{ label: t('queue.retry'), onSelect: () => void retryTask(task) }] : []),
      ...(!terminal ? [{ label: t('queue.cancelPaused'), tone: 'err' as const, onSelect: () => void cancelTask(task), disabled: busy }] : []),
      ...(terminal ? [{ label: t('queue.delete'), tone: 'err' as const, onSelect: () => void deleteTask(task) }] : []),
    ]
  }

  // ── summary card ────────────────────────────────────────────────────────
  const now = Date.now() / 1000
  const runPct = monitor?.step != null && monitor.total_steps ? Math.min(100, (monitor.step / monitor.total_steps) * 100) : null
  const runFinish = runningTask ? finishOf.get(runningTask.id) ?? null : null
  const segments = forecast.map((f) => {
    const start = startOf.get(f.task.id) ?? null
    const span = start != null && f.finishesAt != null ? Math.max(60, f.finishesAt - start) : null
    return { task: f.task, span, finishesAt: f.finishesAt }
  })
  const allSpans = segments.every((s) => s.span != null)

  return (
    <div className="fade-in">
      <PageHead
        eyebrow={t('queue.eyebrow')}
        title={t('queue.title')}
        subtitle={t('queue.description')}
        tools={
          <>
            {holdState && !holdState.held && (
              <button type="button" className="ds-ctl" onClick={() => setHoldModalOpen(true)} disabled={busy} data-testid="queue-hold-btn">
                {t('queue.holdQueue')}
              </button>
            )}
            {holdState?.held && (
              <button type="button" className="ds-ctl" onClick={() => void releaseQueue()} disabled={busy} data-testid="queue-release-btn">
                {t('queue.releaseQueue')}
              </button>
            )}
            <KebabMenu
              trigger="icon"
              label={t('queue.moreActions')}
              items={[
                {
                  label: t('queue.pause'),
                  onSelect: () => runningTask && setPauseConfirmTaskId(runningTask.id),
                  disabled: !runningTask?.is_pausable || pausingTaskId !== null || pauseConfirmTaskId !== null,
                },
                {
                  label: t('queue.cancelCurrent'),
                  tone: 'err',
                  onSelect: () => runningTask && void cancelTask(runningTask),
                  disabled: !runningTask || busy,
                },
                { label: t('common.refresh'), onSelect: () => void reload() },
              ]}
            />
          </>
        }
      />

      <div className="ds-scroll">
        {/* ADR §4.1 队列挂起 banner — 仅 held=true 时显示。 */}
        {holdState?.held && (
          <div className="ds-note ds-warn" style={{ alignItems: 'center' }} data-testid="queue-hold-banner">
            <span style={{ flex: 1 }}>{t('queue.heldBanner')}</span>
            <button type="button" className="ds-ctl ds-ghost" style={{ height: 26 }} onClick={() => void releaseQueue()}>
              {t('queue.releaseQueue')}
            </button>
          </div>
        )}
        {error && <div className="ds-note ds-warn font-mono">{error}</div>}

        <div className="ds-card" style={{ padding: '16px 18px 14px' }} data-testid="queue-summary">
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1px minmax(0,1fr) 1px auto', alignItems: 'stretch', gap: 26 }}>
            <div style={{ minWidth: 150 }}>
              <div className="ds-cap">{t('queue.summaryNow')}</div>
              <div style={{ fontSize: 26, fontWeight: 600, letterSpacing: '-.035em', marginTop: 6, lineHeight: 1 }}>
                {t('queue.summaryCount', { count: forecast.length })}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 8 }}>
                <span className="ds-legend"><s style={{ background: 'var(--green-600)' }} />{t('queue.legendRunning', { count: forecast.length - waitingCount })}</span>
                <span className="ds-legend"><s style={{ background: 'var(--line-3)' }} />{t('queue.legendWaiting', { count: waitingCount })}</span>
              </div>
            </div>
            <div style={{ background: 'var(--line)' }} />
            <div style={{ minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 9 }}>
                <span className="ds-cap" style={{ flex: 1 }}>{t('queue.planTitle')}</span>
                <span className="ds-kpi-meta">
                  {queueFinish ? t('queue.gpuBusyUntil', { time: fmt.clock(queueFinish) }) : forecast.length ? t('queue.summaryUnknown') : t('queue.gpuFree')}
                </span>
              </div>
              {segments.length === 0 ? (
                <div style={{ height: 30, borderRadius: 6, background: 'var(--sunken)', boxShadow: 'inset 0 0 0 1px var(--line-2)', display: 'flex', alignItems: 'center', padding: '0 10px', fontSize: 11, color: 'var(--ink-3)' }}>
                  {t('queue.planEmpty')}
                </div>
              ) : (
                <div style={{ display: 'flex', gap: 3, height: 30 }}>
                  {segments.map((s, i) => {
                    const first = i === 0
                    const last = i === segments.length - 1
                    const radius = `${first ? 6 : 3}px ${last ? 6 : 3}px ${last ? 6 : 3}px ${first ? 6 : 3}px`
                    const run = s.task.status === 'running'
                    const label = run
                      ? `#${s.task.id}`
                      : `#${s.task.id} · ${s.task.name}${s.span != null ? ` · ~${fmt.dur(s.span)}` : ''}`
                    return (
                      <button
                        key={s.task.id}
                        type="button"
                        onClick={() => navigate(`/queue/${s.task.id}`)}
                        title={s.finishesAt ? `${t('queue.summaryTaskFinish')}: ${fmt.clock(s.finishesAt)}` : t('queue.summaryUnknown')}
                        style={{
                          flex: allSpans ? `${s.span} 1 0` : '1 1 0', minWidth: 44, borderRadius: radius, position: 'relative', overflow: 'hidden',
                          background: run ? 'var(--green-soft)' : 'var(--sunken)',
                          boxShadow: `inset 0 0 0 1px ${run ? 'var(--green-line)' : 'var(--line-2)'}`,
                          textAlign: 'left',
                        }}
                      >
                        {run && <span style={{ position: 'absolute', top: 0, bottom: 0, left: 0, width: `${runPct ?? 0}%`, background: 'var(--green)' }} />}
                        <span style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', padding: '0 9px', fontSize: 10.5, fontFamily: 'var(--mono)', color: run ? 'var(--green-ink)' : 'var(--ink-3)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {label}
                        </span>
                      </button>
                    )
                  })}
                </div>
              )}
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 10.5, color: 'var(--ink-3)', fontFamily: 'var(--mono)' }}>
                <span>{t('queue.nowAt', { time: fmt.clock(now) })}</span>
                {segments.map((s) => <span key={s.task.id}>{s.finishesAt ? fmt.clock(s.finishesAt) : '—'}</span>)}
              </div>
            </div>
            <div style={{ background: 'var(--line)' }} />
            <div style={{ minWidth: 132 }}>
              <div className="ds-cap">{t('queue.currentFinish')}</div>
              <div style={{ fontSize: 19, fontWeight: 600, letterSpacing: '-.03em', marginTop: 6 }}>
                {runFinish ? `≈ ${fmt.clock(runFinish)}` : '—'}
              </div>
              <div className="ds-kpi-meta" style={{ marginTop: 3 }}>
                {runFinish
                  ? t('queue.remaining', { time: fmt.dur(runFinish - now) })
                  : runningTask ? t('queue.summaryUnknown') : t('queue.nothingRunning')}
              </div>
            </div>
          </div>
        </div>

        <div className="ds-card">
          <div className="ds-card-head ds-pad">
            <div style={{ flex: 1 }}>
              <div className="ds-card-title">{t('queue.trainTitle')}</div>
              <div className="ds-card-sub">{t('queue.trainSub')}</div>
            </div>
            <div className="ds-card-tools">
              <div className="ds-seg" role="tablist">
                {(['active', 'all', 'done'] as const).map((f) => (
                  <button
                    key={f}
                    type="button"
                    role="tab"
                    aria-selected={filter === f}
                    className={`ds-seg-item${filter === f ? ' ds-is-active' : ''}`}
                    onClick={() => { setFilter(f); setHistoryLimit(HISTORY_STEP) }}
                  >
                    {t(`queue.filter.${f}`)}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div style={{ padding: '0 17px 14px', display: 'flex', flexDirection: 'column', gap: 9 }}>
            {!loaded ? (
              Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="ds-qcard" style={{ height: 88, opacity: 0.4, background: 'var(--sunken)' }} />
              ))
            ) : visible.length === 0 ? (
              <div className="ds-empty">
                <b style={{ color: 'var(--ink-2)', fontWeight: 600 }}>
                  {tasks.length === 0 ? t('queue.empty') : filter === 'done' ? t('queue.noDone') : t('queue.noActive')}
                </b>
                <span>{t('queue.emptyHint')}</span>
              </div>
            ) : (
              visible.map((task) => (
                <QueueCard
                  key={task.id}
                  task={task}
                  monitor={task.id === runningTask?.id ? monitor : null}
                  startsAt={startOf.get(task.id) ?? null}
                  finishesAt={finishOf.get(task.id) ?? null}
                  ahead={pendingOrder.findIndex((x) => x.id === task.id) + (runningTask ? 1 : 0)}
                  held={holdState?.held === true}
                  fmt={fmt}
                  showStrip={stripsShown.has(task.id) && !!task.monitor_state_path}
                  draggable={task.status === 'pending' && pendingOrder.length > 1}
                  dragOver={overId === task.id && dragId !== task.id}
                  isFirstWaiting={pendingOrder[0]?.id === task.id}
                  menuItems={cardMenu(task)}
                  pauseDisabled={busy || pausingTaskId !== null || pauseConfirmTaskId !== null}
                  onOpen={() => navigate(`/queue/${task.id}`)}
                  onContextMenu={(x, y) => setMenu({ x, y, task })}
                  onPause={() => setPauseConfirmTaskId(task.id)}
                  onResume={() => void resumeTask(task)}
                  onRetry={() => void retryTask(task)}
                  onStartNow={() => void startNow(task)}
                  onMoveFirst={() => moveFirst(task)}
                  onMetrics={() => navigate(`/queue/${task.id}#monitor`)}
                  onRender={() => void openRender(task)}
                  onEditNote={() => void editNote(task)}
                  onDragStart={() => setDragId(task.id)}
                  onDragEnd={() => { setDragId(null); setOverId(null) }}
                  onDragOver={() => { if (dragId != null && task.status === 'pending') setOverId(task.id) }}
                  onDrop={() => dropOn(task.id)}
                />
              ))
            )}
            {hiddenHistory > 0 && (
              <button type="button" className="ds-ctl ds-ghost" style={{ alignSelf: 'center' }} onClick={() => setHistoryLimit((n) => n + HISTORY_STEP)}>
                {t('queue.showMore', { n: Math.min(HISTORY_STEP, hiddenHistory), total: hiddenHistory })}
              </button>
            )}
          </div>
        </div>

        <DataTasksCard
          live={dataLive}
          history={dataHistory?.items ?? []}
          projectTitles={projectTitles}
          fmt={fmt}
          onOpen={(task) => navigate(`/queue/${task.id}`)}
          onJump={(path) => navigate(path)}
          onCancel={(task) => void (async () => {
            const ok = await confirm(t('queue.jobs.cancelConfirm', { id: task.id }), { okText: t('queue.jobs.cancelOk') })
            if (!ok) return
            try {
              await api.cancelTask(task.id)
              toast(t('queueDetail.cancelSent'), 'success')
              await reloadData()
            } catch (e) {
              toast(String(e), 'error')
            }
          })()}
        />
      </div>

      {/* 右键菜单 —— 备注 + 采样条 + 详情。点任意处 / ESC 关。 */}
      {menu && (
        <TaskContextMenu
          x={menu.x}
          y={menu.y}
          onClose={() => setMenu(null)}
          items={[
            ...noteItems(menu.task),
            ...stripItem(menu.task),
            { label: t('queue.taskDetailTooltip'), onSelect: () => navigate(`/queue/${menu.task.id}`) },
          ]}
        />
      )}

      {/* ADR Addendum 1 §UI：暂停 confirm modal — 告知用户语义后才调 api。 */}
      {pauseConfirmTaskId !== null && (
        <PauseConfirmModal
          onCancel={() => setPauseConfirmTaskId(null)}
          onConfirm={() => void confirmPause()}
        />
      )}

      {/* ADR §4.3 暂停过程 modal — pausingTaskId 非 null 时全程锁屏。 */}
      {pausingTaskId !== null && (
        <PauseProgressModal
          taskId={pausingTaskId}
          taskName={tasks.find((x) => x.id === pausingTaskId)?.name}
          onClose={() => setPausingTaskId(null)}
        />
      )}

      {/* ADR §4.4 挂起 confirmation modal */}
      {holdModalOpen && (
        <HoldQueueModal
          runningTask={runningTask}
          onCancel={() => setHoldModalOpen(false)}
          onConfirm={onHoldConfirm}
        />
      )}
    </div>
  )
}

type Fmt = ReturnType<typeof useQueueFormat>

// ── QueueCard ───────────────────────────────────────────────────────────────
// One training task as the mockup's .qcard: title row (status, id, time,
// actions), progress row, note row; the sample strip opens on request.

function QueueCard({
  task, monitor, startsAt, finishesAt, ahead, held, fmt, showStrip, draggable, dragOver, isFirstWaiting,
  menuItems, pauseDisabled, onOpen, onContextMenu, onPause, onResume, onRetry, onStartNow, onMoveFirst,
  onMetrics, onRender, onEditNote, onDragStart, onDragEnd, onDragOver, onDrop,
}: {
  task: Task
  monitor: MonitorState | null
  startsAt: number | null
  finishesAt: number | null
  ahead: number
  held: boolean
  fmt: Fmt
  showStrip: boolean
  draggable: boolean
  dragOver: boolean
  isFirstWaiting: boolean
  menuItems: KebabItem[]
  pauseDisabled: boolean
  onOpen: () => void
  onContextMenu: (x: number, y: number) => void
  onPause: () => void
  onResume: () => void
  onRetry: () => void
  onStartNow: () => void
  onMoveFirst: () => void
  onMetrics: () => void
  onRender: () => void
  onEditNote: () => void
  onDragStart: () => void
  onDragEnd: () => void
  onDragOver: () => void
  onDrop: () => void
}) {
  const { t } = useTranslation()
  const now = Date.now() / 1000
  const s = task.status
  const run = s === 'running'
  const terminal = TERMINAL.has(s)

  const steps = run && monitor?.step != null && monitor.total_steps ? { step: monitor.step, total: monitor.total_steps } : null
  const pct = steps ? Math.min(100, Math.round((steps.step / steps.total) * 100)) : s === 'done' ? 100 : 0

  const badge = (() => {
    switch (s) {
      case 'running': return <span className="ds-badge ds-ok"><span className="ds-dot ds-dot-run" />{t('status.running')}</span>
      case 'pending': return <span className="ds-badge ds-warn">{held ? t('queue.waitingForRelease') : t('queue.waitingSlot', { n: ahead })}</span>
      case 'scheduled': return <span className="ds-badge ds-mute">{t('queue.scheduledAt', { time: task.scheduled_at ? fmt.clock(task.scheduled_at) : '—' })}</span>
      case 'paused': return <span className="ds-badge ds-warn">{t('status.paused')}</span>
      case 'done': return <span className="ds-badge ds-mute">{task.finished_at ? t('queue.doneAt', { time: fmt.clock(task.finished_at) }) : t('status.done')}</span>
      case 'failed': return <span className="ds-badge ds-err">{t('status.failed')}</span>
      default: return <span className="ds-badge ds-mute">{t('status.canceled')}</span>
    }
  })()

  const time = (() => {
    if (run) {
      return finishesAt
        ? <><b style={{ color: 'var(--green-text)' }}>~{fmt.dur(finishesAt - now)}</b><i>{t('queue.until', { time: fmt.clock(finishesAt) })}</i></>
        : <><b style={{ color: 'var(--green-text)' }}>{task.started_at ? fmt.dur(now - task.started_at) : '—'}</b><i>{t('queue.elapsed')}</i></>
    }
    if (s === 'pending') {
      return startsAt != null && finishesAt != null
        ? <><b>~{fmt.dur(finishesAt - startsAt)}</b><i>{t('queue.until', { time: fmt.clock(finishesAt) })}</i></>
        : <><b>—</b><i>{t('queue.summaryUnknown')}</i></>
    }
    if (s === 'paused') return <><b>{t('queue.stepN', { n: task.paused_step ?? 0 })}</b><i>{task.paused_at ? fmt.ago(task.paused_at) : ''}</i></>
    if (s === 'scheduled') return <><b>{task.scheduled_at ? fmt.clock(task.scheduled_at) : '—'}</b><i>{t('queue.scheduledStart')}</i></>
    if (terminal && task.started_at && task.finished_at) {
      return <><b>{fmt.dur(task.finished_at - task.started_at)}</b><i>{s === 'done' ? t('queue.finished') : fmt.ago(task.finished_at)}</i></>
    }
    return <><b>—</b><i>{task.finished_at ? fmt.ago(task.finished_at) : ''}</i></>
  })()

  const stop = (fn: () => void) => (e: React.MouseEvent) => { e.stopPropagation(); fn() }
  const actions = (() => {
    if (run) {
      return task.is_pausable
        ? <button type="button" onClick={stop(onPause)} disabled={pauseDisabled} title={t('queue.pauseHint')} data-testid="queue-pause-btn">{Icon.pause}{t('queue.pause')}</button>
        : null
    }
    if (s === 'pending') {
      return !isFirstWaiting ? <button type="button" onClick={stop(onMoveFirst)} title={t('queue.moveFirstHint')}>{t('queue.moveFirst')}</button> : null
    }
    if (s === 'scheduled') return <button type="button" onClick={stop(onStartNow)} title={t('queue.startNowHint')}>{t('queue.startNow')}</button>
    if (s === 'paused' || (terminal && task.is_resumable)) {
      return <button type="button" onClick={stop(onResume)} title={t('queue.resumeHint')} data-testid={`resume-btn-${task.id}`}>{t('queue.resume')}</button>
    }
    if (s === 'done') {
      return (
        <>
          <button type="button" onClick={stop(onMetrics)}>{t('queue.metrics')}</button>
          <button type="button" onClick={stop(onRender)}>{t('queue.jumpGenerate')}</button>
        </>
      )
    }
    return <button type="button" onClick={stop(onRetry)}>{t('queue.retry')}</button>
  })()

  const stats = (() => {
    if (run) {
      return (
        <>
          <b>{steps ? `${pct}%` : '—'}</b>
          {steps && <span>{steps.step.toLocaleString()} / {steps.total.toLocaleString()}</span>}
          {monitor?.epoch != null && monitor.total_epochs ? <span>{t('queue.epochOf', { n: monitor.epoch, total: monitor.total_epochs })}</span> : null}
          {monitor?.speed ? <span>{monitor.speed.toFixed(2)} it/s</span> : null}
        </>
      )
    }
    if (s === 'pending') return <><b>0%</b>{startsAt != null && finishesAt != null && <span>{t('queue.startAt', { time: fmt.clock(startsAt) })}</span>}</>
    if (s === 'paused') return task.last_state_epoch != null ? <span>{t('queue.resumeFromEpoch', { n: task.last_state_epoch })}</span> : null
    if (s === 'scheduled') return <b>0%</b>
    if (s === 'done') return <><b>100%</b>{task.finished_at && <span>{fmt.ago(task.finished_at)}</span>}</>
    return <b>{s === 'failed' ? t('status.failed') : t('status.canceled')}</b>
  })()

  return (
    <div
      className={`ds-qcard${run ? ' ds-is-run' : ''}${terminal ? ' ds-is-done' : ''}`}
      style={dragOver ? { boxShadow: '0 0 0 2px var(--green-600)' } : undefined}
      draggable={draggable}
      onDragStart={(e) => { e.dataTransfer.effectAllowed = 'move'; onDragStart() }}
      onDragEnd={onDragEnd}
      onDragOver={(e) => { if (s === 'pending') { e.preventDefault(); onDragOver() } }}
      onDrop={(e) => { e.preventDefault(); onDrop() }}
      onContextMenu={(e) => { e.preventDefault(); onContextMenu(e.clientX, e.clientY) }}
      data-testid={`queue-task-${task.id}`}
    >
      <div
        className="ds-qcard-top"
        role="link"
        tabIndex={0}
        style={{ cursor: 'pointer' }}
        onClick={onOpen}
        onKeyDown={(e) => {
          if (e.target !== e.currentTarget) return
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen() }
        }}
        title={t('queue.taskDetailTooltip')}
      >
        <span className="ds-qdrag" style={draggable ? { cursor: 'grab' } : undefined} title={draggable ? t('queue.dragHint') : undefined}>
          {s === 'done' ? Icon.done : s === 'failed' ? Icon.failed : Icon.drag}
        </span>
        <span className="ds-qtitle">
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{task.name}</span>
          {badge}
          <span className="ds-qid">#{task.id} · {task.config_name}</span>
        </span>
        <span className="ds-qtime">{time}</span>
        <span className="ds-actgroup" onClick={(e) => e.stopPropagation()}>
          {actions}
          <KebabMenu trigger="group" label={t('queue.moreActions')} items={menuItems} />
        </span>
      </div>
      <div className="ds-qcard-bar">
        <span className="ds-meter">
          {run && !steps
            ? <i className="animate-pulse" style={{ width: '20%', opacity: 0.5 }} />
            : <i style={{ width: `${pct}%`, ...(s === 'failed' ? { background: 'var(--red-text)' } : {}) }} />}
        </span>
        <span className="ds-qcard-stats">{stats}</span>
      </div>
      {task.error_msg && s === 'failed' && (
        <div className="ds-qcard-note" style={{ color: 'var(--red-text)' }}>
          <span className="font-mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={task.error_msg}>{task.error_msg}</span>
        </div>
      )}
      {task.note && (
        <div className="ds-qcard-note" data-testid={`task-note-${task.id}`}>
          {Icon.note}
          <span style={{ flex: 1, minWidth: 0, overflowWrap: 'anywhere' }}>{task.note}</span>
          <button type="button" className="ds-ctl ds-ghost" style={{ height: 22, fontSize: 11 }} onClick={onEditNote}>
            {t('queue.noteEdit')}
          </button>
        </div>
      )}
      {showStrip && <TaskSampleStrip taskId={task.id} live={run} />}
    </div>
  )
}

// ── DataTasksCard ───────────────────────────────────────────────────────────
// Download / preprocess / tagging / reg build / eval: they run next to
// training, so they get their own table under it.

function DataTasksCard({ live, history, projectTitles, fmt, onOpen, onJump, onCancel }: {
  live: Task[]
  history: Task[]
  projectTitles: Record<number, string>
  fmt: Fmt
  onOpen: (task: Task) => void
  onJump: (path: string) => void
  onCancel: (task: Task) => void
}) {
  const { t } = useTranslation()
  const rows = [...live, ...history.filter((h) => !live.some((l) => l.id === h.id))]
  const statusBadge = (task: Task) => {
    switch (task.status) {
      case 'running': return <span className="ds-badge ds-ok"><span className="ds-dot ds-dot-run" />{t('status.running')}</span>
      case 'pending': return <span className="ds-badge ds-warn">{t('status.queued')}</span>
      case 'done': return <span className="ds-badge ds-mute">{t('status.done')}</span>
      case 'failed': return <span className="ds-badge ds-err">{t('status.failed')}</span>
      case 'paused': return <span className="ds-badge ds-warn">{t('status.paused')}</span>
      default: return <span className="ds-badge ds-mute">{t('status.canceled')}</span>
    }
  }
  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div style={{ flex: 1 }}>
          <div className="ds-card-title">{t('queue.dataTitle')}</div>
          <div className="ds-card-sub">{t('queue.dataSub')}</div>
        </div>
      </div>
      {rows.length === 0 ? (
        <div style={{ padding: '0 17px 14px' }}>
          <div className="ds-empty"><span>{t('queue.jobs.empty')}</span></div>
        </div>
      ) : (
        <table className="ds-tbl">
          <thead>
            <tr>
              <th>{t('queue.col.task')}</th><th>{t('queue.col.project')}</th><th>{t('queue.col.status')}</th>
              <th>{t('queue.col.progress')}</th><th>{t('queue.col.started')}</th><th style={{ width: 40 }} />
            </tr>
          </thead>
          <tbody>
            {rows.map((task) => {
              const kind = task.task_type ?? 'train'
              const jump = jobJumpPath(task)
              const isLive = task.status === 'running' || task.status === 'pending'
              return (
                <tr key={task.id} style={{ cursor: 'pointer' }} onClick={() => onOpen(task)} data-testid={`job-row-${task.id}`}>
                  <td>
                    <span className="ds-cell-main">
                      <span className="ds-idchip">#{task.id}</span>
                      <span>{t(`queue.jobs.kind.${kind}`, { defaultValue: kind })}<span className="ds-cell-key">{kind}</span></span>
                    </span>
                  </td>
                  <td>{task.project_id != null ? (projectTitles[task.project_id] ?? `#${task.project_id}`) : '—'}</td>
                  <td>{statusBadge(task)}</td>
                  <td>
                    {task.status === 'running' ? (
                      <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span className="ds-meter" style={{ width: 120 }}><i className="animate-pulse" style={{ width: '35%' }} /></span>
                        <span className="ds-mono" style={{ fontSize: 11 }}>{task.started_at ? fmt.dur(Date.now() / 1000 - task.started_at) : ''}</span>
                      </span>
                    ) : task.started_at && task.finished_at ? (
                      <span className="ds-muted" style={{ fontSize: 11.5 }}>{fmt.dur(task.finished_at - task.started_at)}</span>
                    ) : <span className="ds-muted" style={{ fontSize: 11.5 }}>—</span>}
                  </td>
                  <td className="ds-num">{task.started_at ? fmt.clock(task.started_at) : '—'}</td>
                  <td onClick={(e) => e.stopPropagation()}>
                    <KebabMenu
                      label={t('queue.moreActions')}
                      items={[
                        { label: t('queue.taskDetailTooltip'), onSelect: () => onOpen(task) },
                        ...(jump ? [{ label: t('queue.jobs.jump'), onSelect: () => onJump(jump) }] : []),
                        ...(isLive ? [{ label: t('queue.jobs.cancel'), tone: 'err' as const, onSelect: () => onCancel(task) }] : []),
                      ]}
                    />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}

// ── TaskContextMenu ─────────────────────────────────────────────────────────
// 队列行右键弹出的小菜单。定位用 fixed + 视口边界夹紧（靠右/靠下的行不出屏）。
// 任何一次点击 / 滚动 / ESC 都关闭 —— 菜单本身的点击由条目 onSelect 先跑完。

function TaskContextMenu({ x, y, items, onClose }: {
  x: number
  y: number
  items: KebabItem[]
  onClose: () => void
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [pos, setPos] = useState({ x, y })

  // 挂载后按真实尺寸夹回视口内（菜单高度随条目数变，先渲染再量）。
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setPos({
      x: Math.min(x, window.innerWidth - r.width - 8),
      y: Math.min(y, window.innerHeight - r.height - 8),
    })
  }, [x, y])

  useEffect(() => {
    const close = () => onClose()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    // capture 阶段 + 下一帧注册：避免打开菜单的那次 contextmenu/click 立刻关掉它。
    const tid = window.setTimeout(() => {
      window.addEventListener('click', close)
      window.addEventListener('contextmenu', close)
      window.addEventListener('scroll', close, true)
    }, 0)
    window.addEventListener('keydown', onKey)
    return () => {
      window.clearTimeout(tid)
      window.removeEventListener('click', close)
      window.removeEventListener('contextmenu', close)
      window.removeEventListener('scroll', close, true)
      window.removeEventListener('keydown', onKey)
    }
  }, [onClose])

  return (
    <div
      ref={ref}
      role="menu"
      data-testid="queue-context-menu"
      className="fixed z-[80] py-1 rounded-[10px] border border-dim bg-elevated shadow-lg"
      style={{ left: pos.x, top: pos.y, minWidth: 180 }}
      onContextMenu={(e) => e.preventDefault()}
    >
      {items.map((it) => (
        <button
          key={it.label}
          type="button"
          role="menuitem"
          onClick={() => { it.onSelect(); onClose() }}
          className={`block w-full text-left whitespace-nowrap px-3 py-1.5 text-[12.5px] hover:bg-sunken ${it.tone === 'err' ? 'text-err' : 'text-fg-primary'}`}
        >
          {it.label}
        </button>
      ))}
    </div>
  )
}
