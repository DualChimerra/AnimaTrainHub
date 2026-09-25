import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate } from 'react-router-dom'
import {
  api, PHASE_ORDER,
  type BundleImportResult, type LoraCkpt, type ProjectSummary, type Task, type VersionPhase,
} from '../api/client'
import KebabMenu from '../components/ds/KebabMenu'
import PageHead from '../components/ds/PageHead'
import PathPicker from '../components/PathPicker'
import UploadProgressBar from '../components/UploadProgressBar'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import { splitOptional } from '../lib/labels'
import { useEventStream } from '../lib/useEventStream'
import { useUploadProgress } from '../lib/useUploadProgress'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import { buildTrainingForecast, latestProjectFinish } from '../lib/queueEstimates'

type ProjectSort = 'queue' | 'finish' | 'updated' | 'created' | 'title'
type ProjectFilter = 'all' | 'active' | 'archived'

const DAY = 86400

function formatProjectTime(ts: number): string {
  return new Intl.DateTimeFormat(undefined, {
    day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
  }).format(new Date(ts * 1000))
}

function formatClock(ts: number): string {
  return new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' }).format(new Date(ts * 1000))
}

/** Step key for each version phase (sidebar order, ADR 0010). */
const PHASE_STEP: Record<VersionPhase, { key: string; labelKey: string }> = {
  curating:      { key: 'curate',     labelKey: 'nav.curate' },
  preprocessing: { key: 'preprocess', labelKey: 'nav.preprocess' },
  editing:       { key: 'edit',       labelKey: 'nav.tagEdit' },
  regularizing:  { key: 'reg',        labelKey: 'nav.reg' },
  ready:         { key: 'train',      labelKey: 'nav.train' },
}

const isTrain = (task: Task) => (task.task_type ?? 'train') === 'train'

/** Seconds a training task spent on the GPU inside [from, now]. */
function gpuSeconds(task: Task, from: number, now: number): number {
  if (!task.started_at) return 0
  const end = task.finished_at ?? (task.status === 'running' ? now : task.paused_at ?? null)
  if (end == null) return 0
  return Math.max(0, end - Math.max(task.started_at, from))
}

const Icon = {
  plus: <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>,
  search: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" /></svg>,
  folder: (s = 15) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={s === 16 ? 1.8 : 1.7} strokeLinecap="round" strokeLinejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /></svg>,
  queue: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"><path d="M4 6h16M4 12h10M4 18h16" /><circle cx="18" cy="12" r="2" fill="currentColor" /></svg>,
  clock: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3.5 2" /></svg>,
  hex: (s = 15) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={s === 16 ? 1.8 : 1.7} strokeLinecap="round" strokeLinejoin="round"><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z" /></svg>,
  arrow: <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h13m-5-6 6 6-6 6" /></svg>,
}

interface ProjectExtras { versions: number | null; ckpts: LoraCkpt[] }
interface QueueInfo { order: number; finishesAt: number | null; startsAt: number | null; status: Task['status']; taskId: number }

export default function ProjectsPage() {
  const { t } = useTranslation()
  const [items, setItems] = useState<ProjectSummary[]>([])
  const [tasks, setTasks] = useState<Task[]>([])
  const [extras, setExtras] = useState<Record<number, ProjectExtras>>({})
  const [sort, setSort] = useState<ProjectSort>('queue')
  const [filter, setFilter] = useState<ProjectFilter>('all')
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [busy, setBusy] = useState(false)
  const [importing, setImporting] = useState(false)
  const [showImportDialog, setShowImportDialog] = useState(false)
  const [showImportPicker, setShowImportPicker] = useState(false)
  const navigate = useNavigate()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const uploadProgress = useUploadProgress()

  const refresh = async () => {
    try {
      const [list, queue] = await Promise.all([
        api.listProjects(),
        api.listQueue().catch(() => [] as Task[]),
      ])
      setItems(list)
      setTasks(queue)
      setError(null)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void refresh() }, [])

  useEventStream((evt) => {
    if (evt.type === 'project_state_changed' || evt.type === 'task_state_changed') void refresh()
  })

  // Version counts and LoRA checkpoints per project (cards + KPI). Refetched
  // only when a project actually changed, not on every queue event.
  const extrasKey = items.map((p) => `${p.id}:${p.updated_at}`).join(',')
  useEffect(() => {
    let cancelled = false
    const ids = extrasKey ? extrasKey.split(',').map((s) => Number(s.split(':')[0])) : []
    void Promise.all(ids.map(async (id) => {
      const [detail, groups] = await Promise.all([
        api.getProject(id).catch(() => null),
        api.listProjectLoraCkpts(id).catch(() => []),
      ])
      return [id, { versions: detail ? detail.versions.length : null, ckpts: groups.flatMap((g) => g.items) }] as const
    })).then((entries) => { if (!cancelled) setExtras(Object.fromEntries(entries)) })
    return () => { cancelled = true }
  }, [extrasKey])

  const runningTask = tasks.find((task) => task.status === 'running' && isTrain(task)) ?? null
  const runningTaskId = tasks.find((task) => task.status === 'running')?.id ?? null
  const { state: monitor } = useMonitorProgress(runningTaskId)
  const forecast = useMemo(() => buildTrainingForecast(tasks, monitor), [tasks, monitor])
  const forecastByProject = useMemo(() => {
    const map = new Map<number, QueueInfo>()
    forecast.forEach(({ task, finishesAt }, order) => {
      if (task.project_id != null && !map.has(task.project_id)) {
        const startsAt = order === 0 ? (task.started_at ?? null) : forecast[order - 1].finishesAt
        map.set(task.project_id, { order, finishesAt, startsAt, status: task.status, taskId: task.id })
      }
    })
    return map
  }, [forecast])

  const visibleItems = useMemo(() => {
    const q = query.trim().toLowerCase()
    return items.filter((p) => {
      if (filter === 'active' && p.archived_at != null) return false
      if (filter === 'archived' && p.archived_at == null) return false
      if (!q) return true
      return [p.title, p.slug, p.note ?? ''].some((s) => s.toLowerCase().includes(q))
    })
  }, [items, filter, query])

  const sortedItems = useMemo(() => [...visibleItems].sort((a, b) => {
    const aq = forecastByProject.get(a.id)
    const bq = forecastByProject.get(b.id)
    if (sort === 'queue') {
      if (aq && bq) return aq.order - bq.order
      if (aq) return -1
      if (bq) return 1
      return b.updated_at - a.updated_at
    }
    if (sort === 'finish') {
      if (aq || bq) {
        if (!aq) return 1
        if (!bq) return -1
        if (aq.finishesAt != null && bq.finishesAt != null) return aq.finishesAt - bq.finishesAt
        if (aq.finishesAt != null) return -1
        if (bq.finishesAt != null) return 1
        return aq.order - bq.order
      }
      const af = latestProjectFinish(tasks, a.id)
      const bf = latestProjectFinish(tasks, b.id)
      if (af != null && bf != null) return bf - af
      if (af != null) return -1
      if (bf != null) return 1
      return b.updated_at - a.updated_at
    }
    if (sort === 'created') return b.created_at - a.created_at
    if (sort === 'title') return a.title.localeCompare(b.title)
    return b.updated_at - a.updated_at
  }), [visibleItems, tasks, forecastByProject, sort])

  // ── KPI row (all from real data) ─────────────────────────────────────────
  const now = Date.now() / 1000
  const activeProjects = items.filter((p) => p.archived_at == null)
  const newThisWeek = activeProjects.filter((p) => p.created_at > now - 7 * DAY).length
  const trainQueued = tasks.filter((task) => isTrain(task) && (task.status === 'pending' || task.status === 'scheduled')).length
  const trainRunning = tasks.filter((task) => isTrain(task) && task.status === 'running').length
  const gpuHours = (days: number) =>
    tasks.filter(isTrain).reduce((sum, task) => sum + gpuSeconds(task, now - days * DAY, now), 0) / 3600
  const activeIds = new Set(activeProjects.map((p) => p.id))
  const allCkpts = Object.entries(extras).filter(([id]) => activeIds.has(Number(id))).flatMap(([, e]) => e.ckpts)
  const ckptsThisWeek = allCkpts.filter((c) => c.mtime > now - 7 * DAY).length

  // ── live tasks for "what is happening right now" ──────────────────────────
  const liveTasks = useMemo(() => {
    const order = new Map(forecast.map(({ task }, i) => [task.id, i]))
    const rank = (task: Task) => (task.status === 'running' ? 0 : task.status === 'pending' ? 1 : task.status === 'scheduled' ? 2 : 3)
    return tasks
      .filter((task) => task.status === 'running' || task.status === 'pending' || task.status === 'scheduled' || task.status === 'paused')
      .sort((a, b) => rank(a) - rank(b) || (order.get(a.id) ?? 99) - (order.get(b.id) ?? 99) || a.created_at - b.created_at)
  }, [tasks, forecast])

  const handleCreate = async (form: NewProjectForm) => {
    setBusy(true)
    try {
      const p = await api.createProject({
        title: form.title,
        note: form.note || undefined,
        initial_version_label: form.initial_version_label || 'v1',
      })
      toast(t('projects.created', { title: p.title }), 'success')
      setCreating(false)
      navigate(`/projects/${p.id}`)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const handleDelete = async (p: ProjectSummary) => {
    if (!(await confirm(
      t('projects.deleteConfirm', { title: p.title, slug: p.slug }),
      { tone: 'danger', okText: t('projects.deleteProject') },
    ))) return
    try {
      await api.deleteProject(p.id)
      toast(t('projects.deleted', { title: p.title }), 'success')
      await refresh()
    } catch (err) {
      toast(String(err), 'error')
    }
  }

  const handleArchive = async (p: ProjectSummary) => {
    const archived = p.archived_at != null
    if (!(await confirm(
      archived ? t('projects.unarchiveConfirm', { title: p.title }) : t('projects.archiveConfirm', { title: p.title }),
      { okText: archived ? t('projects.unarchiveTitle') : t('projects.archiveBtn') },
    ))) return
    try {
      if (archived) await api.unarchiveProject(p.id)
      else await api.archiveProject(p.id)
      toast(archived ? t('projects.unarchived', { title: p.title }) : t('projects.archived', { title: p.title }), 'success')
      await refresh()
    } catch (err) {
      toast(String(err), 'error')
    }
  }

  const finishBundleImport = (result: BundleImportResult) => {
    const stats = result.stats
    toast(
      t('projects.importedBundle', {
        title: result.project.title,
        train_count: stats.train_image_count,
        reg_count: stats.reg_image_count,
        preset_count: stats.preset_count,
      }),
      'success',
    )
    navigate(`/projects/${result.project.id}`)
  }

  const runBundleImport = async (job: () => Promise<BundleImportResult>) => {
    setImporting(true)
    try {
      finishBundleImport(await job())
    } catch (e) {
      toast(t('projects.importFailed', { e }), 'error')
    } finally {
      setImporting(false)
    }
  }

  const handleImportPath = async (path: string) => {
    setShowImportPicker(false)
    await runBundleImport(() => api.importBundleFromPath(path))
  }

  const handleImportUpload = async (file: File | null | undefined) => {
    if (!file) return
    // dialog 不立即关：进度条要显示在 dialog 里直到完成 / 失败
    uploadProgress.start(file.size)
    setImporting(true)
    try {
      const result = await api.importBundleUpload(file, uploadProgress.onProgress)
      uploadProgress.finish()
      setShowImportDialog(false)
      uploadProgress.reset()
      finishBundleImport(result)
    } catch (e) {
      uploadProgress.fail(e)
      toast(t('projects.importFailed', { e }), 'error')
    } finally {
      setImporting(false)
    }
  }

  const projectsById = new Map(items.map((p) => [p.id, p]))
  const trainPct = runningTask && monitor?.step != null && monitor.total_steps
    ? Math.round((monitor.step / monitor.total_steps) * 100) : null

  return (
    <div className="fade-in">
      <PageHead
        eyebrow={t('projects.eyebrow')}
        title={t('projects.title')}
        subtitle={t('projects.subtitle')}
        tools={
          <>
            <button
              type="button"
              className="ds-ctl"
              onClick={() => setShowImportDialog(true)}
              disabled={importing}
              title={importing ? t('projects.importing') : t('projects.importZipHint')}
            >
              {importing ? t('projects.importing') : t('projects.importZip')}
            </button>
            <button type="button" className="ds-btn-primary" onClick={() => setCreating(true)}>
              {Icon.plus}{t('projects.newProject')}
            </button>
          </>
        }
      />

      <div className="ds-scroll">
        {error && <div className="ds-note ds-warn font-mono">{error}</div>}

        <div className="ds-toolbar">
          <span className="ds-search" style={{ flex: 1, minWidth: 0 }}>
            {Icon.search}
            <input
              className="ds-inp"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t('projects.searchPlaceholder')}
              aria-label={t('projects.searchPlaceholder')}
            />
          </span>
          <select
            aria-label={t('projects.sortLabel')}
            className="ds-inp"
            style={{ width: 210, flex: 'none', height: 34 }}
            value={sort}
            onChange={(event) => setSort(event.target.value as ProjectSort)}
          >
            <option value="queue">{t('projects.sort_queue')}</option>
            <option value="finish">{t('projects.sort_finish')}</option>
            <option value="updated">{t('projects.sort_updated')}</option>
            <option value="created">{t('projects.sort_created')}</option>
            <option value="title">{t('projects.sort_title')}</option>
          </select>
          <div className="ds-seg" style={{ flex: 'none' }} role="group">
            {(['all', 'active', 'archived'] as const).map((f) => (
              <button
                key={f}
                type="button"
                aria-pressed={filter === f}
                className={`ds-seg-item${filter === f ? ' ds-is-active' : ''}`}
                onClick={() => setFilter(f)}
              >
                {t(f === 'all' ? 'projects.filterAll' : f === 'active' ? 'projects.filterActive' : 'projects.filterArchived')}
              </button>
            ))}
          </div>
        </div>

        <div className="ds-kpis ds-tight" style={{ gridTemplateColumns: 'repeat(4,minmax(0,1fr))' }}>
          <div className="ds-card ds-kpi">
            <div className="ds-kpi-top">
              <span className="ds-kpi-icon">{Icon.folder()}</span>
              {newThisWeek > 0 && <span style={{ marginLeft: 'auto' }}><span className="ds-delta ds-up">{t('projects.kpiWeek', { n: newThisWeek })}</span></span>}
            </div>
            <div className="ds-kpi-val">{activeProjects.length}</div>
            <div className="ds-kpi-label">{t('projects.kpiActive')}</div>
          </div>
          <div className="ds-card ds-kpi">
            <div className="ds-kpi-top">
              <span className="ds-kpi-icon">{Icon.queue}</span>
              <span style={{ marginLeft: 'auto' }}><span className="ds-kpi-meta">{t('projects.kpiRunning', { n: trainRunning })}</span></span>
            </div>
            <div className="ds-kpi-val">{trainQueued}<small>{t('projects.kpiInQueue')}</small></div>
            <div className="ds-kpi-label">{t('projects.kpiTrainTasks')}</div>
          </div>
          <div className="ds-card ds-kpi">
            <div className="ds-kpi-top">
              <span className="ds-kpi-icon">{Icon.clock}</span>
              <span style={{ marginLeft: 'auto' }}><span className="ds-kpi-meta">{t('projects.kpiGpuWeek', { h: gpuHours(7).toFixed(1) })}</span></span>
            </div>
            <div className="ds-kpi-val">{gpuHours(30).toFixed(1)}<small>{t('projects.kpiHours')}</small></div>
            <div className="ds-kpi-label">{t('projects.kpiGpu')}</div>
          </div>
          <div className="ds-card ds-kpi">
            <div className="ds-kpi-top">
              <span className="ds-kpi-icon">{Icon.hex()}</span>
              {ckptsThisWeek > 0 && <span style={{ marginLeft: 'auto' }}><span className="ds-delta ds-up">+{ckptsThisWeek}</span></span>}
            </div>
            <div className="ds-kpi-val">{allCkpts.length}</div>
            <div className="ds-kpi-label">{t('projects.kpiCkpts')}</div>
          </div>
        </div>

        {loading ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,minmax(0,1fr))', gap: 14 }}>
            {[1, 2, 3].map((i) => (
              <div key={i} className="ds-card" style={{ height: 176, padding: '15px 16px' }}>
                <div className="w-3/5 h-4 rounded bg-sunken mb-2.5" />
                <div className="w-2/5 h-[11px] rounded-sm bg-sunken" />
              </div>
            ))}
          </div>
        ) : items.length === 0 ? (
          <div className="ds-empty">
            <span style={{ fontWeight: 500, color: 'var(--ink-2)' }}>{t('projects.noProjects')}</span>
            <span>{t('projects.noProjectsHint')}</span>
          </div>
        ) : sortedItems.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,minmax(0,1fr))', gap: 14 }}>
            {sortedItems.map((p) => (
              <ProjectCard
                key={p.id}
                project={p}
                extras={extras[p.id]}
                queueInfo={forecastByProject.get(p.id)}
                progress={runningTask?.project_id === p.id && monitor?.step != null && monitor.total_steps
                  ? { step: monitor.step, total: monitor.total_steps, pct: trainPct ?? 0 } : null}
                finishedAt={latestProjectFinish(tasks, p.id)}
                onArchive={() => void handleArchive(p)}
                onDelete={() => void handleDelete(p)}
              />
            ))}
          </div>
        ) : (
          <div className="ds-empty">{t('projects.noMatches')}</div>
        )}

        <div className="ds-card">
          <div className="ds-card-head ds-pad">
            <div>
              <div className="ds-card-title">{t('projects.nowTitle')}</div>
              <div className="ds-card-sub">{t('projects.nowSub')}</div>
            </div>
            <div className="ds-card-tools">
              <Link className="ds-ctl" to="/queue">{t('projects.nowAll')}{Icon.arrow}</Link>
            </div>
          </div>
          {liveTasks.length === 0 ? (
            <div className="ds-card-body" style={{ paddingTop: 0 }}>
              <span className="ds-kpi-meta">{t('projects.nowEmpty')}</span>
            </div>
          ) : (
            <table className="ds-tbl">
              <thead>
                <tr>
                  <th>{t('projects.colTask')}</th><th>{t('projects.colProject')}</th><th>{t('projects.colStatus')}</th>
                  <th>{t('projects.colProgress')}</th><th>{t('projects.colLeft')}</th><th style={{ width: 44 }} />
                </tr>
              </thead>
              <tbody>
                {liveTasks.map((task) => (
                  <LiveTaskRow
                    key={task.id}
                    task={task}
                    project={task.project_id != null ? projectsById.get(task.project_id) ?? null : null}
                    forecast={forecast}
                    runningPct={task.id === runningTaskId ? trainPct : null}
                    onOpen={() => navigate(`/queue/${task.id}`)}
                  />
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {creating && (
        <NewProjectDialog
          busy={busy}
          onCancel={() => setCreating(false)}
          onSubmit={handleCreate}
        />
      )}

      {showImportDialog && (
        <BundleImportDialog
          importing={importing}
          uploadState={uploadProgress.state}
          onUpload={handleImportUpload}
          onPickPath={() => {
            setShowImportDialog(false)
            setShowImportPicker(true)
          }}
          onCancel={() => {
            setShowImportDialog(false)
            uploadProgress.reset()
          }}
        />
      )}

      {showImportPicker && (
        <PathPicker
          dirOnly={false}
          onClose={() => setShowImportPicker(false)}
          onPick={(path) => { void handleImportPath(path) }}
        />
      )}
    </div>
  )
}

/** One row of "what is happening right now": task, project, status, progress, time left. */
function LiveTaskRow({ task, project, forecast, runningPct, onOpen }: {
  task: Task
  project: ProjectSummary | null
  forecast: ReturnType<typeof buildTrainingForecast>
  runningPct: number | null
  onOpen: () => void
}) {
  const { t } = useTranslation()
  const now = Date.now() / 1000
  const kind = task.task_type ?? 'train'
  const kindKey = kind.startsWith('eval_') ? 'eval' : kind
  const idx = forecast.findIndex((f) => f.task.id === task.id)
  const entry = idx >= 0 ? forecast[idx] : null
  const startsAt = idx > 0 ? forecast[idx - 1].finishesAt : null

  const badge = task.status === 'running'
    ? <span className="ds-badge ds-ok"><span className="ds-dot ds-dot-run" />{t('status.running')}</span>
    : task.status === 'paused'
      ? <span className="ds-badge ds-mute">{t('status.paused')}</span>
      : <span className="ds-badge ds-warn">{t(task.status === 'scheduled' ? 'status.scheduled' : 'status.pending')}</span>

  let progress: React.ReactNode = <span className="ds-muted" style={{ fontSize: 11.5 }}>—</span>
  if (task.status === 'running' && runningPct != null) {
    progress = (
      <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span className="ds-meter" style={{ width: 120 }}><i style={{ width: `${runningPct}%` }} /></span>
        <span className="ds-mono" style={{ fontSize: 11 }}>{runningPct}%</span>
      </span>
    )
  } else if (task.status === 'pending' && idx > 0) {
    progress = <span className="ds-muted" style={{ fontSize: 11.5 }}>{t('projects.ahead', { count: idx })}</span>
  } else if (task.status === 'paused' && task.paused_step != null) {
    progress = <span className="ds-muted" style={{ fontSize: 11.5 }}>{t('projects.pausedAt', { step: task.paused_step.toLocaleString() })}</span>
  }

  let left = '—'
  if (task.status === 'running' && entry?.finishesAt) {
    const mins = Math.max(1, Math.round((entry.finishesAt - now) / 60))
    left = mins < 60 ? t('projects.leftMin', { m: mins }) : t('projects.leftHours', { h: Math.floor(mins / 60), m: mins % 60 })
  } else if ((task.status === 'pending') && startsAt) {
    left = t('projects.startAt', { time: formatClock(startsAt) })
  } else if (task.status === 'scheduled' && task.scheduled_at) {
    left = t('projects.startAt', { time: formatProjectTime(task.scheduled_at) })
  }

  return (
    <tr>
      <td>
        <span className="ds-cell-main">
          <span className="ds-idchip">#{task.id}</span>
          <span>{t(`projects.taskKind.${kindKey}`, { defaultValue: kind })}<span className="ds-cell-key">{task.config_name || task.name}</span></span>
        </span>
      </td>
      <td>{project?.title ?? '—'}</td>
      <td>{badge}</td>
      <td>{progress}</td>
      <td className="ds-num">{left}</td>
      <td>
        <KebabMenu label={t('projects.taskActions')} items={[{ label: t('projects.openTask'), onSelect: onOpen }]} />
      </td>
    </tr>
  )
}

function BundleImportDialog({
  importing,
  uploadState,
  onUpload,
  onPickPath,
  onCancel,
}: {
  importing: boolean
  uploadState: ReturnType<typeof useUploadProgress>['state']
  onUpload: (file: File | null | undefined) => void
  onPickPath: () => void
  onCancel: () => void
}) {
  const { t } = useTranslation()

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-40 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel() }}
    >
      <div className="bg-elevated border border-subtle rounded-2xl w-[90%] max-w-[560px] p-6 flex flex-col gap-4 shadow-xl">
        <div>
          <h2 className="m-0 text-lg font-semibold text-fg-primary">
            {t('projects.importBundleTitle')}
          </h2>
          <p className="mt-1 mb-0 text-sm text-fg-secondary">
            {t('projects.importBundleHint')}
          </p>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          <label className={`card p-4 cursor-pointer ${importing ? 'opacity-60 pointer-events-none' : ''}`}>
            <div className="font-medium text-fg-primary mb-1">{t('projects.importUpload')}</div>
            <div className="text-xs text-fg-tertiary mb-3">{t('projects.importUploadHint')}</div>
            <input
              type="file"
              accept=".zip,application/zip"
              className="text-xs text-fg-secondary w-full"
              disabled={importing}
              onChange={(e) => onUpload(e.target.files?.[0])}
            />
            {uploadState.phase !== 'idle' && (
              <UploadProgressBar state={uploadState} className="mt-3" />
            )}
          </label>

          <button
            type="button"
            className="card p-4 text-left cursor-pointer hover:border-dim disabled:opacity-60"
            disabled={importing}
            onClick={onPickPath}
          >
            <div className="font-medium text-fg-primary mb-1">{t('projects.importPath')}</div>
            <div className="text-xs text-fg-tertiary">{t('projects.importPathHint')}</div>
          </button>
        </div>

        <div className="flex justify-end">
          <button type="button" className="btn btn-secondary" onClick={onCancel} disabled={importing}>
            {t('common.cancel')}
          </button>
        </div>
      </div>
    </div>
  )
}


/** Project card from the mockup: tile + name + slug·version + ⋯, status
 *  badges, the 5-step progress bar with a caption, and the continue action. */
function ProjectCard({ project: p, extras, queueInfo, progress, finishedAt, onArchive, onDelete }: {
  project: ProjectSummary
  extras?: ProjectExtras
  queueInfo?: QueueInfo
  progress: { step: number; total: number; pct: number } | null
  finishedAt: number | null
  onArchive: () => void
  onDelete: () => void
}) {
  const { t } = useTranslation()
  const status = p.active_version_status
  // The phase cursor only means something while the version is being
  // prepared; once it trains (or trained) it sits on step 5.
  const phase: VersionPhase = status === 'preparing' || status == null ? (p.active_version_phase ?? 'curating') : 'ready'
  const phaseIdx = Math.max(0, PHASE_ORDER.indexOf(phase))
  const running = queueInfo?.status === 'running'
  const queued = queueInfo != null && !running
  const trainingNow = running || status === 'training'
  const stepLabel = (ph: VersionPhase) => splitOptional(t(PHASE_STEP[ph].labelKey)).short.toLowerCase()

  // 5 segments: finished phases full, the training step follows live progress.
  const segments = [0, 1, 2, 3, 4].map((i) => {
    if (status === 'completed') return 100
    if (trainingNow || queued) return i < 4 ? 100 : (i === 4 && progress ? progress.pct : 0)
    return i < phaseIdx ? 100 : 0 // preparing: finished phases; after preparing: 4 of 5
  })

  let caption = ''
  let captionRight = ''
  if (running) {
    caption = t('projects.stepOf', { n: 5, step: stepLabel('ready') })
    captionRight = progress ? `${progress.step.toLocaleString()} / ${progress.total.toLocaleString()}` : ''
  } else if (queued) {
    caption = t('projects.stepOf', { n: 5, step: t('projects.waitingSlot') })
    captionRight = queueInfo?.startsAt ? t('projects.startAt', { time: formatClock(queueInfo.startsAt) }) : ''
  } else if (status === 'completed') {
    caption = finishedAt ? t('projects.doneAt', { when: formatProjectTime(finishedAt) }) : t('versionStatus.completed')
    captionRight = extras ? t('projects.ckpts', { count: extras.ckpts.length }) : ''
  } else if (p.active_version_id != null) {
    caption = t('projects.stepOf', { n: phaseIdx + 1, step: stepLabel(phase) })
  }

  const badge = running
    ? <span className="ds-badge ds-ok"><span className="ds-dot ds-dot-run" />{t('projects.trainingNow')}</span>
    : queued
      ? <span className="ds-badge ds-warn">{t('projects.queuePosition', { position: (queueInfo?.order ?? 0) + 1 })}</span>
      : status === 'completed'
        ? <span className="ds-badge ds-ok">{t('projects.modelReady')}</span>
        : status === 'failed'
          ? <span className="ds-badge ds-err">{t('versionStatus.failed')}</span>
          : status
            ? <span className="ds-badge ds-mute">{t(`versionStatus.${status}`)}</span>
            : null

  const vid = p.active_version_id
  const action = running || (vid != null && phase === 'ready' && status !== 'completed' && !queued)
    ? { to: `/projects/${p.id}/v/${vid}/train`, label: t('projects.continueStep', { step: stepLabel('ready') }) }
    : queued
      ? { to: '/queue', label: t('projects.continueQueue') }
      : status === 'completed'
        ? { to: '/tools/generate', label: t('projects.viewResult') }
        : vid != null
          ? { to: `/projects/${p.id}/v/${vid}/${PHASE_STEP[phase].key}`, label: t('projects.continueStep', { step: stepLabel(phase) }) }
          : { to: `/projects/${p.id}`, label: t('projects.openProject') }

  const tileStyle: React.CSSProperties = { width: 34, height: 34, borderRadius: 10 }
  if (trainingNow) Object.assign(tileStyle, { background: 'var(--green-soft)', color: 'var(--green-text)' })

  return (
    <div className="ds-card" style={{ padding: '15px 16px 14px', display: 'flex', flexDirection: 'column', gap: 11, opacity: p.archived_at != null ? 0.72 : 1 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
        <Link to={`/projects/${p.id}`} style={{ display: 'flex', alignItems: 'flex-start', gap: 10, flex: 1, minWidth: 0 }}>
          <span className="ds-sect-icon" style={tileStyle}>{trainingNow ? Icon.hex(16) : Icon.folder(16)}</span>
          <span style={{ flex: 1, minWidth: 0 }}>
            <span style={{ fontSize: 14, fontWeight: 600, letterSpacing: '-.02em', display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.title}</span>
            <span className="ds-cell-key">{p.slug}{p.active_version_label ? ` · ${p.active_version_label}` : ''}</span>
          </span>
        </Link>
        <KebabMenu
          label={t('projects.actions')}
          items={[
            { label: p.archived_at != null ? t('projects.unarchiveTitle') : t('projects.archiveBtn'), onSelect: onArchive },
            { label: t('projects.deleteProject'), onSelect: onDelete, tone: 'err' },
          ]}
        />
      </div>

      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {badge}
        <span className="ds-badge ds-mute">{t('projects.images', { n: (p.download_image_count ?? 0).toLocaleString() })}</span>
        {extras?.versions != null && <span className="ds-badge ds-mute">{t('projects.versions', { count: extras.versions })}</span>}
      </div>

      <div>
        <div style={{ display: 'flex', gap: 3 }}>
          {segments.map((w, i) => (
            <span key={i} className="ds-meter" style={{ flex: 1 }}><i style={{ width: `${w}%` }} /></span>
          ))}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 11, color: 'var(--ink-3)', gap: 8 }}>
          <span>{caption || ' '}</span>
          {captionRight && <span className="ds-mono">{captionRight}</span>}
        </div>
      </div>

      <Link className="ds-ctl" to={action.to} style={{ justifyContent: 'center', marginTop: 'auto' }}>{action.label}</Link>
    </div>
  )
}

// ── New Project Dialog ──────────────────────────────────────────

interface NewProjectForm {
  title: string
  note: string
  initial_version_label: string
}

function NewProjectDialog({
  busy,
  onCancel,
  onSubmit,
}: {
  busy: boolean
  onCancel: () => void
  onSubmit: (form: NewProjectForm) => void
}) {
  const { t } = useTranslation()
  const [form, setForm] = useState<NewProjectForm>({
    title: '',
    note: '',
    initial_version_label: 'v1',
  })

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!form.title.trim()) return
    onSubmit(form)
  }

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]"
      onClick={onCancel}
    >
      <form
        onClick={(e) => e.stopPropagation()}
        onSubmit={submit}
        className="bg-elevated border border-subtle rounded-2xl p-6 flex flex-col gap-4 shadow-xl"
        style={{ width: '90%', maxWidth: 440 }}
      >
        <h2 className="m-0 text-lg font-semibold">{t('projects.newProject')}</h2>

        <FieldLabel label={t('projects.newProjectTitle')} hint="title">
          <input
            autoFocus
            className="input"
            value={form.title}
            onChange={(e) => setForm({ ...form, title: e.target.value })}
            placeholder={t('projects.titlePlaceholder')}
          />
        </FieldLabel>

        <FieldLabel label={t('projects.versionLabel')} hint="initial_version_label">
          <input
            className="input input-mono"
            value={form.initial_version_label}
            onChange={(e) => setForm({ ...form, initial_version_label: e.target.value })}
            placeholder={t('projects.versionPlaceholder')}
          />
        </FieldLabel>

        <FieldLabel label={t('common.notes')} hint="note">
          <textarea
            className="input"
            value={form.note}
            onChange={(e) => setForm({ ...form, note: e.target.value })}
            placeholder={t('projects.notesPlaceholder')}
            rows={3}
            style={{ resize: 'vertical' }}
          />
        </FieldLabel>

        <div className="flex gap-2 justify-end">
          <button type="button" className="btn btn-secondary" onClick={onCancel}>{t('common.cancel')}</button>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={busy || !form.title.trim()}
          >
            {busy ? t('projects.creating') : t('common.create')}
          </button>
        </div>
      </form>
    </div>
  )
}

function FieldLabel({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-sm font-medium">
        {label}
        {hint && <span className="ml-2 text-xs text-fg-tertiary font-mono">{hint}</span>}
      </span>
      {children}
    </label>
  )
}
