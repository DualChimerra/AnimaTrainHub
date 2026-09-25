/** Project overview, laid out as the approved mockup (Overview / OverviewFiles
 *  artboards): page head with the project actions, the pipeline card, four
 *  KPIs, a Tasks / LoRA files table and a right column with versions, the
 *  train-set composition and the top tags. Every number comes from the API;
 *  a block with nothing to show says so instead of inventing values.
 *
 *  The page shows one version at a time: `?version=N` (links from the queue)
 *  or the active version. Task actions that used to live in the status
 *  banner (pause, cancel, log, monitor) sit in each task's ⋯ menu; the full
 *  output folder (training states, delete, export) stays one click away in
 *  the task's Outputs tab. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate, useOutletContext } from 'react-router-dom'
import {
  api,
  PHASE_ORDER,
  type LoraCkpt,
  type ProjectDetail,
  type Task,
  type TaskOutputs,
  type Version,
  type VersionPhase,
} from '../../api/client'
import KebabMenu, { type KebabItem } from '../../components/ds/KebabMenu'
import PageHead from '../../components/ds/PageHead'
import { useDialog } from '../../components/Dialog'
import { useToast } from '../../components/Toast'
import { useProjectCtx } from '../../context/ProjectContext'
import { splitOptional } from '../../lib/labels'
import { useEventStream } from '../../lib/useEventStream'
import { useMonitorProgress } from '../../lib/useMonitorProgress'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
  onCreateVersion: (forkFromVid?: number) => void
  creatingVersionBusy: boolean
}

type OverviewTab = 'tasks' | 'files'

// ── helpers ──────────────────────────────────────────────────────────────

const DAY = 86400

/** Route segment and sidebar number of each version phase. */
const PHASE_STEP: Record<VersionPhase, { key: string; n: number; labelKey: string }> = {
  curating:      { key: 'curate',     n: 1, labelKey: 'nav.curate' },
  preprocessing: { key: 'preprocess', n: 2, labelKey: 'nav.preprocess' },
  editing:       { key: 'edit',       n: 3, labelKey: 'nav.tagEdit' },
  regularizing:  { key: 'reg',        n: 4, labelKey: 'nav.reg' },
  ready:         { key: 'train',      n: 5, labelKey: 'nav.train' },
}

const phaseIndex = (phase: VersionPhase) => PHASE_ORDER.indexOf(phase)

/** While a version is being prepared only the cursor phase and the ones
 *  before it are reachable (advancing goes through each step's own
 *  "continue" check). Trained / training versions can revisit any step. */
function canGoPhase(version: Version | null, phase: VersionPhase): boolean {
  if (!version) return false
  if (version.status !== 'preparing') return true
  return phaseIndex(phase) <= phaseIndex(version.phase)
}

function startOfDay(ts: number): number {
  const d = new Date(ts * 1000)
  d.setHours(0, 0, 0, 0)
  return d.getTime() / 1000
}

function fmtClock(ts: number, locale: string): string {
  return new Intl.DateTimeFormat(locale, { hour: '2-digit', minute: '2-digit' }).format(new Date(ts * 1000))
}

function fmtDate(ts: number, locale: string, withYear = false): string {
  return new Intl.DateTimeFormat(locale, withYear
    ? { day: 'numeric', month: 'short', year: 'numeric' }
    : { day: 'numeric', month: 'short' }).format(new Date(ts * 1000))
}

function fmtBytes(t: (k: string) => string, n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} ${t('overview.unitGB')}`
  if (n >= 1024 ** 2) return `${Math.round(n / 1024 ** 2)} ${t('overview.unitMB')}`
  if (n >= 1024) return `${Math.round(n / 1024)} ${t('overview.unitKB')}`
  return `${n} ${t('overview.unitB')}`
}

const normPath = (p: string) => p.replace(/\\/g, '/').toLowerCase()

const isTrainTask = (task: Task) => (task.task_type ?? 'train') === 'train'

const Icon = {
  arrow: <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h13m-5-6 6 6-6 6" /></svg>,
  rowgo: <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h13m-5-6 6 6-6 6" /></svg>,
  check: <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round"><path d="m5 13 4 4L19 7" /></svg>,
  image: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="16" rx="2.5" /><path d="m3 15 4.5-4.5 4 4L15 11l6 5.5" /></svg>,
  tag: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M20.6 13.4 12 22l-9-9V3h9l8.6 8.6a2 2 0 0 1 0 2.8Z" /></svg>,
  branch: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="6" r="2.5" /><circle cx="6" cy="18" r="2.5" /><circle cx="18" cy="12" r="2.5" /><path d="M8.5 6h4a3 3 0 0 1 3 3v.5M8.5 18h4a3 3 0 0 0 3-3v-.5" /></svg>,
  hex: (s: number) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={s > 13 ? 1.7 : 1.8} strokeLinecap="round" strokeLinejoin="round"><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z" /></svg>,
}

/** One LoRA file row: from the latest training task's output listing when
 *  there is one (has size, download, zip), else from the version's
 *  checkpoint scan. */
interface LoraRow {
  name: string
  /** Absolute path, used for the generate page and "copy path". */
  path: string
  /** Path relative to output/, only for task-backed rows. */
  relPath: string | null
  size: number | null
  mtime: number
}

// ── main ─────────────────────────────────────────────────────────────────

export default function ProjectOverview() {
  const { t, i18n } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const ctx = useProjectCtx()
  const navigate = useNavigate()
  const { toast } = useToast()
  const { confirm } = useDialog()

  // Shown version: `?version=N` (queue links) first, then the active one.
  // The query is dropped right away so a refresh does not pin the version.
  const [selectedVid, setSelectedVid] = useState<number | null>(() => {
    try {
      const v = new URLSearchParams(window.location.search).get('version')
      if (v && Number.isFinite(Number(v))) return Number(v)
    } catch { /* ignore */ }
    return project.active_version_id ?? activeVersion?.id ?? null
  })
  useEffect(() => {
    try {
      const url = new URL(window.location.href)
      if (url.searchParams.has('version')) {
        url.searchParams.delete('version')
        window.history.replaceState({}, '', url.toString())
      }
    } catch { /* ignore */ }
  }, [])
  // Follow the active version when it changes (sidebar switch, activation
  // from the versions card), but not on mount so `?version=N` survives.
  const prevActiveRef = useRef(project.active_version_id)
  useEffect(() => {
    if (prevActiveRef.current !== project.active_version_id) {
      prevActiveRef.current = project.active_version_id
      setSelectedVid(project.active_version_id)
    }
  }, [project.active_version_id])
  useEffect(() => {
    if (!project.versions.some((v) => v.id === selectedVid)) setSelectedVid(project.active_version_id ?? null)
  }, [project.versions, project.active_version_id, selectedVid])

  const version: Version | null = project.versions.find((v) => v.id === selectedVid) ?? null
  const vid = version?.id ?? null

  // All tasks of the project, newest first.
  const [tasks, setTasks] = useState<Task[]>([])
  const [tasksLoaded, setTasksLoaded] = useState(false)
  const loadTasks = useCallback(() => {
    let cancelled = false
    void api.listQueue()
      .then((items) => {
        if (cancelled) return
        setTasks(items
          .filter((tk) => tk.project_id === project.id)
          .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0)))
      })
      .catch(() => { if (!cancelled) setTasks([]) })
      .finally(() => { if (!cancelled) setTasksLoaded(true) })
    return () => { cancelled = true }
  }, [project.id])
  useEffect(() => loadTasks(), [loadTasks])

  const [outputsKey, setOutputsKey] = useState(0)
  useEventStream((evt) => {
    if (evt.type === 'train_loop_started'
      || evt.type === 'auto_epoch_backup_written'
      || evt.type === 'task_state_changed') {
      loadTasks()
      setOutputsKey((k) => k + 1)
    }
  })

  const versionTasks = useMemo(() => tasks.filter((tk) => tk.version_id === vid), [tasks, vid])
  const latestTrain = useMemo(() => versionTasks.find(isTrainTask) ?? null, [versionTasks])
  const runningTrain = latestTrain?.status === 'running' ? latestTrain : null
  const { state: monitor } = useMonitorProgress(runningTrain?.id ?? null)
  const trainProgress = runningTrain && monitor?.step != null && monitor.total_steps
    ? { step: monitor.step, total: monitor.total_steps } : null

  // LoRA files of the shown version.
  const [outputs, setOutputs] = useState<TaskOutputs | null>(null)
  const [ckpts, setCkpts] = useState<LoraCkpt[]>([])
  const latestTrainId = latestTrain?.id ?? null
  useEffect(() => {
    let cancelled = false
    setOutputs(null)
    if (latestTrainId != null) {
      void api.getTaskOutputs(latestTrainId)
        .then((r) => { if (!cancelled) setOutputs(r) })
        .catch(() => { if (!cancelled) setOutputs(null) })
    }
    if (vid != null) {
      void api.listVersionLoraCkpts(project.id, vid)
        .then((r) => { if (!cancelled) setCkpts(r) })
        .catch(() => { if (!cancelled) setCkpts([]) })
    } else {
      setCkpts([])
    }
    return () => { cancelled = true }
  }, [project.id, vid, latestTrainId, outputsKey])

  const loraRows = useMemo<LoraRow[]>(() => {
    if (outputs?.exists) {
      const dir = outputs.output_dir ?? ''
      return outputs.files
        .filter((f) => f.is_lora)
        .map((f) => ({ name: f.name, path: dir ? `${dir}/${f.path}` : f.path, relPath: f.path, size: f.size, mtime: f.mtime }))
        .sort((a, b) => b.mtime - a.mtime || b.name.localeCompare(a.name, undefined, { numeric: true }))
    }
    return ckpts
      .map((c) => ({ name: c.path.split(/[\\/]/).pop() ?? c.label, path: c.path, relPath: null, size: null, mtime: c.mtime }))
      .sort((a, b) => b.mtime - a.mtime)
  }, [outputs, ckpts])

  // Captions: top tags and the unique-tag count.
  const [tagCounts, setTagCounts] = useState<Array<{ tag: string; n: number }> | null>(null)
  // Train images: resolution buckets.
  const [sizes, setSizes] = useState<Array<{ w: number; h: number }> | null>(null)
  useEffect(() => {
    if (vid == null) { setTagCounts(null); setSizes(null); return }
    let cancelled = false
    void api.listCaptionsFull(project.id, vid)
      .then((res) => {
        if (cancelled) return
        const counter = new Map<string, number>()
        for (const it of res.items) for (const tg of it.tags) counter.set(tg, (counter.get(tg) ?? 0) + 1)
        setTagCounts(Array.from(counter, ([tag, n]) => ({ tag, n }))
          .sort((a, b) => b.n - a.n || a.tag.localeCompare(b.tag)))
      })
      .catch(() => { if (!cancelled) setTagCounts(null) })
    void api.listPreprocessFilesTrain(project.id, vid)
      .then((res) => {
        if (cancelled) return
        setSizes(res.images.flatMap((i) => (i.w && i.h ? [{ w: i.w, h: i.h }] : [])))
      })
      .catch(() => { if (!cancelled) setSizes(null) })
    return () => { cancelled = true }
  }, [project.id, vid, version?.stats?.train_image_count, version?.stats?.tagged_image_count])

  const [tab, setTab] = useState<OverviewTab>('tasks')
  const [editing, setEditing] = useState(false)

  const stepLabel = (phase: VersionPhase) => splitOptional(t(PHASE_STEP[phase].labelKey)).short
  const stepRoute = (v: Version, phase: VersionPhase) => `/projects/${project.id}/v/${v.id}/${PHASE_STEP[phase].key}`
  const cursorPhase = (v: Version): VersionPhase => (v.status === 'preparing' ? v.phase : 'ready')

  const generateUrl = (v: Version, loraPath: string) => {
    const sp = new URLSearchParams()
    sp.set('lora', loraPath)
    sp.set('projectId', String(project.id))
    sp.set('versionId', String(v.id))
    return `/tools/generate?${sp.toString()}`
  }
  const resultPath = version?.output_lora_path || loraRows[0]?.path || null

  // Primary action in the page head.
  let primary: { to: string; label: string } | null = null
  if (version) {
    if (version.status === 'completed' && resultPath) {
      primary = { to: generateUrl(version, resultPath), label: t('projects.viewResult') }
    } else {
      const phase = cursorPhase(version)
      primary = { to: stepRoute(version, phase), label: t('projects.continueStep', { step: stepLabel(phase).toLowerCase() }) }
    }
  }

  const stats = version?.stats
  const train = stats?.train_image_count ?? 0
  const tagged = stats?.tagged_image_count ?? 0
  const reg = stats?.reg_image_count ?? 0

  const subtitle = (
    <>
      {t('overview.downloads', { count: project.download_image_count ?? 0 })}
      {' · '}{t('projects.versions', { count: project.versions.length })}
      {' · '}{t('overview.createdOn', { date: fmtDate(project.created_at, i18n.language, true) })}
      {activeVersion && <>{' · '}{t('overview.activeLabel')} <code>{activeVersion.label}</code></>}
      {project.note && <span style={{ display: 'block', marginTop: 4 }}>{project.note}</span>}
    </>
  )

  return (
    <div className="fade-in">
      <PageHead
        eyebrow={t('overview.eyebrow', { slug: project.slug })}
        title={project.title}
        subtitle={subtitle}
        tools={
          <>
            <button type="button" className="ds-ctl" onClick={() => setEditing(true)}>{t('overview.editProject')}</button>
            <button type="button" className="ds-ctl" onClick={() => ctx?.onCreateVersion()}>{t('overview.newVersion')}</button>
            {primary && <Link className="ds-btn-primary" to={primary.to}>{primary.label}{Icon.arrow}</Link>}
          </>
        }
      />

      <div className="ds-scroll">
        <PipelineCard
          project={project}
          version={version}
          latestTrain={latestTrain}
          trainProgress={trainProgress}
        />

        <Kpis
          project={project}
          version={version}
          uniqueTags={tagCounts?.length ?? null}
          loraRows={loraRows}
        />

        <div className="ds-ov-main">
          <div className="ds-card" style={{ minWidth: 0 }}>
            <div className="ds-tabs" role="tablist">
              <button type="button" role="tab" aria-selected={tab === 'tasks'} className={`ds-tab${tab === 'tasks' ? ' ds-is-active' : ''}`} onClick={() => setTab('tasks')}>
                {t('overview.tabTasks')}<span className="ds-badge ds-mute">{tasks.length}</span>
              </button>
              <button type="button" role="tab" aria-selected={tab === 'files'} className={`ds-tab${tab === 'files' ? ' ds-is-active' : ''}`} onClick={() => setTab('files')}>
                {t('overview.tabOutput')}<span className="ds-badge ds-mute">{loraRows.length}</span>
              </button>
            </div>
            {tab === 'tasks' ? (
              <TasksTable
                project={project}
                tasks={tasks}
                loaded={tasksLoaded}
                onOpen={(id, hash) => navigate(`/queue/${id}${hash ?? ''}`)}
                onPause={(id) => { api.pauseTask(id).then(loadTasks).catch((e) => toast(String(e), 'error')) }}
                onCancel={async (task) => {
                  const ok = await confirm(t('overview.cancelConfirm', { id: task.id }), { tone: 'danger' })
                  if (!ok) return
                  api.cancelTask(task.id).then(loadTasks).catch((e) => toast(String(e), 'error'))
                }}
              />
            ) : (
              <FilesTable
                version={version}
                rows={loraRows}
                outputs={outputs}
                taskId={latestTrainId}
                onRender={(row) => version && navigate(generateUrl(version, row.path))}
              />
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minWidth: 0 }}>
            <VersionsCard
              project={project}
              selectedVid={vid}
              tasks={tasks}
              onOpen={(v) => {
                const isActive = v.id === project.active_version_id
                if (!isActive && v.status === 'completed' && v.output_lora_path) {
                  navigate(generateUrl(v, v.output_lora_path))
                  return
                }
                if (!isActive) ctx?.onSelectVersion(v.id)
                navigate(stepRoute(v, cursorPhase(v)))
              }}
            />
            <CompositionCard version={version} sizes={sizes} />
            <TopTagsCard version={version} tagCounts={tagCounts} train={train} tagged={tagged} reg={reg} />
          </div>
        </div>
      </div>

      {editing && (
        <EditProjectDialog
          project={project}
          onClose={() => setEditing(false)}
          onSaved={async () => { setEditing(false); await reload() }}
        />
      )}
    </div>
  )
}

// ── pipeline ─────────────────────────────────────────────────────────────

function PipelineCard({ project, version, latestTrain, trainProgress }: {
  project: ProjectDetail
  version: Version | null
  latestTrain: Task | null
  trainProgress: { step: number; total: number } | null
}) {
  const { t } = useTranslation()
  const stats = version?.stats
  const train = stats?.train_image_count ?? 0
  const tagged = stats?.tagged_image_count ?? 0
  const reg = stats?.reg_image_count ?? 0
  const preparing = version?.status === 'preparing'
  const ci = version ? (preparing ? phaseIndex(version.phase) : PHASE_ORDER.length) : -1
  const tagsDone = train > 0 && tagged >= train

  type State = 'done' | 'cur' | 'todo'
  const phaseState = (phase: VersionPhase): State => {
    if (!version) return 'todo'
    const i = phaseIndex(phase)
    if (!preparing) return phase === 'ready' ? (version.status === 'completed' ? 'done' : 'cur') : 'done'
    return i < ci ? 'done' : i === ci ? 'cur' : 'todo'
  }

  const editState = phaseState('editing')
  const tagState: State = editState === 'cur' ? (tagsDone ? 'done' : 'cur') : editState
  const tagEditState: State = editState === 'cur' ? (tagsDone ? 'cur' : 'todo') : editState
  const trainState = phaseState('ready')

  const cellStatus = (s: State) => (s === 'done' ? t('overview.cellDone') : s === 'cur' ? t('overview.cellCurrent') : '—')

  let trainCell = cellStatus(trainState)
  let trainTitle: string | undefined
  let trainPct = trainState === 'done' ? 100 : 0
  if (version && trainState !== 'todo' && version.status !== 'preparing') {
    if (trainProgress) {
      trainCell = t('overview.cellStep', { step: trainProgress.step.toLocaleString(), total: trainProgress.total.toLocaleString() })
      trainPct = Math.round((trainProgress.step / trainProgress.total) * 100)
    } else if (latestTrain?.status === 'paused' && latestTrain.paused_step != null) {
      trainCell = t('projects.pausedAt', { step: latestTrain.paused_step.toLocaleString() })
    } else if (latestTrain?.status === 'pending' || latestTrain?.status === 'scheduled') {
      trainCell = t('overview.cellQueued')
    } else if (version.status === 'failed') {
      trainCell = t('overview.cellFailed')
      trainTitle = version.last_failure_reason || latestTrain?.error_msg || undefined
    } else if (version.status === 'canceled') {
      trainCell = t('overview.cellCanceled')
    }
  }

  const steps: Array<{
    key: string; label: string; badge: string; state: State; pct: number
    cell: string; cellTitle?: string; live?: boolean; to: string | null
  }> = [
    {
      key: 'download', label: t('overview.stepDownload'), badge: '·',
      state: (project.download_image_count ?? 0) > 0 || train > 0 ? 'done' : 'todo',
      pct: (project.download_image_count ?? 0) > 0 || train > 0 ? 100 : 0,
      cell: t('projects.images', { n: (project.download_image_count ?? 0).toLocaleString() }),
      to: `/projects/${project.id}/download`,
    },
    {
      key: 'curate', label: t('overview.stepCurate'), badge: '1', state: phaseState('curating'),
      pct: phaseState('curating') === 'done' ? 100 : 0,
      cell: version ? t('projects.images', { n: train.toLocaleString() }) : '—',
      to: version && canGoPhase(version, 'curating') ? `/projects/${project.id}/v/${version.id}/curate` : null,
    },
    {
      key: 'preprocess', label: t('overview.stepPreprocess'), badge: '2', state: phaseState('preprocessing'),
      pct: phaseState('preprocessing') === 'done' ? 100 : 0,
      cell: cellStatus(phaseState('preprocessing')),
      to: version && canGoPhase(version, 'preprocessing') ? `/projects/${project.id}/v/${version.id}/preprocess` : null,
    },
    {
      key: 'tag', label: t('overview.stepTag'), badge: '3', state: tagState,
      pct: train > 0 ? Math.min(100, Math.round((tagged / train) * 100)) : 0,
      cell: version ? `${tagged.toLocaleString()} / ${train.toLocaleString()}` : '—',
      to: version && canGoPhase(version, 'editing') ? `/projects/${project.id}/v/${version.id}/edit` : null,
    },
    {
      key: 'edit', label: t('overview.stepTagEdit'), badge: '3', state: tagEditState,
      pct: tagEditState === 'done' ? 100 : 0,
      cell: cellStatus(tagEditState),
      to: version && canGoPhase(version, 'editing') ? `/projects/${project.id}/v/${version.id}/edit` : null,
    },
    {
      key: 'reg', label: t('overview.stepReg'), badge: '4', state: phaseState('regularizing'),
      pct: phaseState('regularizing') === 'done' ? 100 : 0,
      cell: reg > 0 ? t('projects.images', { n: reg.toLocaleString() })
        : phaseState('regularizing') === 'done' ? t('overview.cellSkipped') : cellStatus(phaseState('regularizing')),
      to: version && canGoPhase(version, 'regularizing') ? `/projects/${project.id}/v/${version.id}/reg` : null,
    },
    {
      key: 'train', label: t('overview.stepTrain'), badge: '5', state: trainState, pct: trainPct,
      cell: trainCell, cellTitle: trainTitle, live: !!trainProgress,
      to: version && canGoPhase(version, 'ready') ? `/projects/${project.id}/v/${version.id}/train` : null,
    },
  ]

  const cursorN = version ? PHASE_STEP[preparing ? version.phase : 'ready'].n : null

  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div>
          <div className="ds-card-title">{t('overview.pipelineProgress')}</div>
          <div className="ds-card-sub">
            {version ? t('overview.pipelineSub', { label: version.label, n: cursorN }) : t('overview.noVersion')}
          </div>
        </div>
      </div>
      <div className="ds-card-body" style={{ paddingTop: 4, overflowX: 'auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7,minmax(0,1fr))', gap: 10, minWidth: 620 }}>
          {steps.map((s) => {
            const body = (
              <>
                <span className="ds-meter"><i style={{ width: `${s.pct}%` }} /></span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className={`ds-step-badge${s.state === 'done' ? ' ds-done' : s.state === 'cur' ? ' ds-cur' : ''}`}>
                    {s.state === 'done' ? Icon.check : s.badge}
                  </span>
                  <span style={{ fontSize: 12, fontWeight: s.state === 'cur' ? 600 : 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.label}</span>
                </span>
                <span className="ds-cell-key" title={s.cellTitle} style={s.live ? { color: 'var(--green-text)' } : undefined}>{s.cell}</span>
              </>
            )
            const style: React.CSSProperties = { display: 'flex', flexDirection: 'column', gap: 7, minWidth: 0 }
            return s.to
              ? <Link key={s.key} to={s.to} style={style}>{body}</Link>
              : <span key={s.key} style={{ ...style, opacity: 0.45 }} aria-disabled="true">{body}</span>
          })}
        </div>
      </div>
    </div>
  )
}

// ── KPIs ─────────────────────────────────────────────────────────────────

function ckptLabel(t: (k: string, o?: Record<string, unknown>) => string, name: string): string {
  const ema = /[_-]ema\b/i.test(name.replace(/\.safetensors$/i, ''))
  const epoch = name.match(/epoch[_-]?0*(\d+)/i) ?? name.match(/-0*(\d+)\.safetensors$/i)
  const step = name.match(/step[_-]?0*(\d+)/i)
  let label = step ? t('overview.ckptStep', { n: Number(step[1]).toLocaleString() })
    : epoch ? t('overview.ckptEpoch', { n: Number(epoch[1]) })
      : t('overview.ckptFinal')
  if (ema) label += ' · EMA'
  return label
}

function Kpis({ project, version, uniqueTags, loraRows }: {
  project: ProjectDetail
  version: Version | null
  uniqueTags: number | null
  loraRows: LoraRow[]
}) {
  const { t } = useTranslation()
  const stats = version?.stats
  const train = stats?.train_image_count ?? 0
  const tagged = stats?.tagged_image_count ?? 0

  // Train-set delta against the version created right before this one.
  const prev = version
    ? [...project.versions]
      .filter((v) => v.created_at < version.created_at && v.stats)
      .sort((a, b) => b.created_at - a.created_at)[0] ?? null
    : null
  const delta = prev && stats ? train - (prev.stats?.train_image_count ?? 0) : null

  const dash = version?.label.indexOf('-') ?? -1
  const labelHead = version ? (dash > 0 ? version.label.slice(0, dash) : version.label) : '—'
  const labelTail = version && dash > 0 ? version.label.slice(dash) : ''
  const isActive = version != null && version.id === project.active_version_id
  const state = version
    ? version.status === 'preparing'
      ? t('overview.kpiPhase', { phase: splitOptional(t(PHASE_STEP[version.phase].labelKey)).short })
      : t(`versionStatus.${version.status}`)
    : ''

  return (
    <div className="ds-kpis ds-ov-kpis">
      <div className="ds-card ds-kpi">
        <div className="ds-kpi-top">
          <span className="ds-kpi-icon">{Icon.image}</span>
          {delta != null && prev && (
            <span style={{ marginLeft: 'auto' }}>
              <span className={`ds-delta ${delta > 0 ? 'ds-up' : delta < 0 ? 'ds-down' : 'ds-mute'}`}>
                {t('overview.kpiDelta', { n: `${delta > 0 ? '+' : delta < 0 ? '−' : '±'}${Math.abs(delta).toLocaleString()}`, label: prev.label })}
              </span>
            </span>
          )}
        </div>
        <div className="ds-kpi-val">{version ? train.toLocaleString() : '—'}</div>
        <div className="ds-kpi-label">{t('overview.kpiTrain')}</div>
      </div>

      <div className="ds-card ds-kpi">
        <div className="ds-kpi-top">
          <span className="ds-kpi-icon">{Icon.tag}</span>
          {version && train > 0 && (
            <span style={{ marginLeft: 'auto' }}>
              <span className="ds-kpi-meta">
                {tagged >= train ? t('overview.kpiAllCaptioned') : t('overview.kpiMissingCaptions', { n: (train - tagged).toLocaleString() })}
              </span>
            </span>
          )}
        </div>
        <div className="ds-kpi-val">{version ? tagged.toLocaleString() : '—'}{version && <small>/ {train.toLocaleString()}</small>}</div>
        <div className="ds-kpi-label">
          {uniqueTags != null ? t('overview.kpiCaptionedUnique', { n: uniqueTags.toLocaleString() }) : t('overview.kpiCaptioned')}
        </div>
      </div>

      <div className="ds-card ds-kpi">
        <div className="ds-kpi-top">
          <span className="ds-kpi-icon">{Icon.branch}</span>
          <span style={{ marginLeft: 'auto' }}>
            <span className="ds-kpi-meta">{t('overview.kpiVersionsTotal', { v: t('projects.versions', { count: project.versions.length }) })}</span>
          </span>
        </div>
        <div className="ds-kpi-val">{labelHead}{labelTail && <small>{labelTail}</small>}</div>
        <div className="ds-kpi-label">{t(isActive ? 'overview.kpiActiveVersion' : 'overview.kpiShownVersion', { state })}</div>
      </div>

      <div className="ds-card ds-kpi">
        <div className="ds-kpi-top">
          <span className="ds-kpi-icon">{Icon.hex(15)}</span>
          <span style={{ marginLeft: 'auto' }}>
            <span className="ds-kpi-meta">
              {loraRows.length > 0 ? t('overview.kpiLastCkpt', { label: ckptLabel(t, loraRows[0].name) }) : t('overview.kpiNoCkpts')}
            </span>
          </span>
        </div>
        <div className="ds-kpi-val">{loraRows.length}</div>
        <div className="ds-kpi-label">{t('overview.kpiCkpts')}</div>
      </div>
    </div>
  )
}

// ── Tasks tab ────────────────────────────────────────────────────────────

function useWhen() {
  const { t, i18n } = useTranslation()
  const loc = i18n.language
  const today = startOfDay(Date.now() / 1000)
  return {
    /** Task start: clock time today, "yesterday", else the date. */
    start: (ts: number | null | undefined) => {
      if (!ts) return '—'
      if (ts >= today) return fmtClock(ts, loc)
      if (ts >= today - DAY) return t('overview.yesterday')
      return fmtDate(ts, loc)
    },
    /** File ready: "today", "yesterday", else the date. */
    day: (ts: number) => (ts >= today ? t('overview.today') : ts >= today - DAY ? t('overview.yesterday') : fmtDate(ts, loc)),
    duration: (task: Task) => {
      if (!task.started_at) return '—'
      const end = task.finished_at ?? (task.status === 'running' ? Date.now() / 1000 : task.paused_at ?? null)
      if (end == null) return '—'
      const mins = Math.floor(Math.max(0, end - task.started_at) / 60)
      if (mins < 1) return t('overview.durLess')
      if (mins < 60) return t('overview.durMin', { m: mins })
      return t('overview.durHours', { h: Math.floor(mins / 60), m: mins % 60 })
    },
  }
}

function TaskStatusBadge({ task }: { task: Task }) {
  const { t } = useTranslation()
  switch (task.status) {
    case 'running':
      return <span className="ds-badge ds-ok"><span className="ds-dot ds-dot-run" />{t('status.running')}</span>
    case 'failed':
      return <span className="ds-badge ds-err" title={task.error_msg ?? undefined}>{t('status.failed')}</span>
    case 'pending':
      return <span className="ds-badge ds-warn">{t('status.pending')}</span>
    case 'scheduled':
      return <span className="ds-badge ds-warn">{t('status.scheduled')}</span>
    case 'paused':
      return <span className="ds-badge ds-warn">{t('status.paused')}</span>
    case 'canceled':
      return <span className="ds-badge ds-mute">{t('status.canceled')}</span>
    default:
      return <span className="ds-badge ds-mute">{t('status.done')}</span>
  }
}

function TasksTable({ project, tasks, loaded, onOpen, onPause, onCancel }: {
  project: ProjectDetail
  tasks: Task[]
  loaded: boolean
  onOpen: (id: number, hash?: string) => void
  onPause: (id: number) => void
  onCancel: (task: Task) => void
}) {
  const { t } = useTranslation()
  const when = useWhen()
  if (!loaded) return <div className="ds-card-body ds-muted" style={{ fontSize: 12.5 }}>{t('common.loading')}</div>
  if (tasks.length === 0) return <div className="ds-card-body ds-muted" style={{ fontSize: 12.5 }}>{t('overview.tasksNone')}</div>

  const labelOf = (vid: number | null | undefined) => project.versions.find((v) => v.id === vid)?.label ?? null

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="ds-tbl">
        <thead>
          <tr>
            <th>{t('projects.colTask')}</th>
            <th>{t('overview.colVersion')}</th>
            <th>{t('projects.colStatus')}</th>
            <th>{t('overview.colStarted')}</th>
            <th>{t('overview.colDuration')}</th>
            <th style={{ width: 40 }} />
          </tr>
        </thead>
        <tbody>
          {tasks.map((task) => {
            const kind = task.task_type ?? 'train'
            const kindKey = kind.startsWith('eval_') ? 'eval' : kind
            const label = labelOf(task.version_id)
            const train = kind === 'train'
            const live = task.status === 'running' || task.status === 'pending' || task.status === 'scheduled'
            const items: KebabItem[] = [
              { label: t('projects.openTask'), onSelect: () => onOpen(task.id) },
              { label: t('overview.banner.viewLog'), onSelect: () => onOpen(task.id, '#log') },
            ]
            if (train && (task.status === 'running' || task.status === 'paused')) {
              items.push({ label: t('overview.banner.openMonitor'), onSelect: () => onOpen(task.id, '#monitor') })
            }
            if (task.status === 'running' && task.is_pausable) {
              items.push({ label: t('overview.banner.pause'), onSelect: () => onPause(task.id) })
            }
            if (live) {
              items.push({
                label: train ? t('overview.banner.cancelTraining') : t('overview.cancelTask'),
                onSelect: () => onCancel(task), tone: 'err',
              })
            }
            return (
              <tr key={task.id}>
                <td>
                  <Link to={`/queue/${task.id}`} className="ds-cell-main">
                    <span className="ds-idchip">#{task.id}</span>
                    <span>{t(`projects.taskKind.${kindKey}`, { defaultValue: kind })}<span className="ds-cell-key">{kind}</span></span>
                  </Link>
                </td>
                <td className={`ds-mono${label ? '' : ' ds-muted'}`}>{label ?? '—'}</td>
                <td><TaskStatusBadge task={task} /></td>
                <td className="ds-num">{when.start(task.started_at)}</td>
                <td className="ds-num">{when.duration(task)}</td>
                <td><KebabMenu label={t('projects.taskActions')} items={items} /></td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── LoRA files tab ───────────────────────────────────────────────────────

function FilesTable({ version, rows, outputs, taskId, onRender }: {
  version: Version | null
  rows: LoraRow[]
  outputs: TaskOutputs | null
  taskId: number | null
  onRender: (row: LoraRow) => void
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const navigate = useNavigate()
  const when = useWhen()
  if (!version) return <div className="ds-card-body ds-muted" style={{ fontSize: 12.5 }}>{t('overview.outputEmpty')}</div>
  if (rows.length === 0) return <div className="ds-card-body ds-muted" style={{ fontSize: 12.5 }}>{t('overview.outputEmptyVersion')}</div>

  const total = rows.reduce((s, r) => s + (r.size ?? 0), 0)
  const sized = rows.every((r) => r.size != null)
  const finalPath = version.output_lora_path ? normPath(version.output_lora_path) : null
  const relPaths = rows.flatMap((r) => (r.relPath ? [r.relPath] : []))

  const openFolder = async () => {
    if (taskId == null) return
    try {
      const r = await api.openTaskFolder(taskId)
      toast(t('queueDetail.folderOpened', { path: r.opened }), 'success')
    } catch (e) { toast(String(e), 'error') }
  }
  const copyPath = async (path: string) => {
    try { await navigator.clipboard.writeText(path); toast(t('queueDetail.pathCopied'), 'success') }
    catch { toast(t('queueDetail.copyFailed'), 'error') }
  }

  return (
    <>
      <div style={{ padding: '13px 17px 0', display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <span className="ds-kpi-meta">
          output/ · {t('projects.ckpts', { count: rows.length })}{sized ? ` · ${fmtBytes(t, total)}` : ''}
        </span>
        <span style={{ flex: 1 }} />
        {taskId != null && relPaths.length > 0 && (
          <a className="ds-ctl" style={{ height: 30 }} href={api.taskOutputsZipUrl(taskId, relPaths)} download>
            {t('overview.downloadAllZip')}
          </a>
        )}
        {taskId != null && outputs?.supports_open_folder && (
          <button type="button" className="ds-ctl ds-ghost" style={{ height: 30 }} onClick={() => void openFolder()}>
            {t('overview.openFolder')}
          </button>
        )}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="ds-tbl">
          <thead>
            <tr>
              <th>{t('common.file')}</th>
              <th>{t('overview.colVersion')}</th>
              <th>{t('common.size')}</th>
              <th>{t('overview.colReady')}</th>
              <th style={{ width: 150 }} />
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const isFinal = finalPath != null && normPath(row.path) === finalPath
              const items: KebabItem[] = []
              if (taskId != null && row.relPath) {
                items.push({
                  label: t('common.download'),
                  onSelect: () => {
                    const a = document.createElement('a')
                    a.href = api.taskOutputDownloadUrl(taskId, row.relPath!)
                    a.download = row.name
                    a.click()
                  },
                })
              }
              items.push({ label: t('overview.copyPath'), onSelect: () => void copyPath(row.path) })
              if (taskId != null) {
                items.push({ label: t('overview.allOutputs'), onSelect: () => navigate(`/queue/${taskId}#outputs`) })
              }
              return (
                <tr key={row.path} style={isFinal ? { background: 'var(--green-soft)' } : undefined}>
                  <td>
                    <span className="ds-cell-main">
                      <span className="ds-sect-icon" style={isFinal ? { background: '#fff' } : undefined}>{Icon.hex(13)}</span>
                      <span style={isFinal ? { fontWeight: 600 } : undefined}>
                        {row.name}
                        <span className="ds-cell-key">{ckptLabel(t, row.name)}{isFinal ? ` · ${t('overview.ckptResult')}` : ''}</span>
                      </span>
                    </span>
                  </td>
                  <td className="ds-mono">{version.label}</td>
                  <td className="ds-num">{row.size != null ? fmtBytes(t, row.size) : '—'}</td>
                  <td className="ds-num">{when.day(row.mtime)}</td>
                  <td>
                    <span style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
                      <button type="button" className="ds-ctl ds-ghost" style={{ height: 26 }} onClick={() => onRender(row)}>{t('overview.render')}</button>
                      <KebabMenu label={t('overview.fileActions')} items={items} />
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </>
  )
}

// ── right column ─────────────────────────────────────────────────────────

function VersionsCard({ project, selectedVid, tasks, onOpen }: {
  project: ProjectDetail
  selectedVid: number | null
  tasks: Task[]
  onOpen: (v: Version) => void
}) {
  const { t } = useTranslation()
  const active = project.versions.find((v) => v.id === project.active_version_id) ?? null
  const versions = [...project.versions].sort((a, b) => b.created_at - a.created_at)

  const badgeOf = (v: Version) => {
    const last = tasks.find((tk) => tk.version_id === v.id && isTrainTask(tk))
    if (v.status === 'training') {
      if (last?.status === 'paused') return <span className="ds-badge ds-warn">{t('overview.verPaused')}</span>
      if (last?.status === 'pending' || last?.status === 'scheduled') return <span className="ds-badge ds-warn">{t('status.queued')}</span>
      return <span className="ds-badge ds-ok"><span className="ds-dot ds-dot-run" />{t('overview.verTraining')}</span>
    }
    if (v.status === 'completed') return <span className="ds-badge ds-mute">{t('overview.verTrained')}</span>
    if (v.status === 'failed') return <span className="ds-badge ds-err" title={v.last_failure_reason ?? undefined}>{t('versionStatus.failed')}</span>
    if (v.status === 'canceled') return <span className="ds-badge ds-mute">{t('versionStatus.canceled')}</span>
    return <span className="ds-badge ds-mute">{t('overview.verPreparing', { phase: splitOptional(t(PHASE_STEP[v.phase].labelKey)).short })}</span>
  }

  const titleOf = (v: Version) => {
    if (v.id === project.active_version_id) return t('overview.openVersion')
    if (v.status === 'completed' && v.output_lora_path) return t('overview.viewOutput')
    return t('overview.activateAndOpen')
  }

  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div>
          <div className="ds-card-title">{t('overview.versions')}</div>
          <div className="ds-card-sub">
            {t('projects.versions', { count: project.versions.length })}
            {active && <>{' · '}{t('overview.activeLabel')} {active.label}</>}
          </div>
        </div>
      </div>
      <div style={{ maxHeight: 360, overflowY: 'auto' }}>
        {versions.length === 0 && <div className="ds-card-body ds-muted" style={{ fontSize: 12.5 }}>{t('overview.noVersion')}</div>}
        {versions.map((v) => {
          const s = v.stats
          const meta: string[] = []
          if (s) {
            meta.push(t('overview.verTrain', { n: s.train_image_count.toLocaleString() }))
            meta.push(s.reg_image_count > 0 ? t('overview.verReg', { n: s.reg_image_count.toLocaleString() }) : t('overview.verNoReg'))
            if (s.has_output) meta.push(t('overview.verCkpt'))
          }
          return (
            <button
              key={v.id}
              type="button"
              className={`ds-verrow${v.id === selectedVid ? ' ds-is-active' : ''}`}
              style={{ width: '100%', textAlign: 'left' }}
              title={titleOf(v)}
              onClick={() => onOpen(v)}
            >
              <span style={{ minWidth: 0 }}>
                <span className="ds-verrow-name"><span className="ds-nm">{v.label}</span>{badgeOf(v)}</span>
                {meta.length > 0 && <span className="ds-cell-key" style={{ display: 'block', marginTop: 4 }}>{meta.join(' · ')}</span>}
              </span>
              <span className="ds-rowgo" aria-hidden="true">{Icon.rowgo}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

const FOLDER_COLORS = ['#a3db52', '#cfe3ad', '#e3ecd4']
const OTHER_COLOR = '#eef1e8'
const REG_COLOR = '#d4d5d1'

function CompositionCard({ version, sizes }: {
  version: Version | null
  sizes: Array<{ w: number; h: number }> | null
}) {
  const { t } = useTranslation()
  const stats = version?.stats
  const train = stats?.train_image_count ?? 0
  const reg = stats?.reg_image_count ?? 0
  const total = train + reg

  // Donut: the three biggest train folders, the rest grouped, then reg.
  const folders = [...(stats?.train_folders ?? [])].filter((f) => f.image_count > 0).sort((a, b) => b.image_count - a.image_count)
  const segs: Array<{ label: string; n: number; color: string }> = folders.slice(0, 3)
    .map((f, i) => ({ label: f.name, n: f.image_count, color: FOLDER_COLORS[i] }))
  const rest = folders.slice(3).reduce((s, f) => s + f.image_count, 0)
  if (rest > 0) segs.push({ label: t('overview.compOther'), n: rest, color: OTHER_COLOR })
  if (reg > 0) segs.push({ label: t('overview.compReg'), n: reg, color: REG_COLOR })
  const segTotal = segs.reduce((s, x) => s + x.n, 0)
  let offset = 25
  const arcs = segs.map((s) => {
    const len = segTotal > 0 ? (s.n / segTotal) * 100 : 0
    const arc = { ...s, dash: `${len} ${100 - len}`, offset }
    offset -= len
    return arc
  })

  // Resolution buckets: top four sizes, the rest grouped.
  const buckets = useMemo(() => {
    if (!sizes || sizes.length === 0) return []
    const m = new Map<string, { w: number; h: number; n: number }>()
    for (const s of sizes) {
      const k = `${s.w}×${s.h}`
      const prev = m.get(k)
      m.set(k, { w: s.w, h: s.h, n: (prev?.n ?? 0) + 1 })
    }
    const sorted = Array.from(m.values()).sort((a, b) => b.n - a.n)
    const out = sorted.slice(0, 4).map((b) => ({
      label: b.w === b.h ? `${b.w}²` : `${b.w}×${b.h}`, title: `${b.w}×${b.h} · ${b.n}`, n: b.n, mute: false,
    }))
    const other = sorted.slice(4).reduce((s, b) => s + b.n, 0)
    if (other > 0) out.push({ label: t('overview.compOther'), title: `${t('overview.compOther')} · ${other}`, n: other, mute: true })
    return out
  }, [sizes, t])
  const maxBucket = Math.max(1, ...buckets.map((b) => b.n))

  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div>
          <div className="ds-card-title">{t('overview.compTitle')}</div>
          <div className="ds-card-sub">
            {version ? t('overview.compSub', { total: total.toLocaleString(), train: train.toLocaleString(), reg: reg.toLocaleString() }) : t('overview.noVersion')}
          </div>
        </div>
      </div>
      <div className="ds-card-body" style={{ paddingTop: 2 }}>
        {segTotal === 0 ? (
          <div className="ds-muted" style={{ fontSize: 12 }}>{t('overview.compEmpty')}</div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: 15 }}>
            <svg className="ds-donut" viewBox="0 0 42 42" aria-hidden="true">
              <circle cx="21" cy="21" r="15.9" fill="none" stroke="#f0f0ee" strokeWidth="6" />
              {arcs.map((a) => (
                <circle key={a.label} cx="21" cy="21" r="15.9" fill="none" stroke={a.color} strokeWidth="6" strokeDasharray={a.dash} strokeDashoffset={a.offset} />
              ))}
              <text x="21" y="20.4" textAnchor="middle" fontSize="7" fontWeight="600" fill="var(--ink)" fontFamily="Geist, sans-serif">{train.toLocaleString()}</text>
              <text x="21" y="25.2" textAnchor="middle" fontSize="3.3" fill="var(--ink-3)" fontFamily="Geist, sans-serif">{t('overview.compInTrain')}</text>
            </svg>
            <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 7 }}>
              {segs.map((s) => (
                <div key={s.label} className="ds-kv">
                  <span className="ds-k ds-legend" style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    <s style={{ background: s.color }} />{s.label}
                  </span>
                  <span className="ds-v">{s.n.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {buckets.length > 0 && (
          <>
            <div className="ds-cap" style={{ margin: '17px 0 10px' }}>{t('overview.compResBuckets')}</div>
            <div className="ds-barset" style={{ height: 50 }}>
              {buckets.map((b) => (
                <i key={b.label} className={b.mute ? 'ds-mute' : undefined} style={{ height: `${Math.max(4, (b.n / maxBucket) * 100)}%` }} title={b.title} />
              ))}
            </div>
            <div className="ds-axis">{buckets.map((b) => <span key={b.label}>{b.label}</span>)}</div>
          </>
        )}
      </div>
    </div>
  )
}

function TopTagsCard({ version, tagCounts, train, tagged }: {
  version: Version | null
  tagCounts: Array<{ tag: string; n: number }> | null
  train: number
  tagged: number
  reg: number
}) {
  const { t } = useTranslation()
  const trigger = version?.trigger_word?.trim() ?? ''
  const top = (tagCounts ?? []).slice(0, 6)
  const base = Math.max(1, train, ...top.map((x) => x.n))

  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div>
          <div className="ds-card-title">{t('overview.topTags')}</div>
          <div className="ds-card-sub">
            {version
              ? t('overview.topTagsSub', { tagged: tagged.toLocaleString(), total: train.toLocaleString(), unique: (tagCounts?.length ?? 0).toLocaleString() })
              : t('overview.noVersion')}
          </div>
        </div>
      </div>
      <div className="ds-card-body" style={{ paddingTop: 2, display: 'flex', flexDirection: 'column', gap: 9 }}>
        {top.length === 0 && <div className="ds-muted" style={{ fontSize: 12 }}>{t('overview.topTagsEmpty')}</div>}
        {top.map((row) => {
          const isTrigger = trigger !== '' && row.tag === trigger
          return (
            <div key={row.tag}>
              <div className="ds-kv">
                <span className="ds-k ds-mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', ...(isTrigger ? { color: 'var(--green-text)' } : {}) }}>{row.tag}</span>
                <span className="ds-v">{row.n.toLocaleString()}</span>
              </div>
              <span className="ds-meter ds-thin" style={{ marginTop: 4 }}><i style={{ width: `${(row.n / base) * 100}%` }} /></span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── edit project ─────────────────────────────────────────────────────────

function EditProjectDialog({ project, onClose, onSaved }: {
  project: ProjectDetail
  onClose: () => void
  onSaved: () => Promise<void>
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [title, setTitle] = useState(project.title)
  const [note, setNote] = useState(project.note ?? '')
  const [busy, setBusy] = useState(false)

  const save = async () => {
    if (!title.trim()) return
    setBusy(true)
    try {
      await api.updateProject(project.id, { title: title.trim(), note: note.trim() })
      toast(t('overview.saved'), 'success')
      await onSaved()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={t('overview.editProject')}
      className="fixed inset-0 z-40 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]"
      onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onClose() }}
      onKeyDown={(e) => { if (e.key === 'Escape' && !busy) onClose() }}
    >
      <form
        className="ds-card"
        style={{ width: '90%', maxWidth: 480, padding: 20, display: 'flex', flexDirection: 'column', gap: 14 }}
        onSubmit={(e) => { e.preventDefault(); void save() }}
      >
        <div className="ds-card-title" style={{ fontSize: 15 }}>{t('overview.editProject')}</div>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <span className="ds-cap">{t('overview.fieldTitle')}</span>
          <input className="input" value={title} onChange={(e) => setTitle(e.target.value)} autoFocus maxLength={200} />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <span className="ds-cap">{t('overview.fieldNote')}</span>
          <textarea className="input" value={note} onChange={(e) => setNote(e.target.value)} rows={3} style={{ resize: 'vertical' }} />
        </label>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="ds-ctl" onClick={onClose} disabled={busy}>{t('common.cancel')}</button>
          <button type="submit" className="ds-btn-primary" disabled={busy || !title.trim()}>{busy ? t('common.saving') : t('common.save')}</button>
        </div>
      </form>
    </div>
  )
}
