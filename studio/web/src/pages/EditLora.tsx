/** Edit LoRA section: the user's image-editing tools, as cards like the Projects
 *  page (name, state, a two-step bar, one next action), and the new-tool dialog. */
import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate } from 'react-router-dom'
import { api, isEditProject, type EditPairsListing, type EditRecipe, type ProjectSummary, type Task } from '../api/client'
import KebabMenu from '../components/ds/KebabMenu'
import PageHead from '../components/ds/PageHead'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import { EditIcon, RecipeChoice } from './edit/shared'

const folderIcon = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /></svg>
)

export default function EditLoraPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const [all, setAll] = useState<ProjectSummary[] | null>(null)
  const [tasks, setTasks] = useState<Task[]>([])
  const [listings, setListings] = useState<Record<number, EditPairsListing | null>>({})
  const [creating, setCreating] = useState(false)
  const [showArchived, setShowArchived] = useState(false)

  const refresh = async () => {
    try {
      const [list, queue] = await Promise.all([api.listProjects(), api.listQueue().catch(() => [] as Task[])])
      setAll(list.filter((p) => isEditProject(p)))
      setTasks(queue)
    } catch (e) {
      toast(String(e), 'error')
      setAll((cur) => cur ?? [])
    }
  }
  useEffect(() => { void refresh() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  useEventStream((evt) => {
    if (evt.type === 'project_state_changed' || evt.type === 'task_state_changed' || evt.type === 'version_state_changed') void refresh()
  })

  // Example counts of each tool's active version.
  const listingKey = (all ?? []).map((p) => `${p.id}:${p.active_version_id}:${p.updated_at}`).join(',')
  useEffect(() => {
    let cancelled = false
    void Promise.all((all ?? []).map(async (p) => {
      if (p.active_version_id == null) return [p.id, null] as const
      return [p.id, await api.listEditPairs(p.id, p.active_version_id).catch(() => null)] as const
    })).then((entries) => { if (!cancelled) setListings(Object.fromEntries(entries)) })
    return () => { cancelled = true }
  }, [listingKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const runningTask = tasks.find((task) => task.status === 'running' && (task.task_type ?? 'train') === 'train') ?? null
  const { state: monitor } = useMonitorProgress(runningTask?.id ?? null)
  const runningPct = runningTask && monitor?.step != null && monitor.total_steps
    ? Math.round((monitor.step / monitor.total_steps) * 100) : null

  const active = useMemo(() => (all ?? []).filter((p) => p.archived_at == null)
    .sort((a, b) => b.updated_at - a.updated_at), [all])
  const archived = useMemo(() => (all ?? []).filter((p) => p.archived_at != null), [all])
  const shown = showArchived ? [...active, ...archived] : active

  const create = async (recipe: EditRecipe, title: string) => {
    const p = await api.createProject({ title, kind: 'edit', initial_version_label: 'v1' })
    const v = p.versions[0]
    if (v) {
      await api.setEditRecipe(p.id, v.id, recipe).catch(() => undefined)
      navigate(`/projects/${p.id}/v/${v.id}/pairs`)
    } else {
      navigate(`/projects/${p.id}`)
    }
  }

  const archive = async (p: ProjectSummary) => {
    const isArchived = p.archived_at != null
    if (!(await confirm(
      isArchived ? t('projects.unarchiveConfirm', { title: p.title }) : t('projects.archiveConfirm', { title: p.title }),
      { okText: isArchived ? t('projects.unarchiveTitle') : t('projects.archiveBtn') },
    ))) return
    try {
      if (isArchived) await api.unarchiveProject(p.id)
      else await api.archiveProject(p.id)
      await refresh()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const remove = async (p: ProjectSummary) => {
    if (!(await confirm(t('projects.deleteConfirm', { title: p.title, slug: p.slug }),
      { tone: 'danger', okText: t('projects.deleteProject') }))) return
    try {
      await api.deleteProject(p.id)
      await refresh()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  return (
    <div className="fade-in">
      <PageHead
        title={t('editLora.title')}
        subtitle={t('editLora.subtitle')}
        tools={(
          <button type="button" className="ds-btn-primary" onClick={() => setCreating(true)}>
            {EditIcon.plus(13)}{t('editLora.newTool')}
          </button>
        )}
      />
      <div className="ds-scroll">
        {all == null ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,minmax(0,1fr))', gap: 14 }}>
            {[0, 1, 2].map((i) => <div key={i} className="ds-card" style={{ height: 176 }} />)}
          </div>
        ) : shown.length === 0 ? (
          <div className="ds-empty">
            <span style={{ fontWeight: 500, color: 'var(--ink-2)' }}>{t('editLora.emptyTitle')}</span>
            <span>{t('editLora.emptyHint')}</span>
            <button type="button" className="ds-btn-primary" style={{ marginTop: 8 }} onClick={() => setCreating(true)}>
              {EditIcon.plus(13)}{t('editLora.newTool')}
            </button>
          </div>
        ) : (
          <div className="ds-edit-tools">
            {shown.map((p) => (
              <ToolCard
                key={p.id}
                project={p}
                listing={listings[p.id]}
                training={runningTask?.project_id === p.id}
                trainPct={runningTask?.project_id === p.id ? runningPct : null}
                queued={tasks.some((task) => task.project_id === p.id && (task.status === 'pending' || task.status === 'scheduled'))}
                onArchive={() => void archive(p)}
                onDelete={() => void remove(p)}
              />
            ))}
          </div>
        )}
        {archived.length > 0 && (
          <button type="button" className="ds-ctl" style={{ alignSelf: 'flex-start' }} onClick={() => setShowArchived((v) => !v)}>
            {showArchived ? t('editLora.hideArchived') : t('editLora.showArchived', { count: archived.length })}
          </button>
        )}
      </div>

      {creating && (
        <NewToolDialog
          onCancel={() => setCreating(false)}
          onCreate={async (recipe, title) => {
            try {
              await create(recipe, title)
            } catch (e) {
              toast(String(e), 'error')
            }
          }}
        />
      )}
    </div>
  )
}

/** One tool, like a project card: name, badges, the two steps, one next action. */
function ToolCard({ project: p, listing, training, trainPct, queued, onArchive, onDelete }: {
  project: ProjectSummary
  listing: EditPairsListing | null | undefined
  training: boolean
  trainPct: number | null
  queued: boolean
  onArchive: () => void
  onDelete: () => void
}) {
  const { t } = useTranslation()
  const vid = p.active_version_id
  const base = `/projects/${p.id}${vid != null ? `/v/${vid}` : ''}`
  const total = listing?.count ?? 0
  const done = listing?.complete ?? 0
  const status = p.active_version_status
  const trained = status === 'completed'
  const atTrain = training || queued || trained || status === 'failed' || (status === 'preparing' && p.active_version_phase === 'ready')
    || (status != null && status !== 'preparing')

  const badge = training
    ? <span className="ds-badge ds-ok"><span className="ds-dot ds-dot-run" />{t('editLora.status_training')}</span>
    : queued
      ? <span className="ds-badge ds-warn">{t('editLora.status_queued')}</span>
      : trained
        ? <span className="ds-badge ds-ok">{t('editLora.status_trained')}</span>
        : status === 'failed'
          ? <span className="ds-badge ds-err">{t('editLora.status_failed')}</span>
          : null

  const segments = [
    total > 0 ? (done / total) * 100 : 0,
    trained ? 100 : training && trainPct != null ? trainPct : 0,
  ]
  const caption = atTrain
    ? t('editLora.stepTrain')
    : t('editLora.stepExamples')
  const captionRight = training && trainPct != null ? `${trainPct}%` : listing ? `${done} / ${total}` : ''

  const action = trained
    ? { to: `/projects/${p.id}`, label: t('projects.viewResult') }
    : queued
      ? { to: '/queue', label: t('projects.continueQueue') }
      : atTrain
        ? { to: `${base}/train`, label: t('projects.continueStep', { step: t('editLora.stepTrainShort') }) }
        : { to: `${base}/pairs`, label: t('projects.continueStep', { step: t('editLora.stepExamplesShort') }) }

  const tileStyle: React.CSSProperties = { width: 34, height: 34, borderRadius: 10 }
  if (training) Object.assign(tileStyle, { background: 'var(--green-soft)', color: 'var(--green-text)' })

  return (
    <div className="ds-card" style={{ padding: '15px 16px 14px', display: 'flex', flexDirection: 'column', gap: 11, opacity: p.archived_at != null ? 0.72 : 1 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
        <Link to={`/projects/${p.id}`} style={{ display: 'flex', alignItems: 'flex-start', gap: 10, flex: 1, minWidth: 0 }}>
          <span className="ds-sect-icon" style={tileStyle}>{folderIcon}</span>
          <span style={{ flex: 1, minWidth: 0 }}>
            <span style={{ fontSize: 14, fontWeight: 600, letterSpacing: '-.02em', display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.title}</span>
            <span className="ds-cell-key">{p.slug}{p.active_version_label ? `, ${p.active_version_label}` : ''}</span>
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
        {listing && <span className="ds-badge ds-mute">{t(`editLora.recipe_${listing.recipe}`)}</span>}
        {listing && <span className="ds-badge ds-mute">{t('editLora.pairsCount', { count: total })}</span>}
      </div>

      <div>
        <div style={{ display: 'flex', gap: 3 }}>
          {segments.map((w, i) => <span key={i} className="ds-meter" style={{ flex: 1 }}><i style={{ width: `${w}%` }} /></span>)}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 11, color: 'var(--ink-3)', gap: 8 }}>
          <span>{caption}</span>
          {captionRight && <span className="ds-mono">{captionRight}</span>}
        </div>
      </div>

      <Link className="ds-ctl" to={action.to} style={{ justifyContent: 'center', marginTop: 'auto' }}>{action.label}</Link>
    </div>
  )
}

function NewToolDialog({ onCancel, onCreate }: {
  onCancel: () => void
  onCreate: (recipe: EditRecipe, title: string) => Promise<void>
}) {
  const { t } = useTranslation()
  const [recipe, setRecipe] = useState<EditRecipe>('text')
  const [title, setTitle] = useState('')
  const [busy, setBusy] = useState(false)
  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim() || busy) return
    setBusy(true)
    void onCreate(recipe, title.trim()).finally(() => setBusy(false))
  }
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]" onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel() }}>
      <form className="ds-modal" style={{ width: 'min(520px, 94vw)' }} onSubmit={submit}>
        <h2 className="ds-modal-title">{t('editLora.createTitle')}</h2>
        <label className="ds-modal-field">
          <span className="ds-modal-label">{t('editLora.createName')}</span>
          <input autoFocus className="ds-inp" style={{ height: 36 }} value={title}
            onChange={(e) => setTitle(e.target.value)} placeholder={t(`editLora.createNamePh_${recipe}`)} />
        </label>
        <div className="ds-modal-field">
          <span className="ds-modal-label">{t('editLora.createType')}</span>
          <RecipeChoice value={recipe} onChange={setRecipe} />
        </div>
        <div className="ds-modal-foot">
          <button type="button" className="ds-ctl" onClick={onCancel} disabled={busy}>{t('common.cancel')}</button>
          <button type="submit" className="ds-btn-primary" disabled={busy || !title.trim()}>
            {busy ? t('projects.creating') : t('common.create')}
          </button>
        </div>
      </form>
    </div>
  )
}
