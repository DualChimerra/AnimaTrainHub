/** Datasets: the shared image library. Each dataset is uploaded once and can
 *  be picked by any training in Train Set Curation; caches the trainings build
 *  stay inside the dataset folder and are reused next time. */
import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate } from 'react-router-dom'
import { api, type DatasetSummary, type ProjectSummary } from '../../api/client'
import KebabMenu from '../../components/ds/KebabMenu'
import PageHead from '../../components/ds/PageHead'
import { useDialog } from '../../components/Dialog'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'
import { useQueueFormat } from '../../lib/queueFormat'

type Sort = 'updated' | 'title' | 'images'

export const DatasetIcon = {
  plus: <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>,
  search: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" /></svg>,
  stack: (s = 15) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /></svg>,
  image: (s = 15) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="16" rx="2.5" /><path d="m3 15 4.5-4.5 4 4L15 11l6 5.5" /></svg>,
  link: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M10 14a4 4 0 0 0 5.66 0l3-3a4 4 0 0 0-5.66-5.66l-1 1" /><path d="M14 10a4 4 0 0 0-5.66 0l-3 3a4 4 0 0 0 5.66 5.66l1-1" /></svg>,
  cache: (s = 15) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><ellipse cx="12" cy="6" rx="8" ry="3" /><path d="M4 6v6c0 1.66 3.58 3 8 3s8-1.34 8-3V6" /><path d="M4 12v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6" /></svg>,
  folder: (s = 15) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /></svg>,
  arrow: <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h13m-5-6 6 6-6 6" /></svg>,
}

export default function DatasetsPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const fmt = useQueueFormat()
  const actions = useDatasetActions()
  const [items, setItems] = useState<DatasetSummary[] | null>(null)
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<Sort>('updated')
  const [creating, setCreating] = useState(false)

  const refresh = async () => {
    try {
      setItems(await api.listLibraryDatasets())
    } catch (e) {
      toast(String(e), 'error')
      setItems((cur) => cur ?? [])
    }
  }
  useEffect(() => { void refresh() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  useEventStream((evt) => {
    if (evt.type === 'project_state_changed' || (evt.type === 'job_state_changed' && evt.status !== 'running')) void refresh()
  })

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase()
    const rows = (items ?? []).filter((d) => !q || [d.title, d.slug, d.note ?? ''].some((s) => s.toLowerCase().includes(q)))
    return rows.sort((a, b) => sort === 'title' ? a.title.localeCompare(b.title)
      : sort === 'images' ? b.image_count - a.image_count
        : b.updated_at - a.updated_at)
  }, [items, query, sort])

  const all = items ?? []
  const totalImages = all.reduce((s, d) => s + d.image_count, 0)
  const totalBytes = all.reduce((s, d) => s + d.image_bytes, 0)
  const cacheBytes = all.reduce((s, d) => s + d.cache_bytes, 0)
  const cacheFiles = all.reduce((s, d) => s + d.cache_files, 0)
  const trainings = new Set(all.flatMap((d) => d.used_by.map((u) => u.version_id))).size
  const inUse = all.filter((d) => d.used_by.length > 0).length

  return (
    <div className="fade-in">
      <PageHead
        title={t('datasets.title')}
        subtitle={t('datasets.subtitle')}
        tools={
          <button type="button" className="ds-btn-primary" onClick={() => setCreating(true)}>
            {DatasetIcon.plus}{t('datasets.newDataset')}
          </button>
        }
      />

      <div className="ds-scroll">
        <div className="ds-toolbar">
          <span className="ds-search" style={{ flex: 1, minWidth: 0 }}>
            {DatasetIcon.search}
            <input
              className="ds-inp"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t('datasets.searchPlaceholder')}
              aria-label={t('datasets.searchPlaceholder')}
            />
          </span>
          <select
            aria-label={t('datasets.sortLabel')}
            className="ds-inp"
            style={{ width: 210, flex: 'none', height: 34 }}
            value={sort}
            onChange={(e) => setSort(e.target.value as Sort)}
          >
            <option value="updated">{t('datasets.sortUpdated')}</option>
            <option value="title">{t('datasets.sortTitle')}</option>
            <option value="images">{t('datasets.sortImages')}</option>
          </select>
        </div>

        <div className="ds-kpis ds-tight" style={{ gridTemplateColumns: 'repeat(4,minmax(0,1fr))' }}>
          <div className="ds-card ds-kpi">
            <div className="ds-kpi-top">
              <span className="ds-kpi-icon">{DatasetIcon.stack()}</span>
              <span style={{ marginLeft: 'auto' }}><span className="ds-kpi-meta">{t('datasets.kpiInUse', { n: inUse })}</span></span>
            </div>
            <div className="ds-kpi-val">{all.length}</div>
            <div className="ds-kpi-label">{t('datasets.kpiDatasets')}</div>
          </div>
          <div className="ds-card ds-kpi">
            <div className="ds-kpi-top">
              <span className="ds-kpi-icon">{DatasetIcon.image()}</span>
              <span style={{ marginLeft: 'auto' }}><span className="ds-kpi-meta">{totalBytes ? fmt.bytes(totalBytes) : '—'}</span></span>
            </div>
            <div className="ds-kpi-val">{totalImages.toLocaleString()}</div>
            <div className="ds-kpi-label">{t('datasets.kpiImages')}</div>
          </div>
          <div className="ds-card ds-kpi">
            <div className="ds-kpi-top"><span className="ds-kpi-icon">{DatasetIcon.link}</span></div>
            <div className="ds-kpi-val">{trainings}</div>
            <div className="ds-kpi-label">{t('datasets.kpiTrainings')}</div>
          </div>
          <div className="ds-card ds-kpi">
            <div className="ds-kpi-top">
              <span className="ds-kpi-icon">{DatasetIcon.cache()}</span>
              <span style={{ marginLeft: 'auto' }}><span className="ds-kpi-meta">{t('datasets.kpiCacheFiles', { count: cacheFiles })}</span></span>
            </div>
            <div className="ds-kpi-val">{cacheBytes ? fmt.bytes(cacheBytes) : '0'}</div>
            <div className="ds-kpi-label">{t('datasets.kpiCache')}</div>
          </div>
        </div>

        {items == null ? (
          <div className="ds-lib-grid">
            {[0, 1, 2].map((i) => <div key={i} className="ds-card" style={{ height: 262 }}><div className="ds-skel" style={{ height: 150, margin: 6, borderRadius: 10 }} /></div>)}
          </div>
        ) : all.length === 0 ? (
          <div className="ds-empty" style={{ padding: '40px 26px' }}>
            <span className="ds-sect-icon" style={{ width: 34, height: 34, borderRadius: 10 }}>{DatasetIcon.stack(16)}</span>
            <span style={{ fontWeight: 500, color: 'var(--ink-2)', marginTop: 4 }}>{t('datasets.emptyTitle')}</span>
            <span style={{ maxWidth: 460 }}>{t('datasets.emptyHint')}</span>
            <button type="button" className="ds-btn-primary" style={{ marginTop: 8 }} onClick={() => setCreating(true)}>
              {DatasetIcon.plus}{t('datasets.newDataset')}
            </button>
          </div>
        ) : shown.length === 0 ? (
          <div className="ds-empty">{t('datasets.noMatches')}</div>
        ) : (
          <div className="ds-lib-grid">
            {shown.map((d) => (
              <DatasetCard
                key={d.id}
                dataset={d}
                fmt={fmt}
                onRename={() => void actions.rename(d).then((ok) => { if (ok) void refresh() })}
                onDelete={() => void actions.remove(d).then((ok) => { if (ok) void refresh() })}
              />
            ))}
          </div>
        )}
      </div>

      {creating && (
        <NewDatasetDialog
          onCancel={() => setCreating(false)}
          onCreated={(d) => { setCreating(false); navigate(`/datasets/${d.id}`) }}
        />
      )}
    </div>
  )
}

/** Rename / delete with the confirmations both the list and the dataset page use. */
export function useDatasetActions() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { confirm, prompt } = useDialog()

  const rename = async (d: DatasetSummary): Promise<boolean> => {
    const title = (await prompt(t('datasets.renamePrompt'), { defaultValue: d.title }))?.trim()
    if (!title || title === d.title) return false
    try {
      await api.updateDataset(d.id, { title })
      return true
    } catch (e) {
      toast(String(e), 'error')
      return false
    }
  }

  const remove = async (d: DatasetSummary): Promise<boolean> => {
    const used = d.used_by.map((u) => `${u.project_title} · ${u.version_label}`)
    const message = used.length > 0
      ? t('datasets.deleteUsedConfirm', { title: d.title, count: used.length, list: used.slice(0, 6).join(', ') + (used.length > 6 ? '…' : '') })
      : t('datasets.deleteConfirm', { title: d.title })
    if (!(await confirm(message, { tone: 'danger', okText: t('datasets.deleteOk') }))) return false
    try {
      await api.deleteDataset(d.id)
      toast(t('datasets.deleted', { title: d.title }), 'success')
      return true
    } catch (e) {
      toast(String(e), 'error')
      return false
    }
  }

  return { rename, remove }
}

/** Card: a mosaic of the newest images, name, counts, where it's used. */
function DatasetCard({ dataset: d, fmt, onRename, onDelete }: {
  dataset: DatasetSummary
  fmt: ReturnType<typeof useQueueFormat>
  onRename: () => void
  onDelete: () => void
}) {
  const { t } = useTranslation()
  const covers = d.covers.slice(0, 3)
  const more = d.image_count - covers.length
  const usedProjects = Array.from(new Set(d.used_by.map((u) => u.project_title)))

  return (
    <Link to={`/datasets/${d.id}`} className="ds-card ds-lib-card">
      {covers.length === 0 ? (
        <div className="ds-lib-empty-cover">
          {DatasetIcon.image(18)}
          <span>{t('datasets.noImagesYet')}</span>
        </div>
      ) : (
        <div className={`ds-lib-cover ds-n${covers.length}`}>
          {covers.map((name, i) => (
            <span key={name}>
              <img src={api.projectThumbUrl(d.id, name, 'download', 384, undefined, true)} alt="" loading="lazy" />
              {i === covers.length - 1 && more > 0 && <span className="ds-lib-more">+{more.toLocaleString()}</span>}
            </span>
          ))}
        </div>
      )}

      <div className="ds-lib-body">
        <div className="ds-lib-head">
          <span style={{ flex: 1, minWidth: 0 }}>
            <span className="ds-lib-title" title={d.title}>{d.title}</span>
            <span className="ds-cell-key">{d.slug}</span>
          </span>
          <KebabMenu
            label={t('datasets.actions')}
            items={[
              { label: t('common.rename'), onSelect: onRename },
              { label: t('datasets.deleteOk'), onSelect: onDelete, tone: 'err', divider: true },
            ]}
          />
        </div>
        {d.note && <div className="ds-lib-note">{d.note}</div>}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <span className="ds-badge ds-mute">{t('projects.images', { n: d.image_count.toLocaleString() })}</span>
          {d.image_bytes > 0 && <span className="ds-badge ds-mute">{fmt.bytes(d.image_bytes)}</span>}
          {d.cache_bytes > 0 && <span className="ds-badge ds-info">{t('datasets.cacheBadge', { size: fmt.bytes(d.cache_bytes) })}</span>}
        </div>
      </div>

      <div className="ds-lib-foot">
        <span title={usedProjects.join(', ')}>
          {d.used_by.length > 0
            ? t('datasets.usedIn', { count: d.used_by.length, list: usedProjects.slice(0, 2).join(', ') })
            : t('datasets.notUsed')}
        </span>
        <span className="ds-mono">{fmt.ago(d.updated_at)}</span>
      </div>
    </Link>
  )
}

/** New dataset: a name, an optional note, and optionally the images of an
 *  existing project to start from. Opens the dataset (its upload page) after. */
export function NewDatasetDialog({ onCancel, onCreated }: {
  onCancel: () => void
  onCreated: (d: DatasetSummary) => void
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [title, setTitle] = useState('')
  const [note, setNote] = useState('')
  const [fromProject, setFromProject] = useState<number | ''>('')
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    void api.listProjects()
      .then((ps) => setProjects(ps.filter((p) => (p.download_image_count ?? 0) > 0 && p.kind !== 'edit')))
      .catch(() => setProjects([]))
  }, [])

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim() || busy) return
    setBusy(true)
    try {
      const d = await api.createDataset({
        title: title.trim(),
        note: note.trim() || undefined,
        from_project_id: fromProject === '' ? undefined : fromProject,
      })
      toast(t('datasets.created', { title: d.title }), 'success')
      onCreated(d)
    } catch (err) {
      toast(String(err), 'error')
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]" onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel() }}>
      <form className="ds-modal" style={{ width: 'min(500px, 94vw)' }} onSubmit={(e) => void submit(e)}>
        <div className="ds-modal-head">
          <span className="ds-sect-icon" style={{ width: 30, height: 30, borderRadius: 9 }}>{DatasetIcon.stack(15)}</span>
          <div>
            <h2 className="ds-modal-title">{t('datasets.createTitle')}</h2>
            <div style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 2 }}>{t('datasets.createSub')}</div>
          </div>
        </div>
        <label className="ds-modal-field">
          <span className="ds-modal-label">{t('datasets.createName')}</span>
          <input autoFocus className="ds-inp" style={{ height: 36 }} value={title} onChange={(e) => setTitle(e.target.value)} placeholder={t('datasets.createNamePh')} />
        </label>
        <label className="ds-modal-field">
          <span className="ds-modal-label">{t('common.notes')}</span>
          <textarea className="ds-inp" style={{ height: 'auto', padding: '8px 11px', resize: 'vertical' }} rows={2} value={note} onChange={(e) => setNote(e.target.value)} placeholder={t('datasets.createNotePh')} />
        </label>
        {projects.length > 0 && (
          <label className="ds-modal-field">
            <span className="ds-modal-label">{t('datasets.createFrom')}</span>
            <select className="ds-inp" style={{ height: 36 }} value={fromProject} onChange={(e) => setFromProject(e.target.value === '' ? '' : Number(e.target.value))}>
              <option value="">{t('datasets.createFromNone')}</option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>{t('datasets.createFromOption', { title: p.title, n: (p.download_image_count ?? 0).toLocaleString() })}</option>
              ))}
            </select>
            <span style={{ fontSize: 11.5, color: 'var(--ink-3)' }}>{t('datasets.createFromHint')}</span>
          </label>
        )}
        <div className="ds-modal-foot">
          <button type="button" className="ds-ctl" onClick={onCancel} disabled={busy}>{t('common.cancel')}</button>
          <button type="submit" className="ds-btn-primary" disabled={busy || !title.trim()}>
            {busy ? t('projects.creating') : t('datasets.createOk')}
          </button>
        </div>
      </form>
    </div>
  )
}
