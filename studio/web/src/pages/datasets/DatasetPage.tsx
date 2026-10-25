/** One shared dataset. "Photos" is the same upload / booru / grid page a
 *  project's dataset step uses; "Files & cache" shows where everything lives:
 *  the image folder, the training caches (per model / format) and which
 *  trainings use the dataset. */
import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  api,
  isDataset,
  type DatasetCacheGroup,
  type DatasetCacheInfo,
  type DatasetSummary,
  type ProjectDetail,
} from '../../api/client'
import KebabMenu from '../../components/ds/KebabMenu'
import PageHead from '../../components/ds/PageHead'
import { useDialog } from '../../components/Dialog'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'
import { useQueueFormat } from '../../lib/queueFormat'
import { DatasetWorkspace } from '../project/steps/Download'
import { DatasetIcon, useDatasetActions } from './DatasetsPage'

type Tab = 'photos' | 'files'

const FAMILY_LABEL: Record<string, string> = {
  anima: 'Anima',
  krea2: 'Krea 2',
  moss_image: 'MOSS Image',
}

const SIZE_COLORS = { photos: 'var(--green)', latents: '#9fc3ea', text: '#e3c77c' }

export default function DatasetPage() {
  const { t } = useTranslation()
  const { did: didParam } = useParams()
  const did = Number(didParam)
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const tab: Tab = params.get('tab') === 'files' ? 'files' : 'photos'
  const fmt = useQueueFormat()
  const actions = useDatasetActions()

  const [project, setProject] = useState<ProjectDetail | null>(null)
  const [summary, setSummary] = useState<DatasetSummary | null>(null)
  const [missing, setMissing] = useState(false)

  const reload = useCallback(async () => {
    try {
      const [p, s] = await Promise.all([api.getProject(did), api.getDataset(did)])
      if (!isDataset(p)) { setMissing(true); return }
      setProject(p)
      setSummary(s)
    } catch {
      setMissing(true)
    }
  }, [did])
  useEffect(() => { void reload() }, [reload])
  useEventStream((evt) => {
    if (evt.type === 'project_state_changed' && evt.project_id === did) void reload()
  })

  const setTab = (next: Tab) => {
    const p = new URLSearchParams(params)
    if (next === 'files') p.set('tab', 'files')
    else p.delete('tab')
    setParams(p, { replace: true })
  }

  if (missing) {
    return (
      <div className="fade-in">
        <PageHead title={t('datasets.notFound')} />
        <div className="ds-scroll">
          <Link className="ds-ctl" to="/datasets" style={{ alignSelf: 'flex-start' }}>{t('datasets.backToList')}</Link>
        </div>
      </div>
    )
  }
  if (!project || !summary) {
    return (
      <div className="fade-in">
        <div className="ds-scroll" style={{ paddingTop: 24 }}>
          <div className="ds-skel" style={{ height: 30, width: 260 }} />
          <div className="ds-skel" style={{ height: 220 }} />
        </div>
      </div>
    )
  }

  const head = (pathButton: React.ReactNode | null) => (
    <PageHead
      eyebrow={<Link to="/datasets" style={{ color: 'inherit' }}>{t('datasets.eyebrow')}</Link>}
      title={summary.title}
      subtitle={summary.note || t('datasets.pageSubtitle')}
      tools={
        <>
          <div className="ds-seg" role="tablist" aria-label={t('datasets.tabsLabel')}>
            <button type="button" role="tab" aria-selected={tab === 'photos'} className={`ds-seg-item${tab === 'photos' ? ' ds-is-active' : ''}`} onClick={() => setTab('photos')}>
              {t('datasets.tabPhotos')}<span className="ds-mono" style={{ opacity: 0.55, marginLeft: 6 }}>{summary.image_count.toLocaleString()}</span>
            </button>
            <button type="button" role="tab" aria-selected={tab === 'files'} className={`ds-seg-item${tab === 'files' ? ' ds-is-active' : ''}`} onClick={() => setTab('files')}>
              {t('datasets.tabFiles')}{summary.cache_bytes > 0 && <span className="ds-mono" style={{ opacity: 0.55, marginLeft: 6 }}>{fmt.bytes(summary.cache_bytes)}</span>}
            </button>
          </div>
          {pathButton}
          <KebabMenu
            trigger="icon"
            label={t('datasets.actions')}
            items={[
              { label: t('common.rename'), onSelect: () => void actions.rename(summary).then((ok) => { if (ok) void reload() }) },
              { label: t('datasets.deleteOk'), tone: 'err', divider: true, onSelect: () => void actions.remove(summary).then((ok) => { if (ok) navigate('/datasets') }) },
            ]}
          />
        </>
      }
    />
  )

  if (tab === 'photos') {
    return (
      <DatasetWorkspace
        project={project}
        activeVersion={null}
        reload={reload}
        renderHead={(pathButton) => head(pathButton)}
      />
    )
  }

  return (
    <div className="fade-in" style={{ minHeight: '100%', display: 'flex', flexDirection: 'column' }}>
      {head(null)}
      <FilesPanel dataset={summary} onChanged={() => void reload()} />
    </div>
  )
}

// ── files & cache ────────────────────────────────────────────────────────

function FilesPanel({ dataset: d, onChanged }: { dataset: DatasetSummary; onChanged: () => void }) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const fmt = useQueueFormat()
  const [cache, setCache] = useState<DatasetCacheInfo | null>(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    try { setCache(await api.getDatasetCache(d.id)) } catch { setCache({ groups: [], total_files: 0, total_bytes: 0, path: '' }) }
  }, [d.id])
  useEffect(() => { void load() }, [load, d.cache_bytes])

  const groups = cache?.groups ?? []
  const sum = (kind: DatasetCacheGroup['kind']) => groups.filter((g) => g.kind === kind)
    .reduce((acc, g) => ({ files: acc.files + g.files, bytes: acc.bytes + g.bytes }), { files: 0, bytes: 0 })
  const latents = sum('latents')
  const text = sum('text')
  const total = d.image_bytes + latents.bytes + text.bytes
  const parts = [
    { key: 'photos', bytes: d.image_bytes, color: SIZE_COLORS.photos, label: t('datasets.partPhotos') },
    { key: 'latents', bytes: latents.bytes, color: SIZE_COLORS.latents, label: t('datasets.partLatents') },
    { key: 'text', bytes: text.bytes, color: SIZE_COLORS.text, label: t('datasets.partText') },
  ].filter((p) => p.bytes > 0)

  const clear = async (g?: DatasetCacheGroup) => {
    const message = g
      ? t('datasets.clearGroupConfirm', { what: kindLabel(t, g.kind), model: familyLabel(g.family) ?? g.group, size: fmt.bytes(g.bytes) })
      : t('datasets.clearAllConfirm', { size: fmt.bytes(cache?.total_bytes ?? 0) })
    if (!(await confirm(message, { tone: 'warn', okText: t('datasets.clearOk') }))) return
    setBusy(true)
    try {
      const r = await api.clearDatasetCache(d.id, g?.kind, g?.group)
      toast(t('datasets.clearedToast', { size: fmt.bytes(r.freed_bytes) }), 'success')
      await load()
      onChanged()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const copyPath = async () => {
    try {
      await navigator.clipboard.writeText(d.path)
      toast(t('datasets.pathCopied'), 'success')
    } catch {
      toast(d.path, 'info')
    }
  }

  const folders = [
    { path: 'download/', what: t('datasets.folderPhotos'), files: d.image_count, bytes: d.image_bytes, color: SIZE_COLORS.photos },
    { path: 'cache/latents/', what: t('datasets.folderLatents'), files: latents.files, bytes: latents.bytes, color: SIZE_COLORS.latents },
    { path: 'cache/text/', what: t('datasets.folderText'), files: text.files, bytes: text.bytes, color: SIZE_COLORS.text },
  ]

  return (
    <div className="ds-scroll" style={{ flex: '1 0 auto' }}>
      <div className="ds-lib-files">
        <div className="ds-lib-col">
          {/* where everything lives */}
          <div className="ds-card">
            <div className="ds-card-head ds-pad">
              <div>
                <div className="ds-card-title">{t('datasets.folderTitle')}</div>
                <div className="ds-card-sub">{t('datasets.folderSub')}</div>
              </div>
              <div className="ds-card-tools">
                <span className="ds-kpi-meta">{t('datasets.totalSize', { size: fmt.bytes(total) })}</span>
              </div>
            </div>
            <div className="ds-lib-path">
              {DatasetIcon.folder(14)}
              <code title={d.path}>{d.path}</code>
              <button type="button" className="ds-ctl ds-ghost" style={{ height: 24, padding: '0 8px', flex: 'none' }} onClick={() => void copyPath()}>{t('datasets.copyPath')}</button>
            </div>
            {parts.length > 0 && (
              <>
                <div className="ds-lib-sizebar">
                  {parts.map((p) => <span key={p.key} style={{ flex: p.bytes, background: p.color }} />)}
                </div>
                <div className="ds-lib-legend">
                  {parts.map((p) => (
                    <span key={p.key} className="ds-legend"><s style={{ background: p.color }} />{p.label}, {fmt.bytes(p.bytes)}, {Math.round((p.bytes / total) * 100)}%</span>
                  ))}
                </div>
              </>
            )}
            <table className="ds-tbl" style={{ borderTop: '1px solid var(--line)' }}>
              <thead>
                <tr><th>{t('datasets.colFolder')}</th><th>{t('datasets.colWhat')}</th><th>{t('datasets.colFiles')}</th><th>{t('common.size')}</th></tr>
              </thead>
              <tbody>
                {folders.map((f) => (
                  <tr key={f.path}>
                    <td><span className="ds-lib-folder"><span className="ds-src-dot" style={{ width: 8, height: 8, borderRadius: 3, background: f.color, flex: 'none' }} /><code>{f.path}</code></span></td>
                    <td className="ds-muted">{f.what}</td>
                    <td className="ds-num">{f.files ? f.files.toLocaleString() : '—'}</td>
                    <td className="ds-num">{f.bytes ? fmt.bytes(f.bytes) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* training caches, one row per format */}
          <div className="ds-card">
            <div className="ds-card-head ds-pad">
              <div>
                <div className="ds-card-title">{t('datasets.cacheTitle')}</div>
                <div className="ds-card-sub">{t('datasets.cacheSub')}</div>
              </div>
            </div>
            {cache == null ? (
              <div className="ds-card-body" style={{ paddingTop: 0 }}><div className="ds-skel" style={{ height: 60 }} /></div>
            ) : groups.length === 0 ? (
              <div className="ds-card-body" style={{ paddingTop: 0 }}>
                <div className="ds-empty">
                  <span className="ds-sect-icon" style={{ width: 30, height: 30, borderRadius: 9 }}>{DatasetIcon.cache(15)}</span>
                  <span style={{ fontWeight: 500, color: 'var(--ink-2)' }}>{t('datasets.cacheEmptyTitle')}</span>
                  <span style={{ maxWidth: 440 }}>{t('datasets.cacheEmptyHint')}</span>
                </div>
              </div>
            ) : (
              <table className="ds-tbl" style={{ borderTop: '1px solid var(--line)' }}>
                <thead>
                  <tr>
                    <th>{t('datasets.colKind')}</th><th>{t('datasets.colModel')}</th><th>{t('datasets.colFormat')}</th>
                    <th>{t('datasets.colFiles')}</th><th>{t('common.size')}</th><th>{t('datasets.colUsed')}</th><th style={{ width: 44 }} />
                  </tr>
                </thead>
                <tbody>
                  {groups.map((g) => (
                    <tr key={`${g.kind}/${g.group}`}>
                      <td><span className={`ds-badge ${g.kind === 'latents' ? 'ds-info' : 'ds-warn'}`}>{kindLabel(t, g.kind)}</span></td>
                      <td>{familyLabel(g.family) ?? <span className="ds-muted">—</span>}</td>
                      <td>
                        <span className="ds-lib-fp" title={`${g.fingerprint ?? g.group}\n${g.path}`}>{g.fingerprint ?? g.group}</span>
                        {g.flip_copies && <span className="ds-badge ds-mute" style={{ marginLeft: 6 }}>{t('datasets.flipCopies')}</span>}
                      </td>
                      <td className="ds-num">{g.files.toLocaleString()}</td>
                      <td className="ds-num">{fmt.bytes(g.bytes)}</td>
                      <td className="ds-num">{g.last_used_at ? fmt.ago(g.last_used_at) : '—'}</td>
                      <td>
                        <KebabMenu label={t('datasets.actions')} items={[{ label: t('datasets.clearGroup'), tone: 'err', onSelect: () => void clear(g) }]} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <div className="ds-cardfoot">
              <span className="ds-hint">{t('datasets.cacheSafeHint')}</span>
              <span className="ds-cardfoot-act">
                <button type="button" className="ds-btn-danger" style={{ height: 30 }} disabled={busy || groups.length === 0} onClick={() => void clear()}>
                  {t('datasets.clearAll')}
                </button>
              </span>
            </div>
          </div>
        </div>

        <div className="ds-lib-col">
          {/* where it is used */}
          <div className="ds-card" style={{ overflow: 'hidden' }}>
            <div className="ds-card-head ds-pad">
              <div>
                <div className="ds-card-title">{t('datasets.usedTitle')}</div>
                <div className="ds-card-sub">{t('datasets.usedSub', { count: d.used_by.length })}</div>
              </div>
            </div>
            {d.used_by.length === 0 ? (
              <div className="ds-card-body" style={{ paddingTop: 0 }}>
                <span className="ds-kpi-meta">{t('datasets.usedEmpty')}</span>
              </div>
            ) : (
              d.used_by.map((u) => (
                <Link key={u.version_id} className="ds-lib-use" to={`/projects/${u.project_id}/v/${u.version_id}/curate`}>
                  <span className="ds-sect-icon">{DatasetIcon.folder(12)}</span>
                  <span title={u.project_title}>{u.project_title}</span>
                  <span className="ds-verpill" style={{ height: 22 }}>{u.version_label}</span>
                </Link>
              ))
            )}
          </div>

          <div className="ds-note ds-info" style={{ flexDirection: 'column', gap: 6 }}>
            <b style={{ fontWeight: 600 }}>{t('datasets.howTitle')}</b>
            <span>{t('datasets.howBody')}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function kindLabel(t: (k: string) => string, kind: DatasetCacheGroup['kind']): string {
  return kind === 'latents' ? t('datasets.kindLatents') : t('datasets.kindText')
}

function familyLabel(family: string | null): string | null {
  if (!family) return null
  return FAMILY_LABEL[family] ?? family
}
