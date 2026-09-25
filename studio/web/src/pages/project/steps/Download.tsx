/** Project dataset (download/) page, laid out as the approved mockup
 *  (Dataset artboard): booru scraping and file import side by side, the
 *  download stats with the format split, and the downloaded images with a
 *  bulk bar, over the data-jobs strip at the bottom.
 *
 *  Booru scraping goes through the existing estimate / download endpoints
 *  (danbooru and gelbooru — the sources the backend supports). The exclusion
 *  chips edit the same global list as the download settings. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type DownloadFile,
  type Job,
  type ProjectDetail,
  type UploadResult,
  type Version,
} from '../../../api/client'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import PathPicker from '../../../components/PathPicker'
import JobLogBar from '../../../components/ds/JobLogBar'
import PageHead from '../../../components/ds/PageHead'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'
import { useUploadProgress, type UseUploadProgress } from '../../../lib/useUploadProgress'

// Same list as studio/datasets.py:IMAGE_EXTS, plus .zip.
const UPLOAD_ACCEPT =
  '.png,.jpg,.jpeg,.webp,.bmp,.gif,.zip,image/png,image/jpeg,image/webp,image/bmp,image/gif,application/zip'

type Source = 'danbooru' | 'gelbooru'
const SOURCES: Source[] = ['danbooru', 'gelbooru']
type SortKey = 'new' | 'old' | 'name' | 'size'
type ViewMode = 'grid' | 'list'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

function fmtBytes(t: (k: string) => string, n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} ${t('overview.unitGB')}`
  if (n >= 1024 ** 2) return `${Math.round(n / 1024 ** 2)} ${t('overview.unitMB')}`
  if (n >= 1024) return `${Math.round(n / 1024)} ${t('overview.unitKB')}`
  return `${n} ${t('overview.unitB')}`
}

const extOf = (name: string) => (name.split('.').pop() ?? '?').toLowerCase().replace('jpeg', 'jpg')

const Icon = {
  upload: <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M12 16V4m0 0-4 4m4-4 4 4" /><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" /></svg>,
  minus: <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M5 12h14" /></svg>,
  plus: <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>,
  x: <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>,
  check: <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3.2" strokeLinecap="round"><path d="m5 13 4 4L19 7" /></svg>,
}

// ── upload flow (shared by the import card and "choose a server file") ──

function useUploader(pid: number, onDone: (r: UploadResult) => void) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const progress: UseUploadProgress = useUploadProgress()
  const [busy, setBusy] = useState(false)
  const [processing, setProcessing] = useState(false)

  // The backend stores the bytes, then processes them (unzip / convert) in
  // an `upload` job; poll its status until it settles.
  const waitForUpload = useCallback(async (): Promise<UploadResult> => {
    for (;;) {
      await new Promise((r) => window.setTimeout(r, 1500))
      const st = await api.getUploadStatus(pid)
      const job = st.job
      if (!job) throw new Error(t('download.uploadJobMissing'))
      if (job.status === 'done') return st.result ?? { added: [], skipped: [] }
      if (job.status === 'failed') throw new Error(job.error_msg || t('download.uploadFailed'))
      if (job.status === 'canceled') throw new Error(t('download.uploadCanceled'))
    }
  }, [pid, t])

  const report = useCallback((r: UploadResult) => {
    const skipped = r.skipped.length
    toast(
      t('download.uploadAdded', { n: r.added.length }) + (skipped ? t('download.uploadSkippedSuffix', { skipped }) : ''),
      r.added.length > 0 ? 'success' : 'error',
    )
    onDone(r)
  }, [onDone, t, toast])

  const uploadFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return false
    setBusy(true)
    setProcessing(false)
    progress.start(files.reduce((s, f) => s + f.size, 0))
    try {
      await api.uploadProjectFiles(pid, files, progress.onProgress)
      progress.finish()
      setProcessing(true)
      report(await waitForUpload())
      window.setTimeout(() => progress.reset(), 800)
      return true
    } catch (e) {
      progress.fail(e)
      toast(String(e), 'error')
      return false
    } finally {
      setProcessing(false)
      setBusy(false)
    }
  }, [pid, progress, report, toast, waitForUpload])

  const importPath = useCallback(async (path: string) => {
    setBusy(true)
    setProcessing(true)
    try {
      await api.uploadProjectFileFromPath(pid, path)
      report(await waitForUpload())
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setProcessing(false)
      setBusy(false)
    }
  }, [pid, report, toast, waitForUpload])

  return { progress, busy, processing, uploadFiles, importPath }
}

// ── page ─────────────────────────────────────────────────────────────────

export default function DownloadPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [files, setFiles] = useState<DownloadFile[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [anchor, setAnchor] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const [lastUpload, setLastUpload] = useState<UploadResult | null>(null)
  const [view, setView] = useState<ViewMode>('grid')
  const [sort, setSort] = useState<SortKey>('new')
  const [showPathPicker, setShowPathPicker] = useState(false)
  const [inTrain, setInTrain] = useState<Set<string>>(new Set())

  const refreshFiles = useCallback(async () => {
    try {
      const r = await api.listFiles(project.id)
      setFiles(r.items)
    } catch { /* ignore */ }
  }, [project.id])
  useEffect(() => { void refreshFiles() }, [refreshFiles])

  // "в train" pins: download names the active version's train set came from.
  const activeVid = activeVersion?.id ?? null
  const refreshTrain = useCallback(async () => {
    if (activeVid == null) { setInTrain(new Set()); return }
    try {
      const view = await api.getCuration(project.id, activeVid)
      const s = new Set<string>()
      for (const items of Object.values(view.right)) for (const it of items) s.add(it.origin ?? it.name)
      setInTrain(s)
    } catch { setInTrain(new Set()) }
  }, [project.id, activeVid])
  useEffect(() => { void refreshTrain() }, [refreshTrain])

  const onUploaded = useCallback((r: UploadResult) => {
    setLastUpload(r)
    void refreshFiles()
    void reload()
  }, [refreshFiles, reload])
  const uploader = useUploader(project.id, onUploaded)

  // Booru download job + its log.
  const [job, setJob] = useState<Job | null>(null)
  const [log, setLog] = useState<string[]>([])
  const refreshJob = useCallback(async () => {
    try {
      const st = await api.getDownloadStatus(project.id)
      setJob(st.job)
      setLog(st.log_tail ? st.log_tail.split('\n').filter(Boolean) : [])
    } catch { /* ignore */ }
  }, [project.id])
  useEffect(() => { void refreshJob() }, [refreshJob])

  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = job?.id ?? null
  useEventStream((evt) => {
    if (evt.type === 'job_log_appended' && evt.job_id != null && evt.job_id === jobIdRef.current) {
      setLog((prev) => [...prev.slice(-400), String(evt.text ?? '')])
    } else if (evt.type === 'job_state_changed' && evt.project_id === project.id && evt.kind === 'download') {
      void refreshJob()
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        void refreshFiles()
        void reload()
      }
    } else if (evt.type === 'project_state_changed' && evt.project_id === project.id) {
      void refreshFiles()
      void refreshTrain()
    }
  })

  const jobActive = job != null && (job.status === 'running' || job.status === 'pending')
  // "[262/400] saved …" lines carry the scrape progress.
  const jobProgress = useMemo(() => {
    for (let i = log.length - 1; i >= 0; i--) {
      const m = log[i].match(/^\[(\d+)\/(\d+)\]/)
      if (m) return { n: Number(m[1]), total: Number(m[2]) }
    }
    return null
  }, [log])

  const sortedFiles = useMemo(() => {
    const arr = [...files]
    switch (sort) {
      case 'new': return arr.sort((a, b) => (b.mtime ?? 0) - (a.mtime ?? 0) || b.name.localeCompare(a.name, undefined, { numeric: true }))
      case 'old': return arr.sort((a, b) => (a.mtime ?? 0) - (b.mtime ?? 0) || a.name.localeCompare(b.name, undefined, { numeric: true }))
      case 'size': return arr.sort((a, b) => b.size - a.size)
      default: return arr.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
    }
  }, [files, sort])

  const deleteSelected = async () => {
    if (selected.size === 0) return
    if (!(await confirm(t('download.confirmDelete', { n: selected.size }), { tone: 'danger', okText: t('common.delete') }))) return
    setDeleting(true)
    try {
      const r = await api.deleteProjectFiles(project.id, Array.from(selected))
      toast(
        t('download.deletedToast', { deleted: r.deleted.length }) + (r.missing.length ? t('download.deletedSkipped', { skipped: r.missing.length }) : ''),
        'success',
      )
      setSelected(new Set())
      setAnchor(null)
      await refreshFiles()
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setDeleting(false)
    }
  }

  const nextTo = activeVersion ? `/projects/${project.id}/v/${activeVersion.id}/curate` : `/projects/${project.id}`
  const uploadDetail = uploader.busy ? (uploader.processing ? t('download.uploadProcessing') : t('download.uploading')) : null
  const logDetail = jobActive && job
    ? `#${job.id} ${t('dataset.logScrape')}${jobProgress ? ` · ${jobProgress.n.toLocaleString()} / ${jobProgress.total.toLocaleString()}` : ''}`
    : uploadDetail
      ? `${t('dataset.logUpload')} · ${uploadDetail}`
      : null

  return (
    <div className="fade-in" style={{ minHeight: '100%', display: 'flex', flexDirection: 'column' }}>
      <PageHead
        accent
        eyebrow={t('dataset.eyebrow')}
        title={t('steps.download.title')}
        subtitle={t('dataset.subtitle')}
        tools={
          <>
            <button type="button" className="ds-ctl" onClick={() => setShowPathPicker(true)} disabled={uploader.busy}>
              {t('download.uploadFromPath')}
            </button>
            <Link className="ds-btn-primary" to={nextTo}>{t('dataset.nextCurate')}</Link>
          </>
        }
      />

      <div className="ds-scroll" style={{ flex: '1 0 auto' }}>
        <div className="ds-ds-top">
          <BooruCard pid={project.id} job={jobActive ? job : null} onStarted={(j) => { setJob(j); setLog([]) }} />
          <UploadCard uploader={uploader} lastUpload={lastUpload} onDismissResult={() => setLastUpload(null)} />
        </div>

        <StatsCard project={project} files={files} selected={selected.size} />

        <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div className="ds-card-head ds-pad">
            <div>
              <div className="ds-card-title">{t('download.sectionTitle')}</div>
              <div className="ds-card-sub">{t('dataset.gridSub', { count: files.length })}</div>
            </div>
            <div className="ds-card-tools">
              <div className="ds-seg" role="group">
                <button type="button" className={`ds-seg-item${view === 'grid' ? ' ds-is-active' : ''}`} aria-pressed={view === 'grid'} onClick={() => setView('grid')}>{t('dataset.viewGrid')}</button>
                <button type="button" className={`ds-seg-item${view === 'list' ? ' ds-is-active' : ''}`} aria-pressed={view === 'list'} onClick={() => setView('list')}>{t('dataset.viewList')}</button>
              </div>
              <select className="ds-inp" style={{ width: 170 }} value={sort} onChange={(e) => setSort(e.target.value as SortKey)} aria-label={t('dataset.sortLabel')}>
                <option value="new">{t('dataset.sortNew')}</option>
                <option value="old">{t('dataset.sortOld')}</option>
                <option value="name">{t('dataset.sortName')}</option>
                <option value="size">{t('dataset.sortSize')}</option>
              </select>
            </div>
          </div>

          {selected.size > 0 && (
            <div className="ds-bulkbar">
              <span className="ds-cbox ds-on" style={{ borderColor: 'var(--green-600)', background: 'var(--green-600)' }}>{Icon.check}</span>
              <span style={{ fontWeight: 500 }}>{t('dataset.selectedN', { n: selected.size })}</span>
              <span style={{ flex: 1 }} />
              {selected.size < files.length && (
                <button type="button" className="ds-ctl ds-ghost" style={{ height: 26, color: 'var(--green-text)' }} onClick={() => setSelected(new Set(files.map((f) => f.name)))} disabled={deleting}>
                  {t('common.selectAll')}
                </button>
              )}
              <button type="button" className="ds-ctl ds-ghost" style={{ height: 26, color: 'var(--green-text)' }} onClick={() => { setSelected(new Set()); setAnchor(null) }} disabled={deleting}>
                {t('dataset.clearSel')}
              </button>
              <button type="button" className="ds-btn-danger" style={{ height: 26 }} onClick={() => void deleteSelected()} disabled={deleting} title={t('download.deleteTitle')}>
                {deleting ? t('download.deleting') : t('download.deleteBtn', { n: selected.size })}
              </button>
            </div>
          )}

          <div className="ds-card-body" style={{ paddingTop: 12 }}>
            {view === 'grid' ? (
              <div style={{ height: files.length === 0 ? 'auto' : 'min(62vh, 640px)' }}>
                <ImageGrid
                  items={sortedFiles.map((f) => ({
                    name: f.name,
                    thumbUrl: api.projectThumbUrl(project.id, f.name),
                    badge: inTrain.has(f.name) ? t('dataset.inTrain') : undefined,
                  }))}
                  selected={selected}
                  onSelect={(name, e) => {
                    const r = applySelection(selected, name, e, sortedFiles.map((f) => f.name), anchor)
                    setSelected(r.next)
                    setAnchor(r.anchor)
                  }}
                  onActivate={(name) => setPreviewIdx(sortedFiles.findIndex((f) => f.name === name))}
                  onPreview={(name) => setPreviewIdx(sortedFiles.findIndex((f) => f.name === name))}
                  clickMode="activate"
                  columnsClass="grid-cols-[repeat(auto-fill,minmax(104px,1fr))]"
                  ariaLabel="downloaded-grid"
                  emptyHint={t('download.emptyHint')}
                />
              </div>
            ) : (
              <FileList
                files={sortedFiles}
                selected={selected}
                inTrain={inTrain}
                onToggle={(name) => {
                  const next = new Set(selected)
                  if (next.has(name)) next.delete(name); else next.add(name)
                  setSelected(next)
                  setAnchor(name)
                }}
                onOpen={(name) => setPreviewIdx(sortedFiles.findIndex((f) => f.name === name))}
              />
            )}
          </div>
        </div>
      </div>

      <JobLogBar
        title={t('dataset.logTitle')}
        running={jobActive || uploader.busy}
        detail={logDetail}
        pct={jobActive && jobProgress && jobProgress.total > 0 ? Math.round((jobProgress.n / jobProgress.total) * 100) : null}
        log={log}
      />

      {previewIdx !== null && previewIdx >= 0 && sortedFiles[previewIdx] && (
        <ImagePreviewModal
          src={api.projectThumbUrl(project.id, sortedFiles[previewIdx].name, 'download', 1600)}
          caption={sortedFiles[previewIdx].name}
          hasPrev={previewIdx > 0}
          hasNext={previewIdx < sortedFiles.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() => previewIdx > 0 && setPreviewIdx(previewIdx - 1)}
          onNext={() => previewIdx < sortedFiles.length - 1 && setPreviewIdx(previewIdx + 1)}
        />
      )}
      {showPathPicker && (
        <PathPicker
          dirOnly={false}
          onClose={() => setShowPathPicker(false)}
          onPick={(path) => { setShowPathPicker(false); void uploader.importPath(path) }}
        />
      )}
    </div>
  )
}

// ── booru ────────────────────────────────────────────────────────────────

function BooruCard({ pid, job, onStarted }: {
  pid: number
  job: Job | null
  onStarted: (job: Job) => void
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { prompt } = useDialog()
  const [source, setSource] = useState<Source>('danbooru')
  const [query, setQuery] = useState('')
  const [estimate, setEstimate] = useState<{ query: string; source: Source; count: number } | null>(null)
  const [searching, setSearching] = useState(false)
  const [limit, setLimit] = useState(100)
  const [starting, setStarting] = useState(false)
  const [exclude, setExclude] = useState<string[] | null>(null)
  const [hasKeys, setHasKeys] = useState<Record<Source, boolean> | null>(null)

  useEffect(() => {
    let cancelled = false
    void api.getSecrets()
      .then((s) => {
        if (cancelled) return
        setExclude(s.download.exclude_tags ?? [])
        setHasKeys({
          danbooru: !!(s.danbooru.username && s.danbooru.api_key),
          gelbooru: !!(s.gelbooru.user_id && s.gelbooru.api_key),
        })
      })
      .catch(() => { if (!cancelled) setExclude([]) })
    return () => { cancelled = true }
  }, [])

  const saveExclude = async (next: string[]) => {
    const prev = exclude
    setExclude(next)
    setEstimate(null)
    try {
      await api.updateSecrets({ download: { exclude_tags: next } })
    } catch (e) {
      setExclude(prev)
      toast(String(e), 'error')
    }
  }

  const search = async () => {
    const tag = query.trim()
    if (!tag) { toast(t('download.tagEmpty'), 'error'); return }
    setSearching(true)
    try {
      const r = await api.estimateDownload(pid, { tag, api_source: source })
      setEstimate({ query: tag, source, count: r.count })
      if (r.count > 0) setLimit((l) => Math.min(Math.max(1, l), r.count))
      if (r.count === 0) toast(t('download.noResults'), 'info')
    } catch (e) {
      setEstimate(null)
      toast(String(e), 'error')
    } finally {
      setSearching(false)
    }
  }

  const start = async () => {
    const tag = query.trim()
    if (!tag) { toast(t('download.tagEmpty'), 'error'); return }
    if (limit < 1) { toast(t('download.countMin'), 'error'); return }
    setStarting(true)
    try {
      const j = await api.startDownload(pid, { tag, count: limit, api_source: source })
      toast(t('download.started', { id: j.id }), 'success')
      onStarted(j)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setStarting(false)
    }
  }

  const fresh = estimate && estimate.query === query.trim() && estimate.source === source ? estimate : null
  const found = fresh && fresh.count > 0 ? fresh.count : null
  const pct = found ? Math.round((Math.min(limit, found) / found) * 100) : null
  const desc = found == null
    ? t('dataset.limitDescNoEstimate')
    : pct != null && pct < 100 ? t('dataset.limitDesc', { pct }) : t('dataset.limitDescAll')

  return (
    <div className="ds-card" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="ds-card-head ds-pad">
        <div>
          <div className="ds-card-title">{t('dataset.booruTitle')}</div>
          <div className="ds-card-sub">{t('dataset.booruSub')}</div>
        </div>
        <div className="ds-card-tools">
          <span className="ds-cap">{t('dataset.source')}</span>
          <div className="ds-seg" role="group" aria-label={t('dataset.source')}>
            {SOURCES.map((s) => (
              <button key={s} type="button" className={`ds-seg-item${source === s ? ' ds-is-active' : ''}`} aria-pressed={source === s} onClick={() => setSource(s)}>{s}</button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ padding: '0 17px 14px', display: 'flex', flexDirection: 'column', gap: 12 }}>
        <form className="ds-searchgroup" onSubmit={(e) => { e.preventDefault(); void search() }}>
          <span className="ds-q">
            <input className="ds-mono" value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t('dataset.queryPlaceholder')} style={{ paddingLeft: 12 }} aria-label={t('dataset.queryPlaceholder')} />
          </span>
          <button type="submit" disabled={searching}>{searching ? t('download.querying') : t('dataset.find')}</button>
        </form>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7, flexWrap: 'wrap' }} title={t('dataset.excludeHint')}>
          <span className="ds-cap" style={{ flex: 'none' }}>{t('dataset.exclude')}</span>
          {(exclude ?? []).map((tag) => (
            <span key={tag} className="ds-chip">
              {tag}
              <button type="button" className="ds-chip-x" aria-label={`${t('dataset.excludeRemove')} ${tag}`} onClick={() => void saveExclude((exclude ?? []).filter((x) => x !== tag))}>{Icon.x}</button>
            </span>
          ))}
          <button
            type="button"
            className="ds-chip-add"
            disabled={exclude == null}
            onClick={async () => {
              const v = (await prompt(t('dataset.excludePrompt')))?.trim()
              if (v && !(exclude ?? []).includes(v)) void saveExclude([...(exclude ?? []), v])
            }}
          >
            {t('dataset.excludeAdd')}
          </button>
        </div>
        {hasKeys && !hasKeys[source] && (
          <div className="ds-note ds-warn" style={{ fontSize: 11.5 }}>{t('dataset.noCredentials', { source })}</div>
        )}
      </div>

      <div className="ds-field" style={{ borderTop: '1px solid var(--line)', borderBottom: 0 }}>
        <div className="ds-field-txt">
          <div className="ds-field-name"><span className="ds-label">{t('dataset.limitLabel')}</span><span className="ds-key">limit</span></div>
          <div className="ds-field-desc">{desc}</div>
        </div>
        <div className="ds-field-ctl" style={{ width: 156, justifyContent: 'center' }}>
          <span className="ds-stepper" style={{ width: 112 }}>
            <input
              inputMode="numeric"
              value={limit}
              onChange={(e) => setLimit(Math.max(0, Number(e.target.value.replace(/\D/g, '')) || 0))}
              aria-label={t('dataset.limitLabel')}
            />
            <button type="button" aria-label={t('dataset.less')} onClick={() => setLimit((l) => Math.max(1, l - 10))}>{Icon.minus}</button>
            <button type="button" aria-label={t('dataset.more')} onClick={() => setLimit((l) => (found ? Math.min(found, l + 10) : l + 10))}>{Icon.plus}</button>
          </span>
        </div>
      </div>

      <div className="ds-cardfoot" style={{ marginTop: 'auto' }}>
        <span className="ds-hint">
          {t('dataset.matches')}
          <b className="ds-mono" style={{ color: 'var(--ink)', fontWeight: 500 }}>
            {fresh ? (fresh.count >= 0 ? fresh.count.toLocaleString() : t('dataset.matchesUnknown')) : '—'}
          </b>
          {job && <span className="ds-badge ds-ok"><span className="ds-dot ds-dot-run" />{t('dataset.taskBadge', { id: job.id })}</span>}
        </span>
        <span className="ds-cardfoot-act">
          <button type="button" className="ds-btn-primary" style={{ height: 34 }} onClick={() => void start()} disabled={starting || !query.trim() || limit < 1}>
            {starting ? t('download.downloading') : t('dataset.downloadN', { n: limit.toLocaleString() })}
          </button>
        </span>
      </div>
    </div>
  )
}

// ── import ───────────────────────────────────────────────────────────────

function UploadCard({ uploader, lastUpload, onDismissResult }: {
  uploader: ReturnType<typeof useUploader>
  lastUpload: UploadResult | null
  onDismissResult: () => void
}) {
  const { t } = useTranslation()
  const inputRef = useRef<HTMLInputElement>(null)
  const [picked, setPicked] = useState<File[]>([])
  const [dragging, setDragging] = useState(false)
  const { progress, busy, processing } = uploader

  const choose = (fl: FileList | null) => { if (fl && fl.length > 0) setPicked(Array.from(fl)) }
  const reset = () => { setPicked([]); if (inputRef.current) inputRef.current.value = '' }
  const totalBytes = picked.reduce((s, f) => s + f.size, 0)
  const st = progress.state
  const pct = st.phase === 'uploading' && st.total > 0 ? Math.round((st.loaded / st.total) * 100)
    : st.phase === 'processing' || processing ? 100 : null

  return (
    <div className="ds-card" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="ds-card-head ds-pad">
        <div>
          <div className="ds-card-title">{t('download.uploadPanel')}</div>
          <div className="ds-card-sub">{t('dataset.uploadSub')}</div>
        </div>
      </div>
      <div className="ds-card-body" style={{ paddingTop: 0, display: 'flex', flexDirection: 'column', gap: 10, flex: 1 }}>
        <label
          className="ds-empty"
          style={{
            padding: 22, minHeight: 170, flex: 1, gap: 6, justifyContent: 'center', cursor: busy ? 'default' : 'pointer',
            ...(dragging ? { borderColor: 'var(--green-600)', background: 'var(--green-soft)', color: 'var(--green-text)' } : {}),
          }}
          onDragOver={(e) => { e.preventDefault(); if (!busy) setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); if (!busy && e.dataTransfer.files?.length) choose(e.dataTransfer.files) }}
        >
          <input ref={inputRef} type="file" multiple accept={UPLOAD_ACCEPT} className="hidden" disabled={busy} onChange={(e) => choose(e.target.files)} />
          {Icon.upload}
          <span style={{ fontWeight: 500, color: 'var(--ink-2)' }}>{t('dataset.dropHint')}</span>
          {picked.length > 0 && (
            <span style={{ fontSize: 11 }} title={picked.map((f) => f.name).join(', ')}>
              {t('dataset.filesPicked', { count: picked.length, size: fmtBytes(t, totalBytes) })}
            </span>
          )}
        </label>

        {(picked.length > 0 || busy) && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {pct != null ? (
              <>
                <span className="ds-meter" style={{ flex: 1 }}><i style={{ width: `${pct}%` }} /></span>
                <span className="ds-mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>
                  {st.phase === 'uploading' ? t('dataset.uploadPct', { pct }) : t('dataset.processing')}
                </span>
              </>
            ) : (
              <span style={{ flex: 1 }} />
            )}
            {!busy && <button type="button" className="ds-ctl ds-ghost" onClick={reset}>{t('common.cancel')}</button>}
            <button
              type="button"
              className="ds-btn-primary"
              disabled={busy || picked.length === 0}
              onClick={async () => { if (await uploader.uploadFiles(picked)) reset() }}
            >
              {processing ? t('download.uploadProcessing') : busy ? t('download.uploading') : t('download.uploadCount', { n: picked.length })}
            </button>
          </div>
        )}
        {st.phase === 'error' && st.error && <div className="ds-note ds-warn" style={{ fontSize: 11.5 }}>{st.error}</div>}

        {lastUpload && (
          <details className="ds-note" style={{ fontSize: 11.5 }}>
            <summary style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8, listStyle: 'none' }}>
              <span>
                {t('download.added')} <b>{lastUpload.added.length}</b>
                {lastUpload.skipped.length > 0 && <> · {t('download.skipped')} <b>{lastUpload.skipped.length}</b></>}
              </span>
              <span style={{ flex: 1 }} />
              <button type="button" aria-label={t('common.close')} onClick={(e) => { e.preventDefault(); onDismissResult() }}>{Icon.x}</button>
            </summary>
            {lastUpload.skipped.length > 0 ? (
              <ul className="ds-mono" style={{ margin: '8px 0 0', padding: 0, listStyle: 'none', maxHeight: 140, overflow: 'auto' }}>
                {lastUpload.skipped.map((s, i) => <li key={`${s.name}-${i}`}>{s.name} — {s.reason}</li>)}
              </ul>
            ) : (
              <div style={{ marginTop: 6 }}>{t('download.allSucceeded')}</div>
            )}
          </details>
        )}
      </div>
    </div>
  )
}

// ── stats ────────────────────────────────────────────────────────────────

const FORMAT_COLORS = ['var(--green)', '#cfe3ad', 'var(--line-2)', '#e3ecd4', '#d4d5d1']

function StatsCard({ project, files, selected }: {
  project: ProjectDetail
  files: DownloadFile[]
  selected: number
}) {
  const { t } = useTranslation()
  const [convertsToPng, setConvertsToPng] = useState<boolean | null>(null)
  useEffect(() => {
    let cancelled = false
    void api.getSecrets().then((s) => { if (!cancelled) setConvertsToPng(!!s.download.convert_to_png) }).catch(() => {})
    return () => { cancelled = true }
  }, [])

  const formats = useMemo(() => {
    const m = new Map<string, number>()
    for (const f of files) m.set(extOf(f.name), (m.get(extOf(f.name)) ?? 0) + 1)
    return Array.from(m, ([ext, n]) => ({ ext, n })).sort((a, b) => b.n - a.n)
  }, [files])
  const total = files.reduce((s, f) => s + f.size, 0)

  return (
    <div className="ds-card">
      <div className="ds-card-head ds-pad">
        <div>
          <div className="ds-card-title">{t('dataset.statsTitle')}</div>
          <div className="ds-card-sub">{t('dataset.statsSub')}</div>
        </div>
      </div>
      <div className="ds-statgrid ds-ds-stats" style={{ borderTop: '1px solid var(--line)', marginTop: 13 }}>
        <div><div className="ds-cap">{t('dataset.statTotal')}</div><div className="ds-stat-v">{(project.download_image_count ?? files.length).toLocaleString()}</div></div>
        <div><div className="ds-cap">{t('dataset.statVisible')}</div><div className="ds-stat-v">{files.length.toLocaleString()}</div></div>
        <div><div className="ds-cap">{t('dataset.statSelected')}</div><div className="ds-stat-v" style={{ color: 'var(--green-text)' }}>{selected.toLocaleString()}</div></div>
        <div><div className="ds-cap">{t('dataset.statSize')}</div><div className="ds-stat-v">{files.length ? fmtBytes(t, total) : '—'}</div></div>
      </div>
      {formats.length > 0 && (
        <div style={{ borderTop: '1px solid var(--line)', padding: '13px 17px 16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 9 }}>
            <span className="ds-cap" style={{ flex: 1 }}>{t('dataset.formats')}</span>
            {convertsToPng && formats.length > 1 && <span className="ds-kpi-meta">{t('dataset.formatsPngNote')}</span>}
          </div>
          <div style={{ display: 'flex', gap: 3, height: 12 }}>
            {formats.map((f, i) => (
              <span
                key={f.ext}
                style={{
                  flex: f.n, background: FORMAT_COLORS[i % FORMAT_COLORS.length],
                  borderRadius: `${i === 0 ? 4 : 0}px ${i === formats.length - 1 ? 4 : 0}px ${i === formats.length - 1 ? 4 : 0}px ${i === 0 ? 4 : 0}px`,
                }}
              />
            ))}
          </div>
          <div style={{ display: 'flex', gap: 18, marginTop: 9, flexWrap: 'wrap' }}>
            {formats.map((f, i) => (
              <span key={f.ext} className="ds-legend">
                <s style={{ background: FORMAT_COLORS[i % FORMAT_COLORS.length] }} />
                {f.ext} · {f.n.toLocaleString()} · {Math.round((f.n / files.length) * 100)}%
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── list view ────────────────────────────────────────────────────────────

function FileList({ files, selected, inTrain, onToggle, onOpen }: {
  files: DownloadFile[]
  selected: Set<string>
  inTrain: Set<string>
  onToggle: (name: string) => void
  onOpen: (name: string) => void
}) {
  const { t } = useTranslation()
  if (files.length === 0) return <div className="ds-muted" style={{ fontSize: 12.5 }}>{t('download.emptyHint')}</div>
  return (
    <div style={{ maxHeight: 'min(62vh, 640px)', overflow: 'auto', margin: '0 -17px' }}>
      <table className="ds-tbl">
        <thead>
          <tr>
            <th style={{ width: 40 }} />
            <th>{t('common.file')}</th>
            <th>{t('dataset.colTrain')}</th>
            <th>{t('dataset.colMeta')}</th>
            <th>{t('common.size')}</th>
          </tr>
        </thead>
        <tbody>
          {files.map((f) => {
            const on = selected.has(f.name)
            return (
              <tr key={f.name} style={on ? { background: 'var(--green-soft)' } : undefined}>
                <td>
                  <button type="button" className={`ds-cbox${on ? ' ds-on' : ''}`} aria-pressed={on} aria-label={`${on ? t('common.deselect') : t('common.select')} ${f.name}`} onClick={() => onToggle(f.name)}>
                    {Icon.check}
                  </button>
                </td>
                <td><button type="button" className="ds-mono" style={{ fontSize: 12, textAlign: 'left' }} onClick={() => onOpen(f.name)}>{f.name}</button></td>
                <td>{inTrain.has(f.name) ? <span className="ds-badge ds-ok">{t('dataset.inTrain')}</span> : <span className="ds-muted">—</span>}</td>
                <td className="ds-muted">{f.has_meta ? 'booru' : '—'}</td>
                <td className="ds-num">{fmtBytes(t, f.size)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
