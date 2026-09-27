import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Outlet, useLocation, useMatch, useNavigate, useParams } from 'react-router-dom'
import { api, type ProjectDetail } from '../../api/client'
import { useProjectCtxSetter, useSelectedProjectSetter } from '../../context/ProjectContext'
import { useDialog } from '../../components/Dialog'
import { useToast } from '../../components/Toast'
import { useEventStream } from '../../lib/useEventStream'
import ExportBundleDialog, { type BundleExportOpts } from '../../components/ExportBundleDialog'

export default function ProjectLayout() {
  const { t } = useTranslation()
  const { pid } = useParams()
  const projectId = pid ? Number(pid) : NaN
  const navigate = useNavigate()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const setCtx = useProjectCtxSetter()
  const setSelected = useSelectedProjectSetter()
  const [project, setProject] = useState<ProjectDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState<{ forkFrom: number | null } | null>(null)
  const [creatingBusy, setCreatingBusy] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [showExportDialog, setShowExportDialog] = useState(false)
  const projectRef = useRef<ProjectDetail | null>(null)
  projectRef.current = project
  // Version-switch request sequence number: on rapid successive switches only the result of the
  // last switch counts, so a late-arriving response/rollback can't overwrite the newer choice.
  const switchSeqRef = useRef(0)
  // Version-switch guard: step pages with state that needs confirmation before discarding
  // (e.g. unsaved edits in TagEdit) register here. Returning false cancels the switch. Switching
  // versions remounts the step page without going through route navigation, so useBlocker can't
  // catch it -- hence this separate channel.
  const switchGuardRef = useRef<(() => Promise<boolean>) | null>(null)
  const setVersionSwitchGuard = useCallback(
    (g: (() => Promise<boolean>) | null) => { switchGuardRef.current = g },
    [],
  )
  // Under the version-scoped route (v/:vid/*), the Outlet is keyed on activeVersion.id: switching
  // versions forces the step page to remount, so local state / caches all get replaced, ruling out
  // "still showing old data under the new version" (e.g. Curation's view-cache guard wouldn't
  // otherwise refetch on a vid change). Overview / Download are project-scoped (Overview also has
  // its own local selectedVid) and don't remount on switch.
  const inVersionScope = useMatch('/projects/:pid/v/:vid/*') != null
  const location = useLocation()

  const reload = useCallback(async () => {
    if (!Number.isFinite(projectId)) return
    try {
      const p = await api.getProject(projectId)
      setProject(p)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [projectId])

  useEffect(() => {
    void reload()
  }, [reload])

  useEventStream((evt) => {
    if (
      (evt.type === 'project_state_changed' && evt.project_id === projectId) ||
      (evt.type === 'version_state_changed' && evt.project_id === projectId)
    ) {
      void reload()
    } else if (
      (
        evt.type === 'version_train_zip_ready' ||
        evt.type === 'version_train_zip_failed' ||
        evt.type === 'version_bundle_zip_ready' ||
        evt.type === 'version_bundle_zip_failed'
      ) &&
      evt.project_id === projectId
    ) {
      setExporting(false)
      if (evt.type === 'version_train_zip_failed' || evt.type === 'version_bundle_zip_failed') {
        const err = typeof evt.error === 'string' ? evt.error : '?'
        toast(t('layout.exportFailed', { error: err }), 'error')
      }
    }
  })

  useEffect(() => {
    if (!exporting) return
    const tid = window.setTimeout(() => setExporting(false), 60_000)
    return () => window.clearTimeout(tid)
  }, [exporting])

  const activeVersion = useMemo(() => {
    if (!project) return null
    const aid = project.active_version_id
    return project.versions.find((v) => v.id === aid) ?? project.versions[0] ?? null
  }, [project])

  const handleSelectVersion = useCallback(async (vid: number) => {
    const prev = projectRef.current
    if (!prev || prev.active_version_id === vid) return
    const guard = switchGuardRef.current
    if (guard && !(await guard())) return
    const prevVid = prev.active_version_id
    const seq = ++switchSeqRef.current
    // Optimistic update: switch locally first, then wait for the backend. If activeVersion stayed
    // at the old value during the activate round-trip, "switch version then immediately start
    // training" would enqueue the old version (#386).
    setProject((cur) => (cur ? { ...cur, active_version_id: vid } : cur))
    try {
      await api.activateVersion(prev.id, vid)
      // Don't apply the response on success (thin response): the optimistic value is already the
      // new server state; full data converges via project_state_changed → reload.
    } catch (e) {
      if (seq === switchSeqRef.current) {
        // Roll back only the active_version_id field, not the whole object -- avoids swallowing other updates from an in-flight reload.
        setProject((cur) => (cur ? { ...cur, active_version_id: prevVid } : cur))
        toast(String(e), 'error')
      }
    }
  }, [toast])

  const handleExportTrain = useCallback(() => {
    if (!projectRef.current || exporting) return
    setShowExportDialog(true)
  }, [exporting])

  const handleExportBundleConfirm = useCallback(async (opts: BundleExportOpts) => {
    setShowExportDialog(false)
    if (!projectRef.current) return
    const av = projectRef.current.versions.find(
      (v) => v.id === projectRef.current!.active_version_id
    ) ?? projectRef.current.versions[0] ?? null
    if (!av) return
    setExporting(true)
    const bundleOpts = {
      train: opts.train,
      trainCaptions: opts.trainCaptions,
      reg: opts.reg,
      regCaptions: opts.regCaptions,
      includeConfig: opts.includeConfig,
      trainLatentCache: opts.trainLatentCache,
      regLatentCache: opts.regLatentCache,
      trainMasks: opts.trainMasks,
    }
    if (opts.destination === 'download') {
      const filename = `${projectRef.current.slug}-${av.label}.bundle.zip`
      const a = document.createElement('a')
      a.href = api.versionBundleZipUrl(projectRef.current.id, av.id, bundleOpts)
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      return
    }
    try {
      const result = await api.exportBundleToDataExports(projectRef.current.id, av.id, bundleOpts)
      toast(t('layout.exportSavedToDataExports', { filename: result.filename, path: result.path }), 'success')
      setExporting(false)
    } catch (e) {
      setExporting(false)
      toast(t('layout.exportFailed', { error: String(e) }), 'error')
    }
  }, [t, toast])

  const handleDeleteVersion = useCallback(async (vid: number) => {
    if (!projectRef.current) return
    const v = projectRef.current.versions.find((x) => x.id === vid)
    if (!v) return
    if (!(await confirm(t('layout.deleteVersionConfirm', { label: v.label }), { tone: 'danger', okText: t('layout.deleteVersionOk') }))) return
    const pid = projectRef.current.id
    try {
      await api.deleteVersion(pid, vid)
      await reload()
      toast(t('layout.deleteVersionDone', { label: v.label }), 'success')
      navigate(`/projects/${pid}`)
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [reload, toast, navigate, confirm, t])

  const handleCreateVersion = useCallback(async (label: string, forkFromVersionId: number | null) => {
    if (!projectRef.current || creatingBusy) return
    // Creating a new version activates it → step page remounts, so it also needs to pass the switch guard (canceling leaves the dialog in place).
    const guard = switchGuardRef.current
    if (guard && !(await guard())) return
    setCreatingBusy(true)
    try {
      const body: { label: string; fork_from_version_id?: number } = { label }
      if (forkFromVersionId !== null) body.fork_from_version_id = forkFromVersionId
      const v = await api.createVersion(projectRef.current.id, body)
      await api.activateVersion(projectRef.current.id, v.id)
      await reload()
      setCreating(null)
      toast(
        forkFromVersionId !== null
          ? t('layout.versionCreatedFromFork', { label })
          : t('layout.versionCreated', { label }),
        'success',
      )
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setCreatingBusy(false)
    }
  }, [creatingBusy, reload, toast, t])

  useEffect(() => {
    if (!project || !setCtx) return
    setCtx({
      project,
      activeVersion,
      reload,
      onSelectVersion: handleSelectVersion,
      onCreateVersion: (forkFromVid?: number) => setCreating({ forkFrom: forkFromVid ?? null }),
      onExportTrain: handleExportTrain,
      onDeleteVersion: handleDeleteVersion,
      exporting,
    })
  }, [project, activeVersion, reload, handleSelectVersion, handleExportTrain, handleDeleteVersion, exporting, setCtx])


  useEffect(() => {
    return () => { setCtx?.(null) }
  }, [setCtx])

  // Sticky snapshot: written on load/refresh, and deliberately **not cleared** when leaving the
  // project page (see the ProjectContext comment), so the sidebar keeps the selected project
  // across pages for navigation. Opening a different project overwrites this snapshot.
  useEffect(() => {
    if (project) setSelected?.({ project, activeVersion })
  }, [project, activeVersion, setSelected])

  if (error) {
    return (
      <div className="m-4 p-3 rounded-md border border-err bg-err-soft text-err font-mono text-sm">
        {error}
      </div>
    )
  }
  if (!project) {
    return <p className="p-6 text-fg-tertiary">{t('layout.loading')}</p>
  }

  return (
    <div className="flex flex-col h-full">
      <StepOutlet stepKey={`${inVersionScope ? activeVersion?.id ?? -1 : 'project'}:${location.pathname}`} context={{
        project,
        activeVersion,
        reload,
        onCreateVersion: (forkFromVid?: number) => setCreating({ forkFrom: forkFromVid ?? null }),
        creatingVersionBusy: creatingBusy,
        setVersionSwitchGuard,
      }} remountKey={inVersionScope ? activeVersion?.id ?? -1 : 'project'} />
      {creating && (
        <NewVersionDialog
          existingLabels={project.versions.map((v) => v.label)}
          existingVersions={project.versions.map((v) => ({ id: v.id, label: v.label }))}
          initialForkFrom={creating.forkFrom}
          busy={creatingBusy}
          onCancel={() => { if (creatingBusy) return; setCreating(null) }}
          onSubmit={handleCreateVersion}
        />
      )}
      {showExportDialog && (
        <ExportBundleDialog
          onConfirm={handleExportBundleConfirm}
          onCancel={() => setShowExportDialog(false)}
        />
      )}
    </div>
  )
}

/** Step outlet that eases in on every step change; `remountKey` still forces a
 *  full remount when the active version changes (see inVersionScope above). */
function StepOutlet({ stepKey, remountKey, context }: {
  stepKey: string
  remountKey: string | number
  context: unknown
}) {
  return (
    <div key={stepKey} className="ds-page-anim flex flex-col flex-1 min-h-0">
      <Outlet key={remountKey} context={context} />
    </div>
  )
}

export function NewVersionDialog({
  existingLabels,
  existingVersions,
  initialForkFrom = null,
  busy = false,
  onCancel,
  onSubmit,
}: {
  existingLabels: string[]
  existingVersions: { id: number; label: string }[]
  /** forkFrom version id pre-filled when the dialog opens (null = not pre-filled, user picks their own). */
  initialForkFrom?: number | null
  busy?: boolean
  onCancel: () => void
  onSubmit: (label: string, forkFromVersionId: number | null) => void
}) {
  const { t } = useTranslation()
  const [label, setLabel] = useState('')
  const [forkFrom, setForkFrom] = useState<string>(initialForkFrom != null ? String(initialForkFrom) : '')
  const [err, setErr] = useState<string | null>(null)

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (busy) return
    const l = label.trim()
    if (!l) return setErr(t('layout.labelEmpty'))
    if (!/^[A-Za-z0-9_.-]+$/.test(l))
      return setErr(t('layout.labelInvalid'))
    if (existingLabels.includes(l)) return setErr(t('layout.labelExists'))
    const fid = forkFrom === '' ? null : Number(forkFrom)
    onSubmit(l, fid)
  }

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]"
      onClick={onCancel}
    >
      <form
        onClick={(e) => e.stopPropagation()}
        onSubmit={submit}
        className="ds-modal"
        style={{ width: 'min(460px, 92vw)' }}
      >
        <div className="ds-modal-head">
          <span className="ds-optcard-ico" style={{ background: 'var(--green)', color: 'var(--green-ink)' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="6" r="2.4" /><circle cx="6" cy="18" r="2.4" /><circle cx="18" cy="8" r="2.4" /><path d="M6 8.4v7.2" /><path d="M18 10.4c0 3.4-3.2 3.9-6 4.4" /></svg>
          </span>
          <h2 className="ds-modal-title">{t('layout.newVersionTitle')}</h2>
        </div>
        <label className="ds-modal-field">
          <span className="ds-modal-label">{t('layout.versionName')}</span>
          <input
            autoFocus
            value={label}
            onChange={(e) => { setLabel(e.target.value); setErr(null) }}
            className="ds-inp"
            style={{ height: 36 }}
            placeholder={t('layout.labelPlaceholder')}
          />
        </label>
        {existingVersions.length > 0 && (
          <label className="ds-modal-field">
            <span className="ds-modal-label">{t('layout.forkFrom')}</span>
            <select
              value={forkFrom}
              onChange={(e) => setForkFrom(e.target.value)}
              className="ds-inp"
              style={{ height: 36 }}
            >
              <option value="">{t('layout.forkBlank')}</option>
              {existingVersions.map((v) => (
                <option key={v.id} value={String(v.id)}>
                  {t('layout.forkFromVersion', { label: v.label })}
                </option>
              ))}
            </select>
            {forkFrom !== '' && (
              <span className="ds-kpi-meta">{t('layout.forkNote')}</span>
            )}
          </label>
        )}
        {err && <div className="ds-note ds-err" style={{ padding: '8px 11px' }}>{err}</div>}
        <div className="ds-modal-foot">
          <button type="button" onClick={onCancel} disabled={busy} className="ds-ctl">
            {t('common.cancel')}
          </button>
          <button type="submit" disabled={busy} className="ds-btn-primary" style={{ minWidth: 110 }}>
            {busy ? t('layout.creatingBtn') : t('common.create')}
          </button>
        </div>
      </form>
    </div>
  )
}

export interface ProjectLayoutContext {
  project: ProjectDetail
  activeVersion: ReturnType<typeof Object.assign>
  reload: () => Promise<void>
}
