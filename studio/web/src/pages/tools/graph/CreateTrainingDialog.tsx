import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { api, type ApiError, type ConfigData, type ProjectSummary, type Task, type Version } from '../../../api/client'
import { activeValues, configPatch, formatNumber, paramById, sameConfigValue, valuesFromConfig } from './model'
import type { Param, Values } from './types'
import { Modal, Switch, useValueLabel } from './ui'

const LABEL_RE = /^[A-Za-z0-9_.-]+$/

function show(v: unknown): string {
  if (v === undefined) return '—'
  if (typeof v === 'number') return formatNumber(v)
  if (Array.isArray(v)) return `[${v.map(show).join(', ')}]`
  return String(v)
}

/**
 * New version from an empty place (or a card): copies a base version — its
 * dataset, reg set and config — writes the place's parameters into the config
 * and can queue it right away. The card on the board gets linked to the new
 * version / task so it shows up as "in the queue".
 */
export default function CreateTrainingDialog({ values, params, onClose, onCreated }: {
  values: Values
  params: Param[]
  onClose: () => void
  onCreated: (r: {
    projectId: number; versionId: number; task: Task | null; label: string; projectTitle: string
    /** Card values as the saved config actually has them. */
    values: Values
    /** Config keys the server changed on save (incompatible combinations). */
    adjusted: string[]
  }) => Promise<void> | void
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const label = useValueLabel()
  const byId = useMemo(() => paramById(params), [params])
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null)
  const [pid, setPid] = useState<number | null>(null)
  const [versions, setVersions] = useState<Version[]>([])
  const [baseVid, setBaseVid] = useState<number | null>(null)
  const [baseCfg, setBaseCfg] = useState<ConfigData | null>(null)
  const [cfgState, setCfgState] = useState<'idle' | 'loading' | 'none' | 'ok'>('idle')
  const [newLabel, setNewLabel] = useState('')
  const [queue, setQueue] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const act = activeValues(values, params)
  const { patch, skipped } = useMemo(() => configPatch(values, params), [values, params])

  useEffect(() => {
    api.listProjects().then((ps) => {
      const live = ps.filter((p) => !p.archived_at)
      setProjects(live)
      if (live.length) setPid(live[0].id)
    }).catch(() => setProjects([]))
  }, [])

  useEffect(() => {
    if (pid == null) return
    let alive = true
    setVersions([])
    setBaseVid(null)
    api.listVersions(pid).then((vs) => {
      if (!alive) return
      setVersions(vs)
      const proj = projects?.find((p) => p.id === pid)
      const def = vs.find((v) => v.id === proj?.active_version_id) ?? vs[vs.length - 1]
      setBaseVid(def?.id ?? null)
    }).catch(() => { if (alive) setVersions([]) })
    return () => { alive = false }
  }, [pid, projects])

  useEffect(() => {
    if (pid == null || baseVid == null) { setBaseCfg(null); setCfgState('idle'); return }
    let alive = true
    setCfgState('loading')
    api.getVersionConfig(pid, baseVid).then((r) => {
      if (!alive) return
      setBaseCfg(r.config)
      setCfgState(r.has_config && r.config ? 'ok' : 'none')
    }).catch(() => { if (alive) { setBaseCfg(null); setCfgState('none') } })
    const base = versions.find((v) => v.id === baseVid)
    if (base) {
      const taken = new Set(versions.map((v) => v.label))
      let n = 1
      while (taken.has(`${base.label}-g${n}`)) n++
      setNewLabel(`${base.label}-g${n}`)
    }
    return () => { alive = false }
  }, [pid, baseVid, versions])

  const labelError = !newLabel ? t('graph.labelRequired')
    : !LABEL_RE.test(newLabel) ? t('graph.labelInvalid')
      : versions.some((v) => v.label === newLabel) ? t('graph.labelTaken') : null

  const changes = Object.entries(patch).map(([k, v]) => ({ key: k, from: baseCfg?.[k], to: v }))
  const canCreate = pid != null && baseVid != null && cfgState === 'ok' && !labelError && !busy

  const create = async () => {
    if (pid == null || baseVid == null) return
    setBusy(true)
    setError(null)
    let created: Version | null = null
    let stage: 'version' | 'config' | 'queue' = 'version'
    const proj = projects?.find((p) => p.id === pid)
    let saved: ConfigData = {}
    const done = (task: Task | null) => {
      const adjusted = Object.keys(patch).filter((k) => !sameConfigValue(saved[k], patch[k]))
      return onCreated({
        projectId: pid, versionId: created!.id, task, label: newLabel, projectTitle: proj?.title ?? '',
        values: { ...values, ...valuesFromConfig(saved, params) }, adjusted,
      })
    }
    try {
      created = await api.createVersion(pid, { label: newLabel, fork_from_version_id: baseVid, note: t('graph.versionNote') })
      stage = 'config'
      let cfg = (await api.getVersionConfig(pid, created.id)).config ?? {}
      const family = patch.model_family
      if (typeof family === 'string' && family !== cfg.model_family) {
        cfg = (await api.switchModelFamily(family, cfg)).config
      }
      saved = (await api.putVersionConfig(pid, created.id, { ...cfg, ...patch })).config
      stage = 'queue'
      const task = queue ? await api.enqueueVersionTraining(pid, created.id) : null
      await done(task)
      onClose()
    } catch (e) {
      const msg = (e as ApiError)?.message || t('graph.createFailed')
      if (created && stage === 'config') {
        // The config was rejected: remove the half-made version so it doesn't linger.
        await api.deleteVersion(pid, created.id).catch(() => undefined)
        setError(msg)
      } else if (created && stage === 'queue') {
        // The version is fine, only queueing failed: keep it and link the card.
        await done(null)
        setError(t('graph.queueFailed', { msg }))
      } else {
        setError(msg)
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal title={t('graph.createTitle')} sub={t('graph.createSub')} onClose={onClose} width={620}
      foot={<>
        <button type="button" className="ds-ctl ds-ghost" onClick={onClose}>{t('common.cancel')}</button>
        <button type="button" className="ds-btn-primary" disabled={!canCreate} onClick={() => void create()}>
          {busy ? t('graph.creating') : queue ? t('graph.createAndQueue') : t('graph.createVersion')}
        </button>
      </>}>
      <div className="gr-form">
        <div className="gr-kv gr-kv-box">
          {params.filter((p) => act[p.id] !== undefined).map((p) => (
            <span key={p.id} className="gr-kv-row"><span className="gr-kv-k">{p.name}</span><span className="gr-kv-v">{label(p, act[p.id])}</span></span>
          ))}
          {!Object.keys(act).length && <span className="ds-cell-key">{t('graph.noValuesSet')}</span>}
        </div>

        {projects && projects.length === 0 ? (
          <div className="ds-note ds-info"><span>{t('graph.noProjects')}</span>
            <button type="button" className="gr-link" onClick={() => navigate('/')}>{t('graph.goProjects')}</button></div>
        ) : (
          <>
            <label className="gr-mfield">
              <span className="gr-pfield-name">{t('graph.project')}</span>
              <select className="ds-inp" value={pid ?? ''} onChange={(e) => setPid(Number(e.target.value))} disabled={!projects}>
                {(projects ?? []).map((p) => <option key={p.id} value={p.id}>{p.title}</option>)}
              </select>
            </label>
            <label className="gr-mfield">
              <span className="gr-pfield-name">{t('graph.baseVersion')}</span>
              <select className="ds-inp" value={baseVid ?? ''} onChange={(e) => setBaseVid(Number(e.target.value))} disabled={!versions.length}>
                {versions.map((v) => <option key={v.id} value={v.id}>{v.label}</option>)}
              </select>
              <span className="ds-ctl-note">{t('graph.baseVersionHint')}</span>
            </label>
            {cfgState === 'none' && <div className="ds-note ds-warn"><span>{t('graph.baseNoConfig')}</span></div>}
            <label className="gr-mfield">
              <span className="gr-pfield-name">{t('graph.newVersionLabel')}</span>
              <input className={`ds-inp ds-mono${labelError ? ' gr-bad' : ''}`} value={newLabel} onChange={(e) => setNewLabel(e.target.value.trim())} />
              {labelError && newLabel && <span className="ds-ctl-note" style={{ color: 'var(--red-text)' }}>{labelError}</span>}
            </label>

            {cfgState === 'ok' && (
              <div className="gr-mfield">
                <span className="gr-pfield-name">{t('graph.configChanges')}</span>
                <div className="gr-changes">
                  {changes.length === 0 && <span className="ds-cell-key">{t('graph.noConfigChanges')}</span>}
                  {changes.map((c) => {
                    const same = JSON.stringify(c.from) === JSON.stringify(c.to)
                    return (
                      <div key={c.key} className={`gr-change${same ? ' gr-same' : ''}`}>
                        <span className="gr-mono">{c.key}</span>
                        <span className="gr-change-v">{same ? show(c.to) : <><s>{show(c.from)}</s> → <b>{show(c.to)}</b></>}</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
            {skipped.length > 0 && (
              <div className="ds-note ds-info"><span>{t('graph.notApplied', { names: skipped.map((id) => byId.get(id)?.name ?? id).join(', ') })}</span></div>
            )}
            <Switch on={queue} onChange={setQueue} label={t('graph.queueNow')} />
          </>
        )}
        {error && <div className="ds-note ds-warn"><span>{error}</span></div>}
      </div>
    </Modal>
  )
}
