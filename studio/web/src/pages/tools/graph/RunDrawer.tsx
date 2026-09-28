import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import FieldLabel from '../../../components/ds/FieldLabel'
import KebabMenu from '../../../components/ds/KebabMenu'
import {
  formatNumber, isActive, knownValues, oneStepAway, optionOf, paramById, parseNumber, scientific,
  taskMatches,
} from './model'
import type { GImage, GraphTask, Param, Run, RunStatus, RunValue, Values } from './types'
import { RUN_STATUSES } from './types'
import { Ico, Stars, Thumb, useValueLabel } from './ui'

export default function RunDrawer({
  run, runs, params, tasks, taskValues, onClose, onUpdate, onUpload, onOpenImage, onImportSamples,
  onDuplicate, onDelete, onCreateTraining, onDownloadConfig, onOpenRun, onShowNeighbours, onLinkTask,
}: {
  run: Run
  runs: Run[]
  params: Param[]
  tasks: GraphTask[]
  taskValues: Map<number, Values>
  onClose: () => void
  onUpdate: (patch: Partial<Run>) => void
  onUpload: (files: File[], step: number | null) => void
  onOpenImage: (img: GImage) => void
  onImportSamples: (taskId: number) => void
  onDuplicate: () => void
  onDelete: () => void
  onCreateTraining: () => void
  onDownloadConfig: () => void
  onOpenRun: (r: Run) => void
  onShowNeighbours: () => void
  onLinkTask: (task: GraphTask | null) => void
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const label = useValueLabel()
  const byId = useMemo(() => paramById(params), [params])
  const fileRef = useRef<HTMLInputElement>(null)
  const [uploadStep, setUploadStep] = useState('')
  const [dragOver, setDragOver] = useState(false)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || document.querySelector('.gr-modal-back')) return
      onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const setValue = (id: string, v: RunValue | undefined) => {
    const values = { ...run.values }
    if (v === undefined) delete values[id]
    else values[id] = v
    onUpdate({ values })
  }

  const linked = run.task_id != null ? tasks.find((x) => x.id === run.task_id) : undefined
  const usedTaskIds = useMemo(() => new Set(runs.map((r) => r.task_id).filter((x): x is number => x != null)), [runs])
  const suggestions = useMemo(() => (run.task_id != null ? [] : tasks.filter((task) =>
    !usedTaskIds.has(task.id) && taskMatches(run.values, taskValues.get(task.id) ?? {}, params)).slice(0, 4)),
  [run.task_id, run.values, tasks, usedTaskIds, taskValues, params])
  const neighbours = useMemo(() => oneStepAway(run, runs, params), [run, runs, params])

  const archivedWithValues = params.filter((p) => p.archived && run.values[p.id] !== undefined)

  const byStep = useMemo(() => {
    const m = new Map<string, GImage[]>()
    for (const img of [...run.images].sort((a, b) => (a.step ?? 1e12) - (b.step ?? 1e12) || a.sort - b.sort)) {
      const k = img.step == null ? '∅' : String(img.step)
      m.set(k, [...(m.get(k) ?? []), img])
    }
    return [...m.entries()]
  }, [run.images])

  const upload = (files: File[]) => {
    const imgs = files.filter((f) => /^image\/(png|jpeg|webp)$/.test(f.type) || /\.(png|jpe?g|webp)$/i.test(f.name))
    if (!imgs.length) return
    onUpload(imgs, parseNumber(uploadStep))
  }

  return (
    <aside className="gr-drawer" aria-label={t('graph.cardDetails')}
      onDragOver={(e) => { if (e.dataTransfer.types.includes('Files')) { e.preventDefault(); setDragOver(true) } }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setDragOver(false) }}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); upload(Array.from(e.dataTransfer.files)) }}
    >
      <input ref={fileRef} type="file" accept="image/png,image/jpeg,image/webp" multiple hidden
        onChange={(e) => { upload(Array.from(e.target.files ?? [])); e.target.value = '' }} />
      <div className="gr-drawer-head">
        <NameInput value={run.name} onSave={(name) => onUpdate({ name })} placeholder={t('graph.untitled')} />
        <KebabMenu label={t('graph.cardActions')} items={[
          { label: t('graph.duplicate'), onSelect: onDuplicate },
          { label: t('graph.createTraining'), onSelect: onCreateTraining },
          { label: t('graph.downloadConfig'), onSelect: onDownloadConfig },
          { label: t('common.delete'), onSelect: onDelete, tone: 'err', divider: true },
        ]} />
        <button type="button" className="ds-kebab" onClick={onClose} aria-label={t('common.close')}>{Ico.close}</button>
      </div>

      <div className="gr-drawer-body">
        {/* ── result ── */}
        <section className="ds-fgroup">
          <div className="ds-fgroup-head"><span className="ds-fgroup-title">{t('graph.result')}</span></div>
          <Row label={t('common.status')}>
            <select className="ds-inp" value={run.status} aria-label={t('common.status')}
              onChange={(e) => onUpdate({ status: e.target.value as RunStatus })}>
              {RUN_STATUSES.map((s) => <option key={s} value={s}>{t(`graph.status_${s}`)}</option>)}
            </select>
          </Row>
          <Row label={t('graph.rating')}>
            <span className="gr-row-inline">
              <span className="ds-cell-key">{run.rating ? t(`graph.rating_${run.rating}`) : t('graph.notRated')}</span>
              <Stars value={run.rating} onChange={(n) => onUpdate({ rating: n })} size={18} />
            </span>
          </Row>
          <Row label={t('graph.favorite')}>
            <label className={`ds-switch${run.favorite ? ' ds-on' : ''}`} style={{ cursor: 'pointer' }}>
              <input type="checkbox" className="sr-only" checked={run.favorite} onChange={(e) => onUpdate({ favorite: e.target.checked })} aria-label={t('graph.favorite')} />
              <i />
            </label>
          </Row>
        </section>

        {/* ── linked training ── */}
        <section className="ds-fgroup">
          <div className="ds-fgroup-head">
            <span className="ds-fgroup-title">{t('graph.training')}</span>
            {(run.task_id != null || run.version_id != null) && (
              <span className="ds-fgroup-count">
                <KebabMenu label={t('graph.training')} items={[{ label: t('graph.unlinkTask'), onSelect: () => onLinkTask(null), tone: 'err' }]} />
              </span>
            )}
          </div>
          {linked || run.task_id != null || (run.project_id && run.version_id) ? (
            <>
              <Row label={linked?.project_title ? `${linked.project_title}${linked.version_label ? ` / ${linked.version_label}` : ''}` : t('graph.training')}
                tip={run.task_id != null ? `#${run.task_id}` : undefined}>
                <span className="gr-row-inline">
                  {linked
                    ? <span className="ds-badge ds-mute">#{linked.id} {t(`graph.task_${linked.status}`, { defaultValue: linked.status })}</span>
                    : <span className="ds-cell-key">{run.task_id != null ? t('graph.taskGone') : t('graph.versionOnly')}</span>}
                </span>
              </Row>
              <div className="gr-group-acts">
                {linked && <button type="button" className="ds-ctl" onClick={() => navigate(`/queue/${linked.id}`)}>{Ico.queue}{t('graph.openTraining')}</button>}
                {(linked?.project_id ?? run.project_id) && (linked?.version_id ?? run.version_id) && (
                  <button type="button" className="ds-ctl"
                    onClick={() => navigate(`/projects/${linked?.project_id ?? run.project_id}/v/${linked?.version_id ?? run.version_id}/train`)}>
                    {Ico.link}{t('graph.openVersion')}
                  </button>
                )}
                {linked && <button type="button" className="ds-ctl" onClick={() => onImportSamples(linked.id)}>{Ico.download}{t('graph.importSamples')}</button>}
              </div>
            </>
          ) : (
            <>
              {suggestions.map((task) => (
                <Row key={task.id} label={t('graph.looksLike', { id: task.id })}
                  tip={`${task.project_title ?? ''}${task.version_label ? ` / ${task.version_label}` : ''}, ${t(`graph.task_${task.status}`, { defaultValue: task.status })}`}>
                  <button type="button" className="ds-ctl" onClick={() => onLinkTask(task)}>{t('graph.link')}</button>
                </Row>
              ))}
              <Row label={t('graph.linkTask')}>
                <select className="ds-inp" value="" aria-label={t('graph.linkTask')}
                  onChange={(e) => { const task = tasks.find((x) => x.id === Number(e.target.value)); if (task) onLinkTask(task) }}>
                  <option value="" disabled>{t('graph.pickTask')}</option>
                  {tasks.map((task) => (
                    <option key={task.id} value={task.id}>
                      #{task.id} {task.project_title ?? task.name}{task.version_label ? ` / ${task.version_label}` : ''}
                    </option>
                  ))}
                </select>
              </Row>
              <div className="gr-group-acts">
                <button type="button" className="ds-btn-primary" onClick={onCreateTraining}>{Ico.play}{t('graph.createTraining')}</button>
              </div>
            </>
          )}
        </section>

        {/* ── samples ── */}
        <section className={`ds-fgroup${dragOver ? ' gr-drop' : ''}`}>
          <div className="ds-fgroup-head">
            <span className="ds-fgroup-title">{t('graph.samples')}</span>
            <span className="ds-fgroup-count">{t('graph.nSamples', { count: run.images.length })}</span>
          </div>
          {run.images.length > 0 && (
            <div className="gr-samples">
              {byStep.flatMap(([, imgs]) => imgs).map((img) => (
                <button key={img.id} type="button" className={`gr-sample${img.step != null && run.best_step === img.step ? ' gr-best' : ''}`}
                  onClick={() => onOpenImage(img)} title={[img.prompt, img.comment].filter(Boolean).join('\n')}>
                  <Thumb id={img.id} />
                  <span className="gr-step">{img.step != null ? t('graph.stepN', { n: img.step }) : t('graph.noStep')}</span>
                  {img.rating ? <span className="gr-sample-r">{'★'.repeat(img.rating)}</span> : null}
                </button>
              ))}
            </div>
          )}
          {/* One strip for adding: drop / paste / pick, and the step the new images belong to. */}
          <div className="ds-dropstrip gr-addstrip">
            {Ico.image}
            <span className="gr-addstrip-t"><b>{t('graph.dropShort')}</b><span>{t('graph.dropShortSub')}</span></span>
            <label className="gr-addstrip-step" title={t('graph.stepForNewHint')}>
              <span>{t('graph.step')}</span>
              <input className="ds-inp" value={uploadStep} onChange={(e) => setUploadStep(e.target.value)} placeholder={t('graph.auto')} inputMode="numeric" />
            </label>
            <button type="button" className="ds-ctl" onClick={() => fileRef.current?.click()}>{t('graph.pickFiles')}</button>
          </div>
        </section>

        {/* ── parameters ── */}
        <section className="ds-fgroup">
          <div className="ds-fgroup-head"><span className="ds-fgroup-title">{t('graph.parameters')}</span></div>
          {params.filter((p) => !p.archived && !p.hidden && isActive(p, run.values, byId)).map((p) => (
            <Row key={p.id} label={p.name} tip={p.description}>
              <ValueEditor param={p} value={run.values[p.id]} runs={runs} onChange={(v) => setValue(p.id, v)} />
            </Row>
          ))}
          {params.some((p) => p.hidden && !p.archived && run.values[p.id] !== undefined) && (
            <div className="gr-group-note">
              {t('graph.hiddenValues')}: {params.filter((p) => p.hidden && !p.archived && run.values[p.id] !== undefined).map((p) => `${p.name} = ${label(p, run.values[p.id])}`).join(', ')}
            </div>
          )}
          {archivedWithValues.length > 0 && (
            <div className="gr-group-note">
              {t('graph.archivedParams')}: {archivedWithValues.map((p) => `${p.name} = ${label(p, run.values[p.id])}`).join(', ')}
            </div>
          )}
        </section>

        {/* ── one parameter away ── */}
        <section className="ds-fgroup">
          <div className="ds-fgroup-head">
            <span className="ds-fgroup-title">{t('graph.oneAway')}</span>
            <span className="ds-fgroup-count">
              {neighbours.length > 0
                ? <button type="button" className="gr-link" onClick={onShowNeighbours}>{t('graph.showOnBoard')}</button>
                : neighbours.length}
            </span>
          </div>
          {neighbours.length === 0 ? <div className="gr-group-note">{t('graph.noNeighbours')}</div> : neighbours.map(({ run: r, param }) => {
            const p = byId.get(param)
            return (
              <button key={r.id} type="button" className="ds-field gr-neigh-row" onClick={() => onOpenRun(r)}>
                <span className="ds-field-txt"><span className="ds-label">{r.name || t('graph.untitled')}</span></span>
                <span className="gr-neigh-diff">{p?.name}: <s>{label(p, p && isActive(p, run.values, byId) ? run.values[param] : undefined)}</s> → <b>{label(p, p && isActive(p, r.values, byId) ? r.values[param] : undefined)}</b></span>
              </button>
            )
          })}
        </section>

        {/* ── notes ── */}
        <section className="ds-fgroup">
          <div className="ds-fgroup-head"><span className="ds-fgroup-title">{t('common.notes')}</span></div>
          <div className="ds-field ds-stack">
            <TextArea value={run.notes} onSave={(notes) => onUpdate({ notes })} placeholder={t('graph.notesPh')} />
          </div>
          <Row label={t('graph.resultsLink')}>
            <LineInput value={run.link} onSave={(link) => onUpdate({ link })} placeholder={t('graph.resultsLinkPh')} mono />
          </Row>
          <Row label={t('graph.tags')}>
            <LineInput value={run.tags.join(', ')} onSave={(s) => onUpdate({ tags: s.split(',').map((x) => x.trim()).filter(Boolean) })} placeholder={t('graph.tagsPh')} />
          </Row>
        </section>
      </div>
    </aside>
  )
}

/** A form row exactly like the training config form: name left, control right. */
function Row({ label, tip, children }: { label: string; tip?: string; children: React.ReactNode }) {
  return (
    <div className="ds-field">
      <div className="ds-field-txt">
        <div className="ds-field-name"><FieldLabel label={label} tip={tip || undefined} /></div>
      </div>
      <div className="ds-field-ctl">{children}</div>
    </div>
  )
}

/** Editor for one parameter value; never turns "not set" into No / 0. */
export function ValueEditor({ param, value, runs, onChange }: {
  param: Param; value: RunValue | undefined; runs: Run[]; onChange: (v: RunValue | undefined) => void
}) {
  const { t } = useTranslation()
  const label = useValueLabel()
  if (param.type === 'bool') {
    return (
      <select className="ds-inp" value={value === undefined ? '' : String(value)} aria-label={param.name}
        onChange={(e) => onChange(e.target.value === '' ? undefined : e.target.value)}>
        <option value="">{t('graph.unset')}</option>
        {param.options.filter((o) => !o.archived || o.id === value).map((o) => <option key={o.id} value={o.id}>{label(param, o.id)}</option>)}
      </select>
    )
  }
  if (param.type === 'list') {
    const current = optionOf(param, value)
    return (
      <select className="ds-inp" value={value === undefined ? '' : String(value)}
        onChange={(e) => onChange(e.target.value === '' ? undefined : e.target.value)} aria-label={param.name}>
        <option value="">{t('graph.unset')}</option>
        {param.options.filter((o) => !o.archived || o.id === current?.id).map((o) => (
          <option key={o.id} value={o.id}>{label(param, o.id)}{o.archived ? ` (${t('graph.archived')})` : ''}</option>
        ))}
      </select>
    )
  }
  if (param.type === 'number') {
    return <NumberInput param={param} value={typeof value === 'number' ? value : undefined} runs={runs} onChange={onChange} />
  }
  return <LineInput value={typeof value === 'string' ? value : ''} onSave={(s) => onChange(s.trim() ? s.trim() : undefined)} placeholder={t('graph.unset')} />
}

function NumberInput({ param, value, runs, onChange }: {
  param: Param; value: number | undefined; runs: Run[]; onChange: (v: number | undefined) => void
}) {
  const { t } = useTranslation()
  const [text, setText] = useState(value === undefined ? '' : formatNumber(value))
  const [bad, setBad] = useState(false)
  useEffect(() => { setText(value === undefined ? '' : formatNumber(value)); setBad(false) }, [value])
  const listId = `gr-dl-${param.id}`
  const commit = () => {
    if (!text.trim()) { setBad(false); if (value !== undefined) onChange(undefined); return }
    const n = parseNumber(text)
    if (n === null) { setBad(true); return }
    setBad(false)
    if (value === undefined || n !== value) onChange(n)
    setText(formatNumber(n))
  }
  const sci = value !== undefined ? scientific(value) : null
  return (
    <span className="gr-num-inp">
      <input className={`ds-inp ds-mono${bad ? ' gr-bad' : ''}`} value={text} list={listId} inputMode="decimal"
        placeholder={t('graph.unset')} onChange={(e) => setText(e.target.value)} onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }} aria-label={param.name}
        aria-invalid={bad} title={bad ? t('graph.badNumber') : undefined} />
      <datalist id={listId}>
        {knownValues(param, runs).map((v) => <option key={String(v)} value={formatNumber(v as number)} />)}
      </datalist>
      {sci && <span className="gr-sci">{sci}</span>}
    </span>
  )
}

function NameInput({ value, onSave, placeholder }: { value: string; onSave: (v: string) => void; placeholder: string }) {
  const [v, setV] = useState(value)
  useEffect(() => setV(value), [value])
  return (
    <input className="gr-name-inp" value={v} placeholder={placeholder} onChange={(e) => setV(e.target.value)}
      onBlur={() => { if (v !== value) onSave(v) }}
      onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }} aria-label={placeholder} />
  )
}

export function LineInput({ value, onSave, placeholder, mono }: { value: string; onSave: (v: string) => void; placeholder?: string; mono?: boolean }) {
  const [v, setV] = useState(value)
  useEffect(() => setV(value), [value])
  return (
    <input className={`ds-inp${mono ? ' ds-mono' : ''}`} value={v} placeholder={placeholder} onChange={(e) => setV(e.target.value)}
      onBlur={() => { if (v !== value) onSave(v) }}
      onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }} aria-label={placeholder} />
  )
}

function TextArea({ value, onSave, placeholder }: { value: string; onSave: (v: string) => void; placeholder: string }) {
  const [v, setV] = useState(value)
  useEffect(() => setV(value), [value])
  return (
    <textarea className="ds-inp gr-notes" value={v} placeholder={placeholder} rows={4}
      onChange={(e) => setV(e.target.value)} onBlur={() => { if (v !== value) onSave(v) }} aria-label={placeholder} />
  )
}

