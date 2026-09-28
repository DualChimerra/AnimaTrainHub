import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api } from '../../../api/client'
import { graphApi } from './api'
import type { GraphTask, Param, Run, TaskSampleInfo, Values } from './types'
import { Ico, Modal, fmtDate, useValueLabel } from './ui'

/** Pick a queue training to turn into a card. */
export function ImportTaskDialog({ tasks, runs, params, taskValues, onClose, onPick }: {
  tasks: GraphTask[]
  runs: Run[]
  params: Param[]
  taskValues: Map<number, Values>
  onClose: () => void
  onPick: (task: GraphTask) => void
}) {
  const { t } = useTranslation()
  const label = useValueLabel()
  const [q, setQ] = useState('')
  const linked = useMemo(() => new Set(runs.map((r) => r.task_id).filter((x) => x != null)), [runs])
  const list = tasks.filter((task) => {
    const s = `${task.id} ${task.name} ${task.project_title ?? ''} ${task.version_label ?? ''} ${task.note}`.toLowerCase()
    return !q.trim() || s.includes(q.trim().toLowerCase())
  })
  return (
    <Modal title={t('graph.importTitle')} sub={t('graph.importSub')} onClose={onClose} width={720}>
      <div className="ds-search" style={{ marginBottom: 10 }}>
        {Ico.search}
        <input className="ds-inp" value={q} onChange={(e) => setQ(e.target.value)} placeholder={t('graph.searchTasks')} autoFocus />
      </div>
      {list.length === 0 && <div className="ds-empty">{t('graph.noTasks')}</div>}
      <div className="gr-tasklist">
        {list.map((task) => {
          const vals = taskValues.get(task.id) ?? {}
          const known = params.filter((p) => vals[p.id] !== undefined)
          const isLinked = linked.has(task.id)
          return (
            <button key={task.id} type="button" className="gr-taskrow" disabled={isLinked} onClick={() => onPick(task)}>
              <div className="gr-taskrow-top">
                <b>#{task.id}</b>
                <span className="gr-taskrow-name">{task.project_title ?? task.name}{task.version_label ? ` / ${task.version_label}` : ''}</span>
                <span className="ds-badge ds-mute">{t(`graph.task_${task.status}`, { defaultValue: task.status })}</span>
                <span className="ds-cell-key">{fmtDate(task.created_at)}</span>
              </div>
              <div className="gr-kv gr-kv-inline">
                {known.map((p) => (
                  <span key={p.id} className="gr-kv-row"><span className="gr-kv-k">{p.name}</span><span className="gr-kv-v">{label(p, vals[p.id])}</span></span>
                ))}
                {!known.length && <span className="ds-cell-key">{task.has_config ? t('graph.nothingRecognised') : t('graph.noTaskConfig')}</span>}
              </div>
              {isLinked && <span className="ds-cell-key">{t('graph.alreadyOnBoard')}</span>}
            </button>
          )
        })}
      </div>
    </Modal>
  )
}

/** Choose which of a task's samples to copy into a card, by step. */
export function SamplePickerDialog({ taskId, run, onClose, onImport }: {
  taskId: number
  run: Run
  onClose: () => void
  onImport: (names: string[]) => Promise<void>
}) {
  const { t } = useTranslation()
  const [items, setItems] = useState<TaskSampleInfo[] | null>(null)
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const have = useMemo(() => new Set(run.images.map((i) => i.source)), [run.images])

  useEffect(() => {
    graphApi.taskSamples(taskId).then((r) => setItems(r.items)).catch(() => setItems([]))
  }, [taskId])

  const groups = useMemo(() => {
    const m = new Map<string, TaskSampleInfo[]>()
    for (const it of items ?? []) {
      const k = it.step != null ? `s${it.step}` : it.epoch != null ? `e${it.epoch}` : 'x'
      m.set(k, [...(m.get(k) ?? []), it])
    }
    return [...m.entries()].sort(([, a], [, b]) => (a[0].step ?? 1e12) - (b[0].step ?? 1e12))
  }, [items])

  const taken = (n: string) => have.has(`task:${taskId}/${n}`)
  const toggle = (names: string[], on: boolean) => setSel((s) => {
    const next = new Set(s)
    for (const n of names) { if (taken(n)) continue; if (on) next.add(n); else next.delete(n) }
    return next
  })
  const free = (items ?? []).filter((i) => !taken(i.filename)).map((i) => i.filename)

  return (
    <Modal title={t('graph.pickSamplesTitle', { id: taskId })} sub={t('graph.pickSamplesSub')} onClose={onClose} width={860}
      foot={<>
        <button type="button" className="gr-link" style={{ marginRight: 'auto' }} disabled={!free.length}
          onClick={() => toggle(free, sel.size < free.length)}>
          {sel.size < free.length ? t('common.selectAll') : t('common.deselect')}
        </button>
        <button type="button" className="ds-ctl ds-ghost" onClick={onClose}>{t('common.cancel')}</button>
        <button type="button" className="ds-btn-primary" disabled={!sel.size || busy}
          onClick={async () => { setBusy(true); await onImport([...sel]); setBusy(false); onClose() }}>
          {busy ? t('graph.importing') : t('graph.importN', { count: sel.size })}
        </button>
      </>}>
      {items === null ? <div className="ds-cell-key">{t('common.loading')}</div>
        : items.length === 0 ? <div className="ds-empty">{t('graph.taskNoSamples')}</div> : (
          <div className="gr-steps gr-steps-wrap">
            {groups.map(([k, list]) => {
              const names = list.map((i) => i.filename)
              const allOn = names.every((n) => sel.has(n) || taken(n))
              const first = list[0]
              return (
                <div key={k} className="gr-step-row">
                  <button type="button" className="gr-step-label gr-step-pick" onClick={() => toggle(names, !allOn)}>
                    <span className={`ds-cbox${allOn ? ' ds-on' : ''}`}><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.4" strokeLinecap="round"><path d="m5 12.5 4.5 4.5L19 7" /></svg></span>
                    {first.step != null ? t('graph.stepN', { n: first.step }) : first.epoch != null ? t('graph.epochN', { n: first.epoch }) : t('graph.noStep')}
                    {first.step != null && first.epoch != null && <span className="ds-cell-key">{t('graph.epochN', { n: first.epoch })}</span>}
                  </button>
                  <div className="gr-step-imgs">
                    {list.map((it) => {
                      const on = sel.has(it.filename)
                      const done = taken(it.filename)
                      return (
                        <button key={it.filename} type="button" disabled={done}
                          className={`gr-sample gr-sample-lg${on ? ' gr-on' : ''}${done ? ' gr-done' : ''}`}
                          onClick={() => toggle([it.filename], !on)} title={[it.filename, it.prompt].filter(Boolean).join('\n')}>
                          <img src={api.sampleImageUrl(it.filename, taskId, 256)} alt="" loading="lazy" decoding="async" />
                          <span className="gr-step">{it.step != null ? t('graph.stepN', { n: it.step }) : it.filename}</span>
                          {done && <span className="gr-sample-done">{t('graph.alreadyAdded')}</span>}
                        </button>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        )}
    </Modal>
  )
}
