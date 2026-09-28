import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { bestRun, coverImage, isActive, knownValues, paramById, sameValue, visibleParams } from './model'
import type { GraphTask, Param, Run, RunValue, Values, ViewState } from './types'
import type { CardActions } from './CardsView'
import { Ico, NumberTitle, Stars, Thumb, useValueLabel } from './ui'

const UNSET = '__unset__'
type AxisValue = RunValue | typeof UNSET

/**
 * Two parameters as axes, the rest as filters: every cell is one combination.
 * The cell shows the best card there (by rating), tinted by that rating — so
 * the direction where results get better is visible at a glance, and blank
 * cells are what hasn't been tried.
 */
export default function MatrixView({ runs, params, view, setView, fixed, actions, tasksFor, thumb }: {
  runs: Run[]
  params: Param[]
  view: ViewState
  setView: (fn: (v: ViewState) => ViewState) => void
  /** Values every cell shares (single-value filters), used when planning. */
  fixed: Values
  actions: CardActions
  tasksFor: (values: Values) => GraphTask[]
  thumb: number
}) {
  const { t } = useTranslation()
  const label = useValueLabel()
  const byId = useMemo(() => paramById(params), [params])
  const choices = visibleParams(params).filter((p) => p.type !== 'text')
  const px = view.matrixX ? byId.get(view.matrixX) : undefined
  const py = view.matrixY ? byId.get(view.matrixY) : undefined

  const axis = (p: Param | undefined): AxisValue[] => {
    if (!p) return []
    const f = view.filters[p.id]?.values
    const vals: AxisValue[] = f?.length ? [...f] : knownValues(p, runs)
    if (!f?.length && runs.some((r) => !isActive(p, r.values, byId) || r.values[p.id] === undefined)) vals.push(UNSET)
    if (p.type === 'number') {
      vals.sort((a, b) => (a === UNSET ? 1 : b === UNSET ? -1 : (a as number) - (b as number)))
    }
    return vals
  }
  const xs = axis(px)
  const ys = py ? axis(py) : [UNSET]

  const fits = (p: Param | undefined, r: Run, v: AxisValue) => {
    if (!p) return true
    const active = isActive(p, r.values, byId)
    if (v === UNSET) return !active || r.values[p.id] === undefined
    return active && sameValue(r.values[p.id], v)
  }

  const cellValues = (x: AxisValue, y: AxisValue): Values => {
    const out: Values = { ...fixed }
    if (px && x !== UNSET) out[px.id] = x
    if (py && y !== UNSET) out[py.id] = y
    return out
  }

  const axisLabel = (p: Param | undefined, v: AxisValue) =>
    v === UNSET ? <i className="ds-muted">{t('graph.unset')}</i> : <>{label(p, v)}<NumberTitle v={v} /></>

  const picker = (value: string | null, onPick: (id: string | null) => void, allowNone: boolean, aria: string) => (
    <select className="ds-inp gr-axis-sel" value={value ?? ''} aria-label={aria}
      onChange={(e) => onPick(e.target.value || null)}>
      {allowNone && <option value="">{t('graph.matrixNone')}</option>}
      {!allowNone && !value && <option value="">{t('graph.matrixPick')}</option>}
      {choices.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
    </select>
  )

  const tried = xs.length * ys.length
  let filled = 0

  const body = ys.map((y) => (
    <tr key={String(y)}>
      {py && <th className="gr-m-yh" scope="row">{axisLabel(py, y)}</th>}
      {xs.map((x) => {
        const here = runs.filter((r) => fits(px, r, x) && fits(py, r, y))
        const best = bestRun(here)
        if (here.length) filled++
        const cover = best ? coverImage(best) : null
        const vals = cellValues(x, y)
        const queued = here.length ? [] : tasksFor(vals)
        const tone = best ? (best.rating ? `gr-m-r${best.rating}` : here.every((r) => r.status === 'planned') ? 'gr-m-plan' : 'gr-m-r0') : 'gr-m-none'
        return (
          <td key={String(x)} className={`gr-m-cell ${tone}`}>
            {best ? (
              <button type="button" className="gr-m-btn" style={{ width: thumb * 0.8 }}
                onClick={() => (here.length === 1 ? actions.open(best) : actions.compare(here))}
                title={here.map((r) => r.name || `#${r.id}`).join('\n')}>
                <span className="gr-m-img" style={{ height: thumb * 0.8 }}>
                  {cover ? <Thumb id={cover.id} /> : <span className="gr-m-noimg">{Ico.image}</span>}
                  {cover?.step != null && <span className="gr-step">{t('graph.stepN', { n: cover.step })}</span>}
                  {here.length > 1 && <span className="gr-count">×{here.length}</span>}
                </span>
                <span className="gr-m-foot">
                  <Stars value={best.rating} size={11} />
                  <span className="gr-m-name">{best.name || t('graph.untitled')}</span>
                </span>
              </button>
            ) : (
              <div className="gr-m-empty" style={{ width: thumb * 0.8, minHeight: thumb * 0.8 }}>
                {queued.length > 0 && (
                  <button type="button" className="gr-task-pill" onClick={() => actions.adoptTask(queued[0], vals)} title={t('graph.adoptTaskHint')}>
                    {Ico.queue}<span className="gr-task-pill-t">#{queued[0].id}</span>
                    <span className="ds-cell-key">{t(`graph.task_${queued[0].status}`, { defaultValue: queued[0].status })}</span>
                  </button>
                )}
                <button type="button" className="gr-m-add" onClick={() => actions.createTraining(vals)} title={t('graph.createTraining')}>
                  {Ico.play}<span>{t('graph.createTrainingShort')}</span>
                </button>
                <button type="button" className="gr-link" onClick={() => actions.plan(vals)}>{t('graph.plan')}</button>
              </div>
            )}
          </td>
        )
      })}
    </tr>
  ))

  return (
    <div className="gr-matrix">
      <div className="gr-matrix-bar ds-card">
        <label className="gr-axis"><span className="ds-cap">{t('graph.axisX')}</span>
          {picker(view.matrixX, (id) => setView((v) => ({ ...v, matrixX: id })), false, t('graph.axisX'))}
        </label>
        <button type="button" className="ds-iconbtn" title={t('graph.swapAxes')} aria-label={t('graph.swapAxes')}
          onClick={() => setView((v) => ({ ...v, matrixX: v.matrixY ?? v.matrixX, matrixY: v.matrixY ? v.matrixX : v.matrixY }))}>
          ⇄
        </button>
        <label className="gr-axis"><span className="ds-cap">{t('graph.axisY')}</span>
          {picker(view.matrixY, (id) => setView((v) => ({ ...v, matrixY: id })), true, t('graph.axisY'))}
        </label>
        <div className="gr-legend">
          <span className="gr-lg gr-m-r3" />{t('graph.rating_3')}
          <span className="gr-lg gr-m-r2" />{t('graph.rating_2')}
          <span className="gr-lg gr-m-r1" />{t('graph.rating_1')}
          <span className="gr-lg gr-m-r0" />{t('graph.notRated')}
          <span className="gr-lg gr-m-none" />{t('graph.untried')}
        </div>
      </div>
      {!px ? (
        <div className="ds-empty">{t('graph.matrixHint')}</div>
      ) : (
        <>
          <div className="gr-m-scroll ds-card">
            <table className="gr-m-table">
              <thead>
                <tr>
                  {py && <th className="gr-m-corner"><span>{py.name} ↓</span><span>{px.name} →</span></th>}
                  {xs.map((x) => <th key={String(x)} className="gr-m-xh">{axisLabel(px, x)}</th>)}
                </tr>
              </thead>
              <tbody>{body}</tbody>
            </table>
          </div>
          <div className="gr-m-progress">
            <div className="gr-progress"><i style={{ width: `${tried ? (filled / tried) * 100 : 0}%` }} /></div>
            <span className="ds-cell-key">{t('graph.coverage', { filled, total: tried })}</span>
          </div>
        </>
      )}
    </div>
  )
}
