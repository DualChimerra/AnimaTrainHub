import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { coverImage, isActive, paramById, visibleParams, type Combo } from './model'
import type { Param, Run, SortKey, Values } from './types'
import { Ico, StatusBadge, Stars, Thumb, useValueLabel } from './ui'
import type { CardActions } from './CardsView'

/** Compact spreadsheet: one row per card, empty places as dashed rows. */
export default function TableView({ combos, params, sort, onSort, selected, actions }: {
  combos: Combo[]
  params: Param[]
  sort: SortKey[]
  onSort: (key: string) => void
  selected: Set<number>
  actions: CardActions
}) {
  const { t } = useTranslation()
  const label = useValueLabel()
  const byId = useMemo(() => paramById(params), [params])
  const cols = visibleParams(params)

  const head = (key: string, text: string, num = false) => {
    const s = sort[0]?.key === key ? sort[0].dir : null
    return (
      <th key={key} className={num ? 'gr-num' : undefined}>
        <button type="button" className="gr-th" onClick={() => onSort(key)}>
          {text}
          {s && <span className="gr-th-dir">{s === 'asc' ? Ico.up : Ico.down}</span>}
        </button>
      </th>
    )
  }

  const cell = (p: Param, values: Values) => {
    if (!isActive(p, values, byId)) return <span className="gr-na">—</span>
    const v = values[p.id]
    if (v === undefined) return <span className="gr-unset">{t('graph.unset')}</span>
    return <span className={p.type === 'number' ? 'gr-mono' : undefined}>{label(p, v)}</span>
  }

  const row = (r: Run) => {
    const cover = coverImage(r)
    const sel = selected.has(r.id)
    return (
      <tr key={r.id} className={sel ? 'gr-row-sel' : undefined} onClick={() => actions.open(r)}>
        <td onClick={(e) => { e.stopPropagation(); actions.toggleSelect(r) }} className="gr-td-check">
          <span className={`ds-cbox${sel ? ' ds-on' : ''}`} role="checkbox" aria-checked={sel} aria-label={t('graph.selectCompare')}>
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.4" strokeLinecap="round"><path d="m5 12.5 4.5 4.5L19 7" /></svg>
          </span>
        </td>
        <td className="gr-td-thumb">
          {cover ? <Thumb id={cover.id} className="gr-tthumb" /> : <span className="gr-tthumb gr-tthumb-empty">{Ico.image}</span>}
        </td>
        <td className="gr-td-name">
          <span className="gr-tname">{r.favorite && <span className="gr-fav-mini">{Ico.heartFill}</span>}{r.name || <span className="ds-muted">{t('graph.untitled')}</span>}</span>
          <span className="ds-cell-key">{t('graph.nSamples', { count: r.images.length })}</span>
        </td>
        <td><StatusBadge status={r.status} /></td>
        <td onClick={(e) => e.stopPropagation()}><Stars value={r.rating} onChange={(n) => actions.rate(r, n)} size={12} /></td>
        {cols.map((p) => <td key={p.id} className={p.type === 'number' ? 'gr-num' : undefined}>{cell(p, r.values)}</td>)}
      </tr>
    )
  }

  const empty = (c: Combo) => (
    <tr key={c.key} className="gr-row-empty">
      <td />
      <td className="gr-td-thumb"><span className="gr-tthumb gr-tthumb-slot">?</span></td>
      <td className="gr-td-name">
        <span className="gr-tname ds-muted">{t('graph.emptyPlace')}</span>
        <span className="gr-row-acts">
          <button type="button" className="gr-link" onClick={() => actions.plan(c.values)}>{t('graph.plan')}</button>
          <button type="button" className="gr-link gr-link-strong" onClick={() => actions.createTraining(c.values)}>{t('graph.createTrainingShort')}</button>
        </span>
      </td>
      <td /><td />
      {cols.map((p) => <td key={p.id} className={p.type === 'number' ? 'gr-num' : undefined}>{c.values[p.id] !== undefined ? cell(p, c.values) : ''}</td>)}
    </tr>
  )

  return (
    <div className="gr-table-wrap ds-card">
      <table className="ds-tbl gr-table">
        <thead>
          <tr>
            <th />
            <th />
            {head('_name', t('common.name'))}
            {head('_status', t('common.status'))}
            {head('_rating', t('graph.rating'))}
            {cols.map((p) => head(p.id, p.name, p.type === 'number'))}
          </tr>
        </thead>
        <tbody>
          {combos.flatMap((c) => (c.runs.length ? c.runs.map(row) : [empty(c)]))}
        </tbody>
      </table>
    </div>
  )
}
