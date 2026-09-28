import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { filterIsSet, formatNumber, isActive, knownValues, paramById, parseNumber, sameValue, visibleParams } from './model'
import type { Param, ParamFilter, Run, RunValue, ViewState } from './types'
import { RUN_STATUSES } from './types'
import { Ico, Switch, useValueLabel } from './ui'

type SetView = (fn: (v: ViewState) => ViewState) => void

/**
 * Left column. Every parameter is one collapsed row showing what's picked;
 * opening it lists the values as checkboxes with how many cards use each —
 * a 0 is a value that hasn't been tried yet.
 */
export default function FilterPanel({ params, runs, view, setView }: {
  params: Param[]
  runs: Run[]
  view: ViewState
  setView: SetView
}) {
  const { t } = useTranslation()
  const label = useValueLabel()
  const byId = useMemo(() => paramById(params), [params])
  // Sections with an active filter start open; the rest stay folded.
  const [open, setOpen] = useState<Set<string>>(() => new Set(Object.keys(view.filters)))
  const toggle = (id: string) => setOpen((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n })

  const setFilter = (id: string, fn: (f: ParamFilter) => ParamFilter) => setView((v) => {
    const next = fn(v.filters[id] ?? {})
    const filters = { ...v.filters }
    if (filterIsSet(next)) filters[id] = next
    else delete filters[id]
    return { ...v, filters }
  })

  const anySet = view.statuses.length > 0 || view.samples !== 'any' || view.favorites || view.minRating > 0
    || Object.values(view.filters).some(filterIsSet)

  const statusSummary = view.statuses.map((s) => t(`graph.status_${s}`)).join(', ')
  const moreSummary = [
    view.favorites && t('graph.fFavorites'),
    view.samples === 'with' && t('graph.fWithSamples'),
    view.samples === 'without' && t('graph.fWithoutSamples'),
    view.minRating > 0 && `★ ${view.minRating}+`,
  ].filter(Boolean).join(', ')

  return (
    <div className="gr-filters">
      <div className="gr-f-head">
        <span className="gr-f-title">{t('graph.filters')}</span>
        {anySet && (
          <button type="button" className="gr-link" onClick={() => setView((v) => ({ ...v, filters: {}, statuses: [], samples: 'any', favorites: false, minRating: 0 }))}>
            {t('graph.resetFilters')}
          </button>
        )}
      </div>

      {(view.mode === 'cards' || view.mode === 'table') && (
        <div className="gr-f-gaps">
          <Switch on={view.gaps} onChange={(on) => setView((v) => ({ ...v, gaps: on }))} label={t('graph.showGaps')} />
          <span className="ds-cell-key">{t('graph.showGapsHint')}</span>
        </div>
      )}

      <Section id="_status" title={t('common.status')} summary={statusSummary} open={open.has('_status')} onToggle={toggle}>
        {RUN_STATUSES.map((s) => (
          <Check key={s} on={view.statuses.includes(s)} count={runs.filter((r) => r.status === s).length}
            onClick={() => setView((v) => ({ ...v, statuses: v.statuses.includes(s) ? v.statuses.filter((x) => x !== s) : [...v.statuses, s] }))}>
            {t(`graph.status_${s}`)}
          </Check>
        ))}
      </Section>

      <Section id="_more" title={t('graph.fMore')} summary={moreSummary} open={open.has('_more')} onToggle={toggle}>
        <Check on={view.favorites} count={runs.filter((r) => r.favorite).length} onClick={() => setView((v) => ({ ...v, favorites: !v.favorites }))}>
          {t('graph.fFavorites')}
        </Check>
        <Check on={view.samples === 'with'} count={runs.filter((r) => r.images.length).length}
          onClick={() => setView((v) => ({ ...v, samples: v.samples === 'with' ? 'any' : 'with' }))}>{t('graph.fWithSamples')}</Check>
        <Check on={view.samples === 'without'} count={runs.filter((r) => !r.images.length).length}
          onClick={() => setView((v) => ({ ...v, samples: v.samples === 'without' ? 'any' : 'without' }))}>{t('graph.fWithoutSamples')}</Check>
        {[3, 2, 1].map((n) => (
          <Check key={n} on={view.minRating === n} count={runs.filter((r) => (r.rating ?? 0) >= n).length}
            onClick={() => setView((v) => ({ ...v, minRating: v.minRating === n ? 0 : n }))}>
            <span className="gr-stars-txt">{'★'.repeat(n)}<i>{'★'.repeat(3 - n)}</i></span>{n < 3 ? ` ${t('graph.andHigher')}` : ''}
          </Check>
        ))}
      </Section>

      <div className="gr-f-divider" />

      {visibleParams(params).map((p) => {
        const f = view.filters[p.id]
        const valueOf = (r: Run) => (isActive(p, r.values, byId) ? r.values[p.id] : undefined)
        const count = (v: RunValue | undefined) => runs.filter((r) => (v === undefined ? valueOf(r) === undefined : sameValue(valueOf(r), v))).length
        const parts: string[] = [...(f?.values ?? []).map((v) => label(p, v))]
        if (f?.unset) parts.push(t('graph.unset'))
        if (f?.min != null || f?.max != null) parts.push(`${f?.min != null ? formatNumber(f.min) : '…'}–${f?.max != null ? formatNumber(f.max) : '…'}`)
        if (f?.text?.trim()) parts.push(`“${f.text.trim()}”`)
        const unset = p.type === 'text' ? 0 : count(undefined)
        return (
          <Section key={p.id} id={p.id} title={p.name} summary={parts.join(', ')} hint={p.description}
            gate={p.condition ? byId.get(p.condition.param)?.name : undefined}
            open={open.has(p.id)} onToggle={toggle} onClear={filterIsSet(f) ? () => setFilter(p.id, () => ({})) : undefined}>
            {p.type === 'text' ? (
              <input className="ds-inp gr-f-inp" value={f?.text ?? ''} placeholder={t('graph.fContains')}
                onChange={(e) => setFilter(p.id, (x) => ({ ...x, text: e.target.value }))} />
            ) : (
              <>
                {knownValues(p, runs).map((val) => {
                  const on = !!f?.values?.some((x) => sameValue(x, val))
                  return (
                    <Check key={String(val)} on={on} count={count(val)}
                      onClick={() => setFilter(p.id, (x) => ({ ...x, values: on ? (x.values ?? []).filter((y) => !sameValue(y, val)) : [...(x.values ?? []), val] }))}>
                      {label(p, val)}
                    </Check>
                  )
                })}
                {unset > 0 && (
                  <Check on={!!f?.unset} count={unset} onClick={() => setFilter(p.id, (x) => ({ ...x, unset: !x.unset }))}>
                    <i className="ds-muted">{t('graph.unset')}</i>
                  </Check>
                )}
                {p.type === 'number' && (
                  <RangeRow key={`${f?.min ?? ''}|${f?.max ?? ''}`} f={f} onChange={(min, max) => setFilter(p.id, (x) => ({ ...x, min, max }))} />
                )}
              </>
            )}
          </Section>
        )
      })}
    </div>
  )
}

function Section({ id, title, summary, hint, gate, open, onToggle, onClear, children }: {
  id: string; title: string; summary: string; hint?: string; gate?: string
  open: boolean; onToggle: (id: string) => void; onClear?: () => void; children: React.ReactNode
}) {
  const { t } = useTranslation()
  return (
    <section className={`gr-fsec${open ? ' gr-open' : ''}${summary ? ' gr-set' : ''}`}>
      <button type="button" className="gr-fsec-head" onClick={() => onToggle(id)} aria-expanded={open} title={hint}>
        <span className="gr-fsec-caret">{Ico.chev}</span>
        <span className="gr-fsec-main">
          <span className="gr-fsec-title">
            {title}
            {gate && <span className="gr-fsec-gate">{t('graph.ifShort', { name: gate })}</span>}
          </span>
          {summary && <span className="gr-fsec-sum">{summary}</span>}
        </span>
      </button>
      {summary && onClear && (
        <button type="button" className="gr-fsec-x" onClick={onClear} aria-label={t('graph.clear')} title={t('graph.clear')}>{Ico.close}</button>
      )}
      {open && <div className="gr-fsec-body">{children}</div>}
    </section>
  )
}

function Check({ on, onClick, count, children }: { on: boolean; onClick: () => void; count?: number; children: React.ReactNode }) {
  return (
    <button type="button" role="checkbox" aria-checked={on} className={`gr-fcheck${on ? ' gr-on' : ''}`} onClick={onClick}>
      <span className={`ds-cbox${on ? ' ds-on' : ''}`}>
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.4" strokeLinecap="round" strokeLinejoin="round"><path d="m5 12.5 4.5 4.5L19 7" /></svg>
      </span>
      <span className="gr-fcheck-l">{children}</span>
      {count !== undefined && <span className={`gr-fcheck-n${count === 0 ? ' gr-zero' : ''}`}>{count}</span>}
    </button>
  )
}

function RangeRow({ f, onChange }: { f: ParamFilter | undefined; onChange: (min: number | null, max: number | null) => void }) {
  const { t } = useTranslation()
  const [min, setMin] = useState(f?.min != null ? formatNumber(f.min) : '')
  const [max, setMax] = useState(f?.max != null ? formatNumber(f.max) : '')
  const commit = () => onChange(parseNumber(min), parseNumber(max))
  return (
    <div className="gr-range">
      <input className="ds-inp" placeholder={t('graph.from')} value={min} onChange={(e) => setMin(e.target.value)} onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') commit() }} aria-label={t('graph.from')} />
      <span className="ds-muted">–</span>
      <input className="ds-inp" placeholder={t('graph.to')} value={max} onChange={(e) => setMax(e.target.value)} onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') commit() }} aria-label={t('graph.to')} />
    </div>
  )
}
