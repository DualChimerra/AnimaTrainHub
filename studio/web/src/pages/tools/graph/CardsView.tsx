import { memo, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import KebabMenu, { type KebabItem } from '../../../components/ds/KebabMenu'
import { coverImage, isActive, paramById, stepFromFilename, varyingParams, type Combo, type Group } from './model'
import type { GImage, GraphTask, Param, Run, Values } from './types'
import { Ico, NumberTitle, StatusBadge, Stars, Thumb, useValueLabel } from './ui'

export interface CardActions {
  open: (run: Run) => void
  openImage: (run: Run, img: GImage) => void
  toggleSelect: (run: Run) => void
  rate: (run: Run, n: number | null) => void
  favorite: (run: Run) => void
  drop: (run: Run, files: File[]) => void
  pickFiles: (run: Run) => void
  duplicate: (run: Run) => void
  neighbours: (run: Run) => void
  createTraining: (values: Values, run?: Run) => void
  plan: (values: Values) => void
  downloadConfig: (run: Run) => void
  remove: (run: Run) => void
  adoptTask: (task: GraphTask, values: Values) => void
  compare: (runs: Run[]) => void
}

export interface CardsProps {
  params: Param[]
  shown: string[]
  thumb: number
  selected: Set<number>
  actions: CardActions
  tasksFor: (values: Values) => GraphTask[]
  taskById: Map<number, GraphTask>
  highlight?: Map<number, string>
}

/** Nested collapsible groups of places. */
export function GroupedCombos({ groups, combos, collapsed, onToggle, depth = 0, ...rest }: CardsProps & {
  groups: Group<Combo>[] | null
  combos: Combo[]
  collapsed: string[]
  onToggle: (key: string) => void
  depth?: number
}) {
  const { t } = useTranslation()
  const label = useValueLabel()
  const byId = useMemo(() => paramById(rest.params), [rest.params])
  if (!groups) return <ComboGrid combos={combos} {...rest} />
  return (
    <div className="gr-groups" data-depth={depth}>
      {groups.map((g) => {
        const p = byId.get(g.paramId)
        const isOpen = !collapsed.includes(g.key)
        const runCount = g.items.reduce((n, c) => n + c.runs.length, 0)
        const empty = g.items.filter((c) => c.runs.length === 0).length
        return (
          <section key={g.key} className="gr-group">
            <button type="button" className="gr-group-head" onClick={() => onToggle(g.key)} aria-expanded={isOpen}>
              <span className={`gr-caret${isOpen ? ' gr-open' : ''}`}>{Ico.chev}</span>
              <span className="gr-group-name">{p?.name}</span>
              <span className="gr-group-val">{label(p, g.value)}<NumberTitle v={g.value} /></span>
              <span className="ds-cell-key">
                {t('graph.nCards', { count: runCount })}
                {empty > 0 && <>, <span className="gr-empty-count">{t('graph.nEmpty', { count: empty })}</span></>}
              </span>
            </button>
            {isOpen && (
              <div className="gr-group-body">
                <GroupedCombos groups={g.children} combos={g.items} collapsed={collapsed} onToggle={onToggle} depth={depth + 1} {...rest} />
              </div>
            )}
          </section>
        )
      })}
    </div>
  )
}

const GAP = 12

/** Column count of the card grid, so stacks of identical runs can span
 *  exactly as many columns as they have cards and stay on the grid. */
function useColumns(thumb: number) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [cols, setCols] = useState(1)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const measure = () => setCols(Math.max(1, Math.floor((el.clientWidth + GAP) / (thumb + GAP))))
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [thumb])
  return { ref, cols }
}

function ComboGrid({ combos, ...rest }: CardsProps & { combos: Combo[] }) {
  const { ref, cols } = useColumns(rest.thumb)
  return (
    <div ref={ref} className="gr-grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {combos.map((c) =>
        c.runs.length === 0 ? <SlotCard key={c.key} combo={c} {...rest} />
          : c.runs.length === 1 ? <RunCard key={c.key} run={c.runs[0]} {...rest} />
            : <ComboStack key={c.key} combo={c} cols={cols} {...rest} />,
      )}
    </div>
  )
}

/** Several runs in one place: side by side on the grid, with what differs
 *  between them highlighted. */
function ComboStack({ combo, cols, ...rest }: CardsProps & { combo: Combo; cols: number }) {
  const { t } = useTranslation()
  const diff = useMemo(() => varyingParams(combo.runs.map((r) => r.values), rest.params), [combo.runs, rest.params])
  const shown = useMemo(() => [...new Set([...diff, ...rest.shown])], [diff, rest.shown])
  const span = Math.min(combo.runs.length, cols)
  return (
    <div className="gr-stack" style={{ gridColumn: `span ${span}` }}>
      <div className="gr-stack-head">
        <span className="ds-badge ds-mute">{t('graph.nRuns', { count: combo.runs.length })}</span>
        {span > 1 && <span className="ds-cell-key gr-stack-hint">{diff.length ? t('graph.stackDiffers') : t('graph.stackSame')}</span>}
        <button type="button" className="gr-link" style={{ marginLeft: 'auto' }} onClick={() => rest.actions.compare(combo.runs)}>
          {t('graph.compare')}
        </button>
      </div>
      <div className="gr-stack-row" style={{ gridTemplateColumns: `repeat(${span}, minmax(0, 1fr))` }}>
        {combo.runs.map((r) => (
          <RunCard key={r.id} run={r} {...rest} shown={shown} diffKeys={diff} />
        ))}
      </div>
    </div>
  )
}

export const RunCard = memo(function RunCard({ run, params, shown, selected, actions, taskById, highlight, diffKeys }: CardsProps & {
  run: Run
  diffKeys?: string[]
}) {
  const { t } = useTranslation()
  const label = useValueLabel()
  const byId = useMemo(() => paramById(params), [params])
  const [over, setOver] = useState(false)
  const navigate = useNavigate()
  const cover = coverImage(run)
  const task = run.task_id != null ? taskById.get(run.task_id) : undefined
  const hl = highlight?.get(run.id)
  const isSel = selected.has(run.id)
  const chips = shown
    .map((id) => byId.get(id))
    .filter((p): p is Param => !!p && !p.archived && isActive(p, run.values, byId))

  const menu: KebabItem[] = [
    { label: t('graph.open'), onSelect: () => actions.open(run) },
    { label: t('graph.addSamples'), onSelect: () => actions.pickFiles(run) },
    { label: t('graph.duplicate'), onSelect: () => actions.duplicate(run) },
    { label: t('graph.oneAway'), onSelect: () => actions.neighbours(run) },
    { label: isSel ? t('graph.unselect') : t('graph.selectCompare'), onSelect: () => actions.toggleSelect(run) },
    { label: t('graph.createTraining'), onSelect: () => actions.createTraining(run.values, run), divider: true },
    { label: t('graph.downloadConfig'), onSelect: () => actions.downloadConfig(run) },
    { label: t('common.delete'), onSelect: () => actions.remove(run), tone: 'err', divider: true },
  ]

  return (
    <article
      className={`gr-card${isSel ? ' gr-sel' : ''}${over ? ' gr-drop' : ''}${hl ? ' gr-hl' : ''}`}
      // Anywhere on the card opens it; buttons and links inside do their own thing.
      onClick={(e) => { if (!(e.target as HTMLElement).closest('button, a, input, select, textarea')) actions.open(run) }}
      onDragOver={(e) => { if (e.dataTransfer.types.includes('Files')) { e.preventDefault(); setOver(true) } }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const files = Array.from(e.dataTransfer.files)
        if (files.length) actions.drop(run, files)
      }}
    >
      <div className="gr-card-img" role="button" tabIndex={0}
        // The picture opens the picture; an empty card opens the card.
        onClick={(e) => { e.stopPropagation(); if (cover) actions.openImage(run, cover); else actions.open(run) }}
        onKeyDown={(e) => { if (e.key === 'Enter') { if (cover) actions.openImage(run, cover); else actions.open(run) } }}
        aria-label={cover ? t('graph.openFull') : t('graph.openCard', { name: run.name || `#${run.id}` })}>
        {cover ? (
          <>
            <Thumb id={cover.id} className="gr-card-thumb" />
            {cover.step != null && <span className="gr-step">{t('graph.stepN', { n: cover.step })}</span>}
            {run.images.length > 1 && <span className="gr-count">{Ico.image}{run.images.length}</span>}
          </>
        ) : (
          <div className="gr-card-noimg">
            {Ico.image}
            <span>{over ? t('graph.dropHere') : t('graph.noSamples')}</span>
          </div>
        )}
        <button
          type="button" className={`gr-check${isSel ? ' gr-on' : ''}`}
          onClick={(e) => { e.stopPropagation(); actions.toggleSelect(run) }}
          aria-pressed={isSel} aria-label={t('graph.selectCompare')} title={t('graph.selectCompare')}
        >
          <span className={`ds-cbox${isSel ? ' ds-on' : ''}`}>
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.4" strokeLinecap="round" strokeLinejoin="round"><path d="m5 12.5 4.5 4.5L19 7" /></svg>
          </span>
        </button>
        <button
          type="button" className={`gr-fav${run.favorite ? ' gr-on' : ''}`}
          onClick={(e) => { e.stopPropagation(); actions.favorite(run) }}
          aria-pressed={run.favorite} aria-label={t('graph.favorite')} title={t('graph.favorite')}
        >{run.favorite ? Ico.heartFill : Ico.heart}</button>
      </div>
      <div className="gr-card-body">
        <div className="gr-card-top">
          <span className="gr-card-name" title={run.name}>
            {run.name || <span className="ds-muted">{t('graph.untitled')}</span>}
          </span>
          <KebabMenu label={t('graph.cardActions')} items={menu} />
        </div>
        <div className="gr-card-meta">
          <StatusBadge status={run.status} />
          <Stars value={run.rating} onChange={(n) => actions.rate(run, n)} size={13} />
        </div>
        {hl && (
          <div className="gr-hl-note">{t('graph.differsIn', { name: byId.get(hl)?.name ?? hl })}</div>
        )}
        {chips.length > 0 && (
          <div className="gr-kv">
            {chips.map((p) => (
              <span key={p.id} className={`gr-kv-row${diffKeys?.includes(p.id) || hl === p.id ? ' gr-diff' : ''}`}>
                <span className="gr-kv-k">{p.name}</span>
                <span className="gr-kv-v">{label(p, run.values[p.id])}</span>
              </span>
            ))}
          </div>
        )}
        {task && (
          <button type="button" className="gr-task-line" title={t('graph.openTraining')} onClick={() => navigate(`/queue/${task.id}`)}>
            {Ico.queue}<span>#{task.id} {t(`graph.task_${task.status}`, { defaultValue: task.status })}</span>
          </button>
        )}
      </div>
    </article>
  )
})

/** An untried combination: what it would be, plus ways to fill it. */
function SlotCard({ combo, params, actions, tasksFor }: CardsProps & { combo: Combo }) {
  const { t } = useTranslation()
  const label = useValueLabel()
  const matching = tasksFor(combo.values)
  const entries = params.filter((p) => combo.values[p.id] !== undefined && !p.archived)
  return (
    <article className="gr-slot">
      <div className="gr-slot-title">{t('graph.emptyPlace')}</div>
      <div className="gr-kv">
        {entries.map((p) => (
          <span key={p.id} className="gr-kv-row">
            <span className="gr-kv-k">{p.name}</span>
            <span className="gr-kv-v">{label(p, combo.values[p.id])}</span>
          </span>
        ))}
      </div>
      {matching.length > 0 && (
        <div className="gr-slot-tasks">
          <span className="ds-cell-key">{t('graph.inQueue')}</span>
          <div className="gr-slot-pills">
            {matching.slice(0, 4).map((task) => (
              <button key={task.id} type="button" className="gr-qpill" onClick={() => actions.adoptTask(task, combo.values)}
                title={`${t('graph.adoptTaskHint')}
#${task.id} ${task.project_title ?? task.name}${task.version_label ? ` / ${task.version_label}` : ''}, ${t(`graph.task_${task.status}`, { defaultValue: task.status })}`}>
                <span className={`gr-qdot gr-q-${task.status}`} />#{task.id}
              </button>
            ))}
            {matching.length > 4 && <span className="ds-cell-key">+{matching.length - 4}</span>}
          </div>
        </div>
      )}
      <div className="gr-slot-actions">
        <button type="button" className="ds-ctl" onClick={() => actions.plan(combo.values)}>{t('graph.plan')}</button>
        <button type="button" className="ds-btn-primary" onClick={() => actions.createTraining(combo.values)}>
          {Ico.play}{t('graph.createTrainingShort')}
        </button>
      </div>
    </article>
  )
}

/** Files a user drops or pastes → step guessed from the name when present. */
export function groupFilesByStep(files: File[]): { step: number | null; files: File[] }[] {
  const map = new Map<number | null, File[]>()
  for (const f of files) {
    const s = stepFromFilename(f.name)
    map.set(s, [...(map.get(s) ?? []), f])
  }
  return [...map.entries()].map(([step, fs]) => ({ step, files: fs }))
}
