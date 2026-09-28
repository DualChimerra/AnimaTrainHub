import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useDialog } from '../../components/Dialog'
import KebabMenu from '../../components/ds/KebabMenu'
import PageHead from '../../components/ds/PageHead'
import Popover, { MenuItems } from '../../components/ds/Popover'
import { useToast } from '../../components/Toast'
import { graphApi } from './graph/api'
import { GroupedCombos as CardsView, groupFilesByStep, type CardActions } from './graph/CardsView'
import CompareView from './graph/CompareView'
import CreateTrainingDialog from './graph/CreateTrainingDialog'
import FilterPanel from './graph/FilterPanel'
import ImageDialog from './graph/ImageDialog'
import { ImportTaskDialog, SamplePickerDialog } from './graph/ImportDialogs'
import MatrixView from './graph/MatrixView'
import {
  activeValues, buildGaps, combosOfIdentical, comparisonParams, configPatch, filterRuns, GAP_LIMIT, groupItems,
  newId, normalizeView, oneStepAway, paramById, sortRuns, statusFromTask, taskMatches,
  taskValuesMap, visibleParams, type Combo,
} from './graph/model'
import ParamsManager, { type Remap } from './graph/ParamsManager'
import RunDrawer from './graph/RunDrawer'
import TableView from './graph/TableView'
import type { Board, GImage, GraphTask, Param, Run, SortKey, Values, ViewMode, ViewState } from './graph/types'
import { Ico, Modal, useValueLabel } from './graph/ui'
import { useGraph } from './graph/useGraph'
import WebView from './graph/WebView'

const MODES: ViewMode[] = ['cards', 'table', 'matrix', 'web']

export default function GraphPage() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const dialog = useDialog()
  const label = useValueLabel()
  const g = useGraph()
  const { board, runs, view, setView, tasks } = g
  // Stable callbacks (the hook's result object itself is new on every render).
  const { createRun, updateRun, duplicateRun, deleteRun, uploadImages } = g
  const params = useMemo(() => board?.params ?? [], [board])
  const byId = useMemo(() => paramById(params), [params])

  const [openRunId, setOpenRunId] = useState<number | null>(null)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [neighboursOf, setNeighboursOf] = useState<number | null>(null)
  const [compare, setCompare] = useState<number[] | null>(null)
  const [image, setImage] = useState<{ runId: number; imageId: number } | null>(null)
  const [createFor, setCreateFor] = useState<{ values: Values; runId?: number } | null>(null)
  const [paramsOpen, setParamsOpen] = useState(false)
  const [importOpen, setImportOpen] = useState(false)
  const [samplePick, setSamplePick] = useState<{ runId: number; taskId: number } | null>(null)
  const [newBoardOpen, setNewBoardOpen] = useState(false)
  const [filtersOpen, setFiltersOpen] = useState(() => window.innerWidth > 900)
  const fileRef = useRef<HTMLInputElement>(null)
  const fileTarget = useRef<number | null>(null)

  const openRun = runs.find((r) => r.id === openRunId) ?? null
  // Close panels that point at cards that no longer exist.
  useEffect(() => { if (openRunId != null && !openRun) setOpenRunId(null) }, [openRunId, openRun])
  useEffect(() => { setSelected(new Set()); setNeighboursOf(null); setOpenRunId(null) }, [board?.id])

  // ── queue tasks → values, and which ones are not on the board yet ─────────
  const taskValues = useMemo(() => taskValuesMap(tasks, params), [tasks, params])
  const taskById = useMemo(() => new Map(tasks.map((x) => [x.id, x])), [tasks])
  const freeTasks = useMemo(() => {
    const used = new Set(runs.map((r) => r.task_id).filter((x): x is number => x != null))
    return tasks.filter((x) => !used.has(x.id) && x.has_config)
  }, [tasks, runs])
  const tasksFor = useCallback((values: Values): GraphTask[] =>
    freeTasks.filter((x) => taskMatches(values, taskValues.get(x.id) ?? {}, params)), [freeTasks, taskValues, params])

  // ── what's on screen ──────────────────────────────────────────────────────
  const neighbourInfo = useMemo(() => {
    const base = runs.find((r) => r.id === neighboursOf)
    if (!base) return null
    const list = oneStepAway(base, runs, params)
    return { base, list, highlight: new Map(list.map((x) => [x.run.id, x.param])) }
  }, [neighboursOf, runs, params])

  const filtered = useMemo(() => {
    if (!view) return []
    if (neighbourInfo) return [neighbourInfo.base, ...neighbourInfo.list.map((x) => x.run)]
    return filterRuns(runs, params, view)
  }, [runs, params, view, neighbourInfo])

  const sorted = useMemo(() => (view ? sortRuns(filtered, view.sort, params) : []), [filtered, view, params])
  // Empty places need at least one parameter narrowed in the filters; without
  // one there is nothing to span and the board shows cards as usual.
  const gapAxes = view ? comparisonParams(params, view).length : 0
  const gaps = useMemo(() => (view?.gaps && !neighbourInfo && gapAxes > 0 ? buildGaps(sorted, params, view) : undefined),
    [view, sorted, params, neighbourInfo, gapAxes])
  const combos: Combo[] = useMemo(() => {
    if (gaps) return gaps
    if (neighbourInfo) return sorted.map((r) => ({ key: `r${r.id}`, values: activeValues(r.values, params), runs: [r] }))
    return combosOfIdentical(sorted, params)
  }, [gaps, sorted, params, neighbourInfo])
  const groups = useMemo(() => (view && !neighbourInfo ? groupItems(combos, view.groupBy, params, (c) => c.values) : null),
    [combos, view, params, neighbourInfo])

  /** Single-value filters = what every place on screen shares. */
  const fixed = useMemo(() => {
    const out: Values = {}
    if (!view) return out
    for (const p of params) {
      const vals = view.filters[p.id]?.values
      if (!p.archived && vals?.length === 1) out[p.id] = vals[0]
    }
    return out
  }, [view, params])

  // ── card actions ──────────────────────────────────────────────────────────
  const autoName = useCallback((values: Values) => {
    const act = activeValues(values, params)
    const ids = (view?.shownParams ?? []).filter((id) => act[id] !== undefined)
    return ids.slice(0, 4).map((id) => label(byId.get(id), act[id])).join(', ')
  }, [params, view, label, byId])

  // Read through a ref so card actions don't change identity on every filter tweak.
  const autoNameRef = useRef(autoName)
  autoNameRef.current = autoName

  const uploadTo = useCallback(async (run: Run, files: File[], step?: number | null) => {
    const groupsByStep = step != null ? [{ step, files }] : groupFilesByStep(files)
    let n = 0
    for (const gr of groupsByStep) {
      const added = await uploadImages(run.id, gr.files, gr.step != null ? { step: gr.step } : undefined)
      n += added.length
    }
    if (n) toast(t('graph.uploadedN', { count: n }), 'success')
  }, [uploadImages, toast, t])

  const downloadConfig = useCallback(async (run: Run) => {
    const save = (blob: Blob, name: string) => {
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = name
      document.body.appendChild(a)
      a.click()
      a.remove()
      setTimeout(() => URL.revokeObjectURL(a.href), 2000)
    }
    const base = (run.name || `run_${run.id}`).replace(/[^\w.-]+/g, '_')
    try {
      const resp = await fetch(graphApi.runConfigUrl(run.id))
      if (resp.ok) { save(await resp.blob(), `${base}.yaml`); return }
    } catch { /* fall through to the fragment */ }
    const { patch } = configPatch(run.values, params)
    if (!Object.keys(patch).length) { toast(t('graph.noConfigToDownload'), 'error'); return }
    const lines = [`# ${t('graph.fragmentHeader')}`, ...Object.entries(patch).map(([k, v]) => `${k}: ${JSON.stringify(v)}`)]
    save(new Blob([lines.join('\n') + '\n'], { type: 'application/x-yaml' }), `${base}.partial.yaml`)
    toast(t('graph.fragmentSaved'), 'info')
  }, [params, toast, t])

  const adoptTask = useCallback(async (task: GraphTask, values: Values = {}) => {
    const vals = { ...values, ...(taskValues.get(task.id) ?? {}) }
    const r = await createRun({
      name: `${[task.project_title, task.version_label].filter(Boolean).join(' / ') || task.name} #${task.id}`,
      values: vals, task_id: task.id, project_id: task.project_id, version_id: task.version_id,
      status: statusFromTask(task.status),
    })
    if (r) {
      setOpenRunId(r.id)
      setSamplePick({ runId: r.id, taskId: task.id })
    }
  }, [createRun, taskValues])

  const actions: CardActions = useMemo(() => ({
    open: (r) => setOpenRunId(r.id),
    openImage: (r, img) => setImage({ runId: r.id, imageId: img.id }),
    toggleSelect: (r) => setSelected((s) => { const n = new Set(s); if (n.has(r.id)) n.delete(r.id); else n.add(r.id); return n }),
    rate: (r, n) => void updateRun(r.id, { rating: n }),
    favorite: (r) => void updateRun(r.id, { favorite: !r.favorite }),
    drop: (r, files) => void uploadTo(r, files),
    pickFiles: (r) => { fileTarget.current = r.id; fileRef.current?.click() },
    duplicate: async (r) => {
      const c = await duplicateRun(r.id)
      if (c) { setOpenRunId(c.id); toast(t('graph.duplicated'), 'success') }
    },
    neighbours: (r) => { setNeighboursOf(r.id); setOpenRunId(null) },
    createTraining: (values, r) => setCreateFor({ values, runId: r?.id }),
    plan: async (values) => {
      const r = await createRun({ values, status: 'planned', name: autoNameRef.current(values) })
      if (r) toast(t('graph.planned'), 'success')
    },
    downloadConfig: (r) => void downloadConfig(r),
    remove: async (r) => {
      const ok = await dialog.confirm(t('graph.deleteCardConfirm', { name: r.name || `#${r.id}`, count: r.images.length }), { tone: 'danger', okText: t('common.delete') })
      if (ok && await deleteRun(r.id)) setSelected((s) => { const n = new Set(s); n.delete(r.id); return n })
    },
    adoptTask: (task, values) => void adoptTask(task, values),
    compare: (list) => setCompare(list.map((r) => r.id)),
  }), [updateRun, duplicateRun, createRun, deleteRun, uploadTo, toast, t, downloadConfig, dialog, adoptTask])

  // Paste an image from the clipboard into the open card (or the one selected card).
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return
      const files = Array.from(e.clipboardData?.files ?? []).filter((f) => f.type.startsWith('image/'))
      if (!files.length) return
      const targetId = openRunId ?? (selected.size === 1 ? [...selected][0] : null)
      const run = runs.find((r) => r.id === targetId)
      e.preventDefault()
      if (!run) { toast(t('graph.pasteWhere'), 'info'); return }
      const named = files.map((f, i) => new File([f], f.name && f.name !== 'image.png' ? f.name : `pasted-${Date.now()}-${i}.png`, { type: f.type }))
      void uploadTo(run, named)
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [openRunId, selected, runs, uploadTo, toast, t])

  // ── params save (remaps first, so card values follow the edit) ────────────
  const saveParams = async (next: typeof params, remaps: Remap[]) => {
    if (!board) return false
    try {
      const byParam = new Map<string, { from: unknown; to: unknown }[]>()
      for (const r of remaps) byParam.set(r.paramId, [...(byParam.get(r.paramId) ?? []), { from: r.from, to: r.to }])
      for (const [pid, mapping] of byParam) await graphApi.remap(board.id, pid, mapping)
    } catch (e) {
      toast((e as Error).message, 'error')
      return false
    }
    const ok = await g.saveParams(next)
    if (ok) {
      if (remaps.length) await g.loadBoard(board.id)
      toast(t('graph.paramsSaved'), 'success')
    }
    return ok
  }

  // ── boards ────────────────────────────────────────────────────────────────
  const createBoard = async (name: string, mode: 'default' | 'empty' | number) => {
    try {
      const r = await graphApi.createBoard({
        name, empty: mode === 'empty', copy_params_from: typeof mode === 'number' ? mode : undefined,
      })
      await g.refreshBoards()
      g.setBoardId(r.board.id)
      setNewBoardOpen(false)
    } catch (e) {
      toast((e as Error).message, 'error')
    }
  }

  if (g.boards === null || (g.loading && !board)) {
    return <div className="gr-page"><PageHead title={t('graph.title')} /><div className="ds-scroll"><div className="ds-skel" style={{ height: 240, borderRadius: 16 }} /></div></div>
  }

  if (!board || !view) {
    return (
      <div className="gr-page">
        <PageHead title={t('graph.title')} subtitle={t('graph.subtitle')} />
        <div className="ds-scroll">
          <div className="gr-welcome ds-card">
            <div className="gr-welcome-ico">{Ico.matrix}</div>
            <div className="ds-card-title">{t('graph.welcomeTitle')}</div>
            <p className="gr-welcome-p">{t('graph.welcomeBody')}</p>
            <ol className="gr-welcome-steps">
              <li>{t('graph.welcome1')}</li>
              <li>{t('graph.welcome2')}</li>
              <li>{t('graph.welcome3')}</li>
            </ol>
            <button type="button" className="ds-btn-primary" onClick={() => setNewBoardOpen(true)}>{Ico.plus}{t('graph.newBoard')}</button>
          </div>
        </div>
        {newBoardOpen && <NewBoardDialog boards={g.boards ?? []} onClose={() => setNewBoardOpen(false)} onCreate={createBoard} />}
      </div>
    )
  }

  const setV = (fn: (v: ViewState) => ViewState) => setView(fn)
  const total = runs.length
  const withSamples = runs.filter((r) => r.images.length).length
  const plannedN = runs.filter((r) => r.status === 'planned').length
  const selectedRuns = runs.filter((r) => selected.has(r.id))
  const gapTotal = gaps?.length ?? 0
  const gapFilled = gaps?.filter((c) => c.runs.length).length ?? 0
  const needGapHint = view.gaps && !neighbourInfo && gapAxes === 0
  const compareRuns = compare ? compare.map((id) => runs.find((r) => r.id === id)).filter((r): r is Run => !!r) : []
  const imageRun = image ? runs.find((r) => r.id === image.runId) : undefined

  return (
    <div className="gr-page">
      <input ref={fileRef} type="file" accept="image/png,image/jpeg,image/webp" multiple hidden
        onChange={(e) => {
          const r = runs.find((x) => x.id === fileTarget.current)
          const files = Array.from(e.target.files ?? [])
          e.target.value = ''
          if (r && files.length) void uploadTo(r, files)
        }} />
      <PageHead
        title={t('graph.title')}
        subtitle={t('graph.stats', { total, withSamples, planned: plannedN })}
        tools={<>
          <select className="ds-inp gr-board-sel" value={board.id} aria-label={t('graph.board')}
            onChange={(e) => g.setBoardId(Number(e.target.value))}>
            {(g.boards ?? []).map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}
          </select>
          <KebabMenu label={t('graph.boards')} trigger="icon" items={[
            { label: t('graph.parameters'), icon: Ico.sliders, onSelect: () => setParamsOpen(true) },
            { label: t('graph.newBoard'), onSelect: () => setNewBoardOpen(true), divider: true },
            { label: t('graph.renameBoard'), onSelect: async () => {
              const name = await dialog.prompt(t('graph.renameBoard'), { defaultValue: board.name, validate: (v) => (v.trim() ? null : t('graph.nameRequired')) })
              if (name) await g.saveBoardMeta({ name: name.trim() })
            } },
            { label: t('graph.deleteBoard'), tone: 'err', divider: true, onSelect: async () => {
              const ok = await dialog.confirm(t('graph.deleteBoardConfirm', { name: board.name, runs: runs.length, images: runs.reduce((n, r) => n + r.images.length, 0) }), { tone: 'danger', okText: t('common.delete') })
              if (!ok) return
              await graphApi.deleteBoard(board.id)
              const list = await g.refreshBoards()
              g.setBoardId(list[0]?.id ?? null)
            } },
          ]} />
          <AddMenu
            onBlank={async () => {
              const r = await g.createRun({ values: fixed, status: 'planned', name: '' })
              if (r) setOpenRunId(r.id)
            }}
            onFromQueue={() => { void g.refreshTasks(); setImportOpen(true) }} />
        </>}
      />

      <div className="gr-toolbar-wrap">
        <div className="ds-toolbar gr-toolbar">
          <button type="button" className={`ds-iconbtn${filtersOpen ? ' gr-on' : ''}`} onClick={() => setFiltersOpen((v) => !v)}
            title={t('graph.toggleFilters')} aria-label={t('graph.toggleFilters')} aria-pressed={filtersOpen}>{Ico.filter}</button>
          <span className="ds-search" style={{ flex: 1, minWidth: 0 }}>
            {Ico.search}
            <input className="ds-inp" value={view.search} onChange={(e) => setV((v) => ({ ...v, search: e.target.value }))}
              placeholder={t('graph.searchPh')} aria-label={t('graph.searchPh')} />
          </span>
          {(view.mode === 'cards' || view.mode === 'table') && (
            <select className="ds-inp gr-tsel" aria-label={t('graph.sort')} value={sortValue(view.sort[0])}
              onChange={(e) => { const k = parseSort(e.target.value); setV((v) => ({ ...v, sort: [k, ...v.sort.slice(1, 2).filter((x) => x.key !== k.key)] })) }}>
              {sortOptions(params, t).map(([val, text]) => <option key={val} value={val}>{text}</option>)}
            </select>
          )}
          <div className="ds-seg" style={{ flex: 'none' }} role="tablist" aria-label={t('graph.viewMode')}>
            {MODES.map((m) => (
              <button key={m} type="button" role="tab" aria-selected={view.mode === m}
                className={`ds-seg-item${view.mode === m ? ' ds-is-active' : ''}`}
                onClick={() => setV((v) => ({ ...v, mode: m }))}>
                {t(`graph.view_${m}`)}
              </button>
            ))}
          </div>
          <ViewSettings board={board} view={view} params={params} setView={setV}
            onApply={(st) => setView(normalizeView(st, params))}
            onSave={async (name) => g.saveBoardMeta({ views: [...board.views, { id: newId('v'), name, state: view }] })}
            onDelete={async (id) => g.saveBoardMeta({ views: board.views.filter((x) => x.id !== id) })} />
        </div>
      </div>

      <div className="gr-body">
        {filtersOpen && (
          <aside className="gr-side" aria-label={t('graph.filters')}>
            <FilterPanel params={params} runs={runs} view={view} setView={setV} />
          </aside>
        )}
        <main className="gr-main">
          {neighbourInfo && (
            <div className="ds-note ds-info gr-banner">
              <span style={{ flex: 1 }}>{t('graph.neighboursBanner', { name: neighbourInfo.base.name || `#${neighbourInfo.base.id}`, count: neighbourInfo.list.length })}</span>
              {neighbourInfo.list.length > 0 && <button type="button" className="ds-ctl" onClick={() => setCompare(filtered.map((r) => r.id))}>{Ico.compare}{t('graph.compareAll')}</button>}
              <button type="button" className="ds-ctl ds-ghost" onClick={() => setNeighboursOf(null)}>{t('graph.backToBoard')}</button>
            </div>
          )}
          {view.gaps && !neighbourInfo && (view.mode === 'cards' || view.mode === 'table') && (
            gaps === null ? (
              <div className="ds-note ds-warn gr-banner"><span>{t('graph.tooManyGaps', { n: GAP_LIMIT })}</span></div>
            ) : needGapHint ? (
              <div className="ds-note ds-info gr-banner"><span>{t('graph.gapsHint')}</span></div>
            ) : gaps && (
              <div className="gr-gapbar">
                <span className="ds-cell-key">{t('graph.coverage', { filled: gapFilled, total: gapTotal })}</span>
                <div className="gr-progress"><i style={{ width: `${gapTotal ? (gapFilled / gapTotal) * 100 : 0}%` }} /></div>
              </div>
            )
          )}

          {total === 0 && !view.gaps && view.mode !== 'matrix' ? (
            <div className="gr-welcome ds-card">
              <div className="ds-card-title">{t('graph.emptyBoardTitle')}</div>
              <p className="gr-welcome-p">{t('graph.emptyBoardBody')}</p>
              <div className="gr-drawer-row" style={{ justifyContent: 'center' }}>
                <button type="button" className="ds-btn-primary" onClick={async () => { const r = await g.createRun({ values: {}, name: '' }); if (r) setOpenRunId(r.id) }}>{Ico.plus}{t('graph.addCard')}</button>
                <button type="button" className="ds-ctl" onClick={() => setImportOpen(true)}>{Ico.queue}{t('graph.fromQueue')}</button>
                <button type="button" className="ds-ctl" onClick={() => setV((v) => ({ ...v, mode: 'matrix' }))}>{Ico.matrix}{t('graph.planInMatrix')}</button>
              </div>
            </div>
          ) : filtered.length === 0 && !(view.gaps && gaps?.length) && view.mode !== 'matrix' ? (
            <div className="ds-empty">{t('graph.nothingMatches')}
              <button type="button" className="gr-link" onClick={() => setV((v) => ({ ...v, filters: {}, search: '', statuses: [], samples: 'any', favorites: false, minRating: 0 }))}>{t('graph.resetFilters')}</button>
            </div>
          ) : view.mode === 'cards' ? (
            <CardsView groups={groups} combos={combos} collapsed={view.collapsed}
              onToggle={(key) => setV((v) => ({ ...v, collapsed: v.collapsed.includes(key) ? v.collapsed.filter((k) => k !== key) : [...v.collapsed, key] }))}
              params={params} shown={view.shownParams} thumb={view.thumb} selected={selected} actions={actions}
              tasksFor={tasksFor} taskById={taskById} highlight={neighbourInfo?.highlight} />
          ) : view.mode === 'table' ? (
            <TableView combos={combos} params={params} sort={view.sort} selected={selected} actions={actions}
              onSort={(key) => setV((v) => {
                const cur = v.sort[0]
                const dir: SortKey['dir'] = cur?.key === key && cur.dir === 'asc' ? 'desc' : 'asc'
                return { ...v, sort: [{ key, dir }, ...v.sort.filter((s) => s.key !== key)].slice(0, 4) }
              })} />
          ) : view.mode === 'matrix' ? (
            <MatrixView runs={filtered} params={params} view={view} setView={setV} fixed={fixed} actions={actions} tasksFor={tasksFor} thumb={view.thumb} />
          ) : (
            <WebView runs={filtered} params={params} view={view} setView={setV} onOpen={(r) => setOpenRunId(r.id)} />
          )}
        </main>

        {openRun && <div className="gr-drawer-scrim" onClick={() => setOpenRunId(null)} aria-hidden="true" />}
        {openRun && (
          <RunDrawer
            key={openRun.id}
            run={openRun} runs={runs} params={params} tasks={tasks} taskValues={taskValues}
            onClose={() => setOpenRunId(null)}
            onUpdate={(patch) => void g.updateRun(openRun.id, patch)}
            onUpload={(files, step) => void uploadTo(openRun, files, step)}
            onOpenImage={(img) => setImage({ runId: openRun.id, imageId: img.id })}
            onImportSamples={(taskId) => setSamplePick({ runId: openRun.id, taskId })}
            onDuplicate={() => actions.duplicate(openRun)}
            onDelete={() => actions.remove(openRun)}
            onCreateTraining={() => setCreateFor({ values: openRun.values, runId: openRun.id })}
            onDownloadConfig={() => void downloadConfig(openRun)}
            onOpenRun={(r) => setOpenRunId(r.id)}
            onShowNeighbours={() => { setNeighboursOf(openRun.id); setOpenRunId(null) }}
            onLinkTask={(task) => void g.updateRun(openRun.id, task
              ? { task_id: task.id, project_id: task.project_id, version_id: task.version_id, status: statusFromTask(task.status) }
              : { task_id: null })}
          />
        )}
      </div>

      {selected.size > 0 && (
        <div className="gr-selbar ds-card">
          <span>{t('graph.selectedN', { count: selected.size })}</span>
          <button type="button" className="ds-btn-primary" disabled={selected.size < 2} onClick={() => setCompare([...selected])}>{Ico.compare}{t('graph.compare')}</button>
          {selected.size === 1 && <button type="button" className="ds-ctl" onClick={() => { setNeighboursOf([...selected][0]); setSelected(new Set()) }}>{t('graph.oneAway')}</button>}
          <button type="button" className="ds-ctl ds-ghost" onClick={() => setSelected(new Set())}>{t('graph.clearSelection')}</button>
          {selectedRuns.length === 1 && <span className="ds-cell-key">{t('graph.pasteTarget')}</span>}
        </div>
      )}

      {compare && compareRuns.length > 0 && (
        <CompareView runs={compareRuns} params={params} onClose={() => setCompare(null)}
          onOpenRun={(r) => { setCompare(null); setOpenRunId(r.id) }}
          onOpenImage={(r, img) => setImage({ runId: r.id, imageId: img.id })} />
      )}
      {image && imageRun && (
        <ImageDialog run={imageRun} imageId={image.imageId} onClose={() => setImage(null)}
          onUpdate={(img: GImage, patch) => void g.updateImage(img, patch)}
          onDelete={(img) => void g.deleteImage(img)}
          onBestStep={(step) => void g.updateRun(imageRun.id, { best_step: step })} />
      )}
      {createFor && (
        <CreateTrainingDialog values={createFor.values} params={params} onClose={() => setCreateFor(null)}
          onCreated={async ({ projectId, versionId, task, label: vlabel, projectTitle, values: saved, adjusted }) => {
            const link = {
              project_id: projectId, version_id: versionId, task_id: task?.id ?? null,
              status: task ? statusFromTask(task.status) : 'planned' as const,
            }
            if (createFor.runId != null) {
              const cur = runs.find((r) => r.id === createFor.runId)
              // A card that already has a training keeps it; the new one gets its own card.
              if (cur && cur.task_id == null && cur.version_id == null) await g.updateRun(cur.id, { ...link, values: saved })
              else await g.createRun({ ...link, values: saved, name: `${projectTitle} / ${vlabel}` })
            } else {
              await g.createRun({ ...link, values: saved, name: `${projectTitle} / ${vlabel}` })
            }
            void g.refreshTasks()
            toast(task ? t('graph.trainingQueued', { label: vlabel }) : t('graph.versionCreated', { label: vlabel }), 'success')
            if (adjusted.length) toast(t('graph.configAdjusted', { keys: adjusted.join(', ') }), 'error')
          }} />
      )}
      {paramsOpen && <ParamsManager params={params} runs={runs} onClose={() => setParamsOpen(false)} onSave={saveParams} />}
      {importOpen && (
        <ImportTaskDialog tasks={tasks} runs={runs} params={params} taskValues={taskValues} onClose={() => setImportOpen(false)}
          onPick={(task) => { setImportOpen(false); void adoptTask(task) }} />
      )}
      {samplePick && (() => {
        const r = runs.find((x) => x.id === samplePick.runId)
        return r ? (
          <SamplePickerDialog taskId={samplePick.taskId} run={r} onClose={() => setSamplePick(null)}
            onImport={async (names) => {
              const res = await g.importSamples(r.id, samplePick.taskId, names)
              if (res) toast(t('graph.importedN', { count: res.items.length }), 'success')
            }} />
        ) : null
      })()}
      {newBoardOpen && <NewBoardDialog boards={g.boards ?? []} onClose={() => setNewBoardOpen(false)} onCreate={createBoard} />}
    </div>
  )
}

// ── toolbar helpers ─────────────────────────────────────────────────────────

const sortValue = (k: SortKey | undefined) => (k ? `${k.key}:${k.dir}` : '_created:desc')
const parseSort = (s: string): SortKey => {
  const i = s.lastIndexOf(':')
  return { key: s.slice(0, i), dir: s.slice(i + 1) === 'asc' ? 'asc' : 'desc' }
}

/** A short list: the useful built-ins plus number parameters (LR, steps…). */
function sortOptions(params: Param[], t: (k: string, o?: Record<string, unknown>) => string): [string, string][] {
  return [
    ['_created:desc', t('graph.sortNewest')],
    ['_created:asc', t('graph.sortOldest')],
    ['_rating:desc', t('graph.sortBest')],
    ['_name:asc', t('graph.sortName')],
    ...visibleParams(params).filter((p) => p.type === 'number').map((p) => [`${p.id}:asc`, t('graph.sortByParam', { name: p.name })] as [string, string]),
  ]
}

const CheckMark = (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.4" strokeLinecap="round" strokeLinejoin="round"><path d="m5 12.5 4.5 4.5L19 7" /></svg>
)

/** The settings button: the less frequent view settings in one place. */
function ViewSettings({ board, view, params, setView, onApply, onSave, onDelete }: {
  board: Board
  view: ViewState
  params: Param[]
  setView: (fn: (v: ViewState) => ViewState) => void
  onApply: (s: ViewState) => void
  onSave: (name: string) => Promise<void>
  onDelete: (id: string) => Promise<void>
}) {
  const { t } = useTranslation()
  const dialog = useDialog()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLButtonElement | null>(null)
  const vis = visibleParams(params)
  return (
    <>
      <button ref={ref} type="button" className={`ds-iconbtn${open ? ' ds-is-open' : ''}`} onClick={() => setOpen((v) => !v)}
        title={t('graph.viewSettings')} aria-label={t('graph.viewSettings')} aria-expanded={open}>{Ico.sliders}</button>
      {open && ref.current && (
        <Popover anchor={ref.current} align="end" minWidth={300} maxHeight={560} role="dialog" ariaLabel={t('graph.viewSettings')} onClose={() => setOpen(false)}>
          <div className="gr-vset">
            {view.mode === 'cards' && (
              <>
                <label className="gr-set-row">
                  <span className="gr-set-label">{t('graph.thumbSize')}</span>
                  <input type="range" className="ds-range" min={140} max={340} step={10} value={view.thumb}
                    style={{ '--p': `${((view.thumb - 140) / 200) * 100}%` } as React.CSSProperties}
                    onChange={(e) => setView((v) => ({ ...v, thumb: Number(e.target.value) }))} />
                </label>
                <label className="gr-set-row">
                  <span className="gr-set-label">{t('graph.groupBy')}</span>
                  <select className="ds-inp" value={view.groupBy[0] ?? ''}
                    onChange={(e) => { const id = e.target.value; setView((v) => ({ ...v, groupBy: id ? [id, ...v.groupBy.slice(1, 2).filter((x) => x !== id)] : [], collapsed: [] })) }}>
                    <option value="">{t('graph.none')}</option>
                    {vis.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                  </select>
                </label>
                <label className="gr-set-row">
                  <span className="gr-set-label">{t('graph.thenGroup')}</span>
                  <select className="ds-inp" value={view.groupBy[1] ?? ''} disabled={!view.groupBy[0]}
                    onChange={(e) => { const id = e.target.value; setView((v) => ({ ...v, groupBy: id ? [v.groupBy[0], id] : v.groupBy.slice(0, 1), collapsed: [] })) }}>
                    <option value="">{t('graph.none')}</option>
                    {vis.filter((p) => p.id !== view.groupBy[0]).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                  </select>
                </label>
              </>
            )}
            {(view.mode === 'cards' || view.mode === 'table') && (
              <label className="gr-set-row">
                <span className="gr-set-label">{t('graph.thenSort')}</span>
                <select className="ds-inp" value={view.sort[1] ? sortValue(view.sort[1]) : ''}
                  onChange={(e) => { const val = e.target.value; setView((v) => ({ ...v, sort: val ? [v.sort[0], parseSort(val)] : v.sort.slice(0, 1) })) }}>
                  <option value="">{t('graph.none')}</option>
                  {sortOptions(params, t).filter(([val]) => val !== sortValue(view.sort[0])).map(([val, text]) => <option key={val} value={val}>{text}</option>)}
                </select>
              </label>
            )}
            {view.mode === 'cards' && (
              <>
                <div className="gr-set-label">{t('graph.cardFieldsTitle')}</div>
                <div className="gr-set-fields">
                  {vis.map((p) => {
                    const on = view.shownParams.includes(p.id)
                    return (
                      <button key={p.id} type="button" role="checkbox" aria-checked={on} className={`gr-fcheck${on ? ' gr-on' : ''}`}
                        onClick={() => setView((v) => ({ ...v, shownParams: on ? v.shownParams.filter((x) => x !== p.id) : [...v.shownParams, p.id] }))}>
                        <span className={`ds-cbox${on ? ' ds-on' : ''}`}>{CheckMark}</span>
                        <span className="gr-fcheck-l">{p.name}</span>
                      </button>
                    )
                  })}
                </div>
              </>
            )}
            <div className="ds-menu-sep" />
            <div className="gr-set-label">{t('graph.viewsTitle')}</div>
            {board.views.length === 0 && <div className="ds-cell-key">{t('graph.noViews')}</div>}
            {board.views.map((sv) => (
              <div key={sv.id} className="gr-set-view">
                <button type="button" className="gr-set-view-name" onClick={() => { onApply(sv.state); setOpen(false) }}>{sv.name}</button>
                <button type="button" className="ds-kebab" onClick={() => void onDelete(sv.id)} aria-label={t('common.delete')}>{Ico.close}</button>
              </div>
            ))}
            <button type="button" className="ds-ctl" style={{ justifyContent: 'center' }}
              onClick={async () => {
                setOpen(false)
                const name = await dialog.prompt(t('graph.saveViewAs'), { placeholder: t('graph.saveViewPh'), validate: (v) => (v.trim() ? null : t('graph.nameRequired')) })
                if (name) await onSave(name.trim())
              }}>{Ico.plus}{t('graph.saveView')}</button>
          </div>
        </Popover>
      )}
    </>
  )
}

/** One primary button with the two ways to add a training. */
function AddMenu({ onBlank, onFromQueue }: { onBlank: () => void; onFromQueue: () => void }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLButtonElement | null>(null)
  return (
    <>
      <button ref={ref} type="button" className="ds-btn-primary" onClick={() => setOpen((v) => !v)} aria-expanded={open} aria-haspopup="menu">
        {Ico.plus}{t('graph.addCard')}
      </button>
      {open && ref.current && (
        <Popover anchor={ref.current} align="end" minWidth={230} ariaLabel={t('graph.addCard')} onClose={() => setOpen(false)}>
          <MenuItems onClose={() => setOpen(false)} items={[
            { label: <>{t('graph.addBlank')}<span className="ds-menu-sub">{t('graph.addBlankSub')}</span></>, tall: true, icon: Ico.plus, onSelect: onBlank },
            { label: <>{t('graph.fromQueue')}<span className="ds-menu-sub">{t('graph.fromQueueSub')}</span></>, tall: true, icon: Ico.queue, onSelect: onFromQueue },
          ]} />
        </Popover>
      )}
    </>
  )
}

function NewBoardDialog({ boards, onClose, onCreate }: {
  boards: import('./graph/types').BoardSummary[]
  onClose: () => void
  onCreate: (name: string, mode: 'default' | 'empty' | number) => Promise<void>
}) {
  const { t } = useTranslation()
  const [name, setName] = useState('')
  const [mode, setMode] = useState<'default' | 'empty' | number>('default')
  const [busy, setBusy] = useState(false)
  return (
    <Modal title={t('graph.newBoard')} sub={t('graph.newBoardSub')} onClose={onClose} width={480}
      foot={<>
        <button type="button" className="ds-ctl ds-ghost" onClick={onClose}>{t('common.cancel')}</button>
        <button type="button" className="ds-btn-primary" disabled={!name.trim() || busy}
          onClick={async () => { setBusy(true); await onCreate(name.trim(), mode); setBusy(false) }}>{t('common.create')}</button>
      </>}>
      <div className="gr-form">
        <label className="gr-mfield">
          <span className="gr-pfield-name">{t('common.name')}</span>
          <input className="ds-inp" autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder={t('graph.boardNamePh')} />
        </label>
        <div className="gr-mfield">
          <span className="gr-pfield-name">{t('graph.startWith')}</span>
          <div className="gr-radios">
            <label><input type="radio" checked={mode === 'default'} onChange={() => setMode('default')} /> {t('graph.startDefault')}</label>
            <label><input type="radio" checked={mode === 'empty'} onChange={() => setMode('empty')} /> {t('graph.startEmpty')}</label>
            {boards.map((b) => (
              <label key={b.id}><input type="radio" checked={mode === b.id} onChange={() => setMode(b.id)} /> {t('graph.startCopy', { name: b.name })}</label>
            ))}
          </div>
        </div>
      </div>
    </Modal>
  )
}
