/** Edit LoRA examples, laid out like Tag Edit: a grid of examples on the left, the
 *  picked one on the right (its pictures and its text). Everything saves as you go. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate, useOutletContext } from 'react-router-dom'
import {
  api,
  type EditPair,
  type EditPairsListing,
  type EditRecipe,
  type EditSlot,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import StepShell from '../../../components/StepShell'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'
import { EDIT_RECIPES, EditIcon, RECIPE_ORDER, pairCover, pairImageUrl, recipeSlots } from '../../edit/shared'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

const IMAGE_ACCEPT = 'image/png,image/jpeg,image/webp,image/bmp,image/gif'
const FOLDER = '1_data'

const newStem = () => `ex_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 5)}`

/** Rename a file to `<stem><ext>` so it lands in the given example. */
function renamed(file: File, stem: string): File {
  const dot = file.name.lastIndexOf('.')
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : '.png'
  return new File([file], `${stem}${ext}`, { type: file.type })
}

function emptyExample(stem: string, recipe: EditRecipe): EditPair {
  return {
    folder: FOLDER, stem, target: null, source: null, source_folder: null, source2: null,
    source2_folder: null, caption: '', mtime: 0,
    missing: recipe === 'image' ? ['source', 'source2', 'target'] : ['source', 'caption', 'target'],
  }
}

export default function PairsPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const vid = activeVersion?.id ?? null
  const locked = activeVersion?.status === 'training'

  const [data, setData] = useState<EditPairsListing | null>(null)
  const [drafts, setDrafts] = useState<string[]>([])
  const [todoOnly, setTodoOnly] = useState(false)
  const [active, setActive] = useState<string | null>(null)
  const [dialog, setDialog] = useState<null | 'bulk' | 'text'>(null)
  const [bust, setBust] = useState(0)

  const refresh = useCallback(async () => {
    if (vid == null) return
    try {
      setData(await api.listEditPairs(project.id, vid))
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [project.id, vid, toast])

  useEffect(() => { void refresh() }, [refresh])
  useEventStream((evt) => {
    if (evt.type === 'version_state_changed' && evt.version_id === vid) void refresh()
  })

  const afterChange = async () => {
    setBust((n) => n + 1)
    await Promise.all([refresh(), reload()])
  }

  const recipe: EditRecipe = data?.recipe ?? 'text'

  // Server examples, then the empty ones just added (an example exists on the
  // server once its first file or text lands).
  const all = useMemo(() => {
    const server = data?.items ?? []
    const known = new Set(server.map((p) => p.stem))
    return [...server, ...drafts.filter((s) => !known.has(s)).map((s) => emptyExample(s, recipe))]
  }, [data, drafts, recipe])
  const shown = todoOnly ? all.filter((p) => p.missing.length > 0) : all

  // Keep a picked example: the first one until the user picks.
  useEffect(() => {
    if (all.length === 0) { setActive(null); return }
    if (!active || !all.some((p) => p.stem === active)) setActive(all[0].stem)
  }, [all, active])
  const activeIdx = all.findIndex((p) => p.stem === active)
  const activePair = activeIdx >= 0 ? all[activeIdx] : null

  const addExample = () => {
    const stem = newStem()
    setTodoOnly(false)
    setDrafts((d) => [...d, stem])
    setActive(stem)
  }

  const uploadSlot = async (stem: string, role: EditSlot, file: File) => {
    if (vid == null) return
    try {
      const res = await api.uploadEditFiles(project.id, vid, role, [renamed(file, stem)])
      if (res.errors.length) toast(res.errors.join('\n'), 'error')
      await afterChange()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const clearSlot = async (pair: EditPair, role: EditSlot) => {
    if (vid == null) return
    try {
      await api.deleteEditPairs(project.id, vid, { slots: [{ role, folder: pair.folder, stem: pair.stem }] })
      await afterChange()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const removeExample = async (pair: EditPair) => {
    if (vid == null) return
    if (!pair.mtime && !pair.caption) { setDrafts((d) => d.filter((s) => s !== pair.stem)); return }
    if (!(await confirm(t('pairs.deleteConfirm'), { tone: 'danger', okText: t('common.delete') }))) return
    try {
      await api.deleteEditPairs(project.id, vid, { items: [{ folder: pair.folder, stem: pair.stem }] })
      setDrafts((d) => d.filter((s) => s !== pair.stem))
      await afterChange()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const changeRecipe = async (next: EditRecipe) => {
    if (vid == null || next === recipe) return
    try {
      await api.setEditRecipe(project.id, vid, next)
      await afterChange()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // Image prompts left over after switching to "image + text": they block training.
  const strayPrompts = recipe === 'text' ? (data?.items ?? []).filter((p) => p.source2) : []
  const dropStrayPrompts = async () => {
    if (vid == null) return
    try {
      await api.deleteEditPairs(project.id, vid, {
        slots: strayPrompts.map((p) => ({ role: 'source2' as const, folder: p.folder, stem: p.stem })),
      })
      await afterChange()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const goTrain = async () => {
    if (vid == null) return
    try {
      const res = await api.advanceVersionPhase(project.id, vid)
      await reload()
      if (!res.ok) { toast(res.reason || t('sidebar.advanceFailed'), 'error'); return }
      navigate(`/projects/${project.id}/v/${vid}/train`)
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  if (vid == null) {
    return <div className="p-6 text-sm text-fg-tertiary">{t('pairs.noVersion')}</div>
  }

  const total = data?.count ?? 0
  const done = data?.complete ?? 0
  const todo = total - done
  const ready = total > 0 && todo === 0 && strayPrompts.length === 0
  const preparing = activeVersion?.status === 'preparing' && activeVersion.phase === 'curating'

  return (
    <StepShell
      idx={1}
      title={t('pairs.title')}
      subtitle={t(`pairs.subtitle_${recipe}`)}
      actions={
        <>
          <button type="button" className="ds-ctl" disabled={locked} onClick={() => setDialog('bulk')}>
            {EditIcon.up(13)}{t('pairs.bulk')}
          </button>
          {EDIT_RECIPES[recipe].text === 'required' && (
            <button type="button" className="ds-ctl" disabled={locked || !total} onClick={() => setDialog('text')}>
              {EditIcon.text(13)}{t('pairs.instructionAll')}
            </button>
          )}
          {preparing ? (
            <button type="button" className="ds-btn-primary" disabled={!ready || locked} onClick={() => void goTrain()}
              title={ready ? undefined : t('pairs.notReady')}>
              {t('pairs.toTrain')}{EditIcon.arrow(12)}
            </button>
          ) : (
            <Link className="ds-btn-primary" to={`/projects/${project.id}/v/${vid}/train`}>{t('pairs.openTrain')}{EditIcon.arrow(12)}</Link>
          )}
        </>
      }
    >
      {(locked || strayPrompts.length > 0) && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
          {locked && <div className="ds-note ds-warn">{t('pairs.locked')}</div>}
          {strayPrompts.length > 0 && (
            <div className="ds-note ds-warn" style={{ alignItems: 'center' }}>
              <span style={{ flex: 1 }}>{t('pairs.strayPrompts', { count: strayPrompts.length })}</span>
              <button type="button" className="ds-ctl" style={{ height: 28 }} disabled={locked} onClick={() => void dropStrayPrompts()}>
                {t('pairs.strayPromptsDrop')}
              </button>
            </div>
          )}
        </div>
      )}
      <div className="ds-ex-grid">
        <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div className="ds-card-head ds-pad">
            <div>
              <div className="ds-card-title">{t('pairs.listTitle')}</div>
              <div className="ds-card-sub">
                {total ? t('pairs.listSub', { done, total }) : t('pairs.listEmpty')}
              </div>
            </div>
            <div className="ds-card-tools">
              <select className="ds-inp" style={{ width: 190 }} value={recipe} disabled={locked}
                aria-label={t('pairs.typeLabel')} onChange={(e) => void changeRecipe(e.target.value as EditRecipe)}>
                {RECIPE_ORDER.map((r) => <option key={r} value={r}>{t(`editLora.recipe_${r}`)}</option>)}
              </select>
              {todo > 0 && (
                <div className="ds-seg" role="group">
                  <button type="button" aria-pressed={!todoOnly} className={`ds-seg-item${!todoOnly ? ' ds-is-active' : ''}`} onClick={() => setTodoOnly(false)}>
                    {t('pairs.filterAll')}
                  </button>
                  <button type="button" aria-pressed={todoOnly} className={`ds-seg-item${todoOnly ? ' ds-is-active' : ''}`} onClick={() => setTodoOnly(true)}>
                    {t('pairs.filterTodo', { count: todo })}
                  </button>
                </div>
              )}
            </div>
          </div>
          <div className="ds-ex-tiles">
            {shown.map((pair) => {
              const n = all.indexOf(pair) + 1
              const src = pairCover(project.id, vid, pair, 256, bust)
              return (
                <button key={pair.stem} type="button"
                  className={`ds-thumb ds-ex-tile${pair.stem === active ? ' ds-sel' : ''}`}
                  onClick={() => setActive(pair.stem)} aria-label={t('pairs.exampleN', { n })}>
                  {src ? <img src={src} alt="" loading="lazy" /> : <span>{n}</span>}
                  {pair.missing.length > 0 && <span className="ds-ex-tile-todo" title={t('pairs.notFilled')} />}
                </button>
              )
            })}
            {!todoOnly && (
              <button type="button" className="ds-ex-tile ds-ex-tile-add" disabled={locked} onClick={addExample}
                aria-label={t('pairs.addExample')} title={t('pairs.addExample')}>
                {EditIcon.plus(18)}
              </button>
            )}
          </div>
        </div>

        <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          {activePair ? (
            <ExampleEditor
              key={activePair.stem}
              n={activeIdx + 1}
              count={all.length}
              pid={project.id}
              vid={vid}
              pair={activePair}
              recipe={recipe}
              bust={bust}
              locked={locked}
              onPrev={() => setActive(all[(activeIdx - 1 + all.length) % all.length].stem)}
              onNext={() => setActive(all[(activeIdx + 1) % all.length].stem)}
              onUpload={(role, f) => void uploadSlot(activePair.stem, role, f)}
              onClear={(role) => void clearSlot(activePair, role)}
              onDelete={() => void removeExample(activePair)}
              onSaved={() => void refresh()}
            />
          ) : (
            <div className="ds-ex-intro">
              <div className="ds-card-title">{t('pairs.introTitle')}</div>
              <p>{t(`pairs.introText_${recipe}`)}</p>
              <p>{t('pairs.introTips')}</p>
              <button type="button" className="ds-btn-primary" style={{ alignSelf: 'flex-start' }} disabled={locked} onClick={addExample}>
                {EditIcon.plus(13)}{t('pairs.addExample')}
              </button>
            </div>
          )}
        </div>
      </div>

      {dialog === 'bulk' && (
        <BulkDialog
          recipe={recipe}
          onCancel={() => setDialog(null)}
          onUpload={async (role, files) => {
            const res = await api.uploadEditFiles(project.id, vid, role, files)
            if (res.errors.length) toast(res.errors.slice(0, 3).join('\n'), 'error')
            toast(t('pairs.uploaded', { count: res.saved.filter((s) => s.kind !== 'caption').length }), 'success')
            await afterChange()
          }}
        />
      )}
      {dialog === 'text' && (
        <InstructionDialog
          recipe={recipe}
          onCancel={() => setDialog(null)}
          onSubmit={async (instruction, mode) => {
            try {
              const res = await api.setEditInstruction(project.id, vid, instruction, mode)
              toast(t('pairs.instructionDone', { count: res.updated }), 'success')
              setDialog(null)
              await afterChange()
            } catch (e) {
              toast(String(e), 'error')
            }
          }}
        />
      )}
    </StepShell>
  )
}

/** One picture of an example: the image with replace / remove on hover, or a drop target. */
function Slot({ label, src, disabled, onFile, onClear }: {
  label: string; src: string | null; disabled?: boolean; onFile: (f: File) => void; onClear: () => void
}) {
  const { t } = useTranslation()
  const ref = useRef<HTMLInputElement | null>(null)
  const [over, setOver] = useState(false)
  return (
    <div className="ds-ex-slot">
      <span className="ds-ex-slot-label">{label}</span>
      <div
        className={`ds-ex-slot-box${src ? '' : ' ds-empty-box'}${over ? ' ds-over' : ''}`}
        onDragOver={(e) => { if (!disabled) { e.preventDefault(); setOver(true) } }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault(); setOver(false)
          const f = e.dataTransfer.files[0]
          if (f && !disabled) onFile(f)
        }}
      >
        {src ? (
          <>
            <img src={src} alt={label} />
            {!disabled && (
              <span className="ds-ex-slot-tools">
                <button type="button" onClick={() => ref.current?.click()}>{t('pairs.replace')}</button>
                <button type="button" onClick={onClear} aria-label={t('common.delete')} title={t('common.delete')}>{EditIcon.x()}</button>
              </span>
            )}
          </>
        ) : (
          <button type="button" disabled={disabled} onClick={() => ref.current?.click()}>{t('pairs.dropHere')}</button>
        )}
      </div>
      <input ref={ref} type="file" hidden accept={IMAGE_ACCEPT}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = '' }} />
    </div>
  )
}

function ExampleEditor({ n, count, pid, vid, pair, recipe, bust, locked, onPrev, onNext, onUpload, onClear, onDelete, onSaved }: {
  n: number; count: number; pid: number; vid: number; pair: EditPair; recipe: EditRecipe; bust: number; locked: boolean
  onPrev: () => void; onNext: () => void
  onUpload: (role: EditSlot, f: File) => void; onClear: (role: EditSlot) => void
  onDelete: () => void; onSaved: () => void
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [caption, setCaption] = useState(pair.caption)
  const [saving, setSaving] = useState(false)
  useEffect(() => { setCaption(pair.caption) }, [pair.caption])

  const save = async () => {
    if (caption.trim() === pair.caption.trim()) return
    setSaving(true)
    try {
      await api.setEditCaption(pid, vid, pair.folder, pair.stem, caption)
      onSaved()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setSaving(false)
    }
  }

  const status = pair.missing.length === 0
    ? t('pairs.ready')
    : t('pairs.needs', { what: pair.missing.map((m) => t(`pairs.need_${m}`)).join(', ') })

  return (
    <>
      <div className="ds-card-head ds-pad">
        <div style={{ minWidth: 0 }}>
          <div className="ds-card-title">{t('pairs.exampleN', { n })}</div>
          <div className="ds-card-sub">{saving ? t('pairs.saving') : status}</div>
        </div>
        <div className="ds-card-tools" style={{ flexWrap: 'nowrap' }}>
          <button type="button" className="ds-kebab" onClick={onPrev} disabled={count < 2} aria-label={t('pairs.prev')}>{EditIcon.prev()}</button>
          <span className="ds-kpi-meta ds-mono">{n} / {count}</span>
          <button type="button" className="ds-kebab" onClick={onNext} disabled={count < 2} aria-label={t('pairs.next')}>{EditIcon.next()}</button>
        </div>
      </div>
      <div className="ds-ex-edit">
        <div className={`ds-ex-slots ds-n${recipeSlots(recipe).length}`}>
          {recipeSlots(recipe).map((role) => (
            <Slot key={role} label={t(`pairs.col_${role}`)} disabled={locked}
              src={pairImageUrl(pid, vid, pair, role, 512, bust)}
              onFile={(f) => onUpload(role, f)} onClear={() => onClear(role)} />
          ))}
        </div>
        <label className="ds-modal-field">
          <span className="ds-ex-slot-label">{t(recipe === 'text' ? 'pairs.col_text' : 'pairs.col_textOptional')}</span>
          <textarea
            className="ds-inp ds-ex-text"
            rows={4}
            value={caption}
            disabled={locked}
            placeholder={t(`pairs.textPlaceholder_${recipe}`)}
            onChange={(e) => setCaption(e.target.value)}
            onBlur={() => void save()}
          />
        </label>
        <button type="button" className="ds-ctl" style={{ alignSelf: 'flex-start', marginTop: 'auto' }} disabled={locked} onClick={onDelete}>
          {t('pairs.deleteExample')}
        </button>
      </div>
    </>
  )
}

function DropZone({ title, hint, withCaptions, onFiles }: {
  title: string; hint: string; withCaptions?: boolean; onFiles: (files: File[]) => Promise<void>
}) {
  const [over, setOver] = useState(false)
  const [busy, setBusy] = useState(false)
  const ref = useRef<HTMLInputElement | null>(null)
  const run = (files: File[]) => {
    if (!files.length) return
    setBusy(true)
    void onFiles(files).finally(() => setBusy(false))
  }
  return (
    <button
      type="button"
      disabled={busy}
      className={`ds-ex-drop${over ? ' ds-is-hover' : ''}`}
      onClick={() => ref.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); run(Array.from(e.dataTransfer.files)) }}
    >
      <span style={{ fontWeight: 600, color: 'var(--ink)', fontSize: 13 }}>{busy ? '…' : title}</span>
      <span>{hint}</span>
      <input ref={ref} type="file" multiple hidden accept={withCaptions ? `${IMAGE_ACCEPT},.txt` : IMAGE_ACCEPT}
        onChange={(e) => { run(Array.from(e.target.files ?? [])); e.target.value = '' }} />
    </button>
  )
}

function Modal({ title, children, onCancel, width = 560 }: {
  title: string; children: React.ReactNode; onCancel: () => void; width?: number
}) {
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]" onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel() }}>
      <div className="ds-modal" style={{ width: `min(${width}px, 94vw)` }}>
        <h2 className="ds-modal-title">{title}</h2>
        {children}
      </div>
    </div>
  )
}

function BulkDialog({ recipe, onCancel, onUpload }: {
  recipe: EditRecipe; onCancel: () => void; onUpload: (role: EditSlot, files: File[]) => Promise<void>
}) {
  const { t } = useTranslation()
  const zones = recipeSlots(recipe)
  return (
    <Modal title={t('pairs.bulkTitle')} onCancel={onCancel} width={zones.length > 2 ? 720 : 560}>
      <span className="ds-kpi-meta">{t('pairs.bulkHint')}</span>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${zones.length},1fr)`, gap: 10 }}>
        {zones.map((role) => (
          <DropZone key={role} title={t(`pairs.col_${role}`)} withCaptions={role === 'target'}
            hint={role === 'target' ? t(`pairs.bulkTargetHint_${recipe}`) : t('pairs.bulkInputHint')}
            onFiles={(files) => onUpload(role, files)} />
        ))}
      </div>
      <div className="ds-modal-foot">
        <button type="button" className="ds-btn-primary" onClick={onCancel}>{t('pairs.bulkDone')}</button>
      </div>
    </Modal>
  )
}

function InstructionDialog({ recipe, onCancel, onSubmit }: {
  recipe: EditRecipe
  onCancel: () => void
  onSubmit: (instruction: string, mode: 'replace' | 'prepend' | 'append') => Promise<void>
}) {
  const { t } = useTranslation()
  const [text, setText] = useState('')
  const [mode, setMode] = useState<'replace' | 'prepend' | 'append'>('replace')
  const [busy, setBusy] = useState(false)
  return (
    <Modal title={t('pairs.instructionTitle')} onCancel={onCancel} width={520}>
      <span className="ds-kpi-meta">{t('pairs.instructionHint')}</span>
      <textarea autoFocus className="ds-inp" style={{ height: 'auto', padding: '8px 11px' }} rows={3}
        value={text} onChange={(e) => setText(e.target.value)} placeholder={t(`pairs.textPlaceholder_${recipe}`)} />
      <div className="ds-seg" role="group" style={{ alignSelf: 'flex-start' }}>
        {(['replace', 'prepend', 'append'] as const).map((m) => (
          <button key={m} type="button" aria-pressed={mode === m}
            className={`ds-seg-item${mode === m ? ' ds-is-active' : ''}`} onClick={() => setMode(m)}>
            {t(`pairs.instr_${m}`)}
          </button>
        ))}
      </div>
      <div className="ds-modal-foot">
        <button type="button" className="ds-ctl" onClick={onCancel} disabled={busy}>{t('common.cancel')}</button>
        <button type="button" className="ds-btn-primary" disabled={busy || !text.trim()}
          onClick={() => { setBusy(true); void onSubmit(text, mode).finally(() => setBusy(false)) }}>
          {t('pairs.instructionApply')}
        </button>
      </div>
    </Modal>
  )
}
