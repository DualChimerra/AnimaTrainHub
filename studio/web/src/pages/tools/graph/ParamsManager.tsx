import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { formatNumber, newId, numberRange, parseNumber, sameValue, scientific } from './model'
import type { Param, ParamOption, ParamType, Run, RunValue } from './types'
import { Chip, Ico, Modal, useValueLabel } from './ui'
import HelpTip from '../../../components/ds/HelpTip'

export interface Remap { paramId: string; from: RunValue; to: RunValue | null }

const TYPES: ParamType[] = ['list', 'number', 'bool', 'text']

/**
 * Edit the board's parameters on a draft; nothing is written until Save.
 * Anything that would change existing cards (deleting a used value, editing
 * a used number) is spelled out first and offers archiving instead.
 */
export default function ParamsManager({ params, runs, onClose, onSave }: {
  params: Param[]
  runs: Run[]
  onClose: () => void
  onSave: (params: Param[], remaps: Remap[]) => Promise<boolean>
}) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState<Param[]>(() => structuredClone(params))
  const [remaps, setRemaps] = useState<Remap[]>([])
  const [openId, setOpenId] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [newName, setNewName] = useState('')
  const [newType, setNewType] = useState<ParamType>('list')

  const dirty = JSON.stringify(draft) !== JSON.stringify(params) || remaps.length > 0

  const usage = (pid: string, v?: RunValue) =>
    runs.filter((r) => r.values[pid] !== undefined && (v === undefined || sameValue(r.values[pid], v))).length

  const patch = (id: string, fn: (p: Param) => Param) => setDraft((d) => d.map((p) => (p.id === id ? fn(p) : p)))
  const move = (id: string, dir: -1 | 1) => setDraft((d) => {
    const i = d.findIndex((p) => p.id === id)
    const j = i + dir
    if (i < 0 || j < 0 || j >= d.length) return d
    const next = [...d]
    ;[next[i], next[j]] = [next[j], next[i]]
    return next
  })

  const add = () => {
    const name = newName.trim()
    if (!name) return
    const id = newId('p')
    const options: ParamOption[] = newType === 'bool' ? [{ id: 'yes', label: 'Yes' }, { id: 'no', label: 'No' }] : []
    setDraft((d) => [...d, { id, name, type: newType, options }])
    setNewName('')
    setOpenId(id)
  }

  const removeParam = (p: Param, mode: 'archive' | 'delete') => {
    if (mode === 'archive') patch(p.id, (x) => ({ ...x, archived: true }))
    else {
      setDraft((d) => d.filter((x) => x.id !== p.id).map((x) =>
        x.condition?.param === p.id ? { ...x, condition: undefined } : x))
      const used = runs.filter((r) => r.values[p.id] !== undefined)
      const seen: RunValue[] = []
      for (const r of used) if (!seen.some((v) => sameValue(v, r.values[p.id]))) seen.push(r.values[p.id])
      setRemaps((rs) => [...rs, ...seen.map((from) => ({ paramId: p.id, from, to: null }))])
    }
    setOpenId(null)
  }

  const save = async () => {
    setSaving(true)
    const ok = await onSave(draft, remaps)
    setSaving(false)
    if (ok) onClose()
  }

  const active = draft.filter((p) => !p.archived)
  const archived = draft.filter((p) => p.archived)

  return (
    <Modal title={t('graph.paramsTitle')} sub={t('graph.paramsSub')} onClose={onClose} width={760}
      foot={<>
        {remaps.length > 0 && <span className="ds-cell-key" style={{ marginRight: 'auto' }}>{t('graph.remapsPending', { count: remaps.length })}</span>}
        <button type="button" className="ds-ctl ds-ghost" onClick={onClose}>{t('common.cancel')}</button>
        <button type="button" className="ds-btn-primary" disabled={!dirty || saving} onClick={() => void save()}>
          {saving ? t('common.saving') : t('common.save')}
        </button>
      </>}>
      <div className="gr-pm">
        {active.map((p, i) => (
          <ParamRow key={p.id} p={p} all={draft} open={openId === p.id} usage={usage(p.id)}
            first={i === 0} last={i === active.length - 1}
            onToggle={() => setOpenId((o) => (o === p.id ? null : p.id))}
            onMove={(dir) => move(p.id, dir)}
            onChange={(fn) => patch(p.id, fn)}
            onRemove={(mode) => removeParam(p, mode)}
            usageOf={(v) => usage(p.id, v)}
            onRemap={(r) => setRemaps((rs) => [...rs, r])}
          />
        ))}

        <div className="gr-pm-add">
          <input className="ds-inp" value={newName} onChange={(e) => setNewName(e.target.value)} placeholder={t('graph.newParamName')}
            onKeyDown={(e) => { if (e.key === 'Enter') add() }} aria-label={t('graph.newParamName')} />
          <select className="ds-inp" value={newType} onChange={(e) => setNewType(e.target.value as ParamType)} style={{ width: 150 }} aria-label={t('graph.paramType')}>
            {TYPES.map((ty) => <option key={ty} value={ty}>{t(`graph.type_${ty}`)}</option>)}
          </select>
          <button type="button" className="ds-ctl" onClick={add} disabled={!newName.trim()}>{Ico.plus}{t('graph.addParam')}</button>
        </div>

        {archived.length > 0 && (
          <div className="gr-pm-archive">
            <div className="ds-cap">{t('graph.archive')}</div>
            {archived.map((p) => (
              <div key={p.id} className="gr-pm-arow">
                <span>{p.name}</span>
                <span className="ds-cell-key">{t('graph.usedIn', { count: usage(p.id) })}</span>
                <button type="button" className="gr-link" onClick={() => patch(p.id, (x) => ({ ...x, archived: false }))}>{t('common.restore')}</button>
              </div>
            ))}
          </div>
        )}
      </div>
    </Modal>
  )
}

function ParamRow({ p, all, open, usage, first, last, onToggle, onMove, onChange, onRemove, usageOf, onRemap }: {
  p: Param
  all: Param[]
  open: boolean
  usage: number
  first: boolean
  last: boolean
  onToggle: () => void
  onMove: (dir: -1 | 1) => void
  onChange: (fn: (p: Param) => Param) => void
  onRemove: (mode: 'archive' | 'delete') => void
  usageOf: (v: RunValue) => number
  onRemap: (r: Remap) => void
}) {
  const { t } = useTranslation()
  const [confirmDel, setConfirmDel] = useState(false)
  const [advanced, setAdvanced] = useState(false)
  const parents = all.filter((x) => x.id !== p.id && !x.archived && (x.type === 'list' || x.type === 'bool'))
  const parent = p.condition ? all.find((x) => x.id === p.condition!.param) : undefined
  const label = useValueLabel()

  return (
    <div className={`gr-pm-row${open ? ' gr-open' : ''}${p.hidden ? ' gr-hidden' : ''}`}>
      <div className="gr-pm-head">
        <span className="gr-pm-move">
          <button type="button" className="ds-kebab" disabled={first} onClick={() => onMove(-1)} aria-label={t('graph.moveUp')}>{Ico.up}</button>
          <button type="button" className="ds-kebab" disabled={last} onClick={() => onMove(1)} aria-label={t('graph.moveDown')}>{Ico.down}</button>
        </span>
        <button type="button" className="gr-pm-title" onClick={onToggle} aria-expanded={open}>
          <span className="gr-pm-name">{p.name}</span>
          <span className="ds-tag ds-adv">{t(`graph.type_${p.type}`)}</span>
          {parent && <span className="ds-tag ds-gate">{t('graph.ifShort', { name: parent.name })}</span>}
          <span className="ds-cell-key">{t('graph.usedIn', { count: usage })}</span>
        </button>
        <button type="button" className="ds-kebab" onClick={() => onChange((x) => ({ ...x, hidden: !x.hidden }))}
          title={p.hidden ? t('graph.showParam') : t('graph.hideParam')} aria-label={p.hidden ? t('graph.showParam') : t('graph.hideParam')}>
          {p.hidden ? Ico.eyeOff : Ico.eye}
        </button>
        <button type="button" className="ds-kebab" onClick={onToggle} aria-label={t('graph.edit')}>{open ? Ico.chevD : Ico.chev}</button>
      </div>

      {open && (
        <div className="gr-pm-body">
          <div className="gr-pm-grid">
            <label className="gr-mfield">
              <span className="gr-pfield-name">{t('common.name')}</span>
              <input className="ds-inp" value={p.name} onChange={(e) => onChange((x) => ({ ...x, name: e.target.value }))} />
            </label>
            <label className="gr-mfield">
              <span className="gr-pfield-name">{t('graph.description')}</span>
              <input className="ds-inp" value={p.description ?? ''} placeholder={t('graph.descriptionPh')}
                onChange={(e) => onChange((x) => ({ ...x, description: e.target.value || undefined }))} />
            </label>
          </div>

          <div className="gr-mfield">
            <span className="gr-pfield-name">{t('graph.condition')}</span>
            <div className="gr-cond">
              <select className="ds-inp" value={p.condition?.param ?? ''} style={{ width: 200 }} aria-label={t('graph.condition')}
                onChange={(e) => onChange((x) => ({ ...x, condition: e.target.value ? { param: e.target.value, options: [] } : undefined }))}>
                <option value="">{t('graph.alwaysShown')}</option>
                {parents.map((x) => <option key={x.id} value={x.id}>{t('graph.whenParam', { name: x.name })}</option>)}
              </select>
              {parent && parent.options.filter((o) => !o.archived).map((o) => {
                const on = p.condition!.options.includes(o.id)
                return (
                  <Chip key={o.id} on={on} onClick={() => onChange((x) => ({
                    ...x, condition: { param: parent.id, options: on ? x.condition!.options.filter((y) => y !== o.id) : [...x.condition!.options, o.id] },
                  }))}>{label(parent, o.id)}</Chip>
                )
              })}
            </div>
            {parent && !p.condition!.options.length && <span className="ds-ctl-note">{t('graph.conditionPick')}</span>}
          </div>

          {p.type === 'number'
            ? <NumberOptions p={p} onChange={onChange} usageOf={usageOf} onRemap={onRemap} />
            : p.type !== 'text' && <ListOptions p={p} onChange={onChange} usageOf={usageOf} onRemap={onRemap} advanced={advanced} />}

          <div className="gr-pm-foot">
            {p.type !== 'text' && (
              <button type="button" className="gr-link" onClick={() => setAdvanced((v) => !v)}>
                {advanced ? t('graph.hideConfigLink') : t('graph.showConfigLink')}
              </button>
            )}
            <span style={{ flex: 1 }} />
            {!confirmDel ? (
              <button type="button" className="ds-btn-danger" onClick={() => (usage ? setConfirmDel(true) : onRemove('delete'))}>
                {Ico.trash}{t('graph.deleteParam')}
              </button>
            ) : (
              <div className="ds-note ds-warn gr-confirm">
                <span style={{ flex: 1 }}>{t('graph.deleteParamUsed', { count: usage })}</span>
                <button type="button" className="ds-ctl" onClick={() => onRemove('archive')}>{Ico.archive}{t('graph.archiveInstead')}</button>
                <button type="button" className="ds-btn-danger" onClick={() => onRemove('delete')}>{t('graph.deleteAnyway')}</button>
                <button type="button" className="ds-ctl ds-ghost" onClick={() => setConfirmDel(false)}>{t('common.cancel')}</button>
              </div>
            )}
          </div>
          {advanced && p.type === 'number' && (
            <div className="gr-cfglink">
              <span className="ds-cell-key">{t('graph.configLinkHelp')}</span>
              <div className="gr-cond">
                <input className="ds-inp ds-mono" style={{ width: 220 }} value={p.config?.key ?? ''} placeholder="learning_rate"
                  onChange={(e) => onChange((x) => ({ ...x, config: e.target.value ? { key: e.target.value.trim(), list: x.config?.list } : undefined }))}
                  aria-label={t('graph.configKey')} />
                <label className="gr-switch-l">
                  <input type="checkbox" checked={!!p.config?.list} disabled={!p.config?.key}
                    onChange={(e) => onChange((x) => ({ ...x, config: x.config ? { ...x.config, list: e.target.checked || undefined } : undefined }))} />
                  {t('graph.configList')}
                </label>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ListOptions({ p, onChange, usageOf, onRemap, advanced }: {
  p: Param
  onChange: (fn: (p: Param) => Param) => void
  usageOf: (v: RunValue) => number
  onRemap: (r: Remap) => void
  advanced: boolean
}) {
  const { t } = useTranslation()
  const [text, setText] = useState('')
  const [pending, setPending] = useState<string | null>(null)
  const setOpt = (id: string, fn: (o: ParamOption) => ParamOption) =>
    onChange((x) => ({ ...x, options: x.options.map((o) => (o.id === id ? fn(o) : o)) }))
  const moveOpt = (id: string, dir: -1 | 1) => onChange((x) => {
    const i = x.options.findIndex((o) => o.id === id)
    const j = i + dir
    if (j < 0 || j >= x.options.length) return x
    const options = [...x.options]
    ;[options[i], options[j]] = [options[j], options[i]]
    return { ...x, options }
  })
  const add = () => {
    const labels = text.split(/[,;\n]/).map((s) => s.trim()).filter(Boolean)
    if (!labels.length) return
    onChange((x) => ({ ...x, options: [...x.options, ...labels.map((l) => ({ id: newId('o'), label: l }))] }))
    setText('')
  }
  const remove = (o: ParamOption, mode: 'archive' | 'delete') => {
    if (mode === 'archive') setOpt(o.id, (x) => ({ ...x, archived: true }))
    else {
      onChange((x) => ({ ...x, options: x.options.filter((y) => y.id !== o.id) }))
      if (usageOf(o.id)) onRemap({ paramId: p.id, from: o.id, to: null })
    }
    setPending(null)
  }
  return (
    <div className="gr-mfield">
      <span className="gr-pfield-name">{t('graph.values')}</span>
      <div className="gr-opts">
        {p.options.map((o, i) => {
          const n = usageOf(o.id)
          return (
            <div key={o.id} className={`gr-opt${o.archived ? ' gr-archived-opt' : ''}`}>
              <div className="gr-opt-row">
                <span className="gr-pm-move">
                  <button type="button" className="ds-kebab" disabled={i === 0} onClick={() => moveOpt(o.id, -1)} aria-label={t('graph.moveUp')}>{Ico.up}</button>
                  <button type="button" className="ds-kebab" disabled={i === p.options.length - 1} onClick={() => moveOpt(o.id, 1)} aria-label={t('graph.moveDown')}>{Ico.down}</button>
                </span>
                <input className="ds-inp" value={o.label ?? ''} onChange={(e) => setOpt(o.id, (x) => ({ ...x, label: e.target.value }))} aria-label={t('graph.valueLabel')} />
                <span className="ds-cell-key gr-opt-n">{t('graph.usedIn', { count: n })}</span>
                {o.archived ? (
                  <button type="button" className="gr-link" onClick={() => setOpt(o.id, (x) => ({ ...x, archived: undefined }))}>{t('common.restore')}</button>
                ) : p.type !== 'bool' && (
                  <button type="button" className="ds-kebab" onClick={() => (n ? setPending(o.id) : remove(o, 'delete'))}
                    aria-label={t('common.delete')} title={t('common.delete')}>{Ico.trash}</button>
                )}
              </div>
              {pending === o.id && (
                <div className="ds-note ds-warn gr-confirm">
                  <span style={{ flex: 1 }}>{t('graph.deleteValueUsed', { count: n })}</span>
                  <button type="button" className="ds-ctl" onClick={() => remove(o, 'archive')}>{Ico.archive}{t('graph.archiveInstead')}</button>
                  <button type="button" className="ds-btn-danger" onClick={() => remove(o, 'delete')}>{t('graph.deleteAnyway')}</button>
                  <button type="button" className="ds-ctl ds-ghost" onClick={() => setPending(null)}>{t('common.cancel')}</button>
                </div>
              )}
              {advanced && <OptionConfig o={o} onChange={(c) => setOpt(o.id, (x) => ({ ...x, config: c }))} />}
            </div>
          )
        })}
      </div>
      {p.type !== 'bool' && (
        <div className="gr-cond">
          <input className="ds-inp" value={text} onChange={(e) => setText(e.target.value)} placeholder={t('graph.addValuesPh')}
            onKeyDown={(e) => { if (e.key === 'Enter') add() }} aria-label={t('graph.addValues')} />
          <button type="button" className="ds-ctl" onClick={add} disabled={!text.trim()}>{Ico.plus}{t('graph.addValues')}</button>
        </div>
      )}
    </div>
  )
}

function OptionConfig({ o, onChange }: { o: ParamOption; onChange: (c: ParamOption['config']) => void }) {
  const { t } = useTranslation()
  const [apply, setApply] = useState(JSON.stringify(o.config?.apply ?? {}))
  const [match, setMatch] = useState(o.config?.match ? JSON.stringify(o.config.match) : '')
  const [err, setErr] = useState<string | null>(null)
  const commit = (a: string, m: string) => {
    try {
      const pa = a.trim() ? JSON.parse(a) : {}
      const pm = m.trim() ? JSON.parse(m) : undefined
      if (typeof pa !== 'object' || Array.isArray(pa) || (pm !== undefined && (typeof pm !== 'object' || Array.isArray(pm)))) throw new Error('obj')
      setErr(null)
      onChange(Object.keys(pa).length || pm ? { apply: pa, ...(pm ? { match: pm } : {}) } : undefined)
    } catch {
      setErr(t('graph.badJson'))
    }
  }
  return (
    <div className="gr-cfglink">
      <label className="gr-mfield"><span className="ds-cell-key">{t('graph.configApply')}</span>
        <input className="ds-inp ds-mono" value={apply} onChange={(e) => setApply(e.target.value)} onBlur={() => commit(apply, match)} placeholder='{"lora_type": "lokr"}' />
      </label>
      <label className="gr-mfield"><span className="ds-cell-key">{t('graph.configMatch')}</span>
        <input className="ds-inp ds-mono" value={match} onChange={(e) => setMatch(e.target.value)} onBlur={() => commit(apply, match)} placeholder='{"weight_decay": {"$gt": 0}}' />
      </label>
      {err && <span className="ds-ctl-note" style={{ color: 'var(--red-text)' }}>{err}</span>}
    </div>
  )
}

function NumberOptions({ p, onChange, usageOf, onRemap }: {
  p: Param
  onChange: (fn: (p: Param) => Param) => void
  usageOf: (v: RunValue) => number
  onRemap: (r: Remap) => void
}) {
  const { t } = useTranslation()
  const [text, setText] = useState('')
  const [range, setRange] = useState({ from: '', to: '', step: '' })
  const [editing, setEditing] = useState<{ id: string; text: string } | null>(null)
  const [pending, setPending] = useState<string | null>(null)
  const sorted = useMemo(() => [...p.options].sort((a, b) => (a.value ?? 0) - (b.value ?? 0)), [p.options])

  const addValues = (vals: number[]) => onChange((x) => {
    const options = [...x.options]
    for (const v of vals) if (!options.some((o) => o.value !== undefined && sameValue(o.value, v))) options.push({ id: newId('n'), value: v })
    return { ...x, options }
  })
  const addText = () => {
    const vals = text.split(/[\s,;]+/).map(parseNumber).filter((n): n is number => n !== null)
    if (vals.length) addValues(vals)
    setText('')
  }
  const rangeVals = (() => {
    const a = parseNumber(range.from)
    const b = parseNumber(range.to)
    const s = parseNumber(range.step)
    return a !== null && b !== null && s !== null ? numberRange(a, b, Math.abs(s)) : []
  })()
  const commitEdit = () => {
    if (!editing) return
    const o = p.options.find((x) => x.id === editing.id)
    const n = parseNumber(editing.text)
    if (o && n !== null && o.value !== undefined && !sameValue(o.value, n)) {
      if (usageOf(o.value)) onRemap({ paramId: p.id, from: o.value, to: n })
      onChange((x) => ({ ...x, options: x.options.map((y) => (y.id === o.id ? { ...y, value: n } : y)) }))
    }
    setEditing(null)
  }
  const remove = (o: ParamOption, mode: 'archive' | 'delete') => {
    if (mode === 'archive') onChange((x) => ({ ...x, options: x.options.map((y) => (y.id === o.id ? { ...y, archived: true } : y)) }))
    else {
      onChange((x) => ({ ...x, options: x.options.filter((y) => y.id !== o.id) }))
      if (o.value !== undefined && usageOf(o.value)) onRemap({ paramId: p.id, from: o.value, to: null })
    }
    setPending(null)
  }

  return (
    <div className="gr-mfield">
      <span className="gr-pfield-name">{t('graph.values')}<HelpTip>{t('graph.numberValuesHint')}</HelpTip></span>
      <div className="gr-numvals">
        {sorted.length === 0 && <span className="ds-cell-key">{t('graph.noValuesYet')}</span>}
        {sorted.map((o) => {
          const n = o.value !== undefined ? usageOf(o.value) : 0
          if (editing?.id === o.id) {
            return (
              <input key={o.id} className="ds-inp ds-mono gr-numedit" autoFocus value={editing.text}
                onChange={(e) => setEditing({ id: o.id, text: e.target.value })} onBlur={commitEdit}
                onKeyDown={(e) => { if (e.key === 'Enter') commitEdit(); if (e.key === 'Escape') setEditing(null) }} />
            )
          }
          return (
            <span key={o.id} className={`gr-numval${o.archived ? ' gr-archived-opt' : ''}`}>
              <button type="button" className="gr-numval-v" onClick={() => setEditing({ id: o.id, text: formatNumber(o.value!) })} title={t('graph.editValue')}>
                {formatNumber(o.value!)}{scientific(o.value!) && <span className="gr-sci">{scientific(o.value!)}</span>}
              </button>
              {n > 0 && <b>{n}</b>}
              {o.archived
                ? <button type="button" className="gr-link" onClick={() => onChange((x) => ({ ...x, options: x.options.map((y) => (y.id === o.id ? { ...y, archived: undefined } : y)) }))}>{t('common.restore')}</button>
                : <button type="button" className="ds-chip-x" onClick={() => (n ? setPending(o.id) : remove(o, 'delete'))} aria-label={t('common.delete')}>{Ico.close}</button>}
            </span>
          )
        })}
      </div>
      {pending && (() => {
        const o = p.options.find((x) => x.id === pending)!
        return (
          <div className="ds-note ds-warn gr-confirm">
            <span style={{ flex: 1 }}>{t('graph.deleteValueUsed', { count: usageOf(o.value!) })}</span>
            <button type="button" className="ds-ctl" onClick={() => remove(o, 'archive')}>{Ico.archive}{t('graph.archiveInstead')}</button>
            <button type="button" className="ds-btn-danger" onClick={() => remove(o, 'delete')}>{t('graph.deleteAnyway')}</button>
            <button type="button" className="ds-ctl ds-ghost" onClick={() => setPending(null)}>{t('common.cancel')}</button>
          </div>
        )
      })()}
      <div className="gr-cond">
        <input className="ds-inp ds-mono" value={text} onChange={(e) => setText(e.target.value)} placeholder="0.0001, 2e-5"
          onKeyDown={(e) => { if (e.key === 'Enter') addText() }} aria-label={t('graph.addValues')} />
        <button type="button" className="ds-ctl" onClick={addText} disabled={!text.trim()}>{Ico.plus}{t('graph.addValues')}</button>
      </div>
      <div className="gr-cond">
        <span className="ds-cell-key">{t('graph.range')}</span>
        <input className="ds-inp ds-mono gr-rng" value={range.from} placeholder={t('graph.from')} onChange={(e) => setRange((r) => ({ ...r, from: e.target.value }))} aria-label={t('graph.from')} />
        <input className="ds-inp ds-mono gr-rng" value={range.to} placeholder={t('graph.to')} onChange={(e) => setRange((r) => ({ ...r, to: e.target.value }))} aria-label={t('graph.to')} />
        <input className="ds-inp ds-mono gr-rng" value={range.step} placeholder={t('graph.stepSize')} onChange={(e) => setRange((r) => ({ ...r, step: e.target.value }))} aria-label={t('graph.stepSize')} />
        <button type="button" className="ds-ctl" disabled={!rangeVals.length}
          onClick={() => { addValues(rangeVals); setRange({ from: '', to: '', step: '' }) }}>
          {rangeVals.length ? t('graph.addNValues', { count: rangeVals.length }) : t('graph.addValues')}
        </button>
      </div>
    </div>
  )
}
