import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { LoraEntry, XYAxisType } from '../../../api/client'
import PathPicker from '../../../components/PathPicker'
import AddSlotButton from './AddSlotButton'
import InlineLoraPicker from './InlineLoraPicker'
import NumberListInput from './NumberListInput'
import type { LoraCatalog } from './useLoraCatalog'
import { AXIS_VALUE_TYPE, REQUIRES_LORA_INDEX, axisLabel, ckptStemFromPath, type XYAxisDraft } from './xy'

const ALL_AXES: XYAxisType[] = ['lora_ckpt', 'lora_scale', 'cfg_scale', 'steps']

function placeholderFor(axis: XYAxisType): string {
  const t = AXIS_VALUE_TYPE[axis]
  if (t === 'int') return '20, 25, 30'
  return '0.6, 0.8, 1.0'
}

/** Inline picker for the lora_ckpt axis: multi-select ckpts + auto push/update loras[] + set axis.raw + axis.loraIndex.
 *
 * Flow: the user multi-selects chips in the picker -> confirms -> we take all the picks'
 * (pid, vid) -- multi-select always stays under one (pid, vid), since the picker clears picked
 * when pid/vid switches. We look for a matching (project_id, version_id) in loras[]; if there's
 * none, push a new entry (path from picks[0], scale=1.0). axis.loraIndex then points at that
 * slot, and axis.raw is picks.map(path).join(', ').
 */
function AxisLoraCkptPicker({
  draft, onDraftChange, loras, onLorasChange, catalog,
}: {
  draft: XYAxisDraft
  onDraftChange: (d: XYAxisDraft) => void
  loras: LoraEntry[]
  onLorasChange: (l: LoraEntry[]) => void
  catalog: LoraCatalog
}) {
  const { t } = useTranslation()
  const [externalOpen, setExternalOpen] = useState(false)

  const commitPicks = (picks: { path: string; projectId: number | null; versionId: number | null }[]) => {
    if (picks.length === 0) {
      // In live mode, the picker sends an empty set whenever every chip is deselected or
      // pid/vid switches -> clear the axis binding. The previous entry in loras[] is left
      // untouched (the picker might reuse it).
      onDraftChange({ ...draft, loraIndex: null, raw: '' })
      return
    }
    const pid = picks[0].projectId
    const vid = picks[0].versionId
    // Look for a slot in loras[] already bound to this (pid, vid)
    let idx = loras.findIndex(
      (l) => l.project_id === pid && l.version_id === vid && pid !== null && vid !== null
    )
    let nextLoras = loras
    if (idx < 0) {
      // Not bound yet -> push a new entry as anchor (path from picks[0]; the backend mutates path per cell)
      const newEntry: LoraEntry = {
        path: picks[0].path,
        scale: 1.0,
        project_id: pid,
        version_id: vid,
      }
      nextLoras = [...loras, newEntry]
      idx = loras.length
      onLorasChange(nextLoras)
    }
    onDraftChange({
      ...draft,
      loraIndex: idx,
      raw: picks.map((p) => p.path).join(', '),
    })
  }

  // Show a summary of the bound LoRA (when raw is non-empty + loraIndex is valid). The
  // project/version name comes from the catalog (the picker mount below lazily fetches the
  // matching project/versions); before that's loaded it falls back to ckptStemFromPath(bound.path).
  const bound = draft.loraIndex !== null && draft.loraIndex < loras.length
    ? loras[draft.loraIndex]
    : null
  const boundProject = bound ? catalog.projects.find((p) => p.id === bound.project_id) : null
  const boundVersion = bound && bound.project_id != null
    ? catalog.versionsOf(bound.project_id)?.find((v) => v.id === bound.version_id)
    : undefined
  const matchedLabel = boundProject && boundVersion
    ? `${boundProject.title} / ${boundVersion.label}`
    : null
  const pickedCount = draft.raw.trim() ? draft.raw.split(',').filter((s) => s.trim()).length : 0

  // Controlled sync into InlineLoraPicker: the raw string is fed in as a path/basename list,
  // anchored to the bound LoRA's (pid, vid), so history backfill highlights the right picker
  // chips and auto-upgrades any basename to a full path.
  const selectedPaths = useMemo(
    () => draft.raw.split(',').map((s) => s.trim()).filter(Boolean),
    [draft.raw],
  )

  return (
    <div className="flex flex-col gap-1.5">
      {bound && pickedCount > 0 && (
        <div
          className="flex items-center gap-2 text-2xs"
          style={{
            padding: '4px 8px',
            borderRadius: 'var(--r-sm)',
            background: 'var(--bg-sunken)',
            color: 'var(--fg-tertiary)',
          }}
        >
          <span>{t('generate.sweep')}</span>
          <span className="font-medium" style={{ color: 'var(--fg-secondary)' }}>
            {matchedLabel ?? ckptStemFromPath(bound.path)}
          </span>
          <span className="font-mono">{t('generate.nCkpts', { count: pickedCount })}</span>
        </div>
      )}
      <InlineLoraPicker
        mode="multi"
        live
        catalog={catalog}
        existingPaths={new Set()}
        showWeight={false}
        selectedPaths={selectedPaths}
        initialPid={bound?.project_id ?? null}
        initialVid={bound?.version_id ?? null}
        onPick={commitPicks}
        onClose={() => { /* always present in the axis card, nothing to close */ }}
        onPickExternal={() => setExternalOpen(true)}
      />
      {externalOpen && (
        <PathPicker
          dirOnly={false}
          onPick={(p) => {
            commitPicks([{ path: p, projectId: null, versionId: null }])
            setExternalOpen(false)
          }}
          onClose={() => setExternalOpen(false)}
        />
      )}
    </div>
  )
}

function AxisCard({
  label, draft, onChange, onRemove, loras, onLorasChange, catalog,
}: {
  label: 'X' | 'Y'
  draft: XYAxisDraft
  onChange: (d: XYAxisDraft) => void
  onRemove?: () => void
  loras: LoraEntry[]
  onLorasChange: (l: LoraEntry[]) => void
  catalog: LoraCatalog
}) {
  const { t } = useTranslation()
  const isCkpt = draft.axis === 'lora_ckpt'

  // Same frame as a LoRA slot card.
  return (
    <div className="ds-card ds-flat" style={{ padding: '10px 11px', border: '1px solid var(--line-2)', display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* The axis name gets its own row as the card's title (the axis-type dropdown and values
          below both belong to it); x sits at the far right of this row, opposite the axis name
          -- previously the single letter "X" shared a row with the dropdown and looked like a
          close button, which was confusing. */}
      <div className="flex items-center justify-between gap-2">
        <span className="ds-badge ds-mute">
          {t('generate.xyAxis', { label })}
        </span>
        {onRemove && (
          <button
            onClick={onRemove}
            className="ds-kebab"
            title={t('generate.xyRemoveAxisTitle', { label })}
            aria-label={t('generate.xyRemoveAxisAria', { label })}
          >
            ×
          </button>
        )}
      </div>

      <select
        className="ds-inp"
        value={draft.axis}
        onChange={(e) => {
          const newAxis = e.target.value as XYAxisType
          onChange({
            ...draft,
            axis: newAxis,
            raw: newAxis === 'lora_ckpt' ? '' : draft.raw,
            loraIndex: REQUIRES_LORA_INDEX.has(newAxis)
              ? (loras.length > 0 ? 0 : null)
              : null,
          })
        }}
      >
        {ALL_AXES.map((a) => (
          <option key={a} value={a}>{axisLabel(a)}</option>
        ))}
      </select>

      {/* A thin line separates the axis-type dropdown from the values (no indentation, no
          horizontal space used); the values depend on the axis type selected above:
          lora_ckpt -> a multi-select chip picker (produces raw + loraIndex);
          lora_scale -> plain numeric chips (not bound to any LoRA); steps/cfg -> text. */}
      {isCkpt ? (
        <AxisLoraCkptPicker
          draft={draft}
          onDraftChange={onChange}
          loras={loras}
          onLorasChange={onLorasChange}
          catalog={catalog}
        />
      ) : draft.axis === 'lora_scale' ? (
        <NumberListInput
          raw={draft.raw}
          onChange={(raw) => onChange({ ...draft, raw })}
          placeholder="0.6, 0.8, 1.0"
        />
      ) : (
        <input
          type="text"
          className="ds-inp ds-mono"
          placeholder={placeholderFor(draft.axis)}
          value={draft.raw}
          onChange={(e) => onChange({ ...draft, raw: e.target.value })}
        />
      )}
    </div>
  )
}

/** Sidebar's XY axis configuration area (rendered only when mode=xy).
 *
 * 4 axis types:
 *   - LoRA (lora_ckpt): a picker multi-selects ckpts as grid points; each cell mutates path and re-injects it
 *   - Weight (lora_scale): a plain numeric axis, overriding every LoRA's multiplier globally within a cell
 *   - CFG / steps: a text field for a number list
 */
export default function SidebarXYAxes({
  xDraft, yDraft, onXChange, onYChange,
  loras, onLorasChange, catalog,
}: {
  xDraft: XYAxisDraft
  yDraft: XYAxisDraft | null
  onXChange: (d: XYAxisDraft) => void
  onYChange: (d: XYAxisDraft | null) => void
  loras: LoraEntry[]
  onLorasChange: (l: LoraEntry[]) => void
  catalog: LoraCatalog
}) {
  const { t } = useTranslation()
  return (
    // No outer .card wrapper here -- the parent sidebar is already a unified card, so this only groups content to avoid a double border.
    <div>
      <div className="ds-cap" style={{ marginBottom: 8 }}>{t('generate.xyAxes')}</div>
      <div className="flex flex-col gap-2">
        <AxisCard
          label="X" draft={xDraft} onChange={onXChange}
          loras={loras} onLorasChange={onLorasChange} catalog={catalog}
        />
        {yDraft ? (
          <AxisCard
            label="Y" draft={yDraft}
            onChange={(d) => onYChange(d)}
            onRemove={() => onYChange(null)}
            loras={loras} onLorasChange={onLorasChange} catalog={catalog}
          />
        ) : (
          <AddSlotButton
            onClick={() => onYChange({
              axis: 'lora_scale',
              raw: '1',  // Start with one value; the user adds more with +
              loraIndex: null,
            })}
          >
            {t('generate.addYAxis')}
          </AddSlotButton>
        )}
      </div>
    </div>
  )
}
