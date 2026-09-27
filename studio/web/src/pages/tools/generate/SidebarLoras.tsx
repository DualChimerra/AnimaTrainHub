import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { LoraEntry } from '../../../api/client'
import PathPicker from '../../../components/PathPicker'
import AddSlotButton from './AddSlotButton'
import InlineLoraPicker, { type PickedLora } from './InlineLoraPicker'
import type { LoraCatalog } from './useLoraCatalog'

/** Sidebar's LoRA section: each LoRA = one persistent picker slot (project dropdown + ckpt chips + weight + x).
 *
 * Unified data model: every picker slot is represented in loras[], with path='' meaning an
 * "empty slot" (the user clicked + Add LoRA but hasn't picked a ckpt yet). This means:
 *   - "+ Add LoRA" pushes an empty entry -> a new picker appears (stable key, no flicker)
 *   - Deselecting (clicking an already-selected chip) sets the slot's path back to '', and the
 *     picker keeps rendering -> visually identical to a freshly opened slot
 *   - x actually removes the entry from the array
 * Generate.tsx's handleGenerate runs `loras.filter((l) => l.path.trim())` before sending to the
 * backend, filtering out empty slots without affecting the enqueue. */
export default function SidebarLoras({
  loras, onChange, catalog,
}: {
  loras: LoraEntry[]
  onChange: (l: LoraEntry[]) => void
  catalog: LoraCatalog
}) {
  const { t } = useTranslation()
  const [externalForIdx, setExternalForIdx] = useState<number | null>(null)
  // Slot whose checkpoint picker is unfolded (empty slots always show it).
  const [editingIdx, setEditingIdx] = useState<number | null>(null)

  // Already-selected paths (mutually disable each other to avoid adding duplicates) -- empty slots excluded
  const existingPaths = useMemo(
    () => new Set(loras.filter((l) => l.path).map((l) => l.path)),
    [loras],
  )

  const handleSlotChange = (i: number, picked: PickedLora | null, weight: number) => {
    const entry: LoraEntry = picked
      ? {
          path: picked.path,
          scale: weight,
          project_id: picked.projectId,
          version_id: picked.versionId,
        }
      : {
          // Deselect: keep the slot but clear path, picker still renders (visually identical to a freshly opened empty slot)
          path: '',
          scale: weight,
          project_id: null,
          version_id: null,
        }
    onChange(loras.map((l, idx) => (idx === i ? entry : l)))
    if (picked) setEditingIdx(null)
  }

  const handleWeight = (i: number, w: number) => {
    onChange(loras.map((l, idx) => (idx === i ? { ...l, scale: w } : l)))
  }

  const handleSlotRemove = (i: number) => {
    onChange(loras.filter((_, idx) => idx !== i))
    if (externalForIdx === i) setExternalForIdx(null)
    setEditingIdx(null)
  }

  const handleAddSlot = () => {
    onChange([
      ...loras,
      { path: '', scale: 1.0, project_id: null, version_id: null },
    ])
  }

  return (
    <div className="flex flex-col gap-2">
      {loras.map((l, i) => {
        const hasCkpt = !!l.path
        // Decision #8 / plan §3: a LoRA that failed to resolve after history backfill
        // (path='' && name retained) renders a warning placeholder card prompting the user to
        // re-pick it -- don't silently leave path empty and let the user wonder why
        if (!hasCkpt && l.name) {
          return (
            <PlaceholderLoraCard
              key={`lora-${i}`}
              name={l.name}
              onPick={() => {
                // Clear name so InlineLoraPicker shows up for the user to re-pick
                onChange(loras.map((lo, idx) => (
                  idx === i ? { ...lo, name: null } : lo
                )))
              }}
              onRemove={() => handleSlotRemove(i)}
              t={t}
            />
          )
        }
        const open = !hasCkpt || editingIdx === i
        return (
          <div
            key={`lora-${i}`}
            className="ds-card ds-flat"
            style={{ padding: '10px 11px', border: '1px solid var(--line-2)', display: 'flex', flexDirection: 'column', gap: 8 }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="ds-badge ds-mute">{t('generate.slotN', { n: i + 1 })}</span>
              <button
                type="button"
                className="ds-mono"
                style={{ fontSize: 11.5, flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', textAlign: 'left', color: hasCkpt ? 'var(--ink)' : 'var(--ink-3)' }}
                title={hasCkpt ? l.path : undefined}
                aria-expanded={open}
                onClick={() => { if (hasCkpt) setEditingIdx(editingIdx === i ? null : i) }}
              >
                {hasCkpt ? slotName(l.path) : t('generate.loraNotPicked')}
              </button>
              <button type="button" className="ds-kebab" aria-label={t('generate.removeSlot')} title={t('generate.removeSlot')} onClick={() => handleSlotRemove(i)}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>
              </button>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
              <input
                type="range"
                className="ds-range"
                min={0} max={1.5} step={0.05}
                value={l.scale}
                style={{ '--p': `${(l.scale / 1.5) * 100}%` } as React.CSSProperties}
                onChange={(e) => handleWeight(i, Number(e.target.value))}
                aria-label={t('generate.loraWeight')}
              />
              <span className="ds-mono" style={{ fontSize: 11.5, width: 34, textAlign: 'right' }}>{l.scale.toFixed(2)}</span>
            </div>
            {open && (
              <InlineLoraPicker
                mode="single"
                embedded
                catalog={catalog}
                value={
                  hasCkpt
                    ? { path: l.path, projectId: l.project_id ?? null, versionId: l.version_id ?? null }
                    : null
                }
                weight={l.scale}
                onChange={(p, w) => handleSlotChange(i, p, w)}
                onClose={() => handleSlotRemove(i)}
                onPickExternal={() => setExternalForIdx(i)}
              />
            )}
          </div>
        )
      })}

      <AddSlotButton onClick={handleAddSlot}>{t('generate.addLora')}</AddSlotButton>

      {/* External-file picker for a slot */}
      {externalForIdx !== null && (
        <PathPicker
          dirOnly={false}
          onPick={(p) => {
            const entry: LoraEntry = {
              path: p,
              scale: 1.0,
              project_id: null,
              version_id: null,
            }
            // Overwrite the target slot's contents; existingPaths already excludes empty paths, so external files can stack
            void existingPaths
            onChange(loras.map((l, idx) => (idx === externalForIdx ? entry : l)))
            setExternalForIdx(null)
          }}
          onClose={() => setExternalForIdx(null)}
        />
      )}
    </div>
  )
}

/** Slot title: the checkpoint file name without extension. */
function slotName(path: string): string {
  return (path.split(/[\\/]/).pop() ?? path).replace(/\.safetensors$/i, '')
}

/** Renders a LoRA slot that failed to resolve after history backfill (decision #8 / plan §3).
 *  Styled to match InlineLoraPicker (card style + border), but shows a warning + name +
 *  [Re-pick] [Remove] instead; it doesn't block submit (path='' gets filtered out). */
function PlaceholderLoraCard({
  name, onPick, onRemove, t,
}: {
  name: string
  onPick: () => void
  onRemove: () => void
  t: (key: string) => string
}) {
  return (
    <div
      className="flex items-center gap-2"
      style={{
        border: '1px solid var(--border-warn, var(--border-subtle))',
        background: 'var(--bg-warn-soft, var(--bg-sunken))',
        borderRadius: 'var(--r-md)',
        padding: '8px 10px',
        fontSize: 12,
      }}
    >
      <span aria-hidden="true" style={{ flexShrink: 0 }}>⚠</span>
      <div className="flex-1 min-w-0 flex flex-col gap-0.5">
        <div className="font-mono truncate" style={{ fontSize: 11, color: 'var(--fg-secondary)' }}>
          {name}
        </div>
        <div className="text-2xs text-fg-tertiary">{t('generate.loraNotFoundHint')}</div>
      </div>
      <button
        type="button"
        onClick={onPick}
        className="font-mono"
        style={{
          border: '1px solid var(--border-subtle)',
          background: 'var(--bg-elevated)',
          borderRadius: 'var(--r-sm)',
          padding: '3px 8px',
          fontSize: 11,
          color: 'var(--fg-secondary)',
          cursor: 'pointer',
          flexShrink: 0,
        }}
      >
        {t('generate.repickLora')}
      </button>
      <button
        type="button"
        onClick={onRemove}
        title={t('common.delete')}
        aria-label={t('common.delete')}
        style={{
          width: 20, height: 20,
          display: 'grid', placeItems: 'center',
          borderRadius: 999,
          border: 0,
          background: 'transparent',
          color: 'var(--fg-tertiary)',
          cursor: 'pointer',
          fontSize: 14,
          lineHeight: 1,
          padding: 0,
          flexShrink: 0,
        }}
      >
        ×
      </button>
    </div>
  )
}
