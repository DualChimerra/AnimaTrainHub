// RuleImpactDialog -- confirmation dialog for rule-cascaded value changes
// (polish pass 2 / R6, D6).
//
// When the user changes a gate field (e.g. enabling InfoNoise, switching the
// optimizer to prodigy) and that triggers a disable_when rule takeover, if
// any of the resulting writes are "lossy" (the target field's current value
// differs from the value about to be written), SchemaForm intercepts at the
// setField entry point: it keeps the invalid state out of form state and
// first shows this dialog listing every change (from -> to + the domain
// reason). Only on confirm does it apply (the trigger change plus every
// cascaded write, committed together); cancel does nothing. Lossless changes
// (the cascade target is already at the pinned value) apply silently with no
// dialog.
//
// FamilySwitchDialog is a family-switch special case of the same interaction
// pattern (it needs the backend to recompute paths, this one is pure
// frontend metadata evaluation); the UI skeleton stays consistent.
import { useTranslation } from 'react-i18next'
import { fieldLabel } from '../lib/schema'

export interface RuleImpactChange {
  field: string
  from: unknown
  to: unknown
  /** Domain reason (the field's disable_hint / suggestion text), shown below the change row. */
  reason?: string
}

interface Props {
  /** The user's original triggering change (the gate field), shown in the title area. */
  trigger: { field: string; from: unknown; to: unknown }
  changes: RuleImpactChange[]
  onApply: () => void
  onCancel: () => void
}

export default function RuleImpactDialog({ trigger, changes, onApply, onCancel }: Props) {
  const { t } = useTranslation()

  const fmt = (v: unknown): string => {
    if (v === null || v === undefined || v === '') return t('familySwitch.empty')
    if (typeof v === 'boolean') return v ? t('field.yes') : t('field.no')
    return String(v)
  }

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]"
      onClick={onCancel}
    >
      <div
        className="bg-elevated border border-subtle rounded-2xl w-[92%] max-w-[560px] max-h-[80vh] p-6 flex flex-col gap-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <h3 className="m-0 text-base font-semibold text-fg-primary">
            {t('ruleImpact.title')}
          </h3>
          <p className="m-0 mt-1 text-sm text-fg-secondary">
            {t('ruleImpact.intro', {
              field: fieldLabel(trigger.field),
              value: fmt(trigger.to),
            })}
          </p>
        </div>

        <div className="overflow-y-auto flex flex-col gap-3 pr-1">
          {changes.map((c) => (
            <div key={c.field} className="text-sm">
              <div className="flex items-baseline gap-2">
                <span className="font-medium text-fg-secondary">{fieldLabel(c.field)}</span>
                <span className="text-fg-tertiary">{fmt(c.from)}</span>
                <span className="text-accent">→</span>
                <span className="text-fg-primary">{fmt(c.to)}</span>
              </div>
              {c.reason && (
                <div className="mt-0.5 text-xs text-fg-tertiary">{c.reason}</div>
              )}
            </div>
          ))}
        </div>

        <div className="flex gap-2 justify-end mt-1">
          <button
            type="button"
            onClick={onCancel}
            className="btn btn-secondary min-w-[96px] justify-center"
          >
            {t('common.cancel')}
          </button>
          <button
            type="button"
            onClick={onApply}
            className="btn btn-primary min-w-[96px] justify-center"
          >
            {t('ruleImpact.ok')}
          </button>
        </div>
      </div>
    </div>
  )
}
