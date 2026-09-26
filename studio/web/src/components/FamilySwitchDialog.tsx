// FamilySwitchDialog -- confirmation dialog for switching model family (multi-model P4-3).
//
// Flipping model_family isn't a plain field edit: paths get recomputed for the
// target family, family-flavor fields get reset, and capability fields the
// target family doesn't support get turned off. This component calls the
// backend preview computation on open (/api/models/family-switch, pure
// computation, nothing written to disk), shows the changes structured into two
// sections ("model paths / parameter adjustments"), and only on confirm does
// it hand the fully recomputed config back to the caller (which then goes
// through that page's normal save flow).
//
// Doesn't use the generic Dialog.confirm text slot: the change list is
// structured data (long paths + old/new comparisons) that becomes an
// unreadable jumble of line breaks stuffed into plain text -- per Dialog.tsx's
// own convention, complex content goes through a declarative JSX modal (same
// pattern as NewVersionDialog).
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type ConfigData,
  type FamilySwitchChange,
} from '../api/client'
import { fieldLabel, schemaEnumLabel } from '../lib/schema'

interface Props {
  /** Target family id (the new value the user picked in the dropdown). */
  target: string
  /** Current config (before the switch, model_family is still the old value). */
  config: ConfigData
  /** User confirmed: apply the fully recomputed config from the backend. */
  onApply: (switched: ConfigData) => void
  /** User canceled / preview failed: caller keeps the old value unchanged. */
  onCancel: () => void
}

/** The 4 weight path fields -- shown with monospace font + a stacked before/after layout. */
const PATH_FIELDS = new Set([
  'transformer_path', 'vae_path', 'text_encoder_path', 't5_tokenizer_path',
])

function useSwitchPreview(target: string, config: ConfigData) {
  const [preview, setPreview] = useState<{
    config: ConfigData
    changes: FamilySwitchChange[]
  } | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let alive = true
    api.switchModelFamily(target, config)
      .then((r) => { if (alive) setPreview(r) })
      .catch((e) => { if (alive) setError(String(e)) })
    return () => { alive = false }
    // The config reference doesn't change over the dialog's lifetime (snapshotted on open)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target])
  return { preview, error }
}

export default function FamilySwitchDialog({ target, config, onApply, onCancel }: Props) {
  const { t } = useTranslation()
  const { preview, error } = useSwitchPreview(target, config)

  const fmt = (v: unknown): string => {
    if (v === null || v === undefined || v === '') return t('familySwitch.empty')
    if (typeof v === 'boolean') return v ? t('field.yes') : t('field.no')
    return String(v)
  }

  const changes = (preview?.changes ?? []).filter((c) => c.field !== 'model_family')
  const pathChanges = changes.filter((c) => PATH_FIELDS.has(c.field))
  const paramChanges = changes.filter((c) => !PATH_FIELDS.has(c.field))
  const fromLabel = schemaEnumLabel('model_family', String(config.model_family ?? 'anima'), t)
  const toLabel = schemaEnumLabel('model_family', target, t)

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]"
      onClick={onCancel}
    >
      <div
        className="bg-elevated border border-subtle rounded-2xl w-[92%] max-w-[640px] max-h-[85vh] p-6 flex flex-col gap-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <h3 className="m-0 text-base font-semibold text-fg-primary">
            {t('familySwitch.title')}
          </h3>
          <p className="m-0 mt-1 text-sm text-fg-secondary">
            {t('familySwitch.intro', { from: fromLabel, to: toLabel })}
          </p>
        </div>

        {error ? (
          <p className="m-0 text-sm text-err">{t('familySwitch.failed', { error })}</p>
        ) : !preview ? (
          <p className="m-0 text-sm text-fg-tertiary">{t('familySwitch.loading')}</p>
        ) : changes.length === 0 ? (
          <p className="m-0 text-sm text-fg-secondary">{t('familySwitch.noChanges')}</p>
        ) : (
          <div className="overflow-y-auto flex flex-col gap-4 pr-1">
            {pathChanges.length > 0 && (
              <section>
                <div className="caption mb-2">
                  {t('familySwitch.pathsSection')}
                </div>
                <div className="flex flex-col gap-2.5">
                  {pathChanges.map((c) => (
                    <div key={c.field} className="text-sm">
                      <div className="font-medium text-fg-secondary mb-0.5">
                        {fieldLabel(c.field)}
                      </div>
                      <div className="font-mono text-xs break-all text-fg-tertiary">
                        {fmt(c.from)}
                      </div>
                      <div className="font-mono text-xs break-all text-fg-primary">
                        <span className="text-accent mr-1">→</span>
                        {fmt(c.to)}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
            {paramChanges.length > 0 && (
              <section>
                <div className="caption mb-2">
                  {t('familySwitch.paramsSection')}
                </div>
                <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
                  {paramChanges.map((c) => (
                    <div key={c.field} className="contents">
                      <div className="font-medium text-fg-secondary">
                        {fieldLabel(c.field)}
                      </div>
                      <div className="text-fg-primary">
                        <span className="text-fg-tertiary">{fmt(c.from)}</span>
                        <span className="text-accent mx-1.5">→</span>
                        {fmt(c.to)}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>
        )}

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
            disabled={!preview || !!error}
            onClick={() => preview && onApply(preview.config)}
            className="btn btn-primary min-w-[96px] justify-center"
          >
            {t('familySwitch.ok')}
          </button>
        </div>
      </div>
    </div>
  )
}
