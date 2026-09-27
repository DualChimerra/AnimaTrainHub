import { useEffect, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import type { SchemaResponse, ConfigData } from '../api/client'
import { evalShowWhen, schemaAltDescription, schemaDisableHint, schemaDescription, schemaGroupLabel } from '../lib/schema'
import Field from './Field'
import RuleImpactDialog, { type RuleImpactChange } from './RuleImpactDialog'

interface Props {
  schema: SchemaResponse
  values: ConfigData
  onChange: (values: ConfigData) => void
  /** These field names render as readonly / disabled (project-specific / global control). */
  disabledFields?: string[]
  /** Badge for each disabled field; defaults to Field's built-in "auto - project controlled".
   * Supports ReactNode so a clickable link can be embedded (e.g. jump to the matching Settings section). */
  disabledHints?: Record<string, React.ReactNode>
  /** A field isn't disabled but still gets a badge (e.g. "auto - project setting" meaning the project pre-filled it,
   * but the user can still edit it). Priority: disabledHints > autoHints. */
  autoHints?: Record<string, React.ReactNode>
  /** Extra button slot to the right of a field (e.g. "reset to global default"). Only applies to string/path fields;
   * looked up by field name. */
  fieldSuffixes?: Record<string, React.ReactNode>
  /** false = hide fields marked advanced. The app always shows everything. */
  advancedMode?: boolean
  /** Render only these schema groups (a tab of the Train page). */
  groupKeys?: string[]
  /** Extra per-field filter (changed only / non-default / locked). */
  fieldFilter?: (name: string) => boolean
  /** Values before this session's edits; fields that differ get marked. */
  baseline?: ConfigData | null
  /** Rows rendered right after a field (e.g. the trigger word after DOP). */
  afterField?: Record<string, ReactNode>
  /** Shown when the filters leave nothing to render. */
  emptyHint?: string
}

/** Computes which groups have at least one visible field under the current advancedMode (used for sidebar anchor navigation).
 * Kept consistent with the buckets logic below: skips hidden / skips advanced (in simple mode).
 * Doesn't account for show_when -- that's per-field and dynamic; the section header still renders by bucket. */
export function visibleSchemaGroups(
  schema: SchemaResponse,
  advancedMode: boolean,
): Array<{ key: string; label: string }> {
  const counts = new Map<string, number>()
  for (const [, prop] of Object.entries(schema.schema.properties)) {
    if (prop.hidden) continue
    if (prop.advanced && !advancedMode) continue
    const g = prop.group ?? 'misc'
    counts.set(g, (counts.get(g) ?? 0) + 1)
  }
  return schema.groups
    .filter((g) => (counts.get(g.key) ?? 0) > 0)
    .map((g) => ({ key: g.key, label: g.label }))
}

/**
 * Renders the form partitioned by schema.groups: each group gets a small heading (name - key - field count), followed
 * by the field rows (mockup .subgroup / .field). show_when uses evalShowWhen for conditional display,
 * depending on the current values.
 */
export default function SchemaForm({
  schema, values, onChange, disabledFields, disabledHints, autoHints, fieldSuffixes, advancedMode = true,
  groupKeys, fieldFilter, baseline, afterField, emptyHint,
}: Props) {
  const { t } = useTranslation()
  const disabledSet = new Set(disabledFields ?? [])
  const dHints = disabledHints ?? {}
  const aHints = autoHints ?? {}
  const suffixes = fieldSuffixes ?? {}
  const props = schema.schema.properties
  const shouldDisableField = (prop: typeof props[string]) =>
    !!prop.disable_when && evalShowWhen(prop.disable_when, values)
  const takeoverValueForField = (prop: typeof props[string]) =>
    prop.disable_value ?? prop.default

  /** Pending change for the R6 confirmation dialog (renders RuleImpactDialog when non-null). */
  const [pendingImpact, setPendingImpact] = useState<{
    trigger: { field: string; from: unknown; to: unknown }
    next: ConfigData
    writes: RuleImpactChange[]
  } | null>(null)

  /** Which fields a rule will write once a change lands (a lossy list: only included if target value != current value).
   * Sources: (1) disable_when takeover (resets to disable_value / default);
   * (2) advisory rewrite -- switching to automagic suggests learning_rate 1e-6
   * (matches upstream ostris/ai-toolkit + diffusion-pipe defaults; starting at AdamW-scale lr would make
   * sign-agreement's adaptation converge ~100x slower). */
  const computeRuleWrites = (
    next: ConfigData, triggerField: string,
  ): RuleImpactChange[] => {
    const writes: RuleImpactChange[] = []
    for (const [name, prop] of Object.entries(props)) {
      if (!prop.disable_when || !evalShowWhen(prop.disable_when, next)) continue
      const target = takeoverValueForField(prop)
      if (target !== undefined && next[name] !== target) {
        writes.push({
          field: name, from: next[name], to: target,
          reason: schemaDisableHint(name, prop.disable_hint, t),
        })
      }
    }
    if (
      triggerField === 'optimizer_type' && next.optimizer_type === 'automagic'
      && Number(next.learning_rate) > 1e-5
    ) {
      writes.push({
        field: 'learning_rate', from: next.learning_rate, to: 1e-6,
        reason: t('ruleImpact.automagicLr'),
      })
    }
    return writes
  }

  // setField entry point interception (R6, D6): a violating state never enters form state -- a lossy chained
  // change pops a confirmation first (confirm = trigger the change + commit every chained write in one go;
  // cancel = nothing happens), a lossless one (the chained target is already pinned to that value) applies silently.
  // model_family never appears in any disable_when, family switching is still intercepted by the caller (Train/Presets' onFormChange) into FamilySwitchDialog.
  const setField = (name: string, v: unknown) => {
    const next = { ...values, [name]: v }
    const writes = computeRuleWrites(next, name)
    if (writes.length === 0) {
      onChange(next)
      return
    }
    setPendingImpact({
      trigger: { field: name, from: values[name], to: v },
      next,
      writes,
    })
  }

  // Fallback silent takeover: a violating state brought in via an external write path (config load / disabledFields
  // change) is corrected directly, no dialog (the user didn't trigger an action, so a dialog would have nothing to
  // "cancel"). The normal interaction path never reaches here once intercepted by setField. Mutual exclusion on an
  // old config that has both set is handled by the backend's _tolerant_validate gate-first fix + the defaulted_fields banner.
  useEffect(() => {
    let nextValues = values
    let changed = false
    for (const [name, prop] of Object.entries(props)) {
      if (!shouldDisableField(prop)) continue
      const takeoverValue = takeoverValueForField(prop)
      if (takeoverValue !== undefined && values[name] !== takeoverValue) {
        nextValues = { ...nextValues, [name]: takeoverValue }
        changed = true
      }
    }
    if (changed) onChange(nextValues)
    // Intentionally only watches values; onChange / props references are stable, adding them would cause an infinite loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values])

  // Bucketed by group. Fields with hidden=true are skipped outright: the value still passes through via ConfigData
  // (not lost on PUT), it's just not rendered in the UI. If every field in a group is hidden, `fields.length === 0`
  // below makes the whole section disappear automatically.
  const buckets = new Map<string, string[]>()
  for (const [name, prop] of Object.entries(props)) {
    if (prop.hidden) continue
    if (prop.advanced && !advancedMode) continue
    const g = prop.group ?? 'misc'
    if (!buckets.has(g)) buckets.set(g, [])
    buckets.get(g)!.push(name)
  }
  const groupSet = groupKeys ? new Set(groupKeys) : null

  let rendered = 0
  const sections = schema.groups.map(({ key, label }) => {
    if (groupSet && !groupSet.has(key)) return null
    const groupLabel = schemaGroupLabel(key, label, t)
    const fields = (buckets.get(key) ?? []).filter((name) =>
      evalShowWhen(props[name].show_when, values) && (!fieldFilter || fieldFilter(name))
    )
    if (fields.length === 0) return null
    rendered += fields.length
    return (
      <section key={key} id={`schema-group-${key}`} className="ds-fgroup">
        <div className="ds-fgroup-head">
          <span className="ds-fgroup-title">{groupLabel}</span>
          <span className="ds-fgroup-count">{t('schema.fieldCount', { n: fields.length, count: fields.length })}</span>
        </div>
        {fields.map((name) => {
          const prop = props[name]
          // disable_when (schema-driven conditional disable, e.g. Prodigy -> lr_scheduler)
          // has lower priority than the global disabledFields (project pre-fill).
          const conditionallyDisabled = shouldDisableField(prop)
          const isDisabled =
            disabledSet.has(name) || conditionallyDisabled
          const hint = disabledSet.has(name)
            ? dHints[name]
            : conditionallyDisabled
              ? schemaDisableHint(name, prop.disable_hint, t)
              : aHints[name]
          const descriptionOverride =
            prop.alt_description_when &&
            evalShowWhen(prop.alt_description_when, values)
              ? schemaAltDescription(name, prop.alt_description, t)
              : schemaDescription(name, prop.description, t)
          // option_show_when: filters the dropdown options by the current values (multi-model P4-2).
          // The currently selected value is kept even if gated out -- the form should reflect the config as-is,
          // cross-family values are reported by backend validation, not silently dropped from the UI.
          const gates = prop.option_show_when
          const enumOptions = gates
            ? (prop.enum ?? []).filter(
                (opt) =>
                  evalShowWhen(gates[String(opt)], values) ||
                  String(opt) === String(values[name] ?? '')
              )
            : undefined
          // option_disable_when: matched options are grayed out and unselectable (D4: not hidden,
          // the user can see why it's disabled -- the title shows disable_hint).
          const dGates = prop.option_disable_when
          const disabledEnumOptions = dGates
            ? Object.keys(dGates).filter((opt) =>
                evalShowWhen(dGates[opt], values)
              )
            : undefined
          const changed = baseline && name in baseline && !sameValue(baseline[name], values[name])
          return (
            <div key={name}>
              <Field
                name={name}
                prop={prop}
                value={values[name]}
                onChange={(v) => setField(name, v)}
                disabled={isDisabled}
                hint={hint}
                descriptionOverride={descriptionOverride}
                suffix={suffixes[name]}
                enumOptions={enumOptions}
                disabledEnumOptions={disabledEnumOptions}
                disabledOptionHint={schemaDisableHint(name, prop.disable_hint, t)}
                changedFrom={changed ? { value: baseline![name] } : undefined}
              />
              {afterField?.[name]}
            </div>
          )
        })}
      </section>
    )
  })

  return (
    <div className="ds-fgroups">
      {sections}
      {rendered === 0 && emptyHint && (
        <div className="ds-empty">{emptyHint}</div>
      )}
      {pendingImpact && (
        <RuleImpactDialog
          trigger={pendingImpact.trigger}
          changes={pendingImpact.writes}
          onApply={() => {
            const applied = { ...pendingImpact.next }
            for (const w of pendingImpact.writes) applied[w.field] = w.to
            setPendingImpact(null)
            onChange(applied)
          }}
          onCancel={() => setPendingImpact(null)}
        />
      )}
    </div>
  )
}

/** Value equality for config fields (numbers, strings, lists, objects). */
export function sameValue(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (a == null && b == null) return true
  if (typeof a === 'object' || typeof b === 'object') return JSON.stringify(a) === JSON.stringify(b)
  return false
}
