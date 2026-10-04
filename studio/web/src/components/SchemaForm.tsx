import { Fragment, useEffect, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import type { SchemaResponse, ConfigData } from '../api/client'
import { controlKind, evalShowWhen, schemaAltDescription, schemaDisableHint, schemaDescription, schemaGroupLabel } from '../lib/schema'
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
  /** Fields kept in the config but not rendered (e.g. model paths while they
   * are synced from global Settings). */
  hiddenFields?: string[]
}

/** Schema groups shown as several cards. Titles: schema.subgroups.<key>.
 *  `rest` takes the group's fields no card lists (new optimizer / scheduler
 *  options land there); `switchesFirst` puts the card's switches above the
 *  line, for cards whose switch turns the rest on (EMA, DOP). */
interface Subgroup { key: string; fields: string[]; rest?: boolean; switchesFirst?: boolean }
const rank = (list: string[], f: string) => { const i = list.indexOf(f); return i < 0 ? list.length : i }
const SUBGROUPS: Record<string, Subgroup[]> = {
  lora: [
    { key: 'loraMain', fields: ['lora_type', 'lora_rank', 'lora_alpha', 'lokr_factor', 'tlora_min_rank', 'tlora_alpha_rank_scale', 'tlora_use_ortho'], rest: true },
    { key: 'loraDropout', fields: ['lora_dropout', 'lora_rank_dropout', 'lora_module_dropout'] },
    { key: 'loraExtra', fields: ['lora_dora', 'lora_rs'] },
    { key: 'loraBlocks', fields: ['lora_reg_dims'] },
  ],
  dataset: [
    { key: 'datasetMain', fields: ['resolution', 'aspect_ratio_limit', 'bucket_min_reso', 'bucket_max_reso', 'bucket_step'], rest: true },
    { key: 'regSet', fields: ['reg_caption', 'reg_weight'] },
  ],
  training: [
    { key: 'trainMain', fields: ['epochs', 'max_steps', 'batch_size', 'grad_accum', 'grad_checkpoint'] },
    { key: 'trainLr', fields: ['learning_rate', 'lr_scheduler', 'lr_scheduler_eta_min', 'optimizer_type'], rest: true },
    { key: 'trainEma', fields: ['ema_enabled', 'ema_decay', 'ema_start_ratio'], switchesFirst: true },
  ],
  loss: [
    { key: 'lossMain', fields: [], rest: true },
    { key: 'lossDop', fields: ['dop_enabled', 'dop_weight', 'dop_ratio'], switchesFirst: true },
  ],
  system: [
    { key: 'sysMain', fields: ['mixed_precision', 'attention_backend', 'num_workers', 'cache_latents', 'block_swap_preflight', 'cache_encode_tiled', 'cache_encode_tile_px', 'cache_encode_tile_overlap', 'cache_encode_max_pixels'] },
    { key: 'sysOpt', fields: ['vae_cache_batch_size', 'blocks_to_swap', 'vae_tiling', 'navit_packing', 'kv_trim'], rest: true },
  ],
  output: [
    { key: 'outMain', fields: ['save_every_epochs', 'save_every_steps', 'save_state_every_epochs', 'save_state_every_steps', 'seed'], rest: true },
    { key: 'outResume', fields: ['resume_lora', 'resume_state'] },
  ],
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
  groupKeys, fieldFilter, baseline, afterField, emptyHint, hiddenFields,
}: Props) {
  const { t } = useTranslation()
  const disabledSet = new Set(disabledFields ?? [])
  const hiddenSet = new Set(hiddenFields ?? [])
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
    // Simplified AdEMAMix sums raw gradients, so its step is ~1/(1-β1) times AdamW's:
    // an AdamW-scale lr (1e-4) diverges; the equivalent at β1=0.99 is 1e-6.
    if (
      triggerField === 'optimizer_type' && next.optimizer_type === 'simplified_ademamix'
      && Number(next.learning_rate) > 1e-5
    ) {
      writes.push({
        field: 'learning_rate', from: next.learning_rate, to: 1e-6,
        reason: t('ruleImpact.simplifiedAdemamixLr'),
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
    if (prop.hidden || hiddenSet.has(name)) continue
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
    // A group can be split into several cards (SUBGROUPS); fields not listed
    // there land in the first card. The first card carries the group anchor.
    const subs = SUBGROUPS[key]
    const restIdx = subs ? Math.max(0, subs.findIndex((x) => x.rest)) : 0
    const cards: Array<{ key: string; title: string; fields: string[]; switchesFirst?: boolean }> = subs
      ? subs.map((s, i) => ({
          key: s.key,
          title: t(`schema.subgroups.${s.key}`),
          switchesFirst: s.switchesFirst,
          fields: fields.filter((f) => s.fields.includes(f)
            || (i === restIdx && !subs.some((x) => x.fields.includes(f))))
            // listed fields in the listed order, the rest after them in schema order
            .sort((a, b) => rank(s.fields, a) - rank(s.fields, b)),
        })).filter((c) => c.fields.length > 0)
      : [{ key, title: groupLabel, fields }]
    return (
      <Fragment key={key}>
        {cards.map((c, ci) => (
          <section key={c.key} id={ci === 0 ? `schema-group-${key}` : undefined} className="ds-fgroup">
            <div className="ds-fgroup-head">
              <span className="ds-fgroup-title">{c.title}</span>
            </div>
            {renderGrid(c.fields, c.switchesFirst)}
          </section>
        ))}
      </Fragment>
    )
  })

  // Inputs go into the grid; switches get their own compact list, split from
  // the inputs by a full-width line: under them, or above them when the switch
  // turns the card on. Rows attached to a switch (afterField) join the grid.
  function renderGrid(fields: string[], switchesFirst = false) {
    const isBool = (n: string) => controlKind(props[n]) === 'bool'
    const inputs = fields.filter((n) => !isBool(n))
    const bools = fields.filter(isBool)
    const attached = bools.filter((n) => afterField?.[n])
    const hasGrid = inputs.length > 0 || attached.length > 0
    const grid = hasGrid && (
      <div className="ds-fgrid">
        {inputs.map((n) => renderField(n))}
        {attached.map((n) => <div key={`after-${n}`} className="ds-fcell ds-full">{afterField![n]}</div>)}
      </div>
    )
    const switches = bools.length > 0 && (
      <div className={`ds-switches${hasGrid ? (switchesFirst ? ' ds-before' : ' ds-after') : ''}`}>
        {bools.map((n) => renderField(n, true))}
      </div>
    )
    return switchesFirst ? <>{switches}{grid}</> : <>{grid}{switches}</>
  }

  function renderField(name: string, inSwitchList = false) {
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
            <Fragment key={name}>
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
              {!inSwitchList && afterField?.[name] && <div className="ds-fcell ds-full">{afterField[name]}</div>}
            </Fragment>
          )
  }

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
