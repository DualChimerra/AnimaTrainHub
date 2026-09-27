import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { SchemaProperty } from '../api/client'
import { useProjectCtx } from '../context/ProjectContext'
import { controlKind, schemaEnumLabel, schemaFieldLabel } from '../lib/schema'
import { useAutoGrowTextarea } from '../lib/useAutoGrowTextarea'
import FieldLabel from './ds/FieldLabel'
import PathPicker from './PathPicker'
import ResumeFieldPicker from './ResumeFieldPicker'

interface Props {
  name: string
  prop: SchemaProperty
  value: unknown
  onChange: (v: unknown) => void
  /** disabled state (auto-controlled fields render grayed-out readonly). */
  disabled?: boolean
  /** Small badge after the field label (e.g. "auto · global setting" /
   * "auto · project setting"). Decoupled from disabled: a field can stay
   * editable with just an informational badge attached, or the badge can pair
   * with disabled to say "this field is auto-filled and you can't change it".
   * Supports ReactNode so it can embed a clickable link (e.g. jumping to the
   * matching Settings section). */
  hint?: React.ReactNode
  /** Overrides prop.description's help text (used for conditional contextual descriptions). */
  descriptionOverride?: string
  /** Extra button slot on the right of a path field (e.g. "reset to global
   * default"). Only rendered for string/path fields; other field types ignore it. */
  suffix?: React.ReactNode
  /** Visible option override for select (the enum subset after option_show_when
   * filtering, computed by SchemaForm from the current values). Defaults to
   * rendering the full prop.enum. */
  enumOptions?: unknown[]
  /** Options matched by option_disable_when (D4: grayed out and unselectable,
   * not hidden), computed by SchemaForm from the current values; title shows
   * disabledOptionHint explaining why it can't be selected. */
  disabledEnumOptions?: string[]
  disabledOptionHint?: string
  /** Value the field had before this session's edits; set only when it
   *  differs, and shown as a green dot plus "was: …" under the control. */
  changedFrom?: { value: unknown }
}

function formatWas(v: unknown): string {
  if (v === null || v === undefined || v === '') return '—'
  if (Array.isArray(v)) return v.join(', ')
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

/** One setting row as in the mockup (.field): name, parameter key and badges,
 *  and the control on the right. The explanation opens on hover over the name.
 *  Multi-line controls take the full width under the text instead (stack). */
function FieldShell({
  name, label, hintNode, helpNode, changedFrom, stack, children, below,
}: {
  name: string
  label: string
  hintNode: React.ReactNode
  helpNode: React.ReactNode
  changedFrom?: { value: unknown }
  stack?: boolean
  children: React.ReactNode
  /** Rendered after the row (pickers anchored to it). */
  below?: React.ReactNode
}) {
  const { t } = useTranslation()
  return (
    <div className={`ds-field${stack ? ' ds-stack' : ''}`} data-field={name} style={{ position: 'relative' }}>
      <div className="ds-field-txt">
        <div className="ds-field-name">
          {changedFrom && <span className="ds-changed-dot" aria-hidden="true" />}
          {/* The config key is for reference, not for reading: it sits in the
              hover tip under the explanation instead of crowding the row. */}
          <FieldLabel label={label} tip={<>{helpNode}<span className="ds-tip-key">{name}</span></>} />
          {hintNode}
        </div>
      </div>
      <div className="ds-field-ctl">
        {children}
        {changedFrom && <span className="ds-ctl-note">{t('field.was', { value: formatWas(changedFrom.value) })}</span>}
      </div>
      {below}
    </div>
  )
}

/** Select fields rendered as option cards; value = i18n prefix of the
 *  one-line description under each option. */
const CARD_FIELDS: Record<string, string> = {
  model_family: 'field.familyDesc',
}

const LockIcon = (
  <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" aria-hidden="true">
    <rect x="5" y="11" width="14" height="10" rx="2" /><path d="M8 11V7a4 4 0 0 1 8 0v4" />
  </svg>
)

/** A single form field, rendered by dispatching on the control kind. */
export default function Field({
  name, prop, value, onChange, disabled = false, hint, descriptionOverride, suffix,
  enumOptions, disabledEnumOptions, disabledOptionHint, changedFrom,
}: Props) {
  const { t } = useTranslation()
  const kind = controlKind(prop)
  const label = schemaFieldLabel(name, t)
  const rawHelp = descriptionOverride ?? prop.description
  // A description that only repeats the name adds nothing to the hover tip.
  const help = rawHelp && rawHelp.trim().toLowerCase() !== label.trim().toLowerCase() ? rawHelp : undefined
  // Recommended value lives in the locale files (schema.recommend.<field>) and is
  // shown in the hover tip after the description. It is advice, not a constraint: nothing reads it
  // back, so a field without one simply renders as before.
  const recommend = t(`schema.recommend.${name}`, { defaultValue: '' })
  const helpNode = (help || recommend) ? (
    <>
      {help && <div>{help}</div>}
      {recommend && (
        <div className="ds-field-rec">
          <span style={{ opacity: 0.75 }}>{t('field.recommended')}: </span>{recommend}
        </div>
      )}
    </>
  ) : null
  const hintText = hint ?? (disabled ? t('field.autoProject') : null)
  const hintNode = hintText
    ? disabled
      ? <span className="ds-lockchip">{LockIcon}{hintText}</span>
      : <span className="ds-tag ds-auto">{hintText}</span>
    : null
  const shell = { name, label, hintNode, helpNode, changedFrom }

  // bool ----------------------------------------------------------------
  if (kind === 'bool') {
    const on = Boolean(value)
    return (
      <FieldShell {...shell}>
        <label className={`ds-switch${on ? ' ds-on' : ''}`} style={disabled ? { opacity: 0.5, cursor: 'not-allowed' } : { cursor: 'pointer' }}>
          <input
            type="checkbox"
            className="sr-only"
            checked={on}
            onChange={(e) => onChange(e.target.checked)}
            disabled={disabled}
            aria-label={label}
          />
          <i />
        </label>
      </FieldShell>
    )
  }

  // tristate (Optional[bool]: null / true / false) ----------------------
  if (kind === 'tristate') {
    const triValue = value === true ? 'true' : value === false ? 'false' : ''
    return (
      <FieldShell {...shell}>
        <select
          value={triValue}
          onChange={(e) => {
            const v = e.target.value
            onChange(v === 'true' ? true : v === 'false' ? false : null)
          }}
          disabled={disabled}
          className="ds-inp"
          aria-label={label}
        >
          <option value="">{t('field.useGlobal')}</option>
          <option value="true">{t('field.yes')}</option>
          <option value="false">{t('field.no')}</option>
        </select>
      </FieldShell>
    )
  }

  // select shown as option cards (few choices that change a lot) ----------
  const cards = kind === 'select' ? CARD_FIELDS[name] : undefined
  if (cards) {
    const opts = (enumOptions ?? prop.enum ?? []).map(String)
    const cur = String(value ?? '')
    return (
      <FieldShell {...shell} stack>
        <div className="ds-optcards ds-compact" role="radiogroup" aria-label={label} style={{ gridTemplateColumns: `repeat(${Math.min(opts.length, 3)}, minmax(0, 1fr))` }}>
          {opts.map((opt) => {
            const on = opt === cur
            const optDisabled = disabled || (disabledEnumOptions?.includes(opt) && !on)
            return (
              <button
                key={opt}
                type="button"
                role="radio"
                aria-checked={on}
                disabled={optDisabled}
                title={optDisabled && !disabled ? disabledOptionHint : undefined}
                className={`ds-optcard${on ? ' ds-is-on' : ''}`}
                style={optDisabled ? { opacity: 0.5, cursor: 'not-allowed' } : undefined}
                onClick={() => { if (!on) onChange(opt) }}
              >
                <span className={`ds-radio${on ? ' ds-on' : ''}`} aria-hidden="true" />
                <span className="ds-optcard-txt">
                  <span className="ds-optcard-name">{schemaEnumLabel(name, opt, t)}</span>
                  <span className="ds-optcard-desc">{t(`${cards}.${opt}`, { defaultValue: '' })}</span>
                </span>
              </button>
            )
          })}
        </div>
      </FieldShell>
    )
  }

  // select --------------------------------------------------------------
  if (kind === 'select') {
    return (
      <FieldShell {...shell}>
        <select
          value={String(value ?? '')}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className="ds-inp"
          aria-label={label}
        >
          {(enumOptions ?? prop.enum ?? []).map((opt) => {
            // The currently selected value stays rendered as selectable even if
            // disabled (the form should reflect the real config; illegal
            // combinations are reported as backend validation errors, not
            // silently cleared in the UI)
            const optDisabled =
              disabledEnumOptions?.includes(String(opt)) &&
              String(opt) !== String(value ?? '')
            return (
              <option
                key={String(opt)}
                value={String(opt)}
                disabled={optDisabled}
                title={optDisabled ? disabledOptionHint : undefined}
              >
                {schemaEnumLabel(name, opt, t)}
              </option>
            )
          })}
        </select>
      </FieldShell>
    )
  }

  // textarea ------------------------------------------------------------
  if (kind === 'textarea') {
    return (
      <FieldShell {...shell} stack>
        <TextareaField value={value} onChange={onChange} disabled={disabled} label={label} />
      </FieldShell>
    )
  }

  // string-list ---------------------------------------------------------
  if (kind === 'string-list') {
    return (
      <FieldShell {...shell} stack>
        <StringListField value={value} onChange={onChange} disabled={disabled} label={label} />
        <span className="ds-ctl-note">{t('field.multilineHint')}</span>
      </FieldShell>
    )
  }

  // int-list (e.g. resolution: [512, 768, 1024]) -----------------------
  if (kind === 'int-list') {
    return (
      <FieldShell {...shell}>
        <IntListField value={value} defaultValue={prop.default} onChange={onChange} disabled={disabled} label={label} />
      </FieldShell>
    )
  }

  // float-list (e.g. per-cycle learning rates) -------------------------
  if (kind === 'float-list') {
    return (
      <FieldShell {...shell}>
        <FloatListField value={value} defaultValue={prop.default} onChange={onChange} disabled={disabled} label={label} />
      </FieldShell>
    )
  }

  // code ----------------------------------------------------------------
  if (kind === 'code') {
    return (
      <FieldShell {...shell} stack>
        <JsonCodeField value={value} onChange={onChange} disabled={disabled} label={label} />
      </FieldShell>
    )
  }

  // int / float ---------------------------------------------------------
  if (kind === 'int' || kind === 'float') {
    return (
      <FieldShell {...shell}>
        <NumberField
          kind={kind}
          value={value}
          defaultValue={prop.default}
          minimum={prop.minimum}
          maximum={prop.maximum}
          onChange={onChange}
          disabled={disabled}
          label={label}
        />
      </FieldShell>
    )
  }

  // string / path -------------------------------------------------------
  return (
    <PathStringField
      shell={shell}
      name={name}
      kind={kind}
      value={value}
      onChange={onChange}
      disabled={disabled}
      suffix={suffix}
      label={label}
    />
  )
}

interface TextareaFieldProps {
  value: unknown
  onChange: (v: unknown) => void
  disabled?: boolean
  label: string
}

const areaStyle = (disabled: boolean): React.CSSProperties => ({
  height: 'auto', padding: '8px 11px', lineHeight: 1.55, resize: 'none', overflow: 'hidden',
  ...(disabled ? { opacity: 0.55, cursor: 'not-allowed' } : null),
})

function TextareaField({ value, onChange, disabled = false, label }: TextareaFieldProps) {
  const taRef = useRef<HTMLTextAreaElement>(null)
  const text = String(value ?? '')
  useAutoGrowTextarea(taRef, text)
  return (
    <textarea
      ref={taRef}
      rows={5}
      value={text}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      aria-label={label}
      className="ds-inp ds-mono"
      style={areaStyle(disabled)}
    />
  )
}

/** String-list input (one entry per line). The textarea display goes through a
 *  local raw buffer: if the controlled value echoed back straight from
 *  join('\n'), a freshly typed newline (a trailing blank line) would get eaten
 *  by split+filter and the cursor couldn't move to a new line. raw keeps the
 *  user's original input; the parsed array is still synced to the parent on
 *  every keystroke, and raw gets normalized (blank lines / leading-trailing
 *  whitespace stripped) on blur. */
function StringListField({ value, onChange, disabled = false, label }: TextareaFieldProps) {
  const joined = Array.isArray(value) ? (value as string[]).join('\n') : ''
  const [raw, setRaw] = useState<string>(joined)
  const taRef = useRef<HTMLTextAreaElement>(null)
  useAutoGrowTextarea(taRef, raw)

  useEffect(() => {
    if (document.activeElement !== taRef.current) setRaw(joined)
  }, [joined])

  const parse = (text: string) =>
    text.split('\n').map((s) => s.trim()).filter((s) => s.length > 0)

  return (
    <textarea
      ref={taRef}
      rows={5}
      value={raw}
      onChange={(e) => {
        setRaw(e.target.value)
        onChange(parse(e.target.value))
      }}
      onBlur={() => setRaw(parse(raw).join('\n'))}
      disabled={disabled}
      aria-label={label}
      className="ds-inp ds-mono"
      style={areaStyle(disabled)}
    />
  )
}

function formatJsonCode(v: unknown): string {
  if (v === null || v === undefined || v === '') return ''
  if (typeof v === 'string') return v
  return JSON.stringify(v, null, 2)
}

function JsonCodeField({ value, onChange, disabled = false, label }: TextareaFieldProps) {
  const [raw, setRaw] = useState<string>(() => formatJsonCode(value))
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    if (document.activeElement !== inputRef.current) {
      setRaw(formatJsonCode(value))
      setError(null)
    }
  }, [value])

  const commit = () => {
    const text = raw.trim()
    if (text === '') {
      onChange(null)
      setError(null)
      return
    }
    try {
      const parsed = JSON.parse(text) as unknown
      if (parsed === null || typeof parsed !== 'object') {
        setError('JSON must be an object or array')
        return
      }
      onChange(parsed)
      setRaw(JSON.stringify(parsed, null, 2))
      setError(null)
    } catch {
      setError('Invalid JSON')
    }
  }

  return (
    <>
      <textarea
        ref={inputRef}
        rows={Math.max(3, raw.split('\n').length + 1)}
        value={raw}
        onChange={(e) => {
          setRaw(e.target.value)
          setError(null)
        }}
        onBlur={commit}
        disabled={disabled}
        aria-label={label}
        className="ds-inp ds-mono"
        style={{ ...areaStyle(disabled), overflow: 'auto' }}
      />
      {error && <span className="ds-ctl-note" style={{ color: 'var(--red-text)' }}>{error}</span>}
    </>
  )
}

interface ListFieldProps {
  value: unknown
  defaultValue: unknown
  onChange: (v: unknown) => void
  disabled?: boolean
  label: string
}

/** Integer-list input (e.g. resolution: [512, 768, 1024]). Comma or
 *  space-separated; the backend validator handles snap/clamp, the frontend
 *  just collects numbers. Falls back to the default value when cleared
 *  (consistent with NumberField). */
function IntListField({ value, defaultValue, onChange, disabled = false, label }: ListFieldProps) {
  const fmt = (v: unknown) =>
    Array.isArray(v) ? (v as number[]).join(', ') : v === null || v === undefined ? '' : String(v)
  const [raw, setRaw] = useState<string>(() => fmt(value))
  const inputRef = useRef<HTMLInputElement | null>(null)
  // placeholder is derived purely from this field's default (a generic component, no field-specific value hardcoded)
  const placeholder = fmt(defaultValue)

  useEffect(() => {
    if (document.activeElement !== inputRef.current) setRaw(fmt(value))
  }, [value])

  const commit = () => {
    const nums = raw
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map((s) => parseInt(s, 10))
      .filter((n) => Number.isFinite(n))
    // Cleared -> fall back to the default value (avoids storing an empty list)
    if (nums.length === 0 && Array.isArray(defaultValue)) {
      onChange(defaultValue)
      setRaw(fmt(defaultValue))
      return
    }
    onChange(nums)
    setRaw(nums.join(', '))
  }

  return (
    <input
      ref={inputRef}
      type="text"
      inputMode="numeric"
      value={raw}
      onChange={(e) => setRaw(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault()
          commit()
        }
      }}
      disabled={disabled}
      aria-label={label}
      className="ds-inp ds-mono"
      placeholder={placeholder}
    />
  )
}

/** Decimal-number list. Scientific notation is accepted, which is important
 * for LR values such as 1e-4 and 5e-5. */
function FloatListField({ value, defaultValue, onChange, disabled = false, label }: ListFieldProps) {
  const fmt = (v: unknown) =>
    Array.isArray(v) ? (v as number[]).join(', ') : v === null || v === undefined ? '' : String(v)
  const [raw, setRaw] = useState<string>(() => fmt(value))
  const inputRef = useRef<HTMLInputElement | null>(null)
  const placeholder = fmt(defaultValue)

  useEffect(() => {
    if (document.activeElement !== inputRef.current) setRaw(fmt(value))
  }, [value])

  const commit = () => {
    const nums = raw
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map((s) => Number(s))
      .filter((n) => Number.isFinite(n))
    if (nums.length === 0) {
      const fallback = Array.isArray(defaultValue) ? defaultValue : []
      onChange(fallback)
      setRaw(fmt(fallback))
      return
    }
    onChange(nums)
    setRaw(nums.join(', '))
  }

  return (
    <input
      ref={inputRef}
      type="text"
      inputMode="decimal"
      value={raw}
      onChange={(e) => setRaw(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault()
          commit()
        }
      }}
      disabled={disabled}
      aria-label={label}
      className="ds-inp ds-mono"
      placeholder={placeholder}
    />
  )
}

interface NumberFieldProps {
  kind: 'int' | 'float'
  value: unknown
  defaultValue: unknown
  minimum?: number
  maximum?: number
  onChange: (v: unknown) => void
  disabled?: boolean
  label: string
}

/** Number input that commits on blur / Enter (typing "0.05" must not be cut
 *  short at "0."). Integers get the mockup's − / + stepper. */
function NumberField({
  kind, value, defaultValue, minimum, maximum, onChange, disabled = false, label,
}: NumberFieldProps) {
  const { t } = useTranslation()
  const formatNum = (v: unknown) =>
    v === null || v === undefined ? '' : String(v)
  const [raw, setRaw] = useState<string>(() => formatNum(value))
  const inputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    if (document.activeElement !== inputRef.current) {
      setRaw(formatNum(value))
    }
  }, [value])

  const commit = () => {
    if (raw === '') {
      onChange(defaultValue)
      setRaw(formatNum(defaultValue))
      return
    }
    const num = kind === 'int' ? parseInt(raw, 10) : parseFloat(raw)
    if (Number.isNaN(num)) {
      setRaw(formatNum(value))
      return
    }
    if (
      (minimum !== undefined && num < minimum) ||
      (maximum !== undefined && num > maximum)
    ) {
      setRaw(formatNum(value))
      return
    }
    onChange(num)
    setRaw(formatNum(num))
  }

  const bump = (d: number) => {
    const base = parseInt(raw, 10)
    let next = (Number.isNaN(base) ? Number(defaultValue) || 0 : base) + d
    if (minimum !== undefined) next = Math.max(minimum, next)
    if (maximum !== undefined) next = Math.min(maximum, next)
    onChange(next)
    setRaw(String(next))
  }

  const input = (
    <input
      ref={inputRef}
      type="text"
      inputMode={kind === 'int' ? 'numeric' : 'decimal'}
      value={raw}
      onChange={(e) => setRaw(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault()
          commit()
        }
      }}
      disabled={disabled}
      aria-label={label}
      className={kind === 'int' ? undefined : 'ds-inp ds-mono'}
    />
  )
  if (kind !== 'int') return input
  return (
    <span className="ds-stepper" style={disabled ? { opacity: 0.55 } : undefined}>
      {input}
      <button type="button" aria-label={t('field.less')} onClick={() => bump(-1)} disabled={disabled}>
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M5 12h14" /></svg>
      </button>
      <button type="button" aria-label={t('field.more')} onClick={() => bump(1)} disabled={disabled}>
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>
      </button>
    </span>
  )
}

interface PathFieldProps {
  shell: {
    name: string
    label: string
    hintNode: React.ReactNode
    helpNode: React.ReactNode
    changedFrom?: { value: unknown }
  }
  /** Schema field name, lets a path field decide whether to use a dedicated picker (resume_state / resume_lora). */
  name: string
  kind: 'path' | 'string'
  value: unknown
  onChange: (v: unknown) => void
  disabled?: boolean
  /** Extra button slot on the right of the input row (e.g. a reset button). */
  suffix?: React.ReactNode
  label: string
}

function PathStringField({
  shell, name, kind, value, onChange, disabled = false, suffix, label,
}: PathFieldProps) {
  const { t } = useTranslation()
  const [picking, setPicking] = useState(false)
  const text = value === null || value === undefined ? '' : String(value)
  const browseBtnRef = useRef<HTMLButtonElement | null>(null)
  const projectCtx = useProjectCtx()

  // resume_state / resume_lora: use the in-project semantic picker (dropdown),
  // so the user never sees the deep path. For external files, the user just
  // types the path into the input directly.
  const resumeKind: 'state' | 'lora' | null =
    name === 'resume_state' ? 'state' :
    name === 'resume_lora' ? 'lora' : null
  const useResumePicker = kind === 'path' && resumeKind !== null && projectCtx !== null

  return (
    <FieldShell
      {...shell}
      // Paths are long: full width under the name, not squeezed into the side column.
      stack={kind === 'path'}
      below={
        <>
          {/* resume_state / resume_lora: a dropdown anchored to the field, listing files grouped by version */}
          {useResumePicker && picking && !disabled && (
            <ResumeFieldPicker
              pid={projectCtx!.project.id}
              kind={resumeKind!}
              value={text}
              onChange={onChange as (v: string) => void}
              onClose={() => setPicking(false)}
              anchorRef={browseBtnRef}
            />
          )}
          {/* Other path fields: keep the PathPicker modal (external model paths and similar cases) */}
          {!useResumePicker && picking && !disabled && (
            <PathPicker
              initialPath={text || undefined}
              onPick={(p) => {
                onChange(p)
                setPicking(false)
              }}
              onClose={() => setPicking(false)}
            />
          )}
        </>
      }
    >
      {kind === 'path' ? (
        // input + folder button on one row, as in the mockup
        <span style={{ display: 'flex', gap: 6, width: '100%' }}>
          <input
            type="text"
            value={text}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
            aria-label={label}
            title={text || undefined}
            className="ds-inp ds-code"
            style={{ flex: 1 }}
          />
          <button
            ref={browseBtnRef}
            type="button"
            onClick={() => setPicking((p) => !p)}
            disabled={disabled}
            className="ds-iconbtn"
            aria-label={useResumePicker ? t('field.browseProject') : t('field.browse')}
            title={useResumePicker ? t('field.browseProject') : t('field.browse')}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /></svg>
          </button>
        </span>
      ) : (
        <input
          type="text"
          value={text}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          aria-label={label}
          title={text || undefined}
          className="ds-inp"
        />
      )}
      {suffix && (
        <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>{suffix}</span>
      )}
    </FieldShell>
  )
}
