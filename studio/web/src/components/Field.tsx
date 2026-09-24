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
  /** disabled 状态（自动控制字段灰显 readonly）。 */
  disabled?: boolean
  /** 字段标签后的小徽章（如「自动 · 全局设置」/「自动 · 项目设置」）。
   * 与 disabled 解耦：可以让字段保持可编辑只挂个徽章作信息提示，也可以
   * 配合 disabled 来表达「这字段被自动填且不让你改」。支持 ReactNode 以便
   * 嵌入可点击链接（如跳转到 Settings 对应区段）。 */
  hint?: React.ReactNode
  /** 覆盖 prop.description 的说明文字（用于条件上下文描述）。 */
  descriptionOverride?: string
  /** path 字段右侧额外按钮槽（如「↺ 重置为全局默认」）。仅对 string/path
   * 字段渲染；其他类型字段忽略。 */
  suffix?: React.ReactNode
  /** select 的可见选项覆盖（option_show_when 过滤后的 enum 子集，由
   * SchemaForm 按当前 values 计算）。缺省渲染 prop.enum 全量。 */
  enumOptions?: unknown[]
  /** option_disable_when 命中的选项（D4：灰显不可选、不隐藏），由 SchemaForm
   * 按当前 values 计算；title 显示 disabledOptionHint 解释为什么不可选。 */
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
          <FieldLabel label={label} tip={helpNode} />
          <span className="ds-key">{name}</span>
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

const LockIcon = (
  <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" aria-hidden="true">
    <rect x="5" y="11" width="14" height="10" rx="2" /><path d="M8 11V7a4 4 0 0 1 8 0v4" />
  </svg>
)

/** 单个表单字段，按 control kind 分发渲染。 */
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
            // 当前已选中的值即使被禁也保持可选中状态渲染（表单如实反映
            // config；非法组合由后端校验报错，不在 UI 里凭空清值）
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

/** 字符串列表输入（每行一条）。textarea 显示走本地 raw 缓冲：受控值若直接用
 *  join('\n') 回显，刚敲的换行（尾部空行）会被 split+filter 吃掉、光标换不了
 *  行。raw 保留用户原始输入，解析后的数组仍每次击键同步给父级，blur 时把
 *  raw 归一化（去空行 / 首尾空白）。 */
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

/** 整数列表输入（如 resolution: [512, 768, 1024]）。逗号或空格分隔；后端 validator
 *  负责 snap/clamp，前端只收集数字。清空后回落到默认值（与 NumberField 一致）。 */
function IntListField({ value, defaultValue, onChange, disabled = false, label }: ListFieldProps) {
  const fmt = (v: unknown) =>
    Array.isArray(v) ? (v as number[]).join(', ') : v === null || v === undefined ? '' : String(v)
  const [raw, setRaw] = useState<string>(() => fmt(value))
  const inputRef = useRef<HTMLInputElement | null>(null)
  // placeholder 纯由该字段的 default 派生（通用组件，不写死任何字段专属值）
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
    // 清空 → 回落默认值（避免存空列表）
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
  /** schema 字段名，让 path 字段判定是否走专用 picker（resume_state / resume_lora）。 */
  name: string
  kind: 'path' | 'string'
  value: unknown
  onChange: (v: unknown) => void
  disabled?: boolean
  /** 输入行右侧额外按钮槽（如重置按钮）。 */
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

  // resume_state / resume_lora：走项目内语义 picker（dropdown），用户看不到深路径。
  // 外部文件用户直接在 input 手填即可。
  const resumeKind: 'state' | 'lora' | null =
    name === 'resume_state' ? 'state' :
    name === 'resume_lora' ? 'lora' : null
  const useResumePicker = kind === 'path' && resumeKind !== null && projectCtx !== null

  return (
    <FieldShell
      {...shell}
      below={
        <>
          {/* resume_state / resume_lora：贴字段的 dropdown，按 version 分组列文件 */}
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
          {/* 其它 path 字段：保留 PathPicker 模态框（外部模型路径等场景） */}
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
      <input
        type="text"
        value={text}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        aria-label={label}
        title={text || undefined}
        className={'ds-inp' + (kind === 'path' ? ' ds-mono' : '')}
      />
      {(kind === 'path' || suffix) && (
        <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          {kind === 'path' && (
            <button
              ref={browseBtnRef}
              type="button"
              onClick={() => setPicking((p) => !p)}
              disabled={disabled}
              className="ds-ctl ds-ghost"
              style={{ height: 24 }}
            >
              {useResumePicker ? t('field.browseProject') : t('field.browse')}
            </button>
          )}
          {suffix}
        </span>
      )}
    </FieldShell>
  )
}
