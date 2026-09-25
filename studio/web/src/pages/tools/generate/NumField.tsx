import { useTranslation } from 'react-i18next'
import FieldLabel from '../../../components/ds/FieldLabel'

/** 带 cap 标题的数值 stepper（mockup：输入框 + 右侧「+」按钮；seed 用「随机」按钮）。 */
export default function NumField({ label, tip, value, onChange, min, max, step, onRandom }: {
  label: string
  tip?: string
  value: number
  onChange: (v: number) => void
  min?: number; max?: number; step?: number
  /** 传入时右侧按钮变成「随机」（seed）。 */
  onRandom?: () => void
}) {
  const { t } = useTranslation()
  const bump = () => {
    const s = step ?? 1
    let next = Math.round((value + s) / s) * s
    if (max != null) next = Math.min(max, next)
    if (min != null) next = Math.max(min, next)
    onChange(Number(next.toFixed(6)))
  }
  return (
    <div style={{ minWidth: 0 }}>
      <div className="ds-cap" style={{ marginBottom: 6 }}>
        <FieldLabel label={label} tip={tip} />
      </div>
      <span className="ds-stepper">
        <input
          type="number"
          aria-label={label}
          min={min} max={max} step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        {onRandom ? (
          <button type="button" aria-label={t('generate.randomSeed')} title={t('generate.randomSeed')} onClick={onRandom}>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 4v4h4" /></svg>
          </button>
        ) : (
          <button type="button" aria-label={t('field.more')} onClick={bump}>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>
          </button>
        )}
      </span>
    </div>
  )
}
