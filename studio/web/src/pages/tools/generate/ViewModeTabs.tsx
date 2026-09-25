import { useTranslation } from 'react-i18next'

/** 单图 / XY 矩阵 模式切换（页头 segmented control）。
 *  compare 是 xy 内部 sub-view（选 2 张自动进入），不在此列出。 */
export type ViewMode = 'single' | 'xy'

export default function ViewModeTabs({
  mode, onModeChange,
}: {
  mode: ViewMode
  onModeChange: (m: ViewMode) => void
}) {
  const { t } = useTranslation()
  const tab = (m: ViewMode, label: string) => (
    <button
      type="button"
      aria-pressed={mode === m}
      onClick={() => onModeChange(m)}
      className={`ds-seg-item${mode === m ? ' ds-is-active' : ''}`}
    >
      {label}
    </button>
  )
  return (
    <div className="ds-seg">
      {tab('single', t('generate.singleMode'))}
      {tab('xy', t('generate.xyMode'))}
    </div>
  )
}
