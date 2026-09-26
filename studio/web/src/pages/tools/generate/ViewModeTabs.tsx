import { useTranslation } from 'react-i18next'

/** Single-image / XY-grid mode switch (the page header's segmented control).
 *  compare is an internal xy sub-view (entered automatically when 2 are selected), not listed here. */
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
