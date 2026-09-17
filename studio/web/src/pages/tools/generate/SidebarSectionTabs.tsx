import { useTranslation } from 'react-i18next'
import type { ViewMode } from './ViewModeTabs'

export type SidebarTab = 'lora' | 'prompts' | 'config'

/** 左侧配置区分页 segmented 控件：LoRA/XY · 提示词 · 配置。
 *
 * 三块原本竖向堆叠靠滚动浏览，改成三页平铺，控件放进底部 footer（「开始生成」上方）。
 * 分段控件而非独立按钮：sunken 轨道把三段一起包住（未选中也有容器），等宽
 * （flex-1）；选中段用 surface 底 + 细边 + 轻阴影抬起，**刻意不复用 btn-primary
 * 橙色**，免得跟正下方的橙色「开始生成」按钮撞脸误认。第一段跟随 mode：
 * single → LoRA，xy → XY。 */
export default function SidebarSectionTabs({
  tab, onTabChange, mode,
}: {
  tab: SidebarTab
  onTabChange: (t: SidebarTab) => void
  mode: ViewMode
}) {
  const { t } = useTranslation()
  const seg = (key: SidebarTab, label: string) => {
    const active = tab === key
    return (
      <button
        onClick={() => onTabChange(key)}
        aria-selected={active}
        className={`seg-item flex-1 min-w-0 min-h-[28px] ${active ? 'is-active' : ''}`}
      >
        {label}
      </button>
    )
  }
  return (
    <div
      role="tablist"
      className="seg flex w-full"
    >
      {seg('lora', mode === 'single' ? 'LoRA' : 'XY')}
      {seg('prompts', t('generate.prompts'))}
      {seg('config', t('generate.samplingParams'))}
    </div>
  )
}
