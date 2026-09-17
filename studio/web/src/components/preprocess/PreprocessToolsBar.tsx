import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'

export type PreprocessTool = 'overview' | 'dedupe' | 'upscale' | 'crop' | 'inpaint'

interface ToolDef {
  id: PreprocessTool
  /** i18n key suffix under `preprocess.tools.*`. */
  i18nKey: string
  /** Disabled in this milestone; tab is dim placeholder. */
  disabled?: boolean
}

/** Overview comes first — it's the gallery + multi-select + undo entry that
 *  governs the dataset, not a transform like upscale/crop/inpaint. */
const TOOLS: ReadonlyArray<ToolDef> = [
  { id: 'overview', i18nKey: 'overview' },
  { id: 'dedupe',   i18nKey: 'dedupe' },
  { id: 'upscale',  i18nKey: 'upscale' },
  { id: 'crop',     i18nKey: 'crop' },
  { id: 'inpaint',  i18nKey: 'inpaint' },
]

interface Props {
  current: PreprocessTool
  projectId: number
  versionId: number
}

/** 预处理工具切换条：贴 header 下方的全宽下划线 tab（对齐正则集页 belowHeader nav）。
 *
 *  工具（总览 / 去重 / 放大 / 裁剪 / 涂抹）是平级 **tool**、不是流水阶段——没有完成
 *  态、任意时刻任意顺序都能用。每个 tab 是路由 `<Link>`（`?tool=`），不是本地 state；
 *  inpaint 是占位禁用。URL 约定（ADR 0010）：
 *  `/projects/:pid/v/:vid/preprocess?tool=...`——query string 让 sidebar 的
 *  `/preprocess` matcher 保持简单，同时切换工具时父路由不卸载。
 *
 *  设计为直接塞进 StepShell 的 `belowHeader`（自带 `border-b px-6` 全宽条）。 */
export default function PreprocessToolsBar({ current, projectId, versionId }: Props) {
  const { t } = useTranslation()
  const base = `/projects/${projectId}/v/${versionId}/preprocess`
  return (
    <div className="px-6 pt-4 shrink-0">
    <nav className="seg">
      <span className="caption mx-2">
        {t('preprocess.toolsLabel')}
      </span>
      {TOOLS.map((tool) => {
        const label = t(`preprocess.tools.${tool.i18nKey}`)
        const isActive = tool.id === current
        if (tool.disabled) {
          return (
            <span
              key={tool.id}
              className="seg-item is-disabled"
              title={t(`preprocess.tools.${tool.i18nKey}Title`, { defaultValue: '' })}
            >{label}</span>
          )
        }
        if (isActive) {
          // Segmented active：白底 pill + 阴影（redesign 原型 Segmented）
          return (
            <span
              key={tool.id}
              className="seg-item is-active"
              aria-current="page"
            >{label}</span>
          )
        }
        // overview is the default tool (no ?tool= query); everyone else needs a tool param
        const href = tool.id === 'overview' ? base : `${base}?tool=${tool.id}`
        return (
          <Link
            key={tool.id}
            to={href}
            className="seg-item"
          >{label}</Link>
        )
      })}
    </nav>
    </div>
  )
}
