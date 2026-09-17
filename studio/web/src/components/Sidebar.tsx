import { Fragment, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { api, PHASE_ORDER, PHASE_SKIPPABLE, type Version, type VersionPhase, type VersionStatus } from '../api/client'
import { useSettingsDrawer } from '../lib/SettingsDrawer'
import { useToast } from './Toast'

/** ADR-0007 §11.2 / §11.5: cursor 派生 step 完成态。
 *
 * STEPS 顺序（ADR 0010 后）：0 download / 1 curate / 2 preprocess / 3 tag / 4 edit / 5 reg / 6 train
 * 项目级 ①：`download_image_count > 0` 派生
 * version 级 ②-⑦：`PHASE_ORDER.indexOf(STEP_KEY_TO_PHASE[key]) < cursorIdx`
 * （ADR 0010 把 preprocess 从 project scope 移到 version scope，curate 之后）
 */
const STEP_KEY_TO_PHASE: Record<string, VersionPhase> = {
  curate:     'curating',
  preprocess: 'preprocessing',
  edit:       'editing',
  reg:        'regularizing',
  train:      'ready',
}

const PHASE_TO_STEP_KEY: Record<VersionPhase, string> = {
  curating:      'curate',
  preprocessing: 'preprocess',
  editing:       'edit',
  regularizing:  'reg',
  ready:         'train',
}
import { useProjectCtx, useSelectedProject } from '../context/ProjectContext'
import type { ProjectCtxValue } from '../context/ProjectContext'

/** 侧边栏项目区数据源：在项目页内用 live ProjectContext（带版本管理回调，
 *  interactive=true）；离开后回退到只读粘性快照（interactive=false）。两者都
 *  没有 → null（不渲染项目区）。 */
type ProjectView =
  | (ProjectCtxValue & { interactive: true })
  | {
      project: ProjectCtxValue['project']
      activeVersion: ProjectCtxValue['activeVersion']
      interactive: false
    }
function useProjectView(): ProjectView | null {
  const live = useProjectCtx()
  const sticky = useSelectedProject()
  if (live) return { ...live, interactive: true }
  if (sticky) return { project: sticky.project, activeVersion: sticky.activeVersion, interactive: false }
  return null
}

// ── icons ──────────────────────────────────────────────────────────────────
const I = {
  folder:  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>,
  queue:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M4 6h16M4 12h10M4 18h16"/><circle cx="18" cy="12" r="2" fill="currentColor"/></svg>,
  preset:  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><line x1="6" y1="4" x2="6" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/><line x1="18" y1="4" x2="18" y2="20"/><circle cx="6" cy="9" r="2" fill="var(--bg-sunken)"/><circle cx="12" cy="15" r="2" fill="var(--bg-sunken)"/><circle cx="18" cy="7" r="2" fill="var(--bg-sunken)"/></svg>,
  monitor: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l4-6 4 3 5-9 5 7"/></svg>,
  cog:     <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>,
  image:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="16" height="16" rx="2"/><circle cx="9" cy="9" r="1.6" fill="currentColor"/><path d="m21 15-5-5L5 21"/></svg>,
  soup:    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 11h18a9 9 0 0 1-9 9 9 9 0 0 1-9-9Z"/><path d="M2 21h20"/><path d="M8 7c0-1.2 1-1.8 1-3"/><path d="M12 6c0-1.2 1-1.8 1-3"/><path d="M16 7c0-1.2 1-1.8 1-3"/></svg>,
  check:   <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="m4 12 5 5 11-12"/></svg>,
  chevL:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M15 6l-6 6 6 6"/></svg>,
  chevR:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M9 6l6 6-6 6"/></svg>,
  download:<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 4v12m0 0-4-4m4 4 4-4M4 20h16"/></svg>,
  upscale: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="13" width="8" height="8" rx="1"/><path d="M14 10V4h-6"/><path d="M21 3 14 10"/></svg>,
  filter:  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 5h18l-7 9v6l-4-2v-4z"/></svg>,
  tag:     <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M20 12 12 20l-9-9V3h8z"/><circle cx="7" cy="7" r="1.5" fill="currentColor"/></svg>,
  edit:    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 21h4l11-11-4-4L3 17z"/><path d="m14 5 4 4"/></svg>,
  reg:     <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><circle cx="17.5" cy="17.5" r="3.5"/></svg>,
  train:   <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 18 9 12l4 4 8-9"/><path d="M15 7h6v6"/></svg>,
  overview:<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg>,
  plus:    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M12 5v14M5 12h14"/></svg>,
  sun:     <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>,
  moon:    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>,
  branch:  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="6" r="2.4"/><circle cx="6" cy="18" r="2.4"/><circle cx="18" cy="8" r="2.4"/><path d="M6 8.4v7.2"/><path d="M18 10.4c0 3.4-3.2 3.9-6 4.4"/></svg>,
  swap:    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"><path d="m17 3 4 4-4 4"/><path d="M21 7H8"/><path d="M7 21l-4-4 4-4"/><path d="M3 17h13"/></svg>,
  trash:   <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18"/><path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/><path d="M6 6l1 14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-14"/></svg>,
}

// ── version status dot (ADR-0007 §11.3-B) ─────────────────────────────────
const STATUS_DOT: Record<VersionStatus, string> = {
  preparing: 'dot dot-warn',
  training:  'dot dot-running',
  completed: 'dot dot-ok',
  failed:    'dot dot-err',
  canceled:  'dot dot-neutral',
}

// ── nav item ───────────────────────────────────────────────────────────────
function navItemClass(active: boolean, collapsed: boolean, prominent: boolean): string {
  return [
    'flex w-full items-center gap-2.5 rounded-md no-underline transition-colors relative border-none cursor-pointer text-sm',
    collapsed ? 'h-8 px-0 justify-center' : prominent ? 'h-8 px-2.5 justify-start' : 'h-8 px-2.5 justify-start',
    active
      ? 'bg-surface text-fg-primary font-semibold shadow-sm ring-1 ring-[rgb(9_9_11/0.06)] [&>svg]:text-fg-primary'
      : 'bg-transparent text-fg-secondary font-medium hover:bg-overlay hover:text-fg-primary [&>svg]:text-fg-tertiary',
  ].join(' ')
}

function NavItem({ to, label, icon, active, collapsed, prominent = false }: {
  to: string; label: string; icon: React.ReactNode; active: boolean; collapsed: boolean
  /** 顶级 tab 用更大字号 + 更大 padding，跟项目下属 sub-nav 区分。 */
  prominent?: boolean
}) {
  return (
    <Link
      to={to}
      title={collapsed ? label : undefined}
      className={navItemClass(active, collapsed, prominent)}
    >
      {icon}
      {!collapsed && <span className="flex-1">{label}</span>}
    </Link>
  )
}

/** NavItem 的 button 变体 —— 给设置抽屉用，不走路由。 */
function NavButton({ onClick, label, icon, active, collapsed, prominent = false }: {
  onClick: () => void; label: string; icon: React.ReactNode; active: boolean; collapsed: boolean
  prominent?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={collapsed ? label : undefined}
      className={navItemClass(active, collapsed, prominent) + ' text-left'}
    >
      {icon}
      {!collapsed && <span className="flex-1">{label}</span>}
    </button>
  )
}

// ── project info block (项目名 + active version label，放最上) ──────────────
function ProjectInfoBlock({ collapsed }: { collapsed: boolean }) {
  const view = useProjectView()
  if (!view) return null
  if (collapsed) return null
  const { project } = view
  return (
    <div className="px-2.5 pt-2 pb-1.5">
      <div className="font-semibold text-fg-primary text-sm leading-5 overflow-hidden text-ellipsis whitespace-nowrap" title={project.title}>
        {project.title}
      </div>
      <div className="font-mono text-2xs text-fg-tertiary overflow-hidden text-ellipsis whitespace-nowrap" title={project.slug}>
        slug / {project.slug}
      </div>
    </div>
  )
}

// ── version picker block（⑂ 版本名 + action：切换 popover / 新建 / 导出 / 删除）─
// 层级锚点：这里是 project scope → version scope 的边界，正名为「版本选择器」
// （不再叫"训练版本"——训练是第 6 步 STEP）。离开项目页（只读粘性快照）时退化为
// 只读身份行、action 全部收起；要管理版本回到项目内。
//   常驻：⑂ 版本名 [⇄切换]（切换仅多版本时出现）
//   hover / 键盘 focus 才浮现：[＋新建] [⬇导出] [🗑删除]（删除仅多版本时出现）
function VersionPickerBlock({ collapsed }: { collapsed: boolean }) {
  const { t } = useTranslation()
  const view = useProjectView()
  const [switching, setSwitching] = useState(false)
  const rowRef = useRef<HTMLDivElement | null>(null)
  if (!view) return null
  if (collapsed) return null // 折叠态不显示（版本没有独立页面）
  const { project, activeVersion } = view
  const multiVersion = project.versions.length > 1

  const idIcon = <span className="w-5 h-5 rounded-[5px] bg-surface ring-1 ring-[rgb(9_9_11/0.08)] grid place-items-center text-fg-tertiary shrink-0 [&>svg]:w-3 [&>svg]:h-3">{I.branch}</span>
  const label = (
    <span
      className="flex-1 min-w-0 overflow-hidden text-ellipsis whitespace-nowrap font-mono text-fg-primary"
      title={activeVersion?.label}
    >
      {activeVersion?.label ?? '—'}
    </span>
  )

  // 只读（离开项目）：身份 + 状态点，无 action。
  if (!view.interactive) {
    return (
      <div className="flex items-center gap-2 rounded-md h-8 px-2.5 text-sm text-fg-secondary">
        {idIcon}
        {label}
        {activeVersion && <span className={STATUS_DOT[activeVersion.status] ?? 'dot dot-neutral'} />}
      </div>
    )
  }

  const { onSelectVersion, onCreateVersion, onExportTrain, onDeleteVersion, exporting } = view

  return (
    <div
      ref={rowRef}
      className="group relative flex items-center gap-2 rounded-md h-8 px-2.5 text-sm text-fg-secondary hover:bg-overlay transition-colors"
    >
      {idIcon}
      {label}

      <div className="flex items-center gap-0.5 shrink-0">
        {/* 新建 / 导出 / 删除：hover（或键盘 focus）才浮现。用 hidden→flex（而非
            opacity）让它们平时不占位，版本名 flex-1 得以伸展；hover 时才挤占空间。 */}
        <span className="hidden group-hover:flex group-focus-within:flex items-center gap-0.5">
          <VerAction icon={I.plus} title={t('sidebar.newVersion')} onClick={() => onCreateVersion()} />
          <VerAction icon={I.download} title={t('sidebar.exportTitle')} onClick={onExportTrain} disabled={exporting} />
          {multiVersion && (
            <VerAction
              icon={I.trash}
              title={t('sidebar.deleteVersionTitle')}
              danger
              onClick={() => { if (activeVersion) onDeleteVersion(activeVersion.id) }}
            />
          )}
        </span>
        {/* 切换：常驻（多版本才有意义）。 */}
        {multiVersion && (
          <VerAction
            icon={I.swap}
            title={t('sidebar.switchVersion')}
            active={switching}
            onClick={() => setSwitching((v) => !v)}
          />
        )}
      </div>

      {switching && (
        <VersionSwitchPopover
          versions={project.versions}
          activeId={project.active_version_id}
          anchorRef={rowRef}
          onPick={(vid) => { onSelectVersion(vid); setSwitching(false) }}
          onClose={() => setSwitching(false)}
        />
      )}
    </div>
  )
}

/** 版本行的 icon-only action 按钮。 */
function VerAction({ icon, title, onClick, disabled = false, active = false, danger = false }: {
  icon: React.ReactNode; title: string; onClick: () => void
  disabled?: boolean; active?: boolean; danger?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={[
        'w-6 h-6 grid place-items-center rounded-sm bg-transparent border-none transition-colors shrink-0',
        disabled ? 'opacity-40 cursor-default' : 'cursor-pointer',
        active ? 'text-accent bg-surface shadow-sm' : 'text-fg-tertiary',
        !disabled && !active ? (danger ? 'hover:bg-surface hover:text-err' : 'hover:bg-surface hover:text-fg-primary') : '',
      ].join(' ')}
    >
      {icon}
    </button>
  )
}

/** 切换版本 popover（照 ResumeFieldPicker 范式：点外 / Esc 关闭，anchor 到版本行）。 */
function VersionSwitchPopover({ versions, activeId, anchorRef, onPick, onClose }: {
  versions: Version[]
  activeId: number | null
  anchorRef: React.RefObject<HTMLElement | null>
  onPick: (vid: number) => void
  onClose: () => void
}) {
  const popRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (popRef.current?.contains(e.target as Node)) return
      if (anchorRef.current?.contains(e.target as Node)) return
      onClose()
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [onClose, anchorRef])

  return (
    <div
      ref={popRef}
      className="absolute z-40 left-0 right-0 top-full mt-1 max-h-[300px] overflow-y-auto rounded-lg border border-subtle bg-elevated shadow-lg p-1"
    >
      {versions.map((v) => {
        const isActive = v.id === activeId
        return (
          <button
            key={v.id}
            type="button"
            onClick={() => onPick(v.id)}
            className={[
              'w-full text-left px-2 h-8 rounded-[6px] font-mono text-xs flex items-center gap-2 border-none cursor-pointer transition-colors',
              isActive ? 'bg-overlay text-fg-primary font-medium' : 'text-fg-secondary bg-transparent hover:bg-overlay hover:text-fg-primary',
            ].join(' ')}
          >
            <span className={STATUS_DOT[v.status] ?? 'dot dot-neutral'} />
            <span className="flex-1 overflow-hidden text-ellipsis whitespace-nowrap">{v.label}</span>
            {isActive && <span className="text-accent shrink-0">{I.check}</span>}
          </button>
        )
      })}
    </div>
  )
}

// ── project stepper nav ────────────────────────────────────────────────────
function ProjectStepperNav({ pid, activeVid, currentStep, version, collapsed, inRoute }: {
  pid: string
  activeVid: string | null
  currentStep: string | null
  version: Version | null
  collapsed: boolean
  /** 是否当前正处于该项目的路由内（决定"概览"是否高亮；离开项目页只导航不高亮）。 */
  inRoute: boolean
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const ctx = useProjectCtx()

  // 项目级 ① 跟"概览"同款：圆盘里放 icon、无序号、无完成绿色态。
  // version 级 phase 重编号 1-6（每个 version 自己一段流水线，ADR 0010 加
  // preprocess phase 后是 6 个 version 级 step）。
  const STEPS = [
    { key: 'download',   labelKey: 'nav.download',   idx: '',  icon: I.download, scope: 'project' as const },
    { key: 'curate',     labelKey: 'nav.curate',     idx: '1', icon: I.filter,   scope: 'version' as const },
    { key: 'preprocess', labelKey: 'nav.preprocess', idx: '2', icon: I.upscale,  scope: 'version' as const },
    { key: 'edit',       labelKey: 'nav.tagEdit',    idx: '3', icon: I.edit,     scope: 'version' as const },
    { key: 'reg',        labelKey: 'nav.reg',        idx: '4', icon: I.reg,      scope: 'version' as const },
    { key: 'train',      labelKey: 'nav.train',      idx: '5', icon: I.train,    scope: 'version' as const },
  ]

  const overviewActive = inRoute && currentStep === null
  // ADR-0007 §11.5 cursor 派生：cursor 之前的 phase = done（仅 version 级）。
  // 项目级 ①② 不参与 cursor、不显示完成态。
  const cursorPhase: VersionPhase = (version?.phase as VersionPhase | undefined) ?? 'curating'
  const cursorIdx = PHASE_ORDER.indexOf(cursorPhase)

  const isStepDone = (key: string): boolean => {
    const phase = STEP_KEY_TO_PHASE[key]
    if (!phase) return false
    return PHASE_ORDER.indexOf(phase) < cursorIdx
  }

  // ADR-0007 §11.5-A: 推进 cursor 的唯一入口 —— 点击 cursor+1 那一行触发。
  // skippable 调 skip，必经调 advance；校验失败给 warning toast。
  const handleAdvanceToNext = async () => {
    if (!activeVid) return
    const nextIdx = cursorIdx + 1
    if (nextIdx >= PHASE_ORDER.length) return
    const isSkippable = PHASE_SKIPPABLE.includes(cursorPhase)
    try {
      const res = isSkippable
        ? await api.skipVersionPhase(Number(pid), Number(activeVid))
        : await api.advanceVersionPhase(Number(pid), Number(activeVid))
      // SSE 不可达时（如 Colab 代理）version_state_changed 不会推到前端，
      // cursor 会永远停在旧 phase——主动 reload 兜底，成功失败都要同步。
      await ctx?.reload()
      if (!res.ok) {
        toast(res.reason || t('sidebar.advanceFailed'), 'error')
        return
      }
      // 用后端返回的 new_phase 导航（而不是本地 cursorIdx+1 推算），防止
      // 本地 phase 已过期时跳错页。
      const landed = res.new_phase ?? PHASE_ORDER[nextIdx]
      navigate(`/projects/${pid}/v/${activeVid}/${PHASE_TO_STEP_KEY[landed]}`)
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const linkCls = (active: boolean, indent = false) => [
    'flex items-center gap-2.5 rounded-md h-8 text-sm no-underline transition-colors border-none',
    collapsed ? 'px-0 justify-center' : 'px-2.5 justify-start',
    !collapsed && indent ? 'ml-3' : '',
    active ? 'bg-surface text-fg-primary font-semibold shadow-sm ring-1 ring-[rgb(9_9_11/0.06)]' : 'bg-transparent text-fg-secondary font-normal hover:bg-overlay hover:text-fg-primary',
  ].join(' ')

  return (
    <div className="flex flex-col gap-px">
      {/* 项目名 + 当前 version label，放最顶 */}
      <ProjectInfoBlock collapsed={collapsed} />

      <Link
        to={`/projects/${pid}`}
        title={collapsed ? t('nav.overview') : undefined}
        className={linkCls(overviewActive)}
      >
        <span className={`w-5 h-5 rounded-[5px] grid place-items-center shrink-0 ${overviewActive ? 'bg-accent-soft text-accent' : 'bg-overlay text-fg-tertiary'}`}>
          {I.overview}
        </span>
        {!collapsed && <span className="flex-1">{t('nav.overview')}</span>}
      </Link>

      {STEPS.map((s, i) => {
        const label = t(s.labelKey)
        const isActive = s.key === currentStep
        const isProject = s.scope === 'project'
        const phase = STEP_KEY_TO_PHASE[s.key]
        const phaseIdx = phase != null ? PHASE_ORDER.indexOf(phase) : -1
        // ADR-0007 §11.5-A: sidebar 是 cursor 推进主通道。
        // - phase_idx < cursorIdx  → done (绿数字，Link)
        // - phase_idx == cursorIdx → 当前 cursor (Link)
        // - phase_idx == cursorIdx+1 → "下一步"，呼吸 button，点击触发 advance/skip
        // - phase_idx > cursorIdx+1 → disabled
        const isCursorCurrent = !isProject && phaseIdx === cursorIdx
        const isNextStep = !isProject && phaseIdx === cursorIdx + 1
        const isFuturePhase = !isProject && phaseIdx > cursorIdx + 1
        const isDone = isProject ? false : isStepDone(s.key)

        const href = (isFuturePhase || isNextStep)
          ? null
          : isProject
            ? `/projects/${pid}/${s.key}`
            : activeVid ? `/projects/${pid}/v/${activeVid}/${s.key}` : null

        // 视觉强度：cursor 当前（实心 accent）> cursor+1（弱底 + 静态 ring）> 已完成（绿）> 默认
        // 注：之前 cursor+1 用 animate-pulse 反而盖过 cursor 当前，去掉动画；改用 ring 静态提示。
        const badgeCls = isDone
          ? 'bg-ok-soft text-ok'
          : isCursorCurrent
            ? 'bg-accent text-accent-fg'
            : isNextStep
              ? 'bg-surface text-accent ring-1 ring-inset ring-accent'
              : isActive
                ? 'bg-accent-soft text-accent'
                : 'bg-overlay text-fg-tertiary'

        // 项目级 ①② badge 内放 icon（跟"概览" ≡ 同款），不要数字 / 不要绿色态。
        // version 级 badge 始终放数字；完成时数字变绿；cursor+1 时背景 accent + 呼吸。
        const badgeContent = isProject ? s.icon : s.idx

        const inner = (
          <>
            <span className={`w-5 h-5 rounded-[5px] grid place-items-center text-[11px] font-semibold tabular-nums shrink-0 [&>svg]:w-3 [&>svg]:h-3 ${badgeCls}`}>
              {badgeContent}
            </span>
            {!collapsed && <span className="flex-1 text-left">{label}</span>}
            {!collapsed && isActive && <span className="dot dot-running" />}
          </>
        )

        // version 级 step（筛选→训练）整体再缩进一层，表达从属于上方的 version 选择器。
        const indent = s.scope === 'version'

        let stepNode: React.ReactNode
        if (isNextStep) {
          // 推进入口：button + onClick
          stepNode = (
            <button
              key={s.key}
              type="button"
              onClick={() => void handleAdvanceToNext()}
              title={collapsed ? `→ ${label}` : t('sidebar.advanceToTitle', { label })}
              className={linkCls(false, indent) + ' cursor-pointer text-fg-primary font-medium'}
            >
              {inner}
            </button>
          )
        } else if (!href) {
          stepNode = (
            <span key={s.key} title={collapsed ? (s.idx ? `${s.idx}. ${label}` : label) : undefined}
              className={linkCls(false, indent) + ' opacity-40 cursor-default'}>
              {inner}
            </span>
          )
        } else {
          stepNode = (
            <Link
              key={s.key}
              to={href}
              title={collapsed ? (s.idx ? `${s.idx}. ${label}` : label) : undefined}
              className={linkCls(isActive, indent)}
            >
              {inner}
            </Link>
          )
        }

        // 在 scope 从 project 切到 version 的边界（"预处理"和"筛选"之间）
        // 插入 VersionPickerBlock —— 版本选择紧靠 version 级 phase 上方。
        const prev = STEPS[i - 1]
        const isBoundary = prev && prev.scope === 'project' && s.scope === 'version'
        if (!isBoundary) return stepNode
        return (
          <Fragment key={s.key}>
            <VersionPickerBlock collapsed={collapsed} />
            {stepNode}
          </Fragment>
        )
      })}
    </div>
  )
}

// ── sidebar ────────────────────────────────────────────────────────────────
const SIDEBAR_KEY = 'studio.sidebar.expanded'

/** Props are phone-only. On desktop `mobile` is false and every branch below
 *  collapses to exactly what the component rendered before. */
export default function Sidebar({
  mobile = false,
  mobileOpen = false,
  onMobileClose,
}: {
  mobile?: boolean
  mobileOpen?: boolean
  onMobileClose?: () => void
} = {}) {
  const { t } = useTranslation()
  const location = useLocation()
  const view = useProjectView()
  const settingsDrawer = useSettingsDrawer()

  // 路由内的 pid（用于步骤高亮）；离开项目页后回退到粘性快照的项目 id，让项目区
  // 跨页保留用于导航。
  const routePid = location.pathname.match(/^\/projects\/([^/]+)/)?.[1] ?? null
  const urlVid = location.pathname.match(/\/v\/([^/]+)/)?.[1] ?? null
  const stepMatch = location.pathname.match(/\/v\/[^/]+\/([^/]+)$/)
  // ADR 0010: preprocess 从 project scope 移到 version scope；project scope 只剩 download
  const projectScopeStep = location.pathname.match(/^\/projects\/[^/]+\/(download)$/)?.[1] ?? null
  const currentStep = stepMatch?.[1] ?? projectScopeStep

  const inRoute = routePid !== null
  const pid = routePid ?? view?.project.id?.toString() ?? null
  const activeVid = view?.activeVersion?.id?.toString() ?? urlVid

  // 有项目（路由内 live 或粘性快照）就展示项目区
  const inProject = view !== null && pid !== null

  const [expandedOverride, setExpandedOverride] = useState<boolean | null>(() => {
    try {
      const v = sessionStorage.getItem(SIDEBAR_KEY)
      return v === '1' ? true : v === '0' ? false : null
    } catch { return null }
  })

  const expanded = mobile ? true : (expandedOverride ?? true)
  const collapsed = !expanded

  const toggle = () => {
    const next = !expanded
    setExpandedOverride(next)
    try { sessionStorage.setItem(SIDEBAR_KEY, next ? '1' : '0') } catch { /* ignore */ }
  }

  const isMain = (path: string) => {
    if (path === '/') return location.pathname === '/'
    return location.pathname.startsWith(path)
  }

  // On a phone the rail is always full width (a 58px icon rail plus a drawer is
  // the worst of both) and slides in over the content instead of squeezing it.
  const asideClass = mobile
    ? 'fixed inset-y-0 left-0 z-50 w-[min(86vw,var(--sidebar-w))] bg-sunken border-r border-subtle flex flex-col overflow-hidden shadow-xl transition-transform duration-200 ease-out'
      + (mobileOpen ? ' translate-x-0' : ' -translate-x-full')
    : 'shrink-0 bg-sunken flex flex-col overflow-hidden h-full transition-[width] duration-[160ms] ease-in-out'

  return (
    <aside
      className={asideClass}
      style={mobile ? undefined : { width: collapsed ? 'var(--sidebar-collapsed-w)' : 'var(--sidebar-w)' }}
      aria-hidden={mobile && !mobileOpen ? true : undefined}
      // `inert` keeps the off-screen drawer out of the tab order. React 18's
      // typings predate the attribute, so it goes in as a spread.
      {...((mobile && !mobileOpen ? { inert: '' } : {}) as Record<string, unknown>)}
    >
      {!mobile && (
      <div
        className={`flex items-center gap-2.5 shrink-0 ${collapsed ? 'justify-center px-2' : 'px-[22px]'}`}
        style={{ height: 'var(--topbar-h)' }}
      >
        <span className="w-6 h-6 rounded-[7px] bg-fg-primary text-fg-inverse grid place-items-center shrink-0 shadow-sm" aria-hidden>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"><path d="M5 20 12 4l7 16"/><path d="M8.2 13h7.6"/></svg>
        </span>
        {!collapsed && (
          <span className="text-sm font-semibold tracking-tight text-fg-primary whitespace-nowrap">
            Anima <span className="text-fg-tertiary font-medium">Studio</span>
          </span>
        )}
      </div>
      )}
      {mobile && (
        <div className="flex items-center justify-between px-3 h-[var(--topbar-h)] border-b border-subtle shrink-0">
          <span className="text-sm font-semibold">{t('nav.projects')}</span>
          <button
            onClick={onMobileClose}
            aria-label={t('common.close')}
            className="w-10 h-10 -mr-1 flex items-center justify-center rounded-md text-fg-secondary hover:bg-overlay"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}
      <nav style={{ scrollbarGutter: 'auto' }} className={`flex-1 flex flex-col gap-0.5 overflow-y-auto overflow-x-hidden ${collapsed ? 'px-2 pb-2' : 'px-3 pb-3.5'}`}>
        <NavItem to="/" label={t('nav.projects')} icon={I.folder} active={!inProject && location.pathname === '/'} collapsed={collapsed} />

        {/* 当前项目下的全部内容（概览 + ①② + VersionPanel + ③-⑦）夹在 项目 / 队列 之间。
            sub-nav 性质，缩进表达从属（折叠态不缩进）。
            VersionPanel 由 ProjectStepperNav 在 project→version scope 切换点插入。 */}
        {inProject && pid && (
          <div className={`flex flex-col gap-0.5 ${collapsed ? '' : 'ml-3'}`}>
            <ProjectStepperNav pid={pid} activeVid={activeVid} currentStep={currentStep} version={view?.activeVersion ?? null} collapsed={collapsed} inRoute={inRoute} />
          </div>
        )}

        <NavItem to="/queue" label={t('nav.queue')} icon={I.queue} active={isMain('/queue')} collapsed={collapsed} />
        <NavItem to="/tools/generate" label={t('nav.generate')} icon={I.image} active={isMain('/tools/generate')} collapsed={collapsed} />
        <NavItem to="/tools/soup" label={t('nav.soup')} icon={I.soup} active={isMain('/tools/soup')} collapsed={collapsed} />
      </nav>

      <div className={`border-t border-subtle flex flex-col gap-0.5 shrink-0 ${collapsed ? 'px-2 py-2' : 'px-3 py-2.5'}`}>
        <NavItem to="/tools/presets" label={t('nav.presets')} icon={I.preset} active={isMain('/tools/presets')} collapsed={collapsed} />
        <NavItem to="/tools/monitor" label={t('nav.monitor')} icon={I.monitor} active={isMain('/tools/monitor')} collapsed={collapsed} />
        <NavButton
          onClick={() => settingsDrawer.isOpen ? void settingsDrawer.close() : settingsDrawer.open()}
          label={t('nav.settings')}
          icon={I.cog}
          active={false}
          collapsed={collapsed}
        />
        <button
          onClick={toggle}
          title={collapsed ? t('sidebar.expand') : t('sidebar.collapse')}
          className={`text-fg-tertiary bg-transparent border-none rounded-md cursor-pointer hover:bg-overlay hover:text-fg-secondary transition-colors h-8 ${collapsed ? 'flex items-center justify-center' : 'flex items-center gap-1.5 px-2.5 text-xs font-medium'}`}
        >
          {collapsed ? I.chevR : <>{I.chevL}<span>{t('sidebar.collapseLabel')}</span></>}
        </button>
      </div>
    </aside>
  )
}
