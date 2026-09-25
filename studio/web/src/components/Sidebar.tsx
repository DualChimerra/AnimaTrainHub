import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { api, PHASE_ORDER, PHASE_SKIPPABLE, type Task, type Version, type VersionPhase, type VersionStatus } from '../api/client'
import { splitOptional } from '../lib/labels'
import { useEventStream, type StudioEvent } from '../lib/useEventStream'
import { useMonitorProgress } from '../lib/useMonitorProgress'
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

// ── icons (mockup sizes: 15px nav icons, 1.7 stroke) ────────────────────────
const svg = (d: React.ReactNode, size = 15, w = 1.7) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={w} strokeLinecap="round" strokeLinejoin="round">{d}</svg>
)
const I = {
  folder:   svg(<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />),
  folderTile: svg(<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />, 15, 1.9),
  queue:    svg(<><path d="M4 6h16M4 12h10M4 18h16" /><circle cx="18" cy="12" r="2" fill="currentColor" /></>),
  preset:   svg(<><path d="M6 4v16M12 4v16M18 4v16" /><circle cx="6" cy="9" r="2" fill="#fff" /><circle cx="12" cy="15" r="2" fill="#fff" /><circle cx="18" cy="7" r="2" fill="#fff" /></>),
  monitor:  svg(<path d="M3 17l4-6 4 3 5-9 5 7" />),
  cog:      svg(<><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33 1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82 1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" /></>),
  image:    svg(<><rect x="3" y="3" width="16" height="16" rx="2" /><circle cx="9" cy="9" r="1.6" fill="currentColor" /><path d="m21 15-5-5L5 21" /></>),
  soup:     svg(<><path d="M3 11h18a9 9 0 0 1-9 9 9 9 0 0 1-9-9Z" /><path d="M2 21h20" /><path d="M8 7c0-1.2 1-1.8 1-3" /><path d="M12 6c0-1.2 1-1.8 1-3" /><path d="M16 7c0-1.2 1-1.8 1-3" /></>),
  overview: svg(<><rect x="3" y="3" width="7" height="9" rx="1" /><rect x="14" y="3" width="7" height="5" rx="1" /><rect x="14" y="12" width="7" height="9" rx="1" /><rect x="3" y="16" width="7" height="5" rx="1" /></>),
  download: svg(<path d="M12 4v12m0 0-4-4m4 4 4-4M4 20h16" />),
  branch:   svg(<><circle cx="6" cy="6" r="2.4" /><circle cx="6" cy="18" r="2.4" /><circle cx="18" cy="8" r="2.4" /><path d="M6 8.4v7.2" /><path d="M18 10.4c0 3.4-3.2 3.9-6 4.4" /></>, 11, 2),
  chevD:    svg(<path d="m6 9 6 6 6-6" />, 11, 2.2),
  check:    svg(<path d="m5 13 4 4L19 7" />, 9, 3.4),
  checkSm:  svg(<path d="m4 12 5 5 11-12" />, 12, 2.4),
  collapse: svg(<><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M10 4v16" /><path d="M6.7 10.4 5.1 12l1.6 1.6" /></>, 15, 1.8),
  expand:   svg(<><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M10 4v16" /><path d="M5.1 10.4 6.7 12l-1.6 1.6" /></>, 15, 1.8),
  plus:     svg(<path d="M12 5v14M5 12h14" />, 13, 2),
  export:   svg(<path d="M12 4v12m0 0-4-4m4 4 4-4M4 20h16" />, 13, 1.8),
  trash:    svg(<><path d="M3 6h18" /><path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" /><path d="M6 6l1 14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-14" /></>, 13, 1.8),
  updown:   <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#878e89" strokeWidth="2" strokeLinecap="round"><path d="m8 9 4-4 4 4M8 15l4 4 4-4" /></svg>,
  brand:    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"><path d="M5 20 12 4l7 16" /><path d="M8.2 13h7.6" /></svg>,
}

// ── version status dot (ADR-0007 §11.3-B) ─────────────────────────────────
const STATUS_DOT: Record<VersionStatus, string> = {
  preparing: 'ds-dot ds-dot-warn',
  training:  'ds-dot ds-dot-run',
  completed: 'ds-dot ds-dot-ok',
  failed:    'ds-dot ds-dot-err',
  canceled:  'ds-dot ds-dot-mute',
}

// ── nav item ───────────────────────────────────────────────────────────────
function navCls(active: boolean, collapsed: boolean): string {
  return ['ds-nav-item', active ? 'ds-is-active' : '', collapsed ? 'justify-center' : ''].filter(Boolean).join(' ')
}

function NavItem({ to, label, icon, active, collapsed, tail }: {
  to: string; label: string; icon: React.ReactNode; active: boolean; collapsed: boolean
  /** Right-aligned count, as in the mockup (Датасет 532, Очередь 2). */
  tail?: React.ReactNode
}) {
  return (
    <Link to={to} title={collapsed ? label : undefined} aria-label={collapsed ? label : undefined} className={navCls(active, collapsed)}>
      {icon}
      {!collapsed && <span>{label}</span>}
      {!collapsed && tail != null && <span className="ds-nav-tail">{tail}</span>}
    </Link>
  )
}

/** NavItem 的 button 变体 —— 给设置抽屉用，不走路由。 */
function NavButton({ onClick, label, icon, active, collapsed }: {
  onClick: () => void; label: string; icon: React.ReactNode; active: boolean; collapsed: boolean
}) {
  return (
    <button type="button" onClick={onClick} title={collapsed ? label : undefined} aria-label={collapsed ? label : undefined} className={navCls(active, collapsed)}>
      {icon}
      {!collapsed && <span>{label}</span>}
    </button>
  )
}

// ── version bar（ВЕРСИЯ + pill；hover 才浮现 新建 / 导出 / 删除）────────────
// 层级锚点：这里是 project scope → version scope 的边界。离开项目页（只读粘性
// 快照）时退化为只读 pill、action 全部收起；要管理版本回到项目内。
//   pill：多版本时点击打开切换 popover（title=切换版本）
//   hover / 键盘 focus 才浮现：[＋新建] [⬇导出] [🗑删除]（删除仅多版本时出现）
function VersionBar({ collapsed }: { collapsed: boolean }) {
  const { t } = useTranslation()
  const view = useProjectView()
  const [switching, setSwitching] = useState(false)
  const rowRef = useRef<HTMLDivElement | null>(null)
  if (!view) return null
  if (collapsed) return null // 折叠态不显示（版本没有独立页面）
  const { project, activeVersion } = view
  const multiVersion = project.versions.length > 1
  const canSwitch = view.interactive && multiVersion

  const pillInner = (
    <>
      {I.branch}
      <span className="ds-nm" title={activeVersion?.label}>{activeVersion?.label ?? '—'}</span>
      {activeVersion && <span className={STATUS_DOT[activeVersion.status] ?? 'ds-dot ds-dot-mute'} />}
      {canSwitch && I.chevD}
    </>
  )

  return (
    <div ref={rowRef} className="ds-verbar group relative">
      <span className="ds-cap">{t('sidebar.versionsLabel')}</span>
      {canSwitch ? (
        <button type="button" className="ds-verpill" title={t('sidebar.switchVersion')} aria-expanded={switching} onClick={() => setSwitching((v) => !v)}>
          {pillInner}
        </button>
      ) : (
        <span className="ds-verpill">{pillInner}</span>
      )}

      {view.interactive && (
        // 新建 / 导出 / 删除：hover（或键盘 focus）才浮现，平时不占位。
        <span className="ml-auto hidden group-hover:flex group-focus-within:flex items-center gap-0.5 shrink-0">
          <VerAction icon={I.plus} title={t('sidebar.newVersion')} onClick={() => view.onCreateVersion()} />
          <VerAction icon={I.export} title={t('sidebar.exportTitle')} onClick={view.onExportTrain} disabled={view.exporting} />
          {multiVersion && (
            <VerAction
              icon={I.trash}
              title={t('sidebar.deleteVersionTitle')}
              danger
              onClick={() => { if (activeVersion) view.onDeleteVersion(activeVersion.id) }}
            />
          )}
        </span>
      )}

      {switching && view.interactive && (
        <VersionSwitchPopover
          versions={project.versions}
          activeId={project.active_version_id}
          anchorRef={rowRef}
          onPick={(vid) => { view.onSelectVersion(vid); setSwitching(false) }}
          onClose={() => setSwitching(false)}
        />
      )}
    </div>
  )
}

/** 版本行的 icon-only action 按钮。 */
function VerAction({ icon, title, onClick, disabled = false, danger = false }: {
  icon: React.ReactNode; title: string; onClick: () => void
  disabled?: boolean; danger?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={title}
      className={[
        'w-6 h-6 grid place-items-center rounded-[6px] transition-colors shrink-0',
        disabled ? 'opacity-40 cursor-default' : 'cursor-pointer',
        'text-fg-tertiary',
        !disabled ? (danger ? 'hover:bg-sunken hover:text-err' : 'hover:bg-sunken hover:text-fg-primary') : '',
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
      className="absolute z-40 left-0 right-0 top-full mt-1 max-h-[300px] overflow-y-auto rounded-[12px] border border-dim bg-elevated shadow-lg p-1"
    >
      {versions.map((v) => {
        const isActive = v.id === activeId
        return (
          <button
            key={v.id}
            type="button"
            onClick={() => onPick(v.id)}
            className={[
              'w-full text-left px-2 h-8 rounded-[7px] font-mono text-xs flex items-center gap-2 cursor-pointer transition-colors',
              isActive ? 'bg-sunken text-fg-primary font-medium' : 'text-fg-secondary hover:bg-sunken hover:text-fg-primary',
            ].join(' ')}
          >
            <span className={STATUS_DOT[v.status] ?? 'ds-dot ds-dot-mute'} />
            <span className="flex-1 overflow-hidden text-ellipsis whitespace-nowrap">{v.label}</span>
            {isActive && <span className="text-accent shrink-0">{I.checkSm}</span>}
          </button>
        )
      })}
    </div>
  )
}

// ── queue data shared by the sidebar (pending count + running task) ─────────
function useQueueSnapshot(): { pending: number; running: Task | null } {
  const [pending, setPending] = useState(0)
  const [running, setRunning] = useState<Task | null>(null)
  const refresh = useCallback(async () => {
    try {
      const [run, pend] = await Promise.all([api.listQueue('running'), api.listQueue('pending')])
      setRunning(run.length > 0 ? run[0] : null)
      setPending(pend.length)
    } catch { /* offline / tests: keep the last known values */ }
  }, [])
  useEffect(() => { void refresh() }, [refresh])
  useEventStream(
    (evt: StudioEvent) => { if (evt.type === 'task_state_changed') void refresh() },
    { onOpen: () => void refresh() },
  )
  return { pending, running }
}

// ── project stepper nav ────────────────────────────────────────────────────
function ProjectStepperNav({ pid, activeVid, currentStep, version, collapsed, inRoute, trainPct }: {
  pid: string
  activeVid: string | null
  currentStep: string | null
  version: Version | null
  collapsed: boolean
  /** 是否当前正处于该项目的路由内（决定"概览"是否高亮；离开项目页只导航不高亮）。 */
  inRoute: boolean
  /** Training progress of the active version, when its task is running. */
  trainPct: number | null
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { toast } = useToast()
  const ctx = useProjectCtx()
  const view = useProjectView()
  const project = view?.project ?? null
  const stats = version?.stats

  // version 级 phase 编号 1-5（每个 version 自己一段流水线）。
  const STEPS = [
    { key: 'curate',     labelKey: 'nav.curate',     idx: '1', tail: stats?.train_image_count },
    { key: 'preprocess', labelKey: 'nav.preprocess', idx: '2', tail: project?.preprocess_image_count },
    { key: 'edit',       labelKey: 'nav.tagEdit',    idx: '3', tail: stats?.tagged_image_count },
    { key: 'reg',        labelKey: 'nav.reg',        idx: '4', tail: stats?.reg_image_count },
    { key: 'train',      labelKey: 'nav.train',      idx: '5', tail: trainPct != null ? `${trainPct}%` : undefined },
  ]

  const overviewActive = inRoute && currentStep === null
  // ADR-0007 §11.5 cursor 派生：cursor 之前的 phase = done。
  // The cursor only means something while the version is being prepared; a
  // trained / training version has every step behind it (same as Overview).
  const cursorPhase: VersionPhase = !version
    ? 'curating'
    : version.status === 'preparing' ? version.phase : 'ready'
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

  return (
    <>
      {!collapsed && project && (
        <Link to={`/projects/${pid}`} className="ds-projcard">
          <span className="ds-projcard-tile">{I.folderTile}</span>
          <span className="ds-projcard-txt">
            <span className="ds-projcard-name" title={project.title}>{project.title}</span>
            <span className="ds-projcard-slug" title={project.slug}>{project.slug}</span>
          </span>
          {I.updown}
        </Link>
      )}

      <NavItem to={`/projects/${pid}`} label={t('nav.overview')} icon={I.overview} active={overviewActive} collapsed={collapsed} />
      <NavItem
        to={`/projects/${pid}/download`}
        label={t('nav.download')}
        icon={I.download}
        active={currentStep === 'download'}
        collapsed={collapsed}
        tail={project?.download_image_count}
      />

      <VersionBar collapsed={collapsed} />

      <div className="ds-steps">
        {STEPS.map((s) => {
          const full = t(s.labelKey)
          const { short, optional } = splitOptional(full)
          const isActive = s.key === currentStep
          const phase = STEP_KEY_TO_PHASE[s.key]
          const phaseIdx = phase != null ? PHASE_ORDER.indexOf(phase) : -1
          // ADR-0007 §11.5-A: sidebar 是 cursor 推进主通道。
          // - phase_idx < cursorIdx  → done
          // - phase_idx == cursorIdx → 当前 cursor
          // - phase_idx == cursorIdx+1 → "下一步"，button，点击触发 advance/skip
          // - phase_idx > cursorIdx+1 → disabled
          const isCursorCurrent = phaseIdx === cursorIdx
          const isNextStep = phaseIdx === cursorIdx + 1
          const isFuturePhase = phaseIdx > cursorIdx + 1
          const isDone = isStepDone(s.key)

          const href = (isFuturePhase || isNextStep || !activeVid)
            ? null
            : `/projects/${pid}/v/${activeVid}/${s.key}`

          const cls = [
            'ds-step',
            isDone ? 'ds-is-done' : '',
            isCursorCurrent ? 'ds-is-cur' : '',
            isActive ? 'ds-is-active' : '',
            optional ? 'ds-is-opt' : '',
            collapsed ? 'justify-center' : '',
          ].filter(Boolean).join(' ')

          const inner = (
            <>
              <span className="ds-step-node">{isDone ? I.check : isCursorCurrent ? s.idx : ''}</span>
              {!collapsed && <span className="ds-step-label">{short}</span>}
              {!collapsed && s.tail != null && <span className="ds-step-tail">{s.tail}</span>}
            </>
          )
          const title = collapsed ? `${s.idx}. ${full}` : optional ? full : undefined

          if (isNextStep) {
            return (
              <button key={s.key} type="button" onClick={() => void handleAdvanceToNext()}
                title={t('sidebar.advanceToTitle', { label: full })} className={cls}>
                {inner}
              </button>
            )
          }
          if (!href) {
            return (
              <span key={s.key} title={title} aria-disabled="true" className={cls + ' opacity-40 cursor-default'}>
                {inner}
              </span>
            )
          }
          return (
            <Link key={s.key} to={href} title={title} className={cls}>
              {inner}
            </Link>
          )
        })}
      </div>
    </>
  )
}

// ── sidebar ────────────────────────────────────────────────────────────────
const SIDEBAR_KEY = 'studio.sidebar.expanded'

/** Props are phone-only. On desktop `mobile` is false. */
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
  const queue = useQueueSnapshot()

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

  // Training progress for the active version's step tail.
  const runningForVersion = queue.running && activeVid && String(queue.running.version_id ?? '') === activeVid
    ? queue.running : null
  const { state: monitor } = useMonitorProgress(runningForVersion?.id ?? null)
  const trainPct = runningForVersion && monitor?.step != null && monitor.total_steps
    ? Math.round((monitor.step / monitor.total_steps) * 100)
    : null

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

  // On a phone the rail is always full width and slides in over the content.
  const asideClass = mobile
    ? 'ds-side fixed inset-y-0 left-0 z-50 w-[min(86vw,var(--sidebar-w))] bg-surface shadow-xl transition-transform duration-200 ease-out'
      + (mobileOpen ? ' translate-x-0' : ' -translate-x-full')
    : 'ds-side overflow-hidden h-full transition-[width] duration-[160ms] ease-in-out'

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
        <div className={`ds-brand ${collapsed ? 'justify-center !px-0' : ''}`}>
          {!collapsed && (
            <>
              <span className="ds-brand-mark" aria-hidden>{I.brand}</span>
              <span className="ds-brand-name">Anima<i>TrainHub</i></span>
            </>
          )}
          <button
            type="button"
            onClick={toggle}
            className={`ds-brand-collapse ${collapsed ? '!ml-0' : ''} hover:bg-sunken hover:text-fg-primary transition-colors`}
            title={collapsed ? t('sidebar.expand') : t('sidebar.collapse')}
            aria-label={collapsed ? t('sidebar.expand') : t('sidebar.collapse')}
          >
            {collapsed ? I.expand : I.collapse}
          </button>
        </div>
      )}
      {mobile && (
        <div className="ds-brand">
          <span className="ds-brand-mark" aria-hidden>{I.brand}</span>
          <span className="ds-brand-name">Anima<i>TrainHub</i></span>
          <button
            type="button"
            onClick={onMobileClose}
            aria-label={t('common.close')}
            className="ds-brand-collapse hover:bg-sunken"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}

      <nav className={`ds-sidenav overflow-y-auto overflow-x-hidden ${collapsed ? '!px-2' : ''}`} style={{ scrollbarGutter: 'auto' }}>
        <NavItem to="/" label={t('nav.projects')} icon={I.folder} active={!inProject && location.pathname === '/'} collapsed={collapsed} />

        {inProject && pid && (
          <ProjectStepperNav
            pid={pid}
            activeVid={activeVid}
            currentStep={currentStep}
            version={view?.activeVersion ?? null}
            collapsed={collapsed}
            inRoute={inRoute}
            trainPct={trainPct}
          />
        )}

        <NavItem to="/queue" label={t('nav.queue')} icon={I.queue} active={isMain('/queue')} collapsed={collapsed} tail={queue.pending > 0 ? queue.pending : undefined} />
        <NavItem to="/tools/generate" label={t('nav.generate')} icon={I.image} active={isMain('/tools/generate')} collapsed={collapsed} />
        <NavItem to="/tools/soup" label={t('nav.soup')} icon={I.soup} active={isMain('/tools/soup')} collapsed={collapsed} />
      </nav>

      <div className={`ds-sidefoot ${collapsed ? '!px-2' : ''}`}>
        <NavItem to="/tools/presets" label={t('nav.presets')} icon={I.preset} active={isMain('/tools/presets')} collapsed={collapsed} />
        <NavItem to="/tools/monitor" label={t('nav.monitor')} icon={I.monitor} active={isMain('/tools/monitor')} collapsed={collapsed} />
        <NavButton
          onClick={() => settingsDrawer.isOpen ? void settingsDrawer.close() : settingsDrawer.open()}
          label={t('nav.settings')}
          icon={I.cog}
          active={false}
          collapsed={collapsed}
        />
      </div>
    </aside>
  )
}
