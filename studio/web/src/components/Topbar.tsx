import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useProjectCtx } from '../context/ProjectContext'
import { api, type Task } from '../api/client'
import { useEventStream, type StudioEvent } from '../lib/useEventStream'
import { useMonitorProgress } from '../lib/useMonitorProgress'
import { useRuntimeModeOptional } from '../lib/RuntimeMode'
import { useSettingsDrawer } from '../lib/SettingsDrawer'
import { splitOptional } from '../lib/labels'
import CommandPalette from './CommandPalette'
import SystemStats from './SystemStats'

const SearchIcon = (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" />
  </svg>
)

/** 运行模式徽标（本 fork）。点开设置抽屉的运行模式区。
 *
 *  常驻显示而不是"仅 colab 时显示"：模式选错的症状（本地起了 0.0.0.0、云端不
 *  开浏览器）远看都像别的 bug，把当前值一直摆在眼前是最省事的排查入口。 */
function RuntimeModeBadge() {
  const { t } = useTranslation()
  const runtime = useRuntimeModeOptional()
  const drawer = useSettingsDrawer()
  const info = runtime?.info
  const mode = runtime?.mode ?? 'local'
  if (!info) return null
  return (
    <button
      type="button"
      onClick={() => drawer.open({ section: 'runtime-mode' })}
      title={t('runtimeMode.current')}
      className="ds-tb-chip shrink-0"
    >
      <span className={`ds-dot ${mode === 'colab' ? 'ds-dot-run' : 'ds-dot-ok'}`} />
      {t(`runtimeMode.${mode}.name`)}
    </button>
  )
}

const QueueIcon = (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
    <rect x="2" y="3" width="20" height="14" rx="2" />
    <path d="M8 21h8M12 17v4" />
  </svg>
)


// ── 格式化工具 ──────────────────────────────────────────────────────────────

function formatETA(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  const h = Math.floor(seconds / 3600)
  const m = Math.round((seconds % 3600) / 60)
  return m > 0 ? `${h}h${m}m` : `${h}h`
}

function formatElapsed(from: number): string {
  const s = Math.max(0, (Date.now() / 1000) - from)
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.round(s / 60)}m`
  const h = Math.floor(s / 3600)
  const m = Math.round((s % 3600) / 60)
  return m > 0 ? `${h}h${m}m` : `${h}h`
}

// ── breadcrumb ──────────────────────────────────────────────────────────────

interface Crumb { label: string; mono?: boolean; to?: string }

function useBreadcrumbs(): Crumb[] {
  const { t } = useTranslation()
  const { pathname } = useLocation()
  const ctx = useProjectCtx()
  const parts = pathname.split('/').filter(Boolean)

  if (parts.length === 0) return [{ label: t('breadcrumb.projects'), to: '/' }]

  if (parts[0] === 'queue') {
    if (parts.length === 1) return [{ label: t('breadcrumb.queue'), to: '/queue' }]
    return [{ label: t('breadcrumb.queue'), to: '/queue' }, { label: `#${parts[1]}`, mono: true }]
  }

  if (parts[0] === 'tools') {
    const labels: Record<string, string> = {
      presets: t('breadcrumb.presets'),
      monitor: t('breadcrumb.monitor'),
      settings: t('breadcrumb.settings'),
      generate: t('breadcrumb.generate'),
      soup: t('breadcrumb.soup'),
    }
    return [{ label: labels[parts[1]] ?? parts[1] }]
  }

  if (parts[0] === 'projects') {
    const crumbs: Crumb[] = [{ label: t('breadcrumb.projects'), to: '/' }]

    const projectLabel = ctx?.project?.title ?? (parts[1] ? `#${parts[1]}` : null)
    const projectId = parts[1]
    if (projectLabel) crumbs.push({ label: projectLabel, to: projectId ? `/projects/${projectId}` : undefined })

    const vIdx = parts.indexOf('v')
    if (vIdx !== -1 && parts[vIdx + 1]) {
      const versionLabel = ctx?.activeVersion?.label ?? `v${parts[vIdx + 1]}`
      const vid = parts[vIdx + 1]
      crumbs.push({ label: versionLabel, mono: true })
      // Step crumbs read exactly like the sidebar steps (short form).
      const stepLabels: Record<string, string> = {
        curate: splitOptional(t('nav.curate')).short,
        preprocess: splitOptional(t('nav.preprocess')).short,
        edit: splitOptional(t('nav.tagEdit')).short,
        reg: splitOptional(t('nav.reg')).short,
        train: splitOptional(t('nav.train')).short,
      }
      const step = parts[vIdx + 2]
      if (step && stepLabels[step]) {
        crumbs.push({ label: stepLabels[step], to: `/projects/${projectId}/v/${vid}/${step}` })
      }
    } else if (parts[2] === 'download') {
      crumbs.push({ label: t('nav.download'), to: `/projects/${projectId}/download` })
    }
    return crumbs
  }

  return [{ label: pathname }]
}

// ── Topbar ──────────────────────────────────────────────────────────────────

export default function Topbar({
  mobile = false,
  onOpenNav,
}: { mobile?: boolean; onOpenNav?: () => void } = {}) {
  const { t } = useTranslation()
  const crumbs = useBreadcrumbs()
  const navigate = useNavigate()
  const ctx = useProjectCtx()
  const [paletteOpen, setPaletteOpen] = useState(false)
  const searchBtnRef = useRef<HTMLButtonElement>(null)

  const [runningTask, setRunningTask] = useState<Task | null>(null)
  const [pendingCount, setPendingCount] = useState(0)

  const { state: monitor } = useMonitorProgress(runningTask?.id ?? null)

  const refreshQueue = useCallback(async () => {
    try {
      const [running, pending] = await Promise.all([
        api.listQueue('running'),
        api.listQueue('pending'),
      ])
      setRunningTask(running.length > 0 ? running[0] : null)
      setPendingCount(pending.length)
    } catch {
      // 忽略
    }
  }, [])

  useEffect(() => {
    void refreshQueue()
  }, [refreshQueue])

  useEventStream(
    (evt: StudioEvent) => {
      if (evt.type === 'task_state_changed') void refreshQueue()
    },
    { onOpen: () => void refreshQueue() },
  )

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setPaletteOpen((p) => !p)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const progress = (() => {
    if (!monitor || !runningTask) return null

    if (monitor.step != null && monitor.total_steps != null && monitor.total_steps > 0) {
      const pct = Math.round((monitor.step / monitor.total_steps) * 100)
      let eta = ''
      if (monitor.speed && monitor.speed > 0) {
        const remaining = (monitor.total_steps - monitor.step) / monitor.speed
        eta = `~${formatETA(remaining)}`
      }
      return { pct, current: monitor.step, total: monitor.total_steps, eta, unit: 'step' as const }
    }

    if (monitor.epoch != null && monitor.total_epochs != null && monitor.total_epochs > 0) {
      const pct = Math.round((monitor.epoch / monitor.total_epochs) * 100)
      return { pct, current: monitor.epoch, total: monitor.total_epochs, unit: 'epoch' as const }
    }

    if (monitor.start_time) {
      const elapsed = formatElapsed(monitor.start_time)
      return { currentUnit: t('topbar.running', { elapsed }) } as const
    }

    return null
  })()

  const projectLabel = (ctx && runningTask?.project_id != null && runningTask.project_id === ctx.project.id)
    ? ctx.project.title
    : null
  const configName = runningTask ? (runningTask.config_name || runningTask.name || '—') : ''
  const taskLabel = projectLabel ? `${projectLabel} / ${configName}` : configName

  const progressSuffix = (() => {
    if (!progress) return runningTask?.started_at ? ` · ${t('topbar.running', { elapsed: formatElapsed(runningTask.started_at) })}` : ''
    if ('pct' in progress) {
      const p = progress as { current: number; total: number; unit: string; eta?: string }
      const nums = `${p.current.toLocaleString()} / ${p.total.toLocaleString()}`
      return ` · ${p.unit} ${nums}${p.eta ? ` ${p.eta}` : ''}`
    }
    if ('currentUnit' in progress) return ` · ${(progress as { currentUnit: string }).currentUnit}`
    return ''
  })()

  return (
    <>
      <header className="ds-topbar" style={{ height: 'var(--topbar-h)' }}>
        {mobile && (
          <button
            type="button"
            onClick={onOpenNav}
            aria-label={t('nav.menu')}
            className="ds-tb-icon shrink-0"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M3 6h18M3 12h18M3 18h18" />
            </svg>
          </button>
        )}
        <div className="ds-crumbs flex-1 overflow-hidden">
          {(mobile ? crumbs.slice(-1) : crumbs).map((b, i, shown) => {
            const isLast = i === shown.length - 1
            const cls = [b.mono ? 'ds-mono' : '', 'truncate', mobile ? 'max-w-full' : 'max-w-[280px]'].filter(Boolean).join(' ')
            return (
              <span key={i} className={`flex items-center gap-[7px] ${mobile ? 'min-w-0' : i === 0 || isLast ? 'shrink-0' : 'min-w-0'}`}>
                {i > 0 && <span className="ds-crumb-sep" aria-hidden>/</span>}
                {isLast
                  ? <b className={cls}>{b.label}</b>
                  : b.to
                    ? <Link to={b.to} className={cls + ' hover:text-fg-primary transition-colors'}>{b.label}</Link>
                    : <span className={cls}>{b.label}</span>}
              </span>
            )
          })}
        </div>

        <div className="ds-tb-right">
          <SystemStats />

          {runningTask && (
            <button
              type="button"
              onClick={() => navigate(`/queue/${runningTask.id}`)}
              className="ds-tb-chip shrink-0 max-w-[340px]"
              title={`${t('topbar.trainingLabel')} · ${taskLabel}${progressSuffix}`}
              aria-label={`${t('topbar.trainingLabel')} · ${taskLabel}`}
            >
              <span className="ds-dot ds-dot-run" />
              <span className="truncate">
                #{runningTask.id}
                {progress && 'pct' in progress
                  ? ` · ${(progress as { current: number }).current.toLocaleString()} / ${(progress as { total: number }).total.toLocaleString()}`
                  : ''}
              </span>
            </button>
          )}

          {!runningTask && pendingCount > 0 && (
            <button
              type="button"
              onClick={() => navigate('/queue')}
              className="ds-tb-chip shrink-0"
            >
              {QueueIcon}
              <span>{t('topbar.pendingCount', { n: pendingCount })}</span>
            </button>
          )}

          {/* 本 fork：公告栏铃铛随 in-app updater/announcements 移除 */}

          {!mobile && <RuntimeModeBadge />}

          <button
            type="button"
            ref={searchBtnRef}
            onClick={() => setPaletteOpen(true)}
            title={t('topbar.search')}
            aria-label={t('topbar.searchAriaLabel')}
            className="ds-tb-icon shrink-0"
          >
            {SearchIcon}
          </button>
        </div>
      </header>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
      />
    </>
  )
}
