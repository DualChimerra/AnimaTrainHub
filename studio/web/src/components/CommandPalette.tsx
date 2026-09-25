import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { api, type CaptionEntry, type PresetSummary, type ProjectSummary, type Task } from '../api/client'
import { useProjectCtx } from '../context/ProjectContext'
import { useSettingsDrawer } from '../lib/SettingsDrawer'

type IconKey = 'folder' | 'queue' | 'preset' | 'monitor' | 'cog' | 'image' | 'step' | 'tag'
/** Right-hand label of a row (commandPalette.kind.*). */
type Kind = 'project' | 'step' | 'task' | 'tag' | 'page' | 'preset' | 'setting'

interface Item {
  id: string
  label: string
  sub?: string
  group: string
  kind: Kind
  icon: IconKey
  /** 路由跳转。跟 action 二选一。 */
  path?: string
  /** 自定义动作（如打开抽屉）。优先于 path。 */
  action?: () => void
}

const SEARCH_ICON = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" />
  </svg>
)

/** 每个结果左侧的小图标（prototype CommandPalette：icon + label + sub + chevron）。 */
const ITEM_ICONS: Record<IconKey, React.ReactNode> = {
  folder:  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>,
  queue:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M4 6h16M4 12h10M4 18h16"/><circle cx="18" cy="12" r="2" fill="currentColor"/></svg>,
  preset:  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><line x1="6" y1="4" x2="6" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/><line x1="18" y1="4" x2="18" y2="20"/><circle cx="6" cy="9" r="2" fill="var(--bg-surface)"/><circle cx="12" cy="15" r="2" fill="var(--bg-surface)"/><circle cx="18" cy="7" r="2" fill="var(--bg-surface)"/></svg>,
  monitor: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M3 17l4-6 4 3 5-9 5 7"/></svg>,
  cog:     <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>,
  image:   <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="1.6" fill="currentColor"/><path d="m21 15-5-5L5 21"/></svg>,
  step:    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9 12h6"/></svg>,
  tag:     <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2H2v10l9.29 9.29a1 1 0 0 0 1.41 0l7.3-7.3a1 1 0 0 0 0-1.41L12 2z"/><path d="M7 7h.01"/></svg>,
}

interface Props {
  open: boolean
  onClose: () => void
}

export default function CommandPalette({ open, onClose }: Props) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const ctx = useProjectCtx()
  const settingsDrawer = useSettingsDrawer()
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const [query, setQuery] = useState('')
  const [activeIdx, setActiveIdx] = useState(0)

  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [projectsLoaded, setProjectsLoaded] = useState(false)

  const [presets, setPresets] = useState<PresetSummary[]>([])
  const [presetsLoaded, setPresetsLoaded] = useState(false)

  const [tasks, setTasks] = useState<Task[]>([])

  const [captions, setCaptions] = useState<CaptionEntry[]>([])
  const [captionsCacheKey, setCaptionsCacheKey] = useState<string | null>(null)
  const [captionsLoading, setCaptionsLoading] = useState(false)

  useEffect(() => {
    if (open) {
      setQuery('')
      setActiveIdx(0)
      setTimeout(() => inputRef.current?.focus(), 40)
    } else {
      setProjectsLoaded(false)
      setPresetsLoaded(false)
    }
  }, [open])

  useEffect(() => {
    if (!open || projectsLoaded) return
    let cancelled = false
    api.listProjects().then((items) => {
      if (!cancelled) { setProjects(items); setProjectsLoaded(true) }
    }).catch(() => {
      if (!cancelled) setProjectsLoaded(true)
    })
    return () => { cancelled = true }
  }, [open, projectsLoaded])

  useEffect(() => {
    if (!open) return
    let cancelled = false
    api.listQueue().then((items) => { if (!cancelled) setTasks(items) }).catch(() => {})
    return () => { cancelled = true }
  }, [open])

  useEffect(() => {
    if (!open || presetsLoaded) return
    let cancelled = false
    api.listPresets().then((items) => {
      if (!cancelled) { setPresets(items); setPresetsLoaded(true) }
    }).catch(() => {
      if (!cancelled) setPresetsLoaded(true)
    })
    return () => { cancelled = true }
  }, [open, presetsLoaded])

  const pid = ctx?.project?.id
  const vid = ctx?.activeVersion?.id
  const queryEnoughForTags = query.length >= 2

  useEffect(() => {
    if (!open || !queryEnoughForTags || !pid || !vid) return
    const key = `${pid}:${vid}`
    if (captionsCacheKey === key) return

    let cancelled = false
    setCaptionsLoading(true)
    api.listCaptionsFull(pid, vid).then((result) => {
      if (!cancelled) {
        setCaptions(result.items)
        setCaptionsCacheKey(key)
        setCaptionsLoading(false)
      }
    }).catch(() => {
      if (!cancelled) setCaptionsLoading(false)
    })
    return () => { cancelled = true }
  }, [open, queryEnoughForTags, pid, vid, captionsCacheKey])

  const allItems = useMemo<Item[]>(() => {
    const items: Item[] = []
    const gProjects = t('commandPalette.groupProjects')
    const gTasks = t('commandPalette.groupTasks')
    const gActions = t('commandPalette.groupActions')

    for (const p of projects) {
      items.push({
        id: `project:${p.id}`,
        label: p.title || `#${p.id}`,
        sub: p.slug ? `/${p.slug}` : t('commandPalette.projectItem', { id: p.id }),
        group: gProjects,
        kind: 'project',
        icon: 'folder',
        path: `/projects/${p.id}`,
      })
    }

    if (ctx) {
      const cpid = ctx.project.id
      const cvid = ctx.activeVersion?.id
      const sub = ctx.activeVersion ? `${ctx.project.title} / ${ctx.activeVersion.label}` : ctx.project.title
      items.push({ id: `overview:${cpid}`, label: t('nav.overview'), sub: ctx.project.title, group: gProjects, kind: 'step', icon: 'folder', path: `/projects/${cpid}` })
      items.push({ id: `download:${cpid}`, label: t('nav.download'), sub: ctx.project.title, group: gProjects, kind: 'step', icon: 'step', path: `/projects/${cpid}/download` })
      if (cvid) {
        const base = `/projects/${cpid}/v/${cvid}`
        items.push({ id: `curate:${cpid}`, label: t('nav.curate'),  sub, group: gProjects, kind: 'step', icon: 'step', path: `${base}/curate` })
        items.push({ id: `edit:${cpid}`,   label: t('nav.tagEdit'), sub, group: gProjects, kind: 'step', icon: 'step', path: `${base}/edit` })
        items.push({ id: `reg:${cpid}`,    label: t('nav.reg'),     sub, group: gProjects, kind: 'step', icon: 'step', path: `${base}/reg` })
        items.push({ id: `train:${cpid}`,  label: t('nav.train'),   sub, group: gProjects, kind: 'step', icon: 'step', path: `${base}/train` })
      }
    }

    for (const task of tasks) {
      items.push({
        id: `task:${task.id}`,
        label: `#${task.id} · ${task.name}`,
        sub: t(`status.${task.status === 'pending' ? 'queued' : task.status}`),
        group: gTasks,
        kind: 'task',
        icon: 'queue',
        path: `/queue/${task.id}`,
      })
    }

    items.push({ id: 'home',     label: t('commandPalette.home'), sub: t('commandPalette.homeSub'),     group: gActions, kind: 'page', icon: 'folder',  path: '/' })
    items.push({ id: 'queue',    label: t('nav.queue'),           sub: t('commandPalette.queueSub'),    group: gActions, kind: 'page', icon: 'queue',   path: '/queue' })
    items.push({ id: 'generate', label: t('nav.generate'),        sub: t('commandPalette.generateSub'), group: gActions, kind: 'page', icon: 'image',   path: '/tools/generate' })
    items.push({ id: 'presets',  label: t('nav.presets'),         sub: t('commandPalette.presetsSub'),  group: gActions, kind: 'page', icon: 'preset',  path: '/tools/presets' })
    items.push({ id: 'monitor',  label: t('nav.monitor'),         sub: t('commandPalette.monitorSub'),  group: gActions, kind: 'page', icon: 'monitor', path: '/tools/monitor' })
    items.push({ id: 'settings', label: t('nav.settings'),        sub: t('commandPalette.settingsSub'), group: gActions, kind: 'setting', icon: 'cog', action: () => settingsDrawer.open() })

    for (const p of presets) {
      items.push({
        id: `preset:${p.name}`,
        label: p.name,
        sub: t('commandPalette.presetItem'),
        group: gActions,
        kind: 'preset',
        icon: 'preset',
        path: '/tools/presets',
      })
    }

    return items
  }, [projects, presets, tasks, ctx, t, settingsDrawer])

  const filteredNav = useMemo(() => {
    if (!query.trim()) return allItems
    const q = query.toLowerCase()
    return allItems.filter(
      (item) =>
        item.label.toLowerCase().includes(q) ||
        (item.sub ?? '').toLowerCase().includes(q) ||
        item.group.toLowerCase().includes(q),
    )
  }, [allItems, query])

  const tagItems = useMemo<Item[]>(() => {
    if (!queryEnoughForTags || !ctx?.activeVersion || captions.length === 0) return []
    const cpid = ctx.project.id
    const cvid = ctx.activeVersion.id
    const q = query.toLowerCase()
    const tagCounts = new Map<string, number>()

    for (const c of captions) {
      for (const tag of c.tags) {
        if (tag.toLowerCase().includes(q)) {
          tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1)
        }
      }
    }

    return Array.from(tagCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12)
      .map(([tag, count]): Item => ({
        id: `tag:${tag}`,
        label: tag,
        sub: t('commandPalette.imageCount', { n: count }),
        group: t('commandPalette.groupTasks'),
        kind: 'tag',
        icon: 'tag',
        path: `/projects/${cpid}/v/${cvid}/edit`,
      }))
  }, [captions, queryEnoughForTags, query, ctx, t])

  const filtered = useMemo(() => [...filteredNav, ...tagItems], [filteredNav, tagItems])

  const grouped = useMemo(() => {
    const map = new Map<string, Item[]>()
    for (const g of [t('commandPalette.groupProjects'), t('commandPalette.groupTasks'), t('commandPalette.groupActions')]) {
      if (filtered.some((i) => i.group === g)) map.set(g, [])
    }
    for (const item of filtered) {
      if (!map.has(item.group)) map.set(item.group, [])
      map.get(item.group)!.push(item)
    }
    return map
  }, [filtered, t])

  const flatItems = useMemo(() => {
    const out: Item[] = []
    for (const [, items] of grouped) out.push(...items)
    return out
  }, [grouped])

  const select = useCallback(
    (item: Item) => {
      if (item.action) item.action()
      else if (item.path) navigate(item.path)
      onClose()
    },
    [navigate, onClose],
  )

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIdx((i) => Math.min(i + 1, flatItems.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIdx((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (flatItems[activeIdx]) select(flatItems[activeIdx])
    } else if (e.key === 'Escape') {
      onClose()
    }
  }

  useEffect(() => {
    const el = listRef.current
    if (!el) return
    const active = el.querySelector(`[data-palette-idx="${activeIdx}"]`) as HTMLElement | null
    if (active) active.scrollIntoView({ block: 'nearest' })
  }, [activeIdx])

  if (!open) return null

  const q = query.trim()
  return (
    <div
      className="fixed inset-0 z-[80]"
      style={{ background: 'rgba(20,22,20,.18)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        className="ds-palette"
        style={{ width: 'min(640px, calc(100vw - 32px))', top: '12vh' }}
        role="dialog"
        aria-modal="true"
        aria-label={t('commandPalette.placeholder')}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="ds-palette-q">
          <span style={{ color: 'var(--ink-3)', display: 'grid' }}>{SEARCH_ICON}</span>
          <input
            ref={inputRef}
            type="text"
            aria-label={t('commandPalette.placeholder')}
            placeholder={t('commandPalette.placeholder')}
            value={query}
            onChange={(e) => { setQuery(e.target.value); setActiveIdx(0) }}
            onKeyDown={handleKeyDown}
          />
          {captionsLoading && <span className="ds-kpi-meta">{t('commandPalette.searchingTags')}</span>}
          {q && <span className="ds-badge ds-mute">{t('commandPalette.matches', { count: flatItems.length })}</span>}
        </div>

        <div ref={listRef} style={{ maxHeight: 'min(56vh, 460px)', overflowY: 'auto', paddingBottom: 6 }}>
          {filtered.length === 0 ? (
            <div className="ds-empty">{t('commandPalette.noResults')}</div>
          ) : (
            [...grouped.entries()].map(([group, items]) => (
              <div key={group}>
                <div className="ds-cap ds-palette-sect">{group}</div>
                {items.map((item) => {
                  const idx = flatItems.indexOf(item)
                  return (
                    <button
                      key={item.id}
                      type="button"
                      data-palette-idx={idx}
                      onClick={() => select(item)}
                      onMouseEnter={() => setActiveIdx(idx)}
                      className={`ds-palette-row${idx === activeIdx ? ' ds-is-on' : ''}`}
                      style={{ width: '100%', textAlign: 'left' }}
                    >
                      <span className="ds-palette-ico">{ITEM_ICONS[item.icon]}</span>
                      <span className="ds-palette-name">
                        <Highlight text={item.label} q={q} />
                        {item.sub && <span style={{ color: 'var(--ink-4)' }}> · <Highlight text={item.sub} q={q} /></span>}
                      </span>
                      <span className="ds-palette-kind">{t(`commandPalette.kind.${item.kind}`)}</span>
                    </button>
                  )
                })}
              </div>
            ))
          )}
        </div>

        <div className="ds-palette-foot">
          <span><span className="ds-kbd">↑</span><span className="ds-kbd">↓</span>{t('commandPalette.navigate')}</span>
          <span><span className="ds-kbd">Enter</span>{t('commandPalette.select')}</span>
          <span><span className="ds-kbd">Esc</span>{t('commandPalette.close')}</span>
          <span style={{ marginLeft: 'auto' }}>
            {queryEnoughForTags && ctx?.activeVersion
              ? t('commandPalette.searchTagsHint', { n: captions.length })
              : t('commandPalette.shortcutHint')}
          </span>
        </div>
      </div>
    </div>
  )
}

/** Marks the first case-insensitive occurrence of `q` in `text`. */
function Highlight({ text, q }: { text: string; q: string }) {
  if (!q) return <>{text}</>
  const i = text.toLowerCase().indexOf(q.toLowerCase())
  if (i < 0) return <>{text}</>
  return <>{text.slice(0, i)}<em>{text.slice(i, i + q.length)}</em>{text.slice(i + q.length)}</>
}
