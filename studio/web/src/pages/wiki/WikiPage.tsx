// Wiki (experimental) — a searchable encyclopedia of training parameters,
// guides, recipes and terms. Self-contained: everything lives in this folder;
// the app only adds a route and a sidebar link (both marked "Wiki (experimental)").
//
// Layout follows a docs site inside one card: a rail with search and the
// section tree on the left, the page in the middle, and on wide screens an
// aside with "On this page" (articles) or the facts box (parameters).
import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { useSearchParams } from 'react-router-dom'
import { api, type SchemaResponse } from '../../api/client'
import PageHead from '../../components/ds/PageHead'
import Popover, { MenuItems } from '../../components/ds/Popover'
import { useIsMobile } from '../../lib/useMediaQuery'
import { controlKind } from '../../lib/schema'
import { ARTICLES_EN } from './content.en'
import { ARTICLES_RU } from './content.ru'
import {
  buildArticleEntries,
  buildParamEntries,
  exprKeys,
  highlightParts,
  linksOf,
  scoreEntry,
  sectionLabel,
  sectionRank,
  store,
  stripTags,
  tokenize,
  type WikiBlock,
  type WikiEntry,
  type WikiKind,
} from './model'
import './strings'
import './wiki.css'

type KindFilter = 'all' | 'article' | 'param' | 'fav'

const KIND_FILTERS: KindFilter[] = ['all', 'article', 'param', 'fav']
const KIND_ORDER: Record<WikiKind, number> = { recipe: 0, guide: 1, term: 2, param: 3 }

function parseKind(k: string | null): KindFilter {
  if (k === 'guide' || k === 'recipe' || k === 'term') return 'article'
  return (KIND_FILTERS as string[]).includes(k ?? '') ? (k as KindFilter) : 'all'
}

// ── icons ──────────────────────────────────────────────────────────────────
const svg = (d: ReactNode, size = 14, w = 1.8) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={w} strokeLinecap="round" strokeLinejoin="round" aria-hidden>{d}</svg>
)
const KIND_ICON: Record<WikiKind, ReactNode> = {
  param: svg(<><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12" /><circle cx="16" cy="6" r="2" /><circle cx="10" cy="12" r="2" /><circle cx="18" cy="18" r="2" /></>, 13),
  guide: svg(<><path d="M2 4h6a4 4 0 0 1 4 4v13a3 3 0 0 0-3-3H2z" /><path d="M22 4h-6a4 4 0 0 0-4 4v13a3 3 0 0 1 3-3h7z" /></>, 13),
  recipe: svg(<><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-9V3" /><path d="M7.5 15h9" /></>, 13),
  term: svg(<><path d="M4 19 9 5l5 14M6 14h6" /><path d="M15 15.5a2.5 2.5 0 1 0 5 0v-3M20 19v-3.5" /></>, 13),
}
const SECTION_ICON: Record<string, ReactNode> = {
  basics: KIND_ICON.guide,
  model: svg(<><path d="M21 8 12 3 3 8v8l9 5 9-5z" /><path d="m3 8 9 5 9-5M12 13v8" /></>),
  dataset: svg(<><rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="9" cy="9" r="2" /><path d="m21 15-5-5L5 21" /></>),
  caption: svg(<path d="M4 6h16M4 12h16M4 18h10" />),
  lora: svg(<><circle cx="6" cy="6" r="2.5" /><circle cx="18" cy="6" r="2.5" /><circle cx="12" cy="18" r="2.5" /><path d="M8 7.5 11 16M16 7.5 13 16M8.5 6h7" /></>),
  training: svg(<><path d="m3 17 6-6 4 4 8-8" /><path d="M14 7h7v7" /></>),
  noise_augmentation: svg(<path d="M2 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0" />),
  timestep_sampling: svg(<><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>),
  loss: svg(<><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1" /></>),
  system: svg(<><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" /></>),
  output: svg(<><path d="M12 3v12M7 10l5 5 5-5" /><path d="M5 21h14" /></>),
  sample: svg(<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />),
  eval_validation: svg(<><circle cx="12" cy="12" r="9" /><path d="m8 12 3 3 5-6" /></>),
  monitor: svg(<path d="M3 12h4l3 8 4-16 3 8h4" />),
  troubleshooting: svg(<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.4-.6-.6-2.4z" />),
}
const FOLDER_ICON = svg(<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />)
const sectionIcon = (key: string) => SECTION_ICON[key] ?? FOLDER_ICON

const I = {
  search: svg(<><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></>, 14, 2),
  x: svg(<path d="M18 6 6 18M6 6l12 12" />, 12, 2.2),
  star: (on: boolean, size = 14) => <svg width={size} height={size} viewBox="0 0 24 24" fill={on ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" aria-hidden><path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1.1 6.2L12 17.3 6.4 20.2l1.1-6.2L3 9.6l6.2-.9z" /></svg>,
  copy: svg(<><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V6a2 2 0 0 1 2-2h9" /></>, 13, 1.9),
  check: svg(<path d="m5 12.5 4.5 4.5L19 7" />, 13, 2.2),
  back: svg(<path d="m15 18-6-6 6-6" />, 14, 2),
  chev: svg(<path d="m9 6 6 6-6 6" />, 12, 2.2),
  left: svg(<path d="M19 12H5M11 6l-6 6 6 6" />, 13, 2),
  right: svg(<path d="M5 12h14M13 6l6 6-6 6" />, 13, 2),
  home: svg(<><path d="M3 11 12 4l9 7" /><path d="M5 10v10h14V10" /></>),
  filter: svg(<path d="M4 5h16l-6 7.5V19l-4 1v-7.5z" />, 14, 1.9),
  spark: svg(<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />, 15, 1.9),
}

// ── data ───────────────────────────────────────────────────────────────────
function useSchema(): { schema: SchemaResponse | null; offline: boolean } {
  const [state, setState] = useState<{ schema: SchemaResponse | null; offline: boolean }>({ schema: null, offline: false })
  useEffect(() => {
    let alive = true
    api.schema()
      .then((s) => { if (alive) setState({ schema: s, offline: false }) })
      .catch(() => { if (alive) setState({ schema: null, offline: true }) })
    return () => { alive = false }
  }, [])
  return state
}

function fmtValue(v: unknown): string {
  if (v === undefined) return ''
  if (v === null) return 'null'
  if (typeof v === 'string') return v === '' ? '""' : v
  if (Array.isArray(v)) return v.length === 0 ? '[]' : v.map((x) => fmtValue(x)).join(', ')
  return String(v)
}

/** Readable form of a show_when / disable_when expression. */
function fmtExpr(expr: string): string {
  return expr.replace(/\s*\|\|\s*/g, ' · ').replace(/\s*&&\s*/g, ' & ').replace(/==/g, ' = ').replace(/!=/g, ' ≠ ')
}

const headingId = (i: number) => `wk-h-${i}`

// ── rich text ──────────────────────────────────────────────────────────────
interface LinkCtx {
  byId: Map<string, WikiEntry>
  open: (id: string) => void
}

/** Inline markup: `code`, **bold**, [[id]] / [[id|label]]. */
function Inline({ text, ctx }: { text: string; ctx: LinkCtx }) {
  const out: ReactNode[] = []
  const re = /(`[^`]+`|\*\*[^*]+\*\*|\[\[[^\]]+\]\])/g
  let last = 0
  let i = 0
  for (const m of text.matchAll(re)) {
    if (m.index! > last) out.push(<Fragment key={i++}>{text.slice(last, m.index)}</Fragment>)
    const tok = m[0]
    if (tok.startsWith('`')) {
      out.push(<code key={i++} className="wk-code">{tok.slice(1, -1)}</code>)
    } else if (tok.startsWith('**')) {
      out.push(<b key={i++}><Inline text={tok.slice(2, -2)} ctx={ctx} /></b>)
    } else {
      const [id, label] = tok.slice(2, -2).split('|')
      const target = ctx.byId.get(id)
      const shown = label ?? (target ? (target.kind === 'param' ? target.key! : target.title) : id.replace(/^\w+:/, ''))
      out.push(target ? (
        <a
          key={i++}
          href={`?e=${encodeURIComponent(id)}`}
          className={`wk-link${target.kind === 'param' && !label ? ' wk-link-key' : ''}`}
          title={target.kind === 'param' ? target.title : stripTags(target.summary)}
          onClick={(e) => { e.preventDefault(); ctx.open(id) }}
        >
          {shown}
        </a>
      ) : (
        <span key={i++} className="wk-link wk-dead">{shown}</span>
      ))
    }
    last = m.index! + tok.length
  }
  if (last < text.length) out.push(<Fragment key={i++}>{text.slice(last)}</Fragment>)
  return <>{out}</>
}

function CodeBlock({ code }: { code: string }) {
  const { t } = useTranslation()
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try { await navigator.clipboard.writeText(code) } catch { /* clipboard blocked */ }
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1400)
  }
  return (
    <div className="wk-pre">
      <button type="button" className="wk-pre-copy" onClick={() => void copy()} aria-label={t('wiki.copied')}>
        {copied ? I.check : I.copy}
      </button>
      <pre>{code}</pre>
    </div>
  )
}

function Blocks({ blocks, ctx }: { blocks: WikiBlock[]; ctx: LinkCtx }) {
  return (
    <div className="wk-prose">
      {blocks.map((b, i) => {
        if ('h' in b) return <h2 key={i} id={headingId(i)}>{b.h}</h2>
        if ('p' in b) return <p key={i}><Inline text={b.p} ctx={ctx} /></p>
        if ('code' in b) return <CodeBlock key={i} code={b.code} />
        if ('note' in b) {
          return (
            <div key={i} className={`ds-note ds-${b.tone ?? 'info'}`}>
              <span><Inline text={b.note} ctx={ctx} /></span>
            </div>
          )
        }
        if ('list' in b) {
          const items = b.list.map((x, j) => <li key={j}><Inline text={x} ctx={ctx} /></li>)
          return b.ordered ? <ol key={i}>{items}</ol> : <ul key={i}>{items}</ul>
        }
        return (
          <div key={i} className="wk-tablewrap">
            <table className="ds-tbl">
              <thead><tr>{b.table.head.map((h, j) => <th key={j}><Inline text={h} ctx={ctx} /></th>)}</tr></thead>
              <tbody>
                {b.table.rows.map((r, j) => (
                  <tr key={j}>{r.map((c, k) => <td key={k}><Inline text={c} ctx={ctx} /></td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      })}
    </div>
  )
}

function Hl({ text, tokens }: { text: string; tokens: string[] }) {
  return <>{highlightParts(text, tokens).map((p, i) => p.hit ? <mark key={i} className="wk-mark">{p.s}</mark> : <Fragment key={i}>{p.s}</Fragment>)}</>
}

/** Articles first (recipes → guides → terms, by title), then parameters in form order. */
function orderSection(list: WikiEntry[], lang: string): WikiEntry[] {
  const arts = list.filter((e) => e.kind !== 'param')
    .sort((a, b) => KIND_ORDER[a.kind] - KIND_ORDER[b.kind] || a.title.localeCompare(b.title, lang))
  return [...arts, ...list.filter((e) => e.kind === 'param')]
}

// ── page ───────────────────────────────────────────────────────────────────
export default function WikiPage() {
  const { t, i18n } = useTranslation()
  const lang = i18n.language || 'en'
  const isMobile = useIsMobile()
  const [params, setParams] = useSearchParams()
  const { schema, offline } = useSchema()

  const openId = params.get('e')
  const q = params.get('q') ?? ''
  const kind = parseKind(params.get('k'))
  const section = params.get('g') ?? ''
  const styleOnly = params.get('st') === '1'
  const basicOnly = params.get('b') === '1'

  const [favorites, setFavorites] = useState<string[]>(() => store.favorites())
  const [recent, setRecent] = useState<string[]>(() => store.recent())
  const [cursor, setCursor] = useState(0)
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({})
  const [filterAt, setFilterAt] = useState<HTMLElement | null>(null)
  const [mainEl, setMainEl] = useState<HTMLElement | null>(null)
  const searchRef = useRef<HTMLInputElement | null>(null)
  const railRef = useRef<HTMLDivElement | null>(null)

  const update = useCallback((patch: Record<string, string | null>, push = false) => {
    setParams((prev) => {
      const next = new URLSearchParams(prev)
      for (const [k, v] of Object.entries(patch)) {
        if (v == null || v === '') next.delete(k)
        else next.set(k, v)
      }
      return next
    }, { replace: !push })
  }, [setParams])

  // ── entries ──
  const entries = useMemo(() => {
    const articles = lang.startsWith('ru') ? ARTICLES_RU : ARTICLES_EN
    return [...buildArticleEntries(articles), ...buildParamEntries(schema, t)]
  }, [lang, schema, t])
  const byId = useMemo(() => new Map(entries.map((e) => [e.id, e])), [entries])

  // Backlinks: which articles mention each id.
  const backlinks = useMemo(() => {
    const m = new Map<string, string[]>()
    for (const e of entries) {
      if (e.kind === 'param') continue
      for (const id of linksOf(e)) {
        if (id === e.id) continue
        const arr = m.get(id) ?? []
        arr.push(e.id)
        m.set(id, arr)
      }
    }
    return m
  }, [entries])

  /** Every section with its entries in reading order. */
  const groups = useMemo(() => {
    const m = new Map<string, WikiEntry[]>()
    for (const e of entries) {
      const arr = m.get(e.section) ?? []
      arr.push(e)
      m.set(e.section, arr)
    }
    return [...m.entries()]
      .sort((a, b) => sectionRank(a[0]) - sectionRank(b[0]) || a[0].localeCompare(b[0]))
      .map(([key, list]) => ({ key, list: orderSection(list, lang) }))
  }, [entries, lang])

  const favSet = useMemo(() => new Set(favorites), [favorites])
  const tokens = useMemo(() => tokenize(q), [q])
  const searching = tokens.length > 0
  const treeFiltered = kind !== 'all' || styleOnly || basicOnly

  const passes = useCallback((e: WikiEntry) => {
    if (kind === 'fav' && !favSet.has(e.id)) return false
    if (kind === 'article' && e.kind === 'param') return false
    if (kind === 'param' && e.kind !== 'param') return false
    if (styleOnly && !e.styleRec) return false
    if (basicOnly && e.advanced) return false
    return true
  }, [kind, favSet, styleOnly, basicOnly])

  const tree = useMemo(
    () => groups.map((g) => ({ key: g.key, list: g.list.filter(passes) })).filter((g) => g.list.length > 0),
    [groups, passes],
  )

  const results = useMemo(() => {
    if (!searching) return []
    return entries
      .filter(passes)
      .map((e) => ({ e, score: scoreEntry(e, tokens) }))
      .filter((r) => r.score > 0)
      .sort((a, b) => b.score - a.score || a.e.title.localeCompare(b.e.title, lang))
      .map((r) => r.e)
  }, [searching, entries, passes, tokens, lang])

  const kindCounts = useMemo(() => {
    const c: Record<KindFilter, number> = { all: entries.length, article: 0, param: 0, fav: favorites.filter((id) => byId.has(id)).length }
    for (const e of entries) c[e.kind === 'param' ? 'param' : 'article']++
    return c
  }, [entries, favorites, byId])

  const current = openId ? byId.get(openId) ?? null : null
  const shownSection = current ? '' : section
  const activeSection = current?.section ?? section

  const open = useCallback((id: string) => {
    update({ e: id, g: null }, true)
    setRecent(store.pushRecent(id))
  }, [update])
  const showSection = useCallback((key: string) => update({ g: key, e: null }, true), [update])
  const goHome = useCallback(() => update({ e: null, g: null }, true), [update])

  const toggleFav = useCallback((id: string) => {
    setFavorites((prev) => {
      const next = prev.includes(id) ? prev.filter((x) => x !== id) : [id, ...prev]
      store.setFavorites(next)
      return next
    })
  }, [])

  const isGroupOpen = (key: string) => openGroups[key] ?? (treeFiltered || key === activeSection)
  const onGroup = (key: string) => {
    const wasOpen = isGroupOpen(key)
    setOpenGroups((p) => ({ ...p, [key]: !wasOpen }))
    if (!wasOpen && !isMobile) showSection(key)
  }

  // The section of whatever is open is always unfolded in the tree.
  useEffect(() => {
    if (activeSection) setOpenGroups((p) => (p[activeSection] ? p : { ...p, [activeSection]: true }))
  }, [activeSection])

  // Keyboard cursor over search results; follows the open entry.
  useEffect(() => {
    const idx = current ? results.findIndex((e) => e.id === current.id) : -1
    setCursor(idx >= 0 ? idx : 0)
  }, [results, current])

  useEffect(() => {
    const rail = railRef.current
    const el = rail?.querySelector<HTMLElement>('.wk-is-cursor') ?? rail?.querySelector<HTMLElement>('.wk-leaf.ds-is-active')
    el?.scrollIntoView?.({ block: 'nearest' })
  }, [cursor, openId, tree, searching])

  useEffect(() => { mainEl?.scrollTo?.({ top: 0 }) }, [openId, shownSection, mainEl])

  // "/" focuses the search from anywhere on the page.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) return
      const el = e.target as HTMLElement | null
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)) return
      e.preventDefault()
      searchRef.current?.focus()
      searchRef.current?.select()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const onSearchKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setCursor((c) => Math.min(results.length - 1, c + 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setCursor((c) => Math.max(0, c - 1)) }
    else if (e.key === 'Enter') { const hit = results[cursor]; if (hit) open(hit.id) }
    else if (e.key === 'Escape' && q) { e.preventDefault(); update({ q: null }) }
  }

  const ctx: LinkCtx = { byId, open }
  const extraFilters = (styleOnly ? 1 : 0) + (basicOnly ? 1 : 0)

  const showRail = !isMobile || (!current && !shownSection)
  const showMain = !isMobile || !!current || !!shownSection

  const crumbs: Array<{ label: string; onClick?: () => void }> = [{ label: t('wiki.title'), onClick: current || shownSection ? goHome : undefined }]
  if (current) crumbs.push({ label: sectionLabel(current.section, t, lang), onClick: () => showSection(current.section) }, { label: current.title })
  else if (shownSection) crumbs.push({ label: sectionLabel(shownSection, t, lang) })

  return (
    <div className="wk-page fade-in">
      <PageHead
        eyebrow={t('wiki.eyebrow')}
        title={t('wiki.title')}
        subtitle={t('wiki.subtitle')}
        tools={<span className="ds-badge ds-mute">{t('wiki.entriesBadge', { count: entries.length })}</span>}
      />

      <div className="ds-scroll ds-tight" style={{ minHeight: 0 }}>
        <div className={`ds-card wk-shell${isMobile ? ' wk-mobile' : ''}`}>
          {showRail && (
            <aside className="wk-rail">
              <div className="wk-rail-head">
                <label className="ds-search wk-search">
                  {I.search}
                  <input
                    ref={searchRef}
                    className="ds-inp"
                    value={q}
                    onChange={(e) => update({ q: e.target.value })}
                    onKeyDown={onSearchKey}
                    placeholder={t('wiki.searchPlaceholder')}
                    aria-label={t('wiki.searchPlaceholder')}
                    title={t('wiki.searchHint')}
                  />
                  {q ? (
                    <button type="button" className="wk-search-x" onClick={() => update({ q: null })} aria-label={t('wiki.clearSearch')}>{I.x}</button>
                  ) : !isMobile && <span className="ds-kbd wk-search-kbd">/</span>}
                </label>
                <div className="wk-filterrow">
                  <div className="ds-seg" role="tablist" aria-label={t('wiki.title')}>
                    {KIND_FILTERS.map((k) => (
                      <button
                        key={k}
                        type="button"
                        role="tab"
                        aria-selected={kind === k}
                        aria-label={k === 'fav' ? t('wiki.kind.fav') : undefined}
                        title={k === 'fav' ? t('wiki.kind.fav') : undefined}
                        className={`ds-seg-item${kind === k ? ' ds-is-active' : ''}${k === 'fav' ? ' wk-seg-fav' : ''}`}
                        onClick={() => update({ k: k === 'all' ? null : k })}
                      >
                        {k === 'fav' ? <>{I.star(kind === 'fav', 12)}{kindCounts.fav > 0 && <span className="wk-seg-n">{kindCounts.fav}</span>}</>
                          : k === 'article' ? t('wiki.kindArticle') : t(`wiki.kind.${k}`)}
                      </button>
                    ))}
                  </div>
                  <button
                    type="button"
                    className={`ds-iconbtn wk-filterbtn${filterAt ? ' ds-is-open' : ''}${extraFilters ? ' wk-on' : ''}`}
                    aria-label={t('wiki.filters')}
                    title={t('wiki.filters')}
                    onClick={(e) => setFilterAt(filterAt ? null : e.currentTarget)}
                  >
                    {I.filter}
                    {extraFilters > 0 && <span className="wk-filterbtn-n">{extraFilters}</span>}
                  </button>
                </div>
              </div>

              <div className="wk-tree" ref={railRef}>
                {searching ? (
                  <>
                    <div className="wk-tree-cap"><span className="ds-cap">{t('wiki.listSub', { count: results.length })}</span></div>
                    {results.length === 0 ? (
                      <div className="wk-tree-empty">{t('wiki.noMatch', { q })}</div>
                    ) : (
                      <div role="listbox" aria-label={t('wiki.listTitle')}>
                        {results.map((e, i) => (
                          <button
                            key={e.id}
                            type="button"
                            role="option"
                            aria-selected={e.id === current?.id}
                            className={`wk-hit${e.id === current?.id ? ' ds-is-active' : ''}${i === cursor ? ' wk-is-cursor' : ''}`}
                            onClick={() => open(e.id)}
                          >
                            <span className="wk-hit-ico" title={t(`wiki.kindOne.${e.kind}`)}>{KIND_ICON[e.kind]}</span>
                            <span className="wk-hit-txt">
                              <span className="wk-hit-title"><Hl text={e.title} tokens={tokens} /></span>
                              {e.kind === 'param'
                                ? <span className="wk-hit-sub wk-mono"><Hl text={e.key!} tokens={tokens} /></span>
                                : <span className="wk-hit-sub"><Hl text={stripTags(e.summary)} tokens={tokens} /></span>}
                            </span>
                          </button>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <nav aria-label={t('wiki.listTitle')}>
                    {!isMobile && (
                      <button type="button" className={`wk-navrow wk-homerow${!current && !shownSection ? ' ds-is-active' : ''}`} onClick={goHome}>
                        <span className="wk-navrow-ico">{I.home}</span>
                        <span className="wk-navrow-label">{t('wiki.home')}</span>
                      </button>
                    )}
                    <div className="wk-tree-cap"><span className="ds-cap">{t('wiki.browse')}</span></div>
                    {tree.length === 0 && (
                      <div className="wk-tree-empty">{kind === 'fav' ? t('wiki.favEmpty') : t('wiki.noMatchFilters')}</div>
                    )}
                    {tree.map(({ key, list }) => {
                      const isOpen = isGroupOpen(key)
                      const firstParam = list.findIndex((e) => e.kind === 'param')
                      return (
                        <div key={key} className="wk-group">
                          <button
                            type="button"
                            className={`wk-navrow wk-group-btn${shownSection === key ? ' ds-is-active' : ''}`}
                            aria-expanded={isOpen}
                            onClick={() => onGroup(key)}
                          >
                            <span className="wk-navrow-ico">{sectionIcon(key)}</span>
                            <span className="wk-navrow-label">{sectionLabel(key, t, lang)}</span>
                            <span className="wk-navrow-n">{list.length}</span>
                            <span className={`wk-chev${isOpen ? ' wk-open' : ''}`}>{I.chev}</span>
                          </button>
                          {isOpen && (
                            <div className="wk-leaves">
                              {list.map((e, i) => (
                                <Fragment key={e.id}>
                                  {i === firstParam && firstParam > 0 && <div className="wk-leaf-cap">{t('wiki.paramsTitle')}</div>}
                                  <button
                                    type="button"
                                    className={`wk-leaf${e.id === current?.id ? ' ds-is-active' : ''}${e.kind === 'param' ? ' wk-leaf-param' : ''}`}
                                    title={e.kind === 'param' ? e.key : stripTags(e.summary)}
                                    onClick={() => open(e.id)}
                                  >
                                    <span className="wk-leaf-label">{e.title}</span>
                                    {e.styleRec && <span className="wk-leaf-dot" aria-label={t('wiki.styleRec')} />}
                                  </button>
                                </Fragment>
                              ))}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </nav>
                )}
              </div>

              {!isMobile && searching && <div className="wk-rail-foot">{t('wiki.keysTip')}</div>}
            </aside>
          )}

          {showMain && (
            <main className="wk-main" ref={setMainEl}>
              <div className="wk-bar">
                {isMobile && (
                  <button type="button" className="ds-iconbtn wk-back" onClick={() => update({ e: null, g: null }, true)} aria-label={t('wiki.back')}>{I.back}</button>
                )}
                <nav className="wk-crumbs" aria-label="breadcrumbs">
                  {crumbs.map((c, i) => (
                    <Fragment key={i}>
                      {i > 0 && <span className="wk-crumb-sep">/</span>}
                      {c.onClick ? <button type="button" onClick={c.onClick}>{c.label}</button> : <b>{c.label}</b>}
                    </Fragment>
                  ))}
                </nav>
              </div>

              <div className="wk-body">
                {offline && <div className="ds-note ds-warn wk-offline"><span>{t('wiki.schemaOffline')}</span></div>}
                {current ? (
                  <EntryDetail
                    entry={current}
                    entries={entries}
                    groups={groups}
                    backlinks={backlinks.get(current.id) ?? []}
                    ctx={ctx}
                    lang={lang}
                    scrollEl={mainEl}
                    fav={favSet.has(current.id)}
                    onFav={() => toggleFav(current.id)}
                    onSection={showSection}
                  />
                ) : shownSection ? (
                  <SectionView
                    sectionKey={shownSection}
                    list={(groups.find((g) => g.key === shownSection)?.list ?? []).filter(passes)}
                    lang={lang}
                    onOpen={open}
                  />
                ) : (
                  <Home
                    entries={entries}
                    byId={byId}
                    groups={groups}
                    favorites={favorites}
                    recent={recent}
                    lang={lang}
                    onOpen={open}
                    onSection={showSection}
                    onSearch={(s) => { update({ q: s }); searchRef.current?.focus() }}
                  />
                )}
              </div>
            </main>
          )}
        </div>
      </div>

      {filterAt && (
        <Popover anchor={filterAt} onClose={() => setFilterAt(null)} align="end" minWidth={240} ariaLabel={t('wiki.filters')}>
          <div className="ds-menu-group">{t('wiki.filters')}</div>
          <MenuItems
            onClose={() => setFilterAt(null)}
            items={[
              { label: t('wiki.onlyStyle'), checked: styleOnly, onSelect: () => update({ st: styleOnly ? null : '1' }) },
              { label: t('wiki.hideAdvanced'), checked: basicOnly, onSelect: () => update({ b: basicOnly ? null : '1' }) },
              { label: t('wiki.filtersReset'), divider: true, disabled: !treeFiltered && !q, onSelect: () => update({ q: null, k: null, st: null, b: null }) },
            ]}
          />
        </Popover>
      )}
    </div>
  )
}

// ── shared bits ────────────────────────────────────────────────────────────
function KindBadge({ kind }: { kind: WikiKind }) {
  const { t } = useTranslation()
  return <span className={`wk-kind wk-kind-${kind}`}>{KIND_ICON[kind]}{t(`wiki.kindOne.${kind}`)}</span>
}

function LinkCard({ e, onOpen }: { e: WikiEntry; onOpen: (id: string) => void }) {
  const { t } = useTranslation()
  return (
    <button type="button" className="wk-linkcard" onClick={() => onOpen(e.id)}>
      <span className="wk-linkcard-ico">{KIND_ICON[e.kind]}</span>
      <span className="wk-linkcard-txt">
        <span className="wk-linkcard-title">{e.title}</span>
        <span className={`wk-linkcard-sub${e.kind === 'param' ? ' wk-mono' : ''}`}>{e.kind === 'param' ? e.key : t(`wiki.kindOne.${e.kind}`)}</span>
      </span>
    </button>
  )
}

function ArticleCards({ list, onOpen }: { list: WikiEntry[]; onOpen: (id: string) => void }) {
  return (
    <div className="wk-cards">
      {list.map((e) => (
        <button key={e.id} type="button" className="ds-optcard wk-card" onClick={() => onOpen(e.id)}>
          <span className={`ds-optcard-ico wk-ico-${e.kind}`}>{KIND_ICON[e.kind]}</span>
          <span className="ds-optcard-txt">
            <span className="ds-optcard-name">{e.title}</span>
            <span className="ds-optcard-desc">{stripTags(e.summary)}</span>
          </span>
        </button>
      ))}
    </div>
  )
}

function ParamTable({ list, onOpen }: { list: WikiEntry[]; onOpen: (id: string) => void }) {
  const { t } = useTranslation()
  return (
    <div className="wk-tablewrap">
      <table className="ds-tbl wk-rowtbl">
        <thead><tr><th>{t('wiki.param')}</th><th>{t('wiki.fDefault')}</th><th /></tr></thead>
        <tbody>
          {list.map((e) => (
            <tr key={e.id} onClick={() => onOpen(e.id)}>
              <td>
                <span className="wk-rowtbl-title">{e.title}</span>
                <span className="wk-rowtbl-key">{e.key}</span>
              </td>
              <td className="wk-mono">{e.prop ? (e.prop.default === undefined ? '—' : fmtValue(e.prop.default)) : '—'}</td>
              <td className="wk-rowtbl-tags">
                {e.styleRec && <span className="ds-tag ds-edited">{t('wiki.styleShort')}</span>}
                {e.advanced && <span className="ds-tag ds-adv">adv</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── detail ─────────────────────────────────────────────────────────────────
function EntryDetail({ entry, entries, groups, backlinks, ctx, lang, scrollEl, fav, onFav, onSection }: {
  entry: WikiEntry
  entries: WikiEntry[]
  groups: Array<{ key: string; list: WikiEntry[] }>
  backlinks: string[]
  ctx: LinkCtx
  lang: string
  scrollEl: HTMLElement | null
  fav: boolean
  onFav: () => void
  onSection: (key: string) => void
}) {
  const { t } = useTranslation()
  const [copied, setCopied] = useState(false)
  useEffect(() => { setCopied(false) }, [entry.id])

  const copyKey = async () => {
    if (!entry.key) return
    try { await navigator.clipboard.writeText(entry.key) } catch { /* clipboard blocked: nothing to do */ }
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1400)
  }

  const prop = entry.prop
  const relations = useMemo(() => {
    if (entry.kind !== 'param' || !entry.key) return null
    const key = entry.key
    const dependsOn = [...new Set([...exprKeys(prop?.show_when), ...exprKeys(prop?.disable_when)])]
      .filter((k) => k !== key).map((k) => `param:${k}`)
    const controls = entries
      .filter((e) => e.kind === 'param' && e.key !== key && (exprKeys(e.prop?.show_when).includes(key) || exprKeys(e.prop?.disable_when).includes(key)))
      .map((e) => e.id)
    return { dependsOn, controls }
  }, [entry, entries, prop])

  const typeLabel = (() => {
    if (!prop) return null
    const k = controlKind(prop)
    if (k === 'bool' || k === 'tristate') return t('wiki.typeBool')
    if (k === 'int') return t('wiki.typeInt')
    if (k === 'float') return t('wiki.typeFloat')
    if (k === 'select') return t('wiki.typeSelect')
    if (k === 'path') return t('wiki.typePath')
    if (k.endsWith('list')) return t('wiki.typeList')
    return t('wiki.typeString')
  })()
  const min = prop?.minimum ?? prop?.exclusiveMinimum
  const max = prop?.maximum ?? prop?.exclusiveMaximum
  const range = min != null || max != null ? `${min ?? '−∞'} … ${max ?? '∞'}` : null

  const facts: Array<[string, string]> = []
  if (entry.kind === 'param' && prop) {
    if (typeLabel) facts.push([t('wiki.fType'), typeLabel])
    facts.push([t('wiki.fDefault'), prop.default === undefined ? t('wiki.noDefault') : fmtValue(prop.default)])
    if (range) facts.push([t('wiki.fRange'), range])
    facts.push([t('wiki.fMode'), prop.advanced ? t('wiki.modeAdvanced') : t('wiki.modeBasic')])
    if (prop.show_when) facts.push([t('wiki.fShownWhen'), fmtExpr(prop.show_when)])
    if (prop.disable_when) facts.push([t('wiki.fDisabledWhen'), fmtExpr(prop.disable_when)])
    if (prop.cli_alias) facts.push([t('wiki.fCli'), prop.cli_alias])
  }
  // Facts render twice: inline under the lead, and in the aside; CSS container
  // queries on the main column show exactly one of them.
  const factsBox = (inline: boolean) => facts.length > 0 && (
    <dl className={`wk-facts${inline ? ' wk-facts-inline' : ''}`}>
      {!inline && <div className="wk-aside-title"><span className="ds-cap">{t('wiki.facts')}</span></div>}
      {facts.map(([k, v]) => (
        <div key={k} className="wk-fact"><dt>{k}</dt><dd>{v}</dd></div>
      ))}
    </dl>
  )

  const headings = entry.kind === 'param' ? [] : entry.body.flatMap((b, i) => ('h' in b ? [{ id: headingId(i), text: b.h }] : []))

  const linkGroups: Array<[string, string[]]> = [
    [t('wiki.dependsOn'), relations?.dependsOn ?? []],
    [t('wiki.controls'), relations?.controls ?? []],
    [t('wiki.mentionedIn'), backlinks],
    [t('wiki.related'), entry.related],
  ]
  const realGroups = linkGroups
    .map(([label, ids]) => [label, ids.filter((id) => ctx.byId.has(id))] as const)
    .filter(([, ids]) => ids.length > 0)

  const siblings = groups.find((g) => g.key === entry.section)?.list ?? []
  const idx = siblings.findIndex((e) => e.id === entry.id)
  const prev = idx > 0 ? siblings[idx - 1] : null
  const next = idx >= 0 && idx < siblings.length - 1 ? siblings[idx + 1] : null

  const aside = entry.kind === 'param' ? factsBox(false) : headings.length >= 2 && <Toc headings={headings} scrollEl={scrollEl} />

  return (
    <div className={`wk-doc${aside ? ' wk-has-aside' : ''}`}>
      <article className="wk-doc-main">
        <header className="wk-hero">
          <div className="wk-kindline">
            <KindBadge kind={entry.kind} />
            <button type="button" className="wk-kindline-sect" onClick={() => onSection(entry.section)}>
              {sectionIcon(entry.section)}{sectionLabel(entry.section, t, lang)}
            </button>
            <button type="button" className={`ds-ctl wk-fav${fav ? ' wk-on' : ''}`} aria-pressed={fav} onClick={onFav}>
              {I.star(fav)}{fav ? t('wiki.saved') : t('wiki.save')}
            </button>
          </div>
          <h1>{entry.title}</h1>
          {entry.key && (
            <div className="wk-keyrow">
              <button type="button" className="wk-keychip" onClick={() => void copyKey()} title={t('wiki.copyKey')}>
                <span>{entry.key}</span>
                {copied ? I.check : I.copy}
              </button>
              {copied && <span className="wk-copied">{t('wiki.copied')}</span>}
              {entry.advanced && <span className="ds-tag ds-adv">{t('wiki.modeAdvanced')}</span>}
            </div>
          )}
          {entry.kind !== 'param' && <p className="wk-lead"><Inline text={entry.summary} ctx={ctx} /></p>}
        </header>

        {/* A parameter's description reads first, like the lead of an article. */}
        {entry.kind === 'param' && entry.body.length > 0 && <Blocks blocks={entry.body} ctx={ctx} />}

        {entry.styleRec && (
          <div className="wk-callout">
            <span className="wk-callout-ico">{I.spark}</span>
            <div>
              <div className="wk-callout-title">{t('wiki.styleRec')}</div>
              <div className="wk-callout-text"><Inline text={entry.styleRec} ctx={ctx} /></div>
            </div>
          </div>
        )}

        {factsBox(true)}

        {entry.kind !== 'param' && entry.body.length > 0 && <Blocks blocks={entry.body} ctx={ctx} />}

        {entry.options && entry.options.length > 0 && (
          <section className="wk-block">
            <h2 className="wk-block-title">{t('wiki.options')}</h2>
            <div className="wk-tablewrap">
              <table className="ds-tbl">
                <thead><tr><th>{t('wiki.optValue')}</th><th>{t('wiki.optLabel')}</th></tr></thead>
                <tbody>
                  {entry.options.map((o) => (
                    <tr key={o.value}>
                      <td><code className="wk-code">{o.value}</code></td>
                      <td>
                        {o.label}
                        {String(prop?.default) === o.value && <span className="ds-badge ds-mute wk-optdef">{t('wiki.optDefault')}</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {realGroups.length > 0 && (
          <section className="wk-block">
            <h2 className="wk-block-title">{t('wiki.links')}</h2>
            {realGroups.map(([label, ids]) => (
              <div key={label} className="wk-linkgroup">
                <div className="wk-linkgroup-label"><span className="ds-cap">{label}</span></div>
                <div className="wk-linkgrid">
                  {ids.map((id) => <LinkCard key={id} e={ctx.byId.get(id)!} onOpen={ctx.open} />)}
                </div>
              </div>
            ))}
          </section>
        )}

        {(prev || next) && (
          <nav className="wk-pager">
            {prev ? (
              <button type="button" className="wk-pager-btn" onClick={() => ctx.open(prev.id)}>
                <span className="wk-pager-dir">{I.left}{t('wiki.prev')}</span>
                <span className="wk-pager-title">{prev.title}</span>
              </button>
            ) : <span />}
            {next && (
              <button type="button" className="wk-pager-btn wk-next" onClick={() => ctx.open(next.id)}>
                <span className="wk-pager-dir">{t('wiki.next')}{I.right}</span>
                <span className="wk-pager-title">{next.title}</span>
              </button>
            )}
          </nav>
        )}

        {entry.source && <div className="wk-source">{t('wiki.sourceNote')}: {entry.source}</div>}
      </article>

      {aside && <aside className="wk-aside">{aside}</aside>}
    </div>
  )
}

/** "On this page" with a scroll spy over the main column. */
function Toc({ headings, scrollEl }: { headings: Array<{ id: string; text: string }>; scrollEl: HTMLElement | null }) {
  const { t } = useTranslation()
  const [active, setActive] = useState(headings[0]?.id ?? '')

  useEffect(() => {
    if (!scrollEl) return
    const onScroll = () => {
      const top = scrollEl.getBoundingClientRect().top + 90
      let cur = headings[0]?.id ?? ''
      for (const h of headings) {
        const el = document.getElementById(h.id)
        if (el && el.getBoundingClientRect().top <= top) cur = h.id
      }
      // At the very bottom the last heading may never reach the line.
      if (scrollEl.scrollTop + scrollEl.clientHeight >= scrollEl.scrollHeight - 4) cur = headings[headings.length - 1].id
      setActive(cur)
    }
    onScroll()
    scrollEl.addEventListener('scroll', onScroll, { passive: true })
    return () => scrollEl.removeEventListener('scroll', onScroll)
  }, [scrollEl, headings])

  return (
    <nav className="wk-toc" aria-label={t('wiki.onThisPage')}>
      <div className="wk-aside-title"><span className="ds-cap">{t('wiki.onThisPage')}</span></div>
      {headings.map((h) => (
        <a
          key={h.id}
          href={`#${h.id}`}
          className={h.id === active ? 'ds-is-active' : ''}
          onClick={(e) => {
            e.preventDefault()
            document.getElementById(h.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
          }}
        >
          {h.text}
        </a>
      ))}
    </nav>
  )
}

// ── section overview ───────────────────────────────────────────────────────
function SectionView({ sectionKey, list, lang, onOpen }: {
  sectionKey: string
  list: WikiEntry[]
  lang: string
  onOpen: (id: string) => void
}) {
  const { t } = useTranslation()
  const arts = list.filter((e) => e.kind !== 'param')
  const params = list.filter((e) => e.kind === 'param')
  return (
    <div className="wk-wide-page">
      <header className="wk-hero">
        <div className="wk-sect-hero">
          <span className="wk-sect-hero-ico">{sectionIcon(sectionKey)}</span>
          <div>
            <div className="ds-eyebrow">{t('wiki.sectionEyebrow')}</div>
            <h1>{sectionLabel(sectionKey, t, lang)}</h1>
            <div className="wk-hero-sub">{t('wiki.sectionStats', { a: arts.length, p: params.length })}</div>
          </div>
        </div>
      </header>
      {list.length === 0 && <div className="ds-empty">{t('wiki.sectionEmpty')}</div>}
      {arts.length > 0 && (
        <section className="wk-block">
          <h2 className="wk-block-title">{t('wiki.articlesTitle')}</h2>
          <ArticleCards list={arts} onOpen={onOpen} />
        </section>
      )}
      {params.length > 0 && (
        <section className="wk-block">
          <h2 className="wk-block-title">{t('wiki.paramsTitle')}</h2>
          <ParamTable list={params} onOpen={onOpen} />
        </section>
      )}
    </div>
  )
}

// ── home ───────────────────────────────────────────────────────────────────
const HERO_ID = 'recipe:style-lora'
const START_IDS = ['recipe:character-lora', 'guide:pipeline', 'guide:timesteps', 'guide:captions', 'recipe:vram-12gb', 'guide:overfitting']
const POPULAR = {
  en: ['learning rate', 'LoKr', 'timestep', 'EMA', 'captions', 'OOM', 'overfitting'],
  ru: ['learning rate', 'LoKr', 'timestep', 'EMA', 'капшены', 'OOM', 'переобучение'],
}

function Home({ entries, byId, groups, favorites, recent, lang, onOpen, onSection, onSearch }: {
  entries: WikiEntry[]
  byId: Map<string, WikiEntry>
  groups: Array<{ key: string; list: WikiEntry[] }>
  favorites: string[]
  recent: string[]
  lang: string
  onOpen: (id: string) => void
  onSection: (g: string) => void
  onSearch: (q: string) => void
}) {
  const { t } = useTranslation()
  const cheat = useMemo(
    () => entries.filter((e) => e.kind === 'param' && e.styleRec).sort((a, b) => sectionRank(a.section) - sectionRank(b.section)),
    [entries],
  )
  const favs = favorites.filter((id) => byId.has(id))
  const recents = recent.filter((id) => byId.has(id)).slice(0, 6)
  const ctx: LinkCtx = { byId, open: onOpen }
  const start = START_IDS.filter((id) => byId.has(id)).map((id) => byId.get(id)!)
  const hero = byId.get(HERO_ID)

  return (
    <div className="wk-wide-page">
      {hero && (
        <div className="wk-herocard">
          <span className="wk-herocard-ico">{I.spark}</span>
          <div className="wk-herocard-txt">
            <div className="wk-herocard-eyebrow">{t('wiki.heroEyebrow')}</div>
            <div className="wk-herocard-title">{hero.title}</div>
            <div className="wk-herocard-desc">{stripTags(hero.summary)}</div>
          </div>
          <div className="wk-herocard-act">
            <button type="button" className="ds-btn-primary" onClick={() => onOpen(hero.id)}>{t('wiki.heroOpen')}{I.right}</button>
            {cheat.length > 0 && (
              <button type="button" className="ds-ctl" onClick={() => document.getElementById('wk-cheat')?.scrollIntoView({ behavior: 'smooth', block: 'start' })}>
                {t('wiki.heroCheat')}
              </button>
            )}
          </div>
        </div>
      )}

      <div className="wk-popular">
        <span className="ds-cap">{t('wiki.popular')}</span>
        {(lang.startsWith('ru') ? POPULAR.ru : POPULAR.en).map((s) => (
          <button key={s} type="button" className="ds-chip wk-popchip" onClick={() => onSearch(s)}>{I.search}{s}</button>
        ))}
      </div>

      <section className="wk-block">
        <h2 className="wk-block-title">{t('wiki.homeTitle')}</h2>
        <div className="wk-block-sub">{t('wiki.homeSub')}</div>
        <ArticleCards list={start} onOpen={onOpen} />
      </section>

      {(favs.length > 0 || recents.length > 0) && (
        <div className="wk-twocol">
          {favs.length > 0 && (
            <section className="wk-block">
              <h2 className="wk-block-title">{t('wiki.favTitle')}</h2>
              <div className="wk-linkgrid wk-linkgrid-1">{favs.map((id) => <LinkCard key={id} e={byId.get(id)!} onOpen={onOpen} />)}</div>
            </section>
          )}
          {recents.length > 0 && (
            <section className="wk-block">
              <h2 className="wk-block-title">{t('wiki.recentTitle')}</h2>
              <div className="wk-linkgrid wk-linkgrid-1">{recents.map((id) => <LinkCard key={id} e={byId.get(id)!} onOpen={onOpen} />)}</div>
            </section>
          )}
        </div>
      )}

      <section className="wk-block">
        <h2 className="wk-block-title">{t('wiki.browse')}</h2>
        <div className="wk-block-sub">{t('wiki.browseSub')}</div>
        <div className="wk-tiles">
          {groups.map(({ key, list }) => {
            const arts = list.filter((e) => e.kind !== 'param').length
            return (
              <button key={key} type="button" className="wk-tile" onClick={() => onSection(key)}>
                <span className="ds-optcard-ico wk-tile-ico">{sectionIcon(key)}</span>
                <span className="wk-tile-txt">
                  <span className="wk-tile-name">{sectionLabel(key, t, lang)}</span>
                  <span className="wk-tile-sub">{t('wiki.sectionStats', { a: arts, p: list.length - arts })}</span>
                </span>
                <span className="wk-tile-go">{I.chev}</span>
              </button>
            )
          })}
        </div>
      </section>

      {cheat.length > 0 && (
        <section className="wk-block" id="wk-cheat">
          <h2 className="wk-block-title">{t('wiki.styleCheatTitle')}</h2>
          <div className="wk-block-sub">{t('wiki.styleCheatSub')}</div>
          <div className="wk-tablewrap">
            <table className="ds-tbl wk-rowtbl wk-cheat">
              <thead><tr><th>{t('wiki.param')}</th><th>{t('wiki.value')}</th></tr></thead>
              <tbody>
                {cheat.map((e) => (
                  <tr key={e.id} onClick={() => onOpen(e.id)}>
                    <td>
                      <span className="wk-rowtbl-title">{e.title}</span>
                      <span className="wk-rowtbl-key">{e.key}</span>
                    </td>
                    <td><Inline text={e.styleRec!} ctx={ctx} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
