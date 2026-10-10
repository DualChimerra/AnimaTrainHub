// Wiki (experimental) — data model, entry building and search.
//
// Two sources feed the encyclopedia:
//  * parameters — built live from GET /api/schema plus the schema.* strings the
//    training form already uses (label, description, style recommendation), so
//    the wiki can never drift from the form;
//  * articles — hand-written guides, recipes and glossary terms (content.*.ts).
import type { TFunction } from 'i18next'
import type { SchemaProperty, SchemaResponse } from '../../api/client'
import { SCHEMA_GROUP_LABEL_KEYS, schemaEnumLabel, schemaFieldLabel } from '../../lib/schema'

export type WikiKind = 'param' | 'guide' | 'recipe' | 'term'

/** Rich text blocks. Strings support `code`, **bold** and [[id]] / [[id|label]] links. */
export type WikiBlock =
  | { h: string }
  | { p: string }
  | { list: string[]; ordered?: boolean }
  | { table: { head: string[]; rows: string[][] } }
  | { code: string }
  | { note: string; tone?: 'info' | 'warn' | 'ok' }

/** A hand-written article as stored in content.*.ts. */
export interface WikiArticle {
  id: string
  kind: Exclude<WikiKind, 'param'>
  title: string
  summary: string
  /** Topic key — the same keys as schema groups, so filters cover both. */
  section: string
  /** Extra search words (synonyms, other spellings, abbreviations). */
  aliases?: string[]
  styleRec?: string
  body: WikiBlock[]
  related?: string[]
  source?: string
}

export interface ParamOption {
  value: string
  label: string
}

export interface WikiEntry {
  id: string
  kind: WikiKind
  title: string
  summary: string
  section: string
  /** Config key for parameters. */
  key?: string
  aliases: string[]
  styleRec?: string
  body: WikiBlock[]
  related: string[]
  source?: string
  // parameter facts
  prop?: SchemaProperty
  options?: ParamOption[]
  advanced?: boolean
  /** Lower-cased haystack, built once. */
  hay: { title: string; key: string; aliases: string; summary: string; body: string }
}

/** Section keys in the order the training form shows them, plus wiki-only topics. */
export const SECTION_ORDER = [
  'basics',
  'model',
  'dataset',
  'caption',
  'lora',
  'training',
  'noise_augmentation',
  'timestep_sampling',
  'loss',
  'system',
  'output',
  'sample',
  'eval_validation',
  'monitor',
  'troubleshooting',
] as const

const EXTRA_SECTION_LABELS: Record<string, { en: string; ru: string }> = {
  basics: { en: 'Basics', ru: 'Основы' },
  troubleshooting: { en: 'Troubleshooting', ru: 'Решение проблем' },
}

export function sectionLabel(key: string, t: TFunction, lang: string): string {
  const extra = EXTRA_SECTION_LABELS[key]
  if (extra) return lang.startsWith('ru') ? extra.ru : extra.en
  const k = SCHEMA_GROUP_LABEL_KEYS[key]
  return k ? t(k) : t('wiki.otherSection')
}

export function sectionRank(key: string): number {
  const i = (SECTION_ORDER as readonly string[]).indexOf(key)
  return i < 0 ? SECTION_ORDER.length : i
}

/** Hidden schema fields that are still worth documenting (set elsewhere in the UI). */
const HIDDEN_ALLOW = new Set(['trigger_word'])

export function stripTags(s: string): string {
  return s.replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, '$2').replace(/\[\[([^\]]+)\]\]/g, (_, id: string) => id.replace(/^\w+:/, '')).replace(/[`*]/g, '')
}

function blockStrings(b: WikiBlock): string[] {
  if ('h' in b) return [b.h]
  if ('p' in b) return [b.p]
  if ('list' in b) return b.list
  if ('table' in b) return [...b.table.head, ...b.table.rows.flat()]
  if ('code' in b) return [b.code]
  return [b.note]
}

function blocksText(blocks: WikiBlock[]): string {
  return stripTags(blocks.flatMap(blockStrings).join(' '))
}

const norm = (s: string) => s.toLowerCase().replace(/ё/g, 'е')

function withHay(e: Omit<WikiEntry, 'hay'>): WikiEntry {
  return {
    ...e,
    hay: {
      title: norm(e.title),
      key: norm(e.key ?? ''),
      aliases: norm(e.aliases.join(' ')),
      summary: norm(stripTags(e.summary)),
      body: norm(blocksText(e.body) + ' ' + (e.styleRec ?? '')),
    },
  }
}

/** First sentence-ish chunk, for the list's one-line summary. */
function firstSentence(s: string): string {
  const m = s.match(/^(.{20,180}?[.!?;:])(\s|$)/)
  return (m ? m[1] : s).replace(/[.;:]$/, '')
}

/** Parameters from the live schema (or, offline, from the description strings alone). */
export function buildParamEntries(schema: SchemaResponse | null, t: TFunction): WikiEntry[] {
  const props: Record<string, SchemaProperty> = schema?.schema.properties ?? {}
  const names = schema
    ? Object.keys(props).filter((n) => !props[n].hidden || HIDDEN_ALLOW.has(n))
    : Object.keys((t('schema.descriptions', { returnObjects: true }) as Record<string, string>) ?? {})

  return names.map((name) => {
    const prop = props[name] as SchemaProperty | undefined
    const desc = t(`schema.descriptions.${name}`, { defaultValue: '' }) || prop?.description || ''
    const alt = prop?.alt_description
      ? t(`schema.altDescriptions.${name}`, { defaultValue: '' }) || prop.alt_description
      : ''
    const rec = t(`schema.recommend.${name}`, { defaultValue: '' }) || undefined
    const options = prop?.enum?.map((v) => ({ value: String(v), label: schemaEnumLabel(name, v, t) }))
    const body: WikiBlock[] = []
    if (desc) body.push({ p: desc })
    if (alt && prop?.alt_description_when) body.push({ note: `${t('wiki.whenAlt', { cond: prop.alt_description_when })}: ${alt}`, tone: 'info' })
    return withHay({
      id: `param:${name}`,
      kind: 'param',
      title: schemaFieldLabel(name, t),
      summary: firstSentence(desc.replace(/^\[[^\]]+\]\s*/, '')),
      section: prop?.group ?? 'other',
      key: name,
      aliases: [name.replace(/_/g, ' '), prop?.cli_alias ?? '', ...(options?.flatMap((o) => [o.value, o.label]) ?? [])],
      styleRec: rec,
      body,
      related: [],
      prop,
      options,
      advanced: !!prop?.advanced,
    })
  })
}

export function buildArticleEntries(articles: WikiArticle[]): WikiEntry[] {
  return articles.map((a) => withHay({
    id: a.id,
    kind: a.kind,
    title: a.title,
    summary: a.summary,
    section: a.section,
    aliases: a.aliases ?? [],
    styleRec: a.styleRec,
    body: a.body,
    related: a.related ?? [],
    source: a.source,
  }))
}

/** Every [[id]] mentioned in an entry (body, summary, related). */
export function linksOf(e: WikiEntry): string[] {
  const text = [e.summary, e.styleRec ?? '', ...e.body.flatMap(blockStrings)].join(' ')
  const ids = new Set<string>(e.related)
  for (const m of text.matchAll(/\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)) ids.add(m[1])
  return [...ids]
}

/** Keys referenced by a show_when / disable_when expression. */
export function exprKeys(expr: string | undefined): string[] {
  if (!expr) return []
  return [...new Set(expr.split(/\|\||&&/).map((c) => c.split(/==|!=/)[0].trim()).filter(Boolean))]
}

// ── search ────────────────────────────────────────────────────────────────

export function tokenize(q: string): string[] {
  return norm(q).split(/[\s,]+/).map((s) => s.trim()).filter(Boolean)
}

/** Score an entry for a query; 0 = no match. Every token has to hit somewhere. */
export function scoreEntry(e: WikiEntry, tokens: string[]): number {
  if (tokens.length === 0) return 1
  const { title, key, aliases, summary, body } = e.hay
  let score = 0
  for (const tok of tokens) {
    const underscored = tok.replace(/\s+/g, '_')
    let s = 0
    if (key && (key === tok || key === underscored)) s = 120
    else if (title === tok) s = 110
    else if (title.startsWith(tok)) s = 70
    else if (key.startsWith(tok)) s = 65
    else if (title.includes(tok)) s = 50
    else if (key.includes(tok)) s = 45
    else if (aliases.includes(tok)) s = 35
    else if (summary.includes(tok)) s = 20
    else if (body.includes(tok)) s = 8
    if (s === 0) return 0
    score += s
  }
  // Articles explain things in depth; nudge them above a stray parameter hit.
  if (e.kind !== 'param') score += 4
  return score
}

/** Wrap matches of the query tokens in <mark>. Returns string parts. */
export function highlightParts(text: string, tokens: string[]): Array<{ s: string; hit: boolean }> {
  if (tokens.length === 0 || !text) return [{ s: text, hit: false }]
  const lower = norm(text)
  const marks = new Array<boolean>(text.length).fill(false)
  for (const tok of tokens) {
    if (!tok) continue
    let i = lower.indexOf(tok)
    while (i >= 0) {
      for (let j = i; j < i + tok.length; j++) marks[j] = true
      i = lower.indexOf(tok, i + tok.length)
    }
  }
  const parts: Array<{ s: string; hit: boolean }> = []
  for (let i = 0; i < text.length; i++) {
    const last = parts[parts.length - 1]
    if (last && last.hit === marks[i]) last.s += text[i]
    else parts.push({ s: text[i], hit: marks[i] })
  }
  return parts
}

// ── persisted per-viewer conveniences ──────────────────────────────────────

const FAV_KEY = 'studio.wiki.favorites'
const RECENT_KEY = 'studio.wiki.recent'

function readList(key: string): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(key) ?? '[]')
    return Array.isArray(v) ? v.filter((x) => typeof x === 'string') : []
  } catch { return [] }
}
function writeList(key: string, v: string[]) {
  try { localStorage.setItem(key, JSON.stringify(v)) } catch { /* private mode */ }
}

export const store = {
  favorites: () => readList(FAV_KEY),
  setFavorites: (v: string[]) => writeList(FAV_KEY, v),
  recent: () => readList(RECENT_KEY),
  pushRecent: (id: string) => {
    const next = [id, ...readList(RECENT_KEY).filter((x) => x !== id)].slice(0, 8)
    writeList(RECENT_KEY, next)
    return next
  },
}
