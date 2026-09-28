/**
 * Graph — pure logic (no React): value formatting, conditional parameters,
 * filtering, sorting, grouping, empty places, "differs by one parameter",
 * and the bridge to training configs.
 *
 * Conventions
 * - A run value is absent when it is not set. "Not set" is never `No` or 0.
 * - Numbers compare with a relative tolerance, so 0.0001 and 1e-4 are equal.
 * - A parameter is *active* for a run when it is not archived and its
 *   condition (if any) holds for that run's values.
 */
import type {
  GImage, GraphTask, Param, ParamFilter, ParamOption, Run, RunStatus, RunValue, SortKey,
  Values, ViewState,
} from './types'

// ── numbers ─────────────────────────────────────────────────────────────────

/** Strip float noise (0.1 + 0.2) without ever losing a real digit. */
export function cleanNumber(n: number): number {
  return Number(n.toPrecision(12))
}

export function sameNumber(a: number, b: number): boolean {
  if (a === b) return true
  return Math.abs(a - b) <= 1e-9 * Math.max(Math.abs(a), Math.abs(b))
}

export function sameValue(a: RunValue | undefined, b: RunValue | undefined): boolean {
  if (typeof a === 'number' && typeof b === 'number') return sameNumber(a, b)
  return a === b
}

/** Full-precision decimal: 0.00002 stays "0.00002", never "2e-5". */
export function formatNumber(n: number): string {
  if (!Number.isFinite(n)) return String(n)
  if (Number.isInteger(n)) return String(n)
  return cleanNumber(n).toLocaleString('en-US', { useGrouping: false, maximumSignificantDigits: 15 })
}

/** Scientific form, shown next to small values ("1e-4"). */
export function scientific(n: number): string | null {
  if (n === 0 || Math.abs(n) >= 0.001 || Number.isInteger(n)) return null
  const [m, e] = cleanNumber(n).toExponential().split('e')
  const mant = m.replace(/\.?0+$/, '')
  return `${mant}e${Number(e)}`
}

/** Accepts "0.0001", "1e-4", "1,5", " 2E-5 ". Returns null when not a number. */
export function parseNumber(s: string): number | null {
  const t = s.trim().replace(',', '.')
  if (!t) return null
  const n = Number(t)
  return Number.isFinite(n) ? cleanNumber(n) : null
}

/** Inclusive range from..to by step, rounded so 1e-5 steps don't drift. */
export function numberRange(from: number, to: number, step: number, cap = 200): number[] {
  if (!(step > 0) || !Number.isFinite(from) || !Number.isFinite(to)) return []
  const out: number[] = []
  const dir = to >= from ? 1 : -1
  for (let i = 0; i < cap; i++) {
    const v = cleanNumber(from + dir * i * step)
    if (dir > 0 ? v > to + step * 1e-9 : v < to - step * 1e-9) break
    out.push(v)
  }
  return out
}

export function newId(prefix: string): string {
  return `${prefix}${Date.now().toString(36).slice(-4)}${Math.random().toString(36).slice(2, 7)}`
}

// ── parameters ──────────────────────────────────────────────────────────────

export function paramById(params: Param[]): Map<string, Param> {
  return new Map(params.map((p) => [p.id, p]))
}

/** Parameters shown in the UI (not archived, not hidden), in board order. */
export function visibleParams(params: Param[]): Param[] {
  return params.filter((p) => !p.archived && !p.hidden)
}

/** Does `param` apply given these values? Follows condition chains. */
export function isActive(param: Param, values: Values, byId: Map<string, Param>, depth = 0): boolean {
  if (param.archived) return false
  const c = param.condition
  if (!c || !c.param) return true
  const parent = byId.get(c.param)
  if (!parent || depth > 8) return true
  if (!isActive(parent, values, byId, depth + 1)) return false
  const v = values[parent.id]
  return v !== undefined && c.options.some((o) => sameValue(o as RunValue, v))
}

/** Values restricted to parameters that are active for them. */
export function activeValues(values: Values, params: Param[]): Values {
  const byId = paramById(params)
  const out: Values = {}
  for (const p of params) {
    if (values[p.id] !== undefined && isActive(p, values, byId)) out[p.id] = values[p.id]
  }
  return out
}

export function optionOf(param: Param, v: RunValue | undefined): ParamOption | undefined {
  if (v === undefined) return undefined
  if (param.type === 'number') return param.options.find((o) => o.value !== undefined && sameValue(o.value, v))
  return param.options.find((o) => o.id === v)
}

/** Every value a parameter can take in a picker: its options (not archived)
 *  plus, for numbers, any value some card already uses. Numbers sort ascending. */
export function knownValues(param: Param, runs: Run[], includeArchived = false): RunValue[] {
  if (param.type === 'text') {
    const set = new Set<string>()
    for (const r of runs) {
      const v = r.values[param.id]
      if (typeof v === 'string' && v) set.add(v)
    }
    return [...set].sort()
  }
  if (param.type === 'number') {
    const vals: number[] = []
    const push = (n: number) => { if (!vals.some((x) => sameNumber(x, n))) vals.push(n) }
    for (const o of param.options) if (o.value !== undefined && (includeArchived || !o.archived)) push(o.value)
    for (const r of runs) { const v = r.values[param.id]; if (typeof v === 'number') push(v) }
    return vals.sort((a, b) => a - b)
  }
  return param.options.filter((o) => includeArchived || !o.archived).map((o) => o.id)
}

// ── filtering ───────────────────────────────────────────────────────────────

export const EMPTY_FILTER: ParamFilter = {}

export function filterIsSet(f: ParamFilter | undefined): boolean {
  if (!f) return false
  return !!(f.values?.length || f.unset || f.min != null || f.max != null || (f.text && f.text.trim()))
}

export function matchesParamFilter(param: Param, run: Run, f: ParamFilter, byId: Map<string, Param>): boolean {
  if (!filterIsSet(f)) return true
  const active = isActive(param, run.values, byId)
  const v = active ? run.values[param.id] : undefined
  if (v === undefined) return !!f.unset
  if (f.values?.length && !f.values.some((x) => sameValue(x, v))) return false
  if (typeof v === 'number') {
    if (f.min != null && v < f.min && !sameNumber(v, f.min)) return false
    if (f.max != null && v > f.max && !sameNumber(v, f.max)) return false
  }
  if (f.text && f.text.trim() && !String(v).toLowerCase().includes(f.text.trim().toLowerCase())) return false
  // "only unset" when unset is the sole criterion
  if (f.unset && !f.values?.length && f.min == null && f.max == null && !(f.text && f.text.trim())) return false
  return true
}

export function filterRuns(runs: Run[], params: Param[], view: ViewState): Run[] {
  const byId = paramById(params)
  const q = view.search.trim().toLowerCase()
  const filtered = params.filter((p) => !p.archived && filterIsSet(view.filters[p.id]))
  return runs.filter((r) => {
    if (q && !`${r.name}\n${r.notes}\n${r.tags.join(' ')}`.toLowerCase().includes(q)) return false
    if (view.statuses.length && !view.statuses.includes(r.status)) return false
    if (view.samples === 'with' && r.images.length === 0) return false
    if (view.samples === 'without' && r.images.length > 0) return false
    if (view.favorites && !r.favorite) return false
    if (view.minRating > 0 && (r.rating ?? 0) < view.minRating) return false
    return filtered.every((p) => matchesParamFilter(p, r, view.filters[p.id], byId))
  })
}

// ── sorting ─────────────────────────────────────────────────────────────────

const STATUS_ORDER: Record<RunStatus, number> = { training: 0, planned: 1, done: 2, stopped: 3 }

/** Rank of a value inside its parameter: option order for lists, the number
 *  itself for numbers, text for text. Unset sorts last in both directions. */
function valueRank(param: Param, v: RunValue | undefined): number | string | null {
  if (v === undefined) return null
  if (param.type === 'number') return typeof v === 'number' ? v : Number(v)
  if (param.type === 'text') return String(v).toLowerCase()
  const i = param.options.findIndex((o) => o.id === v)
  return i < 0 ? 1e9 : i
}

export function compareRuns(a: Run, b: Run, keys: SortKey[], byId: Map<string, Param>): number {
  for (const k of keys) {
    const sign = k.dir === 'desc' ? -1 : 1
    let x: number | string | null
    let y: number | string | null
    switch (k.key) {
      case '_name': x = a.name.toLowerCase(); y = b.name.toLowerCase(); break
      case '_created': x = a.created_at; y = b.created_at; break
      case '_rating': x = a.rating ?? 0; y = b.rating ?? 0; break
      case '_status': x = STATUS_ORDER[a.status]; y = STATUS_ORDER[b.status]; break
      case '_samples': x = a.images.length; y = b.images.length; break
      default: {
        const p = byId.get(k.key)
        if (!p) continue
        x = isActive(p, a.values, byId) ? valueRank(p, a.values[p.id]) : null
        y = isActive(p, b.values, byId) ? valueRank(p, b.values[p.id]) : null
      }
    }
    if (x === null && y === null) continue
    if (x === null) return 1
    if (y === null) return -1
    if (x < y) return -sign
    if (x > y) return sign
  }
  return a.created_at - b.created_at
}

export function sortRuns(runs: Run[], keys: SortKey[], params: Param[]): Run[] {
  const byId = paramById(params)
  return [...runs].sort((a, b) => compareRuns(a, b, keys, byId))
}

// ── combinations (a group of runs that share values) ────────────────────────

/** A place on the board: fixed values plus the runs that sit there. Empty
 *  runs = an untried combination. */
export interface Combo {
  key: string
  values: Values
  runs: Run[]
}

export function valuesKey(values: Values, paramIds: string[]): string {
  return paramIds.map((id) => {
    const v = values[id]
    if (v === undefined) return '∅'
    return typeof v === 'number' ? `n${cleanNumber(v)}` : `s${v}`
  }).join('|')
}

/** Runs that have exactly the same active values sit together. */
export function combosOfIdentical(runs: Run[], params: Param[]): Combo[] {
  const ids = params.filter((p) => !p.archived).map((p) => p.id)
  const map = new Map<string, Combo>()
  for (const r of runs) {
    const vals = activeValues(r.values, params)
    const key = valuesKey(vals, ids)
    const c = map.get(key)
    if (c) c.runs.push(r)
    else map.set(key, { key, values: vals, runs: [r] })
  }
  return [...map.values()]
}

/** The parameters a gap comparison spans: every non-archived parameter the
 *  filter narrows to explicit values (1 value = fixed, several = varied). */
export function comparisonParams(params: Param[], view: ViewState): Param[] {
  return params.filter((p) => !p.archived && p.type !== 'text' && (view.filters[p.id]?.values?.length ?? 0) > 0)
}

export const GAP_LIMIT = 400

/**
 * Every combination of the filtered values, conditional parameters expanded
 * only where their condition holds (Logit-normal gets no SNR variants). Each
 * combination collects the runs that match it; the rest are empty places.
 * Returns null when the product is larger than GAP_LIMIT.
 */
export function buildGaps(runs: Run[], params: Param[], view: ViewState): Combo[] | null {
  const byId = paramById(params)
  const axes = comparisonParams(params, view)
  let combos: Values[] = [{}]
  for (const p of axes) {
    const choices = view.filters[p.id]!.values!
    const next: Values[] = []
    for (const c of combos) {
      if (!isActive(p, c, byId)) { next.push(c); continue }
      for (const v of choices) next.push({ ...c, [p.id]: v })
    }
    combos = next
    if (combos.length > GAP_LIMIT) return null
  }
  const ids = axes.map((p) => p.id)
  const out: Combo[] = combos.map((values) => ({ key: valuesKey(values, ids), values, runs: [] }))
  for (const r of runs) {
    for (const c of out) {
      if (ids.every((id) => {
        const p = byId.get(id)!
        const want = c.values[id]
        const active = isActive(p, r.values, byId)
        if (want === undefined) return !active || r.values[id] === undefined
        return active && sameValue(r.values[id], want)
      })) { c.runs.push(r); break }
    }
  }
  return out
}

// ── grouping ────────────────────────────────────────────────────────────────

export interface Group<T> {
  key: string
  paramId: string
  value: RunValue | undefined
  items: T[]
  children: Group<T>[] | null
}

/** Nested groups in `groupBy` order. Value order follows the parameter
 *  (option order / ascending numbers); "not set" comes last. */
export function groupItems<T>(
  items: T[], groupBy: string[], params: Param[], valuesOf: (t: T) => Values, prefix = '',
): Group<T>[] | null {
  const byId = paramById(params)
  const ids = groupBy.filter((id) => byId.has(id) && !byId.get(id)!.archived)
  if (!ids.length) return null
  const [head, ...rest] = ids
  const p = byId.get(head)!
  const buckets = new Map<string, { value: RunValue | undefined; items: T[] }>()
  for (const it of items) {
    const vals = valuesOf(it)
    const v = isActive(p, vals, byId) ? vals[head] : undefined
    const k = v === undefined ? '∅' : typeof v === 'number' ? `n${cleanNumber(v)}` : `s${v}`
    const b = buckets.get(k)
    if (b) b.items.push(it)
    else buckets.set(k, { value: v, items: [it] })
  }
  const ordered = [...buckets.entries()].sort(([, a], [, b]) => {
    const x = valueRank(p, a.value)
    const y = valueRank(p, b.value)
    if (x === null) return 1
    if (y === null) return -1
    return x < y ? -1 : x > y ? 1 : 0
  })
  return ordered.map(([k, b]) => {
    const key = `${prefix}${head}=${k}`
    return {
      key, paramId: head, value: b.value, items: b.items,
      children: groupItems(b.items, rest, params, valuesOf, `${key}/`),
    }
  })
}

// ── differences ─────────────────────────────────────────────────────────────

/**
 * Parameters whose value differs between two runs. Name, notes and status
 * never count. A dependent parameter that only appears / disappears because
 * its condition parameter changed is folded into that change (Sampling
 * switching to Logit-normal drops SNR — one difference, not three).
 */
export function differences(a: Values, b: Values, params: Param[]): string[] {
  const byId = paramById(params)
  const out: string[] = []
  for (const p of params) {
    if (p.archived) continue
    const ia = isActive(p, a, byId)
    const ib = isActive(p, b, byId)
    if (!ia && !ib) continue
    if (ia !== ib) {
      // Appeared / vanished with its condition: the parent already counts.
      if (p.condition && byId.has(p.condition.param)) continue
      out.push(p.id)
      continue
    }
    if (!sameValue(a[p.id], b[p.id])) out.push(p.id)
  }
  return out
}

/** Runs that differ from `run` in exactly one parameter, with that parameter. */
export function oneStepAway(run: Run, runs: Run[], params: Param[]): { run: Run; param: string }[] {
  const out: { run: Run; param: string }[] = []
  for (const r of runs) {
    if (r.id === run.id) continue
    const d = differences(run.values, r.values, params)
    if (d.length === 1) out.push({ run: r, param: d[0] })
  }
  return out
}

/** Parameters whose values are not identical across all given value sets. */
export function varyingParams(sets: Values[], params: Param[]): string[] {
  if (sets.length < 2) return []
  const seen = new Set<string>()
  for (let i = 1; i < sets.length; i++) {
    for (const id of differences(sets[0], sets[i], params)) seen.add(id)
  }
  // A parameter can vary between items 1 and 2 without differing from item 0.
  for (let i = 1; i < sets.length; i++) {
    for (let j = i + 1; j < sets.length; j++) {
      for (const id of differences(sets[i], sets[j], params)) seen.add(id)
    }
  }
  return params.filter((p) => seen.has(p.id)).map((p) => p.id)
}

// ── training config bridge ──────────────────────────────────────────────────

export function sameConfigValue(actual: unknown, want: unknown): boolean {
  if (want !== null && typeof want === 'object' && !Array.isArray(want)) {
    const op = want as Record<string, unknown>
    if ('$truthy' in op) {
      const truthy = Array.isArray(actual) ? actual.length > 0 : !!actual
      return truthy === !!op.$truthy
    }
    if ('$gt' in op) return typeof actual === 'number' && actual > Number(op.$gt)
    if ('$gte' in op) return typeof actual === 'number' && actual >= Number(op.$gte)
    return false
  }
  if (typeof want === 'number' && typeof actual === 'number') return sameNumber(want, actual)
  if (Array.isArray(want) && Array.isArray(actual)) {
    return want.length === actual.length && want.every((w, i) => sameConfigValue(actual[i], w))
  }
  return actual === want
}

function matchPatch(opt: ParamOption): Record<string, unknown> | null {
  const c = opt.config
  if (!c) return null
  const m = c.match ?? c.apply
  return m && Object.keys(m).length ? m : null
}

/** Is this parameter tied to the training config at all? */
export function isLinked(p: Param): boolean {
  if (p.type === 'number') return !!p.config?.key
  if (p.type === 'text') return false
  return p.options.some((o) => matchPatch(o) !== null)
}

/** Can the Graph write this parameter into a config (not only read it)? */
export function isApplicable(p: Param): boolean {
  if (p.type === 'number') return !!p.config?.key
  if (p.type === 'text') return false
  return p.options.some((o) => o.config && Object.keys(o.config.apply ?? {}).length > 0)
}

/** Config keys the Graph needs from a task to recognise it. */
export function configKeys(params: Param[]): string[] {
  const keys = new Set<string>()
  for (const p of params) {
    if (p.config?.key) keys.add(p.config.key)
    for (const o of p.options) {
      for (const k of Object.keys(o.config?.apply ?? {})) keys.add(k)
      for (const k of Object.keys(o.config?.match ?? {})) keys.add(k)
    }
  }
  return [...keys].sort()
}

/** Read a task config back into Graph values (only what can be recognised). */
export function valuesFromConfig(cfg: Record<string, unknown>, params: Param[]): Values {
  const out: Values = {}
  for (const p of params) {
    if (p.archived) continue
    if (p.type === 'number' && p.config?.key) {
      let v = cfg[p.config.key]
      if (Array.isArray(v)) v = v.length === 1 ? v[0] : undefined
      if (typeof v === 'number' && Number.isFinite(v)) out[p.id] = cleanNumber(v)
      continue
    }
    if (p.type === 'list' || p.type === 'bool') {
      let best: ParamOption | null = null
      let bestSize = 0
      for (const o of p.options) {
        const m = matchPatch(o)
        if (!m) continue
        const keys = Object.keys(m)
        if (keys.every((k) => sameConfigValue(cfg[k], m[k])) && keys.length > bestSize) {
          best = o
          bestSize = keys.length
        }
      }
      if (best) out[p.id] = best.id
    }
  }
  // Drop values whose condition does not hold (SNR under Logit-normal).
  return activeValues(out, params)
}

/** The config patch for a set of values, plus the parameters it can't write. */
export function configPatch(values: Values, params: Param[]): { patch: Record<string, unknown>; skipped: string[] } {
  const patch: Record<string, unknown> = {}
  const skipped: string[] = []
  const active = activeValues(values, params)
  for (const p of params) {
    const v = active[p.id]
    if (v === undefined) continue
    if (p.type === 'number' && p.config?.key && typeof v === 'number') {
      patch[p.config.key] = p.config.list ? [v] : v
    } else if (p.type === 'list' || p.type === 'bool') {
      const o = optionOf(p, v)
      const apply = o?.config?.apply
      if (apply && Object.keys(apply).length) Object.assign(patch, apply)
      else skipped.push(p.id)
    } else {
      skipped.push(p.id)
    }
  }
  return { patch, skipped }
}

/**
 * Does a task fit a place? Every *linked* value the place fixes must be what
 * the task ran with. Unlinked parameters can't be checked and are ignored;
 * a place with nothing checkable matches nothing.
 */
export function taskMatches(place: Values, taskValues: Values, params: Param[]): boolean {
  const byId = paramById(params)
  let checked = 0
  for (const [id, want] of Object.entries(place)) {
    const p = byId.get(id)
    if (!p || p.archived || !isLinked(p)) continue
    if (!isActive(p, place, byId)) continue
    checked++
    if (!sameValue(taskValues[id], want)) return false
  }
  return checked > 0
}

/** Queue status → card status. */
export function statusFromTask(status: string): RunStatus {
  if (status === 'running') return 'training'
  if (status === 'done') return 'done'
  if (status === 'failed' || status === 'canceled' || status === 'paused') return 'stopped'
  return 'planned' // pending / scheduled
}

export function taskValuesMap(tasks: GraphTask[], params: Param[]): Map<number, Values> {
  return new Map(tasks.map((t) => [t.id, valuesFromConfig(t.config, params)]))
}

// ── images ──────────────────────────────────────────────────────────────────

/** Cover image of a card: best step, then best rated, then the latest step. */
export function coverImage(run: Run): GImage | null {
  if (!run.images.length) return null
  if (run.best_step != null) {
    const at = run.images.filter((i) => i.step === run.best_step)
    if (at.length) return at[0]
  }
  const rated = [...run.images].filter((i) => i.rating).sort((a, b) => (b.rating ?? 0) - (a.rating ?? 0))
  if (rated.length) return rated[0]
  return [...run.images].sort((a, b) => (b.step ?? -1) - (a.step ?? -1) || b.sort - a.sort)[0]
}

/** Step hidden in a file name: step_400, s400, lora-000400.png, "400steps". */
export function stepFromFilename(name: string): number | null {
  const base = name.replace(/\.[a-z0-9]+$/i, '')
  const m = base.match(/(?:^|[^a-z])(?:step|steps|s)[_-]?(\d{1,7})(?!\d)/i)
    ?? base.match(/(\d{1,7})[_-]?steps?\b/i)
    ?? base.match(/[-_](\d{4,7})$/)
  return m ? Number(m[1]) : null
}

/** Best run of a set: rating, then favourite, then newest. */
export function bestRun(runs: Run[]): Run | null {
  if (!runs.length) return null
  return [...runs].sort((a, b) =>
    (b.rating ?? 0) - (a.rating ?? 0)
    || Number(b.favorite) - Number(a.favorite)
    || b.images.length - a.images.length
    || b.created_at - a.created_at)[0]
}

// ── defaults ────────────────────────────────────────────────────────────────

export function defaultView(params: Param[]): ViewState {
  const shown = ['lora_type', 'rank_alpha', 'optimizer', 'scheduler', 'lr', 'steps']
    .filter((id) => params.some((p) => p.id === id))
  return {
    mode: 'cards',
    filters: {},
    search: '',
    statuses: [],
    samples: 'any',
    favorites: false,
    minRating: 0,
    groupBy: [],
    sort: [{ key: '_created', dir: 'desc' }],
    shownParams: shown.length ? shown : visibleParams(params).slice(0, 5).map((p) => p.id),
    thumb: 180,
    gaps: false,
    collapsed: [],
    matrixX: params.some((p) => p.id === 'lr') ? 'lr' : null,
    matrixY: params.some((p) => p.id === 'scheduler') ? 'scheduler' : null,
    webColor: params.some((p) => p.id === 'lora_type') ? 'lora_type' : null,
  }
}

/** Fill in fields an older saved view may lack. */
export function normalizeView(v: Partial<ViewState> | null | undefined, params: Param[]): ViewState {
  const d = defaultView(params)
  if (!v || typeof v !== 'object') return d
  return {
    ...d,
    ...v,
    filters: v.filters && typeof v.filters === 'object' ? v.filters : {},
    statuses: Array.isArray(v.statuses) ? v.statuses : [],
    groupBy: Array.isArray(v.groupBy) ? v.groupBy : [],
    sort: Array.isArray(v.sort) && v.sort.length ? v.sort : d.sort,
    shownParams: Array.isArray(v.shownParams) ? v.shownParams : d.shownParams,
    collapsed: Array.isArray(v.collapsed) ? v.collapsed : [],
    thumb: typeof v.thumb === 'number' ? Math.max(110, Math.min(360, v.thumb)) : d.thumb,
  }
}

export function viewHasFilters(v: ViewState): boolean {
  return !!(v.search.trim() || v.statuses.length || v.samples !== 'any' || v.favorites || v.minRating
    || Object.values(v.filters).some(filterIsSet))
}
