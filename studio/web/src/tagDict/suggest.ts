/** Tag autocomplete — pure functions.
 *
 * 1. extractCurrentToken: from the input text and cursor position, the token
 *    being typed and its range in the string (used to splice the pick in).
 *    Commas and newlines (multi-line textareas) separate tokens.
 * 2. findSuggestions: prefix matches first, then substring matches, each group
 *    in list order (the default source is sorted by post count).
 */
import type { TagSuggestion } from './types'

const SEPARATORS = new Set([',', '\n'])

export interface ExtractedToken {
  /** The trimmed query text (what findSuggestions gets). */
  token: string
  /** Start index of the token's slot in the string (leading spaces included). */
  start: number
  /** End index (exclusive; trailing spaces included, the separator not). */
  end: number
}

/** Scan from the cursor to the nearest separator on both sides.
 *
 * The range includes the whitespace around the token, so a pick replaces it
 * too and no double spaces are left. The separators themselves stay. */
export function extractCurrentToken(value: string, cursor: number): ExtractedToken {
  const cur = Math.max(0, Math.min(cursor, value.length))
  let start = cur
  while (start > 0 && !SEPARATORS.has(value[start - 1])) start--
  let end = cur
  while (end < value.length && !SEPARATORS.has(value[end])) end++
  const raw = value.slice(start, end)
  const token = raw.trim()
  return { token, start, end }
}

/** The slice of the store findSuggestions needs; separate for tests. */
export interface SuggestStore {
  tagKeys: string[]
  /** tagKeys without spaces / underscores, same indices (precomputed on load). */
  compactedKeys: string[]
}

/** Candidates for a query token; an empty token or empty list → []. */
export function findSuggestions(
  rawToken: string,
  store: SuggestStore,
  limit = 8,
): TagSuggestion[] {
  const token = rawToken.trim().toLowerCase()
  if (!token || !store.tagKeys.length) return []

  // Both sides compare in compact form (no spaces / underscores): "redey",
  // "red_ey" and "red ey" all find "red eyes". A plain startsWith/includes
  // match is always a compact match too, so one check covers both.
  const compactToken = token.replace(/[\s_]/g, '')
  if (!compactToken) return []

  const out: TagSuggestion[] = []
  for (let i = 0; i < store.tagKeys.length && out.length < limit; i++) {
    if (store.compactedKeys[i].startsWith(compactToken)) {
      out.push({ tag: store.tagKeys[i], matchType: 'prefix' })
    }
  }
  for (let i = 0; i < store.tagKeys.length && out.length < limit; i++) {
    const compacted = store.compactedKeys[i]
    if (!compacted.startsWith(compactToken) && compacted.includes(compactToken)) {
      out.push({ tag: store.tagKeys[i], matchType: 'substring' })
    }
  }
  return out
}
