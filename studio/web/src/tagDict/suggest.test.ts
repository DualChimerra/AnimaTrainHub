import { describe, expect, it } from 'vitest'
import { extractCurrentToken, findSuggestions, type SuggestStore } from './suggest'

// ---------------------------------------------------------------------------
// extractCurrentToken
// ---------------------------------------------------------------------------

describe('extractCurrentToken', () => {
  it('returns whole string when no separator', () => {
    expect(extractCurrentToken('1gir', 4)).toEqual({ token: '1gir', start: 0, end: 4 })
  })

  it('extracts token after a comma', () => {
    // cursor at the end of "1girl,solo": the current token is "solo"
    expect(extractCurrentToken('1girl,solo', 10)).toEqual({ token: 'solo', start: 6, end: 10 })
  })

  it('extracts token after a space-comma sequence', () => {
    // cursor at the end of "1girl, sol"; start includes the leading space
    expect(extractCurrentToken('1girl, sol', 10)).toEqual({ token: 'sol', start: 6, end: 10 })
  })

  it('handles cursor in the middle of a token', () => {
    // value="1girl,solo,long"; cursor=8 is inside "solo"
    const r = extractCurrentToken('1girl,solo,long', 8)
    expect(r.token).toBe('solo')
    expect(r.start).toBe(6)
    expect(r.end).toBe(10)
  })

  it('handles newline as separator (textarea multiline)', () => {
    const r = extractCurrentToken('foo\nbar', 7)
    expect(r).toEqual({ token: 'bar', start: 4, end: 7 })
  })

  it('returns empty token when cursor right after comma', () => {
    const r = extractCurrentToken('1girl,', 6)
    expect(r).toEqual({ token: '', start: 6, end: 6 })
  })

  it('clamps cursor to string range', () => {
    expect(extractCurrentToken('abc', 100)).toEqual({ token: 'abc', start: 0, end: 3 })
    expect(extractCurrentToken('abc', -5)).toEqual({ token: 'abc', start: 0, end: 3 })
  })
})

// ---------------------------------------------------------------------------
// findSuggestions
// ---------------------------------------------------------------------------

function buildStore(tags: string[]): SuggestStore {
  // Same as the production store: tagKeys keep list (popularity) order.
  return { tagKeys: tags, compactedKeys: tags.map((t) => t.replace(/[\s_]/g, '')) }
}

describe('findSuggestions', () => {
  const store = buildStore(['1girl', 'girl', 'long hair', 'longest day', 'solo'])

  it('returns [] for empty token', () => {
    expect(findSuggestions('', store)).toEqual([])
  })

  it('returns [] when the list is empty', () => {
    expect(findSuggestions('foo', buildStore([]))).toEqual([])
  })

  it('finds prefix matches in list order', () => {
    const r = findSuggestions('lon', store)
    expect(r.map((s) => s.tag)).toEqual(['long hair', 'longest day'])
    expect(r[0].matchType).toBe('prefix')
  })

  it('orders prefix matches by list (popularity) order, not length', () => {
    // 'red eyes' comes first in the list (more popular) but is longer than 'red'
    expect(findSuggestions('re', buildStore(['red eyes', 'red'])).map((x) => x.tag)).toEqual(['red eyes', 'red'])
  })

  it('case-insensitive', () => {
    const r = findSuggestions('GIRL', store)
    expect(r.map((s) => s.tag)).toContain('girl')
    expect(r.map((s) => s.tag)).toContain('1girl')
  })

  it('falls back to substring after prefix matches', () => {
    // 'girl' is the prefix match; '1girl' comes through the substring pass
    const r = findSuggestions('girl', store, 5)
    expect(r.find((s) => s.tag === '1girl')?.matchType).toBe('substring')
  })

  it('matches compacted form (skip spaces/_): "redey" → "red eyes"', () => {
    const r = findSuggestions('redey', buildStore(['1girl', 'red eyes', 'red hair', 'solo']))
    expect(r.map((s) => s.tag)).toContain('red eyes')
    expect(r[0].matchType).toBe('prefix')
  })

  it('matches when token uses booru underscore form: "red_ey" → "red eyes"', () => {
    const r = findSuggestions('red_ey', buildStore(['red eyes', 'solo']))
    expect(r.map((x) => x.tag)).toContain('red eyes')
    expect(r[0].matchType).toBe('prefix')
  })

  it('returns [] for token made of separators only', () => {
    // an empty compact token must bail out, or ''.startsWith would match everything
    expect(findSuggestions('_', store)).toEqual([])
  })

  it('honors limit', () => {
    const big = buildStore(Array.from({ length: 20 }, (_, i) => `xtag${i}`))
    expect(findSuggestions('xtag', big, 3)).toHaveLength(3)
  })
})
