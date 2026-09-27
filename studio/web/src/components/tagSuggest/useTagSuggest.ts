/** Autocomplete behavior hook -- shared by input / textarea.
 *
 * Design principle: the hook never mutates the user's value, it only calls back `onPick` when the user selects a
 * candidate. The caller decides how it lands in the data (replace the token range / push to a tags array / append).
 *
 * Popup rule: candidates only pop up after the input changes (notifyChange); focusing / clicking to move the cursor
 * never pops one, and clicking also closes an already-open popup. The global toggle (Settings' "Tag translation
 * dictionary" section) disables popups from every entry point when off.
 *
 * Two token modes:
 *   - `wholeAsToken: false` (default): computes the current token from the cursor + comma boundaries, and gives
 *     the caller a `range` on commit for slice-and-replace.
 *   - `wholeAsToken: true`: the whole value is treated as a single token. Used by TagEditor's chip-mode input
 *     (the input is a draft, already a single segment; commit calls addTag(s.tag) directly).
 */
import { useEffect, useMemo, useState } from 'react'
import type React from 'react'

import { useTagAutocompleteEnabled } from '../../tagDict/autocompleteToggle'
import { useTagDict } from '../../tagDict/store'
import { extractCurrentToken, findSuggestions } from '../../tagDict/suggest'
import type { TagSuggestion } from '../../tagDict/types'

export interface TagSuggestPick {
  /** The selected candidate. */
  suggestion: TagSuggestion
  /** Range of the current token within the original value; [0, value.length] when wholeAsToken. */
  range: { start: number; end: number }
}

interface Args {
  value: string
  inputRef: React.RefObject<HTMLInputElement | HTMLTextAreaElement | null>
  onPick: (pick: TagSuggestPick) => void
  wholeAsToken?: boolean
  /** Disables autocomplete (dict not loaded, field disabled, etc.). */
  disabled?: boolean
}

export interface TagSuggestApi {
  open: boolean
  suggestions: TagSuggestion[]
  activeIdx: number
  setActiveIdx: (i: number) => void
  setOpen: (open: boolean) => void
  /** Current caret position (state); passed to TagSuggestList as its positionDep. */
  cursor: number
  /** Call as the first statement in the input's onKeyDown; returns true if handled (the caller should return). */
  handleKeyDown: (e: React.KeyboardEvent) => boolean
  /** Call in the input's onChange (tracks cursor + auto-opens). The only entry point that opens the popup. */
  notifyChange: () => void
  /** Call in the input's onFocus (only tracks the cursor, doesn't open the popup). */
  notifyFocus: () => void
  /** Call in the input's onBlur (closes with a delay, to give a click time to land). */
  notifyBlur: () => void
  /** Call in the input's onClick: a mouse click moving the cursor -> closes an already-open popup. */
  notifyClick: () => void
  /** Call in the input's onKeyUp (tracks the cursor when the keyboard moves it). */
  notifySelect: () => void
  /** For mouse pick / programmatic triggering. */
  pickAt: (i: number) => void
}

export function useTagSuggest({
  value, inputRef, onPick, wholeAsToken = false, disabled = false,
}: Args): TagSuggestApi {
  const dict = useTagDict()
  const [acEnabled] = useTagAutocompleteEnabled()
  const off = disabled || !acEnabled
  const [open, setOpen] = useState(false)
  const [activeIdx, setActiveIdx] = useState(0)
  const [cursor, setCursor] = useState(0)

  const tokenInfo = useMemo(() => {
    if (wholeAsToken) return { token: value.trim(), start: 0, end: value.length }
    return extractCurrentToken(value, cursor)
  }, [value, cursor, wholeAsToken])

  const suggestions = useMemo(() => {
    if (off || !open || dict.status !== 'ready' || !tokenInfo.token) return []
    return findSuggestions(tokenInfo.token, {
      tagKeys: dict.tagKeys,
      compactedKeys: dict.compactedKeys,
    })
  }, [off, open, tokenInfo.token, dict.status, dict.tagKeys, dict.compactedKeys])

  // Resets active when the suggestions list changes
  const sugKey = suggestions.map((s) => s.tag).join('|')
  useEffect(() => { setActiveIdx(0) }, [sugKey])

  const syncCursor = () => {
    const el = inputRef.current
    if (!el) return
    setCursor(el.selectionStart ?? el.value.length)
  }

  const pickAt = (i: number) => {
    const s = suggestions[i]
    if (!s) return
    onPick({ suggestion: s, range: { start: tokenInfo.start, end: tokenInfo.end } })
    setOpen(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent): boolean => {
    if (off) return false
    if (!open || suggestions.length === 0) return false
    if (e.key === 'ArrowDown') {
      e.preventDefault(); setActiveIdx((activeIdx + 1) % suggestions.length); return true
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIdx((activeIdx - 1 + suggestions.length) % suggestions.length); return true
    }
    if (e.key === 'Enter' || e.key === 'Tab') {
      e.preventDefault(); pickAt(activeIdx); return true
    }
    if (e.key === 'Escape') {
      e.preventDefault(); setOpen(false); return true
    }
    return false
  }

  return {
    open, suggestions, activeIdx, setActiveIdx, setOpen,
    cursor,
    handleKeyDown,
    notifyChange: () => { syncCursor(); if (!off) setOpen(true) },
    // focus only tracks the cursor: candidates only pop up after an input change, not just from clicking into the middle of a prompt
    notifyFocus: () => { syncCursor() },
    // 100ms delay: gives onMouseDown(pick) time to complete; the popover won't get stuck if the user switches elsewhere
    notifyBlur: () => { setTimeout(() => setOpen(false), 120) },
    // A mouse click = the user is moving the cursor, not completing -> close the candidates
    notifyClick: () => { syncCursor(); setOpen(false) },
    notifySelect: () => { syncCursor() },
    pickAt,
  }
}
