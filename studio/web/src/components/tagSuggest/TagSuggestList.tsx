/** Autocomplete suggestion list -- Portal + caret-anchored fixed positioning.
 *
 * The old version anchored to the input container with absolute +
 * align="top|bottom"; for a tall textarea the popover could fly off the top
 * or bottom of the screen. New version: use the mirror-div algorithm to get
 * the caret's pixel coordinates inside the input/textarea, portal to body,
 * and position it fixed right below the cursor; flips to above the cursor
 * automatically when there isn't enough room at the bottom of the viewport.
 */
import { useLayoutEffect, useState } from 'react'
import type { RefObject } from 'react'
import { createPortal } from 'react-dom'

import { getCaretCoordinates } from '../../tagDict/caretCoords'
import type { TagSuggestion } from '../../tagDict/types'

interface Props {
  open: boolean
  suggestions: TagSuggestion[]
  activeIdx: number
  onPick: (s: TagSuggestion) => void
  onHover: (idx: number) => void
  inputRef: RefObject<HTMLInputElement | HTMLTextAreaElement | null>
  /** Pass the cursor position in as a dep; optional (add deps as needed). */
  cursor?: number
  /** Extra dependency that triggers a position recompute (e.g. the value string); re-measures the caret when it changes. */
  positionDeps?: ReadonlyArray<unknown>
}

interface Position {
  top: number
  left: number
  /** True when flipped above the cursor (the popover's bottom aligns with the caret's top). */
  flipUp: boolean
}

export function TagSuggestList({
  open, suggestions, activeIdx, onPick, onHover, inputRef, cursor, positionDeps = [],
}: Props) {
  const [pos, setPos] = useState<Position | null>(null)

  useLayoutEffect(() => {
    if (!open || !inputRef.current || suggestions.length === 0) {
      setPos(null)
      return
    }
    const el = inputRef.current
    const cursorPos = cursor ?? el.selectionStart ?? el.value.length
    const caret = getCaretCoordinates(el, cursorPos)
    const rect = el.getBoundingClientRect()

    // caret's y/x in viewport coordinates
    const caretTopVp = rect.top + caret.top - el.scrollTop
    const caretLeftVp = rect.left + caret.left - el.scrollLeft

    // estimated popover height (~26px per row + 8px padding, capped at 260)
    const POPOVER_H = Math.min(260, suggestions.length * 26 + 8)
    const SPACE_BELOW = window.innerHeight - (caretTopVp + caret.height) - 8
    const SPACE_ABOVE = caretTopVp - 8
    const flipUp = SPACE_BELOW < POPOVER_H && SPACE_ABOVE > SPACE_BELOW

    const top = flipUp ? caretTopVp - POPOVER_H - 4 : caretTopVp + caret.height + 4
    // keep the popover from crossing the right edge (rough: stay within a 200px minimum width)
    const left = Math.min(caretLeftVp, window.innerWidth - 220)

    setPos({ top, left: Math.max(8, left), flipUp })
    // dependency list: recompute whenever any of open / suggestions count / cursor / value changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, suggestions.length, cursor, ...positionDeps])

  if (!open || !pos || suggestions.length === 0) return null

  return createPortal(
    <ul
      className="bg-elevated border border-subtle rounded-sm shadow-lg max-h-[260px] overflow-y-auto min-w-[220px] list-none p-1 m-0"
      role="listbox"
      style={{ position: 'fixed', top: pos.top, left: pos.left, zIndex: 1000 }}
    >
      {suggestions.map((s, i) => (
        <li
          key={s.tag}
          role="option"
          aria-selected={i === activeIdx}
          onMouseEnter={() => onHover(i)}
          // onMouseDown + preventDefault: keeps the input from losing focus
          onMouseDown={(e) => { e.preventDefault(); onPick(s) }}
          className={
            'px-2.5 py-1 text-xs font-mono cursor-pointer rounded-sm flex items-center gap-2 ' +
            (i === activeIdx ? 'bg-overlay text-fg-primary' : 'text-fg-secondary hover:bg-overlay')
          }
        >
          <span>{s.tag}</span>
        </li>
      ))}
    </ul>,
    document.body,
  )
}
