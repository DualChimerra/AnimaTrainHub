import { useEffect, useRef, useState } from 'react'

import { TagSuggestList } from './tagSuggest/TagSuggestList'
import { useTagSuggest } from './tagSuggest/useTagSuggest'

/** Comma-separated string -> normalized tag array (trims edges / drops empty segments). */
export function parseTags(s: string): string[] {
  return s.split(',').map((t) => t.trim()).filter(Boolean)
}

/** Comma-separated tag list input (no label). Two states:
 *
 * - **Editing (focus)**: a plain-text `<input>`, commas / spaces typed freely. Controlled by the text, not the
 *   array, to avoid a "join-back on every keystroke" wiping out a comma / trailing space being typed. An autocomplete
 *   popover pops up below the input; keyboard up/down + Enter/Tab select.
 * - **Idle (blur)**: renders tags as chips, scannable at a glance; shows the Chinese translation when there's a match.
 *
 * On blur the text is normalized to `tags.join(', ')`, so re-entering edit mode shows the canonical form.
 * Use this directly for cases that already have their own outer label (Settings' SettingsField); for a
 * 140px grid label use {@link TagsInput} below. */
export function TagListInput({ value, onChange, placeholder, disabled, className = '', style, commitOnBlur = false }: {
  value: string[]
  onChange: (v: string[]) => void
  placeholder?: string
  disabled?: boolean
  className?: string
  /** Inline style passthrough (used by the tagging page's TagField block to align visually with the training-config page's controls). */
  style?: React.CSSProperties
  /** true: editing only updates local text, and only reports up to the parent once on blur (used by the instant-apply
   *  settings pages to avoid per-keystroke commit -> per-keystroke PUT). Default false keeps per-keystroke reporting (used by local-editing scenes like the tagging page). */
  commitOnBlur?: boolean
}) {
  const [text, setText] = useState(value.join(', '))
  const [editing, setEditing] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const suggest = useTagSuggest({
    value: text,
    inputRef,
    disabled,
    onPick: ({ suggestion, range }) => {
      const before = text.slice(0, range.start)
      const after = text.slice(range.end)
      // Replace the token range with `tag, `; whatever already follows is appended right after, with leading whitespace normalized once
      const cleanAfter = after.replace(/^[,]\s*/, '')
      const next = `${before}${suggestion.tag}, ${cleanAfter}`
      setText(next)
      if (!commitOnBlur) onChange(parseTags(next))
      // Moves the cursor to right after the ", " following the newly inserted tag
      const newCursor = before.length + suggestion.tag.length + 2
      // Waits for React to finish flushing before setSelectionRange, otherwise the controlled update would push the cursor to the end
      requestAnimationFrame(() => {
        const el = inputRef.current
        if (el) { el.focus(); el.setSelectionRange(newCursor, newCursor) }
      })
    },
  })

  // External value change (restoring a default / switching forms) that disagrees with the current text's parse result -> resync
  // the text. A value change triggered by the user's own typing never reaches here (parseTags(text) always equals value then).
  useEffect(() => {
    if (editing) return  // Don't let an external value overwrite local text while editing (especially needed under commitOnBlur)
    if (JSON.stringify(parseTags(text)) !== JSON.stringify(value)) setText(value.join(', '))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, editing])

  // Entering edit mode -> puts the cursor in the input.
  useEffect(() => {
    if (editing) inputRef.current?.focus()
  }, [editing])

  if (editing && !disabled) {
    return (
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={text}
          placeholder={placeholder}
          onChange={(e) => {
            setText(e.target.value)
            if (!commitOnBlur) onChange(parseTags(e.target.value))
            suggest.notifyChange()
          }}
          onKeyDown={(e) => { suggest.handleKeyDown(e) }}
          onKeyUp={() => { suggest.notifySelect() }}
          onClick={() => { suggest.notifyClick() }}
          onFocus={() => { suggest.notifyFocus() }}
          onBlur={() => {
            suggest.notifyBlur()
            // Normalize on blur: underscore -> space (matches training captions' form; the backend match is already
            // _/space-insensitive), chips always display the space form. Under commitOnBlur, this is when the full tags list is reported up to the parent.
            const source = commitOnBlur ? parseTags(text) : value
            const canon = source.map((t) => t.replace(/_/g, ' '))
            if (JSON.stringify(canon) !== JSON.stringify(value)) onChange(canon)
            setText(canon.join(', '))
            setEditing(false)
          }}
          disabled={disabled}
          className={className}
          style={style}
        />
        <TagSuggestList
          open={suggest.open}
          suggestions={suggest.suggestions}
          activeIdx={suggest.activeIdx}
          onPick={(s) => suggest.pickAt(suggest.suggestions.indexOf(s))}
          onHover={suggest.setActiveIdx}
          inputRef={inputRef}
          cursor={suggest.cursor}
          positionDeps={[text]}
        />
      </div>
    )
  }

  // Idle state: shows chips, click / focus enters editing. min-h-[1.75rem] is a fallback line height for an empty
  // list, so the container doesn't collapse to 0 when only an nbsp renders; combined with the `.input` class's
  // padding it matches a normal input's height.
  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      onClick={() => { if (!disabled) setEditing(true) }}
      onFocus={() => { if (!disabled) setEditing(true) }}
      className={`${className} flex flex-wrap items-center gap-1 min-h-[1.75rem] ${disabled ? '' : 'cursor-text'}`}
      style={style}
    >
      {value.length === 0
        // nbsp placeholder: holds up one line's height, so the container doesn't collapse to a sliver when the list is
        // empty (must be a real U+00A0 -- an ASCII space gets squashed to 0 width in a flex container)
        ? <span className="text-fg-tertiary">{placeholder || ' '}</span>
        : value.map((tag, i) => (
            <span
              key={`${tag}-${i}`}
              className="inline-flex items-center px-2 py-0.5 rounded-full bg-overlay border border-subtle text-xs font-mono text-fg-primary"
            >
              {tag}
            </span>
          ))}
    </div>
  )
}

/** Version with a 140px label (used by the tagging page's grid layout). */
export default function TagsInput({ label, value, placeholder, disabled, onChange, modified, className = '' }: {
  label: string
  value: string[]
  placeholder?: string
  disabled: boolean
  onChange: (v: string[]) => void
  modified?: boolean
  className?: string
}) {
  return (
    <label className={'grid grid-cols-1 sm:grid-cols-[140px_1fr] sm:items-center gap-2 ' + className}>
      <span className="text-fg-tertiary font-mono text-xs">{label}</span>
      <TagListInput
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        disabled={disabled}
        className={`input input-mono ${modified ? 'border-warn' : ''}`}
      />
    </label>
  )
}
