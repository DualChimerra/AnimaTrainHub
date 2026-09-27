import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useRef, useState } from 'react'
import { beforeEach, describe, expect, it } from 'vitest'

import { __setStateForTest } from '../../tagDict/store'
import { TagSuggestList } from './TagSuggestList'
import { useTagSuggest } from './useTagSuggest'

/** Seed a small tag list (no network) so findSuggestions has matches. */
function seedDict() {
  const tagKeys = ['solo', 'long hair']
  __setStateForTest({
    status: 'ready',
    tagKeys,
    compactedKeys: tagKeys.map((t) => t.replace(/[\s_]/g, '')),
    meta: null,
    error: null,
  })
}

/** Mirrors the real caller's wiring (isomorphic to PromptList / TagsInput). */
function Harness({ initial = '' }: { initial?: string }) {
  const [value, setValue] = useState(initial)
  const inputRef = useRef<HTMLInputElement>(null)
  const suggest = useTagSuggest({
    value,
    inputRef,
    onPick: ({ suggestion }) => setValue(suggestion.tag),
  })
  return (
    <div>
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => { setValue(e.target.value); suggest.notifyChange() }}
        onKeyDown={(e) => { suggest.handleKeyDown(e) }}
        onKeyUp={() => suggest.notifySelect()}
        onClick={() => suggest.notifyClick()}
        onFocus={() => suggest.notifyFocus()}
        onBlur={() => suggest.notifyBlur()}
      />
      <TagSuggestList
        open={suggest.open}
        suggestions={suggest.suggestions}
        activeIdx={suggest.activeIdx}
        onPick={(s) => suggest.pickAt(suggest.suggestions.indexOf(s))}
        onHover={suggest.setActiveIdx}
        inputRef={inputRef}
        cursor={suggest.cursor}
        positionDeps={[value]}
      />
    </div>
  )
}

describe('useTagSuggest popup rules', () => {
  beforeEach(() => {
    localStorage.clear()
    seedDict()
  })

  it('clicking into an input that already has content does not pop suggestions (only a value change does)', async () => {
    const user = userEvent.setup()
    render(<Harness initial="sol" />)
    await user.click(screen.getByRole('textbox'))
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('pops suggestions after the value changes', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    await user.type(screen.getByRole('textbox'), 'sol')
    expect(await screen.findByRole('listbox')).toBeInTheDocument()
    expect(screen.getByText('solo')).toBeInTheDocument()
  })

  it('clicking the input to move the caret after suggestions are open closes them', async () => {
    const user = userEvent.setup()
    const input = () => screen.getByRole('textbox')
    render(<Harness />)
    await user.type(input(), 'sol')
    expect(await screen.findByRole('listbox')).toBeInTheDocument()
    await user.click(input())
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('typing does not pop suggestions once the global switch is off', async () => {
    localStorage.setItem('studio.tag.autocomplete', '0')
    const user = userEvent.setup()
    render(<Harness />)
    await user.type(screen.getByRole('textbox'), 'sol')
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('clicking a suggestion still works (onMouseDown pick is not blocked by click-close)', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    await user.type(screen.getByRole('textbox'), 'sol')
    const option = await screen.findByText('solo')
    await user.click(option)
    expect(screen.getByRole('textbox')).toHaveValue('solo')
    expect(screen.queryByRole('listbox')).toBeNull()
  })
})
