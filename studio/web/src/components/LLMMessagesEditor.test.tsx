import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import LLMMessagesEditor from './LLMMessagesEditor'
import type { LLMMessage } from '../api/client'

// Controlled textarea: the parent component holds state and writes back each edit, mirroring real Settings usage.
function Harness() {
  const [messages, setMessages] = useState<LLMMessage[]>([
    { type: 'text', role: 'system', content: '' },
    { type: 'text', role: 'user', content: '' },
  ])
  return <LLMMessagesEditor messages={messages} onChange={setMessages} />
}

describe('LLMMessagesEditor', () => {
  // regression: after an immutable update swaps the message object reference, if the
  // stable id isn't carried over to the new object, idOf issues a new id -> the key
  // changes -> SortableMessage (with its textarea) remounts -> typing loses focus after
  // one character and continuous input is impossible.
  it('editing message content does not remount the textarea, keeps focus, allows continuous typing', async () => {
    const user = userEvent.setup()
    render(<Harness />)

    // two messages -> two textareas; the second is the user message body.
    const before = screen.getAllByRole('textbox')[1] as HTMLTextAreaElement
    before.focus()
    expect(before).toHaveFocus()

    await user.type(before, 'hello')

    const after = screen.getAllByRole('textbox')[1] as HTMLTextAreaElement
    expect(after).toBe(before)         // same DOM node - no remount
    expect(after).toHaveFocus()        // focus retained
    expect(after).toHaveValue('hello') // every typed character lands, not just the first
  })
})
