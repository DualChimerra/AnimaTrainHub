import { render, screen } from '@testing-library/react'
import userEvent, { type UserEvent } from '@testing-library/user-event'
import type { ComponentProps } from 'react'
import { describe, expect, it, vi } from 'vitest'

import BulkActionBar from './BulkActionBar'
import { DialogProvider } from './Dialog'
import { ToastProvider } from './Toast'

const cache = new Map<string, string[]>([
  ['a.png', ['cat', 'solo']],
  ['b.png', ['cat']],
  ['c.png', ['dog']],
])

function renderBar(overrides: Partial<ComponentProps<typeof BulkActionBar>> = {}) {
  const onApply = vi.fn()
  const props: ComponentProps<typeof BulkActionBar> = {
    cache,
    selectedKeys: ['a.png', 'b.png'],
    onApply,
    tagSuggestions: ['cat', 'dog', 'solo', 'warm'],
    onSelectAll: vi.fn(),
    onClearSelection: vi.fn(),
    totalCount: 3,
    ...overrides,
  }
  render(
    <ToastProvider>
      <DialogProvider>
        <BulkActionBar {...props} />
      </DialogProvider>
    </ToastProvider>,
  )
  return { onApply }
}

// Triggering an op pops a confirm modal - only clicking its "Confirm" button actually applies it.
const confirmInModal = async (user: UserEvent) => {
  const ok = await screen.findByRole('button', { name: 'Confirm' })
  await user.click(ok)
}

describe('BulkActionBar (with confirm modal)', () => {
  it('add at front (default position) prepends after confirm', async () => {
    const user = userEvent.setup()
    const { onApply } = renderBar()

    await user.type(screen.getByLabelText('Tags to add'), 'warm')
    await user.click(screen.getByRole('button', { name: 'Add' }))
    await confirmInModal(user)

    expect(onApply).toHaveBeenCalledOnce()
    const updates = onApply.mock.calls[0][0] as Map<string, string[]>
    expect(updates.get('a.png')).toEqual(['warm', 'cat', 'solo'])
    expect(updates.get('b.png')).toEqual(['warm', 'cat'])
    expect(updates.has('c.png')).toBe(false)
  })

  it('switching position toggle to "Back" appends instead of prepends', async () => {
    const user = userEvent.setup()
    const { onApply } = renderBar()

    await user.click(screen.getByRole('button', { name: 'Back' }))
    await user.type(screen.getByLabelText('Tags to add'), 'warm')
    await user.click(screen.getByRole('button', { name: 'Add' }))
    await confirmInModal(user)

    const updates = onApply.mock.calls[0][0] as Map<string, string[]>
    expect(updates.get('a.png')).toEqual(['cat', 'solo', 'warm'])
  })

  it('remove drops typed tag from selected only after confirm', async () => {
    const user = userEvent.setup()
    const { onApply } = renderBar()

    await user.type(screen.getByLabelText('Tags to remove'), 'cat')
    await user.click(screen.getByRole('button', { name: 'Remove' }))
    await confirmInModal(user)

    const updates = onApply.mock.calls[0][0] as Map<string, string[]>
    expect(updates.get('a.png')).toEqual(['solo'])
    expect(updates.get('b.png')).toEqual([])
    expect(updates.has('c.png')).toBe(false)
  })

  it('replace swaps old→new after confirm', async () => {
    const user = userEvent.setup()
    const { onApply } = renderBar()

    await user.type(screen.getByPlaceholderText('old'), 'cat')
    await user.type(screen.getByPlaceholderText('new'), 'feline')
    await user.click(screen.getByRole('button', { name: 'Replace' }))
    await confirmInModal(user)

    const updates = onApply.mock.calls[0][0] as Map<string, string[]>
    expect(updates.get('a.png')).toEqual(['feline', 'solo'])
    expect(updates.get('b.png')).toEqual(['feline'])
  })

  it('dedupe runs after confirm with no input needed', async () => {
    const dupCache = new Map<string, string[]>([
      ['a.png', ['cat', 'cat', 'solo']],
    ])
    const user = userEvent.setup()
    const { onApply } = renderBar({ cache: dupCache, selectedKeys: ['a.png'] })

    await user.click(screen.getByRole('button', { name: 'Dedupe' }))
    await confirmInModal(user)

    const updates = onApply.mock.calls[0][0] as Map<string, string[]>
    expect(updates.get('a.png')).toEqual(['cat', 'solo'])
  })

  it('cancelling the confirm modal skips apply', async () => {
    const user = userEvent.setup()
    const { onApply } = renderBar()

    await user.type(screen.getByLabelText('Tags to add'), 'warm')
    await user.click(screen.getByRole('button', { name: 'Add' }))

    const cancel = await screen.findByRole('button', { name: 'Cancel' })
    await user.click(cancel)

    expect(onApply).not.toHaveBeenCalled()
  })

  it('all op buttons disabled when nothing selected', () => {
    renderBar({ selectedKeys: [] })
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Remove' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Replace' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Dedupe' })).toBeDisabled()
  })

  it('select-all-images button uses image-specific label', () => {
    renderBar()
    expect(screen.getByRole('button', { name: 'Select all images' })).toBeInTheDocument()
  })
})
