import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { NewVersionDialog } from './Layout'

describe('NewVersionDialog (PP10.1)', () => {
  it('hides the "Fork from…" dropdown when no existing versions', () => {
    render(
      <NewVersionDialog
        existingLabels={[]}
        existingVersions={[]}
        onCancel={() => {}}
        onSubmit={() => {}}
      />,
    )
    expect(screen.queryByText(/Fork from…/)).toBeNull()
  })

  it('shows dropdown with existing versions and submits with null forkFrom by default', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(
      <NewVersionDialog
        existingLabels={['baseline']}
        existingVersions={[{ id: 7, label: 'baseline' }]}
        onCancel={() => {}}
        onSubmit={onSubmit}
      />,
    )
    expect(screen.getByText(/Fork from…/)).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Start from scratch' })).toBeInTheDocument()
    expect(
      screen.getByRole('option', { name: /Copy from baseline/ }),
    ).toBeInTheDocument()

    await user.type(screen.getByPlaceholderText(/baseline/), 'high-lr')
    await user.click(screen.getByRole('button', { name: 'Create' }))
    expect(onSubmit).toHaveBeenCalledWith('high-lr', null)
  })

  it('passes fork_from_version_id when user picks a source version', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(
      <NewVersionDialog
        existingLabels={['baseline']}
        existingVersions={[{ id: 7, label: 'baseline' }]}
        onCancel={() => {}}
        onSubmit={onSubmit}
      />,
    )
    await user.type(screen.getByPlaceholderText(/baseline/), 'forked')
    await user.selectOptions(screen.getByRole('combobox'), '7')
    // A hint should appear once a source version is picked
    expect(screen.getByText(/Copies train\/, reg\//)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Create' }))
    expect(onSubmit).toHaveBeenCalledWith('forked', 7)
  })

  it('rejects duplicate label', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(
      <NewVersionDialog
        existingLabels={['baseline']}
        existingVersions={[{ id: 7, label: 'baseline' }]}
        onCancel={() => {}}
        onSubmit={onSubmit}
      />,
    )
    await user.type(screen.getByPlaceholderText(/baseline/), 'baseline')
    await user.click(screen.getByRole('button', { name: 'Create' }))
    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.getByText(/Label already exists/)).toBeInTheDocument()
  })
})
