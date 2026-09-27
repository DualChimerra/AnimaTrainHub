import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import TaskLogDrawer, { type LogSource } from './TaskLogDrawer'

function makeSource(overrides: Partial<LogSource> = {}): LogSource {
  return {
    key: 'tag',
    label: 'Tagging task',
    status: 'running',
    lines: ['line a', 'line b'],
    startedAt: 1700000000,
    finishedAt: null,
    ...overrides,
  }
}

describe('TaskLogDrawer (issue #251)', () => {
  it('hides entirely when there is no source', () => {
    const { container } = render(<TaskLogDrawer sources={[null, false, undefined]} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('mounting already running (replay scenario) auto-expands, the log panel rises and shows the tail', () => {
    render(<TaskLogDrawer sources={[makeSource()]} />)
    expect(screen.getByRole('button', { name: /Tagging task/ })).toHaveAttribute(
      'aria-expanded',
      'true',
    )
    expect(screen.getByTestId('log-drawer-body')).toHaveStyle({ height: '40vh' })
    expect(screen.getByText(/line a\s+line b/)).toBeInTheDocument()
  })

  it('mounting already done (historical replay) defaults to collapsed (body height 0), last line shown on the strip', () => {
    render(
      <TaskLogDrawer
        sources={[makeSource({ status: 'done', finishedAt: 1700000010 })]}
      />,
    )
    const strip = screen.getByRole('button', { name: /Tagging task/ })
    expect(strip).toHaveAttribute('aria-expanded', 'false')
    // The body stays mounted for the height animation; collapsed = height 0;
    // the collapsed strip shows the last line.
    expect(screen.getByTestId('log-drawer-body')).toHaveStyle({ height: '0px' })
    expect(screen.getByText('line b')).toBeInTheDocument()
  })

  it('does not auto-collapse when the task ends (done / failed both stay expanded; collapsing is only manual or on page unmount)', () => {
    const { rerender } = render(<TaskLogDrawer sources={[makeSource()]} />)
    const strip = () => screen.getByRole('button', { name: /Tagging task/ })
    expect(strip()).toHaveAttribute('aria-expanded', 'true')

    rerender(<TaskLogDrawer sources={[makeSource({ status: 'done' })]} />)
    expect(strip()).toHaveAttribute('aria-expanded', 'true')

    rerender(<TaskLogDrawer sources={[makeSource()]} />)
    rerender(<TaskLogDrawer sources={[makeSource({ status: 'failed' })]} />)
    expect(strip()).toHaveAttribute('aria-expanded', 'true')
  })

  it('clicking the collapsed strip toggles expand/collapse and keeps the manual state', async () => {
    const user = userEvent.setup()
    render(<TaskLogDrawer sources={[makeSource()]} />)
    const strip = screen.getByRole('button', { name: /Tagging task/ })
    await user.click(strip)
    expect(strip).toHaveAttribute('aria-expanded', 'false')
    await user.click(strip)
    expect(strip).toHaveAttribute('aria-expanded', 'true')
  })

  it('multiple sources, single display: a live task takes priority over a later-started terminal task', () => {
    render(
      <TaskLogDrawer
        sources={[
          makeSource({
            key: 'reg_build',
            label: 'Booru set build',
            status: 'done',
            startedAt: 1700009999,
          }),
          makeSource({ key: 'reg_ai', label: 'AI prior generation', status: 'running' }),
        ]}
      />,
    )
    expect(screen.getByText('AI prior generation')).toBeInTheDocument()
    expect(screen.queryByText('Booru set build')).toBeNull()
  })

  it('picks the most recently started task when all are terminal', () => {
    render(
      <TaskLogDrawer
        sources={[
          makeSource({ key: 'a', label: 'Old task', status: 'done', startedAt: 100 }),
          makeSource({ key: 'b', label: 'New task', status: 'failed', startedAt: 200 }),
        ]}
      />,
    )
    expect(screen.getByText('New task')).toBeInTheDocument()
    expect(screen.queryByText('Old task')).toBeNull()
  })

  it('shows a cancel button when live and onCancel is provided; clicking it does not toggle expand/collapse', async () => {
    const onCancel = vi.fn()
    const user = userEvent.setup()
    render(<TaskLogDrawer sources={[makeSource({ onCancel })]} />)
    const strip = screen.getByRole('button', { name: /Tagging task/ })
    expect(strip).toHaveAttribute('aria-expanded', 'true')
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onCancel).toHaveBeenCalledOnce()
    expect(strip).toHaveAttribute('aria-expanded', 'true')
  })

  it('does not show a cancel button in a terminal state', () => {
    render(
      <TaskLogDrawer sources={[makeSource({ status: 'done', onCancel: () => {} })]} />,
    )
    expect(screen.queryByRole('button', { name: 'Cancel' })).toBeNull()
  })

  it('shows a retry button when failed and onRetry is provided; clicking it does not toggle expand/collapse', async () => {
    const onRetry = vi.fn()
    const user = userEvent.setup()
    render(<TaskLogDrawer sources={[makeSource({ status: 'failed', onRetry })]} />)
    const strip = screen.getByRole('button', { name: /Tagging task/ })
    expect(strip).toHaveAttribute('aria-expanded', 'false')
    await user.click(screen.getByRole('button', { name: 'Retry' }))
    expect(onRetry).toHaveBeenCalledOnce()
    expect(strip).toHaveAttribute('aria-expanded', 'false')
  })

  it('does not show a retry button when not failed', () => {
    render(
      <TaskLogDrawer sources={[makeSource({ status: 'done', onRetry: () => {} })]} />,
    )
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  })
})
