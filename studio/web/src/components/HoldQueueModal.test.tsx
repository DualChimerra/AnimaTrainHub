/**
 * HoldQueueModal unit tests (ADR 0006 PR-4 4.4).
 *
 * Verifies:
 *   - no running task -> simple confirm modal, no radio
 *   - running task -> renders radio, defaulting to "let it finish"
 *   - main button text tracks the radio selection
 *   - confirm callback dispatches hold-only vs hold-and-pause based on the radio choice
 *   - cancel callback
 */
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { HoldQueueModal } from './HoldQueueModal'
import type { Task } from '../api/client'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (k: string, opts?: Record<string, string | number>) => {
      if (opts && typeof opts === 'object') {
        return `${k}<${JSON.stringify(opts)}>`
      }
      return k
    },
  }),
}))

const mkTask = (overrides?: Partial<Task>): Task => ({
  id: 42,
  name: 'my_lora_v1',
  config_name: 'cfg',
  status: 'running',
  priority: 0,
  created_at: 0,
  started_at: null,
  finished_at: null,
  pid: null,
  exit_code: null,
  output_dir: null,
  error_msg: null,
  ...overrides,
})

describe('HoldQueueModal', () => {
  it('no running task: only confirm/cancel, no radio', () => {
    const onCancel = vi.fn()
    const onConfirm = vi.fn()
    render(<HoldQueueModal runningTask={null} onCancel={onCancel} onConfirm={onConfirm} />)

    expect(screen.getByTestId('hold-queue-modal')).toBeInTheDocument()
    expect(screen.queryByTestId('hold-opt-let-run')).toBeNull()
    expect(screen.queryByTestId('hold-opt-pause-too')).toBeNull()
    fireEvent.click(screen.getByTestId('hold-confirm-btn'))
    expect(onConfirm).toHaveBeenCalledWith({ kind: 'hold-only' })
  })

  it('with running task: renders radio defaulting to "let run"', () => {
    render(
      <HoldQueueModal runningTask={mkTask()} onCancel={vi.fn()} onConfirm={vi.fn()} />,
    )
    const letRun = screen.getByTestId('hold-opt-let-run') as HTMLInputElement
    const pauseToo = screen.getByTestId('hold-opt-pause-too') as HTMLInputElement
    expect(letRun.checked).toBe(true)
    expect(pauseToo.checked).toBe(false)
  })

  it('confirm with let-run selected -> hold-only decision', () => {
    const onConfirm = vi.fn()
    render(
      <HoldQueueModal runningTask={mkTask()} onCancel={vi.fn()} onConfirm={onConfirm} />,
    )
    fireEvent.click(screen.getByTestId('hold-confirm-btn'))
    expect(onConfirm).toHaveBeenCalledWith({ kind: 'hold-only' })
  })

  it('confirm with pause-too selected -> hold-and-pause with task id', () => {
    const onConfirm = vi.fn()
    render(
      <HoldQueueModal runningTask={mkTask({ id: 7 })} onCancel={vi.fn()} onConfirm={onConfirm} />,
    )
    fireEvent.click(screen.getByTestId('hold-opt-pause-too'))
    fireEvent.click(screen.getByTestId('hold-confirm-btn'))
    expect(onConfirm).toHaveBeenCalledWith({ kind: 'hold-and-pause', taskId: 7 })
  })

  it('cancel button triggers onCancel', () => {
    const onCancel = vi.fn()
    render(
      <HoldQueueModal runningTask={null} onCancel={onCancel} onConfirm={vi.fn()} />,
    )
    fireEvent.click(screen.getByTestId('hold-cancel-btn'))
    expect(onCancel).toHaveBeenCalled()
  })

  it('confirm button text mirrors radio selection', () => {
    render(
      <HoldQueueModal runningTask={mkTask({ id: 9 })} onCancel={vi.fn()} onConfirm={vi.fn()} />,
    )
    const btn = screen.getByTestId('hold-confirm-btn')
    // default let-run -> confirmLetRun
    expect(btn.textContent).toContain('confirmLetRun')
    fireEvent.click(screen.getByTestId('hold-opt-pause-too'))
    expect(btn.textContent).toContain('confirmPauseToo')
    fireEvent.click(screen.getByTestId('hold-opt-let-run'))
    expect(btn.textContent).toContain('confirmLetRun')
  })
})
