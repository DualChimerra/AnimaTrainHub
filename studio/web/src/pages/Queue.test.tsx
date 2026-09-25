/** 队列页：行内联采样条 + 右键备注菜单（_v20 tasks.note）。 */
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { api, type Task } from '../api/client'
import { DialogProvider } from '../components/Dialog'
import { ToastProvider } from '../components/Toast'
import QueuePage from './Queue'

class FakeEventSource {
  static readonly OPEN = 1
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  readyState = FakeEventSource.OPEN
  close(): void { this.readyState = 2 }
}

function makeTask(over: Partial<Task> = {}): Task {
  return {
    id: 1, name: 'lora run', config_name: 'train_lora', task_type: 'train',
    status: 'done', priority: 0,
    created_at: 900, started_at: 1000, finished_at: 1200,
    pid: null, exit_code: 0, output_dir: null, error_msg: null,
    monitor_state_path: 'C:/x/monitor_state.json',
    note: null,
    ...over,
  }
}

beforeEach(() => {
  vi.stubGlobal('EventSource', FakeEventSource)
  vi.spyOn(api, 'getQueueHold').mockResolvedValue({ held: false, pending_waiting: 0 } as never)
  vi.spyOn(api, 'listTaskSamples').mockResolvedValue({ items: [], total: 0 })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function renderQueue() {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <DialogProvider>
          <QueuePage />
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>,
  )
}

it('shows an existing note on the row', async () => {
  vi.spyOn(api, 'listQueue').mockResolvedValue([makeTask({ note: 'alpha 16' })])
  renderQueue()
  const note = await screen.findByTestId('task-note-1')
  expect(note).toHaveTextContent('alpha 16')
})

it('right-click opens the menu and saves a note', async () => {
  vi.spyOn(api, 'listQueue').mockResolvedValue([makeTask()])
  const setNote = vi.spyOn(api, 'setTaskNote').mockResolvedValue(
    makeTask({ note: 'dataset v3' }),
  )
  renderQueue()

  fireEvent.contextMenu(await screen.findByText('lora run'))
  const menu = await screen.findByTestId('queue-context-menu')
  // 没备注时是「添加备注」，没有「删除备注」项
  expect(menu).toHaveTextContent('添加备注')
  expect(menu).not.toHaveTextContent('删除备注')

  fireEvent.click(screen.getByText('添加备注'))
  const input = await screen.findByRole('textbox')
  fireEvent.change(input, { target: { value: 'dataset v3' } })
  fireEvent.click(screen.getByText('保存'))

  await waitFor(() => expect(setNote).toHaveBeenCalledWith(1, 'dataset v3'))
  expect(await screen.findByTestId('task-note-1')).toHaveTextContent('dataset v3')
})

it('removes a note straight from the menu', async () => {
  vi.spyOn(api, 'listQueue').mockResolvedValue([makeTask({ note: 'old' })])
  const setNote = vi.spyOn(api, 'setTaskNote').mockResolvedValue(makeTask({ note: null }))
  renderQueue()

  fireEvent.contextMenu(await screen.findByText('lora run'))
  fireEvent.click(await screen.findByText('删除备注'))

  await waitFor(() => expect(setNote).toHaveBeenCalledWith(1, ''))
  await waitFor(() => expect(screen.queryByTestId('task-note-1')).not.toBeInTheDocument())
})

it('opens the row samples on request and hides them again', async () => {
  vi.spyOn(api, 'listQueue').mockResolvedValue([makeTask()])
  vi.spyOn(api, 'listTaskSamples').mockResolvedValue({
    items: [{ filename: 'epoch_1.png', mtime: 1, size: 1, epoch: 1, step: null }],
    total: 1,
  })
  renderQueue()

  // Cards carry no strip until asked (the mockup's queue cards).
  fireEvent.contextMenu(await screen.findByText('lora run'))
  expect(screen.queryByTestId('sample-strip-1')).not.toBeInTheDocument()
  fireEvent.click(await screen.findByText('显示采样图'))
  expect(await screen.findByTestId('sample-strip-1')).toBeInTheDocument()

  fireEvent.contextMenu(screen.getByText('lora run'))
  fireEvent.click(await screen.findByText('隐藏采样图'))
  await waitFor(() => expect(screen.queryByTestId('sample-strip-1')).not.toBeInTheDocument())
})

it('does not fetch samples for tasks without a monitor state', async () => {
  const list = vi.spyOn(api, 'listTaskSamples')
  vi.spyOn(api, 'listQueue').mockResolvedValue([makeTask({ monitor_state_path: null })])
  renderQueue()

  await screen.findByText('lora run')
  expect(list).not.toHaveBeenCalled()
})
