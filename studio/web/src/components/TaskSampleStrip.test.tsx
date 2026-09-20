/** 队列行内联采样条：清单 → 缩略图 → 点开灯箱（含前后切）。 */
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { api, type TaskSample } from '../api/client'
import TaskSampleStrip from './TaskSampleStrip'

class FakeEventSource {
  static readonly OPEN = 1
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  readyState = FakeEventSource.OPEN
  close(): void { this.readyState = 2 }
}

function sample(over: Partial<TaskSample> = {}): TaskSample {
  return { filename: 'epoch_1_a.png', mtime: 1, size: 10, epoch: 1, step: null, ...over }
}

beforeEach(() => {
  vi.stubGlobal('EventSource', FakeEventSource)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it('renders one thumbnail per sample', async () => {
  vi.spyOn(api, 'listTaskSamples').mockResolvedValue({
    items: [sample(), sample({ filename: 'epoch_2_a.png', epoch: 2, mtime: 2 })],
    total: 2,
  })
  render(<TaskSampleStrip taskId={7} />)

  const strip = await screen.findByTestId('sample-strip-7')
  await waitFor(() => expect(strip.querySelectorAll('img')).toHaveLength(2))
  // 缩略图走 /samples/{file}?task_id=&w= 代理（带 task_id，否则落全局兜底目录）
  expect(strip.querySelector('img')!.getAttribute('src'))
    .toContain('/samples/epoch_1_a.png?task_id=7&w=')
})

it('renders nothing when the task has no samples', async () => {
  vi.spyOn(api, 'listTaskSamples').mockResolvedValue({ items: [], total: 0 })
  const { container } = render(<TaskSampleStrip taskId={7} />)
  await waitFor(() => expect(container).toBeEmptyDOMElement())
})

it('stays quiet when the listing fails', async () => {
  vi.spyOn(api, 'listTaskSamples').mockRejectedValue(new Error('boom'))
  const { container } = render(<TaskSampleStrip taskId={7} />)
  await waitFor(() => expect(container).toBeEmptyDOMElement())
})

it('opens the lightbox on a thumbnail and steps through the feed', async () => {
  vi.spyOn(api, 'listTaskSamples').mockResolvedValue({
    items: [sample(), sample({ filename: 'epoch_2_a.png', epoch: 2, mtime: 2 })],
    total: 2,
  })
  render(<TaskSampleStrip taskId={7} />)

  const strip = await screen.findByTestId('sample-strip-7')
  await waitFor(() => expect(strip.querySelectorAll('button')).toHaveLength(2))
  fireEvent.click(strip.querySelectorAll('button')[0])

  // 灯箱：计数器 + 原图（无 w= 参数）
  expect(await screen.findByText('1 / 2')).toBeInTheDocument()
  const full = screen.getByAltText(/epoch_1_a\.png/)
  expect(full.getAttribute('src')).toBe('/samples/epoch_1_a.png?task_id=7')

  fireEvent.click(screen.getByLabelText('Next'))
  expect(await screen.findByText('2 / 2')).toBeInTheDocument()

  fireEvent.click(screen.getByLabelText('Close'))
  await waitFor(() => expect(screen.queryByText('2 / 2')).not.toBeInTheDocument())
})
