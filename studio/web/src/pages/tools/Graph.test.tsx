/** GraphPage smoke: a board renders its cards in every view, empty places
 *  appear for the filtered values, and a queue task that matches an empty
 *  place is offered there. */
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { DialogProvider } from '../../components/Dialog'
import { ToastProvider } from '../../components/Toast'
import GraphPage from './Graph'
import type { Param, Run } from './graph/types'

const PARAMS: Param[] = [
  { id: 'lr', name: 'LR', type: 'number', config: { key: 'learning_rate' }, options: [
    { id: 'a', value: 0.0001 }, { id: 'b', value: 0.00002 },
  ] },
  { id: 'scheduler', name: 'Scheduler', type: 'list', options: [
    { id: 'constant', label: 'Constant', config: { apply: { lr_scheduler: 'none' } } },
    { id: 'cosine', label: 'Cosine', config: { apply: { lr_scheduler: 'cosine' } } },
  ] },
]

function run(id: number, values: Run['values'], extra: Partial<Run> = {}): Run {
  return {
    id, board_id: 1, name: `run ${id}`, values, status: 'done', notes: '', link: '', favorite: false,
    rating: null, best_step: null, tags: [], task_id: null, project_id: null, version_id: null,
    created_at: id, updated_at: id, images: [], ...extra,
  }
}

const RUNS = [
  run(1, { lr: 0.0001, scheduler: 'cosine' }, { rating: 3 }),
  run(2, { lr: 0.00002, scheduler: 'cosine' }),
]

const fetchMock = vi.fn()
function json(body: unknown) {
  return Promise.resolve({
    ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body),
    headers: new Headers({ 'content-type': 'application/json' }),
  } as Response)
}

beforeEach(() => {
  localStorage.clear()
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string) => {
    if (url.endsWith('/api/graph/boards')) {
      return json({ items: [{ id: 1, name: 'Style', note: '', run_count: 2, image_count: 0, created_at: 1, updated_at: 1 }] })
    }
    if (url.endsWith('/api/graph/boards/1')) {
      return json({ board: { id: 1, name: 'Style', note: '', params: PARAMS, views: [], created_at: 1, updated_at: 1 }, runs: RUNS })
    }
    if (url.includes('/api/graph/tasks')) {
      return json({ items: [{
        id: 77, name: 't', status: 'pending', created_at: 1, started_at: null, finished_at: null,
        project_id: 1, version_id: 2, project_title: 'Proj', version_label: 'v2', note: '',
        has_config: true, config: { learning_rate: 1e-4, lr_scheduler: 'none' },
      }] })
    }
    return json({})
  })
})

function renderPage() {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <DialogProvider>
          <GraphPage />
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>,
  )
}

describe('GraphPage', () => {
  it('shows the board and its cards', async () => {
    renderPage()
    expect(await screen.findByText('run 1')).toBeInTheDocument()
    expect(screen.getByText('run 2')).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: 'Board' })).toHaveValue('1')
  })

  it('lays out empty places for the filtered values and offers matching queue tasks', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByText('run 1')
    // LR: both values, Scheduler: both values -> 4 places, 2 filled
    await user.click(screen.getByRole('button', { name: /^LR/, expanded: false }))
    const lrBlock = screen.getByText('LR', { selector: '.gr-fsec-title' }).closest('section')!
    for (const box of within(lrBlock).getAllByRole('checkbox').slice(0, 2)) await user.click(box)
    await user.click(screen.getByRole('button', { name: /^Scheduler/, expanded: false }))
    const schBlock = screen.getByText('Scheduler', { selector: '.gr-fsec-title' }).closest('section')!
    await user.click(within(schBlock).getByRole('checkbox', { name: /Constant/ }))
    await user.click(within(schBlock).getByRole('checkbox', { name: /Cosine/ }))
    await user.click(screen.getByRole('button', { name: /Empty places/ }))
    await waitFor(() => expect(screen.getByText(/2 of 4 combinations tried/)).toBeInTheDocument())
    expect(screen.getAllByText('Not tried yet')).toHaveLength(2)
    // the pending task has LR 1e-4 + Constant: it sits in that empty place
    expect(await screen.findByText(/#77/)).toBeInTheDocument()
  })

  it('renders the table, matrix and web views', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByText('run 1')
    await user.click(screen.getByRole('tab', { name: /Table/ }))
    expect(screen.getByRole('table')).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: /Matrix/ }))
    await user.click(screen.getByRole('tab', { name: /Web/ }))
    expect(screen.getByText(/2 cards · 1 links/)).toBeInTheDocument()
  })
})
