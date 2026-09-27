/** R-5 -- data task view (shares /api/queue with the GPU view, resource_class=data):
 *  sectioned rendering / row click goes to the unified detail page /queue/:id /
 *  cancel goes through cancelTask (with confirm). */
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DialogProvider } from '../../components/Dialog'
import { ToastProvider } from '../../components/Toast'
import { api, type Task, type TaskType } from '../../api/client'
import DataJobsPanel from './DataJobsPanel'

function makeJobTask(overrides: Partial<Task> = {}): Task {
  return {
    id: 1, name: 'tag', config_name: 'tag', task_type: 'tag',
    status: 'running', priority: 0,
    created_at: 900, started_at: 1000, finished_at: null,
    pid: 123, exit_code: null, output_dir: null, error_msg: null,
    project_id: 1, version_id: 2,
    params: '{}', params_decoded: { tagger: 'wd14' },
    ...overrides,
  }
}

class FakeEventSource {
  static readonly OPEN = 1
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  readyState = FakeEventSource.OPEN
  close(): void { this.readyState = 2 }
}

beforeEach(() => {
  vi.stubGlobal('EventSource', FakeEventSource)
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
    ok: false, status: 404, json: async () => null, text: async () => '',
    headers: new Headers(),
  } as Response)))
  vi.spyOn(api, 'listProjects').mockResolvedValue([
    { id: 1, title: 'MyProj' } as never,
  ])
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function renderPanel(kind: TaskType | null = null, q?: string) {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <DialogProvider>
          <Routes>
            <Route
              path="/"
              element={
                <DataJobsPanel
                  kind={kind} q={q} historyPage={1} pageSize={20}
                  onHistoryTotal={() => {}} refreshToken={0}
                />
              }
            />
            <Route path="/queue/:id" element={<div data-testid="task-detail-route" />} />
          </Routes>
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>,
  )
}

describe('DataJobsPanel', () => {
  it('renders active/history sections; each row shows the kind label and project name', async () => {
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([
      makeJobTask({ id: 10, task_type: 'tag', status: 'running' }),
    ])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [makeJobTask({
        id: 9, task_type: 'download', status: 'done', finished_at: 2000,
      })],
      total: 1, page: 1, page_size: 20,
    })

    renderPanel()

    await waitFor(() => expect(screen.getByTestId('job-row-10')).toBeInTheDocument())
    expect(within(screen.getByTestId('job-row-10')).getByText('Tagging')).toBeInTheDocument()
    expect(within(screen.getByTestId('job-row-9')).getByText('Download')).toBeInTheDocument()
    await waitFor(() => expect(screen.getAllByText(/MyProj/).length).toBeGreaterThan(0))
  })

  it('data source is /api/queue resource_class=data (kind/q passed through)', async () => {
    const liveSpy = vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    const histSpy = vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderPanel('download', 'usa')
    await waitFor(() =>
      expect(liveSpy).toHaveBeenCalledWith('usa', 'download', 'data'),
    )
    expect(histSpy).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'download', q: 'usa', resourceClass: 'data' }),
    )
  })

  it('clicking a row navigates to the unified detail page /queue/{id} (isomorphic to GPU tasks)', async () => {
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([makeJobTask({ id: 10 })])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })

    renderPanel()
    await waitFor(() => expect(screen.getByTestId('job-row-10')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('job-row-10'))
    await waitFor(() => expect(screen.getByTestId('task-detail-route')).toBeInTheDocument())
  })

  it('cancel requires confirm; confirming calls cancelTask (the unified queue cancel endpoint)', async () => {
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([makeJobTask({ id: 10 })])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [], total: 0, page: 1, page_size: 20,
    })
    const cancelSpy = vi.spyOn(api, 'cancelTask').mockResolvedValue({
      task_id: 10, canceled: true,
    })

    renderPanel()
    await waitFor(() => expect(screen.getByTestId('job-cancel-btn-10')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('job-cancel-btn-10'))

    await waitFor(() => expect(screen.getByRole('dialog')).toBeInTheDocument())
    expect(cancelSpy).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('Cancel task', { selector: 'button[type="submit"]' }))
    await waitFor(() => expect(cancelSpy).toHaveBeenCalledWith(10))
  })

  it('a done row has no cancel button but has a jump button', async () => {
    vi.spyOn(api, 'listQueueLive').mockResolvedValue([])
    vi.spyOn(api, 'listQueueHistory').mockResolvedValue({
      items: [makeJobTask({ id: 9, status: 'done', finished_at: 2000 })],
      total: 1, page: 1, page_size: 20,
    })

    renderPanel()
    await waitFor(() => expect(screen.getByTestId('job-row-9')).toBeInTheDocument())
    expect(screen.queryByTestId('job-cancel-btn-9')).not.toBeInTheDocument()
    expect(screen.getByTestId('job-jump-btn-9')).toBeInTheDocument()
  })
})
