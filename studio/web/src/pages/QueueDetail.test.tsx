/** QueueDetail page component-level regression tests.
 *
 *  Currently only covers the SnapshotConfigTab refetch trap: the parent
 *  component shallow-clones task every 2s for the elapsed-time tick; the old
 *  implementation used [task] as a dep, which made the snapshot config
 *  refetch along with it every 2s -- browser stutter, loading flash. */
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ToastProvider } from '../components/Toast'
import { DialogProvider } from '../components/Dialog'
import type { Task } from '../api/client'
import QueueDetailPage, { SnapshotConfigTab } from './QueueDetail'

const SNAPSHOT_URL_PREFIX = '/api/queue/'
const SNAPSHOT_URL_SUFFIX = '/snapshot/config'

const fetchMock = vi.fn()

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: 1, name: 'train', config_name: 'train',
    status: 'running', priority: 0,
    created_at: 1000, started_at: 1100, finished_at: null,
    pid: 1234, exit_code: null, output_dir: null, error_msg: null,
    ...overrides,
  }
}

function snapshotResponse() {
  const body = { yaml: 'key: val\n', config: { key: 'val' } }
  return {
    ok: true, status: 200,
    json: async () => body,
    text: async () => JSON.stringify(body),
    headers: new Headers({ 'content-type': 'application/json' }),
  } as Response
}

function snapshotCallCount(): number {
  return fetchMock.mock.calls.filter(([url]) =>
    typeof url === 'string'
    && url.startsWith(SNAPSHOT_URL_PREFIX)
    && url.endsWith(SNAPSHOT_URL_SUFFIX),
  ).length
}

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string) => {
    if (url.startsWith(SNAPSHOT_URL_PREFIX) && url.endsWith(SNAPSHOT_URL_SUFFIX)) {
      return Promise.resolve(snapshotResponse())
    }
    return Promise.resolve({
      ok: false, status: 404, json: async () => null, text: async () => '',
      headers: new Headers(),
    } as Response)
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function setup(task: Task | null) {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <SnapshotConfigTab task={task} />
      </ToastProvider>
    </MemoryRouter>
  )
}

describe('SnapshotConfigTab', () => {
  it('a 2s shallow clone of task by the parent does not trigger a refetch -- the snapshot is immutable', async () => {
    const task = makeTask()
    const view = setup(task)

    await waitFor(() => expect(snapshotCallCount()).toBe(1))

    // Simulate the parent's 2s tick: a shallow clone produces a new reference,
    // id / started_at stay the same
    for (let i = 0; i < 5; i++) {
      view.rerender(
        <MemoryRouter>
          <ToastProvider>
            <SnapshotConfigTab task={{ ...task }} />
          </ToastProvider>
        </MemoryRouter>
      )
    }

    // Wait a bit for any extra useEffect to settle
    await new Promise((r) => setTimeout(r, 20))
    expect(snapshotCallCount()).toBe(1)
  })

  it('a pending -> running transition (started_at null->number) triggers one refetch', async () => {
    const view = setup(makeTask({ status: 'pending', started_at: null }))
    await waitFor(() => expect(snapshotCallCount()).toBe(1))

    view.rerender(
      <MemoryRouter>
        <ToastProvider>
          <SnapshotConfigTab task={makeTask({ status: 'running', started_at: 1234 })} />
        </ToastProvider>
      </MemoryRouter>
    )

    await waitFor(() => expect(snapshotCallCount()).toBe(2))
  })
})

// ── SSE refresh of the pause button (QueueDetailPage header) ────────────────
//
// regression: after resume/start, is_pausable flips to true via
// train_loop_started + auto_epoch_backup_written, but the header's shared task
// used to reload only on task_state_changed, missing these two events -> the
// pause button never appeared until navigating to /queue and back (full page
// remount).
//
// jsdom has no EventSource by default (useEventStream's internal typeof guard
// short-circuits), so we install a fake one to make the hook actually
// subscribe, then manually drive an event to verify the component re-fetches
// getTask and shows the button.
class FakeEventSource {
  static instances: FakeEventSource[] = []
  static readonly OPEN = 1
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  readyState = FakeEventSource.OPEN
  constructor(public url: string) { FakeEventSource.instances.push(this) }
  close(): void { this.readyState = 2 }
  emit(evt: unknown): void { this.onmessage?.({ data: JSON.stringify(evt) }) }
}

const QUEUE_ITEM_URL = '/api/queue/119'

function queueItemResponse(task: Task) {
  return {
    ok: true, status: 200,
    json: async () => task,
    text: async () => JSON.stringify(task),
    headers: new Headers({ 'content-type': 'application/json' }),
  } as Response
}

function renderDetailPage() {
  return render(
    <MemoryRouter initialEntries={['/queue/119']}>
      <ToastProvider>
        <DialogProvider>
          <Routes>
            <Route path="/queue/:id" element={<QueueDetailPage />} />
          </Routes>
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>
  )
}

function getTaskCalls(): number {
  return fetchMock.mock.calls.filter(([u]) => u === QUEUE_ITEM_URL).length
}

describe('QueueDetailPage pause button SSE refresh', () => {
  beforeEach(() => {
    FakeEventSource.instances = []
    vi.stubGlobal('EventSource', FakeEventSource)
  })

  it('an auto_epoch_backup_written event triggers a refetch -> pause button appears', async () => {
    // getTask: first fetch has is_pausable=false (train loop / first epoch
    // backup not ready yet), then is_pausable=true (first epoch backup written).
    let pausable = false
    fetchMock.mockImplementation((url: string) => {
      if (url === QUEUE_ITEM_URL) {
        return Promise.resolve(queueItemResponse(makeTask({
          id: 119, status: 'running', is_pausable: pausable,
        })))
      }
      return Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })

    renderDetailPage()

    // Initially: the running header has rendered (title includes the task
    // name), but is_pausable=false -> pause button absent
    await waitFor(() => expect(screen.getByText(/^#119 · /)).toBeInTheDocument())
    expect(screen.queryByTestId('detail-pause-btn')).not.toBeInTheDocument()

    // Backend writes the first epoch backup -> is_pausable flips; push an SSE event
    pausable = true
    await waitFor(() => expect(FakeEventSource.instances.length).toBeGreaterThan(0))
    FakeEventSource.instances[0].emit({ type: 'auto_epoch_backup_written', task_id: 119 })

    // After the refetch, the pause button appears (no page remount needed)
    await waitFor(() => expect(screen.getByTestId('detail-pause-btn')).toBeInTheDocument())
  })

  it('an event for a different task does not trigger a refetch', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url === QUEUE_ITEM_URL) {
        return Promise.resolve(queueItemResponse(makeTask({
          id: 119, status: 'running', is_pausable: false,
        })))
      }
      return Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })

    renderDetailPage()
    await waitFor(() => expect(screen.getByText(/^#119 · /)).toBeInTheDocument())
    const before = getTaskCalls()

    await waitFor(() => expect(FakeEventSource.instances.length).toBeGreaterThan(0))
    // A backup event for a different task (task_id=999) -- should not trigger a refetch on this page
    FakeEventSource.instances[0].emit({ type: 'auto_epoch_backup_written', task_id: 999 })
    await new Promise((r) => setTimeout(r, 150))

    expect(getTaskCalls()).toBe(before)
  })
})

// ── P-H task-type-specific tabs ──────────────────────────────────────────────
describe('QueueDetailPage task-type-specific tabs (P-H)', () => {
  it('a generate task only keeps overview+log, hides monitor/eval/snapshot, and deep-links to the generate result', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url === QUEUE_ITEM_URL) {
        return Promise.resolve(queueItemResponse(makeTask({
          id: 119, task_type: 'generate', status: 'done', finished_at: 1200,
        })))
      }
      return Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })

    renderDetailPage()

    // The deep-link button appears once the task loads (proving it's a
    // generate task and has hydrated)
    await waitFor(() => expect(screen.getByTestId('detail-view-generate')).toBeInTheDocument())
    // Train-only tabs are hidden
    expect(screen.queryByText('Monitor')).not.toBeInTheDocument()
    expect(screen.queryByText('Metrics')).not.toBeInTheDocument()
    expect(screen.queryByText('Snapshot config')).not.toBeInTheDocument()
    expect(screen.queryByText('Outputs')).not.toBeInTheDocument()
    // The overview tab is still there
    expect(screen.getByText('Overview')).toBeInTheDocument()
  })
})
