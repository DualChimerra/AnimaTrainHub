/** ProjectOverview regression test.
 *
 *  The pause action of a running training task must appear live: the task
 *  list is Overview's own local state (not a project prop), so Layout's
 *  version_state_changed reload never touches it. is_pausable flips to true
 *  on train_loop_started + the first epoch's auto_epoch_backup_written, so
 *  Overview has to subscribe to those SSE events and refetch the queue;
 *  otherwise the pause item never shows up until a version switch or reload.
 *
 *  jsdom has no EventSource (useEventStream's typeof guard short-circuits), so
 *  a fake is installed to make the hook subscribe, then events are driven by
 *  hand to check that the component refetches listQueue and shows the item. */
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Outlet, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DialogProvider } from '../../components/Dialog'
import { ToastProvider } from '../../components/Toast'
import { api, type ProjectDetail, type Task, type Version } from '../../api/client'
import ProjectOverview from './Overview'

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

function makeVersion(overrides: Partial<Version> = {}): Version {
  return {
    id: 7, project_id: 3, label: 'v1', config_name: 'train',
    status: 'training', phase: 'ready', last_failure_reason: null,
    created_at: 1000, output_lora_path: null, note: null, trigger_word: '',
    ...overrides,
  }
}

function makeProject(overrides: Partial<ProjectDetail> = {}): ProjectDetail {
  return {
    id: 3, slug: 'proj', title: 'Proj', active_version_id: 7,
    active_version_label: 'v1', active_version_status: 'training',
    active_version_phase: null, created_at: 1000, updated_at: 1000,
    archived_at: null, note: null,
    download_image_count: 0, preprocess_image_count: 0,
    versions: [makeVersion()],
    ...overrides,
  }
}

function makeTrainTask(is_pausable: boolean): Task {
  return {
    id: 42, name: 'train', config_name: 'train', status: 'running', priority: 0,
    created_at: 1000, started_at: 1100, finished_at: null, pid: 1234,
    exit_code: null, output_dir: null, error_msg: null,
    project_id: 3, version_id: 7, is_pausable,
  }
}

function renderOverview(project: ProjectDetail) {
  const ctxValue = {
    project,
    activeVersion: project.versions[0] ?? null,
    reload: async () => {},
    onCreateVersion: () => {},
    creatingVersionBusy: false,
  }
  return render(
    <MemoryRouter initialEntries={['/projects/3']}>
      <ToastProvider>
        <DialogProvider>
          <Routes>
            <Route element={<Outlet context={ctxValue} />}>
              <Route path="/projects/:id" element={<ProjectOverview />} />
            </Route>
          </Routes>
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>
  )
}

beforeEach(() => {
  FakeEventSource.instances = []
  vi.stubGlobal('EventSource', FakeEventSource)
  // Captions / outputs / checkpoints all have a .catch: a blanket 404 lets them
  // fail quietly; listQueue is spied separately to control what it returns.
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
    ok: false, status: 404, json: async () => null, text: async () => '',
    headers: new Headers(),
  } as Response)))
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('ProjectOverview pause action refreshes over SSE', () => {
  it('auto_epoch_backup_written refetches the tasks and the pause item appears', async () => {
    // listQueue: is_pausable=false at first (no epoch backup on disk yet), true afterwards.
    let pausable = false
    const listSpy = vi.spyOn(api, 'listQueue')
      .mockImplementation(async () => [makeTrainTask(pausable)])

    renderOverview(makeProject())

    // Open the task's ⋯ menu: cancel is there, pause is not (is_pausable=false).
    const kebab = await screen.findByRole('button', { name: 'Actions' })
    fireEvent.click(kebab)
    await waitFor(() => expect(screen.getByText('取消训练')).toBeInTheDocument())
    expect(screen.queryByText('暂停')).not.toBeInTheDocument()
    const callsBefore = listSpy.mock.calls.length

    // The first epoch backup lands on disk -> is_pausable flips; push an SSE event.
    pausable = true
    await waitFor(() => expect(FakeEventSource.instances.length).toBeGreaterThan(0))
    FakeEventSource.instances[0].emit({ type: 'auto_epoch_backup_written', task_id: 42 })

    // After the refetch the open menu shows pause (no version switch / reload needed).
    await waitFor(() => expect(screen.getByText('暂停')).toBeInTheDocument())
    expect(listSpy.mock.calls.length).toBeGreaterThan(callsBefore)
  })
})
