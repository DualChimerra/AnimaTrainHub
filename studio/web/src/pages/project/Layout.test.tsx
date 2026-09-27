/**
 * ProjectLayout version-switch regression test (issue #386).
 *
 * Scenario: switching versions then immediately clicking "Start training"
 * while the activate request is still in flight -- the activeVersion the
 * Train page reads must already be the new version (optimistic update),
 * otherwise it would enqueue the old version.
 */
import { useEffect } from 'react'
import { act, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useOutletContext } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  ProjectSetterContext,
  type ProjectCtxValue,
} from '../../context/ProjectContext'
import type { ProjectDetail, Version } from '../../api/client'
import { DialogProvider } from '../../components/Dialog'

const toastMock = vi.fn()
vi.mock('../../components/Toast', () => ({
  useToast: () => ({ toast: toastMock }),
}))

vi.mock('../../lib/useEventStream', () => ({
  useEventStream: () => {},
}))

const getProjectMock = vi.fn()
const activateVersionMock = vi.fn()
vi.mock('../../api/client', () => ({
  api: {
    getProject: (pid: number) => getProjectMock(pid),
    activateVersion: (pid: number, vid: number) => activateVersionMock(pid, vid),
  },
}))

import ProjectLayout from './Layout'

function makeVersion(id: number, label: string): Version {
  return {
    id, project_id: 3, label, config_name: null, status: 'preparing',
    phase: 'curating', last_failure_reason: null, created_at: 0,
    output_lora_path: null, note: null, trigger_word: '',
  }
}

const V1 = makeVersion(1, 'v1')
const V2 = makeVersion(2, 'v2')
const V3 = makeVersion(3, 'v3')

function makeProject(activeVid: number, versions: Version[]): ProjectDetail {
  return {
    id: 3, slug: 'ganyu', title: 'Ganyu', active_version_id: activeVid,
    active_version_label: 'v2', active_version_status: 'preparing',
    active_version_phase: 'curating', created_at: 0, updated_at: 0,
    archived_at: null, note: null, versions,
    download_image_count: 0, preprocess_image_count: 0,
  }
}

type OutletCtx = {
  activeVersion: Version | null
  setVersionSwitchGuard: (g: (() => Promise<boolean>) | null) => void
}

let lastOutletCtx: OutletCtx | null = null
let probeMounts = 0

/** Simulates the Train and other step pages: reads activeVersion from the Outlet
 * context (this is exactly what Train.tsx uses to enqueue). */
function Probe() {
  const octx = useOutletContext<OutletCtx>()
  lastOutletCtx = octx
  useEffect(() => { probeMounts += 1 }, [])
  return <div data-testid="active-vid">{octx.activeVersion ? String(octx.activeVersion.id) : 'none'}</div>
}

let lastCtx: ProjectCtxValue | null = null

function renderLayout(path = '/projects/3') {
  return render(
    <MemoryRouter
      initialEntries={[path]}
      future={{ v7_relativeSplatPath: true, v7_startTransition: true }}
    >
      <DialogProvider>
        <ProjectSetterContext.Provider value={(v) => { lastCtx = v }}>
          <Routes>
            <Route path="/projects/:pid" element={<ProjectLayout />}>
              <Route index element={<Probe />} />
              <Route path="v/:vid">
                <Route path="train" element={<Probe />} />
              </Route>
            </Route>
          </Routes>
        </ProjectSetterContext.Provider>
      </DialogProvider>
    </MemoryRouter>
  )
}

function deferred<T>() {
  let resolve!: (v: T) => void
  let reject!: (e: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

/** Waits for the Layout's first load to finish: Probe has rendered **and** the
 * setCtx effect has flushed. On slow environments (CI), getProject can resolve
 * before findByTestId's first synchronous check, so the element matches
 * immediately without going through the act-wrapped polling path -> the
 * passive effect hasn't run yet and lastCtx is still null, so we must
 * explicitly wait some more. */
async function ready() {
  await screen.findByTestId('active-vid')
  await waitFor(() => expect(lastCtx).not.toBeNull())
}

describe('ProjectLayout version switching (#386)', () => {
  beforeEach(() => {
    lastCtx = null
    lastOutletCtx = null
    probeMounts = 0
    toastMock.mockClear()
    getProjectMock.mockReset()
    activateVersionMock.mockReset()
    getProjectMock.mockResolvedValue(makeProject(2, [V1, V2]))
  })

  it('activeVersion optimistically switches to the new version while activate is in flight', async () => {
    const d = deferred<{ active_version_id: number }>()
    activateVersionMock.mockReturnValue(d.promise)
    renderLayout()
    await ready()
    expect(screen.getByTestId('active-vid')).toHaveTextContent('2')

    act(() => { lastCtx!.onSelectVersion(1) })
    // The backend hasn't responded yet, but locally it's already v1 -- clicking
    // "Start training" right now would enqueue v1.
    expect(screen.getByTestId('active-vid')).toHaveTextContent('1')

    await act(async () => { d.resolve({ active_version_id: 1 }) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('1')
  })

  it('rolls back to the original version and toasts on activate failure', async () => {
    const d = deferred<{ active_version_id: number }>()
    activateVersionMock.mockReturnValue(d.promise)
    renderLayout()
    await ready()

    act(() => { lastCtx!.onSelectVersion(1) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('1')

    await act(async () => { d.reject(new Error('boom')) })
    await waitFor(() => expect(screen.getByTestId('active-vid')).toHaveTextContent('2'))
    expect(toastMock).toHaveBeenCalled()
  })

  it('switching twice in quick succession: the first failure does not overwrite the second choice', async () => {
    getProjectMock.mockResolvedValue(makeProject(2, [V1, V2, V3]))
    const d1 = deferred<{ active_version_id: number }>()
    const d2 = deferred<{ active_version_id: number }>()
    activateVersionMock.mockImplementation((_pid: number, vid: number) =>
      vid === 1 ? d1.promise : d2.promise)
    renderLayout()
    await ready()

    act(() => { lastCtx!.onSelectVersion(1) })
    act(() => { lastCtx!.onSelectVersion(3) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('3')

    // The first switch's failure lands first: its sequence number is already
    // stale, so it neither rolls back nor toasts.
    await act(async () => { d1.reject(new Error('stale boom')) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('3')
    expect(toastMock).not.toHaveBeenCalled()

    await act(async () => { d2.resolve({ active_version_id: 3 }) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('3')
  })

  it('version-scoped routes: switching versions remounts the step page', async () => {
    activateVersionMock.mockResolvedValue({ active_version_id: 1 })
    renderLayout('/projects/3/v/2/train')
    await ready()
    expect(probeMounts).toBe(1)

    await act(async () => { lastCtx!.onSelectVersion(1) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('1')
    // The old step page instance unmounts and a new one remounts -- local
    // cache / selection state is fully replaced.
    expect(probeMounts).toBe(2)
  })

  it('project-scoped routes (overview): switching versions does not remount', async () => {
    activateVersionMock.mockResolvedValue({ active_version_id: 1 })
    renderLayout()
    await ready()

    await act(async () => { lastCtx!.onSelectVersion(1) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('1')
    expect(probeMounts).toBe(1)
  })

  it('cancels the switch when the switch guard returns false', async () => {
    renderLayout('/projects/3/v/2/train')
    await ready()
    act(() => { lastOutletCtx!.setVersionSwitchGuard(() => Promise.resolve(false)) })

    await act(async () => { lastCtx!.onSelectVersion(1) })
    expect(screen.getByTestId('active-vid')).toHaveTextContent('2')
    expect(activateVersionMock).not.toHaveBeenCalled()
  })
})
