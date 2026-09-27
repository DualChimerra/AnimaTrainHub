import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { SettingsDrawerProvider } from '../lib/SettingsDrawer'
import {
  ProjectContext,
  SelectedProjectContext,
  type ProjectCtxValue,
  type SelectedProjectValue,
} from '../context/ProjectContext'
import type { ProjectDetail, Version } from '../api/client'
import { DialogProvider } from './Dialog'
import { ToastProvider } from './Toast'
import Sidebar from './Sidebar'

function renderAt(path: string, sticky: SelectedProjectValue | null = null) {
  return render(
    <MemoryRouter
      initialEntries={[path]}
      future={{ v7_relativeSplatPath: true, v7_startTransition: true }}
    >
      <ToastProvider>
        <DialogProvider>
          <SettingsDrawerProvider>
            <SelectedProjectContext.Provider value={sticky}>
              <Sidebar />
            </SelectedProjectContext.Provider>
          </SettingsDrawerProvider>
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>
  )
}

const MOCK_VERSION: Version = {
  id: 7, project_id: 3, label: 'v1', config_name: null, status: 'preparing',
  phase: 'curating', last_failure_reason: null, created_at: 0,
  output_lora_path: null, note: null, trigger_word: '',
}
const MOCK_PROJECT: ProjectDetail = {
  id: 3, slug: 'ganyu', title: 'Ganyu', active_version_id: 7,
  active_version_label: 'v1', active_version_status: 'preparing',
  active_version_phase: 'curating', created_at: 0, updated_at: 0,
  archived_at: null, note: null, versions: [MOCK_VERSION],
  download_image_count: 0, preprocess_image_count: 0,
}
const STICKY: SelectedProjectValue = { project: MOCK_PROJECT, activeVersion: MOCK_VERSION }

const V2: Version = {
  ...MOCK_VERSION, id: 8, label: 'v2-exp', status: 'completed', phase: 'ready',
}

// Live state (inside a project): injects a ProjectContext with callbacks (interactive=true).
function renderLive(path: string, ctx: ProjectCtxValue) {
  return render(
    <MemoryRouter
      initialEntries={[path]}
      future={{ v7_relativeSplatPath: true, v7_startTransition: true }}
    >
      <ToastProvider>
        <DialogProvider>
          <SettingsDrawerProvider>
            <ProjectContext.Provider value={ctx}>
              <Sidebar />
            </ProjectContext.Provider>
          </SettingsDrawerProvider>
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>
  )
}

function makeCtx(versions: Version[]): ProjectCtxValue {
  const project: ProjectDetail = { ...MOCK_PROJECT, versions, active_version_id: versions[0].id }
  return {
    project,
    activeVersion: versions[0],
    reload: vi.fn(),
    onSelectVersion: vi.fn(),
    onCreateVersion: vi.fn(),
    onExportTrain: vi.fn(),
    onDeleteVersion: vi.fn(),
    exporting: false,
  }
}

describe('Sidebar (PP0)', () => {
  it('shows main items + tools with all 5 destinations', () => {
    renderAt('/')
    // main nav
    expect(screen.getByRole('link', { name: /Projects/ })).toHaveAttribute(
      'href',
      '/'
    )
    expect(screen.getByRole('link', { name: /Queue/ })).toHaveAttribute(
      'href',
      '/queue'
    )
    // tools area (redesign dropped the "Tools" group label, just a border-top divider now)
    expect(screen.getByRole('link', { name: /Presets/ })).toHaveAttribute(
      'href',
      '/tools/presets'
    )
    expect(screen.getByRole('link', { name: /Monitor/ })).toHaveAttribute(
      'href',
      '/tools/monitor'
    )
    // Settings is no longer a route link, just a button that opens the right-hand drawer; no href
    expect(screen.getByRole('button', { name: /Settings/ })).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: /Settings/ })).toBeNull()
  })

  it('marks the active route', () => {
    renderAt('/tools/presets')
    const link = screen.getByRole('link', { name: /Presets/ })
    // active link carries ds-is-active (2026 design: white card background + thin border)
    expect(link.className).toMatch(/ds-is-active/)
    // inactive link doesn't
    const queue = screen.getByRole('link', { name: /Queue/ })
    expect(queue.className).not.toMatch(/ds-is-active/)
  })

  it('does not include the removed Datasets link', () => {
    renderAt('/')
    expect(screen.queryByRole('link', { name: /Datasets/ })).toBeNull()
    expect(screen.queryByRole('link', { name: /Config/ })).toBeNull()
  })

  // Sticky "selected project": leaving the project page (e.g. to the queue page) keeps the
  // project section around, for cross-page navigation.
  it('keeps the selected project section on a global page (queue)', () => {
    renderAt('/queue', STICKY)
    // project name still shows
    expect(screen.getByText('Ganyu')).toBeInTheDocument()
    // overview link points back to that project, clickable
    const overview = screen.getByRole('link', { name: /Overview/ })
    expect(overview).toHaveAttribute('href', '/projects/3')
  })

  it('does not highlight overview when off the project route', () => {
    renderAt('/queue', STICKY)
    // on the queue page: queue is highlighted, overview is not (gated by inRoute, to avoid
    // a false positive from currentStep === null)
    const queue = screen.getByRole('link', { name: /Queue/ })
    expect(queue.className).toMatch(/ds-is-active/)
    const overview = screen.getByRole('link', { name: /Overview/ })
    expect(overview.className).not.toMatch(/ds-is-active/)
  })

  it('shows no project section without a sticky selection', () => {
    renderAt('/queue')
    expect(screen.queryByText('Ganyu')).toBeNull()
    expect(screen.queryByRole('link', { name: /Overview/ })).toBeNull()
  })

  // Read-only state (outside the project): the version row shows only the label, the switch
  // pill is not interactive.
  it('read-only version row off the project route: no switch pill', () => {
    renderAt('/queue', STICKY)
    expect(screen.getByText('v1')).toBeInTheDocument()
    expect(screen.queryByTitle('Switch version')).toBeNull()
  })
})

describe('Sidebar version row (live / in project)', () => {
  it('single version: switch pill opens a popover with just "New version"', () => {
    renderLive('/projects/3', makeCtx([MOCK_VERSION]))
    const pill = screen.getByTitle('Switch version')
    expect(pill).toBeInTheDocument()
    fireEvent.click(pill)
    expect(screen.getByRole('menuitem', { name: /New version/ })).toBeInTheDocument()
    expect(screen.queryByRole('menuitem', { name: 'v2-exp' })).toBeNull()
  })

  it('"New version" menu item invokes onCreateVersion', () => {
    const ctx = makeCtx([MOCK_VERSION])
    renderLive('/projects/3', ctx)
    fireEvent.click(screen.getByTitle('Switch version'))
    fireEvent.click(screen.getByRole('menuitem', { name: /New version/ }))
    expect(ctx.onCreateVersion).toHaveBeenCalled()
  })

  it('multi version: switch opens popover and picks a version', () => {
    const ctx = makeCtx([MOCK_VERSION, V2])
    renderLive('/projects/3', ctx)
    const sw = screen.getByTitle('Switch version')
    expect(sw).toBeInTheDocument()
    fireEvent.click(sw)
    // popover lists both versions; clicking the non-active v2-exp -> onSelectVersion(8)
    fireEvent.click(screen.getByRole('menuitem', { name: 'v2-exp' }))
    expect(ctx.onSelectVersion).toHaveBeenCalledWith(8)
  })
})
