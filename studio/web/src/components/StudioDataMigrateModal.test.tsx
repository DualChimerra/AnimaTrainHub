/** StudioDataMigrateModal: confirm-state info display / start call / uncloseable while running.
 *  SSE doesn't connect under jsdom (useEventStream's internal EventSource guard), so phase
 *  progression is only tested up to running - done/error are driven by SSE events, covered
 *  by the backend tests that publish them. */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach } from 'vitest'

import StudioDataMigrateModal from './StudioDataMigrateModal'

const mockApi = {
  getStudioDataInfo: vi.fn(),
  startStudioDataMigrate: vi.fn(),
  getStudioDataMigrateStatus: vi.fn(),
}
vi.mock('../api/client', () => ({
  get api() { return mockApi },
}))

const INFO = {
  current: 'G:\\AnimaLoraStudio\\studio_data',
  default: 'G:\\AnimaLoraStudio\\studio_data',
  is_custom: false,
  scan: {
    total_files: 42,
    total_bytes: 5 * 1024 * 1024,
    entries: [
      { name: 'projects', is_dir: true, files: 30, bytes: 4 * 1024 * 1024 },
      { name: 'studio.db', is_dir: false, files: 1, bytes: 1024 * 1024 },
    ],
  },
}

describe('StudioDataMigrateModal', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.getStudioDataInfo.mockResolvedValue(INFO)
    mockApi.startStudioDataMigrate.mockResolvedValue({ ok: true })
  })

  it('confirm state shows source/target paths + file count & size + top-level entries', async () => {
    render(
      <StudioDataMigrateModal target="D:\data" onClose={() => {}} onRestart={() => {}} />,
    )
    await waitFor(() => {
      expect(screen.getByText(/42 files/)).toBeInTheDocument()
    })
    expect(screen.getByText(/5\.0 MB/)).toBeInTheDocument()
    // target shows the actual landing directory target\studio_data (the user picked the parent dir)
    expect(screen.getByText('D:\\data\\studio_data')).toBeInTheDocument()
    expect(screen.getByText('projects/')).toBeInTheDocument()
    expect(screen.getByText('studio.db')).toBeInTheDocument()
  })

  it('clicking start migration -> calls startStudioDataMigrate(target) and enters running (uncloseable)', async () => {
    const onClose = vi.fn()
    render(
      <StudioDataMigrateModal target="D:\data" onClose={onClose} onRestart={() => {}} />,
    )
    await screen.findByText('Start migration')
    await userEvent.click(screen.getByText('Start migration'))
    expect(mockApi.startStudioDataMigrate).toHaveBeenCalledWith('D:\\data')
    await screen.findByText('Copying…')
    // running state: the header's close x is not rendered
    expect(screen.queryByLabelText('Close')).not.toBeInTheDocument()
    expect(onClose).not.toHaveBeenCalled()
  })

  it('confirm state cancel -> onClose', async () => {
    const onClose = vi.fn()
    render(
      <StudioDataMigrateModal target="D:\data" onClose={onClose} onRestart={() => {}} />,
    )
    await screen.findByText('Cancel')
    await userEvent.click(screen.getByText('Cancel'))
    expect(onClose).toHaveBeenCalled()
  })

  it('start rejected by the backend (422) -> error state shows the reason, closeable', async () => {
    mockApi.startStudioDataMigrate.mockRejectedValue(new Error('Target directory not empty'))
    render(
      <StudioDataMigrateModal target="D:\data" onClose={() => {}} onRestart={() => {}} />,
    )
    await screen.findByText('Start migration')
    await userEvent.click(screen.getByText('Start migration'))
    await screen.findByText(/Target directory not empty/)
    expect(screen.getByLabelText('Close')).toBeInTheDocument()
  })
})
