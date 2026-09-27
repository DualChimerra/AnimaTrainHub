/** ModelsRootMigrateModal: confirm -> start / 409 target_conflict three-way choice
 *  flow (issue #351). SSE doesn't connect under jsdom (useEventStream's internal
 *  EventSource guard), so phase progression is only tested up to running --
 *  done/error are driven by SSE events, which the backend tests cover. */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach } from 'vitest'

import ModelsRootMigrateModal from './ModelsRootMigrateModal'

const mockApi = {
  getModelsRootInfo: vi.fn(),
  startModelsRootMigrate: vi.fn(),
  getModelsRootMigrateStatus: vi.fn(),
}
vi.mock('../api/client', () => ({
  get api() { return mockApi },
}))

const INFO = {
  current: 'G:\\AnimaLoraStudio\\models',
  default: 'G:\\AnimaLoraStudio\\models',
  is_custom: false,
  scan: {
    total_files: 7,
    total_bytes: 3 * 1024 * 1024,
    entries: [
      { name: 'diffusion_models', is_dir: true, files: 2, bytes: 2 * 1024 * 1024 },
      { name: 'vae', is_dir: true, files: 5, bytes: 1024 * 1024 },
    ],
  },
}

/** Shape of the ApiError for a backend 409 models_root.target_conflict (as produced by makeApiError). */
const conflictError = () =>
  Object.assign(new Error('Target already contains a non-empty models directory'), {
    status: 409,
    code: 'models_root.target_conflict',
    detail: {
      target: 'D:\\newroot\\models',
      existing_files: 12,
      existing_bytes: 2 * 1024 * 1024,
      same_name_files: 3,
    },
  })

describe('ModelsRootMigrateModal', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockApi.getModelsRootInfo.mockResolvedValue(INFO)
    mockApi.startModelsRootMigrate.mockResolvedValue({ ok: true })
  })

  it('clicking start migration in confirm phase calls startModelsRootMigrate(target, undefined) and moves to running', async () => {
    render(<ModelsRootMigrateModal target="D:\newroot" onClose={() => {}} onDone={() => {}} />)
    await screen.findByText('Start migration')
    await userEvent.click(screen.getByText('Start migration'))
    expect(mockApi.startModelsRootMigrate).toHaveBeenCalledWith('D:\\newroot', undefined)
    await screen.findByText('Copying…')
  })

  it('409 target_conflict shows stats in conflict phase; skip existing files re-sends with skip and moves to running', async () => {
    mockApi.startModelsRootMigrate
      .mockRejectedValueOnce(conflictError())
      .mockResolvedValueOnce({ ok: true })
    render(<ModelsRootMigrateModal target="D:\newroot" onClose={() => {}} onDone={() => {}} />)
    await screen.findByText('Start migration')
    await userEvent.click(screen.getByText('Start migration'))

    await screen.findByText('Target already contains models data')
    // Interpolated stats: target path + existing file count/size + same-name count
    expect(screen.getByText(/already has 12 files \(2\.0 MB\), 3 of which/)).toBeInTheDocument()

    await userEvent.click(screen.getByText('Skip existing files'))
    await waitFor(() => {
      expect(mockApi.startModelsRootMigrate).toHaveBeenLastCalledWith('D:\\newroot', 'skip')
    })
    await screen.findByText('Copying…')
  })

  it('conflict phase overwrite existing files re-sends with overwrite', async () => {
    mockApi.startModelsRootMigrate
      .mockRejectedValueOnce(conflictError())
      .mockResolvedValueOnce({ ok: true })
    render(<ModelsRootMigrateModal target="D:\newroot" onClose={() => {}} onDone={() => {}} />)
    await screen.findByText('Start migration')
    await userEvent.click(screen.getByText('Start migration'))
    await screen.findByText('Overwrite existing files')

    await userEvent.click(screen.getByText('Overwrite existing files'))
    await waitFor(() => {
      expect(mockApi.startModelsRootMigrate).toHaveBeenLastCalledWith('D:\\newroot', 'overwrite')
    })
  })

  it('cancel in conflict phase returns to confirm (modal stays open, no further requests)', async () => {
    mockApi.startModelsRootMigrate.mockRejectedValueOnce(conflictError())
    const onClose = vi.fn()
    render(<ModelsRootMigrateModal target="D:\newroot" onClose={onClose} onDone={() => {}} />)
    await screen.findByText('Start migration')
    await userEvent.click(screen.getByText('Start migration'))
    await screen.findByText('Cancel')

    await userEvent.click(screen.getByText('Cancel'))
    await screen.findByText('Start migration')
    expect(onClose).not.toHaveBeenCalled()
    expect(mockApi.startModelsRootMigrate).toHaveBeenCalledTimes(1)
  })

  it('a non-conflict error shows the reason in error phase', async () => {
    mockApi.startModelsRootMigrate.mockRejectedValue(
      Object.assign(new Error('Invalid target location'), { status: 422, code: 'models_root.target_invalid' }),
    )
    render(<ModelsRootMigrateModal target="D:\newroot" onClose={() => {}} onDone={() => {}} />)
    await screen.findByText('Start migration')
    await userEvent.click(screen.getByText('Start migration'))
    await screen.findByText(/Invalid target location/)
  })
})
