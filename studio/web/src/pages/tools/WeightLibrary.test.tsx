import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useTranslation } from 'react-i18next'
import { ModelsSection } from './Settings'
import { ToastProvider } from '../../components/Toast'
import { api, type ModelsCatalog } from '../../api/client'

const reload = vi.fn(async () => {})
const start = vi.fn(async () => {})
const row = (label: string, families: string[], exists = true) => ({
  label, path: `C:/weights/${label}`, value: `C:/weights/${label}`, families, exists,
  is_current: false, size: 1024, download_id: null, download_variant: null, status_key: null,
})
const catalog = {
  models_root: 'C:/weights', downloads: {}, t5_tokenizer: { name: 'T5 tokenizer', files: [] },
  weight_library: {
    checkpoints: { folders: ['C:/weights/diffusion_models'], errors: [], rows: [row('moss.safetensors', ['moss_image']), row('unknown.safetensors', [])] },
    vae: { folders: ['C:/weights/vae'], errors: [], rows: [row('flux2.safetensors', ['moss_image'])] },
    text_encoders: { folders: ['C:/weights/text_encoders'], errors: [], rows: [row('qwen35', ['moss_image'], false)] },
  },
} as unknown as ModelsCatalog

function Harness() {
  const { t } = useTranslation()
  return <ToastProvider><ModelsSection catalog={catalog} busy={new Set()} start={start} reloadCatalog={reload} catalogError={null} t={t} /></ToastProvider>
}

beforeEach(() => {
  vi.restoreAllMocks()
  reload.mockClear()
  vi.spyOn(api, 'getSecrets').mockResolvedValue({ models: { root: null, auto_sync_paths: true } } as Awaited<ReturnType<typeof api.getSecrets>>)
  vi.spyOn(api, 'updateSecrets').mockResolvedValue({} as Awaited<ReturnType<typeof api.updateSecrets>>)
  vi.spyOn(api, 'selectWeight').mockResolvedValue(catalog)
})

describe('weight library', () => {
  it('keeps three generic categories and disables unknown or missing weights', async () => {
    render(<Harness />)
    for (const name of ['Checkpoints', 'VAE', 'Text Encoders']) expect(screen.getByRole('heading', { name })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'moss.safetensors' })).toBeEnabled()
    expect(screen.getByRole('radio', { name: 'unknown.safetensors' })).toBeDisabled()
    expect(screen.getByRole('radio', { name: 'qwen35' })).toBeDisabled()
    await waitFor(() => expect(api.getSecrets).toHaveBeenCalled())
  })

  it('selects an exact path through compatibility validation', async () => {
    render(<Harness />)
    fireEvent.click(screen.getByRole('radio', { name: 'moss.safetensors' }))
    await waitFor(() => expect(api.selectWeight).toHaveBeenCalledWith('checkpoints', 'C:/weights/moss.safetensors'))
    await waitFor(() => expect(reload).toHaveBeenCalled())
  })
})
