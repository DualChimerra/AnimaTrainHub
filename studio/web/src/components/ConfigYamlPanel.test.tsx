import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ConfigData } from '../api/client'
import { api } from '../api/client'
import ConfigYamlPanel from './ConfigYamlPanel'
import { ToastProvider } from './Toast'

// R4(D3): the component no longer prunes/renders config locally (pruneInactiveConfig/
// configToYaml were removed); instead it calls POST /api/schema/preview-yaml after a
// 300ms debounce, the same serialization path used on disk. The test mocks that
// endpoint and pins down "the request carries the current config + shows the returned text".

function renderPanel(config: ConfigData, hint?: string) {
  return render(
    <ToastProvider>
      <ConfigYamlPanel config={config} fileLabel="config.yaml" hint={hint} />
    </ToastProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('ConfigYamlPanel', () => {
  it('debounces then renders backend preview text and counts top-level keys', async () => {
    const spy = vi
      .spyOn(api, 'previewConfigYaml')
      .mockResolvedValue({ yaml: 'optimizer_type: adamw\nlora_rank: 64\n' })
    renderPanel({ optimizer_type: 'adamw', lora_rank: 64 })
    await waitFor(
      () => {
        const pre = document.querySelector('pre')
        expect(pre?.textContent).toContain('optimizer_type: adamw')
      },
      { timeout: 2000 },
    )
    expect(spy).toHaveBeenCalledWith({ optimizer_type: 'adamw', lora_rank: 64 })
    expect(screen.getByText('config.yaml')).toBeInTheDocument()
    // schema.fieldCount -> 2 top-level keys (top-level line count)
    expect(screen.getByText('2 items')).toBeInTheDocument()
  })

  it('shows the hint when provided', async () => {
    vi.spyOn(api, 'previewConfigYaml').mockResolvedValue({ yaml: 'lora_rank: 64\n' })
    renderPanel({ lora_rank: 64 }, 'Has unsaved changes')
    expect(screen.getByText('Has unsaved changes')).toBeInTheDocument()
    await waitFor(
      () => expect(document.querySelector('pre')?.textContent).toContain('lora_rank'),
      { timeout: 2000 },
    )
  })
})
