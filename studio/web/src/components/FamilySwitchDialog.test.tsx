import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import '../i18n'

vi.mock('../api/client', () => ({
  api: { switchModelFamily: vi.fn() },
}))

import { api } from '../api/client'
import FamilySwitchDialog from './FamilySwitchDialog'

const switched = { model_family: 'krea2', sample_sampler_name: 'euler' }
const response = {
  config: switched,
  changes: [
    { field: 'model_family', from: 'anima', to: 'krea2' },
    { field: 'transformer_path', from: 'G:/models/anima.safetensors', to: 'G:/models/krea2-raw-bf16.safetensors' },
    { field: 't5_tokenizer_path', from: 'G:/models/t5_tokenizer', to: '' },
    { field: 'sample_sampler_name', from: 'er_sde', to: 'euler' },
    { field: 'shuffle_caption', from: true, to: false },
  ],
}

describe('FamilySwitchDialog', () => {
  it('renders grouped changes and applies switched config on confirm', async () => {
    vi.mocked(api.switchModelFamily).mockResolvedValue(response)
    const onApply = vi.fn()
    render(
      <FamilySwitchDialog
        target="krea2"
        config={{ model_family: 'anima' }}
        onApply={onApply}
        onCancel={() => {}}
      />,
    )
    // paths section (monospace old/new comparison) and params section render as separate
    // groups; the model_family field's own row is not listed
    await screen.findByText('G:/models/krea2-raw-bf16.safetensors')
    expect(screen.getByText('Transformer Path')).toBeInTheDocument()
    expect(screen.getByText('Sample Sampler Name')).toBeInTheDocument()
    expect(screen.queryByText('Model Family')).not.toBeInTheDocument()
    // empty-value placeholder + localized boolean
    expect(screen.getByText('(empty)')).toBeInTheDocument()
    expect(screen.getByText(/Yes/)).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Switch' }))
    expect(onApply).toHaveBeenCalledWith(switched)
  })

  it('cancel keeps old values and never applies', async () => {
    vi.mocked(api.switchModelFamily).mockResolvedValue(response)
    const onApply = vi.fn()
    const onCancel = vi.fn()
    render(
      <FamilySwitchDialog
        target="krea2"
        config={{ model_family: 'anima' }}
        onApply={onApply}
        onCancel={onCancel}
      />,
    )
    await screen.findByText('Transformer Path')
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onCancel).toHaveBeenCalled()
    expect(onApply).not.toHaveBeenCalled()
  })

  it('disables confirm and surfaces error when preview fails', async () => {
    vi.mocked(api.switchModelFamily).mockRejectedValue(new Error('boom'))
    render(
      <FamilySwitchDialog
        target="krea2"
        config={{ model_family: 'anima' }}
        onApply={() => {}}
        onCancel={() => {}}
      />,
    )
    await waitFor(() => {
      expect(screen.getByText(/boom/)).toBeInTheDocument()
    })
    expect(screen.getByRole('button', { name: 'Switch' })).toBeDisabled()
  })
})
