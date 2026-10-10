import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, expect, it, vi } from 'vitest'
import type { SchemaResponse } from '../../../api/client'
import { configPatch, valuesFromConfig } from './model'
import ParamsManager, { type Remap } from './ParamsManager'
import type { Param } from './types'

const schema: SchemaResponse = {
  groups: [{ key: 'lora', label: 'Network' }, { key: 'training', label: 'Training' }, { key: 'sample', label: 'Sampling' }],
  schema: { properties: {
    lora_type: { type: 'string', enum: ['lora', 'lokr'], group: 'lora' },
    learning_rate: { type: 'number', default: 0.0001, group: 'training' },
    seed: { type: 'integer', default: 42, group: 'training' },
    sample_prompt: { type: 'string', group: 'sample' },
    sample_prompts: { type: 'array', items: { type: 'string' }, group: 'sample' },
    internal: { type: 'string', hidden: true, group: 'training' },
  } },
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(schema), {
    headers: { 'Content-Type': 'application/json' },
  })))
})

it('groups real fields like the training form, prevents duplicates, and saves usable config links', async () => {
  const user = userEvent.setup()
  const save = vi.fn<(params: Param[], remaps: Remap[]) => Promise<boolean>>().mockResolvedValue(true)
  const existing: Param = { id: 'lr', name: 'LR', type: 'number', options: [], config: { key: 'learning_rate' } }
  render(<ParamsManager params={[existing]} runs={[]} onClose={vi.fn()} onSave={save} />)
  const select = screen.getByRole('combobox', { name: 'Training parameter' })
  await waitFor(() => expect(select).toBeEnabled())
  expect(Array.from(select.querySelectorAll('optgroup')).map((g) => g.label)).toEqual([
    'Network settings', 'Training parameters', 'Sampling',
  ])
  expect(select.querySelector('option[value="internal"]')).toBeNull()
  expect(select.querySelector('option[value="learning_rate"]')).toBeDisabled()

  for (const key of ['lora_type', 'seed', 'sample_prompt', 'sample_prompts']) {
    await user.selectOptions(select, key)
    await user.click(within(select.closest('.gr-pm-add-config')!).getByRole('button', { name: 'Add' }))
    expect(select.querySelector(`option[value="${key}"]`)).toBeDisabled()
  }
  await user.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(save).toHaveBeenCalledOnce())
  const params: Param[] = save.mock.calls[0][0]
  expect(params[0]).toEqual(existing)
  const cfg = { learning_rate: 0.00002, lora_type: 'lokr', seed: 123, sample_prompt: 'cat', sample_prompts: ['cat', 'dog'] }
  expect(configPatch(valuesFromConfig(cfg, params), params)).toEqual({ patch: cfg, skipped: [] })
})
