import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { SchemaResponse } from '../api/client'
import i18n from '../i18n'
import { schemaEnumLabel } from '../lib/schema'
import SchemaForm from './SchemaForm'

const parameters = {
  peak1_position: 0.525, peak1_width: 0.55, peak1_weight: 0.15,
  peak2_position: 0.85, peak2_width: 1.2, peak2_weight: 0.35,
  background_mean: -3.2, background_width: 2.2, background_weight: 0.45,
  uniform_weight: 0.05,
}

const schema: SchemaResponse = {
  groups: [{ key: 'timestep_sampling', label: 'Timestep sampling' }],
  schema: { properties: {
    timestep_sampling: {
      type: 'string', enum: ['logit_normal', 'style_friendly', 'dual_peak'],
      default: 'logit_normal', group: 'timestep_sampling', advanced: true,
    },
    timestep_shift: {
      type: 'number', default: 3, group: 'timestep_sampling', advanced: true,
      show_when: 'timestep_sampling!=uniform&&timestep_sampling!=style_friendly&&timestep_sampling!=dual_peak',
    },
    ...Object.fromEntries(Object.entries(parameters).map(([key, value]) => [
      `dual_peak_${key}`, {
        type: 'number', default: value, group: 'timestep_sampling', advanced: true,
        show_when: 'timestep_sampling==dual_peak',
      },
    ])),
  } },
}

afterEach(async () => { await i18n.changeLanguage('zh') })

describe('Dual Peak sampling controls', () => {
  it.each(['ru', 'en', 'zh'])('has a translated option and parameter help in %s', async (lang) => {
    await i18n.changeLanguage(lang)
    expect(schemaEnumLabel('timestep_sampling', 'dual_peak', i18n.t)).toContain('Dual Peak')
    for (const key of Object.keys(parameters)) {
      expect(i18n.exists(`schema.descriptions.dual_peak_${key}`, { lng: lang, fallbackLng: false })).toBe(true)
    }
  })

  it('shows custom controls only for dual_peak, preserves values and edits the second peak', async () => {
    await i18n.changeLanguage('ru')
    const onChange = vi.fn()
    const values = {
      timestep_sampling: 'dual_peak',
      ...Object.fromEntries(Object.entries(parameters).map(([k, v]) => [`dual_peak_${k}`, v])),
    }
    const { rerender } = render(<SchemaForm schema={schema} values={values} onChange={onChange} advancedMode />)
    expect(screen.getByRole('option', { name: 'Dual Peak — два пика для стиля' })).toBeInTheDocument()
    expect(screen.queryByText('timestep_shift')).not.toBeInTheDocument()
    expect(screen.getAllByRole('textbox')).toHaveLength(10)
    const secondPeak = screen.getByDisplayValue('0.85')
    fireEvent.change(secondPeak, { target: { value: '0.82' } })
    fireEvent.blur(secondPeak)
    expect(onChange).toHaveBeenCalledWith({ ...values, dual_peak_peak2_position: 0.82 })
    onChange.mockClear()
    rerender(<SchemaForm schema={schema} values={{ ...values, timestep_sampling: 'logit_normal' }} onChange={onChange} advancedMode />)
    expect(screen.queryByText('dual_peak_peak1_position')).not.toBeInTheDocument()
    expect(screen.getByText('timestep_shift')).toBeInTheDocument()
    expect(onChange).not.toHaveBeenCalled()
  })
})
