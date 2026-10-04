import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { ConfigData, SchemaResponse } from '../api/client'
import SchemaForm from './SchemaForm'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, opts?: { defaultValue?: string; n?: number }) => {
      const dict: Record<string, string> = {
        'schema.groups.training': 'Training',
        'schema.groups.timestepSampling': 'Timestep Sampling',
        'schema.disableHints.learning_rate': 'Prodigy controls the learning rate',
        'schema.disableHints.lr_scheduler': 'Schedule-Free has its own scheduler',
        'schema.disableHints.timestep_sampling': 'InfoNoise controls timestep sampling',
        'schema.descriptions.timestep_sampling': 'Normal timestep description',
        'schema.altDescriptions.timestep_sampling': 'InfoNoise takes over timestep sampling',
        'schema.enums.optimizer_type.adamw': 'AdamW',
        'schema.enums.optimizer_type.prodigy': 'Prodigy',
        'schema.enums.optimizer_type.prodigy_plus_schedulefree': 'Prodigy+ Schedule-Free',
        'schema.enums.lr_scheduler.none': 'None',
        'schema.enums.lr_scheduler.cosine': 'Cosine',
        'schema.enums.timestep_sampling.logit_normal': 'Logit Normal',
        'schema.enums.timestep_sampling.uniform': 'Uniform',
        'field.useGlobal': 'Use global',
        'field.yes': 'Yes',
        'field.no': 'No',
      }
      if (key === 'schema.fieldCount') return `${opts?.n ?? 0} fields`
      return dict[key] ?? opts?.defaultValue ?? key
    },
  }),
}))

const schema: SchemaResponse = {
  groups: [
    { key: 'training', label: 'Training' },
    { key: 'timestep_sampling', label: 'Timestep Sampling' },
  ],
  schema: {
    properties: {
      optimizer_type: {
        type: 'string',
        enum: ['adamw', 'prodigy', 'prodigy_plus_schedulefree', 'automagic', 'simplified_ademamix'],
        default: 'adamw',
        group: 'training',
        description: 'Optimizer',
      },
      learning_rate: {
        type: 'number',
        default: 0.0001,
        group: 'training',
        description: 'Learning rate',
        disable_when: 'optimizer_type==prodigy||optimizer_type==prodigy_plus_schedulefree',
        disable_value: 1,
      },
      lr_scheduler: {
        type: 'string',
        enum: ['none', 'cosine'],
        default: 'none',
        group: 'training',
        description: 'Scheduler',
        disable_when: 'optimizer_type==prodigy_plus_schedulefree',
      },
      timestep_sampling: {
        type: 'string',
        enum: ['logit_normal', 'uniform'],
        default: 'logit_normal',
        group: 'timestep_sampling',
        description: 'Backend normal description',
        alt_description_when: 'infonoise_enabled==true',
        disable_when: 'infonoise_enabled==true',
        advanced: true,
      },
      infonoise_enabled: {
        type: 'boolean',
        default: false,
        group: 'timestep_sampling',
        description: 'InfoNoise',
        advanced: true,
      },
      wandb_enabled: {
        anyOf: [{ type: 'boolean' }, { type: 'null' }],
        default: null,
        group: 'training',
        description: 'WandB',
      },
    },
  },
}

describe('SchemaForm takeover behavior', () => {
  it('resets and disables fields taken over by disable_when', async () => {
    const onChange = vi.fn()
    const values: ConfigData = {
      optimizer_type: 'prodigy_plus_schedulefree',
      learning_rate: 0.0001,
      lr_scheduler: 'cosine',
      infonoise_enabled: false,
      timestep_sampling: 'logit_normal',
    }

    render(
      <SchemaForm
        schema={schema}
        values={values}
        onChange={onChange}
        advancedMode
      />,
    )

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith({
        ...values,
        learning_rate: 1,
        lr_scheduler: 'none',
      })
    })
    const learningRateInput = screen.getByRole('textbox') as HTMLInputElement
    expect(learningRateInput).toBeDisabled()
    expect(learningRateInput.value).toBe('0.0001')
    expect(screen.getByText('Prodigy controls the learning rate')).toBeInTheDocument()
    const schedulerSelect = screen.getAllByRole('combobox')[1] as HTMLSelectElement
    expect(schedulerSelect).toBeDisabled()
    expect(schedulerSelect.value).toBe('cosine')
    expect(screen.getByText('Schedule-Free has its own scheduler')).toBeInTheDocument()
  })

  it('locks learning rate to 1 for regular Prodigy', async () => {
    const onChange = vi.fn()
    const values: ConfigData = {
      optimizer_type: 'prodigy',
      learning_rate: 0.0001,
      lr_scheduler: 'none',
      infonoise_enabled: false,
      timestep_sampling: 'logit_normal',
    }

    render(
      <SchemaForm
        schema={schema}
        values={values}
        onChange={onChange}
        advancedMode
      />,
    )

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith({
        ...values,
        learning_rate: 1,
      })
    })
    expect(screen.getByRole('textbox')).toBeDisabled()
    expect(screen.getByText('Prodigy controls the learning rate')).toBeInTheDocument()
  })

  it('renders nullable booleans as tri-state selects without coercing null to false', () => {
    const onChange = vi.fn()

    render(
      <SchemaForm
        schema={schema}
        values={{
          optimizer_type: 'adamw',
          learning_rate: 0.0001,
          lr_scheduler: 'none',
          infonoise_enabled: false,
          timestep_sampling: 'logit_normal',
          wandb_enabled: null,
        }}
        onChange={onChange}
        advancedMode
      />,
    )

    const wandbSelect = screen.getByDisplayValue('Use global') as HTMLSelectElement
    expect(wandbSelect.value).toBe('')
    expect(screen.queryByRole('checkbox', { name: /wandb/i })).not.toBeInTheDocument()
    expect(onChange).not.toHaveBeenCalled()
  })

  it('disables InfoNoise-controlled timestep fields and uses frontend alt descriptions', () => {
    const { container } = render(
      <SchemaForm
        schema={schema}
        values={{
          optimizer_type: 'adamw',
          learning_rate: 0.0001,
          lr_scheduler: 'none',
          infonoise_enabled: true,
          timestep_sampling: 'uniform',
        }}
        onChange={() => {}}
        advancedMode
      />,
    )

    const timestepSelect = screen.getAllByRole('combobox').find((el) => (el as HTMLSelectElement).value === 'uniform') as HTMLSelectElement
    expect(timestepSelect).toBeDisabled()
    expect(screen.getByText('InfoNoise controls timestep sampling')).toBeInTheDocument()
    // The description lives in the hover tip on the field name.
    fireEvent.mouseEnter(container.querySelector('[data-field="timestep_sampling"] .ds-label') as Element)
    expect(screen.getByRole('tooltip')).toHaveTextContent('InfoNoise takes over timestep sampling')
    expect(screen.queryByText('Normal timestep description')).not.toBeInTheDocument()
  })

})

// --- option_show_when: enum options filtered by model_family (multi-model P4-2) ---

const gatedSchema: SchemaResponse = {
  groups: [{ key: 'sample', label: 'Sampling' }],
  schema: {
    properties: {
      model_family: {
        type: 'string',
        enum: ['anima', 'krea2'],
        default: 'anima',
        group: 'sample',
        description: 'Family',
      },
      sample_sampler_name: {
        type: 'string',
        enum: ['er_sde', 'dpmpp_3m_sde', 'euler'],
        default: 'er_sde',
        group: 'sample',
        description: 'Sampler',
        option_show_when: {
          er_sde: 'model_family==anima',
          dpmpp_3m_sde: 'model_family==anima',
          euler: 'model_family==krea2',
        },
      },
    },
  },
}

describe('SchemaForm option_show_when filtering', () => {
  const samplerSelect = () =>
    screen
      .getAllByRole('combobox')
      .find((el) =>
        Array.from((el as HTMLSelectElement).options).some((o) =>
          ['er_sde', 'euler'].includes(o.value),
        ),
      ) as HTMLSelectElement

  const optionValues = (el: HTMLSelectElement) =>
    Array.from(el.options).map((o) => o.value)

  it('hides options gated to another family', () => {
    render(
      <SchemaForm
        schema={gatedSchema}
        values={{ model_family: 'anima', sample_sampler_name: 'er_sde' }}
        onChange={() => {}}
      />,
    )
    expect(optionValues(samplerSelect())).toEqual(['er_sde', 'dpmpp_3m_sde'])
  })

  it('shows only the owning family options after switching family', () => {
    render(
      <SchemaForm
        schema={gatedSchema}
        values={{ model_family: 'krea2', sample_sampler_name: 'euler' }}
        onChange={() => {}}
      />,
    )
    expect(optionValues(samplerSelect())).toEqual(['euler'])
  })

  it('keeps the currently selected value visible even when gated out', () => {
    // config holds a value from another family (e.g. hand-edited yaml / family switch
    // that wasn't migrated): the form shows it as-is instead of hiding it - rejecting
    // the value is the backend validator's job
    render(
      <SchemaForm
        schema={gatedSchema}
        values={{ model_family: 'krea2', sample_sampler_name: 'er_sde' }}
        onChange={() => {}}
      />,
    )
    expect(optionValues(samplerSelect())).toEqual(['er_sde', 'euler'])
    expect(samplerSelect().value).toBe('er_sde')
  })
})

// --- R6 (D6): the setField entrypoint intercepts lossy linked changes with a confirm first ---
// The mocked t() returns the raw key for unknown keys; RuleImpactDialog's button/title assert on the key.

describe('SchemaForm rule takeover confirm (R6)', () => {
  const base: ConfigData = {
    optimizer_type: 'adamw',
    learning_rate: 0.0001,
    lr_scheduler: 'none',
    infonoise_enabled: false,
    timestep_sampling: 'uniform',
  }

  it('intercepts lossy takeover with a confirm dialog; apply commits all writes', () => {
    const onChange = vi.fn()
    render(<SchemaForm schema={schema} values={base} onChange={onChange} advancedMode />)
    // enabling InfoNoise -> timestep_sampling(uniform) gets pinned back to logit_normal = lossy
    fireEvent.click(screen.getByRole('checkbox'))
    expect(onChange).not.toHaveBeenCalled()  // the violating state never enters form state
    expect(screen.getByText('ruleImpact.title')).toBeInTheDocument()
    fireEvent.click(screen.getByText('ruleImpact.ok'))
    expect(onChange).toHaveBeenCalledWith({
      ...base,
      infonoise_enabled: true,
      timestep_sampling: 'logit_normal',
    })
  })

  it('cancel leaves the form untouched', () => {
    const onChange = vi.fn()
    render(<SchemaForm schema={schema} values={base} onChange={onChange} advancedMode />)
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(screen.getByText('common.cancel'))
    expect(onChange).not.toHaveBeenCalled()
    expect(screen.queryByText('ruleImpact.title')).not.toBeInTheDocument()
  })

  it('applies silently when linked fields are already at pinned values', () => {
    const onChange = vi.fn()
    render(
      <SchemaForm
        schema={schema}
        values={{ ...base, timestep_sampling: 'logit_normal' }}
        onChange={onChange}
        advancedMode
      />,
    )
    fireEvent.click(screen.getByRole('checkbox'))
    // lossless (the target is already at its pinned value) -> commits directly, no dialog
    expect(screen.queryByText('ruleImpact.title')).not.toBeInTheDocument()
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ infonoise_enabled: true }),
    )
  })

  it('advisory automagic learning-rate rewrite goes through the same dialog', () => {
    const onChange = vi.fn()
    const values: ConfigData = { ...base }
    render(<SchemaForm schema={schema} values={values} onChange={onChange} advancedMode />)
    const optimizerSelect = screen
      .getAllByRole('combobox')
      .find((el) =>
        Array.from((el as HTMLSelectElement).options).some((o) => o.value === 'adamw'),
      ) as HTMLSelectElement
    fireEvent.change(optimizerSelect, { target: { value: 'automagic' } })
    // learning_rate 0.0001 > 1e-5 -> advisory rewrites it to 1e-6, lossy -> dialog
    expect(screen.getByText('ruleImpact.title')).toBeInTheDocument()
    fireEvent.click(screen.getByText('ruleImpact.ok'))
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ optimizer_type: 'automagic', learning_rate: 1e-6 }),
    )
  })

  it('switching to simplified_ademamix suggests an lr scaled by (1 - beta1)', () => {
    const onChange = vi.fn()
    const values: ConfigData = { ...base }
    render(<SchemaForm schema={schema} values={values} onChange={onChange} advancedMode />)
    const optimizerSelect = screen
      .getAllByRole('combobox')
      .find((el) =>
        Array.from((el as HTMLSelectElement).options).some((o) => o.value === 'adamw'),
      ) as HTMLSelectElement
    fireEvent.change(optimizerSelect, { target: { value: 'simplified_ademamix' } })
    expect(screen.getByText('ruleImpact.simplifiedAdemamixLr')).toBeInTheDocument()
    fireEvent.click(screen.getByText('ruleImpact.ok'))
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ optimizer_type: 'simplified_ademamix', learning_rate: 1e-6 }),
    )
  })
})
