import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { type LoraCkpt } from '../../../api/client'
import InlineLoraPicker, { projectAbbr, type PickedLora } from './InlineLoraPicker'
import type { ProjectLora } from './types'
import type { LoraCatalog, LoraVersionOption } from './useLoraCatalog'

/** Builds an already-loaded LoraCatalog from samples (ProjectLora[]) -- a stand-in
 *  for the lazy-cascade catalog under test: projects / versionsByPid are pre-populated
 *  in sync, and fetchCkpts returns fixed ckpts. */
function catalogFrom(samples: ProjectLora[], ckpts: LoraCkpt[]): LoraCatalog {
  const projects = Array.from(
    new Map(samples.map((s) => [s.projectId, { id: s.projectId, title: s.projectTitle }])).values(),
  )
  const versionsByPid: Record<number, LoraVersionOption[]> = {}
  for (const s of samples) {
    (versionsByPid[s.projectId] ??= []).push({ id: s.versionId, label: s.versionLabel, status: s.status })
  }
  return {
    projects,
    projectsLoading: false,
    ensureProjects: () => {},
    loadProjects: () => Promise.resolve(projects),
    versionsOf: (pid) => versionsByPid[pid],
    ensureVersions: () => {},
    fetchCkpts: () => Promise.resolve(ckpts),
  }
}

const sample: ProjectLora[] = [
  {
    projectId: 1, projectTitle: 'cute_chibi',
    versionId: 11, versionLabel: 'v3', status: 'training',
    path: '/loras/cute_chibi/v3.safetensors', createdAt: 300,
  },
  {
    projectId: 1, projectTitle: 'cute_chibi',
    versionId: 12, versionLabel: 'v2', status: 'completed',
    path: '/loras/cute_chibi/v2.safetensors', createdAt: 200,
  },
  {
    projectId: 2, projectTitle: 'noir_portrait',
    versionId: 21, versionLabel: 'v1', status: 'completed',
    path: '/loras/noir/v1.safetensors', createdAt: 100,
  },
]

const ckptsV3: LoraCkpt[] = [
  { kind: 'final', value: 0, label: 'final', path: '/loras/cute_chibi/v3/final.safetensors', mtime: 300 },
  { kind: 'step', value: 2000, label: 'step 2000', path: '/loras/cute_chibi/v3/step_2000.safetensors', mtime: 250 },
  { kind: 'step', value: 1000, label: 'step 1000', path: '/loras/cute_chibi/v3/step_1000.safetensors', mtime: 200 },
]

describe('projectAbbr', () => {
  it('extracts first 2 alphanumerics, uppercase', () => {
    expect(projectAbbr('cute_chibi')).toBe('CU')
    expect(projectAbbr('noir_portrait')).toBe('NO')
  })
  it('falls back to ?? when empty', () => {
    expect(projectAbbr('___')).toBe('??')
    expect(projectAbbr('')).toBe('??')
  })
})

describe('InlineLoraPicker — multi mode (default)', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  function renderMulti(overrides: Partial<{
    projectLoras: ProjectLora[]
    existingPaths: Set<string>
    showWeight: boolean
    ckpts: LoraCkpt[]
    live: boolean
  }> = {}) {
    const catalog = catalogFrom(overrides.projectLoras ?? sample, overrides.ckpts ?? ckptsV3)
    const onPick = vi.fn()
    const onClose = vi.fn()
    const onPickExternal = vi.fn()
    const utils = render(
      <InlineLoraPicker
        mode="multi"
        catalog={catalog}
        existingPaths={overrides.existingPaths ?? new Set()}
        showWeight={overrides.showWeight ?? true}
        live={overrides.live ?? false}
        onPick={onPick}
        onClose={onClose}
        onPickExternal={onPickExternal}
      />
    )
    return { ...utils, onPick, onClose, onPickExternal }
  }

  it('renders project + version dropdowns from projectLoras', async () => {
    renderMulti()
    expect(screen.getByLabelText('Pick a project')).toBeInTheDocument()
    expect(screen.getByText('cute_chibi')).toBeInTheDocument()
    expect(screen.getByText('noir_portrait')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText(/v3 \(training\)/)).toBeInTheDocument())
  })

  it('auto-loads ckpts for the default project/version (as chips)', async () => {
    renderMulti()
    await waitFor(() => {
      expect(screen.getByText('final')).toBeInTheDocument()
      expect(screen.getByText('step 2000')).toBeInTheDocument()
      expect(screen.getByText('step 1000')).toBeInTheDocument()
    })
  })

  it('click toggles picked; "Add N" commits + closes', async () => {
    const user = userEvent.setup()
    const { onPick, onClose } = renderMulti()
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 2000').closest('button')!)
    await user.click(screen.getByText('step 1000').closest('button')!)
    expect(screen.getByText(/2 picked/)).toBeInTheDocument()
    await user.click(screen.getByText(/Add 2/))
    expect(onPick).toHaveBeenCalledTimes(1)
    const [picks, weight] = onPick.mock.calls[0]
    expect(picks).toHaveLength(2)
    expect(picks.map((p: PickedLora) => p.path).sort()).toEqual([
      '/loras/cute_chibi/v3/step_1000.safetensors',
      '/loras/cute_chibi/v3/step_2000.safetensors',
    ])
    expect(weight).toBe(1.0)
    expect(onClose).toHaveBeenCalled()
  })

  it('commits picks in ckpt display order, not click order ("Add N")', async () => {
    const user = userEvent.setup()
    const { onPick } = renderMulti()
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 1000').closest('button')!)
    await user.click(screen.getByText('final').closest('button')!)
    await user.click(screen.getByText('step 2000').closest('button')!)
    await user.click(screen.getByText(/Add 3/))
    const [picks] = onPick.mock.calls[0]
    expect(picks.map((p: PickedLora) => p.path)).toEqual([
      '/loras/cute_chibi/v3/final.safetensors',
      '/loras/cute_chibi/v3/step_2000.safetensors',
      '/loras/cute_chibi/v3/step_1000.safetensors',
    ])
  })

  it('live mode commits in display order on each toggle (XY ckpt axis is monotonic)', async () => {
    const user = userEvent.setup()
    const { onPick } = renderMulti({ live: true, showWeight: false })
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 1000').closest('button')!)
    await user.click(screen.getByText('step 2000').closest('button')!)
    await user.click(screen.getByText('final').closest('button')!)
    const lastPicks = onPick.mock.calls[onPick.mock.calls.length - 1][0]
    expect(lastPicks.map((p: PickedLora) => p.path)).toEqual([
      '/loras/cute_chibi/v3/final.safetensors',
      '/loras/cute_chibi/v3/step_2000.safetensors',
      '/loras/cute_chibi/v3/step_1000.safetensors',
    ])
  })

  it('existingPaths disables the chip; click does not toggle', async () => {
    const user = userEvent.setup()
    const { onPick } = renderMulti({
      existingPaths: new Set(['/loras/cute_chibi/v3/step_2000.safetensors']),
    })
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    const btn = screen.getByText('step 2000').closest('button')!
    expect(btn).toBeDisabled()
    await user.click(btn)
    expect(onPick).not.toHaveBeenCalled()
  })

  it('shows empty state when projectLoras is empty', () => {
    renderMulti({ projectLoras: [] })
    expect(screen.getByText(/No trained LoRAs yet/)).toBeInTheDocument()
  })

  it('shows no-ckpt hint when version has no ckpts', async () => {
    renderMulti({ ckpts: [] })
    await waitFor(() =>
      expect(screen.getByText(/No checkpoint files found in this version/)).toBeInTheDocument()
    )
  })

  it('search filters chip list', async () => {
    const user = userEvent.setup()
    renderMulti()
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.type(screen.getByPlaceholderText('Search checkpoint file name\u2026'), '2000')
    expect(screen.queryByText('final')).not.toBeInTheDocument()
    expect(screen.queryByText('step 1000')).not.toBeInTheDocument()
    expect(screen.getByText('step 2000')).toBeInTheDocument()
  })

  it('× triggers onClose', async () => {
    const user = userEvent.setup()
    const { onClose } = renderMulti()
    await user.click(screen.getByLabelText('Close the picker'))
    expect(onClose).toHaveBeenCalled()
  })

  it('"External file" triggers onPickExternal', async () => {
    const user = userEvent.setup()
    const { onPickExternal } = renderMulti()
    await user.click(screen.getByText('External file'))
    expect(onPickExternal).toHaveBeenCalled()
  })

  it('changing project resets picked', async () => {
    const user = userEvent.setup()
    renderMulti()
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 2000').closest('button')!)
    expect(screen.getByText(/1 picked/)).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Pick a project'), '2')
    expect(screen.queryByText(/1 picked/)).not.toBeInTheDocument()
  })

  it('switching project does not fetch ckpts with (new pid, old vid) (regression: avoid 404)', async () => {
    const user = userEvent.setup()
    const calls: string[] = []
    const base = catalogFrom(sample, ckptsV3)
    const catalog: LoraCatalog = {
      ...base,
      fetchCkpts: (pid, vid) => { calls.push(`${pid}:${vid}`); return base.fetchCkpts(pid, vid) },
    }
    render(
      <InlineLoraPicker
        mode="multi"
        catalog={catalog}
        existingPaths={new Set()}
        showWeight
        onPick={vi.fn()}
        onClose={vi.fn()}
        onPickExternal={vi.fn()}
      />,
    )
    await waitFor(() => expect(calls).toContain('1:11'))
    await user.selectOptions(screen.getByLabelText('Pick a project'), '2')
    await waitFor(() => expect(calls).toContain('2:21'))
    const valid = new Set(['1:11', '1:12', '2:21'])
    expect(calls.every((c) => valid.has(c))).toBe(true)
  })

  it('keeps old chips while the version is loading (stale-while-revalidate, no flash-empty)', async () => {
    const user = userEvent.setup()
    const ckptsV2: LoraCkpt[] = [
      { kind: 'final', value: 0, label: 'v2-final', path: '/loras/cute_chibi/v2/final.safetensors', mtime: 1 },
    ]
    let resolveSecond: (v: LoraCkpt[]) => void = () => {}
    let call = 0
    const base = catalogFrom(sample, ckptsV3)
    const catalog: LoraCatalog = {
      ...base,
      fetchCkpts: () => {
        call += 1
        if (call === 1) return Promise.resolve(ckptsV3)
        return new Promise<LoraCkpt[]>((res) => { resolveSecond = res })
      },
    }
    render(
      <InlineLoraPicker
        mode="multi"
        catalog={catalog}
        existingPaths={new Set()}
        showWeight
        onPick={vi.fn()}
        onClose={vi.fn()}
        onPickExternal={vi.fn()}
      />,
    )
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.selectOptions(screen.getByLabelText('Pick a version'), '12')
    expect(screen.getByText('step 2000')).toBeInTheDocument()
    resolveSecond(ckptsV2)
    await waitFor(() => expect(screen.getByText('v2-final')).toBeInTheDocument())
    expect(screen.queryByText('step 2000')).not.toBeInTheDocument()
  })

  it('weight slider value used in onPick', async () => {
    const user = userEvent.setup()
    const { onPick } = renderMulti()
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 2000').closest('button')!)
    const weightInput = screen.getByLabelText('LoRA weight value')
    await user.clear(weightInput)
    await user.type(weightInput, '0.75')
    await user.click(screen.getByText(/Add 1/))
    const [, weight] = onPick.mock.calls[0]
    expect(weight).toBe(0.75)
  })

  it('showWeight=false hides weight slider (XY axis use)', async () => {
    const user = userEvent.setup()
    renderMulti({ showWeight: false })
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 2000').closest('button')!)
    expect(screen.queryByLabelText('LoRA weight value')).not.toBeInTheDocument()
  })
})

describe('InlineLoraPicker — single mode (controlled slot)', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  function renderSingle(overrides: Partial<{
    value: PickedLora | null
    weight: number
    ckpts: LoraCkpt[]
  }> = {}) {
    const catalog = catalogFrom(sample, overrides.ckpts ?? ckptsV3)
    const onChange = vi.fn()
    const onClose = vi.fn()
    const onPickExternal = vi.fn()
    const utils = render(
      <InlineLoraPicker
        mode="single"
        catalog={catalog}
        value={overrides.value ?? null}
        weight={overrides.weight ?? 1.0}
        onChange={onChange}
        onClose={onClose}
        onPickExternal={onPickExternal}
      />
    )
    return { ...utils, onChange, onClose, onPickExternal }
  }

  it('click ckpt chip → onChange(pick, weight) — does not auto-close', async () => {
    const user = userEvent.setup()
    const { onChange, onClose } = renderSingle()
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 2000').closest('button')!)
    expect(onChange).toHaveBeenCalledTimes(1)
    const [pick, weight] = onChange.mock.calls[0]
    expect(pick).toEqual({
      path: '/loras/cute_chibi/v3/step_2000.safetensors',
      projectId: 1,
      versionId: 11,
    })
    expect(weight).toBe(1.0)
    expect(onClose).not.toHaveBeenCalled()
  })

  it('click currently-selected chip -> onChange(null, weight) deselects (SidebarLoras clears the path but keeps the slot)', async () => {
    const user = userEvent.setup()
    const { onChange, onClose } = renderSingle({
      value: {
        path: '/loras/cute_chibi/v3/step_2000.safetensors',
        projectId: 1, versionId: 11,
      },
    })
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
    await user.click(screen.getByText('step 2000').closest('button')!)
    expect(onChange).toHaveBeenCalledWith(null, 1.0)
    expect(onClose).not.toHaveBeenCalled()
  })

  it('weight slider change → onChange(value, new_weight)', async () => {
    const value: PickedLora = {
      path: '/loras/cute_chibi/v3/step_2000.safetensors',
      projectId: 1, versionId: 11,
    }
    const { onChange } = renderSingle({ value, weight: 0.8 })
    await waitFor(() => expect(screen.getByLabelText('LoRA weight value')).toBeInTheDocument())
    const weightInput = screen.getByLabelText('LoRA weight value') as HTMLInputElement
    expect(weightInput.value).toBe('0.8')
    fireEvent.change(weightInput, { target: { value: '1.2' } })
    expect(onChange).toHaveBeenCalledWith(value, 1.2)
  })

  it('× → onClose (deletes slot from parent)', async () => {
    const user = userEvent.setup()
    const { onClose } = renderSingle()
    await user.click(screen.getByLabelText('Remove LoRA'))
    expect(onClose).toHaveBeenCalled()
  })

  it('weight slider always visible in single mode (even without selection)', async () => {
    renderSingle({ value: null })
    expect(await screen.findByLabelText('LoRA weight value')).toBeInTheDocument()
  })
})

describe('InlineLoraPicker - controlled sync (Step 6 / decision #8)', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('rerender with new props.value → project/version dropdowns reflect new ids', async () => {
    const catalog = catalogFrom(sample, ckptsV3)
    const onChange = vi.fn()
    const onClose = vi.fn()
    const initialValue: PickedLora = {
      path: '/loras/cute_chibi/v3/final.safetensors',
      projectId: 1, versionId: 11,
    }
    const { rerender } = render(
      <InlineLoraPicker
        mode="single"
        catalog={catalog}
        value={initialValue}
        weight={1.0}
        onChange={onChange}
        onClose={onClose}
      />
    )
    await waitFor(() => {
      const projectSelect = screen.getAllByRole('combobox')[0] as HTMLSelectElement
      expect(projectSelect.value).toBe('1')
    })

    const newValue: PickedLora = {
      path: '/loras/noir/v1/final.safetensors',
      projectId: 2, versionId: 21,
    }
    rerender(
      <InlineLoraPicker
        mode="single"
        catalog={catalog}
        value={newValue}
        weight={1.0}
        onChange={onChange}
        onClose={onClose}
      />
    )
    await waitFor(() => {
      const projectSelect = screen.getAllByRole('combobox')[0] as HTMLSelectElement
      expect(projectSelect.value).toBe('2')
    })
  })

  it('value=null does not sync, keeps the fallback default (projects[0] ckpts shown)', async () => {
    const catalog = catalogFrom(sample, ckptsV3)
    const onChange = vi.fn()
    const onClose = vi.fn()
    render(
      <InlineLoraPicker
        mode="single"
        catalog={catalog}
        value={null}
        weight={1.0}
        onChange={onChange}
        onClose={onClose}
      />
    )
    await waitFor(() => expect(screen.getByText('step 2000')).toBeInTheDocument())
  })
})
