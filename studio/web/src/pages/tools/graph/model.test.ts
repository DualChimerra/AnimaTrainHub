import { describe, expect, it } from 'vitest'
import {
  buildGaps, configPatch, defaultView, differences, filterRuns, formatNumber, groupItems,
  numberRange, oneStepAway, parseNumber, sameValue, scientific, sortRuns, stepFromFilename,
  taskMatches, valuesFromConfig, varyingParams,
} from './model'
import type { Param, Run, Values } from './types'

const P: Param[] = [
  { id: 'lora', name: 'LoRA type', type: 'list', options: [
    { id: 'tlora', label: 'TLoRA', config: { apply: { lora_type: 'tlora' } } },
    { id: 'lokr', label: 'LoKr', config: { apply: { lora_type: 'lokr' } } },
  ] },
  { id: 'lr', name: 'LR', type: 'number', config: { key: 'learning_rate' }, options: [
    { id: 'a', value: 0.0001 }, { id: 'b', value: 0.00002 },
  ] },
  { id: 'res', name: 'Resolution', type: 'number', config: { key: 'resolution', list: true }, options: [] },
  { id: 'sched', name: 'Scheduler', type: 'list', options: [
    { id: 'c', label: 'Constant', config: { apply: { lr_scheduler: 'none' } } },
    { id: 'cos', label: 'Cosine', config: { apply: { lr_scheduler: 'cosine' } } },
  ] },
  { id: 'wd', name: 'Weight decay', type: 'bool', options: [
    { id: 'yes', label: 'Yes', config: { apply: { weight_decay: 0.01 }, match: { weight_decay: { $gt: 0 } } } },
    { id: 'no', label: 'No', config: { apply: { weight_decay: 0 } } },
  ] },
  { id: 'samp', name: 'Sampling', type: 'list', options: [
    { id: 'style', label: 'Style-friendly', config: { apply: { timestep_sampling: 'style_friendly' } } },
    { id: 'logit', label: 'Logit-normal', config: { apply: { timestep_sampling: 'logit_normal' } } },
  ] },
  { id: 'snr', name: 'SNR mean', type: 'number', config: { key: 'style_snr_mean' }, options: [],
    condition: { param: 'samp', options: ['style'] } },
  { id: 'note', name: 'Dataset', type: 'text', options: [] },
]

let seq = 0
function run(values: Values, extra: Partial<Run> = {}): Run {
  seq++
  return {
    id: seq, board_id: 1, name: `r${seq}`, values, status: 'done', notes: '', link: '',
    favorite: false, rating: null, best_step: null, tags: [], task_id: null, project_id: null,
    version_id: null, created_at: seq, updated_at: seq, images: [], ...extra,
  }
}

describe('numbers', () => {
  it('treats 1e-4 and 0.0001 as the same and keeps small LR precise', () => {
    expect(sameValue(1e-4, 0.0001)).toBe(true)
    expect(parseNumber('1e-4')).toBe(0.0001)
    expect(parseNumber('0,5')).toBe(0.5)
    expect(parseNumber('abc')).toBeNull()
    expect(formatNumber(0.00002)).toBe('0.00002')
    expect(formatNumber(1e-7)).toBe('0.0000001')
    expect(scientific(0.00003)).toBe('3e-5')
    expect(scientific(1024)).toBeNull()
  })
  it('builds a drift-free range', () => {
    expect(numberRange(0.00001, 0.00005, 0.00001)).toEqual([0.00001, 0.00002, 0.00003, 0.00004, 0.00005])
    expect(numberRange(1500, 3000, 500)).toEqual([1500, 2000, 2500, 3000])
    expect(numberRange(1, 2, 0)).toEqual([])
  })
})

describe('differences', () => {
  it('folds dependent SNR into the Sampling change', () => {
    const a = { samp: 'style', snr: -6, lr: 0.0001 }
    const b = { samp: 'logit', lr: 0.0001 }
    expect(differences(a, b, P)).toEqual(['samp'])
  })
  it('counts unset vs set as a difference, not as No', () => {
    expect(differences({ wd: 'no' }, {}, P)).toEqual(['wd'])
  })
  it('finds runs one parameter away', () => {
    const base = run({ lora: 'lokr', lr: 0.0001 })
    const lr = run({ lora: 'lokr', lr: 0.00002 })
    const two = run({ lora: 'tlora', lr: 0.00002 })
    expect(oneStepAway(base, [base, lr, two], P).map((x) => [x.run.id, x.param])).toEqual([[lr.id, 'lr']])
  })
  it('lists params varying anywhere in a set', () => {
    expect(varyingParams([{ lr: 1 }, { lr: 1, wd: 'yes' }, { lr: 2, wd: 'yes' }], P)).toEqual(['lr', 'wd'])
  })
})

describe('filter / sort / group', () => {
  const runs = [
    run({ lora: 'lokr', lr: 0.00002, sched: 'cos' }, { name: 'alpha', notes: 'crispy lines' }),
    run({ lora: 'tlora', lr: 0.0001 }, { rating: 3 }),
    run({ lora: 'lokr', lr: 0.0001, samp: 'logit', snr: -6 }),
  ]
  it('filters by values, unset and search', () => {
    const v = defaultView(P)
    expect(filterRuns(runs, P, { ...v, filters: { lr: { values: [1e-4] } } }).length).toBe(2)
    expect(filterRuns(runs, P, { ...v, filters: { sched: { unset: true } } }).length).toBe(2)
    expect(filterRuns(runs, P, { ...v, search: 'CRISPY' }).map((r) => r.name)).toEqual(['alpha'])
    // SNR is inactive under Logit-normal, so a SNR filter never matches it
    expect(filterRuns(runs, P, { ...v, filters: { snr: { values: [-6] } } }).length).toBe(0)
    expect(filterRuns(runs, P, { ...v, minRating: 2 }).length).toBe(1)
  })
  it('sorts LR numerically with unset last', () => {
    const r = sortRuns([...runs, run({})], [{ key: 'lr', dir: 'asc' }], P)
    expect(r.map((x) => x.values.lr)).toEqual([0.00002, 0.0001, 0.0001, undefined])
  })
  it('nests groups in the chosen order', () => {
    const g = groupItems(runs, ['lora', 'lr'], P, (r) => r.values)!
    expect(g.map((x) => x.value)).toEqual(['tlora', 'lokr'])
    expect(g[1].children!.map((x) => x.value)).toEqual([0.00002, 0.0001])
  })
})

describe('gaps', () => {
  it('spans the filtered values and places runs', () => {
    const runs = [run({ lr: 0.0001, sched: 'cos' }), run({ lr: 0.0001, sched: 'cos' })]
    const v = { ...defaultView(P), filters: { lr: { values: [0.0001, 0.00002] }, sched: { values: ['c', 'cos'] } } }
    const g = buildGaps(runs, P, v)!
    expect(g.length).toBe(4)
    expect(g.filter((c) => c.runs.length).length).toBe(1)
    expect(g.find((c) => c.runs.length)!.runs.length).toBe(2)
  })
  it('does not create SNR variants for Logit-normal', () => {
    const v = { ...defaultView(P), filters: { samp: { values: ['style', 'logit'] }, snr: { values: [-6, -4] } } }
    const g = buildGaps([], P, v)!
    expect(g.length).toBe(3)
  })
})

describe('config bridge', () => {
  it('reads a task config back, most specific option wins', () => {
    const v = valuesFromConfig({
      lora_type: 'lokr', learning_rate: 1e-4, resolution: [1024], lr_scheduler: 'cosine',
      weight_decay: 0.05, timestep_sampling: 'logit_normal', style_snr_mean: -6,
    }, P)
    expect(v).toEqual({ lora: 'lokr', lr: 0.0001, res: 1024, sched: 'cos', wd: 'yes', samp: 'logit' })
  })
  it('ignores multi-resolution lists', () => {
    expect(valuesFromConfig({ resolution: [512, 1024] }, P).res).toBeUndefined()
  })
  it('builds a patch and reports what it cannot write', () => {
    const { patch, skipped } = configPatch({ lora: 'tlora', res: 1536, samp: 'logit', snr: -6, note: 'x' }, P)
    expect(patch).toEqual({ lora_type: 'tlora', resolution: [1536], timestep_sampling: 'logit_normal' })
    expect(skipped).toEqual(['note'])
  })
  it('matches tasks on the linked values a place fixes', () => {
    expect(taskMatches({ lr: 0.0001, note: 'x' }, { lr: 1e-4, lora: 'lokr' }, P)).toBe(true)
    expect(taskMatches({ lr: 0.00002 }, { lr: 1e-4 }, P)).toBe(false)
    expect(taskMatches({ note: 'x' }, { lr: 1e-4 }, P)).toBe(false)
  })
})

it('reads steps out of file names', () => {
  expect(stepFromFilename('step_400.png')).toBe(400)
  expect(stepFromFilename('my_lora-000800.png')).toBe(800)
  expect(stepFromFilename('sample 1200steps.webp')).toBe(1200)
  expect(stepFromFilename('cat.png')).toBeNull()
})
