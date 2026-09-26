/** paramsSnapshot unit tests: the applySnapshot reducer + resolveSnapshotLora's
 * three-tier fallback.
 *
 * Decision #8 (single snapshot-apply entry point) / plan §3 LoRA placeholder fallback.
 */
import { describe, expect, it } from 'vitest'
import {
  applySnapshot, buildCellSnapshot, loraBasename, resolveLoraFromCkpts, transformAxisRawForSnapshot,
  type GenerateParamsSnapshot, type SnapshotLora, type SnapshotLoraResolver,
} from './paramsSnapshot'
import type { XYAxisDraft } from './xy'

// Lazy cascade: applySnapshot no longer takes the full projectLoras; instead it
// gets an injected resolver (fetches a version's ckpts on demand, then calls
// resolveLoraFromCkpts) + projectExists. Simulated here with a fixed mapping.
const CKPTS_BY_VERSION: Record<string, { path: string }[]> = {
  '1:11': [{ path: '/loras/cute_chibi/v3.safetensors' }],
  '2:21': [{ path: '/loras/noir/v1.safetensors' }],
}
const resolver: SnapshotLoraResolver = (snap) => {
  if (snap.project_id == null || snap.version_id == null) {
    return Promise.resolve(resolveLoraFromCkpts(snap, []))
  }
  const ckpts = CKPTS_BY_VERSION[`${snap.project_id}:${snap.version_id}`] ?? []
  return Promise.resolve(resolveLoraFromCkpts(snap, ckpts))
}
const projectExists = (pid: number): boolean => pid === 1 || pid === 2

function snapshot(overrides: Partial<GenerateParamsSnapshot> = {}): GenerateParamsSnapshot {
  return {
    schema_version: 1,
    mode: 'single',
    prompts: ['1girl'],
    negative_prompt: 'blurry',
    width: 1024,
    height: 768,
    steps: 20,
    cfg_scale: 7,
    count: 1,
    seed: 42,
    loras: [],
    xy_draft: null,
    dataset_pick: null,
    ...overrides,
  }
}

describe('loraBasename', () => {
  it('strips POSIX path', () => {
    expect(loraBasename('/a/b/c/my.safetensors')).toBe('my.safetensors')
  })
  it('strips Windows path', () => {
    expect(loraBasename('G:\\a\\b\\my.safetensors')).toBe('my.safetensors')
  })
  it('no separator -> returns the whole thing', () => {
    expect(loraBasename('only.safetensors')).toBe('only.safetensors')
  })
})

describe('transformAxisRawForSnapshot', () => {
  it('lora_ckpt: raw paths -> basename list', () => {
    const draft: XYAxisDraft = {
      axis: 'lora_ckpt',
      raw: '/a/b/step_1000.safetensors, /a/b/step_2000.safetensors',
      loraIndex: 0,
    }
    expect(transformAxisRawForSnapshot(draft).raw).toBe('step_1000.safetensors, step_2000.safetensors')
  })
  it('non lora_ckpt axes: raw passes through unchanged', () => {
    const draft: XYAxisDraft = { axis: 'steps', raw: '10, 20, 30', loraIndex: null }
    expect(transformAxisRawForSnapshot(draft).raw).toBe('10, 20, 30')
  })
})

describe('resolveLoraFromCkpts', () => {
  const ckpts = [
    { path: '/loras/cute/v3/final.safetensors' },
    { path: '/loras/cute/v3/step_1000.safetensors' },
  ]

  it('an exact basename match -> returns that ckpt path + keeps the snapshot\'s scale/ids', () => {
    const snap: SnapshotLora = {
      name: 'step_1000.safetensors', scale: 0.8,
      project_id: 1, version_id: 11,
    }
    const r = resolveLoraFromCkpts(snap, ckpts)
    expect(r.path).toBe('/loras/cute/v3/step_1000.safetensors')
    expect(r.scale).toBe(0.8)
    expect(r.project_id).toBe(1)
    expect(r.version_id).toBe(11)
  })

  it('no basename match but the version has ckpts -> falls back to the version\'s ckpts[0] (list is already sorted final -> step descending)', () => {
    const snap: SnapshotLora = {
      name: 'gone.safetensors', scale: 1.0,
      project_id: 1, version_id: 11,
    }
    const r = resolveLoraFromCkpts(snap, ckpts)
    expect(r.path).toBe('/loras/cute/v3/final.safetensors')
    expect(r.project_id).toBe(1)
    expect(r.version_id).toBe(11)
  })

  it('empty ckpts (version has no outputs / was deleted) -> placeholder: empty path + name kept + original ids', () => {
    const snap: SnapshotLora = {
      name: 'gone.safetensors', scale: 0.5,
      project_id: 9, version_id: 9,
    }
    const r = resolveLoraFromCkpts(snap, [])
    expect(r.path).toBe('')
    expect(r.name).toBe('gone.safetensors')  // the placeholder UI reads this field
    expect(r.project_id).toBe(9)
    expect(r.version_id).toBe(9)
    expect(r.scale).toBe(0.5)
  })
})

describe('applySnapshot (async + injected resolver/projectExists)', () => {
  it('single mode: all fields are filled in + loras replaces singleLoras', async () => {
    const snap = snapshot({
      mode: 'single',
      seed: 42,
      loras: [{ name: 'v3.safetensors', scale: 0.7, project_id: 1, version_id: 11 }],
    })
    const r = await applySnapshot(snap, resolver, projectExists)
    expect(r.mode).toBe('single')
    expect(r.seed).toBe(42)
    expect(r.loras).toHaveLength(1)
    expect(r.loras[0].path).toBe('/loras/cute_chibi/v3.safetensors')
    expect(r.unresolvedLoraCount).toBe(0)
    expect(r.xDraft).toBeUndefined()  // single mode does not fill xDraft
    expect(r.yDraft).toBeUndefined()
  })

  it('base_model backfill: an explicit value passes through, missing -> null', async () => {
    expect((await applySnapshot(snapshot({ base_model: 'preview2' }), resolver, projectExists)).baseModel).toBe('preview2')
    expect((await applySnapshot(snapshot({ base_model: '/loras/ft.safetensors' }), resolver, projectExists)).baseModel)
      .toBe('/loras/ft.safetensors')
    // An old snapshot without this field -> null (falls back to the settings page's default base model)
    expect((await applySnapshot(snapshot(), resolver, projectExists)).baseModel).toBeNull()
  })

  it('compare mode maps to xy (the sub-view has no selectedIndices so it cannot enter directly)', async () => {
    const snap = snapshot({ mode: 'compare' })
    expect((await applySnapshot(snap, resolver, projectExists)).mode).toBe('xy')
  })

  it('xy mode: xDraft + yDraft are filled in', async () => {
    const snap = snapshot({
      mode: 'xy',
      xy_draft: {
        x: { axis: 'cfg_scale', raw: '4, 5, 6', loraIndex: null },
        y: { axis: 'steps', raw: '10, 20', loraIndex: null },
      },
    })
    const r = await applySnapshot(snap, resolver, projectExists)
    expect(r.mode).toBe('xy')
    expect(r.xDraft?.axis).toBe('cfg_scale')
    expect(r.xDraft?.raw).toBe('4, 5, 6')
    expect(r.yDraft?.axis).toBe('steps')
  })

  it('dataset_pick fallback: a projectExists hit -> datasetPick is kept', async () => {
    const snap = snapshot({
      dataset_pick: {
        projectId: 1, versionId: 11,
        name: '0001.txt', tags: ['tag-a', 'tag-b'],
      },
    })
    const r = await applySnapshot(snap, resolver, projectExists)
    expect(r.datasetPick?.projectId).toBe(1)
    expect(r.prompts).toEqual(['1girl'])  // prompts is not polluted
  })

  it('dataset_pick fallback: a projectExists miss -> tags are appended to the first prompt + datasetPick=null', async () => {
    const snap = snapshot({
      prompts: ['base prompt'],
      dataset_pick: {
        projectId: 999, versionId: 88,  // projectExists returns false
        name: '0001.txt', tags: ['fall-tag-1', 'fall-tag-2'],
      },
    })
    const r = await applySnapshot(snap, resolver, projectExists)
    expect(r.datasetPick).toBeNull()
    expect(r.prompts[0]).toBe('base prompt, fall-tag-1, fall-tag-2')
  })

  it('dataset_pick fallback: an empty first prompt -> becomes just the tags string', async () => {
    const snap = snapshot({
      prompts: [''],
      dataset_pick: {
        projectId: 999, versionId: 88,
        name: '0001.txt', tags: ['x', 'y'],
      },
    })
    const r = await applySnapshot(snap, resolver, projectExists)
    expect(r.prompts[0]).toBe('x, y')
  })

  it('dataset_pick fallback: tags already at the end of the first prompt -> not appended again (guards against double-clicking the same history entry)', async () => {
    const snap = snapshot({
      prompts: ['base, x, y'],
      dataset_pick: {
        projectId: 999, versionId: 88,
        name: '0001.txt', tags: ['x', 'y'],
      },
    })
    const r = await applySnapshot(snap, resolver, projectExists)
    expect(r.prompts[0]).toBe('base, x, y')
  })

  it('an unresolved LoRA (version has no ckpts) -> unresolvedLoraCount > 0', async () => {
    const snap = snapshot({
      loras: [
        { name: 'gone1.safetensors', scale: 1, project_id: 99, version_id: 99 },
        { name: 'v3.safetensors', scale: 1, project_id: 1, version_id: 11 },
        { name: 'gone2.safetensors', scale: 1, project_id: 88, version_id: 88 },
      ],
    })
    const r = await applySnapshot(snap, resolver, projectExists)
    expect(r.unresolvedLoraCount).toBe(2)
    expect(r.loras[0].path).toBe('')
    expect(r.loras[1].path).toBe('/loras/cute_chibi/v3.safetensors')
    expect(r.loras[2].path).toBe('')
  })
})

describe('buildCellSnapshot', () => {
  const baseXy = snapshot({
    mode: 'xy',
    steps: 20,
    cfg_scale: 5,
    loras: [{ name: 'a.safetensors', scale: 0.5 }, { name: 'b.safetensors', scale: 0.7 }],
  })

  it('steps axis: overrides top-level steps; mode -> single; xy_draft -> null; xy_origin records the position', () => {
    const cell = buildCellSnapshot(baseXy, { xi: 2, yi: 0 }, {
      x: { axis: 'steps', loraIndex: null, value: 30 },
      y: null,
    })
    expect(cell.mode).toBe('single')
    expect(cell.steps).toBe(30)
    expect(cell.cfg_scale).toBe(5)  // untouched by the axis
    expect(cell.xy_draft).toBeNull()
    expect(cell.xy_origin).toEqual({
      xi: 2, yi: 0, xv: 30, yv: null, x_axis: 'steps', y_axis: null,
    })
  })

  it('cfg_scale axis: overrides top-level cfg_scale (a string input is accepted too)', () => {
    const cell = buildCellSnapshot(baseXy, { xi: 1, yi: 0 }, {
      x: { axis: 'cfg_scale', loraIndex: null, value: '7.5' },
      y: null,
    })
    expect(cell.cfg_scale).toBe(7.5)
    expect(cell.steps).toBe(20)
  })

  it('lora_scale axis: all LoRAs share the cell value (not changed individually by loraIndex)', () => {
    const cell = buildCellSnapshot(baseXy, { xi: 0, yi: 0 }, {
      x: { axis: 'lora_scale', loraIndex: null, value: 0.9 },
      y: null,
    })
    expect(cell.loras).toEqual([
      { name: 'a.safetensors', scale: 0.9 },
      { name: 'b.safetensors', scale: 0.9 },
    ])
  })

  it('lora_ckpt axis: only the name at the given loraIndex changes to the basename, its original ids are invalidated', () => {
    const cell = buildCellSnapshot(baseXy, { xi: 1, yi: 0 }, {
      x: { axis: 'lora_ckpt', loraIndex: 0, value: '/some/path/new.safetensors' },
      y: null,
    })
    expect(cell.loras[0]).toEqual({
      name: 'new.safetensors', scale: 0.5, project_id: null, version_id: null,
    })
    // loras[1] is untouched
    expect(cell.loras[1].name).toBe('b.safetensors')
  })

  it('2D: both x and y axes are materialized', () => {
    const cell = buildCellSnapshot(baseXy, { xi: 1, yi: 2 }, {
      x: { axis: 'steps', loraIndex: null, value: 25 },
      y: { axis: 'cfg_scale', loraIndex: null, value: 8.0 },
    })
    expect(cell.steps).toBe(25)
    expect(cell.cfg_scale).toBe(8.0)
    expect(cell.xy_origin?.yi).toBe(2)
    expect(cell.xy_origin?.y_axis).toBe('cfg_scale')
  })

  it('does not pollute the original XY snapshot (loras is deep-copied)', () => {
    const before = JSON.parse(JSON.stringify(baseXy.loras))
    buildCellSnapshot(baseXy, { xi: 0, yi: 0 }, {
      x: { axis: 'lora_scale', loraIndex: null, value: 0.1 },
      y: null,
    })
    expect(baseXy.loras).toEqual(before)
  })
})
