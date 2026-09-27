import { describe, expect, it } from 'vitest'
import type { LoraEntry } from '../../../api/client'
import { buildXYMatrix, cellCount, draftToSpec, parseAxisValues, type XYAxisDraft } from './xy'

describe('parseAxisValues', () => {
  it('parses int axis (steps) splits by comma', () => {
    expect(parseAxisValues('steps', '20, 25, 30')).toEqual([20, 25, 30])
  })

  it('parses float axis (cfg_scale)', () => {
    expect(parseAxisValues('cfg_scale', '3.0, 4.5, 5')).toEqual([3.0, 4.5, 5])
  })

  it('parses string axis (lora_ckpt path)', () => {
    expect(parseAxisValues('lora_ckpt', '/a/step100.safetensors, /a/step200.safetensors'))
      .toEqual(['/a/step100.safetensors', '/a/step200.safetensors'])
  })

  it('rejects empty values', () => {
    expect(() => parseAxisValues('steps', '')).toThrow()
    expect(() => parseAxisValues('steps', ' , , ')).toThrow()
  })

  it('rejects non-numeric on int axis', () => {
    expect(() => parseAxisValues('steps', '20, foo, 30')).toThrow(/is not a valid number/)
  })

  it('rejects float on int axis', () => {
    expect(() => parseAxisValues('steps', '20, 25.5')).toThrow(/must be an integer/)
  })

  it('accepts whitespace and trims', () => {
    expect(parseAxisValues('steps', '  20  ,30 , 40 ')).toEqual([20, 30, 40])
  })
})

describe('draftToSpec', () => {
  const loras: LoraEntry[] = [
    { path: '/a.safetensors', scale: 1.0 },
    { path: '/b.safetensors', scale: 0.8 },
  ]

  it('builds spec for non-lora axis without lora_index', () => {
    const d: XYAxisDraft = { axis: 'steps', raw: '20, 30', loraIndex: null }
    const s = draftToSpec(d, loras)
    expect(s.axis).toBe('steps')
    expect(s.values).toEqual([20, 30])
    expect(s.lora_index).toBeUndefined()
  })

  it('lora_scale is now a global axis: no longer requires lora_index (passes through spec.lora_index=undefined)', () => {
    const d: XYAxisDraft = { axis: 'lora_scale', raw: '0.5, 1.0', loraIndex: null }
    const s = draftToSpec(d, loras)
    expect(s.axis).toBe('lora_scale')
    expect(s.values).toEqual([0.5, 1.0])
    expect(s.lora_index).toBeUndefined()
  })

  it('the lora_ckpt axis still requires loraIndex (points at the anchor slot the caller pushed itself)', () => {
    const d: XYAxisDraft = { axis: 'lora_ckpt', raw: '/x/step.safetensors', loraIndex: null }
    expect(() => draftToSpec(d, loras)).toThrow(/must bind a LoRA/)
  })

  it('lora_ckpt axis with an out-of-range lora_index throws', () => {
    const d: XYAxisDraft = { axis: 'lora_ckpt', raw: '/x/step.safetensors', loraIndex: 5 }
    expect(() => draftToSpec(d, loras)).toThrow(/does not exist/)
  })

  it('lora_ckpt axis with a valid lora_index fills in spec.lora_index', () => {
    const d: XYAxisDraft = { axis: 'lora_ckpt', raw: '/x/step.safetensors', loraIndex: 1 }
    const s = draftToSpec(d, loras)
    expect(s.axis).toBe('lora_ckpt')
    expect(s.lora_index).toBe(1)
    expect(s.values).toEqual(['/x/step.safetensors'])
  })
})

describe('buildXYMatrix (only sends anchors referenced by an axis, drops orphans left over from the picker)', () => {
  const CHEN = { path: 'G:/chen-bin_v3.4.safetensors', scale: 1, project_id: 2, version_id: 4 }
  const ORPHAN = { path: 'G:/chen-bin_v3.2.safetensors', scale: 1, project_id: 1, version_id: 1 }
  const HOSHI = { path: 'G:/hoshi.safetensors', scale: 1, project_id: 3, version_id: 9 }

  it('a non-lora_ckpt axis (steps) -> lora_configs is empty (orphans are not sent as base LoRAs)', () => {
    // Reproduces the root cause: xyLoras contains ORPHAN (left over from the
    // picker), but the current X axis is steps and doesn't reference it.
    const x: XYAxisDraft = { axis: 'steps', raw: '20, 25, 30', loraIndex: null }
    const { xy_matrix, loraConfigs } = buildXYMatrix(x, null, [ORPHAN, HOSHI])
    expect(loraConfigs).toEqual([]) // before the fix this would send the whole [ORPHAN, HOSHI] bucket
    expect(xy_matrix.x.values).toEqual([20, 25, 30])
    expect(xy_matrix.y).toBeNull()
  })

  it('a lora_ckpt axis only sends the anchor it references, loraIndex remaps to 0', () => {
    // xyLoras=[ORPHAN, CHEN], X axis loraIndex=1 points at CHEN; ORPHAN is
    // unreferenced -> dropped.
    const x: XYAxisDraft = { axis: 'lora_ckpt', raw: CHEN.path, loraIndex: 1 }
    const { xy_matrix, loraConfigs } = buildXYMatrix(x, null, [ORPHAN, CHEN])
    expect(loraConfigs).toEqual([CHEN]) // the never-picked ORPHAN(v3.2) does not leak in
    expect(xy_matrix.x.lora_index).toBe(0) // 1 -> 0 remap
    expect(xy_matrix.x.values).toEqual([CHEN.path])
  })

  it('X/Y are both lora_ckpt referencing different anchors -> both are kept, each remapped', () => {
    const x: XYAxisDraft = { axis: 'lora_ckpt', raw: CHEN.path, loraIndex: 2 }
    const y: XYAxisDraft = { axis: 'lora_ckpt', raw: HOSHI.path, loraIndex: 0 }
    // loras=[HOSHI, ORPHAN, CHEN]: X -> idx2 (CHEN), Y -> idx0 (HOSHI), ORPHAN (idx1) dropped
    const { xy_matrix, loraConfigs } = buildXYMatrix(x, y, [HOSHI, ORPHAN, CHEN])
    expect(loraConfigs).toEqual([CHEN, HOSHI]) // in order of appearance: X first -> CHEN=0, Y -> HOSHI=1
    expect(xy_matrix.x.lora_index).toBe(0)
    expect(xy_matrix.y?.lora_index).toBe(1)
  })

  it('X/Y reference the same anchor -> dedupe to one entry, both axes point at the same index', () => {
    const x: XYAxisDraft = { axis: 'lora_ckpt', raw: CHEN.path, loraIndex: 0 }
    const y: XYAxisDraft = { axis: 'lora_scale', raw: '0.6, 0.8', loraIndex: 0 }
    const { loraConfigs, xy_matrix } = buildXYMatrix(x, y, [CHEN])
    expect(loraConfigs).toEqual([CHEN])
    expect(xy_matrix.x.lora_index).toBe(0)
    // lora_scale does not require lora_index, passes through undefined
    expect(xy_matrix.y?.lora_index).toBeUndefined()
  })

  it('an out-of-range loraIndex on a lora_ckpt axis throws (not silently swallowed)', () => {
    const x: XYAxisDraft = { axis: 'lora_ckpt', raw: CHEN.path, loraIndex: 5 }
    expect(() => buildXYMatrix(x, null, [CHEN])).toThrow(/does not exist/)
  })
})

describe('cellCount', () => {
  it('returns x for y=null (single-axis degeneration)', () => {
    expect(cellCount(3, null)).toBe(3)
  })

  it('returns x*y for 2D matrix', () => {
    expect(cellCount(3, 4)).toBe(12)
    expect(cellCount(5, 5)).toBe(25)
  })

  it('handles 0 length gracefully (callers guard)', () => {
    expect(cellCount(0, 3)).toBe(0)
  })
})
