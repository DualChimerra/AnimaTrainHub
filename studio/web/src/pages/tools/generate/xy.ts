/** XY mode local state + values parsing/validation utilities.
 *
 * The backend schema requires the values type to be derived per axis (int / float / string);
 * the frontend parses the user's comma-separated input string into the correct type, and
 * produces an error message when parsing fails. */

import type { LoraEntry, XYAxisSpec, XYAxisType, XYMatrixSpec } from '../../../api/client'
import i18n from '../../../i18n'

/** UI-side axis state: raw is the user's comma-separated input string (not parsed live, for easier editing). */
export interface XYAxisDraft {
  axis: XYAxisType
  raw: string
  loraIndex: number | null
}

/** Field-type mapping that goes with XYAxisSpec (shares its source with schema._check_axis_values). */
export const AXIS_VALUE_TYPE: Record<XYAxisType, 'int' | 'float' | 'string'> = {
  steps: 'int',
  cfg_scale: 'float',
  lora_scale: 'float',
  lora_ckpt: 'string',  // ckpt path
}

export const AXIS_LABEL_KEYS: Record<XYAxisType, string> = {
  steps: 'generate.axisSteps',
  cfg_scale: 'generate.axisCfgScale',
  lora_scale: 'generate.axisLoraScale',
  lora_ckpt: 'generate.axisLora',
}

export function axisLabel(axis: XYAxisType): string {
  return i18n.t(AXIS_LABEL_KEYS[axis])
}

/** Only lora_ckpt needs loraIndex (identifies which LoRA's path gets mutated within a cell).
 *  lora_scale was changed to a global axis (all LoRAs share the cell value), no longer bound to a specific LoRA. */
export const REQUIRES_LORA_INDEX: Set<XYAxisType> = new Set(['lora_ckpt'])

/** Parses a comma-separated raw string into axis values. Throws a string error on failure. */
export function parseAxisValues(axis: XYAxisType, raw: string): Array<number | string> {
  const parts = raw.split(',').map((s) => s.trim()).filter((s) => s.length > 0)
  if (parts.length === 0) {
    throw i18n.t('generate.axisValueRequired', { axis: axisLabel(axis) })
  }
  const t = AXIS_VALUE_TYPE[axis]
  if (t === 'string') {
    return parts
  }
  const out: number[] = []
  for (const p of parts) {
    const n = Number(p)
    if (!Number.isFinite(n)) {
      throw i18n.t('generate.axisValueInvalidNumber', { axis: axisLabel(axis), value: p })
    }
    if (t === 'int' && !Number.isInteger(n)) {
      throw i18n.t('generate.axisValueMustBeInteger', { axis: axisLabel(axis), value: p })
    }
    out.push(n)
  }
  return out
}

/** Converts a draft into an XYAxisSpec -- a client-side sanity check before schema validation. */
export function draftToSpec(
  draft: XYAxisDraft,
  loras: LoraEntry[],
): XYAxisSpec {
  const values = parseAxisValues(draft.axis, draft.raw)
  const spec: XYAxisSpec = { axis: draft.axis, values }
  if (REQUIRES_LORA_INDEX.has(draft.axis)) {
    if (draft.loraIndex === null) {
      throw i18n.t('generate.axisRequiresLora', { axis: axisLabel(draft.axis) })
    }
    if (draft.loraIndex >= loras.length) {
      throw i18n.t('generate.axisLoraMissing', { axis: axisLabel(draft.axis), n: draft.loraIndex + 1 })
    }
    spec.lora_index = draft.loraIndex
  }
  return spec
}

/** Builds xy_matrix + the lora_configs actually sent, at XY submit time.
 *
 * **Why we can't just send the whole `loras` bucket**: in XY mode the only entry point that adds
 * an anchor to loras (xyLoras) is the lora_ckpt axis picker, and it only pushes, never prunes
 * (SidebarXYAxes.commitPicks) -- switching project/version, changing an axis type, or deleting the
 * Y axis all leave old anchors behind as "orphans". The backend stacks every entry in
 * lora_configs onto each cell as a base LoRA (anima_daemon._run_xy), so orphans would leak into
 * every image (the root cause of "picked chenbin V3.4 but got v3.2, which was never selected, mixed in / a hoshi from a different mode").
 *
 * Here we keep only the anchors referenced by the current X/Y axes' loraIndex, remapping indices
 * in order of appearance and dropping every orphan. A non-lora_ckpt axis contributes no anchor →
 * lora_configs ends up empty. Validation failures (missing axis values / loraIndex out of range)
 * still throw a string error via draftToSpec. */
export function buildXYMatrix(
  xDraft: XYAxisDraft,
  yDraft: XYAxisDraft | null,
  loras: LoraEntry[],
): { xy_matrix: XYMatrixSpec; loraConfigs: LoraEntry[] } {
  const remap = new Map<number, number>()
  const loraConfigs: LoraEntry[] = []
  const remapDraft = (d: XYAxisDraft): XYAxisDraft => {
    if (d.axis !== 'lora_ckpt' || d.loraIndex == null) return d
    const entry = loras[d.loraIndex]
    if (!entry || !entry.path.trim()) return d // draftToSpec will throw axisLoraMissing
    if (!remap.has(d.loraIndex)) {
      remap.set(d.loraIndex, loraConfigs.length)
      loraConfigs.push(entry)
    }
    return { ...d, loraIndex: remap.get(d.loraIndex) ?? null }
  }
  const x = draftToSpec(remapDraft(xDraft), loraConfigs)
  const y = yDraft ? draftToSpec(remapDraft(yDraft), loraConfigs) : null
  return { xy_matrix: { x, y }, loraConfigs }
}

/** Computes the total cell count (degrades to 1×N when y=null). */
export function cellCount(xLen: number, yLen: number | null): number {
  return xLen * (yLen ?? 1)
}

/** path → a "short name" stripped of directory prefix and the .safetensors suffix (used by the XY header / LoRA cards). */
export function ckptStemFromPath(path: string): string {
  const filename = path.split(/[\\/]/).pop() ?? path
  return filename.replace(/\.safetensors$/i, '')
}

/** If axis is lora_ckpt (value is a path), display using the stem; other types are returned as-is. */
export function formatAxisValue(axis: XYAxisType, value: string): string {
  if (axis === 'lora_ckpt') return ckptStemFromPath(value)
  return value
}
