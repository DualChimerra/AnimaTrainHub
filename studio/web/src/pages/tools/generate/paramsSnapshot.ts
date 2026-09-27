/** Test-image params snapshot: shared by the saved PNG metadata and IndexedDB HistoryEntry.params.
 *
 *
 *
 */
import type { LoraEntry, XYAxisType } from '../../../api/client'
import type { DatasetPick } from './PromptFromDatasetPicker'
import {
  SAMPLER_OPTIONS_BY_FAMILY, SCHEDULER_OPTIONS_BY_FAMILY,
  type SamplerName, type SchedulerName,
} from './types'
import type { XYAxisDraft } from './xy'

export const PARAMS_SNAPSHOT_VERSION = 1

export interface SnapshotLora {
  name: string
  scale: number
  project_id?: number | null
  version_id?: number | null
}

export interface SnapshotXYAxis {
  axis: XYAxisType
  raw: string
  loraIndex: number | null
}

/** XY cell PNGs only: points back to the cell's position in its XY plot (so Comfy / A1111
 *  can tell "this is XY cell (xi,yi)"; our own restore goes through mode='single' and ignores it). */
export interface XYCellOrigin {
  xi: number
  yi: number
  xv: string | number
  yv: string | number | null
  x_axis: XYAxisType
  y_axis: XYAxisType | null
}

export interface GenerateParamsSnapshot {
  schema_version: number
  mode: 'single' | 'xy' | 'compare'
  model_family?: 'anima' | 'krea2'
  prompts: string[]
  negative_prompt: string
  width: number
  height: number
  steps: number
  cfg_scale: number
  sampler_name?: string
  scheduler?: string
  /** Base model chosen at the time (official variant key or local custom path); null/missing = follow
   *  the Settings default. Old snapshots lack this field and restore to null (use default). */
  base_model?: string | null
  /** Text encoder variant chosen at the time (krea2): 'bf16' | 'fp8'; missing = the default back then.
   *  Display only; old snapshots lack this field. */
  text_encoder?: 'bf16' | 'fp8'
  count: number
  seed: number
  loras: SnapshotLora[]
  xy_draft?: { x: SnapshotXYAxis; y: SnapshotXYAxis | null } | null
  /** Training-set caption picker selection (keeps the picker UI context).
   *  name is a relative path (e.g. "5_concept/0001.txt"), never a local absolute path. */
  dataset_pick?: DatasetPick | null
  /** XY cell PNGs only; always undefined for composite / single PNGs.
   *  Forward-compat field: old code ignores it and v2 migrate passes it through. */
  xy_origin?: XYCellOrigin | null
}

/** path -> basename (directory stripped), keeping suffixes like .safetensors.
 *  Used when writing metadata so local path structure doesn't leak. */
export function loraBasename(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

/** Raw string of the XY lora_ckpt axis (comma-separated ckpt paths) -> list of basenames.
 *  Other axes have numeric raw strings and are returned as is. */
export function transformAxisRawForSnapshot(draft: XYAxisDraft): SnapshotXYAxis {
  if (draft.axis !== 'lora_ckpt') {
    return { axis: draft.axis, raw: draft.raw, loraIndex: draft.loraIndex }
  }
  const raw = draft.raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(loraBasename)
    .join(', ')
  return { axis: draft.axis, raw, loraIndex: draft.loraIndex }
}

/** Restore: given the ckpts of a (project, version), resolve snapshot LoRAs to paths on this machine.
 *  `ckpts` is the full projectLoras list fetched once on mount. No ids / external LoRA -> caller passes []. */
export function resolveLoraFromCkpts(
  snap: SnapshotLora, ckpts: readonly { path: string }[],
): LoraEntry {
  const byName = ckpts.find((c) => loraBasename(c.path) === snap.name)
  const path = byName?.path ?? ckpts[0]?.path ?? ''
  if (path) return {
    path, scale: snap.scale,
    project_id: snap.project_id ?? null, version_id: snap.version_id ?? null,
  }
  return {
    path: '', scale: snap.scale,
    project_id: snap.project_id ?? null, version_id: snap.version_id ?? null,
    name: snap.name,
  }
}

/** Resolve one snapshot LoRA to a LoraEntry on this machine. Generate implements it with catalog.fetchCkpts
 *  (lazily fetch that version's ckpts, then resolveLoraFromCkpts); injected so this module has no network dependency. */
export type SnapshotLoraResolver = (snap: SnapshotLora) => Promise<LoraEntry>

/** applySnapshot output (decision #8 / Arch v2 Step 3): turns a snapshot into a "prefs field patch".
 *
 */
export interface AppliedSnapshot {
  mode: 'single' | 'xy'
  modelFamily: 'anima' | 'krea2'
  prompts: string[]
  negPrompt: string
  width: number
  height: number
  steps: number
  cfgScale: number
  samplerName: SamplerName
  scheduler: SchedulerName
  count: number
  seed: number
  baseModel: string | null
  datasetPick: DatasetPick | null
  loras: LoraEntry[]
  xDraft?: SnapshotXYAxis
  yDraft?: SnapshotXYAxis | null
  unresolvedLoraCount: number
}

/** Fallback when dataset_pick fails: append the tags to the end of the first prompt.
 *  Joined the same way as `datasetSuffix` in handleGenerate (join(', ')) to avoid visual differences. */
function mergeTagsIntoFirstPrompt(prompts: string[], tags: string[]): string[] {
  if (tags.length === 0) return prompts
  const suffix = tags.join(', ')
  const first = (prompts[0] ?? '').trimEnd()
  if (first.endsWith(suffix)) return prompts
  const sep = first === '' ? '' : (first.endsWith(',') ? ' ' : ', ')
  return [`${first}${sep}${suffix}`, ...prompts.slice(1)]
}

/** Normalize the snapshot's sampler/scheduler to valid values: old snapshots missing fields, or external
 *  PNGs with values we don't support, fall back to the family default instead of putting invalid values into prefs. */
function coerceSampler(v: string | undefined, family: 'anima' | 'krea2'): SamplerName {
  const allowed = SAMPLER_OPTIONS_BY_FAMILY[family] as readonly string[]
  return allowed.includes(v ?? '')
    ? (v as SamplerName)
    : (allowed[0] as SamplerName)
}
function coerceScheduler(v: string | undefined, family: 'anima' | 'krea2'): SchedulerName {
  const allowed = SCHEDULER_OPTIONS_BY_FAMILY[family] as readonly string[]
  return allowed.includes(v ?? '')
    ? (v as SchedulerName)
    : (allowed[0] as SchedulerName)
}

export async function applySnapshot(
  snap: GenerateParamsSnapshot,
  resolveLora: SnapshotLoraResolver,
  projectExists: (projectId: number) => boolean,
): Promise<AppliedSnapshot> {
  const resolved = await Promise.all(snap.loras.map((l) => resolveLora(l)))
  const unresolved = resolved.filter((l) => !l.path).length
  const mode: 'single' | 'xy' = snap.mode === 'single' ? 'single' : 'xy'

  let prompts = snap.prompts
  let datasetPick = snap.dataset_pick ?? null
  if (datasetPick && datasetPick.tags.length > 0 && !projectExists(datasetPick.projectId)) {
    prompts = mergeTagsIntoFirstPrompt(prompts, datasetPick.tags)
    datasetPick = null
  }

  const family: 'anima' | 'krea2' =
    snap.model_family === 'krea2' ? 'krea2' : 'anima'
  const applied: AppliedSnapshot = {
    mode,
    modelFamily: family,
    prompts,
    negPrompt: snap.negative_prompt,
    width: snap.width,
    height: snap.height,
    steps: snap.steps,
    cfgScale: snap.cfg_scale,
    samplerName: coerceSampler(snap.sampler_name, family),
    scheduler: coerceScheduler(snap.scheduler, family),
    count: snap.count,
    seed: snap.seed,
    baseModel: snap.base_model ?? null,
    datasetPick,
    loras: resolved,
    unresolvedLoraCount: unresolved,
  }
  if (mode === 'xy' && snap.xy_draft) {
    applied.xDraft = snap.xy_draft.x
    applied.yDraft = snap.xy_draft.y
  }
  return applied
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

/** Materialize an XY snapshot at (xi, yi) into a single-cell snapshot.
 *
 *
 *
 */
export function buildCellSnapshot(
  xy: GenerateParamsSnapshot,
  cellPos: { xi: number; yi: number },
  axes: {
    x: { axis: XYAxisType; loraIndex: number | null; value: string | number }
    y: { axis: XYAxisType; loraIndex: number | null; value: string | number } | null
  },
): GenerateParamsSnapshot {
  const out: GenerateParamsSnapshot = {
    ...xy,
    mode: 'single',
    xy_draft: null,
    loras: xy.loras.map((l) => ({ ...l })),
  }
  applyAxisToCell(out, axes.x.axis, axes.x.value, axes.x.loraIndex)
  if (axes.y) {
    applyAxisToCell(out, axes.y.axis, axes.y.value, axes.y.loraIndex)
  }
  out.xy_origin = {
    xi: cellPos.xi,
    yi: cellPos.yi,
    xv: axes.x.value,
    yv: axes.y?.value ?? null,
    x_axis: axes.x.axis,
    y_axis: axes.y?.axis ?? null,
  }
  return out
}

function applyAxisToCell(
  snap: GenerateParamsSnapshot,
  axis: XYAxisType,
  value: string | number,
  loraIndex: number | null,
): void {
  switch (axis) {
    case 'steps':
      snap.steps = Math.trunc(Number(value))
      return
    case 'cfg_scale':
      snap.cfg_scale = Number(value)
      return
    case 'lora_scale': {
      const scale = Number(value)
      snap.loras = snap.loras.map((l) => ({ ...l, scale }))
      return
    }
    case 'lora_ckpt': {
      if (loraIndex == null || !snap.loras[loraIndex]) return
      const name = loraBasename(String(value))
      snap.loras = snap.loras.map((l, i) =>
        i === loraIndex ? { ...l, name, project_id: null, version_id: null } : l,
      )
      return
    }
  }
}
