/** Auto-save test-generated images to disk (when Settings → testing → save_test_images is on).
 *
 * Paths:
 *  - single: `studio_data/test/<YYYY-MM-DD>/single/single image N.png`
 *  - xy:     `studio_data/test/<YYYY-MM-DD>/xy/xy plot N/{xy plot.png + cell x{xi} y{yi}.png ...}`
 *
 * single: each sample is uploaded separately (one image per POST);
 * xy: sent in one multipart request: composite + N cells + cells_manifest (a per-cell
 * materialized single-snapshot); the server writes them all into the same folder, atomic across the whole commit.
 *
 * The upload includes the params JSON (GenerateParamsSnapshot): the backend writes it into the
 * PNG's `anima_params` tEXt chunk, and for single also writes the a1111 `parameters` block, for
 * the history rail to look back at / for the user to copy the PNG and reuse its parameters.
 *
 * Failure / a backend 403 (the toggle is off) are both swallowed silently -- doesn't interrupt the user's main flow.
 */
import { api } from '../../../api/client'
import { composeXYMatrix, type ExportInput } from './exportXY'
import { buildCellSnapshot, type GenerateParamsSnapshot } from './paramsSnapshot'

interface SingleSaveResult {
  path: string
  index: number
  filename: string
}

interface XYSaveResult {
  folder: string
  index: number
  composite: string
  cells: string[]
}

/** Saves every sample in single mode to disk. Returns each image's server path (same order as
 *  filenames; a failed slot is null). The caller uses the 0th path as entry.diskPath (a dedup key). */
export async function saveSingleSamples(
  taskId: number,
  filenames: string[],
  params: GenerateParamsSnapshot,
): Promise<Array<string | null>> {
  const paths: Array<string | null> = []
  for (const fn of filenames) {
    try {
      const res = await fetch(api.generateSampleUrl(taskId, fn))
      if (!res.ok) { paths.push(null); continue }
      const blob = await res.blob()
      const fd = new FormData()
      fd.append('mode', 'single')
      fd.append('image', blob, 'single.png')
      fd.append('params', JSON.stringify(params))
      // 0.17 item3: include task_id so the server enriches it into the PNG's anima_params; only
      // then can the `?task=` deep link look-back match this disk entry by task_id (previously
      // omitted → saved images had no task_id → look-back failed).
      fd.append('task_id', String(taskId))
      const r = await fetch('/api/generate/save', { method: 'POST', body: fd })
      if (!r.ok) { paths.push(null); continue }
      const data = await r.json() as SingleSaveResult
      paths.push(data.path)
    } catch {
      paths.push(null)
    }
  }
  return paths
}

/** Saves an xy folder to disk (composite + each cell's original image). Returns the server-side
 *  folder path (null on failure). `xySnapshot.mode` must be 'xy'; this function derives the per-cell single-snapshot. */
export async function saveXYMatrix(
  input: ExportInput,
  xySnapshot: GenerateParamsSnapshot,
): Promise<string | null> {
  try {
    const { samples, taskId, xAxis, yAxis, xValues, yValues } = input
    const xLoraIndex = xySnapshot.xy_draft?.x.loraIndex ?? null
    const yLoraIndex = xySnapshot.xy_draft?.y?.loraIndex ?? null

    // 1) Fetch every cell's PNG bytes + build a per-cell single-snapshot
    type CellEntry = { xi: number; yi: number; blob: Blob; params: GenerateParamsSnapshot }
    const cellEntries: CellEntry[] = []
    for (const s of samples) {
      const fn = s.path.split(/[\\/]/).pop()
      if (!fn) continue
      try {
        const res = await fetch(api.generateSampleUrl(taskId, fn))
        if (!res.ok) continue
        const blob = await res.blob()
        const xv = xValues[s.xy.xi] ?? ''
        const yv = yAxis ? (yValues[s.xy.yi] ?? '') : null
        const cellParams = buildCellSnapshot(xySnapshot, s.xy, {
          x: { axis: xAxis, loraIndex: xLoraIndex, value: xv },
          y: yAxis ? { axis: yAxis, loraIndex: yLoraIndex, value: yv ?? '' } : null,
        })
        cellEntries.push({ xi: s.xy.xi, yi: s.xy.yi, blob, params: cellParams })
      } catch {
        // A single cell's fetch failed, skip it; the server validates against the manifest, and a missing cell just isn't sent
      }
    }
    if (cellEntries.length === 0) return null

    // 2) Composite the full image (reuse the existing composeXYMatrix)
    const composite = await composeXYMatrix(input)

    // 3) Send everything in one multipart request: composite + N cells + cells_manifest
    const fd = new FormData()
    fd.append('mode', 'xy')
    fd.append('image', composite, 'xy plot.png')
    fd.append('params', JSON.stringify(xySnapshot))
    fd.append('task_id', String(taskId))  // 0.17 item3: same as single, so `?task=` deep-link look-back matches
    const manifest = cellEntries.map(({ xi, yi, params }) => ({ xi, yi, params }))
    fd.append('cells_manifest', JSON.stringify(manifest))
    for (const { xi, yi, blob } of cellEntries) {
      fd.append('cells', blob, `cell x${xi} y${yi}.png`)
    }
    const r = await fetch('/api/generate/save', { method: 'POST', body: fd })
    if (!r.ok) return null
    const data = await r.json() as XYSaveResult
    return data.folder
  } catch {
    return null
  }
}
