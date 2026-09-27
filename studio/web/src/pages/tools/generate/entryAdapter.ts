/** History entry source adapter (plan decision #12).
 *
 * All logic that branches on entry.source is collected into this one file -- UI / handlers /
 * hooks all go through the helpers here, and **never switch on entry.source directly**.
 *
 * Adding a third source in the future ('upload' / 'remote') only touches this file; consumers need no changes.
 *
 * Why function helpers instead of object methods (`entry.imageUrl()`):
 * - entries must be serializable into sessionStorage (undo snapshots) and passed across hooks; object methods can't be serialized
 * - function helpers fit the React/TS style better
 * - mocking a helper in tests is cleaner than mocking a class method
 */
import { api } from '../../../api/client'
import type { GenerateParamsSnapshot } from './paramsSnapshot'

/** Axis metadata for XY history playback (shared by CacheEntry + DiskEntry -- used to rebuild PreviewXYGrid).
 *
 *  - CacheEntry: samples[].path is a cache filename; imageUrl is left empty, and PreviewXYGrid
 *    falls back to `api.generateSampleUrl(taskId, filename)`
 *  - DiskEntry: the server already URL-encodes imageUrl
 *    (`/api/generate/disk/image/<date>/xy/<folder>/<cell>`), and PreviewXYGrid uses it directly */
export interface HistoryXYMeta {
  xAxis: string
  yAxis: string | null
  xValues: string[]
  yValues: Array<string | null>
  samples: Array<{
    path: string
    xy: { xi: number; yi: number; xv: string | number; yv: string | number | null }
    /** Already URL-encoded by the server for disk-served entries; undefined for cache -> falls back to api.generateSampleUrl */
    imageUrl?: string
  }>
}

/** Persistent entry: a PNG / folder on disk, pure derived data returned by the server's disk-history endpoint.
 *
 * single: one PNG (filename + imageUrl pointing at it)
 * xy: an `xy plot N/` folder (folder + imageUrl pointing at the composite,
 *     xyMeta.samples holding each cell's info + a direct URL).
 */
export interface DiskEntry {
  source: 'disk'
  /** Stable id returned by the server: 'disk:<sha1-12>' (decision #12, avoids stuffing a filename with spaces into a React key) */
  id: string
  mode: 'single' | 'xy'
  /** YYYY-MM-DD, the corresponding folder */
  date: string
  /** single: the PNG filename (with extension); xy: undefined (use `folder` instead) */
  filename?: string
  /** xy: the folder name `xy plot <N>`; single: undefined */
  folder?: string
  /** Large-image URL, usable directly with <img src=...> (already URL-encoded server-side). Points at the composite in xy mode. */
  imageUrl: string
  /** Thumbnail URL, server-side resized + ETag */
  thumbUrl: string
  /** PNG / composite mtime */
  createdAt: number
  /** Parameter snapshot parsed out of the PNG's anima_params (in xy mode, the composite's XY snapshot) */
  params: GenerateParamsSnapshot
  /** Per-cell metadata for xy mode (used to rebuild PreviewXYGrid); undefined in single mode */
  xyMeta?: HistoryXYMeta
}

/** Transient entry: lives only in the server's in-memory cache, for the current session.
 *  Lost when the tab closes, the server restarts, or it's evicted by LRU. */
export interface CacheEntry {
  source: 'cache'
  id: string  // uuid
  mode: 'single' | 'xy'
  /** daemon task id, used to build the cache URL */
  taskId: number
  createdAt: number
  /** Filenames in the server's cache (XY has N per-cell files; single has 1) */
  filenames: string[]
  /** Parameter snapshot built at generation time (not read back from the cache) */
  params: GenerateParamsSnapshot
  /** XY mode's axis + per-cell metadata, used to rebuild PreviewXYGrid */
  xyMeta?: HistoryXYMeta
}

export type HistoryEntry = DiskEntry | CacheEntry

// ---------------------------------------------------------------------------
// Helpers that branch on source -- the **only** place allowed to switch on entry.source
// ---------------------------------------------------------------------------

/** The large-image URL for the entry's cell at `idx`. */
export function entryImageUrl(e: HistoryEntry, idx = 0): string {
  switch (e.source) {
    case 'disk':
      return e.imageUrl  // Already encoded by the server
    case 'cache': {
      const fn = e.filenames[idx] ?? e.filenames[0] ?? ''
      return api.generateSampleUrl(e.taskId, fn)
    }
  }
}

/** The entry's thumbnail URL (for the small-image rail). */
export function entryThumbUrl(e: HistoryEntry): string {
  switch (e.source) {
    case 'disk':
      return e.thumbUrl
    case 'cache':
      // CacheEntry has no server-side thumbnail -- just use the full-image URL with CSS scaling
      // (not many images are generated within a session, so a browser loading a few originals
      // is acceptable; not worth adding IDB / server-side thumbnail complexity for such a short-lived entry)
      return entryImageUrl(e, 0)
  }
}

/** The params snapshot carried by the entry (used to backfill on a history click). */
export function entryParams(e: HistoryEntry): GenerateParamsSnapshot {
  return e.params  // Field name is the same across both sources
}

/** The generate task id for the entry (0.17 P-H's `?task=` deep link matches by this).
 *  cache carries taskId at the top level; disk hides it in the task_id the server enriches into the PNG's anima_params. */
export function entryTaskId(e: HistoryEntry): number | undefined {
  switch (e.source) {
    case 'cache':
      return e.taskId
    case 'disk':
      return (e.params as { task_id?: number }).task_id
  }
}

/** The entry's display label (for PreviewHistoryRail / badge text). */
export function entryDisplayLabel(e: HistoryEntry): string {
  switch (e.source) {
    case 'disk':
      // xy uses the folder name ("xy plot 3"); single uses the filename minus the .png extension
      if (e.mode === 'xy' && e.folder) return e.folder
      return (e.filename ?? '').replace(/\.png$/i, '')
    case 'cache':
      return `#${e.taskId}`
  }
}

/** The badge for an XY entry in the history rail ("XY 5x3"). */
export function entryBadge(e: HistoryEntry): string | undefined {
  if (e.mode !== 'xy') return undefined
  if (e.source === 'cache' && e.xyMeta) {
    const xs = new Set(e.xyMeta.samples.map((s) => s.xy.xi))
    const ys = new Set(e.xyMeta.samples.map((s) => s.xy.yi))
    return `XY ${xs.size}×${ys.size || 1}`
  }
  if (e.source === 'disk' && e.params.xy_draft) {
    const xLen = e.params.xy_draft.x.raw.split(',').filter((s) => s.trim()).length
    const yLen = e.params.xy_draft.y?.raw.split(',').filter((s) => s.trim()).length ?? 1
    return `XY ${xLen}×${yLen}`
  }
  return 'XY'
}

/** 删除 entry：
 *  - DiskEntry single：单文件 DELETE `/api/generate/disk/<date>/single/<encoded>`
 *  - DiskEntry xy：整文件夹 DELETE `/api/generate/disk/<date>/xy/<encoded folder>`
 *  - CacheEntry：仅本地 splice（无 server 删）
 *  返回 server 端是否真删了（CacheEntry 永远 false）。 */
export async function entryDelete(e: HistoryEntry): Promise<{ removed: boolean }> {
  if (e.source !== 'disk') return { removed: false }
  let url: string
  if (e.mode === 'xy') {
    if (!e.folder) throw new Error('xy disk entry missing folder')
    url = `/api/generate/disk/${e.date}/xy/${encodeURIComponent(e.folder)}`
  } else {
    if (!e.filename) throw new Error('single disk entry missing filename')
    url = `/api/generate/disk/${e.date}/single/${encodeURIComponent(e.filename)}`
  }
  const r = await fetch(url, { method: 'DELETE' })
  if (!r.ok) throw new Error(`delete failed: ${r.status}`)
  const data = await r.json() as { ok: boolean; noop?: boolean }
  return { removed: !data.noop }
}
