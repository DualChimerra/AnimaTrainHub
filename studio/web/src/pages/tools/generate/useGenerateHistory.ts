/** Test-image history rail.
 *
 *
 *
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type CacheGenerateHistoryEntry } from '../../../api/client'
import {
  entryDelete,
  type CacheEntry,
  type DiskEntry,
  type HistoryEntry,
  type HistoryXYMeta,
} from './entryAdapter'
import type { GenerateParamsSnapshot } from './paramsSnapshot'

export type { CacheEntry, DiskEntry, HistoryEntry, HistoryXYMeta } from './entryAdapter'

interface DiskHistoryServerXYMeta {
  x_axis: string | null
  y_axis: string | null
  x_values: string[]
  y_values: Array<string | null>
  samples: Array<{
    path: string
    xy: { xi: number; yi: number; xv: string | null; yv: string | null }
    image_url: string
  }>
}

interface DiskHistoryServerEntry {
  id: string
  date: string
  mode: 'single' | 'xy'
  filename?: string
  folder?: string
  path: string
  image_url: string
  thumb_url: string
  created_at: number
  schema_version: number
  params: unknown
  xy_meta?: DiskHistoryServerXYMeta | null
}

interface DiskHistoryResponse {
  entries: DiskHistoryServerEntry[]
}

function xyMetaFromServer(meta: DiskHistoryServerXYMeta): HistoryXYMeta {
  return {
    xAxis: meta.x_axis ?? '',
    yAxis: meta.y_axis,
    xValues: meta.x_values,
    yValues: meta.y_values,
    samples: meta.samples.map((s) => ({
      path: s.path,
      xy: { xi: s.xy.xi, yi: s.xy.yi, xv: s.xy.xv ?? '', yv: s.xy.yv },
      imageUrl: s.image_url,
    })),
  }
}

function diskEntryFromServer(d: DiskHistoryServerEntry): DiskEntry {
  return {
    source: 'disk',
    id: d.id,
    mode: d.mode,
    date: d.date,
    filename: d.filename,
    folder: d.folder,
    imageUrl: d.image_url,
    thumbUrl: d.thumb_url,
    createdAt: d.created_at * 1000,
    params: d.params as GenerateParamsSnapshot,
    xyMeta: d.xy_meta ? xyMetaFromServer(d.xy_meta) : undefined,
  }
}

/** server cache index -> frontend CacheEntry. XY mode rebuilds xyMeta from server samples;
 *  axis metadata comes from params.xy_draft (same scheme as entryBadge). */
function cacheEntryFromServer(c: CacheGenerateHistoryEntry): CacheEntry {
  const params = c.params as unknown as GenerateParamsSnapshot
  let xyMeta: HistoryXYMeta | undefined
  if (c.mode === 'xy' && c.samples && c.samples.length > 0) {
    const xDraft = params.xy_draft?.x
    const yDraft = params.xy_draft?.y
    const xValues = xDraft?.raw.split(',').map((s) => s.trim()).filter(Boolean) ?? []
    const yValues = yDraft
      ? yDraft.raw.split(',').map((s) => s.trim()).filter(Boolean)
      : [null as string | null]
    xyMeta = {
      xAxis: xDraft?.axis ?? '',
      yAxis: yDraft?.axis ?? null,
      xValues,
      yValues,
      samples: c.samples.map((s) => ({
        path: s.filename,
        xy: {
          xi: s.xy.xi, yi: s.xy.yi,
          xv: typeof s.xy.xv === 'number' ? s.xy.xv : (s.xy.xv ?? ''),
          yv: s.xy.yv,
        },
      })),
    }
  }
  return {
    source: 'cache',
    id: c.id,
    mode: c.mode,
    taskId: c.taskId,
    createdAt: c.createdAt,
    filenames: c.filenames,
    params,
    xyMeta,
  }
}

export interface UseGenerateHistoryResult {
  entries: HistoryEntry[]
  loading: boolean
  remove: (id: string) => Promise<void>
  refresh: () => Promise<void>
  refreshCache: () => Promise<void>
}

export function useGenerateHistory(): UseGenerateHistoryResult {
  const [diskEntries, setDiskEntries] = useState<DiskEntry[]>([])
  const [cacheEntries, setCacheEntries] = useState<CacheEntry[]>([])
  const [loading, setLoading] = useState(true)
  const loadedRef = useRef(false)

  const fetchDisk = async () => {
    try {
      const r = await fetch('/api/generate/disk/history')
      if (!r.ok) return
      const data = (await r.json()) as DiskHistoryResponse
      setDiskEntries(data.entries.map(diskEntryFromServer))
    } catch {
    }
  }

  const fetchCache = async () => {
    try {
      const data = await api.listCacheGenerateHistory()
      setCacheEntries(data.entries.map(cacheEntryFromServer))
    } catch {
    }
  }

  useEffect(() => {
    if (loadedRef.current) return
    loadedRef.current = true
    setLoading(true)
    void Promise.all([fetchDisk(), fetchCache()]).finally(() => setLoading(false))
  }, [])

  const entries = useMemo<HistoryEntry[]>(
    () => [...diskEntries, ...cacheEntries].sort((a, b) => b.createdAt - a.createdAt),
    [diskEntries, cacheEntries],
  )

  const remove = async (id: string) => {
    const target = entries.find((e) => e.id === id)
    if (!target) return
    if (target.source === 'disk') {
      try {
        await entryDelete(target)
      } catch {
      }
      setDiskEntries((prev) => prev.filter((e) => e.id !== id))
    } else {
      setCacheEntries((prev) => prev.filter((e) => e.id !== id))
    }
  }

  return { entries, loading, remove, refresh: fetchDisk, refreshCache: fetchCache }
}
