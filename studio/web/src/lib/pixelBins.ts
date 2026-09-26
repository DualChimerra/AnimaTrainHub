/** Pixel-area binning -- maps to common LoRA training resolutions (shared by
 *  the Preprocess pages and the Overview detail tab). Bin edges are split by
 *  total pixel count (w*h), matching training resolution. */

export const PX_BINS = [
  { id: 'lt-512',    label: '< 512²',         lo: 0,           hi: 512 * 512,   sortKey: 0 },
  { id: '512-768',   label: '512² – 768²',    lo: 512 * 512,   hi: 768 * 768,   sortKey: 512 * 512 },
  { id: '768-1024',  label: '768² – 1024²',   lo: 768 * 768,   hi: 1024 * 1024, sortKey: 768 * 768 },
  { id: '1024-1536', label: '1024² – 1536²',  lo: 1024 * 1024, hi: 1536 * 1536, sortKey: 1024 * 1024 },
  { id: '1536-2048', label: '1536² – 2048²',  lo: 1536 * 1536, hi: 2048 * 2048, sortKey: 1536 * 1536 },
  { id: 'gt-2048',   label: '> 2048²',        lo: 2048 * 2048, hi: Infinity,    sortKey: 2048 * 2048 },
] as const

export type PxBinId = (typeof PX_BINS)[number]['id']

export function pxBinFor(w: number | null, h: number | null): PxBinId | null {
  if (w == null || h == null) return null
  const area = w * h
  const bin = PX_BINS.find((b) => area >= b.lo && area < b.hi)
  return bin?.id ?? PX_BINS[PX_BINS.length - 1].id
}

export interface PixelHistBin {
  id: PxBinId
  label: string
  n: number
}

/** Given a set of (w, h) pairs, returns histogram data for the non-empty bins (fits BarHistogram). */
export function computePixelHist(items: ReadonlyArray<{ w: number | null; h: number | null }>): PixelHistBin[] {
  const counts = new Map<PxBinId, number>(PX_BINS.map((b) => [b.id, 0]))
  for (const it of items) {
    const id = pxBinFor(it.w, it.h)
    if (id) counts.set(id, (counts.get(id) ?? 0) + 1)
  }
  return PX_BINS
    .map((b) => ({ id: b.id, label: b.label, n: counts.get(b.id) ?? 0 }))
    .filter((b) => b.n > 0)
}
