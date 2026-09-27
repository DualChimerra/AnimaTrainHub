import { memo, useEffect, useRef, useState, type SyntheticEvent } from 'react'
import { useTranslation } from 'react-i18next'
import { VirtuosoGrid } from 'react-virtuoso'

export interface ImageGridItem {
  name: string
  thumbUrl: string
  /** Small text shown on the corner badge on hover (optional): e.g. a tag preview. */
  meta?: string
  /** Always-visible small badge, bottom-right of the cell (optional). E.g. "processed", used to distinguish state in a merged view. */
  badge?: string
  /** Always-visible caption along the bottom edge (e.g. "1024×1408"). */
  caption?: string
}

interface Props {
  items: ImageGridItem[]
  selected: Set<string>
  /** Click = toggle checkbox; shift+click = range-select; see applySelection. */
  onSelect: (name: string, e: React.MouseEvent) => void
  /** Hover callback: used to drive an external "large preview panel". */
  onHover?: (name: string) => void
  /** Fullscreen modal preview, triggered by the cell's magnifier button (optional). */
  onPreview?: (name: string) => void
  /** Primary click behavior: selects by default; in activate mode, a plain click is handed to the caller to open/activate. */
  onActivate?: (name: string) => void
  clickMode?: 'select' | 'activate'
  emptyHint?: string
  /** aria-label for the grid, passed in for tests / long-list scenarios. */
  ariaLabel?: string
  /** Column count (auto-fits by width by default). Narrow columns like FolderColumn pass 2-3. */
  columnsClass?: string
  /** The current "active" item (e.g. the one TagEdit is editing), matched exactly against item.name.
   *
   * **Passing** this prop enables "decoupled visuals" mode:
   * - border / ring follow only activeName (marks the active item)
   * - checkbox follows only selected (marks multi-select)
   *
   * **Not passing** it keeps the old behavior: selected drives both border and checkbox (other pages using
   * ImageGrid, like Curation / Download / Reg, never adopted the "active item" concept -- unchanged). */
  activeName?: string
}

// Auto-fills by container width by default: 120px minimum per cell, remaining width split across the last column;
// more columns as the container widens, no breakpoint switching needed.
const DEFAULT_COLUMNS = 'grid-cols-[repeat(auto-fill,minmax(120px,1fr))]'

// Virtual-scroll buffer: about 5-6 rows of cells. The dark theme's cell background is #110f0b (near black);
// a newly mounted cell flashes black while the img is still decoding when scrolling. A large enough overscan
// lets buffer-region images decode ahead of time so they show immediately on entering the viewport instead of
// "black then image". Cost: ~50 extra DOM nodes -- thumbnails are lightweight, acceptable.
const OVERSCAN_PX = 600

// Dominant-color placeholder cache: thumbUrl -> '#RRGGBB'.
//
// Module-level Map (not React state) -- persists across ImageGrid instances and across cell unmount/mount.
// On a cell's first mount, a cache lookup miss shows the default bg-sunken (black); once the img
// onLoad fires, the dominant color is sampled via canvas into the map + setState; if the user scrolls away
// and back, the cell remounts and the lookup hits, immediately filling the background with the dominant
// color, so the user sees a color block instead of black while the img decodes -- the "black flash on scroll" is gone.
//
// Downside: on first browse (cold cache) it's still black -> image; but on regular browsing (scrolling back and
// forth) it has color from the second time on. Not persisted to sessionStorage -- the cache refills within a
// few seconds of every page refresh anyway, negligible cost; persisting it would also mean handling mtime invalidation.
const colorCache = new Map<string, string>()

// Reuses a single 1x1 canvas: drawImage(img, 0, 0, 1, 1) lets the browser do the full downscale-and-average
// internally (1-2 orders of magnitude faster than manually iterating ImageData); getImageData on that one pixel ~= the average color.
let _colorCanvas: HTMLCanvasElement | null = null
function extractAvgColor(img: HTMLImageElement): string | null {
  try {
    if (!_colorCanvas) {
      _colorCanvas = document.createElement('canvas')
      _colorCanvas.width = 1
      _colorCanvas.height = 1
    }
    const ctx = _colorCanvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) return null
    ctx.drawImage(img, 0, 0, 1, 1)
    const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
  } catch {
    // canvas tainted (cross-origin) / jsdom has no real canvas / other failure -- silently fall back
    // to the default bg-sunken. Same-origin thumbs in production are never tainted.
    return null
  }
}

export default function ImageGrid({
  items,
  selected,
  onSelect,
  onHover,
  onPreview,
  onActivate,
  clickMode = 'select',
  emptyHint,
  ariaLabel,
  columnsClass = DEFAULT_COLUMNS,
  activeName,
}: Props) {
  const { t } = useTranslation()
  if (items.length === 0) {
    return <p className="text-fg-tertiary text-sm py-2">{emptyHint ?? t('imageGrid.noImages')}</p>
  }
  const decoupled = activeName !== undefined

  // role="grid" + aria-label go on the outer wrapper: VirtuosoGrid wraps its internal List/Item in several
  // layers of div (scroller/list/item); putting the role directly on an inner layer would get its className/
  // style rewritten by Virtuoso. The outer wrapper is stable. getAllByRole('gridcell') pierces the intermediate
  // divs to find the role="gridcell" on Cell -- unaffected for AT / tests.
  return (
    <div role="grid" aria-label={ariaLabel} className="h-full">
      <VirtuosoGrid
        style={{ height: '100%' }}
        totalCount={items.length}
        overscan={OVERSCAN_PX}
        listClassName={`grid ${columnsClass} gap-[8px]`}
        itemContent={(index) => {
          const it = items[index]
          const isSel = selected.has(it.name)
          const isActive = decoupled && it.name === activeName
          // border follows selected in the old behavior; follows activeName when decoupled
          const borderHighlight = decoupled ? isActive : isSel
          return (
            <Cell
              item={it}
              selected={isSel}
              borderHighlight={borderHighlight}
              onSelect={onSelect}
              onHover={onHover}
              onPreview={onPreview}
              onActivate={onActivate}
              clickMode={clickMode}
            />
          )
        }}
      />
    </div>
  )
}

/** Cell is wrapped in memo: the parent re-renders every time hover changes focus, but most
 * cells' selected / onSelect / item references don't change, so the re-render can be skipped, avoiding
 * recreating the DOM for all N thumbnails.
 *
 * `borderHighlight` controls the accent border + ring (the "highlighted" look), decoupled from `selected`
 * (checkbox state): they match on the old path, but in TagEdit's decoupled mode border follows
 * activeName while checkbox follows multi-select. */
const Cell = memo(function Cell({
  item,
  selected,
  borderHighlight,
  onSelect,
  onHover,
  onPreview,
  onActivate,
  clickMode,
}: {
  item: ImageGridItem
  selected: boolean
  borderHighlight: boolean
  onSelect: (name: string, e: React.MouseEvent) => void
  onHover?: (name: string) => void
  onPreview?: (name: string) => void
  onActivate?: (name: string) => void
  clickMode: 'select' | 'activate'
}) {
  const { t } = useTranslation()
  // Synchronously check the cache on mount -- a hit fills the background with the dominant color right away
  // (avoiding the black flash); a miss leaves undefined -> falls back to bg-sunken in the class name. Lazy init keeps this a one-time lookup.
  const [bg, setBg] = useState<string | undefined>(() => colorCache.get(item.thumbUrl))
  // Whether the img has loaded -- false by default (opacity-0), true after onLoad (opacity-100).
  // Paired with transition-opacity so "color block -> image" is a 150ms fade rather than a snap, further softening
  // the visual jump on a cache hit; on a cache miss it also turns "black -> image pops in" into
  // "black -> image fades in".
  const [loaded, setLoaded] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)
  // On unmount, abort any thumbnail still downloading (src='' is the only way to cancel an <img> request).
  // The browser doesn't cancel the request just because the node was removed -- after switching routes,
  // dozens of half-loaded thumbs would keep occupying the same-origin HTTP/1.1 connection limit of 6,
  // starving the new page's /api fetches at the back of the queue. From the user's view it looks like
  // "the route is stuck behind images". A canceled image doesn't enter the HTTP cache, but the backend's
  // thumb_cache is already on disk + a fully-loaded one gets a 304, so re-entering the page is cheap.
  useEffect(() => {
    // Grab the element on mount: by unmount React has already nulled the ref, so reading
    // imgRef.current in the cleanup wouldn't get the node. Changing src on a detached <img> still aborts the request.
    const img = imgRef.current
    // StrictMode (dev) runs mount → cleanup → mount: the cleanup below has
    // already blanked src, and React will not set an unchanged prop again.
    if (img && img.getAttribute('src') !== item.thumbUrl) img.src = item.thumbUrl
    return () => {
      if (img && !img.complete) img.src = ''
    }
  }, [item.thumbUrl])
  const handleCellClick = (e: React.MouseEvent) => {
    if (clickMode === 'activate' && onActivate && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      onActivate(item.name)
      return
    }
    onSelect(item.name, e)
  }

  const handleSelectionClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    onSelect(item.name, e)
  }

  // img finished loading: (1) sample the color into the cache (if not already there); (2) mark loaded -> triggers the fade-in.
  const handleImgLoad = (e: SyntheticEvent<HTMLImageElement>) => {
    if (!colorCache.has(item.thumbUrl)) {
      const color = extractAvgColor(e.currentTarget)
      if (color) {
        colorCache.set(item.thumbUrl, color)
        setBg(color)
      }
    }
    setLoaded(true)
  }

  return (
    <div
      role="gridcell"
      aria-selected={selected}
      onMouseEnter={onHover ? () => onHover(item.name) : undefined}
      onClick={handleCellClick}
      title={item.meta ? `${item.name}\n${item.meta}` : item.name}
      style={bg ? { background: bg } : undefined}
      className={'ds-thumb group aspect-square cursor-pointer select-none' + (borderHighlight ? ' ds-sel' : '')}
    >
      {/* Can't use loading="lazy" in a virtualized context: the browser won't proactively load a cell as soon as
       * it enters the DOM (including the overscan zone) -- it waits until it actually enters the viewport, which
       * defeats the whole point of overscan pre-warming. Switched to eager (the default) here so the browser starts
       * fetching + decoding as soon as Virtuoso mounts the cell, which combined with a large overscan makes scroll flashing nearly invisible. */}
      <img
        ref={imgRef}
        src={item.thumbUrl}
        alt={item.name}
        decoding="async"
        draggable={false}
        // Lets thumb requests that haven't started yet queue behind data fetches, so in-page actions aren't starved by images.
        // React 18 only passes through lowercase unknown attributes (camelCase fetchPriority is
        // a React 19 thing); spread bypasses TS's check on the unknown prop.
        {...{ fetchpriority: 'low' }}
        onLoad={handleImgLoad}
        className={
          'absolute inset-0 w-full h-full object-cover pointer-events-none transition-opacity duration-150 ' +
          (loaded ? 'opacity-100' : 'opacity-0')
        }
      />
      <button
        type="button"
        onClick={handleSelectionClick}
        aria-label={`${selected ? t('common.deselect') : t('common.select')} ${item.name}`}
        className={'ds-mark flex transition-opacity ' + (selected ? 'opacity-100' : 'opacity-0 group-hover:opacity-100')}
      >
        <span className={'ds-cbox' + (selected ? ' ds-on' : '')}>
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round"><path d="m5 13 4 4L19 7" /></svg>
        </span>
      </button>
      {/* Magnifier: appears on hover, click triggers the fullscreen modal preview (doesn't affect selection state) */}
      {onPreview && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            onPreview(item.name)
          }}
          aria-label={`${t('common.preview')} ${item.name}`}
          className={
            'absolute w-5 h-5 rounded-[5px] bg-black/60 text-white text-[11px] opacity-0 group-hover:opacity-100 hover:bg-black/80 ' +
            // the pin owns the top-right corner when there is one
            (item.badge ? 'right-1.5 ' + (item.caption ? 'bottom-6' : 'bottom-1.5') : 'top-1.5 right-1.5')
          }
        >
          ⤢
        </button>
      )}
      {item.badge && <span className="ds-pin pointer-events-none">{item.badge}</span>}
      {item.caption && <span className="ds-cap pointer-events-none">{item.caption}</span>}
    </div>
  )
})

/** Utility for callers: click = toggle checkbox; shift+click = range-select.
 *
 * - Click an already-selected item -> deselect it
 * - Click an unselected item -> select it
 * - shift+click: selects every item between the anchor and the current position (doesn't deselect already-selected ones)
 *
 * Note: no longer requires ctrl/cmd -- a plain click toggles, matching "checkbox multi-select" UX.
 */
export function applySelection(
  current: Set<string>,
  name: string,
  e: React.MouseEvent,
  names: string[],
  lastAnchor: string | null
): { next: Set<string>; anchor: string } {
  if (e.shiftKey && lastAnchor && names.includes(lastAnchor)) {
    const i = names.indexOf(lastAnchor)
    const j = names.indexOf(name)
    if (j === -1) return _toggle(current, name)
    const [lo, hi] = i < j ? [i, j] : [j, i]
    const next = new Set(current)
    for (let k = lo; k <= hi; k++) next.add(names[k])
    return { next, anchor: name }
  }
  return _toggle(current, name)
}

function _toggle(
  current: Set<string>,
  name: string
): { next: Set<string>; anchor: string } {
  const next = new Set(current)
  if (next.has(name)) next.delete(name)
  else next.add(name)
  return { next, anchor: name }
}
