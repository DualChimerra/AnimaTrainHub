import type { ReactNode } from 'react'

/** Filmstrip only depends on `name` (selection key + tooltip); the thumbnail URL is supplied by the caller's closure. */
export interface FilmstripItemBase {
  name: string
}

/** The left-column vertical nav shared by the preprocess sub-pages
 *  (extracted from the crop page, reused by the inpaint page in PR-A).
 *
 *  3-col vertical grid with square cover thumbs. Squaring is intentional --
 *  when a dataset mixes portrait/landscape ARs, a square cover-crop keeps
 *  the grid tidy; the full AR is visible on the main canvas. Page-specific
 *  badges (the crop rect overlay / the inpaint "edited" dot) are injected
 *  via renderOverlay -- this component has no notion of crop / stroke.
 *
 *  The empty state renders the same container too -- when the caller places
 *  this component inside a multi-column grid, conditionally unmounting it
 *  would collapse the column layout (learned the hard way on the crop
 *  page's 264-image dataset).
 */
export default function Filmstrip<T extends FilmstripItemBase>({
  items,
  activeName,
  onSelect,
  thumbUrl,
  emptyHint,
  renderOverlay,
}: {
  items: T[]
  activeName: string | null
  onSelect: (name: string) => void
  thumbUrl: (im: T) => string
  emptyHint?: string
  renderOverlay?: (im: T) => ReactNode
}) {
  if (items.length === 0) {
    return (
      <div className="flex items-center justify-center bg-sunken/40 border border-subtle rounded p-3 h-full text-center text-fg-tertiary text-[11px] leading-snug">
        {emptyHint ?? ''}
      </div>
    )
  }
  return (
    <div className="grid grid-cols-3 gap-1 overflow-y-auto pr-1 bg-sunken/40 border border-subtle rounded p-1.5 h-full content-start">
      {items.map((im) => {
        const isActive = im.name === activeName
        return (
          <div key={im.name} className="fs-thumb-sq-cell">
            <button
              onClick={() => onSelect(im.name)}
              className={'fs-thumb-sq ' + (isActive ? 'is-active' : '')}
              title={im.name}
            >
              {/* <img> instead of background-image: browsers honour Cache-Control
                  + ETag for <img src> reliably; CSS background-image hits the
                  in-memory decoded-image cache and can keep showing stale bytes
                  after an in-place crop output. object-fit: cover preserves the
                  original squared-thumbnail look. */}
              <img
                src={thumbUrl(im)}
                alt=""
                draggable={false}
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  objectPosition: 'center',
                  pointerEvents: 'none',
                }}
              />
              {renderOverlay?.(im)}
            </button>
          </div>
        )
      })}
    </div>
  )
}
