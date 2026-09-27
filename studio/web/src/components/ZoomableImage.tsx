import { useLayoutEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useZoomPan } from '../lib/useZoomPan'

/** Zoomable single-image viewer (a viewer wrapper around useZoomPan).
 *
 *  Fills the parent container (the parent decides overall size), with the
 *  same structure used by the inpaint / crop pages: a viewport (rounded
 *  border, bg-sunken, wheel = zoom around the pointer, left-drag = pan,
 *  double-click = toggle fit<->100%) plus a bottom readout strip (zoom% /
 *  fit / 100% / image size / usage hint). Refitting happens automatically
 *  when src changes.
 *
 *  For pure viewing scenarios (FullscreenViewer / ImagePreviewModal / the
 *  TagEdit single image / the test page's output preview); brush-style
 *  tools (InpaintCanvas) compose useZoomPan directly instead.
 */
export default function ZoomableImage({
  src,
  alt,
  className,
  style,
  onError,
  onNaturalSize,
  readout = true,
}: {
  src: string
  alt?: string
  className?: string
  style?: React.CSSProperties
  /** Called when the img fails to load (for the caller to switch to a placeholder UI). */
  onError?: () => void
  /** Called with the natural pixel size once the image has loaded. */
  onNaturalSize?: (w: number, h: number) => void
  /** Zoom readout strip under the viewport; off for small previews. */
  readout?: boolean
}) {
  const { t } = useTranslation()
  // Not reset when src changes: the browser keeps painting the current image until the
  // next one has loaded and decoded, so flipping through a sequence never flashes an
  // empty viewport. onLoad then stores the new size and we refit before the frame paints.
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null)
  const zp = useZoomPan({
    contentW: nat?.w ?? 0,
    contentH: nat?.h ?? 0,
    primaryButtonPans: true,
  })
  const { fit } = zp
  useLayoutEffect(() => { if (nat) fit() }, [nat, fit])

  return (
    <div
      className={'flex flex-col gap-1.5 w-full h-full min-h-0 ' + (className ?? '')}
      style={style}
    >
      <div
        ref={zp.wrapRef}
        {...zp.handlers}
        onDoubleClick={() => (zp.zoomPct === 100 ? zp.fit() : zp.reset100())}
        className="relative flex-1 min-h-0 overflow-hidden rounded border border-subtle bg-sunken"
        style={{ touchAction: 'none', cursor: 'grab' }}
      >
        <img
          ref={(el) => { zp.contentRef.current = el }}
          src={src}
          alt={alt}
          draggable={false}
          decoding="async"
          onLoad={(e) => {
            const w = e.currentTarget.naturalWidth
            const h = e.currentTarget.naturalHeight
            setNat({ w, h })
            onNaturalSize?.(w, h)
          }}
          onError={onError}
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            transformOrigin: '0 0',
            maxWidth: 'none',
            maxHeight: 'none',
            willChange: 'transform',
            visibility: nat ? 'visible' : 'hidden',
          }}
        />
      </div>

      {/* readout strip (same layout as the inpaint / crop pages) */}
      {readout && (
      <div className="shrink-0 flex items-center gap-2 text-[11px] font-mono text-fg-tertiary px-1">
        <span>{zp.zoomPct}%</span>
        <button
          type="button"
          className="px-1.5 py-0.5 rounded hover:bg-overlay hover:text-fg-primary"
          onClick={() => zp.fit()}
        >{t('common.zoomFit')}</button>
        <button
          type="button"
          className="px-1.5 py-0.5 rounded hover:bg-overlay hover:text-fg-primary"
          onClick={() => zp.reset100()}
        >100%</button>
        <span className="flex-1" />
        {nat && <span>{nat.w}×{nat.h}</span>}
        <span className="text-fg-disabled">{t('common.zoomHint')}</span>
      </div>)}
    </div>
  )
}
