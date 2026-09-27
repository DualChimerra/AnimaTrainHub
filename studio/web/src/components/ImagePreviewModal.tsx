import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'

import ZoomableImage from './ZoomableImage'

interface Props {
  src: string
  /** Comparison image: if passed, switches to a left/right split layout (src on
   *  the left, compareSrc on the right). Stacks vertically on narrow screens. */
  compareSrc?: string
  /** Small label above the left image in split layout (e.g. "original"). */
  srcLabel?: string
  /** Small label above the right image in split layout (e.g. "processed"). */
  compareLabel?: string
  caption?: string
  /** 0-based position in the list; when passed together with total, shows an "index+1 / total" counter at the bottom. */
  index?: number
  total?: number
  hasPrev?: boolean
  hasNext?: boolean
  onClose: () => void
  onPrev?: () => void
  onNext?: () => void
  onAccept?: () => void
  onDelete?: () => void
  shortcutHint?: string
  /** URLs to warm up in the background (usually the prev/next image), so flipping
   *  with the arrows shows an already-decoded image instead of loading from scratch. */
  preload?: string[]
}

export default function ImagePreviewModal({
  src,
  compareSrc,
  srcLabel,
  compareLabel,
  caption,
  index,
  total,
  hasPrev,
  hasNext,
  onClose,
  onPrev,
  onNext,
  onAccept,
  onDelete,
  shortcutHint,
  preload,
}: Props) {
  // Handlers live in a ref so the keydown listener binds once: callers pass fresh
  // inline closures on every render (and some re-render on every training tick).
  const keysRef = useRef({ hasPrev, hasNext, onPrev, onNext, onClose, onAccept, onDelete })
  keysRef.current = { hasPrev, hasNext, onPrev, onNext, onClose, onAccept, onDelete }
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const { hasPrev, hasNext, onPrev, onNext, onClose, onAccept, onDelete } = keysRef.current
      if (e.key === 'Escape') {
        e.preventDefault()
        onClose()
      } else if (e.key === 'ArrowLeft' && hasPrev && onPrev) {
        e.preventDefault()
        onPrev()
      } else if (e.key === 'ArrowRight' && hasNext && onNext) {
        e.preventDefault()
        onNext()
      } else if ((e.key === 'Enter' || e.key === ' ') && onAccept) {
        e.preventDefault()
        onAccept()
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && onDelete) {
        e.preventDefault()
        onDelete()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  // The page behind stays put while the lightbox is open (no scroll-through on wheel / touch).
  useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = prev }
  }, [])

  // Warm up the neighbours; keyed by the joined URLs so a re-render with the same list is a no-op.
  const preloadKey = (preload ?? []).filter(Boolean).join('\n')
  useEffect(() => {
    if (!preloadKey) return
    const t = window.setTimeout(() => {
      for (const u of preloadKey.split('\n')) {
        const img = new Image()
        img.decoding = 'async'
        img.src = u
      }
    }, 150)
    return () => window.clearTimeout(t)
  }, [preloadKey])

  const counter = index != null && total != null ? `${index + 1} / ${total}` : null

  // Portaled to <body>: callers mount this deep inside cards, and an ancestor with opacity /
  // transform (e.g. a finished queue card at opacity .72) would otherwise make the fixed
  // overlay translucent, trap it under later siblings and composite the page behind every frame.
  return createPortal(
    <div
      className="fixed inset-0 z-50 bg-black flex flex-col"
      onClick={onClose}
    >
      <div className="relative flex-1 min-h-0 flex items-center justify-center p-4 sm:p-6">
        <button
          onClick={(e) => {
            e.stopPropagation()
            onClose()
          }}
          className="absolute top-3 right-4 z-10 rounded bg-black/50 px-3 py-1 text-slate-300 hover:text-white text-2xl"
          aria-label="Close"
        >
          ×
        </button>
        {hasPrev && onPrev && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              onPrev()
            }}
            className="absolute left-4 top-1/2 z-10 -translate-y-1/2 text-slate-300 hover:text-white text-5xl px-4 py-3 bg-black/30 rounded"
            aria-label="Previous"
          >
            ‹
          </button>
        )}
        {hasNext && onNext && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              onNext()
            }}
            className="absolute right-4 top-1/2 z-10 -translate-y-1/2 text-slate-300 hover:text-white text-5xl px-4 py-3 bg-black/30 rounded"
            aria-label="Next"
          >
            ›
          </button>
        )}
        {compareSrc ? (
          // The wrapper doesn't stopPropagation -- so clicking between panes /
          // on blank space inside a pane bubbles up to the outer onClose; only
          // the img itself stops it (clicking the image doesn't close).
          <div className="w-full h-full flex flex-col md:flex-row items-stretch justify-center gap-2 md:gap-4">
            <SplitPane src={src} label={srcLabel} altFallback={caption} />
            <SplitPane src={compareSrc} label={compareLabel} altFallback={caption} />
          </div>
        ) : (
          // Single image: a zoomable viewport (wheel / drag / double-click,
          // useZoomPan). Clicking the viewport doesn't close it (drag-to-pan and
          // click-to-close can't coexist) -- the × button / ESC / edge backdrop
          // can still close it.
          <div className="w-full h-full" onClick={(e) => e.stopPropagation()}>
            <ZoomableImage src={src} alt={caption ?? 'preview'} />
          </div>
        )}
      </div>
      {(counter || caption || shortcutHint) && (
        <div className="shrink-0 border-t border-white/10 bg-black px-4 py-2 flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-xs text-slate-400">
          {counter && <div className="font-mono text-slate-300">{counter}</div>}
          {caption && <div className="font-mono text-slate-300">{caption}</div>}
          {shortcutHint && <div>{shortcutHint}</div>}
        </div>
      )}
    </div>,
    document.body,
  )
}

function SplitPane({
  src,
  label,
  altFallback,
}: { src: string; label?: string; altFallback?: string }) {
  return (
    <div className="flex-1 min-h-0 min-w-0 flex flex-col items-center justify-center gap-1.5 overflow-hidden">
      {label && (
        <div className="shrink-0 text-[11px] font-mono uppercase tracking-wider text-slate-400">
          {label}
        </div>
      )}
      <img
        src={src}
        alt={label ?? altFallback ?? 'preview'}
        onClick={(e) => e.stopPropagation()}
        decoding="async"
        className="max-w-full max-h-full object-contain"
      />
    </div>
  )
}
