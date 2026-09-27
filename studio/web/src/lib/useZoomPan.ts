/**
 * useZoomPan -- image / canvas viewing viewport: wheel zoom around the
 * pointer + drag pan + fit/100%.
 *
 * Extracted from the inpaint page's InpaintCanvas view layer (PR-B series),
 * reused by every single-image viewing scenario: FullscreenViewer /
 * ImagePreviewModal / the embedded single image in TagEdit / InpaintCanvas.
 *
 * Usage (viewer, left-drag = pan):
 *   const zp = useZoomPan({ contentW, contentH, primaryButtonPans: true })
 *   <div ref={zp.wrapRef} {...zp.handlers} style={{ touchAction: 'none', ... }}>
 *     <img ref={zp.contentRef} style={{ position:'absolute', left:0, top:0,
 *          transformOrigin:'0 0' }} ... />
 *   </div>
 *
 * Usage (brush-style tools, left button reserved for the brush, spacebar /
 * middle button = pan): primaryButtonPans defaults to false; the caller asks
 * panPointerDown/Move first in its own pointer handler -- returning true
 * means this event was claimed as a pan gesture and the brush logic should
 * skip it.
 *
 * Behavior conventions (matching the inpaint page):
 * - wheel zoom is centered on the pointer (non-passive, preventDefault to stop page scroll)
 * - holding spacebar = the pan modifier key (not hijacked while a form element is focused)
 * - on container resize, stays fit as long as the user hasn't manually touched the view
 * - contentW/H changes (switching images) trigger an automatic refit
 * - view state lives in a ref and writes straight to the DOM transform --
 *   the pan/zoom hot path never goes through React rendering; only zoomPct
 *   (for the readout) goes through state
 */
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

export interface ZoomPanView {
  scale: number
  tx: number
  ty: number
}

export interface UseZoomPanOptions {
  /** The content's native size (image naturalWidth/Height or canvas size). Treated as not-ready when 0. */
  contentW: number
  contentH: number
  /** true = left-drag pans (pure viewer); false = spacebar / middle button only (brush-style tools). */
  primaryButtonPans?: boolean
  /** How zoom is applied to the DOM:
   *  - 'transform' (default): translate+scale, pure compositing, fastest;
   *    scales the content as a whole (including borders / children).
   *  - 'size': changes width/height + translates -- content reflows at the
   *    new size, while px-based borders / handles / labels don't scale with
   *    it (used by DOM overlay editors, e.g. FreeCropEditor's %-positioned
   *    rects). Triggers layout every frame, acceptable for a small subtree. */
  applyMode?: 'transform' | 'size'
  /** Padding factor used when fitting, defaults to 0.98. */
  fitPadding?: number
  /** Zoom bounds (scale values). */
  minScale?: number
  maxScale?: number
}

export function useZoomPan({
  contentW,
  contentH,
  primaryButtonPans = false,
  applyMode = 'transform',
  fitPadding = 0.98,
  minScale = 0.02,
  maxScale = 32,
}: UseZoomPanOptions) {
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const contentRef = useRef<HTMLElement | null>(null)
  const viewRef = useRef<ZoomPanView>({ scale: 1, tx: 0, ty: 0 })
  const interactedRef = useRef(false)
  const spaceRef = useRef(false)
  const panRef = useRef<{ x: number; y: number } | null>(null)
  // Whether the current pan gesture involved any dragging (distinguishes
  // "clicked the backdrop to close" from "dragged then released" -- used by
  // the viewer modal)
  const draggedRef = useRef(false)
  const [zoomPct, setZoomPct] = useState(100)

  const applyToDom = useCallback(() => {
    const v = viewRef.current
    const el = contentRef.current
    if (!el) return
    if (applyMode === 'size') {
      el.style.width = `${contentW * v.scale}px`
      el.style.height = `${contentH * v.scale}px`
      el.style.transform = `translate(${v.tx}px, ${v.ty}px)`
    } else {
      el.style.transform = `translate(${v.tx}px, ${v.ty}px) scale(${v.scale})`
    }
  }, [applyMode, contentW, contentH])

  const applyView = useCallback(() => {
    applyToDom()
    setZoomPct(Math.round(viewRef.current.scale * 100))
  }, [applyToDom])

  const fit = useCallback(() => {
    const wrap = wrapRef.current
    if (!wrap || contentW <= 0 || contentH <= 0) return
    const rect = wrap.getBoundingClientRect()
    if (rect.width < 10 || rect.height < 10) return
    const scale = Math.min(rect.width / contentW, rect.height / contentH) * fitPadding
    viewRef.current = {
      scale,
      tx: (rect.width - contentW * scale) / 2,
      ty: (rect.height - contentH * scale) / 2,
    }
    interactedRef.current = false
    applyView()
  }, [contentW, contentH, fitPadding, applyView])

  /** Returns to 100%, anchored on the viewport center. */
  const reset100 = useCallback(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const rect = wrap.getBoundingClientRect()
    const v = viewRef.current
    const cx = rect.width / 2
    const cy = rect.height / 2
    viewRef.current = {
      scale: 1,
      tx: cx - (cx - v.tx) / v.scale,
      ty: cy - (cy - v.ty) / v.scale,
    }
    interactedRef.current = true
    applyView()
  }, [applyView])

  /** Screen coordinates -> content coordinates (native pixels). Returns null if the container isn't mounted. */
  const toContentPoint = useCallback((clientX: number, clientY: number) => {
    const wrap = wrapRef.current
    if (!wrap) return null
    const rect = wrap.getBoundingClientRect()
    const v = viewRef.current
    return {
      x: (clientX - rect.left - v.tx) / v.scale,
      y: (clientY - rect.top - v.ty) / v.scale,
    }
  }, [])

  // Auto-fit when switching images / on initial readiness. Layout effect: the new
  // transform lands before the browser paints, so the image never shows a frame at the old scale.
  useLayoutEffect(() => {
    fit()
  }, [fit])

  // Container resize: stays fit as long as the user hasn't manually touched the view
  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const ro = new ResizeObserver(() => {
      if (!interactedRef.current) fit()
    })
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [fit])

  // wheel zoom -- React's onWheel is passive, so this must be attached manually as non-passive to allow preventDefault
  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = wrap.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      const v = viewRef.current
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
      const next = Math.min(maxScale, Math.max(minScale, v.scale * factor))
      viewRef.current = {
        scale: next,
        tx: mx - ((mx - v.tx) * next) / v.scale,
        ty: my - ((my - v.ty) * next) / v.scale,
      }
      interactedRef.current = true
      applyView()
    }
    wrap.addEventListener('wheel', onWheel, { passive: false })
    return () => wrap.removeEventListener('wheel', onWheel)
  }, [applyView, minScale, maxScale])

  // Spacebar = the pan modifier key (not hijacked while a form element is focused)
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return
      const el = e.target as HTMLElement | null
      if (
        el &&
        (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
      ) {
        return
      }
      spaceRef.current = true
      e.preventDefault()
    }
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') spaceRef.current = false
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  /** Whether a pointerdown is a pan gesture; if so, claims it (the caller should skip its own logic). */
  const panPointerDown = useCallback((e: React.PointerEvent): boolean => {
    const isPan =
      spaceRef.current ||
      e.button === 1 ||
      (primaryButtonPans && e.button === 0)
    if (!isPan) return false
    panRef.current = { x: e.clientX, y: e.clientY }
    draggedRef.current = false
    return true
  }, [primaryButtonPans])

  /** Movement while a pan is in progress; returns whether this event was consumed. */
  const panPointerMove = useCallback((e: React.PointerEvent): boolean => {
    if (!panRef.current) return false
    const dx = e.clientX - panRef.current.x
    const dy = e.clientY - panRef.current.y
    if (dx !== 0 || dy !== 0) draggedRef.current = true
    panRef.current = { x: e.clientX, y: e.clientY }
    viewRef.current = {
      ...viewRef.current,
      tx: viewRef.current.tx + dx,
      ty: viewRef.current.ty + dy,
    }
    interactedRef.current = true
    applyToDom()
    return true
  }, [applyToDom])

  /** Ends the pan; returns whether this gesture involved any dragging (the modal uses this to decide whether to treat it as a click-to-close). */
  const endPan = useCallback((): boolean => {
    panRef.current = null
    const dragged = draggedRef.current
    draggedRef.current = false
    return dragged
  }, [])

  /** All-in-one binding for viewers (don't use for brush-style tools -- compose panPointerDown/Move/endPan yourself instead). */
  const handlers = {
    onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => {
      if (panPointerDown(e)) e.currentTarget.setPointerCapture(e.pointerId)
    },
    onPointerMove: (e: React.PointerEvent<HTMLDivElement>) => {
      panPointerMove(e)
    },
    onPointerUp: () => {
      endPan()
    },
    onPointerCancel: () => {
      endPan()
    },
  }

  return {
    wrapRef,
    contentRef,
    viewRef,
    zoomPct,
    fit,
    reset100,
    toContentPoint,
    interactedRef,
    spaceRef,
    panPointerDown,
    panPointerMove,
    endPan,
    handlers,
  }
}
