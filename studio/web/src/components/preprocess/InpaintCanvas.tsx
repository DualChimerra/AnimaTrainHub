import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react'
import { useTranslation } from 'react-i18next'
import { useZoomPan } from '../../lib/useZoomPan'

/** A single paint / mask stroke. Coordinates / diameter are always in **source-image pixel** units -- view zoom only
 *  affects display; stroke data is zoom-independent, so offscreen replay ("save all") matches what the canvas shows. */
export interface InpaintStroke {
  color: string
  /** Brush diameter (source-image px). */
  size: number
  /** 1 = solid; below 1 the edge is softened by a blur with radius size*(1-hardness)/2. */
  hardness: number
  /** Eraser (destination-out): in paint mode erases unsaved paint strokes, in mask mode erases the mask. */
  erase?: boolean
  points: { x: number; y: number }[]
}

export type InpaintMode = 'paint' | 'mask'

export interface InpaintCanvasHandle {
  /** Composites the current image + all paint strokes and exports a PNG. Returns null if the image hasn't finished loading. */
  exportBlob: () => Promise<Blob | null>
  /** Exports the mask layer as a grayscale PNG (255=learn 0=don't-learn) + coverage.
   *  An empty mask (learn everything) -> null (the caller should DELETE rather than write an all-white file). */
  exportMaskBlob: () => Promise<{ blob: Blob; coverage: number } | null>
}

/** Display color for the mask on the canvas (export only looks at alpha, the color value doesn't matter). */
const MASK_COLOR = '#ff2d2d'
const MASK_VIEW_ALPHA = 0.45

function strokePath(ctx: CanvasRenderingContext2D, s: InpaintStroke, color?: string): void {
  const c = color ?? s.color
  ctx.strokeStyle = c
  ctx.fillStyle = c
  ctx.lineWidth = s.size
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  const pts = s.points
  if (pts.length === 0) return
  if (pts.length === 1) {
    ctx.beginPath()
    ctx.arc(pts[0].x, pts[0].y, s.size / 2, 0, Math.PI * 2)
    ctx.fill()
    return
  }
  ctx.beginPath()
  ctx.moveTo(pts[0].x, pts[0].y)
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
  ctx.stroke()
}

type ScratchRef = { current: HTMLCanvasElement | null }

/** Draws one stroke (soft edges included: composited via a reusable scratch canvas + blur filter). */
function drawOneStroke(
  ctx: CanvasRenderingContext2D,
  s: InpaintStroke,
  color: string,
  holder: ScratchRef,
): void {
  if (s.hardness >= 1 || s.points.length === 0) {
    strokePath(ctx, s, color)
    return
  }
  const w = ctx.canvas.width
  const h = ctx.canvas.height
  if (!holder.current || holder.current.width !== w || holder.current.height !== h) {
    holder.current = document.createElement('canvas')
    holder.current.width = w
    holder.current.height = h
  }
  const sctx = holder.current.getContext('2d')
  if (!sctx) {
    strokePath(ctx, s, color)
    return
  }
  sctx.clearRect(0, 0, w, h)
  strokePath(sctx, s, color)
  const blur = (s.size * (1 - s.hardness)) / 2
  ctx.save()
  ctx.filter = `blur(${blur}px)`
  ctx.drawImage(holder.current, 0, 0)
  ctx.restore()
}

/** Replays strokes onto a separate layer (erase goes through destination-out -- for paint, erasing reveals the
 *  base image under unsaved strokes; for mask, erasing clears the mask -- both share the same semantics). */
function drawStrokesToLayer(
  ctx: CanvasRenderingContext2D,
  strokes: InpaintStroke[],
  scratchRef: ScratchRef,
  colorOf: (s: InpaintStroke) => string,
): void {
  for (const s of strokes) {
    ctx.save()
    ctx.globalCompositeOperation = s.erase ? 'destination-out' : 'source-over'
    drawOneStroke(ctx, s, colorOf(s), scratchRef)
    ctx.restore()
  }
}

function drawPaintStrokes(
  ctx: CanvasRenderingContext2D,
  strokes: InpaintStroke[],
  scratchRef: ScratchRef,
): void {
  drawStrokesToLayer(ctx, strokes, scratchRef, (s) => s.color)
}

function drawMaskStrokes(
  ctx: CanvasRenderingContext2D,
  strokes: InpaintStroke[],
  scratchRef: ScratchRef,
): void {
  drawStrokesToLayer(ctx, strokes, scratchRef, () => MASK_COLOR)
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`image load failed: ${url}`))
    img.src = url
  })
}

/** Server grayscale mask (255=learn 0=don't-learn) -> canvas mask-layer bitmap (red + alpha=don't-learn amount). */
async function loadMaskBase(url: string, w: number, h: number): Promise<HTMLCanvasElement | null> {
  let img: HTMLImageElement
  try {
    img = await loadImage(url)
  } catch {
    return null // 404 = no mask
  }
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d')
  if (!ctx) return null
  ctx.drawImage(img, 0, 0, w, h)
  const data = ctx.getImageData(0, 0, w, h)
  const px = data.data
  for (let i = 0; i < px.length; i += 4) {
    const v = px[i] // grayscale value (R channel)
    px[i] = 255
    px[i + 1] = 45
    px[i + 2] = 45
    px[i + 3] = 255 - v
  }
  ctx.putImageData(data, 0, 0)
  return c
}

/** Rebuilds the mask layer: base image (existing server mask) + local strokes. */
function rebuildMaskLayer(
  layer: HTMLCanvasElement,
  base: HTMLCanvasElement | null,
  strokes: InpaintStroke[],
  scratchRef: ScratchRef,
): void {
  const ctx = layer.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, layer.width, layer.height)
  if (base) ctx.drawImage(base, 0, 0)
  drawMaskStrokes(ctx, strokes, scratchRef)
}

/** mask layer -> grayscale PNG (255=learn 0=don't-learn) + coverage. Fully empty -> null. */
async function maskLayerToGray(
  layer: HTMLCanvasElement,
): Promise<{ blob: Blob; coverage: number } | null> {
  const ctx = layer.getContext('2d')
  if (!ctx) return null
  const data = ctx.getImageData(0, 0, layer.width, layer.height)
  const px = data.data
  let sum = 0
  for (let i = 0; i < px.length; i += 4) {
    const a = px[i + 3]
    sum += a
    const v = 255 - a
    px[i] = v
    px[i + 1] = v
    px[i + 2] = v
    px[i + 3] = 255
  }
  const n = px.length / 4
  const coverage = sum / 255 / n
  if (sum === 0) return null
  const out = document.createElement('canvas')
  out.width = layer.width
  out.height = layer.height
  const octx = out.getContext('2d')
  if (!octx) return null
  octx.putImageData(data, 0, 0)
  const blob = await new Promise<Blob | null>((resolve) => {
    out.toBlob((b) => resolve(b), 'image/png')
  })
  return blob ? { blob, coverage } : null
}

/** Rebuilds and exports the mask offscreen (used by "save all" for the non-active image). null = an empty mask. */
export async function renderMaskBlob(
  maskBaseUrl: string | null,
  w: number,
  h: number,
  strokes: InpaintStroke[],
): Promise<{ blob: Blob; coverage: number } | null> {
  const layer = document.createElement('canvas')
  layer.width = w
  layer.height = h
  const base = maskBaseUrl ? await loadMaskBase(maskBaseUrl, w, h) : null
  rebuildMaskLayer(layer, base, strokes, { current: null })
  return await maskLayerToGray(layer)
}

/** Replays paint offscreen: loads the source image -> replays strokes on a layer (erase included) -> composites a PNG blob. */
export async function renderInpaintedBlob(
  imageUrl: string,
  w: number,
  h: number,
  strokes: InpaintStroke[],
): Promise<Blob> {
  const img = await loadImage(imageUrl)
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas 2d context unavailable')
  ctx.drawImage(img, 0, 0, w, h)
  const layer = document.createElement('canvas')
  layer.width = w
  layer.height = h
  const lctx = layer.getContext('2d')
  if (!lctx) throw new Error('canvas 2d context unavailable')
  drawPaintStrokes(lctx, strokes, { current: null })
  ctx.drawImage(layer, 0, 0)
  return await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('toBlob failed'))),
      'image/png',
    )
  })
}

function toHex(r: number, g: number, b: number): string {
  const c = (v: number) => v.toString(16).padStart(2, '0')
  return `#${c(r)}${c(g)}${c(b)}`
}

/** Main paint canvas: a canvas at source-image resolution + the useZoomPan viewport (space / middle-click pan).
 *
 *  Two independently-controlled data planes:
 *  - Paint strokes (strokes): drawn directly over pixels, redraw order is img -> strokes.
 *  - Mask strokes (maskStrokes) + server base image (maskBaseUrl): composited onto a separate
 *    maskLayer (a red-alpha bitmap), overlaid on the main canvas at partial opacity last.
 *
 *  While drawing, a preview segment is painted incrementally on the main canvas (a mask eraser shows as translucent
 *  white); pointerup commits it and a prop change then triggers a full redraw to correct it.
 */
const InpaintCanvas = forwardRef<
  InpaintCanvasHandle,
  {
    imageUrl: string
    imageW: number
    imageH: number
    mode: InpaintMode
    strokes: InpaintStroke[]
    maskStrokes: InpaintStroke[]
    /** URL of the server's existing mask; null = no base image. */
    maskBaseUrl: string | null
    brush: { color: string; size: number; hardness: number }
    /** Whether the current tool is the eraser (shared by both paint / mask modes). */
    erase: boolean
    onStrokeEnd: (s: InpaintStroke) => void
    onMaskStrokeEnd: (s: InpaintStroke) => void
    onPickColor: (hex: string) => void
  }
>(function InpaintCanvas(
  {
    imageUrl, imageW, imageH, mode, strokes, maskStrokes, maskBaseUrl,
    brush, erase, onStrokeEnd, onMaskStrokeEnd, onPickColor,
  },
  ref,
) {
  const { t } = useTranslation()
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const cursorRef = useRef<HTMLDivElement | null>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const scratchRef = useRef<HTMLCanvasElement | null>(null)
  const paintLayerRef = useRef<HTMLCanvasElement | null>(null)
  const maskLayerRef = useRef<HTMLCanvasElement | null>(null)
  const maskBaseRef = useRef<HTMLCanvasElement | null>(null)

  // The viewport (zoom / pan / fit / coordinate conversion) goes through the shared hook; a brush-tool scene leaves
  // the left click to the brush (primaryButtonPans defaults to false), pan is triggered by space / middle-click instead.
  const zp = useZoomPan({ contentW: imageW, contentH: imageH })
  const { wrapRef, contentRef, toContentPoint } = zp

  // Locks ownership (paint / mask) at pen-down; commits on release using this -- never guesses mode after the fact
  const drawingRef = useRef<{ stroke: InpaintStroke; target: InpaintMode } | null>(null)

  const [loaded, setLoaded] = useState(false)
  const [maskBaseTick, setMaskBaseTick] = useState(0)
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null)

  const strokesRef = useRef(strokes)
  strokesRef.current = strokes
  const maskStrokesRef = useRef(maskStrokes)
  maskStrokesRef.current = maskStrokes
  const brushRef = useRef(brush)
  brushRef.current = brush
  const modeRef = useRef(mode)
  modeRef.current = mode
  const eraseRef = useRef(erase)
  eraseRef.current = erase

  const ensureLayer = useCallback((
    holder: React.MutableRefObject<HTMLCanvasElement | null>,
  ): HTMLCanvasElement => {
    if (
      !holder.current ||
      holder.current.width !== imageW ||
      holder.current.height !== imageH
    ) {
      const c = document.createElement('canvas')
      c.width = imageW
      c.height = imageH
      holder.current = c
    }
    return holder.current
  }, [imageW, imageH])

  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    const paintLayer = paintLayerRef.current
    if (paintLayer) ctx.drawImage(paintLayer, 0, 0)
    const layer = maskLayerRef.current
    if (layer) {
      ctx.save()
      ctx.globalAlpha = MASK_VIEW_ALPHA
      ctx.drawImage(layer, 0, 0)
      ctx.restore()
    }
  }, [])

  // Image loading (imageUrl change = a new image or an mtime refresh after saving)
  useEffect(() => {
    let cancelled = false
    setLoaded(false)
    imgRef.current = null
    loadImage(imageUrl).then(
      (img) => {
        if (cancelled) return
        imgRef.current = img
        setLoaded(true)
        zp.fit()
        redraw()
      },
      () => {
        /* Leave the canvas blank on failure; switching images in the filmstrip can recover */
      },
    )
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageUrl, zp.fit, redraw])

  // Mask base-image loading (URL change = new image / refresh after saving / local clear -> null)
  useEffect(() => {
    let cancelled = false
    maskBaseRef.current = null
    if (!maskBaseUrl) {
      setMaskBaseTick((v) => v + 1)
      return
    }
    void loadMaskBase(maskBaseUrl, imageW, imageH).then((base) => {
      if (cancelled) return
      maskBaseRef.current = base
      setMaskBaseTick((v) => v + 1)
    })
    return () => {
      cancelled = true
    }
  }, [maskBaseUrl, imageW, imageH])

  // Mask layer rebuild (base image / strokes changed) -> redraw the main canvas
  useEffect(() => {
    const layer = ensureLayer(maskLayerRef)
    rebuildMaskLayer(layer, maskBaseRef.current, maskStrokes, scratchRef)
    redraw()
  }, [maskStrokes, maskBaseTick, ensureLayer, redraw])

  // Paint layer rebuild (undo / redo / stroke commit / clear -- including the eraser composite)
  useEffect(() => {
    const layer = ensureLayer(paintLayerRef)
    const ctx = layer.getContext('2d')
    if (ctx) {
      ctx.clearRect(0, 0, layer.width, layer.height)
      drawPaintStrokes(ctx, strokes, scratchRef)
    }
    redraw()
  }, [strokes, ensureLayer, redraw])

  useImperativeHandle(ref, () => ({
    exportBlob: async () => {
      const canvas = canvasRef.current
      const img = imgRef.current
      if (!canvas || !img) return null
      // Export excludes the mask overlay: cleanly rebuilds img + paint layer (including the eraser composite)
      const out = document.createElement('canvas')
      out.width = canvas.width
      out.height = canvas.height
      const ctx = out.getContext('2d')
      if (!ctx) return null
      ctx.drawImage(img, 0, 0, out.width, out.height)
      const layer = document.createElement('canvas')
      layer.width = out.width
      layer.height = out.height
      const lctx = layer.getContext('2d')
      if (!lctx) return null
      drawPaintStrokes(lctx, strokesRef.current, scratchRef)
      ctx.drawImage(layer, 0, 0)
      return await new Promise<Blob | null>((resolve) => {
        out.toBlob((b) => resolve(b), 'image/png')
      })
    },
    exportMaskBlob: async () => {
      const layer = ensureLayer(maskLayerRef)
      rebuildMaskLayer(layer, maskBaseRef.current, maskStrokesRef.current, scratchRef)
      return await maskLayerToGray(layer)
    },
  }), [ensureLayer])

  // Brush-circle cursor (mutates style on the ref directly; diameter = brush size x current scale)
  const lastPointerRef = useRef<{ x: number; y: number } | null>(null)
  const updateCursor = useCallback((clientX: number, clientY: number) => {
    const cur = cursorRef.current
    const wrap = wrapRef.current
    if (!cur || !wrap) return
    lastPointerRef.current = { x: clientX, y: clientY }
    const rect = wrap.getBoundingClientRect()
    const d = brushRef.current.size * zp.viewRef.current.scale
    cur.style.left = `${clientX - rect.left - d / 2}px`
    cur.style.top = `${clientY - rect.top - d / 2}px`
    cur.style.width = `${d}px`
    cur.style.height = `${d}px`
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // After a wheel zoom, the cursor's diameter must follow scale (the hook doesn't know about the cursor -- driven
  // by the zoomPct change + one extra update using the last pointer position)
  useEffect(() => {
    const p = lastPointerRef.current
    if (p) updateCursor(p.x, p.y)
  }, [zp.zoomPct, updateCursor])

  const pickColor = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current
      const pt = toContentPoint(clientX, clientY)
      if (!canvas || !pt) return
      const x = Math.round(pt.x)
      const y = Math.round(pt.y)
      if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const d = ctx.getImageData(x, y, 1, 1).data
      onPickColor(toHex(d[0], d[1], d[2]))
    },
    [toContentPoint, onPickColor],
  )

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!loaded) return
      e.currentTarget.setPointerCapture(e.pointerId)
      // Pan gestures (space / middle-click) are handled by the viewport hook; this component only handles the brush
      if (zp.panPointerDown(e)) return
      if (e.button !== 0) return
      if (e.altKey && modeRef.current === 'paint') {
        pickColor(e.clientX, e.clientY)
        return
      }
      const pt = toContentPoint(e.clientX, e.clientY)
      if (!pt) return
      const b = brushRef.current
      const isMask = modeRef.current === 'mask'
      const stroke: InpaintStroke = {
        color: isMask ? MASK_COLOR : b.color,
        size: b.size,
        hardness: b.hardness,
        ...(eraseRef.current ? { erase: true } : {}),
        points: [pt],
      }
      drawingRef.current = { stroke, target: isMask ? 'mask' : 'paint' }
      // A single point is visible immediately (a preview; the eraser shows as translucent white, corrected by a full redraw on release)
      const ctx = canvasRef.current?.getContext('2d')
      if (ctx) {
        ctx.save()
        if (stroke.erase) {
          ctx.globalAlpha = 0.5
          strokePath(ctx, stroke, '#ffffff')
        } else if (isMask) {
          ctx.globalAlpha = MASK_VIEW_ALPHA
          strokePath(ctx, stroke, MASK_COLOR)
        } else {
          strokePath(ctx, stroke)
        }
        ctx.restore()
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [loaded, pickColor, toContentPoint, zp.panPointerDown],
  )

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      updateCursor(e.clientX, e.clientY)
      const pt = toContentPoint(e.clientX, e.clientY)
      if (pt) {
        setCursorPos({
          x: Math.max(0, Math.min(imageW, Math.round(pt.x))),
          y: Math.max(0, Math.min(imageH, Math.round(pt.y))),
        })
      }
      if (zp.panPointerMove(e)) return
      const drawing = drawingRef.current
      if (!drawing || !pt) return
      const stroke = drawing.stroke
      const prev = stroke.points[stroke.points.length - 1]
      stroke.points.push(pt)
      const ctx = canvasRef.current?.getContext('2d')
      if (ctx) {
        ctx.save()
        const isMask = drawing.target === 'mask'
        if (stroke.erase) {
          ctx.strokeStyle = '#ffffff'
          ctx.globalAlpha = 0.5
        } else if (isMask) {
          ctx.strokeStyle = MASK_COLOR
          ctx.globalAlpha = MASK_VIEW_ALPHA
        } else {
          ctx.strokeStyle = stroke.color
        }
        ctx.lineWidth = stroke.size
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        ctx.beginPath()
        ctx.moveTo(prev.x, prev.y)
        ctx.lineTo(pt.x, pt.y)
        ctx.stroke()
        ctx.restore()
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [updateCursor, toContentPoint, imageW, imageH, zp.panPointerMove],
  )

  const endStroke = useCallback(() => {
    zp.endPan()
    const drawing = drawingRef.current
    drawingRef.current = null
    if (!drawing) return
    if (drawing.target === 'mask') onMaskStrokeEnd(drawing.stroke)
    else onStrokeEnd(drawing.stroke)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onStrokeEnd, onMaskStrokeEnd, zp.endPan])

  return (
    <div className="flex flex-col h-full min-h-0 gap-1.5">
      <div
        ref={wrapRef}
        className="relative flex-1 min-h-0 overflow-hidden rounded border border-subtle bg-sunken"
        style={{ touchAction: 'none', cursor: 'none' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endStroke}
        onPointerCancel={endStroke}
        onPointerLeave={() => {
          const cur = cursorRef.current
          if (cur) {
            cur.style.width = '0px'
            cur.style.height = '0px'
          }
          setCursorPos(null)
        }}
      >
        <canvas
          ref={(el) => {
            canvasRef.current = el
            contentRef.current = el
          }}
          width={imageW}
          height={imageH}
          style={{ position: 'absolute', left: 0, top: 0, transformOrigin: '0 0' }}
        />
        {/* Brush-circle cursor (red outline for the mask brush / dashed white for the eraser) */}
        <div
          ref={cursorRef}
          className="absolute pointer-events-none rounded-full"
          style={{
            border: mode === 'mask' && !erase
              ? '1.5px solid rgba(255,45,45,0.95)'
              : `1.5px ${erase ? 'dashed' : 'solid'} rgba(255,255,255,0.9)`,
            outline: '1px solid rgba(0,0,0,0.6)',
          }}
        />
        {!loaded && (
          <div className="absolute inset-0 flex items-center justify-center text-fg-tertiary text-sm">
            {t('preprocessInpaint.canvasLoading')}
          </div>
        )}
      </div>

      {/* Readout strip: zoom / view controls / cursor pixel coordinates */}
      <div className="shrink-0 flex items-center gap-2 text-[11px] font-mono text-fg-tertiary px-1">
        <span>{zp.zoomPct}%</span>
        <button
          type="button"
          className="px-1.5 py-0.5 rounded hover:bg-overlay hover:text-fg-primary"
          onClick={() => zp.fit()}
        >{t('preprocessInpaint.zoomFit')}</button>
        <button
          type="button"
          className="px-1.5 py-0.5 rounded hover:bg-overlay hover:text-fg-primary"
          onClick={() => zp.reset100()}
        >100%</button>
        <span className="flex-1" />
        {cursorPos && (
          <span>{cursorPos.x}, {cursorPos.y}</span>
        )}
        <span>{imageW}×{imageH}</span>
        <span className="text-fg-disabled">{t('preprocessInpaint.canvasHint')}</span>
      </div>
    </div>
  )
})

export default InpaintCanvas
