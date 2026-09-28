import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { coverImage, differences, isActive, knownValues, paramById, sameValue, visibleParams } from './model'
import type { Param, Run, ViewState } from './types'
import { graphApi } from './api'
import { THUMB_PX, useValueLabel } from './ui'

interface Node { id: number; x: number; y: number; vx: number; vy: number; run: Run }
interface Edge { a: number; b: number; param: string }

const PALETTE = ['#7fb33a', '#4f7fc9', '#d08a2e', '#b3402f', '#8a5cc7', '#2f9c8f', '#c7577f', '#6b726e', '#a8a23a', '#3f5f8f']
const MAX_NODES = 400

/** Deterministic pseudo-random so the layout does not jump between visits. */
function rand(seed: number) {
  const x = Math.sin(seed * 9301 + 49297) * 233280
  return x - Math.floor(x)
}

function layout(runs: Run[], edges: Edge[]): Map<number, Node> {
  const nodes = new Map<number, Node>()
  const n = runs.length
  runs.forEach((r, i) => {
    const a = (i / Math.max(1, n)) * Math.PI * 2
    const rad = 200 + 70 * Math.sqrt(n)
    nodes.set(r.id, { id: r.id, run: r, x: Math.cos(a) * rad + rand(r.id) * 20, y: Math.sin(a) * rad + rand(r.id + 7) * 20, vx: 0, vy: 0 })
  })
  const list = [...nodes.values()]
  const ticks = n > 200 ? 160 : 260
  for (let t = 0; t < ticks; t++) {
    const k = 1 - t / ticks
    for (let i = 0; i < list.length; i++) {
      const p = list[i]
      for (let j = i + 1; j < list.length; j++) {
        const q = list[j]
        let dx = p.x - q.x
        let dy = p.y - q.y
        let d2 = dx * dx + dy * dy
        if (d2 < 1) { dx = rand(i + j) - 0.5; dy = rand(i * j + 3) - 0.5; d2 = 1 }
        const f = 14000 / d2
        const d = Math.sqrt(d2)
        p.vx += (dx / d) * f; p.vy += (dy / d) * f
        q.vx -= (dx / d) * f; q.vy -= (dy / d) * f
      }
    }
    for (const e of edges) {
      const p = nodes.get(e.a)!
      const q = nodes.get(e.b)!
      const dx = q.x - p.x
      const dy = q.y - p.y
      const d = Math.max(1, Math.sqrt(dx * dx + dy * dy))
      const f = (d - 170) * 0.04
      p.vx += (dx / d) * f; p.vy += (dy / d) * f
      q.vx -= (dx / d) * f; q.vy -= (dy / d) * f
    }
    for (const p of list) {
      p.vx -= p.x * 0.004
      p.vy -= p.y * 0.004
      p.x += Math.max(-30, Math.min(30, p.vx)) * k
      p.y += Math.max(-30, Math.min(30, p.vy)) * k
      p.vx *= 0.5; p.vy *= 0.5
    }
  }
  return nodes
}

/**
 * The web: every card is a dot, and a line joins two cards that differ in
 * exactly one parameter. Chains of lines are single-parameter sweeps; islands
 * are cards nothing was compared against yet.
 */
export default function WebView({ runs, params, view, setView, onOpen }: {
  runs: Run[]
  params: Param[]
  view: ViewState
  setView: (fn: (v: ViewState) => ViewState) => void
  onOpen: (run: Run) => void
}) {
  const { t } = useTranslation()
  const label = useValueLabel()
  const byId = useMemo(() => paramById(params), [params])
  const shown = runs.slice(0, MAX_NODES)
  const sig = shown.map((r) => `${r.id}:${JSON.stringify(r.values)}`).join(';')

  const edges = useMemo(() => {
    const out: Edge[] = []
    for (let i = 0; i < shown.length; i++) {
      for (let j = i + 1; j < shown.length; j++) {
        const d = differences(shown[i].values, shown[j].values, params)
        if (d.length === 1) out.push({ a: shown[i].id, b: shown[j].id, param: d[0] })
      }
    }
    return out
    // `sig` captures everything the edges depend on.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig, params])

  const [nodes, setNodes] = useState<Map<number, Node>>(new Map())
  useEffect(() => {
    const laid = layout(shown, edges)
    setNodes(laid)
    // Start zoomed so the whole web fits the canvas.
    const el = wrapRef.current
    const pts = [...laid.values()]
    if (!el || !pts.length) return
    const xs = pts.map((p) => p.x)
    const ys = pts.map((p) => p.y)
    const pad = 110
    const bw = Math.max(...xs) - Math.min(...xs) + pad * 2
    const bh = Math.max(...ys) - Math.min(...ys) + pad * 2
    const k = Math.max(0.35, Math.min(1.2, el.clientWidth / bw, el.clientHeight / bh))
    const cx = (Math.max(...xs) + Math.min(...xs)) / 2
    const cy = (Math.max(...ys) + Math.min(...ys)) / 2
    const fitTf = { k, x: -cx * k, y: -cy * k }
    fitRef.current = fitTf
    setTf(fitTf)
  },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [edges])

  const colorParam = view.webColor ? byId.get(view.webColor) : undefined
  const colorValues = useMemo(() => (colorParam ? knownValues(colorParam, runs) : []), [colorParam, runs])
  const colorOf = (r: Run) => {
    if (!colorParam || !isActive(colorParam, r.values, byId)) return '#c9ccc8'
    const v = r.values[colorParam.id]
    const i = colorValues.findIndex((x) => sameValue(x, v))
    return i < 0 ? '#c9ccc8' : PALETTE[i % PALETTE.length]
  }

  // pan / zoom
  const [tf, setTf] = useState({ x: 0, y: 0, k: 1 })
  const fitRef = useRef({ x: 0, y: 0, k: 1 })
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const drag = useRef<{ kind: 'pan' | 'node'; id?: number; sx: number; sy: number; ox: number; oy: number; moved: boolean } | null>(null)
  const [hover, setHover] = useState<number | null>(null)
  const [size, setSize] = useState({ w: 800, h: 560 })

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const mx = e.clientX - rect.left - rect.width / 2
      const my = e.clientY - rect.top - rect.height / 2
      setTf((cur) => {
        const k = Math.max(0.25, Math.min(3, cur.k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)))
        return { k, x: mx - ((mx - cur.x) * k) / cur.k, y: my - ((my - cur.y) * k) / cur.k }
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => { ro.disconnect(); el.removeEventListener('wheel', onWheel) }
  }, [])

  const onMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    const dx = e.clientX - d.sx
    const dy = e.clientY - d.sy
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true
    if (d.kind === 'pan') setTf((cur) => ({ ...cur, x: d.ox + dx, y: d.oy + dy }))
    else if (d.id != null) {
      setNodes((m) => {
        const next = new Map(m)
        const n = next.get(d.id!)
        if (n) next.set(d.id!, { ...n, x: d.ox + dx / tf.k, y: d.oy + dy / tf.k })
        return next
      })
    }
  }

  const hoverEdges = hover != null ? edges.filter((e) => e.a === hover || e.b === hover) : []
  const linked = new Set(hoverEdges.flatMap((e) => [e.a, e.b]))
  const isolated = shown.filter((r) => !edges.some((e) => e.a === r.id || e.b === r.id)).length
  const R = 36

  return (
    <div className="gr-web">
      {runs.length > MAX_NODES && <div className="ds-note ds-warn"><span>{t('graph.webCapped', { n: MAX_NODES })}</span></div>}
      <div
        ref={wrapRef} className="gr-web-canvas ds-card"
        onPointerDown={(e) => {
          if (e.button !== 0) return
          ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
          drag.current = { kind: 'pan', sx: e.clientX, sy: e.clientY, ox: tf.x, oy: tf.y, moved: false }
        }}
        onPointerMove={onMove}
        onPointerUp={() => {
          const d = drag.current
          drag.current = null
          if (d?.kind === 'node' && !d.moved && d.id != null) {
            const n = nodes.get(d.id)
            if (n) onOpen(n.run)
          }
        }}
      >
        {/* Legend in the top-left corner of the canvas. */}
        <div className="gr-web-legend" onPointerDown={(e) => e.stopPropagation()}>
          <select className="ds-inp gr-web-color" value={view.webColor ?? ''} aria-label={t('graph.webColor')}
            onChange={(e) => setView((v) => ({ ...v, webColor: e.target.value || null }))}>
            <option value="">{t('graph.colorNone')}</option>
            {visibleParams(params).filter((p) => p.type !== 'text').map((p) => <option key={p.id} value={p.id}>{t('graph.colorBy', { name: p.name })}</option>)}
          </select>
          {colorParam && colorValues.map((v, i) => (
            <span key={String(v)} className="gr-web-legend-i"><span className="gr-web-dot" style={{ background: PALETTE[i % PALETTE.length] }} />{label(colorParam, v)}</span>
          ))}
        </div>
        {/* Stats + reset in the bottom-right corner. */}
        <div className="gr-web-corner" onPointerDown={(e) => e.stopPropagation()}>
          <span title={t('graph.webHelp')}>{t('graph.webStats', { nodes: shown.length, edges: edges.length, alone: isolated })}</span>
          <button type="button" className="ds-ctl" onClick={() => setTf(fitRef.current)}>{t('graph.webReset')}</button>
        </div>
        {shown.length === 0 ? <div className="ds-empty" style={{ margin: 24 }}>{t('graph.webEmpty')}</div> : (
          <svg width={size.w} height={size.h} role="img" aria-label={t('graph.viewWeb')}>
            <defs>
              <clipPath id="gr-node-clip"><circle r={R} cx="0" cy="0" /></clipPath>
            </defs>
            <g transform={`translate(${size.w / 2 + tf.x} ${size.h / 2 + tf.y}) scale(${tf.k})`}>
              {edges.map((e, i) => {
                const a = nodes.get(e.a)
                const b = nodes.get(e.b)
                if (!a || !b) return null
                const on = hover != null && (e.a === hover || e.b === hover)
                return (
                  <g key={i}>
                    <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                      stroke={on ? '#4a7429' : '#d4d5d1'} strokeWidth={on ? 2 : 1.2} opacity={hover != null && !on ? 0.35 : 1} />
                    {on && (
                      <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 4} className="gr-web-elabel" textAnchor="middle">
                        {byId.get(e.param)?.name}
                      </text>
                    )}
                  </g>
                )
              })}
              {[...nodes.values()].map((n) => {
                const cover = coverImage(n.run)
                const dim = hover != null && hover !== n.id && !linked.has(n.id)
                return (
                  <g key={n.id} transform={`translate(${n.x} ${n.y})`} opacity={dim ? 0.3 : 1}
                    className="gr-web-node"
                    onPointerDown={(e) => {
                      e.stopPropagation()
                      ;(e.currentTarget.closest('.gr-web-canvas') as HTMLElement | null)?.setPointerCapture(e.pointerId)
                      drag.current = { kind: 'node', id: n.id, sx: e.clientX, sy: e.clientY, ox: n.x, oy: n.y, moved: false }
                    }}
                    onPointerEnter={() => setHover(n.id)}
                    onPointerLeave={() => setHover((h) => (h === n.id ? null : h))}
                  >
                    {/* colored ring (by the chosen parameter) → white outline → picture */}
                    <circle r={R + 5} fill={colorOf(n.run)} />
                    <circle r={R + 1.5} fill="#fff" />
                    {cover
                      ? <image href={graphApi.imageUrl(cover.id, THUMB_PX)} x={-R} y={-R} width={R * 2} height={R * 2} clipPath="url(#gr-node-clip)" preserveAspectRatio="xMidYMid slice" />
                      : <circle r={R} fill="var(--sunken)" />}
                    {hover === n.id && <circle r={R + 5} fill="none" stroke="#16181a" strokeWidth={1.5} />}
                    {n.run.rating ? <text y={-R - 10} textAnchor="middle" className="gr-web-stars">{'★'.repeat(n.run.rating)}</text> : null}
                    {n.run.favorite && <text x={R} y={-R + 8} className="gr-web-fav">♥</text>}
                    <text y={R + 20} textAnchor="middle" className="gr-web-label">{shortName(n.run.name || `#${n.run.id}`)}</text>
                  </g>
                )
              })}
            </g>
          </svg>
        )}
      </div>
    </div>
  )
}

function shortName(s: string): string {
  return s.length > 22 ? `${s.slice(0, 21)}…` : s
}
