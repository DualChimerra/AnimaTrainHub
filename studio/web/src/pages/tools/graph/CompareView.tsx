import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'
import { graphApi } from './api'
import { activeValues, paramById, varyingParams } from './model'
import type { GImage, Param, Run } from './types'
import { Ico, Stars, Switch, useValueLabel } from './ui'
import HelpTip from '../../../components/ds/HelpTip'

/** Zoom state in image fractions, so images of different sizes show the
 *  same region when synced. z=1 fits the whole image. */
interface ZView { z: number; cx: number; cy: number }
const FIT: ZView = { z: 1, cx: 0.5, cy: 0.5 }

/** Sample size presets: cell height / minimum column width. */
const SIZES = { s: { h: 200, w: 150 }, m: { h: 300, w: 210 }, l: { h: 480, w: 320 } } as const

type StepChoice = number | 'all' | 'best'
const ANY = '__any__'

export default function CompareView({ runs: initial, params, onClose, onOpenRun, onOpenImage }: {
  runs: Run[]
  params: Param[]
  onClose: () => void
  onOpenRun: (r: Run) => void
  onOpenImage: (run: Run, img: GImage) => void
}) {
  const { t } = useTranslation()
  const label = useValueLabel()
  const byId = useMemo(() => paramById(params), [params])
  const [ids, setIds] = useState(() => initial.map((r) => r.id))
  const runs = ids.map((id) => initial.find((r) => r.id === id)).filter((r): r is Run => !!r)

  const diff = useMemo(() => varyingParams(runs.map((r) => r.values), params), [runs, params])
  const common = useMemo(() => {
    if (!runs.length) return []
    const a = activeValues(runs[0].values, params)
    return params.filter((p) => !p.archived && !diff.includes(p.id) && a[p.id] !== undefined)
  }, [runs, params, diff])

  const allImgs = runs.flatMap((r) => r.images)
  const steps = [...new Set(allImgs.map((i) => i.step).filter((s): s is number => s != null))].sort((a, b) => a - b)
  const prompts = [...new Set(allImgs.map((i) => i.prompt).filter(Boolean))]
  const seeds = [...new Set(allImgs.map((i) => i.seed).filter((s): s is number => s != null))].sort((a, b) => a - b)

  // Default: the last step every run has; otherwise all steps as rows.
  const [step, setStep] = useState<StepChoice>(() => {
    const shared = steps.filter((s) => runs.every((r) => r.images.some((i) => i.step === s)))
    return shared.length ? shared[shared.length - 1] : steps.length ? 'all' : 'best'
  })
  const [prompt, setPrompt] = useState<string>(ANY)
  const [seed, setSeed] = useState<string>(ANY)
  const [sync, setSync] = useState(true)
  const [shared, setShared] = useState<ZView>(FIT)
  const [own, setOwn] = useState<Record<string, ZView>>({})
  const [showCommon, setShowCommon] = useState(false)
  const [size, setSize] = useState<keyof typeof SIZES>(() => {
    try { const v = localStorage.getItem('studio:graph:cmpSize') as keyof typeof SIZES | null; return v && v in SIZES ? v : 'm' } catch { return 'm' }
  })
  const pickSize = (v: keyof typeof SIZES) => { setSize(v); try { localStorage.setItem('studio:graph:cmpSize', v) } catch { /* private mode */ } }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !document.querySelector('.gr-viewer')) { e.stopPropagation(); onClose() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const pick = (r: Run, s: number | null | 'best'): GImage[] => r.images.filter((i) => {
    if (s === 'best') {
      if (r.best_step != null) { if (i.step !== r.best_step) return false }
    } else if (i.step !== s) return false
    if (prompt !== ANY && i.prompt !== prompt) return false
    if (seed !== ANY && String(i.seed) !== seed) return false
    return true
  })

  const hasNoStep = allImgs.some((i) => i.step == null)
  const rows: (number | null | 'best')[] = step === 'all' ? [...steps, ...(hasNoStep ? [null] : [])] : step === 'best' ? ['best'] : [step]

  const viewFor = (key: string) => (sync ? shared : own[key] ?? FIT)
  const setViewFor = (key: string, v: ZView) => (sync ? setShared(v) : setOwn((m) => ({ ...m, [key]: v })))

  const cols = `repeat(${Math.max(1, runs.length)}, minmax(${SIZES[size].w}px, ${SIZES[size].w * 1.6}px))`

  return createPortal(
    <div className="gr-compare ds-backdrop-anim" role="dialog" aria-modal="true" aria-label={t('graph.compareTitle')}>
      <div className="gr-compare-head">
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="ds-modal-title">{t('graph.compareTitle')} ({runs.length})<HelpTip>{t('graph.compareSub')}</HelpTip></div>
        </div>
        <label className="gr-axis"><span className="ds-cap">{t('graph.step')}</span>
          <select className="ds-inp gr-axis-sel" value={String(step)} aria-label={t('graph.step')}
            onChange={(e) => { const v = e.target.value; setStep(v === 'all' || v === 'best' ? v : Number(v)) }}>
            <option value="all">{t('graph.allSteps')}</option>
            <option value="best">{t('graph.bestOfEach')}</option>
            {steps.map((s) => <option key={s} value={s}>{t('graph.stepN', { n: s })}</option>)}
          </select>
        </label>
        <label className="gr-axis"><span className="ds-cap">{t('graph.prompt')}</span>
          <select className="ds-inp gr-axis-sel" value={prompt} onChange={(e) => setPrompt(e.target.value)} aria-label={t('graph.prompt')}>
            <option value={ANY}>{t('graph.any')}</option>
            {prompts.map((p) => <option key={p} value={p}>{p.length > 48 ? `${p.slice(0, 48)}…` : p}</option>)}
          </select>
        </label>
        <label className="gr-axis"><span className="ds-cap">{t('graph.seed')}</span>
          <select className="ds-inp gr-axis-sel" value={seed} onChange={(e) => setSeed(e.target.value)} aria-label={t('graph.seed')}>
            <option value={ANY}>{t('graph.any')}</option>
            {seeds.map((s) => <option key={s} value={String(s)}>{s}</option>)}
          </select>
        </label>
        <div className="ds-seg" role="group" aria-label={t('graph.sampleSize')}>
          {(Object.keys(SIZES) as (keyof typeof SIZES)[]).map((k) => (
            <button key={k} type="button" className={`ds-seg-item${size === k ? ' ds-is-active' : ''}`} aria-pressed={size === k} onClick={() => pickSize(k)}>{t(`graph.size_${k}`)}</button>
          ))}
        </div>
        <Switch on={sync} onChange={(v) => { setSync(v); setShared(FIT); setOwn({}) }} label={t('graph.syncZoom')} />
        <button type="button" className="ds-ctl" onClick={() => { setShared(FIT); setOwn({}) }}>{t('graph.fit')}</button>
        <button type="button" className="ds-kebab" onClick={onClose} aria-label={t('common.close')}>{Ico.close}</button>
      </div>

      <div className="gr-compare-body">
        <div className="gr-common">
          <button type="button" className="gr-link" onClick={() => setShowCommon((v) => !v)}>
            {showCommon ? Ico.chevD : Ico.chev} {t('graph.commonParams', { count: common.length })}
          </button>
          {showCommon && (
            <div className="gr-common-list">
              {common.map((p) => (
                <span key={p.id} className="gr-kv-row"><span className="gr-kv-k">{p.name}</span><span className="gr-kv-v">{label(p, runs[0].values[p.id])}</span></span>
              ))}
            </div>
          )}
        </div>

        <div className="gr-cmp-grid" style={{ gridTemplateColumns: step === 'all' ? `64px ${cols}` : cols, ['--cmp-h' as string]: `${SIZES[size].h}px` }}>
          {step === 'all' && <div />}
          {runs.map((r) => (
            <div key={r.id} className="gr-cmp-col-head">
              <div className="gr-cmp-name">
                <button type="button" className="gr-card-name" onClick={() => onOpenRun(r)} title={r.name}>{r.name || t('graph.untitled')}</button>
                <Stars value={r.rating} size={12} />
                {runs.length > 2 && (
                  <button type="button" className="ds-kebab" onClick={() => setIds((l) => l.filter((x) => x !== r.id))} aria-label={t('graph.removeFromCompare')} title={t('graph.removeFromCompare')}>{Ico.close}</button>
                )}
              </div>
              <div className="gr-kv">
                {diff.map((pid) => {
                  const p = byId.get(pid)!
                  const vals = activeValues(r.values, params)
                  return (
                    <span key={pid} className="gr-kv-row gr-diff">
                      <span className="gr-kv-k">{p.name}</span>
                      <span className="gr-kv-v">{vals[pid] === undefined ? <i>{t('graph.unset')}</i> : label(p, vals[pid])}</span>
                    </span>
                  )
                })}
              </div>
            </div>
          ))}

          {rows.map((s) => (
            <RowCells key={String(s)} s={s} runs={runs} withLabel={step === 'all'} pick={pick}
              viewFor={viewFor} setViewFor={setViewFor} onOpenImage={onOpenImage} />
          ))}
        </div>
        {!allImgs.length && <div className="ds-empty" style={{ marginTop: 16 }}>{t('graph.compareNoImages')}</div>}
      </div>
    </div>,
    document.body,
  )
}

function RowCells({ s, runs, withLabel, pick, viewFor, setViewFor, onOpenImage }: {
  s: number | null | 'best'
  runs: Run[]
  withLabel: boolean
  pick: (r: Run, s: number | null | 'best') => GImage[]
  viewFor: (key: string) => ZView
  setViewFor: (key: string, v: ZView) => void
  onOpenImage: (run: Run, img: GImage) => void
}) {
  const { t } = useTranslation()
  return (
    <>
      {withLabel && <div className="gr-cmp-rowlabel">{s === 'best' ? t('graph.bestStep') : s == null ? t('graph.noStep') : t('graph.stepN', { n: s })}</div>}
      {runs.map((r) => {
        const imgs = pick(r, s)
        if (s === 'best') imgs.sort((a, b) => (b.rating ?? 0) - (a.rating ?? 0) || (b.step ?? -1) - (a.step ?? -1))
        const key = `${r.id}:${String(s)}`
        const img = imgs[0]
        return (
          <div key={key} className="gr-cmp-cell">
            {img ? (
              <>
                <SyncImage img={img} view={viewFor(key)} onView={(v) => setViewFor(key, v)} />
                <div className="gr-cmp-cap">
                  {img.step != null && <span>{t('graph.stepN', { n: img.step })}</span>}
                  {img.seed != null && <span>seed {img.seed}</span>}
                  {imgs.length > 1 && <span title={t('graph.moreMatches')}>+{imgs.length - 1}</span>}
                  <button type="button" className="gr-cmp-open" onClick={() => onOpenImage(r, img)} title={t('graph.openFull')} aria-label={t('graph.openFull')}>⤢</button>
                </div>
              </>
            ) : (
              <div className="gr-cmp-missing">
                {s === 'best' || s == null ? t('graph.noSample') : t('graph.noSampleAt', { n: s })}
              </div>
            )}
          </div>
        )
      })}
    </>
  )
}

function SyncImage({ img, view, onView }: { img: GImage; view: ZView; onView: (v: ZView) => void }) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [box, setBox] = useState({ w: 0, h: 0 })
  const [nat, setNat] = useState<{ w: number; h: number } | null>(
    img.width && img.height ? { w: img.width, h: img.height } : null,
  )
  const viewRef = useRef(view)
  viewRef.current = view
  const onViewRef = useRef(onView)
  onViewRef.current = onView
  const drag = useRef<{ x: number; y: number; v: ZView } | null>(null)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    setBox({ w: el.clientWidth, h: el.clientHeight })
    return () => ro.disconnect()
  }, [])

  const s0 = nat && box.w ? Math.min(box.w / nat.w, box.h / nat.h) : 0
  useEffect(() => {
    const el = ref.current
    if (!el || !nat || !s0) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const v = viewRef.current
      const r = el.getBoundingClientRect()
      const mx = e.clientX - r.left - r.width / 2
      const my = e.clientY - r.top - r.height / 2
      const scale = s0 * v.z
      const fx = v.cx + mx / (nat.w * scale)
      const fy = v.cy + my / (nat.h * scale)
      const z = Math.max(1, Math.min(24, v.z * (e.deltaY < 0 ? 1.18 : 1 / 1.18)))
      const ns = s0 * z
      const next = z === 1 ? FIT : { z, cx: fx - mx / (nat.w * ns), cy: fy - my / (nat.h * ns) }
      // Several wheel ticks can land before React re-renders; build on the
      // latest value, not the one from the last render.
      viewRef.current = next
      onViewRef.current(next)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [nat, s0])

  const scale = s0 * view.z
  const style: React.CSSProperties = nat && s0 ? {
    position: 'absolute',
    width: nat.w * scale,
    height: nat.h * scale,
    left: box.w / 2 - view.cx * nat.w * scale,
    top: box.h / 2 - view.cy * nat.h * scale,
    maxWidth: 'none',
  } : { width: '100%', height: '100%', objectFit: 'contain' }
  // Thumbnail while fitted, the original once zoomed in.
  const src = view.z > 1.3 ? graphApi.imageUrl(img.id) : graphApi.imageUrl(img.id, 1024)

  return (
    <div
      ref={ref} className="gr-sync" style={{ cursor: view.z > 1 ? 'grab' : 'zoom-in' }}
      onPointerDown={(e) => { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); drag.current = { x: e.clientX, y: e.clientY, v: viewRef.current } }}
      onPointerMove={(e) => {
        const d = drag.current
        if (!d || !nat || !s0 || d.v.z <= 1) return
        const sc = s0 * d.v.z
        onView({ ...d.v, cx: d.v.cx - (e.clientX - d.x) / (nat.w * sc), cy: d.v.cy - (e.clientY - d.y) / (nat.h * sc) })
      }}
      onPointerUp={() => { drag.current = null }}
      onDoubleClick={() => onView(view.z > 1 ? FIT : { z: 3, cx: 0.5, cy: 0.5 })}
    >
      <img src={src} alt={img.prompt} draggable={false} decoding="async" style={style}
        onLoad={(e) => { if (!nat) setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight }) }} />
    </div>
  )
}
