import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'
import ZoomableImage from '../../../components/ZoomableImage'
import { useDialog } from '../../../components/Dialog'
import { graphApi } from './api'
import { parseNumber } from './model'
import type { GImage, Run } from './types'
import { Ico, Stars } from './ui'

/** Full-size viewer (whole image, zoom / pan) with the sample's metadata. */
export default function ImageDialog({ run, imageId, onClose, onUpdate, onDelete, onBestStep }: {
  run: Run
  imageId: number
  onClose: () => void
  onUpdate: (img: GImage, patch: Partial<GImage>) => void
  onDelete: (img: GImage) => void
  onBestStep: (step: number | null) => void
}) {
  const { t } = useTranslation()
  const dialog = useDialog()
  const list = [...run.images].sort((a, b) => (a.step ?? 1e12) - (b.step ?? 1e12) || a.sort - b.sort)
  const [id, setId] = useState(imageId)
  const idx = Math.max(0, list.findIndex((i) => i.id === id))
  const img = list[idx]

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return
      if (e.key === 'Escape') { e.stopPropagation(); onClose() }
      if (e.key === 'ArrowRight' && idx < list.length - 1) setId(list[idx + 1].id)
      if (e.key === 'ArrowLeft' && idx > 0) setId(list[idx - 1].id)
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [idx, list, onClose])

  useEffect(() => { if (!img) onClose() }, [img, onClose])
  if (!img) return null

  return createPortal(
    <div className="gr-viewer ds-backdrop-anim" role="dialog" aria-modal="true" aria-label={t('graph.sample')}>
      <div className="gr-viewer-main">
        <ZoomableImage key={img.id} src={graphApi.imageUrl(img.id)} alt={img.prompt} />
        {idx > 0 && <button type="button" className="gr-viewer-nav gr-prev" onClick={() => setId(list[idx - 1].id)} aria-label={t('graph.prev')}>‹</button>}
        {idx < list.length - 1 && <button type="button" className="gr-viewer-nav gr-next" onClick={() => setId(list[idx + 1].id)} aria-label={t('graph.next')}>›</button>}
      </div>
      <aside className="gr-viewer-side">
        <div className="gr-drawer-head">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="ds-card-title">{run.name || t('graph.untitled')}</div>
            <div className="ds-cell-key">{idx + 1} / {list.length}{img.width ? `, ${img.width}×${img.height}` : ''}</div>
          </div>
          <button type="button" className="ds-kebab" onClick={onClose} aria-label={t('common.close')}>{Ico.close}</button>
        </div>
        <div className="gr-drawer-body" key={img.id}>
          <Field label={t('graph.step')} hint={t('graph.stepHint')}>
            <MetaInput value={img.step == null ? '' : String(img.step)} mono
              onSave={(s) => { const n = parseNumber(s); onUpdate(img, { step: n == null ? null : Math.round(n) }) }} />
          </Field>
          <Field label={t('graph.prompt')}>
            <MetaInput value={img.prompt} area onSave={(s) => onUpdate(img, { prompt: s })} />
          </Field>
          <Field label={t('graph.seed')}>
            <MetaInput value={img.seed == null ? '' : String(img.seed)} mono
              onSave={(s) => { const n = parseNumber(s); onUpdate(img, { seed: n == null ? null : Math.round(n) }) }} />
          </Field>
          <Field label={t('graph.comment')}>
            <MetaInput value={img.comment} area onSave={(s) => onUpdate(img, { comment: s })} />
          </Field>
          <Field label={t('graph.rating')}>
            <Stars value={img.rating} onChange={(n) => onUpdate(img, { rating: n })} size={17} />
          </Field>
          <div className="gr-drawer-row" style={{ marginTop: 6 }}>
            {img.step != null && (
              run.best_step === img.step
                ? <button type="button" className="ds-ctl" onClick={() => onBestStep(null)}>{t('graph.unsetBestStep')}</button>
                : <button type="button" className="ds-ctl" onClick={() => onBestStep(img.step)}>{t('graph.setBestStep')}</button>
            )}
            <a className="ds-ctl ds-ghost" href={graphApi.imageUrl(img.id)} download>{Ico.download}{t('common.download')}</a>
            <button type="button" className="ds-btn-danger" style={{ marginLeft: 'auto' }}
              onClick={async () => {
                if (await dialog.confirm(t('graph.deleteSampleConfirm'), { tone: 'danger' })) onDelete(img)
              }}>{t('common.delete')}</button>
          </div>
          {img.source && <div className="ds-cell-key" style={{ marginTop: 10, wordBreak: 'break-all' }}>{t('graph.source')}: {img.source}</div>}
        </div>
      </aside>
    </div>,
    document.body,
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="gr-mfield">
      <span className="gr-pfield-name" title={hint}>{label}</span>
      {children}
    </label>
  )
}

function MetaInput({ value, onSave, area, mono }: { value: string; onSave: (s: string) => void; area?: boolean; mono?: boolean }) {
  const [v, setV] = useState(value)
  useEffect(() => setV(value), [value])
  const common = {
    value: v,
    onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => setV(e.target.value),
    onBlur: () => { if (v !== value) onSave(v) },
  }
  return area
    ? <textarea className="ds-inp gr-notes" rows={3} {...common} />
    : <input className={`ds-inp${mono ? ' ds-mono' : ''}`} {...common}
      onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }} />
}
