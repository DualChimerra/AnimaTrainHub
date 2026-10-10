import { useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api } from '../../../api/client'
import { useToast } from '../../../components/Toast'

export interface GenerateReference {
  id: string
  width: number
  height: number
}

/** Edit LoRA source image: the picture the LoRA edits (cosmos temporal reference). */
export default function ReferenceImageInput({ value, onChange, matchSize, onMatchSize, hint }: {
  value: GenerateReference | null
  onChange: (ref: GenerateReference | null) => void
  /** Only the first input decides the output size. */
  matchSize?: boolean
  onMatchSize?: (v: boolean) => void
  hint?: string
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const input = useRef<HTMLInputElement | null>(null)
  const [busy, setBusy] = useState(false)
  const [over, setOver] = useState(false)

  const upload = async (file: File | undefined) => {
    if (!file) return
    setBusy(true)
    try {
      onChange(await api.uploadGenerateReference(file))
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'stretch' }}>
        <button
          type="button"
          className={`ds-thumb${over ? ' ds-sel' : ''}`}
          style={{ width: 84, height: 84, flex: 'none', cursor: 'pointer' }}
          onClick={() => input.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true) }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); void upload(e.dataTransfer.files[0]) }}
          title={t('generate.refPick')}
        >
          {value
            ? <img src={api.generateReferenceUrl(value.id)} alt="" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            : <span>{busy ? '…' : t('generate.refEmpty')}</span>}
        </button>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0, fontSize: 12 }}>
          <span className="ds-kpi-meta">{hint ?? t('generate.refHint')}</span>
          {value && <span className="ds-cell-key">{value.width}×{value.height}</span>}
          {onMatchSize && (
            <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={!!matchSize} onChange={(e) => onMatchSize(e.target.checked)} />
              {t('generate.refMatchSize')}
            </label>
          )}
          {value && (
            <button type="button" className="ds-ctl" style={{ height: 26, alignSelf: 'flex-start' }} onClick={() => onChange(null)}>
              {t('generate.refClear')}
            </button>
          )}
        </div>
      </div>
      <input ref={input} type="file" hidden accept="image/*"
        onChange={(e) => { void upload(e.target.files?.[0]); e.target.value = '' }} />
    </div>
  )
}
