import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type TriggerDetectResult, type Version } from '../api/client'
import { useToast } from './Toast'

/** Version trigger word, edited on the Train page.
 *
 *  The trigger lives on the version row (the old Tagging step used to write it)
 *  and is forced into config.yaml on every save, so it cannot be typed into the
 *  schema form. It is prepended to sample prompts and, more importantly, DOP
 *  cannot run without it — its preservation branch is "the caption with the
 *  trigger removed".
 *
 *  Captions in this studio always carry the trigger already, so the card offers
 *  to read it back from them instead of making the user retype it. */
export default function TriggerWordCard({
  projectId,
  version,
  dopEnabled,
  onSaved,
}: {
  projectId: number
  version: Version
  dopEnabled: boolean
  onSaved: () => Promise<void> | void
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const saved = version.trigger_word ?? ''
  const [value, setValue] = useState(saved)
  const [busy, setBusy] = useState(false)
  const [detecting, setDetecting] = useState(false)
  const [detected, setDetected] = useState<TriggerDetectResult | null>(null)

  useEffect(() => { setValue(saved) }, [saved, version.id])

  const detect = useCallback(async (fill: boolean) => {
    setDetecting(true)
    try {
      const r = await api.detectTrigger(projectId, version.id)
      setDetected(r)
      if (fill && r.suggested) setValue(r.suggested)
    } catch (e) {
      toast(String((e as Error).message || e), 'error')
    } finally {
      setDetecting(false)
    }
  }, [projectId, version.id, toast])

  // DOP on and nothing saved: look at the captions straight away so the fix is
  // one click (Save) rather than a hunt for where the trigger is supposed to go.
  useEffect(() => {
    if (dopEnabled && !saved.trim()) void detect(true)
  }, [dopEnabled, saved, detect])

  const dirty = value.trim() !== saved.trim()
  const save = async () => {
    setBusy(true)
    try {
      await api.updateVersion(projectId, version.id, { trigger_word: value.trim() })
      await onSaved()
      toast(t('trigger.saved'), 'success')
    } catch (e) {
      toast(t('trigger.saveFailed', { error: String((e as Error).message || e) }), 'error')
    } finally {
      setBusy(false)
    }
  }

  const missing = dopEnabled && !saved.trim()
  const inCaptions = detected?.candidates.find((c) => c.word.toLowerCase() === value.trim().toLowerCase())

  // A field row like the schema ones; under DOP it is the gated row the mockup
  // draws with a green rule, since DOP cannot run without it.
  return (
    <div
      className="ds-field"
      data-testid="trigger-word-card"
      style={dopEnabled ? { background: 'var(--area)', borderLeft: '2px solid var(--green-600)', paddingLeft: 26 } : undefined}
    >
      <div className="ds-field-txt">
        <div className="ds-field-name">
          <span className="ds-label">{t('trigger.title')}</span>
          <span className="ds-key">trigger_word</span>
          {dopEnabled && <span className="ds-tag ds-gate">{t('trigger.gateDop')}</span>}
        </div>
        <div className="ds-field-desc">{t('trigger.hint')}</div>
        {detected && detected.candidates.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
            <span className="ds-kpi-meta">{t('trigger.candidates', { n: detected.total })}</span>
            {detected.candidates.map((c) => (
              <button
                key={c.word}
                type="button"
                onClick={() => setValue(c.word)}
                className={`ds-chip ds-mono${c.word === value.trim() ? ' ds-is-active' : ''}`}
                style={{ height: 24, paddingRight: 9, fontSize: 11 }}
                title={t('trigger.coverage', { pct: Math.round(c.coverage * 100) })}
              >
                {c.word} <b>{Math.round(c.coverage * 100)}%</b>
              </button>
            ))}
          </div>
        )}
        {detected && detected.total > 0 && detected.candidates.length === 0 && (
          <div className="ds-field-desc">{t('trigger.noneFound')}</div>
        )}
        {detected && value.trim() && !inCaptions && detected.total > 0 && (
          <div className="ds-field-desc" style={{ color: 'var(--amber-text)' }}>{t('trigger.notInCaptions')}</div>
        )}
        {missing && <div className="ds-field-desc" style={{ color: 'var(--amber-text)' }}>{t('trigger.dopNeedsTrigger')}</div>}
      </div>
      <div className="ds-field-ctl">
        <input
          className="ds-inp ds-mono"
          value={value}
          placeholder={t('trigger.placeholder')}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && dirty && !busy) void save() }}
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          aria-label={t('trigger.title')}
          style={missing ? { borderColor: 'var(--amber-line)' } : undefined}
        />
        <span style={{ display: 'flex', alignItems: 'center', gap: 7, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          {inCaptions && <span className="ds-badge ds-ok">{t('trigger.inCaptionsPct', { pct: Math.round(inCaptions.coverage * 100) })}</span>}
          <button type="button" className="ds-ctl ds-ghost" style={{ height: 24 }} onClick={() => void detect(true)} disabled={detecting}>
            {detecting ? t('trigger.detecting') : t('trigger.detect')}
          </button>
          <button type="button" className="ds-btn-primary" style={{ height: 24 }} onClick={() => void save()} disabled={!dirty || busy}>
            {busy ? t('common.saving') : t('common.save')}
          </button>
        </span>
      </div>
    </div>
  )
}
