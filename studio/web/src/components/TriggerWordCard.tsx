import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type TriggerDetectResult, type Version } from '../api/client'
import FieldLabel from './ds/FieldLabel'
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

  // Same cell as the schema settings around it: name on top, the input with
  // its actions in one line under it, then the caption candidates.
  return (
    <div className="ds-trigger" data-testid="trigger-word-card">
      <div className="ds-fcell-head">
        <FieldLabel label={t('trigger.title')} tip={t('trigger.hint')} />
        {dopEnabled && <span className="ds-tag ds-gate">{t('trigger.gateDop')}</span>}
        {inCaptions && <span className="ds-badge ds-ok">{t('trigger.inCaptionsPct', { pct: Math.round(inCaptions.coverage * 100) })}</span>}
      </div>
      <div className="ds-trigger-row">
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
        <button type="button" className="ds-ctl" onClick={() => void detect(true)} disabled={detecting}>
          {detecting ? t('trigger.detecting') : t('trigger.detect')}
        </button>
        <button type="button" className="ds-btn-primary" onClick={() => void save()} disabled={!dirty || busy}>
          {busy ? t('common.saving') : t('common.save')}
        </button>
      </div>
      {detected && detected.candidates.length > 0 && (
        <div className="ds-trigger-cands">
          <span className="ds-muted">{t('trigger.candidates', { n: detected.total })}</span>
          {detected.candidates.map((c) => (
            <button
              key={c.word}
              type="button"
              onClick={() => setValue(c.word)}
              className={`ds-chip ds-mono${c.word === value.trim() ? ' ds-is-active' : ''}`}
              title={t('trigger.coverage', { pct: Math.round(c.coverage * 100) })}
            >
              {c.word} <b>{Math.round(c.coverage * 100)}%</b>
            </button>
          ))}
        </div>
      )}
      {detected && detected.total > 0 && detected.candidates.length === 0 && (
        <div className="ds-trigger-msg">{t('trigger.noneFound')}</div>
      )}
      {detected && value.trim() && !inCaptions && detected.total > 0 && (
        <div className="ds-trigger-msg ds-warn">{t('trigger.notInCaptions')}</div>
      )}
      {missing && <div className="ds-trigger-msg ds-warn">{t('trigger.dopNeedsTrigger')}</div>}
    </div>
  )
}
