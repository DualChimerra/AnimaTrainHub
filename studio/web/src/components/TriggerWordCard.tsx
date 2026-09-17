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

  return (
    <section
      className={`rounded-md border px-3.5 py-3 flex flex-col gap-2 shrink-0 ${
        missing ? 'border-warn bg-warn-soft' : 'border-subtle bg-surface'
      }`}
      data-testid="trigger-word-card"
    >
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-sm font-semibold text-fg-primary">{t('trigger.title')}</span>
        {dopEnabled && <span className="badge badge-accent">DOP</span>}
      </div>
      <p className="m-0 text-xs text-fg-tertiary">{t('trigger.hint')}</p>
      <div className="flex items-center gap-2 flex-wrap">
        <input
          className="input input-mono flex-1 min-w-[160px]"
          value={value}
          placeholder={t('trigger.placeholder')}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && dirty && !busy) void save() }}
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          aria-label={t('trigger.title')}
        />
        <button
          type="button" className="btn btn-secondary btn-sm"
          onClick={() => void detect(true)} disabled={detecting}
        >
          {detecting ? t('trigger.detecting') : t('trigger.detect')}
        </button>
        <button
          type="button" className="btn btn-primary btn-sm"
          onClick={() => void save()} disabled={!dirty || busy}
        >
          {busy ? t('common.saving') : t('common.save')}
        </button>
      </div>

      {detected && detected.candidates.length > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap text-xs">
          <span className="text-fg-tertiary">{t('trigger.candidates', { n: detected.total })}</span>
          {detected.candidates.map((c) => (
            <button
              key={c.word} type="button"
              onClick={() => setValue(c.word)}
              className={`px-2 py-0.5 rounded-md font-mono border transition-colors ${
                c.word === value.trim() ? 'border-accent bg-accent-soft text-fg-primary' : 'border-subtle bg-canvas text-fg-secondary hover:border-bold'
              }`}
              title={t('trigger.coverage', { pct: Math.round(c.coverage * 100) })}
            >
              {c.word} <span className="text-fg-tertiary">{Math.round(c.coverage * 100)}%</span>
            </button>
          ))}
        </div>
      )}
      {detected && detected.total > 0 && detected.candidates.length === 0 && (
        <p className="m-0 text-xs text-fg-tertiary">{t('trigger.noneFound')}</p>
      )}
      {detected && value.trim() && !inCaptions && detected.total > 0 && (
        <p className="m-0 text-xs text-warn">{t('trigger.notInCaptions')}</p>
      )}
      {missing && <p className="m-0 text-xs text-warn">{t('trigger.dopNeedsTrigger')}</p>}
    </section>
  )
}
