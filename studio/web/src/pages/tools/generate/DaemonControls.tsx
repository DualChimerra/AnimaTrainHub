import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type DaemonStatus } from '../../../api/client'
import { useEventStream } from '../../../lib/useEventStream'
import { useToast } from '../../../components/Toast'

/** Bottom status strip of the Generate page (mockup .logbar): inference daemon
 *  state, model loaded / queue size, "clear VRAM" and the log drawer toggle. */
export default function DaemonControls({ queued, logOpen, onToggleLog }: {
  /** Generate tasks waiting in the queue. */
  queued: number
  logOpen: boolean
  onToggleLog: () => void
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [status, setStatus] = useState<DaemonStatus | null>(null)
  const [unloading, setUnloading] = useState(false)

  useEffect(() => {
    void api.getDaemonStatus()
      .then(setStatus)
      .catch(() => { /* startup flicker */ })
  }, [])

  useEventStream((evt) => {
    if (evt.type === 'daemon_state_changed') {
      setStatus({
        state: evt.state,
        model_loaded: !!evt.model_loaded,
        busy: !!evt.busy,
        alive: evt.state !== 'stopped',
      } as DaemonStatus)
    }
  })

  const handleUnload = async () => {
    setUnloading(true)
    try {
      const r = await api.unloadDaemon()
      toast(r.noop ? t('generate.vramAlreadyFree') : t('generate.vramUnloadRequested'), r.noop ? 'info' : 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setUnloading(false)
    }
  }

  const canUnload = !!(status && status.model_loaded && !status.busy && status.state !== 'unloading')
  const dot = !status || status.state === 'stopped'
    ? 'ds-dot-mute'
    : status.busy || status.state === 'starting' || status.state === 'unloading' ? 'ds-dot-run' : 'ds-dot-ok'
  const detail = [
    status ? t(`generate.daemonState.${status.state}`) : t('generate.daemonLoading'),
    status && (status.model_loaded ? t('generate.daemonModelLoaded') : t('generate.daemonModelNotLoaded')),
    t('generate.daemonQueue', { n: queued }),
  ].filter(Boolean).join(' · ')

  return (
    <div className="ds-logbar">
      <span className={`ds-dot ${dot}`} />
      <span>{t('generate.daemonTitle')}</span>
      <span className="ds-mono" style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{detail}</span>
      <button
        type="button"
        className="ds-mono"
        style={{ marginLeft: 'auto', color: 'inherit', opacity: canUnload && !unloading ? 1 : 0.45 }}
        onClick={handleUnload}
        disabled={!canUnload || unloading}
        title={
          !status ? t('generate.daemonLoading')
            : status.busy ? t('generate.daemonBusy')
              : !status.model_loaded ? t('generate.daemonNoModel')
                : t('generate.daemonUnloadTitle')
        }
      >
        {unloading ? t('generate.unloadingVram') : t('generate.unloadVram')}
      </button>
      <button type="button" className="ds-mono" style={{ color: 'inherit' }} aria-expanded={logOpen} onClick={onToggleLog}>
        {t('dataset.logToggle')} {logOpen ? '↓' : '↑'}
      </button>
    </div>
  )
}
