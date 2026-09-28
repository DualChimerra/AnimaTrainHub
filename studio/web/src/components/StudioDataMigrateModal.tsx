/** studio_data migration confirm + progress modal (Settings -> System -> Storage location).
 *
 * Four phases: loading (fetches the scan info) -> confirm (file count / size
 * / top-level breakdown + confirm) -> running (progress bar, SSE-driven) ->
 * done / error.
 *
 * The modal can't be closed while running (it's a one-shot maintenance
 * operation with a bounded copy duration; the user just waits it out -- no
 * "keep migrating in the background, reopen to check progress" limbo state
 * is introduced). The new location only **takes effect after a server
 * restart** (the pointer file is resolved at import time), so the done
 * phase offers "restart now".
 */
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { api, type StudioDataInfo } from '../api/client'
import { formatBytes } from '../lib/useUploadProgress'
import { useEventStream } from '../lib/useEventStream'

type Phase = 'loading' | 'confirm' | 'running' | 'done' | 'error'

interface Progress {
  doneFiles: number
  totalFiles: number
  doneBytes: number
  totalBytes: number
  currentFile: string
}

const EMPTY_PROGRESS: Progress = {
  doneFiles: 0, totalFiles: 0, doneBytes: 0, totalBytes: 0, currentFile: '',
}

export default function StudioDataMigrateModal({ target, onClose, onRestart }: {
  target: string
  onClose: () => void
  /** "Restart now" in the done phase -- reuses the Settings page's existing restart + health-poll logic */
  onRestart: () => void
}) {
  const { t } = useTranslation()
  // The user picks the parent directory; the backend actually copies data to
  // target/studio_data/ (for display only -- the API still sends `target` and
  // the backend appends the suffix).
  const sep = target.includes('\\') ? '\\' : '/'
  const destination = target.endsWith(sep) ? `${target}studio_data` : `${target}${sep}studio_data`
  const [phase, setPhase] = useState<Phase>('loading')
  const [info, setInfo] = useState<StudioDataInfo | null>(null)
  const [progress, setProgress] = useState<Progress>(EMPTY_PROGRESS)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    void api.getStudioDataInfo().then((i) => {
      if (cancelled) return
      setInfo(i)
      setPhase('confirm')
    }).catch((e) => {
      if (cancelled) return
      setError(String(e))
      setPhase('error')
    })
    return () => { cancelled = true }
  }, [])

  // SSE: live progress + completion events (only acted on while running, to keep unrelated noise from flipping the phase)
  useEventStream((evt) => {
    if (evt.type === 'studio_data_migrate_progress') {
      setProgress({
        doneFiles: Number(evt.done_files) || 0,
        totalFiles: Number(evt.total_files) || 0,
        doneBytes: Number(evt.done_bytes) || 0,
        totalBytes: Number(evt.total_bytes) || 0,
        currentFile: typeof evt.current_file === 'string' ? evt.current_file : '',
      })
    } else if (evt.type === 'studio_data_migrate_done') {
      setPhase((p) => {
        if (p !== 'running') return p
        if (evt.ok) return 'done'
        setError(typeof evt.error === 'string' ? evt.error : 'unknown')
        return 'error'
      })
    }
  }, {
    // A done event can be lost while SSE is disconnected/reconnecting, which
    // would leave the running phase stuck (the modal can't be closed) --
    // fetch a fresh status snapshot on reconnect to catch up.
    onOpen: () => {
      void api.getStudioDataMigrateStatus().then((s) => {
        setPhase((p) => {
          if (p !== 'running') return p
          if (s.state === 'done') return 'done'
          if (s.state === 'error') { setError(s.error); return 'error' }
          return p
        })
      }).catch(() => { /* retry on the next reconnect */ })
    },
  })

  const handleStart = async () => {
    setPhase('running')
    setProgress(EMPTY_PROGRESS)
    try {
      await api.startStudioDataMigrate(target)
    } catch (e) {
      setError(String(e))
      setPhase('error')
    }
  }

  const closable = phase !== 'running'
  const pct = progress.totalBytes > 0
    ? Math.min(100, Math.round((progress.doneBytes / progress.totalBytes) * 100))
    : 0

  return (
    <div
      className="fixed inset-0 z-50 bg-zinc-950/40 backdrop-blur-[2px] flex items-center justify-center"
      onClick={closable ? onClose : undefined}
    >
      <div
        className="bg-elevated border border-subtle rounded-2xl shadow-xl w-[560px] max-h-[80vh] m-modal flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="px-4 py-3 border-b border-subtle flex items-center gap-2 shrink-0">
          <h3 className="m-0 text-sm font-semibold flex-1 text-fg-primary">
            {t('settings.storage.migrateTitle')}
          </h3>
          {closable && (
            <button className="btn btn-ghost text-xs" onClick={onClose} aria-label={t('common.close')}>×</button>
          )}
        </header>

        <div className="p-4 flex flex-col gap-3 overflow-y-auto">
          {phase === 'loading' && (
            <div className="text-sm text-fg-tertiary py-6 text-center">
              {t('settings.storage.scanning')}
            </div>
          )}

          {phase === 'confirm' && info?.scan && (
            <>
              <div className="text-xs text-fg-secondary flex flex-col gap-1">
                <div>
                  <span className="text-fg-tertiary">{t('settings.storage.from')}</span>{' '}
                  <code className="font-mono">{info.current}</code>
                </div>
                <div>
                  <span className="text-fg-tertiary">{t('settings.storage.to')}</span>{' '}
                  <code className="font-mono">{destination}</code>
                </div>
              </div>
              <div className="text-sm font-semibold">
                {t('settings.storage.totalLine', {
                  files: info.scan.total_files,
                  size: formatBytes(info.scan.total_bytes),
                })}
              </div>
              <div
                className="bg-sunken border border-subtle rounded-md text-xs font-mono overflow-y-auto"
                style={{ maxHeight: 200 }}
              >
                {info.scan.entries.map((e) => (
                  <div key={e.name} className="flex justify-between gap-2 px-2.5 py-1 border-b border-subtle last:border-b-0">
                    <span className="truncate">{e.is_dir ? `${e.name}/` : e.name}</span>
                    <span className="text-fg-tertiary shrink-0">
                      {t('settings.storage.entryMeta', { files: e.files, size: formatBytes(e.bytes) })}
                    </span>
                  </div>
                ))}
              </div>
              <div className="text-xs text-fg-tertiary">
                {t('settings.storage.keepOriginalNote')}
              </div>
            </>
          )}

          {phase === 'running' && (
            <>
              <div className="text-sm">{t('settings.storage.migrating')}</div>
              <div className="h-2 rounded-full bg-sunken border border-subtle overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${pct}%`,
                    background: 'var(--accent)',
                    transition: 'width 200ms linear',
                  }}
                />
              </div>
              <div className="text-xs text-fg-tertiary font-mono flex justify-between gap-2">
                <span className="truncate">{progress.currentFile}</span>
                <span className="shrink-0">
                  {progress.doneFiles}/{progress.totalFiles}, {formatBytes(progress.doneBytes)}/{formatBytes(progress.totalBytes)}, {pct}%
                </span>
              </div>
            </>
          )}

          {phase === 'done' && (
            <>
              <div className="text-sm text-ok">{t('settings.storage.doneTitle')}</div>
              <div className="text-xs text-fg-secondary">
                {t('settings.storage.doneRestartNote')}
              </div>
            </>
          )}

          {phase === 'error' && (
            <div className="text-sm text-err break-all">
              {t('settings.storage.failed', { error })}
            </div>
          )}
        </div>

        <footer className="px-4 py-3 border-t border-subtle flex justify-end gap-2 shrink-0">
          {phase === 'confirm' && (
            <>
              <button className="btn btn-ghost" onClick={onClose}>{t('common.cancel')}</button>
              <button className="btn btn-primary" onClick={() => void handleStart()}>
                {t('settings.storage.startMigrate')}
              </button>
            </>
          )}
          {phase === 'done' && (
            <>
              <button className="btn btn-ghost" onClick={onClose}>{t('common.close')}</button>
              <button className="btn btn-primary" onClick={onRestart}>
                {t('settings.storage.restartNow')}
              </button>
            </>
          )}
          {phase === 'error' && (
            <button className="btn btn-ghost" onClick={onClose}>{t('common.close')}</button>
          )}
        </footer>
      </div>
    </div>
  )
}
