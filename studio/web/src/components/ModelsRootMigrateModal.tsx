/** Models root directory migration confirm + progress modal (Settings -> System -> Storage location -> models root).
 *
 * Mirrors StudioDataMigrateModal; the difference: the copy **takes effect
 * immediately, no restart needed** (models_root() reads secrets.models.root
 * live). So the done phase skips "restart now" and just offers "done" +
 * close, and calls onDone so the parent refreshes the displayed current path.
 *
 * Phases: loading (fetches the scan info) -> confirm (file count / size /
 * top-level breakdown + confirm) -> running (progress bar, SSE-driven) ->
 * done / error. The modal can't be closed while running. If the target
 * already has models data, the backend returns 409 target_conflict -> the
 * conflict phase (issue #351): "skip existing files" (merge in, keep the
 * target's existing version for same-named files) / "overwrite existing
 * files" / cancel.
 */
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { api, type ApiError, type ModelsRootInfo } from '../api/client'
import { formatBytes } from '../lib/useUploadProgress'
import { useEventStream } from '../lib/useEventStream'

type Phase = 'loading' | 'confirm' | 'conflict' | 'running' | 'done' | 'error'

interface ConflictInfo {
  existingFiles: number
  existingBytes: number
  sameNameFiles: number
}

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

export default function ModelsRootMigrateModal({ target, onClose, onDone }: {
  target: string
  onClose: () => void
  /** Called on successful migration (the parent refetches getModelsRootInfo to refresh the current path; no restart needed) */
  onDone: () => void
}) {
  const { t } = useTranslation()
  // The user picks the parent directory; the backend actually copies data to
  // target/models/ (for display only -- the API still sends `target` and the
  // backend appends the suffix).
  const sep = target.includes('\\') ? '\\' : '/'
  const destination = target.endsWith(sep) ? `${target}models` : `${target}${sep}models`
  const [phase, setPhase] = useState<Phase>('loading')
  const [info, setInfo] = useState<ModelsRootInfo | null>(null)
  const [progress, setProgress] = useState<Progress>(EMPTY_PROGRESS)
  const [conflict, setConflict] = useState<ConflictInfo | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    void api.getModelsRootInfo().then((i) => {
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
    if (evt.type === 'models_root_migrate_progress') {
      setProgress({
        doneFiles: Number(evt.done_files) || 0,
        totalFiles: Number(evt.total_files) || 0,
        doneBytes: Number(evt.done_bytes) || 0,
        totalBytes: Number(evt.total_bytes) || 0,
        currentFile: typeof evt.current_file === 'string' ? evt.current_file : '',
      })
    } else if (evt.type === 'models_root_migrate_done') {
      setPhase((p) => {
        if (p !== 'running') return p
        if (evt.ok) { onDone(); return 'done' }
        setError(typeof evt.error === 'string' ? evt.error : 'unknown')
        return 'error'
      })
    }
  }, {
    // A done event can be lost while SSE is disconnected/reconnecting, which
    // would leave the running phase stuck (the modal can't be closed) --
    // fetch a fresh status snapshot on reconnect to catch up.
    onOpen: () => {
      void api.getModelsRootMigrateStatus().then((s) => {
        setPhase((p) => {
          if (p !== 'running') return p
          if (s.state === 'done') { onDone(); return 'done' }
          if (s.state === 'error') { setError(s.error); return 'error' }
          return p
        })
      }).catch(() => { /* retry on the next reconnect */ })
    },
  })

  const handleStart = async (onConflict?: 'skip' | 'overwrite') => {
    setPhase('running')
    setProgress(EMPTY_PROGRESS)
    try {
      await api.startModelsRootMigrate(target, onConflict)
    } catch (e) {
      const err = e as ApiError
      if (err.code === 'models_root.target_conflict') {
        const d = (err.detail ?? {}) as Record<string, unknown>
        setConflict({
          existingFiles: Number(d.existing_files) || 0,
          existingBytes: Number(d.existing_bytes) || 0,
          sameNameFiles: Number(d.same_name_files) || 0,
        })
        setPhase('conflict')
        return
      }
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
            {t('settings.storage.modelsMigrateTitle')}
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
                {t('settings.storage.modelsKeepOriginalNote')}
              </div>
            </>
          )}

          {phase === 'conflict' && conflict && (
            <>
              <div className="text-sm font-semibold">
                {t('settings.storage.conflictTitle')}
              </div>
              <div className="text-xs text-fg-secondary">
                {t('settings.storage.conflictSummary', {
                  path: destination,
                  files: conflict.existingFiles,
                  size: formatBytes(conflict.existingBytes),
                  same: conflict.sameNameFiles,
                })}
              </div>
              <div className="text-xs text-fg-tertiary">
                {t('settings.storage.conflictHint')}
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
                  {progress.doneFiles}/{progress.totalFiles} · {formatBytes(progress.doneBytes)}/{formatBytes(progress.totalBytes)} · {pct}%
                </span>
              </div>
            </>
          )}

          {phase === 'done' && (
            <>
              <div className="text-sm text-ok">{t('settings.storage.doneTitle')}</div>
              <div className="text-xs text-fg-secondary">
                {t('settings.storage.modelsDoneNote')}
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
          {phase === 'conflict' && (
            <>
              <button className="btn btn-ghost" onClick={() => setPhase('confirm')}>
                {t('common.cancel')}
              </button>
              <button className="btn btn-danger" onClick={() => void handleStart('overwrite')}>
                {t('settings.storage.conflictOverwrite')}
              </button>
              <button className="btn btn-primary" onClick={() => void handleStart('skip')}>
                {t('settings.storage.conflictSkip')}
              </button>
            </>
          )}
          {(phase === 'done' || phase === 'error') && (
            <button className="btn btn-primary" onClick={onClose}>{t('common.close')}</button>
          )}
        </footer>
      </div>
    </div>
  )
}
