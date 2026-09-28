// PauseProgressModal -- the ADR 0006 PR-4 SS4.3 pause-in-progress modal.
//
// Design rationale: after the user clicks "pause", handle_interrupt takes a few to a dozen seconds to write to disk
// (state + LoRA + wandb finish); the UI must give transparent feedback and block misclicks during that window. Clicking
// pause locks the screen with a modal that guides the whole process; only __EVENT__:pause_state counts as "pause complete"
// (rc=0 isn't enough -- rc is unreliable when the Windows wrapper rewrites it).
//
// State machine (self-managed inside the modal, decoupled from task.status):
//   - 'saving': sent the pause request, waiting for the child process to emit pause_state
//   - 'saved': success, task_state_changed -> paused, close the modal + toast
//   - 'timeout': no event received after 30s, shows a three-way choice (wait more / force-quit keeping progress / terminate)
//   - 'failed': the child process exited abnormally (rc != 0 + status='failed')
//
// SSE subscription: uses the global useEventStream, filtered to this task_id's pause_state /
// task_state_changed / pause_failed events.
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api } from '../api/client'
import { useEventStream, type StudioEvent } from '../lib/useEventStream'
import { useToast } from './Toast'

const TIMEOUT_MS = 30_000

type PhaseState =
  | { phase: 'saving' }
  | { phase: 'saved'; step: number }
  | { phase: 'timeout' }
  | { phase: 'failed'; exitCode: number | null }

export interface PauseProgressModalProps {
  taskId: number
  /** Task display name (used for the modal title) */
  taskName?: string
  /** Modal close callback */
  onClose: () => void
}

export function PauseProgressModal({ taskId, taskName, onClose }: PauseProgressModalProps) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [state, setState] = useState<PhaseState>({ phase: 'saving' })
  const [elapsedSec, setElapsedSec] = useState(0)
  const startedAt = useRef(Date.now())
  // Shares phase across onEvent / the timer, to avoid closure traps.
  const phaseRef = useRef<PhaseState['phase']>('saving')
  phaseRef.current = state.phase

  // -- elapsed counter (used for the copy display + timeout detection)
  useEffect(() => {
    const id = window.setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - startedAt.current) / 1000))
    }, 1000)
    return () => window.clearInterval(id)
  }, [])

  // -- 30s timeout -> escalate to the timeout state (only fires during the saving phase)
  useEffect(() => {
    const id = window.setTimeout(() => {
      if (phaseRef.current === 'saving') {
        setState({ phase: 'timeout' })
      }
    }, TIMEOUT_MS)
    return () => window.clearTimeout(id)
  }, [])

  // -- SSE listener
  useEventStream(
    useCallback((evt: StudioEvent) => {
      if (evt.task_id !== taskId) return
      if (evt.type === 'pause_state') {
        // The child process has persisted .pt + snapshot to disk -> mark saved; step comes from the payload (PR-2 emit)
        const step = typeof evt.step === 'number' ? evt.step : 0
        setState({ phase: 'saved', step })
        return
      }
      if (evt.type === 'task_state_changed') {
        if (evt.status === 'paused' && phaseRef.current !== 'saved') {
          // Fallback: if the pause_state event got lost, task_state_changed='paused' also counts as success
          setState({ phase: 'saved', step: 0 })
        } else if (evt.status === 'failed' || evt.status === 'canceled') {
          // The child process exited abnormally / the user canceled from elsewhere -> failed state
          if (phaseRef.current === 'saving' || phaseRef.current === 'timeout') {
            setState({ phase: 'failed', exitCode: null })
          }
        }
      }
    }, [taskId]),
  )

  // -- user actions
  const handleClose = () => onClose()

  const handleWaitMore = () => {
    // Wait another 30 seconds: reset the timer + roll the state back to saving
    startedAt.current = Date.now()
    setElapsedSec(0)
    setState({ phase: 'saving' })
  }

  const handleForceCancelKeep = async () => {
    // ADR SS4.3 "force-cancel, keep progress": sends a hard interrupt; marks paused if the pause file
    // has already landed on disk, otherwise degrades to canceled -- that degrade logic is already implemented
    // in supervisor's three-way _finish_slot branch, the frontend only needs to call cancel.
    try {
      await api.cancelTask(taskId)
      toast(t('queue.pauseSent'), 'info')
    } catch (e) {
      toast(String(e), 'error')
    }
    onClose()
  }

  const handleTerminate = async () => {
    try {
      await api.cancelTask(taskId)
    } catch (e) {
      toast(String(e), 'error')
    }
    onClose()
  }

  const handleViewLogs = () => {
    // Open the QueueDetail logs tab
    window.location.href = `/queue/${taskId}?tab=logs`
  }

  // -- render
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="pause-progress-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]"
      data-testid="pause-progress-modal"
    >
      <div className="w-[90%] max-w-[520px] flex flex-col gap-6 p-7 bg-elevated border border-subtle rounded-2xl shadow-xl">
        <h2
          id="pause-progress-title"
          className="m-0 text-lg font-semibold text-fg-primary"
        >
          {t('queue.pauseProgress.title', { id: taskId })}
          {taskName ? `, ${taskName}` : ''}
        </h2>

        {state.phase === 'saving' && (
          <div className="flex flex-col gap-3" data-testid="pause-saving">
            <div className="flex items-center gap-3">
              <span className="inline-block w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-fg-secondary">{t('queue.pauseProgress.saving')}</span>
            </div>
            <span className="text-xs text-fg-tertiary">
              {t('queue.pauseProgress.elapsedSeconds', { n: elapsedSec })}
            </span>
          </div>
        )}

        {state.phase === 'saved' && (
          <div className="flex flex-col gap-3" data-testid="pause-saved">
            <span className="text-sm text-ok">
              ✓ {t('queue.pauseProgress.saved', { step: state.step })}
            </span>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={handleClose}
                className="btn btn-primary"
              >
                {t('queue.pauseProgress.ok')}
              </button>
            </div>
          </div>
        )}

        {state.phase === 'timeout' && (
          <div className="flex flex-col gap-4" data-testid="pause-timeout">
            <span className="text-sm font-medium text-warn">
              ⚠️ {t('queue.pauseProgress.timeoutTitle')}
            </span>
            <span className="text-xs text-fg-secondary">
              {t('queue.pauseProgress.timeoutDesc', { n: elapsedSec })}
            </span>
            <div className="flex flex-col gap-2">
              <button
                type="button"
                onClick={handleWaitMore}
                className="px-3 py-2 text-sm rounded border border-dim bg-surface hover:bg-surface-hover text-left"
              >
                {t('queue.pauseProgress.waitMore')}
              </button>
              <button
                type="button"
                onClick={handleForceCancelKeep}
                className="px-3 py-2 text-sm rounded border border-dim bg-surface hover:bg-surface-hover text-left"
                title={t('queue.pauseProgress.forceCancelKeepHint')}
              >
                {t('queue.pauseProgress.forceCancelKeep')}
              </button>
              <button
                type="button"
                onClick={handleTerminate}
                className="px-3 py-2 text-sm rounded border border-err text-err hover:bg-err-soft text-left"
                title={t('queue.pauseProgress.terminateHint')}
              >
                {t('queue.pauseProgress.terminate')}
              </button>
            </div>
          </div>
        )}

        {state.phase === 'failed' && (
          <div className="flex flex-col gap-3" data-testid="pause-failed">
            <span className="text-sm font-medium text-err">
              ✗ {t('queue.pauseProgress.failedTitle')}
            </span>
            <span className="text-xs text-fg-secondary">
              {t('queue.pauseProgress.failedDesc', { code: state.exitCode ?? '?' })}
            </span>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={handleViewLogs}
                className="btn btn-secondary"
              >
                {t('queue.pauseProgress.viewLogs')}
              </button>
              <button
                type="button"
                onClick={handleClose}
                className="btn btn-primary"
              >
                {t('queue.pauseProgress.ok')}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
