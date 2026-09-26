// HoldQueueModal -- confirmation modal for holding the queue, ADR 0006 PR-4 §4.4.
//
// Case A: no running task -> simple yes/no confirmation
// Case B: a running task exists -> radio options let the user choose "let it
//   finish" vs "pause it too". The primary button's text updates with the
//   radio choice, so clicking doesn't leave the user unsure what they just confirmed.
//
// The decision is handed to the caller via onConfirm, which calls hold + an optional pause API itself.
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { Task } from '../api/client'

export type HoldDecision =
  | { kind: 'hold-only' }
  | { kind: 'hold-and-pause'; taskId: number }

export interface HoldQueueModalProps {
  /** The current running task; null = case A. */
  runningTask: Task | null
  onCancel: () => void
  onConfirm: (decision: HoldDecision) => void
}

export function HoldQueueModal({ runningTask, onCancel, onConfirm }: HoldQueueModalProps) {
  const { t } = useTranslation()
  // Only used in case B: defaults to "let it finish"
  const [pauseToo, setPauseToo] = useState(false)
  const hasRunning = runningTask !== null
  const runningName = runningTask?.name ?? ''
  const runningId = runningTask?.id ?? 0

  const handleConfirm = () => {
    if (hasRunning && pauseToo) {
      onConfirm({ kind: 'hold-and-pause', taskId: runningId })
    } else {
      onConfirm({ kind: 'hold-only' })
    }
  }

  const confirmText = !hasRunning
    ? t('queue.holdModal.confirmSimple')
    : pauseToo
      ? t('queue.holdModal.confirmPauseToo', { id: runningId })
      : t('queue.holdModal.confirmLetRun', { id: runningId })

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="hold-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]"
      data-testid="hold-queue-modal"
    >
      <div className="w-[90%] max-w-[520px] flex flex-col gap-5 p-7 bg-elevated border border-subtle rounded-2xl shadow-xl">
        <h2
          id="hold-modal-title"
          className="m-0 text-lg font-semibold text-fg-primary"
        >
          {t('queue.holdModal.title')}
        </h2>

        <p className="m-0 text-sm text-fg-secondary leading-relaxed">
          {hasRunning
            ? t('queue.holdModal.descWithRunning')
            : t('queue.holdModal.descNoRunning')}
        </p>

        {hasRunning && (
          <div className="flex flex-col gap-3">
            <p className="m-0 text-sm text-fg-primary">
              {t('queue.holdModal.currentlyRunning', { id: runningId, name: runningName })}
            </p>
            <div className="flex flex-col gap-2 pl-1">
              <label className="flex items-start gap-2 cursor-pointer text-sm text-fg-secondary">
                <input
                  type="radio"
                  name="hold-option"
                  checked={!pauseToo}
                  onChange={() => setPauseToo(false)}
                  className="mt-1"
                  data-testid="hold-opt-let-run"
                />
                <span>{t('queue.holdModal.optionLetRun', { id: runningId })}</span>
              </label>
              <label className="flex items-start gap-2 cursor-pointer text-sm text-fg-secondary">
                <input
                  type="radio"
                  name="hold-option"
                  checked={pauseToo}
                  onChange={() => setPauseToo(true)}
                  className="mt-1"
                  data-testid="hold-opt-pause-too"
                />
                <span>{t('queue.holdModal.optionPauseToo', { id: runningId })}</span>
              </label>
            </div>
          </div>
        )}

        <div className="flex justify-end gap-2 pt-1">
          {/* Unified modal footer pattern: btn-secondary / btn-primary (the
              previous bare tailwind classes had a different size/radius than
              the .btn system). */}
          <button
            type="button"
            onClick={onCancel}
            className="btn btn-secondary"
            data-testid="hold-cancel-btn"
          >
            {t('common.cancel', { defaultValue: 'Cancel' })}
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            className="btn btn-primary"
            data-testid="hold-confirm-btn"
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  )
}
