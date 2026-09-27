// PauseConfirmModal -- pause confirm modal per ADR 0006 Addendum 1, UI decision #5.
//
// Design rationale: option Delta downgrades pause semantics to "release the
// GPU immediately + discard the current round's progress", which doesn't
// match the user's intuition of "save progress then exit". A confirm modal
// before every pause click makes this explicit:
// (1) some experimental params (InfoNoise / Prodigy-family / cosine LR) may
//     be affected by pausing
// (2) resuming loses the current round's progress and continues from the end
//     of the previous epoch
// Only after the user confirms does it actually call the pause API and enter
// the PauseProgressModal lock screen.
//
// Copy deliberately carries no dynamic fields (no epoch N / task name) --
// ADR decision #5 states explicitly: "the user doesn't need to know what the
// actual logic looks like".
import { useTranslation } from 'react-i18next'

export interface PauseConfirmModalProps {
  onCancel: () => void
  onConfirm: () => void
}

export function PauseConfirmModal({ onCancel, onConfirm }: PauseConfirmModalProps) {
  const { t } = useTranslation()

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="pause-confirm-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]"
      data-testid="pause-confirm-modal"
    >
      <div className="w-[90%] max-w-[480px] flex flex-col gap-5 p-7 bg-elevated border border-subtle rounded-2xl shadow-xl">
        <h2
          id="pause-confirm-title"
          className="m-0 text-lg font-semibold text-fg-primary"
        >
          {t('queue.pauseConfirm.title')}
        </h2>

        <p className="m-0 text-sm text-fg-secondary leading-relaxed">
          {t('queue.pauseConfirm.adaptiveWarning')}
        </p>

        <p className="m-0 text-sm text-fg-secondary leading-relaxed">
          {t('queue.pauseConfirm.epochLoss')}
        </p>

        <div className="flex justify-end gap-3 mt-2">
          {/* Unified modal footer pattern: full-size btn-secondary / btn-primary
              (same as Dialog / the new-project modal; previously a smaller
              ghost+sm pair). */}
          <button
            type="button"
            onClick={onCancel}
            className="btn btn-secondary"
            data-testid="pause-confirm-cancel"
          >
            {t('queue.pauseConfirm.cancel')}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="btn btn-primary"
            data-testid="pause-confirm-ok"
          >
            {t('queue.pauseConfirm.confirm')}
          </button>
        </div>
      </div>
    </div>
  )
}
