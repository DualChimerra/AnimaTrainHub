import { useTranslation } from 'react-i18next'

/** Task 状态徽章（pending / running / done / failed / canceled）。 */
export default function StatusBadge({ status }: { status: string }) {
  const { t } = useTranslation()
  const cls =
    status === 'done'    ? 'ds-badge ds-ok'
    : status === 'running'  ? 'ds-badge ds-info'
    : status === 'failed'   ? 'ds-badge ds-err'
    : 'ds-badge ds-mute'
  const label =
    status === 'done'    ? t('status.done')
    : status === 'running'  ? t('status.generating')
    : status === 'failed'   ? t('status.failed')
    : status === 'pending'  ? t('status.queued')
    : status === 'canceled' ? t('status.canceled')
    : status
  return <span className={cls}>{label}</span>
}
