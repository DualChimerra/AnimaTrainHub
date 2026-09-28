/** ADR-0007 §11.3-B: version.status (5-value enum) -> color mapping.
 *
 *  status: preparing / training / completed / failed / canceled
 *  Coexists with the old StageBadge for now; after the PR-5 v9 destructive
 *  migration, StageBadge gets removed along with the stage field and
 *  VersionStatusBadge becomes the sole version status display component.
 */
import { useTranslation } from 'react-i18next'
import type { VersionPhase, VersionStatus } from '../api/client'

const DOT_RUNNING = (
  <span className="dot dot-running" style={{ flexShrink: 0 }} />
)

type StatusEntry = { badge: string; key: string; dot?: true }

const STATUS_MAP: Record<VersionStatus, StatusEntry> = {
  preparing: { badge: 'badge-warn',    key: 'versionStatus.preparing' },
  training:  { badge: 'badge-accent',  key: 'versionStatus.training', dot: true },
  completed: { badge: 'badge-ok',      key: 'versionStatus.completed' },
  failed:    { badge: 'badge-err',     key: 'versionStatus.failed' },
  canceled:  { badge: 'badge-neutral', key: 'versionStatus.canceled' },
}

/** Phase text suffixed onto the badge while preparing (per PR #265 review:
 * the optional preprocessing / regularizing steps are shown too -- whatever
 * the cursor is on). */
const PHASE_SUFFIX_KEY: Record<VersionPhase, string> = {
  curating: 'versionPhase.curating',
  preprocessing: 'versionPhase.preprocessing',
  editing: 'versionPhase.editing',
  regularizing: 'versionPhase.regularizing',
  ready: 'versionPhase.ready',
}

export default function VersionStatusBadge({
  status,
  phase,
}: {
  status: VersionStatus | null | undefined
  /** When passed and status=preparing, shows a "Preparing · tagging"-style suffix (used by the project card). */
  phase?: VersionPhase | null
}) {
  const { t } = useTranslation()
  if (!status) return null
  const entry = STATUS_MAP[status] ?? { badge: 'badge-neutral', key: status }
  const suffixKey =
    status === 'preparing' && phase ? PHASE_SUFFIX_KEY[phase] : undefined
  return (
    <span className={`badge ${entry.badge}`}>
      {entry.dot && DOT_RUNNING}
      {STATUS_MAP[status] ? t(entry.key) : status}
      {suffixKey ? `, ${t(suffixKey)}` : ''}
    </span>
  )
}
