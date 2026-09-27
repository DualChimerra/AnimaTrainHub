import i18n from '../../i18n'
/** Shared utilities for data jobs (used by both DataJobsPanel + QueueDetail).
 *  After the R-5 ledger merge, a job is just a task (task_type = kind); every utility reads fields off the Task shape. */
import type { Task, TaskType } from '../../api/client'

/** The full set of kinds for the data view (light + io tiers). eval_samples is an exclusive
 *  tier, belonging to the GPU view (anchor §4-2), and is not included here. */
export const DATA_VIEW_KINDS: TaskType[] = [
  'download', 'preprocess', 'tag', 'reg_build',
  'eval_clip', 'eval_dino', 'eval_tag', 'eval_ccip',
]

/** All job kinds (for iterating over i18n labels; includes eval_samples). */
export const JOB_KINDS: TaskType[] = ['eval_samples', ...DATA_VIEW_KINDS]

export const JOB_STATUS_TONE: Record<string, string> = {
  pending: 'neutral', running: 'accent', done: 'ok', failed: 'err', canceled: 'neutral',
}

export function fmtJobAgo(ts: number): string {
  const sec = Math.max(0, Date.now() / 1000 - ts)
  if (sec < 60) return 'just now'
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`
  return `${Math.floor(sec / 86400)}d ago`
}

export function fmtJobDuration(start: number | null, end: number | null): string {
  if (!start) return '—'
  const e = end ?? Date.now() / 1000
  const sec = Math.max(0, e - start)
  if (sec < 60) return `${sec.toFixed(0)}s`
  const m = Math.floor(sec / 60); const s = Math.floor(sec % 60)
  if (m < 60) return `${m}m ${s}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

export function fmtJobTime(ts: number | null | undefined): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString(i18n.language, { hour12: false })
}

/** Job kind → deep link to the native step page (download is project-level, the rest are
 *  version-level; eval_* lands on the Train page). Non-job types return null (train/generate have their own separate jump link). */
export function jobJumpPath(task: Task): string | null {
  const kind = task.task_type ?? 'train'
  const pid = task.project_id
  const vid = task.version_id
  if (!pid || !JOB_KINDS.includes(kind)) return null
  if (kind === 'download') return `/projects/${pid}/download`
  if (!vid) return null
  switch (kind) {
    case 'preprocess': return `/projects/${pid}/v/${vid}/preprocess`
    case 'tag': return `/projects/${pid}/v/${vid}/tag`
    case 'reg_build': return `/projects/${pid}/v/${vid}/reg`
    default: return `/projects/${pid}/v/${vid}/train`
  }
}

/** params value → human-readable string. Booleans go through field.yes/no, arrays join with commas, objects truncate as JSON. */
export function fmtParamValue(
  v: unknown, t: (key: string) => string,
): string {
  if (typeof v === 'boolean') return v ? t('field.yes') : t('field.no')
  if (v == null) return '—'
  if (Array.isArray(v)) return v.length ? v.map(String).join(', ') : '—'
  if (typeof v === 'object') {
    const s = JSON.stringify(v)
    return s.length > 200 ? `${s.slice(0, 200)}…` : s
  }
  const s = String(v)
  return s.length > 200 ? `${s.slice(0, 200)}…` : s
}

/** param key → human-readable label; falls back to the raw key when no mapping exists (every field is shown, none hidden). */
export function paramLabel(key: string, t: (k: string) => string): string {
  const i18nKey = `queue.jobs.param.${key}`
  const label = t(i18nKey)
  return label === i18nKey ? key : label
}
