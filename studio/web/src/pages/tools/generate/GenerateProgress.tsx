import { useTranslation } from 'react-i18next'

/** Generation progress: covers the **whole pipeline** (not just the sample step).
 *
 * Source: daemon-pushed SSE events
 *   - generate_phase          { name: 'load'|'clip'|'sample'|'vae' }  <- non-sample phases
 *   - generate_image_started  { batch_idx, batch_total, total_steps }
 *   - generate_preview_step   { step, total, image_b64? }
 *
 * Generate.tsx aggregates these into a `progress` prop, rendered in the
 * result card's footer (meter + status + params).
 */
export type GeneratePhase = 'load' | 'clip' | 'sample' | 'vae'

export interface GenerateProgress {
  /** Current phase (load/clip/sample/vae); null = unknown (fall back to step estimate) */
  phase: GeneratePhase | null
  /** Which image is running (multi-image / XY runs; single image has batchTotal=1) */
  batchIdx: number | null
  batchTotal: number | null
  /** Current sample step for this image */
  currentStep: number | null
  totalSteps: number | null
}

// Base progress fraction each phase occupies within a single image (sample spans 0.20-0.92 linearly by step)
const PHASE_BASE: Record<GeneratePhase, number> = {
  load: 0.03,
  clip: 0.12,
  sample: 0.20,
  vae: 0.95,
}

/** Result card footer: meter, status text and the run's parameters. */
export default function GenerateProgressBar({
  busy, progress, status, params,
}: {
  busy: boolean
  progress: GenerateProgress
  /** Text when nothing is in flight ("done · 6.4 s", an error, …). */
  status?: { text: string; tone?: 'ok' | 'err' } | null
  /** Right-aligned parameter line. */
  params?: string
}) {
  const { t } = useTranslation()
  const active = busy || progress.currentStep != null || progress.phase != null

  const stepFrac =
    progress.currentStep != null && progress.totalSteps && progress.totalSteps > 0
      ? Math.min(1, progress.currentStep / progress.totalSteps)
      : 0

  // Single-image progress: sample phase spans 0.20-0.92 by step, other phases use their base;
  // fall back to the step fraction when phase is unknown.
  let frac: number
  if (progress.phase === 'sample') frac = PHASE_BASE.sample + stepFrac * 0.72
  else if (progress.phase) frac = PHASE_BASE[progress.phase]
  else frac = stepFrac > 0 ? PHASE_BASE.sample + stepFrac * 0.72 : 0

  // Multi-image (batch / XY): fold this image's fraction into the overall progress
  const bt = progress.batchTotal
  const bi = progress.batchIdx
  const overall = bt && bt > 1 && bi != null ? Math.min(1, (bi + frac) / bt) : frac
  const pct = Math.round(overall * 100)

  let phaseLabel: string
  if (progress.phase === 'load') phaseLabel = t('generate.phaseLoad')
  else if (progress.phase === 'clip') phaseLabel = t('generate.phaseClip')
  else if (progress.phase === 'vae') phaseLabel = t('generate.phaseVae')
  else if (progress.phase === 'sample' || stepFrac > 0)
    phaseLabel = t('generate.phaseSample', { step: progress.currentStep ?? 0, total: progress.totalSteps ?? 0 })
  else phaseLabel = t('generate.progressPreparing')

  const batchTag = bt && bt > 1 && bi != null ? `${bi + 1}/${bt} · ` : ''

  if (!active && !status && !params) return null
  const meterPct = active ? pct : status?.tone === 'ok' ? 100 : status ? 0 : null

  return (
    <div style={{ borderTop: '1px solid var(--line)', padding: '11px 17px', display: 'flex', alignItems: 'center', gap: 12, flex: 'none' }}>
      {meterPct != null && (
        <span className="ds-meter" style={{ width: 180, flex: 'none' }}>
          <i style={{ width: `${meterPct}%`, transition: 'width 150ms linear' }} />
        </span>
      )}
      {active ? (
        <span className="ds-mono" style={{ fontSize: 11.5, color: 'var(--ink-2)', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {batchTag}{phaseLabel} · {pct}%
        </span>
      ) : status ? (
        <span
          className="ds-mono"
          style={{ fontSize: 11.5, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: status.tone === 'err' ? 'var(--red-text)' : status.tone === 'ok' ? 'var(--green-text)' : 'var(--ink-3)' }}
          title={status.text}
        >
          {status.text}
        </span>
      ) : null}
      {params && (
        <span className="ds-mono ds-muted" style={{ fontSize: 11.5, marginLeft: 'auto', whiteSpace: 'nowrap' }}>{params}</span>
      )}
    </div>
  )
}
