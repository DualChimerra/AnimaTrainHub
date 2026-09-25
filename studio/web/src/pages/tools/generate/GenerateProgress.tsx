import { useTranslation } from 'react-i18next'

/** 出图进度：覆盖**全流程**（不再只有采样 step）。
 *
 * 来源：daemon 推 SSE
 *   - generate_phase          { name: 'load'|'clip'|'sample'|'vae' }  ← 覆盖非采样阶段
 *   - generate_image_started  { batch_idx, batch_total, total_steps }
 *   - generate_preview_step   { step, total, image_b64? }
 *
 * Generate.tsx 聚合成 progress prop；渲染在「结果」卡片底栏（mockup：meter + 状态 + 参数）。
 */
export type GeneratePhase = 'load' | 'clip' | 'sample' | 'vae'

export interface GenerateProgress {
  /** 当前阶段（load/clip/sample/vae）；null = 未知（退回按 step 估算） */
  phase: GeneratePhase | null
  /** 当前在跑哪一张（多张图 / XY 时；单图 batchTotal=1） */
  batchIdx: number | null
  batchTotal: number | null
  /** 当前图采样到第几步 */
  currentStep: number | null
  totalSteps: number | null
}

// 各阶段在「单张图」里占的总进度基点（sample 段按 step 线性铺开 0.20→0.92）
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

  // 单张图的进度：sample 段按 step 铺 0.20→0.92，其余阶段用基点；无 phase 时退回 step。
  let frac: number
  if (progress.phase === 'sample') frac = PHASE_BASE.sample + stepFrac * 0.72
  else if (progress.phase) frac = PHASE_BASE[progress.phase]
  else frac = stepFrac > 0 ? PHASE_BASE.sample + stepFrac * 0.72 : 0

  // 多图（batch / XY）：把当前图的 frac 摊进整体
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
