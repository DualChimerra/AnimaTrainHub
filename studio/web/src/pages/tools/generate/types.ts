
import type { VersionStatus } from '../../../api/client'

export interface ProjectLora {
  projectId: number
  projectTitle: string
  versionId: number
  versionLabel: string
  status: VersionStatus
  path: string
  createdAt: number
}

export const DEFAULT_NEG =
  'worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, bad anatomy, bad hands, bad feet, missing fingers, extra fingers, text, watermark, logo, signature'

/** Sampler / scheduler options, kept in sync with the backend GenerateRequest Literal
 *  (first item = family default); out-of-family values get a 422 from the backend. */
export type GenerateFamily = 'anima' | 'krea2'

export const SAMPLER_OPTIONS_BY_FAMILY = {
  anima: ['er_sde', 'dpmpp_3m_sde'],
  krea2: ['euler'],
} as const satisfies Record<GenerateFamily, readonly string[]>

export const SCHEDULER_OPTIONS_BY_FAMILY = {
  anima: ['simple', 'sgm_uniform'],
  krea2: ['simple'],
} as const satisfies Record<GenerateFamily, readonly string[]>

export const FAMILY_GENERATE_DEFAULTS = {
  anima: { steps: 25, cfgScale: 4.0 },
  krea2: { steps: 28, cfgScale: 4.5 },
} as const satisfies Record<GenerateFamily, { steps: number; cfgScale: number }>

export const DISTILLED_GENERATE_DEFAULTS = { steps: 8, cfgScale: 0.0 } as const

export const SAMPLER_OPTIONS = ['er_sde', 'dpmpp_3m_sde', 'euler'] as const
export type SamplerName = (typeof SAMPLER_OPTIONS)[number]
export const DEFAULT_SAMPLER: SamplerName = 'er_sde'

export const SCHEDULER_OPTIONS = ['simple', 'sgm_uniform'] as const
export type SchedulerName = (typeof SCHEDULER_OPTIONS)[number]
export const DEFAULT_SCHEDULER: SchedulerName = 'simple'
