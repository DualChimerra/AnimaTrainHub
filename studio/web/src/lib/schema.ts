// schema.ts -- interprets the JSON Schema returned by FastAPI into the shape the frontend form needs.
import type { TFunction } from 'i18next'
import type { SchemaProperty } from '../api/client'

export type ControlKind =
  | 'bool'
  | 'select'
  | 'tristate'
  | 'int'
  | 'float'
  | 'string'
  | 'path'
  | 'textarea'
  | 'code'
  | 'string-list'
  | 'int-list'
  | 'float-list'

/**
 * Infers a field's control type. Prefers the schema's custom `control` meta
 * field when present, otherwise infers from JSON Schema's type / enum / anyOf.
 */
export function controlKind(prop: SchemaProperty): ControlKind {
  if (prop.control && prop.control !== 'auto') {
    if (
      prop.control === 'path' ||
      prop.control === 'textarea' ||
      prop.control === 'code' ||
      prop.control === 'string-list' ||
      prop.control === 'tristate'
    )
      return prop.control
  }

  if (prop.enum && prop.enum.length > 0) return 'select'

  // Unwrap the nullable anyOf: [X, null] shape
  let type = prop.type
  if (!type && prop.anyOf) {
    const hasNull = prop.anyOf.some((a) => a.type === 'null')
    const nonNull = prop.anyOf.find((a) => a.type && a.type !== 'null')
    if (hasNull && nonNull?.type === 'boolean') return 'tristate'
    type = nonNull?.type
  }

  if (type === 'boolean') return 'bool'
  if (type === 'integer') return 'int'
  if (type === 'number') return 'float'
  if (type === 'array') {
    const itemType = prop.items?.type
    if (itemType === 'integer') return 'int-list'
    if (itemType === 'number') return 'float-list'
    return 'string-list'
  }
  return 'string'
}

/**
 * Simple show_when parser: supports `key==value` / `key!=value`, plus `||` combinations.
 */
export function evalShowWhen(
  expr: string | undefined,
  values: Record<string, unknown>
): boolean {
  if (!expr) return true
  const branches = expr.split('||').map((part) => part.trim()).filter(Boolean)
  if (branches.length > 1) {
    return branches.some((branch) => evalShowWhen(branch, values))
  }
  const ands = expr.split('&&').map((part) => part.trim()).filter(Boolean)
  if (ands.length > 1) {
    return ands.every((clause) => evalShowWhen(clause, values))
  }
  const eq = expr.split('==')
  if (eq.length === 2) {
    return String(values[eq[0].trim()]) === eq[1].trim()
  }
  const ne = expr.split('!=')
  if (ne.length === 2) {
    return String(values[ne[0].trim()]) !== ne[1].trim()
  }
  return true
}

// pruneInactiveConfig (the frontend mirror of the backend's config_prune) has
// been removed (polish pass 2 / R4, D3): the YAML preview now goes through
// POST /api/schema/preview-yaml, the same serialization path used on disk.
// evalShowWhen is still the evaluator for the form's live visibility, kept
// as a word-for-word mirror of the backend's config_rules.eval_show_when
// (the last remaining cross-language dual implementation).

/** A field's human-readable label: capitalize first letters + turn underscores into spaces. */
export function fieldLabel(name: string): string {
  return name
    .split('_')
    .map((w) => (w.length > 0 ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ')
}

export const SCHEMA_GROUP_LABEL_KEYS: Record<string, string> = {
  model: 'schema.groups.model',
  dataset: 'schema.groups.dataset',
  caption: 'schema.groups.caption',
  lora: 'schema.groups.lora',
  training: 'schema.groups.training',
  noise_augmentation: 'schema.groups.noiseAugmentation',
  timestep_sampling: 'schema.groups.timestepSampling',
  loss: 'schema.groups.loss',
  system: 'schema.groups.system',
  output: 'schema.groups.output',
  sample: 'schema.groups.sample',
  eval_validation: 'schema.groups.evalValidation',
  monitor: 'schema.groups.monitor',
}

export const SCHEMA_ENUM_LABEL_KEYS: Record<string, Record<string, string>> = {
  model_family: {
    anima: 'schema.enums.modelFamily.anima',
    krea2: 'schema.enums.modelFamily.krea2',
  },
  lora_type: {
    lora: 'schema.enums.loraType.lora',
    lokr: 'schema.enums.loraType.lokr',
    loha: 'schema.enums.loraType.loha',
    ortho: 'schema.enums.loraType.ortho',
    tlora: 'schema.enums.loraType.tlora',
  },
  lr_scheduler: {
    none: 'schema.enums.lrScheduler.none',
    cosine: 'schema.enums.lrScheduler.cosine',
    cosine_with_restart: 'schema.enums.lrScheduler.cosineWithRestart',
    cosine_with_warmup: 'schema.enums.lrScheduler.cosineWithWarmup',
    cosine_cycles: 'schema.enums.lrScheduler.cosineCycles',
    constant_then_cosine: 'schema.enums.lrScheduler.constantThenCosine',
  },
  optimizer_type: {
    adamw: 'schema.enums.optimizerType.adamw',
    adamw8bit: 'schema.enums.optimizerType.adamw8bit',
    automagic: 'schema.enums.optimizerType.automagic',
    came: 'schema.enums.optimizerType.came',
    lion: 'schema.enums.optimizerType.lion',
    prodigy: 'schema.enums.optimizerType.prodigy',
    prodigy_plus_schedulefree: 'schema.enums.optimizerType.prodigyPlusSchedulefree',
  },
  timestep_sampling: {
    logit_normal: 'schema.enums.timestepSampling.logitNormal',
    uniform: 'schema.enums.timestepSampling.uniform',
    logit_normal_low: 'schema.enums.timestepSampling.logitNormalLow',
    mode: 'schema.enums.timestepSampling.mode',
    style_friendly: 'schema.enums.timestepSampling.styleFriendly',
    dual_peak: 'schema.enums.timestepSampling.dualPeak',
  },
  loss_weighting: {
    none: 'schema.enums.lossWeighting.none',
    min_snr: 'schema.enums.lossWeighting.minSnr',
    detail_inv_t: 'schema.enums.lossWeighting.detailInvT',
    cosmap: 'schema.enums.lossWeighting.cosmap',
  },
  leap_variant: {
    original: 'schema.enums.leapVariant.original',
    sparse: 'schema.enums.leapVariant.sparse',
    bridge: 'schema.enums.leapVariant.bridge',
    lagrange: 'schema.enums.leapVariant.lagrange',
  },
  mixed_precision: {
    bf16: 'schema.enums.mixedPrecision.bf16',
    fp16: 'schema.enums.mixedPrecision.fp16',
    no: 'schema.enums.mixedPrecision.no',
  },
  attention_backend: {
    none: 'schema.enums.attentionBackend.none',
    xformers: 'schema.enums.attentionBackend.xformers',
    flash_attn: 'schema.enums.attentionBackend.flashAttn',
  },
  noise_enhancement_type: {
    none: 'schema.enums.noiseEnhancementType.none',
    offset: 'schema.enums.noiseEnhancementType.offset',
    pyramid: 'schema.enums.noiseEnhancementType.pyramid',
  },
  sample_sampler_name: {
    er_sde: 'schema.enums.sampler.erSde',
    dpmpp_3m_sde: 'schema.enums.sampler.dpmpp3mSde',
    euler: 'schema.enums.sampler.euler',
  },
  sample_scheduler: {
    simple: 'schema.enums.scheduler.simple',
    sgm_uniform: 'schema.enums.scheduler.sgmUniform',
  },
}

/** Human label of a schema field: schema.labels.<field> when the locale has
 *  one, else the key in title case ("lora_rank" -> "Lora Rank"). */
export function schemaFieldLabel(name: string, t: TFunction): string {
  return t(`schema.labels.${name}`, { defaultValue: fieldLabel(name) })
}

export function schemaGroupLabel(key: string, fallback: string, t: TFunction): string {
  const labelKey = SCHEMA_GROUP_LABEL_KEYS[key]
  return labelKey ? t(labelKey) : fallback
}

export function schemaEnumLabel(fieldName: string, value: unknown, t: TFunction): string {
  const raw = String(value)
  const labelKey = SCHEMA_ENUM_LABEL_KEYS[fieldName]?.[raw]
  return labelKey ? t(labelKey) : raw
}

export function schemaDescription(name: string, fallback: string | undefined, t: TFunction): string | undefined {
  const translated = t(`schema.descriptions.${name}`, { defaultValue: '' })
  return translated || fallback
}

export function schemaAltDescription(name: string, fallback: string | undefined, t: TFunction): string | undefined {
  const translated = t(`schema.altDescriptions.${name}`, { defaultValue: '' })
  return translated || fallback
}

export function schemaDisableHint(name: string, fallback: string | undefined, t: TFunction): string | undefined {
  const translated = t(`schema.disableHints.${name}`, { defaultValue: '' })
  return translated || fallback
}
