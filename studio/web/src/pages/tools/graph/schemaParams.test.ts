import { expect, it } from 'vitest'
import i18n from '../../../i18n'
import { configPatch, taskMatches, valuesFromConfig } from './model'
import { paramFromSchema } from './schemaParams'

it('preserves false, nullable auto, numeric lists and free text from a training config', () => {
  const params = [
    paramFromSchema('cache_latents', { type: 'boolean' }, i18n.t),
    paramFromSchema('optional', { anyOf: [{ type: 'boolean' }, { type: 'null' }] }, i18n.t),
    paramFromSchema('resolution', { type: 'array', items: { type: 'integer' }, default: [1024] }, i18n.t),
    paramFromSchema('sample_prompt', { type: 'string' }, i18n.t),
  ]
  const cfg = { cache_latents: false, optional: null, resolution: [768], sample_prompt: 'a cat' }
  const values = valuesFromConfig(cfg, params)
  expect(Object.values(values)).toEqual(['no', 'auto', 768, 'a cat'])
  expect(configPatch(values, params)).toEqual({ patch: cfg, skipped: [] })
  expect(taskMatches(values, values, params)).toBe(true)
})

it('reports malformed structured text instead of writing an invalid config', () => {
  const p = paramFromSchema('sample_prompts', { type: 'array', items: { type: 'string' } }, i18n.t)
  expect(configPatch({ [p.id]: '[broken' }, [p])).toEqual({ patch: {}, skipped: [p.id] })
})
