import { describe, expect, it } from 'vitest'
import { evalProgressFromResults } from './useEvalProgress'
import type { EvalMetricResult } from '../api/client'

function result(status: string, metricStatuses: Record<string, string> = {}): EvalMetricResult {
  return {
    schema_version: 1,
    has_metrics: true,
    status,
    run_id: 'r',
    metrics: {},
    metric_states: Object.fromEntries(
      Object.entries(metricStatuses).map(([k, s]) => [k, { key: k, status: s, value: null }]),
    ),
  } as EvalMetricResult
}

describe('evalProgressFromResults', () => {
  it('empty results = no evaluation', () => {
    expect(evalProgressFromResults([])).toEqual({ active: false, done: 0, total: 0 })
  })

  it('run-level running counts as active (covers the image-gen phase, before metrics start)', () => {
    expect(evalProgressFromResults([result('running')])).toEqual({ active: true, done: 0, total: 1 })
  })

  it('a pending metric counts as active (covers the queued/computing-metrics phase)', () => {
    expect(
      evalProgressFromResults([result('done', { clip_t: 'pending', clip_i: 'done', dino_i: 'done' })]),
    ).toEqual({ active: true, done: 0, total: 1 })
  })

  it('all terminal states = not active, counts done', () => {
    expect(
      evalProgressFromResults([
        result('done', { clip_t: 'done', clip_i: 'done', dino_i: 'done' }),
        result('done', { clip_t: 'done', clip_i: 'failed', dino_i: 'done' }),
      ]),
    ).toEqual({ active: false, done: 2, total: 2 })
  })

  it('mixed: one still evaluating, one finished -> active with done=1/total=2', () => {
    expect(
      evalProgressFromResults([
        result('done', { clip_t: 'done', clip_i: 'done', dino_i: 'done' }),
        result('running', { clip_t: 'running' }),
      ]),
    ).toEqual({ active: true, done: 1, total: 2 })
  })
})
