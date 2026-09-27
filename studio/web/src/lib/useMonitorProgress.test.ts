import { describe, expect, it } from 'vitest'
import { _mergeDeltaForTest as mergeDelta } from './useMonitorProgress'

describe('mergeDelta (PR #37 delta protocol)', () => {
  it('initial merge populates state from null prev', () => {
    const out = mergeDelta(null, {
      step: 5, total_steps: 100, epoch: 1,
      appended_losses: [{ step: 1, loss: 0.5 }, { step: 5, loss: 0.3 }],
      appended_lr: [{ step: 1, lr: 1e-4 }],
      appended_optimizer_metrics: [{ step: 1, actual_lr: 1e-4, d: 1e-4 }],
      appended_samples: [{ path: '/a.png', step: 5 }],
      config: { model: 'X' },
    })
    expect(out.step).toBe(5)
    expect(out.losses).toHaveLength(2)
    expect(out.lr_history).toHaveLength(1)
    expect(out.optimizer_metrics_history).toHaveLength(1)
    expect(out.optimizer_metrics_history?.[0].d).toBe(1e-4)
    expect(out.samples).toHaveLength(1)
    expect(out.config).toEqual({ model: 'X' })
  })

  it('appends new losses to existing array', () => {
    const out = mergeDelta(
      {
        step: 1, losses: [{ step: 1, loss: 0.5 }], lr_history: [], samples: [],
      },
      {
        step: 3,
        appended_losses: [{ step: 2, loss: 0.4 }, { step: 3, loss: 0.3 }],
      },
    )
    expect(out.losses).toHaveLength(3)
    expect(out.losses?.map((l) => l.step)).toEqual([1, 2, 3])
    expect(out.step).toBe(3)
  })

  it('dedups losses whose step <= last known step', () => {
    // Simulates a reconnect scenario: the snapshot already has steps 1-3, and
    // the delta re-pushes steps 2-4
    const out = mergeDelta(
      {
        step: 3,
        losses: [
          { step: 1, loss: 0.5 },
          { step: 2, loss: 0.4 },
          { step: 3, loss: 0.3 },
        ],
        lr_history: [],
        samples: [],
      },
      {
        step: 4,
        appended_losses: [
          { step: 2, loss: 0.4 },
          { step: 3, loss: 0.3 },
          { step: 4, loss: 0.2 },
        ],
      },
    )
    // Only keeps > 3, i.e. step 4
    expect(out.losses).toHaveLength(4)
    expect(out.losses?.map((l) => l.step)).toEqual([1, 2, 3, 4])
  })

  it('appends and dedups optimizer metrics by step', () => {
    const out = mergeDelta(
      {
        step: 2,
        losses: [],
        lr_history: [],
        optimizer_metrics_history: [
          { step: 1, actual_lr: 1e-4, d: 1e-4 },
          { step: 2, actual_lr: 2e-4, d: 2e-4 },
        ],
        samples: [],
      },
      {
        step: 3,
        appended_optimizer_metrics: [
          { step: 2, actual_lr: 2e-4, d: 2e-4 },
          { step: 3, actual_lr: 3e-4, d: 3e-4 },
        ],
      },
    )
    expect(out.optimizer_metrics_history).toHaveLength(3)
    expect(out.optimizer_metrics_history?.map((m) => m.step)).toEqual([1, 2, 3])
    expect(out.optimizer_metrics_history?.[2].d).toBe(3e-4)
  })

  it('dedups samples by (step, path) tuple', () => {
    const out = mergeDelta(
      {
        samples: [{ path: '/a.png', step: 1 }, { path: '/b.png', step: 1 }],
        losses: [], lr_history: [],
      },
      {
        appended_samples: [
          { path: '/b.png', step: 1 },  // dup
          { path: '/c.png', step: 1 },  // same step, different path -> new
          { path: '/d.png', step: 2 },  // new step
        ],
      },
    )
    expect(out.samples).toHaveLength(4)
    expect(out.samples?.map((s) => s.path)).toEqual(['/a.png', '/b.png', '/c.png', '/d.png'])
  })

  it('caps losses at MAX_LOSSES=50000 (matches backend disk cap)', () => {
    // Long training run + full snapshot scenario: the old 5000 cap would
    // immediately slice off history. Now aligned with the backend
    // train_monitor's 50000 cap; the frontend only trims as a last resort
    // when volume truly explodes.
    const prev = {
      losses: Array.from({ length: 49500 }, (_, i) => ({ step: i, loss: 0.0 })),
      lr_history: [],
      samples: [],
    }
    const out = mergeDelta(prev, {
      appended_losses: Array.from({ length: 1000 }, (_, i) => ({ step: 49500 + i, loss: 0.0 })),
    })
    expect(out.losses).toHaveLength(50000)
    // Keeps the tail
    expect(out.losses?.[0].step).toBe(500)
    expect(out.losses?.[49999].step).toBe(50499)
  })

  it('keeps 10k step training intact without truncation', () => {
    // Regression for a bug where a cold-start full fetch was immediately
    // truncated: 10k history + one or two deltas should never be sliced.
    const prev = {
      losses: Array.from({ length: 10000 }, (_, i) => ({ step: i, loss: 0.0 })),
      lr_history: [],
      samples: [],
    }
    const out = mergeDelta(prev, {
      appended_losses: [{ step: 10000, loss: 0.1 }, { step: 10001, loss: 0.1 }],
    })
    expect(out.losses).toHaveLength(10002)
    expect(out.losses?.[0].step).toBe(0)
    expect(out.losses?.[10001].step).toBe(10001)
  })

  it('caps samples at MAX_SAMPLES=50 (same as backend)', () => {
    const prev = {
      losses: [], lr_history: [],
      samples: Array.from({ length: 45 }, (_, i) => ({ path: `/p${i}`, step: i })),
    }
    const out = mergeDelta(prev, {
      appended_samples: Array.from({ length: 10 }, (_, i) => ({ path: `/n${i}`, step: 45 + i })),
    })
    expect(out.samples).toHaveLength(50)
    expect(out.samples?.[0].path).toBe('/p5')  // head got trimmed
    expect(out.samples?.[49].path).toBe('/n9')
  })

  it('replaces scalar fields each merge', () => {
    const out = mergeDelta(
      { step: 1, speed: 0.5, losses: [], lr_history: [], samples: [] },
      { step: 10, speed: 2.0 },
    )
    expect(out.step).toBe(10)
    expect(out.speed).toBe(2.0)
  })

  it('keeps old config when delta has no config', () => {
    const out = mergeDelta(
      { config: { rank: 32 }, losses: [], lr_history: [], samples: [] },
      { step: 5 },
    )
    expect(out.config).toEqual({ rank: 32 })
  })

  it('keeps task project/version metadata across deltas', () => {
    const out = mergeDelta(
      {
        task_id: 12,
        project_id: 28,
        project_slug: 'xi410',
        version_id: 3,
        version_label: 'v1',
        losses: [],
        lr_history: [],
        samples: [],
      },
      { step: 6 },
    )
    expect(out.project_id).toBe(28)
    expect(out.project_slug).toBe('xi410')
    expect(out.version_id).toBe(3)
    expect(out.version_label).toBe('v1')
  })
})
