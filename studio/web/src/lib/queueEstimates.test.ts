import { describe, expect, it } from 'vitest'
import type { Task } from '../api/client'
import { buildTrainingForecast, typicalTrainingDuration } from './queueEstimates'

function task(over: Partial<Task>): Task {
  return {
    id: 1,
    name: 'training',
    config_name: 'train',
    task_type: 'train',
    status: 'pending',
    priority: 0,
    created_at: 1,
    started_at: null,
    finished_at: null,
    pid: null,
    exit_code: null,
    output_dir: null,
    error_msg: null,
    ...over,
  }
}

describe('queue training forecast', () => {
  it('uses the median completed duration for waiting tasks', () => {
    const tasks = [
      task({ id: 1, status: 'done', started_at: 10, finished_at: 110 }),
      task({ id: 2, status: 'done', started_at: 10, finished_at: 310 }),
      task({ id: 3, status: 'done', started_at: 10, finished_at: 210 }),
    ]
    expect(typicalTrainingDuration(tasks)).toBe(200)
  })

  it('orders by scheduler priority and adds estimates cumulatively', () => {
    const tasks = [
      task({ id: 1, status: 'done', started_at: 0.1, finished_at: 100.1 }),
      task({ id: 2, status: 'running', started_at: 900, created_at: 1 }),
      task({ id: 3, priority: 1, created_at: 3 }),
      task({ id: 4, priority: 0, created_at: 2 }),
    ]
    const result = buildTrainingForecast(tasks, { step: 50, total_steps: 100, speed: 1 }, 1000)

    expect(result.map(({ task }) => task.id)).toEqual([2, 3, 4])
    expect(result.map(({ finishesAt }) => finishesAt)).toEqual([1050, 1150, 1250])
  })
})
