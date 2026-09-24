import type { MonitorState, Task } from '../api/client'

export interface TaskForecast {
  task: Task
  finishesAt: number | null
}

const isTrainingTask = (task: Task) => (task.task_type ?? 'train') === 'train'

export function typicalTrainingDuration(tasks: Task[]): number | null {
  const durations = tasks
    .filter((task) => isTrainingTask(task) && task.status === 'done' && task.started_at && task.finished_at)
    .map((task) => task.finished_at! - task.started_at!)
    .filter((seconds) => seconds > 0)
    .sort((a, b) => a - b)

  if (!durations.length) return null
  const middle = Math.floor(durations.length / 2)
  return durations.length % 2
    ? durations[middle]
    : (durations[middle - 1] + durations[middle]) / 2
}

export function buildTrainingForecast(
  tasks: Task[],
  monitor: MonitorState | null,
  now = Date.now() / 1000,
): TaskForecast[] {
  const live = tasks
    .filter((task) => isTrainingTask(task) && (task.status === 'running' || task.status === 'pending'))
    .sort((a, b) => {
      if (a.status !== b.status) return a.status === 'running' ? -1 : 1
      return b.priority - a.priority || a.created_at - b.created_at
    })
  const typical = typicalTrainingDuration(tasks)
  let cursor = now
  let canEstimate = true

  return live.map((task) => {
    let remaining: number | null = typical
    if (task.status === 'running') {
      if (
        monitor?.speed && monitor.speed > 0 &&
        monitor.step != null && monitor.total_steps != null &&
        monitor.total_steps >= monitor.step
      ) {
        remaining = (monitor.total_steps - monitor.step) / monitor.speed
      } else if (typical != null && task.started_at) {
        remaining = Math.max(0, typical - (now - task.started_at))
      }
    }

    if (!canEstimate || remaining == null) {
      canEstimate = false
      return { task, finishesAt: null }
    }
    cursor += remaining
    return { task, finishesAt: cursor }
  })
}

export function latestProjectFinish(tasks: Task[], projectId: number): number | null {
  let latest: number | null = null
  for (const task of tasks) {
    if (!isTrainingTask(task) || task.project_id !== projectId || !task.finished_at) continue
    latest = latest == null ? task.finished_at : Math.max(latest, task.finished_at)
  }
  return latest
}
