import { makeApiError } from '../../../api/client'
import type {
  Board, BoardSummary, GImage, GraphTask, Param, Run, SavedView, TaskSampleInfo,
} from './types'

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const isForm = init?.body instanceof FormData
  const resp = await fetch(path, {
    headers: {
      Accept: 'application/json',
      ...(init?.body && !isForm ? { 'Content-Type': 'application/json' } : {}),
    },
    ...init,
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => null)
    throw makeApiError(resp.status, resp.statusText, body, resp.headers.get('X-Trace-Id'))
  }
  return (await resp.json()) as T
}

const json = (method: string, body?: unknown): RequestInit =>
  ({ method, ...(body !== undefined ? { body: JSON.stringify(body) } : {}) })

export const graphApi = {
  listBoards: () => call<{ items: BoardSummary[] }>('/api/graph/boards').then((r) => r.items),
  createBoard: (body: { name: string; empty?: boolean; copy_params_from?: number }) =>
    call<{ board: Board; runs: Run[] }>('/api/graph/boards', json('POST', body)),
  getBoard: (id: number) => call<{ board: Board; runs: Run[] }>(`/api/graph/boards/${id}`),
  updateBoard: (id: number, patch: Partial<{ name: string; note: string; params: Param[]; views: SavedView[] }>) =>
    call<Board>(`/api/graph/boards/${id}`, json('PATCH', patch)),
  deleteBoard: (id: number) => call<{ ok: boolean }>(`/api/graph/boards/${id}`, json('DELETE')),
  remap: (id: number, paramId: string, mapping: { from: unknown; to: unknown }[]) =>
    call<{ changed: number }>(`/api/graph/boards/${id}/remap`, json('POST', { param_id: paramId, mapping })),

  createRun: (boardId: number, body: Partial<Run>) =>
    call<Run>(`/api/graph/boards/${boardId}/runs`, json('POST', body)),
  updateRun: (id: number, patch: Partial<Run>) => call<Run>(`/api/graph/runs/${id}`, json('PATCH', patch)),
  duplicateRun: (id: number, overrides?: Partial<Run>) =>
    call<Run>(`/api/graph/runs/${id}/duplicate`, json('POST', { overrides })),
  deleteRun: (id: number) => call<{ ok: boolean }>(`/api/graph/runs/${id}`, json('DELETE')),
  runConfigUrl: (id: number) => `/api/graph/runs/${id}/config`,

  uploadImages: (runId: number, files: File[], meta?: Partial<GImage>) => {
    const fd = new FormData()
    for (const f of files) fd.append('files', f, f.name || 'pasted.png')
    if (meta) fd.append('meta', JSON.stringify(meta))
    return call<{ items: GImage[]; errors: { name: string; reason: string }[] }>(
      `/api/graph/runs/${runId}/images`, { method: 'POST', body: fd },
    )
  },
  importSamples: (runId: number, taskId: number, filenames: string[]) =>
    call<{ items: GImage[]; skipped: number }>(
      `/api/graph/runs/${runId}/import-samples`, json('POST', { task_id: taskId, filenames }),
    ),
  updateImage: (id: number, patch: Partial<GImage>) =>
    call<GImage>(`/api/graph/images/${id}`, json('PATCH', patch)),
  deleteImage: (id: number) => call<{ ok: boolean }>(`/api/graph/images/${id}`, json('DELETE')),
  imageUrl: (id: number, w?: number) => `/api/graph/images/${id}/file${w ? `?w=${w}` : ''}`,

  listTasks: (keys: string[]) =>
    call<{ items: GraphTask[] }>(`/api/graph/tasks?keys=${encodeURIComponent(keys.join(','))}`).then((r) => r.items),
  taskSamples: (taskId: number) =>
    call<{ items: TaskSampleInfo[]; total: number }>(`/api/graph/tasks/${taskId}/samples`),
}
