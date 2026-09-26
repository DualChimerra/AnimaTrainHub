/** entryTaskId -- the matching logic for the 0.17 P-H `?task=` deep link to hit
 *  a history entry by task_id. Cache entries carry taskId at the top level;
 *  disk entries get task_id enriched into the PNG's anima_params by the server. */
import { describe, expect, it } from 'vitest'
import { entryTaskId, type HistoryEntry } from './entryAdapter'

describe('entryTaskId', () => {
  it('a cache entry reads the top-level taskId', () => {
    const e = { source: 'cache', taskId: 42 } as HistoryEntry
    expect(entryTaskId(e)).toBe(42)
  })

  it('a disk entry reads params.task_id', () => {
    const e = { source: 'disk', params: { task_id: 7 } } as unknown as HistoryEntry
    expect(entryTaskId(e)).toBe(7)
  })

  it('a disk entry with no task_id -> undefined (old disk images have no enrichment)', () => {
    const e = { source: 'disk', params: {} } as unknown as HistoryEntry
    expect(entryTaskId(e)).toBeUndefined()
  })
})
