import { useCallback, useRef, useState } from 'react'

/** The minimal common shape of Job / Task -- the replay guard only looks at these three fields. */
interface ReplayableItem {
  id: number
  status: string
  version_id?: number | null
}

export const splitLog = (log: string): string[] => (log ? log.split('\n') : [])

/**
 * "Latest task + log replay" state container (shared by Tagging / Regularization).
 *
 * On page entry / SSE reconnect (onOpen), calls refresh() to hydrate the
 * most recent task and its full log from the server; incremental SSE log
 * lines keep flowing through the usual setItem / setLogs append.
 *
 * refresh has three layers of anti-regression guards:
 * 1. While locally tracking a running/pending task, don't let it get
 *    clobbered by a result for a different id
 * 2. For the same id, don't overwrite if the server's log is shorter than
 *    the local one (the file write to disk can lag behind)
 * 3. When the server has no task, only clear leftover state that "belongs
 *    to a different version"
 *
 * onHydrated fires whenever refresh actually rewrites state (passing null on
 * clear), letting the caller keep derived state (like aiBusy) in sync.
 * fetchLatest / onHydrated go through refs, so the caller doesn't need to
 * memoize them.
 */
export function useLatestJobReplay<T extends ReplayableItem>(
  vid: number | null,
  fetchLatest: (vid: number) => Promise<{ item: T | null; log: string }>,
  onHydrated?: (item: T | null) => void,
) {
  const [item, setItem] = useState<T | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const itemRef = useRef<T | null>(null)
  const logsRef = useRef<string[]>([])
  const itemIdRef = useRef<number | null>(null)
  itemRef.current = item
  logsRef.current = logs
  itemIdRef.current = item?.id ?? null
  const fetchRef = useRef(fetchLatest)
  fetchRef.current = fetchLatest
  const onHydratedRef = useRef(onHydrated)
  onHydratedRef.current = onHydrated

  const refresh = useCallback(async () => {
    if (!vid) return
    try {
      const r = await fetchRef.current(vid)
      if (!r.item) {
        if (itemRef.current?.version_id !== vid) {
          setItem(null)
          setLogs([])
          onHydratedRef.current?.(null)
        }
        return
      }
      const current = itemRef.current
      if (
        current?.version_id === vid &&
        current.id !== r.item.id &&
        (current.status === 'pending' || current.status === 'running')
      ) return
      const hydratedLogs = splitLog(r.log)
      if (
        current?.version_id === vid &&
        current.id === r.item.id &&
        logsRef.current.length > hydratedLogs.length
      ) return
      setItem(r.item)
      setLogs(hydratedLogs)
      onHydratedRef.current?.(r.item)
    } catch { /* a hydrate failure shouldn't block the page */ }
  }, [vid])

  return { item, logs, setItem, setLogs, itemIdRef, refresh }
}
