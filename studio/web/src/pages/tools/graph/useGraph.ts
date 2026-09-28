import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ApiError } from '../../../api/client'
import { useToast } from '../../../components/Toast'
import { graphApi } from './api'
import { configKeys, normalizeView, statusFromTask } from './model'
import type { Board, BoardSummary, GImage, GraphTask, Param, Run, ViewState } from './types'

const BOARD_KEY = 'studio:graph:board'
const viewKey = (id: number) => `studio:graph:view:${id}`

function readJson<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : null
  } catch {
    return null
  }
}
function writeJson(key: string, v: unknown) {
  try { localStorage.setItem(key, JSON.stringify(v)) } catch { /* quota / private mode */ }
}

const TASK_POLL_MS = 15000

/** Everything the Graph page needs: boards, the open board, its cards, the
 *  queue tasks it recognises, and the persisted view. */
export function useGraph() {
  const { toast } = useToast()
  const [boards, setBoards] = useState<BoardSummary[] | null>(null)
  const [boardId, setBoardIdState] = useState<number | null>(() => readJson<number>(BOARD_KEY))
  const [board, setBoard] = useState<Board | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [loading, setLoading] = useState(true)
  const [tasks, setTasks] = useState<GraphTask[]>([])
  const [view, setViewState] = useState<ViewState | null>(null)

  const fail = useCallback((e: unknown, fallback: string) => {
    toast((e as ApiError)?.message || fallback, 'error')
  }, [toast])

  const refreshBoards = useCallback(async () => {
    try {
      const list = await graphApi.listBoards()
      setBoards(list)
      return list
    } catch (e) {
      fail(e, 'Failed to load boards')
      setBoards([])
      return []
    }
  }, [fail])

  const setBoardId = useCallback((id: number | null) => {
    setBoardIdState(id)
    writeJson(BOARD_KEY, id)
  }, [])

  // First load: boards, then pick the remembered one (or the newest).
  useEffect(() => {
    void refreshBoards().then((list) => {
      setBoardIdState((cur) => {
        if (cur != null && list.some((b) => b.id === cur)) return cur
        return list[0]?.id ?? null
      })
      if (!list.length) setLoading(false)
    })
  }, [refreshBoards])

  const loadBoard = useCallback(async (id: number) => {
    setLoading(true)
    try {
      const r = await graphApi.getBoard(id)
      setBoard(r.board)
      setRuns(r.runs)
      setViewState(normalizeView(readJson<ViewState>(viewKey(id)), r.board.params))
    } catch (e) {
      fail(e, 'Failed to load board')
      setBoard(null)
      setRuns([])
    } finally {
      setLoading(false)
    }
  }, [fail])

  useEffect(() => {
    if (boardId == null) { setBoard(null); setRuns([]); return }
    void loadBoard(boardId)
  }, [boardId, loadBoard])

  const setView = useCallback((next: ViewState | ((v: ViewState) => ViewState)) => {
    setViewState((prev) => {
      if (!prev) return prev
      const v = typeof next === 'function' ? next(prev) : next
      if (board) writeJson(viewKey(board.id), v)
      return v
    })
  }, [board])

  // ── queue tasks (read-only, polled while the page is visible) ─────────────
  const keys = useMemo(() => (board ? configKeys(board.params) : []), [board])
  const keysSig = keys.join(',')
  const tasksSig = useRef('')
  const refreshTasks = useCallback(async () => {
    if (!keysSig) { setTasks([]); return }
    try {
      const next = await graphApi.listTasks(keysSig.split(','))
      // The poll usually returns the same list; keep the old array so nothing re-renders.
      const sig = JSON.stringify(next)
      if (sig !== tasksSig.current) {
        tasksSig.current = sig
        setTasks(next)
      }
    } catch {
      /* the queue bridge is optional; the board works without it */
    }
  }, [keysSig])
  useEffect(() => {
    void refreshTasks()
    const id = window.setInterval(() => {
      if (document.visibilityState === 'visible') void refreshTasks()
    }, TASK_POLL_MS)
    return () => window.clearInterval(id)
  }, [refreshTasks])

  // ── mutations ─────────────────────────────────────────────────────────────
  const replaceRun = useCallback((r: Run) => {
    setRuns((list) => list.some((x) => x.id === r.id) ? list.map((x) => (x.id === r.id ? r : x)) : [...list, r])
  }, [])

  const createRun = useCallback(async (body: Partial<Run>) => {
    if (!board) return null
    try {
      const r = await graphApi.createRun(board.id, body)
      replaceRun(r)
      return r
    } catch (e) { fail(e, 'Failed to create card'); return null }
  }, [board, replaceRun, fail])

  const updateRun = useCallback(async (id: number, patch: Partial<Run>) => {
    // Optimistic: the card changes at once, the server answer settles it.
    setRuns((list) => list.map((x) => (x.id === id ? { ...x, ...patch } : x)))
    try {
      const r = await graphApi.updateRun(id, patch)
      replaceRun(r)
      return r
    } catch (e) {
      fail(e, 'Failed to save')
      if (board) void loadBoard(board.id)
      return null
    }
  }, [replaceRun, fail, board, loadBoard])

  const duplicateRun = useCallback(async (id: number) => {
    try {
      const r = await graphApi.duplicateRun(id)
      replaceRun(r)
      return r
    } catch (e) { fail(e, 'Failed to copy'); return null }
  }, [replaceRun, fail])

  const deleteRun = useCallback(async (id: number) => {
    try {
      await graphApi.deleteRun(id)
      setRuns((list) => list.filter((x) => x.id !== id))
      return true
    } catch (e) { fail(e, 'Failed to delete'); return false }
  }, [fail])

  const patchImages = useCallback((runId: number, fn: (imgs: GImage[]) => GImage[]) => {
    setRuns((list) => list.map((x) => (x.id === runId ? { ...x, images: fn(x.images) } : x)))
  }, [])

  const uploadImages = useCallback(async (runId: number, files: File[], meta?: Partial<GImage>) => {
    if (!files.length) return []
    try {
      const r = await graphApi.uploadImages(runId, files, meta)
      patchImages(runId, (imgs) => [...imgs, ...r.items])
      for (const err of r.errors) toast(`${err.name}: ${err.reason}`, 'error')
      return r.items
    } catch (e) { fail(e, 'Upload failed'); return [] }
  }, [patchImages, toast, fail])

  const importSamples = useCallback(async (runId: number, taskId: number, names: string[]) => {
    try {
      const r = await graphApi.importSamples(runId, taskId, names)
      patchImages(runId, (imgs) => [...imgs, ...r.items])
      return r
    } catch (e) { fail(e, 'Import failed'); return null }
  }, [patchImages, fail])

  const updateImage = useCallback(async (img: GImage, patch: Partial<GImage>) => {
    patchImages(img.run_id, (imgs) => imgs.map((i) => (i.id === img.id ? { ...i, ...patch } : i)))
    try {
      const r = await graphApi.updateImage(img.id, patch)
      patchImages(img.run_id, (imgs) => imgs.map((i) => (i.id === r.id ? r : i)))
    } catch (e) { fail(e, 'Failed to save') }
  }, [patchImages, fail])

  const deleteImage = useCallback(async (img: GImage) => {
    try {
      await graphApi.deleteImage(img.id)
      patchImages(img.run_id, (imgs) => imgs.filter((i) => i.id !== img.id))
    } catch (e) { fail(e, 'Failed to delete') }
  }, [patchImages, fail])

  const saveParams = useCallback(async (params: Param[]) => {
    if (!board) return false
    try {
      const b = await graphApi.updateBoard(board.id, { params })
      setBoard(b)
      return true
    } catch (e) { fail(e, 'Failed to save parameters'); return false }
  }, [board, fail])

  const saveBoardMeta = useCallback(async (patch: Partial<Pick<Board, 'name' | 'note' | 'views'>>) => {
    if (!board) return
    try {
      const b = await graphApi.updateBoard(board.id, patch)
      setBoard(b)
      void refreshBoards()
    } catch (e) { fail(e, 'Failed to save') }
  }, [board, fail, refreshBoards])

  // ── keep linked cards' status in step with their queue task ───────────────
  const syncing = useRef(new Set<number>())
  useEffect(() => {
    if (!tasks.length) return
    const byId = new Map(tasks.map((t) => [t.id, t]))
    for (const r of runs) {
      if (r.task_id == null || syncing.current.has(r.id)) continue
      const task = byId.get(r.task_id)
      if (!task) continue
      const s = statusFromTask(task.status)
      if (s !== r.status) {
        syncing.current.add(r.id)
        void updateRun(r.id, { status: s }).finally(() => syncing.current.delete(r.id))
      }
    }
  }, [tasks, runs, updateRun])

  return {
    boards, boardId, setBoardId, refreshBoards, board, setBoard, runs, setRuns, loading, loadBoard,
    tasks, refreshTasks, view, setView,
    createRun, updateRun, duplicateRun, deleteRun,
    uploadImages, importSamples, updateImage, deleteImage,
    saveParams, saveBoardMeta,
  }
}

export type GraphState = ReturnType<typeof useGraph>
