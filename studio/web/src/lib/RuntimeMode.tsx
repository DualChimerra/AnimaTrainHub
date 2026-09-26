// RuntimeMode.tsx -- global data layer for the Colab / Local runtime mode (this fork).
//
// The backend's `/api/runtime` is the single source of truth (see
// studio/infrastructure/runtime_mode.py). This layer does three things:
//   1. Fetches the state once on first paint, and turns on `needsPick` when
//      `mode === ''` (the user never chose) and it isn't pinned by env --
//      `<RuntimeModeGate>` pops the picker based on that;
//   2. Exposes `setMode`, shared by the picker and the settings section;
//   3. Broadcasts the `effective` mode to components that need to change
//      copy / behavior by mode (the Topbar badge, the local-path hint in
//      settings, etc.).
//
// A fetch failure does **not** block the app: right after the backend comes
// up, the SPA may be ready before the router is, and the mode only affects
// hints and defaults -- hard-blocking first paint over it isn't worth it. On
// failure, `info` stays null and `needsPick` stays false; the user proceeds
// normally and gets asked again on the next refresh.
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

import { api, type RuntimeInfo, type RuntimeMode } from '../api/client'

interface RuntimeModeValue {
  info: RuntimeInfo | null
  /** The effective mode; falls back to 'local' before state is fetched (local is the overwhelming majority case). */
  mode: RuntimeMode
  /** true = needs to pop the picker (the user has never chosen, and it isn't pinned by ALS_RUNTIME_MODE). */
  needsPick: boolean
  loading: boolean
  error: string | null
  /** Persists the choice; needsPick drops once it succeeds. Throws are left for the caller to display. */
  setMode: (mode: RuntimeMode) => Promise<RuntimeInfo>
  reload: () => Promise<void>
}

const Ctx = createContext<RuntimeModeValue | null>(null)

export function RuntimeModeProvider({ children }: { children: ReactNode }) {
  const [info, setInfo] = useState<RuntimeInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const reload = useCallback(async () => {
    try {
      const next = await api.getRuntime()
      setInfo(next)
      setError(null)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void reload() }, [reload])

  const setMode = useCallback(async (mode: RuntimeMode) => {
    const next = await api.setRuntimeMode(mode)
    setInfo(next)
    setError(null)
    return next
  }, [])

  const value = useMemo<RuntimeModeValue>(() => ({
    info,
    mode: info?.effective ?? 'local',
    // Never asks when locked (injected via env) -- the environment already answered for the user.
    needsPick: !!info && !info.locked && info.mode === '',
    loading,
    error,
    setMode,
    reload,
  }), [info, loading, error, setMode, reload])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useRuntimeMode(): RuntimeModeValue {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useRuntimeMode must be used inside <RuntimeModeProvider>')
  return ctx
}

/** Same as above, but returns null instead of throwing when there's no Provider.
 *
 *  For widgets that might get rendered standalone, outside the Provider --
 *  the Topbar's mode badge and the settings mode section both fall into
 *  this category: in the real app they're always inside the Provider
 *  (main.tsx mounts it at the root), but component tests often render just
 *  a single page/component. Letting a decorative badge crash the whole
 *  page's render is a bad trade-off; these call sites already have to
 *  handle a "state not fetched yet" branch when `info` is null, so one more
 *  source of null adds no extra complexity. */
export function useRuntimeModeOptional(): RuntimeModeValue | null {
  return useContext(Ctx)
}
