// SettingsData.tsx -- the global Settings data layer.
//
// Lifts secrets / catalog / downloadBusy / the SSE subscription out of
// SettingsPage into a root-level Provider, so SettingsPage itself can
// unmount + remount without paying the cost of refetching data:
// - secrets: fetched once, lives in context; updated via setSecrets by
//   SettingsPage after a save
// - catalog: reloadCatalog + the model_download_changed SSE subscription
//   live persistently, shared with the download components
// - downloadBusy: the in-flight Set paired with startDownload
//
// This layer only holds data, it renders no UI. SettingsPage unmounts when
// SettingsDrawer closes; on the next open it renders instantly since the
// data is already in context.
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { useTranslation } from 'react-i18next'
import { api, type ModelsCatalog, type Secrets, type SecretsPatch } from '../api/client'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import { useEventStream } from './useEventStream'

// Global "saved" status indicator (replaces the old save button's dirty state under instant-apply).
export type SaveStatus =
  | { state: 'idle' }
  | { state: 'saving' }
  | { state: 'saved'; at: number }
  | { state: 'error'; error: string }

// Shallow-merges a single-field patch into local secrets (for optimistic
// updates). secrets is a two-level structure (section -> fields); top-level
// scalar fields (like download_source) are overwritten directly.
function mergePatchLocal(base: Secrets, patch: SecretsPatch): Secrets {
  const out = { ...base } as Record<string, unknown>
  for (const key of Object.keys(patch)) {
    const pv = (patch as Record<string, unknown>)[key]
    const bv = out[key]
    if (pv && typeof pv === 'object' && !Array.isArray(pv)
        && bv && typeof bv === 'object' && !Array.isArray(bv)) {
      out[key] = { ...(bv as object), ...(pv as object) }
    } else {
      out[key] = pv
    }
  }
  return out as unknown as Secrets
}

interface SettingsData {
  secrets: Secrets | null
  secretsError: string | null
  setSecrets: (s: Secrets) => void
  /** Unified write entry point for instant-apply: optimistic update + a serialized single-field PUT patch. */
  commitSecrets: (patch: SecretsPatch) => void
  /** Wraps a one-off immediate PUT (independent saves like download source / main model / upscaler), driving the saveStatus indicator. */
  runSave: <T>(fn: () => Promise<T>) => Promise<T>
  saveStatus: SaveStatus
  catalog: ModelsCatalog | null
  catalogError: string | null
  reloadCatalog: () => Promise<ModelsCatalog | null>
  downloadBusy: Set<string>
  startDownload: (model_id: string, variant?: string) => Promise<void>
  /** The inverse of downloading (confirm -> DELETE -> refresh catalog), shared by every section of the download center. */
  deleteAsset: (model_id: string, variant: string | undefined, name: string) => Promise<void>
  setDownloadSource: (type: string, source: string) => Promise<void>
}

const Ctx = createContext<SettingsData | null>(null)

export function SettingsDataProvider({ children }: { children: ReactNode }) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const dialog = useDialog()
  const [secrets, setSecrets] = useState<Secrets | null>(null)
  const [secretsError, setSecretsError] = useState<string | null>(null)
  const [catalog, setCatalog] = useState<ModelsCatalog | null>(null)
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const [downloadBusy, setDownloadBusy] = useState<Set<string>>(new Set())
  const [saveStatus, setSaveStatus] = useState<SaveStatus>({ state: 'idle' })
  // Serialized PUT queue + in-flight counter: preserves ordering to avoid a backend read-modify-write race.
  const saveQueueRef = useRef<Promise<unknown>>(Promise.resolve())
  const pendingRef = useRef(0)

  useEffect(() => {
    api.getSecrets()
      .then((s) => { setSecrets(s); setSecretsError(null) })
      .catch((e) => setSecretsError(String(e)))
  }, [])

  const reloadCatalog = useCallback(async (): Promise<ModelsCatalog | null> => {
    try {
      const c = await api.getModelsCatalog()
      setCatalog(c)
      setCatalogError(null)
      return c
    } catch (e) {
      setCatalogError(String(e))
      return null
    }
  }, [])

  useEffect(() => { void reloadCatalog() }, [reloadCatalog])

  // model_download_changed both drives the catalog refresh and is the only
  // global signal for a download failure: downloads run on a background
  // thread, and the failure reason (e.g. a gated repo missing a token) only
  // lands in download status -- it never makes startDownload's await throw.
  // On failed, pop an error toast surfacing the backend's actionable
  // message directly to the user -- otherwise all they see is a red badge
  // on the card, with the actual reason buried in a collapsed "download
  // log" on another tab (or only in the terminal).
  useEventStream((evt) => {
    if (evt.type !== 'model_download_changed') return
    void reloadCatalog().then((c) => {
      if (evt.status !== 'failed' || !c) return
      const key = String(evt.key ?? '')
      const dl = c.downloads[key]
      toast(dl?.message || t('settings.downloadFailed', { error: key }), 'error')
    })
  })

  const startDownload = useCallback(async (model_id: string, variant?: string) => {
    const key = variant ? `${model_id}:${variant}` : model_id
    setDownloadBusy((s) => new Set(s).add(key))
    try {
      await api.startModelDownload({ model_id, variant })
      toast(t('settings.downloadStarted', { name: key }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setDownloadBusy((s) => { const n = new Set(s); n.delete(key); return n })
    }
  }, [reloadCatalog, t, toast])

  // Deletes a downloaded asset (the inverse of a download): confirm -> DELETE -> refresh catalog.
  // Shared by every section of the download center (training models / tagging / eval / upscaler).
  const deleteAsset = useCallback(async (model_id: string, variant: string | undefined, name: string) => {
    if (!(await dialog.confirm(t('settings.confirmDeleteAsset', { name }), { tone: 'danger' }))) return
    try {
      await api.deleteModelAsset(model_id, variant)
      toast(t('settings.assetDeleted', { name }), 'success')
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [dialog, reloadCatalog, t, toast])

  // Picking a download source by type: saved instantly (an immediate action
  // like "download" / models.root, not part of the form draft). Deliberately
  // does not call setSecrets -- otherwise it would desync SettingsPage's
  // draft/server state, and the form's Save would clobber this change back.
  // The dropdown's current value reads from catalog (refreshed via
  // reloadCatalog), independent of the form's secrets.
  // Immediate-save wrapper: also drives the top-right saveStatus indicator
  // for independent PUTs that don't go through the commitSecrets queue
  // (toggles like download source / main model / upscaler / auto_sync),
  // keeping feedback consistent.
  const runSave = useCallback(async <T,>(fn: () => Promise<T>): Promise<T> => {
    setSaveStatus({ state: 'saving' })
    try {
      const r = await fn()
      setSaveStatus({ state: 'saved', at: Date.now() })
      return r
    } catch (e) {
      setSaveStatus({ state: 'error', error: String(e) })
      throw e
    }
  }, [])

  const setDownloadSource = useCallback(async (type: string, source: string) => {
    try {
      await runSave(() => api.updateSecrets({ download_sources: { [type]: source } }))
      await reloadCatalog()
    } catch (e) {
      toast(String(e), 'error')
    }
  }, [runSave, reloadCatalog, toast])

  // Unified instant-apply write: optimistically updates local secrets so
  // controls reflect the change immediately, and queues a single-field PUT
  // patch onto the serial queue. Once the queue fully drains, writes back
  // once with the backend's authoritative result (which carries validator
  // normalization + sensitive-field masking) -- this avoids an earlier PUT's
  // authoritative result overwriting a later field's optimistic value
  // (a mid-flight flicker) when multiple fields change in a row. On
  // failure, refetches secrets to restore consistency.
  const commitSecrets = useCallback((patch: SecretsPatch) => {
    setSecrets((s) => (s ? mergePatchLocal(s, patch) : s))
    setSaveStatus({ state: 'saving' })
    pendingRef.current += 1
    saveQueueRef.current = saveQueueRef.current
      .then(() => api.updateSecrets(patch))
      .then((authoritative) => {
        pendingRef.current -= 1
        // Refresh the "saved" timestamp on every PUT completion, so
        // back-to-back saves each get visible feedback; the authoritative
        // result is only written back once the queue drains, to avoid
        // overwriting later fields' optimistic values mid-flight.
        if (pendingRef.current === 0) setSecrets(authoritative)
        setSaveStatus({ state: 'saved', at: Date.now() })
      })
      .catch((e) => {
        pendingRef.current -= 1
        setSaveStatus({ state: 'error', error: String(e) })
        toast(String(e), 'error')
        api.getSecrets().then((s) => setSecrets(s)).catch(() => {})
      })
  }, [toast])

  return (
    <Ctx.Provider value={{
      secrets, secretsError, setSecrets, commitSecrets, runSave, saveStatus,
      catalog, catalogError, reloadCatalog,
      downloadBusy, startDownload, deleteAsset, setDownloadSource,
    }}>
      {children}
    </Ctx.Provider>
  )
}

export function useSettingsData(): SettingsData {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useSettingsData must be used inside <SettingsDataProvider>')
  return ctx
}
