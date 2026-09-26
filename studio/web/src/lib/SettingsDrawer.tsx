// SettingsDrawer.tsx -- the global open/close store for the Settings drawer.
//
// The old routed settings page (/tools/settings) has been removed; every
// "open settings" action now goes through useSettingsDrawer().open({
// section? }). State lives in memory and defaults to closed on page
// refresh, matching the user's mental model of a "drawer".
//
// dirty guard: SettingsPage registers a query function via
// registerDirtyGuard; the drawer calls it before closing to decide whether
// to pop a confirm. This keeps the guard logic on the form's side (which
// already knows whether it's dirty), while the drawer's job is just to ask
// the user.
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
import { useDialog } from '../components/Dialog'

interface OpenOptions {
  /** Jumps to the given section (matches a DOM id inside SettingsPage); keeps the last tab if omitted. */
  section?: string
}

interface SettingsDrawerApi {
  isOpen: boolean
  /** The section passed to the last open() call; an effect inside
   *  SettingsPage watches it to scrollIntoView. Every open() gets a new
   *  reference even for the same section, so the effect always fires. */
  sectionRequest: { section: string; nonce: number } | null
  open: (opts?: OpenOptions) => void
  close: () => void
  /** SettingsPage registers a query function for "is the current draft dirty" on mount; passes null on unmount. */
  registerDirtyGuard: (fn: (() => boolean) | null) => void
}

const Ctx = createContext<SettingsDrawerApi | null>(null)

export function SettingsDrawerProvider({ children }: { children: ReactNode }) {
  const { t } = useTranslation()
  const { confirm } = useDialog()
  const [isOpen, setIsOpen] = useState(false)
  const [sectionRequest, setSectionRequest] = useState<SettingsDrawerApi['sectionRequest']>(null)
  const dirtyGuardRef = useRef<(() => boolean) | null>(null)
  const nonceRef = useRef(0)

  const open = useCallback((opts?: OpenOptions) => {
    // Reset sectionRequest on every open: explicitly clear it when no section
    // is passed, so a section left over from a previous open doesn't get
    // consumed again by a stale effect the next time it opens with "no
    // section -> SettingsPage remounts", causing an unexpected scroll.
    if (opts?.section) {
      nonceRef.current += 1
      setSectionRequest({ section: opts.section, nonce: nonceRef.current })
    } else {
      setSectionRequest(null)
    }
    setIsOpen(true)
  }, [])

  const close = useCallback(async () => {
    if (dirtyGuardRef.current?.()) {
      const ok = await confirm(t('settings.closeDirtyConfirm'), {
        tone: 'warn',
        title: t('settings.closeDirtyTitle'),
        okText: t('settings.closeDirtyOk'),
      })
      if (!ok) return
    }
    setIsOpen(false)
  }, [confirm, t])

  const registerDirtyGuard = useCallback((fn: (() => boolean) | null) => {
    dirtyGuardRef.current = fn
  }, [])

  // Close on ESC: lives in the Provider rather than the Drawer component, so the shortcut still works during lazy loading.
  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        void close()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isOpen, close])

  return (
    <Ctx.Provider value={{ isOpen, sectionRequest, open, close, registerDirtyGuard }}>
      {children}
    </Ctx.Provider>
  )
}

export function useSettingsDrawer(): SettingsDrawerApi {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useSettingsDrawer must be used inside <SettingsDrawerProvider>')
  return ctx
}
