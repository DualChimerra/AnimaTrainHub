// SettingsDrawer.tsx -- the shell for the settings drawer that slides in from
// the right.
//
// Rendering strategy (solves the "mounting 1000+ lines of Settings on open
// causes a frame stall" problem):
//
// 1. Code splitting (React.lazy): Settings isn't in the first-load bundle,
//    it's only downloaded on first open
// 2. Skeleton-first: the drawer shell slides in immediately showing a
//    skeleton; Settings is only actually mounted on the next frame (rAF).
//    The animation runs on the GPU (transform + opacity), so the mount
//    stall is masked by the animation
// 3. Symmetric close: when isOpen=false, Settings is unmounted immediately,
//    then the slide-out plays; this avoids React unmounting 8 tabs' worth
//    of component trees fighting the animation for the main thread
//
// Settings data (secrets / catalog / SSE) lives persistently in
// SettingsDataProvider, so mounting/unmounting Settings here only costs a
// React render, not a data refetch.
import { lazy, Suspense, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useSettingsDrawer } from '../lib/SettingsDrawer'

const SettingsPageLazy = lazy(() => import('../pages/tools/Settings'))

// Width: matches the Claude Design prototype's compact right drawer
// (prototype is 520px). We give real, dense settings content a bit more
// room -> 600px, with 94vw as the small-screen fallback. Much narrower than
// the old 1024px, single-column, no more PAGE INDEX (the prototype drawer
// doesn't have one) -- closer to the prototype's chrome.
const DRAWER_WIDTH_CLASS = 'w-[min(920px,calc(100vw-40px))]'
const ANIM_MS = 220

export default function SettingsDrawer() {
  const { isOpen, close } = useSettingsDrawer()
  const [contentReady, setContentReady] = useState(false)
  const [shouldRender, setShouldRender] = useState(isOpen)
  const [active, setActive] = useState(isOpen)

  useEffect(() => {
    if (isOpen) {
      setShouldRender(true)
    } else {
      setActive(false)
      const timer = setTimeout(() => {
        setShouldRender(false)
      }, ANIM_MS)
      return () => clearTimeout(timer)
    }
  }, [isOpen])

  useEffect(() => {
    if (shouldRender && isOpen) {
      const id = requestAnimationFrame(() => {
        setActive(true)
      })
      return () => cancelAnimationFrame(id)
    }
  }, [shouldRender, isOpen])

  // open: let the shell slide in first (GPU animation), mount Settings on the next frame
  // close: unmount Settings immediately, the shell's slide-out plays over the empty skeleton
  useEffect(() => {
    if (!isOpen) {
      setContentReady(false)
      return
    }
    const id = requestAnimationFrame(() => setContentReady(true))
    return () => cancelAnimationFrame(id)
  }, [isOpen])

  if (!shouldRender) return null

  return (
    <div
      aria-hidden={!isOpen}
      className={`fixed inset-0 z-40 ${active ? '' : 'pointer-events-none'}`}
    >
      {/* backdrop: absolute, fills the viewport, so the underlying page never
       *  peeks through on the right while the panel is mid slide-in. It used
       *  to be flex + flex-1, which only covered the strip left of the
       *  panel; the panel's own slot on the right went uncovered during the
       *  slide -> the page underneath showed through / flashed black-white. */}
      <div
        onClick={() => void close()}
        className="absolute inset-0 transition-colors ease-out"
        style={{
          background: active ? 'rgba(20,22,20,.18)' : 'transparent',
          transitionDuration: `${ANIM_MS}ms`,
        }}
        aria-label="close settings"
      />
      <aside
        role="dialog"
        aria-modal="true"
        className={`ds-drawer transition-transform ease-out m-drawer-full ${DRAWER_WIDTH_CLASS} ${
          active ? 'translate-x-0' : 'translate-x-full'
        }`}
        style={{ transitionDuration: `${ANIM_MS}ms` }}
      >
        {contentReady && isOpen ? (
          <Suspense fallback={<DrawerSkeleton />}>
            <SettingsPageLazy />
          </Suspense>
        ) : (
          <DrawerSkeleton />
        )}
      </aside>
    </div>
  )
}

function DrawerSkeleton() {
  const { t } = useTranslation()
  return (
    <div className="flex flex-col h-full">
      {/* Same-structure placeholder bar as PageHeader -- height matches, so
          switching from slide-in to real content doesn't jump */}
      <div className="px-6 pt-5 pb-4 bg-canvas border-b border-subtle">
        <div className="h-7 w-24 rounded bg-overlay" />
        <div className="mt-4 flex gap-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-7 w-16 rounded bg-overlay/60" />
          ))}
        </div>
      </div>
      <div className="flex-1 grid place-items-center text-fg-tertiary text-sm">
        {t('settings.drawerLoading')}
      </div>
    </div>
  )
}
