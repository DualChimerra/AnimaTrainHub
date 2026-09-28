import { useEffect, useState } from 'react'
import {
  createBrowserRouter,
  Navigate,
  Outlet,
  RouterProvider,
  useLocation,
  useNavigate,
} from 'react-router-dom'
import SettingsDrawer from './components/SettingsDrawer'
import Sidebar from './components/Sidebar'
import { useIsMobile } from './lib/useMediaQuery'
import Topbar from './components/Topbar'
import {
  ProjectContext,
  ProjectSetterContext,
  SelectedProjectContext,
  SelectedProjectSetterContext,
  type ProjectCtxValue,
  type SelectedProjectValue,
} from './context/ProjectContext'
import { useSettingsDrawer } from './lib/SettingsDrawer'
import ProjectsPage from './pages/Projects'
import QueuePage from './pages/Queue'
import QueueDetailPage from './pages/QueueDetail'
import ProjectLayout from './pages/project/Layout'
import ProjectOverview from './pages/project/Overview'
import CurationPage from './pages/project/steps/Curation'
import DownloadPage from './pages/project/steps/Download'
import PreprocessHub from './pages/project/steps/PreprocessHub'
import RegularizationPage from './pages/project/steps/Regularization'
import TagEditPage from './pages/project/steps/TagEdit'
import TrainPage from './pages/project/steps/Train'
import GeneratePage from './pages/tools/Generate'
import MonitorPage from './pages/tools/Monitor'
import PresetsPage from './pages/tools/Presets'
import SoupPage from './pages/tools/Soup'
import GraphPage from './pages/tools/Graph'

/**
 * Compatibility redirect for the old `/tools/settings?section=…` path: jumps to
 * the home page and opens the drawer (keeping the `section` param). Settings no
 * longer has its own URL; old bookmarks / the Topbar notification button link
 * here without 404ing, but land on the drawer instead of a full page.
 */
function SettingsRedirect() {
  const drawer = useSettingsDrawer()
  const navigate = useNavigate()
  const location = useLocation()
  useEffect(() => {
    const section = new URLSearchParams(location.search).get('section')
    drawer.open(section ? { section } : undefined)
    navigate('/', { replace: true })
    // Runs once on mount only; the empty deps array is intentional — later
    // location.search changes are caused by navigate('/') itself, so open()
    // must not fire again.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return null
}

/**
 * Compatibility redirect for the old `/queue/:id/log` and `/queue/:id/monitor`
 * paths: keeps the URL alive and forwards to the matching tab on the new detail
 * page (the tab is expressed as a hash), so bookmarks / saved links don't break.
 */
function QueueDetailRedirect({ tab }: { tab: 'log' | 'monitor' }) {
  const path = window.location.pathname
  const id = path.match(/\/queue\/(\d+)/)?.[1]
  if (!id) return <Navigate to="/queue" replace />
  return (
    <Navigate to={{ pathname: `/queue/${id}`, hash: tab }} replace />
  )
}

/** Animation key of a route: project pages share one per project (the steps
 *  animate inside ProjectLayout), every other page gets its own. */
function pageKey(path: string): string {
  return path.match(/^\/projects\/[^/]+/)?.[0] ?? path
}

/** Sidebar + Topbar shell; every route element renders into <Outlet />.
 *  SettingsDrawer uses fixed inset-0 to cover the whole viewport (including the
 *  Sidebar on the left) -- so clicking the backdrop anywhere (Sidebar area
 *  included) closes the drawer. The column's position:relative stays as the
 *  anchor for absolute elements inside <main> (e.g. the task log drawer pinned
 *  to the bottom footer). */
function RootLayout() {
  // Phone layout: the sidebar leaves the flex row and becomes an off-canvas
  // drawer, so the content column gets the whole width. Nothing about the
  // desktop structure changes — `mobile` is false there and both components
  // render exactly as before.
  const isMobile = useIsMobile()
  const [navOpen, setNavOpen] = useState(false)
  const location = useLocation()

  // Navigating from inside the drawer should close it; otherwise the freshly
  // opened page sits hidden behind the overlay.
  useEffect(() => { setNavOpen(false) }, [location.pathname])

  // Same for the settings drawer: opening it from the nav drawer (or anywhere)
  // must not leave the nav stacked on top of it.
  const settingsDrawer = useSettingsDrawer()
  useEffect(() => { if (settingsDrawer.isOpen) setNavOpen(false) }, [settingsDrawer.isOpen])

  // A drawer over the page must not let the page scroll underneath it.
  useEffect(() => {
    if (!isMobile || !navOpen) return
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = previous }
  }, [isMobile, navOpen])

  useEffect(() => {
    if (!navOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setNavOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [navOpen])

  return (
    // The whole app is one white panel on the warm canvas (mockup .viewport /
    // .app): sidebar and main column share it. On a phone the panel is
    // full-bleed and the sidebar slides in as a drawer.
    <div className="ds-viewport">
      <div className="ds-app">
        <Sidebar mobile={isMobile} mobileOpen={navOpen} onMobileClose={() => setNavOpen(false)} />
        {isMobile && navOpen && (
          <div
            className="fixed inset-0 z-40 bg-zinc-950/40 backdrop-blur-[2px]"
            onClick={() => setNavOpen(false)}
            aria-hidden="true"
          />
        )}
        <div className="ds-main relative overflow-hidden">
          <Topbar mobile={isMobile} onOpenNav={() => setNavOpen(true)} />
          <main style={{ flex: 1, overflow: 'auto' }}>
            {/* Each top-level page eases in; a project keeps one key so its
                layout (and loaded project) survives step changes. */}
            <div key={pageKey(location.pathname)} className="ds-page-anim" style={{ height: '100%' }}>
              <Outlet />
            </div>
          </main>
          <SettingsDrawer />
        </div>
      </div>
    </div>
  )
}

// Singleton DataRouter: migrated from the classic BrowserRouter so react-router
// v6's useBlocker works (BrowserRouter doesn't support it), which pages like
// TagEdit need for their "unsaved changes" navigation prompt. The structure is
// unchanged -- RootLayout wraps Sidebar/Topbar, and the app routes render as
// children into <Outlet />.
const router = createBrowserRouter(
  [
    {
      element: <RootLayout />,
      children: [
        { path: '/', element: <ProjectsPage /> },
        { path: '/queue', element: <QueuePage /> },
        { path: '/queue/:id', element: <QueueDetailPage /> },
        { path: '/queue/:id/log', element: <QueueDetailRedirect tab="log" /> },
        { path: '/queue/:id/monitor', element: <QueueDetailRedirect tab="monitor" /> },
        {
          path: '/projects/:pid',
          element: <ProjectLayout />,
          children: [
            { index: true, element: <ProjectOverview /> },
            { path: 'download', element: <DownloadPage /> },
            {
              path: 'v/:vid',
              children: [
                { path: 'curate', element: <CurationPage /> },
                // ADR 0010: preprocess moved from project scope to version scope
                { path: 'preprocess', element: <PreprocessHub /> },
                { path: 'edit', element: <TagEditPage /> },
                { path: 'reg', element: <RegularizationPage /> },
                { path: 'train', element: <TrainPage /> },
              ],
            },
          ],
        },
        { path: '/tools/presets', element: <PresetsPage /> },
        { path: '/tools/monitor', element: <MonitorPage /> },
        { path: '/tools/settings', element: <SettingsRedirect /> },
        { path: '/tools/generate', element: <GeneratePage /> },
        { path: '/tools/soup', element: <SoupPage /> },
        { path: '/tools/graph', element: <GraphPage /> },
        { path: '/configs', element: <Navigate to="/tools/presets" replace /> },
        { path: '/monitor', element: <Navigate to="/tools/monitor" replace /> },
        { path: '/datasets', element: <Navigate to="/" replace /> },
      ],
    },
  ],
  {
    // ADR 0012: the SPA mounts at the root path, no more /studio subpath prefix.
    basename: '/',
    future: { v7_relativeSplatPath: true },
  },
)

export default function App() {
  const [projectCtx, setProjectCtx] = useState<ProjectCtxValue | null>(null)
  // The "selected project" snapshot kept across pages (see ProjectContext comment)
  const [selectedProject, setSelectedProject] = useState<SelectedProjectValue | null>(null)

  return (
    <ProjectContext.Provider value={projectCtx}>
      <ProjectSetterContext.Provider value={setProjectCtx}>
        <SelectedProjectContext.Provider value={selectedProject}>
          <SelectedProjectSetterContext.Provider value={setSelectedProject}>
            <RouterProvider router={router} />
          </SelectedProjectSetterContext.Provider>
        </SelectedProjectContext.Provider>
      </ProjectSetterContext.Provider>
    </ProjectContext.Provider>
  )
}
