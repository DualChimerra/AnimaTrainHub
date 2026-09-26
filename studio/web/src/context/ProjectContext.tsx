import { createContext, useContext } from 'react'
import type { ProjectDetail, Version } from '../api/client'

export interface ProjectCtxValue {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
  onSelectVersion: (vid: number) => void
  /** Opens the "new version" dialog. Passing forkFromVid preselects the forkFrom dropdown (used by the "duplicate config into new version" CTA). */
  onCreateVersion: (forkFromVid?: number) => void
  onExportTrain: () => void
  onDeleteVersion: (vid: number) => void
  exporting: boolean
}

export const ProjectContext = createContext<ProjectCtxValue | null>(null)
export const useProjectCtx = () => useContext(ProjectContext)

// Setter lives at App level so Layout.tsx can push context up to Sidebar
export const ProjectSetterContext = createContext<((v: ProjectCtxValue | null) => void) | null>(null)
export const useProjectCtxSetter = () => useContext(ProjectSetterContext)

// ── sticky "selected project" ──────────────────────────────────────────────
// ProjectContext gets cleared when leaving the /projects/:pid route (Layout
// unmounts), which used to make the sidebar lose the current project after
// switching to a global page like Queue or Generate. This read-only snapshot
// is written when Layout mounts and **not cleared** on unmount, so the
// sidebar keeps the "selected project" for navigation across pages;
// interactions that need a live Layout (add/remove version, export, etc.)
// are still only available within a project (while ProjectContext exists).
// Switching projects still goes through manual selection on the project
// list page -- opening another project overwrites this snapshot with the
// new one.
export interface SelectedProjectValue {
  project: ProjectDetail
  activeVersion: Version | null
}
export const SelectedProjectContext = createContext<SelectedProjectValue | null>(null)
export const useSelectedProject = () => useContext(SelectedProjectContext)
export const SelectedProjectSetterContext = createContext<
  ((v: SelectedProjectValue | null) => void) | null
>(null)
export const useSelectedProjectSetter = () => useContext(SelectedProjectSetterContext)
