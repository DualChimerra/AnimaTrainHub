import { useCallback, useMemo, useRef, useState } from 'react'
import { api, type LoraCkpt, type VersionStatus } from '../../../api/client'

/** Lazy cascading cache for the test page's LoRA picker data.
 *
 *
 */

export interface LoraProjectOption {
  id: number
  title: string
}

export interface LoraVersionOption {
  id: number
  label: string
  status: VersionStatus
}

export interface LoraCatalog {
  projects: LoraProjectOption[]
  projectsLoading: boolean
  ensureProjects: () => void
  loadProjects: () => Promise<LoraProjectOption[]>
  versionsOf: (pid: number) => LoraVersionOption[] | undefined
  ensureVersions: (pid: number) => void
  fetchCkpts: (pid: number, vid: number) => Promise<LoraCkpt[]>
}

export function useLoraCatalog(): LoraCatalog {
  const [projects, setProjects] = useState<LoraProjectOption[]>([])
  const [projectsLoading, setProjectsLoading] = useState(false)
  const projectsReq = useRef<Promise<LoraProjectOption[]> | null>(null)

  const [versionsByPid, setVersionsByPid] = useState<Record<number, LoraVersionOption[]>>({})
  const versionReqs = useRef<Record<number, Promise<LoraVersionOption[]>>>({})

  const ckptCache = useRef<Record<string, LoraCkpt[]>>({})
  const ckptReqs = useRef<Record<string, Promise<LoraCkpt[]>>>({})

  const loadProjects = useCallback((): Promise<LoraProjectOption[]> => {
    if (projectsReq.current) return projectsReq.current
    setProjectsLoading(true)
    const req = api.listProjects()
      .then((ps) => {
        const opts = ps.map((p) => ({ id: p.id, title: p.title }))
        setProjects(opts)
        return opts
      })
      .catch(() => {
        projectsReq.current = null
        return [] as LoraProjectOption[]
      })
      .finally(() => setProjectsLoading(false))
    projectsReq.current = req
    return req
  }, [])

  const ensureProjects = useCallback(() => { void loadProjects() }, [loadProjects])

  const loadVersions = useCallback((pid: number): Promise<LoraVersionOption[]> => {
    const existing = versionReqs.current[pid]
    if (existing) return existing
    const req = api.getProject(pid)
      .then((d) => {
        const opts = d.versions.map((v) => ({ id: v.id, label: v.label, status: v.status }))
        setVersionsByPid((m) => ({ ...m, [pid]: opts }))
        return opts
      })
      .catch(() => {
        delete versionReqs.current[pid]
        return [] as LoraVersionOption[]
      })
    versionReqs.current[pid] = req
    return req
  }, [])

  const ensureVersions = useCallback((pid: number) => { void loadVersions(pid) }, [loadVersions])

  const versionsOf = useCallback(
    (pid: number): LoraVersionOption[] | undefined => versionsByPid[pid],
    [versionsByPid],
  )

  const fetchCkpts = useCallback((pid: number, vid: number): Promise<LoraCkpt[]> => {
    const key = `${pid}:${vid}`
    const cached = ckptCache.current[key]
    if (cached) return Promise.resolve(cached)
    const inflight = ckptReqs.current[key]
    if (inflight) return inflight
    const req = api.listVersionLoraCkpts(pid, vid)
      .then((items) => {
        ckptCache.current[key] = items
        delete ckptReqs.current[key]
        return items
      })
      .catch((e) => {
        delete ckptReqs.current[key]
        throw e
      })
    ckptReqs.current[key] = req
    return req
  }, [])

  return useMemo<LoraCatalog>(
    () => ({
      projects, projectsLoading, ensureProjects, loadProjects,
      versionsOf, ensureVersions, fetchCkpts,
    }),
    [projects, projectsLoading, ensureProjects, loadProjects, versionsOf, ensureVersions, fetchCkpts],
  )
}
