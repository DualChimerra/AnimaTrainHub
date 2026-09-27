import { useEffect, useState } from 'react'
import { api } from '../../../api/client'
import type { ProjectLora } from './types'

/** Fetches every project's LoRA versions once at startup (including ones still training).
 *
 * The previous implementation filtered with `if (!v.output_lora_path) continue` -- but
 * output_lora_path is only backfilled once training completes (status=done); for a version
 * still training, that field is null. On disk, anima_train has already written step
 * checkpoints (_step1500.safetensors etc.) under versions/{label}/output/, so the picker should list those too.
 *
 * Current strategy:
 *   1. Prefer v.output_lora_path (training finished, path = _final.safetensors)
 *   2. Fallback: fetch listVersionLoraCkpts and take the latest (most recent step / epoch checkpoint)
 *   3. Neither → version's output/ has no checkpoint at all, skip it
 *
 * Doesn't throw on failure -- the user falls back to the "External file..." PathPicker. This is
 * an N+1 call: acceptable for the typical user's project count (< 20); loaded once at startup, and the picker doesn't refresh live (matches user expectations).
 */
export function useProjectLoras(): ProjectLora[] {
  const [items, setItems] = useState<ProjectLora[]>([])
  useEffect(() => {
    void (async () => {
      try {
        const projects = await api.listProjects()
        const details = await Promise.all(
          projects.map((p) => api.getProject(p.id).catch(() => null))
        )

        // Phase 1: build directly from v.output_lora_path (already-finished ones) + collect ones needing fallback
        const out: ProjectLora[] = []
        const fallbacks: Array<{ pid: number; vid: number; meta: ProjectLora }> = []

        for (const d of details) {
          if (!d) continue
          for (const v of d.versions) {
            if (v.output_lora_path) {
              out.push({
                projectId: d.id,
                projectTitle: d.title,
                versionId: v.id,
                versionLabel: v.label,
                status: v.status,
                path: v.output_lora_path,
                createdAt: v.created_at,
              })
            } else {
              // Still training or no final yet → try the latest step/epoch checkpoint
              fallbacks.push({
                pid: d.id, vid: v.id,
                meta: {
                  projectId: d.id,
                  projectTitle: d.title,
                  versionId: v.id,
                  versionLabel: v.label,
                  status: v.status,
                  path: '',  // filled in by the fallback
                  createdAt: v.created_at,
                },
              })
            }
          }
        }

        // Phase 2: concurrently fetch the checkpoint lists for everything needing a fallback, take the latest
        const fbResults = await Promise.all(
          fallbacks.map(async (fb) => {
            try {
              const ckpts = await api.listVersionLoraCkpts(fb.pid, fb.vid)
              if (ckpts.length === 0) return null
              return { ...fb.meta, path: ckpts[0].path }
            } catch {
              return null
            }
          })
        )
        for (const r of fbResults) {
          if (r) out.push(r)
        }

        out.sort((a, b) => b.createdAt - a.createdAt)
        setItems(out)
      } catch {
        /* a startup failure shouldn't block anything */
      }
    })()
  }, [])
  return items
}
