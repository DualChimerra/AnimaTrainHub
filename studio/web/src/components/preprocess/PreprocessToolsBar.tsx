import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate } from 'react-router-dom'
import { api, type CropWorkspaceItem, type TrainImage } from '../../api/client'
import { useDialog } from '../Dialog'
import { useToast } from '../Toast'
import { useEventStream } from '../../lib/useEventStream'

export type PreprocessTool = 'overview' | 'dedupe' | 'upscale' | 'crop' | 'inpaint'

/** Tab order of the mockup. Overview is the default tool (no `?tool=`). */
const TABS: ReadonlyArray<PreprocessTool> = ['overview', 'upscale', 'crop', 'inpaint', 'dedupe']

interface Props {
  current: PreprocessTool
  projectId: number
  versionId: number
}

const toolHref = (projectId: number, versionId: number, tool: PreprocessTool) => {
  const base = `/projects/${projectId}/v/${versionId}/preprocess`
  return tool === 'overview' ? base : `${base}?tool=${tool}`
}

/** Train-set numbers behind the tool cards and the head's restore action.
 *  Refetched on preprocess / project events so every tool page stays in sync. */
function useTrainSummary(projectId: number, versionId: number) {
  const [images, setImages] = useState<TrainImage[] | null>(null)
  const [crop, setCrop] = useState<CropWorkspaceItem[] | null>(null)
  const refresh = useCallback(async () => {
    if (!versionId) return
    const [files, ws] = await Promise.all([
      api.listPreprocessFilesTrain(projectId, versionId).catch(() => null),
      api.listCropWorkspaceTrain(projectId, versionId).catch(() => null),
    ])
    setImages(files?.images ?? null)
    setCrop(ws?.images ?? null)
  }, [projectId, versionId])
  useEffect(() => { void refresh() }, [refresh])
  useEventStream((evt) => {
    if (evt.project_id !== projectId) return
    if (evt.type === 'preprocess_progress' || evt.type === 'project_state_changed'
      || (evt.type === 'job_state_changed' && evt.status !== 'running' && evt.status !== 'pending')) {
      void refresh()
    }
  })
  return useMemo(() => {
    const live = (images ?? []).filter((i) => !i.duplicate_removed)
    const processed = live.filter((i) => i.processed)
    const aspects = new Set(live.flatMap((i) => (i.w && i.h ? [Math.round((i.w / i.h) * 20) / 20] : [])))
    return {
      loaded: images != null,
      total: live.length,
      processed: processed.length,
      processedNames: processed.map((i) => i.name),
      removed: (images ?? []).filter((i) => i.duplicate_removed).length,
      masks: (crop ?? []).filter((c) => c.mask_mtime != null).length,
      aspects: aspects.size,
      refresh,
    }
  }, [images, crop, refresh])
}

const Icon = {
  upscale: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" /></svg>,
  crop: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M6 2v14a2 2 0 0 0 2 2h14" /><path d="M18 22V8a2 2 0 0 0-2-2H2" /></svg>,
  inpaint: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="m14.5 4.5 5 5L9 20H4v-5L14.5 4.5Z" /><path d="m12 7 5 5" /></svg>,
  dedupe: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><rect x="8" y="8" width="13" height="13" rx="2.5" /><path d="M16 8V5.5A2.5 2.5 0 0 0 13.5 3h-8A2.5 2.5 0 0 0 3 5.5v8A2.5 2.5 0 0 0 5.5 16H8" /></svg>,
}

/** Mockup's row of four tool cards under the step head (upscale · crop ·
 *  inpaint · duplicates) with each tool's state; the open tool is ringed.
 *  Goes into StepShell's `belowHeader`. */
export default function PreprocessToolsBar({ current, projectId, versionId }: Props) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const s = useTrainSummary(projectId, versionId)
  const pct = (n: number) => (s.total > 0 ? Math.round((n / s.total) * 100) : 0)

  const cards: Array<{ tool: PreprocessTool; icon: ReactNode; label: string; cell: string; fill: number; tone?: 'warn' }> = [
    { tool: 'upscale', icon: Icon.upscale, label: t('preprocess.tools.upscale'), cell: t('ppFrame.sumUpscale', { n: s.processed, total: s.total }), fill: pct(s.processed) },
    { tool: 'crop', icon: Icon.crop, label: t('preprocess.tools.crop'), cell: t('ppFrame.sumCrop', { n: s.aspects }), fill: 0 },
    { tool: 'inpaint', icon: Icon.inpaint, label: t('preprocess.tools.inpaint'), cell: t('ppFrame.sumInpaint', { n: s.masks }), fill: pct(s.masks) },
    { tool: 'dedupe', icon: Icon.dedupe, label: t('preprocess.tools.dedupe'), cell: t('ppFrame.sumDedupe', { n: s.removed }), fill: pct(s.removed), tone: 'warn' },
  ]

  return (
    <div className="ds-pp-cards" style={{ padding: '0 24px 12px' }}>
      {cards.map((c) => {
        const active = c.tool === current
        const tileStyle: React.CSSProperties = { width: 30, height: 30, borderRadius: 9 }
        if (active) Object.assign(tileStyle, { background: 'var(--green-soft)', color: 'var(--green-text)' })
        else if (c.tone === 'warn' && s.removed > 0) Object.assign(tileStyle, { background: 'var(--amber-soft)', color: 'var(--amber-text)' })
        return (
          <button
            key={c.tool}
            type="button"
            className="ds-card"
            aria-current={active ? 'page' : undefined}
            onClick={() => navigate(toolHref(projectId, versionId, c.tool))}
            style={{
              padding: '14px 16px', display: 'flex', gap: 11, alignItems: 'flex-start', textAlign: 'left',
              ...(active ? { boxShadow: '0 0 0 1px var(--green-600), var(--sh-card)' } : {}),
            }}
          >
            <span className="ds-sect-icon" style={tileStyle}>{c.icon}</span>
            <span style={{ flex: 1, minWidth: 0 }}>
              <span style={{ fontSize: 13, fontWeight: 600, display: 'block' }}>{c.label}</span>
              <span className="ds-cell-key">{s.loaded ? c.cell : '—'}</span>
              <span className="ds-meter ds-thin" style={{ marginTop: 7, display: 'block' }}>
                <i className={c.tone === 'warn' ? 'ds-warn' : undefined} style={{ width: `${c.fill}%` }} />
              </span>
            </span>
          </button>
        )
      })}
    </div>
  )
}

/** The mockup's tool card: underline tabs across the tools, the open tool's
 *  content below. Tabs are route links (`?tool=`), so switching keeps the
 *  parent route mounted. */
export function PreprocessCard({ current, projectId, versionId, children }: Props & { children: ReactNode }) {
  const { t } = useTranslation()
  return (
    <div className="ds-card" style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <nav className="ds-tabs" aria-label={t('preprocess.toolsLabel')}>
        {TABS.map((tool) => (
          <Link
            key={tool}
            to={toolHref(projectId, versionId, tool)}
            className={`ds-tab${tool === current ? ' ds-is-active' : ''}`}
            aria-current={tool === current ? 'page' : undefined}
          >
            {t(`preprocess.tools.${tool}`)}
          </Link>
        ))}
      </nav>
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>{children}</div>
    </div>
  )
}

/** Page head tools of every preprocess tool: restore all processed images,
 *  then on to the tag step. */
export function PreprocessHeadTools({ projectId, versionId }: { projectId: number; versionId: number }) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const s = useTrainSummary(projectId, versionId)
  const [busy, setBusy] = useState(false)

  const restoreAll = async () => {
    const names = s.processedNames
    if (names.length === 0 || busy) return
    if (!(await confirm(t('preprocessOverview.confirmRestore', { n: names.length }), { tone: 'danger', okText: t('preprocessOverview.confirmRestoreOk') }))) return
    setBusy(true)
    try {
      const r = await api.restorePreprocessFilesTrain(projectId, versionId, names)
      toast(t('ppFrame.restoredN', { n: r.restored.length }), 'success')
      await s.refresh()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <button
        type="button"
        className="ds-ctl"
        onClick={() => void restoreAll()}
        disabled={busy || s.processed === 0}
        title={t('ppFrame.restoreAllTitle')}
      >
        {t('ppFrame.restoreAll')}{s.processed > 0 ? ` · ${s.processed}` : ''}
      </button>
      <Link className="ds-btn-primary" to={`/projects/${projectId}/v/${versionId}/edit`}>{t('ppFrame.next')}</Link>
    </>
  )
}
