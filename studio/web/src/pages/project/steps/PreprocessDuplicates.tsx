import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOutletContext } from 'react-router-dom'
import {
  api,
  type DuplicateScanOptions,
  type DuplicateScanResult,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import DuplicateReviewPanel, {
  DEFAULT_DUPLICATE_OPTIONS,
} from '../../../components/DuplicateReviewPanel'
import { useDialog } from '../../../components/Dialog'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import StepShell from '../../../components/StepShell'
import PreprocessToolsBar, { PreprocessCard, PreprocessHeadTools } from '../../../components/preprocess/PreprocessToolsBar'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

interface DuplicateLog {
  ts: number
  text: string
  status?: string
}

export default function PreprocessDuplicatesPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const vid = activeVersion?.id ?? 0
  const [options, setOptions] = useState<DuplicateScanOptions>(DEFAULT_DUPLICATE_OPTIONS)
  const [result, setResult] = useState<DuplicateScanResult | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const [logs, setLogs] = useState<DuplicateLog[]>([])
  const [scanLogVisible, setScanLogVisible] = useState(false)
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const lastLogAtRef = useRef(0)

  useEffect(() => {
    lastLogAtRef.current = 0
    setLogs([])
    setScanLogVisible(false)
  }, [project.id])

  useEventStream((evt) => {
    if (evt.type !== 'duplicate_scan_progress' || evt.project_id !== project.id) return
    const now = Date.now()
    const status = String(evt.status ?? '')
    if (status === 'running' && now - lastLogAtRef.current < 1000) return
    lastLogAtRef.current = now
    setLogs((prev) => [
      ...prev.slice(-119),
      {
        ts: now,
        status,
        text: String(evt.text ?? status),
      },
    ])
  })

  // 裁剪/缩放候选已并入分组（更严格的重复判断），故预览名单直接取分组成员即可。
  const previewNames = useMemo(
    () => result
      ? Array.from(new Set(
          result.groups.flatMap((group) => group.items.map((item) => item.name)),
        ))
      : [],
    [result],
  )

  const scan = async () => {
    if (busy) return
    setBusy(true)
    setResult(null)
    setSelected(new Set())
    setScanLogVisible(true)
    setLogs([{ ts: Date.now(), status: 'running', text: t('duplicates.logStarted') }])
    try {
      const next = await api.scanDuplicatesTrain(project.id, vid, options)
      setResult(next)
      setSelected(new Set(
        next.groups.flatMap((group) =>
          group.items.filter((item) => !item.keep).map((item) => item.name),
        ),
      ))
      toast(
        t('duplicates.scanDone', {
          groups: next.group_count,
          candidates: next.candidate_count,
          crops: next.crop_relation_count,
        }),
        'success',
      )
    } catch (e) {
      toast(String(e), 'error')
      setLogs((prev) => [...prev, { ts: Date.now(), status: 'error', text: String(e) }])
    } finally {
      setBusy(false)
    }
  }

  const apply = async () => {
    if (busy || selected.size === 0) return
    const names = Array.from(selected)
    const ok = await confirm(
      t('duplicates.confirmApply', { n: names.length }),
      { tone: 'warn', okText: t('duplicates.applyOk') },
    )
    if (!ok) return
    setBusy(true)
    try {
      const res = await api.applyDuplicateActionTrain(project.id, vid, { names })
      toast(
        t('duplicates.appliedToast', { n: res.removed.length }) +
          (res.skipped.length ? t('duplicates.appliedSkipped', { n: res.skipped.length }) : '') +
          (res.missing.length ? t('duplicates.appliedMissing', { n: res.missing.length }) : ''),
        'success',
      )
      setSelected(new Set())
      setResult(null)
      setPreviewIdx(null)
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const openPreview = (name: string) => {
    const index = previewNames.indexOf(name)
    setPreviewIdx(index >= 0 ? index : 0)
  }

  // ADR 0010: hooks 之后再做 vid guard
  if (!activeVersion) {
    return (
      <div className="p-6 text-fg-secondary">
        {t('projectStepper.selectVersion')}
      </div>
    )
  }

  return (
    <StepShell
      idx={2}
      eyebrow={t('ppFrame.eyebrow')}
      title={t('ppFrame.title')}
      subtitle={t('ppFrame.subtitle')}
      actions={<PreprocessHeadTools projectId={project.id} versionId={vid} />}
      belowHeader={<PreprocessToolsBar current="dedupe" projectId={project.id} versionId={vid} />}
      logSources={[
        scanLogVisible && logs.length > 0 && {
          key: 'dup_scan',
          label: t('logDrawer.dupScan'),
          // 扫描是同步 HTTP + SSE 进度，前端合成状态：跑着 = running，
          // 否则按最后一条日志判 failed/done。不可取消。
          status: busy
            ? ('running' as const)
            : logs[logs.length - 1]?.status === 'error'
              ? ('failed' as const)
              : ('done' as const),
          lines: logs.map((l) => l.text),
          startedAt: logs[0] ? logs[0].ts / 1000 : null,
          finishedAt: busy ? null : logs[logs.length - 1] ? logs[logs.length - 1].ts / 1000 : null,
        },
      ]}
    >
      <PreprocessCard current="dedupe" projectId={project.id} versionId={vid}>
        <div className="ds-pp-split">
          <DuplicateOperationPanel
            options={options}
            busy={busy}
            onOptionsChange={setOptions}
            result={result}
            selectedCount={selected.size}
            onScan={() => void scan()}
            onApply={() => void apply()}
          />
          <DuplicateReviewPanel
            projectId={project.id}
            versionId={vid}
            result={result}
            selected={selected}
            busy={busy}
            onSelect={setSelected}
            onPreview={openPreview}
          />
        </div>
      </PreprocessCard>

      {previewIdx !== null && previewNames[previewIdx] && (() => {
        const rel = previewNames[previewIdx]
        const i = rel.lastIndexOf('/')
        const folder = i >= 0 ? rel.slice(0, i) : ''
        const filename = i >= 0 ? rel.slice(i + 1) : rel
        return (
        <ImagePreviewModal
          src={api.versionThumbUrl(project.id, vid, 'train', filename, folder, 1600)}
          caption={rel}
          index={previewIdx}
          total={previewNames.length}
          hasPrev={previewIdx > 0}
          hasNext={previewIdx < previewNames.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() => previewIdx > 0 && setPreviewIdx(previewIdx - 1)}
          onNext={() => previewIdx < previewNames.length - 1 && setPreviewIdx(previewIdx + 1)}
          shortcutHint={t('duplicates.previewHint')}
        />
        )
      })()}
    </StepShell>
  )
}

// Three sensitivity levels; they only drive the variant / crop checks
// (backend variant_score + crop_score), so strict-duplicate mode locks them.
const SENSITIVITY_OPTIONS = [
  { id: 'loose', key: 'sensitivityLoose' },
  { id: 'standard', key: 'sensitivityStandard' },
  { id: 'strict', key: 'sensitivityStrict' },
] as const

interface DuplicateOperationPanelProps {
  options: DuplicateScanOptions
  busy: boolean
  onOptionsChange: (next: DuplicateScanOptions) => void
  result: DuplicateScanResult | null
  selectedCount: number
  onScan: () => void
  onApply: () => void
}

/** Left settings column of the duplicates tool (mockup PreprocessDupes):
 *  how to match, sensitivity, the scan numbers, then scan / remove. */
function DuplicateOperationPanel({ options, busy, onOptionsChange, result, selectedCount, onScan, onApply }: DuplicateOperationPanelProps) {
  const { t } = useTranslation()
  const patch = <K extends keyof DuplicateScanOptions>(key: K, value: DuplicateScanOptions[K]) => {
    onOptionsChange({ ...options, [key]: value })
  }
  const sensitivityLocked = options.match_scope !== 'both'
  const total = result?.total_images ?? 0

  return (
    <div className="ds-pp-side">
      <div>
        <div className="ds-cap" style={{ marginBottom: 8 }}>{t('ppFrame.dupHow')}</div>
        <div className="ds-seg" style={{ display: 'flex' }} role="group" aria-label={t('duplicates.scope')}>
          {(['strict', 'both'] as const).map((id) => (
            <button key={id} type="button" className={`ds-seg-item${options.match_scope === id ? ' ds-is-active' : ''}`} style={{ flex: 1 }} aria-pressed={options.match_scope === id} disabled={busy} onClick={() => patch('match_scope', id)}>
              {id === 'strict' ? t('duplicates.scopeStrict') : t('duplicates.scopeBoth')}
            </button>
          ))}
        </div>
        <div className="ds-ctl-note" style={{ marginTop: 6 }}>{options.match_scope === 'strict' ? t('ppFrame.dupHowStrict') : t('ppFrame.dupHowBoth')}</div>
      </div>

      <div title={sensitivityLocked ? t('duplicates.sensitivityLockedHint') : undefined} style={sensitivityLocked ? { opacity: 0.5 } : undefined}>
        <div className="ds-cap" style={{ marginBottom: 8 }}>{t('duplicates.sensitivity')}</div>
        <div className="ds-seg" style={{ display: 'flex' }} role="group" aria-label={t('duplicates.sensitivity')}>
          {SENSITIVITY_OPTIONS.map(({ id, key }) => (
            <button key={id} type="button" className={`ds-seg-item${options.sensitivity === id ? ' ds-is-active' : ''}`} style={{ flex: 1 }} aria-pressed={options.sensitivity === id} disabled={busy || sensitivityLocked} onClick={() => patch('sensitivity', id)}>
              {t(`duplicates.${key}`)}
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="ds-cap" style={{ marginBottom: 6 }}>{t('duplicates.statsTitle')}</div>
        <div className="ds-kv"><span className="ds-k">{t('duplicates.statsGroups')}</span><span className="ds-v">{result?.group_count ?? '—'}</span></div>
        <div className="ds-kv"><span className="ds-k">{t('duplicates.statsCandidates')}</span><span className="ds-v">{result?.candidate_count ?? '—'}</span></div>
        <div className="ds-kv"><span className="ds-k">{t('duplicates.statsCrops')}</span><span className="ds-v">{result?.crop_relation_count ?? '—'}</span></div>
        <div className="ds-kv"><span className="ds-k">{t('duplicates.statsSelected')}</span><span className="ds-v">{selectedCount}</span></div>
        <div className="ds-kv"><span className="ds-k">{t('duplicates.statsAfter')}</span><span className="ds-v ds-acc">{result ? Math.max(0, total - selectedCount) : '—'}</span></div>
        <div className="ds-kv"><span className="ds-k">{t('duplicates.statsCompared')}</span><span className="ds-v">{result?.stats.compared_pairs ?? '—'}</span></div>
        <div className="ds-kv"><span className="ds-k">{t('duplicates.statsElapsed')}</span><span className="ds-v">{result ? `${result.elapsed_seconds}s` : '—'}</span></div>
      </div>

      <div className="ds-note ds-info" style={{ fontSize: 11.5 }}>{t('ppFrame.dupNote')}</div>

      <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
        <button type="button" className="ds-ctl" style={{ justifyContent: 'center', height: 34 }} onClick={onScan} disabled={busy}>
          {busy ? t('duplicates.scanning') : t('duplicates.scanBtn')}
        </button>
        <button type="button" className="ds-btn-primary" style={{ justifyContent: 'center', height: 34 }} onClick={onApply} disabled={busy || selectedCount === 0}>
          {t('ppFrame.dupApply', { count: selectedCount })}
        </button>
      </div>
    </div>
  )
}
