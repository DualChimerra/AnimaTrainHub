import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import {
  api,
  type ApiError,
  type LoraCkpt,
  type ProjectSummary,
  type SoupCompatibility,
  type SoupFile,
  type SoupMergeResult,
  type SoupSourceInfo,
  type VersionCkptGroup,
} from '../../api/client'
import { useDialog } from '../../components/Dialog'
import FieldLabel from '../../components/ds/FieldLabel'
import KebabMenu from '../../components/ds/KebabMenu'
import PageHead from '../../components/ds/PageHead'
import { useToast } from '../../components/Toast'

/** One ingredient in the soup: a checkpoint plus how much of it to use. */
interface Ingredient {
  /** Absolute path — also the identity, a file can only be picked once. */
  path: string
  /** What to show; the absolute path is long and mostly noise. */
  label: string
  /** Where it came from, for the second line ("project · version", "upload"). */
  origin: string
  weight: number
}

type Method = 'average' | 'sum'

function fmtSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—'
  const mb = bytes / 1024 ** 2
  return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`
}

function basename(p: string): string {
  const parts = p.split(/[\\/]/)
  return parts[parts.length - 1] || p
}

/** Weights as the merge will actually apply them. Mirrors
 *  `studio/services/soup.normalize_weights` so this preview cannot disagree
 *  with the file that gets written. */
function effectiveWeights(items: Ingredient[], method: Method): number[] {
  const raw = items.map((i) => i.weight)
  if (method === 'sum') return raw
  const total = raw.reduce((a, b) => a + b, 0)
  if (Math.abs(total) < 1e-9) return raw.map(() => 0)
  return raw.map((w) => w / total)
}

export default function SoupPage() {
  const { t } = useTranslation()
  const { toast } = useToast()
  const dialog = useDialog()
  const navigate = useNavigate()

  const [items, setItems] = useState<Ingredient[]>([])
  const [method, setMethod] = useState<Method>('average')
  const [name, setName] = useState('')
  const [verdict, setVerdict] = useState<SoupCompatibility | null>(null)
  const [checking, setChecking] = useState(false)
  const [merging, setMerging] = useState(false)
  const [result, setResult] = useState<SoupMergeResult | null>(null)
  const [outputs, setOutputs] = useState<SoupFile[]>([])
  const [uploads, setUploads] = useState<SoupFile[]>([])
  const [pickerOpen, setPickerOpen] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const refreshSources = useCallback(async () => {
    try {
      const r = await api.listSoupSources()
      setUploads(r.uploads)
      setOutputs(r.outputs)
    } catch {
      /* the picker degrades to project checkpoints only */
    }
  }, [])

  useEffect(() => { void refreshSources() }, [refreshSources])

  // Compatibility is re-checked whenever the ingredient *set* changes — not on
  // every weight keystroke, since weights cannot make files incompatible.
  const pathKey = useMemo(() => JSON.stringify(items.map((i) => i.path)), [items])
  useEffect(() => {
    const paths = JSON.parse(pathKey) as string[]
    if (paths.length < 2) { setVerdict(null); return }
    let alive = true
    setChecking(true)
    api.inspectSoupSources(paths)
      .then((v) => { if (alive) setVerdict(v) })
      .catch((e: ApiError) => {
        if (alive) setVerdict({ ok: false, errors: [e.message], warnings: [], items: [] })
      })
      .finally(() => { if (alive) setChecking(false) })
    return () => { alive = false }
  }, [pathKey])

  const eff = useMemo(() => effectiveWeights(items, method), [items, method])
  const canMerge = items.length >= 2 && !!verdict?.ok && !merging && !checking

  const add = (ing: Ingredient) => {
    setItems((arr) => (arr.some((i) => i.path === ing.path) ? arr : [...arr, ing]))
    setResult(null)
  }
  const remove = (path: string) => {
    setItems((arr) => arr.filter((i) => i.path !== path))
    setResult(null)
  }
  const setWeight = (path: string, weight: number) =>
    setItems((arr) => arr.map((i) => (i.path === path ? { ...i, weight } : i)))

  const suggestedName = useMemo(() => {
    if (items.length < 2) return ''
    // "a + b" reads better than a timestamp, and the full recipe is written
    // into the file's metadata anyway.
    const stems = items.map((i) => i.label.replace(/\.safetensors$/i, ''))
    const joined = stems.join(' + ')
    return joined.length <= 80 ? joined : `soup of ${items.length}`
  }, [items])

  const handleUpload = async (files: FileList | null) => {
    if (!files?.length) return
    for (const file of Array.from(files)) {
      try {
        const saved = await api.uploadSoupSource(file)
        add({ path: saved.path, label: saved.name, origin: t('soup.originUpload'), weight: 1 })
        toast(t('soup.uploaded', { name: saved.name }), 'success')
      } catch (e) {
        toast((e as ApiError).message || t('soup.uploadFailed'), 'error')
      }
    }
    await refreshSources()
    if (fileRef.current) fileRef.current.value = ''
  }

  const handleMerge = async () => {
    const outName = (name.trim() || suggestedName).trim()
    if (!outName) { toast(t('soup.needName'), 'error'); return }
    setMerging(true)
    try {
      const r = await api.mergeSoup({
        inputs: items.map((i) => ({ path: i.path, weight: i.weight })),
        name: outName,
        method,
      })
      setResult(r)
      toast(t('soup.merged', { name: r.name }), 'success')
      await refreshSources()
    } catch (e) {
      toast((e as ApiError).message || t('soup.mergeFailed'), 'error')
    } finally {
      setMerging(false)
    }
  }

  const handleTest = (path: string) => {
    navigate(`/tools/generate?lora=${encodeURIComponent(path)}`)
  }

  const handleDeleteOutput = async (file: SoupFile) => {
    if (!(await dialog.confirm(t('soup.deleteConfirm', { name: file.name })))) return
    try {
      await api.deleteSoupOutput(file.name)
      if (result?.name === file.name) setResult(null)
      await refreshSources()
    } catch (e) {
      toast((e as ApiError).message || t('soup.deleteFailed'), 'error')
    }
  }

  const [dragOver, setDragOver] = useState(false)
  const infoByPath = useMemo(() => new Map((verdict?.items ?? []).map((i) => [i.path, i])), [verdict])
  const sumWeights = items.reduce((a, i) => a + i.weight, 0)

  return (
    <div className="fade-in" style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <input
        ref={fileRef} type="file" accept=".safetensors" multiple hidden
        onChange={(e) => void handleUpload(e.target.files)}
      />
      <PageHead
        eyebrow={t('soup.eyebrow')}
        title={t('soup.title')}
        subtitle={t('soup.subtitle')}
        tools={<span className="ds-badge ds-mute">{t('soup.compatBadge')}</span>}
      />

      <div className="ds-scroll" style={{ minHeight: 0, overflowY: 'auto' }}>
        <div className="ds-soup-grid">

          {/* ── the pot ── */}
          <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <div className="ds-card-head ds-pad">
              <div style={{ flex: 1 }}>
                <div className="ds-card-title">{t('soup.potTitle')}</div>
                <div className="ds-card-sub">
                  {items.length === 0
                    ? t('soup.emptyBody')
                    : t(method === 'average' ? 'soup.potSubAverage' : 'soup.potSubSum', { count: items.length })}
                </div>
              </div>
              <div className="ds-card-tools">
                <button type="button" className="ds-ctl ds-ghost" onClick={() => setPickerOpen(true)}>+ {t('soup.addCheckpoint')}</button>
              </div>
            </div>

            <div style={{ padding: '0 17px 14px', display: 'flex', flexDirection: 'column', gap: 10 }}>
              {items.map((ing, idx) => (
                <IngredientRow
                  key={ing.path}
                  ing={ing}
                  info={infoByPath.get(ing.path)}
                  share={eff[idx]}
                  method={method}
                  onWeight={(w) => setWeight(ing.path, w)}
                  onRemove={() => remove(ing.path)}
                />
              ))}

              <div
                className="ds-dropstrip"
                style={dragOver ? { borderColor: 'var(--green-600)', background: 'var(--green-soft)' } : undefined}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); void handleUpload(e.dataTransfer.files) }}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M12 16V4M7 9l5-5 5 5M5 20h14" /></svg>
                <span style={{ flex: 1 }}><b>{t('soup.dropHint')}</b></span>
                <button type="button" className="ds-ctl ds-ghost" style={{ height: 26 }} onClick={() => fileRef.current?.click()}>{t('soup.pickFile')}</button>
              </div>

              <Verdict verdict={verdict} checking={checking} count={items.length} />
            </div>

            <div style={{ marginTop: 'auto', borderTop: '1px solid var(--line)', padding: '14px 17px', display: 'flex', flexDirection: 'column', gap: 11 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span className="ds-cap" style={{ width: 104, flex: 'none' }}>
                  <FieldLabel label={t('soup.method')} tip={t(`soup.methodHelp_${method}`)} />
                </span>
                <div className="ds-seg">
                  {(['average', 'sum'] as Method[]).map((m) => (
                    <button
                      key={m}
                      type="button"
                      className={`ds-seg-item${method === m ? ' ds-is-active' : ''}`}
                      aria-pressed={method === m}
                      onClick={() => setMethod(m)}
                    >
                      {t(`soup.method_${m}`)}
                    </button>
                  ))}
                </div>
                <span style={{ flex: 1 }} />
                {items.length > 0 && (
                  <span className="ds-badge ds-mute">
                    {method === 'average' ? t('soup.sumAverage') : t('soup.sumRaw', { sum: sumWeights.toFixed(2) })}
                  </span>
                )}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span className="ds-cap" style={{ width: 104, flex: 'none' }}>{t('soup.outputName')}</span>
                <input
                  className="ds-inp ds-mono"
                  style={{ flex: 1, width: 'auto' }}
                  value={name}
                  placeholder={suggestedName || t('soup.outputNamePlaceholder')}
                  onChange={(e) => setName(e.target.value)}
                  aria-label={t('soup.outputName')}
                />
                <button type="button" className="ds-btn-primary" style={{ height: 34, flex: 'none' }} disabled={!canMerge} onClick={() => void handleMerge()}>
                  {merging ? t('soup.merging') : items.length >= 2 ? t('soup.mergeN', { count: items.length }) : t('soup.merge')}
                </button>
              </div>
            </div>
          </div>

          {/* ── result + past soups ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minHeight: 0 }}>
            <div className="ds-card">
              <div className="ds-card-head ds-pad">
                <div style={{ minWidth: 0 }}>
                  <div className="ds-card-title">{t('soup.resultTitle')}</div>
                  <div className="ds-card-sub ds-mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {result ? `${result.name} · ${fmtSize(result.size)}` : t('soup.resultEmpty')}
                  </div>
                </div>
              </div>
              {result && (
                <div className="ds-card-body" style={{ paddingTop: 2 }}>
                  {result.warnings.map((w, i) => (
                    <div key={i} className="ds-note ds-warn" style={{ marginBottom: 8 }}><span>{w}</span></div>
                  ))}
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button type="button" className="ds-ctl" style={{ flex: 1, justifyContent: 'center' }} onClick={() => handleTest(result.path)}>
                      {t('soup.testNow')}
                    </button>
                    <a className="ds-ctl ds-ghost" href={api.soupDownloadUrl(result.name)} download>{t('soup.download')}</a>
                  </div>
                </div>
              )}
            </div>

            <div className="ds-card">
              <div className="ds-card-head ds-pad">
                <div>
                  <div className="ds-card-title">{t('soup.previousSoups')}</div>
                  <div className="ds-card-sub">{t('soup.previousSoupsSub')}</div>
                </div>
              </div>
              <div style={{ padding: '0 0 10px' }}>
                {outputs.length === 0 ? (
                  <div className="ds-cell-key" style={{ padding: '0 17px 6px' }}>{t('soup.noSoups')}</div>
                ) : outputs.map((f, i) => (
                  <div
                    key={f.path}
                    style={{ padding: '10px 17px', borderBottom: i < outputs.length - 1 ? '1px solid var(--line)' : undefined, display: 'flex', alignItems: 'center', gap: 10 }}
                  >
                    <span className="ds-mono" style={{ fontSize: 12, flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={f.name}>
                      {f.name.replace(/\.safetensors$/i, '')}
                    </span>
                    <span className="ds-cell-key">{fmtSize(f.size)}</span>
                    <KebabMenu
                      label={t('soup.actions', { name: f.name })}
                      items={[
                        { label: t('soup.test'), onSelect: () => handleTest(f.path) },
                        { label: t('soup.download'), onSelect: () => downloadFile(api.soupDownloadUrl(f.name), f.name) },
                        { label: t('common.delete'), onSelect: () => void handleDeleteOutput(f), tone: 'err' },
                      ]}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {pickerOpen && (
        <SourcePicker
          uploads={uploads}
          outputs={outputs}
          chosen={new Set(items.map((i) => i.path))}
          onPick={add}
          onClose={() => setPickerOpen(false)}
          onUploadsChanged={refreshSources}
        />
      )}
    </div>
  )
}

function downloadFile(href: string, name: string) {
  const a = document.createElement('a')
  a.href = href
  a.download = name
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

// ── pieces ──────────────────────────────────────────────────────────────────

function IngredientRow({ ing, info, share, method, onWeight, onRemove }: {
  ing: Ingredient
  info?: SoupSourceInfo
  share: number
  method: Method
  onWeight: (w: number) => void
  onRemove: () => void
}) {
  const { t } = useTranslation()
  const net = info?.algo ? `${info.algo}${info.rank != null ? ` r${info.rank}` : ''}` : null
  return (
    <div className="ds-card ds-flat" style={{ border: '1px solid var(--line-2)', padding: '12px 14px', display: 'flex', alignItems: 'center', gap: 12 }}>
      <span className="ds-sect-icon">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M4 7h16M4 12h16M4 17h10" /></svg>
      </span>
      <span style={{ flex: 1, minWidth: 0 }}>
        <span className="ds-mono" style={{ fontSize: 12.5, fontWeight: 600, display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={ing.path}>{ing.label}</span>
        {/* The share is what the merge really applies — in average mode a weight
            of 1 next to a weight of 3 means 25%, not 100%. */}
        <span className="ds-cell-key">
          {[ing.origin, net].filter(Boolean).join(' · ')} · <span>{method === 'average' ? `${Math.round(share * 100)}%` : `×${share.toFixed(2)}`}</span>
        </span>
      </span>
      <span style={{ width: 170, display: 'flex', alignItems: 'center', gap: 9 }}>
        <input
          type="range"
          className="ds-range"
          min={0} max={2} step={0.05}
          value={Math.min(2, ing.weight)}
          style={{ '--p': `${(Math.min(2, ing.weight) / 2) * 100}%` } as React.CSSProperties}
          onChange={(e) => onWeight(Number(e.target.value))}
          aria-label={t('soup.weightSlider')}
        />
        <input
          type="number"
          min={0} step={0.05}
          value={ing.weight}
          onChange={(e) => onWeight(Number(e.target.value))}
          aria-label={t('soup.weight')}
          className="ds-mono"
          style={{ width: 42, fontSize: 11.5, textAlign: 'right', border: 0, background: 'transparent', padding: 0, outline: 'none' }}
        />
      </span>
      <button type="button" className="ds-kebab" onClick={onRemove} aria-label={t('common.delete')} title={t('common.delete')}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>
      </button>
    </div>
  )
}

function Verdict({ verdict, checking, count }: {
  verdict: SoupCompatibility | null; checking: boolean; count: number
}) {
  const { t } = useTranslation()
  if (count < 2) return <div className="ds-ctl-note">{t('soup.needTwo')}</div>
  if (checking) return <div className="ds-note ds-info"><span>{t('soup.checking')}</span></div>
  if (!verdict) return null
  return (
    <>
      {verdict.errors.map((e, i) => (
        <div key={`e${i}`} className="ds-note ds-err"><span>{e}</span></div>
      ))}
      {verdict.warnings.map((w, i) => (
        <div key={`w${i}`} className="ds-note ds-warn"><span>{w}</span></div>
      ))}
      {verdict.ok && verdict.warnings.length === 0 && (
        <div className="ds-note ds-ok"><span>{t('soup.compatible')}</span></div>
      )}
    </>
  )
}

/** Modal picker: project checkpoints (grouped by version), uploads, past soups. */
function SourcePicker({ uploads, outputs, chosen, onPick, onClose, onUploadsChanged }: {
  uploads: SoupFile[]
  outputs: SoupFile[]
  chosen: Set<string>
  onPick: (i: Ingredient) => void
  onClose: () => void
  onUploadsChanged: () => Promise<void>
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [projectId, setProjectId] = useState<number | null>(null)
  const [groups, setGroups] = useState<VersionCkptGroup<LoraCkpt>[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api.listProjects()
      .then((ps) => {
        setProjects(ps)
        if (ps.length) setProjectId(ps[0].id)
      })
      .catch(() => setProjects([]))
  }, [])

  useEffect(() => {
    if (projectId == null) { setGroups([]); return }
    let alive = true
    setLoading(true)
    api.listProjectLoraCkpts(projectId)
      .then((g) => { if (alive) setGroups(g) })
      .catch(() => { if (alive) setGroups([]) })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [projectId])

  const project = projects.find((p) => p.id === projectId)
  const pick = (ing: Ingredient) => { onPick(ing); onClose() }

  const removeUpload = async (file: SoupFile) => {
    try {
      await api.deleteSoupUpload(file.name)
      await onUploadsChanged()
    } catch (e) {
      toast((e as ApiError).message || t('soup.deleteFailed'), 'error')
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-0 sm:p-6"
      style={{ background: 'rgba(20,22,20,.18)' }}
      onClick={onClose}
    >
      <div
        className="ds-card w-full sm:max-w-[640px] max-h-[85vh] overflow-auto"
        style={{ boxShadow: '0 24px 48px -24px rgba(20,22,20,.3)' }}
        onClick={(e) => e.stopPropagation()}
        role="dialog" aria-modal="true" aria-label={t('soup.pickerTitle')}
      >
        <div className="ds-card-head ds-pad" style={{ position: 'sticky', top: 0, background: 'var(--panel)', zIndex: 1, borderBottom: '1px solid var(--line)' }}>
          <div className="ds-card-title" style={{ flex: 1 }}>{t('soup.pickerTitle')}</div>
          <button type="button" className="ds-kebab" onClick={onClose} aria-label={t('common.cancel')}>✕</button>
        </div>

        <div className="p-5 flex flex-col gap-5">
          <div className="flex flex-col gap-2">
            <span className="ds-cap">{t('soup.fromProject')}</span>
            {projects.length === 0 ? (
              <p className="m-0 text-xs text-fg-muted">{t('soup.noProjects')}</p>
            ) : (
              <select
                className="ds-inp" value={projectId ?? ''}
                onChange={(e) => setProjectId(Number(e.target.value))}
              >
                {projects.map((p) => (
                  <option key={p.id} value={p.id}>{p.title}</option>
                ))}
              </select>
            )}
            {loading && <p className="m-0 text-xs text-fg-muted">{t('common.loading')}</p>}
            {!loading && groups.map((g) => (
              <div key={g.version_id} className="flex flex-col gap-1">
                <div className="ds-cap mt-2">{g.label}</div>
                {g.items.length === 0 && (
                  <p className="m-0 text-xs text-fg-muted">{t('soup.noCheckpoints')}</p>
                )}
                {g.items.map((c) => (
                  <PickRow
                    key={c.path}
                    label={basename(c.path)}
                    sub={`${project?.title ?? ''} · ${g.label} · ${c.label}`}
                    disabled={chosen.has(c.path)}
                    onPick={() => pick({
                      path: c.path,
                      label: basename(c.path),
                      origin: `${project?.title ?? ''} · ${g.label}`,
                      weight: 1,
                    })}
                  />
                ))}
              </div>
            ))}
          </div>

          <Section title={t('soup.fromUploads')} empty={t('soup.noUploads')} items={uploads}>
            {(f) => (
              <PickRow
                key={f.path} label={f.name} sub={fmtSize(f.size)}
                disabled={chosen.has(f.path)}
                onPick={() => pick({ path: f.path, label: f.name, origin: t('soup.originUpload'), weight: 1 })}
                onRemove={() => void removeUpload(f)}
              />
            )}
          </Section>

          <Section title={t('soup.fromSoups')} empty={t('soup.noSoups')} items={outputs}>
            {(f) => (
              <PickRow
                key={f.path} label={f.name} sub={fmtSize(f.size)}
                disabled={chosen.has(f.path)}
                onPick={() => pick({ path: f.path, label: f.name, origin: t('soup.originSoup'), weight: 1 })}
              />
            )}
          </Section>
        </div>
      </div>
    </div>
  )
}

function Section({ title, empty, items, children }: {
  title: string; empty: string; items: SoupFile[]
  children: (f: SoupFile) => React.ReactNode
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="ds-cap">{title}</span>
      {items.length === 0
        ? <p className="m-0 text-xs text-fg-muted">{empty}</p>
        : items.map(children)}
    </div>
  )
}

function PickRow({ label, sub, disabled, onPick, onRemove }: {
  label: string; sub: string; disabled?: boolean
  onPick: () => void; onRemove?: () => void
}) {
  const { t } = useTranslation()
  return (
    <div className="flex items-center gap-2">
      <button
        type="button" disabled={disabled} onClick={onPick}
        className="flex-1 min-w-0 text-left p-2.5 rounded-md hover:bg-sunken disabled:opacity-40 disabled:cursor-default"
      >
        <div className="truncate ds-mono" style={{ fontSize: 12.5 }}>{label}</div>
        <div className="truncate ds-cell-key">{disabled ? t('soup.alreadyAdded') : sub}</div>
      </button>
      {onRemove && (
        <button type="button" className="ds-kebab" onClick={onRemove} aria-label={t('common.delete')}>✕</button>
      )}
    </div>
  )
}
