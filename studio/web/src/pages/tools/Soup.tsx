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
  type VersionCkptGroup,
} from '../../api/client'
import { useDialog } from '../../components/Dialog'
import PageHeader from '../../components/PageHeader'
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

  return (
    <div className="flex flex-col h-full overflow-auto">
      <PageHeader
        eyebrow={t('soup.eyebrow')}
        title={t('soup.title')}
        subtitle={t('soup.subtitle')}
        actions={
          <>
            <input
              ref={fileRef} type="file" accept=".safetensors" multiple hidden
              onChange={(e) => void handleUpload(e.target.files)}
            />
            <button className="btn btn-secondary btn-sm" onClick={() => fileRef.current?.click()}>
              {t('soup.upload')}
            </button>
            <button className="btn btn-primary btn-sm" onClick={() => setPickerOpen(true)}>
              {t('soup.addCheckpoint')}
            </button>
          </>
        }
      />

      <div className="flex-1 p-6 flex flex-col lg:flex-row gap-5 items-start">
        {/* ── ingredients ─────────────────────────────────────────────── */}
        <div className="flex-1 min-w-0 w-full flex flex-col gap-4">
          {items.length === 0 ? (
            <EmptyState
              title={t('soup.emptyTitle')}
              body={t('soup.emptyBody')}
              cta={t('soup.addCheckpoint')}
              onCta={() => setPickerOpen(true)}
            />
          ) : (
            <div className="card p-0 overflow-hidden">
              {items.map((ing, idx) => (
                <IngredientRow
                  key={ing.path}
                  ing={ing}
                  share={eff[idx]}
                  method={method}
                  onWeight={(w) => setWeight(ing.path, w)}
                  onRemove={() => remove(ing.path)}
                />
              ))}
            </div>
          )}

          {outputs.length > 0 && (
            <div className="card p-5">
              <div className="caption mb-3">{t('soup.previousSoups')}</div>
              <div className="flex flex-col gap-2">
                {outputs.map((f) => (
                  <div key={f.path} className="flex items-center gap-3 flex-wrap">
                    <div className="flex-1 min-w-0">
                      <div className="truncate text-sm">{f.name}</div>
                      <div className="text-xs text-fg-muted">{fmtSize(f.size)}</div>
                    </div>
                    <button className="btn btn-ghost btn-sm" onClick={() => handleTest(f.path)}>
                      {t('soup.test')}
                    </button>
                    <a className="btn btn-ghost btn-sm" href={api.soupDownloadUrl(f.name)} download>
                      {t('soup.download')}
                    </a>
                    <button
                      className="btn btn-ghost btn-sm" style={{ color: 'var(--err)' }}
                      onClick={() => void handleDeleteOutput(f)}
                    >
                      {t('common.delete')}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── recipe ──────────────────────────────────────────────────── */}
        <div className="w-full lg:w-[380px] shrink-0 flex flex-col gap-4">
          <div className="card p-5 flex flex-col gap-4">
            <div>
              <div className="caption mb-2">{t('soup.method')}</div>
              <div className="flex gap-2">
                {(['average', 'sum'] as Method[]).map((m) => (
                  <button
                    key={m}
                    className={`btn btn-sm flex-1 ${method === m ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setMethod(m)}
                  >
                    {t(`soup.method_${m}`)}
                  </button>
                ))}
              </div>
              <p className="mt-2 mb-0 text-xs text-fg-muted">{t(`soup.methodHelp_${method}`)}</p>
            </div>

            <label className="flex flex-col gap-1.5">
              <span className="caption">{t('soup.outputName')}</span>
              <input
                className="input" value={name}
                placeholder={suggestedName || t('soup.outputNamePlaceholder')}
                onChange={(e) => setName(e.target.value)}
              />
            </label>

            <Verdict verdict={verdict} checking={checking} count={items.length} />

            <button className="btn btn-primary" disabled={!canMerge} onClick={() => void handleMerge()}>
              {merging ? t('soup.merging') : t('soup.merge')}
            </button>
          </div>

          {result && (
            <div className="card p-5 flex flex-col gap-3">
              <div className="caption">{t('soup.resultTitle')}</div>
              <div>
                <div className="font-medium break-all">{result.name}</div>
                <div className="text-xs text-fg-muted">{fmtSize(result.size)}</div>
              </div>
              {result.warnings.map((w, i) => (
                <p key={i} className="m-0 text-xs" style={{ color: 'var(--warn)' }}>{w}</p>
              ))}
              <button className="btn btn-primary btn-sm" onClick={() => handleTest(result.path)}>
                {t('soup.testNow')}
              </button>
              <a className="btn btn-secondary btn-sm" href={api.soupDownloadUrl(result.name)} download>
                {t('soup.download')}
              </a>
            </div>
          )}
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

// ── pieces ──────────────────────────────────────────────────────────────────

function EmptyState({ title, body, cta, onCta }: {
  title: string; body: string; cta: string; onCta: () => void
}) {
  return (
    <div className="card p-10 text-center">
      <div className="text-lg font-semibold mb-2">{title}</div>
      <p className="text-fg-secondary text-sm max-w-[460px] mx-auto mb-5">{body}</p>
      <button className="btn btn-primary btn-sm" onClick={onCta}>{cta}</button>
    </div>
  )
}

function IngredientRow({ ing, share, method, onWeight, onRemove }: {
  ing: Ingredient
  share: number
  method: Method
  onWeight: (w: number) => void
  onRemove: () => void
}) {
  const { t } = useTranslation()
  return (
    <div className="p-4 border-b border-subtle last:border-b-0 flex flex-col sm:flex-row sm:items-center gap-3">
      <div className="flex-1 min-w-0">
        <div className="truncate font-medium text-sm">{ing.label}</div>
        <div className="text-xs text-fg-muted truncate">{ing.origin}</div>
      </div>
      <div className="flex items-center gap-3">
        <input
          type="range" min={0} max={2} step={0.05} value={ing.weight}
          onChange={(e) => onWeight(Number(e.target.value))}
          className="w-[120px]" aria-label={t('soup.weight')}
        />
        <input
          type="number" min={0} step={0.05} value={ing.weight}
          onChange={(e) => onWeight(Number(e.target.value))}
          className="input w-[76px]" aria-label={t('soup.weight')}
        />
        {/* The share is what the merge really applies — in average mode a weight
            of 1 next to a weight of 3 means 25%, not 100%. */}
        <div className="w-[52px] text-right text-xs text-fg-muted tabular-nums">
          {method === 'average' ? `${Math.round(share * 100)}%` : `×${share.toFixed(2)}`}
        </div>
        <button className="btn btn-ghost btn-sm" onClick={onRemove} aria-label={t('common.delete')}>✕</button>
      </div>
    </div>
  )
}

function Verdict({ verdict, checking, count }: {
  verdict: SoupCompatibility | null; checking: boolean; count: number
}) {
  const { t } = useTranslation()
  if (count < 2) return <p className="m-0 text-xs text-fg-muted">{t('soup.needTwo')}</p>
  if (checking) return <p className="m-0 text-xs text-fg-muted">{t('soup.checking')}</p>
  if (!verdict) return null
  return (
    <div className="flex flex-col gap-1.5">
      {verdict.errors.map((e, i) => (
        <p key={`e${i}`} className="m-0 text-xs" style={{ color: 'var(--err)' }}>{e}</p>
      ))}
      {verdict.warnings.map((w, i) => (
        <p key={`w${i}`} className="m-0 text-xs" style={{ color: 'var(--warn)' }}>{w}</p>
      ))}
      {verdict.ok && verdict.warnings.length === 0 && (
        <p className="m-0 text-xs" style={{ color: 'var(--ok)' }}>{t('soup.compatible')}</p>
      )}
    </div>
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
      className="fixed inset-0 z-50 bg-black/50 flex items-end sm:items-center justify-center p-0 sm:p-6"
      onClick={onClose}
    >
      <div
        className="card w-full sm:max-w-[640px] max-h-[85vh] overflow-auto p-0"
        onClick={(e) => e.stopPropagation()}
        role="dialog" aria-modal="true" aria-label={t('soup.pickerTitle')}
      >
        <div className="p-5 border-b border-subtle flex items-center gap-3 sticky top-0 bg-surface z-[1]">
          <div className="flex-1 font-semibold">{t('soup.pickerTitle')}</div>
          <button className="btn btn-ghost btn-sm" onClick={onClose} aria-label={t('common.cancel')}>✕</button>
        </div>

        <div className="p-5 flex flex-col gap-5">
          <div className="flex flex-col gap-2">
            <span className="caption">{t('soup.fromProject')}</span>
            {projects.length === 0 ? (
              <p className="m-0 text-xs text-fg-muted">{t('soup.noProjects')}</p>
            ) : (
              <select
                className="input" value={projectId ?? ''}
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
                <div className="caption mt-2">{g.label}</div>
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
      <span className="caption">{title}</span>
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
        className="flex-1 min-w-0 text-left p-2.5 rounded-md hover:bg-elevated disabled:opacity-40 disabled:cursor-default"
      >
        <div className="truncate text-sm">{label}</div>
        <div className="truncate text-xs text-fg-muted">{disabled ? t('soup.alreadyAdded') : sub}</div>
      </button>
      {onRemove && (
        <button className="btn btn-ghost btn-sm" onClick={onRemove} aria-label={t('common.delete')}>✕</button>
      )}
    </div>
  )
}
