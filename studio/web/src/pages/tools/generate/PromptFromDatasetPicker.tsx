import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type CaptionEntry, type ProjectSummary } from '../../../api/client'
import { useLocalStorageState } from '../../../lib/useLocalStorageState'
import ImagePreviewModal from '../../../components/ImagePreviewModal'

// Naming prefix matches useAdvancedMode's `studio:` convention (PR #66 P1-4). The old
// `anima.generate.promptDataset.*` keys are migrated once on mount, then discarded.
const LAST_PROJECT_KEY = 'studio:generate:promptDataset:projectId'
const LAST_VERSION_KEY = 'studio:generate:promptDataset:versionId'
const LEGACY_PROJECT_KEY = 'anima.generate.promptDataset.projectId'
const LEGACY_VERSION_KEY = 'anima.generate.promptDataset.versionId'

function migrateLegacyKey(legacyKey: string, newKey: string): void {
  if (typeof window === 'undefined') return
  if (window.localStorage.getItem(newKey) !== null) return
  const raw = window.localStorage.getItem(legacyKey)
  if (raw === null) return
  const n = Number(raw)
  if (Number.isFinite(n)) {
    window.localStorage.setItem(newKey, JSON.stringify(n))
  }
  window.localStorage.removeItem(legacyKey)
}

export interface DatasetPick {
  projectId: number
  versionId: number
  /** Training-set image filename (CaptionEntry.name, e.g. "0001.png") */
  name: string
  /**
   * Subfolder the image lives in (CaptionEntry.folder, e.g. "5_concept"). Optional: older
   * snapshots (serialized before this field existed) lack it, so the thumbnail just doesn't
   * render -- it doesn't affect tag concatenation.
   */
  folder?: string
  /** Tag list parsed out of the caption text, in the training set's original order */
  tags: string[]
}

/** Pick one caption from the training set to use as a prompt suffix for generation (not written into the sidebar's "positive" textarea).
 *
 * Controlled single-select:
 * - The parent controls open/close (x fires onClose); generating doesn't auto-close it.
 * - Selection state (DatasetPick) is held by the parent -- **the parent must also clear
 *   `value` when it closes the picker**, otherwise datasetPick.tags keeps getting appended to
 *   the prompt by handleGenerate while the user thinks nothing is selected anymore.
 * - Clicking a list row: unselected -> activates it; already-selected same row -> deselects it.
 * - The selected caption's tags are shown in a read-only textarea at the bottom, never written into the prompt box above.
 *
 * pid/vid represent "currently browsing" state, decoupled from `value` -- browsing to a
 * different project/version doesn't affect `value`; persisted across sessions via localStorage
 * to remember the browsing position.
 */
export default function PromptFromDatasetPicker({
  value, onChange, onClose,
}: {
  /** Currently selected caption (null = none) */
  value: DatasetPick | null
  onChange: (next: DatasetPick | null) => void
  onClose: () => void
}) {
  const { t } = useTranslation()
  // One-time migration of the old anima.* keys to the studio: naming (PR #66 P1-4 convention);
  // calling it at the top of the module body is fine -- it has no React state read/write side
  // effects, it only touches localStorage, so it doesn't need a useEffect.
  if (typeof window !== 'undefined') {
    migrateLegacyKey(LEGACY_PROJECT_KEY, LAST_PROJECT_KEY)
    migrateLegacyKey(LEGACY_VERSION_KEY, LAST_VERSION_KEY)
  }

  const [projects, setProjects] = useState<ProjectSummary[]>([])
  // useLocalStorageState's default only takes effect when storage has no value; when `value` is set, use it as the initial "browsing position"
  const [pid, setPid] = useLocalStorageState<number | null>(LAST_PROJECT_KEY, value?.projectId ?? null)
  const [vid, setVid] = useLocalStorageState<number | null>(LAST_VERSION_KEY, value?.versionId ?? null)
  // History backfill: when value switches to a (projectId, versionId), the browsing pid/vid
  // follows it -- otherwise the caption list would stay on the version the user last browsed,
  // never highlight the current value.name row, and show the bottom tags orphaned with nothing
  // matching them on screen. This doesn't persist to the outside state (localStorage); it only
  // updates in-memory state, so closing and reopening the picker still returns to wherever the
  // user last browsed by hand.
  useEffect(() => {
    if (value == null) return
    setPid((cur) => (cur === value.projectId ? cur : value.projectId))
    setVid((cur) => (cur === value.versionId ? cur : value.versionId))
  }, [value?.projectId, value?.versionId])  // eslint-disable-line react-hooks/exhaustive-deps
  const [versions, setVersions] = useState<Array<{ id: number; label: string }>>([])
  const [captions, setCaptions] = useState<CaptionEntry[]>([])
  // The (pid, vid) that `captions` actually belongs to. Row/preview thumbnail URLs always use
  // this, never the live pid/vid -- during the frame where a project/version switch is in
  // flight, the old captions are still rendering, and using the live pid/vid would attach stale
  // filenames to the new project, 404'ing the whole column (black images). Once bound to their
  // source, old images keep pointing at their own (pid, vid) until replaced by new data, so they
  // always resolve to a real file. Written atomically with `captions` when the response lands, so the two never drift apart.
  const [loaded, setLoaded] = useState<{ pid: number; vid: number } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  // The hovered caption key, driving the bottom large preview (moving away falls back to the selected value's image)
  const [hoveredKey, setHoveredKey] = useState<string | null>(null)
  // Clicking the bottom large preview zooms it into a fullscreen modal. What's stored is a
  // snapshot of the preview's location info at click time, not the live previewMeta -- moving
  // the mouse into the fullscreen-covering modal leaves the picker, triggering the root's
  // onMouseLeave to clear hoveredKey; an image that tracked the live value would vanish
  // instantly. Decoupling it into a snapshot keeps it stable until closed manually.
  const [zoomMeta, setZoomMeta] = useState<
    { pid: number; vid: number; name: string; folder?: string } | null
  >(null)

  // 1. Fetch the project list; if the remembered pid no longer exists in it, clear it to avoid a ghost selection
  useEffect(() => {
    void api.listProjects()
      .then((items) => {
        setProjects(items)
        if (pid != null && !items.some((p) => p.id === pid)) setPid(null)
      })
      .catch((e) => setError(String(e)))
    // Putting pid in the deps would trigger repeated refetches; once on mount is enough
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 2. Once a project is picked, fetch its versions; prefer reusing vid if that version still exists in the new project
  useEffect(() => {
    if (!pid) { setVersions([]); setVid(null); return }
    // Same stale-response guard: rapidly switching projects can let an old getProject call
    // resolve late and feed another project's versions / default vid in, which would then
    // indirectly leak into the captions effect below.
    let cancelled = false
    void api.getProject(pid)
      .then((p) => {
        if (cancelled) return
        const vs = p.versions.map((v) => ({ id: v.id, label: v.label }))
        setVersions(vs)
        if (vs.length > 0) {
          setVid((cur) => (cur && vs.some((v) => v.id === cur) ? cur : vs[0].id))
        } else {
          setVid(null)
        }
      })
      .catch((e) => { if (!cancelled) setError(String(e)) })
    return () => { cancelled = true }
    // Same as above: vid is only read inside the effect, kept out of the deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid])

  // 3. Once a version is picked, fetch captions
  useEffect(() => {
    if (!pid || !vid) { setCaptions([]); setLoaded(null); return }
    // Stale-response guard: rapidly switching project/version can make an old request resolve
    // after the new one; without this guard the old captions would overwrite the new ones and
    // mismatch the current selection (showing another project's images). Matches the same guard
    // in InlineLoraPicker. `captions` is written together with its source (pid, vid), which
    // anchors thumbnail URLs (see the `loaded` comment) -- the two update atomically and never drift.
    let cancelled = false
    setLoading(true)
    setError(null)
    void api.listCaptionsFull(pid, vid)
      .then((r) => {
        if (cancelled) return
        setCaptions(r.items)
        setLoaded({ pid, vid })
        setLoading(false)
      })
      .catch((e) => {
        if (cancelled) return
        setError(String(e))
        setLoading(false)
      })
    return () => { cancelled = true }
  }, [pid, vid])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return captions
    return captions.filter((c) =>
      c.name.toLowerCase().includes(q) ||
      c.tags.some((t) => t.toLowerCase().includes(q))
    )
  }, [captions, search])

  // The key in the current list matching the selected caption (highlighted only when the
  // loaded captions and `value` share the same (pid, vid) -- uses `loaded`, not the live
  // pid/vid, to stay consistent with the list/thumbnail source)
  const selectedKeyInList = useMemo(() => {
    if (!value || !loaded || value.projectId !== loaded.pid || value.versionId !== loaded.vid) return null
    return value.name
  }, [value, loaded])

  const tagsText = value ? value.tags.join(', ') : ''

  // Bottom large-preview source: prefers the hovered row (browsing pid/vid), then falls back to
  // the selected value (using value's own project/version/folder, decoupled from the browsing
  // position). A value from an old snapshot with no folder simply doesn't render.
  const hoveredCaption = hoveredKey
    ? captions.find((c) => `${c.folder}/${c.name}` === hoveredKey) ?? null
    : null
  // Location info for the current preview image, shared by the thumbnail (512) and the zoomed view (1600); null = nothing to show.
  const previewMeta =
    hoveredCaption && loaded
      ? { pid: loaded.pid, vid: loaded.vid, name: hoveredCaption.name, folder: hoveredCaption.folder }
      : value && value.folder
        ? { pid: value.projectId, vid: value.versionId, name: value.name, folder: value.folder }
        : null
  const previewSrc = previewMeta
    ? api.versionThumbUrl(previewMeta.pid, previewMeta.vid, 'train', previewMeta.name, previewMeta.folder, 512)
    : ''

  const handleRowClick = (c: CaptionEntry) => {
    // The row belongs to the `loaded` set of captions, so the selection must also land on
    // loaded's (pid, vid), not the live pid/vid (the two can briefly disagree mid-switch,
    // which would otherwise write the wrong project into datasetPick).
    if (!loaded) return
    if (
      value
      && value.projectId === loaded.pid
      && value.versionId === loaded.vid
      && value.name === c.name
    ) {
      // Deselect
      onChange(null)
      return
    }
    onChange({
      projectId: loaded.pid,
      versionId: loaded.vid,
      name: c.name,
      folder: c.folder,
      tags: c.tags,
    })
  }

  return (
    <>
    {/* onMouseLeave is bound to the whole picker (not just the list): moving from a list row to
        the bottom large preview to click and zoom it must not lose hover -- otherwise, in the
        unselected case the preview and its zoom button would vanish (clearing on hoveredKey)
        before they could be clicked, and in the selected case it would fall back to zooming a
        different image (value's). Only leaving the whole picker clears it and falls back to value. */}
    <div
      className="rounded-md border border-subtle bg-overlay p-2.5 flex flex-col gap-2"
      data-testid="prompt-dataset-picker"
      onMouseLeave={() => setHoveredKey(null)}
    >
      {/* header */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-fg-secondary shrink-0">{t('generate.datasetPromptTitle')}</span>
        <span className="flex-1" />
        {value && (
          <button
            onClick={() => onChange(null)}
            className="btn btn-ghost btn-sm text-2xs text-fg-tertiary"
            title={t('generate.clearDatasetPickTitle')}
          >
            {t('generate.clearDatasetPick')}
          </button>
        )}
        <button
          onClick={onClose}
          className="btn btn-ghost btn-sm text-fg-tertiary px-1.5"
          title={t('generate.closeDatasetPickerTitle')}
          aria-label={t('common.close')}
        >
          ×
        </button>
      </div>

      {/* project / version selection */}
      <div className="flex gap-2">
        <select
          className="input text-xs flex-1"
          value={pid ?? ''}
          onChange={(e) => setPid(e.target.value ? Number(e.target.value) : null)}
          aria-label={t('generate.selectProjectAria')}
        >
          <option value="">{t('generate.selectProject')}</option>
          {projects.map((p) => (
            <option key={p.id} value={p.id}>{p.title}</option>
          ))}
        </select>
        <select
          className="input text-xs flex-1"
          value={vid ?? ''}
          onChange={(e) => setVid(e.target.value ? Number(e.target.value) : null)}
          disabled={versions.length === 0}
          aria-label={t('generate.selectVersionAria')}
        >
          <option value="">{t('generate.selectVersion')}</option>
          {versions.map((v) => (
            <option key={v.id} value={v.id}>{v.label}</option>
          ))}
        </select>
      </div>

      {/* search */}
      <input
        type="text"
        className="input text-xs"
        placeholder={t('generate.searchFilenameTag')}
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        disabled={!pid || !vid || captions.length === 0}
      />

      {error && <div className="text-2xs text-err">{error}</div>}

      {/* caption list */}
      <div
        className="flex flex-col gap-px overflow-y-auto"
        style={{ maxHeight: 320 }}
      >
        {loading && <div className="text-2xs text-fg-tertiary">{t('common.loading')}</div>}
        {!loading && pid && vid && captions.length === 0 && !error && (
          <div className="text-2xs text-fg-tertiary">{t('generate.noCaptions')}</div>
        )}
        {!loading && filtered.map((c) => {
          const k = `${c.folder}/${c.name}`
          const active = selectedKeyInList === c.name
          // In-row thumbnails request 64px (~2x display size) so high-DPI screens stay sharp;
          // native lazy loading means rows that haven't scrolled into view don't fire a request,
          // so long lists don't pull every image down at once. The URL uses `loaded`, not the
          // live pid/vid, so the filename and (pid, vid) always belong to the same set and never 404.
          const thumb = loaded
            ? api.versionThumbUrl(loaded.pid, loaded.vid, 'train', c.name, c.folder, 64)
            : ''
          return (
            <button
              key={k}
              onClick={() => handleRowClick(c)}
              onMouseEnter={() => setHoveredKey(k)}
              className="flex items-center gap-2 px-2 py-1.5 rounded text-xs text-left border-none transition-colors"
              style={{
                background: active ? 'var(--accent-soft)' : 'transparent',
                color: active ? 'var(--accent)' : 'var(--fg-secondary)',
                cursor: 'pointer',
              }}
            >
              {thumb && (
                <img
                  src={thumb}
                  alt=""
                  loading="lazy"
                  className="shrink-0 rounded object-cover bg-sunken"
                  style={{ width: 36, height: 36 }}
                  onError={(e) => { e.currentTarget.style.visibility = 'hidden' }}
                />
              )}
              <span className="font-mono text-2xs shrink-0">{active ? '✓' : '+'}</span>
              <div className="flex-1 min-w-0">
                <div className="font-medium truncate">{c.name}</div>
                <div className="text-2xs text-fg-tertiary truncate">
                  {c.tags.slice(0, 6).join(', ')}{c.tags.length > 6 ? ` (+${c.tags.length - 6})` : ''}
                </div>
              </div>
            </button>
          )
        })}
      </div>

      <label className="caption block mt-1">{t('generate.selectedDatasetTagsLabel')}</label>
      {/* Top: the training-set large image (hovered row / selected row), for eyeballing against
          generated results; bottom: read-only tags. object-contain shows the whole image
          uncropped (matches the large-preview convention used by TagEdit / Preprocess). */}
      <div className="flex flex-col gap-2">
        <div
          className="rounded border border-subtle bg-sunken overflow-hidden flex items-center justify-center"
          style={{ height: 240 }}
        >
          {previewSrc ? (
            <button
              type="button"
              onClick={() => previewMeta && setZoomMeta(previewMeta)}
              className="w-full h-full flex items-center justify-center border-none bg-transparent p-0 cursor-zoom-in"
              title={t('generate.datasetPreviewZoomTitle')}
              aria-label={t('generate.datasetPreviewZoomTitle')}
            >
              <img
                src={previewSrc}
                alt={t('generate.datasetPreviewAlt')}
                loading="lazy"
                className="w-full h-full object-contain"
                onError={(e) => { e.currentTarget.style.visibility = 'hidden' }}
              />
            </button>
          ) : (
            <span className="text-2xs text-fg-tertiary px-1.5 text-center leading-snug">
              {t('generate.datasetPreviewEmpty')}
            </span>
          )}
        </div>
        {/* Minimum height matches "positive" (PromptList rows=5 text-sm): 5x20 line-height + .input's 14 padding + 2 border = 116 */}
        <textarea
          className="input w-full font-mono text-xs resize-y"
          style={{ minHeight: 116 }}
          value={tagsText}
          readOnly
          placeholder={t('generate.selectedDatasetTagsPlaceholder')}
          aria-label={t('generate.selectedDatasetTagsAria')}
        />
      </div>
    </div>
    {zoomMeta && (
      <ImagePreviewModal
        src={api.versionThumbUrl(zoomMeta.pid, zoomMeta.vid, 'train', zoomMeta.name, zoomMeta.folder, 1600)}
        caption={zoomMeta.name}
        onClose={() => setZoomMeta(null)}
      />
    )}
    </>
  )
}
