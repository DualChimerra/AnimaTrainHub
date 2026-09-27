import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { type LoraCkpt } from '../../../api/client'
import type { LoraCatalog } from './useLoraCatalog'

function basenameOf(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

/** Project abbreviation icon (2-char uppercase). Referenced by SidebarLoras / legacy code. */
export function projectAbbr(title: string): string {
  const cleaned = title.replace(/[^a-zA-Z0-9]/g, '')
  return (cleaned.slice(0, 2) || '??').toUpperCase()
}

export interface PickedLora {
  path: string
  projectId: number | null
  versionId: number | null
}

interface CommonProps {
  /** Lazy cascading data source: the picker fetches projects / versions / ckpts on demand (see useLoraCatalog) */
  catalog: LoraCatalog
  /** × button callback: single mode = delete the whole slot; multi mode = close the inline panel */
  onClose: () => void
  onPickExternal?: () => void
  /** Rendered inside a LoRA slot card that already has its own name row, ×
   *  and weight slider: drop the frame, title, × and weight. */
  embedded?: boolean
}

interface SingleModeProps extends CommonProps {
  mode: 'single'
  /** The ckpt bound to this slot (null = empty slot). Controlled. */
  value: PickedLora | null
  /** Current weight. Controlled. */
  weight: number
  /** Fires on any value/weight change. */
  onChange: (next: PickedLora | null, weight: number) => void
  /** showWeight is forced to true (single mode = one LoRA slot, always has a weight). */
  showWeight?: never
  existingPaths?: never
}

interface MultiModeProps extends CommonProps {
  mode?: 'multi'
  /** Paths already picked by the caller (other LoRA slots / other axes) — shown checked and disabled in the list */
  existingPaths?: Set<string>
  /** Fires on chip toggle / weight / pid-vid change.
   *
   * - Normal multi mode: fires when the user clicks "Add N" to commit; the picker then auto-closes.
   * - live mode: fires immediately on every chip toggle / weight change / pid-vid switch (no commit button).
   */
  onPick: (picks: PickedLora[], weight: number) => void
  /** Hide the weight row for XY axis bindings (the axis card has its own lora_scale control) */
  showWeight?: boolean
  defaultWeight?: number
  /** Live commit: every chip toggle calls onPick immediately instead of rendering an "Add N" commit footer.
   *  Used by XY axis cards, where the picker stays mounted and the user expects what-you-see-is-what-you-get. */
  live?: boolean
  /** Controlled selection set (only meaningful in live mode): the picker keeps its internal `picked` in sync with this array.
   *
   * Elements can be a full path or a basename: a raw string can just be split and passed in, and the picker
   * matches basenames against ckpts to highlight the full path. On a basename match it immediately calls onPick
   * to write back the full path (fixes history backfill sending a basename, which the daemon reports as "path not found").
   *
   * undefined = the picker uses its own internal picked state (active task / bulk-add flow). */
  selectedPaths?: string[]
  /** Controlled initial pid/vid (live mode only): paired with selectedPaths so the picker anchors to the
   *  right project/version on mount, otherwise it falls back to projects[0]. */
  initialPid?: number | null
  initialVid?: number | null
  value?: never
  weight?: never
  onChange?: never
}

type Props = SingleModeProps | MultiModeProps

/** Inline LoRA picker: project + version dropdowns -> ckpt chip list -> single/multi select + weight.
 *
 * **single mode** (controlled): one picker = one LoRA slot. Clicking a chip swaps the slot's ckpt;
 *   clicking the same chip again clears it (empty slot). Changing the weight slider fires onChange
 *   immediately. × deletes the slot.
 *
 * **multi mode** (XY axis / bulk add): toggle multi-select + a weight footer + an "Add N" button that
 *   commits once; onClose fires automatically after commit. × cancels the inline panel. XY usage passes
 *   `showWeight=false` to hide the weight row.
 */
export default function InlineLoraPicker(props: Props) {
  const { t } = useTranslation()
  const { catalog, onClose, onPickExternal, embedded = false } = props
  // Destructure the stable loaders (useCallback) plus reactive data; effect deps use plain identifiers.
  const { projects, ensureProjects, ensureVersions, versionsOf, fetchCkpts } = catalog
  const isSingle = props.mode === 'single'
  const showWeight = isSingle ? true : (props.showWeight ?? true)
  const existingPaths = isSingle ? new Set<string>() : (props.existingPaths ?? new Set<string>())

  // Project dropdown: fetch the project list lazily on mount (catalog caches + dedupes, so it's a no-op if the picker isn't open)
  useEffect(() => { ensureProjects() }, [ensureProjects])

  // Multi mode controlled anchor: adopt caller-supplied initialPid/Vid directly (history backfill goes through here)
  const multiInitialPid = !isSingle ? (props as MultiModeProps).initialPid ?? null : null
  const multiInitialVid = !isSingle ? (props as MultiModeProps).initialVid ?? null : null
  // Initial pid/vid: single mode uses `value`; multi mode falls back to initialPid/Vid.
  // Under lazy cascading the project list isn't there yet on mount -> start at null;
  // once projects arrive, the effect below auto-anchors to the first project (preserving
  // the existing UX of "open the picker and immediately see the first project's ckpts").
  const initialPid = isSingle
    ? (props.value?.projectId ?? null)
    : (multiInitialPid ?? null)
  const initialVid = isSingle
    ? (props.value?.versionId ?? null)
    : (multiInitialVid ?? null)

  const [pid, setPid] = useState<number | null>(initialPid)

  // Once pid settles (anchor backfill or user pick) -> lazily fetch that project's versions
  useEffect(() => {
    if (pid != null) ensureVersions(pid)
  }, [ensureVersions, pid])

  // Decision #8 (plan §9.2): single mode is controlled, so pid must stay in sync with
  // props.value.projectId — otherwise a history backfill / URL ?lora= flowing into a new
  // LoraEntry would leave the dropdown stuck on the old value (previously papered over by
  // the parent bumping urlConsumedKey to force a remount; removed in Step 6).
  // Functional setPid update + skip when the value is unchanged, to avoid an infinite loop.
  // Not synced when value=null (keeps the fallback of showing projects[0]'s ckpts when no LoRA is picked)
  const singleValue = isSingle ? props.value : null
  useEffect(() => {
    if (!isSingle || singleValue == null) return
    const next = singleValue.projectId
    setPid((cur) => (cur === next ? cur : next))
  }, [isSingle, singleValue])

  // Multi mode: when the caller supplies initialPid (XY history backfill), pid follows the prop
  useEffect(() => {
    if (isSingle || multiInitialPid == null) return
    setPid((cur) => (cur === multiInitialPid ? cur : multiInitialPid))
  }, [isSingle, multiInitialPid])

  // With no anchor (single value=null / multi with no initialPid), once projects lazily load,
  // auto-anchor to the first project — preserves the existing UX of seeing the first project's
  // ckpts as soon as the picker opens, without having to pick one first. Skipped when an anchor
  // exists, since the two sync effects above already handle that case.
  const hasAnchor = isSingle ? singleValue != null : multiInitialPid != null
  useEffect(() => {
    if (hasAnchor) return
    if (projects.length === 0) return
    setPid((cur) => (cur != null ? cur : projects[0].id))
  }, [hasAnchor, projects])

  // Version dropdown: sourced from the catalog (fetched lazily by the effect above once pid settles). Empty array until it arrives.
  const versions = useMemo(
    () => (pid != null ? versionsOf(pid) ?? [] : []),
    [versionsOf, pid],
  )

  const [vid, setVid] = useState<number | null>(initialVid)
  // Same as pid: in single mode, when value is non-null, vid stays in sync with props.value.versionId
  useEffect(() => {
    if (!isSingle || singleValue == null) return
    const next = singleValue.versionId
    setVid((cur) => (cur === next ? cur : next))
  }, [isSingle, singleValue])

  // Multi mode: controlled vid (pairs with multiInitialPid)
  useEffect(() => {
    if (isSingle || multiInitialVid == null) return
    setVid((cur) => (cur === multiInitialVid ? cur : multiInitialVid))
  }, [isSingle, multiInitialVid])
  // When versions change and the current vid isn't in the new list, auto-pick the first one
  // (triggered when the user switches the project dropdown)
  useEffect(() => {
    if (versions.length === 0) {
      setVid((cur) => (cur === null ? cur : null))
    } else if (!versions.some((v) => v.id === vid)) {
      setVid(versions[0].id)
    }
  }, [versions, vid])

  // Fetch ckpts
  const [ckpts, setCkpts] = useState<LoraCkpt[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // vid invariant: either null (still resolving) or belonging to the current pid's versions
  // (the dropdown onChange clears it on project switch, and the auto-vid effect picks one from
  // versions). So we only fetch ckpts once vid is non-null — the (newPid, oldVid) combo that
  // would trigger a 404 can't happen.
  useEffect(() => {
    if (pid === null) {
      setCkpts([])  // No project selected -> clear
      return
    }
    // Transition frame while switching projects (vid is momentarily null, waiting for auto-vid
    // to settle): keep the old chips instead of clearing them, and let `settling` show the
    // loading state. Otherwise there'd be a flash of "loading/empty" before the new chips appear.
    if (vid === null) return
    let cancelled = false
    setLoading(true)
    setError(null)
    void fetchCkpts(pid, vid)
      .then((items) => {
        if (cancelled) return
        setCkpts(items)
        setLoading(false)
      })
      .catch((e) => {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
        setCkpts([])
        setLoading(false)
      })
    return () => { cancelled = true }
  }, [pid, vid, fetchCkpts])

  // Narrow window after switching projects while the new project's versions are still loading
  // over the network (vid not yet settled) -> show "loading" instead of a long blank gap. This
  // is the only case; switching versions directly (vid changes without going through null) does
  // not trigger it, so swapping checkpoints within the same project doesn't flash. A project
  // that's loaded but genuinely has no versions (versionsOf returns []) doesn't count either.
  const settling = pid !== null && vid === null && versionsOf(pid) === undefined
  // Updating: a ckpt request is in flight, or we're waiting on new versions after a project switch.
  // Old chips stay visible but clicks are disabled during this (to avoid picking the stale ckpt),
  // with a small hint that doesn't shift the layout.
  const updating = loading || settling

  // Search filter
  const [search, setSearch] = useState('')
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return ckpts
    return ckpts.filter((c) =>
      c.label.toLowerCase().includes(q) || c.path.toLowerCase().includes(q)
    )
  }, [ckpts, search])

  // Weight: controlled in single mode; internal state in multi mode
  const [internalWeight, setInternalWeight] = useState<number>(
    isSingle ? props.weight : (props.mode === 'multi' ? props.defaultWeight ?? 1.0 : 1.0)
  )
  // In single mode weight follows props; in multi mode singleWeight is always 0 and never triggers the sync
  const singleWeight = isSingle ? props.weight : 0
  useEffect(() => {
    if (isSingle) setInternalWeight(singleWeight)
  }, [isSingle, singleWeight])

  // Multi mode's current session selection (unused in single mode, which is controlled via props.value)
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const isLive = !isSingle && (props as MultiModeProps).live === true
  // Controlled selection (live mode only): when the caller supplies selectedPaths, the picker treats it as the single source of truth
  const multiSelectedPaths = !isSingle ? (props as MultiModeProps).selectedPaths : undefined
  const isControlled = !isSingle && multiSelectedPaths !== undefined
  // Distinguish "pid/vid changed because a prop synced" from "the user clicked the dropdown" — the former must not clear the axis
  const lastPropPidVid = useRef({ pid: multiInitialPid, vid: multiInitialVid })
  useEffect(() => {
    lastPropPidVid.current = { pid: multiInitialPid, vid: multiInitialVid }
  }, [multiInitialPid, multiInitialVid])
  useEffect(() => {
    if (isSingle) return
    // Controlled: a pid/vid change caused by a prop sync doesn't clear picked (let the selectedPaths sync handle it)
    if (isControlled && pid === lastPropPidVid.current.pid && vid === lastPropPidVid.current.vid) return
    setPicked(new Set())  // Clear the UI selection when pid/vid switches
    // Live mode: clear the axis too when pid/vid switches (the old path is meaningless under the
    // new version); this also runs once on initial mount, but draft.raw is usually already empty
    // by then so committing an empty set is a no-op.
    if (isLive) {
      (props as MultiModeProps).onPick([], internalWeight)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid, vid, isSingle, isControlled])

  // Controlled sync: selectedPaths + ckpts determine picked.
  //   - A selectedPaths element that's a full path hits directly.
  //   - A basename (the form used by history backfill, to avoid leaking absolute paths into
  //     snapshots) is matched against ckpts by basename; picked stores the full path, and we
  //     immediately call onPick to write the full path back into raw — otherwise the daemon
  //     receives a basename and reports "LoRA path not found".
  useEffect(() => {
    if (!isControlled || !multiSelectedPaths || loading) return
    if (pid === null || vid === null) return
    const ckptByPath = new Map(ckpts.map((c) => [c.path, c]))
    const ckptByBasename = new Map<string, LoraCkpt>()
    for (const c of ckpts) ckptByBasename.set(basenameOf(c.path), c)
    const resolvedPaths: string[] = []
    let needUpgrade = false
    for (const v of multiSelectedPaths) {
      if (!v) continue
      if (ckptByPath.has(v)) {
        resolvedPaths.push(v)
        continue
      }
      const ck = ckptByBasename.get(basenameOf(v))
      if (ck) {
        resolvedPaths.push(ck.path)
        needUpgrade = true  // raw holds a basename / stale path, needs upgrading to the full path on this machine
      }
      // Not found -> not among the current ckpts, skip it (user may have switched projects or the file was deleted)
    }
    const resolvedSet = new Set(resolvedPaths)
    // Avoid an infinite loop: skip setState when picked hasn't actually changed
    setPicked((prev) => {
      if (prev.size === resolvedSet.size && [...prev].every((p) => resolvedSet.has(p))) return prev
      return resolvedSet
    })
    if (needUpgrade && isLive) {
      const picks = ckpts
        .filter((c) => resolvedSet.has(c.path))
        .map((c) => ({ path: c.path, projectId: pid, versionId: vid }))
      ;(props as MultiModeProps).onPick(picks, internalWeight)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isControlled, multiSelectedPaths, ckpts, loading, pid, vid, isLive])

  const currentVersion = versions.find((v) => v.id === vid)

  // Selection set -> picks: ordered by how ckpts are displayed (list_lora_ckpts' canonical sort:
  // final -> step desc -> epoch desc), not click order. The XY ckpt axis must be monotonic,
  // otherwise grid columns/rows jump around based on click order and the overfitting inflection
  // point becomes unreadable (ep60 should always sit between ep80 and ep40).
  const orderedPicks = (sel: Set<string>): PickedLora[] =>
    ckpts
      .filter((c) => sel.has(c.path))
      .map((c) => ({ path: c.path, projectId: pid, versionId: vid }))

  // Chip click
  const onChipClick = (c: LoraCkpt) => {
    if (existingPaths.has(c.path)) return
    if (isSingle) {
      const { value } = props
      const isCurrent = value && value.path === c.path
      if (isCurrent) {
        // Deselect: clear the slot's ckpt (visually identical to a freshly opened empty slot);
        // SidebarLoras receiving null does not delete the whole slot, only clears the path (only × deletes the slot).
        props.onChange(null, internalWeight)
        return
      }
      props.onChange(
        { path: c.path, projectId: pid, versionId: vid },
        internalWeight,
      )
      return
    }
    setPicked((s) => {
      const next = new Set(s)
      if (next.has(c.path)) next.delete(c.path); else next.add(c.path)
      // Live mode: commit immediately on every chip toggle, without waiting for "Add N"
      if (isLive && pid !== null && vid !== null) {
        ;(props as MultiModeProps).onPick(orderedPicks(next), internalWeight)
      }
      return next
    })
  }

  const onWeightChange = (w: number) => {
    if (isSingle) {
      props.onChange(props.value, w)
    } else {
      setInternalWeight(w)
      if (isLive && pid !== null && vid !== null && picked.size > 0) {
        ;(props as MultiModeProps).onPick(orderedPicks(picked), w)
      }
    }
  }

  const commitMulti = () => {
    if (isSingle) return
    if (picked.size === 0) return
    props.onPick(orderedPicks(picked), internalWeight)
    onClose()
  }

  // Single mode: the selected ckpt path (used for chip highlighting)
  const selectedPath = isSingle ? props.value?.path ?? null : null

  return (
    <div
      className={embedded ? 'flex flex-col gap-2' : 'rounded-md border border-subtle bg-overlay p-2.5 flex flex-col gap-2'}
      data-testid="inline-lora-picker"
    >
      {/* header */}
      <div className="flex items-center gap-2">
        {!embedded && <span className="text-xs font-semibold text-fg-secondary shrink-0">{t('generate.pickLora')}</span>}
        {/* While updating (old chips stay in place), show a small hint that doesn't shift the layout or replace the grid */}
        {updating && ckpts.length > 0 && (
          <span className="text-2xs text-fg-tertiary shrink-0">{t('common.loading')}</span>
        )}
        <span className="flex-1" />
        {onPickExternal && (
          <button
            onClick={onPickExternal}
            className={embedded ? 'ds-ctl-note' : 'btn btn-ghost btn-sm text-2xs text-fg-tertiary'}
            style={embedded ? { color: 'var(--green-text)' } : undefined}
            title={t('generate.externalFileHint')}
          >
            {t('generate.externalFile')}
          </button>
        )}
        {!embedded && <button
          onClick={onClose}
          className="btn btn-ghost btn-sm text-fg-tertiary px-1.5"
          title={isSingle ? t('generate.removeSlot') : t('generate.closePanel')}
          aria-label={isSingle ? t('generate.removeLora') : t('generate.closePicker')}
        >
          ×
        </button>}
      </div>

      {/* project / version dropdowns */}
      <div className="flex gap-2">
        <select
          className="ds-inp"
          value={pid ?? ''}
          onChange={(e) => {
            // Switching projects: update pid + vid in the same batch, clearing vid to null
            // right away (committed together with setPid) to avoid a (newPid, oldVid) frame
            // that would make the ckpt effect fetch the wrong thing and hit a 404. Once the
            // new versions lazily load, the auto-vid effect picks that project's version.
            setPid(e.target.value ? Number(e.target.value) : null)
            setVid(null)
          }}
          aria-label={t('generate.pickProject')}
        >
          <option value="">{t('generate.pickProjectPlaceholder')}</option>
          {projects.map((p) => (
            <option key={p.id} value={p.id}>{p.title}</option>
          ))}
        </select>
        <select
          className="ds-inp"
          value={vid ?? ''}
          onChange={(e) => setVid(e.target.value ? Number(e.target.value) : null)}
          disabled={versions.length === 0}
          aria-label={t('generate.pickVersion')}
        >
          <option value="">{t('generate.pickVersionPlaceholder')}</option>
          {versions.map((v) => (
            <option key={v.id} value={v.id}>
              {v.status === 'training' ? t('generate.versionTraining', { label: v.label }) : v.label}
            </option>
          ))}
        </select>
      </div>

      {/* search */}
      <input
        type="text"
        className="ds-inp"
        placeholder={t('generate.searchCkpt')}
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        disabled={!pid || !vid || ckpts.length === 0}
      />

      {error && <div className="text-2xs text-err">{error}</div>}
      {currentVersion?.status === 'training' && (
        <div className="text-2xs text-fg-tertiary">
          <span className="badge badge-info" style={{ fontSize: 10, marginRight: 4 }}>{t('generate.trainingNow')}</span>
          {t('generate.ckptsRefresh')}
        </div>
      )}

      {/* ckpt chip list -- an equal-width grid (auto-fill) keeps names of varying length
          aligned into tidy columns; long names truncate in-cell with a title for the full text,
          avoiding a ragged flowing layout. */}
      <div
        className="grid gap-1.5 overflow-y-auto"
        style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', maxHeight: 280, padding: 2 }}
      >
        {/* "Loading" only appears when there's really nothing to show (first fetch). When
            switching version/project the old chips still render (filtered.map below is no
            longer gated by loading), and get replaced in place once new results arrive —
            no flash of an empty list. */}
        {updating && ckpts.length === 0 && <div className="text-2xs text-fg-tertiary px-1 py-2" style={{ gridColumn: '1 / -1' }}>{t('common.loading')}</div>}
        {!loading && projects.length === 0 && (
          <div className="text-fg-tertiary text-xs px-1 py-4 text-center" style={{ gridColumn: '1 / -1' }}>
            {onPickExternal ? t('generate.noLorasYetExternal') : t('generate.noLorasYet')}
          </div>
        )}
        {!loading && projects.length > 0 && pid !== null && vid !== null && ckpts.length === 0 && !error && (
          <div className="text-2xs text-fg-tertiary px-1 py-4 text-center" style={{ gridColumn: '1 / -1' }}>
            {t('generate.noCkptsInVersion')}
          </div>
        )}
        {filtered.map((c) => {
          const isExisting = existingPaths.has(c.path)
          const isPicked = isSingle ? c.path === selectedPath : picked.has(c.path)
          const marker = isExisting ? '✓' : (isPicked ? '✓' : '+')
          return (
            <button
              key={c.path}
              type="button"
              onClick={() => onChipClick(c)}
              disabled={isExisting || updating}
              className="font-mono flex items-center gap-1 min-w-0"
              style={{
                fontSize: 11,
                padding: '4px 8px',
                borderRadius: 'var(--r-md)',
                border: isPicked
                  ? '1px solid transparent'
                  : (isExisting ? '1px dashed var(--border-default)' : '1px solid var(--border-subtle)'),
                background: isExisting
                  ? 'var(--bg-sunken)'
                  : (isPicked ? 'var(--accent-soft)' : 'var(--bg-sunken)'),
                color: isExisting
                  ? 'var(--fg-tertiary)'
                  : (isPicked ? 'var(--accent)' : 'var(--fg-secondary)'),
                cursor: isExisting ? 'not-allowed' : 'pointer',
              }}
              title={c.path}
            >
              <span className="shrink-0">{marker}</span>
              <span className="truncate flex-1 text-left">{c.label}</span>
            </button>
          )
        })}
        {!loading && ckpts.length > 0 && filtered.length === 0 && (
          <div className="text-fg-tertiary text-xs px-1 py-4 text-center" style={{ gridColumn: '1 / -1' }}>{t('generate.noMatchingCkpt')}</div>
        )}
      </div>

      {/* Weight slider -- always shown in single mode; in multi mode shown when showWeight and something is picked */}
      {showWeight && !embedded && (isSingle || picked.size > 0) && (
        <div
          className="flex items-center gap-2 pt-1"
          style={{ borderTop: '1px solid var(--border-subtle)' }}
        >
          <span
            className="font-mono text-fg-tertiary shrink-0"
            style={{ fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase' }}
          >
            {t('generate.weight')}
          </span>
          <input
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={internalWeight}
            onChange={(e) => onWeightChange(Number(e.target.value))}
            className="flex-1"
            aria-label={t('generate.loraWeight')}
            style={{ accentColor: 'var(--accent)' }}
          />
          <input
            type="number"
            min={0}
            max={1.5}
            step={0.05}
            value={internalWeight}
            onChange={(e) => onWeightChange(Number(e.target.value))}
            className="input font-mono text-center"
            style={{ width: 54, padding: '3px 6px', fontSize: 12 }}
            aria-label={t('generate.loraWeightValue')}
          />
        </div>
      )}

      {/* Multi mode: commit footer (not rendered in live mode, where chips are already what-you-see-is-what-you-get) */}
      {!isSingle && !isLive && picked.size > 0 && (
        <div className="flex items-center gap-2 justify-end">
          <span className="text-2xs text-fg-tertiary mr-auto">{t('generate.pickedCount', { count: picked.size })}</span>
          <button
            onClick={() => setPicked(new Set())}
            className="btn btn-ghost btn-sm text-xs"
          >
            {t('common.cancel')}
          </button>
          <button
            onClick={commitMulti}
            className="btn btn-primary btn-sm text-xs"
          >
            {t('generate.addN', { count: picked.size })}
          </button>
        </div>
      )}
    </div>
  )
}
