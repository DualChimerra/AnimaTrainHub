import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type DuplicateGroup,
  type DuplicateItem,
  type DuplicateScanOptions,
  type DuplicateScanResult,
} from '../api/client'

export const DEFAULT_DUPLICATE_OPTIONS: DuplicateScanOptions = {
  match_scope: 'both',
  sensitivity: 'standard',
}

interface Props {
  projectId: number
  versionId: number
  result: DuplicateScanResult | null
  selected: Set<string>
  busy: boolean
  onSelect: (next: Set<string>) => void
  onPreview: (name: string) => void
}

type GroupFilter = 'all' | 'exact' | 'similar'

/** Train rel path "1_data/X.png" → folder + file name for the thumb URL. */
function splitRel(rel: string): { folder: string; filename: string } {
  const i = rel.lastIndexOf('/')
  return i >= 0
    ? { folder: rel.slice(0, i), filename: rel.slice(i + 1) }
    : { folder: '', filename: rel }
}

const isExact = (g: DuplicateGroup) => g.best?.match_type === 'strict-duplicate'

/** Duplicate review column (mockup PreprocessDupes): group filter chips and
 *  the scan summary, then one card per group. Selected images are the ones
 *  to remove; each image toggles on its own, "Keep best" re-applies the
 *  scanner's suggestion for that group. */
export default function DuplicateReviewPanel({
  projectId,
  versionId,
  result,
  selected,
  busy,
  onSelect,
  onPreview,
}: Props) {
  const { t } = useTranslation()
  const [filter, setFilter] = useState<GroupFilter>('all')
  const suggested = useMemo(
    () => (result ? result.groups.flatMap((g) => g.items.filter((it) => !it.keep).map((it) => it.name)) : []),
    [result],
  )
  const groups = useMemo(() => {
    const all = result?.groups ?? []
    if (filter === 'exact') return all.filter(isExact)
    if (filter === 'similar') return all.filter((g) => !isExact(g))
    return all
  }, [result, filter])
  const exactCount = (result?.groups ?? []).filter(isExact).length

  const toggleName = (name: string) => {
    const next = new Set(selected)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    onSelect(next)
  }
  const keepBest = (g: DuplicateGroup) => {
    const next = new Set(selected)
    for (const it of g.items) {
      if (it.keep) next.delete(it.name)
      else next.add(it.name)
    }
    onSelect(next)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, minWidth: 0 }}>
      <div style={{ padding: '12px 17px', borderBottom: '1px solid var(--line)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        {([
          ['all', t('ppFrame.dupAllGroups'), result?.group_count ?? 0],
          ['exact', t('ppFrame.dupExact'), exactCount],
          ['similar', t('ppFrame.dupSimilar'), (result?.group_count ?? 0) - exactCount],
        ] as Array<[GroupFilter, string, number]>).map(([id, label, n]) => (
          <button key={id} type="button" className={`ds-chip${filter === id ? ' ds-is-active' : ''}`} style={{ paddingRight: 6 }} aria-pressed={filter === id} onClick={() => setFilter(id)} disabled={!result}>
            {label} <b>{n}</b>
          </button>
        ))}
        <span style={{ flex: 1 }} />
        {result && (
          <span className="ds-kpi-meta">
            {t('ppFrame.dupChecked', { n: result.readable_images, total: result.total_images, sel: selected.size })}
          </span>
        )}
        {selected.size > 0 ? (
          <button type="button" className="ds-ctl ds-ghost" style={{ height: 28 }} onClick={() => onSelect(new Set())} disabled={busy}>{t('common.deselect')}</button>
        ) : (
          <button type="button" className="ds-ctl ds-ghost" style={{ height: 28 }} onClick={() => onSelect(new Set(suggested))} disabled={busy || suggested.length === 0}>{t('duplicates.selectSuggested')}</button>
        )}
      </div>

      <div style={{ padding: '14px 17px', flex: 1, minHeight: 0, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 11 }}>
        {!result ? (
          <div className="ds-empty" style={{ margin: 'auto 0' }}>
            <span style={{ fontWeight: 500, color: 'var(--ink-2)' }}>{t('duplicates.emptyTitle')}</span>
            <span style={{ fontSize: 11.5 }}>{t('ppFrame.dupEmpty')}</span>
          </div>
        ) : groups.length === 0 ? (
          <div className="ds-empty" style={{ margin: 'auto 0' }}>{t('duplicates.noGroups', { total: result.total_images })}</div>
        ) : (
          groups.map((group, i) => (
            <DuplicateGroupCard
              key={group.group_id}
              index={i + 1}
              projectId={projectId}
              versionId={versionId}
              group={group}
              selected={selected}
              busy={busy}
              onToggle={toggleName}
              onKeepBest={() => keepBest(group)}
              onPreview={onPreview}
            />
          ))
        )}
      </div>
    </div>
  )
}

function DuplicateGroupCard({ index, projectId, versionId, group, selected, busy, onToggle, onKeepBest, onPreview }: {
  index: number
  projectId: number
  versionId: number
  group: DuplicateGroup
  selected: Set<string>
  busy: boolean
  onToggle: (name: string) => void
  onKeepBest: () => void
  onPreview: (name: string) => void
}) {
  const { t } = useTranslation()
  const score = group.best ? Math.round(group.best.score) : null
  return (
    <div className="ds-card" style={{ border: '1px solid var(--line-2)', boxShadow: 'none', padding: '12px 14px', flex: 'none' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
        <span className="ds-mono" style={{ fontSize: 12, fontWeight: 600 }}>{t('ppFrame.dupGroup', { n: index })}</span>
        {score != null && (
          <span className={`ds-badge ${isExact(group) ? 'ds-err' : 'ds-warn'}`} title={group.best?.match_type}>
            {t('ppFrame.dupSimilarity', { n: score })}
          </span>
        )}
        <span className="ds-kpi-meta">{t('ppFrame.dupFiles', { count: group.items.length })}</span>
        <span style={{ flex: 1 }} />
        <button type="button" className="ds-ctl ds-ghost" style={{ height: 26 }} onClick={onKeepBest} disabled={busy}>{t('ppFrame.dupKeepBest')}</button>
      </div>
      <div className="ds-grid-thumbs" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(96px, 104px))' }}>
        {group.items.map((item) => (
          <DuplicateItemCell
            key={item.name}
            projectId={projectId}
            versionId={versionId}
            item={item}
            remove={selected.has(item.name)}
            busy={busy}
            onToggle={() => onToggle(item.name)}
            onPreview={() => onPreview(item.name)}
          />
        ))}
      </div>
    </div>
  )
}

function DuplicateItemCell({ projectId, versionId, item, remove, busy, onToggle, onPreview }: {
  projectId: number
  versionId: number
  item: DuplicateItem
  remove: boolean
  busy: boolean
  onToggle: () => void
  onPreview: () => void
}) {
  const { t } = useTranslation()
  const { folder, filename } = splitRel(item.name)
  const title = `${item.name}\n${item.width}×${item.height} · ${item.filesize_kb} KB${item.metrics ? `\n${item.metrics.match_type} · ${item.metrics.score}` : ''}`
  return (
    <div className={`ds-thumb group${remove ? '' : ' ds-sel'}`} style={{ aspectRatio: '1', opacity: remove ? 0.55 : 1 }} title={title}>
      <button type="button" className="absolute inset-0" onClick={onToggle} disabled={busy}
        aria-label={`${remove ? t('duplicates.restoreCandidate') : t('duplicates.removeCandidate')} ${item.name}`}>
        <img
          src={api.versionThumbUrl(projectId, versionId, 'train', filename, folder, 256)}
          alt={item.name}
          loading="lazy"
          decoding="async"
          className="w-full h-full object-cover"
        />
      </button>
      <span className="ds-pin pointer-events-none" style={remove ? { background: 'var(--err)' } : undefined}>
        {remove ? t('duplicates.selectedRemove') : item.keep ? t('ppFrame.dupKeep') : t('duplicates.keep')}
      </span>
      <button
        type="button"
        onClick={onPreview}
        aria-label={`${t('common.preview')} ${item.name}`}
        className="absolute bottom-6 right-1.5 w-5 h-5 rounded-[5px] bg-black/60 text-white text-[11px] opacity-0 group-hover:opacity-100 hover:bg-black/80"
      >⤢</button>
      <span className="ds-cap pointer-events-none">{filename}</span>
    </div>
  )
}
