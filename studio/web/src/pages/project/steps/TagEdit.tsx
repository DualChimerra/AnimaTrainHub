import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useBlocker, useOutletContext } from 'react-router-dom'
import {
  api,
  type CommitItem,
  type ProjectDetail,
  type Version,
} from '../../../api/client'
import BulkActionBar from '../../../components/BulkActionBar'
import { useDialog } from '../../../components/Dialog'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import SaveBar from '../../../components/SaveBar'
import StepShell from '../../../components/StepShell'
import TagEditor from '../../../components/TagEditor'
import TagStatsPanel from '../../../components/TagStatsPanel'
import { useToast } from '../../../components/Toast'
import ZoomableImage from '../../../components/ZoomableImage'
import { useEventStream } from '../../../lib/useEventStream'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
  setVersionSwitchGuard: (g: (() => Promise<boolean>) | null) => void
}

const keyOf = (folder: string, name: string) => `${folder}/${name}`

interface CaptionMeta {
  folder: string
  name: string
  format: 'txt' | 'json' | 'none'
}

function arraysEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

export default function TagEditPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload, setVersionSwitchGuard } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const versionId = activeVersion?.id ?? null

  const [cache, setCache] = useState<Map<string, string[]>>(new Map())
  const [initial, setInitial] = useState<Map<string, string[]>>(new Map())
  const [meta, setMeta] = useState<Map<string, CaptionMeta>>(new Map())
  const [keys, setKeys] = useState<string[]>([])

  const [activeKey, setActiveKey] = useState<string>('')
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [anchor, setAnchor] = useState<string | null>(null)
  // '' = all; otherwise scoped to that folder (1_data / 2_data ...). Deliberately named
  // differently from `activeFolder` used below during editing (that one is the folder of the currently-edited image, purely for display).
  const [folderFilter, setFolderFilter] = useState<string>('')
  // tag picked in the statistics card; its remove / replace work on it
  const [pickedTag, setPickedTag] = useState<string | null>(null)
  // natural size of the active image, read from the loaded original
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null)

  const reloadCache = useCallback(async () => {
    if (versionId == null) return
    try {
      const r = await api.listCaptionsFull(project.id, versionId)
      const c = new Map<string, string[]>()
      const m = new Map<string, CaptionMeta>()
      const ks: string[] = []
      for (const it of r.items) {
        const k = keyOf(it.folder, it.name)
        c.set(k, it.tags)
        m.set(k, { folder: it.folder, name: it.name, format: it.format })
        ks.push(k)
      }
      setCache(c); setInitial(new Map(c)); setMeta(m); setKeys(ks)
    } catch (e) { toast(String(e), 'error') }
  }, [project.id, versionId, toast])

  useEffect(() => { void reloadCache() }, [reloadCache])

  useEventStream((evt) => {
    if (
      evt.type === 'version_state_changed' &&
      versionId != null &&
      evt.version_id === versionId
    ) {
      void reloadCache(); void reload()
    } else if (
      evt.type === 'job_state_changed' &&
      evt.project_id === project.id &&
      (evt.status === 'done' || evt.status === 'failed')
    ) {
      void reloadCache(); void reload()
    }
  })

  const dirtyKeys = useMemo(() => {
    const out: string[] = []
    for (const k of keys) {
      const cur = cache.get(k) ?? []
      const ini = initial.get(k) ?? []
      if (!arraysEqual(cur, ini)) out.push(k)
    }
    return out
  }, [cache, initial, keys])
  const dirty = dirtyKeys.length > 0

  useEffect(() => {
    if (!dirty) return
    const handler = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = '' }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [dirty])

  // In-app React-Router navigation doesn't fire beforeunload, so we need useBlocker (v6.4+).
  // When dirty=false the blocker auto-allows navigation; when dirty=true it intercepts navigation
  // → confirm popup → the user choosing "discard" calls proceed(), "stay" calls reset().
  const blocker = useBlocker(dirty)
  useEffect(() => {
    if (blocker.state !== 'blocked') return
    let cancelled = false
    void confirm(
      t('tagEdit.unsavedConfirmMessage', { n: dirtyKeys.length }),
      {
        tone: 'danger',
        title: t('tagEdit.unsavedConfirmTitle'),
        okText: t('tagEdit.unsavedConfirmDiscard'),
        cancelText: t('tagEdit.unsavedConfirmStay'),
      },
    ).then((ok) => {
      if (cancelled) return
      if (ok) blocker.proceed?.()
      else blocker.reset?.()
    })
    return () => { cancelled = true }
  }, [blocker, confirm, t, dirtyKeys.length])

  // Switching versions remounts this page (Layout's Outlet key), which doesn't go through route
  // navigation, so useBlocker can't catch it; register a switch guard while dirty, reusing the same confirm copy.
  useEffect(() => {
    if (!dirty) return
    setVersionSwitchGuard(() =>
      confirm(
        t('tagEdit.unsavedConfirmMessage', { n: dirtyKeys.length }),
        {
          tone: 'danger',
          title: t('tagEdit.unsavedConfirmTitle'),
          okText: t('tagEdit.unsavedConfirmDiscard'),
          cancelText: t('tagEdit.unsavedConfirmStay'),
        },
      )
    )
    return () => setVersionSwitchGuard(null)
  }, [dirty, dirtyKeys.length, setVersionSwitchGuard, confirm, t])

  // folder list + each folder's raw image count (unaffected by filterTag, so the tab numbers stay
  // stable and don't jitter -- same style as the Preprocess chips). UI hides the tabs for single-folder projects.
  const folderNames = useMemo(() => {
    const set = new Set<string>()
    for (const m of meta.values()) set.add(m.folder)
    return Array.from(set).sort()
  }, [meta])
  const folderCounts = useMemo(() => {
    const c = new Map<string, number>()
    for (const m of meta.values()) c.set(m.folder, (c.get(m.folder) ?? 0) + 1)
    return c
  }, [meta])

  const filteredKeys = useMemo(() => {
    if (!folderFilter) return keys
    return keys.filter((k) => meta.get(k)?.folder === folderFilter)
  }, [keys, meta, folderFilter])

  const captionItems = useMemo(
    () =>
      filteredKeys.map((k) => {
        const m = meta.get(k)!
        const tags = cache.get(k) ?? []
        return {
          name: k,
          thumbUrl:
            activeVersion != null
              ? api.versionThumbUrl(project.id, activeVersion.id, 'train', m.name, m.folder)
              : '',
          meta: tags.slice(0, 5).join(', '),
        }
      }),
    [filteredKeys, meta, cache, project.id, activeVersion]
  )

  const selectedKeys = useMemo(
    () => filteredKeys.filter((k) => sel.has(k)),
    [filteredKeys, sel]
  )
  const navKeys = selectedKeys.length > 0 ? selectedKeys : filteredKeys
  const activeIndex = activeKey ? navKeys.indexOf(activeKey) : -1

  // the active-image card always shows one image (mockup): fall back to the
  // first visible one when nothing is active or the active one went away
  useEffect(() => {
    if ((!activeKey || !meta.has(activeKey)) && filteredKeys.length > 0) setActiveKey(filteredKeys[0])
  }, [activeKey, meta, filteredKeys])
  useEffect(() => { setDims(null) }, [activeKey])

  const tagSuggestions = useMemo(() => {
    const set = new Set<string>()
    for (const tags of cache.values()) for (const tag of tags) set.add(tag)
    return Array.from(set).sort((a, b) => a.localeCompare(b))
  }, [cache])

  const handlePickTag = useCallback(
    (tag: string) => {
      const matched = new Set<string>()
      for (const k of keys) {
        if ((cache.get(k) ?? []).includes(tag)) matched.add(k)
      }
      setSel(matched); setAnchor(null); setPickedTag(tag)
      toast(t('tagEdit.selectedContaining', { tag, n: matched.size }), 'success')
    },
    [keys, cache, toast, t]
  )

  if (!activeVersion) {
    return <p className="text-fg-tertiary p-6">{t('tagEdit.noVersion')}</p>
  }

  const handleClick = (key: string, e: React.MouseEvent) => {
    const r = applySelection(sel, key, e, filteredKeys, anchor)
    setSel(r.next); setAnchor(r.anchor)
  }

  const navActive = (delta: number) => {
    if (navKeys.length === 0) return
    const i = activeKey ? navKeys.indexOf(activeKey) : -1
    const next = i < 0 ? 0 : (i + delta + navKeys.length) % navKeys.length
    setActiveKey(navKeys[next])
  }

  const updateActiveTags = (tags: string[]) => {
    if (!activeKey) return
    setCache((prev) => {
      const next = new Map(prev); next.set(activeKey, [...tags]); return next
    })
  }

  const applyBulkUpdates = (updates: Map<string, string[]>) => {
    setCache((prev) => {
      const next = new Map(prev)
      for (const [k, v] of updates) next.set(k, v)
      return next
    })
  }

  // Triggered by the × inline in the tag distribution row: removes that tag from the currently
  // selected images. pre-compute updates gets the real affected count → the confirm modal shows
  // the exact number of images → apply only happens after the user confirms.
  const removeTagFromSelected = async (tag: string) => {
    if (selectedKeys.length === 0) return
    const updates = new Map<string, string[]>()
    for (const k of selectedKeys) {
      const cur = cache.get(k) ?? []
      if (!cur.includes(tag)) continue
      updates.set(k, cur.filter((tt) => tt !== tag))
    }
    if (updates.size === 0) return
    const ok = await confirm(
      t('bulkAction.confirmMessage', {
        op: t('bulkAction.opLabelRemove', { tags: tag }),
        n: updates.size,
      }),
      { tone: 'danger', title: t('bulkAction.confirmTitle') },
    )
    if (!ok) return
    applyBulkUpdates(updates)
    if (tag === pickedTag) setPickedTag(null)
    toast(t('tagEdit.removedFromN', { tag, n: updates.size }), 'success')
  }

  // Submit from the ✎ inline edit in the tag distribution row: replaces oldTag with newTag across the selected images, deduplicating.
  const replaceTagInSelected = async (oldTag: string, newTag: string) => {
    if (selectedKeys.length === 0 || !newTag || newTag === oldTag) return
    const updates = new Map<string, string[]>()
    for (const k of selectedKeys) {
      const cur = cache.get(k) ?? []
      if (!cur.includes(oldTag)) continue
      const next: string[] = []
      const seen = new Set<string>()
      for (const tt of cur) {
        const out = tt === oldTag ? newTag : tt
        if (seen.has(out)) continue
        seen.add(out); next.push(out)
      }
      updates.set(k, next)
    }
    if (updates.size === 0) return
    const ok = await confirm(
      t('bulkAction.confirmMessage', {
        op: t('bulkAction.opLabelReplace', { from: oldTag, to: newTag }),
        n: updates.size,
      }),
      { tone: 'danger', title: t('bulkAction.confirmTitle') },
    )
    if (!ok) return
    applyBulkUpdates(updates)
    if (oldTag === pickedTag) setPickedTag(newTag)
    toast(t('tagEdit.replacedInN', { from: oldTag, to: newTag, n: updates.size }), 'success')
  }

  const onSave = async () => {
    if (!dirty || versionId == null) return
    const items: CommitItem[] = dirtyKeys.map((k) => {
      const m = meta.get(k)!
      return { folder: m.folder, name: m.name, tags: cache.get(k) ?? [] }
    })
    try {
      const r = await api.commitCaptions(project.id, versionId, items)
      setInitial(new Map(cache))
      toast(t('tagEdit.savedToast', { written: r.written, id: r.snapshot.id }), 'success')
      void reload()
    } catch (e) { toast(String(e), 'error') }
  }

  const onAfterRestore = async () => {
    await reloadCache()
    setActiveKey('')
    setSel(new Set())
    setAnchor(null)
    setFolderFilter('')
    setPickedTag(null)
    await reload()
  }

  const stats = activeVersion.stats
  const trainTotal = stats?.train_image_count ?? 0
  const taggedTotal = stats?.tagged_image_count ?? 0
  const allTagged = trainTotal > 0 && taggedTotal >= trainTotal

  const activeMeta = activeKey ? meta.get(activeKey) : undefined
  const activeFolder = activeMeta?.folder ?? ''
  const activeName = activeMeta?.name ?? ''
  const activeTags = activeKey ? cache.get(activeKey) ?? [] : []
  const pickedCount = pickedTag
    ? filteredKeys.reduce((n, k) => n + ((cache.get(k) ?? []).includes(pickedTag) ? 1 : 0), 0)
    : 0

  return (
    <StepShell
      idx={4}
      mobilePageScroll
      eyebrow={t('steps.eyebrowStep', { n: 3, label: activeVersion.label })}
      title={t('tagEdit.title')}
      subtitle={t('tagEdit.subtitle')}
      actions={
        <>
          {stats && (
            <span className={`ds-badge ${allTagged ? 'ds-ok' : 'ds-mute'}`}>
              {t('tagEdit.taggedBadge', { tagged: taggedTotal, total: trainTotal })}
            </span>
          )}
          <a
            className="ds-ctl"
            href={dirty ? undefined : api.versionTrainZipUrl(project.id, activeVersion.id)}
            download={dirty ? undefined : true}
            aria-disabled={dirty}
            title={dirty ? t('tagEdit.saveThenDownload') : t('tagEdit.downloadTitle')}
            onClick={(e) => { if (dirty) { e.preventDefault(); toast(t('tagEdit.saveThenDownloadToast'), 'error') } }}
          >{t('tagEdit.downloadZip')}</a>
          <Link className="ds-btn-primary" to={`/projects/${project.id}/v/${activeVersion.id}/reg`}>
            {t('tagEdit.next')}
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h13m-5-6 6 6-6 6" /></svg>
          </Link>
        </>
      }
      footer={
        <SaveBar
          pid={project.id}
          vid={activeVersion.id}
          dirtyCount={dirtyKeys.length}
          onSave={onSave}
          onDiscard={() => setCache(new Map(initial))}
          onAfterRestore={onAfterRestore}
        />
      }
    >
      <div className="ds-te-grid">
        <TagStatsPanel
          cache={cache}
          allKeys={filteredKeys}
          selectedKeys={selectedKeys}
          triggerWord={activeVersion.trigger_word || undefined}
          pickedTag={pickedTag}
          onPickTag={handlePickTag}
          onRemoveTag={removeTagFromSelected}
          onReplaceTag={replaceTagInSelected}
        />

        <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div className="ds-card-head ds-pad">
            <div>
              <div className="ds-card-title">{t('tagEdit.imagesTitle')}</div>
              <div className="ds-card-sub">
                {t('tagEdit.imagesSub', { n: filteredKeys.length })}
                {pickedTag && ` · ${t('tagEdit.withTag', { n: pickedCount, tag: pickedTag })}`}
              </div>
            </div>
            <div className="ds-card-tools">
              <select
                className="ds-inp"
                style={{ width: 170 }}
                value={folderFilter}
                onChange={(e) => setFolderFilter(e.target.value)}
                aria-label={t('tagEdit.folderLabel')}
                disabled={folderNames.length < 2}
              >
                <option value="">{t('tagEdit.allFolders', { n: keys.length })}</option>
                {folderNames.map((f) => (
                  <option key={f} value={f}>{f} · {folderCounts.get(f) ?? 0}</option>
                ))}
              </select>
            </div>
          </div>
          <div style={{ flex: 1, minHeight: 0, padding: '2px 17px 14px' }}>
            <ImageGrid
              items={captionItems}
              selected={sel}
              activeName={activeKey || undefined}
              onSelect={handleClick}
              onActivate={setActiveKey}
              clickMode="activate"
              columnsClass="grid-cols-[repeat(auto-fill,minmax(76px,1fr))]"
              ariaLabel="tag-edit-grid"
              emptyHint={
                folderFilter
                  ? t('tagEdit.noImagesInFolder', { folder: folderFilter })
                  : t('tagEdit.noImagesHint')
              }
            />
          </div>
          {selectedKeys.length > 0 && (
            <BulkActionBar
              variant="bar"
              cache={cache}
              selectedKeys={selectedKeys}
              onApply={applyBulkUpdates}
              tagSuggestions={tagSuggestions}
              onClearSelection={() => { setSel(new Set()); setAnchor(null); setPickedTag(null) }}
              onSelectAll={() => setSel(new Set(filteredKeys))}
              totalCount={filteredKeys.length}
            />
          )}
        </div>

        <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div className="ds-card-head ds-pad">
            <div style={{ minWidth: 0 }}>
              <div className="ds-card-title" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={activeName}>
                {activeName || t('tagEdit.noActive')}
              </div>
              <div className="ds-card-sub">
                {activeKey
                  ? [dims && `${dims.w} × ${dims.h}`, activeFolder, activeIndex >= 0 && `${activeIndex + 1} / ${navKeys.length}`].filter(Boolean).join(' · ')
                  : '—'}
              </div>
            </div>
            <div className="ds-card-tools" style={{ flexWrap: 'nowrap' }}>
              <button type="button" className="ds-kebab" onClick={() => navActive(-1)} disabled={navKeys.length === 0} aria-label={t('tagEdit.prevImage')}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M15 6l-6 6 6 6" /></svg>
              </button>
              <button type="button" className="ds-kebab" onClick={() => navActive(1)} disabled={navKeys.length === 0} aria-label={t('tagEdit.nextImage')}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M9 6l6 6-6 6" /></svg>
              </button>
            </div>
          </div>
          {activeKey ? (
            <>
              <div style={{ padding: '0 17px 12px', flex: 'none' }}>
                {/* Full-resolution image straight out + wheel zoom / drag pan / double-click fit↔100% (for checking fine-detail tags) */}
                <div style={{ height: 186 }}>
                  <ZoomableImage
                    key={activeKey}
                    src={api.versionThumbUrl(project.id, activeVersion.id, 'train', activeName, activeFolder, 0)}
                    alt={activeName}
                    readout={false}
                    onNaturalSize={(w, h) => setDims({ w, h })}
                  />
                </div>
              </div>
              <div style={{ padding: '0 17px 14px', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                <TagEditor tags={activeTags} onChange={updateActiveTags} triggerWord={activeVersion.trigger_word || undefined} />
              </div>
            </>
          ) : (
            <div className="ds-empty" style={{ margin: 'auto 17px' }}>{t('tagEdit.noActiveHint')}</div>
          )}
        </div>
      </div>
    </StepShell>
  )
}
