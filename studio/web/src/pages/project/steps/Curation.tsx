import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type CurationItem,
  type CurationValidationView,
  type CurationView,
  type ProjectDetail,
  type ValidationItem,
  type Version,
} from '../../../api/client'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import PageHead from '../../../components/ds/PageHead'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { parseFolderMeta } from '../../../lib/folderMeta'
import { useEventStream } from '../../../lib/useEventStream'

// ---------- 排序 ----------
type SortMode =
  | 'id-asc'
  | 'id-desc'
  | 'name-asc'
  | 'name-desc'
  | 'mtime-asc'
  | 'mtime-desc'

const SORT_STORAGE_KEY = 'curation:sort'
const DEFAULT_SORT: SortMode = 'id-asc'

// 右栏目标桶：训练集（默认，现行为）/ 验证集（held-out，扁平无文件夹）
type Bucket = 'train' | 'validation'
const BUCKET_STORAGE_KEY = 'curation:bucket'

function numericIdKey(name: string): number {
  const stem = name.replace(/\.[^.]+$/, '')
  return /^\d+$/.test(stem) ? Number(stem) : Number.POSITIVE_INFINITY
}

function compareItems(a: CurationItem, b: CurationItem, mode: SortMode): number {
  switch (mode) {
    case 'id-asc':
    case 'id-desc': {
      const ka = numericIdKey(a.name)
      const kb = numericIdKey(b.name)
      const d = ka === kb ? a.name.localeCompare(b.name) : ka - kb
      return mode === 'id-asc' ? d : -d
    }
    case 'name-asc':
      return a.name.localeCompare(b.name)
    case 'name-desc':
      return b.name.localeCompare(a.name)
    case 'mtime-asc':
      return a.mtime - b.mtime || a.name.localeCompare(b.name)
    case 'mtime-desc':
      return b.mtime - a.mtime || a.name.localeCompare(b.name)
  }
}

function normalizeItem(it: CurationItem | string | undefined): CurationItem {
  if (typeof it === 'string') return { name: it, mtime: 0 }
  if (it && typeof it.name === 'string')
    return { name: it.name, mtime: typeof it.mtime === 'number' ? it.mtime : 0 }
  return { name: '', mtime: 0 }
}

function sortItems(
  items: (CurationItem | string)[],
  mode: SortMode
): CurationItem[] {
  return items.map(normalizeItem).sort((a, b) => compareItems(a, b, mode))
}

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

interface Preview {
  side: 'left' | 'right'
  name: string
  folder?: string
  url: string
  caption: string
  list: string[]
  index: number
  resolve: (name: string) => string
}

type Focus =
  | { side: 'left'; name: string; url: string }
  | { side: 'right'; folder: string; name: string; url: string }

const FOLDER_PATTERN = /^([0-9]+_)?[A-Za-z][A-Za-z0-9_-]*$/

export default function CurationPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const dialog = useDialog()
  const [view, setView] = useState<CurationView | null>(null)
  const [valView, setValView] = useState<CurationValidationView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const [bucket, setBucket] = useState<Bucket>(() => {
    if (typeof window === 'undefined') return 'train'
    const v = window.localStorage.getItem(BUCKET_STORAGE_KEY)
    return v === 'validation' ? 'validation' : 'train'
  })
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(BUCKET_STORAGE_KEY, bucket)
    }
  }, [bucket])

  const SORT_OPTIONS: { value: SortMode; label: string }[] = [
    { value: 'id-asc', label: 'ID ↑' },
    { value: 'id-desc', label: 'ID ↓' },
    { value: 'name-asc', label: t('common.filename') + ' ↑' },
    { value: 'name-desc', label: t('common.filename') + ' ↓' },
    { value: 'mtime-asc', label: t('curate.downloadTime') + ' ↑' },
    { value: 'mtime-desc', label: t('curate.downloadTime') + ' ↓' },
  ]

  const [leftSel, setLeftSel] = useState<Set<string>>(new Set())
  const [leftAnchor, setLeftAnchor] = useState<string | null>(null)
  const [rightFolder, setRightFolder] = useState<string>('')
  const [rightSel, setRightSel] = useState<Set<string>>(new Set())
  const [rightAnchor, setRightAnchor] = useState<string | null>(null)

  const [focus, setFocus] = useState<Focus | null>(null)
  const [altHeld, setAltHeld] = useState(false)
  useEffect(() => {
    const isAlt = (e: KeyboardEvent) =>
      e.key === 'Alt' || e.code === 'AltLeft' || e.code === 'AltRight'
    const down = (e: KeyboardEvent) => {
      if (isAlt(e)) { e.preventDefault(); setAltHeld(true) }
    }
    const up = (e: KeyboardEvent) => {
      if (isAlt(e)) { e.preventDefault(); setAltHeld(false) }
    }
    const move = (e: MouseEvent) => {
      if (e.altKey !== altHeld) setAltHeld(e.altKey)
    }
    const blur = () => setAltHeld(false)
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('mousemove', move)
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('mousemove', move)
      window.removeEventListener('blur', blur)
    }
  }, [altHeld])

  const [renaming, setRenaming] = useState<{ target: string; value: string } | null>(null)
  const [preview, setPreview] = useState<Preview | null>(null)

  const [sortMode, setSortMode] = useState<SortMode>(() => {
    if (typeof window === 'undefined') return DEFAULT_SORT
    const v = window.localStorage.getItem(SORT_STORAGE_KEY)
    return (['id-asc','id-desc','name-asc','name-desc','mtime-asc','mtime-desc'] as SortMode[]).includes(v as SortMode)
      ? (v as SortMode)
      : DEFAULT_SORT
  })
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(SORT_STORAGE_KEY, sortMode)
    }
  }, [sortMode])

  const versionId = activeVersion?.id ?? null

  const fetchTrain = useCallback(async () => {
    if (versionId == null) return
    try {
      const v = await api.getCuration(project.id, versionId)
      setView(v)
      setError(null)
      const fallback = v.folders.includes('1_data') ? '1_data' : v.folders[0] ?? ''
      if (!rightFolder || !v.folders.includes(rightFolder)) {
        setRightFolder(fallback)
        setRightSel(new Set())
        setRightAnchor(null)
      }
    } catch (e) {
      setError(String(e))
    }
  }, [project.id, versionId, rightFolder])

  const fetchValidation = useCallback(async () => {
    if (versionId == null) return
    try {
      const vv = await api.getCurationValidation(project.id, versionId)
      setValView(vv)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [project.id, versionId])

  // 切换 bucket 只在目标视图没缓存时拉一次：左栏 download−train−validation 两个
  // bucket 完全一致，纯来回切不该重拉、不该整页 loading（左栏用已有数据兜底）。
  useEffect(() => {
    if (versionId == null) return
    if (bucket === 'validation') {
      if (valView == null) void fetchValidation()
    } else if (view == null) {
      void fetchTrain()
    }
  }, [bucket, versionId, view, valView, fetchTrain, fetchValidation])

  // 改动后刷新当前 bucket + 作废另一个的缓存：跨 bucket 增删会改共享的左栏候选
  // 池 / 另一侧右栏，下次切过去自动重拉，保证不显示陈旧的左栏。
  const refresh = useCallback(async () => {
    if (bucket === 'validation') {
      await fetchValidation()
      setView(null)
    } else {
      await fetchTrain()
      setValView(null)
    }
  }, [bucket, fetchTrain, fetchValidation])

  useEventStream((evt) => {
    if (
      evt.type === 'version_state_changed' &&
      evt.project_id === project.id &&
      versionId != null &&
      evt.version_id === versionId
    ) {
      void refresh()
    }
  })

  const isVal = bucket === 'validation'
  const folderNames = view?.folders ?? []

  // 左栏候选 download − train − validation，两个 bucket 共用同一池 —— 目标视图
  // 尚未加载时回退到另一视图的（内容相同的）左栏，切换时左栏不空屏、不闪。
  const currentLeft = useMemo(
    () => (isVal ? valView?.left ?? view?.left ?? [] : view?.left ?? valView?.left ?? []),
    [isVal, valView, view]
  )
  const leftSortedNames = useMemo(
    () => sortItems(currentLeft, sortMode).map((e) => e.name),
    [currentLeft, sortMode]
  )

  // train 右栏：当前文件夹的 entries（带 origin）。validation 右栏：全量扁平。
  const trainEntries = useMemo(
    () => (view && rightFolder ? view.right[rightFolder] ?? [] : []),
    [view, rightFolder]
  )
  const valEntries = useMemo<ValidationItem[]>(() => valView?.right ?? [], [valView])
  const rightSortedNames = useMemo(
    () => sortItems(isVal ? valEntries : trainEntries, sortMode).map((e) => e.name),
    [isVal, valEntries, trainEntries, sortMode]
  )

  // ADR 0010 fixup: train 区 thumb 走 download bucket + manifest.origin，
  // 显示"预处理前的样子"。trainEntries 已带 origin（backend list_train 加）。
  // 用 raw=1 跳过 resolve_origin —— 否则老 ADR 0004 设计会 hijack 到
  // preprocess/{派生} 派生（X.jpg → preprocess/X_c0.png），但 ADR 0010
  // 后 preprocess/ 不再被 worker 写 → 404 裂图。
  const rightOriginByName = useMemo(() => {
    const m = new Map<string, string>()
    for (const e of trainEntries) {
      m.set(e.name, e.origin ?? e.name)
    }
    return m
  }, [trainEntries])
  // validation 图无 manifest / 无 origin：按 name → 物理 folder 反查，供
  // 缩略图寻址（version thumb 的 validation bucket 需 folder）+ 精确删除。
  const valFolderByName = useMemo(() => {
    const m = new Map<string, string>()
    for (const e of valEntries) m.set(e.name, e.folder)
    return m
  }, [valEntries])

  const leftItems = useMemo(
    () => leftSortedNames.map((n) => ({
      name: n,
      thumbUrl: api.projectThumbUrl(project.id, n, 'download', 256, undefined, true),
    })),
    [leftSortedNames, project.id]
  )
  const rightItems = useMemo(
    () =>
      versionId == null
        ? []
        : rightSortedNames.map((n) => ({
            name: n,
            thumbUrl: isVal
              ? api.versionThumbUrl(
                  project.id, versionId, 'validation', n, valFolderByName.get(n), 256,
                )
              : api.projectThumbUrl(
                  project.id, rightOriginByName.get(n) ?? n, 'download', 256,
                  undefined, true,
                ),
          })),
    [rightSortedNames, project.id, versionId, isVal, rightOriginByName, valFolderByName]
  )

  const onLeftHover = useCallback(
    (name: string) =>
      setFocus({
        side: 'left', name,
        url: api.projectThumbUrl(project.id, name, 'download', 768, undefined, true),
      }),
    [project.id]
  )

  const onRightHover = useCallback(
    (name: string) => {
      if (versionId == null) return
      if (isVal) {
        const folder = valFolderByName.get(name)
        if (!folder) return
        setFocus({
          side: 'right', folder, name,
          url: api.versionThumbUrl(project.id, versionId, 'validation', name, folder, 768),
        })
        return
      }
      if (!rightFolder) return
      const origin = rightOriginByName.get(name) ?? name
      setFocus({
        side: 'right',
        folder: rightFolder,
        name,
        url: api.projectThumbUrl(project.id, origin, 'download', 768, undefined, true),
      })
    },
    [versionId, project.id, isVal, valFolderByName, rightFolder, rightOriginByName]
  )

  if (!activeVersion) {
    return <p className="text-fg-tertiary p-6">{t('curate.noVersion')}</p>
  }
  if (error) {
    return (
      <div className="p-3 rounded-md bg-err-soft border border-err text-err font-mono text-sm">
        {error}
      </div>
    )
  }
  // 整页 loading 只在首次（两个视图都没数据）出现；切换 bucket 时另一视图已有
  // 数据，页面照常渲染，只有右栏在等自己那侧的数据。
  if (view == null && valView == null) {
    return <p className="text-fg-tertiary p-6">{t('curate.loading')}</p>
  }

  // 当前 bucket 自己那侧的右栏数据是否还在加载（切到一个从没看过的 bucket 时）
  const rightLoading = isVal ? valView == null : view == null
  const downloadTotal = isVal
    ? valView?.download_total ?? view?.download_total ?? 0
    : view?.download_total ?? valView?.download_total ?? 0

  const switchBucket = (next: Bucket) => {
    if (next === bucket) return
    setBucket(next)
    setLeftSel(new Set())
    setLeftAnchor(null)
    setRightSel(new Set())
    setRightAnchor(null)
    setRenaming(null)
  }

  const switchRightFolder = (next: string) => {
    setRightFolder(next)
    setRightSel(new Set())
    setRightAnchor(null)
  }

  const handleLeftClick = (name: string, e: React.MouseEvent) => {
    const r = applySelection(leftSel, name, e, leftSortedNames, leftAnchor)
    setLeftSel(r.next)
    setLeftAnchor(r.anchor)
  }

  const handleRightClick = (name: string, e: React.MouseEvent) => {
    const r = applySelection(rightSel, name, e, rightSortedNames, rightAnchor)
    setRightSel(r.next)
    setRightAnchor(r.anchor)
  }

  const copyLeftFiles = async (files: string[], options: { clearSelection?: boolean } = {}) => {
    if (files.length === 0 || busy) return false
    if (!isVal) {
      if (!rightFolder) { toast(t('curate.noTargetFolder'), 'error'); return false }
      if (!FOLDER_PATTERN.test(rightFolder)) { toast(t('curate.invalidFolder'), 'error'); return false }
    }
    setBusy(true)
    try {
      const r = isVal
        ? await api.copyToValidation(project.id, activeVersion.id, { files })
        : await api.copyToTrain(project.id, activeVersion.id, { files, dest_folder: rightFolder })
      toast(
        t('curate.copiedN', { n: r.copied.length }) +
        (r.skipped.length ? t('curate.copiedSkipped', { n: r.skipped.length }) : ''),
        'success'
      )
      if (options.clearSelection) setLeftSel(new Set())
      await refresh()
      await reload()
      return true
    } catch (e) {
      toast(String(e), 'error')
      return false
    } finally {
      setBusy(false)
    }
  }

  const removeRightFiles = async (
    folder: string,
    files: string[],
    options: { clearSelection?: boolean; confirm?: boolean } = {}
  ) => {
    if (files.length === 0 || busy) return false
    if (!isVal && !folder) return false
    const folderLabel = isVal ? t('curate.bucketValidation') : folder
    if (options.confirm &&
        !(await dialog.confirm(t('curate.confirmRemove', { folder: folderLabel, n: files.length }), { tone: 'warn', okText: t('curate.removeOkText') }))) {
      return false
    }
    setBusy(true)
    try {
      const r = isVal
        ? await api.removeFromValidation(project.id, activeVersion.id, {
            items: files
              .map((name) => ({ folder: valFolderByName.get(name) ?? '', name }))
              .filter((it) => it.folder),
          })
        : await api.removeFromTrain(project.id, activeVersion.id, { folder, files })
      toast(t('curate.removedN', { n: r.removed.length }), 'success')
      if (options.clearSelection) setRightSel(new Set())
      await refresh()
      await reload()
      return true
    } catch (e) {
      toast(String(e), 'error')
      return false
    } finally {
      setBusy(false)
    }
  }

  const doCopy = async () => {
    await copyLeftFiles(Array.from(leftSel), { clearSelection: true })
  }

  const doRemove = async () => {
    await removeRightFiles(rightFolder, Array.from(rightSel), { clearSelection: true, confirm: true })
  }

  const doRenameFolder = async () => {
    if (!renaming) return
    const target = renaming.target
    const next = renaming.value.trim()
    if (!next || next === target) { setRenaming(null); return }
    if (!FOLDER_PATTERN.test(next)) return toast(t('curate.invalidFolder'), 'error')
    setBusy(true)
    try {
      await api.folderOp(project.id, activeVersion.id, { op: 'rename', name: target, new_name: next })
      if (rightFolder === target) switchRightFolder(next)
      setRenaming(null)
      toast(t('curate.renamedToast', { from: target, to: next }), 'success')
      await refresh()
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const doDeleteFolder = async (name: string) => {
    const cnt = view?.right[name]?.length ?? 0
    if (!(await dialog.confirm(
      t('curate.confirmDeleteFolder', { name, n: cnt }),
      { tone: 'warn', okText: t('curate.deleteFolderOkText') },
    ))) return
    setBusy(true)
    try {
      await api.folderOp(project.id, activeVersion.id, { op: 'delete', name })
      if (rightFolder === name) switchRightFolder('')
      await refresh()
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const openLeftPreview = (name: string) => {
    setPreview({
      side: 'left', name,
      url: api.projectThumbUrl(project.id, name, 'download', 1600),
      caption: name,
      list: leftSortedNames,
      index: leftSortedNames.indexOf(name),
      resolve: (n) => api.projectThumbUrl(project.id, n, 'download', 1600),
    })
  }
  const openRightPreview = (name: string) => {
    if (versionId == null) return
    if (isVal) {
      const folder = valFolderByName.get(name) ?? ''
      setPreview({
        side: 'right', name, folder,
        url: api.versionThumbUrl(project.id, versionId, 'validation', name, folder, 1600),
        caption: name,
        list: rightSortedNames,
        index: rightSortedNames.indexOf(name),
        resolve: (n) =>
          api.versionThumbUrl(project.id, versionId, 'validation', n, valFolderByName.get(n) ?? '', 1600),
      })
      return
    }
    const folder = rightFolder
    setPreview({
      side: 'right', name, folder,
      url: api.versionThumbUrl(project.id, versionId, 'train', name, folder, 1600),
      caption: `${folder}/${name}`,
      list: rightSortedNames,
      index: rightSortedNames.indexOf(name),
      resolve: (n) => api.versionThumbUrl(project.id, versionId, 'train', n, folder, 1600),
    })
  }
  const stepPreview = (delta: number) => {
    if (!preview) return
    const idx = preview.index + delta
    if (idx < 0 || idx >= preview.list.length) return
    const name = preview.list[idx]
    setPreview({
      ...preview, name,
      url: preview.resolve(name),
      caption: preview.side === 'right' && preview.folder && !isVal ? `${preview.folder}/${name}` : name,
      index: idx,
    })
  }

  const advancePreviewAfterAction = (doneName: string) => {
    if (!preview) return
    const list = preview.list.filter((name) => name !== doneName)
    if (list.length === 0) { setPreview(null); return }
    const index = Math.min(preview.index, list.length - 1)
    const name = list[index]
    setPreview({
      ...preview, name,
      url: preview.resolve(name),
      caption: preview.side === 'right' && preview.folder && !isVal ? `${preview.folder}/${name}` : name,
      list, index,
    })
  }

  const copyPreviewImage = async () => {
    if (!preview || preview.side !== 'left' || busy) return
    const name = preview.name
    if (await copyLeftFiles([name])) advancePreviewAfterAction(name)
  }

  const removePreviewImage = async () => {
    if (!preview || preview.side !== 'right' || !preview.folder || busy) return
    const folder = preview.folder
    const name = preview.name
    if (await removeRightFiles(folder, [name])) advancePreviewAfterAction(name)
  }

  // Train → validation: the copies leave the train set (with their edits)
  // and the originals go to the held-out set.
  const doMoveToValidation = async () => {
    if (isVal || !rightFolder || rightSel.size === 0 || busy) return
    const files = Array.from(rightSel)
    if (!(await dialog.confirm(
      t('curate.confirmMoveToVal', { n: files.length, folder: rightFolder }),
      { tone: 'warn', okText: t('curate.moveToValOk') },
    ))) return
    setBusy(true)
    try {
      await api.removeFromTrain(project.id, activeVersion.id, { folder: rightFolder, files })
      const origins = files.map((n) => rightOriginByName.get(n) ?? n)
      const r = await api.copyToValidation(project.id, activeVersion.id, { files: origins })
      toast(t('curate.movedToValN', { n: r.copied.length }), 'success')
      setRightSel(new Set())
      setView(null)
      setValView(null)
      await fetchTrain()
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const doNewFolder = async () => {
    const name = (await dialog.prompt(t('curate.newFolderPrompt'), { defaultValue: '' }))?.trim()
    if (!name) return
    if (!FOLDER_PATTERN.test(name)) { toast(t('curate.invalidFolder'), 'error'); return }
    setBusy(true)
    try {
      await api.folderOp(project.id, activeVersion.id, { op: 'create', name })
      switchRightFolder(name)
      await refresh()
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const trainTotal = view?.train_total ?? activeVersion.stats?.train_image_count ?? 0
  const valTotal = valView?.val_total ?? activeVersion.stats?.validation_image_count ?? 0
  const effective = folderNames.reduce(
    (s, f) => s + parseFolderMeta(f).repeat * (view?.right[f]?.length ?? 0), 0,
  )
  const valPct = downloadTotal > 0 ? ((valTotal / downloadTotal) * 100).toFixed(1) : null

  return (
    <div className="fade-in" style={{ minHeight: '100%', display: 'flex', flexDirection: 'column' }}>
      <PageHead
        accent
        eyebrow={t('steps.eyebrowStep', { n: 1, label: activeVersion.label })}
        title={t('steps.curate.title')}
        subtitle={t('curate.subtitleLong')}
        tools={
          <>
            <div className="ds-seg" role="group" aria-label={t('curate.bucketLabel')} data-testid="curate-bucket-toggle">
              <button type="button" className={`ds-seg-item${!isVal ? ' ds-is-active' : ''}`} aria-pressed={!isVal} onClick={() => switchBucket('train')}>{t('curate.bucketTrain')}</button>
              <button type="button" className={`ds-seg-item${isVal ? ' ds-is-active' : ''}`} aria-pressed={isVal} onClick={() => switchBucket('validation')}>{t('curate.bucketValShort')}</button>
            </div>
            <Link className="ds-btn-primary" to={`/projects/${project.id}/v/${activeVersion.id}/preprocess`}>{t('curate.next')}</Link>
          </>
        }
      />

      <div className="ds-scroll ds-tight" style={{ flex: '1 0 auto' }}>
        <div className="ds-kpis ds-tight ds-ov-kpis">
          <div className="ds-card ds-kpi">
            <div className="ds-kpi-top"><span className="ds-kpi-icon">{Icon.image}</span><span style={{ marginLeft: 'auto' }}><span className="ds-kpi-meta">{t('curate.kpiUnused', { n: currentLeft.length.toLocaleString() })}</span></span></div>
            <div className="ds-kpi-val" style={{ marginTop: 14 }}>{downloadTotal.toLocaleString()}</div>
            <div className="ds-kpi-label">{t('curate.kpiTotal')}</div>
          </div>
          <div className="ds-card ds-kpi">
            <div className="ds-kpi-top"><span className="ds-kpi-icon">{Icon.stack}</span><span style={{ marginLeft: 'auto' }}><span className="ds-kpi-meta">{t('curate.kpiGroups', { count: folderNames.length })}</span></span></div>
            <div className="ds-kpi-val" style={{ marginTop: 14 }}>{trainTotal.toLocaleString()}</div>
            <div className="ds-kpi-label">{t('curate.kpiTrain')}</div>
          </div>
          <div className="ds-card ds-kpi">
            <div className="ds-kpi-top"><span className="ds-kpi-icon">{Icon.shield}</span><span style={{ marginLeft: 'auto' }}>{valPct != null && <span className="ds-kpi-meta">{t('curate.kpiValPct', { pct: valPct })}</span>}</span></div>
            <div className="ds-kpi-val" style={{ marginTop: 14 }}>{valTotal.toLocaleString()}</div>
            <div className="ds-kpi-label">{t('curate.kpiVal')}</div>
          </div>
          <div className="ds-card ds-kpi">
            <div className="ds-kpi-top"><span className="ds-kpi-icon">{Icon.repeat}</span><span style={{ marginLeft: 'auto' }}><span className="ds-kpi-meta">{t('curate.kpiEffectiveMeta')}</span></span></div>
            <div className="ds-kpi-val" style={{ marginTop: 14 }}>{view ? effective.toLocaleString() : '—'}</div>
            <div className="ds-kpi-label">{t('curate.kpiEffective')}</div>
          </div>
        </div>

        <div className="ds-cur-main">
          {/* left: the dataset pool */}
          <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <div className="ds-card-head ds-pad">
              <div>
                <div className="ds-card-title">{t('curate.downloadPanelTitle')}</div>
                <div className="ds-card-sub">{t('curate.downloadSubtitle', { unused: currentLeft.length, total: downloadTotal, sel: leftSel.size })}</div>
              </div>
              <div className="ds-card-tools">
                <select
                  className="ds-inp"
                  style={{ width: 150, height: 30 }}
                  value={sortMode}
                  onChange={(e) => setSortMode(e.target.value as SortMode)}
                  title={t('curate.sortTitle')}
                  aria-label={t('curate.sortLabel')}
                >
                  {SORT_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
                {leftSel.size > 0 ? (
                  <button type="button" className="ds-ctl ds-ghost" onClick={() => setLeftSel(new Set())} disabled={busy}>{t('curate.deselect')}</button>
                ) : (
                  <button type="button" className="ds-ctl ds-ghost" onClick={() => setLeftSel(new Set(leftSortedNames))} disabled={busy || leftSortedNames.length === 0}>{t('curate.selectAll')}</button>
                )}
              </div>
            </div>
            <div className="ds-card-body" style={{ paddingTop: 2, flex: 1, minHeight: 0 }}>
              <div className="ds-cur-grid">
                <ImageGrid
                  items={leftItems}
                  selected={leftSel}
                  activeName={preview?.side === 'left' ? preview.name : undefined}
                  onSelect={handleLeftClick}
                  onHover={onLeftHover}
                  onPreview={openLeftPreview}
                  onActivate={openLeftPreview}
                  clickMode="activate"
                  columnsClass="grid-cols-[repeat(auto-fill,minmax(84px,1fr))]"
                  ariaLabel="download-grid"
                  emptyHint={t('curate.downloadEmptyHint')}
                />
              </div>
            </div>
            <div className="ds-cardfoot">
              {isVal ? (
                <button type="button" className="ds-btn-primary" style={{ height: 34 }} onClick={doCopy} disabled={busy || leftSel.size === 0} title={t('curate.copyToValTitle')}>
                  {t('curate.addNToVal', { n: leftSel.size })}
                </button>
              ) : (
                <button
                  type="button"
                  className="ds-btn-primary"
                  style={{ height: 34 }}
                  onClick={doCopy}
                  disabled={busy || leftSel.size === 0 || !rightFolder}
                  title={rightFolder ? t('curate.copyToTitle', { folder: rightFolder }) : t('curate.noFolderTitle')}
                >
                  {t('curate.addNTo', { n: leftSel.size, folder: rightFolder || '?' })}
                </button>
              )}
              <span className="ds-cardfoot-act" />
            </div>
          </div>

          {/* right: the version's train (or validation) set */}
          <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <div className="ds-card-head ds-pad">
              <div>
                <div className="ds-card-title">{isVal ? t('curate.valPanelTitle') : t('curate.trainPanelTitle')}</div>
                <div className="ds-card-sub">
                  {isVal
                    ? t('curate.valSubtitle', { total: valView?.val_total ?? 0, sel: rightSel.size })
                    : t('curate.trainSubtitle', { total: view?.train_total ?? 0, folders: folderNames.length, sel: rightSel.size })}
                </div>
              </div>
              <div className="ds-card-tools">
                {rightSel.size > 0 ? (
                  <button type="button" className="ds-ctl ds-ghost" onClick={() => setRightSel(new Set())} disabled={busy}>{t('curate.deselect')}</button>
                ) : (
                  <button type="button" className="ds-ctl ds-ghost" onClick={() => setRightSel(new Set(rightSortedNames))} disabled={busy || rightSortedNames.length === 0}>{t('curate.selectAll')}</button>
                )}
                {!isVal && (
                  <button type="button" className="ds-ctl ds-ghost" onClick={() => rightFolder && setRenaming({ target: rightFolder, value: rightFolder })} disabled={busy || !rightFolder}>
                    {t('common.rename')}
                  </button>
                )}
              </div>
            </div>

            {isVal ? (
              <div style={{ padding: '0 17px 10px' }}><div className="ds-note" style={{ fontSize: 11.5 }}>{t('curate.valHint')}</div></div>
            ) : (
              <div style={{ padding: '0 17px 10px', display: 'flex', gap: 7, flexWrap: 'wrap', alignItems: 'center' }}>
                {folderNames.length === 0 && <span className="ds-muted" style={{ fontSize: 12 }}>{t('curate.noTrainFolders')}</span>}
                {folderNames.map((f) => {
                  const active = f === rightFolder
                  return (
                    <span key={f} className={`ds-chip${active ? ' ds-is-active' : ''}`}>
                      <button type="button" onClick={() => switchRightFolder(f)} title={active ? t('curate.folderActiveTitle') : t('curate.folderSwitchTitle')} style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
                        {f} <b>{(view?.right[f]?.length ?? 0).toLocaleString()}</b>
                      </button>
                      <button type="button" className="ds-chip-x" aria-label={`${t('curate.deleteFolderOkText')} ${f}`} title={t('curate.deleteFolderOkText')} onClick={() => void doDeleteFolder(f)} disabled={busy}>{Icon.x}</button>
                    </span>
                  )
                })}
                <button type="button" className="ds-chip-add" onClick={() => void doNewFolder()} disabled={busy}>{t('curate.newFolderPlaceholder')}</button>
              </div>
            )}

            {renaming && !isVal && (
              <div style={{ padding: '0 17px 10px', display: 'flex', alignItems: 'center', gap: 8, fontSize: 12.5 }}>
                <span className="ds-muted">{t('curate.renameLabel', { name: renaming.target })}</span>
                <input
                  autoFocus
                  className="ds-inp ds-mono"
                  style={{ width: 190, height: 30 }}
                  value={renaming.value}
                  onChange={(e) => setRenaming({ ...renaming, value: e.target.value })}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') void doRenameFolder()
                    if (e.key === 'Escape') setRenaming(null)
                  }}
                />
                <button type="button" className="ds-btn-primary" style={{ height: 30 }} onClick={() => void doRenameFolder()} disabled={busy}>{t('curate.renameOk')}</button>
                <button type="button" className="ds-ctl ds-ghost" style={{ height: 30 }} onClick={() => setRenaming(null)}>{t('common.cancel')}</button>
              </div>
            )}

            <div className="ds-card-body" style={{ paddingTop: 0, flex: 1, minHeight: 0 }}>
              <div className="ds-cur-grid">
                <ImageGrid
                  items={rightItems}
                  selected={rightSel}
                  activeName={preview?.side === 'right' ? preview.name : undefined}
                  onSelect={handleRightClick}
                  onHover={onRightHover}
                  onPreview={openRightPreview}
                  onActivate={openRightPreview}
                  clickMode="activate"
                  columnsClass="grid-cols-[repeat(auto-fill,minmax(84px,1fr))]"
                  ariaLabel={isVal ? 'validation-grid' : 'train-grid'}
                  emptyHint={
                    rightLoading
                      ? t('curate.loading')
                      : isVal
                        ? t('curate.valEmpty')
                        : rightFolder
                          ? t('curate.trainEmptyFolder', { folder: rightFolder })
                          : t('curate.trainNoFolder')
                  }
                />
              </div>
            </div>
            <div className="ds-cardfoot">
              <button type="button" className="ds-ctl" style={{ height: 34 }} onClick={doRemove} disabled={busy || rightSel.size === 0 || (!isVal && !rightFolder)}>
                {rightSel.size > 0 ? t('curate.removeFromSetN', { n: rightSel.size }) : t('curate.removeFromSet')}
              </button>
              <span className="ds-cardfoot-act">
                {!isVal && (
                  <button type="button" className="ds-ctl" style={{ height: 34 }} onClick={() => void doMoveToValidation()} disabled={busy || rightSel.size === 0 || !rightFolder} title={t('curate.moveToValTitle')}>
                    {t('curate.moveToVal')}
                  </button>
                )}
              </span>
            </div>
          </div>
        </div>
      </div>

      {altHeld && focus && <AltHoverPreview focus={focus} isVal={isVal} />}

      {preview && (
        <ImagePreviewModal
          src={preview.url}
          caption={preview.caption}
          index={preview.index}
          total={preview.list.length}
          hasPrev={preview.index > 0}
          hasNext={preview.index < preview.list.length - 1}
          onClose={() => setPreview(null)}
          onPrev={() => stepPreview(-1)}
          onNext={() => stepPreview(1)}
          onAccept={preview.side === 'left' ? copyPreviewImage : undefined}
          onDelete={preview.side === 'right' ? removePreviewImage : undefined}
          shortcutHint={preview.side === 'left' ? t('curate.previewHintLeft') : t('curate.previewHintRight')}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------

const Icon = {
  image: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="16" rx="2.5" /><path d="m3 15 4.5-4.5 4 4L15 11l6 5.5" /></svg>,
  stack: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /></svg>,
  shield: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3 4 6v6c0 4.5 3.4 8.3 8 9 4.6-.7 8-4.5 8-9V6l-8-3Z" /></svg>,
  repeat: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M17 2l4 4-4 4" /><path d="M3 11V9a3 3 0 0 1 3-3h15" /><path d="M7 22l-4-4 4-4" /><path d="M21 13v2a3 3 0 0 1-3 3H3" /></svg>,
  x: <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>,
}

function AltHoverPreview({ focus, isVal }: { focus: Focus; isVal: boolean }) {
  const { t } = useTranslation()
  const sourceLabel = focus.side === 'left'
    ? t('curate.sourceLabelDownload')
    : isVal
      ? t('curate.bucketValidation')
      : t('curate.sourceLabelTrain', { folder: focus.folder })
  return (
    <div aria-hidden className="fixed inset-0 z-40 pointer-events-none flex items-center justify-center p-6">
      <div className="relative flex flex-col overflow-hidden rounded-[14px] max-w-[95vw] max-h-[95vh] bg-black/90 shadow-xl">
        <img src={focus.url} alt={focus.name} className="max-w-[95vw] max-h-[88vh] object-contain" />
        <div className="flex items-center gap-2 shrink-0 px-3 py-1.5 border-t border-white/[0.08]">
          <span className={`ds-badge ${focus.side === 'left' ? 'ds-ok' : 'ds-info'}`}>{sourceLabel}</span>
          <code className="ds-mono truncate flex-1 min-w-0 text-white text-sm">{focus.name}</code>
          <span className="text-xs shrink-0 text-white/40">{t('curate.altHoverClose')}</span>
        </div>
      </div>
    </div>
  )
}
