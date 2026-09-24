import { useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import { useTranslation } from 'react-i18next'

import { TranslatedTag } from './tagDisplay/TranslatedTag'
import { TagSuggestList } from './tagSuggest/TagSuggestList'
import { useTagSuggest } from './tagSuggest/useTagSuggest'

type Sort = 'count_desc' | 'count_asc' | 'name_asc' | 'name_desc'

interface Props {
  cache: Map<string, string[]>
  /** Every image key of the version; counted when nothing is selected. */
  allKeys: string[]
  selectedKeys: string[]
  /** Tag written first into every caption; highlighted in the list. */
  triggerWord?: string
  /** Tag picked in the list; the footer actions work on it. */
  pickedTag: string | null
  /** 点 tag 文字 = 选中所有含该 tag 的图（保留旧行为，仍是 tag 浏览的主路径）。 */
  onPickTag: (tag: string) => void
  /** 从当前选中图中删除该 tag。 */
  onRemoveTag: (tag: string) => void
  /** 从当前选中图中把 oldTag 替换为 newTag。 */
  onReplaceTag: (oldTag: string, newTag: string) => void
}

/** Tag statistics card (mockup TagEdit, left column): filter, one meter per
 *  tag, and remove / replace for the picked tag. Counts cover the selected
 *  images, or the whole version when nothing is selected. */
export default function TagStatsPanel({
  cache,
  allKeys,
  selectedKeys,
  triggerWord,
  pickedTag,
  onPickTag,
  onRemoveTag,
  onReplaceTag,
}: Props) {
  const { t } = useTranslation()
  const [filter, setFilter] = useState('')
  const [sort, setSort] = useState<Sort>('count_desc')
  // inline replace state：editingTag = 正在 inline 改名的 tag；editValue 是
  // input 当前值。按回车提交；Esc / 失焦取消。
  const [editingTag, setEditingTag] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const editInputRef = useRef<HTMLInputElement>(null)

  const usingSelection = selectedKeys.length > 0
  const countedKeys = usingSelection ? selectedKeys : allKeys

  const items = useMemo(() => {
    const counter = new Map<string, number>()
    for (const k of countedKeys) {
      for (const tag of cache.get(k) ?? []) counter.set(tag, (counter.get(tag) ?? 0) + 1)
    }
    return Array.from(counter.entries())
  }, [cache, countedKeys])

  const sorted = useMemo(() => {
    const out = [...items]
    if (sort === 'count_desc') out.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    else if (sort === 'count_asc') out.sort((a, b) => a[1] - b[1] || a[0].localeCompare(b[0]))
    else if (sort === 'name_asc') out.sort((a, b) => a[0].localeCompare(b[0]))
    else out.sort((a, b) => b[0].localeCompare(a[0]))
    return out
  }, [items, sort])

  const filtered = useMemo(() => {
    const f = filter.trim().toLowerCase()
    if (!f) return sorted
    return sorted.filter(([tag]) => tag.toLowerCase().includes(f))
  }, [sorted, filter])

  const maxCount = countedKeys.length || 1

  // 进入 edit mode 时 focus + select 输入框
  useEffect(() => {
    if (editingTag && editInputRef.current) {
      editInputRef.current.focus()
      editInputRef.current.select()
    }
  }, [editingTag])

  const startEdit = (tag: string) => { setEditingTag(tag); setEditValue(tag) }
  const cancelEdit = () => { setEditingTag(null); setEditValue('') }
  const commitEdit = () => {
    const o = editingTag
    const n = editValue.trim()
    if (!o || !n || n === o) { cancelEdit(); return }
    onReplaceTag(o, n)
    cancelEdit()
  }

  const canAct = pickedTag != null && usingSelection

  return (
    <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div className="ds-card-head ds-pad">
        <div>
          <div className="ds-card-title">{t('tagStats.title')}</div>
          <div className="ds-card-sub">
            {t('tagStats.unique', { n: items.length })}
            {usingSelection && ` · ${t('tagStats.inSelected', { n: selectedKeys.length })}`}
          </div>
        </div>
      </div>
      <div style={{ padding: '0 17px 10px', display: 'flex', gap: 6 }}>
        <span className="ds-search" style={{ flex: 1 }}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" /></svg>
          <input className="ds-inp" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder={t('tagStats.filterPlaceholder')} aria-label={t('tagStats.filterPlaceholder')} />
        </span>
        <select className="ds-inp" style={{ width: 74, flex: 'none' }} value={sort} onChange={(e) => setSort(e.target.value as Sort)} aria-label={t('tagStats.sortLabel')}>
          <option value="count_desc">{t('tagStats.sortCountDesc')}</option>
          <option value="count_asc">{t('tagStats.sortCountAsc')}</option>
          <option value="name_asc">{t('tagStats.sortNameAsc')}</option>
          <option value="name_desc">{t('tagStats.sortNameDesc')}</option>
        </select>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '0 17px 14px', display: 'flex', flexDirection: 'column', gap: 3 }}>
        {filtered.length === 0 ? (
          <p className="ds-muted" style={{ fontSize: 12, margin: 0 }}>
            {filter.trim() ? t('tagStats.noMatch') : usingSelection ? t('tagStats.noTagsSelected') : t('tagStats.noTags')}
          </p>
        ) : filtered.map(([tag, c]) => {
          if (editingTag === tag) {
            return (
              <div key={tag} style={{ padding: '3px 0' }}>
                <EditTagInput
                  editInputRef={editInputRef}
                  value={editValue}
                  setValue={setEditValue}
                  onCommit={commitEdit}
                  onCancel={cancelEdit}
                />
              </div>
            )
          }
          const picked = tag === pickedTag
          const isTrigger = !!triggerWord && tag === triggerWord
          return (
            <button
              key={tag}
              type="button"
              onClick={() => onPickTag(tag)}
              onDoubleClick={() => startEdit(tag)}
              title={t('tagStats.pickTagTitle', { tag })}
              aria-pressed={picked}
              style={{ display: 'block', textAlign: 'left', width: '100%', padding: '3px 6px', margin: '0 -6px', borderRadius: 7, background: picked ? 'var(--sunken)' : undefined, boxSizing: 'content-box' }}
            >
              <div className="ds-kv" style={{ padding: '0 0 3px' }}>
                <span className="ds-k ds-mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: isTrigger ? 'var(--green-text)' : picked ? 'var(--ink)' : undefined }}>
                  <TranslatedTag tag={tag} />
                </span>
                <span className="ds-v">{c}</span>
              </div>
              <span className="ds-meter ds-thin"><i style={{ width: `${Math.max((c / maxCount) * 100, 1)}%` }} /></span>
            </button>
          )
        })}
      </div>

      <div style={{ borderTop: '1px solid var(--line)', padding: '10px 17px', display: 'flex', gap: 7 }}>
        <button
          type="button"
          className="ds-ctl ds-ghost"
          style={{ flex: 1, justifyContent: 'center' }}
          disabled={!canAct}
          onClick={() => pickedTag && onRemoveTag(pickedTag)}
          title={pickedTag ? t('tagStats.removeTitle', { tag: pickedTag }) : t('tagStats.pickFirst')}
        >{t('tagStats.removeBtn')}</button>
        <button
          type="button"
          className="ds-ctl ds-ghost"
          style={{ flex: 1, justifyContent: 'center' }}
          disabled={!canAct}
          onClick={() => pickedTag && startEdit(pickedTag)}
          title={pickedTag ? t('tagStats.replaceTitle', { tag: pickedTag }) : t('tagStats.pickFirst')}
        >{t('tagStats.replaceBtn')}</button>
      </div>
    </div>
  )
}

/** Inline tag rename input + 翻译 autocomplete。抽出来是为了能放 useTagSuggest hook —
 *  父组件 .map 内部不能调 hook。 */
function EditTagInput({ editInputRef, value, setValue, onCommit, onCancel }: {
  editInputRef: RefObject<HTMLInputElement>
  value: string
  setValue: (v: string) => void
  onCommit: () => void
  onCancel: () => void
}) {
  const { t } = useTranslation()
  const suggest = useTagSuggest({
    value,
    inputRef: editInputRef,
    wholeAsToken: true,
    onPick: ({ suggestion }) => setValue(suggestion.tag),
  })
  return (
    <div style={{ position: 'relative' }}>
      <input
        ref={editInputRef}
        value={value}
        onChange={(e) => { setValue(e.target.value); suggest.notifyChange() }}
        onKeyDown={(e) => {
          if (suggest.handleKeyDown(e)) return
          if (e.key === 'Enter') onCommit()
          else if (e.key === 'Escape') onCancel()
        }}
        onClick={() => suggest.notifyClick()}
        onFocus={() => suggest.notifyFocus()}
        onBlur={() => { suggest.notifyBlur(); onCancel() }}
        className="ds-inp ds-mono"
        style={{ height: 28 }}
        aria-label={t('tagStats.replaceWith')}
        placeholder={t('tagStats.replaceWith')}
      />
      <TagSuggestList
        open={suggest.open}
        suggestions={suggest.suggestions}
        activeIdx={suggest.activeIdx}
        onPick={(s) => suggest.pickAt(suggest.suggestions.indexOf(s))}
        onHover={suggest.setActiveIdx}
        inputRef={editInputRef}
        cursor={suggest.cursor}
        positionDeps={[value]}
      />
    </div>
  )
}
