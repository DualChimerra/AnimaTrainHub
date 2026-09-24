import { useEffect, useMemo, useRef, useState } from 'react'
import {
  DndContext,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'
import {
  SortableContext,
  arrayMove,
  rectSortingStrategy,
  useSortable,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useTranslation } from 'react-i18next'

import { TranslatedTag } from './tagDisplay/TranslatedTag'
import { TagSuggestList } from './tagSuggest/TagSuggestList'
import { useTagSuggest } from './tagSuggest/useTagSuggest'

interface Props {
  tags: string[]
  natural?: boolean
  onChange: (tags: string[]) => void
  onSave?: () => void | Promise<void>
  saving?: boolean
  dirty?: boolean
  /** Trigger word; its chip is shown in green. */
  triggerWord?: string
}

type Mode = 'chip' | 'text'

const parseLine = (raw: string): string[] =>
  raw.split(/[,，\n]/).map((t) => t.trim()).filter(Boolean)

export default function TagEditor({
  tags, natural, onChange, onSave, saving, dirty, triggerWord,
}: Props) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState('')
  const tagsJoined = useMemo(() => tags.join(', '), [tags])
  const [mode, setMode] = useState<Mode>(natural ? 'text' : 'chip')
  const [textBuf, setTextBuf] = useState(() => tagsJoined)
  const draftInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // PointerSensor + 6px 启动距离：拖拽手感不会跟「点 × 删除」/ 误触冲突。
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
  )

  // Reset draft when image switches
  useEffect(() => { setDraft('') }, [tags])

  // Sync textBuf when tags change WHILE in text mode (image switch)
  const prevTagsJoinedRef = useRef(tagsJoined)
  useEffect(() => {
    if (mode === 'text' && tagsJoined !== prevTagsJoinedRef.current) {
      setTextBuf(tagsJoined)
    }
    prevTagsJoinedRef.current = tagsJoined
  }, [tagsJoined, mode])

  const addTag = (raw: string) => {
    const t = raw.trim().replace(/^[,，]+|[,，]+$/g, '')
    if (!t) return
    if (tags.includes(t)) { setDraft(''); return }
    // 加到末尾：跟 chip 拖拽重排的心智一致（新东西落在底部，用户拖到想要的位置）
    onChange([...tags, t])
    setDraft('')
  }

  // chip 模式 input：draft 整体当一个 token；选中候选直接 addTag。
  const draftSuggest = useTagSuggest({
    value: draft,
    inputRef: draftInputRef,
    wholeAsToken: true,
    onPick: ({ suggestion }) => { addTag(suggestion.tag) },
  })

  // text 模式 textarea：根据 cursor 算 token range，替换为 `tag, ` 并保持光标。
  const textSuggest = useTagSuggest({
    value: textBuf,
    inputRef: textareaRef,
    onPick: ({ suggestion, range }) => {
      const before = textBuf.slice(0, range.start)
      const after = textBuf.slice(range.end)
      const cleanAfter = after.replace(/^[,，]\s*/, '')
      const next = `${before}${suggestion.tag}, ${cleanAfter}`
      setTextBuf(next)
      const newCursor = before.length + suggestion.tag.length + 2
      requestAnimationFrame(() => {
        const el = textareaRef.current
        if (el) { el.focus(); el.setSelectionRange(newCursor, newCursor) }
      })
    },
  })

  const removeTag = (t: string) => {
    onChange(tags.filter((x) => x !== t))
  }

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event
    if (!over || active.id === over.id) return
    const oldIndex = tags.indexOf(String(active.id))
    const newIndex = tags.indexOf(String(over.id))
    if (oldIndex < 0 || newIndex < 0) return
    onChange(arrayMove(tags, oldIndex, newIndex))
  }

  const commitText = () => {
    const next: string[] = []
    const seen = new Set<string>()
    for (const t of parseLine(textBuf)) {
      if (seen.has(t)) continue
      seen.add(t); next.push(t)
    }
    if (JSON.stringify(next) !== JSON.stringify(tags)) onChange(next)
  }

  const switchToText = () => {
    if (mode === 'text') return
    setTextBuf(tagsJoined) // sync immediately, no double-render via effect
    setMode('text')
  }

  const switchToChip = () => {
    if (mode === 'chip') return
    commitText()
    setMode('chip')
  }

  if (natural) {
    return (
      <div className="flex flex-col gap-2 flex-1 min-h-0">
        <textarea
          value={tags[0] ?? ''}
          onChange={(e) => onChange([e.target.value])}
          placeholder={t('tagEditor.naturalPlaceholder')}
          className="input input-mono text-sm flex-1 resize-none"
        />
        {onSave && (
          <button
            disabled={saving || !dirty}
            onClick={onSave}
            className={`self-start ${dirty ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm'}`}
          >
            {saving ? t('common.saving') : dirty ? t('common.save') : t('saveBar.saved')}
          </button>
        )}
      </div>
    )
  }

  // Caption block of the active-image card (mockup TagEdit): a caption line
  // with the tag count, the chips, and a dashed "+ tag" field at the end.
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, flex: 1, minHeight: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 'none' }}>
        <span className="ds-cap">{t('tagEditor.captionCount', { count: tags.length })}</span>
        <span style={{ flex: 1 }} />
        <div className="ds-seg" role="group" aria-label={t('tagEditor.modeLabel')}>
          <ModeBtn active={mode === 'chip'} onClick={switchToChip}>{t('tagEditor.modeChip')}</ModeBtn>
          <ModeBtn active={mode === 'text'} onClick={switchToText}>{t('tagEditor.modeText')}</ModeBtn>
        </div>
      </div>

      {mode === 'chip' ? (
        <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext items={tags} strategy={rectSortingStrategy}>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignContent: 'flex-start' }}>
                {tags.map((tag) => (
                  <SortableChip key={tag} id={tag} trigger={!!triggerWord && tag === triggerWord} onRemove={() => removeTag(tag)} />
                ))}
                <span style={{ position: 'relative', display: 'inline-flex' }}>
                  <input
                    ref={draftInputRef}
                    value={draft}
                    onChange={(e) => { setDraft(e.target.value); draftSuggest.notifyChange() }}
                    onKeyDown={(e) => {
                      if (draftSuggest.handleKeyDown(e)) return
                      if (e.key === 'Enter' || e.key === ',' || e.key === '，') {
                        e.preventDefault(); addTag(draft)
                      }
                    }}
                    onClick={() => draftSuggest.notifyClick()}
                    onFocus={() => draftSuggest.notifyFocus()}
                    onBlur={() => draftSuggest.notifyBlur()}
                    placeholder={t('tagEditor.addPlaceholder')}
                    aria-label={t('tagEditor.addPlaceholder')}
                    className="ds-chip-add ds-mono"
                    style={{ width: draft ? Math.max(90, draft.length * 7.2 + 26) : 72, outline: 'none', background: 'transparent', color: 'var(--ink)' }}
                  />
                  <TagSuggestList
                    open={draftSuggest.open}
                    suggestions={draftSuggest.suggestions}
                    activeIdx={draftSuggest.activeIdx}
                    onPick={(s) => draftSuggest.pickAt(draftSuggest.suggestions.indexOf(s))}
                    onHover={draftSuggest.setActiveIdx}
                    inputRef={draftInputRef}
                    cursor={draftSuggest.cursor}
                    positionDeps={[draft]}
                  />
                </span>
              </div>
            </SortableContext>
          </DndContext>
          {onSave && (
            <button
              type="button"
              disabled={saving || !dirty}
              onClick={onSave}
              className={dirty ? 'ds-btn-primary' : 'ds-ctl'}
              style={{ marginTop: 10 }}
            >
              {saving ? t('common.saving') : dirty ? t('common.save') : t('saveBar.saved')}
            </button>
          )}
        </div>
      ) : (
        <>
          <div style={{ position: 'relative', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <textarea
              ref={textareaRef}
              value={textBuf}
              onChange={(e) => { setTextBuf(e.target.value); textSuggest.notifyChange() }}
              onKeyDown={(e) => { textSuggest.handleKeyDown(e) }}
              onKeyUp={() => textSuggest.notifySelect()}
              onClick={() => textSuggest.notifyClick()}
              onFocus={() => textSuggest.notifyFocus()}
              onBlur={() => { textSuggest.notifyBlur(); commitText() }}
              placeholder={t('tagEditor.textPlaceholder')}
              className="ds-inp ds-mono"
              style={{ flex: 1, minHeight: 90, height: 'auto', padding: '8px 11px', resize: 'none', lineHeight: 1.55 }}
            />
            <TagSuggestList
              open={textSuggest.open}
              suggestions={textSuggest.suggestions}
              activeIdx={textSuggest.activeIdx}
              onPick={(s) => textSuggest.pickAt(textSuggest.suggestions.indexOf(s))}
              onHover={textSuggest.setActiveIdx}
              inputRef={textareaRef}
              cursor={textSuggest.cursor}
              positionDeps={[textBuf]}
            />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, flex: 'none' }}>
            <button type="button" onClick={commitText} className="ds-ctl ds-ghost" style={{ height: 28 }}>{t('tagEditor.sync')}</button>
            {onSave && (
              <button
                type="button"
                disabled={saving || !dirty}
                onClick={async () => { commitText(); await onSave() }}
                className={dirty ? 'ds-btn-primary' : 'ds-ctl'}
                style={{ height: 28 }}
              >
                {saving ? t('common.saving') : dirty ? t('common.save') : t('saveBar.saved')}
              </button>
            )}
          </div>
        </>
      )}
    </div>
  )
}

function ModeBtn({ active, onClick, children }: {
  active: boolean; onClick: () => void; children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`ds-seg-item${active ? ' ds-is-active' : ''}`}
      style={{ height: 22, padding: '0 9px', fontSize: 11 }}
    >
      {children}
    </button>
  )
}

/** 单个可拖拽 chip。dnd-kit 用 useSortable 给我们 setNodeRef / 拖拽 listeners /
 * transform / transition;CSS.Transform.toString 把 dnd-kit 算出的 (x,y,scale)
 * 翻译成 CSS transform 字符串。
 *
 * × 删除按钮要 stopPropagation onPointerDown —— 否则 6px 移动阈值过后 × 也成了
 * 拖拽起点,点 × 反而触发拖拽。
 */
function SortableChip({ id, trigger, onRemove }: { id: string; trigger?: boolean; onRemove: () => void }) {
  const { t } = useTranslation()
  const {
    attributes, listeners, setNodeRef, transform, transition, isDragging,
  } = useSortable({ id })
  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
    zIndex: isDragging ? 1 : undefined,
    cursor: isDragging ? 'grabbing' : 'grab',
    touchAction: 'none',
    userSelect: 'none',
    ...(trigger ? { background: 'var(--green-soft)', color: 'var(--green-text)' } : null),
  }
  return (
    <span
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className="ds-chip"
    >
      <TranslatedTag tag={id} />
      <button
        type="button"
        onPointerDown={(e) => e.stopPropagation()}
        onClick={onRemove}
        aria-label={t('tagEditor.deleteTag', { tag: id })}
        className="ds-chip-x"
      >
        <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>
      </button>
    </span>
  )
}
