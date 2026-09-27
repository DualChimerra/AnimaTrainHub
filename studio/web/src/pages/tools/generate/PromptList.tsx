import { useRef } from 'react'
import { useTranslation } from 'react-i18next'

import { TagSuggestList } from '../../../components/tagSuggest/TagSuggestList'
import { useTagSuggest } from '../../../components/tagSuggest/useTagSuggest'
import { useAutoGrowTextarea } from '../../../lib/useAutoGrowTextarea'
import { useTokenCount } from '../../../lib/useTokenCount'

/** Positive prompt input.
 *
 * Previously supported cycling through multiple prompts ("+ Add prompt"); per a product
 * decision to hide that cycling feature from the frontend, this was simplified to a single
 * textarea. The backend still takes list[str], so it's still wrapped into an array when the request is sent.
 *
 * Wired up with tag autocomplete: the token under the cursor triggers suggestions; up/down/Tab/Enter picks and inserts one.
 */
export default function PromptList({ prompts, onChange, modelFamily = 'anima' }: {
  prompts: string[]
  onChange: (p: string[]) => void
  /** The family used for token counting (selects the matching tokenizer); defaults to anima if omitted. */
  modelFamily?: string
}) {
  const { t } = useTranslation()
  const taRef = useRef<HTMLTextAreaElement>(null)
  // Currently only the first prompt is shown; editing it syncs back as [value]
  const value = prompts[0] ?? ''
  const suggest = useTagSuggest({
    value,
    inputRef: taRef,
    onPick: ({ suggestion, range }) => {
      const before = value.slice(0, range.start)
      const after = value.slice(range.end)
      const cleanAfter = after.replace(/^[,]\s*/, '')
      const next = `${before}${suggestion.tag}, ${cleanAfter}`
      onChange([next])
      const newCursor = before.length + suggestion.tag.length + 2
      requestAnimationFrame(() => {
        const el = taRef.current
        if (el) { el.focus(); el.setSelectionRange(newCursor, newCursor) }
      })
    },
  })
  useAutoGrowTextarea(taRef, value)
  const tokenCount = useTokenCount(value, modelFamily)
  return (
    <div className="relative">
      <textarea
        ref={taRef}
        className="ds-inp"
        style={{ height: 'auto', minHeight: 84, padding: '9px 11px 20px', fontSize: 12, lineHeight: 1.55, resize: 'none', overflow: 'hidden' }}
        rows={4}
        value={value}
        onChange={(e) => { onChange([e.target.value]); suggest.notifyChange() }}
        onKeyDown={(e) => { suggest.handleKeyDown(e) }}
        onKeyUp={() => suggest.notifySelect()}
        onClick={() => suggest.notifyClick()}
        onFocus={() => suggest.notifyFocus()}
        onBlur={() => suggest.notifyBlur()}
        placeholder={t('generate.positivePlaceholder')}
      />
      {tokenCount != null && (
        <span className="absolute bottom-1.5 right-2 text-2xs text-fg-tertiary pointer-events-none select-none">
          {t('generate.tokenCount', { n: tokenCount })}
        </span>
      )}
      <TagSuggestList
        open={suggest.open}
        suggestions={suggest.suggestions}
        activeIdx={suggest.activeIdx}
        onPick={(s) => suggest.pickAt(suggest.suggestions.indexOf(s))}
        onHover={suggest.setActiveIdx}
        inputRef={taRef}
        cursor={suggest.cursor}
        positionDeps={[value]}
      />
    </div>
  )
}
