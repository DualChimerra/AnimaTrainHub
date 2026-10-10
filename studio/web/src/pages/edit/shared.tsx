/** Edit LoRA pieces shared by the tools list, the examples page and the project overview. */
import { useTranslation } from 'react-i18next'
import { api, type EditPair, type EditRecipe, type EditSlot } from '../../api/client'

export const RECIPE_ORDER: EditRecipe[] = ['text', 'image']

/** What each tool type asks for in an example. */
export const EDIT_RECIPES: Record<EditRecipe, { inputs: 1 | 2; text: 'required' | 'optional' }> = {
  text:  { inputs: 1, text: 'required' },
  image: { inputs: 2, text: 'optional' },
}

/** Picture slots of an example, in reading order. */
export const recipeSlots = (recipe: EditRecipe): EditSlot[] =>
  recipe === 'image' ? ['source', 'source2', 'target'] : ['source', 'target']

export const EditIcon = {
  up: (s = 14) => (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 16V4m0 0-4 4m4-4 4 4M4 20h16" />
    </svg>
  ),
  plus: (s = 14) => (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden><path d="M12 5v14M5 12h14" /></svg>
  ),
  arrow: (s = 13) => (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M5 12h13m-5-6 6 6-6 6" /></svg>
  ),
  text: (s = 14) => (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden><path d="M4 6h16M4 12h10M4 18h13" /></svg>
  ),
  x: (s = 11) => (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" aria-hidden><path d="M18 6 6 18M6 6l12 12" /></svg>
  ),
  prev: (s = 14) => (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M15 6l-6 6 6 6" /></svg>
  ),
  next: (s = 14) => (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M9 6l6 6-6 6" /></svg>
  ),
}

/** The two tool types as plain choices (new tool dialog). */
export function RecipeChoice({ value, onChange }: { value: EditRecipe; onChange: (r: EditRecipe) => void }) {
  const { t } = useTranslation()
  return (
    <div className="ds-optcards" role="radiogroup">
      {RECIPE_ORDER.map((r) => (
        <button key={r} type="button" role="radio" aria-checked={value === r}
          className={`ds-optcard${value === r ? ' ds-is-on' : ''}`} onClick={() => onChange(r)}>
          <span className="ds-optcard-txt">
            <span className="ds-optcard-name">{t(`editLora.recipe_${r}`)}</span>
            <span className="ds-optcard-desc">{t(`editLora.recipeDesc_${r}`)}</span>
          </span>
        </button>
      ))}
    </div>
  )
}

const BUCKET: Record<EditSlot, 'train' | 'control' | 'control2'> = { target: 'train', source: 'control', source2: 'control2' }

/** Thumbnail URL of one image of an example (null when that slot is empty). */
export function pairImageUrl(pid: number, vid: number, pair: EditPair, slot: EditSlot, size = 384, bust = 0): string | null {
  const name = slot === 'target' ? pair.target : slot === 'source' ? pair.source : pair.source2
  if (!name) return null
  const folder = slot === 'target' ? pair.folder : slot === 'source' ? pair.source_folder : pair.source2_folder
  return `${api.versionThumbUrl(pid, vid, BUCKET[slot], name, folder || undefined, size)}&v=${Math.round(pair.mtime)}-${bust}`
}

/** The picture that stands for an example in a grid: its result, else its input. */
export function pairCover(pid: number, vid: number, pair: EditPair, size = 256, bust = 0): string | null {
  return pairImageUrl(pid, vid, pair, 'target', size, bust) ?? pairImageUrl(pid, vid, pair, 'source', size, bust)
}
