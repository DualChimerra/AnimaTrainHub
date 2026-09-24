/** Optional steps carry a trailing parenthetical in every locale:
 *  "Препроцесс (необязательно)", "Preprocess (optional)", "预处理（可选）".
 *  The sidebar and breadcrumbs show the short form; the full label stays
 *  available as a tooltip. */
const OPTIONAL_SUFFIX = /^(.*?)\s*[(（][^()（）]+[)）]\s*$/

export function splitOptional(label: string): { short: string; optional: boolean } {
  const m = label.match(OPTIONAL_SUFFIX)
  return m ? { short: m[1], optional: true } : { short: label, optional: false }
}
