import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type ModelsCatalog } from '../api/client'

/** One option in the base model dropdown: value = official variant key or a local custom absolute path. */
export interface BaseModelOption {
  value: string
  label: string
  /** Purpose declaration for an official variant (krea2: raw=training / turbo=inference);
   *  custom weights carry no such metadata. Pages can use it to apply distilled inference defaults. */
  purpose?: 'training' | 'inference'
}

/** Model families that support base-model selection. Catalog section key = `${family}_main`. */
export type BaseModelFamily = 'anima' | 'krea2'

interface FamilyMainSection {
  variants: Array<{
    variant: string
    exists: boolean
    /** Starting with krea2, variants carry a purpose declaration (raw=training / turbo=inference). */
    purpose?: 'training' | 'inference'
  }>
  custom: Array<{ path: string; name: string; exists: boolean }>
  selected: string
}

function mainSection(
  catalog: ModelsCatalog | null, family: BaseModelFamily,
): FamilyMainSection | null {
  if (!catalog) return null
  const section = family === 'krea2' ? catalog.krea2_main : catalog.anima_main
  return (section as FamilyMainSection | undefined) ?? null
}

/** Pulls the list of "downloaded main models for the given family" from the model
 *  catalog, plus the Settings page's currently selected value.
 *
 *  options only includes official variants that exist on disk plus registered
 *  local customs (ones not downloaded are omitted, to avoid picking a variant
 *  whose weights can't be fetched); defaultValue = the base model currently
 *  selected on the Settings page for that family, used as the dropdown's
 *  initial / fallback value. krea2 variants carry a purpose badge (raw=training
 *  base model / turbo=inference base model, both are selectable -- A1 does not
 *  add a whitelist). */
export function useBaseModelOptions(family: BaseModelFamily = 'anima'): {
  options: BaseModelOption[]
  defaultValue: string | null
  loaded: boolean
} {
  const { t } = useTranslation()
  const [catalog, setCatalog] = useState<ModelsCatalog | null>(null)
  useEffect(() => {
    let alive = true
    api.getModelsCatalog().then((c) => { if (alive) setCatalog(c) }).catch(() => {})
    return () => { alive = false }
  }, [])
  const options = useMemo<BaseModelOption[]>(() => {
    const section = mainSection(catalog, family)
    if (!section) return []
    const out: BaseModelOption[] = []
    for (const v of section.variants) {
      if (!v.exists) continue
      const badge = v.purpose
        ? ` · ${t(`baseModel.purpose.${v.purpose}`)}`
        : ''
      out.push({
        value: v.variant,
        label: `${v.variant}${badge}`,
        purpose: v.purpose,
      })
    }
    for (const c of section.custom) {
      if (c.exists) out.push({ value: c.path, label: c.name })
    }
    return out
  }, [catalog, family, t])
  return {
    options,
    defaultValue: mainSection(catalog, family)?.selected ?? null,
    loaded: catalog !== null,
  }
}

/** krea2 TE option state: whether the fp8 directory is ready (weights + config
 *  downloaded, which decides whether fp8 is selectable in the test page
 *  dropdown) + the default variant selected in the download center (dropdown
 *  initial value). */
/** The krea2 TE currently selected on the Settings page. `'custom'` = a locally
 *  registered encoder directory is selected -- in that case the generate
 *  request omits text_encoder and lets the server resolve it from selected_te
 *  (the request field can only carry an official variant key, see the Literal
 *  in api/schemas/generate.py). */
export type Krea2TeSelection = 'bf16' | 'fp8' | 'custom'

export function useKrea2TeOptions(): {
  fp8Ready: boolean; selected: Krea2TeSelection
} {
  const [state, setState] = useState<{
    fp8Ready: boolean; selected: Krea2TeSelection
  }>({
    fp8Ready: false, selected: 'bf16',
  })
  useEffect(() => {
    let alive = true
    api.getModelsCatalog().then((c) => {
      if (!alive) return
      const files = c.krea2_text_encoder_fp8?.files ?? []
      const existing = new Set(files.filter((f) => f.exists).map((f) => f.name))
      const fp8Ready = (
        files.some((f) => f.name.endsWith('.safetensors') && f.exists)
        && existing.has('config.json')
        && existing.has('tokenizer.json')
      )
      const sel = c.krea2_text_encoder?.selected
      const selected: Krea2TeSelection = (
        sel === 'fp8' ? 'fp8' : sel === 'bf16' || !sel ? 'bf16' : 'custom'
      )
      setState({ fp8Ready, selected })
    }).catch(() => {})
    return () => { alive = false }
  }, [])
  return state
}

function basename(p: string): string {
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'))
  return i >= 0 ? p.slice(i + 1) : p
}

/** Base model dropdown. Controlled: `value` is a "temporary override for this
 *  instance" (null = follow the Settings page default).
 *
 *  `family` decides which family's main models to list (defaults to anima, for
 *  backward compatibility with existing callers). `className` lets each page
 *  align the select's style with its other inputs (the regularization set page
 *  uses "select input", the test page uses "input text-xs w-full"). */
export default function BaseModelSelect({
  value, onChange, family = 'anima', className = 'select input', style, ariaLabel,
}: {
  value: string | null
  onChange: (v: string) => void
  family?: BaseModelFamily
  className?: string
  /** Inline style pass-through (the regularization set page uses it to align
   *  visuals with the training config page's controls). */
  style?: React.CSSProperties
  ariaLabel?: string
}) {
  const { options, defaultValue } = useBaseModelOptions(family)
  // Effective value: explicit override wins, otherwise follow the Settings page default.
  const effective = value ?? defaultValue ?? ''
  // When effective isn't in options (e.g. the variant selected on the Settings
  // page hasn't been downloaded yet), add an extra entry so the select doesn't
  // fall back to the first item and show something other than what's actually in effect.
  const missing = effective !== '' && !options.some((o) => o.value === effective)
  return (
    <select
      className={className}
      style={style}
      value={effective}
      onChange={(e) => onChange(e.target.value)}
      aria-label={ariaLabel}
    >
      {missing && <option value={effective}>{basename(effective)}</option>}
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  )
}
