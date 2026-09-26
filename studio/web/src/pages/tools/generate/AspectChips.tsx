/** Aspect-ratio preset cards (8 common ratios + a Swap for landscape/portrait).
 *
 * Each card: a glyph (a small rectangle drawn to the ratio) + the ratio number + name + WxH.
 * Selected state: accent-colored outline + the ratio number in accent color.
 * Laid out grid-cols-4 over two rows, 4x2=8 cards; fits within the 420px sidebar.
 *
 * Swap is its own button: physically swaps W<->H, and if that hits a preset the aspect switches to match
 * (an exact w/h match finds the current chip); falls back to aspect='custom' when nothing matches.
 */

export interface AspectPreset {
  /** Unique key, also the ratio label shown in the large font. */
  ratio: string
  /** 'Square' / 'Landscape' / 'Portrait' */
  kind: 'Square' | 'Landscape' | 'Portrait'
  w: number
  h: number
}

export const ASPECT_PRESETS: AspectPreset[] = [
  { ratio: '3:2',   kind: 'Landscape', w: 1254, h: 836  },
  { ratio: '1:1',   kind: 'Square',    w: 1024, h: 1024 },
  { ratio: '4:5',   kind: 'Portrait',  w: 915,  h: 1144 },
  { ratio: '7:9',   kind: 'Portrait',  w: 896,  h: 1152 },
  { ratio: '3:4',   kind: 'Portrait',  w: 768,  h: 1024 },
  { ratio: '2:3',   kind: 'Portrait',  w: 832,  h: 1216 },
  { ratio: '9:16',  kind: 'Portrait',  w: 768,  h: 1344 },
  { ratio: '5:12',  kind: 'Portrait',  w: 640,  h: 1536 },
]

export type AspectName = string  // uses "ratio" as the name; custom = 'custom'

/** Reverse-looks-up the matching preset ratio for a given w/h; returns 'custom' when nothing matches. */
export function aspectFromDimensions(w: number, h: number): AspectName {
  for (const p of ASPECT_PRESETS) {
    if (p.w === w && p.h === h) return p.ratio
  }
  return 'custom'
}

/** Aspect presets as a row of pills; the active one is dark, the size shows on hover. */
export default function AspectChips({
  aspect, onPick,
}: {
  aspect: AspectName
  onPick: (ratio: AspectName, w?: number, h?: number) => void
}) {
  return (
    <div className="ds-pills" style={{ flexWrap: 'wrap' }}>
      {ASPECT_PRESETS.map((p) => (
        <button
          key={p.ratio}
          type="button"
          className={`ds-pill${aspect === p.ratio ? ' ds-is-active' : ''}`}
          style={{ height: 26, padding: '0 10px', fontFamily: 'var(--mono)', fontSize: 11.5 }}
          title={`${p.kind} · ${p.w}×${p.h}`}
          aria-pressed={aspect === p.ratio}
          onClick={() => onPick(p.ratio, p.w, p.h)}
        >
          {p.ratio}
        </button>
      ))}
    </div>
  )
}
