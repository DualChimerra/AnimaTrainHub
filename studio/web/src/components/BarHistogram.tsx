/** Generic horizontal bar histogram -- reused by:
 *  - Preprocess pixel distribution (PixelHist)
 *  - PreprocessCrop aspect ratio distribution (ArHist)
 *  - The same stats in the Overview detail tab
 *
 *  Rendering: each row is a grid of 96px label / 1fr progress bar / 30px number.
 *  The bar uses the global `.ar-bar` / `.ar-bar-fill` CSS classes.
 */
interface HistBin {
  /** React key (optional, falls back to label) */
  key?: string
  label: string
  n: number
}

interface Props {
  bins: HistBin[]
  /** Placeholder text shown when bins is empty */
  emptyHint?: string
}

export default function BarHistogram({ bins, emptyHint }: Props) {
  if (bins.length === 0) {
    return emptyHint ? (
      <p className="text-xs text-fg-tertiary italic m-0">{emptyHint}</p>
    ) : null
  }
  const max = Math.max(1, ...bins.map((b) => b.n))
  return (
    <div className="flex flex-col gap-1">
      {bins.map((b) => (
        <div
          key={b.key ?? b.label}
          className="grid items-center gap-1.5 text-[11px]"
          style={{ gridTemplateColumns: '96px 1fr 30px' }}
        >
          <span className="text-fg-tertiary font-mono">{b.label}</span>
          <div className="ar-bar">
            <div className="ar-bar-fill" style={{ width: `${(b.n / max) * 100}%` }} />
          </div>
          <span className="font-mono text-right text-fg-secondary">{b.n}</span>
        </div>
      ))}
    </div>
  )
}
