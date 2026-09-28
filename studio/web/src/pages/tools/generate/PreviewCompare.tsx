import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type MonitorState } from '../../../api/client'
import FullscreenViewer from './FullscreenViewer'
import { axisLabel, formatAxisValue, type XYAxisDraft } from './xy'

type Sample = NonNullable<MonitorState['samples']>[number]

function labelOf(s: Sample, xDraft: XYAxisDraft, yDraft: XYAxisDraft | null): string {
  if (!s.xy) return ''
  const x = `${axisLabel(xDraft.axis)}=${formatAxisValue(xDraft.axis, String(s.xy.xv ?? ''))}`
  if (yDraft && s.xy.yv != null) {
    return `${x}, ${axisLabel(yDraft.axis)}=${formatAxisValue(yDraft.axis, String(s.xy.yv))}`
  }
  return x
}

/** XY mode's internal sub-view: pick 2 cells and compare them side by side.
 *
 * - The "<- Back to grid" header clears selectedIndices and returns to the grid view
 * - Double-clicking a single image opens it fullscreen (FullscreenViewer, closes on ESC)
 */
export default function PreviewCompare({
  samples, taskId, selectedIndices, xDraft, yDraft, onBack,
}: {
  samples: NonNullable<MonitorState['samples']>
  taskId: number
  selectedIndices: [number, number]
  xDraft: XYAxisDraft
  yDraft: XYAxisDraft | null
  onBack: () => void
}) {
  const { t } = useTranslation()
  const [aIdx, bIdx] = selectedIndices
  const sampleA = samples[aIdx]
  const sampleB = samples[bIdx]
  // P1-G: track fullscreen by side so left/right arrows can switch between A and B (not just view one passively)
  const [fullscreenSide, setFullscreenSide] = useState<'A' | 'B' | null>(null)

  if (!sampleA || !sampleB) {
    return (
      <div className="flex flex-col gap-3 flex-1 min-h-0">
        <button onClick={onBack} className="self-start btn btn-ghost btn-sm text-xs">
          {t('generate.backToGrid')}
        </button>
        <div className="flex-1 grid place-items-center rounded-md border border-subtle bg-sunken text-fg-tertiary text-sm">
          {t('generate.selectedSamplesUnavailable')}
        </div>
      </div>
    )
  }

  const fnA = sampleA.path.split(/[\\/]/).pop() ?? sampleA.path
  const fnB = sampleB.path.split(/[\\/]/).pop() ?? sampleB.path
  const urlA = api.generateSampleUrl(taskId, fnA)
  const urlB = api.generateSampleUrl(taskId, fnB)
  const captionA = labelOf(sampleA, xDraft, yDraft) || fnA
  const captionB = labelOf(sampleB, xDraft, yDraft) || fnB

  return (
    <div className="flex flex-col gap-3 flex-1 min-h-0">
      <div className="flex items-center justify-between shrink-0">
        <button
          onClick={onBack}
          className="btn btn-ghost btn-sm text-xs text-fg-secondary"
        >
          {t('generate.backToGridClear')}
        </button>
        <span className="text-2xs text-fg-tertiary">
          {t('generate.doubleClickFullscreen')}
        </span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 flex-1 min-h-0">
        {[
          { sample: sampleA, fn: fnA, url: urlA, caption: captionA, side: 'A' as const },
          { sample: sampleB, fn: fnB, url: urlB, caption: captionB, side: 'B' as const },
        ].map(({ fn, url, caption, side }) => (
          <div key={side} className="flex flex-col gap-2 min-h-0">
            <div className="flex items-center gap-2 text-2xs shrink-0">
              <span className="badge badge-info shrink-0">{side}</span>
              <span className="font-mono text-fg-tertiary truncate">{caption}</span>
            </div>
            <button
              onDoubleClick={() => setFullscreenSide(side)}
              className="flex-1 min-h-0 flex items-center justify-center bg-sunken rounded-md border border-subtle p-0 cursor-zoom-in"
              title={t('generate.doubleClickFullscreenTitle')}
            >
              <img
                src={url}
                className="rounded-md object-contain"
                style={{ maxWidth: '100%', maxHeight: '100%' }}
                alt={fn}
              />
            </button>
          </div>
        ))}
      </div>
      {fullscreenSide && (
        <FullscreenViewer
          src={fullscreenSide === 'A' ? urlA : urlB}
          caption={fullscreenSide === 'A' ? captionA : captionB}
          onClose={() => setFullscreenSide(null)}
          // Both A and B exist (guaranteed by the early-return branch) -> both arrow directions are valid
          hasPrev
          hasNext
          onPrev={() => setFullscreenSide((s) => (s === 'B' ? 'A' : 'B'))}
          onNext={() => setFullscreenSide((s) => (s === 'A' ? 'B' : 'A'))}
          shortcutHint={t('generate.compareShortcutHint', { direction: fullscreenSide === 'A' ? 'A→B' : 'B→A' })}
        />
      )}
    </div>
  )
}
