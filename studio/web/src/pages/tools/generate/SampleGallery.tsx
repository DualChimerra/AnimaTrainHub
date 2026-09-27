import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api } from '../../../api/client'
import ZoomableImage from '../../../components/ZoomableImage'

/** Large preview of the test task's most recent image.
 *
 * Previously also rendered a thumbnail row; per user feedback ("we already have a dedicated
 * history rail, SampleGallery's thumbnail column is redundant") that was removed. This
 * component now only shows the **latest** image, filling the container (with built-in
 * zoom/pan via useZoomPan); the empty state is handled by the caller.
 */
export default function SampleGallery({ samples, taskId }: {
  samples: Array<{ path: string; step?: number }>
  taskId: number
}) {
  const { t } = useTranslation()
  const [errored, setErrored] = useState(false)

  const cur = samples[samples.length - 1]
  const filename = cur ? (cur.path.split(/[\\/]/).pop() ?? cur.path) : ''
  const fullUrl = cur ? api.generateSampleUrl(taskId, filename) : ''
  // Reset error when the image changes (the original implementation cleared it via onLoad;
  // ZoomableImage's internal onLoad is used to read the natural size, so this was switched to
  // an src-driven reset instead -- the retry-on-failure semantics are unchanged)
  useEffect(() => { setErrored(false) }, [fullUrl])

  if (!samples.length) return null

  if (errored) {
    return (
      <div className="flex-1 grid place-items-center rounded-md border border-subtle bg-sunken text-fg-tertiary text-sm">
        {t('generate.imageCachePending')}
      </div>
    )
  }

  return (
    // ZoomableImage already has its own viewport border/background + readout bar; the old
    // filename footer ("single image N.png") duplicated the readout, so it was removed
    <div className="flex-1 min-h-0 w-full">
      <ZoomableImage
        key={fullUrl}
        src={fullUrl}
        alt={filename}
        onError={() => setErrored(true)}
      />
    </div>
  )
}
