import { useSyncExternalStore } from 'react'
import { sampleHover } from '../lib/sampleHover'
import SampleHoverPreview from './SampleHoverPreview'

const closePreview = () => sampleHover.close()

/** All thumbnails share one overlay, including during fast pointer movement. */
export default function SampleHoverHost() {
  const sample = useSyncExternalStore(sampleHover.subscribe, sampleHover.snapshot, () => null)
  return sample ? <SampleHoverPreview sample={sample} onClose={closePreview} /> : null
}
