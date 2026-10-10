import { useEffect } from 'react'
import { createPortal } from 'react-dom'

export interface SampleHover {
  src: string
  caption: string
  x: number
  y: number
  width: number
  height: number
  anchor: HTMLElement
}

/** A mouse preview stays beside the pointer and never intercepts it. */
export default function SampleHoverPreview({ sample, onClose }: { sample: SampleHover; onClose: () => void }) {
  useEffect(() => {
    const onBlur = () => onClose()
    const onPointer = (e: PointerEvent) => {
      if (!(e.target instanceof Node) || !sample.anchor.contains(e.target)) onClose()
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    const onScroll = (e: Event) => {
      if (e.target === document || (e.target instanceof Element && e.target.contains(sample.anchor))) onClose()
    }
    window.addEventListener('scroll', onScroll, true)
    window.addEventListener('resize', onClose)
    window.addEventListener('keydown', onKey)
    window.addEventListener('blur', onBlur)
    window.addEventListener('pointermove', onPointer)
    return () => {
      window.removeEventListener('scroll', onScroll, true)
      window.removeEventListener('resize', onClose)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('blur', onBlur)
      window.removeEventListener('pointermove', onPointer)
    }
  }, [onClose, sample.anchor])

  const margin = 12
  const scale = Math.min(Math.max(5, 420 / sample.width), (window.innerWidth - margin * 2) / sample.width, (window.innerHeight - margin * 2) / sample.height)
  const width = sample.width * scale
  const height = sample.height * scale
  const right = sample.x + 18
  const left = Math.max(margin, Math.min(right + width <= window.innerWidth - margin ? right : sample.x - width - 18,
    window.innerWidth - width - margin))
  const top = Math.max(margin, Math.min(sample.y - height / 2, window.innerHeight - height - margin))

  return createPortal(
    <div className="ds-sample-hover" role="tooltip" aria-label={sample.caption} style={{ left, top, width, height }}>
      <img key={sample.src} src={sample.src} alt="" decoding="async" />
      <span>{sample.caption}</span>
    </div>,
    document.body,
  )
}
