import { useEffect, useRef, type CSSProperties, type ReactNode } from 'react'
import { sampleHover } from '../lib/sampleHover'

/** Shared training thumbnail: portrait image, top caption and mouse preview. */
export default function SampleThumbnail({ src, previewSrc, caption, latest = false, selected = false, onClick, style, title, className = '', disabled = false, children }: {
  src: string
  previewSrc: string
  caption: string
  latest?: boolean
  selected?: boolean
  onClick?: () => void
  style?: CSSProperties
  title?: string
  className?: string
  disabled?: boolean
  children?: ReactNode
}) {
  const anchorRef = useRef<HTMLButtonElement | null>(null)
  const compact = typeof style?.width === 'number' && style.width < 64
  const label = compact ? caption.replace(/^(\S{2})\S*\s+(\d)/, '$1. $2') : caption
  useEffect(() => {
    const anchor = anchorRef.current
    return () => { if (anchor) sampleHover.close(anchor) }
  }, [])
  return (
    <button type="button"
      ref={anchorRef}
      className={`ds-thumb ds-training-sample${latest ? ' ds-latest' : ''}${selected ? ' ds-sel' : ''}${compact ? ' ds-compact-sample' : ''} ${className}`}
      disabled={disabled}
      style={style} aria-label={title || caption}
      onPointerEnter={(e) => {
        if (e.pointerType !== 'mouse') return
        const rect = e.currentTarget.getBoundingClientRect()
        sampleHover.show({ src: previewSrc, caption, x: e.clientX, y: e.clientY,
          width: rect.width, height: rect.height, anchor: e.currentTarget })
      }}
      onPointerMove={(e) => { if (e.pointerType === 'mouse') sampleHover.move(e.currentTarget, e.clientX, e.clientY) }}
      onPointerLeave={(e) => sampleHover.close(e.currentTarget)}
      onPointerDown={() => sampleHover.close()}
      onClick={() => { sampleHover.close(); onClick?.() }}>
      <img src={src} alt="" loading="lazy" decoding="async" />
      <span className="ds-sample-label">{label}</span>
      {children}
    </button>
  )
}
