import type { SampleHover } from '../components/SampleHoverPreview'

let current: SampleHover | null = null
let frame: number | null = null
let pendingMove: { anchor: HTMLElement; x: number; y: number } | null = null
const listeners = new Set<() => void>()
const publish = () => { for (const listener of listeners) listener() }

export const sampleHover = {
  subscribe(listener: () => void) { listeners.add(listener); return () => { listeners.delete(listener) } },
  snapshot: () => current,
  show(sample: SampleHover) {
    if (frame != null) cancelAnimationFrame(frame)
    frame = null
    pendingMove = null
    current = sample
    publish()
  },
  move(anchor: HTMLElement, x: number, y: number) {
    if (current?.anchor !== anchor) return
    pendingMove = { anchor, x, y }
    if (frame != null) return
    frame = requestAnimationFrame(() => {
      frame = null
      const move = pendingMove
      pendingMove = null
      if (!move || current?.anchor !== move.anchor) return
      current = { ...current, x: move.x, y: move.y }
      publish()
    })
  },
  close(anchor?: HTMLElement) {
    if (anchor && current?.anchor !== anchor) return
    if (frame != null) cancelAnimationFrame(frame)
    frame = null
    pendingMove = null
    if (!current) return
    current = null
    publish()
  },
}
