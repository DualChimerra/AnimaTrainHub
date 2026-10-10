import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { sampleHover } from '../lib/sampleHover'
import SampleHoverHost from './SampleHoverHost'
import SampleThumbnail from './SampleThumbnail'

afterEach(() => { sampleHover.close(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

it('keeps one preview during fast switching and ignores a late leave from the previous thumbnail', () => {
  class MousePointerEvent extends MouseEvent {
    pointerType: string
    constructor(type: string, init: PointerEventInit = {}) { super(type, init); this.pointerType = init.pointerType ?? 'mouse' }
  }
  vi.stubGlobal('PointerEvent', MousePointerEvent)
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
    x: 100, y: 100, left: 100, top: 100, right: 184, bottom: 212, width: 84, height: 112, toJSON: () => ({}),
  })
  const view = render(<>
    <SampleThumbnail src="/first-thumb.png" previewSrc="/first.png" caption="First" />
    <SampleThumbnail src="/second-thumb.png" previewSrc="/second.png" caption="Second" />
    <SampleHoverHost />
  </>)
  const first = screen.getByRole('button', { name: 'First' })
  const second = screen.getByRole('button', { name: 'Second' })
  fireEvent.pointerEnter(first, { pointerType: 'mouse', clientX: 100, clientY: 150 })
  expect(screen.getAllByRole('tooltip')).toHaveLength(1)
  const firstImage = screen.getByRole('tooltip').querySelector('img')
  fireEvent.pointerMove(first, { pointerType: 'mouse', clientX: 110, clientY: 150 })
  fireEvent.pointerEnter(second, { pointerType: 'mouse', clientX: 200, clientY: 150 })
  fireEvent.pointerLeave(first, { pointerType: 'mouse' })
  expect(screen.getAllByRole('tooltip')).toHaveLength(1)
  expect(screen.getByRole('tooltip')).toHaveAccessibleName('Second')
  expect(screen.getByRole('tooltip').querySelector('img')).not.toBe(firstImage)
  expect(screen.getByRole('tooltip').querySelector('img')).toHaveAttribute('src', '/second.png')
  fireEvent.pointerMove(document.body, { pointerType: 'mouse' })
  expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  fireEvent.pointerEnter(second, { pointerType: 'mouse' })
  fireEvent.resize(window)
  expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  fireEvent.pointerEnter(second, { pointerType: 'mouse' })
  view.unmount()
  expect(sampleHover.snapshot()).toBeNull()
})
