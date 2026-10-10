import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import SampleHoverPreview from './SampleHoverPreview'

it('shows a fivefold preview and keeps it open while live logs scroll', () => {
  const anchor = document.createElement('button')
  const close = vi.fn()
  const view = render(<div>
    <div ref={(el) => el?.appendChild(anchor)} />
    <div data-testid="logs" />
    <SampleHoverPreview sample={{ src: '/sample.png', caption: 'Epoch 1', x: 100, y: 250, width: 84, height: 112, anchor }} onClose={close} />
  </div>)
  const preview = screen.getByRole('tooltip', { name: 'Epoch 1' })
  expect(preview).toHaveStyle({ width: '420px', height: '560px' })
  fireEvent.scroll(screen.getByTestId('logs'))
  expect(close).not.toHaveBeenCalled()
  fireEvent.scroll(anchor.parentElement!)
  expect(close).toHaveBeenCalledOnce()
  view.unmount()
  fireEvent.resize(window)
  expect(close).toHaveBeenCalledOnce()
})

it('keeps a preview inside the screen near the bottom right corner', () => {
  const close = vi.fn()
  render(<SampleHoverPreview sample={{ src: '/sample.png', caption: 'Epoch 2',
    x: window.innerWidth - 20, y: window.innerHeight - 20, width: 84, height: 112,
    anchor: document.createElement('button'),
  }} onClose={close} />)
  const preview = screen.getByRole('tooltip')
  expect(parseFloat(preview.style.left)).toBeGreaterThanOrEqual(12)
  expect(parseFloat(preview.style.left) + parseFloat(preview.style.width)).toBeLessThanOrEqual(window.innerWidth - 12)
  expect(parseFloat(preview.style.top) + parseFloat(preview.style.height)).toBeLessThanOrEqual(window.innerHeight - 12)
  fireEvent.keyDown(window, { key: 'Escape' })
  expect(close).toHaveBeenCalledOnce()
})
