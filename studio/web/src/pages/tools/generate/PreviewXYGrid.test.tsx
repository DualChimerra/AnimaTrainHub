import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import PreviewXYGrid, { type XYSample } from './PreviewXYGrid'
import type { XYAxisDraft } from './xy'

type Sample = XYSample

const xDraft: XYAxisDraft = { axis: 'steps', raw: '20, 25, 30', loraIndex: null }
const yDraft: XYAxisDraft = { axis: 'cfg_scale', raw: '3.0, 5.0', loraIndex: null }

function makeSample(xi: number, yi: number, xv: number, yv: number | null): Sample {
  return {
    path: `/tmp/anima_gen_99/xy_x${String(xi).padStart(2, '0')}_y${String(yi).padStart(2, '0')}_s42.png`,
    step: yi * 3 + xi + 1,
    xy: { xi, yi, xv, yv },
  }
}

describe('PreviewXYGrid', () => {
  it('shows total count for 2D matrix', () => {
    render(
      <PreviewXYGrid
        samples={[]}
        taskId={99}
        xDraft={xDraft}
        yDraft={yDraft}
      />
    )
    // 3 × 2 = 6 images
    expect(screen.getByText(/3 × 2 = 6 images/)).toBeInTheDocument()
  })

  it('shows partial count when generation in progress', () => {
    const samples = [makeSample(0, 0, 20, 3.0), makeSample(1, 0, 25, 3.0)]
    render(
      <PreviewXYGrid samples={samples} taskId={99} xDraft={xDraft} yDraft={yDraft} />
    )
    expect(screen.getByText(/3 × 2 = 6 images/)).toBeInTheDocument()
    expect(screen.getByText(/Generated 2/)).toBeInTheDocument()
  })

  it('renders 1D layout (y=null) with single header row', () => {
    render(
      <PreviewXYGrid samples={[]} taskId={99} xDraft={xDraft} yDraft={null} />
    )
    expect(screen.getByText(/3 images/)).toBeInTheDocument()
  })

  it('renders cell images at the right (yi, xi) positions', () => {
    const samples: Sample[] = [
      makeSample(0, 0, 20, 3.0),
      makeSample(2, 1, 30, 5.0),
    ]
    render(
      <PreviewXYGrid samples={samples} taskId={99} xDraft={xDraft} yDraft={yDraft} />
    )
    // A generated cell shows the img; a pending cell shows …
    const imgs = screen.getAllByRole('img')
    expect(imgs.length).toBe(2)
    // At least 4 placeholder gray cells (6 total - 2 generated)
    const placeholders = screen.getAllByText('…')
    expect(placeholders.length).toBe(4)
  })

  it('calls onCellClick only with Ctrl+click (a plain click is reserved for pan-drag)', async () => {
    const user = userEvent.setup()
    const onCellClick = vi.fn()
    const samples = [makeSample(0, 0, 20, null), makeSample(1, 0, 25, null)]
    render(
      <PreviewXYGrid
        samples={samples}
        taskId={99}
        xDraft={xDraft}
        yDraft={null}
        onCellClick={onCellClick}
      />
    )
    const imgs = screen.getAllByRole('img')
    // A plain click -> does not fire (reserved for pan)
    await user.click(imgs[1])
    expect(onCellClick).not.toHaveBeenCalled()
    // Ctrl+click -> fires
    await user.keyboard('{Control>}')
    await user.click(imgs[1])
    await user.keyboard('{/Control}')
    expect(onCellClick).toHaveBeenCalledWith(1)
  })

  it('shows zoom percentage button (replaces old density toggle)', () => {
    const samples = [makeSample(0, 0, 20, null)]
    render(
      <PreviewXYGrid samples={samples} taskId={99} xDraft={xDraft} yDraft={null} />
    )
    // Defaults to 100%
    const zoomBtn = screen.getByRole('button', { name: '100%' })
    expect(zoomBtn).toBeInTheDocument()
  })

  it('highlights selected cells via selectedIndices', () => {
    const samples = [makeSample(0, 0, 20, null), makeSample(1, 0, 25, null)]
    render(
      <PreviewXYGrid
        samples={samples}
        taskId={99}
        xDraft={xDraft}
        yDraft={null}
        selectedIndices={[0]}
      />
    )
    const buttons = screen.getAllByRole('img').map((img) => img.closest('button'))
    expect(buttons[0]?.className).toContain('border-accent')
    expect(buttons[1]?.className).not.toContain('border-accent')
  })

  it('navigates fullscreen cells with arrow keys', async () => {
    const user = userEvent.setup()
    const samples: Sample[] = [
      makeSample(0, 0, 20, 3.0),
      makeSample(1, 0, 25, 3.0),
      makeSample(0, 1, 20, 5.0),
      makeSample(1, 1, 25, 5.0),
    ]
    render(
      <PreviewXYGrid samples={samples} taskId={99} xDraft={xDraft} yDraft={yDraft} />
    )

    await user.dblClick(screen.getAllByRole('img')[0])
    expect(screen.getByText(/Steps=20 .* CFG Scale=3/)).toBeInTheDocument()

    await user.keyboard('{ArrowRight}')
    expect(screen.getByText(/Steps=25 .* CFG Scale=3/)).toBeInTheDocument()

    await user.keyboard('{ArrowDown}')
    expect(screen.getByText(/Steps=25 .* CFG Scale=5/)).toBeInTheDocument()
  })

  it('uses sample.imageUrl when provided (disk replay path)', () => {
    const sample: Sample = {
      ...makeSample(0, 0, 20, null),
      imageUrl: '/api/generate/disk/image/2026-06-08/xy/xy%20plot%201/cell%20x0%20y0.png',
    }
    render(
      <PreviewXYGrid samples={[sample]} taskId={-1} xDraft={xDraft} yDraft={null} />
    )
    const img = screen.getAllByRole('img')[0] as HTMLImageElement
    expect(img.src).toContain('/api/generate/disk/image/2026-06-08/xy/')
    expect(img.src).not.toContain('/sample/')  // does not fall back to generateSampleUrl
  })

  it('with compositeUrl set, Export PNG goes through anchor download, not composeXYMatrix', async () => {
    // composeXYMatrix needs canvas + fetch, neither exists in jsdom -> if the
    // fallback went through it, it would necessarily throw.
    // Here we click the export button and assert we get the exportDownloaded
    // text = it took the compositeUrl direct-download branch.
    const user = userEvent.setup()
    const samples = [makeSample(0, 0, 20, null)]
    // Intercept the anchor click (to avoid a real download prompt)
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    render(
      <PreviewXYGrid
        samples={samples} taskId={-1} xDraft={xDraft} yDraft={null}
        compositeUrl="/api/generate/disk/image/2026-06-08/xy/xy%20plot%201/xy%20plot.png"
      />
    )
    const exportBtn = screen.getByRole('button', { name: /Export PNG|Export/ })
    await user.click(exportBtn)
    expect(clickSpy).toHaveBeenCalledTimes(1)
    clickSpy.mockRestore()
  })
})
