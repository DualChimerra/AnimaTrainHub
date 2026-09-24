import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import SystemStats from './SystemStats'
import { api, type SystemStats as Stats } from '../api/client'

// useEventStream 在 jsdom 下不会真起 EventSource (源码有 typeof 守卫)，所以
// 这里测的主要是：mount 时 GET 一次冷启动 + 各种 stats 形态下的渲染。SSE
// delta 的合并行为另测（手动验证或 e2e）。

function makeStats(overrides: Partial<Stats> = {}): Stats {
  return {
    cpu_pct: 12.5,
    ram_used_gb: 8.0,
    ram_total_gb: 32.0,
    gpu: [
      {
        index: 0,
        name: 'Test GPU',
        util_pct: 50,
        vram_used_gb: 4.0,
        vram_total_gb: 24.0,
        temp_c: 55,
      },
    ],
    ...overrides,
  }
}

describe('SystemStats', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders nothing before first fetch resolves', () => {
    vi.spyOn(api, 'systemStats').mockReturnValue(new Promise(() => {}))
    const { container } = render(<SystemStats />)
    expect(container.firstChild).toBeNull()
  })

  it('shows GPU / VRAM / CPU / RAM readouts after mount fetch', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats())
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('CPU')).toBeInTheDocument())
    // 2026 mockup: label + value + 20px bar; GPU/CPU as %, VRAM/RAM as used / total GB
    expect(screen.getByText('13%')).toBeInTheDocument()  // cpu 12.5 → 13
    expect(screen.getByText('GPU')).toBeInTheDocument()
    expect(screen.getByText('50%')).toBeInTheDocument()  // gpu util 50
    expect(screen.getByText('VRAM')).toBeInTheDocument()
    expect(screen.getByText('4.0 / 24')).toBeInTheDocument()
    expect(screen.getByText('RAM')).toBeInTheDocument()
    expect(screen.getByText('8 / 32')).toBeInTheDocument()
  })

  it('hides GPU / VRAM when stats.gpu is null', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({ gpu: null }))
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('CPU')).toBeInTheDocument())
    expect(screen.queryByText('GPU')).toBeNull()
    expect(screen.queryByText('VRAM')).toBeNull()
  })

  it('hides GPU / VRAM when stats.gpu is empty array', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({ gpu: [] }))
    render(<SystemStats />)
    await waitFor(() => expect(screen.getByText('CPU')).toBeInTheDocument())
    expect(screen.queryByText('GPU')).toBeNull()
    expect(screen.queryByText('VRAM')).toBeNull()
  })

  it('marks the bar hot when load exceeds 90%', async () => {
    vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats({ cpu_pct: 95 }))
    render(<SystemStats />)
    const el = await screen.findByText('95%')
    // the 2026 mockup paints a loaded bar amber via .ds-hot on the fill
    const item = el.parentElement as HTMLElement
    const fill = item.querySelector('.ds-bar > i')
    expect(fill?.className).toContain('ds-hot')
    expect(fill?.getAttribute('style')).toContain('width: 95%')
  })

  it('only fetches once on mount (SSE 化后无轮询)', async () => {
    const spy = vi.spyOn(api, 'systemStats').mockResolvedValue(makeStats())
    render(<SystemStats />)
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1))
    // 等一段实际时间让任何潜在的轮询有机会触发
    await new Promise((r) => setTimeout(r, 200))
    expect(spy).toHaveBeenCalledTimes(1)
  })
})
