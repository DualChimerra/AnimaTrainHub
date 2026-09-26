/** GeneratePage 端到端 smoke：mock fetch，验证 single / xy / 多 prompt+xy
 *  三个关键路径的 enqueue payload 行为。 */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ToastProvider } from '../../components/Toast'
import GeneratePage from './Generate'
import { useMonitorProgress } from '../../lib/useMonitorProgress'

vi.mock('../../lib/useMonitorProgress', () => {
  const NULL_MONITOR = { state: null }
  return { useMonitorProgress: vi.fn(() => NULL_MONITOR) }
})

const fetchMock = vi.fn()
let lastEnqueueBody: Record<string, unknown> | null = null

beforeEach(() => {
  lastEnqueueBody = null
  window.localStorage.clear()
  vi.mocked(useMonitorProgress).mockReturnValue({ state: null } as never)
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string, init?: RequestInit) => {
    if (url.endsWith('/api/projects') && (init?.method ?? 'GET') === 'GET') {
      return Promise.resolve({
        ok: true, status: 200,
        json: async () => ({ items: [] }),
        text: async () => '{"items":[]}',
        headers: new Headers({ 'content-type': 'application/json' }),
      } as Response)
    }
    if (url.startsWith('/api/queue') && (init?.method ?? 'GET') === 'GET') {
      return Promise.resolve({
        ok: true, status: 200,
        json: async () => ({ items: [] }),
        text: async () => '{"items":[]}',
        headers: new Headers({ 'content-type': 'application/json' }),
      } as Response)
    }
    // enqueueGenerate
    if (url.endsWith('/api/generate') && init?.method === 'POST') {
      lastEnqueueBody = JSON.parse(String(init.body))
      const taskStub = {
        id: 1, name: 'generate', config_name: 'generate', status: 'pending',
        priority: 0, created_at: 0, started_at: null, finished_at: null,
        pid: null, exit_code: null, output_dir: null, error_msg: null,
      }
      return Promise.resolve({
        ok: true, status: 200,
        json: async () => taskStub,
        text: async () => JSON.stringify(taskStub),
        headers: new Headers({ 'content-type': 'application/json' }),
      } as Response)
    }
    return Promise.resolve({
      ok: false, status: 404,
      json: async () => null,
      text: async () => '',
      headers: new Headers(),
    } as Response)
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function setup() {
  return render(
    <ToastProvider>
      <GeneratePage />
    </ToastProvider>
  )
}

async function waitForInitialLorasLoad() {
  await screen.findByRole('button', { name: /开始生成/ })
}

// The settings column shows prompts, LoRA and parameters at once (no tabs any
// more); kept as a no-op so the call sites read the same.
async function openPromptsTab(_user: ReturnType<typeof userEvent.setup>) {
  await Promise.resolve()
}

describe('GeneratePage 端到端 smoke', () => {
  it('mode=single：enqueue payload 含 xy_matrix=null + 完整字段', async () => {
    const user = userEvent.setup()
    setup()

    const btn = screen.getByRole('button', { name: /开始生成/ })
    await user.click(btn)

    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    const body = lastEnqueueBody!
    expect(body.xy_matrix).toBeNull()
    expect(body.prompts).toEqual(['newest, safe, 1girl, masterpiece, best quality'])
    expect(body.count).toBe(1)
    expect(body.attention_backend).toBeUndefined()
  })

  it('多任务（P-I）：running + pending → 排队列表带取消，提交按钮不禁用', async () => {
    const genTask = (id: number, status: string) => ({
      id, name: 'generate', config_name: 'generate', status, priority: 0,
      created_at: 0, started_at: status === 'running' ? 1 : null, finished_at: null,
      pid: null, exit_code: null, output_dir: null, error_msg: null,
    })
    const jsonOk = (body: unknown) => Promise.resolve({
      ok: true, status: 200, json: async () => body,
      text: async () => JSON.stringify(body),
      headers: new Headers({ 'content-type': 'application/json' }),
    } as Response)
    fetchMock.mockImplementation((url: string) => {
      if (url.endsWith('/api/projects')) return jsonOk({ items: [] })
      // group=live&types=generate → running #5 + pending #6,#7
      if (url.includes('group=live')) {
        return jsonOk({ items: [genTask(5, 'running'), genTask(6, 'pending'), genTask(7, 'pending')] })
      }
      if (url.startsWith('/api/queue')) return jsonOk({ items: [] })
      return Promise.resolve({ ok: false, status: 404, json: async () => null, text: async () => '', headers: new Headers() } as Response)
    })

    setup()
    await waitFor(() => expect(screen.getByTestId('timeline-cancel-6')).toBeInTheDocument())
    expect(screen.getByTestId('timeline-cancel-7')).toBeInTheDocument()
    expect(screen.getByTestId('timeline-cancel-5')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /开始生成/ })).not.toBeDisabled()
  })

  it('mode=xy 默认 X=steps 20,25,30：按钮显示「开始生成 · 3 张」并 enqueue 正确 xy_matrix', async () => {
    const user = userEvent.setup()
    setup()

    await user.click(screen.getByRole('button', { name: 'XY 矩阵' }))

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /开始生成 · 3 张/ })).toBeInTheDocument()
    )

    await user.click(screen.getByRole('button', { name: /开始生成 · 3 张/ }))

    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    const body = lastEnqueueBody!
    const xy = body.xy_matrix as { x: { axis: string; values: number[] }; y: unknown }
    expect(xy).not.toBeNull()
    expect(xy.x.axis).toBe('steps')
    expect(xy.x.values).toEqual([20, 25, 30])
    expect(xy.y).toBeNull()
    expect(body.count).toBe(1)
  })

  it('多 prompt 轮换功能已隐藏：只有一个 textarea，"添加 prompt"按钮不存在', async () => {
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()
    await openPromptsTab(user)
    const promptInputs = screen.getAllByPlaceholderText('输入正向提示词…')
    expect(promptInputs.length).toBe(1)
    expect(screen.queryByRole('button', { name: /添加 prompt/ })).toBeNull()
  })

  it('切到 xy 再切回 single：sidebar 已填的 prompts/seed 等保留', async () => {
    const user = userEvent.setup()
    setup()
    await openPromptsTab(user)

    const promptArea = screen.getAllByPlaceholderText('输入正向提示词…')[0]
    await user.clear(promptArea)
    await user.type(promptArea, 'my custom prompt')

    await user.click(screen.getByRole('button', { name: 'XY 矩阵' }))
    await user.click(screen.getByRole('button', { name: '单图' }))

    expect(promptArea).toHaveValue('my custom prompt')
  })

  it('训练 / reg-ai 等任务在跑时，按钮可用（提交排队）+ tooltip 说明会排队', async () => {
    const previousImpl = fetchMock.getMockImplementation()
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (
        url.startsWith('/api/queue') && !url.includes('group=live')
        && (init?.method ?? 'GET') === 'GET'
      ) {
        const running = {
          id: 42, name: 'train', config_name: 'train', status: 'running',
          priority: 0, created_at: 0, started_at: 0, finished_at: null,
          pid: 1234, exit_code: null, output_dir: null, error_msg: null,
        }
        return Promise.resolve({
          ok: true, status: 200,
          json: async () => ({ items: [running] }),
          text: async () => `{"items":[${JSON.stringify(running)}]}`,
          headers: new Headers({ 'content-type': 'application/json' }),
        } as Response)
      }
      return previousImpl ? previousImpl(url, init) : Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })

    setup()

    const btn = await screen.findByRole('button', { name: /开始生成/ })
    await waitFor(() =>
      expect(btn).toHaveAttribute('title', expect.stringContaining('#42')),
    )
    expect(btn).not.toBeDisabled()
  })

  it('URL ?lora= 进入时 replace 缓存 LoRA list + clamp xDraft.loraIndex', async () => {
    window.localStorage.setItem(
      'studio:generate:params:v1',
      JSON.stringify({
        mode: 'single',
        prompts: ['persist'],
        negPrompt: '',
        aspect: '1:1',
        width: 1024, height: 1024,
        steps: 25, cfgScale: 4, count: 1, seed: 0,
        loras: [
          { path: 'G:/old/cached.safetensors', scale: 1, project_id: 1, version_id: 1 },
        ],
        xDraft: { axis: 'lora_ckpt', raw: 'a, b', loraIndex: 5 },
        yDraft: { axis: 'lora_scale', raw: '0.5, 1.0', loraIndex: 3 },
        datasetPick: null,
      })
    )

    const newLoraPath = 'G:/new/from_project.safetensors'
    const search = `?lora=${encodeURIComponent(newLoraPath)}&projectId=2&versionId=3`
    window.history.replaceState({}, '', `/tools/generate${search}`)

    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /开始生成/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    const body = lastEnqueueBody!
    expect(body.lora_configs).toEqual([
      { path: newLoraPath, scale: 1.0, project_id: 2, version_id: 3 },
    ])

    const stored = JSON.parse(window.localStorage.getItem('studio:generate:params:v1')!)
    expect(stored.xDraft.loraIndex).toBe(0)
    expect(stored.yDraft.loraIndex).toBe(0)
    expect(window.location.search).toBe('')
  })

  it('刷新后恢复左侧生成参数，但不恢复当前生成结果', async () => {
    const user = userEvent.setup()
    const first = setup()
    await waitForInitialLorasLoad()
    await openPromptsTab(user)

    const promptArea = screen.getAllByPlaceholderText('输入正向提示词…')[0]
    await user.clear(promptArea)
    await user.type(promptArea, 'persist me')
    await user.click(screen.getByRole('button', { name: 'XY 矩阵' }))

    first.unmount()
    setup()
    await waitForInitialLorasLoad()

    expect(screen.getAllByPlaceholderText('输入正向提示词…')[0]).toHaveValue('persist me')
    expect(screen.getByRole('button', { name: /开始生成 · 3 张/ })).toBeInTheDocument()
    expect(screen.queryByText('#1')).toBeNull()
    expect(screen.getByText('填写参数后点击「开始生成」')).toBeInTheDocument()
  })


  const A = { path: 'G:/a.safetensors', scale: 1, project_id: null, version_id: null }
  const B = { path: 'G:/b.safetensors', scale: 1, project_id: null, version_id: null }
  const seedPrefs = (over: Record<string, unknown>) =>
    window.localStorage.setItem(
      'studio:generate:params:v1',
      JSON.stringify({
        mode: 'single', prompts: ['x'], negPrompt: '',
        aspect: '1:1', width: 1024, height: 1024,
        steps: 25, cfgScale: 4, count: 1, seed: 0,
        xDraft: { axis: 'steps', raw: '20, 25, 30', loraIndex: null },
        yDraft: null, datasetPick: null,
        ...over,
      })
    )

  it('single 提交只用 singleLoras（不带 xyLoras）', async () => {
    seedPrefs({ mode: 'single', singleLoras: [A], xyLoras: [B] })
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /开始生成/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    expect(lastEnqueueBody!.lora_configs).toEqual([A])
    expect(lastEnqueueBody!.xy_matrix).toBeNull()
  })

  it('xy 提交不带 singleLoras，也不带未被轴引用的 xyLoras 孤儿', async () => {
    seedPrefs({ mode: 'xy', singleLoras: [A], xyLoras: [B] })
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /开始生成/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    expect(lastEnqueueBody!.lora_configs).toEqual([])
    expect(lastEnqueueBody!.xy_matrix).not.toBeNull()
  })

  it('老版本共享 loras 迁移：拆成 singleLoras/xyLoras 各一份，不丢已选 LoRA', async () => {
    seedPrefs({ mode: 'single', loras: [A] })
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /开始生成/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    expect(lastEnqueueBody!.lora_configs).toEqual([A])

    const stored = JSON.parse(window.localStorage.getItem('studio:generate:params:v1')!)
    expect(stored.singleLoras).toEqual([A])
    expect(stored.xyLoras).toEqual([A])
  })

  it('点击 XY 落盘历史 → 左侧 XY 轴 dropdown 切到 LoRA + raw 写入', async () => {
    seedPrefs({ mode: 'xy' })
    const xySnapshotParams = {
      schema_version: 1,
      mode: 'xy',
      prompts: ['recall-prompt'],
      negative_prompt: 'recall-neg',
      width: 768, height: 1344,
      steps: 25, cfg_scale: 5, count: 1, seed: 7,
      loras: [
        { name: 'chen-bin_V3.7_step5500.safetensors', scale: 1,
          project_id: 19, version_id: 44 },
      ],
      xy_draft: {
        x: {
          axis: 'lora_ckpt',
          raw: 'epoch40.safetensors, epoch38.safetensors, epoch24.safetensors',
          loraIndex: 0,
        },
        y: null,
      },
      dataset_pick: null,
    }
    const diskEntry = {
      id: 'disk:abc123',
      date: '2026-06-09',
      mode: 'xy',
      folder: 'xy plot 1',
      path: '/tmp/test/2026-06-09/xy/xy plot 1',
      image_url: '/api/generate/disk/image/2026-06-09/xy/xy%20plot%201/xy%20plot.png',
      thumb_url: '/api/generate/disk/thumb/2026-06-09/xy/xy%20plot%201/xy%20plot.png?w=128',
      created_at: 1717900000,
      schema_version: 2,
      params: xySnapshotParams,
      xy_meta: {
        x_axis: 'lora_ckpt',
        y_axis: null,
        x_values: ['epoch40.safetensors', 'epoch38.safetensors', 'epoch24.safetensors'],
        y_values: [null],
        samples: [],
      },
    }
    const previousImpl = fetchMock.getMockImplementation()
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/api/generate/disk/history') && (init?.method ?? 'GET') === 'GET') {
        return Promise.resolve({
          ok: true, status: 200,
          json: async () => ({ entries: [diskEntry] }),
          text: async () => JSON.stringify({ entries: [diskEntry] }),
          headers: new Headers({ 'content-type': 'application/json' }),
        } as Response)
      }
      return previousImpl ? previousImpl(url, init) : Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })

    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    const initialAxisInput = await screen.findByDisplayValue(/20, 25, 30/)
    expect(initialAxisInput).toBeInTheDocument()

    const thumb = await screen.findByTitle(/xy plot 1 ·/)
    await user.click(thumb)

    await waitFor(() => {
      const xLabel = screen.getAllByText('X 轴')[0]
      // AxisCard uses the same frame as a LoRA slot card (.ds-card.ds-flat)
      const card = xLabel.closest('div.ds-card')!
      const axisSelect = card.querySelector('select') as HTMLSelectElement
      expect(axisSelect.value).toBe('lora_ckpt')
    })
    expect(screen.queryByDisplayValue(/20, 25, 30/)).not.toBeInTheDocument()
  })

  it('XY 开始后改 X 轴：sidebar 改了但右侧结果网格冻结（30 列仍在）', async () => {
    vi.mocked(useMonitorProgress).mockReturnValue({
      state: {
        samples: [
          { path: 'cell x0 y0.png', xy: { xi: 0, yi: 0, xv: 20, yv: null } },
          { path: 'cell x1 y0.png', xy: { xi: 1, yi: 0, xv: 25, yv: null } },
          { path: 'cell x2 y0.png', xy: { xi: 2, yi: 0, xv: 30, yv: null } },
        ],
      },
    } as never)
    seedPrefs({ mode: 'xy' })  // 默认 X=steps raw "20, 25, 30"
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /开始生成 · 3 张/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())

    await waitFor(() => expect(screen.getByText('30')).toBeInTheDocument())

    const axisInput = screen.getByDisplayValue('20, 25, 30')
    await user.clear(axisInput)
    await user.type(axisInput, '20, 25')

    await waitFor(() => expect(axisInput).toHaveValue('20, 25'))
    expect(screen.getByText('30')).toBeInTheDocument()
  })

  it('回看 XY 历史时点开始生成：清掉历史 override，结果区回到实时新任务', async () => {
    vi.mocked(useMonitorProgress).mockReturnValue({
      state: {
        samples: [{ path: 'cell x0 y0.png', xy: { xi: 0, yi: 0, xv: 20, yv: null } }],
      },
    } as never)
    seedPrefs({ mode: 'xy' })  // 默认 X=steps raw "20, 25, 30"
    const xySnapshotParams = {
      schema_version: 1, mode: 'xy',
      prompts: ['recall'], negative_prompt: '',
      width: 1024, height: 1024, steps: 25, cfg_scale: 4, count: 1, seed: 0,
      loras: [],
      xy_draft: { x: { axis: 'steps', raw: '20, 25, 30', loraIndex: null }, y: null },
      dataset_pick: null,
    }
    const diskEntry = {
      id: 'disk:xy1', date: '2026-06-09', mode: 'xy', folder: 'xy plot 1',
      path: '/tmp/test/2026-06-09/xy/xy plot 1',
      image_url: '/api/generate/disk/image/2026-06-09/xy/xy%20plot%201/xy%20plot.png',
      thumb_url: '/api/generate/disk/thumb/2026-06-09/xy/xy%20plot%201/xy%20plot.png?w=128',
      created_at: 1717900000, schema_version: 2,
      params: xySnapshotParams,
      xy_meta: {
        x_axis: 'steps', y_axis: null,
        x_values: ['20', '25', '30'], y_values: [null], samples: [],
      },
    }
    const previousImpl = fetchMock.getMockImplementation()
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/api/generate/disk/history') && (init?.method ?? 'GET') === 'GET') {
        return Promise.resolve({
          ok: true, status: 200,
          json: async () => ({ entries: [diskEntry] }),
          text: async () => JSON.stringify({ entries: [diskEntry] }),
          headers: new Headers({ 'content-type': 'application/json' }),
        } as Response)
      }
      return previousImpl ? previousImpl(url, init) : Promise.resolve({
        ok: false, status: 404, json: async () => null, text: async () => '',
        headers: new Headers(),
      } as Response)
    })

    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    const thumb = await screen.findByTitle(/xy plot 1 ·/)
    await user.click(thumb)
    await waitFor(() => expect(screen.getByText('xy plot 1')).toBeInTheDocument())

    await user.click(screen.getByRole('button', { name: /开始生成 · 3 张/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    await waitFor(() => expect(screen.queryByText('xy plot 1')).not.toBeInTheDocument())
  })
})
