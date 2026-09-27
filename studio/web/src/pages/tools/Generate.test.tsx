/** GeneratePage end-to-end smoke: mocks fetch and verifies the enqueue payload
 *  behavior for the single / xy / multi-prompt+xy code paths. */
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
  await screen.findByRole('button', { name: /Start generate/ })
}

// The settings column shows prompts, LoRA and parameters at once (no tabs any
// more); kept as a no-op so the call sites read the same.
async function openPromptsTab(_user: ReturnType<typeof userEvent.setup>) {
  await Promise.resolve()
}

describe('GeneratePage end-to-end smoke', () => {
  it('mode=single: enqueue payload has xy_matrix=null + full fields', async () => {
    const user = userEvent.setup()
    setup()

    const btn = screen.getByRole('button', { name: /Start generate/ })
    await user.click(btn)

    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    const body = lastEnqueueBody!
    expect(body.xy_matrix).toBeNull()
    expect(body.prompts).toEqual(['newest, safe, 1girl, masterpiece, best quality'])
    expect(body.count).toBe(1)
    expect(body.attention_backend).toBeUndefined()
  })

  it('multi-task (P-I): running + pending -> queue list has cancel, submit button stays enabled', async () => {
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
    expect(screen.getByRole('button', { name: /Start generate/ })).not.toBeDisabled()
  })

  it('mode=xy defaults to X=steps 20,25,30: button shows "Start generate . 3 images" and enqueues the correct xy_matrix', async () => {
    const user = userEvent.setup()
    setup()

    await user.click(screen.getByRole('button', { name: 'XY Matrix' }))

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Start generate . 3 images/ })).toBeInTheDocument()
    )

    await user.click(screen.getByRole('button', { name: /Start generate . 3 images/ }))

    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    const body = lastEnqueueBody!
    const xy = body.xy_matrix as { x: { axis: string; values: number[] }; y: unknown }
    expect(xy).not.toBeNull()
    expect(xy.x.axis).toBe('steps')
    expect(xy.x.values).toEqual([20, 25, 30])
    expect(xy.y).toBeNull()
    expect(body.count).toBe(1)
  })

  it('multi-prompt rotation is hidden: only one textarea, no "Add prompt" button', async () => {
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()
    await openPromptsTab(user)
    const promptInputs = screen.getAllByPlaceholderText('Enter positive prompt…')
    expect(promptInputs.length).toBe(1)
    expect(screen.queryByRole('button', { name: /Add prompt/ })).toBeNull()
  })

  it('switching to xy and back to single: sidebar-filled prompts/seed etc. are kept', async () => {
    const user = userEvent.setup()
    setup()
    await openPromptsTab(user)

    const promptArea = screen.getAllByPlaceholderText('Enter positive prompt…')[0]
    await user.clear(promptArea)
    await user.type(promptArea, 'my custom prompt')

    await user.click(screen.getByRole('button', { name: 'XY Matrix' }))
    await user.click(screen.getByRole('button', { name: 'Single' }))

    expect(promptArea).toHaveValue('my custom prompt')
  })

  it('while a training / reg-ai task etc. is running, the button stays enabled (submits to queue) + tooltip explains queueing', async () => {
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

    const btn = await screen.findByRole('button', { name: /Start generate/ })
    await waitFor(() =>
      expect(btn).toHaveAttribute('title', expect.stringContaining('#42')),
    )
    expect(btn).not.toBeDisabled()
  })

  it('entering via URL ?lora= replaces the cached LoRA list + clamps xDraft.loraIndex', async () => {
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

    await user.click(await screen.findByRole('button', { name: /Start generate/ }))
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

  it('after a refresh, the left-side generate params are restored, but the current generation result is not', async () => {
    const user = userEvent.setup()
    const first = setup()
    await waitForInitialLorasLoad()
    await openPromptsTab(user)

    const promptArea = screen.getAllByPlaceholderText('Enter positive prompt…')[0]
    await user.clear(promptArea)
    await user.type(promptArea, 'persist me')
    await user.click(screen.getByRole('button', { name: 'XY Matrix' }))

    first.unmount()
    setup()
    await waitForInitialLorasLoad()

    expect(screen.getAllByPlaceholderText('Enter positive prompt…')[0]).toHaveValue('persist me')
    expect(screen.getByRole('button', { name: /Start generate . 3 images/ })).toBeInTheDocument()
    expect(screen.queryByText('#1')).toBeNull()
    expect(screen.getByText('Fill parameters, then click \u201cStart generate\u201d')).toBeInTheDocument()
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

  it('single submit uses only singleLoras (no xyLoras)', async () => {
    seedPrefs({ mode: 'single', singleLoras: [A], xyLoras: [B] })
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /Start generate/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    expect(lastEnqueueBody!.lora_configs).toEqual([A])
    expect(lastEnqueueBody!.xy_matrix).toBeNull()
  })

  it('xy submit has no singleLoras, and no orphan xyLoras unreferenced by any axis', async () => {
    seedPrefs({ mode: 'xy', singleLoras: [A], xyLoras: [B] })
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /Start generate/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    expect(lastEnqueueBody!.lora_configs).toEqual([])
    expect(lastEnqueueBody!.xy_matrix).not.toBeNull()
  })

  it('legacy shared-loras migration: splits into one copy each of singleLoras/xyLoras, keeping selected LoRAs', async () => {
    seedPrefs({ mode: 'single', loras: [A] })
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /Start generate/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    expect(lastEnqueueBody!.lora_configs).toEqual([A])

    const stored = JSON.parse(window.localStorage.getItem('studio:generate:params:v1')!)
    expect(stored.singleLoras).toEqual([A])
    expect(stored.xyLoras).toEqual([A])
  })

  it('clicking a persisted XY history entry -> the left XY axis dropdown switches to LoRA + writes raw', async () => {
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
      const xLabel = screen.getAllByText('X axis')[0]
      // AxisCard uses the same frame as a LoRA slot card (.ds-card.ds-flat)
      const card = xLabel.closest('div.ds-card')!
      const axisSelect = card.querySelector('select') as HTMLSelectElement
      expect(axisSelect.value).toBe('lora_ckpt')
    })
    expect(screen.queryByDisplayValue(/20, 25, 30/)).not.toBeInTheDocument()
  })

  it('changing the X axis after XY starts: sidebar updates but the right-side result grid stays frozen (30 columns remain)', async () => {
    vi.mocked(useMonitorProgress).mockReturnValue({
      state: {
        samples: [
          { path: 'cell x0 y0.png', xy: { xi: 0, yi: 0, xv: 20, yv: null } },
          { path: 'cell x1 y0.png', xy: { xi: 1, yi: 0, xv: 25, yv: null } },
          { path: 'cell x2 y0.png', xy: { xi: 2, yi: 0, xv: 30, yv: null } },
        ],
      },
    } as never)
    seedPrefs({ mode: 'xy' })  // defaults to X=steps raw "20, 25, 30"
    const user = userEvent.setup()
    setup()
    await waitForInitialLorasLoad()

    await user.click(await screen.findByRole('button', { name: /Start generate . 3 images/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())

    await waitFor(() => expect(screen.getByText('30')).toBeInTheDocument())

    const axisInput = screen.getByDisplayValue('20, 25, 30')
    await user.clear(axisInput)
    await user.type(axisInput, '20, 25')

    await waitFor(() => expect(axisInput).toHaveValue('20, 25'))
    expect(screen.getByText('30')).toBeInTheDocument()
  })

  it('clicking start generate while viewing XY history: clears the history override, result area returns to a live new task', async () => {
    vi.mocked(useMonitorProgress).mockReturnValue({
      state: {
        samples: [{ path: 'cell x0 y0.png', xy: { xi: 0, yi: 0, xv: 20, yv: null } }],
      },
    } as never)
    seedPrefs({ mode: 'xy' })  // defaults to X=steps raw "20, 25, 30"
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

    await user.click(screen.getByRole('button', { name: /Start generate . 3 images/ }))
    await waitFor(() => expect(lastEnqueueBody).not.toBeNull())
    await waitFor(() => expect(screen.queryByText('xy plot 1')).not.toBeInTheDocument())
  })
})
