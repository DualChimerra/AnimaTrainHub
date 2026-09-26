import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type CaptionEntry, type ProjectDetail, type ProjectSummary } from '../../../api/client'
import PromptFromDatasetPicker from './PromptFromDatasetPicker'


type CaptionsResult = { folder: null; items: CaptionEntry[] }

function deferred<T>() {
  let resolve!: (v: T) => void
  const promise = new Promise<T>((r) => { resolve = r })
  return { promise, resolve }
}

function cap(name: string, tag: string): CaptionEntry {
  return {
    name, folder: '2_data', tag_count: 1, tags_preview: [tag],
    has_caption: true, tags: [tag], format: 'txt',
  }
}

const projects = [{ id: 1, slug: 'a', title: 'projA' }] as unknown as ProjectSummary[]
const projectDetail = {
  id: 1, slug: 'a', title: 'projA',
  versions: [{ id: 11, label: 'v1' }, { id: 12, label: 'v2' }],
} as unknown as ProjectDetail

function rowThumb() {
  const img = document.querySelector('img')
  if (!img) throw new Error('no row thumbnail rendered')
  return img as HTMLImageElement
}

describe('PromptFromDatasetPicker — thumbnail / pid·vid 同步', () => {
  beforeEach(() => { localStorage.clear() })
  afterEach(() => { vi.restoreAllMocks(); localStorage.clear() })

  async function selectProjectAndV1() {
    const user = userEvent.setup()
    render(<PromptFromDatasetPicker value={null} onChange={vi.fn()} onClose={vi.fn()} />)
    await screen.findByRole('option', { name: 'projA' })
    await user.selectOptions(screen.getByLabelText('选择项目'), '1')
    await waitFor(() => expect(api.listCaptionsFull).toHaveBeenCalledWith(1, 11))
    return user
  }

  it('行缩略图 URL 锚定当前 (pid, vid) + 行文件名', async () => {
    vi.spyOn(api, 'listProjects').mockResolvedValue(projects)
    vi.spyOn(api, 'getProject').mockResolvedValue(projectDetail)
    vi.spyOn(api, 'listCaptionsFull').mockResolvedValue({ folder: null, items: [cap('only.png', 'x')] })

    await selectProjectAndV1()

    await waitFor(() => expect(screen.getByText('only.png')).toBeInTheDocument())
    const src = rowThumb().getAttribute('src') ?? ''
    expect(src).toContain('/projects/1/versions/11/thumb')
    expect(src).toContain('name=only.png')
    expect(src).toContain('folder=2_data')
  })

  it('迟到的旧响应不覆盖当前 captions（regression：thumb 集体 404 黑图）', async () => {
    vi.spyOn(api, 'listProjects').mockResolvedValue(projects)
    vi.spyOn(api, 'getProject').mockResolvedValue(projectDetail)

    const d11 = deferred<CaptionsResult>()
    const d12 = deferred<CaptionsResult>()
    vi.spyOn(api, 'listCaptionsFull').mockImplementation((_pid, vid) =>
      vid === 11 ? d11.promise : d12.promise
    )

    const user = await selectProjectAndV1()

    await user.selectOptions(screen.getByLabelText('选择版本'), '12')
    await waitFor(() => expect(api.listCaptionsFull).toHaveBeenCalledWith(1, 12))

    await act(async () => { d12.resolve({ folder: null, items: [cap('b_v12.png', 'y')] }) })
    await waitFor(() => expect(screen.getByText('b_v12.png')).toBeInTheDocument())

    await act(async () => { d11.resolve({ folder: null, items: [cap('a_v11.png', 'x')] }) })

    expect(screen.getByText('b_v12.png')).toBeInTheDocument()
    expect(screen.queryByText('a_v11.png')).not.toBeInTheDocument()
    const src = rowThumb().getAttribute('src') ?? ''
    expect(src).toContain('/versions/12/thumb')
    expect(src).toContain('name=b_v12.png')
    expect(src).not.toContain('name=a_v11.png')
  })

  it('新版本 captions 加载失败时旧行缩略图仍锚定旧 (pid,vid)（不套 live vid 出 404）', async () => {
    vi.spyOn(api, 'listProjects').mockResolvedValue(projects)
    vi.spyOn(api, 'getProject').mockResolvedValue(projectDetail)
    vi.spyOn(api, 'listCaptionsFull').mockImplementation((_pid, vid) =>
      vid === 11
        ? Promise.resolve({ folder: null, items: [cap('only.png', 'x')] })
        : Promise.reject(new Error('boom'))
    )

    const user = await selectProjectAndV1()
    await waitFor(() => expect(screen.getByText('only.png')).toBeInTheDocument())

    await user.selectOptions(screen.getByLabelText('选择版本'), '12')
    await waitFor(() => expect(api.listCaptionsFull).toHaveBeenCalledWith(1, 12))

    expect(screen.getByText('only.png')).toBeInTheDocument()
    const src = rowThumb().getAttribute('src') ?? ''
    expect(src).toContain('/versions/11/thumb')
    expect(src).not.toContain('/versions/12/thumb')
  })

  it('点击底部大图预览放大成全屏 modal（复用 ImagePreviewModal，请求 1600 大图）', async () => {
    vi.spyOn(api, 'listProjects').mockResolvedValue(projects)
    vi.spyOn(api, 'getProject').mockResolvedValue(projectDetail)
    vi.spyOn(api, 'listCaptionsFull').mockResolvedValue({ folder: null, items: [cap('shot.png', 'x')] })

    const user = await selectProjectAndV1()
    await waitFor(() => expect(screen.getByText('shot.png')).toBeInTheDocument())

    await user.hover(screen.getByText('shot.png'))
    const zoomBtn = await screen.findByRole('button', { name: '点击放大' })

    fireEvent.click(zoomBtn)
    const modalImg = await screen.findByAltText('shot.png')
    const src = modalImg.getAttribute('src') ?? ''
    expect(src).toContain('/projects/1/versions/11/thumb')
    expect(src).toContain('name=shot.png')
    expect(src).toContain('size=1600')

    await user.keyboard('{Escape}')
    await waitFor(() => expect(screen.queryByAltText('shot.png')).not.toBeInTheDocument())
  })
})
