/** SoupPage smoke: the two things a user can get badly wrong here are
 *  (a) not seeing that a weight of 1 next to a weight of 3 means 25%, and
 *  (b) being allowed to merge checkpoints the backend will refuse. Both are
 *  pinned below, together with the "test it now" hand-off to the generate page. */
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { DialogProvider } from '../../components/Dialog'
import { ToastProvider } from '../../components/Toast'
import SoupPage from './Soup'

const navigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return { ...actual, useNavigate: () => navigate }
})

const UPLOADS = [
  { name: 'a.safetensors', path: '/soup/uploads/a.safetensors', size: 1024 ** 2, mtime: 2 },
  { name: 'b.safetensors', path: '/soup/uploads/b.safetensors', size: 1024 ** 2, mtime: 1 },
]

let inspectOk = true
const fetchMock = vi.fn()

function json(body: unknown) {
  return Promise.resolve({
    ok: true, status: 200,
    json: async () => body,
    text: async () => JSON.stringify(body),
    headers: new Headers({ 'content-type': 'application/json' }),
  } as Response)
}

beforeEach(() => {
  inspectOk = true
  navigate.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  fetchMock.mockReset()
  fetchMock.mockImplementation((url: string, init?: RequestInit) => {
    if (url.endsWith('/api/soup/sources')) return json({ uploads: UPLOADS, outputs: [] })
    if (url.endsWith('/api/soup/inspect')) {
      return json(inspectOk
        ? { ok: true, errors: [], warnings: [], items: [] }
        : { ok: false, errors: ['different rank'], warnings: [], items: [] })
    }
    if (url.endsWith('/api/soup/merge') && init?.method === 'POST') {
      return json({
        name: 'mix.safetensors', path: '/soup/output/mix.safetensors',
        size: 2 * 1024 ** 2, mtime: 3, warnings: [], sources: [],
      })
    }
    if (url.endsWith('/api/projects')) return json({ items: [] })
    return json({})
  })
})

function renderPage() {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <DialogProvider>
          <SoupPage />
        </DialogProvider>
      </ToastProvider>
    </MemoryRouter>,
  )
}

/** Opens the picker and adds both uploaded files. */
async function addTwo(user: ReturnType<typeof userEvent.setup>) {
  for (const name of ['a.safetensors', 'b.safetensors']) {
    await user.click((await screen.findAllByRole('button', { name: /add checkpoint/i }))[0])
    const dialog = await screen.findByRole('dialog')
    await user.click(within(dialog).getByText(name))
  }
}

describe('SoupPage', () => {
  it('needs two checkpoints before it will merge', async () => {
    const user = userEvent.setup()
    renderPage()
    const merge = await screen.findByRole('button', { name: /^merge$/i })
    expect(merge).toBeDisabled()
    await addTwo(user)
    await waitFor(() => expect(merge).toBeEnabled())
  })

  it('shows each checkpoint as its real share, not its raw weight', async () => {
    const user = userEvent.setup()
    renderPage()
    await addTwo(user)
    // Two equal weights → half each.
    await waitFor(() => expect(screen.getAllByText('50%')).toHaveLength(2))
    const weights = screen.getAllByRole('spinbutton', { name: /weight/i })
    await user.clear(weights[0])
    await user.type(weights[0], '3')
    await waitFor(() => {
      expect(screen.getByText('75%')).toBeInTheDocument()
      expect(screen.getByText('25%')).toBeInTheDocument()
    })
  })

  it('refuses to merge what the backend calls incompatible', async () => {
    inspectOk = false
    const user = userEvent.setup()
    renderPage()
    await addTwo(user)
    expect(await screen.findByText('different rank')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^merge$/i })).toBeDisabled()
  })

  it('merges and hands the result to the generate page', async () => {
    const user = userEvent.setup()
    renderPage()
    await addTwo(user)
    const merge = await screen.findByRole('button', { name: /^merge$/i })
    await waitFor(() => expect(merge).toBeEnabled())
    await user.click(merge)

    const body = JSON.parse(String(
      fetchMock.mock.calls.find(([u]) => String(u).endsWith('/api/soup/merge'))?.[1]?.body,
    ))
    expect(body.method).toBe('average')
    expect(body.inputs.map((i: { path: string }) => i.path)).toEqual(UPLOADS.map((u) => u.path))
    // Name defaults to the suggestion rather than making the user invent one.
    expect(body.name).toBe('a + b')

    await user.click(await screen.findByRole('button', { name: /test it now/i }))
    expect(navigate).toHaveBeenCalledWith(
      `/tools/generate?lora=${encodeURIComponent('/soup/output/mix.safetensors')}`,
    )
  })
})
