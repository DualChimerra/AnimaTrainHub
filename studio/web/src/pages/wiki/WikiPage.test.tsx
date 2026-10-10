import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import en from '../../i18n/locales/en.json'
import { ARTICLES_EN } from './content.en'
import { ARTICLES_RU } from './content.ru'
import { buildArticleEntries, linksOf, scoreEntry, tokenize } from './model'
import WikiPage from './WikiPage'

const SCHEMA = {
  groups: [{ key: 'lora', label: 'Network Settings' }, { key: 'timestep_sampling', label: 'Timestep Sampling' }],
  schema: {
    properties: {
      lora_rank: { type: 'integer', default: 32, minimum: 4, group: 'lora', description: 'Rank' },
      timestep_sampling: { type: 'string', default: 'logit_normal', enum: ['logit_normal', 'style_friendly'], group: 'timestep_sampling', description: 'Mode' },
      style_snr_mean: { type: 'number', default: -6, group: 'timestep_sampling', advanced: true, show_when: 'timestep_sampling==style_friendly', description: 'Mean' },
      no_progress: { type: 'boolean', hidden: true, group: 'monitor', description: 'x' },
    },
  },
}

describe('wiki content', () => {
  const paramKeys = new Set(Object.keys(en.schema.descriptions))
  const ids = new Set(ARTICLES_EN.map((a) => a.id))

  it('ships the same articles in both languages', () => {
    expect(ARTICLES_RU.map((a) => a.id).sort()).toEqual(ARTICLES_EN.map((a) => a.id).sort())
  })

  it.each([['en', ARTICLES_EN], ['ru', ARTICLES_RU]] as const)('every [[link]] in %s resolves', (_lang, articles) => {
    const dead: string[] = []
    for (const e of buildArticleEntries([...articles])) {
      for (const id of linksOf(e)) {
        const ok = id.startsWith('param:') ? paramKeys.has(id.slice(6)) : ids.has(id)
        if (!ok) dead.push(`${e.id} → ${id}`)
      }
    }
    expect(dead).toEqual([])
  })

  it('ranks a title hit above a body hit', () => {
    const entries = buildArticleEntries(ARTICLES_EN)
    const toks = tokenize('optimizer')
    const best = entries.map((e) => ({ id: e.id, s: scoreEntry(e, toks) })).sort((a, b) => b.s - a.s)[0]
    expect(best.id).toMatch(/optimizer/)
  })
})

describe('WikiPage', () => {
  beforeEach(() => {
    try { localStorage.clear() } catch { /* ignore */ }
    vi.mocked(fetch).mockImplementation(async (url) =>
      String(url).includes('/api/schema')
        ? new Response(JSON.stringify(SCHEMA), { status: 200, headers: { 'content-type': 'application/json' } })
        : new Response('', { status: 404 }))
  })

  const renderAt = (url = '/wiki') => render(<MemoryRouter initialEntries={[url]}><WikiPage /></MemoryRouter>)

  it('lists schema parameters next to articles and hides hidden fields', async () => {
    const user = userEvent.setup()
    renderAt()
    const nav = await screen.findByRole('navigation', { name: 'Contents' })
    const lora = await within(nav).findByRole('button', { name: /network/i })
    await user.click(lora)
    await waitFor(() => expect(within(nav).getByTitle('lora_rank')).toBeInTheDocument())
    expect(within(nav).queryByTitle('no_progress')).toBeNull()
    await user.click(within(nav).getByRole('button', { name: /Basics/ }))
    expect(within(nav).getByText('Style LoRA: starter recipe')).toBeInTheDocument()
  })

  it('searches and opens a parameter with its facts and dependencies', async () => {
    const user = userEvent.setup()
    renderAt()
    await screen.findByRole('navigation', { name: 'Contents' })
    await user.type(screen.getByRole('textbox', { name: /search/i }), 'style_snr_mean{Enter}')
    const article = await screen.findByRole('article')
    expect(within(article).getByText('-6')).toBeInTheDocument()
    expect(within(article).getByText('timestep_sampling = style_friendly')).toBeInTheDocument()
    expect(within(article).getAllByRole('button', { name: /Timestep Sampling/ }).length).toBeGreaterThan(0)
  })

  it('opens an article from the URL and follows an internal link', async () => {
    const user = userEvent.setup()
    renderAt('/wiki?e=recipe:style-lora')
    const article = await screen.findByRole('article')
    expect(within(article).getByRole('heading', { name: 'Style LoRA: starter recipe' })).toBeInTheDocument()
    await user.click(within(article).getAllByRole('link', { name: 'Captions and the trigger' })[0])
    expect(await screen.findByRole('heading', { name: 'Captions and the trigger' })).toBeInTheDocument()
  })

  it('saves an entry to favourites', async () => {
    const user = userEvent.setup()
    renderAt('/wiki?e=term:lora')
    const article = await screen.findByRole('article')
    await user.click(within(article).getByRole('button', { name: /Save/ }))
    expect(within(article).getByRole('button', { name: /Saved/ })).toHaveAttribute('aria-pressed', 'true')
    await user.click(screen.getByRole('tab', { name: /Saved/ }))
    const nav = screen.getByRole('navigation', { name: 'Contents' })
    expect(within(nav).getAllByTitle(/.+/).filter((el) => el.classList.contains('wk-leaf'))).toHaveLength(1)
  })
})
