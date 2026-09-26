import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import TagsInput, { parseTags } from './TagsInput'

/** Controlled harness: simulates the parent component holding string[] state, mirroring the real controlled loop. */
function Harness({ initial = [] as string[] }: { initial?: string[] }) {
  const [tags, setTags] = useState<string[]>(initial)
  return (
    <>
      <TagsInput label="bl" value={tags} disabled={false} onChange={setTags} />
      <output data-testid="tags">{JSON.stringify(tags)}</output>
    </>
  )
}

/** The resting state is a role=button chip container; clicking it enters edit state, which is when the textbox appears. */
async function enterEdit(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button'))
  return screen.getByRole('textbox')
}

describe('TagsInput', () => {
  it('editing lets you type comma-separated tags - the comma is not eaten immediately', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const input = await enterEdit(user)
    await user.type(input, 'monochrome, greyscale')
    expect(input).toHaveValue('monochrome, greyscale')
    expect(screen.getByTestId('tags')).toHaveTextContent('["monochrome","greyscale"]')
  })

  it('editing lets you type a multi-word tag with spaces - the space is not trimmed', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const input = await enterEdit(user)
    await user.type(input, 'blue eyes')
    expect(input).toHaveValue('blue eyes')
    expect(screen.getByTestId('tags')).toHaveTextContent('["blue eyes"]')
  })

  it('blur exits edit state, renders as chips, normalizes the text', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const input = await enterEdit(user)
    await user.type(input, 'x,,y , ')
    expect(input).toHaveValue('x,,y , ')   // edit state keeps the raw text
    await user.tab()                       // blur
    expect(screen.queryByRole('textbox')).toBeNull()   // back to the chip resting state
    expect(screen.getByText('x')).toBeInTheDocument()
    expect(screen.getByText('y')).toBeInTheDocument()
    expect(screen.getByTestId('tags')).toHaveTextContent('["x","y"]')
    // re-entering edit state shows the normalized form
    const input2 = await enterEdit(user)
    expect(input2).toHaveValue('x, y')
  })

  it('blur normalizes underscores to spaces (chips use a unified space form)', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const input = await enterEdit(user)
    await user.type(input, 'cat_girl, blue_eyes')
    expect(input).toHaveValue('cat_girl, blue_eyes')   // edit state shows it verbatim
    await user.tab()                                    // blur normalizes
    expect(screen.getByText('cat girl')).toBeInTheDocument()
    expect(screen.getByText('blue eyes')).toBeInTheDocument()
    expect(screen.getByTestId('tags')).toHaveTextContent('["cat girl","blue eyes"]')
  })

  it('the resting state renders each tag as its own chip', () => {
    render(<Harness initial={['monochrome', 'blue eyes']} />)
    expect(screen.queryByRole('textbox')).toBeNull()
    expect(screen.getByText('monochrome')).toBeInTheDocument()
    expect(screen.getByText('blue eyes')).toBeInTheDocument()
  })

  it('syncs the chip display when the external value changes (restore defaults)', async () => {
    function ResetHarness() {
      const [tags, setTags] = useState<string[]>(['a', 'b'])
      return (
        <>
          <TagsInput label="bl" value={tags} disabled={false} onChange={setTags} />
          <button onClick={() => setTags(['reset'])}>do-reset</button>
        </>
      )
    }
    const user = userEvent.setup()
    render(<ResetHarness />)
    expect(screen.getByText('a')).toBeInTheDocument()
    expect(screen.getByText('b')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'do-reset' }))
    expect(screen.getByText('reset')).toBeInTheDocument()
    expect(screen.queryByText('a')).toBeNull()
  })

  it('parseTags trims leading/trailing whitespace and drops empty segments', () => {
    expect(parseTags('  a , , b ,')).toEqual(['a', 'b'])
    expect(parseTags('')).toEqual([])
    expect(parseTags('blue eyes, red_hair')).toEqual(['blue eyes', 'red_hair'])
  })
})
