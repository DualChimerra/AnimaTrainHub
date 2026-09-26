import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import TagEditor from './TagEditor'

describe('TagEditor (PP4 chip mode)', () => {
  it('renders chips for each tag', () => {
    render(<TagEditor tags={['a', 'b']} onChange={() => {}} />)
    expect(screen.getByText('a')).toBeInTheDocument()
    expect(screen.getByText('b')).toBeInTheDocument()
  })

  it('Enter adds a tag at the end (chip drag mental model - new things land at the bottom)', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TagEditor tags={['a']} onChange={onChange} />)
    const input = screen.getByPlaceholderText(/\+ tag/)
    await user.type(input, 'new{Enter}')
    expect(onChange).toHaveBeenCalledWith(['a', 'new'])
  })

  it('comma also adds a tag', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TagEditor tags={[]} onChange={onChange} />)
    await user.type(screen.getByPlaceholderText(/\+ tag/), 'foo,')
    expect(onChange).toHaveBeenCalledWith(['foo'])
  })

  it('clicking × removes a tag', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TagEditor tags={['a', 'b']} onChange={onChange} />)
    await user.click(screen.getByLabelText('Delete a'))
    expect(onChange).toHaveBeenCalledWith(['b'])
  })

  it('refuses duplicates silently', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TagEditor tags={['a']} onChange={onChange} />)
    await user.type(screen.getByPlaceholderText(/\+ tag/), 'a{Enter}')
    expect(onChange).not.toHaveBeenCalled()
  })

  it('natural mode renders textarea', () => {
    render(<TagEditor natural tags={['a long sentence']} onChange={() => {}} />)
    expect(
      screen.getByPlaceholderText(/Natural language/)
    ).toBeInTheDocument()
  })
})
