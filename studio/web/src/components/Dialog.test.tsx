import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { DialogProvider, useDialog } from './Dialog'
import { useState } from 'react'

// Test harness: the outer component triggers confirm/prompt/alert, puts the resolved
// Promise value on screen, and the test asserts on the user interaction's result.
function Harness({
  run,
}: {
  run: (api: ReturnType<typeof useDialog>) => Promise<unknown>
}) {
  const api = useDialog()
  const [result, setResult] = useState<string>('')
  return (
    <div>
      <button
        onClick={() =>
          void run(api).then((v) => setResult('resolved:' + JSON.stringify(v)))
        }
      >
        trigger
      </button>
      <div data-testid="result">{result}</div>
    </div>
  )
}

const wrap = (run: (api: ReturnType<typeof useDialog>) => Promise<unknown>) =>
  render(
    <DialogProvider>
      <Harness run={run} />
    </DialogProvider>,
  )

describe('Dialog (useDialog API)', () => {
  it('confirm resolves true on OK click', async () => {
    const user = userEvent.setup()
    wrap((api) => api.confirm('Delete?'))
    await user.click(screen.getByText('trigger'))
    await user.click(screen.getByText('Confirm'))
    expect(screen.getByTestId('result')).toHaveTextContent('resolved:true')
  })

  it('confirm resolves false on cancel click', async () => {
    const user = userEvent.setup()
    wrap((api) => api.confirm('Delete?'))
    await user.click(screen.getByText('trigger'))
    await user.click(screen.getByText('Cancel'))
    expect(screen.getByTestId('result')).toHaveTextContent('resolved:false')
  })

  it('confirm resolves false when closed with Escape', async () => {
    const user = userEvent.setup()
    wrap((api) => api.confirm('Delete?'))
    await user.click(screen.getByText('trigger'))
    await user.keyboard('{Escape}')
    expect(screen.getByTestId('result')).toHaveTextContent('resolved:false')
  })

  it('confirm supports custom button labels', async () => {
    const user = userEvent.setup()
    wrap((api) => api.confirm('Delete?', { okText: 'Burn it', cancelText: 'Nevermind' }))
    await user.click(screen.getByText('trigger'))
    expect(screen.getByText('Burn it')).toBeInTheDocument()
    expect(screen.getByText('Nevermind')).toBeInTheDocument()
  })

  it('confirm OK button is always btn-primary (tone does not change the button color)', async () => {
    const user = userEvent.setup()
    wrap((api) => api.confirm('Delete?', { tone: 'danger' }))
    await user.click(screen.getByText('trigger'))
    expect(screen.getByText('Confirm')).toHaveClass('btn-primary')
    expect(screen.getByText('Confirm')).not.toHaveClass('btn-danger')
  })

  it('prompt resolves the typed string on OK click', async () => {
    const user = userEvent.setup()
    wrap((api) => api.prompt('Name'))
    await user.click(screen.getByText('trigger'))
    const input = screen.getByRole('textbox')
    // Click to focus before typing: avoids userEvent dropping the first character
    // right as the modal opens / before autofocus settles (occasional "y-preset" on slow CI).
    await user.click(input)
    await user.type(input, 'my-preset')
    await user.click(screen.getByText('OK'))
    expect(screen.getByTestId('result')).toHaveTextContent('resolved:"my-preset"')
  })

  it('prompt resolves null on cancel', async () => {
    const user = userEvent.setup()
    wrap((api) => api.prompt('Name'))
    await user.click(screen.getByText('trigger'))
    await user.click(screen.getByText('Cancel'))
    expect(screen.getByTestId('result')).toHaveTextContent('resolved:null')
  })

  it('prompt defaultValue prefills the input', async () => {
    const user = userEvent.setup()
    wrap((api) => api.prompt('Name', { defaultValue: 'v3' }))
    await user.click(screen.getByText('trigger'))
    expect(screen.getByRole('textbox')).toHaveValue('v3')
  })

  it('prompt validate blocks submit and shows the error', async () => {
    const user = userEvent.setup()
    wrap((api) =>
      api.prompt('Name', { validate: (v) => (v.length < 3 ? 'At least 3 characters' : null) }),
    )
    await user.click(screen.getByText('trigger'))
    await user.type(screen.getByRole('textbox'), 'ab')
    await user.click(screen.getByText('OK'))
    expect(screen.getByText('At least 3 characters')).toBeInTheDocument()
    // dialog stays open after the error - result is still empty
    expect(screen.getByTestId('result')).toHaveTextContent('')
  })

  it('alert resolves on Got it click', async () => {
    const user = userEvent.setup()
    wrap((api) => api.alert('Done'))
    await user.click(screen.getByText('trigger'))
    expect(screen.getByText('Got it')).toBeInTheDocument()
    await user.click(screen.getByText('Got it'))
    expect(screen.getByTestId('result')).toHaveTextContent('resolved:')
  })

  it('alert has no cancel button', async () => {
    const user = userEvent.setup()
    wrap((api) => api.alert('Done'))
    await user.click(screen.getByText('trigger'))
    expect(screen.queryByText('Cancel')).not.toBeInTheDocument()
  })
})
