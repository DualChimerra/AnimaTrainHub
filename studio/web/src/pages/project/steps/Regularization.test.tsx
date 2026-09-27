import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ExcludeTags } from './Regularization'

// Upstream v0.20.2 (#454)'s fix for excluded-tag overflow. The same PR's
// SourceSegmented pill-radio test case does not apply to this fork: the
// source picker here is the fork's own segmented control, it doesn't use the
// pill-radio / vs-channel-radio styling, so there's no matching regression.
describe('regularization exclusion chips', () => {
  it('keeps a long natural-language tag inside a single-height chip', () => {
    const longTag = 'a single girl with very long pale lavender hair faces toward the viewer while the background extends across the entire frame'
    render(
      <ExcludeTags
        trainTags={[{ tag: longTag, count: 1 }]}
        excluded={new Set()}
        onToggle={vi.fn()}
      />,
    )

    const text = screen.getByText(longTag)
    const textContainer = text
    const chip = text.closest('button')

    expect(chip).toHaveClass('ds-chip', 'max-w-full', 'overflow-hidden', 'whitespace-nowrap')
    expect(textContainer).toHaveClass('min-w-0', 'truncate', 'text-left')
    expect(textContainer).toHaveAttribute('title', longTag)
  })
})
