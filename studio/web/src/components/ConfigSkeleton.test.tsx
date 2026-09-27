/**
 * ConfigSkeleton unit tests.
 *
 * Verifies:
 *   - the card variant's outer element is a <section>, the flat variant's is a <div>
 *   - role="status" + aria-label + sr-only text are consistent
 *   - the groups prop controls the number of groups, row count controls rows per group
 *   - label passes through to aria + sr-only
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import ConfigSkeleton from './ConfigSkeleton'

describe('ConfigSkeleton', () => {
  it('renders card variant with section + per-group border', () => {
    const { container } = render(<ConfigSkeleton groups={[2, 3]} />)
    expect(container.querySelector('section')).toBeTruthy()
    // 2 groups x 1 bordered card each
    const cards = container.querySelectorAll('.border.border-subtle.bg-surface')
    expect(cards.length).toBe(2)
  })

  it('renders flat variant with div and no per-group border', () => {
    const { container } = render(<ConfigSkeleton variant="flat" groups={[2, 3]} />)
    expect(container.querySelector('section')).toBeFalsy()
    expect(container.querySelector('div[role="status"]')).toBeTruthy()
    expect(container.querySelectorAll('.border.border-subtle.bg-surface').length).toBe(0)
  })

  it('uses label for aria-label and sr-only text', () => {
    render(<ConfigSkeleton label="Loading training config" />)
    const status = screen.getByRole('status')
    expect(status.getAttribute('aria-label')).toBe('Loading training config')
    expect(screen.getByText('Loading training config...')).toBeInTheDocument()
  })

  it('defaults to 4 groups when groups prop is omitted', () => {
    const { container } = render(<ConfigSkeleton />)
    // section > 4 children groups + 1 sr-only span = 5 direct children
    const section = container.querySelector('section')
    expect(section).toBeTruthy()
    // 4 group cards
    expect(container.querySelectorAll('.border.border-subtle.bg-surface').length).toBe(4)
  })

  it('renders the requested row count per group', () => {
    const { container } = render(<ConfigSkeleton groups={[1, 5]} />)
    const groups = container.querySelectorAll('section > div')
    expect(groups.length).toBe(2)
    // group 1 has 1 row = 1 label bar + 1 input bar = 2 divs inside the row container
    // group 2 has 5 rows = 5 row containers (2 bars each)
    const rowsInGroup1 = groups[0].querySelectorAll('.flex.flex-col.gap-1')
    const rowsInGroup2 = groups[1].querySelectorAll('.flex.flex-col.gap-1')
    expect(rowsInGroup1.length).toBe(1)
    expect(rowsInGroup2.length).toBe(5)
  })
})
