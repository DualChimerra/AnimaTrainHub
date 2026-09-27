import i18n from '../i18n'
import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import '../i18n'
import VersionStatusBadge from './VersionStatusBadge'

describe('VersionStatusBadge', () => {
  it('renders nothing when status is null', () => {
    const { container } = render(<VersionStatusBadge status={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders nothing when status is undefined', () => {
    const { container } = render(<VersionStatusBadge status={undefined} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders preparing with warn badge', () => {
    const { container } = render(<VersionStatusBadge status="preparing" />)
    const badge = container.querySelector('.badge')
    expect(badge).not.toBeNull()
    expect(badge?.className).toContain('badge-warn')
  })

  it('renders training with accent badge + running dot', () => {
    const { container } = render(<VersionStatusBadge status="training" />)
    const badge = container.querySelector('.badge')
    expect(badge?.className).toContain('badge-accent')
    expect(container.querySelector('.dot.dot-running')).not.toBeNull()
  })

  it('renders completed with ok badge', () => {
    const { container } = render(<VersionStatusBadge status="completed" />)
    expect(container.querySelector('.badge')?.className).toContain('badge-ok')
  })

  it('renders failed with err badge', () => {
    const { container } = render(<VersionStatusBadge status="failed" />)
    expect(container.querySelector('.badge')?.className).toContain('badge-err')
  })

  it('renders canceled with neutral badge', () => {
    const { container } = render(<VersionStatusBadge status="canceled" />)
    expect(container.querySelector('.badge')?.className).toContain('badge-neutral')
  })

  // v12 - shows a phase suffix while preparing (project card "Preparing - Tag edit").
  // Asserts against the composed i18n value rather than a literal: this test used to
  // hardcode the editing phase's old copy ("Tagging"); once that copy was renamed it
  // stayed red forever even though the component itself was correct.
  it('appends phase suffix when preparing', () => {
    const { container } = render(
      <VersionStatusBadge status="preparing" phase="editing" />
    )
    expect(container.querySelector('.badge')?.textContent).toBe(
      `${i18n.t('versionStatus.preparing')} · ${i18n.t('versionPhase.editing')}`
    )
  })

  it('shows suffix for optional phases too (PR #265 review)', () => {
    const cases = [
      ['preprocessing', 'Preparing · Preprocess'],
      ['regularizing', 'Preparing · Regularization'],
    ] as const
    for (const [phase, text] of cases) {
      const { container } = render(
        <VersionStatusBadge status="preparing" phase={phase} />
      )
      expect(container.querySelector('.badge')?.textContent).toBe(text)
    }
  })

  it('ignores phase when status is not preparing', () => {
    const { container } = render(
      <VersionStatusBadge status="training" phase="editing" />
    )
    expect(container.querySelector('.badge')?.textContent).toBe('Training')
  })
})
