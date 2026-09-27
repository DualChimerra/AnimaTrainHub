import type { ReactNode } from 'react'

interface Props {
  title: string
  subtitle?: string
  /** Caption above the title (mono, uppercase, gray). E.g. "GENERATE" / "STEP 5 · …". */
  eyebrow?: string
  /** Renders the eyebrow in the accent color instead of gray (e.g. to highlight the current step). */
  accentEyebrow?: boolean
  /** Tab nav bar; when tabs is passed, subtitle is not rendered (tabs take the description's place). */
  tabs?: ReactNode
  actions?: ReactNode
  /** Top-right slot -- a standalone position aligned with the top of the
   *  title (outside the actions row). For auxiliary nav like PhaseHeaderNav
   *  that must stay pinned to the top-right. */
  topRight?: ReactNode
  sticky?: boolean
}

export default function PageHeader({ title, subtitle, eyebrow, accentEyebrow, tabs, actions, topRight, sticky }: Props) {
  return (
    <div className={`page-header px-6 pt-5 pb-4 bg-canvas border-b border-subtle ${sticky ? 'sticky top-0 z-[5]' : 'relative'}`}>
      {topRight && (
        <div className="page-header-topright absolute top-3 right-6 z-[1]">{topRight}</div>
      )}
      <div className="page-header-row flex items-end gap-4 flex-wrap">
        <div className="flex-1 min-w-0">
          {eyebrow && (
            <div className={`caption mb-1 ${accentEyebrow ? 'text-accent' : ''}`}>{eyebrow}</div>
          )}
          <h1 className="m-0 text-2xl font-semibold tracking-[-0.025em] leading-8">{title}</h1>
          {/* tabs sit below the main title in place of subtitle; the two are mutually exclusive (tabs win). */}
          {tabs ? (
            <div className="mt-3">{tabs}</div>
          ) : (
            subtitle && (
              <p className="mt-1 text-fg-tertiary text-sm max-w-[760px] m-0">{subtitle}</p>
            )
          )}
        </div>
        {actions && (
          <div className="page-header-actions flex gap-2 items-center">{actions}</div>
        )}
      </div>
    </div>
  )
}
