import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import PageHead from './ds/PageHead'
import TaskLogDrawer, { type LogSource } from './TaskLogDrawer'

interface Props {
  idx: number | string
  title: string
  subtitle?: ReactNode
  /** Caption above the title; defaults to "STEP N" for a numeric idx. */
  eyebrow?: string
  /** Green eyebrow (the mockup's version steps). On unless set to false. */
  accentEyebrow?: boolean
  actions?: ReactNode
  topRight?: ReactNode
  children: ReactNode
  /** Full-width block between the head and the content (tool summary,
   *  filters); does not scroll with the content. */
  belowHeader?: ReactNode
  /** Page task logs (issue #251 shared drawer); falsy entries are dropped and
   *  nothing renders when all are empty. */
  logSources?: Array<LogSource | null | undefined | false>
  /** Phone only: let the whole page scroll instead of the content pane. For pages
   *  whose content is a plain list; leave it off where the content is a
   *  virtualised grid that needs a fixed-height box. */
  mobilePageScroll?: boolean
  /** Bar pinned under the content, above the job log strip (e.g. the tag
   *  editor's unsaved-changes bar). */
  footer?: ReactNode
}

/** Frame of a version step, as in the mockup: page head with the step
 *  eyebrow and tools, an optional block under it, the content filling the
 *  rest of the height, and the job log strip at the bottom. */
export default function StepShell({ idx, title, subtitle, eyebrow, accentEyebrow = true, actions, topRight, children, belowHeader, logSources, mobilePageScroll, footer }: Props) {
  const { t } = useTranslation()
  const derivedEyebrow = eyebrow ?? (typeof idx === 'number' || /^\d+$/.test(String(idx)) ? t('steps.stepN', { n: idx }) : undefined)
  const tools = actions || topRight ? <>{topRight}{actions}</> : undefined
  return (
    <div className={`step-shell fade-in flex flex-col h-full relative ${mobilePageScroll ? 'is-page-scroll' : ''}`}>
      <PageHead accent={accentEyebrow} eyebrow={derivedEyebrow} title={title} subtitle={subtitle} tools={tools} />
      {belowHeader}
      <div className="step-shell-body flex-1 min-h-0 flex flex-col overflow-hidden" style={{ padding: '0 24px 18px' }}>
        {children}
      </div>
      {footer}
      {logSources && <TaskLogDrawer sources={logSources} />}
    </div>
  )
}
