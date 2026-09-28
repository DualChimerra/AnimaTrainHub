import type { ReactNode } from 'react'
import HelpTip from './ds/HelpTip'

/** Help text next to a field name / section title / card corner. Kept as a
 *  named export for existing callers; it is the same "?" tip used everywhere. */
interface InfoButtonProps {
  children: ReactNode
  ariaLabel?: string
}

export function InfoButton({ children, ariaLabel }: InfoButtonProps) {
  return <HelpTip label={ariaLabel}>{children}</HelpTip>
}

export default InfoButton
