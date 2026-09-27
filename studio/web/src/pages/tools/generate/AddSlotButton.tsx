import type { ReactNode } from 'react'

/** The "+ Add..." button in the sidebar (shared by LoRA slots / XY's "+ Add Y axis"):
 *  the mockup's dashed chip, fills the whole row. */
export default function AddSlotButton({
  onClick, children,
}: {
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="ds-chip-add"
      style={{ width: '100%', justifyContent: 'center' }}
    >
      {children}
    </button>
  )
}
