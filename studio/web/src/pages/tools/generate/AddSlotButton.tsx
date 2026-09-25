import type { ReactNode } from 'react'

/** Sidebar 里「+ 添加 …」按钮（LoRA 槽 / XY 的「+ 添加 Y 轴」共用）：
 *  macket 的 dashed chip，占满整行。 */
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
