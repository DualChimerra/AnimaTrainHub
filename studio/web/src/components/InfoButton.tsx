import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'

/**
 * Click-toggle popover. Attaches help text to a field name / section title /
 * card corner without taking up space in the main UI; only shows when the user needs it.
 *
 * Behavior:
 * - Clicking the trigger toggles it; clicking outside / Esc closes it
 * - aria-expanded for screen readers; the trigger carries its own aria-label
 * - The trigger is an SVG i-in-circle icon (previously used the Unicode ⓘ
 *   character, whose vertical position was inconsistent across fonts and
 *   didn't align with the baseline of adjacent text; SVG with a fixed viewBox centers it precisely)
 *
 * Deliberately not done (YAGNI; extend later if another trigger need shows up):
 * - No hover trigger support (not mobile-friendly)
 * - No dynamic placement calculation (always bottom-left; viewport overflow handled via max-width)
 */
interface InfoButtonProps {
  children: ReactNode
  ariaLabel?: string
}

function InfoIcon() {
  return (
    <svg
      width={12}
      height={12}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx={8} cy={8} r={6.5} />
      <line x1={8} y1={7.5} x2={8} y2={11.5} />
      <circle cx={8} cy={4.8} r={0.6} fill="currentColor" stroke="none" />
    </svg>
  )
}

export function InfoButton({ children, ariaLabel }: InfoButtonProps) {
  const { t } = useTranslation()
  const resolvedAriaLabel = ariaLabel ?? t('infoButton.label')
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <span ref={wrapRef} className="info-btn-anchor">
      <button
        type="button"
        className="info-btn-trigger"
        onClick={(e) => {
          // stopPropagation: avoids triggering the outer toggle when placed inside a <summary> / clickable row
          e.stopPropagation()
          setOpen((v) => !v)
        }}
        aria-expanded={open}
        aria-label={resolvedAriaLabel}
      >
        <InfoIcon />
      </button>
      {open && (
        <div className="info-btn-panel" role="dialog">
          {children}
        </div>
      )}
    </span>
  )
}
