import { useLayoutEffect, type RefObject } from 'react'

/** Grows a textarea to fit its content with no upper bound; the minimum
 * height is whatever the `rows` attribute produces initially.
 *
 * First resets height to auto so the browser falls back to the rows height,
 * then sets it to scrollHeight -- when content is short, scrollHeight is
 * floored by clientHeight, which lands exactly back at the rows minimum.
 * Pair this with resize-none overflow-hidden in className (manual drag-resize
 * would just get overwritten on the next keystroke anyway, so it's disabled
 * outright; hidden prevents a scrollbar flash during the height change).
 *
 * When the element sits inside a display:none container (e.g. an inactive
 * tab pane), scrollHeight is 0 -- bail out immediately rather than
 * collapsing the height to 0. ResizeObserver will fire and recompute once
 * it becomes visible again (size 0 -> actual), so the textarea doesn't stay
 * collapsed after switching back to that tab.
 */
export function useAutoGrowTextarea(
  ref: RefObject<HTMLTextAreaElement>,
  value: string,
): void {
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const fit = () => {
      // offsetParent===null => self or an ancestor is display:none, scrollHeight is unreliable, leave the height alone
      if (el.offsetParent === null) return
      el.style.height = 'auto'
      // scrollHeight excludes the border; add it back for box-sizing:
      // border-box, otherwise a scrollbar appears 2px short each time
      const border = el.offsetHeight - el.clientHeight
      // Add one extra line height: leaves room for the token-count badge in
      // the bottom-right corner (so it doesn't overlap content flush with the bottom)
      const lineHeight = parseFloat(getComputedStyle(el).lineHeight) || 20
      el.style.height = `${el.scrollHeight + border + lineHeight}px`
    }
    fit()
    // Watch for size changes: recomputes when a tab switch makes the element go from hidden to visible, and also on width changes (which affect height via line wrapping)
    const ro = new ResizeObserver(fit)
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref, value])
}
