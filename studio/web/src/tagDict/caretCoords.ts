/** Computes the caret's pixel coordinates inside an input/textarea (relative to the element's top-left, accounting for border / padding).
 *
 * The classic "mirror div" algorithm: build a hidden div copying all of the target element's relevant styles, fill
 * it with content up to the cursor position plus a marker span, and the span's offset is the caret's coordinates within the element.
 *
 * Adapted from component/textarea-caret-position (MIT).
 */

const PROPS_TO_COPY = [
  'direction',
  'boxSizing',
  'width',
  'height',
  'overflowX',
  'overflowY',
  'borderTopWidth',
  'borderRightWidth',
  'borderBottomWidth',
  'borderLeftWidth',
  'borderStyle',
  'paddingTop',
  'paddingRight',
  'paddingBottom',
  'paddingLeft',
  'fontStyle',
  'fontVariant',
  'fontWeight',
  'fontStretch',
  'fontSize',
  'fontSizeAdjust',
  'lineHeight',
  'fontFamily',
  'textAlign',
  'textTransform',
  'textIndent',
  'textDecoration',
  'letterSpacing',
  'wordSpacing',
  'tabSize',
  'MozTabSize',
] as const

export interface CaretCoords {
  /** Top y of the caret (relative to the element's top-left, including border + padding offset). */
  top: number
  /** Left x of the caret. */
  left: number
  /** Single-line line height (lets the caller know where the caret's bottom edge is, for sizing the popover offset). */
  height: number
}

export function getCaretCoordinates(
  element: HTMLInputElement | HTMLTextAreaElement,
  position: number,
): CaretCoords {
  const isInput = element.tagName.toLowerCase() === 'input'
  const div = document.createElement('div')
  document.body.appendChild(div)
  const style = div.style
  const computed = window.getComputedStyle(element)

  style.whiteSpace = isInput ? 'nowrap' : 'pre-wrap'
  if (!isInput) style.wordWrap = 'break-word'
  style.position = 'absolute'
  style.visibility = 'hidden'
  style.top = '0'
  style.left = '0'

  for (const prop of PROPS_TO_COPY) {
    // CSSStyleDeclaration index-signature compatibility issue: just write to it as a plain record
    ;(style as unknown as Record<string, string>)[prop] = computed[prop as keyof CSSStyleDeclaration] as string
  }

  // input: single line forces nowrap; whitespace must become nbsp so the div doesn't collapse it
  let pre = element.value.substring(0, position)
  if (isInput) pre = pre.replace(/\s/g, ' ')
  div.textContent = pre

  const span = document.createElement('span')
    // Uses the character at the cursor as the marker; if there's none (cursor at the end), fall back to '.'
  span.textContent = element.value.substring(position) || '.'
  div.appendChild(span)

  const lineHeight = parseInt(computed.lineHeight || '0', 10) || parseInt(computed.fontSize || '16', 10)
  const result: CaretCoords = {
    top: span.offsetTop + parseInt(computed.borderTopWidth || '0', 10),
    left: span.offsetLeft + parseInt(computed.borderLeftWidth || '0', 10),
    height: lineHeight,
  }

  document.body.removeChild(div)
  return result
}
