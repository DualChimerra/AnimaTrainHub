/**
 * ConfigSkeleton -- loading skeleton for the Schema form.
 *
 * Extracted from Train.tsx's `ConfigSkeleton` and Presets.tsx's `SkeletonGroups`.
 * Both share nearly the same structure (N groups x M rows of "label bar + input
 * bar"), just with a different look:
 *   - Train page: each group has a border + bg-surface card, stronger sense of depth (variant='card')
 *   - Presets page: flat layout to save space (variant='flat')
 *
 * Also fixes the a11y issue Designer P2 flagged: putting role="status" on the
 * outer element doesn't announce "loading complete" on unmount. The sr-only
 * text stays, but callers returning this component should only mount it while
 * `loading === true` (unmounting implicitly means loading finished); a caller
 * page that needs an explicit "loading complete" announcement should mount its
 * own aria-live region at the page's top level.
 */

interface ConfigSkeletonProps {
  /** sr-only / aria-label text, defaults to "Loading config…" */
  label?: string
  /** Row count per group, defaults to [5, 6, 4, 5] */
  groups?: number[]
  /** 'card' = each group has a card border (Train style); 'flat' = flat (Presets style). Defaults to 'card' */
  variant?: 'card' | 'flat'
}

const DEFAULT_GROUPS = [5, 6, 4, 5]

export default function ConfigSkeleton({
  label = 'Loading config…',
  groups = DEFAULT_GROUPS,
  variant = 'card',
}: ConfigSkeletonProps) {
  const isCard = variant === 'card'
  const Wrapper = isCard ? 'section' : 'div'
  const wrapperClass = isCard
    ? 'flex-1 min-h-0 overflow-y-auto pr-1 space-y-3'
    : 'flex flex-col gap-3 animate-pulse'
  const groupClass = isCard
    ? 'animate-pulse rounded-md border border-subtle bg-surface p-3.5'
    : 'flex flex-col gap-2'
  const titleBarClass = isCard
    ? 'h-3.5 w-32 rounded-sm bg-sunken mb-2.5'
    : 'h-3 w-24 rounded-sm bg-sunken opacity-60'
  const rowsContainerClass = isCard
    ? 'flex flex-col gap-2'
    : 'flex flex-col gap-1.5'
  const rowClass = isCard
    ? 'flex flex-col gap-1'
    : 'flex flex-col gap-0.5'
  const labelBarClass = isCard
    ? 'h-2.5 w-24 rounded-sm bg-sunken opacity-70'
    : 'h-2 w-20 rounded-sm bg-sunken opacity-50'
  const inputBarClass = isCard
    ? 'h-7 rounded-sm bg-canvas border border-subtle'
    : 'h-[26px] rounded-sm border border-subtle bg-canvas'

  return (
    <Wrapper className={wrapperClass} role="status" aria-label={label}>
      {groups.map((rows, gi) => (
        <div key={gi} className={groupClass}>
          <div className={titleBarClass} />
          <div className={rowsContainerClass}>
            {Array.from({ length: rows }).map((_, ri) => (
              <div key={ri} className={rowClass}>
                <div className={labelBarClass} />
                <div className={inputBarClass} />
              </div>
            ))}
          </div>
        </div>
      ))}
      <span className="sr-only">{label}...</span>
    </Wrapper>
  )
}
