/** Type-size density (dark theme has been removed, the UI is fixed to a light color scheme).
 *
 * Finalized in the redesign prototype: the UI is permanently density-tight
 * (compact), and the density toggle has been removed (design iteration
 * chat2 #8: "Сделай по умолчанию плотный вариант интерфейса,
 * переключения плотности также не должно быть" -- make the compact
 * variant the default, and there should be no density toggle either).
 *
 * This module's only job is, at boot (`initTheme()` in main.tsx), to clean
 * up any leftover legacy classes and pin density-tight.
 */

export type Density = 'tight' | 'default' | 'loose'

export function applyDensity(d: Density): void {
  const root = document.documentElement
  root.classList.remove('density-tight', 'density-loose')
  if (d === 'tight') root.classList.add('density-tight')
  else if (d === 'loose') root.classList.add('density-loose')
}

// ── boot ───────────────────────────────────────────────────────────────────
export function initTheme(): void {
  // Force the light color scheme: strip any leftover legacy dark-theme class.
  document.documentElement.classList.remove('theme-dark')
  // Density is fixed to compact (finalized in the prototype, no toggle).
  applyDensity('tight')
}
