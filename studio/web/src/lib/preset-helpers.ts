// preset-helpers.ts -- preset-related utilities shared by the Presets page
// and the Train page.
// Split out so Train.tsx's inline "new preset" form doesn't need its own copy.
import type { ConfigData, SchemaResponse } from '../api/client'

/** Legal preset-name characters: letters / digits / _ / -. The
 * /api/presets/<name> route validates names against this same set. */
export const PRESET_NAME_RE = /^[A-Za-z0-9_\-]+$/

const DESC_KEY = 'studio.preset.descriptions'

/** The preset subtitle ("description") is stored in localStorage, indexed by
 * name. The backend preset schema has no description field; this is a
 * purely frontend-display helper string. */
export function loadPresetDescriptions(): Record<string, string> {
  try {
    const raw = localStorage.getItem(DESC_KEY)
    return raw ? (JSON.parse(raw) as Record<string, string>) : {}
  } catch {
    return {}
  }
}

export function savePresetDescriptions(d: Record<string, string>) {
  try {
    localStorage.setItem(DESC_KEY, JSON.stringify(d))
  } catch {
    /* ignore quota errors */
  }
}

/** Extracts the default-value dict from the schema. Used as the form's initial content when creating a new preset. */
export function defaultsFromSchema(schema: SchemaResponse | null): ConfigData {
  if (!schema) return {}
  const out: ConfigData = {}
  for (const [name, prop] of Object.entries(schema.schema.properties)) {
    if (prop.default !== undefined) out[name] = prop.default
  }
  return out
}

/** Returns `base` directly if it doesn't collide with any name in
 * `existing`; otherwise tries `base_1`, `base_2`, ... and returns the first
 * one that's free.
 *
 * Used for auto-naming in the project page's one-click "new preset" -- base
 * looks like `<project_slug>_<version_label>`; on a name collision (the user
 * already created one for the same version before) an underscore suffix is
 * appended, staying compatible with PRESET_NAME_RE. Capped at 999 as a
 * safety net; 999 same-name collisions is unrealistic, and past that it
 * returns a timestamp-suffixed name to avoid an infinite loop.
 */
export function generateUniquePresetName(
  base: string,
  existing: ReadonlyArray<{ name: string }>,
): string {
  const taken = new Set(existing.map((p) => p.name))
  if (!taken.has(base)) return base
  for (let i = 1; i < 1000; i++) {
    const candidate = `${base}_${i}`
    if (!taken.has(candidate)) return candidate
  }
  return `${base}_${Date.now()}`
}
