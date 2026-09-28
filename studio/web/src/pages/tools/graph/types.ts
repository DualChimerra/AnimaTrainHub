/** Graph — data shapes shared by the page, its views and the pure model. */

export type ParamType = 'list' | 'number' | 'bool' | 'text'
export type RunStatus = 'planned' | 'training' | 'done' | 'stopped'
export const RUN_STATUSES: RunStatus[] = ['planned', 'training', 'done', 'stopped']

/** How a list / bool option maps onto the training config. `apply` is what
 *  gets written when a training is created from the Graph; `match` (defaults
 *  to `apply`) recognises a queue task's frozen config. */
export interface ConfigLink {
  apply: Record<string, unknown>
  match?: Record<string, unknown>
}

export interface ParamOption {
  id: string
  /** list / bool options */
  label?: string
  /** number options */
  value?: number
  config?: ConfigLink
  archived?: boolean
}

export interface Param {
  id: string
  name: string
  type: ParamType
  options: ParamOption[]
  description?: string
  hidden?: boolean
  archived?: boolean
  /** Shown / compared only while another parameter has one of these options. */
  condition?: { param: string; options: string[] }
  /** number params: the config key the value is written to / read from.
   *  `list: true` = the key holds a list (resolution) and the value is its only item. */
  config?: { key: string; list?: boolean }
}

/** list / bool / → option id; number → number; text → string. Absent = not set. */
export type RunValue = string | number
export type Values = Record<string, RunValue>

export interface GImage {
  id: number
  run_id: number
  width: number | null
  height: number | null
  step: number | null
  prompt: string
  seed: number | null
  comment: string
  rating: number | null
  source: string
  sort: number
  created_at: number
}

export interface Run {
  id: number
  board_id: number
  name: string
  values: Values
  status: RunStatus
  notes: string
  link: string
  favorite: boolean
  rating: number | null
  best_step: number | null
  tags: string[]
  task_id: number | null
  project_id: number | null
  version_id: number | null
  created_at: number
  updated_at: number
  images: GImage[]
}

export interface SavedView {
  id: string
  name: string
  state: ViewState
}

export interface Board {
  id: number
  name: string
  note: string
  params: Param[]
  views: SavedView[]
  created_at: number
  updated_at: number
}

export interface BoardSummary {
  id: number
  name: string
  note: string
  run_count: number
  image_count: number
  created_at: number
  updated_at: number
}

export type ViewMode = 'cards' | 'table' | 'matrix' | 'web'

export interface ParamFilter {
  /** list / bool option ids, or numbers for number params. */
  values?: RunValue[]
  /** include cards where the parameter is not set */
  unset?: boolean
  min?: number | null
  max?: number | null
  text?: string
}

export interface SortKey {
  /** param id, or one of the built-ins: _name, _created, _rating, _status, _samples */
  key: string
  dir: 'asc' | 'desc'
}

export interface ViewState {
  mode: ViewMode
  filters: Record<string, ParamFilter>
  search: string
  statuses: RunStatus[]
  samples: 'any' | 'with' | 'without'
  favorites: boolean
  minRating: number
  groupBy: string[]
  sort: SortKey[]
  shownParams: string[]
  thumb: number
  /** Show empty places for untried combinations of the filtered values. */
  gaps: boolean
  collapsed: string[]
  matrixX: string | null
  matrixY: string | null
  webColor: string | null
}

/** A queue training task as the Graph sees it (read-only). */
export interface GraphTask {
  id: number
  name: string
  status: string
  created_at: number
  started_at: number | null
  finished_at: number | null
  project_id: number | null
  version_id: number | null
  project_title: string | null
  version_label: string | null
  note: string
  has_config: boolean
  config: Record<string, unknown>
}

export interface TaskSampleInfo {
  filename: string
  mtime: number
  size: number
  step: number | null
  epoch: number | null
  prompt?: string
  seed?: number | null
}
