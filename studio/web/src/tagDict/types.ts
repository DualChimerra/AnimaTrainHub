/** Autocomplete tag list — types.
 *
 * Data flow: the backend's GET /api/tag-dictionary/data returns
 * `{tags, meta}`; the frontend keeps the list in memory for autocomplete.
 */

export interface TagDictMeta {
  source_name: string
  source_url: string
  entry_count: number
  downloaded_at: number
  kind: 'default' | 'user'
}

export interface TagDictPayload {
  /** Tags in file order (the default source is sorted by popularity). */
  tags: string[]
  meta: TagDictMeta
}

export interface TagDictMetaResponse {
  loaded: boolean
  meta: TagDictMeta | null
}

/** One autocomplete candidate. */
export interface TagSuggestion {
  tag: string
  matchType: 'prefix' | 'substring'
}

export type TagDictStatus = 'idle' | 'loading' | 'ready' | 'error' | 'empty'
