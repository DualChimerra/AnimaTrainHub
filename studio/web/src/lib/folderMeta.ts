/** Parses metadata out of a dataset folder name.
 *
 *  SYNC WITH `runtime/training/dataset.py:ImageDataset._parse_folder_meta` --
 *  the two parsers must stay in agreement, otherwise the frontend's display /
 *  upscale target diverges from the trainer's actual behavior.
 *
 *  Grammar `[Npx_][R_]label` (token order, both prefixes optional):
 *   - `\d+px` resolution prefix (snapped to a multiple of 64 + clamped to [256,4096])
 *   - `\d+` Kohya repeat
 *   - the remainder is the label
 */
export interface FolderMeta {
  /** Resolution specified by the px prefix; null = no prefix (fans out over the config's resolution list). */
  reso: number | null
  repeat: number
  label: string
}

export function parseFolderMeta(name: string): FolderMeta {
  let reso: number | null = null
  let repeat = 1
  let rest = name
  let m = rest.match(/^(\d+)px_(.*)$/)
  if (m) {
    reso = Math.max(256, Math.min(4096, Math.round(parseInt(m[1], 10) / 64) * 64))
    rest = m[2]
  }
  m = rest.match(/^(\d+)_(.*)$/)
  if (m) {
    repeat = Math.max(parseInt(m[1], 10), 1)
    rest = m[2]
  }
  return { reso, repeat, label: rest }
}
