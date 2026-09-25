/** 画幅预设卡片（8 个常用比例 + Swap 横竖切换）。
 *
 * 每个卡片：glyph（按比例画的小矩形）+ 比例数字 + 名称 + W×H。
 * 选中态：accent 色描边 + 比例数字 accent 色。
 * grid-cols-4 两行排，4×2=8 张；sidebar 420px 内塞得下。
 *
 * Swap 按钮独立一个：物理交换 W↔H，命中预设则 aspect 跟着切（精确匹配
 * w/h 找当前 chip）；找不到时 aspect='custom'。
 */

export interface AspectPreset {
  /** 唯一 key，也是显示在大字号位置的比例标签 */
  ratio: string
  /** 'Square' / 'Landscape' / 'Portrait' */
  kind: 'Square' | 'Landscape' | 'Portrait'
  w: number
  h: number
}

export const ASPECT_PRESETS: AspectPreset[] = [
  { ratio: '3:2',   kind: 'Landscape', w: 1254, h: 836  },
  { ratio: '1:1',   kind: 'Square',    w: 1024, h: 1024 },
  { ratio: '4:5',   kind: 'Portrait',  w: 915,  h: 1144 },
  { ratio: '7:9',   kind: 'Portrait',  w: 896,  h: 1152 },
  { ratio: '3:4',   kind: 'Portrait',  w: 768,  h: 1024 },
  { ratio: '2:3',   kind: 'Portrait',  w: 832,  h: 1216 },
  { ratio: '9:16',  kind: 'Portrait',  w: 768,  h: 1344 },
  { ratio: '5:12',  kind: 'Portrait',  w: 640,  h: 1536 },
]

export type AspectName = string  // 用 "ratio" 做 name；自定义 = 'custom'

/** 给定 w/h 反查匹配的预设 ratio；不命中返回 'custom'。 */
export function aspectFromDimensions(w: number, h: number): AspectName {
  for (const p of ASPECT_PRESETS) {
    if (p.w === w && p.h === h) return p.ratio
  }
  return 'custom'
}

/** Aspect presets as a row of pills; the active one is dark, the size shows on hover. */
export default function AspectChips({
  aspect, onPick,
}: {
  aspect: AspectName
  onPick: (ratio: AspectName, w?: number, h?: number) => void
}) {
  return (
    <div className="ds-pills" style={{ flexWrap: 'wrap' }}>
      {ASPECT_PRESETS.map((p) => (
        <button
          key={p.ratio}
          type="button"
          className={`ds-pill${aspect === p.ratio ? ' ds-is-active' : ''}`}
          style={{ height: 26, padding: '0 10px', fontFamily: 'var(--mono)', fontSize: 11.5 }}
          title={`${p.kind} · ${p.w}×${p.h}`}
          aria-pressed={aspect === p.ratio}
          onClick={() => onPick(p.ratio, p.w, p.h)}
        >
          {p.ratio}
        </button>
      ))}
    </div>
  )
}
