import { useEffect, useState } from 'react'
import { api, type SystemStats as SystemStatsData } from '../api/client'
import { useEventStream } from '../lib/useEventStream'

/** Topbar system load, as in the mockup: GPU · VRAM · CPU · RAM, each a label,
 *  a value and a 20px bar, separated by full-height hairlines. A bar turns
 *  amber above 90%. */
function MeterItem({ label, value, pct, tooltip }: { label: string; value: string; pct: number; tooltip: string }) {
  const clamped = Math.min(100, Math.max(0, pct))
  return (
    <span title={tooltip}>
      <i className="ds-lbl">{label}</i>
      <b>{value}</b>
      <span className="ds-bar"><i className={clamped > 90 ? 'ds-hot' : undefined} style={{ width: `${clamped}%` }} /></span>
    </span>
  )
}

export default function SystemStats() {
  const [stats, setStats] = useState<SystemStatsData | null>(null)

  // mount 时拉一次冷启动 (避免空白等 2.5s 首个 SSE 事件)，之后纯靠后端
  // sampler 通过 SSE 推送。SSE 重连时 onOpen 也补一次冷启动，防漏。
  useEffect(() => {
    let cancelled = false
    api.systemStats().then((s) => {
      if (!cancelled) setStats(s)
    }).catch(() => {/* 首次失败：等 SSE 第一帧就行 */})
    return () => { cancelled = true }
  }, [])

  useEventStream(
    (evt) => {
      if (evt.type !== 'system_stats_updated') return
      const payload = evt.payload as SystemStatsData | undefined
      if (payload) setStats(payload)
    },
    {
      onOpen: () => {
        // SSE 重连：补一次冷启动；服务端 sampler 仍在跑，下次 tick 会自然推
        // 上来，但这一次显式 GET 让 UI 立刻刷新
        api.systemStats().then((s) => setStats(s)).catch(() => {})
      },
    },
  )

  if (!stats) return null

  const gpu0 = stats.gpu && stats.gpu.length > 0 ? stats.gpu[0] : null
  const ramPct = stats.ram_total_gb > 0 ? (stats.ram_used_gb / stats.ram_total_gb) * 100 : 0
  const vramPct = gpu0 && gpu0.vram_total_gb > 0 ? (gpu0.vram_used_gb / gpu0.vram_total_gb) * 100 : 0

  const gpuExtra = stats.gpu && stats.gpu.length > 1
    ? ` (+${stats.gpu.length - 1} more)`
    : ''
  const gpuTempText = gpu0?.temp_c != null ? ` · ${gpu0.temp_c}°C` : ''
  const gpuLabel = gpu0 ? `${gpu0.name}${gpuTempText}${gpuExtra}` : ''

  return (
    <div className="ds-sysload hidden md:flex shrink-0">
      {gpu0 && (
        <MeterItem
          label="GPU"
          value={`${Math.round(gpu0.util_pct)}%`}
          pct={gpu0.util_pct}
          tooltip={`GPU utilization · ${gpuLabel}`}
        />
      )}
      {gpu0 && (
        <MeterItem
          label="VRAM"
          value={`${gpu0.vram_used_gb.toFixed(1)} / ${Math.round(gpu0.vram_total_gb)}`}
          pct={vramPct}
          tooltip={`VRAM ${gpu0.vram_used_gb.toFixed(1)} / ${gpu0.vram_total_gb.toFixed(1)} GB (${vramPct.toFixed(0)}%) · ${gpuLabel}`}
        />
      )}
      <MeterItem
        label="CPU"
        value={`${Math.round(stats.cpu_pct)}%`}
        pct={stats.cpu_pct}
        tooltip={`CPU usage ${stats.cpu_pct.toFixed(1)}%`}
      />
      <MeterItem
        label="RAM"
        value={`${Math.round(stats.ram_used_gb)} / ${Math.round(stats.ram_total_gb)}`}
        pct={ramPct}
        tooltip={`RAM ${stats.ram_used_gb.toFixed(1)} / ${stats.ram_total_gb.toFixed(1)} GB (${ramPct.toFixed(0)}%)`}
      />
    </div>
  )
}
