import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'

/** Time formatting shared by the queue pages: clock time (with the date when
 *  it is not today), a full date-time, durations and "ago", all in the UI
 *  language. */
export function useQueueFormat() {
  const { t, i18n } = useTranslation()
  const lang = i18n.language
  return useMemo(() => {
    const clock = (ts: number) => {
      const d = new Date(ts * 1000)
      const today = d.toDateString() === new Date().toDateString()
      return new Intl.DateTimeFormat(lang, today
        ? { hour: '2-digit', minute: '2-digit' }
        : { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }).format(d)
    }
    const stamp = (ts: number | null | undefined) => (ts
      ? new Intl.DateTimeFormat(lang, {
        day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit',
      }).format(new Date(ts * 1000))
      : '—')
    const dur = (sec: number) => {
      const s = Math.max(0, Math.round(sec))
      if (s < 60) return t('queue.dur.s', { n: s })
      const m = Math.round(s / 60)
      if (m < 60) return t('queue.dur.m', { n: m })
      return t('queue.dur.hm', { h: Math.floor(m / 60), m: String(m % 60).padStart(2, '0') })
    }
    const rtf = new Intl.RelativeTimeFormat(lang, { numeric: 'auto' })
    const ago = (ts: number) => {
      const s = Math.max(0, Date.now() / 1000 - ts)
      if (s < 60) return rtf.format(0, 'second')
      if (s < 3600) return rtf.format(-Math.floor(s / 60), 'minute')
      if (s < 86400) return rtf.format(-Math.floor(s / 3600), 'hour')
      return rtf.format(-Math.floor(s / 86400), 'day')
    }
    const bytes = (n: number) => {
      const u = t('queueDetail.unitsBytes', { returnObjects: true }) as unknown
      const units = Array.isArray(u) && u.length === 4 ? (u as string[]) : ['B', 'KB', 'MB', 'GB']
      if (n < 1024) return `${n} ${units[0]}`
      if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} ${units[1]}`
      if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(0)} ${units[2]}`
      return `${(n / 1024 ** 3).toFixed(2)} ${units[3]}`
    }
    return { clock, stamp, dur, ago, bytes }
  }, [t, lang])
}

export type QueueFormat = ReturnType<typeof useQueueFormat>
