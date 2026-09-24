import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type CaptionSnapshot } from '../api/client'
import { useDialog } from './Dialog'
import { useToast } from './Toast'

interface Props {
  pid: number
  vid: number
  /** 待保存数：0 = 无 dirty。 */
  dirtyCount: number
  /** 触发保存：父组件提供 commit 实现（已经计算 diff）。 */
  onSave: () => Promise<void>
  /** Drop the local edits and go back to what is on disk. */
  onDiscard: () => void
  /** 触发还原后，父组件需要重新拉缓存。 */
  onAfterRestore: () => Promise<void>
}

function fmtTime(epoch: number): string {
  const d = new Date(epoch * 1000)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}

function fmtSize(b: number): string {
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1024 / 1024).toFixed(1)} MB`
}

/** Bottom bar of the tag editor (mockup TagEdit): green "unsaved" strip with
 *  discard / save while there are local edits, a quiet strip otherwise.
 *  Restore points (auto-created on every save) open above it. */
export default function SaveBar({
  pid, vid, dirtyCount, onSave, onDiscard, onAfterRestore,
}: Props) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const [open, setOpen] = useState(false)
  const [items, setItems] = useState<CaptionSnapshot[]>([])
  const [busyId, setBusyId] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const dirty = dirtyCount > 0

  const refresh = useCallback(async () => {
    try { setItems(await api.listCaptionSnapshots(pid, vid)) }
    catch (e) { toast(String(e), 'error') }
  }, [pid, vid, toast])

  useEffect(() => { if (open) void refresh() }, [open, refresh])

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const save = async () => {
    setSaving(true)
    try { await onSave() } finally { setSaving(false) }
  }

  const discard = async () => {
    if (!(await confirm(t('saveBar.confirmDiscard', { n: dirtyCount }), { tone: 'danger', okText: t('saveBar.discard') }))) return
    onDiscard()
  }

  const restore = async (sid: string) => {
    if (!(await confirm(t('saveBar.confirmRestore'), { tone: 'warn', okText: t('common.restore') }))) return
    setBusyId(sid)
    try {
      const r = await api.restoreCaptionSnapshot(pid, vid, sid)
      toast(t('saveBar.restoreDone', { n: r.written, m: r.removed_old }), 'success')
      setOpen(false)
      await onAfterRestore()
    } catch (e) { toast(String(e), 'error') }
    finally { setBusyId(null) }
  }

  const del = async (sid: string) => {
    if (!(await confirm(t('saveBar.confirmDeletePoint', { id: sid }), { tone: 'danger', okText: t('common.delete') }))) return
    setBusyId(sid)
    try { await api.deleteCaptionSnapshot(pid, vid, sid); await refresh() }
    catch (e) { toast(String(e), 'error') }
    finally { setBusyId(null) }
  }

  return (
    <div
      ref={ref}
      className="ds-logbar"
      style={{ position: 'relative', ...(dirty ? { background: 'var(--green-soft)', borderTopColor: 'var(--green-line)', color: 'var(--green-text)' } : null) }}
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
        <path d="M12 4v8" /><path d="M6.3 7.3a8 8 0 1 0 11.4 0" />
      </svg>
      <span style={{ fontWeight: dirty ? 500 : undefined }}>
        {dirty ? t('saveBar.unsaved', { count: dirtyCount }) : t('saveBar.allSaved')}
      </span>
      <span style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
        <button
          type="button"
          className="ds-ctl ds-ghost"
          style={{ height: 26, color: 'inherit' }}
          onClick={() => setOpen(!open)}
          aria-expanded={open}
        >
          {t('saveBar.restorePoints')} {open ? '↓' : '↑'}
        </button>
        {dirty && (
          <button type="button" className="ds-ctl ds-ghost" style={{ height: 26, color: 'inherit' }} onClick={() => void discard()} disabled={saving}>
            {t('saveBar.discard')}
          </button>
        )}
        <button
          type="button"
          className="ds-btn-primary"
          style={{ height: 26 }}
          onClick={() => void save()}
          disabled={saving || !dirty}
          title={t('saveBar.tooltip')}
        >
          {saving ? t('common.saving') : dirty ? t('saveBar.saveWrite') : t('saveBar.saved')}
        </button>
      </span>

      {open && (
        <div
          role="dialog"
          aria-label="snapshot-list"
          className="ds-card"
          style={{ position: 'absolute', right: 20, bottom: 'calc(100% + 6px)', width: 340, maxHeight: 320, overflowY: 'auto', zIndex: 30, color: 'var(--ink)' }}
        >
          <div className="ds-cap" style={{ padding: '12px 14px 6px' }}>{t('saveBar.restorePoints')}</div>
          {items.length === 0 ? (
            <p className="ds-muted" style={{ padding: '4px 14px 14px', margin: 0, fontSize: 12 }}>
              {t('saveBar.noRestorePoints')}
            </p>
          ) : (
            <ul style={{ listStyle: 'none', padding: '0 0 6px', margin: 0 }}>
              {items.map((s) => (
                <li key={s.id} style={{ padding: '8px 14px', display: 'flex', alignItems: 'center', gap: 8, borderTop: '1px solid var(--line)', fontSize: 12 }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="ds-mono" style={{ fontSize: 11.5 }}>{fmtTime(s.created_at)}</div>
                    <div className="ds-kpi-meta">{t('saveBar.restoreEntry', { n: s.file_count, size: fmtSize(s.size) })}</div>
                  </div>
                  <button type="button" className="ds-ctl" style={{ height: 26 }} onClick={() => void restore(s.id)} disabled={busyId === s.id}>
                    {t('common.restore')}
                  </button>
                  <button type="button" className="ds-kebab" onClick={() => void del(s.id)} disabled={busyId === s.id} aria-label={t('common.delete')}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
