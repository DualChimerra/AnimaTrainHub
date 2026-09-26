import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type CaptionSnapshot } from '../api/client'
import { useDialog } from './Dialog'
import { useToast } from './Toast'
import Popover from './ds/Popover'

interface Props {
  pid: number
  vid: number
  /** Pending save count: 0 = nothing dirty. */
  dirtyCount: number
  /** Triggers save: the parent provides the commit implementation (diff already computed). */
  onSave: () => Promise<void>
  /** Drop the local edits and go back to what is on disk. */
  onDiscard: () => void
  /** After triggering a restore, the parent needs to refetch its cache. */
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
  const pointsBtnRef = useRef<HTMLButtonElement | null>(null)
  const dirty = dirtyCount > 0

  const refresh = useCallback(async () => {
    try { setItems(await api.listCaptionSnapshots(pid, vid)) }
    catch (e) { toast(String(e), 'error') }
  }, [pid, vid, toast])

  // Load on mount (the button shows how many points exist) and on every open.
  useEffect(() => { void refresh() }, [open, refresh])

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
    <div ref={ref} className={`ds-savebar${dirty ? ' ds-is-dirty' : ''}`}>
      <span className="ds-savebar-status">
        <span className={`ds-dot ${dirty ? 'ds-dot-run' : 'ds-dot-ok'}`} />
        <span style={{ fontWeight: dirty ? 600 : 500 }}>
          {dirty ? t('saveBar.unsaved', { count: dirtyCount }) : t('saveBar.allSaved')}
        </span>
      </span>
      <span className="ds-savebar-act">
        <button
          ref={pointsBtnRef}
          type="button"
          className={`ds-ctl${open ? ' ds-is-open' : ''}`}
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          aria-haspopup="dialog"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /><path d="M12 7v5l3 2" /></svg>
          {t('saveBar.restorePoints')}
          {items.length > 0 && <span className="ds-badge ds-mute" style={{ height: 17, padding: '0 6px', fontSize: 10.5 }}>{items.length}</span>}
        </button>
        {dirty && (
          <button type="button" className="ds-ctl" onClick={() => void discard()} disabled={saving}>
            {t('saveBar.discard')}
          </button>
        )}
        <button
          type="button"
          className="ds-btn-primary"
          style={{ minWidth: 120 }}
          onClick={() => void save()}
          disabled={saving || !dirty}
          title={t('saveBar.tooltip')}
        >
          {saving ? t('common.saving') : dirty ? t('saveBar.saveWrite') : t('saveBar.saved')}
        </button>
      </span>

      {open && pointsBtnRef.current && (
        <Popover anchor={pointsBtnRef.current} align="end" minWidth={340} maxHeight={360} role="dialog" ariaLabel={t('saveBar.restorePoints')} onClose={() => setOpen(false)}>
          <div className="ds-menu-head">{t('saveBar.restorePoints')}</div>
          {items.length === 0 ? (
            <p className="ds-muted" style={{ padding: '2px 10px 10px', margin: 0, fontSize: 12 }}>
              {t('saveBar.noRestorePoints')}
            </p>
          ) : (
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              {items.map((s) => (
                <li key={s.id} className="ds-savepoint">
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 12.5, fontWeight: 500 }}>{fmtTime(s.created_at)}</div>
                    <div className="ds-kpi-meta">{t('saveBar.restoreEntry', { n: s.file_count, size: fmtSize(s.size) })}</div>
                  </div>
                  <button type="button" className="ds-ctl ds-sm" onClick={() => void restore(s.id)} disabled={busyId === s.id}>
                    {t('common.restore')}
                  </button>
                  <button type="button" className="ds-kebab" onClick={() => void del(s.id)} disabled={busyId === s.id} aria-label={t('common.delete')} title={t('common.delete')}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Popover>
      )}
    </div>
  )
}
