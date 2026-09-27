import { useTranslation } from 'react-i18next'
import type { SaveStatus } from '../lib/SettingsData'

/**
 * Autosave status indicator (shared by instant-apply / debounce autosave pages):
 * saving -> "Saving..."; saved -> "Saved hh:mm:ss" with a flash animation;
 * error -> red error; idle renders nothing. Originally a private Settings-page
 * component, extracted during UX polish pass 2 and now shared by the Settings
 * topbar, the Train page header and the Presets page header.
 */
export default function SaveIndicator({ status }: { status: SaveStatus }) {
  const { t } = useTranslation()
  if (status.state === 'saving') {
    return <span className="text-xs text-fg-tertiary">{t('settings.saveStatus.saving')}</span>
  }
  if (status.state === 'saved') {
    const time = new Date(status.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    // key={status.at}: remounts on every save -> replays the flash animation, so back-to-back saves are visibly distinct.
    return (
      <span key={status.at} className="settings-saved-flash text-xs inline-flex items-center gap-1">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 6L9 17l-5-5" />
        </svg>
        {t('settings.saveStatus.saved', { time })}
      </span>
    )
  }
  if (status.state === 'error') {
    return <span className="text-xs text-err">{t('settings.saveStatus.error')}</span>
  }
  return null
}
