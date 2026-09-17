import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import zh from './locales/zh.json'
import en from './locales/en.json'
import ru from './locales/ru.json'

const STORAGE_KEY = 'studio.lang'

export function getStoredLang(): string | null {
  try { return localStorage.getItem(STORAGE_KEY) } catch { return null }
}

export function getStoredLangWithDefault(): string {
  return getStoredLang() ?? 'en'
}

export function setStoredLang(lang: string) {
  try { localStorage.setItem(STORAGE_KEY, lang) } catch { /* ignore */ }
}

/** Resolves once i18next finished initialising.
 *
 *  `init()` is asynchronous and applies its `lng` option when it settles, so a
 *  `changeLanguage()` issued before that gets overwritten. Anything that needs a
 *  deterministic locale (the test setup, most of all) awaits this first. */
export const i18nReady = i18n
  .use(initReactI18next)
  .init({
    // ru is intentionally partial: i18next falls back to `en` per key, so an
    // untranslated screen shows English rather than a raw key.
    resources: { zh: { translation: zh }, en: { translation: en }, ru: { translation: ru } },
    lng: getStoredLangWithDefault(),
    fallbackLng: 'en',
    interpolation: { escapeValue: false },
  })

export default i18n
