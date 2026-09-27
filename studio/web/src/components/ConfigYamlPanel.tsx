import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, type ConfigData } from '../api/client'
import { useToast } from './Toast'

/**
 * YAML preview (R4,D3): the current form config is rendered through the backend
 * /api/schema/preview-yaml -- it goes through the same tolerant + pruning +
 * safe_dump serialization path as the file written to disk on save, so the
 * preview is physically identical to what lands on disk (previously the
 * frontend had a pruneInactiveConfig + configToYaml double mirror that only
 * stayed "in sync" by comment discipline; that's been removed). 300ms debounce
 * to stay responsive; keeps the previous text if a request fails.
 * The Presets page wraps this in a centered modal; the Train page uses it as a
 * tab in the right-column preview drawer.
 * Height is controlled by the parent: pass flex layout classes via className,
 * <pre> scrolls on its own; long lines don't wrap and scroll horizontally
 * instead (wrapping long values like paths makes them completely unreadable).
 */
export default function ConfigYamlPanel({
  config,
  fileLabel,
  hint,
  className,
}: {
  config: ConfigData
  /** File-on-disk context, e.g. `config.yaml` / `my-preset.yaml`. */
  fileLabel: string
  /** Top warning strip (e.g. "contains unsaved changes"), hidden by default. */
  hint?: string
  className?: string
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [yamlText, setYamlText] = useState('')

  useEffect(() => {
    let alive = true
    const timer = setTimeout(() => {
      api.previewConfigYaml(config)
        .then((r) => { if (alive) setYamlText(r.yaml) })
        .catch(() => { /* keep previous text on a network hiccup; the next change retries */ })
    }, 300)
    return () => { alive = false; clearTimeout(timer) }
  }, [config])

  // safe_dump top-level keys start at column 0 -- a non-whitespace line start is
  // one field written to disk (continuation lines of multi-line strings are indented)
  const fieldCount = yamlText
    ? yamlText.split('\n').filter((l) => /^[^\s#]/.test(l)).length
    : 0

  return (
    <div className={className ?? 'flex flex-col min-h-0'}>
      <div className="flex items-center gap-2 mb-2 shrink-0">
        <span className="ds-mono truncate" style={{ fontSize: 11.5, fontWeight: 600 }}>{fileLabel}</span>
        <span className="ds-kpi-meta shrink-0">
          {t('schema.fieldCount', { n: fieldCount, count: fieldCount })}
        </span>
        {hint && <span className="truncate" style={{ fontSize: 11, color: 'var(--amber-text)' }}>{hint}</span>}
        <span className="flex-1" />
        <button
          type="button"
          className="ds-ctl ds-ghost shrink-0"
          style={{ height: 26 }}
          onClick={() => {
            navigator.clipboard.writeText(yamlText)
              .then(() => toast(t('presets.copied'), 'success'))
              .catch(() => toast(t('presets.copyFailed'), 'error'))
          }}
        >{t('common.copy')}</button>
      </div>
      <pre className="ds-console flex-1 min-h-0 m-0 whitespace-pre overflow-auto" style={{ lineHeight: 1.7 }}>
        {yamlText}
      </pre>
    </div>
  )
}
