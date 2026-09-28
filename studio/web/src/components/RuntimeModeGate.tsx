// RuntimeModeGate.tsx -- the one-time "Colab or Local" picker shown on first
// launch (this fork).
//
// Why it's a one-shot blocking gate: the sane defaults for the two modes
// **actively conflict** (bind address, whether to open a browser, where to
// put studio_data, whether to nag about syncing to Drive). It used to be a
// CLI flag people had to remember; users had no idea what to pass right
// after install. Now the app asks once on launch, persists the answer, and
// never asks again.
//
// The detection result is only a preselection, never a substitute for the
// user's own click -- setups like a self-hosted JupyterHub or local training
// inside docker will inevitably be misdetected, and silently picking for the
// user makes it harder to correct. The card surfaces what the detection was
// based on.
//
// This component renders nothing (`needsPick` is false) when pinned by
// `ALS_RUNTIME_MODE`; the Colab notebook's startup cell sets it, so cloud
// users aren't blocked and get a working app out of the box.
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import type { RuntimeMode } from '../api/client'
import { useRuntimeMode } from '../lib/RuntimeMode'

function formatBytes(n: number | null | undefined): string {
  if (!n || n <= 0) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1 }
  return `${v >= 10 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`
}

export default function RuntimeModeGate() {
  const { needsPick, info, setMode } = useRuntimeMode()
  const { t } = useTranslation()
  const [choice, setChoice] = useState<RuntimeMode | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Preselect from the detection result. info arrives asynchronously, so this
  // is filled in from an effect rather than as the useState initial value.
  useEffect(() => {
    if (info && choice === null) setChoice(info.detected)
  }, [info, choice])

  if (!needsPick || !info) return null

  const env = info.environment
  const confirm = async () => {
    if (!choice) return
    setSaving(true)
    setError(null)
    try {
      await setMode(choice)
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    // Deliberately no backdrop onClick-to-close and no X -- this is mandatory,
    // and it never appears again once answered.
    <div
      className="fixed inset-0 z-[60] bg-zinc-950/40 backdrop-blur-[2px] flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="runtime-mode-title"
    >
      <div className="bg-elevated border border-subtle rounded-2xl shadow-xl w-[640px] max-w-full max-h-[90vh] flex flex-col overflow-hidden">
        <header className="px-5 pt-5 pb-3 shrink-0">
          <h2 id="runtime-mode-title" className="m-0 text-base font-semibold text-fg-primary">
            {t('runtimeMode.title')}
          </h2>
          <p className="mt-1 mb-0 text-xs text-fg-tertiary">
            {t('runtimeMode.subtitle')}
          </p>
        </header>

        <div className="px-5 pb-4 flex flex-col gap-3 overflow-y-auto">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <ModeCard
              mode="local"
              selected={choice === 'local'}
              recommended={info.detected === 'local'}
              onSelect={() => setChoice('local')}
            />
            <ModeCard
              mode="colab"
              selected={choice === 'colab'}
              recommended={info.detected === 'colab'}
              onSelect={() => setChoice('colab')}
            />
          </div>

          {/* Detection basis: lets the user judge whether the preselection is
              right, and doubles as the entry point for self-diagnosis if the
              wrong mode was picked. */}
          <div className="rounded-md border border-subtle bg-surface px-3 py-2 text-xs text-fg-secondary flex flex-col gap-1">
            <div className="text-fg-tertiary">{t('runtimeMode.detectedAs', {
              mode: t(`runtimeMode.${info.detected}.name`),
            })}</div>
            <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono">
              <span>{env.platform}, Python {env.python}</span>
              {env.gpu ? <span>GPU: {env.gpu}</span> : <span>{t('runtimeMode.noGpu')}</span>}
              <span>{t('runtimeMode.diskFree', {
                free: formatBytes(env.disk_free),
                total: formatBytes(env.disk_total),
              })}</span>
            </div>
            <div className="font-mono break-all">
              studio_data: {env.studio_data}
            </div>
          </div>

          {error && (
            <div className="rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger">
              {error}
            </div>
          )}
        </div>

        <footer className="px-5 py-3 border-t border-subtle flex items-center gap-2 shrink-0">
          <span className="text-xs text-fg-tertiary flex-1">
            {t('runtimeMode.changeLater')}
          </span>
          <button
            className="btn btn-primary text-sm"
            disabled={!choice || saving}
            onClick={() => void confirm()}
          >
            {saving ? t('runtimeMode.saving') : t('runtimeMode.confirm')}
          </button>
        </footer>
      </div>
    </div>
  )
}

function ModeCard({ mode, selected, recommended, onSelect }: {
  mode: RuntimeMode
  selected: boolean
  recommended: boolean
  onSelect: () => void
}) {
  const { t } = useTranslation()
  const bullets = t(`runtimeMode.${mode}.bullets`, { returnObjects: true }) as string[]
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={`text-left rounded-md border p-3 flex flex-col gap-2 transition-colors ${
        selected
          ? 'border-accent bg-accent/10'
          : 'border-subtle bg-surface hover:border-dim'
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-fg-primary">
          {t(`runtimeMode.${mode}.name`)}
        </span>
        {recommended && (
          <span className="text-[10px] uppercase tracking-wide rounded px-1.5 py-0.5 bg-accent/20 text-accent">
            {t('runtimeMode.recommended')}
          </span>
        )}
      </div>
      <p className="m-0 text-xs text-fg-secondary">{t(`runtimeMode.${mode}.summary`)}</p>
      <ul className="m-0 pl-4 text-xs text-fg-tertiary flex flex-col gap-0.5">
        {Array.isArray(bullets) && bullets.map((b, i) => <li key={i}>{b}</li>)}
      </ul>
    </button>
  )
}
