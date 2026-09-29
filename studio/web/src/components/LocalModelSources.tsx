// LocalModelSources.tsx -- the shared piece for the "use your own weights" line on the Settings page.
//
// The backend flattens each weight category's candidates into catalog.model_sources[domain] (see
// _source_row in studio/services/models/catalog.py): built-in preset rows and user-registered
// local rows share the same shape. This file only renders the **local rows** and the "pick from a folder"
// entry point; preset rows are still rendered by their own download cards (they need download buttons / chunk progress).
//
// Mapping between domain and the selected-value field (each of the three call sites writes back its own):
//   anima / krea2  -> secrets.models.selected[family]     main model weights (.safetensors)
//   vae            -> secrets.models.selected_vae         VAE weights (.safetensors)
//   anima_te / krea2_te -> secrets.models.selected_te[family]  text encoder directory
//
// Paths are always **absolute paths on the server machine**: the user's own PC in local mode, the disk
// inside the container in cloud mode -- copy switches by runtime mode so cloud users don't type a local D:\ path.
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { api, type ModelSourceRow } from '../api/client'
import { useToast } from './Toast'
import { useRuntimeModeOptional } from '../lib/RuntimeMode'
import PathPicker from './PathPicker'

/** Shape of local weights: a single file (main model / VAE) or a transformers directory (text encoder). */
export type LocalSourceShape = 'file' | 'dir'

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

interface RowsProps {
  /** Key into catalog.model_sources (anima / krea2 / vae / anima_te / krea2_te). */
  domain: string
  /** All candidate rows for this domain; this component only renders the ones with kind === 'local'. */
  rows: ModelSourceRow[]
  /** Radio group name (the preset radio in the same card must share this name to stay mutually exclusive). */
  radioName: string
  /** Select a row: the caller writes back its own selected-value field (selected / selected_vae / selected_te). */
  onSelect: (value: string) => void | Promise<void>
  /** Refresh the catalog after registering / unregistering (the selected value may have been reset by the server). */
  onChanged: () => void | Promise<void>
  /** Main-model rows additionally offer a "working mode" dropdown: reassigns these weights to a different model family. */
  familyOptions?: { value: string; label: string }[]
  /** Re-select these weights in the **new family** after switching modes (only if they were selected before the switch).
   *  Without this the user would have to switch to the other card and click the radio again. */
  selectInDomain?: (domain: string, value: string) => void | Promise<void>
}

/** A registered local-weights row (radio select + delete + optional mode switch). */
export function LocalModelRows({
  domain, rows, radioName, onSelect, onChanged, familyOptions, selectInDomain,
}: RowsProps) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const [busy, setBusy] = useState<string | null>(null)
  const local = rows.filter((r) => r.kind === 'local')
  if (local.length === 0) return null

  const unregister = async (row: ModelSourceRow) => {
    if (!row.candidate) return
    setBusy(row.value)
    try {
      await api.removeModelSource(domain, row.candidate)
      toast(t('settings.localModelRemoved', { name: row.label }), 'success')
      await onChanged()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(null)
    }
  }

  // Mode switch = re-register the same path under a different domain. If the old domain currently has it selected,
  // the server resets that family's selected value to the official variant (_selected_value_reset), so the
  // catalog must be re-fetched after the switch to see the real state.
  const changeFamily = async (row: ModelSourceRow, nextDomain: string) => {
    if (!row.candidate || nextDomain === domain) return
    setBusy(row.value)
    try {
      await api.addModelSource(nextDomain, row.candidate)
      // Unregistering the old candidate makes the server reset the old family's selected value to the official
      // variant (since these weights are no longer under that family) -- so "was it currently selected" must be
      // read before unregistering, and reselected after.
      const wasCurrent = row.is_current
      await api.removeModelSource(domain, row.candidate)
      if (wasCurrent && selectInDomain) await selectInDomain(nextDomain, row.value)
      toast(t('settings.localModelFamilyChanged', {
        name: row.label,
        family: familyOptions?.find((f) => f.value === nextDomain)?.label ?? nextDomain,
      }), 'success')
      await onChanged()
    } catch (e) {
      toast(String(e), 'error')
      await onChanged()
    } finally {
      setBusy(null)
    }
  }

  return (
    <>
      {local.map((row) => (
        <li
          key={row.value}
          className={`model-row ds-model-row${row.is_current ? ' ds-is-on' : ''}`}
        >
          <input
            type="radio"
            name={radioName}
            checked={row.is_current}
            disabled={!row.exists || busy === row.value}
            onChange={() => void onSelect(row.value)}
            title={row.exists
              ? t('settings.selectLocalModel')
              : t('settings.localModelMissing')}
          />
          <code className="ds-model-name" title={row.value}>{row.label}</code>
          <span className="ds-badge ds-mute">{t('settings.localBaseModel')}</span>
          <span className={`ds-badge ds-mono ${row.exists ? 'ds-ok' : 'ds-err'}`}>
            {row.exists ? `✓ ${fmtBytes(row.size)}` : t('settings.localModelMissing')}
          </span>
          {familyOptions && familyOptions.length > 1 && (
            <select
              value={domain}
              disabled={busy === row.value}
              onChange={(e) => void changeFamily(row, e.target.value)}
              className="ds-inp ds-mono ds-set-minisel"
              title={t('settings.localModelFamilyHint')}
            >
              {familyOptions.map((f) => (
                <option key={f.value} value={f.value}>{f.label}</option>
              ))}
            </select>
          )}
          <button
            onClick={() => void unregister(row)}
            disabled={busy === row.value}
            className="ds-ctl ds-ghost ds-sm"
            title={t('settings.localModelRemoveHint')}
          >
            {t('settings.localModelRemove')}
          </button>
        </li>
      ))}
    </>
  )
}

interface AddProps {
  domain: string
  /** Single file (main model / VAE) or directory (text encoder). */
  shape: LocalSourceShape
  /** PathPicker's starting directory (usually catalog.models_root). */
  initialPath?: string
  onChanged: () => void | Promise<void>
}

/** "Pick one from a folder" entry point: PathPicker selection -> registered as a local candidate. */
export function AddLocalModelButton({
  domain, shape, initialPath, onChanged,
}: AddProps) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const runtime = useRuntimeModeOptional()
  const [picking, setPicking] = useState(false)
  const [saving, setSaving] = useState(false)

  const register = async (path: string) => {
    setPicking(false)
    setSaving(true)
    try {
      await api.addModelSource(domain, { kind: 'local', path })
      toast(t('settings.localModelAdded', { path }), 'success')
      await onChanged()
    } catch (e) {
      // Backend validation errors (bad extension / missing config.json / path doesn't exist) are surfaced as-is
      // so the user can adjust their selection.
      toast(String(e), 'error')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <button
        onClick={() => setPicking(true)}
        disabled={saving}
        className="ds-ctl ds-sm"
        title={runtime?.mode === 'colab'
          ? t('settings.localModelPathHintColab')
          : t('settings.localModelPathHintLocal')}
      >
        {saving
          ? t('common.saving')
          : shape === 'dir'
            ? t('settings.addLocalModelDir')
            : t('settings.addLocalModelFile')}
      </button>
      {picking && (
        <PathPicker
          initialPath={initialPath}
          dirOnly={shape === 'dir'}
          onPick={(p) => void register(p)}
          onClose={() => setPicking(false)}
        />
      )}
    </div>
  )
}
