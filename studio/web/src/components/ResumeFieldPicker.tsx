import { useEffect, useRef, useState } from 'react'
import { api, type LoraCkpt, type StateCkpt, type VersionCkptGroup } from '../api/client'

/**
 * In-field dropdown picker: triggered by the "browse this project" button
 * next to a field, it pops up available files grouped by version; picking
 * one writes the absolute path back into the field.
 *
 * Difference from PathPicker: no filesystem tree, just semantic file labels
 * ("baseline / step 2476"). The user never sees the deep path, but the
 * underlying onChange still writes the real absolute path, keeping the
 * schema field value compatible.
 *
 * resume_state field -> kind="state", lists training_state_step*.pt across
 *   all versions of the project
 * resume_lora  field -> kind="lora", lists *.safetensors (including final)
 *   across all versions of the project
 *
 * External files / checkpoints from other projects can be typed directly
 * into the field input; this picker only covers the most common path of
 * "reuse output from some version of this project".
 */
export default function ResumeFieldPicker({
  pid,
  kind,
  value,
  onChange,
  onClose,
  anchorRef,
}: {
  pid: number
  kind: 'state' | 'lora'
  value: string
  onChange: (path: string) => void
  onClose: () => void
  anchorRef: React.RefObject<HTMLElement | null>
}) {
  const [stateGroups, setStateGroups] = useState<VersionCkptGroup<StateCkpt>[] | null>(null)
  const [loraGroups, setLoraGroups] = useState<VersionCkptGroup<LoraCkpt>[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const popRef = useRef<HTMLDivElement | null>(null)

  // fetch data
  useEffect(() => {
    let alive = true
    if (kind === 'state') {
      api.listProjectStateCkpts(pid)
        .then((g) => { if (alive) setStateGroups(g) })
        .catch((e) => { if (alive) setError(String(e)) })
    } else {
      api.listProjectLoraCkpts(pid)
        .then((g) => { if (alive) setLoraGroups(g) })
        .catch((e) => { if (alive) setError(String(e)) })
    }
    return () => { alive = false }
  }, [pid, kind])

  // Close on outside click + Esc
  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (popRef.current?.contains(e.target as Node)) return
      if (anchorRef.current?.contains(e.target as Node)) return
      onClose()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [onClose, anchorRef])

  const groups = kind === 'state' ? stateGroups : loraGroups
  const loaded = groups !== null
  const totalItems = groups?.reduce((s, g) => s + g.items.length, 0) ?? 0

  return (
    <div
      ref={popRef}
      className="absolute z-40 mt-1 max-h-[360px] overflow-y-auto rounded-lg border border-subtle bg-elevated shadow-lg text-xs"
      style={{ top: '100%', left: 0, width: '100%', minWidth: 'min(420px, calc(100vw - 32px))' }}
    >
      {error ? (
        <div className="px-3 py-2 text-err">{error}</div>
      ) : !loaded ? (
        <div className="px-3 py-2 text-fg-tertiary italic">Loading…</div>
      ) : totalItems === 0 ? (
        <div className="px-3 py-2 text-fg-tertiary italic">
          This project hasn't produced{kind === 'state' ? ' training_state_step*.pt' : ' LoRA ckpt'} yet —
          {kind === 'state' ? 'Run one training round first with save_state_every_steps / save_state_every_epochs' : 'Train at least one ckpt first'}
        </div>
      ) : (
        // Only render versions that have items; skip empty versions to cut noise
        groups!.filter((g) => g.items.length > 0).map((g) => (
          <div key={g.version_id} className="border-b border-subtle last:border-0">
            <div className="px-3 py-1 bg-canvas caption sticky top-0 border-b border-subtle">
              {g.label}
            </div>
            {kind === 'state'
              ? (g.items as StateCkpt[]).map((it) => (
                  <PickRow
                    key={it.path}
                    label={it.label}
                    selected={it.path === value}
                    onPick={() => { onChange(it.path); onClose() }}
                  />
                ))
              : (g.items as LoraCkpt[]).map((it) => (
                  <PickRow
                    key={it.path}
                    label={it.label}
                    selected={it.path === value}
                    onPick={() => { onChange(it.path); onClose() }}
                  />
                ))}
          </div>
        ))
      )}
    </div>
  )
}

function PickRow({
  label, selected, onPick,
}: {
  label: string
  selected: boolean
  onPick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onPick}
      className={
        'w-full text-left px-4 py-1.5 font-mono cursor-pointer transition-colors flex items-center gap-2 ' +
        (selected
          ? 'bg-accent-soft text-accent font-semibold'
          : 'text-fg-primary hover:bg-overlay')
      }
    >
      <span className={'w-3 inline-block ' + (selected ? '' : 'opacity-0')}>✓</span>
      <span>{label}</span>
    </button>
  )
}
