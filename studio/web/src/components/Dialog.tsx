// Dialog.tsx -- imperative confirm / prompt / alert, replacing the browser's
// native window.* trio.
//
// Design motivation: the browser's native dialog styling is completely
// disconnected from the app UI (a dull gray system popup), and it can't
// support custom button text / danger-action coloring / input validation. The
// repo had 20+ scattered confirm/prompt/alert call sites; this centralizes them
// into a single Provider + hook imperative API, with minimal call-site changes
// (`if (!confirm(...))` -> `if (!await confirm(...))`).
//
// API shape:
//   const { confirm, prompt, alert } = useDialog()
//   const ok    = await confirm('Delete version v1?', { tone: 'danger' })   // Promise<boolean>
//   const name  = await prompt('New preset name', { defaultValue, validate }) // Promise<string|null>
//   await alert('Failed to parse JSON', { tone: 'error' })
//
// Not a replacement for: existing complex form dialogs (like NewVersionDialog)
// that have custom fields / embed SchemaForm -- declarative JSX is clearer
// there, so those stay as they are.
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { useTranslation } from 'react-i18next'
import { playSound } from '../lib/sound'

export type DialogTone = 'default' | 'danger' | 'warn'

export interface ConfirmOptions {
  /** Decides the confirm button color: default=accent / danger=err red / warn=warn orange */
  tone?: DialogTone
  okText?: string
  cancelText?: string
  /** Dialog title (defaults to "Confirm action") */
  title?: string
}

export interface PromptOptions {
  defaultValue?: string
  placeholder?: string
  /** Synchronous validation: return null = passes, return string = error message. Runs on every keystroke. */
  validate?: (v: string) => string | null
  okText?: string
  cancelText?: string
  title?: string
  /** Mirrors ConfirmOptions -- for the "submitting this input triggers a dangerous action" case. */
  tone?: DialogTone
}

export interface AlertOptions {
  tone?: DialogTone
  okText?: string
  title?: string
}

type DialogState =
  | {
      type: 'confirm'
      message: string
      options: ConfirmOptions
      resolve: (ok: boolean) => void
    }
  | {
      type: 'prompt'
      label: string
      options: PromptOptions
      resolve: (v: string | null) => void
    }
  | {
      type: 'alert'
      message: string
      options: AlertOptions
      resolve: () => void
    }

interface DialogApi {
  confirm: (message: string, options?: ConfirmOptions) => Promise<boolean>
  prompt: (label: string, options?: PromptOptions) => Promise<string | null>
  alert: (message: string, options?: AlertOptions) => Promise<void>
}

const Ctx = createContext<DialogApi | null>(null)

export function DialogProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<DialogState | null>(null)

  const confirm = useCallback(
    (message: string, options: ConfirmOptions = {}): Promise<boolean> =>
      new Promise((resolve) => {
        setState({ type: 'confirm', message, options, resolve })
      }),
    [],
  )

  const prompt = useCallback(
    (label: string, options: PromptOptions = {}): Promise<string | null> =>
      new Promise((resolve) => {
        setState({ type: 'prompt', label, options, resolve })
      }),
    [],
  )

  const alert = useCallback(
    (message: string, options: AlertOptions = {}): Promise<void> =>
      new Promise((resolve) => {
        setState({ type: 'alert', message, options, resolve })
      }),
    [],
  )

  // Common cancel entry point (ESC / click backdrop / cancel button): confirm ->
  // false, prompt -> null, alert -> resolves directly.
  const cancel = useCallback(() => {
    setState((cur) => {
      if (!cur) return null
      if (cur.type === 'confirm') cur.resolve(false)
      else if (cur.type === 'prompt') cur.resolve(null)
      else cur.resolve()
      return null
    })
  }, [])

  const confirmOk = useCallback((value: boolean | string) => {
    setState((cur) => {
      if (!cur) return null
      if (cur.type === 'confirm') cur.resolve(value as boolean)
      else if (cur.type === 'prompt') cur.resolve(value as string)
      else cur.resolve()
      return null
    })
  }, [])

  // ESC closes (equivalent to cancel)
  useEffect(() => {
    if (!state) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        cancel()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [state, cancel])

  return (
    <Ctx.Provider value={{ confirm, prompt, alert }}>
      {children}
      {state && (
        <DialogRoot state={state} onCancel={cancel} onOk={confirmOk} />
      )}
    </Ctx.Provider>
  )
}

export function useDialog(): DialogApi {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useDialog must be used inside <DialogProvider>')
  return ctx
}

// ────────────────────────────────────────────────────────────────────────────

/** The confirm button uses the primary color; the danger tone switches it to btn-danger. */
function toneButtonClass(tone: DialogTone | undefined): string {
  return tone === 'danger' ? 'btn btn-danger' : 'btn btn-primary'
}

interface RootProps {
  state: DialogState
  onCancel: () => void
  onOk: (value: boolean | string) => void
}

function DialogRoot({ state, onCancel, onOk }: RootProps) {
  const { t } = useTranslation()
  // input value uses a ref rather than state, to avoid triggering a DialogRoot
  // rerender on every keystroke. The error message uses state since it needs to
  // trigger a re-render.
  const [inputValue, setInputValue] = useState(
    state.type === 'prompt' ? (state.options.defaultValue ?? '') : '',
  )
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => { playSound('open') }, [])

  // Prompt auto-focuses + selects the default value, so users can rename it easily
  useEffect(() => {
    if (state.type === 'prompt') {
      requestAnimationFrame(() => {
        inputRef.current?.focus()
        inputRef.current?.select()
      })
    }
  }, [state.type])

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault()
    if (state.type === 'confirm') {
      onOk(true)
    } else if (state.type === 'prompt') {
      const v = inputValue
      const err = state.options.validate?.(v) ?? null
      if (err) {
        setError(err)
        inputRef.current?.focus()
        return
      }
      onOk(v)
    } else {
      onOk(true)
    }
  }

  const title =
    state.options.title ??
    (state.type === 'confirm'
      ? t('dialog.confirmTitle')
      : state.type === 'prompt'
        ? t('dialog.inputTitle')
        : t('dialog.infoTitle'))

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-[60] flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px] ds-backdrop-anim"
      onMouseDown={(e) => {
        // Only close on a click on the backdrop itself -- a mouseDown inside the form doesn't fire this
        if (e.target === e.currentTarget) onCancel()
      }}
    >
      <form
        onSubmit={handleSubmit}
        className="bg-elevated border border-subtle rounded-2xl w-[90%] max-w-[440px] p-6 flex flex-col gap-4 shadow-xl ds-dialog-anim"
      >
        <h2 className="m-0 text-lg font-semibold text-fg-primary">{title}</h2>

        {state.type === 'prompt' ? (
          <label className="flex flex-col gap-1.5">
            <span className="text-sm text-fg-secondary">{state.label}</span>
            <input
              ref={inputRef}
              className="input"
              value={inputValue}
              placeholder={state.options.placeholder}
              onChange={(e) => {
                setInputValue(e.target.value)
                if (error) setError(null)
              }}
            />
            {error && (
              <span className="text-xs text-err">{error}</span>
            )}
          </label>
        ) : (
          <p className="m-0 text-sm text-fg-secondary whitespace-pre-wrap">
            {state.type === 'confirm' ? state.message : state.message}
          </p>
        )}

        {/* min-w-[96px] + justify-center: both buttons stay the same width (96px
            covers the natural ~90px width of a 4-character okText, so "Cancel"
            no longer looks narrower than "Cancel plan"). */}
        <div className="flex gap-2 justify-end mt-1">
          {state.type !== 'alert' && (
            <button
              type="button"
              onClick={onCancel}
              className="btn btn-secondary min-w-[96px] justify-center"
            >
              {state.options.cancelText ?? t('common.cancel')}
            </button>
          )}
          <button
            type="submit"
            className={`${toneButtonClass(state.options.tone)} min-w-[96px] justify-center`}
          >
            {state.options.okText ??
              (state.type === 'confirm'
                ? t('common.confirm')
                : state.type === 'prompt'
                  ? t('common.ok')
                  : t('common.gotIt'))}
          </button>
        </div>
      </form>
    </div>
  )
}
