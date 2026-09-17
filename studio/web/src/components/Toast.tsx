import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'

type Kind = 'info' | 'success' | 'error'

interface ToastItem {
  id: number
  kind: Kind
  message: string
}

interface ToastApi {
  toast: (msg: string, kind?: Kind) => void
}

const Ctx = createContext<ToastApi | null>(null)

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([])

  const toast = useCallback((message: string, kind: Kind = 'info') => {
    const id = Date.now() + Math.random()
    setItems((arr) => [...arr, { id, kind, message }])
    window.setTimeout(() => {
      setItems((arr) => arr.filter((t) => t.id !== id))
    }, kind === 'error' ? 6000 : 3000)
  }, [])

  return (
    <Ctx.Provider value={{ toast }}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 space-y-2 max-w-sm">
        {items.map((t) => (
          <div
            key={t.id}
            className={
              'flex items-start gap-2.5 pl-3.5 pr-4 py-2.5 rounded-lg shadow-lg text-sm border bg-elevated border-subtle fade-in ' +
              (t.kind === 'error' ? 'text-err' : 'text-fg-primary')
            }
          >
            <span
              aria-hidden
              className={
                'dot mt-[7px] ' +
                (t.kind === 'error' ? 'dot-err' : t.kind === 'success' ? 'dot-ok' : 'bg-accent')
              }
            />
            <span className="min-w-0 break-words">{t.message}</span>
          </div>
        ))}
      </div>
    </Ctx.Provider>
  )
}

export function useToast(): ToastApi {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useToast must be used inside <ToastProvider>')
  return ctx
}

/** 把任意 throw 的错误转成 toast，避免 alert/console。 */
export function useReportError() {
  const { toast } = useToast()
  return useCallback(
    (e: unknown) => toast(e instanceof Error ? e.message : String(e), 'error'),
    [toast]
  )
}

/** 副作用：当 deps 变化且 cond 真，弹一条 toast。常用于事件提示。 */
export function useToastOn(cond: boolean, message: string, kind: Kind = 'info') {
  const { toast } = useToast()
  useEffect(() => {
    if (cond) toast(message, kind)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cond])
}
