import { useEffect, useState } from 'react'

/** The prompt's real token count (debounced call to the backend tokenizer,
 *  same source of truth as training/inference).
 *
 * Returns null when unavailable (endpoint failure / tokenizer not ready /
 * empty text), which hides the badge. Purely informational -- no length
 * warning (whether over-length gets truncated is a model concern; the user
 * owns quality).
 */
export function useTokenCount(text: string, modelFamily: string): number | null {
  const [tokens, setTokens] = useState<number | null>(null)
  useEffect(() => {
    if (!text.trim()) {
      setTokens(null)
      return
    }
    let alive = true
    const timer = setTimeout(() => {
      fetch('/api/generate/token_count', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model_family: modelFamily }),
      })
        .then((r) => r.json())
        .then((d: { tokens: number | null }) => {
          if (alive) setTokens(typeof d.tokens === 'number' ? d.tokens : null)
        })
        .catch(() => { if (alive) setTokens(null) })
    }, 500)
    return () => { alive = false; clearTimeout(timer) }
  }, [text, modelFamily])
  return tokens
}
