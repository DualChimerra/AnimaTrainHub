/**
 * Interface sounds — short, soft cues synthesised with Web Audio (no audio
 * files to ship or load). Every cue is a few sine/triangle partials with a
 * fast attack and an exponential tail, mixed well below speech level so it
 * reads as tactile feedback rather than a notification.
 *
 * The browser only allows audio after a user gesture; the context is created
 * lazily on the first cue, which always follows a click or key press.
 *
 * On/off lives in localStorage (`studio.sound`, default on) and is toggled in
 * Settings → Appearance.
 */
import { useEffect, useState } from 'react'

export type SoundName =
  | 'tap'      // buttons, tabs, segmented controls, nav rows
  | 'toggleOn' // switch / checkbox turned on
  | 'toggleOff'
  | 'select'   // an option picked in a dropdown or menu
  | 'open'     // a menu, popover or dialog appears
  | 'success'  // toast: something finished well
  | 'error'    // toast: something failed
  | 'notify'   // background work finished (training done)

const STORAGE_KEY = 'studio.sound'
const CHANGE_EVENT = 'studio:sound-change'

export function isSoundEnabled(): boolean {
  try { return localStorage.getItem(STORAGE_KEY) !== '0' } catch { return true }
}

export function setSoundEnabled(on: boolean) {
  try { localStorage.setItem(STORAGE_KEY, on ? '1' : '0') } catch { /* private mode */ }
  window.dispatchEvent(new Event(CHANGE_EVENT))
}

export function useSoundEnabled(): [boolean, (on: boolean) => void] {
  const [on, setOn] = useState(isSoundEnabled)
  useEffect(() => {
    const sync = () => setOn(isSoundEnabled())
    window.addEventListener(CHANGE_EVENT, sync)
    window.addEventListener('storage', sync)
    return () => {
      window.removeEventListener(CHANGE_EVENT, sync)
      window.removeEventListener('storage', sync)
    }
  }, [])
  return [on, setSoundEnabled]
}

let ctx: AudioContext | null = null
let master: GainNode | null = null

function audio(): { ac: AudioContext; out: GainNode } | null {
  if (typeof window === 'undefined') return null
  const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
  if (!AC) return null
  if (!ctx) {
    try {
      ctx = new AC()
      master = ctx.createGain()
      master.gain.value = 0.9
      // A gentle low-pass takes the digital edge off every partial.
      const lp = ctx.createBiquadFilter()
      lp.type = 'lowpass'
      lp.frequency.value = 5200
      master.connect(lp).connect(ctx.destination)
    } catch {
      return null
    }
  }
  if (ctx.state === 'suspended') void ctx.resume()
  return { ac: ctx, out: master! }
}

interface Partial {
  /** Start frequency, Hz. */
  f: number
  /** End frequency for a glide; defaults to f. */
  to?: number
  /** Offset from the cue start, seconds. */
  at?: number
  /** Tail length, seconds. */
  dur: number
  gain: number
  type?: OscillatorType
}

function play(partials: Partial[]) {
  const a = audio()
  if (!a) return
  const now = a.ac.currentTime + 0.005
  for (const p of partials) {
    const t0 = now + (p.at ?? 0)
    const osc = a.ac.createOscillator()
    const g = a.ac.createGain()
    osc.type = p.type ?? 'sine'
    osc.frequency.setValueAtTime(p.f, t0)
    if (p.to && p.to !== p.f) osc.frequency.exponentialRampToValueAtTime(p.to, t0 + p.dur * 0.6)
    g.gain.setValueAtTime(0.0001, t0)
    g.gain.exponentialRampToValueAtTime(p.gain, t0 + 0.006)
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + p.dur)
    osc.connect(g).connect(a.out)
    osc.start(t0)
    osc.stop(t0 + p.dur + 0.02)
  }
}

const CUES: Record<SoundName, Partial[]> = {
  tap: [
    { f: 1250, to: 950, dur: 0.05, gain: 0.03 },
    { f: 2500, to: 1900, dur: 0.03, gain: 0.008 },
  ],
  toggleOn: [
    { f: 880, to: 1320, dur: 0.07, gain: 0.03 },
    { f: 1760, dur: 0.05, gain: 0.006, at: 0.02 },
  ],
  toggleOff: [
    { f: 1320, to: 880, dur: 0.07, gain: 0.026 },
  ],
  select: [
    { f: 1046.5, dur: 0.09, gain: 0.03, type: 'triangle' },
    { f: 1568, dur: 0.07, gain: 0.01, at: 0.012 },
  ],
  open: [
    { f: 520, to: 780, dur: 0.08, gain: 0.022 },
  ],
  success: [
    { f: 783.99, dur: 0.22, gain: 0.035, type: 'triangle' },
    { f: 1174.66, dur: 0.3, gain: 0.03, type: 'triangle', at: 0.085 },
    { f: 2349.3, dur: 0.18, gain: 0.005, at: 0.085 },
  ],
  error: [
    { f: 392, dur: 0.16, gain: 0.04, type: 'triangle' },
    { f: 311.13, dur: 0.26, gain: 0.04, type: 'triangle', at: 0.11 },
  ],
  notify: [
    { f: 659.25, dur: 0.24, gain: 0.03, type: 'triangle' },
    { f: 880, dur: 0.24, gain: 0.03, type: 'triangle', at: 0.09 },
    { f: 1318.5, dur: 0.4, gain: 0.028, type: 'triangle', at: 0.18 },
  ],
}

let lastAt = 0
let lastName: SoundName | null = null

export function playSound(name: SoundName) {
  if (!isSoundEnabled()) return
  // One gesture can reach several listeners (a tab that is also a button);
  // collapse bursts so a click never sounds twice.
  const t = performance.now()
  if (t - lastAt < 45 && (lastName === name || name === 'tap')) return
  lastAt = t
  lastName = name
  try { play(CUES[name]) } catch { /* audio is a nicety, never an error */ }
}

// ── delegated feedback ─────────────────────────────────────────────────────

/** Controls that answer a click with a tap. Primary actions and toggles get
 *  their own cue elsewhere; plain text links stay silent. */
const TAP_SELECTOR = [
  '.ds-btn-primary', '.ds-btn-dark', '.ds-btn-danger', '.ds-ctl', '.ds-iconbtn',
  '.ds-seg-item', '.ds-pill', '.ds-tab', '.ds-btab', '[role="tab"]',
  '.ds-optcard', '.ds-chip', '.ds-chip-add', '.ds-nav-item', '.ds-step', '.ds-verrow',
  '.ds-rowgo', '.ds-kebab', '.ds-verpill', '.ds-projcard', '.btn',
].join(',')

let installed = false

/** Wires the ambient cues once for the whole app (called from main.tsx). */
export function installSoundFeedback() {
  if (installed || typeof document === 'undefined') return
  installed = true

  document.addEventListener('change', (e) => {
    const el = e.target as HTMLInputElement | null
    if (!el || el.tagName !== 'INPUT') return
    if (el.type === 'checkbox') playSound(el.checked ? 'toggleOn' : 'toggleOff')
    else if (el.type === 'radio') playSound('select')
  }, true)

  document.addEventListener('click', (e) => {
    const el = (e.target as Element | null)?.closest?.(TAP_SELECTOR) as HTMLElement | null
    if (!el) return
    if ((el as HTMLButtonElement).disabled || el.getAttribute('aria-disabled') === 'true') return
    // A switch is a <label> around a checkbox: the change listener speaks for it.
    if (el.querySelector('input[type="checkbox"],input[type="radio"]')) return
    playSound('tap')
  }, true)
}
