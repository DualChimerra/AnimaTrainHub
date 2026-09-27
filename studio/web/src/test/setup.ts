import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

// Test environment default locale = en: tests assert English literals.
// Explicitly force it after import (import gets hoisted, so pre-writing
// localStorage wouldn't help) so the suite doesn't depend on init()'s default.
import i18n, { i18nReady } from '../i18n'
// Both calls are asynchronous and both had to be awaited. `void`-ing
// `changeLanguage` left the locale wherever `init()` put it (the app default,
// 'en'), and awaiting only `changeLanguage` was still not enough: `init()`
// applies its own `lng` when it settles, overwriting an earlier switch.
await i18nReady
await i18n.changeLanguage('en')

// jsdom's fetch actually hits the network. tagDict store / other mount-time
// requests getting a 404 in test mode is an expected branch, but a
// `network error` makes React log an act() warning. Install a default fetch
// stub: any request not explicitly mocked returns 404. Individual tests can
// override this in their own beforeEach with
// vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(...).
if (typeof globalThis.fetch === 'function') {
  vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })))
}

// jsdom has no ResizeObserver (useAutoGrowTextarea etc. `new` it to grow the
// textarea); install a no-op so components don't throw on mount. Tests don't
// verify auto-grow, so the callback never firing is fine.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver
}

// Pre-warm the tagDict store to the 'empty' state: useTagDict's useEffect
// skips loadDict when the status isn't idle, avoiding an async fetch + act()
// warning on component mount. Tests that need to see a ready dict state can
// override this with __setStateForTest.
import { __setStateForTest } from '../tagDict/store'
__setStateForTest({ status: 'empty' })
