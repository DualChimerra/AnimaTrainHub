/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // ── background layers ───────────────────────────────────────
        // Usage: bg-canvas  bg-surface  bg-sunken  bg-overlay  bg-elevated
        canvas:   'var(--bg-canvas)',
        surface:  'var(--bg-surface)',
        sunken:   'var(--bg-sunken)',
        overlay:  'var(--bg-overlay)',
        elevated: 'var(--bg-elevated)',

        // ── foreground text ──────────────────────────────────────────
        // Usage: text-fg-primary  text-fg-secondary  text-fg-tertiary  ...
        'fg-primary':   'var(--fg-primary)',
        'fg-secondary': 'var(--fg-secondary)',
        'fg-tertiary':  'var(--fg-tertiary)',
        'fg-disabled':  'var(--fg-disabled)',
        'fg-inverse':   'var(--fg-inverse)',

        // ── border colors ────────────────────────────────────────────
        // Usage: border-subtle(thin)  border-dim(default)  border-bold(thick)
        // Note: avoid the border-{border-*} naming to prevent a duplicated prefix
        subtle: 'var(--border-subtle)',
        dim:    'var(--border-default)',
        bold:   'var(--border-strong)',

        // ── accent colors ────────────────────────────────────────────
        // Usage: bg-accent  text-accent  bg-accent-soft  text-accent-fg ...
        accent: {
          DEFAULT: 'var(--accent)',
          hover:   'var(--accent-hover)',
          soft:    'var(--accent-soft)',
          fg:      'var(--accent-fg)',
        },

        // ── selected list row ────────────────────────────────────────
        // Usage: bg-selected-soft border-selected (radio row highlight; more
        // muted and desaturated than the accent family in dark mode, see
        // tokens.css --row-selected-*)
        selected: {
          DEFAULT: 'var(--row-selected-border)',
          soft:    'var(--row-selected-bg)',
        },

        // ── status colors ────────────────────────────────────────────
        // Usage: bg-ok  text-ok  bg-ok-soft / bg-err  text-err  bg-err-soft / ...
        ok:   { DEFAULT: 'var(--ok)',   soft: 'var(--ok-soft)',   line: 'var(--ok-line)'   },
        warn: { DEFAULT: 'var(--warn)', soft: 'var(--warn-soft)', line: 'var(--warn-line)' },
        err:  { DEFAULT: 'var(--err)',  soft: 'var(--err-soft)',  line: 'var(--err-line)'  },
        info: { DEFAULT: 'var(--info)', soft: 'var(--info-soft)', line: 'var(--info-line)' },

        // ── shadcn / Tailark compat aliases (usable directly by registry blocks) ──
        background: 'var(--background)',
        foreground: 'var(--foreground)',
        card:       { DEFAULT: 'var(--card)', foreground: 'var(--card-foreground)' },
        popover:    { DEFAULT: 'var(--popover)', foreground: 'var(--popover-foreground)' },
        primary:    { DEFAULT: 'var(--primary)', foreground: 'var(--primary-foreground)' },
        secondary:  { DEFAULT: 'var(--secondary)', foreground: 'var(--secondary-foreground)' },
        muted:      { DEFAULT: 'var(--muted)', foreground: 'var(--muted-foreground)' },
        destructive:{ DEFAULT: 'var(--destructive)', foreground: 'var(--destructive-foreground)' },
        border:     'var(--border)',
        input:      'var(--input)',
        ring:       'var(--ring)',
      },

      // text-accent* read --accent-text when a skin defines it: a light accent
      // can be a good fill yet unreadable as text. Without it they fall back to
      // --accent / --accent-hover, i.e. exactly the old colours.
      textColor: {
        accent: {
          DEFAULT: 'var(--accent-text, var(--accent))',
          hover:   'var(--accent-text, var(--accent-hover))',
          soft:    'var(--accent-soft)',
          fg:      'var(--accent-fg)',
        },
      },

      // Status colors use a light outline when used as border / ring (text-* /
      // bg-* stay solid), so banners like `border-warn bg-warn-soft` get a
      // Tailark-style soft warning look.
      borderColor: {
        ok:   'var(--ok-line)',
        warn: 'var(--warn-line)',
        err:  'var(--err-line)',
        info: 'var(--info-line)',
      },
      ringColor: {
        ok:   'var(--ok-line)',
        warn: 'var(--warn-line)',
        err:  'var(--err-line)',
        info: 'var(--info-line)',
      },

      fontFamily: {
        sans: ['var(--font-sans)'],
        mono: ['var(--font-mono)'],
        code: ['var(--font-code)'],
      },

      // ── radius → CSS variable (Tailark: --radius 0.625rem) ──────────
      borderRadius: {
        sm:      '4px',
        DEFAULT: 'var(--r-sm)',
        md:      'var(--r-md)',
        lg:      'var(--r-lg)',
        xl:      'var(--r-card)',
        '2xl':   'var(--r-xl)',
      },

      // ── font size → CSS variables, supports runtime density adjustment ──
      // A same-named key under extend overrides the Tailwind default:
      //   text-xs   → var(--t-xs,  12px)
      //   text-sm   → var(--t-sm,  14px)
      //   text-base → var(--t-base,15px)  (Tailwind default 16px, changed to 15px here)
      //   text-lg   → var(--t-lg,  18px)  (matches Tailwind default)
      //   text-xl   → var(--t-xl,  22px)  (Tailwind default 20px)
      //   text-2xl  → var(--t-2xl, 28px)  (Tailwind default 24px)
      //   text-3xl  → var(--t-3xl, 36px)  (matches Tailwind default)
      // text-md is our own extension; Tailwind has no such level
      fontSize: {
        '2xs':  ['var(--t-2xs)', { lineHeight: '1.4' }],
        'xs':   ['var(--t-xs)',   { lineHeight: '1.5' }],
        'sm':   ['var(--t-sm)',   { lineHeight: '1.5' }],
        'base': ['var(--t-base)', { lineHeight: '1.6' }],
        'md':   ['var(--t-md)',   { lineHeight: '1.5' }],
        'lg':   ['var(--t-lg)',   { lineHeight: '1.4' }],
        'xl':   ['var(--t-xl)',   { lineHeight: '1.3' }],
        '2xl':  ['var(--t-2xl)', { lineHeight: '1.2'  }],
        '3xl':  ['var(--t-3xl)', { lineHeight: '1.15' }],
      },

      // ── radii → CSS variables (tokens.css --r-*) ────────────────
      // Tailwind used to ship its own 2/4/6/8/12px defaults while the hand-written
      // primitives (.btn/.card/.input) read --r-*. Two parallel scales meant every
      // radius change had to be made twice and one was always forgotten.
      // With this mapping tokens.css is the single source of truth.
      // Usage: rounded-sm(small controls) rounded-md(panels/buttons) rounded-lg(large) rounded-full(pill)
      borderRadius: {
        none: '0px',
        sm:   'var(--r-sm)',
        DEFAULT: 'var(--r-md)',
        md:   'var(--r-md)',
        lg:   'var(--r-lg)',
        xl:   'var(--r-xl)',
        '2xl': 'var(--r-xl)',
        '3xl': 'var(--r-xl)',
        full: 'var(--r-pill)',
      },

      // ── shadow → CSS variables, supports automatic dark mode switching ──
      // Usage: shadow-sm  shadow-md  shadow-lg  shadow-xl
      boxShadow: {
        xs: 'var(--sh-xs)',
        sm: 'var(--sh-sm)',
        md: 'var(--sh-md)',
        lg: 'var(--sh-lg)',
        xl: 'var(--sh-xl)',
      },
    },
  },
  plugins: [],
}
