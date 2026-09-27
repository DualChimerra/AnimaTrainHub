/**
 * LLM tagger config workspace -- implemented per the "LLM Settings redesign.html" mockup.
 *
 * Layout: preset bar on top + a two-column workspace grid (360px left / 1fr
 * right) + a savebar at the bottom.
 * Left column: three independent cards stacked vertically -- Connection (01) /
 * Sampling parameters (03) / Image preprocessing (04)
 * Right column: the Prompt template (02) composer, a large card
 *
 * Design decisions:
 * - No "preview request JSON" / "test run one" / token price stats (per the user's decision)
 * - The savebar only keeps a "discard changes" button; saving relies on the
 *   global Settings page's top "Save" button
 */
import type { TFunction } from 'i18next'
import { Trans, useTranslation } from 'react-i18next'
import { useMemo } from 'react'
import type { LLMPreset } from '../api/client'
import LLMMessagesEditor from './LLMMessagesEditor'

const MASK = '***'

const LLM_PRESET_LABEL_KEYS: Record<string, string> = {
  style_json: 'llmWorkspace.presetLabels.styleJson',
  general_json: 'llmWorkspace.presetLabels.generalJson',
  txt_tags: 'llmWorkspace.presetLabels.txtTags',
  joycaption: 'llmWorkspace.presetLabels.joycaption',
}

function presetLabel(preset: LLMPreset, t: TFunction): string {
  const key = LLM_PRESET_LABEL_KEYS[preset.id]
  return key ? t(key, { defaultValue: preset.label }) : preset.label
}

interface Props {
  /** Section title at the top of the card; visually aligned with SettingsSection's h2. */
  title?: string
  currentPreset: LLMPreset
  serverCurrentPreset?: LLMPreset
  presets: LLMPreset[]
  currentPresetId: string
  onSelectPreset: (id: string) => void
  onUpdatePreset: <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => void
  onResetToBuiltin: () => void
  onSaveAs: () => void
  onAddPreset: () => void
  onDeletePreset: () => void
  llmModelsBusy: boolean
  llmTestBusy: boolean
  onRefreshModels: () => void
  onTestConnection: () => void
}

// -- Style constants mapping mockup fields directly to design tokens -----------------------
const inputStyle: React.CSSProperties = {
  width: '100%',
  background: 'var(--bg-sunken)',
  border: '1px solid var(--border-subtle)',
  borderRadius: 'var(--r-md)',
  padding: '8px 11px',
  fontSize: 'var(--t-sm)',
  color: 'var(--fg-primary)',
  fontFamily: 'var(--font-mono)',
  outline: 'none',
  transition: 'all 100ms',
}

export default function LLMTaggerWorkspace(props: Props) {
  const {
    title,
    currentPreset,
    serverCurrentPreset,
    presets,
    currentPresetId,
    onSelectPreset,
    onUpdatePreset,
    onResetToBuiltin,
    onSaveAs,
    onAddPreset,
    onDeletePreset,
    llmModelsBusy,
    llmTestBusy,
    onRefreshModels,
    onTestConnection,
  } = props

  // dirty diff: compares currentPreset (draft) against serverCurrentPreset (persisted)
  const dirtyCount = useMemo(() => {
    if (!serverCurrentPreset) return 0
    const keys: (keyof LLMPreset)[] = [
      'label', 'base_url', 'api_key', 'model', 'model_ids', 'endpoint',
      'messages', 'output_format', 'temperature', 'max_tokens',
      'max_side', 'jpeg_quality', 'max_image_mb', 'timeout', 'max_retries',
      'concurrency', 'requests_per_second', 'max_requests_per_minute',
    ]
    let n = 0
    for (const k of keys) {
      const a = currentPreset[k]
      const b = serverCurrentPreset[k]
      // Treated as unchanged when api_key is displayed as MASK
      if (k === 'api_key' && a === MASK) continue
      if (JSON.stringify(a) !== JSON.stringify(b)) n += 1
    }
    return n
  }, [currentPreset, serverCurrentPreset])

  return (
    // Outer wrapper for the whole LLM module -- title + preset bar + the workspace's 5 sections all live in the same card
    <div
      className="bg-surface border border-subtle"
      style={{ borderRadius: 'var(--r-lg)', overflow: 'hidden' }}
    >
      {/* Title aligned with SettingsSection's h2: text-sm font-semibold + p-4 spacing */}
      {title && (
        <h2
          className="text-sm font-semibold text-fg-primary m-0"
          style={{
            padding: '14px 16px',
            borderBottom: '1px solid var(--border-subtle)',
          }}
        >
          {title}
        </h2>
      )}
      <PresetBar
        currentPreset={currentPreset}
        presets={presets}
        currentPresetId={currentPresetId}
        onSelectPreset={onSelectPreset}
        onResetToBuiltin={onResetToBuiltin}
        onSaveAs={onSaveAs}
        onAddPreset={onAddPreset}
        onDeletePreset={onDeletePreset}
        onUpdateLabel={(label) => onUpdatePreset('label', label)}
        dirtyCount={dirtyCount}
      />

      {/* Workspace two-column grid: left 360px with three stacked sections / right 1fr composer fills the rest */}
      <div
        className="grid items-stretch"
        style={{ gridTemplateColumns: '360px 1fr' }}
      >
        {/* LEFT column: the 3 sections are separated by border-bottom; the whole column has a right border separating it from the right column */}
        <div
          className="flex flex-col"
          style={{ borderRight: '1px solid var(--border-subtle)' }}
        >
          <ConnectionSection
            preset={currentPreset}
            serverPreset={serverCurrentPreset}
            onUpdate={onUpdatePreset}
            llmModelsBusy={llmModelsBusy}
            llmTestBusy={llmTestBusy}
            onRefreshModels={onRefreshModels}
            onTestConnection={onTestConnection}
            bottomBorder
          />
          <AdvancedSection preset={currentPreset} onUpdate={onUpdatePreset} />
        </div>

        {/* RIGHT column: composer fills the full height; the messages area scrolls internally */}
        <ComposerSection preset={currentPreset} onUpdate={onUpdatePreset} />
      </div>
    </div>
  )
}

// ── Preset bar ──────────────────────────────────────────────────────────
function PresetBar({
  currentPreset,
  presets,
  currentPresetId,
  onSelectPreset,
  onResetToBuiltin,
  onSaveAs,
  onAddPreset,
  onDeletePreset,
  onUpdateLabel,
  dirtyCount,
}: {
  currentPreset: LLMPreset
  presets: LLMPreset[]
  currentPresetId: string
  onSelectPreset: (id: string) => void
  onResetToBuiltin: () => void
  onSaveAs: () => void
  onAddPreset: () => void
  onDeletePreset: () => void
  onUpdateLabel: (s: string) => void
  dirtyCount: number
}) {
  const { t } = useTranslation()
  return (
    <div
      style={{
        padding: '10px 12px 10px 16px',
        display: 'grid',
        gridTemplateColumns: '1fr auto',
        alignItems: 'center',
        gap: 12,
        borderBottom: '1px solid var(--border-subtle)',
      }}
    >
      <div className="flex items-center gap-3.5 min-w-0">
        <Caption>{t('llmWorkspace.preset')}</Caption>
        <div
          className="flex items-center gap-2.5 cursor-pointer"
          style={{
            background: 'var(--bg-sunken)',
            border: '1px solid var(--border-default)',
            borderRadius: 'var(--r-md)',
            padding: '7px 14px 7px 12px',
            fontSize: 'var(--t-sm)',
            color: 'var(--fg-primary)',
            fontWeight: 500,
            minWidth: 260,
            position: 'relative',
          }}
        >
          {currentPreset.builtin && (
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 'var(--t-2xs)',
                padding: '1px 6px',
                borderRadius: 'var(--r-sm)',
                background: 'var(--info-soft)',
                color: 'var(--info)',
                letterSpacing: '0.04em',
              }}
            >
              {t('llmWorkspace.builtin')}
            </span>
          )}
          {/* A select overlays the whole pick control so the user can switch it; the select itself is transparent */}
          <select
            value={currentPresetId}
            onChange={(e) => onSelectPreset(e.target.value)}
            className="absolute inset-0 opacity-0 cursor-pointer"
            aria-label={t('llmWorkspace.selectPreset')}
          >
            {presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.builtin ? t('llmWorkspace.builtinPrefix') : ''}{presetLabel(p, t)}
              </option>
            ))}
          </select>
          <span className="truncate">{presetLabel(currentPreset, t)}</span>
          <span className="ml-auto" style={{ color: 'var(--fg-tertiary)' }}>▾</span>
        </div>
        {dirtyCount > 0 && (
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 'var(--t-xs)',
              color: 'var(--fg-tertiary)',
              whiteSpace: 'nowrap',
            }}
          >
            <Trans
              i18nKey="llmWorkspace.dirtySummary"
              values={{ n: dirtyCount }}
              components={{ count: <b style={{ color: 'var(--fg-secondary)', fontWeight: 500 }} /> }}
            />
          </span>
        )}
        {/* Edit the current preset's label */}
        <input
          type="text"
          value={currentPreset.label}
          onChange={(e) => onUpdateLabel(e.target.value)}
          className="hidden"
          aria-hidden
        />
      </div>
      <div className="flex items-center gap-1.5">
        {currentPreset.builtin && (
          <PBtn variant="danger" onClick={onResetToBuiltin} title={t('llmWorkspace.resetBuiltinTitle')}>
            {t('llmWorkspace.resetBuiltin')}
          </PBtn>
        )}
        <PBtn onClick={onSaveAs} title={t('llmWorkspace.saveAsTitle')}>{t('llmWorkspace.saveAs')}</PBtn>
        <PBtn onClick={onAddPreset}>{t('llmWorkspace.newPreset')}</PBtn>
        {!currentPreset.builtin && presets.length > 1 && (
          <PBtn variant="danger" onClick={onDeletePreset}>{t('llmWorkspace.deletePreset')}</PBtn>
        )}
      </div>
    </div>
  )
}

// ── Connection section (01) ────────────────────────────────────────────
function ConnectionSection({
  preset, serverPreset, onUpdate,
  llmModelsBusy, llmTestBusy,
  onRefreshModels, onTestConnection,
  bottomBorder,
}: {
  preset: LLMPreset
  serverPreset?: LLMPreset
  onUpdate: <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => void
  llmModelsBusy: boolean
  llmTestBusy: boolean
  onRefreshModels: () => void
  onTestConnection: () => void
  bottomBorder?: boolean
}) {
  const { t } = useTranslation()
  return (
    <Section bottomBorder={bottomBorder}>
      <SectionHeader step="01" title={t('llmWorkspace.connection')} hint="openai-compatible" />
      <SectionBody>
        <Field
          label="Base URL"
          required
          help={(
            <Trans
              i18nKey="llmWorkspace.baseUrlHelp"
              components={{ code: <Code /> }}
            />
          )}
        >
          <input
            type="text"
            value={preset.base_url}
            onChange={(e) => onUpdate('base_url', e.target.value)}
            placeholder="https://api.openai.com/v1"
            style={inputStyle}
            onFocus={(e) => e.currentTarget.style.boxShadow = '0 0 0 3px var(--accent-soft)'}
            onBlur={(e) => e.currentTarget.style.boxShadow = 'none'}
          />
        </Field>

        <Field label="API Key" required>
          <InputWithSuffix>
            <SensitiveInput
              value={preset.api_key}
              serverValue={serverPreset?.api_key ?? ''}
              onChange={(v) => onUpdate('api_key', v)}
            />
          </InputWithSuffix>
        </Field>

        <Field label="Model" required>
          <InputWithSuffix
            suffix={
              <ChipButton onClick={onRefreshModels} disabled={llmModelsBusy || !preset.base_url.trim()}>
                {llmModelsBusy ? t('llmWorkspace.fetchingModels') : t('llmWorkspace.fetchModels')}
              </ChipButton>
            }
          >
            {preset.model_ids.length > 0 ? (
              <select
                value={preset.model}
                onChange={(e) => onUpdate('model', e.target.value)}
                style={{ ...inputStyle, paddingRight: 130 }}
              >
                {!preset.model_ids.includes(preset.model) && preset.model && (
                  <option value={preset.model}>{preset.model}</option>
                )}
                {preset.model_ids.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={preset.model}
                onChange={(e) => onUpdate('model', e.target.value)}
                placeholder={t('llmWorkspace.modelPlaceholder')}
                style={{ ...inputStyle, paddingRight: 130 }}
              />
            )}
          </InputWithSuffix>
        </Field>

        <Field label={t('llmWorkspace.endpointStyle')}>
          {/* Test-connection button sits inline at the end of the Segmented control; the result is shown via a toast, not kept on screen. */}
          <div className="flex items-center gap-2">
            <div className="flex-1 min-w-0">
              <Segmented
                value={preset.endpoint}
                onChange={(v) => onUpdate('endpoint', v)}
                options={[
                  { value: 'chat_completions', label: 'CHAT COMPLETIONS' },
                  { value: 'responses', label: 'RESPONSES' },
                ]}
              />
            </div>
            <ChipButton
              onClick={onTestConnection}
              disabled={llmTestBusy || !preset.base_url.trim() || !preset.model.trim()}
            >
              {llmTestBusy ? t('llmWorkspace.testing') : t('llmWorkspace.testConnection')}
            </ChipButton>
          </div>
        </Field>
      </SectionBody>
    </Section>
  )
}

// ── Sampling section (03) ───────────────────────────────────────────────
// -- Advanced params: a details panel collapsed by default, wrapping 03 Sampling + 04 Image preprocessing ------
// Shows a summary line (temp / max / max-side / q) when collapsed; both sub-sections render fully when expanded.
function AdvancedSection({ preset, onUpdate }: {
  preset: LLMPreset
  onUpdate: <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => void
}) {
  const { t } = useTranslation()
  return (
    <details className="group">
      <summary
        className="cursor-pointer list-none flex items-center justify-between gap-2"
        style={{
          padding: '11px 16px 10px',
          // Same bottom border style as SectionHeader; when expanded, the inner SamplingSection's own border continues the separation.
          borderBottom: '1px solid var(--border-subtle)',
        }}
      >
        <h3
          className="m-0 flex items-center gap-2 whitespace-nowrap"
          style={{ fontSize: 'var(--t-md)', fontWeight: 600, letterSpacing: '-0.005em' }}
        >
          <span
            className="text-fg-tertiary text-xs transition-transform group-open:rotate-90 inline-block w-3"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            ▸
          </span>
          <Step>⚙</Step>
          <span>{t('llmWorkspace.advanced')}</span>
        </h3>
        {/* Summary value: shows the current key numbers when collapsed (compact form, so it fits the 360px left column).
         * truncate + min-w-0 makes it ellipsize gracefully instead of forcing the title to wrap. */}
        <span
          className="group-open:hidden truncate min-w-0"
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 'var(--t-xs)',
            color: 'var(--fg-tertiary)',
          }}
          title={`temperature ${preset.temperature} · max_tokens ${preset.max_tokens} · concurrency ${preset.concurrency} · max_requests_per_minute ${preset.max_requests_per_minute || 0} · max_side ${preset.max_side}px · jpeg_quality ${preset.jpeg_quality}`}
        >
          {preset.temperature} · {preset.max_tokens}t · c{preset.concurrency} · m{preset.max_requests_per_minute || 0} · {preset.max_side}px · q{preset.jpeg_quality}
        </span>
      </summary>
      {/* Expanded content: 03 Sampling + 04 Image preprocessing stacked as-is */}
      <SamplingSection preset={preset} onUpdate={onUpdate} bottomBorder />
      <ImageSection preset={preset} onUpdate={onUpdate} />
    </details>
  )
}

function SamplingSection({ preset, onUpdate, bottomBorder }: {
  preset: LLMPreset
  onUpdate: <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => void
  bottomBorder?: boolean
}) {
  const { t } = useTranslation()
  return (
    <Section bottomBorder={bottomBorder}>
      <SectionHeader step="03" title={t('llmWorkspace.sampling')} hint="model-side" />
      <SectionBody>
        <Field label="Temperature" optional={t('llmWorkspace.temperatureHint')}>
          <SliderRow value={preset.temperature} min={0} max={2} step={0.05}
            onChange={(v) => onUpdate('temperature', v)} />
        </Field>
        <Field label="Max tokens">
          <SliderRow value={preset.max_tokens} min={64} max={4096} step={32}
            onChange={(v) => onUpdate('max_tokens', Math.round(v))} integer />
        </Field>
        <Row2>
          <Field label="Timeout" optional="s">
            <input
              type="number" min={5} max={600}
              value={preset.timeout}
              onChange={(e) => onUpdate('timeout', Math.max(5, Number(e.target.value) || 5))}
              style={inputStyle}
            />
          </Field>
          <Field label="Max retries">
            <input
              type="number" min={1} max={10}
              value={preset.max_retries}
              onChange={(e) => onUpdate('max_retries', Math.max(1, Number(e.target.value) || 1))}
              style={inputStyle}
            />
          </Field>
        </Row2>
        <Row2>
          <Field label="Concurrency" optional="requests">
            <input
              type="number" min={1} max={8}
              value={preset.concurrency}
              onChange={(e) => onUpdate('concurrency',
                Math.max(1, Math.min(8, Number(e.target.value) || 1)))}
              style={inputStyle}
            />
          </Field>
          <Field label="Max/min" optional="0 = no limit">
            <input
              type="number" min={0} max={3600}
              value={preset.max_requests_per_minute}
              onChange={(e) => onUpdate('max_requests_per_minute',
                Math.max(0, Math.min(3600, Math.round(Number(e.target.value) || 0))))}
              style={inputStyle}
            />
          </Field>
        </Row2>
        <Row2>
          <Field label="Requests/sec" optional="0 = no limit">
            <input
              type="number" min={0} max={60} step={0.1}
              value={preset.requests_per_second}
              onChange={(e) => onUpdate('requests_per_second',
                Math.max(0, Math.min(60, Number(e.target.value) || 0)))}
              style={inputStyle}
            />
          </Field>
        </Row2>
      </SectionBody>
    </Section>
  )
}

// ── Image preprocessing section (04) ────────────────────────────────────
function ImageSection({ preset, onUpdate, bottomBorder }: {
  preset: LLMPreset
  onUpdate: <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => void
  bottomBorder?: boolean
}) {
  const { t } = useTranslation()
  return (
    <Section bottomBorder={bottomBorder}>
      <SectionHeader step="04" title={t('llmWorkspace.imagePreprocess')} hint="before upload" />
      <SectionBody>
        <Field label="Max side" optional={t('llmWorkspace.maxSideHint')}>
          <SliderRow value={preset.max_side} min={512} max={2048} step={64}
            onChange={(v) => onUpdate('max_side', Math.round(v))} integer />
        </Field>
        <Row2>
          <Field label="JPEG quality">
            <input
              type="number" min={1} max={100}
              value={preset.jpeg_quality}
              onChange={(e) => onUpdate('jpeg_quality',
                Math.max(1, Math.min(100, Number(e.target.value) || 85)))}
              style={inputStyle}
            />
          </Field>
          <Field label="Max size" optional="MB">
            <input
              type="number" min={0.1} max={25} step={0.1}
              value={preset.max_image_mb}
              onChange={(e) => onUpdate('max_image_mb',
                Math.max(0.1, Number(e.target.value) || 5))}
              style={inputStyle}
            />
          </Field>
        </Row2>
        <Help>
          <Trans
            i18nKey="llmWorkspace.imageSizeHint"
            components={{ limit: <b style={{ color: 'var(--fg-secondary)' }} /> }}
          />
        </Help>
      </SectionBody>
    </Section>
  )
}

// ── Composer section (02) — the hero ────────────────────────────────────
// Height fills the left column's total height (grid items-stretch + h-full); the messages area scrolls internally.
function ComposerSection({ preset, onUpdate }: {
  preset: LLMPreset
  onUpdate: <K extends keyof LLMPreset>(field: K, value: LLMPreset[K]) => void
}) {
  const { t } = useTranslation()
  return (
    <div
      className="flex flex-col"
      style={{ height: '100%', minHeight: 0, overflow: 'hidden' }}
    >
      {/* composer-tabbar -- fixed height */}
      <div
        className="flex items-center justify-between shrink-0"
        style={{ padding: '12px 16px', borderBottom: '1px solid var(--border-subtle)' }}
      >
        <div className="flex items-center gap-4">
          <h3 className="flex items-center gap-2 m-0" style={{
            fontSize: 'var(--t-md)', fontWeight: 600, letterSpacing: '-0.005em',
          }}>
            <Step>02</Step>
            <span>{t('llmWorkspace.promptTemplate')}</span>
          </h3>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
            color: 'var(--fg-tertiary)', letterSpacing: '0.04em',
          }}>
            {t('llmWorkspace.messageSummary', { n: preset.messages.length })}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="flex items-center" style={{
            background: 'var(--bg-sunken)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--r-md)',
            padding: '4px 6px 4px 12px',
            gap: 8,
          }}>
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
              color: 'var(--fg-tertiary)', letterSpacing: '0.06em',
              textTransform: 'uppercase',
            }}>{t('llmWorkspace.output')}</span>
            <select
              value={preset.output_format}
              onChange={(e) => onUpdate('output_format', e.target.value as LLMPreset['output_format'])}
              className="cursor-pointer outline-none border-0"
              style={{
                background: 'transparent', color: 'var(--fg-primary)',
                fontSize: 'var(--t-sm)', padding: '4px 6px',
              }}
            >
              <option value="json">{t('llmWorkspace.jsonCaption')}</option>
              <option value="text">{t('llmWorkspace.textCaption')}</option>
            </select>
          </div>
        </div>
      </div>

      {/* msg-list -- flex-1 + scrolls internally */}
      <div
        className="flex-1 min-h-0 overflow-y-auto"
        style={{ padding: '14px 16px' }}
      >
        {preset.endpoint === 'responses' && (
          <div
            style={{
              fontSize: 'var(--t-xs)', color: 'var(--warn)',
              fontFamily: 'var(--font-mono)', letterSpacing: '0.04em',
              marginBottom: 10,
            }}
          >
            {t('llmWorkspace.responsesWarning')}
          </div>
        )}
        <LLMMessagesEditor
          messages={preset.messages}
          onChange={(msgs) => onUpdate('messages', msgs)}
        />
      </div>
    </div>
  )
}

// ── Reusable primitives ─────────────────────────────────────────────────

/** A section within the left column (no separate border + radius; sections are separated by bottomBorder). */
function Section({ children, bottomBorder }: {
  children: React.ReactNode
  bottomBorder?: boolean
}) {
  return (
    <div style={bottomBorder ? { borderBottom: '1px solid var(--border-subtle)' } : undefined}>
      {children}
    </div>
  )
}

function SectionHeader({ step, title, hint }: { step: string; title: string; hint?: string }) {
  return (
    <div
      className="flex items-center justify-between"
      style={{
        padding: '11px 16px 10px',
        borderBottom: '1px solid var(--border-subtle)',
      }}
    >
      <h3 className="m-0 flex items-center gap-2" style={{
        fontSize: 'var(--t-md)', fontWeight: 600, letterSpacing: '-0.005em',
      }}>
        <Step>{step}</Step>
        <span>{title}</span>
      </h3>
      {hint && (
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 'var(--t-xs)',
          color: 'var(--fg-tertiary)',
        }}>
          {hint}
        </span>
      )}
    </div>
  )
}

function SectionBody({ children }: { children: React.ReactNode }) {
  return <div style={{ padding: '12px 16px 14px' }}>{children}</div>
}

function Step({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
      color: 'var(--accent-text, var(--accent))', background: 'var(--accent-soft)',
      borderRadius: 999, padding: '2px 7px', letterSpacing: '0.04em',
    }}>
      {children}
    </span>
  )
}

function Caption({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
      color: 'var(--fg-tertiary)', letterSpacing: '0.08em',
      textTransform: 'uppercase', whiteSpace: 'nowrap',
    }}>
      {children}
    </span>
  )
}

function Field({ label, required, optional, help, children }: {
  label: string
  required?: boolean
  optional?: string
  help?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="grid" style={{ gap: 5, marginBottom: 12 }}>
      <label
        className="flex items-center gap-1.5"
        style={{
          fontFamily: 'var(--font-mono)', fontSize: 'var(--t-2xs)',
          color: 'var(--fg-secondary)', letterSpacing: '0.06em',
          textTransform: 'uppercase',
        }}
      >
        {label}
        {required && <span style={{ color: 'var(--err)' }}>*</span>}
        {optional && (
          <span style={{
            fontWeight: 400, textTransform: 'none', letterSpacing: 0,
            color: 'var(--fg-tertiary)', fontFamily: 'var(--font-sans)',
          }}>
            {optional}
          </span>
        )}
      </label>
      {children}
      {help && <Help>{help}</Help>}
    </div>
  )
}

function Help({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: 'var(--t-xs)', color: 'var(--fg-tertiary)', lineHeight: 1.45,
    }}>
      {children}
    </div>
  )
}

function Code({ children }: { children?: React.ReactNode }) {
  return (
    <code style={{
      fontFamily: 'var(--font-mono)',
      color: 'var(--fg-secondary)',
    }}>
      {children}
    </code>
  )
}

function Row2({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: 12 }}>
      {children}
    </div>
  )
}

function InputWithSuffix({ children, suffix }: {
  children: React.ReactNode
  suffix?: React.ReactNode
}) {
  return (
    <div style={{ position: 'relative' }}>
      {children}
      {suffix && (
        <div
          className="flex items-center gap-0.5"
          style={{
            position: 'absolute', right: 4, top: '50%',
            transform: 'translateY(-50%)',
          }}
        >
          {suffix}
        </div>
      )}
    </div>
  )
}

function ChipButton({ children, onClick, disabled, active }: {
  children: React.ReactNode
  onClick?: () => void
  disabled?: boolean
  active?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      style={{
        background: 'transparent',
        border: '1px solid var(--border-subtle)',
        color: active ? 'var(--accent)' : 'var(--fg-secondary)',
        borderColor: active ? 'var(--accent)' : 'var(--border-subtle)',
        fontSize: 'var(--t-2xs)',
        padding: '4px 8px',
        borderRadius: 'var(--r-sm)',
        fontFamily: 'var(--font-mono)',
        letterSpacing: '0.04em',
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
      onMouseEnter={(e) => {
        if (disabled) return
        e.currentTarget.style.color = 'var(--accent)'
        e.currentTarget.style.borderColor = 'var(--accent)'
      }}
      onMouseLeave={(e) => {
        if (disabled || active) return
        e.currentTarget.style.color = 'var(--fg-secondary)'
        e.currentTarget.style.borderColor = 'var(--border-subtle)'
      }}
    >
      {children}
    </button>
  )
}

function PBtn({ children, onClick, variant, title }: {
  children: React.ReactNode
  onClick?: () => void
  variant?: 'default' | 'primary' | 'danger'
  title?: string
}) {
  const base: React.CSSProperties = {
    background: variant === 'primary' ? 'var(--accent)' : 'transparent',
    border: '1px solid transparent',
    color: variant === 'primary' ? 'var(--accent-fg)' : 'var(--fg-secondary)',
    padding: variant === 'primary' ? '7px 14px' : '6px 10px',
    borderRadius: 'var(--r-md)',
    fontSize: 'var(--t-sm)',
    fontWeight: variant === 'primary' ? 500 : 400,
    display: 'inline-flex', alignItems: 'center', gap: 6,
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  }
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      style={base}
      onMouseEnter={(e) => {
        if (variant === 'primary') {
          e.currentTarget.style.background = 'var(--accent-hover)'
        } else if (variant === 'danger') {
          e.currentTarget.style.background = 'var(--err-soft)'
          e.currentTarget.style.color = 'var(--err)'
        } else {
          e.currentTarget.style.background = 'var(--bg-overlay)'
          e.currentTarget.style.color = 'var(--fg-primary)'
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = variant === 'primary' ? 'var(--accent)' : 'transparent'
        e.currentTarget.style.color = variant === 'primary' ? 'var(--accent-fg)' : 'var(--fg-secondary)'
      }}
    >
      {children}
    </button>
  )
}

function Segmented<T extends string>({ value, onChange, options }: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: string }[]
}) {
  return (
    <div className="seg flex w-full">
      {options.map((opt) => {
        const on = value === opt.value
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={`seg-item flex-1 min-h-[28px] ${on ? 'is-active' : ''}`}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}

function SliderRow({ value, min, max, step, onChange, integer }: {
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  integer?: boolean
}) {
  const clamp = (v: number) => Math.max(min, Math.min(max, v))
  return (
    <div className="grid items-center" style={{ gridTemplateColumns: '1fr 64px', gap: 10 }}>
      <input
        type="range"
        min={min} max={max} step={step ?? 1}
        value={value}
        onChange={(e) => onChange(clamp(Number(e.target.value)))}
        style={{ width: '100%', accentColor: 'var(--accent)' }}
      />
      <input
        type="number"
        min={min} max={max} step={step ?? 1}
        value={integer ? Math.round(value) : value}
        onChange={(e) => onChange(clamp(Number(e.target.value)))}
        style={{
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--r-sm)',
          padding: '4px 8px',
          fontFamily: 'var(--font-mono)',
          fontSize: 'var(--t-xs)',
          color: 'var(--fg-primary)',
          textAlign: 'right',
          outline: 'none',
          width: '100%',
        }}
      />
    </div>
  )
}

function SensitiveInput({ value, serverValue, onChange }: {
  value: string
  serverValue: string
  onChange: (v: string) => void
}) {
  const { t } = useTranslation()
  const masked = value === MASK
  return (
    <input
      type="password"
      value={masked ? '' : value}
      placeholder={serverValue === MASK ? t('llmWorkspace.secretSavedPlaceholder') : ''}
      onChange={(e) => onChange(e.target.value || MASK)}
      autoComplete="new-password"
      data-lpignore="true"
      data-1p-ignore
      data-form-type="other"
      style={inputStyle}
    />
  )
}
