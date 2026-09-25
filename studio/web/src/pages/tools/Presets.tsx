import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  api,
  type ApiError,
  type ConfigData,
  type PresetSummary,
  type SchemaResponse,
} from '../../api/client'
import ConfigSkeleton from '../../components/ConfigSkeleton'
import { useDialog } from '../../components/Dialog'
import KebabMenu, { type KebabItem } from '../../components/ds/KebabMenu'
import PageHead from '../../components/ds/PageHead'
import YamlLines from '../../components/ds/YamlLines'
import PathPicker from '../../components/PathPicker'
import SchemaForm from '../../components/SchemaForm'
import { useToast } from '../../components/Toast'
import { schemaEnumLabel } from '../../lib/schema'
import { useSettingsDrawer } from '../../lib/SettingsDrawer'
import { useAdvancedMode } from '../../lib/useAdvancedMode'
import {
  PRESET_NAME_RE,
  defaultsFromSchema,
  loadPresetDescriptions,
  savePresetDescriptions,
} from '../../lib/preset-helpers'

// ── TOML 生成（键按字母排序，值尽量保留原始类型） ──────────────────────────
function toTomlValue(v: unknown): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number') return String(v)
  if (Array.isArray(v)) return '[' + v.map(toTomlValue).join(', ') + ']'
  if (typeof v === 'object') {
    const lines: string[] = []
    for (const [k, vv] of Object.entries(v as Record<string, unknown>)) {
      lines.push(`  ${k} = ${toTomlValue(vv)}`)
    }
    return '{\n' + lines.join('\n') + '\n}'
  }
  const s = String(v)
  if (/[\n"'#[\]{}]/.test(s)) return `'''\n${s}\n'''`
  if (s.includes(' ') || s === '' || /[^\w.\-]/.test(s)) return `"${s}"`
  return s
}

function generateToml(config: ConfigData): string {
  const keys = Object.keys(config).sort()
  return keys.map((k) => `${k} = ${toTomlValue(config[k])}`).join('\n')
}

// 表格列 / 详情卡用：从一份 config 派生 optimizer / rank / resolution。
function cfgRank(c?: ConfigData): string {
  const v = c?.lora_rank
  return v === undefined || v === null ? '—' : String(v)
}
function cfgAlpha(c?: ConfigData): string {
  const v = c?.lora_alpha
  return v === undefined || v === null ? '—' : String(v)
}
function cfgRes(c?: ConfigData): string {
  const v = c?.resolution
  return v === undefined || v === null ? '—' : String(v)
}

// 预设名校验 / 描述存储 / schema 默认值 抽到 lib/preset-helpers.ts，
// 跟 Train 页面「新建预设」内联表单共享，避免两份维护。

// 上传冲突时,后端 409 body 透传到这里;用户决定覆盖 / 另存为 / 取消。
interface ConflictState {
  config: ConfigData
  desc: string
  suggestedName: string
}

type ConflictChoice =
  | { kind: 'overwrite' }
  | { kind: 'saveAs'; name: string }
  | { kind: 'cancel' }

export default function PresetsPage() {
  const { t, i18n } = useTranslation()
  const { toast } = useToast()
  const { confirm } = useDialog()
  const settingsDrawer = useSettingsDrawer()

  // ── backend state ──
  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [presets, setPresets] = useState<PresetSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [busy, setBusy] = useState(false)
  const [autoSyncPaths, setAutoSyncPaths] = useState<boolean>(true)
  // 4 个模型字段当前 Settings 算出的绝对路径（reset 按钮 + 新建预设默认值）
  const [modelPathDefaults, setModelPathDefaults] = useState<Record<string, string>>({})
  // prototype 表格列（Optimizer/Rank/Res）需要每个 preset 的 config —— 列表 API
  // 只回 name/path/updated，这里按需拉全部 config 填表格（preset 数量很少）。
  const [configCache, setConfigCache] = useState<Record<string, ConfigData>>({})

  // 已保存快照，用于 dirty 判定
  const savedJsonRef = useRef<string | null>(null)
  const [droppedFields, setDroppedFields] = useState<string[]>([])
  const [defaultedFields, setDefaultedFields] = useState<string[]>([])

  // 描述
  const [descriptions, setDescriptions] = useState<Record<string, string>>(loadPresetDescriptions)
  const [descDraft, setDescDraft] = useState('')
  const [descDirty, setDescDirty] = useState(false)

  // 新建模式输入
  const [newName, setNewName] = useState('')
  const [newNameError, setNewNameError] = useState('')
  const isNew = selected === null

  // ── 上传冲突 dialog 状态 + 命令式 resolver ──
  const [conflict, setConflict] = useState<ConflictState | null>(null)
  const conflictResolveRef = useRef<((c: ConflictChoice) => void) | null>(null)
  const askConflict = (state: ConflictState): Promise<ConflictChoice> =>
    new Promise((resolve) => {
      conflictResolveRef.current = resolve
      setConflict(state)
    })
  const resolveConflict = (choice: ConflictChoice) => {
    setConflict(null)
    const r = conflictResolveRef.current
    conflictResolveRef.current = null
    r?.(choice)
  }

  // ── UI 状态 ──
  // editorOpen：prototype 把整套 schema 编辑收进「Edit config / New preset」模态。
  const [editorOpen, setEditorOpen] = useState(false)
  const [tomlOpen, setTomlOpen] = useState(false)
  const [exportDialogOpen, setExportDialogOpen] = useState(false)
  const [showImportPathPicker, setShowImportPathPicker] = useState(false)
  const [filter, setFilter] = useState('')
  const [summaryFormat, setSummaryFormat] = useState<'yaml' | 'toml'>('yaml')
  const [previewYaml, setPreviewYaml] = useState('')
  const [advancedMode, toggleAdvancedMode] = useAdvancedMode()
  const newNameInputRef = useRef<HTMLInputElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // 4 个模型字段（用于新建预设默认值 / reset 按钮）。同 Train.tsx 的 GLOBAL_MODEL_FIELDS。
  const MODEL_PATH_FIELDS = useMemo(() => [
    'transformer_path', 'vae_path', 'text_encoder_path', 't5_tokenizer_path',
  ], [])

  // ── 加载 schema + 预设列表 + Settings toggle + 模型路径默认 ──
  useEffect(() => {
    api.schema().then(setSchema).catch((e) => toast(t('presets.loadSchemaFailed', { error: e }), 'error'))
    refreshList()
    api.getSecrets().then((s) => setAutoSyncPaths(s.models?.auto_sync_paths ?? true)).catch(() => {})
    api.getModelPathDefaults().then(setModelPathDefaults).catch(() => {})
  }, [t, toast])

  const refreshList = () => {
    api.listPresets().then((list) => {
      setPresets(list)
      // 拉每个 preset 的 config 填表格列（best-effort，失败列显示 —）。
      list.forEach((p) => {
        api.getPreset(p.name)
          .then((c) => setConfigCache((m) => ({ ...m, [p.name]: c })))
          .catch(() => {})
      })
    }).catch(() => setPresets([]))
  }

  // ── 选 preset 切换 ──
  useEffect(() => {
    if (!selected) {
      if (schema) {
        const defaults = { ...defaultsFromSchema(schema), ...modelPathDefaults }
        setConfig(defaults)
        savedJsonRef.current = JSON.stringify(defaults)
        setNewName('')
        setDescDraft('')
        setDescDirty(false)
        setDroppedFields([])
        setDefaultedFields([])
      } else {
        setConfig(null)
        savedJsonRef.current = null
        setNewName('')
        setDescDraft('')
        setDescDirty(false)
        setDroppedFields([])
        setDefaultedFields([])
      }
      setNewNameError('')
      return
    }
    api.getPresetWithWarnings(selected).then(({ config: data, dropped_fields, defaulted_fields }) => {
      setConfig(data)
      savedJsonRef.current = JSON.stringify(data)
      setDroppedFields(dropped_fields)
      setDefaultedFields(defaulted_fields)
      setDescDraft(descriptions[selected] ?? '')
      setDescDirty(false)
    }).catch((e) => {
      toast(t('presets.loadFailed', { error: e }), 'error')
      setSelected(null)
    })
    // modelPathDefaults 故意排除：late-arrival 由下一个 useEffect 处理
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, schema, descriptions, t, toast])

  // modelPathDefaults 异步晚到时，新建模式下用户没改过就地覆盖 4 字段为绝对路径。
  useEffect(() => {
    if (selected !== null) return
    if (!schema || !config) return
    if (Object.keys(modelPathDefaults).length === 0) return
    const currentJson = JSON.stringify(config)
    if (currentJson !== savedJsonRef.current) return
    let needsUpdate = false
    for (const f of MODEL_PATH_FIELDS) {
      if (modelPathDefaults[f] && config[f] !== modelPathDefaults[f]) {
        needsUpdate = true
        break
      }
    }
    if (!needsUpdate) return
    const next = { ...config, ...modelPathDefaults }
    setConfig(next)
    savedJsonRef.current = JSON.stringify(next)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelPathDefaults, selected, schema])

  // ── 首次拿到列表后：自动选最近一个，省一次「切换」点击 ──
  const autoSelectedRef = useRef(false)
  useEffect(() => {
    if (autoSelectedRef.current) return
    if (presets.length > 0 && selected === null) {
      autoSelectedRef.current = true
      setSelected(presets[0].name)
    } else if (presets.length === 0 && schema) {
      autoSelectedRef.current = true
    }
  }, [presets, selected, schema])

  // 编辑器开着时 Esc 关闭（dirty 时仍可关 —— 改动留在内存，跟切 preset 一致）。
  useEffect(() => {
    if (!editorOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setEditorOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [editorOpen])

  // ── 派生 ──
  const dirty = useMemo(() => {
    if (!config) return false
    return JSON.stringify(config) !== savedJsonRef.current
  }, [config])
  const hasAnyChange = dirty || descDirty

  // auto_sync_paths ON：预设里 4 模型字段灰显。OFF：可编辑 + 重置按钮。
  const disabledFields = autoSyncPaths ? MODEL_PATH_FIELDS : []
  const disabledHints = useMemo(() => {
    const h: Record<string, React.ReactNode> = {}
    if (autoSyncPaths) {
      const node = (
        <>
          {t('train.globalAutoLockedPrefix')} ·{' '}
          <button
            type="button"
            onClick={() => settingsDrawer.open({ section: 'models' })}
            className="bg-transparent border-none p-0 underline text-warn hover:opacity-80 cursor-pointer"
          >
            {t('train.globalAutoLockedLink')}
          </button>
        </>
      )
      for (const f of MODEL_PATH_FIELDS) h[f] = node
    }
    return h
  }, [t, autoSyncPaths, MODEL_PATH_FIELDS, settingsDrawer])
  const autoHints = useMemo(() => {
    const h: Record<string, string> = {}
    if (!autoSyncPaths) {
      for (const f of MODEL_PATH_FIELDS) h[f] = t('train.globalAutoEditableHint')
    }
    return h
  }, [t, autoSyncPaths, MODEL_PATH_FIELDS])

  const fieldSuffixes = useMemo(() => {
    if (autoSyncPaths) return {}
    if (!config) return {}
    if (Object.keys(modelPathDefaults).length === 0) return {}
    const out: Record<string, React.ReactNode> = {}
    for (const f of MODEL_PATH_FIELDS) {
      const dv = modelPathDefaults[f]
      if (typeof dv !== 'string' || !dv) continue
      out[f] = (
        <button
          type="button"
          onClick={() => setConfig({ ...config, [f]: dv })}
          className="btn btn-ghost btn-sm shrink-0"
          title={t('train.resetToGlobalDefaultTitle')}
        >
          {t('train.resetToGlobalDefault')}
        </button>
      )
    }
    return out
  }, [autoSyncPaths, modelPathDefaults, config, t, MODEL_PATH_FIELDS])

  // ── 操作 ──
  const handleSave = async () => {
    const name = isNew ? newName.trim() : selected
    if (!name) {
      setNewNameError(t('presets.nameRequired'))
      newNameInputRef.current?.focus()
      return
    }
    if (!config) return
    if (isNew) {
      if (!PRESET_NAME_RE.test(name)) { setNewNameError(t('presets.nameInvalid')); return }
      if (presets.find((p) => p.name === name)) { setNewNameError(t('presets.nameExists')); return }
    }
    setBusy(true)
    try {
      await api.savePreset(name, config)
      if (descDraft) {
        const next = { ...descriptions, [name]: descDraft }
        setDescriptions(next); savePresetDescriptions(next)
      } else if (descriptions[name]) {
        const { [name]: _, ...rest } = descriptions
        setDescriptions(rest); savePresetDescriptions(rest)
      }
      savedJsonRef.current = JSON.stringify(config)
      setConfigCache((m) => ({ ...m, [name]: config }))
      setDescDirty(false)
      if (isNew) {
        setSelected(name)
        setNewName('')
        setNewNameError('')
        toast(t('presets.created', { name }), 'success')
      } else {
        toast(t('presets.saved'), 'success')
      }
      setEditorOpen(false)
      refreshList()
    } catch (e) { toast(String(e), 'error') }
    finally { setBusy(false) }
  }

  // "复制副本":Save-As 语义 —— 把当前 config 写到新名字下,refresh + 自动选中。
  const handleDuplicate = async (name: string | null = selected) => {
    if (busy || !name) return
    const src = name === selected && config
      ? config
      : configCache[name] ?? await api.getPreset(name).catch(() => null)
    if (!src) return
    const srcDesc = name === selected ? descDraft : descriptions[name] ?? ''
    const baseName = name
    let candidate = `${baseName}-copy`
    let i = 2
    while (presets.find((p) => p.name === candidate)) {
      candidate = `${baseName}-copy-${i++}`
    }
    setBusy(true)
    try {
      await api.savePreset(candidate, src)
      if (srcDesc) {
        const next = { ...descriptions, [candidate]: srcDesc }
        setDescriptions(next); savePresetDescriptions(next)
      }
      refreshList()
      setSelected(candidate)
      toast(t('presets.duplicated', { name: candidate }), 'success')
    } catch (e) { toast(String(e), 'error') }
    finally { setBusy(false) }
  }

  // 「+ New preset」：进新建模式 + 打开编辑器（schema 默认值由 selected→null effect 预填）。
  const handleNew = () => {
    setSelected(null)
    setEditorOpen(true)
  }

  const handleDelete = async (name: string | null = selected) => {
    if (!name) return
    if (!(await confirm(t('presets.confirmDelete', { name }), { tone: 'danger', okText: t('common.delete') }))) return
    setBusy(true)
    const target = name
    api.deletePreset(target).then(() => {
      const { [target]: _, ...rest } = descriptions
      setDescriptions(rest); savePresetDescriptions(rest)
      setConfigCache((m) => { const { [target]: _drop, ...keep } = m; return keep })
      if (target === selected) setSelected(null)
      refreshList()
      toast(t('presets.deleted'), 'success')
    }).catch((e) => toast(String(e), 'error')).finally(() => setBusy(false))
  }

  const currentExportName = () => (isNew ? newName.trim() : selected) || 'preset'

  const downloadCurrentPreset = () => {
    if (!config) return
    if (isNew || !selected || hasAnyChange) {
      toast(t('presets.saveBeforeDownload'), 'info')
      return
    }
    const a = document.createElement('a')
    a.href = api.presetDownloadUrl(selected)
    a.download = `${selected}.yaml`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const exportCurrentPresetToDataExports = async () => {
    if (!config) return
    setBusy(true)
    try {
      const result = await api.exportPresetToDataExports(currentExportName(), config)
      toast(t('presets.exportedToDataExports', { filename: result.filename, path: result.path }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  // 「导入」：上传 / server path → 后端校验落盘 → refresh + 选中；409 冲突弹三选一。
  const handleImportedPreset = (name: string) => {
    refreshList()
    setSelected(name)
    toast(t('presets.imported', { name }), 'success')
  }

  const handleImportConflict = async (err: ApiError): Promise<boolean> => {
    if (err.status === 409 && err.detail && typeof err.detail === 'object') {
      const d = err.detail as { config?: ConfigData; suggested_name?: string }
      if (!d.config || !d.suggested_name) { toast(String(err), 'error'); return true }
      const choice = await askConflict({
        config: d.config, desc: '', suggestedName: d.suggested_name,
      })
      if (choice.kind === 'cancel') return true
      const target = choice.kind === 'overwrite' ? d.suggested_name : choice.name
      setBusy(true)
      try {
        await api.savePreset(target, d.config)
        handleImportedPreset(target)
      } catch (saveErr) { toast(String(saveErr), 'error') }
      finally { setBusy(false) }
      return true
    }
    return false
  }

  const handleImportFile = async (f: File) => {
    let imported: { name: string }
    try {
      imported = await api.importPreset(f)
    } catch (e) {
      const err = e as ApiError
      if (await handleImportConflict(err)) return
      toast(String(e), 'error')
      return
    }
    handleImportedPreset(imported.name)
  }

  const handleImportFromPath = async (path: string) => {
    setShowImportPathPicker(false)
    setBusy(true)
    try {
      const imported = await api.importPresetFromPath(path)
      handleImportedPreset(imported.name)
    } catch (e) {
      const err = e as ApiError
      if (await handleImportConflict(err)) return
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const onImportClick = () => fileInputRef.current?.click()

  const saveDisabled =
    busy
    || !config
    || (isNew && !newName.trim())
    || (!isNew && !hasAnyChange)

  const selectedConfig = selected ? configCache[selected] : undefined

  // Row click / row menu: switch the selection; drop the previous config first
  // so the editor never shows (or saves) the old preset under the new name.
  const selectPreset = (name: string) => {
    if (name === selected) return
    setConfig(null)
    setSelected(name)
  }

  // Summary YAML: the same pruned dump the preset file gets on save.
  const summarySource = selected ? (selectedConfig ?? config) : null
  useEffect(() => {
    if (!summarySource) { setPreviewYaml(''); return }
    let alive = true
    api.previewConfigYaml(summarySource)
      .then((res) => { if (alive) setPreviewYaml(res.yaml) })
      .catch(() => { if (alive) setPreviewYaml('') })
    return () => { alive = false }
  }, [summarySource])

  const fmtUpdated = (ts: number) => {
    const d = new Date(ts * 1000)
    const time = new Intl.DateTimeFormat(i18n.language, { hour: '2-digit', minute: '2-digit' }).format(d)
    if (d.toDateString() === new Date().toDateString()) return t('presets.todayAt', { time })
    return new Intl.DateTimeFormat(i18n.language, { day: 'numeric', month: 'short' }).format(d)
  }

  // ── 渲染 ──
  const q = filter.trim().toLowerCase()
  const shownPresets = q
    ? presets.filter((p) => p.name.toLowerCase().includes(q) || (descriptions[p.name] ?? '').toLowerCase().includes(q))
    : presets
  const summaryConfig = selected ? (selectedConfig ?? config ?? undefined) : undefined
  const yamlFieldCount = previewYaml ? previewYaml.split('\n').filter((l) => /^[^\s#]/.test(l)).length : 0

  const headMenu: KebabItem[] = [
    { label: t('presets.importUpload'), onSelect: onImportClick, disabled: busy },
    { label: t('presets.importPath'), onSelect: () => setShowImportPathPicker(true), disabled: busy },
    { label: t('presets.exportYaml'), onSelect: () => setExportDialogOpen(true), disabled: busy || !selected || !config },
  ]
  const rowMenu = (name: string): KebabItem[] => [
    { label: t('presets.editConfig'), onSelect: () => { selectPreset(name); setEditorOpen(true) }, disabled: busy },
    { label: t('presets.duplicate'), onSelect: () => void handleDuplicate(name), disabled: busy },
    { label: t('presets.exportYaml'), onSelect: () => { selectPreset(name); setExportDialogOpen(true) }, disabled: busy },
    { label: t('common.delete'), onSelect: () => void handleDelete(name), tone: 'err', disabled: busy },
  ]
  const kv = (k: string, v: React.ReactNode) => (
    <div className="ds-kv" key={k}><span className="ds-k">{k}</span><span className="ds-v">{v}</span></div>
  )
  const raw = (v: unknown) => (v === undefined || v === null || v === '' ? '—' : Array.isArray(v) ? `[${v.join(', ')}]` : String(v))

  return (
    <div className="fade-in" style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,.yaml,.yml"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void handleImportFile(f)
          if (fileInputRef.current) fileInputRef.current.value = ''
        }}
      />

      <PageHead
        eyebrow={t('presets.eyebrow')}
        title={t('presets.title')}
        subtitle={t('presets.subtitle')}
        tools={
          <>
            <KebabMenu trigger="icon" label={t('presets.headMenu')} items={headMenu} />
            <button type="button" onClick={handleNew} disabled={busy} className="ds-btn-primary">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>
              {t('presets.newPresetBtn')}
            </button>
          </>
        }
      />

      <div className="ds-scroll ds-tight" style={{ minHeight: 0 }}>
        <div className="ds-presets-grid">

          {/* ── list ── */}
          <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
            <div className="ds-card-head ds-pad">
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="ds-card-title">{t('presets.allTitle')}</div>
                <div className="ds-card-sub">{t('presets.allSub', { count: presets.length })}</div>
              </div>
              <div className="ds-card-tools">
                <span className="ds-search" style={{ width: 200 }}>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" /></svg>
                  <input className="ds-inp" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder={t('presets.filterPlaceholder')} aria-label={t('presets.filterPlaceholder')} />
                </span>
              </div>
            </div>
            <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
              {presets.length === 0 ? (
                <div className="ds-empty">{t('presets.empty')}</div>
              ) : shownPresets.length === 0 ? (
                <div className="ds-empty">{t('presets.noMatch', { search: filter })}</div>
              ) : (
                <table className="ds-tbl">
                  <thead>
                    <tr>
                      <th>{t('presets.colName')}</th>
                      <th>{t('presets.colType')}</th>
                      <th>{t('presets.colRank')}</th>
                      <th>{t('presets.colRes')}</th>
                      <th>{t('presets.colOptimizer')}</th>
                      <th>{t('presets.colUpdated')}</th>
                      <th style={{ width: 40 }} />
                    </tr>
                  </thead>
                  <tbody>
                    {shownPresets.map((p) => {
                      const active = p.name === selected
                      const c = configCache[p.name]
                      const lt = c?.lora_type as string | undefined
                      return (
                        <tr
                          key={p.name}
                          onClick={() => selectPreset(p.name)}
                          style={{ cursor: 'pointer', background: active ? 'var(--green-soft)' : undefined }}
                          aria-selected={active}
                        >
                          <td>
                            <span className="ds-cell-main">
                              <span className="ds-sect-icon" style={active ? { background: '#fff' } : undefined}>
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><circle cx="6" cy="6" r="2.5" /><circle cx="6" cy="18" r="2.5" /><circle cx="18" cy="12" r="2.5" /><path d="M8.5 6h4a3 3 0 0 1 3 3v.5M8.5 18h4a3 3 0 0 0 3-3v-.5" /></svg>
                              </span>
                              <span className="ds-mono" style={{ fontWeight: 600, minWidth: 0 }}>
                                {p.name}
                                {descriptions[p.name] && <span className="ds-cell-key">{descriptions[p.name]}</span>}
                              </span>
                            </span>
                          </td>
                          <td>
                            {lt ? (
                              <span className={`ds-badge ${c?.model_family === 'krea2' ? 'ds-info' : active ? 'ds-ok' : 'ds-mute'}`}>
                                {schemaEnumLabel('lora_type', lt, t)}
                              </span>
                            ) : '—'}
                          </td>
                          <td className="ds-num">{cfgRank(c)}</td>
                          <td className="ds-num">{cfgRes(c)}</td>
                          <td>{c?.optimizer_type ? schemaEnumLabel('optimizer_type', String(c.optimizer_type), t) : '—'}</td>
                          <td className="ds-muted" style={{ whiteSpace: 'nowrap' }}>{fmtUpdated(p.updated_at)}</td>
                          <td onClick={(e) => e.stopPropagation()}>
                            <KebabMenu label={t('presets.rowMenu', { name: p.name })} items={rowMenu(p.name)} />
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          {/* ── summary ── */}
          <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
            {selected && summaryConfig ? (
              <>
                <div className="ds-card-head ds-pad">
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="ds-card-title ds-mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{selected}</div>
                    <div className="ds-card-sub">
                      {[descriptions[selected], yamlFieldCount ? t('presets.fieldsN', { count: yamlFieldCount }) : null].filter(Boolean).join(' · ')}
                    </div>
                  </div>
                  <div className="ds-card-tools">
                    <button type="button" className="ds-ctl ds-ghost" onClick={() => setEditorOpen(true)} disabled={busy || !config}>
                      {t('presets.edit')}
                    </button>
                    <KebabMenu
                      trigger="icon"
                      label={t('presets.rowMenu', { name: selected })}
                      items={[
                        { label: t('presets.duplicate'), onSelect: () => void handleDuplicate(selected), disabled: busy || !config },
                        { label: t('presets.exportYaml'), onSelect: () => setExportDialogOpen(true), disabled: busy || !config },
                        { label: t('common.delete'), onSelect: () => void handleDelete(selected), tone: 'err', disabled: busy },
                      ]}
                    />
                  </div>
                </div>
                <div style={{ padding: '0 17px 12px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 16px' }}>
                  {kv(t('presets.kvNetwork'), raw(summaryConfig.lora_type))}
                  {kv('rank / alpha', `${cfgRank(summaryConfig)} / ${cfgAlpha(summaryConfig)}`)}
                  {kv(t('presets.colOptimizer'), raw(summaryConfig.optimizer_type))}
                  {kv('learning_rate', raw(summaryConfig.learning_rate))}
                  {kv(t('presets.kvResolution'), raw(summaryConfig.resolution))}
                  {kv('batch × ga', `${raw(summaryConfig.batch_size)} × ${raw(summaryConfig.grad_accum)}`)}
                  {kv(t('presets.kvEpochs'), raw(summaryConfig.epochs))}
                  {kv('Timestep', raw(summaryConfig.timestep_sampling))}
                </div>
                <div className="ds-tabs" style={{ padding: '0 17px' }} role="tablist">
                  {(['yaml', 'toml'] as const).map((f) => (
                    <button
                      key={f}
                      type="button"
                      role="tab"
                      aria-selected={summaryFormat === f}
                      className={`ds-tab${summaryFormat === f ? ' ds-is-active' : ''}`}
                      onClick={() => setSummaryFormat(f)}
                    >
                      {f.toUpperCase()}
                    </button>
                  ))}
                  <span style={{ flex: 1 }} />
                  <button
                    type="button"
                    className="ds-ctl-note"
                    style={{ alignSelf: 'center' }}
                    onClick={() => {
                      const text = summaryFormat === 'yaml' ? previewYaml : generateToml(summaryConfig)
                      navigator.clipboard.writeText(text)
                        .then(() => toast(t('presets.copied'), 'success'))
                        .catch(() => toast(t('presets.copyFailed'), 'error'))
                    }}
                  >
                    {t('common.copy')}
                  </button>
                </div>
                <pre className="ds-yaml" style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
                  {summaryFormat === 'yaml'
                    ? (previewYaml ? <YamlLines text={`# ${t('presets.yamlPruned')}\n${previewYaml}`} /> : t('common.loading'))
                    : generateToml(summaryConfig)}
                </pre>
              </>
            ) : (
              <div className="ds-empty">{t('presets.selectHint')}</div>
            )}
          </div>
        </div>
      </div>

      {/* ── 编辑器模态（New / Edit config） ── */}
      {editorOpen && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-start justify-center"
          style={{ background: 'rgba(23,24,26,0.42)', backdropFilter: 'blur(2px)', paddingTop: '6vh', paddingBottom: '6vh' }}
          onMouseDown={(e) => { if (e.target === e.currentTarget) setEditorOpen(false) }}
        >
          <div
            className="card"
            style={{ width: '92%', maxWidth: 760, maxHeight: '88vh', display: 'flex', flexDirection: 'column', boxShadow: 'var(--sh-xl)', overflow: 'hidden' }}
          >
            {/* header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '18px 24px 16px', borderBottom: '1px solid var(--border-subtle)' }}>
              <h2 style={{ margin: 0, fontSize: 17, fontWeight: 600, letterSpacing: '-.02em' }}>
                {isNew ? t('presets.newPresetBtn') : <>{t('presets.editPrefix')} · <span className="mono">{selected}</span></>}
              </h2>
              <span style={{ flex: 1 }} />
              <span className="ds-seg">
                <button
                  type="button"
                  onClick={() => advancedMode && toggleAdvancedMode()}
                  className={`ds-seg-item${!advancedMode ? ' ds-is-active' : ''}`}
                >
                  {t('train.simpleMode')}
                </button>
                <button
                  type="button"
                  onClick={() => !advancedMode && toggleAdvancedMode()}
                  className={`ds-seg-item${advancedMode ? ' ds-is-active' : ''}`}
                >
                  {t('train.advancedMode')}
                </button>
              </span>
              <button type="button" onClick={() => setEditorOpen(false)} className="ds-kebab" aria-label={t('common.cancel')}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>
              </button>
            </div>

            {/* scroll body */}
            <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '18px 24px 22px', display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* name / description */}
              <div style={{ display: 'flex', gap: 12 }}>
                {isNew ? (
                  <label className="flex flex-col gap-1.5" style={{ flex: 1 }}>
                    <span className="ds-cap">{t('presets.presetName')}</span>
                    <input
                      ref={newNameInputRef}
                      autoFocus
                      className="ds-inp ds-mono"
                      placeholder="my-training-preset"
                      value={newName}
                      onChange={(e) => { setNewName(e.target.value); setNewNameError('') }}
                      disabled={busy}
                    />
                    {newNameError && <span className="text-xs text-err">{newNameError}</span>}
                  </label>
                ) : (
                  <label className="flex flex-col gap-1.5" style={{ flex: 1 }}>
                    <span className="ds-cap">{t('presets.nameReadonly')}</span>
                    <input className="ds-inp ds-mono" value={selected ?? ''} disabled readOnly />
                  </label>
                )}
                <label className="flex flex-col gap-1.5" style={{ flex: 1.5 }}>
                  <span className="ds-cap">{t('presets.description')}</span>
                  <input
                    className="ds-inp"
                    placeholder={t('presets.descPlaceholder')}
                    value={descDraft}
                    onChange={(e) => { setDescDraft(e.target.value); setDescDirty(true) }}
                    disabled={busy}
                  />
                </label>
              </div>

              {(droppedFields.length > 0 || defaultedFields.length > 0) && (
                <div className="ds-note ds-warn" style={{ display: 'block' }}>
                  <span className="font-semibold">{t('presets.compatNoticeTitle')}</span>
                  {droppedFields.length > 0 && (
                    <div>{t('presets.droppedFieldsBody')}<code className="ml-1 text-[11px] opacity-80">{droppedFields.join(', ')}</code></div>
                  )}
                  {defaultedFields.length > 0 && (
                    <div>{t('presets.defaultedFieldsBody')}<code className="ml-1 text-[11px] opacity-80">{defaultedFields.join(', ')}</code></div>
                  )}
                </div>
              )}

              {!schema || !config ? (
                <div className="h-[200px]"><ConfigSkeleton variant="flat" label={t('presets.loadingConfig')} /></div>
              ) : (
                <SchemaForm
                  schema={schema}
                  values={config}
                  onChange={setConfig}
                  disabledFields={disabledFields}
                  disabledHints={disabledHints}
                  autoHints={autoHints}
                  fieldSuffixes={fieldSuffixes}
                  advancedMode={advancedMode}
                />
              )}

              {/* TOML preview（折叠） */}
              {config && Object.keys(config).length > 0 && (
                <section className={`rounded-md border border-subtle bg-surface ${tomlOpen ? 'px-3.5 py-2.5' : 'px-3.5 py-1.5'}`}>
                  <button
                    type="button"
                    onClick={() => setTomlOpen((v) => !v)}
                    className="w-full flex items-center gap-2 bg-transparent border-none p-0 cursor-pointer text-left"
                  >
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-info shrink-0" />
                    <span className="caption">{t('presets.tomlPreview')}</span>
                    <span className="flex-1" />
                    {tomlOpen && (
                      <button
                        className="btn btn-ghost btn-sm text-xs"
                        onClick={(e) => {
                          e.stopPropagation()
                          navigator.clipboard.writeText(generateToml(config))
                            .then(() => toast(t('presets.copied'), 'success'))
                            .catch(() => toast(t('presets.copyFailed'), 'error'))
                        }}
                      >{t('common.copy')}</button>
                    )}
                    <span className="text-fg-tertiary">{tomlOpen ? '▾' : '▸'}</span>
                  </button>
                  {tomlOpen && (
                    <pre className="m-0 mt-2.5 p-3 bg-sunken rounded-sm font-mono text-xs text-fg-secondary leading-[1.7] whitespace-pre-wrap break-words max-h-80 overflow-auto">
                      {generateToml(config)}
                    </pre>
                  )}
                </section>
              )}
            </div>

            {/* footer */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, padding: '14px 24px', borderTop: '1px solid var(--border-subtle)' }}>
              <button type="button" className="ds-ctl" onClick={() => setEditorOpen(false)}>{t('common.cancel')}</button>
              <button type="button" className="ds-btn-primary" onClick={handleSave} disabled={saveDisabled}>
                {isNew ? t('common.create') : t('common.save')}
              </button>
            </div>
          </div>
        </div>
      )}

      {exportDialogOpen && (
        <PresetExportDialog
          onDownload={() => {
            setExportDialogOpen(false)
            downloadCurrentPreset()
          }}
          onDataExports={() => {
            setExportDialogOpen(false)
            void exportCurrentPresetToDataExports()
          }}
          onCancel={() => setExportDialogOpen(false)}
        />
      )}

      {showImportPathPicker && (
        <PathPicker
          dirOnly={false}
          onClose={() => setShowImportPathPicker(false)}
          onPick={(path) => { void handleImportFromPath(path) }}
        />
      )}

      {conflict && (
        <ImportConflictDialog
          suggestedName={conflict.suggestedName}
          existingNames={presets.map((p) => p.name)}
          onDecide={resolveConflict}
        />
      )}
    </div>
  )
}

function PresetExportDialog({
  onDownload,
  onDataExports,
  onCancel,
}: {
  onDownload: () => void
  onDataExports: () => void
  onCancel: () => void
}) {
  const { t } = useTranslation()
  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-[60] flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel() }}
    >
      <div className="bg-elevated border border-subtle rounded-2xl w-[90%] max-w-[420px] p-6 flex flex-col gap-4 shadow-xl">
        <div>
          <h2 className="m-0 text-lg font-semibold text-fg-primary">{t('presets.exportPresetTitle')}</h2>
          <p className="mt-1 mb-0 text-sm text-fg-secondary">{t('presets.exportPresetHint')}</p>
        </div>
        <button type="button" className="card p-4 text-left hover:border-dim" onClick={onDownload}>
          <div className="font-medium text-fg-primary mb-1">{t('presets.exportDownload')}</div>
          <div className="text-xs text-fg-tertiary">{t('presets.exportDownloadHint')}</div>
        </button>
        <button type="button" className="card p-4 text-left hover:border-dim" onClick={onDataExports}>
          <div className="font-medium text-fg-primary mb-1">{t('presets.exportDataExports')}</div>
          <div className="text-xs text-fg-tertiary">{t('presets.exportDataExportsHint')}</div>
        </button>
        <div className="flex justify-end">
          <button type="button" className="btn btn-secondary" onClick={onCancel}>{t('common.cancel')}</button>
        </div>
      </div>
    </div>
  )
}

// ImportConflictDialog —— 上传 preset 名字撞库时弹三选一（覆盖 / 另存为 / 取消）。
function ImportConflictDialog({
  suggestedName,
  existingNames,
  onDecide,
}: {
  suggestedName: string
  existingNames: string[]
  onDecide: (c: ConflictChoice) => void
}) {
  const { t } = useTranslation()
  const [newName, setNewName] = useState(() => {
    let i = 2
    let cand = `${suggestedName}-${i}`
    while (existingNames.includes(cand)) cand = `${suggestedName}-${++i}`
    return cand
  })
  const [error, setError] = useState('')
  const inputRef = useRef<HTMLInputElement | null>(null)
  useEffect(() => {
    requestAnimationFrame(() => { inputRef.current?.focus(); inputRef.current?.select() })
  }, [])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onDecide({ kind: 'cancel' }) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onDecide])

  const submitSaveAs = (e?: React.FormEvent) => {
    e?.preventDefault()
    const v = newName.trim()
    if (!v) { setError(t('presets.nameRequired')); return }
    if (!PRESET_NAME_RE.test(v)) { setError(t('presets.nameInvalid')); return }
    if (existingNames.includes(v)) { setError(t('presets.nameExists')); return }
    onDecide({ kind: 'saveAs', name: v })
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-[60] flex items-center justify-center bg-zinc-950/40 backdrop-blur-[2px]"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onDecide({ kind: 'cancel' }) }}
    >
      <form
        onSubmit={submitSaveAs}
        className="bg-elevated border border-subtle rounded-2xl w-[90%] max-w-[480px] p-6 flex flex-col gap-4 shadow-xl"
      >
        <h2 className="m-0 text-lg font-semibold text-fg-primary">
          {t('presets.importConflictTitle', { name: suggestedName })}
        </h2>
        <p className="m-0 text-sm text-fg-secondary">
          {t('presets.importConflictBody')}
        </p>
        <label className="flex flex-col gap-1.5">
          <span className="text-sm text-fg-secondary">{t('presets.importSaveAsLabel')}</span>
          <input
            ref={inputRef}
            className="input input-mono font-mono"
            value={newName}
            onChange={(e) => { setNewName(e.target.value); if (error) setError('') }}
          />
          {error && <span className="text-xs text-err">{error}</span>}
        </label>
        <div className="flex gap-2 justify-end mt-1">
          <button type="button" onClick={() => onDecide({ kind: 'cancel' })} className="btn btn-secondary">
            {t('common.cancel')}
          </button>
          <button
            type="button"
            onClick={() => onDecide({ kind: 'overwrite' })}
            className="btn btn-warn"
            title={t('presets.importOverwriteTitle', { name: suggestedName })}
          >
            {t('presets.importOverwrite')}
          </button>
          <button type="submit" className="btn btn-primary">
            {t('presets.importSaveAs')}
          </button>
        </div>
      </form>
    </div>
  )
}
