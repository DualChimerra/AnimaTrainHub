import type { TFunction } from 'i18next'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type Job,
  type ProjectDetail,
  type RegAiRequest,
  type RegBuildRequest,
  type RegStatus,
  type RegTagCount,
  type Task,
  type Version,
} from '../../../api/client'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import ImagePreviewModal from '../../../components/ImagePreviewModal'
import StepShell from '../../../components/StepShell'
import FieldLabel from '../../../components/ds/FieldLabel'
import KebabMenu from '../../../components/ds/KebabMenu'
import { TranslatedTag } from '../../../components/tagDisplay/TranslatedTag'
import { TagSuggestList } from '../../../components/tagSuggest/TagSuggestList'
import { useTagSuggest } from '../../../components/tagSuggest/useTagSuggest'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

interface AdvancedParams {
  skip_similar: boolean
  aspect_ratio_filter_enabled: boolean
  min_aspect_ratio: number
  max_aspect_ratio: number
  postprocess_method: 'smart' | 'stretch' | 'crop'
  postprocess_max_crop_ratio: number
}

// batch_size 不暴露 — 多 train 子文件夹（5_concept / 1_general 等）共用同一 batch
// 概念在 UI 上意义不大，保持源脚本默认 5。
const ADVANCED_DEFAULTS: AdvancedParams = {
  skip_similar: true,
  aspect_ratio_filter_enabled: false,
  min_aspect_ratio: 0.5,
  max_aspect_ratio: 2.0,
  postprocess_method: 'smart',
  postprocess_max_crop_ratio: 0.1,
}

export default function RegularizationPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [reg, setReg] = useState<RegStatus | null>(null)
  const [trainTags, setTrainTags] = useState<RegTagCount[]>([])
  // excluded 既包含 train top-tag 上点掉的，也包含「自定义排除」输入框加的（这部分
  // 在 train top-tag 列表里查不到）。后端不存这份选择，切页面回来需要按
  // (project, version) 在 localStorage 恢复，不然用户加的自定义 tag 看着就丢了。
  const [excluded, setExcluded] = useState<Set<string>>(new Set())
  const [autoTag, setAutoTag] = useState(true)
  // A3 — reg 自动打标的 tagger 选择。UI 暴露 wd14 / cltagger；后端 422 校验同。
  const [autoTagKind, setAutoTagKind] = useState<'wd14' | 'cltagger'>('wd14')
  // Auto dedup after the build, on by default. Whether a run keeps the
  // existing images is picked per run: "Top up" (incremental) or "Start" (full).
  const [autoDedup, setAutoDedup] = useState(true)
  // B1（PR-2）— 构建模式 + 目标数（仅 flat 模式生效）。默认 flat，target 留空 = train 总数。
  const [buildMode, setBuildMode] = useState<'mirror' | 'flat'>('flat')
  const [targetCount, setTargetCount] = useState<string>('')  // input value (string for blank → null)
  const [apiSource, setApiSource] = useState<'gelbooru' | 'danbooru'>('gelbooru')
  const [advanced, setAdvanced] = useState<AdvancedParams>(ADVANCED_DEFAULTS)

  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = job?.id ?? null

  // B2（PR-2）：「设置 & 日志」+「先验生成」合并成单 tab「生成」；顶部 source picker
  // 决定渲染 Booru 配置面板还是 AI 配置面板。「开始生成」按钮按 source 调对应 endpoint。
  const [activeTab, setActiveTab] = useState<'generate' | 'images'>('generate')
  // 来源默认 AI 先验（#8 决策 2026-05-30）：对齐 DreamBooth 原论文 neutral prior。
  // Booru 路径保留作"省时间"备选（不烧 GPU、更快出图）。
  const [source, setSource] = useState<'booru' | 'ai'>('ai')

  // 先验生成 — base 模型对每张 train 图反向出对照图，无 LoRA 参数（DreamBooth prior preservation）。
  // excluded tag 复用主组件 `excluded` Set，与 booru tab 双向同步。
  const [aiNeg, setAiNeg] = useState(
    'worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, bad anatomy, bad hands, bad feet'
  )
  const [aiWidth, setAiWidth] = useState(1024)
  const [aiHeight, setAiHeight] = useState(1024)
  const [aiSteps, setAiSteps] = useState(25)
  const [aiCfg, setAiCfg] = useState(4.0)
  const [aiSeed, setAiSeed] = useState(0)
  // reg 子文件夹 repeat 前缀（N_data）。reg 集独立于 train repeat，默认 1
  // （DreamBooth 标准：reg 每张每 epoch 见 1 次）。见 anima-phase-cursor-sse-desync 周边讨论。
  const [aiRepeat, setAiRepeat] = useState(1)
  const [aiTask, setAiTask] = useState<Task | null>(null)
  const [aiLogs, setAiLogs] = useState<string[]>([])
  const [aiBusy, setAiBusy] = useState(false)
  const aiTaskIdRef = useRef<number | null>(null)
  aiTaskIdRef.current = aiTask?.id ?? null

  // 预览 modal
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const [previewCaption, setPreviewCaption] = useState<string>('')

  const vid = activeVersion?.id ?? null

  const refreshReg = useCallback(async () => {
    if (!vid) return
    try {
      const s = await api.getRegStatus(project.id, vid)
      setReg(s)
    } catch (e) {
      toast(t('reg.loadFailed', { error: String(e) }), 'error')
    }
  }, [project.id, vid, t, toast])

  const refreshTrainTags = useCallback(async () => {
    if (!vid) return
    try {
      const items = await api.previewRegTags(project.id, vid, 30)
      setTrainTags(items)
    } catch {
      setTrainTags([])
    }
  }, [project.id, vid])

  useEffect(() => {
    void refreshReg()
    void refreshTrainTags()
  }, [refreshReg, refreshTrainTags])

  // 把 excluded 持久化到 localStorage（按 project + version 隔离），切页面回来也在。
  // 切 version / 进入页面时先 seed 一次；之后随 setExcluded 变化自动保存。
  const excludedStorageKey = vid
    ? `studio.reg.excluded.${project.id}.${vid}`
    : null
  useEffect(() => {
    if (!excludedStorageKey) return
    try {
      const raw = localStorage.getItem(excludedStorageKey)
      if (!raw) {
        setExcluded(new Set())
        return
      }
      const arr = JSON.parse(raw)
      if (Array.isArray(arr)) {
        setExcluded(new Set(arr.filter((x): x is string => typeof x === 'string')))
      } else {
        setExcluded(new Set())
      }
    } catch {
      setExcluded(new Set())
    }
  }, [excludedStorageKey])
  useEffect(() => {
    if (!excludedStorageKey) return
    try {
      localStorage.setItem(excludedStorageKey, JSON.stringify(Array.from(excluded)))
    } catch { /* quota / privacy mode：丢就丢，不打扰用户 */ }
  }, [excludedStorageKey, excluded])

  // 刷新 / 进入页面时回放最近一次 reg_build job：锁回 jid + 回放历史日志
  useEffect(() => {
    if (!vid) return
    void api
      .getLatestVersionJob(project.id, vid, 'reg_build')
      .then((r) => {
        if (!r.job) return
        setJob(r.job)
        setLogs(r.log ? r.log.split('\n') : [])
      })
      .catch(() => {})
  }, [project.id, vid])

  useEventStream((evt) => {
    const jid = jobIdRef.current
    const tid = aiTaskIdRef.current
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      void api.getJob(jid).then(setJob).catch(() => {})
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        void refreshReg()
        void reload()
        if (evt.status === 'done') setActiveTab('images')
      }
    } else if (evt.type === 'task_log_appended' && tid && evt.task_id === tid) {
      setAiLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'task_state_changed' && tid && evt.task_id === tid) {
      void api.getRegPriorTask(project.id, vid!, tid).then((t) => {
        setAiTask(t)
        if (t.status === 'done' || t.status === 'failed' || t.status === 'canceled') {
          setAiBusy(false)
          void refreshReg()
          if (t.status === 'done') setActiveTab('images')
        }
      }).catch(() => {})
    }
  })

  // SSE 不可靠兜底（Colab 代理常断 SSE → 见 anima-phase-cursor-sse-desync）：
  // reg 生成跑着时定时轮询后端状态，不只靠 task_state_changed / job_state_changed。
  // 没有它，SSE 一死 badge 永远停在 "Queued #N"、aiBusy 永远 true（生成按钮锁死），
  // 而 worker 其实正常出图（reg/ 里картинки 在涨）。状态非终态时每 3s 拉一次。
  useEffect(() => {
    const taskLive = aiTask?.status === 'pending' || aiTask?.status === 'running'
    const jobLive = job?.status === 'pending' || job?.status === 'running'
    if ((!taskLive && !jobLive) || !vid) return
    const timer = setInterval(() => {
      const tid = aiTaskIdRef.current
      if (taskLive && tid) {
        void api.getRegPriorTask(project.id, vid, tid).then((t) => {
          setAiTask(t)
          if (t.status === 'done' || t.status === 'failed' || t.status === 'canceled') {
            setAiBusy(false)
            void refreshReg()
            if (t.status === 'done') setActiveTab('images')
          }
        }).catch(() => {})
        // 日志同样不能只靠 SSE task_log_appended（Colab 上 SSE 死 → 日志永远空、
        // 要手动刷新页面才见更新）。拉全量 run.log 整体替换（全量 ⊇ 增量，自愈）。
        void api.getLog(tid).then((r) => {
          setAiLogs(r.content ? r.content.split('\n') : [])
        }).catch(() => {})
      }
      const jid = jobIdRef.current
      if (jobLive && jid) {
        void api.getJob(jid).then((j) => {
          setJob(j)
          if (j.status === 'done' || j.status === 'failed' || j.status === 'canceled') {
            void refreshReg()
            void reload()
            if (j.status === 'done') setActiveTab('images')
          }
        }).catch(() => {})
        void api.getLog(jid).then((r) => {
          setLogs(r.content ? r.content.split('\n') : [])
        }).catch(() => {})
      }
    }, 3000)
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aiTask?.status, job?.status, vid])

  const trainImageCount = activeVersion?.stats?.train_image_count ?? 0
  // 任意一种生成跑着都视为 live —— 防止 booru / AI 并发同时写 reg/。
  const isLive = job?.status === 'running' || job?.status === 'pending' || aiBusy

  // B1（PR-2）— 现有 reg 集结构推断：meta.build_mode 优先（新 meta 写入），
  // 否则看 reg.files 路径前缀（仅 1_data/ → flat；含 N_xxx 多种 → mirror）。
  // 空集 → null（mode 可自由切换）。
  const existingMode = useMemo<'mirror' | 'flat' | null>(() => {
    if (!reg || !reg.exists || reg.image_count === 0) return null
    if (reg.meta?.build_mode === 'mirror' || reg.meta?.build_mode === 'flat') {
      return reg.meta.build_mode
    }
    const prefixes = new Set<string>()
    for (const rel of reg.files) {
      const idx = rel.indexOf('/')
      prefixes.add(idx >= 0 ? rel.slice(0, idx) : '')
    }
    if (prefixes.size === 1 && prefixes.has('1_data')) return 'flat'
    return 'mirror'
  }, [reg])
  // mode 跟现有结构不一致时禁用切换（incremental 沿用会撞结构；用户必须先清空）
  const modeLocked = existingMode !== null && existingMode !== buildMode

  // 现有 reg 集存在时，把 buildMode 自动对齐它（避免切到 version 看到错的初始值）。
  // 用户点 disabled 的下拉看到 tooltip 提示「先清空」。
  useEffect(() => {
    if (existingMode && existingMode !== buildMode) setBuildMode(existingMode)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [existingMode])

  const toggleTag = (tag: string) => {
    setExcluded((prev) => {
      const next = new Set(prev)
      if (next.has(tag)) next.delete(tag)
      else next.add(tag)
      return next
    })
  }

  const handleAiGenerate = async (incremental: boolean) => {
    if (!vid) return
    if (trainImageCount <= 0) {
      toast(t('reg.noTrainForAi'), 'error')
      return
    }
    setAiBusy(true)
    setAiTask(null)
    setAiLogs([])
    try {
      const body: RegAiRequest = {
        excluded_tags: Array.from(excluded),
        negative_prompt: aiNeg,
        width: aiWidth,
        height: aiHeight,
        steps: aiSteps,
        cfg_scale: aiCfg,
        seed: aiSeed,
        incremental,
        repeat: aiRepeat,
      }
      const task = await api.enqueueRegPrior(project.id, vid, body)
      setAiTask(task)
      toast(t('reg.aiEnqueued', { id: task.id }), 'success')
    } catch (e) {
      toast(String(e), 'error')
      setAiBusy(false)
    }
  }

  const startBuild = async (incremental: boolean) => {
    if (!vid) return
    if (trainImageCount <= 0) {
      toast(t('reg.noTrainForBuild'), 'error')
      return
    }
    const parsedTarget = targetCount.trim() === '' ? null : Number(targetCount)
    const body: RegBuildRequest = {
      excluded_tags: Array.from(excluded),
      auto_tag: autoTag,
      auto_tag_kind: autoTagKind,
      api_source: apiSource,
      incremental,
      auto_dedup: autoDedup,
      build_mode: buildMode,
      target_count: parsedTarget,
      ...advanced,
    }
    try {
      const j = await api.startRegBuild(project.id, vid, body)
      setJob(j)
      setLogs([])
      toast(t('reg.enqueued', { id: j.id }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const onDelete = async () => {
    if (!vid) return
    if (!(await confirm(t('reg.confirmDelete'), { tone: 'danger', okText: t('reg.deleteOkText') }))) return
    try {
      await api.deleteReg(project.id, vid)
      toast(t('reg.deleted'), 'success')
      setReg(null)
      void refreshReg()
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // 预览：点击缩略图 → 加载该图 caption → 打开 modal
  const openPreview = useCallback(
    async (idx: number) => {
      if (!reg || !vid) return
      const path = reg.files[idx]
      setPreviewIdx(idx)
      setPreviewCaption(t('reg.captionLoading'))
      try {
        const r = await api.getRegCaption(project.id, vid, path)
        setPreviewCaption(r.tags.length ? r.tags.join(', ') : t('reg.captionEmpty'))
      } catch (e) {
        setPreviewCaption(t('reg.captionFailed', { error: String(e) }))
      }
    },
    [reg, vid, project.id, t]
  )

  if (!activeVersion || !vid) {
    return <p className="text-fg-tertiary p-6">{t('reg.noVersion')}</p>
  }

  const trainUrl = `/projects/${project.id}/v/${vid}/train`
  const regCount = reg?.image_count ?? 0
  // how many images a run aims for: the train total unless a flat booru set
  // was given its own target
  const runTarget = source === 'booru' && buildMode === 'flat' && targetCount.trim() !== ''
    ? Math.max(1, Number(targetCount) || trainImageCount)
    : trainImageCount

  // "Top up" keeps what is there; "Start" rebuilds, so ask first when a set exists
  const run = async (incremental: boolean) => {
    if (!incremental && reg?.exists && regCount > 0) {
      const ok = await confirm(t('reg.confirmFullRun', { n: regCount }), { tone: 'danger', okText: t('reg.startBuildBtn') })
      if (!ok) return
    }
    if (source === 'ai') await handleAiGenerate(incremental)
    else await startBuild(incremental)
  }

  const cancelJob = async () => {
    if (!job) return
    try {
      await api.cancelJob(job.id)
      toast(t('reg.cancelToast'), 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const tabsCard = (
    <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minWidth: 0, minHeight: 0, flex: activeTab === 'images' ? 1 : undefined }}>
      <div className="ds-tabs" role="tablist">
        <button type="button" role="tab" aria-selected={activeTab === 'generate'} className={`ds-tab${activeTab === 'generate' ? ' ds-is-active' : ''}`} onClick={() => setActiveTab('generate')}>
          {t('reg.tabGenerate')}
          {isLive && <span className="ds-badge ds-warn">live</span>}
        </button>
        <button type="button" role="tab" aria-selected={activeTab === 'images'} className={`ds-tab${activeTab === 'images' ? ' ds-is-active' : ''}`} onClick={() => setActiveTab('images')}>
          {t('reg.tabImages')}
          {regCount > 0 && <span className="ds-badge ds-mute">{regCount}</span>}
        </button>
      </div>

      {activeTab === 'generate' ? (
        <>
          <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
            <div style={{ padding: '14px 17px 15px', borderBottom: '1px solid var(--line)' }}>
              <div className="ds-cap" style={{ marginBottom: 10 }}>{t('reg.sourceLabel')}</div>
              <div className="ds-optcards" role="radiogroup" aria-label={t('reg.sourceLabel')}>
                <SourceCard
                  on={source === 'booru'}
                  onClick={() => setSource('booru')}
                  icon={Icon.globe}
                  name={t('reg.sourceBooru')}
                  badge={t('reg.sourceBooruSub')}
                  desc={t('reg.sourceBooruDesc')}
                />
                <SourceCard
                  on={source === 'ai'}
                  onClick={() => setSource('ai')}
                  icon={Icon.spark}
                  name={t('reg.sourceAi')}
                  badge="GPU"
                  desc={t('reg.sourceAiDesc')}
                />
              </div>
            </div>

            {source === 'ai' ? (
              <AiForm
                neg={aiNeg} onNegChange={setAiNeg}
                width={aiWidth} onWidthChange={setAiWidth}
                height={aiHeight} onHeightChange={setAiHeight}
                steps={aiSteps} onStepsChange={setAiSteps}
                cfg={aiCfg} onCfgChange={setAiCfg}
                seed={aiSeed} onSeedChange={setAiSeed}
                repeat={aiRepeat} onRepeatChange={setAiRepeat}
              />
            ) : (
              <BooruForm
                trainImageCount={trainImageCount}
                apiSource={apiSource} onApiSourceChange={setApiSource}
                buildMode={buildMode} onBuildModeChange={setBuildMode}
                modeLocked={modeLocked}
                existingMode={existingMode}
                targetCount={targetCount} onTargetCountChange={setTargetCount}
                autoTag={autoTag} onAutoTagChange={setAutoTag}
                autoTagKind={autoTagKind} onAutoTagKindChange={setAutoTagKind}
                autoDedup={autoDedup} onAutoDedupChange={setAutoDedup}
                advanced={advanced} onAdvancedChange={setAdvanced}
              />
            )}

            <ExcludeTags
              trainTags={trainTags}
              excluded={excluded}
              onToggle={toggleTag}
            />
          </div>

          <div className="ds-cardfoot">
            <span className="ds-hint">
              {trainImageCount <= 0 ? t('reg.noTrainForBuild') : t('reg.collected', { n: regCount, target: runTarget })}
            </span>
            <span className="ds-cardfoot-act">
              <button
                type="button"
                className="ds-ctl"
                style={{ height: 34 }}
                onClick={() => void run(true)}
                disabled={isLive || trainImageCount <= 0 || regCount >= runTarget}
                title={source === 'ai' ? t('reg.modeIncrementalAi') : t('reg.modeIncrementalBooru')}
              >{t('reg.topUpTo', { n: runTarget })}</button>
              <button
                type="button"
                className="ds-btn-primary"
                style={{ height: 34 }}
                onClick={() => void run(false)}
                disabled={isLive || trainImageCount <= 0 || (source === 'booru' && modeLocked)}
                title={source === 'ai' ? t('reg.modeFullAi') : t('reg.modeFullBooru')}
              >{isLive ? t('reg.generatingBtn') : t('reg.startBuildBtn')}</button>
            </span>
          </div>
        </>
      ) : reg && regCount > 0 ? (
        <RegImages
          pid={project.id}
          vid={vid}
          reg={reg}
          isLive={isLive}
          runTarget={runTarget}
          onTopUp={() => void run(true)}
          onPreview={(idx) => void openPreview(idx)}
          onChanged={() => {
            void refreshReg()
            void reload()
          }}
        />
      ) : (
        <div className="ds-empty" style={{ margin: 17 }}>
          <span style={{ fontWeight: 500, color: 'var(--ink-2)' }}>{t('reg.emptyRegTitle')}</span>
          <span style={{ fontSize: 11.5 }}>{t('reg.emptyRegHint')}</span>
        </div>
      )}
    </div>
  )

  return (
    <StepShell
      idx={5}
      eyebrow={t('reg.eyebrow')}
      title={t('steps.reg.title')}
      subtitle={<Trans i18nKey="steps.reg.subtitle" components={{ code: <code /> }} />}
      actions={
        <>
          {!reg?.exists && <Link className="ds-ctl" to={trainUrl}>{t('reg.skipStep')}</Link>}
          <KebabMenu
            label={t('reg.moreActions')}
            trigger="icon"
            items={[{ label: t('reg.deleteBtn'), tone: 'err', onSelect: () => void onDelete(), disabled: isLive || !reg?.exists }]}
          />
          <Link className="ds-btn-primary" to={trainUrl}>
            {t('reg.next')}
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h13m-5-6 6 6-6 6" /></svg>
          </Link>
        </>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, flex: 1, minHeight: 0, overflowY: 'auto', paddingTop: 2 }}>
        <StatusCard reg={reg} existingMode={existingMode} job={job} aiTask={aiTask} />

        {activeTab === 'generate' ? (
          <div className="ds-formsplit ds-even ds-fill" style={{ flex: 1, minHeight: 0 }}>
            {tabsCard}
            <RunLog
              source={source}
              job={job}
              jobLogs={logs}
              aiTask={aiTask}
              aiLogs={aiLogs}
              onCancelJob={cancelJob}
            />
          </div>
        ) : tabsCard}
      </div>

      {previewIdx !== null && reg && reg.files[previewIdx] && (
        <ImagePreviewModal
          src={regOrigUrl(project.id, vid, reg.files[previewIdx])}
          caption={previewCaption}
          hasPrev={previewIdx > 0}
          hasNext={previewIdx < reg.files.length - 1}
          onClose={() => setPreviewIdx(null)}
          onPrev={() =>
            previewIdx > 0 ? void openPreview(previewIdx - 1) : undefined
          }
          onNext={() =>
            previewIdx < reg.files.length - 1
              ? void openPreview(previewIdx + 1)
              : undefined
          }
        />
      )}
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// Pieces of the page (mockup Reg / RegImages)
// ---------------------------------------------------------------------------

const Icon = {
  globe: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" /></svg>
  ),
  spark: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6" /></svg>
  ),
  x: (
    <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round"><path d="M6 6l12 12M18 6 6 18" /></svg>
  ),
  minus: (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M5 12h14" /></svg>
  ),
  plus: (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>
  ),
}

/** Finished-state label shared by the booru job and the prior task. */
function runStatusLabel(status: string, t: TFunction): string {
  if (status === 'done') return t('reg.aiStatusDone')
  if (status === 'running') return t('reg.aiStatusRunning')
  if (status === 'failed') return t('reg.aiStatusFailed')
  if (status === 'pending') return t('reg.aiStatusPending')
  if (status === 'canceled') return t('reg.aiStatusCanceled')
  return status
}

function runBadgeTone(status: string): string {
  if (status === 'done') return 'ds-ok'
  if (status === 'failed') return 'ds-err'
  if (status === 'running' || status === 'pending') return 'ds-info'
  return 'ds-mute'
}

// Four numbers about the current set: size + structure, source + tagger,
// tags the booru search could not use, and when it was last built.
function StatusCard({
  reg, existingMode, job, aiTask,
}: {
  reg: RegStatus | null
  existingMode: 'mirror' | 'flat' | null
  job: Job | null
  aiTask: Task | null
}) {
  const { t, i18n } = useTranslation()
  const m = reg?.exists ? reg.meta : null
  const folderCount = useMemo(() => {
    const s = new Set<string>()
    for (const rel of reg?.files ?? []) {
      const i = rel.indexOf('/')
      s.add(i >= 0 ? rel.slice(0, i) : '')
    }
    return s.size
  }, [reg])
  const failed = m?.failed_tags ?? []
  const latestRun = job ?? aiTask
  const cell = (last: boolean, cap: string, value: React.ReactNode, key: React.ReactNode, big = false, title?: string) => (
    <div style={{ padding: '14px 17px', borderRight: last ? undefined : '1px solid var(--line)' }} title={title}>
      <div className="ds-cap">{cap}</div>
      <div style={big
        ? { fontSize: 18, fontWeight: 600, letterSpacing: '-.03em', marginTop: 5 }
        : { fontSize: 14, fontWeight: 600, marginTop: 6 }}
      >{value}</div>
      <div className="ds-cell-key">{key}</div>
    </div>
  )
  if (!reg) {
    return <div className="ds-card" style={{ padding: '14px 17px', fontSize: 12, color: 'var(--ink-3)' }}>{t('reg.statusLoading')}</div>
  }
  const date = m
    ? new Intl.DateTimeFormat(i18n.language, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }).format(m.generated_at * 1000)
    : '—'
  return (
    <div className="ds-card">
      <div className="ds-reg-status">
        {cell(false, t('reg.statusCellSet'),
          <>{reg.exists ? reg.image_count : 0} <span style={{ fontSize: 12, fontWeight: 400, color: 'var(--ink-3)' }}>{t('reg.nImagesShort')}</span></>,
          !reg.exists ? t('reg.statusNotBuilt')
            : existingMode === 'flat' ? t('reg.structFlat')
              : existingMode === 'mirror' ? t('reg.structMirror', { count: folderCount })
                : '—',
          true)}
        {cell(false, t('reg.statusCellSourceTag'),
          m ? (m.generation_method === 'ai_base' ? t('reg.sourceAi') : t('reg.sourceBooru')) : '—',
          m
            ? [m.generation_method === 'ai_base' ? t('reg.sourceAiSub') : m.api_source,
              m.auto_tagged ? (m.auto_tag_kind ?? t('reg.taggerUnknown')) : t('reg.statusTaggerOff')].join(' · ')
            : '—')}
        {cell(false, t('reg.statusCellInvalidTags'),
          <span style={{ color: failed.length > 0 ? 'var(--amber-text)' : undefined }}>{m ? failed.length : '—'}</span>,
          m ? (failed.length > 0 ? t('reg.invalidKey') : t('reg.invalidNone')) : '—',
          true,
          failed.length > 0 ? t('reg.failedTagsTitle', { tags: failed.join(', ') }) : undefined)}
        {cell(true, t('reg.statusCellLatest'),
          date,
          latestRun ? t('reg.latestRun', { id: latestRun.id, status: runStatusLabel(latestRun.status, t) }) : (m ? formatAgo(m.generated_at, t) : '—'))}
      </div>
    </div>
  )
}

function SourceCard({ on, onClick, icon, name, badge, desc }: {
  on: boolean
  onClick: () => void
  icon: React.ReactNode
  name: string
  badge: string
  desc: string
}) {
  return (
    <button type="button" role="radio" aria-checked={on} className={`ds-optcard${on ? ' ds-is-on' : ''}`} onClick={onClick}>
      <span className="ds-optcard-ico">{icon}</span>
      <span className="ds-optcard-txt">
        <span className="ds-optcard-name">{name}<span className="ds-badge ds-mute">{badge}</span></span>
        <span className="ds-optcard-desc">{desc}</span>
      </span>
      <span className={`ds-radio${on ? ' ds-on' : ''}`} />
    </button>
  )
}

function SubGroup({ name, keyName, meta }: { name: string; keyName?: string; meta?: React.ReactNode }) {
  return (
    <div className="ds-subgroup">
      <span className="ds-sg-name">{name}</span>
      {keyName && <span className="ds-sg-key">{keyName}</span>}
      <span style={{ flex: 1 }} />
      {meta != null && <span className="ds-kpi-meta">{meta}</span>}
    </div>
  )
}

// One setting row: name + parameter key (explanation on hover), control on the right.
function Row({ label, keyName, desc, children }: {
  label: string
  keyName?: string
  desc?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="ds-field">
      <div className="ds-field-txt">
        <div className="ds-field-name"><FieldLabel label={label} tip={desc} />{keyName && <span className="ds-key">{keyName}</span>}</div>
      </div>
      <div className="ds-field-ctl">{children}</div>
    </div>
  )
}

function Stepper({ value, onChange, step = 1, min, max, placeholder, disabled, ariaLabel }: {
  value: string
  onChange: (v: string) => void
  step?: number
  min?: number
  max?: number
  placeholder?: string
  disabled?: boolean
  ariaLabel?: string
}) {
  const { t } = useTranslation()
  const bump = (d: number) => {
    const base = value.trim() === '' ? Number(placeholder) || 0 : Number(value) || 0
    let next = Math.round((base + d) * 1000) / 1000
    if (min != null) next = Math.max(min, next)
    if (max != null) next = Math.min(max, next)
    onChange(String(next))
  }
  return (
    <span className="ds-stepper" style={disabled ? { opacity: 0.55 } : undefined}>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        inputMode="decimal"
        aria-label={ariaLabel}
      />
      <button type="button" aria-label={t('reg.less')} onClick={() => bump(-step)} disabled={disabled}>{Icon.minus}</button>
      <button type="button" aria-label={t('reg.more')} onClick={() => bump(step)} disabled={disabled}>{Icon.plus}</button>
    </span>
  )
}

/** Numeric stepper bound to a number state; out-of-range input is clamped. */
function NumStepper({ value, onChange, step = 1, min, max, ariaLabel }: {
  value: number
  onChange: (v: number) => void
  step?: number
  min?: number
  max?: number
  ariaLabel?: string
}) {
  const [draft, setDraft] = useState(String(value))
  useEffect(() => { setDraft(String(value)) }, [value])
  return (
    <Stepper
      value={draft}
      step={step}
      min={min}
      max={max}
      ariaLabel={ariaLabel}
      onChange={(v) => {
        setDraft(v)
        const n = Number(v)
        if (v.trim() === '' || !Number.isFinite(n)) return
        let c = n
        if (min != null) c = Math.max(min, c)
        if (max != null) c = Math.min(max, c)
        onChange(c)
      }}
    />
  )
}

function Switch({ on, onChange, disabled, label }: { on: boolean; onChange: (v: boolean) => void; disabled?: boolean; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      className={`ds-switch${on ? ' ds-on' : ''}`}
      onClick={() => onChange(!on)}
      disabled={disabled}
      style={disabled ? { opacity: 0.45, cursor: 'not-allowed' } : undefined}
    ><i /></button>
  )
}

function Seg<T extends string>({ value, options, onChange, disabled, label }: {
  value: T
  options: Array<[T, string]>
  onChange: (v: T) => void
  disabled?: boolean
  label: string
}) {
  return (
    <div className="ds-seg" style={{ display: 'flex', width: '100%' }} role="group" aria-label={label}>
      {options.map(([id, text]) => (
        <button key={id} type="button" className={`ds-seg-item${value === id ? ' ds-is-active' : ''}`} style={{ flex: 1 }} aria-pressed={value === id} onClick={() => onChange(id)} disabled={disabled}>
          {text}
        </button>
      ))}
    </div>
  )
}

// AI prior settings: what to generate, then the sampler.
function AiForm({
  neg, onNegChange,
  width, onWidthChange,
  height, onHeightChange,
  steps, onStepsChange,
  cfg, onCfgChange,
  seed, onSeedChange,
  repeat, onRepeatChange,
}: {
  neg: string
  onNegChange: (v: string) => void
  width: number; onWidthChange: (v: number) => void
  height: number; onHeightChange: (v: number) => void
  steps: number; onStepsChange: (v: number) => void
  cfg: number; onCfgChange: (v: number) => void
  seed: number; onSeedChange: (v: number) => void
  repeat: number; onRepeatChange: (v: number) => void
}) {
  const { t } = useTranslation()
  return (
    <>
      <SubGroup name={t('reg.grpAiGen')} keyName="ai_prior" meta={t('reg.nFields', { count: 4 })} />
      <Row label={t('reg.negPrompt')} keyName="negative_prompt" desc={t('reg.negPromptDesc')}>
        <textarea
          className="ds-inp ds-mono"
          style={{ height: 84, padding: '8px 10px', resize: 'vertical', lineHeight: 1.5 }}
          value={neg}
          onChange={(e) => onNegChange(e.target.value)}
          aria-label={t('reg.negPrompt')}
        />
      </Row>
      <Row label={t('reg.widthLabel')} keyName="width" desc={t('reg.sizeDesc')}>
        <NumStepper value={width} onChange={onWidthChange} step={64} min={256} max={4096} ariaLabel={t('reg.widthLabel')} />
      </Row>
      <Row label={t('reg.heightLabel')} keyName="height">
        <NumStepper value={height} onChange={onHeightChange} step={64} min={256} max={4096} ariaLabel={t('reg.heightLabel')} />
      </Row>
      <Row label={t('reg.repeatLabel')} keyName="repeat" desc={t('reg.repeatHint')}>
        <NumStepper value={repeat} onChange={onRepeatChange} min={1} max={100} ariaLabel={t('reg.repeatLabel')} />
      </Row>

      <SubGroup name={t('reg.grpSampling')} keyName="sampler" meta={t('reg.nFields', { count: 3 })} />
      <Row label={t('reg.stepsLabel')} keyName="steps">
        <NumStepper value={steps} onChange={onStepsChange} min={1} max={150} ariaLabel={t('reg.stepsLabel')} />
      </Row>
      <Row label="CFG Scale" keyName="cfg_scale">
        <NumStepper value={cfg} onChange={onCfgChange} step={0.5} min={0} max={20} ariaLabel="CFG Scale" />
      </Row>
      <Row label={t('reg.seedLabel')} keyName="seed" desc={t('reg.seedHintRandom')}>
        <NumStepper value={seed} onChange={onSeedChange} min={0} ariaLabel={t('reg.seedLabel')} />
      </Row>
    </>
  )
}

// Booru scrape settings, then post-processing.
function BooruForm({
  trainImageCount,
  apiSource, onApiSourceChange,
  buildMode, onBuildModeChange, modeLocked, existingMode,
  targetCount, onTargetCountChange,
  autoTag, onAutoTagChange,
  autoTagKind, onAutoTagKindChange,
  autoDedup, onAutoDedupChange,
  advanced, onAdvancedChange,
}: {
  trainImageCount: number
  apiSource: 'gelbooru' | 'danbooru'
  onApiSourceChange: (v: 'gelbooru' | 'danbooru') => void
  buildMode: 'mirror' | 'flat'
  onBuildModeChange: (v: 'mirror' | 'flat') => void
  modeLocked: boolean
  existingMode: 'mirror' | 'flat' | null
  targetCount: string
  onTargetCountChange: (v: string) => void
  autoTag: boolean
  onAutoTagChange: (v: boolean) => void
  autoTagKind: 'wd14' | 'cltagger'
  onAutoTagKindChange: (v: 'wd14' | 'cltagger') => void
  autoDedup: boolean
  onAutoDedupChange: (v: boolean) => void
  advanced: AdvancedParams
  onAdvancedChange: (v: AdvancedParams) => void
}) {
  const { t } = useTranslation()
  const mirror = buildMode === 'mirror'
  const set = <K extends keyof AdvancedParams>(k: K, v: AdvancedParams[K]) =>
    onAdvancedChange({ ...advanced, [k]: v })
  return (
    <>
      <SubGroup name={t('reg.grpBooruScrape')} keyName="booru" meta={t('reg.nFields', { count: 6 })} />
      <Row label={t('reg.source')} keyName="api_source">
        <Seg value={apiSource} options={[['gelbooru', 'Gelbooru'], ['danbooru', 'Danbooru']]} onChange={onApiSourceChange} label={t('reg.source')} />
      </Row>
      <Row label={t('reg.buildModeLabel')} keyName="build_mode" desc={t('reg.buildModeTitle')}>
        <Seg
          value={buildMode}
          options={[['flat', t('reg.buildModeFlatShort')], ['mirror', t('reg.buildModeMirrorShort')]]}
          onChange={onBuildModeChange}
          disabled={modeLocked}
          label={t('reg.buildModeLabel')}
        />
        {modeLocked && <span className="ds-ctl-note">{t('reg.buildModeLocked', { mode: existingMode })}</span>}
      </Row>
      <Row label={t('reg.targetCount')} keyName="target_count" desc={t('reg.targetCountDesc')}>
        <Stepper
          value={mirror ? String(trainImageCount) : targetCount}
          onChange={onTargetCountChange}
          placeholder={String(trainImageCount)}
          min={1}
          disabled={mirror}
          ariaLabel={t('reg.targetCount')}
        />
        <span className="ds-ctl-note">{mirror ? t('reg.targetMirrorLocked', { n: trainImageCount }) : t('reg.targetCountHint')}</span>
      </Row>
      <Row label={t('reg.autoTagLabel')} keyName="auto_tag" desc={t('reg.autoTagDesc')}>
        <Switch on={autoTag} onChange={onAutoTagChange} label={t('reg.autoTagLabel')} />
      </Row>
      <Row label={t('reg.autoTagKindLabel')} keyName="auto_tag_kind">
        <Seg value={autoTagKind} options={[['wd14', 'WD14'], ['cltagger', 'CLTagger']]} onChange={onAutoTagKindChange} disabled={!autoTag} label={t('reg.autoTagKindLabel')} />
        {!autoTag && <span className="ds-ctl-note">{t('reg.autoTagKindDisabled')}</span>}
      </Row>
      <Row label={t('reg.autoDedupLabel')} keyName="auto_dedup" desc={t('reg.autoDedupTitle')}>
        <Switch on={autoDedup} onChange={onAutoDedupChange} label={t('reg.autoDedupLabel')} />
      </Row>

      <SubGroup name={t('reg.grpAdvanced')} keyName="postprocess" meta={t('reg.nFields', { count: 4 })} />
      <Row label={t('reg.aspectFilter')} keyName="aspect_ratio_filter" desc={t('reg.aspectFilterHint')}>
        <Switch on={advanced.aspect_ratio_filter_enabled} onChange={(v) => set('aspect_ratio_filter_enabled', v)} label={t('reg.aspectFilter')} />
        {advanced.aspect_ratio_filter_enabled && (
          <div style={{ display: 'flex', gap: 6, width: '100%' }}>
            <NumStepper value={advanced.min_aspect_ratio} onChange={(v) => set('min_aspect_ratio', v)} step={0.05} min={0.1} max={1} ariaLabel={t('reg.aspectMin')} />
            <NumStepper value={advanced.max_aspect_ratio} onChange={(v) => set('max_aspect_ratio', v)} step={0.1} min={1} max={10} ariaLabel={t('reg.aspectMax')} />
          </div>
        )}
      </Row>
      <Row label={t('reg.postprocess')} keyName="postprocess_method">
        <select
          className="ds-inp"
          value={advanced.postprocess_method}
          onChange={(e) => set('postprocess_method', e.target.value as 'smart' | 'stretch' | 'crop')}
          aria-label={t('reg.postprocess')}
        >
          <option value="smart">{t('reg.postprocessSmart')}</option>
          <option value="stretch">{t('reg.postprocessStretch')}</option>
          <option value="crop">{t('reg.postprocessCrop')}</option>
        </select>
      </Row>
      <Row label={t('reg.maxCropLabel')} keyName="postprocess_max_crop_ratio" desc={t('reg.maxCropTitle')}>
        <NumStepper value={advanced.postprocess_max_crop_ratio} onChange={(v) => set('postprocess_max_crop_ratio', v)} step={0.05} min={0.05} max={0.5} ariaLabel={t('reg.maxCropLabel')} />
      </Row>
      <Row label={t('reg.skipSimilarLabel')} keyName="skip_similar" desc={t('reg.skipSimilarTitle')}>
        <Switch on={advanced.skip_similar} onChange={(v) => set('skip_similar', v)} label={t('reg.skipSimilarLabel')} />
      </Row>
    </>
  )
}

// Excluded tags: the chips that are out (train-derived plain, own ones amber),
// a "+ tag" field, then the frequent train tags to click out.
export function ExcludeTags({
  trainTags, excluded, onToggle,
}: {
  trainTags: RegTagCount[]
  excluded: Set<string>
  onToggle: (tag: string) => void
}) {
  const { t } = useTranslation()
  const [draft, setDraft] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const suggest = useTagSuggest({
    value: draft,
    inputRef,
    wholeAsToken: true,
    // 选中候选时把整段 draft 替换；用户再按 Enter 走 addCustom 走 normalize → 落 booru 形态
    onPick: ({ suggestion }) => { setDraft(suggestion.tag) },
  })
  const trainTagSet = useMemo(
    () => new Set(trainTags.map((x) => x.tag)),
    [trainTags]
  )
  const excludedList = useMemo(
    () => Array.from(excluded).sort((a, b) => Number(!trainTagSet.has(a)) - Number(!trainTagSet.has(b)) || a.localeCompare(b)),
    [excluded, trainTagSet]
  )
  const candidates = trainTags.filter((x) => !excluded.has(x.tag))
  const normalize = (raw: string): string =>
    raw.trim().toLowerCase().replace(/\s+/g, '_')
  const addCustom = () => {
    const items = draft
      .split(/[,，\n]+/)
      .map(normalize)
      .filter(Boolean)
    if (items.length === 0) return
    for (const tag of items) {
      if (!excluded.has(tag)) onToggle(tag)
    }
    setDraft('')
  }

  return (
    <>
      <SubGroup name={t('reg.excludeTitle')} keyName="excluded_tags" meta={t('reg.excludedN', { count: excluded.size })} />
      <div style={{ padding: '12px 17px 14px', display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {excludedList.map((tag) => {
            const custom = !trainTagSet.has(tag)
            return (
              <span
                key={tag}
                className="ds-chip max-w-full overflow-hidden whitespace-nowrap"
                style={custom ? { background: 'var(--amber-soft)', color: 'var(--amber-text)' } : undefined}
                title={custom ? t('reg.excludeCustomRemoveTitle') : undefined}
              >
                <span className="min-w-0 truncate text-left" title={tag.replace(/_/g, ' ')}>
                  <TranslatedTag tag={tag.replace(/_/g, ' ')} />
                </span>
                <button type="button" className="ds-chip-x" onClick={() => onToggle(tag)} aria-label={t('reg.excludeCustomRemoveAria', { tag })}>
                  {Icon.x}
                </button>
              </span>
            )
          })}
          <span style={{ position: 'relative', display: 'inline-flex' }}>
            <input
              ref={inputRef}
              className="ds-chip-add ds-mono"
              style={{ width: draft ? Math.max(110, draft.length * 7.2 + 26) : 80, outline: 'none', background: 'transparent', color: 'var(--ink)' }}
              value={draft}
              onChange={(e) => { setDraft(e.target.value); suggest.notifyChange() }}
              onKeyDown={(e) => {
                if (suggest.handleKeyDown(e)) return
                if (e.key === 'Enter') {
                  e.preventDefault()
                  addCustom()
                }
              }}
              onFocus={() => suggest.notifyFocus()}
              onBlur={() => suggest.notifyBlur()}
              placeholder={t('reg.excludePlaceholder')}
              aria-label={t('reg.excludePlaceholder')}
            />
            <TagSuggestList
              open={suggest.open}
              suggestions={suggest.suggestions}
              activeIdx={suggest.activeIdx}
              onPick={(s) => suggest.pickAt(suggest.suggestions.indexOf(s))}
              onHover={suggest.setActiveIdx}
              inputRef={inputRef}
              cursor={suggest.cursor}
              positionDeps={[draft]}
            />
          </span>
        </div>

        {candidates.length > 0 ? (
          <div>
            <div className="ds-cap" style={{ marginBottom: 8 }}>{t('reg.excludeTrainTitle')}</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {candidates.map((info) => (
                <button
                  key={info.tag}
                  type="button"
                  onClick={() => onToggle(info.tag)}
                  className="ds-chip max-w-full overflow-hidden whitespace-nowrap"
                  style={{ background: 'transparent', boxShadow: 'inset 0 0 0 1px var(--line-2)', paddingRight: 9 }}
                  title={t('reg.excludeClick')}
                >
                  <span className="min-w-0 truncate text-left" title={info.tag.replace(/_/g, ' ')}>
                    <TranslatedTag tag={info.tag.replace(/_/g, ' ')} />
                  </span>
                  <b>{info.count}</b>
                </button>
              ))}
            </div>
          </div>
        ) : trainTags.length === 0 ? (
          <div className="ds-field-desc" style={{ marginTop: 0 }}>{t('reg.excludeNoTags')}</div>
        ) : null}
      </div>
    </>
  )
}

/** Right card of the generate tab: the run's stdout. Shows the run of the
 *  picked source, or whichever run exists. */
function RunLog({
  source, job, jobLogs, aiTask, aiLogs, onCancelJob,
}: {
  source: 'ai' | 'booru'
  job: Job | null
  jobLogs: string[]
  aiTask: Task | null
  aiLogs: string[]
  onCancelJob: () => Promise<void>
}) {
  const { t } = useTranslation()
  const preRef = useRef<HTMLPreElement>(null)
  const showJob = job != null && (source === 'booru' || aiTask == null)
  const lines = showJob ? jobLogs : aiTask ? aiLogs : []
  const run = showJob ? job : aiTask
  useEffect(() => {
    if (preRef.current) preRef.current.scrollTop = preRef.current.scrollHeight
  }, [jobLogs, aiLogs, showJob])
  const jobLive = showJob && (job.status === 'running' || job.status === 'pending')
  return (
    <div className="ds-card" style={{ display: 'flex', flexDirection: 'column', minWidth: 0, minHeight: 0 }}>
      <div className="ds-card-head ds-pad">
        <div>
          <div className="ds-card-title">{t('reg.logTitle')}</div>
          <div className="ds-card-sub">{showJob ? t('reg.logSubBooru') : aiTask ? t('reg.logSubAi') : t('reg.logSubIdle')}</div>
        </div>
        <div className="ds-card-tools">
          {run && (
            <span className={`ds-badge ${runBadgeTone(run.status)}`}>
              {t('reg.latestRun', { id: run.id, status: runStatusLabel(run.status, t) })}
            </span>
          )}
          {jobLive && (
            <button type="button" className="ds-ctl ds-ghost" style={{ height: 26 }} onClick={() => void onCancelJob()}>{t('common.cancel')}</button>
          )}
        </div>
      </div>
      <div className="ds-card-body" style={{ paddingTop: 2, flex: 1, minHeight: 0, display: 'flex' }}>
        {lines.length > 0 ? (
          <pre ref={preRef} className="ds-console" style={{ flex: 1, minHeight: 240, margin: 0, overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            {lines.map((line, i) => (
              <div key={i} className={/error|failed|traceback/i.test(line) ? 'ds-e' : /warn|skip/i.test(line) ? 'ds-w' : /\bdone\b|完成/i.test(line) ? 'ds-g' : undefined}>{line || ' '}</div>
            ))}
          </pre>
        ) : (
          <div className="ds-empty" style={{ flex: 1, justifyContent: 'center', minHeight: 240 }}>{t('reg.logEmpty')}</div>
        )}
      </div>
    </div>
  )
}

// Images tab (mockup RegImages): folder chips + grid on the left, the picked
// image with its caption on the right.
function RegImages({
  pid, vid, reg, isLive, runTarget, onTopUp, onPreview, onChanged,
}: {
  pid: number
  vid: number
  reg: RegStatus
  isLive: boolean
  runTarget: number
  onTopUp: () => void
  onPreview: (idx: number) => void
  onChanged: () => void
}) {
  const { t } = useTranslation()
  const { toast } = useToast()
  const { confirm } = useDialog()
  // reg.files 是相对 reg/ 的路径（含子文件夹镜像 train，例如 "5_concept/2001.png"）
  const allItems = useMemo(
    () =>
      reg.files.map((rel) => {
        const idx = rel.lastIndexOf('/')
        const folder = idx >= 0 ? rel.slice(0, idx) : ''
        const name = idx >= 0 ? rel.slice(idx + 1) : rel
        return {
          name: rel,
          folder,
          file: name,
          thumbUrl: api.versionThumbUrl(pid, vid, 'reg', name, folder),
        }
      }),
    [reg.files, pid, vid]
  )
  // 按子文件夹分 chip；根（""，老 build 才有）放最后
  const folders = useMemo(() => {
    const order = Array.from(new Set(allItems.map((it) => it.folder)))
    order.sort((a, b) => {
      if (a === '' && b !== '') return 1
      if (b === '' && a !== '') return -1
      return a.localeCompare(b)
    })
    return order
  }, [allItems])
  const folderCounts = useMemo(() => {
    const m = new Map<string, number>()
    for (const it of allItems) m.set(it.folder, (m.get(it.folder) ?? 0) + 1)
    return m
  }, [allItems])
  // null = 全部；否则限定到该 folder
  const [activeFolder, setActiveFolder] = useState<string | null>(null)
  const items = useMemo(
    () => (activeFolder === null ? allItems : allItems.filter((it) => it.folder === activeFolder)),
    [allItems, activeFolder]
  )
  const names = useMemo(() => items.map((it) => it.name), [items])
  const allIndexByName = useMemo(() => {
    const m = new Map<string, number>()
    allItems.forEach((it, i) => m.set(it.name, i))
    return m
  }, [allItems])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [anchor, setAnchor] = useState<string | null>(null)
  const [active, setActive] = useState<string | null>(null)
  // 切 folder：清空选择 + anchor。多选只在当前范围内生效。
  useEffect(() => {
    setSelected(new Set())
    setAnchor(null)
  }, [activeFolder])
  // reg.files 变化（删除完 refresh 后）：把已不存在的 name 从 selected 清掉
  useEffect(() => {
    const fileSet = new Set(allItems.map((it) => it.name))
    setSelected((prev) => {
      let changed = false
      const next = new Set<string>()
      for (const n of prev) {
        if (fileSet.has(n)) next.add(n)
        else changed = true
      }
      return changed ? next : prev
    })
    setActive((a) => (a && fileSet.has(a) ? a : allItems[0]?.name ?? null))
  }, [allItems])

  const openFull = (name: string) => {
    const i = allIndexByName.get(name)
    if (i !== undefined) onPreview(i)
  }

  const deleteNames = async (list: string[]) => {
    if (list.length === 0) return
    const ok = await confirm(
      t('reg.confirmDeleteFiles', { n: list.length }),
      { tone: 'danger', okText: t('reg.deleteOkText') }
    )
    if (!ok) return
    try {
      const r = await api.deleteRegFiles(pid, vid, list)
      toast(t('reg.deleteFilesDone', { n: r.count }), 'success')
      setSelected(new Set())
      setAnchor(null)
      onChanged()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // 文件夹改名（对齐 Step 1 train 改名）：主要场景 = 改 Kohya repeat 前缀
  // （2_data → 1_data），调 reg repeat 不必去 Colab 文件系统手动 rename。
  const [renaming, setRenaming] = useState<{ from: string; value: string } | null>(null)
  const [renameBusy, setRenameBusy] = useState(false)
  const doRename = async () => {
    if (!renaming) return
    const next = renaming.value.trim()
    if (!next || next === renaming.from) { setRenaming(null); return }
    setRenameBusy(true)
    try {
      await api.renameRegFolder(pid, vid, renaming.from, next)
      toast(t('reg.renameFolderDone', { from: renaming.from, to: next }), 'success')
      if (activeFolder === renaming.from) setActiveFolder(next)
      setRenaming(null)
      onChanged()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setRenameBusy(false)
    }
  }

  // 自动去重：用默认参数扫，把每组里的"推荐删除"项直接删，没 review panel。
  // reg 集 quality bar 比 train 低，不需要逐组人工选保留。
  const [dedupBusy, setDedupBusy] = useState(false)
  const onDedup = async () => {
    if (dedupBusy || isLive) return
    const ok = await confirm(t('reg.confirmDedup'), {
      tone: 'danger', okText: t('reg.dedupOkText'),
    })
    if (!ok) return
    setDedupBusy(true)
    try {
      const r = await api.dedupPurgeReg(pid, vid)
      toast(
        t('reg.dedupDone', {
          scanned: r.scanned, groups: r.groups, deleted: r.count,
        }),
        'success'
      )
      setSelected(new Set())
      setAnchor(null)
      onChanged()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setDedupBusy(false)
    }
  }

  const busy = isLive || dedupBusy || renameBusy
  const activeItem = active ? allItems.find((it) => it.name === active) ?? null : null

  return (
    <div className="ds-reg-images">
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, minWidth: 0, borderRight: '1px solid var(--line)' }}>
        <div style={{ padding: '12px 17px', borderBottom: '1px solid var(--line)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <button type="button" className={`ds-chip${activeFolder === null ? ' ds-is-active' : ''}`} style={{ paddingRight: 9 }} onClick={() => setActiveFolder(null)} aria-pressed={activeFolder === null}>
            {t('reg.folderAll')} <b>{allItems.length}</b>
          </button>
          {folders.length > 1 && folders.map((f) => (
            <button key={f || '__root__'} type="button" className={`ds-chip ds-mono${activeFolder === f ? ' ds-is-active' : ''}`} style={{ paddingRight: 9 }} onClick={() => setActiveFolder(f)} aria-pressed={activeFolder === f}>
              {f || t('reg.folderRoot')} <b>{folderCounts.get(f) ?? 0}</b>
            </button>
          ))}
          <span style={{ flex: 1 }} />
          {selected.size > 0 ? (
            <button type="button" className="ds-ctl ds-ghost" style={{ height: 28 }} onClick={() => { setSelected(new Set()); setAnchor(null) }}>{t('common.deselect')}</button>
          ) : (
            <button type="button" className="ds-ctl ds-ghost" style={{ height: 28 }} onClick={() => setSelected(new Set(names))} disabled={names.length === 0}>{t('common.selectAll')}</button>
          )}
          <button
            type="button"
            className="ds-ctl ds-ghost"
            style={{ height: 28 }}
            onClick={() => void deleteNames(Array.from(selected))}
            disabled={selected.size === 0 || busy}
            title={t('reg.deleteFilesTitle')}
          >{t('reg.deleteFilesBtn', { n: selected.size })}</button>
        </div>

        {renaming && (
          <div style={{ padding: '10px 17px', borderBottom: '1px solid var(--line)', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="ds-kpi-meta">{t('reg.renameFolderLabel', { name: renaming.from })}</span>
            <input
              autoFocus
              className="ds-inp ds-mono"
              style={{ maxWidth: 180, height: 28 }}
              value={renaming.value}
              onChange={(e) => setRenaming({ from: renaming.from, value: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void doRename()
                if (e.key === 'Escape') setRenaming(null)
              }}
            />
            <button type="button" className="ds-btn-primary" style={{ height: 28 }} onClick={() => void doRename()} disabled={renameBusy}>{t('reg.renameFolderOk')}</button>
            <button type="button" className="ds-ctl ds-ghost" style={{ height: 28 }} onClick={() => setRenaming(null)} disabled={renameBusy}>{t('reg.renameFolderCancel')}</button>
          </div>
        )}

        <div style={{ padding: '14px 17px', flex: 1, minHeight: 0 }}>
          <ImageGrid
            items={items.map((it) => ({ name: it.name, thumbUrl: it.thumbUrl }))}
            selected={selected}
            activeName={active ?? undefined}
            onSelect={(name, e) => {
              const r = applySelection(selected, name, e, names, anchor)
              setSelected(r.next)
              setAnchor(r.anchor)
            }}
            onActivate={setActive}
            onPreview={openFull}
            clickMode="activate"
            columnsClass="grid-cols-[repeat(auto-fill,minmax(84px,1fr))]"
            ariaLabel="reg-preview"
          />
        </div>

        <div style={{ borderTop: '1px solid var(--line)', padding: '10px 17px', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span className="ds-kpi-meta">
            {t('reg.imagesFooter', { n: allItems.length, folders: folders.length })}
            {selected.size > 0 && ` · ${t('reg.regPreviewSelected', { n: selected.size })}`}
          </span>
          <span style={{ flex: 1 }} />
          {activeFolder !== null && activeFolder !== '' && (
            <button
              type="button"
              className="ds-ctl ds-ghost"
              style={{ height: 28 }}
              onClick={() => setRenaming({ from: activeFolder, value: activeFolder })}
              disabled={busy}
              title={t('reg.renameFolderTitle')}
            >{t('reg.renameFolderBtn')}</button>
          )}
          <button type="button" className="ds-ctl ds-ghost" style={{ height: 28 }} onClick={() => void onDedup()} disabled={busy} title={t('reg.dedupTitle')}>
            {dedupBusy ? t('reg.dedupRunning') : t('reg.dedupBtn')}
          </button>
          <button type="button" className="ds-ctl" style={{ height: 28 }} onClick={onTopUp} disabled={busy || allItems.length >= runTarget}>
            {t('reg.topUpTo', { n: runTarget })}
          </button>
        </div>
      </div>

      <RegImageDetail
        pid={pid}
        vid={vid}
        item={activeItem}
        meta={reg.meta}
        busy={busy}
        onOpenFull={() => activeItem && openFull(activeItem.name)}
        onDelete={() => activeItem && void deleteNames([activeItem.name])}
      />
    </div>
  )
}

function RegImageDetail({ pid, vid, item, meta, busy, onOpenFull, onDelete }: {
  pid: number
  vid: number
  item: { name: string; folder: string; file: string } | null
  meta: RegStatus['meta']
  busy: boolean
  onOpenFull: () => void
  onDelete: () => void
}) {
  const { t } = useTranslation()
  const [caption, setCaption] = useState<string | null>(null)
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null)
  const name = item?.name
  useEffect(() => {
    setDims(null)
    if (!name) { setCaption(null); return }
    let alive = true
    setCaption(t('reg.captionLoading'))
    api.getRegCaption(pid, vid, name)
      .then((r) => { if (alive) setCaption(r.tags.length ? r.tags.join(', ') : t('reg.captionEmpty')) })
      .catch((e) => { if (alive) setCaption(t('reg.captionFailed', { error: String(e) })) })
    return () => { alive = false }
  }, [pid, vid, name, t])

  if (!item) {
    return <div className="ds-empty" style={{ margin: 16 }}>{t('reg.pickImage')}</div>
  }
  const repeat = /^(\d+)_/.exec(item.folder)?.[1]
  const source = meta ? (meta.generation_method === 'ai_base' ? t('reg.sourceAi') : meta.api_source) : '—'
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, minWidth: 0 }}>
      <div className="ds-pane-head">
        <div className="ds-card-title ds-mono" style={{ flex: 1, fontSize: 12.5, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={item.name}>{item.file}</div>
        <KebabMenu
          label={t('reg.imageActions')}
          items={[
            { label: t('reg.openFull'), onSelect: onOpenFull },
            { label: t('reg.deleteThis'), tone: 'err', onSelect: onDelete, disabled: busy },
          ]}
        />
      </div>
      <div className="ds-reg-detail" style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 14, overflowY: 'auto', minHeight: 0 }}>
        <button type="button" className="ds-thumb" style={{ aspectRatio: '1', width: '100%', display: 'block', flex: 'none' }} onClick={onOpenFull} aria-label={t('reg.openFull')}>
          <img
            src={regOrigUrl(pid, vid, item.name)}
            alt={item.file}
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
            onLoad={(e) => setDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
          />
        </button>
        <div>
          <div className="ds-cap" style={{ marginBottom: 8 }}>{t('reg.captionTitle')}</div>
          <div className="ds-card ds-flat ds-mono" style={{ padding: '9px 11px', border: '1px solid var(--line-2)', background: 'var(--card)', fontSize: 11.5, lineHeight: 1.55, wordBreak: 'break-word' }}>
            {caption ?? '—'}
          </div>
          <div className="ds-ctl-note" style={{ marginTop: 6 }}>{t('reg.captionNote')}</div>
        </div>
        <div style={{ borderTop: '1px solid var(--line)', paddingTop: 12 }}>
          <div className="ds-kv"><span className="ds-k">{t('reg.kvSource')}</span><span className="ds-v">{source}</span></div>
          <div className="ds-kv"><span className="ds-k">{t('reg.kvSize')}</span><span className="ds-v">{dims ? `${dims.w} × ${dims.h}` : '—'}</span></div>
          <div className="ds-kv"><span className="ds-k">{t('reg.kvFolder')}</span><span className="ds-v">{item.folder || t('reg.folderRoot')}</span></div>
          <div className="ds-kv"><span className="ds-k">{t('reg.repeatLabel')}</span><span className="ds-v">{repeat ?? '—'}</span></div>
        </div>
      </div>
    </div>
  )
}

function regOrigUrl(pid: number, vid: number, rel: string): string {
  const idx = rel.lastIndexOf('/')
  const folder = idx >= 0 ? rel.slice(0, idx) : ''
  const name = idx >= 0 ? rel.slice(idx + 1) : rel
  // 768px 预览（与 PP3 alt-hover 同尺寸）
  return api.versionThumbUrl(pid, vid, 'reg', name, folder, 768)
}

function formatAgo(unix: number, t: TFunction): string {
  const now = Date.now() / 1000
  const dt = now - unix
  if (dt < 60) return t('reg.agoJustNow')
  if (dt < 3600) return t('reg.agoMinutes', { n: Math.floor(dt / 60) })
  if (dt < 86400) return t('reg.agoHours', { n: Math.floor(dt / 3600) })
  return t('reg.agoDays', { n: Math.floor(dt / 86400) })
}
