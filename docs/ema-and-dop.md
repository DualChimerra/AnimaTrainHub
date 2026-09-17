# EMA 与 DOP —— 两个「别把风格 LoRA 训废」的开关

两者都是 opt-in / default-off，不开时全链路与改动前等价。放在一篇是因为它们解决的是
风格 LoRA 的同一类痛：**过拟合与选点靠运气**。

---

## 1. EMA —— 权重指数滑动平均

`ema_enabled` / `ema_decay` / `ema_start_ratio` · 实现：`runtime/training/ema.py`

### 解决什么

训练到某一步的权重，是「最后一次随机批次把参数推到哪」的结果。相邻 epoch 一个偏这边
一个偏那边，「第几个 epoch 最好」于是变成抓阄。

EMA 在训练途中额外维护一份平滑副本，每次 optimizer.step() 之后

```
shadow ← decay · shadow + (1 - decay) · 当前权重
```

正常 checkpoint 照存，**额外**再存一份 `*_ema.safetensors`。两个都能拿去试。

### 怎么设

| 字段 | 含义 | 建议 |
|---|---|---|
| `ema_decay` | 平滑窗口。0.999 ≈ 最近 1000 个 update step | 总步数 2000-3000 → **0.999**；步数很少 → 0.99 |
| `ema_start_ratio` | 从总步数的百分之多少开始累计 | **0**（含 warmup 已经够）；想只平均平台期 → 0.3 |

### 两个实现要点

* **影子是 fp32。** 训练权重 bf16 只有 8 位尾数；decay=0.999 的千分之一增量在 bf16
  累加器里会被直接舍成 0，EMA 静默退化成「一份不动的旧拷贝」。
* **影子从「开始那一刻的权重」起步**，不是从 LoRA 的零初始化起步；配合
  `(1+n)/(10+n)` 的 warmup，早期几乎直接跟随当前权重。否则 2760 步后仍有
  0.999^2760 ≈ 6% 的「零」留在平均里，白白削弱 LoRA 6%。

落盘复用 adapter 自己的 `save()`（`ema.applied()` 上下文临时换入影子），所以 alpha
重写 / `ss_*` metadata / 族标记完全一致。影子随训练状态一起存，resume 不丢。

---

## 2. DOP —— 差分输出保持

`dop_enabled` / `dop_weight` / `dop_ratio` · 实现：`runtime/training/dop.py`
来源：kohya-ss/sd-scripts PR #1710、ostris/ai-toolkit 同名功能。

### 解决什么

风格 LoRA 训完常见两个毛病：

1. 提示词里没有触发词，它照样改图（风格「漏」得到处都是）；
2. 把数据集的**内容**（同一个典型角色、同一种背景）一起搬进生成结果 ——
   「不是把风格贴上去，而是把数据集抄过来」。

传统解法是正则集：另外准备一堆中性图。DOP 不需要额外图片：

1. 拿**同一批训练图**，把 caption 里的触发词去掉；
2. **关掉适配器**跑一次前向 → 底模本来会画成什么样（no_grad，常量目标）；
3. **开着适配器**跑同样的输入 → 现在会画成什么样；
4. 两者 MSE × `dop_weight` 加进总 loss。

于是 LoRA 被同时教两件事：**带触发词 = 我的风格；不带触发词 = 我什么都不改**。
内容在两条分支里完全相同，所以「抄内容」这条路拿不到任何奖励。

### 怎么设

| 字段 | 含义 | 建议 |
|---|---|---|
| `dop_weight` | 保持项相对主 loss 的权重（两者同为 MSE，1.0 = 等权） | **1.0**；风格还在漏 → 2-5；风格学不进去 → 0.3-0.5 |
| `dop_ratio` | 在多大比例的 step 上启用 | **1.0**；嫌慢 → 0.5（约束强度也减半） |

### 前提与代价

* **必须有非空 `trigger_word`。** 保持分支就是「去掉触发词」的那一条；没有触发词
  两条分支完全一样、约束退化成 0。启动期 fail-fast，不静默降级。
  触发词在 Train 页顶部的「Trigger word / 触发词」卡片里填写（写入 version，
  并强制同步进 config.yaml）；「从 caption 识别」会读 train/ 下的 caption，找出
  （几乎）每条都有的那个词，通常就是开头的 `@handle`。入队训练时若 DOP 已开但
  version 没有触发词：先沿用 yaml 里已有的值，再尝试从 caption 自动识别（必须
  100% 覆盖且位于开头或是 `@handle`），都没有才在入队阶段直接报错。
* **每个启用的 step 多两次前向**（一次 no_grad 参照 + 一次带梯度），约 2-2.5× 单步耗时。
* **范围（v1）**：标准 rectified flow 路径。与 LeapAlign（自带目标函数）、NaViT
  打包（逐图 cross 需重新打包）在 schema 层互斥。
* 适配器必须实现 `disabled()`（临时把缩放置 0）。LyCORIS / OrthoLoRA 都已实现；
  其它适配器会在启动期报错而不是训到一半才发现。

监控面板与 wandb 里多一条 `dop_loss` 曲线：它应当先降后平，长期不降说明
`dop_weight` 太小（约束没咬住）或触发词在 caption 里位置不一致。

---

## 3. 两者一起用

互不冲突，推荐组合：DOP 管「风格别乱漏、别抄内容」，EMA 管「哪个 checkpoint 都不用赌」。
风格 LoRA 的一套起手式：

```yaml
timestep_sampling: style_friendly   # 火力压在风格档
style_snr_mean: -6.0
dop_enabled: true                   # 不抄内容、不乱漏
dop_weight: 1.0
ema_enabled: true                   # 选点不靠运气
ema_decay: 0.999
```

测试：`tests/test_ema.py`（12 条）、`tests/test_dop.py`（17 条），均为纯 CPU。
