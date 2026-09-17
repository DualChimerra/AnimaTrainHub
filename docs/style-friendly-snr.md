# Style-Friendly SNR Sampler（风格 LoRA 专用 timestep 采样）

> 状态：**已实现**（`timestep_sampling: style_friendly`），opt-in / default-off。
> 不选本模式时全链路逐字节等价于改动前。
> 论文：[Style-Friendly SNR Sampler for Style-Driven Generation](https://arxiv.org/abs/2411.14793)
> （在 FLUX-dev / SD3.5 上验证；两者与 Anima 同为 rectified flow，数学可直接搬）。

## 1. 解决什么问题

训练一个 step 要先抽一个噪声水平 `t`。**不同噪声档学到的东西不一样**：

| t 区间 | 模型在这一档能看到什么 | 学到的是 |
|---|---|---|
| 高（≈0.8-0.99） | 只剩大团色块与明暗关系 | 用色、光照、构图、块面、整体气质 = **风格** |
| 中（≈0.4-0.7） | 轮廓与形体成形 | 结构、解剖、姿态 |
| 低（≈0.05-0.3） | 细线与纹理 | 笔触颗粒、毛发丝、噪点 = **细节/易过拟合区** |

SD3/Anima 默认的 `logit_normal` 把火力摊在中段，风格档欠采样。论文的观察是：
把 log-SNR 分布整体推向低值（= 高噪声）后风格贴合度显著上升——**rank 32 + 本采样器
超过 rank 128 + SD3 采样器**，即"采到正确的噪声档"比"堆可训练参数"更划算。

对风格 LoRA 的实际意义：想要"整体气质像，但别把数据集的细节死记硬背下来"时，
这是比调 rank / LR / 步数更直接的旋钮。

## 2. 数学

本仓库 rectified flow 约定：`t=0` 数据端、`t=1` 噪声端、`x_t = (1-t)·x0 + t·x1`。

```
SNR = ((1-t)/t)²                  # 信号/噪声功率比
λ   = log-SNR = 2·ln((1-t)/t)
t   = sigmoid(-λ/2)               # 反解
```

采样：`λ ~ N(style_snr_mean, style_snr_sigma²)` → `t = sigmoid(-λ/2)`。

实现：`runtime/training/timestep_sampling.py::sample_t_style_friendly`。

## 3. 怎么开

```yaml
timestep_sampling: style_friendly   # 新增枚举值
style_snr_mean: -6.0                # log-SNR 均值；论文 FLUX/SD3.5 配方
style_snr_sigma: 2.0                # log-SNR 标准差；论文推荐 2.0-3.0
```

| mean | 中位 t | 效果 |
|---|---|---|
| -8 | ≈0.982 | 极端风格档；结构/细节几乎不学 |
| **-6** | **≈0.953** | 论文配方，风格优先 |
| -4 | ≈0.881 | 风格为主 + 一点结构 |
| -2 | ≈0.731 | 约等于现默认 `logit_normal + shift 3`（≈0.75） |
| 0 | 0.5 | 中性 |

`style_snr_sigma` 是窗口宽度：小 = 火力集中但覆盖窄（该档过拟合风险），
大 = 覆盖宽但回到摊薄。先用 2.0。

## 3.5 与现默认的精确关系

现默认 `logit_normal + timestep_shift=s` **恰好是本模式的一个特例**：

```
u = sigmoid(z), z~N(0,1)；Möbius shift 在 log-odds 上是平移 logit(t) = z + ln s
λ = 2·ln((1-t)/t) = -2·logit(t)  ⇒  λ ~ N(-2·ln s, 2²)
```

即 `shift=s` ≡ `style_friendly(mean=-2·ln s, sigma=2)`：

| timestep_shift | 等价 style_snr_mean | 中位 t |
|---|---|---|
| 2.0 | -1.39 | 0.667 |
| 2.5 | -1.83 | 0.714 |
| **3.0（现默认）** | **-2.20** | 0.749 |
| 4.0 | -2.77 | 0.799 |

论文的风格档配方是 **-6**——比现默认低近 4 个 log-SNR 单位。这就是"默认配置离风格档
有多远"的量化答案，也是为什么单纯调大 `timestep_shift` 到不了那儿：要做到 mean=-6
需要 shift≈20，远超字段上限 10（且那条路径的 sigma 被写死为 2，不可调）。

对照（sigma=2.0）：

| style_snr_mean | 中位 t | 样本落在 t>0.9 的比例 |
|---|---|---|
| -8 | 0.982 | 96% |
| -6 | 0.953 | 79% |
| -5 | 0.924 | 62% |
| -4 | 0.881 | 42% |
| -3 | 0.817 | 24% |
| -2.2（= 现默认） | 0.749 | 13% |

契约由 `test_is_a_strict_generalisation_of_logit_normal_shift` 钉死。

## 4. 与既有旋钮的关系

- **`timestep_shift` 不参与**。它是 Möbius 偏移、作用在 logit-normal 内部的 `u` 上；
  本模式的偏移完全由 `style_snr_mean` 给定，两者叠加 = 双重偏移。schema 在选中本模式时
  隐藏该字段，`sample_t` 也不读它（`test_sample_t_dispatches_and_ignores_timestep_shift`）。
- **`timestep_schedule_shift` 仍叠加**（默认 1.0 = 恒等）。它是采样后的全局 σ schedule 偏移，
  与本模式正交；一般保持 1.0。
- **`timestep_shift_resolution_aware` 仍独立生效**：按每图 token 数做分辨率修正。
  注意它与本模式叠加时同样是乘法复合——原生分辨率训练下大图会被再往噪声端推一截。
- **`loss_weighting=detail_inv_t`** 与本模式方向相反（强化低 t 细节）。同开等于一边踩油门
  一边踩刹车，不推荐。
- **InfoNoise**：开启时正式阶段由自适应 CDF 接管分布，本模式只作为热身期 baseline
  （用户设的 mean/sigma 会被透传过去）。

## 5. 测试

`tests/test_style_friendly_sampler.py`（纯 CPU）：约定映射、论文默认落点、
mean 单调性、sigma 宽度、开区间、shift 不参与、schedule_shift 仍叠加、
registry 接线、`mean=0.0` 不被 falsy 默认吞掉、schema 字段与 show_when。
