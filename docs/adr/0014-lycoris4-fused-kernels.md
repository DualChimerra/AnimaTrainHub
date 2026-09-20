# 0014 — LyCORIS 4 fused kernels 与 Windows Triton

**状态**：Accepted

**日期**：2026-09-20
**决策者**：项目维护者

## 背景

训练栈通过 `utils/lycoris_adapter.py` 接入 LyCORIS 3.4。普通 LoRA 已走
bypass forward，但 LoHa / LoKr / DoRA 与 plain T-LoRA 的 rebuild 路径仍可能
物化完整 `ΔW`。LyCORIS 4.0 新增 Triton / TileLang fused kernels，并按
Triton → TileLang → `torch.compile` → eager 的顺序逐调用选择与回退。

Windows 上 PyTorch 不自带可导入的 Triton，缺失时还会让 xformers 在启动期打印
非致命 traceback。当前项目的目标本机环境是 PyTorch 2.11 + CUDA 12.8；该组合
按 triton-windows 的兼容表对应 Triton 3.6。

同时，LyCORIS 4.0 移除了旧 patch 所依赖的内部 `make_kron` /
`rebuild_tucker` import，但 4.0.0 的 LoKr `rank_dropout` 仍会在 CPU 创建 mask。

## 候选方案

1. **保留 LyCORIS 3.4**：风险最低，但无法使用 fused kernels，启动警告保留。
2. **直接升级并删除本地 adapter / patch**：代码最少，但丢失 family preset、
   checkpoint metadata、训练 hook、T-LoRA timestep mask，并重新暴露 LoKr device bug。
3. **升级到 4.0，保留薄 integration adapter，数学运算委托官方 functional API**：
   需要小范围兼容改动，但保留项目协议并获得官方 kernel dispatch。

## 决策

- 固定 `lycoris-lora==4.0.0`，避免未来 major API 静默漂移。
- Windows dependency 声明允许 Triton 3.6–3.8；启动器在导入 xformers/flash-attn
  之前按已安装 PyTorch minor 自动收窄或修复版本（2.10/2.11→3.6、
  2.12/2.13→3.7、2.14→3.8）。
- 保留 `LycorisAdapter`：它只负责 AdapterProtocol、模型族 target preset、保存元数据、
  sample/eval 状态和 T-LoRA step hook；LoRA 数学继续交给 LyCORIS。
- plain T-LoRA 在两个低秩因子上应用 timestep mask。forward 调用官方
  `functional.locon.bypass_forward_diff`，不物化 `ΔW`；merge/save 路径调用
  `functional.locon.diff_weight`。两者都进入 LyCORIS 4 fused/compile/eager dispatch。
- LoKr device patch 改为包装 upstream `get_weight`：只临时关闭坏的 dropout 分支，
  调用 upstream（保留 fused dispatch），然后在 `weight.device` 重放 dropout。
- OrthoLoRA 保持项目自实现。LyCORIS 4 没有与其 Cayley/SVD 参数化等价的官方
  adapter，不能为了去掉 wrapper 改变算法或 checkpoint 语义。

## 理由

方案 3 把外部算法实现尽量留给上游，同时保留 Anima/Krea2 集成必需的边界。
显式 pin 能让 device patch 的适用版本可审计。Triton 缺少或某个 shape 不受支持时，
LyCORIS 自带回退，因此加速失败不会改变训练正确性。

## 后果

- Windows 首次安装多下载约 50 MB，并在首次遇到新 shape 时支付 JIT/tuning 开销。
- fused kernel 的 microbenchmark 提升不等于整步训练等比例提升；DiT 主干仍占大头。
- `rank_dropout`、卷积、过大 rank 等超出 fused scope 的调用会自动回退。
- Triton 与 PyTorch minor 有配对约束；支持表之外的新 PyTorch 会保守回退并要求
  更新 `_WINDOWS_TRITON_FOR_TORCH`，不会猜测 ABI。

## 参考

- [LyCORIS 4.0 fused kernels](https://github.com/KohakuBlueleaf/LyCORIS/blob/v4.0.0/docs/kernels/README.md)
- [LyCORIS backend selection](https://github.com/KohakuBlueleaf/LyCORIS/blob/v4.0.0/docs/kernels/backends.md)
- [triton-windows PyTorch compatibility](https://github.com/triton-lang/triton-windows#3-pytorch)
