"""``Block.forward(packed_tokens=True)`` 派发契约（NaViT 打包路径的钩子前提）。

block swap 把权重取回/放开挂在 nn.Module 的四个钩子上，而钩子**只在
``__call__`` 上触发**。所以打包循环必须走 ``blk(..., packed_tokens=True)``，
不能直接调 ``blk.forward_tokens(...)``——后者会让换出层拿着槽里残留的别层权重
静默算下去（全在卡上，不报错、不 NaN）。

本文件锁两件事，都不需要 CUDA / xformers：
1. 派发本身：``packed_tokens=True`` 的输出 == 直接调 ``forward_tokens``，且钩子触发；
2. 源码不变式：``forward_packed_navit`` 的 block 循环里不出现直调。

数值等价（swap 开/关逐值一致）由 tests/test_block_swap_anima.py 的
``test_navit_packed_forward_backward_matches_without_swap`` 在 GPU 上把关。
"""
from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from modeling.anima.cosmos_predict2_modeling import Block, MiniTrainDIT

D = 32
CTX = 24
COUNTS = [3, 5]


def _block() -> Block:
    torch.manual_seed(0)
    return Block(
        x_dim=D, context_dim=CTX, num_heads=2,
        use_adaln_lora=True, adaln_lora_dim=8,
        # 与 MiniTrainDIT 的 atten_backend 一致；transformer_engine 后端在 CPU/CI 上不可用
        self_attention_backend="torch", cross_attention_backend="torch",
    ).eval()


def _packed_inputs():
    torch.manual_seed(1)
    sn = sum(COUNTS)
    g = len(COUNTS)
    x = torch.randn(1, sn, D)
    cross = torch.randn(1, sum([4] * g), CTX)
    emb = torch.randn(1, g, D)
    lora = torch.randn(1, g, 3 * D)
    mod_index = torch.repeat_interleave(torch.arange(g), torch.tensor(COUNTS))
    return x, emb, cross, lora, mod_index


def test_packed_dispatch_matches_direct_forward_tokens():
    """``blk(..., packed_tokens=True)`` ≡ ``blk.forward_tokens(...)``（逐值）。"""
    blk = _block()
    x, emb, cross, lora, mod_index = _packed_inputs()
    kwargs = dict(token_wise_mod=True, mod_index=mod_index)

    with torch.no_grad():
        direct = blk.forward_tokens(x, emb, cross, adaln_lora_B_T_3D=lora, **kwargs)
        dispatched = blk(x, emb, cross, adaln_lora_B_T_3D=lora, packed_tokens=True, **kwargs)

    torch.testing.assert_close(direct, dispatched)


def test_packed_dispatch_fires_module_hooks():
    """派发路径触发 forward pre/post 钩子——block swap 的取回/放开就挂在这儿。"""
    blk = _block()
    x, emb, cross, lora, mod_index = _packed_inputs()
    fired: list[str] = []
    blk.register_forward_pre_hook(lambda *_a: fired.append("pre"))
    blk.register_forward_hook(lambda *_a: fired.append("post"))

    with torch.no_grad():
        blk(
            x, emb, cross, adaln_lora_B_T_3D=lora, packed_tokens=True,
            token_wise_mod=True, mod_index=mod_index,
        )

    assert fired == ["pre", "post"]


def test_packed_dispatch_backward_hooks_fire():
    """反向钩子同理：少了它梯度会静默算错（attach() 文档，实测偏差 300× 噪声底）。"""
    blk = _block()
    x, emb, cross, lora, mod_index = _packed_inputs()
    fired: list[str] = []
    blk.register_full_backward_pre_hook(lambda *_a: fired.append("bwd_pre"))
    blk.register_full_backward_hook(lambda *_a: fired.append("bwd_post"))

    out = blk(
        x.requires_grad_(True), emb, cross, adaln_lora_B_T_3D=lora,
        packed_tokens=True, token_wise_mod=True, mod_index=mod_index,
    )
    out.square().mean().backward()

    assert "bwd_pre" in fired and "bwd_post" in fired


def test_dense_path_still_rejects_unknown_kwargs():
    """``**packed_kwargs`` 不能把稠密路径的拼写错误吞掉。"""
    blk = _block()
    with pytest.raises(TypeError, match="unexpected keyword"):
        blk(torch.randn(1, 1, 2, 2, D), torch.randn(1, 1, D), torch.randn(1, 4, CTX),
            typo_kwarg=1)


def test_packed_loop_routes_through_call():
    """源码不变式：打包循环不得直调 blk.forward_tokens（钩子会静默失效）。"""
    src = inspect.getsource(MiniTrainDIT.forward_packed_navit)
    # 只看代码行：注释里提到 blk.forward_tokens(...) 是在解释为什么不能那么写
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "blk.forward_tokens(" not in code, (
        "forward_packed_navit 的 block 循环必须走 blk(..., packed_tokens=True)："
        "直调 forward_tokens 会跳过 nn.Module 钩子，block swap 静默用错权重"
    )
    assert "packed_tokens=True" in code
