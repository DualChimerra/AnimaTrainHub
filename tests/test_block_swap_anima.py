"""Anima family block swap wiring (docs/design/block-swap.md B3: the mechanism
is family-agnostic; wiring = loader placement + capability bit + version-aware
discount + sampling offload mutual exclusion).

The mechanism core (four hooks / double buffering / pinned lifecycle) is
already pinned down by test_block_swap*.py; this file only tests the wiring
surface newly added on the anima side.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training.families.anima.loader import (  # noqa: E402
    place_model_for_block_swap,
    swapped_param_ratio_from_header,
)

_CUDA = torch.cuda.is_available()


# ── swapped_param_ratio: counts numel from the safetensors header (no CUDA dependency) ────────

def _write_fake_checkpoint(path: Path, *, num_blocks: int, prefix: str = "") -> dict:
    """Write a small safetensors file whose key shapes resemble a real checkpoint.

    One 64-param tensor per layer + two non-block tensors (x_embedder 32 /
    final_layer 16). ``prefix`` simulates the model./module. style prefixes
    that get stripped by _load_weights_best_effort -- the ratio must be
    prefix-agnostic.
    """
    from safetensors.torch import save_file

    tensors = {
        f"{prefix}x_embedder.proj.1.weight": torch.zeros(4, 8),
        f"{prefix}final_layer.linear.weight": torch.zeros(4, 4),
    }
    for i in range(num_blocks):
        tensors[f"{prefix}blocks.{i}.attn.qkv.weight"] = torch.zeros(8, 8)
    save_file(tensors, str(path))
    return tensors


def test_ratio_counts_trailing_blocks_from_header(tmp_path):
    ckpt = tmp_path / "anima.safetensors"
    _write_fake_checkpoint(ckpt, num_blocks=4)
    # total = 32 + 16 + 4x64 = 304; swapping the trailing 2 layers = 128
    assert swapped_param_ratio_from_header(ckpt, 2) == pytest.approx(128 / 304)
    # swap everything
    assert swapped_param_ratio_from_header(ckpt, 4) == pytest.approx(256 / 304)
    # out-of-range clamps to the total layer count (a cross-version shared
    # setting value: 36 fed into a 28-layer checkpoint)
    assert swapped_param_ratio_from_header(ckpt, 99) == pytest.approx(256 / 304)
    assert swapped_param_ratio_from_header(ckpt, 0) == 0.0


def test_ratio_is_prefix_agnostic(tmp_path):
    """Checkpoint keys carry a model. prefix (stripped only at loader time) -> ratio is unaffected."""
    ckpt = tmp_path / "prefixed.safetensors"
    _write_fake_checkpoint(ckpt, num_blocks=4, prefix="model.")
    assert swapped_param_ratio_from_header(ckpt, 2) == pytest.approx(128 / 304)


def test_family_ratio_degrades_to_conservative_zero(tmp_path):
    """No path / broken file -> 0 (the guard budgets for the full model so it never wrongly lets a config through)."""
    from training.families.anima.family import AnimaFamily

    fam = AnimaFamily()
    assert fam.swapped_param_ratio(14) == 0.0  # no checkpoint_path given
    assert fam.swapped_param_ratio(14, checkpoint_path="") == 0.0
    bad = tmp_path / "not_a_checkpoint.safetensors"
    bad.write_bytes(b"garbage")
    assert fam.swapped_param_ratio(14, checkpoint_path=str(bad)) == 0.0
    ckpt = tmp_path / "ok.safetensors"
    _write_fake_checkpoint(ckpt, num_blocks=4)
    assert fam.swapped_param_ratio(2, checkpoint_path=str(ckpt)) == pytest.approx(128 / 304)


def test_family_reports_swappable_block_count_from_checkpoint(tmp_path):
    """Only the checkpoint itself knows the layer-count ceiling (2B=28 / 14B=36); preflight's recommendation search needs it."""
    from training.families.anima.family import AnimaFamily

    fam = AnimaFamily()
    assert fam.swappable_blocks() == 0                       # no path -> can't decide
    assert fam.swappable_blocks(checkpoint_path="") == 0
    bad = tmp_path / "broken.safetensors"
    bad.write_bytes(b"garbage")
    assert fam.swappable_blocks(checkpoint_path=str(bad)) == 0

    ckpt = tmp_path / "ok.safetensors"
    _write_fake_checkpoint(ckpt, num_blocks=28)
    assert fam.swappable_blocks(checkpoint_path=str(ckpt)) == 28


def test_preflight_lets_anima_run_on_a_6gb_card(tmp_path, monkeypatch):
    """**Regression**: when preflight doesn't pass through ``checkpoint_path``,
    anima's ratio is always 0, so it computes "swapping any number of layers
    saves no VRAM" and judges an otherwise runnable 6GB config as failing.

    Real-world magnitudes here: Anima 2B bf16 weights are about 4GB, and a
    6GB card has about 5.5GB free. With ratio 0, need = 4GB + 3GB base >
    5.5GB -> rejected. With the real ratio (swapping 28/28 layers, non-block
    params are a rounding error), need drops to just over 3GB -> allowed.
    """
    import types

    from training import block_swap_preflight as preflight, sysmem
    from training.families.anima.family import AnimaFamily

    ckpt = tmp_path / "ok.safetensors"
    _write_fake_checkpoint(ckpt, num_blocks=28)
    monkeypatch.setattr(sysmem, "_file_bytes", lambda _p: int(4 * 1024 ** 3))
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: int(5.5 * 1024 ** 3))
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: int(24 * 1024 ** 3))

    fam = AnimaFamily()

    def _ctx(family):
        return types.SimpleNamespace(
            args=types.SimpleNamespace(
                block_swap_preflight=True,
                blocks_to_swap=28,
                transformer_path=str(ckpt),
            ),
            family=family,
        )

    real = types.SimpleNamespace(
        spec=types.SimpleNamespace(capabilities=frozenset({"block_swap"})),
        swapped_param_ratio=fam.swapped_param_ratio,
        swappable_blocks=fam.swappable_blocks,
    )
    preflight.run(_ctx(real))  # wiring is correct -> allowed

    # control: ratio always 0 (= behavior of a missing checkpoint_path) ->
    # must be rejected, proving the above passes for the right reason and
    # not because the criterion is too loose
    blind = types.SimpleNamespace(
        spec=types.SimpleNamespace(capabilities=frozenset({"block_swap"})),
        swapped_param_ratio=lambda _b, *, checkpoint_path=None: 0.0,
        swappable_blocks=fam.swappable_blocks,
    )
    with pytest.raises(RuntimeError):
        preflight.run(_ctx(blind))


# ── loader placement: swapped-out layers never go on the card (§9.4 discipline) ─────────────────────────────────

class _TinyDiT(torch.nn.Module):
    """Minimal stand-in shaped like Anima: .blocks ModuleList + non-block params and buffers."""

    def __init__(self, num_blocks: int = 4) -> None:
        super().__init__()
        self.x_embedder = torch.nn.Linear(8, 8)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8) for _ in range(num_blocks)]
        )
        self.register_buffer("pos_freq", torch.arange(4, dtype=torch.float32))


def test_pinned_budget_guard_fails_fast_before_any_placement(monkeypatch):
    """The budget guard raises before any move / allocation happens (B6); must hold even without CUDA."""
    from training import sysmem

    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 1024)  # 1KB: must reject
    model = _TinyDiT()
    with pytest.raises(RuntimeError, match="内存不足以换出"):
        place_model_for_block_swap(model, "cuda", torch.float32, 2)
    # fail-fast semantics: the model hasn't moved at all (still all on CPU, no marker)
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert getattr(model, "blocks_to_swap", 0) == 0


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_placement_keeps_swapped_blocks_on_cpu():
    """Non-swapped parts go on the card, the trailing N layers stay on CPU, dtype is cast, the marker is set, out-of-range clamps."""
    model = _TinyDiT(num_blocks=4)
    num = place_model_for_block_swap(model, "cuda", torch.bfloat16, 99)
    assert num == 4  # clamps to the total layer count

    model2 = _TinyDiT(num_blocks=4)
    num2 = place_model_for_block_swap(model2, "cuda", torch.bfloat16, 2)
    assert num2 == 2
    assert model2.blocks_to_swap == 2
    for i, block in enumerate(model2.blocks):
        expect = "cpu" if i >= 2 else "cuda"
        for p in block.parameters():
            assert p.device.type == expect, f"blocks.{i} should be on {expect}"
            assert p.dtype == torch.bfloat16
    # non-block params and buffers all go on the card
    assert model2.x_embedder.weight.device.type == "cuda"
    assert model2.pos_freq.device.type == "cuda"


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_anima_forward_backward_matches_without_swap():
    """A real small-scale Anima model: after loader placement + attaching the
    four hooks, the forward output and input gradient match a no-swap
    control (fp32 eliminates nondeterminism; mechanism race conditions are
    guarded by the grad_fidelity real-size test -- this one pins down anima's
    structural compatibility, i.e. the rope/adaln_lora/cross-attn chain of
    prepared tensors all get fed into the block through a manually unrolled
    loop).
    """
    import copy

    from modeling.anima import anima_modeling
    from training.block_swap import PinnedBlockSwap
    from training.families.anima.forward import forward_with_optional_checkpoint

    config = dict(
        max_img_h=64, max_img_w=64, max_frames=4,
        in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=64, num_blocks=4, num_heads=2,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True, adaln_lora_dim=16,
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=1.0,
    )
    torch.manual_seed(0)
    ref = anima_modeling.Anima(**config)
    swapped = copy.deepcopy(ref)

    ref = ref.to("cuda", torch.float32)
    ref.requires_grad_(False)
    place_model_for_block_swap(swapped, "cuda", torch.float32, 2)
    swapped.requires_grad_(False)
    swap = PinnedBlockSwap(swapped.blocks, 2, "cuda")
    swap.attach()

    def _inputs():
        g = torch.Generator(device="cpu").manual_seed(7)
        latents = torch.randn(1, 16, 1, 8, 8, generator=g).to("cuda")
        t = torch.full((1, 1), 0.5, device="cuda")
        cross = torch.randn(1, 32, 1024, generator=g).to("cuda")
        pad = torch.zeros(1, 1, 8, 8, device="cuda")
        return latents.requires_grad_(True), t, cross, pad

    outs, grads = [], []
    for model in (ref, swapped):
        latents, t, cross, pad = _inputs()
        out = forward_with_optional_checkpoint(
            model, latents, t, cross, pad, use_checkpoint=True,
        )
        out.square().mean().backward()
        outs.append(out.detach())
        grads.append(latents.grad.detach().clone())
    torch.cuda.synchronize()

    torch.testing.assert_close(outs[0], outs[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-5)

    swap.close()


# ── sampling-time VAE decode offload and swap are mutually exclusive ─────────────────────────────────

def test_decode_offload_skips_swapped_dit():
    from training.families.anima.sampling import _decode_offload_targets

    model = _TinyDiT()
    qwen = object()
    # no swap: both DiT and Qwen can be offloaded
    assert _decode_offload_targets(model, qwen) == (model, qwen)
    # swap active (marker set by the loader): only offload Qwen -- the
    # blanket .to() during restore would move the pinned CPU master copy
    # back onto the card, so DiT must be skipped
    model.blocks_to_swap = 2
    assert _decode_offload_targets(model, qwen) == (qwen,)


# ── the NaViT packed path must also go through __call__ (otherwise the four hooks silently never fire) ─────────────

@pytest.mark.skipif(not _CUDA, reason="requires CUDA + xformers")
def test_navit_packed_forward_backward_matches_without_swap():
    """Regression: NaViT's block-diagonal packed forward matches a no-swap
    control value-for-value under block swap.

    The old bug: the packing loop called ``blk.forward_tokens(...)``
    directly. nn.Module's forward/backward hooks only fire through
    ``__call__``, so swap's fetch/release never ran even once -- the
    swapped-out block kept pointing at the weights left in the two GPU slots
    by the last standard forward pass (e.g. the step-0 baseline sample).
    **Everything stays on the card, so nothing errors and nothing is NaN**;
    it's just that 26 layers become a duplicate of two, and per attach()'s
    documented semantics the gradient is silently wrong -- after 2-5 epochs
    the output degrades into a mosaic.

    The fix: the packing loop goes through ``blk(..., packed_tokens=True)``,
    the same hook semantics as the standard checkpoint loop in
    ``families/anima/forward.py``.
    """
    import copy
    from types import SimpleNamespace

    pytest.importorskip("xformers")

    from modeling.anima import anima_modeling
    from training.block_swap import PinnedBlockSwap
    from training.families.anima.navit import (
        navit_packed_forward_and_loss,
        pack_cross_embeddings,
    )
    from training.losses import build_loss  # noqa: PLC0415

    config = dict(
        max_img_h=64, max_img_w=64, max_frames=4,
        in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=64, num_blocks=4, num_heads=2,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True, adaln_lora_dim=16,
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=1.0,
    )
    torch.manual_seed(0)
    ref = anima_modeling.Anima(**config)
    swapped = copy.deepcopy(ref)

    ref = ref.to("cuda", torch.float32)
    ref.requires_grad_(False)
    place_model_for_block_swap(swapped, "cuda", torch.float32, 2)
    swapped.requires_grad_(False)
    swap = PinnedBlockSwap(swapped.blocks, 2, "cuda")
    swap.attach()

    # reproduce the real scenario: run one standard forward pass first so
    # the two GPU slots hold the weights of the "last two layers". Before
    # the fix, the packed forward pass right after this is exactly the
    # moment those two weight sets get used to impersonate all swapped layers.
    from training.families.anima.forward import forward_with_optional_checkpoint
    forward_with_optional_checkpoint(
        swapped,
        torch.randn(1, 16, 1, 8, 8, device="cuda"),
        torch.full((1, 1), 0.5, device="cuda"),
        torch.randn(1, 32, 1024, device="cuda"),
        torch.zeros(1, 1, 8, 8, device="cuda"),
        use_checkpoint=False,
    )

    shapes = [(8, 8), (8, 12)]          # G=2 heterogeneous images
    loss_fn = build_loss(SimpleNamespace(loss_type="mse"))

    def _inputs():
        g = torch.Generator(device="cpu").manual_seed(7)
        lats = [
            torch.randn(1, 16, 1, h, w, generator=g).to("cuda").requires_grad_(True)
            for h, w in shapes
        ]
        t = torch.tensor([0.3, 0.7], device="cuda")
        cross = torch.randn(len(shapes), 32, 1024, generator=g).to("cuda")
        return lats, t, cross

    outs, grads = [], []
    for model in (ref, swapped):
        lats, t, cross = _inputs()
        cross_packed, text_seqlens = pack_cross_embeddings(cross, None, False)
        torch.manual_seed(11)           # per-image noising uses the global RNG; both sides need the same seed
        loss, _pred, _info = navit_packed_forward_and_loss(
            model, lats, t, cross_packed, text_seqlens, loss_fn,
            use_checkpoint=True,
        )
        loss.backward()
        outs.append(loss.detach())
        grads.append(torch.cat([l.grad.reshape(-1) for l in lats]).clone())
    torch.cuda.synchronize()

    torch.testing.assert_close(outs[0], outs[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-5)

    swap.close()
