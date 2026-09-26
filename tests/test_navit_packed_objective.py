"""Unit tests for the NaViT / Patch-n-Pack training step: navit_packed_forward_and_loss +
pack_cross_embeddings.

Verifies:
  1) pack_cross_embeddings: the trim / no-trim paths produce correct concat length and text_seqlens.
  2) navit_packed_forward_and_loss: loss is finite, carries gradients, per_image_loss has shape [G].
  3) grad_checkpoint path output == non-checkpoint path, and still backprops.

Needs CUDA + xformers (varlen kernels are GPU-only); skipped otherwise.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xformers")

from modeling.anima.anima_modeling import Anima
from modeling.anima.cosmos_predict2_modeling import set_xformers_enabled
from training.losses.mse import MseLoss
from training.families.anima.navit import navit_packed_forward_and_loss, pack_cross_embeddings

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA + xformers"
)


def _model(dtype):
    torch.manual_seed(0)
    m = Anima(
        max_img_h=64,
        max_img_w=64,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=128,
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2.0,
        crossattn_emb_channels=128,
        pos_emb_cls="rope3d",
    )
    return m.to(device="cuda", dtype=dtype).eval()


@requires_cuda
def test_pack_cross_embeddings_no_trim():
    """Legacy 512-pad: every image carries its full caption, sum(L) = G * 512."""
    G, L, D = 3, 512, 128
    cross = torch.randn(G, L, D)
    t5_attn = torch.ones(G, L)
    packed, seqlens = pack_cross_embeddings(cross, t5_attn, navit_text_trim_padding=False)
    assert packed.shape == (1, G * L, D)
    assert seqlens == [L] * G
    assert sum(seqlens) == packed.shape[1]


@requires_cuda
def test_pack_cross_embeddings_trim():
    """Trim padding: only valid tokens are concatenated, sum(L) = sum(valid per image)."""
    G, L, D = 3, 512, 128
    cross = torch.randn(G, L, D)
    # different valid length per image
    valid_lens = [10, 77, 256]
    t5_attn = torch.zeros(G, L)
    for i, n in enumerate(valid_lens):
        t5_attn[i, :n] = 1.0
    packed, seqlens = pack_cross_embeddings(cross, t5_attn, navit_text_trim_padding=True)
    assert seqlens == valid_lens
    assert sum(seqlens) == packed.shape[1]
    # verify concat order: the first segment == cross[0, :10]
    torch.testing.assert_close(packed[0, :10, :], cross[0, :10, :])


@requires_cuda
def test_navit_forward_and_loss_basic():
    """navit_packed_forward_and_loss: loss is finite, carries gradients, per_image shape is [G]."""
    set_xformers_enabled(True)
    dtype = torch.float16
    model = _model(dtype)
    D = 128

    latent_shapes = [(4, 4), (6, 8), (4, 6)]
    text_lens = [5, 9, 3]
    timesteps = [0.2, 0.6, 0.9]

    latents_list = [
        torch.randn(1, 16, 1, h, w, device="cuda", dtype=dtype)
        for h, w in latent_shapes
    ]
    cross_list = [torch.randn(1, L, D, device="cuda", dtype=dtype) for L in text_lens]
    cross_packed = torch.cat(cross_list, dim=1)
    t = torch.tensor(timesteps, device="cuda", dtype=dtype)

    loss_fn = MseLoss()
    loss, pred, info = navit_packed_forward_and_loss(
        model, latents_list, t, cross_packed, text_lens, loss_fn,
    )

    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert info["per_image_loss"].shape == (len(latent_shapes),)
    assert all(torch.isfinite(v) for v in info["per_image_loss"])
    # predicted token count == sum(N)
    assert pred.shape[1] == sum(info["visual_seqlens"])

    # gradients backprop successfully
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


@requires_cuda
def test_navit_checkpoint_equivalence():
    """grad_checkpoint path == non-checkpoint path, and still backprops."""
    set_xformers_enabled(True)
    dtype = torch.float16
    model = _model(dtype)
    D = 128

    latent_shapes = [(4, 4), (6, 8)]
    text_lens = [5, 9]
    timesteps = [0.3, 0.7]

    latents_list = [
        torch.randn(1, 16, 1, h, w, device="cuda", dtype=dtype)
        for h, w in latent_shapes
    ]
    cross_list = [torch.randn(1, L, D, device="cuda", dtype=dtype) for L in text_lens]
    cross_packed = torch.cat(cross_list, dim=1)
    t = torch.tensor(timesteps, device="cuda", dtype=dtype)
    loss_fn = MseLoss()

    with torch.no_grad():
        torch.manual_seed(42)
        _, pred_plain, _ = navit_packed_forward_and_loss(
            model, latents_list, t, cross_packed, text_lens, loss_fn,
            use_checkpoint=False,
        )
        torch.manual_seed(42)
        _, pred_ckpt, _ = navit_packed_forward_and_loss(
            model, latents_list, t, cross_packed, text_lens, loss_fn,
            use_checkpoint=True,
        )
    torch.testing.assert_close(pred_ckpt, pred_plain, rtol=2e-3, atol=2e-3)

    # checkpoint path still backprops
    torch.manual_seed(42)
    loss, _, _ = navit_packed_forward_and_loss(
        model, latents_list, t, cross_packed, text_lens, loss_fn,
        use_checkpoint=True,
    )
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all()
        for p in model.parameters() if p.grad is not None
    )


@requires_cuda
def test_navit_per_image_weights():
    """per_image_weights (regularization set loss_weight x loss_weighting) scales the loss per image.

    Covers the two weighting factors filled in on the navit path: after passing a [G] weight
    vector, both the per-image loss and the total loss are scaled by the weights (symmetric
    with per-sample weighting on the standard path).
    """
    set_xformers_enabled(True)
    dtype = torch.float16
    model = _model(dtype)
    D = 128

    latent_shapes = [(4, 4), (6, 8)]
    text_lens = [5, 9]
    timesteps = [0.3, 0.7]
    latents_list = [
        torch.randn(1, 16, 1, h, w, device="cuda", dtype=dtype)
        for h, w in latent_shapes
    ]
    cross_list = [torch.randn(1, L, D, device="cuda", dtype=dtype) for L in text_lens]
    cross_packed = torch.cat(cross_list, dim=1)
    t = torch.tensor(timesteps, device="cuda", dtype=dtype)
    loss_fn = MseLoss()

    # baseline (no per-image weights)
    torch.manual_seed(42)
    _, _, info0 = navit_packed_forward_and_loss(
        model, latents_list, t, cross_packed, text_lens, loss_fn,
    )
    base = info0["per_image_loss"].clone().to(torch.float32)  # [G]

    # per-image weights [2.0, 0.5]
    w = torch.tensor([2.0, 0.5], device="cuda", dtype=torch.float32)
    torch.manual_seed(42)
    loss_w, _, info_w = navit_packed_forward_and_loss(
        model, latents_list, t, cross_packed, text_lens, loss_fn,
        per_image_weights=w,
    )
    # per-image loss is scaled by the weights
    torch.testing.assert_close(
        info_w["per_image_loss"].to(torch.float32), base * w, rtol=2e-3, atol=2e-3,
    )
    # total loss = mean(per_image x weights)
    torch.testing.assert_close(
        loss_w.detach().to(torch.float32), (base * w).mean(), rtol=2e-3, atol=2e-3,
    )
