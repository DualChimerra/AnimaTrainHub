"""Core of the NaViT / Patch-n-Pack training step: per-image noising -> block-diagonal packed forward -> per-image loss.

G heterogeneous images are concatenated into a single sequence; each image
only attends to its own tokens (block-diagonal self-attn) and its own
caption (block-diagonal cross-attn), and carries its own timestep (per-token
AdaLN). Zero padding, using the xformers varlen fast kernel.

Public:
- pack_cross_embeddings -- concatenates per-image text embeddings into a single sequence
- navit_packed_forward_and_loss -- one NaViT training step (forward + per-image loss)
"""

from __future__ import annotations

from typing import Sequence

import torch

from training.noise import make_noise


def pack_cross_embeddings(
    cross: torch.Tensor,
    t5_attn: torch.Tensor | None,
    navit_text_trim_padding: bool = False,
) -> tuple[torch.Tensor, list[int]]:
    """Concatenate per-image text embeddings into a single sequence for block-diagonal cross-attn.

    Args:
        cross: ``[G, L, D]`` -- per-image text embedding (L=512 padded).
        t5_attn: ``[G, L]`` -- per-image attention mask (1=valid token).
            Only used when ``navit_text_trim_padding=True``.
        navit_text_trim_padding: when True, truncate padding to each image's
            valid token count (speeds up cross-attn, no attending to padding
            positions); when False, each image carries the full 512-pad
            (same behavior as the standard path).

    Returns:
        (cross_packed ``[1, sum(L), D]``, text_seqlens ``[G]``)
    """
    G, L, D = cross.shape
    if navit_text_trim_padding and t5_attn is not None:
        tlens = t5_attn.sum(dim=1).clamp(min=1).tolist()
        tlens = [min(int(n), L) for n in tlens]
        cross_packed = torch.cat(
            [cross[i : i + 1, : tlens[i], :] for i in range(G)], dim=1
        )
        return cross_packed, tlens
    # Legacy 512-pad path: every image carries the full padded caption.
    cross_packed = cross.reshape(1, G * L, D)
    return cross_packed, [L] * G


def navit_packed_forward_and_loss(
    model,
    latents_list: Sequence[torch.Tensor],
    t_per_image: torch.Tensor,
    cross_packed: torch.Tensor,
    text_seqlens: Sequence[int],
    loss_fn,
    *,
    noise_offset: float = 0.0,
    pyramid_iters: int = 0,
    pyramid_discount: float = 0.35,
    use_checkpoint: bool = False,
    per_image_weights: torch.Tensor | None = None,
):
    """One NaViT/Patch-n-Pack training step: per-image noising -> packed forward -> per-image loss.

    G heterogeneous images are each noised independently (their own t /
    shape) -> patchified -> concatenated into a single sequence ->
    block-diagonal forward via ``forward_packed_navit`` -> per-image loss is
    the segment mean of the per-token loss.

    Args:
        model: ``MiniTrainDIT`` (needs ``patchify_latents_to_tokens`` /
            ``forward_packed_navit``).
        latents_list: G clean latents, each ``[1,C,T,h_i,w_i]`` or ``[C,T,h_i,w_i]``.
        t_per_image: ``[G]`` -- per-image flow-matching timestep.
        cross_packed: ``[1, sum(L), D]`` -- concatenated text embedding sequence.
        text_seqlens: ``[G]`` -- per-image caption token count (sum == sum(L)).
        loss_fn: ``LossProtocol`` -- ``compute(pred, target, t) -> per-element loss``.
        noise_offset / pyramid_iters / pyramid_discount: noise parameters (passed through to ``make_noise``).
        use_checkpoint: per-block gradient checkpointing (peak activation ~= 1 block).
        per_image_weights: ``[G]`` per-image weights (the regularization set's
            ``loss_weight`` times the t-dependent weight from
            ``loss_weighting``, combined per-image-t and passed in by the
            training loop; None=equal weight, neutral behavior).

    Returns:
        (loss, pred, info) -- ``loss`` is the per-image mean (gradient-bearing);
        ``info`` carries ``visual_seqlens`` and a detached ``per_image_loss`` for telemetry.
    """
    G = len(latents_list)
    if G == 0:
        raise ValueError("navit pack is empty")
    if len(text_seqlens) != G:
        raise ValueError(f"text_seqlens has {len(text_seqlens)} entries, expected G={G}")

    t_per_image = t_per_image.reshape(-1)
    if t_per_image.shape[0] != G:
        raise ValueError(f"t_per_image has {t_per_image.shape[0]} entries, expected G={G}")

    noisy_tok_list, target_tok_list, grid_list, vseq = [], [], [], []
    for i, lat in enumerate(latents_list):
        if lat.dim() == 4:
            lat = lat.unsqueeze(0)
        ti = t_per_image[i].to(dtype=lat.dtype)
        noise_i = make_noise(
            lat,
            noise_offset=noise_offset,
            pyramid_iters=pyramid_iters,
            pyramid_discount=pyramid_discount,
        )
        t_exp = ti.view(1, 1, 1, 1, 1)
        noisy_i = (1 - t_exp) * lat + t_exp * noise_i
        target_i = noise_i - lat
        # noisy and target have the same shape; concatenate them along the
        # batch dim into [2,C,T,h,w] and patchify once, then slice: rearrange
        # is independent per batch row -> equivalent to calling it per-item
        # separately, with half as many patchify calls.
        btok, bgrid, _m, bsize = model.patchify_latents_to_tokens(
            torch.cat([noisy_i, target_i], dim=0)
        )
        noisy_tok_list.append(btok[:1])
        target_tok_list.append(btok[1:])
        grid_list.append(bgrid[:1])
        vseq.append(int(btok[:1].shape[1]))

    tokens = torch.cat(noisy_tok_list, dim=1)         # [1, sum(N), M]
    target_tokens = torch.cat(target_tok_list, dim=1)
    grid = torch.cat(grid_list, dim=2)                # [1, 2, sum(N)]

    pred = model.forward_packed_navit(
        tokens, t_per_image, cross_packed, grid, vseq,
        [int(s) for s in text_seqlens],
        use_checkpoint=use_checkpoint,
    )

    # Per-image loss: elementwise loss -> mean over the patch dim -> mean over tokens within each image.
    # loss_fn.compute returns per-element loss (reduction='none'), which holds for any shape.
    # Studio's mse/huber don't use the t argument (constant huber), so passing t_per_image [G] is harmless.
    loss_map = loss_fn.compute(pred.float(), target_tokens.float(), t_per_image)  # [1, sum(N), M]
    token_loss = loss_map.mean(dim=-1)[0]              # [sum(N)] fp32

    counts = torch.tensor(vseq, device=token_loss.device)
    seg_id = torch.repeat_interleave(
        torch.arange(G, device=token_loss.device), counts
    )
    onehot = (
        seg_id.unsqueeze(0) == torch.arange(G, device=token_loss.device).unsqueeze(1)
    ).to(token_loss.dtype)                             # [G, sum(N)]
    per_image = (onehot @ token_loss) / counts.to(token_loss.dtype)  # [G], grad-bearing

    # Per-image weight: regularization set's loss_weight times the
    # timestep-dependent loss_weighting (min_snr / cosmap / detail_inv_t).
    # Combined per-image-t and passed in by the training loop, symmetric
    # with the standard path's per-sample weighting -- NaViT's per-image t
    # maps directly onto the per-sample SNR weighting semantics. When None,
    # behavior is neutral (equivalent to all 1.0).
    if per_image_weights is not None:
        per_image = per_image * per_image_weights.to(
            device=per_image.device, dtype=per_image.dtype
        )

    loss = per_image.mean()
    info = {
        "visual_seqlens": vseq,
        "per_image_loss": per_image.detach(),
    }
    return loss, pred, info
