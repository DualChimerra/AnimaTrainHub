"""MOSS NaViT / Patch-n-Pack training step: per-image noise -> packed forward -> per-image loss.

Same semantics as Anima's ``families/anima/navit.py``: G images of different sizes share one
sequence, each with its own timestep and text. The loss is averaged over all elements within an
image, then across images (equal weight per image, not per token). The packed forward is
``MossImageTransformer.forward_packed``.
"""

from __future__ import annotations

from typing import Sequence

import torch

from training.noise import make_noise


def navit_packed_forward_and_loss(
    model,
    latents_list: Sequence[torch.Tensor],
    t_per_image: torch.Tensor,
    contexts: Sequence[torch.Tensor],
    loss_fn,
    *,
    noise_offset: float = 0.0,
    pyramid_iters: int = 0,
    pyramid_discount: float = 0.35,
    use_checkpoint: bool = False,
    per_image_weights: torch.Tensor | None = None,
):
    """One MOSS NaViT training step.

    Args:
        model: ``MossImageTransformer``.
        latents_list: G clean latents, each ``[1, 32, 1, h_i, w_i]`` or ``[32, 1, h_i, w_i]``
            (FLUX2_F8C32_BN layout; the latter comes from ``collate_fn_navit_pack``).
        t_per_image: ``[G]`` per-image flow-matching timesteps.
        contexts: G tensors of ``[L_i, 4096]`` valid text tokens (padding removed).
        loss_fn: ``LossProtocol`` (``compute(pred, target, t)`` returns element-wise loss).
        noise_offset / pyramid_iters / pyramid_discount: passed to ``make_noise``.
        use_checkpoint: per-block gradient checkpointing.
        per_image_weights: ``[G]`` per-image weights (regularization loss_weight × loss_weighting), None = equal.

    Returns:
        (loss, pred, info): ``pred`` is the list of per-image predictions; ``info`` has
        ``visual_seqlens`` and detached ``per_image_loss``, shaped as in the Anima version.
    """
    count = len(latents_list)
    if count == 0:
        raise ValueError("navit pack is empty")
    if len(contexts) != count:
        raise ValueError(f"contexts has {len(contexts)} entries, expected G={count}")
    t_per_image = t_per_image.reshape(-1)
    if t_per_image.shape[0] != count:
        raise ValueError(f"t_per_image has {t_per_image.shape[0]} entries, expected G={count}")

    noisy_list, target_list = [], []
    for i, latent in enumerate(latents_list):
        if latent.dim() == 4:  # collate_fn_navit_pack yields [C, T, h, w]
            latent = latent.unsqueeze(0)
        noise = make_noise(
            latent,
            noise_offset=noise_offset,
            pyramid_iters=pyramid_iters,
            pyramid_discount=pyramid_discount,
        )
        t = t_per_image[i].to(dtype=latent.dtype)
        noisy_list.append((1 - t) * latent + t * noise)
        target_list.append(noise - latent)

    preds = model.forward_packed(noisy_list, t_per_image, contexts, use_checkpoint=use_checkpoint)

    per_image = torch.stack([
        loss_fn.compute(pred.float(), target.float(), t_per_image[i:i + 1]).mean()
        for i, (pred, target) in enumerate(zip(preds, target_list))
    ])
    if per_image_weights is not None:
        per_image = per_image * per_image_weights.to(device=per_image.device, dtype=per_image.dtype)

    loss = per_image.mean()
    info = {
        "visual_seqlens": [int(p.shape[-2] * p.shape[-1]) // 4 for p in preds],
        "per_image_loss": per_image.detach(),
    }
    return loss, preds, info
