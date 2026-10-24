"""Experimental Riemannion with auxiliary AdamW (arxiv:2507.12142).

Adapted from WalkingMeatAxolotl/AnimaLoraStudio, commit
42f510faa6e0a74676541393ca44fb506d8c2093 (GPL-3.0).
Bogachev et al., "LoRA meets Riemannion: Muon Optimizer for
Parametrization-independent Low-Rank Adapters", Algorithms 1/2/4.

Updates use tangent projection, transported heavy-ball momentum,
Newton-Schulz orthogonalization, and truncated-SVD retraction.
This implementation uses decoupled weight decay and caches momentum before
orthogonalization. It is an experimental adaptation, not official paper code.
Calculations are batched by shape in fp32; auxiliary parameters use AdamW.

Notation (one layer, W is out×in = m×n, rank r):
    X = A_L · Bᵀ             the weight delta; A_L has orthonormal columns (m×r), B is n×r
    B_R                      orthonormal basis of col(B) (n×r)
    ξ = Ȧ · B_Rᵀ + A_L · Ḃᵀ   a tangent vector (Ȧ ⊥ A_L)

Each step works on a 2r×2r core in the bases [A_L, Q_a] × [B_R, Q_b], where
Q_a / Q_b come from QR of Ȧ and of the part of Ḃ orthogonal to B_R:
ξ = [A_L, Q_a]·K·[B_R, Q_b]ᵀ with K = [[Ḃ_Rᵀ, R_bᵀ], [R_a, 0]]. Newton-Schulz
on K orthogonalizes ξ, zeroing the lower-right block projects it back to the
tangent space, and a truncated SVD of X(1−ηγ) − η·ξ retracts to rank r.
Layers too narrow for 2r orthonormal columns use reduced QR of the combined
bases instead.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = [
    "RiemannionWithAuxAdam", "project_lr", "riemannian_grad", "riemannion_update",
]

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)  # same as Muon (Keller Jordan, 2024)
NS_STEPS = 5


def _cholesky_qr(x: Tensor) -> tuple[Tensor, Tensor] | None:
    gram = (x.mT @ x).double()
    chol, info = torch.linalg.cholesky_ex(gram)
    if bool((info != 0).any()):
        return None
    r = chol.mT.to(x.dtype)
    q = torch.linalg.solve_triangular(r, x, upper=True, left=False)
    return q, r


def _qr(x: Tensor) -> tuple[Tensor, Tensor]:
    """Reduced QR of tall matrices ([..., m, k], k≪m): CholeskyQR2, falling back to Householder."""
    first = _cholesky_qr(x)
    if first is not None:
        second = _cholesky_qr(first[0])
        if second is not None:
            return second[0], second[1] @ first[1]
    return torch.linalg.qr(x, mode="reduced")


def _qr_perp(x: Tensor, basis: Tensor) -> tuple[Tensor, Tensor]:
    """x is analytically ⊥ basis; project again after QR to remove numerical leakage, so [basis, Q] is orthonormal."""
    q, r = _qr(x - basis @ (basis.mT @ x))
    q2, r2 = _qr(q - basis @ (basis.mT @ q))
    return q2, r2 @ r


def _polar_newton_schulz(core: Tensor) -> Tensor:
    """Approximate polar factor of a batch of cores (singular values to ~1, zeros stay 0), fp32."""
    a, b, c = NS_COEFFICIENTS
    x = core / core.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    for _ in range(NS_STEPS):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x


def _svd(core: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if core.is_cuda:
        try:
            return torch.linalg.svd(core, driver="gesvda")
        except torch.linalg.LinAlgError:
            # gesvda fails to converge on many near-zero / repeated singular values
            # (e.g. warm start from a LoRA of effective rank well below r); fall back
            # to CPU SVD, paying the cost only on the failing step
            u, s, vh = torch.linalg.svd(core.cpu())
            return u.to(core.device), s.to(core.device), vh.to(core.device)
    return torch.linalg.svd(core)


def riemannian_grad(a_l: Tensor, grad_z1: Tensor, grad_b: Tensor) -> tuple[Tensor, Tensor]:
    """Tangent vector (Ȧ, Ḃ) from the adapter's two gradients: Ȧ = (I−A_L A_Lᵀ)·∇Z1, Ḃ = ∇B."""
    a_dot = grad_z1 - a_l @ (a_l.mT @ grad_z1)
    return a_dot, grad_b


def project_lr(a: Tensor, b: Tensor, a_l: Tensor, b_r: Tensor) -> tuple[Tensor, Tensor]:
    """ProjectLR (Algorithm 2): project Z = A·Bᵀ onto the tangent space at (A_L, B_R).

    P(Z) = A_L A_Lᵀ Z + (I − A_L A_Lᵀ) Z B_R B_Rᵀ
         = (I−A_L A_Lᵀ)·A·(Bᵀ B_R) · B_Rᵀ  +  A_L · (B·(Aᵀ A_L))ᵀ
    """
    a_dot = (a - a_l @ (a_l.mT @ a)) @ (b.mT @ b_r)
    b_dot = b @ (a.mT @ a_l)
    return a_dot, b_dot


def riemannion_update(
    a_l: Tensor, b: Tensor, b_r: Tensor, a_dot: Tensor, b_dot: Tensor,
    lr: float, weight_decay: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """OrthoLR -> tangent projection -> RetractionLR; returns the new (A_L, B, B_R).

    Input tangent vector ξ = Ȧ·B_Rᵀ + A_L·Ḃᵀ (Ȧ ⊥ A_L). All work happens on the
    2r×2r core in the [A_L, Q_a] × [B_R, Q_b] bases (see the module docstring).
    """
    rank = a_l.shape[-1]
    if 2 * rank > min(a_l.shape[-2], b_r.shape[-2]):
        # A rank-r orthogonal complement may not fit in a narrow layer.
        # Reduced QR of the combined bases supports these high-rank cases
        # without pretending that 2r orthonormal columns fit in <2r rows.
        q_a, r_a = torch.linalg.qr(torch.cat([a_l, a_dot], dim=-1), mode="reduced")
        q_b, r_b = torch.linalg.qr(torch.cat([b_r, b_dot], dim=-1), mode="reduced")
        a_basis, b_basis = r_a[..., :rank], r_b[..., :rank]
        core = r_a[..., rank:] @ b_basis.mT + a_basis @ r_b[..., rank:].mT
        step = _polar_newton_schulz(core)
        left = a_basis @ (a_basis.mT @ step)
        step = left + ((step - left) @ b_basis) @ b_basis.mT
        x_core = (1.0 - lr * weight_decay) * (a_basis @ (q_b.mT @ b).mT)
        u, s, vh = torch.linalg.svd(x_core - lr * step, full_matrices=False)
        new_a_l = q_a @ u[..., :rank]
        new_b_r = q_b @ vh[..., :rank, :].mT
        return new_a_l, new_b_r * s[..., None, :rank], new_b_r
    b_dot_r = b_r.mT @ b_dot                       # r×r
    q_a, r_a = _qr_perp(a_dot, a_l)                # Ȧ = Q_a R_a
    q_b, r_b = _qr_perp(b_dot - b_r @ b_dot_r, b_r)  # Ḃ⊥ = Q_b R_b
    zeros = torch.zeros_like(r_a)
    core = torch.cat([
        torch.cat([b_dot_r.mT, r_b.mT], dim=-1),
        torch.cat([r_a, zeros], dim=-1),
    ], dim=-2)
    step = _polar_newton_schulz(core)
    step[..., rank:, rank:] = 0                    # back to the tangent space: drop (I−P_A)·O·(I−P_B)
    x_core = torch.zeros_like(core)
    x_core[..., :rank, :rank] = (1.0 - lr * weight_decay) * (b_r.mT @ b).mT  # X = A_L·Cᵀ·B_Rᵀ
    u, s, vh = _svd(x_core - lr * step)
    new_a_l = torch.cat([a_l, q_a], dim=-1) @ u[..., :rank]
    new_b_r = torch.cat([b_r, q_b], dim=-1) @ vh[..., :rank, :].mT
    return new_a_l, new_b_r * s[..., None, :rank], new_b_r


class RiemannionWithAuxAdam(Optimizer):
    """Riemannion for the adapter's marked B / Z1 parameters, auxiliary AdamW for the rest.

    The adapter tags parameters with ``_riemann_role`` ("b" / "z1") and ``_riemann_layer``
    (a weakref returning the owning layer); the layer has ``a_l`` / ``b_r`` buffers and
    ``lora_b`` / ``lora_z1`` parameters. Momentum lives in ``self.state[lora_b]`` and is
    saved with optimizer.state_dict for resume.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum,
            adam_betas=tuple(adam_betas), adam_eps=adam_eps, use_riemann=True,
        )
        super().__init__(self._split_groups(params), defaults)

    @staticmethod
    def _split_groups(params) -> list[dict]:
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        groups = params if isinstance(params[0], dict) else [{"params": params}]
        split: list[dict] = []
        for group in groups:
            group_params = list(group["params"])
            extra = {k: v for k, v in group.items() if k != "params"}
            riemann = [p for p in group_params if hasattr(p, "_riemann_role")]
            adam = [p for p in group_params if not hasattr(p, "_riemann_role")]
            if riemann:
                split.append({**extra, "params": riemann, "use_riemann": True})
            if adam:
                split.append({**extra, "params": adam, "use_riemann": False})
        return split

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_riemann"]:
                self._riemann_step(group)
            else:
                self._adam_step(group)
        return loss

    def _riemann_step(self, group: dict) -> None:
        lr, wd, beta = group["lr"], group["weight_decay"], group["momentum"]
        buckets: dict[tuple, list] = defaultdict(list)
        for p in group["params"]:
            if p._riemann_role != "b":
                continue
            layer = p._riemann_layer()
            z1 = layer.lora_z1
            if p.grad is None or z1.grad is None:
                continue
            buckets[(tuple(layer.a_l.shape), tuple(p.shape), p.device)].append(layer)

        for layers in buckets.values():
            a_l = torch.stack([ly.a_l for ly in layers]).float()
            b_r = torch.stack([ly.b_r for ly in layers]).float()
            b = torch.stack([ly.lora_b for ly in layers]).float()
            g_z1 = torch.stack([ly.lora_z1.grad for ly in layers]).float()
            g_b = torch.stack([ly.lora_b.grad for ly in layers]).float()

            a_dot, b_dot = riemannian_grad(a_l, g_z1, g_b)
            states = [self.state[ly.lora_b] for ly in layers]
            if any("a_hb" in st for st in states):
                # Some layers may receive their first gradient later than
                # others in the same shape bucket (e.g. optional branches).
                a_shape = (*a_l.shape[1:-1], 2 * a_l.shape[-1])
                b_shape = (*b_r.shape[1:-1], 2 * b_r.shape[-1])
                a_hb = torch.stack([
                    st["a_hb"] if "a_hb" in st else a_l.new_zeros(a_shape) for st in states
                ])
                b_hb = torch.stack([
                    st["b_hb"] if "b_hb" in st else b_r.new_zeros(b_shape) for st in states
                ])
                a_prev, b_prev = project_lr(a_hb, b_hb, a_l, b_r)
                a_dot = beta * a_prev + a_dot
                b_dot = beta * b_prev + b_dot
            # Cache momentum ξ = [Ȧ, A_L]·[B_R, Ḃ]ᵀ (projected to the new tangent space next step)
            a_hb_new = torch.cat([a_dot, a_l], dim=-1)
            b_hb_new = torch.cat([b_r, b_dot], dim=-1)

            new_a_l, new_b, new_b_r = riemannion_update(a_l, b, b_r, a_dot, b_dot, lr, wd)
            # Layers with an all-zero gradient (unused this step) keep their values. A batch
            # mask instead of per-layer .any() avoids one GPU sync per layer
            # (280 layers: optimizer.step 30 ms -> 100 ms)
            active = ((g_z1.abs().amax(dim=(-2, -1)) > 0) | (g_b.abs().amax(dim=(-2, -1)) > 0))
            active = active[:, None, None]
            new_a_l = torch.where(active, new_a_l, a_l)
            new_b = torch.where(active, new_b, b)
            new_b_r = torch.where(active, new_b_r, b_r)

            # Batched write-back: per-layer copy_ would be 3 × layers kernel launches
            torch._foreach_copy_(
                [ly.a_l for ly in layers] + [ly.b_r for ly in layers] + [ly.lora_b for ly in layers],
                [*new_a_l.unbind(0), *new_b_r.unbind(0), *new_b.unbind(0)],
            )
            for st, a_hb, b_hb in zip(states, a_hb_new.unbind(0), b_hb_new.unbind(0)):
                st["a_hb"] = a_hb
                st["b_hb"] = b_hb

    def _adam_step(self, group: dict) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        beta1, beta2 = group["adam_betas"]
        eps = group["adam_eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            step = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.lerp_(p.grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            denom = (exp_avg_sq / (1 - beta2 ** step)).sqrt_().add_(eps)
            p.mul_(1 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-lr / (1 - beta1 ** step))
