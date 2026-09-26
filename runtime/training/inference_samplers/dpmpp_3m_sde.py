"""DPM-Solver++(3M) SDE for CONST flow -- line-by-line aligned with ComfyUI k_diffusion.sample_dpmpp_3m_sde.

Alignment notes (comfy/k_diffusion/sampling.py + comfy/model_sampling.py, GPL-3.0):
- lambda (CONST half-log-SNR) = sigma.logit().neg() = log((1-sigma)/sigma)
- alpha_t = sigma_{i+1} * exp(lambda_t)  (always equals 1-sigma_{i+1} under CONST; kept in the ComfyUI form)
- offset_first_sigma_for_snr: when sigma_0 >= 1, replace it with percent_to_sigma(1e-4)
- Noise: BrownianTreeNoiseSampler, transform=identity (uses sigma directly as the
  time axis, **not** -log(sigma)); BatchedBrownianTree runs on CPU (cpu=True) to
  keep the RNG aligned.

This makes dpmpp_3m_sde's output match ComfyUI, widening the gap from [[er_sde]],
which uses independent Gaussian noise.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch

# When torchsde queries at boundary sigma values (== tree t0/t1), float32
# round-trip error produces a ta<t0 / tb>t1 overshoot of a few ULPs, which
# raises a UserWarning before clamping back to the boundary (numerically
# correct). ComfyUI hits this same warning too, it just doesn't promote it
# to an error. We silence exactly this warning here to avoid log spam.
warnings.filterwarnings(
    "ignore",
    message=r"Should have t[ab][<>]=t[01] but got",
    module="torchsde",
)


class _BrownianTreeNoiseSampler:
    """Aligned with ComfyUI's BrownianTreeNoiseSampler + BatchedBrownianTree.

    transform=identity: t0/t1 are taken directly as sigma_min/sigma_max (no
    -log). The tree is built and queried on CPU, then moved back to the
    original device, replicating ComfyUI's cpu=True RNG behavior.
    """

    def __init__(self, x: torch.Tensor, sigma_min: float, sigma_max: float, seed: Optional[int] = None):
        from torchsde import BrownianTree

        # ComfyUI: transform=identity, t0=sigma_min, t1=sigma_max. Store the
        # boundaries as float32 tensors throughout (matching the .float()
        # precision used when __call__ queries them), to avoid float64
        # boundary values triggering torchsde's ta<t0 / tb>t1 overshoot warning.
        t0 = torch.as_tensor(sigma_min, dtype=torch.float32)
        t1 = torch.as_tensor(sigma_max, dtype=torch.float32)
        self._sign = 1
        if float(t0) > float(t1):
            t0, t1, self._sign = t1, t0, -1
        if seed is None:
            seed = int(torch.randint(0, 2**63 - 1, ()).item())
        # cpu=True: the tree lives on CPU, so w0 is on CPU too
        self._device = x.device
        self._dtype = x.dtype
        self._w0 = torch.zeros_like(x, device="cpu")
        self._tree = BrownianTree(t0, self._w0, t1, entropy=int(seed))

    def __call__(self, sigma: float, sigma_next: float) -> torch.Tensor:
        t0 = torch.as_tensor(sigma, dtype=torch.float32)
        t1 = torch.as_tensor(sigma_next, dtype=torch.float32)
        sign = 1
        if float(t0) > float(t1):
            t0, t1, sign = t1, t0, -1
        delta = float(t1) - float(t0)
        if delta < 1e-12:
            return self._w0.to(device=self._device, dtype=self._dtype)
        w = self._tree(t0, t1).to(device=self._device, dtype=self._dtype) * (self._sign * sign)
        return w / (delta ** 0.5)


def _gaussian_noise_sampler(x: torch.Tensor, seed: Optional[int]):
    """Fallback when torchsde isn't available: independent Gaussian noise."""
    if seed is not None:
        if x.device.type == "cpu":
            seed = int(seed) + 1
        g = torch.Generator(device=x.device)
        g.manual_seed(int(seed))
    else:
        g = None

    def _sample(_sigma, _sigma_next):
        return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=g)

    return _sample


def _build_noise_sampler(
    x: torch.Tensor,
    sigmas: torch.Tensor,
    seed: Optional[int],
    *,
    require_brownian_tree: bool = False,
):
    """Prefer BrownianTree (aligned with ComfyUI); falls back to independent Gaussian noise if torchsde is missing."""
    positive = sigmas[sigmas > 0]
    sigma_min = float(positive.min()) if positive.numel() > 0 else 1e-3
    sigma_max = float(sigmas.max())
    try:
        return _BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed)
    except ImportError as exc:
        if require_brownian_tree:
            raise RuntimeError(
                "Comfy parity dpmpp_3m_sde requires torchsde BrownianTree noise"
            ) from exc
        import logging
        logging.getLogger(__name__).warning(
            "torchsde is not installed; dpmpp_3m_sde is falling back to independent Gaussian noise (behavior will diverge from ComfyUI / er_sde)"
        )
        return _gaussian_noise_sampler(x, seed=seed)


def _offset_first_sigma_for_snr(sigmas: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    """Aligned with ComfyUI's offset_first_sigma_for_snr (CONST).

    sigma_0 >= 1 would make logit blow up to inf; ComfyUI replaces it with
    percent_to_sigma(1e-4) = time_snr_shift(shift, 1 - 1e-4).
    """
    if sigmas.numel() <= 1:
        return sigmas
    if float(sigmas[0]) >= 1.0:
        sigmas = sigmas.clone()
        t = 1.0 - 1e-4
        sigmas[0] = shift * t / (1.0 + (shift - 1.0) * t)
    return sigmas


@torch.no_grad()
def sample(
    denoise_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    seed: Optional[int] = None,
    s_noise: float = 1.0,
    eta: float = 1.0,
    shift: float = 3.0,
    require_brownian_tree: bool = False,
    step_callback=None,
    **_unused,
) -> torch.Tensor:
    """DPM-Solver++(3M) SDE -- aligned with ComfyUI.

    Args:
        denoise_fn: takes (x, sigma) and returns the x0 estimate.
        sigmas: sigma sequence from high to low, ending with 0.0.
        eta: SDE noise strength (1.0 = ComfyUI default; 0.0 degenerates to multistep ODE).
        s_noise: noise scaling factor.
        shift: ModelSamplingDiscreteFlow shift (used for the sigma_0 offset, default 3.0).
        require_brownian_tree: if True, fail immediately when torchsde is missing instead of falling back to Gaussian.
        step_callback: per-step callback (step, total, denoised).
    """
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    if sigmas.numel() <= 1:
        return x

    noise_sampler = _build_noise_sampler(
        x,
        sigmas,
        seed=seed,
        require_brownian_tree=require_brownian_tree,
    )
    # offset_first_sigma_for_snr (applied after computing noise_sampler's
    # sigma_min/sigma_max, matching ComfyUI's order: BrownianTreeNoiseSampler
    # uses the raw sigmas[sigmas>0])
    sigmas = _offset_first_sigma_for_snr(sigmas, shift=shift)
    eps = 1e-7

    def half_log_snr(sigma: torch.Tensor) -> torch.Tensor:
        # CONST: log((1-sigma)/sigma) = logit(sigma).neg()
        s = sigma.clamp(min=eps, max=1.0 - eps)
        return torch.log((1.0 - s) / s)

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in range(len(sigmas) - 1):
        sigma_i = sigmas[i]
        denoised = denoise_fn(x, sigma_i)

        if step_callback is not None:
            try:
                step_callback(i, len(sigmas) - 1, denoised)
            except Exception:
                pass

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            lambda_s = half_log_snr(sigma_i)
            lambda_t = half_log_snr(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1.0)
            alpha_t = sigmas[i + 1] * lambda_t.exp()  # = 1 - sigma_{i+1} (CONST)

            x = sigmas[i + 1] / sigma_i * (-h * eta).exp() * x \
                + alpha_t * (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1.0
                phi_3 = phi_2 / h_eta - 0.5
                x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1.0
                x = x + (alpha_t * phi_2) * d

            if eta:
                x = x + noise_sampler(float(sigma_i), float(sigmas[i + 1])) * sigmas[i + 1] \
                    * (-2 * h * eta).expm1().neg().sqrt() * float(s_noise)

            denoised_1, denoised_2 = denoised, denoised_1
            h_1, h_2 = h, h_1

    return x
