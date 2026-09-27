"""ER-SDE-Solver-3 implementation under the CONST (flow) noise schedule.

Extracted from the original _sample_er_sde_const_x0 in training/sampling.py
(ADR 0003 PR-C moved it into the inference_samplers/ plugin subpackage).

Based on ComfyUI's k_diffusion_sampling.sample_er_sde (with the model_patcher
dependency stripped out).
"""

from __future__ import annotations

from typing import Optional

import torch


def _default_noise_sampler(x: torch.Tensor, seed: Optional[int]):
    """Based on ComfyUI's k_diffusion_sampling.default_noise_sampler"""
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


def _offset_first_sigma_for_snr(sigmas: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    """Aligned with ComfyUI's offset_first_sigma_for_snr (CONST): when sigma_0 >= 1,
    replace it with percent_to_sigma(1e-4) = time_snr_shift(shift, 1 - 1e-4), to
    avoid logSNR blowing up to inf, which would overflow er_lambda -> noise_scaler NaN."""
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
    max_stage: int = 3,
    shift: float = 3.0,
    step_callback=None,
) -> torch.Tensor:
    """ER-SDE-3 stochastic sampler.

    step_callback: an optional hook (used for daemon mid-step previews).
        Signature: callback(step:int, total:int, denoised:torch.Tensor) -> None.
        Called once per step after the x0 estimate is computed; it returns
        synchronously/blocking, so the caller should do lightweight decoding +
        async push and not block inside the callback. Behavior is identical to
        the previous version when this is None (the default).
    """
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    if sigmas.numel() <= 1:
        return x

    noise_sampler = _default_noise_sampler(x, seed=seed)

    # Aligned with ComfyUI: first offset sigma_0 (>= 1 -> ~0.9997), then compute
    # half-log-SNR. Clamping straight to 1-1e-12 would make er_lambda≈1e12,
    # overflowing noise_scaler's exp into NaN.
    sigmas = _offset_first_sigma_for_snr(sigmas, shift=shift)
    # CONST: half_log_snr = log((1 - t) / t) = -logit(t). The trailing 0 is
    # handled by the sigmas[i+1]==0 branch below; here we only clamp the
    # lower bound to avoid log(0).
    eps = 1e-12
    t = sigmas.clamp(min=eps, max=1.0 - eps)
    half_log_snrs = torch.log((1 - t) / t)
    er_lambdas = half_log_snrs.neg().exp()  # er_lambda = t / (1 - t)

    old_denoised = None
    old_denoised_d = None

    def noise_scaler(lam: torch.Tensor) -> torch.Tensor:
        # default_er_sde_noise_scaler
        lam = lam.to(x.device, dtype=torch.float32)
        return lam * ((lam ** 0.3).exp() + 10.0)

    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        denoised = denoise_fn(x, sigma)

        if step_callback is not None:
            try:
                step_callback(i, len(sigmas) - 1, denoised)
            except Exception:
                pass  # a preview failure shouldn't affect sampling

        stage_used = min(int(max_stage), i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = 1.0 - sigmas[i]
            alpha_t = 1.0 - sigmas[i + 1]
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1 (Euler)
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            if stage_used >= 2 and old_denoised is not None:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                # Stage 2
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3 and old_denoised_d is not None:
                    # Stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u

                old_denoised_d = denoised_d

            # Stochastic term
            if s_noise and float(s_noise) > 0:
                noise = noise_sampler(float(sigmas[i]), float(sigmas[i + 1]))
                sde_scale = (er_lambda_t ** 2 - (er_lambda_s ** 2) * (r ** 2)).clamp(min=0).sqrt().nan_to_num(nan=0.0)
                x = x + alpha_t * noise * float(s_noise) * sde_scale

        old_denoised = denoised

    return x
