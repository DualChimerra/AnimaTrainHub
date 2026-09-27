"""InfoNoise adaptive timestep sampler.

Dynamically estimates the information content of each noise interval from the
I-MMSE identity, and concentrates sampling probability on the informative
window (the sigma band that's "neither too easy nor too hard"). Works in
Flow Matching's t in (0,1) space; internally maps to sigma = t/(1-t) and then
splits log-sigma space evenly into K bins.

Reference paper: arxiv 2602.18647 "Information-Guided Noise Allocation for
Efficient Diffusion Training" (Sec 3.1 + Algorithm 1).

Key implementation choices (alignment with / deviation from the paper):

- **EMA smoothing direction**: m_hat_k <- (1-beta)*m_hat_k + beta*l_bar_k,
  beta multiplies the new value. beta=0.9 means the new value gets 90%
  weight; FIFO B=256 already does low-level smoothing, so the EMA is a light
  second-pass smoothing. Paper Sec 3.1 describes a "smoothed binwise
  estimate" but doesn't give the literal formula; this direction is codified
  by test_ema_responsiveness_codifies_design_choice.

- **Entropy rate r_hat_k = mse_k / sigma_k^2** (log-sigma space, paper
  Appendix B.2 Eq 61). Self-consistent with summing over delta-log-sigma.
  Note: the paper's Sec 3 VE-channel formula is mse/sigma^3 (Eq 59), and the
  FM-OT path literally gives (1-u)/u^3 (Eq 64); the sigma^2 log-sigma form
  unifies both paths and avoids the 1/sigma^3 universal tail dominating the
  normalization at the low-sigma end (Sec B.6).

- **Gate pivot c**: defaults to c=0.15 (the value the paper reports for
  CIFAR in Sec 5). Setting gate_pivot_c=0 switches to the literal dynamic
  Eq 87 implementation (sweep from high sigma to low sigma, find the last bin
  with r_norm >= p_onset). The dynamic variant degenerates when the mmse
  shape is monotonically decreasing; c=0.15 is robust across shapes -- see
  tools/infonoise_e2e_verify.py for details.

- **Warmup units**: N_warm is counted in optimizer steps (gated inside
  maybe_refresh), matching the same units as the total_steps x 20% default in
  build. grad_accum>1 doesn't affect warmup duration. _internal_step is a
  record-call counter (micro-batch granularity), used only for diagnostics.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
import torch

from dual_peak_sampling import DualPeakConfig
from training.timestep_samplers.dual_peak import config_from_args

logger = logging.getLogger(__name__)


class InfoNoiseScheduler:
    """InfoNoise adaptive timestep sampler."""

    def __init__(
        self,
        K: int = 64,
        t_min: float = 0.001,
        t_max: float = 0.999,
        N_warm: int = 5000,
        M: int = 100,
        B: int = 256,
        beta: float = 0.9,
        n_gate: int = 3,
        p_onset: float = 0.002,
        N_min: int = 50,
        gate_pivot_c: float = 0.15,
        baseline_shift: float = 3.0,
        baseline_mode: str = "logit_normal",
        baseline_mix_low_prob: float = 0.0,
        baseline_timestep_schedule_shift: float = 1.0,
        baseline_style_snr_mean: float = -6.0,
        baseline_style_snr_sigma: float = 2.0,
        baseline_dual_peak_config: DualPeakConfig | None = None,
    ):
        if not 0.0 < p_onset < 1.0:
            raise ValueError(f"p_onset must be in (0,1), got {p_onset}")
        self.K = K
        self.N_warm = N_warm
        self.M = M
        self.B = B
        self.beta = beta
        self.n_gate = n_gate
        self.p_onset = p_onset
        self.N_min = N_min
        self.gate_pivot_c = gate_pivot_c
        self.baseline_shift = baseline_shift
        self.baseline_mode = baseline_mode
        self.baseline_mix_low_prob = baseline_mix_low_prob
        self.baseline_timestep_schedule_shift = baseline_timestep_schedule_shift
        self.baseline_style_snr_mean = baseline_style_snr_mean
        self.baseline_style_snr_sigma = baseline_style_snr_sigma
        self.baseline_dual_peak_config = baseline_dual_peak_config
        self._internal_step = 0

        sigma_min = t_min / (1.0 - t_min)
        sigma_max = t_max / (1.0 - t_max)
        log_edges = np.linspace(np.log(sigma_min), np.log(sigma_max), K + 1)
        self._log_sigma_edges = log_edges
        self._delta_log_sigma = float(log_edges[1] - log_edges[0])
        self._sigma_centers = np.exp(0.5 * (log_edges[:-1] + log_edges[1:]))

        self._fifo = [deque(maxlen=B) for _ in range(K)]
        self._mse_ema = np.zeros(K, dtype=np.float64)
        self._n_count = np.zeros(K, dtype=np.int32)
        self._cdf_values: Optional[np.ndarray] = None
        # Observability: last_refresh_status exposes the exit reason of the last
        # _refresh call; refresh_attempts counts degraded runs; warned_cold_start
        # prevents log spam.
        self._last_refresh_status: str = "not_refreshed_yet"
        self._refresh_attempts: int = 0
        self._refresh_degraded_count: int = 0
        self._warned_cold_start: bool = False

    def sample(self, bs: int, device, *, token_counts=None) -> torch.Tensor:
        """Sample t in (0,1). Uses the logit-normal baseline during warmup, the adaptive CDF after."""
        if self._cdf_values is None:
            return self._sample_baseline(bs, device)
        u = torch.rand(bs).numpy()
        log_sigma = np.interp(u, self._cdf_values, self._log_sigma_edges)
        sigma = np.exp(log_sigma)
        t = sigma / (1.0 + sigma)
        return torch.tensor(t, device=device, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)

    def _sample_baseline(self, bs: int, device) -> torch.Tensor:
        # P1-3: while warmup / the CDF isn't ready yet, keep using whatever
        # timestep_sampling the user picked in the schema, instead of hardcoding
        # logit_normal_shift. Reuse training.timestep_sampling.sample_t so we
        # don't fork a second copy of the distribution logic.
        from training.timestep_sampling import sample_t
        return sample_t(
            bs,
            device,
            mode=self.baseline_mode,
            shift=self.baseline_shift,
            mix_low_prob=self.baseline_mix_low_prob,
            timestep_schedule_shift=self.baseline_timestep_schedule_shift,
            style_snr_mean=self.baseline_style_snr_mean,
            style_snr_sigma=self.baseline_style_snr_sigma,
            dual_peak_config=self.baseline_dual_peak_config,
        )

    def record(self, t: torch.Tensor, raw_mse: torch.Tensor):
        """Record the per-sample raw MSE (no loss weighting applied) into its bin."""
        t_np = t.detach().cpu().float().numpy()
        mse_np = raw_mse.detach().cpu().float().numpy()
        sigma_np = t_np / np.clip(1.0 - t_np, 1e-8, None)
        log_sigma_np = np.log(np.clip(sigma_np, 1e-8, None))
        edges_inner = self._log_sigma_edges[1:-1]
        for i in range(len(t_np)):
            k = int(np.searchsorted(edges_inner, log_sigma_np[i]))
            self._fifo[k].append(float(mse_np[i]))
            self._n_count[k] = min(self._n_count[k] + 1, self.B)
        self._internal_step += 1   # diagnostics only: micro-batch counter, not used for warmup gating

    def maybe_refresh(self, global_step: int):
        """Refresh the schedule once the conditions are met (every M steps,
        after warmup ends, once every bin has enough samples).

        The warmup gate uses global_step (optimizer step), matching the units
        N_warm is computed in (total_steps x 20%, in build). With
        grad_accum>1, using _internal_step here would end warmup grad_accum
        times too early (PR #TODO fix).
        """
        if global_step < self.N_warm:
            return
        if global_step % self.M != 0:
            return
        if int(np.min(self._n_count)) < self.N_min:
            self._last_refresh_status = "skipped_bins_not_full"
            return
        self._refresh()
        # Cold-start trip wire: a full _refresh ran but the CDF still isn't
        # ready -> InfoNoise is silently falling back to baseline; warn once so
        # the user isn't burning compute for nothing.
        if self._cdf_values is None and not self._warned_cold_start:
            logger.warning(
                "InfoNoise: warmup has finished and every bin has enough "
                "samples, but the first schedule refresh still produced no "
                "valid CDF (reason: %s). Continuing to sample from the "
                "logit-normal baseline; if this persists late into training, "
                "your loss distribution is too flat in log-sigma space (e.g. "
                "an already-converged model), InfoNoise gives no speedup, and "
                "you should disable infonoise_enabled.",
                self._last_refresh_status,
            )
            self._warned_cold_start = True

    def status(self) -> dict:
        """Expose the current scheduler state (for wandb monitoring / debugging)."""
        return {
            "kind": "infonoise",
            "cdf_ready": self._cdf_values is not None,
            "last_refresh_status": self._last_refresh_status,
            "refresh_attempts": self._refresh_attempts,
            "refresh_degraded_count": self._refresh_degraded_count,
            "internal_step": self._internal_step,
        }

    # --- Pause/resume support (ADR 0006 Addendum 1): save the adaptive schedule
    # so resume doesn't lose the CDF. ---
    # Hyperparameters (K/B/N_warm/M/beta/...) are NOT saved -- they're rebuilt
    # from args; only the *learned* state is saved. K/B are recorded so
    # load_state_dict can shape-check and avoid mixing up checkpoints from
    # different configs.

    # v1 = sigma^3 entropy rate + buggy pivot; v2 = sigma^2 log-sigma entropy rate + paper-aligned pivot
    _STATE_VERSION = 2

    def state_dict(self) -> dict:
        return {
            "__version__": self._STATE_VERSION,
            "K": self.K,
            "B": self.B,
            "fifo": [list(buf) for buf in self._fifo],
            "mse_ema": self._mse_ema.copy(),
            "n_count": self._n_count.copy(),
            "cdf_values": None if self._cdf_values is None else self._cdf_values.copy(),
            "internal_step": int(self._internal_step),
            "last_refresh_status": self._last_refresh_status,
            "refresh_attempts": int(self._refresh_attempts),
            "refresh_degraded_count": int(self._refresh_degraded_count),
            "warned_cold_start": bool(self._warned_cold_start),
        }

    def load_state_dict(self, state: dict) -> None:
        saved_version = int(state.get("__version__", 1))
        if saved_version != self._STATE_VERSION:
            # v1 used the sigma^3 formula + a buggy pivot, mathematically
            # incompatible with v2; drop mse_ema/cdf and cold-start instead.
            logger.warning(
                "InfoNoise resume: state_dict version %d (current=%d) -- the "
                "algorithm changed (sigma^3 -> sigma^2 entropy rate + "
                "paper-aligned gate pivot); discarding mse_ema/cdf and "
                "cold-starting warmup.",
                saved_version, self._STATE_VERSION,
            )
            return
        saved_K = int(state.get("K", self.K))
        saved_B = int(state.get("B", self.B))
        if saved_K != self.K or saved_B != self.B:
            # Training may have already run for hours; don't crash just
            # because the config changed -- fall back to cold start and let
            # warmup run again.
            logger.warning(
                "InfoNoise resume: shape mismatch (saved K=%d B=%d, current K=%d B=%d) "
                "-- skipping sampler state load, cold-starting warmup instead.",
                saved_K, saved_B, self.K, self.B,
            )
            return
        self._fifo = [deque(buf, maxlen=self.B) for buf in state["fifo"]]
        self._mse_ema = np.asarray(state["mse_ema"], dtype=np.float64).copy()
        self._n_count = np.asarray(state["n_count"], dtype=np.int32).copy()
        cdf = state.get("cdf_values")
        self._cdf_values = None if cdf is None else np.asarray(cdf, dtype=np.float64).copy()
        self._internal_step = int(state.get("internal_step", 0))
        self._last_refresh_status = str(state.get("last_refresh_status", "not_refreshed_yet"))
        self._refresh_attempts = int(state.get("refresh_attempts", 0))
        self._refresh_degraded_count = int(state.get("refresh_degraded_count", 0))
        self._warned_cold_start = bool(state.get("warned_cold_start", False))

    def _refresh(self):
        self._refresh_attempts += 1

        # Step A+B: average loss + EMA smoothing (paper Sec 3.1 binwise smoothing)
        # Implementation choice: m_hat_k <- (1-beta)*m_hat_k + beta*l_bar_k,
        # beta controls the new value's weight (beta=0.9 -> new value gets 90%).
        # Paper Sec 3.1 describes a "smoothed binwise estimate" but gives no
        # literal formula; test_ema_responsiveness_codifies_design_choice
        # locks in this choice.
        l_bar = np.array([
            float(np.mean(list(buf))) if buf else 0.0
            for buf in self._fifo
        ])
        self._mse_ema = (1.0 - self.beta) * self._mse_ema + self.beta * l_bar

        # Step C: entropy rate r_hat_k = mse_k / sigma_k^2 (log-sigma space, paper Appendix B.2 Eq 61)
        # Note: an earlier implementation used mse/sigma^3 (VE channel Eq 59) +
        # summing delta-log-sigma, missing a sigma Jacobian factor; sigma^2
        # makes the log-sigma-space entropy rate self-consistent with the
        # normalization (M1/M3 fix).
        r_hat = self._mse_ema / (self._sigma_centers ** 2 + 1e-30)

        # Step D: gate pivot c
        # Defaults to gate_pivot_c=0.15 (the paper's Sec 5 CIFAR value);
        # gate_pivot_c=0 uses the literal dynamic Eq 87 implementation (sweep
        # from high sigma to low sigma, find the last bin with r_norm >= p_onset).
        # An earlier implementation took above.argmax(), picking the first
        # "above" bin from the low-sigma end -> c=sigma_min, degenerating the
        # gate into an identity mapping (E2E testing showed 99% of the mass
        # concentrated in the lowest sigma quarter).
        r_max = float(r_hat.max())
        if r_max < 1e-30:
            self._last_refresh_status = "mse_collapsed"
            self._refresh_degraded_count += 1
            return
        if self.gate_pivot_c > 0:
            c = float(self.gate_pivot_c)
        else:
            r_norm = r_hat / r_max
            above = r_norm >= self.p_onset
            # any(above) is always strictly True: r_norm.max() == 1.0 >= p_onset
            # (__init__ already asserts p_onset in (0,1)), so the dynamic path
            # needs no early-exit branch.
            last_above = int(np.where(above)[0][-1])
            c = float(self._sigma_centers[last_above])

        # Step E: gate g(sigma) = sigma^n / (sigma^n + c^n)
        sn = self._sigma_centers ** self.n_gate
        cn = c ** self.n_gate
        r_tilde = r_hat * sn / (sn + cn + 1e-30)

        # Step F+G: normalize + build the CDF (trapezoidal integration in
        # log-sigma space; bins are equal width so this is just a sum)
        q = r_tilde.clip(0.0)
        Z = float(q.sum() * self._delta_log_sigma)
        if Z < 1e-30:
            self._last_refresh_status = "normalizer_too_small"
            self._refresh_degraded_count += 1
            return
        q_norm = q / Z
        cdf = np.concatenate([[0.0], np.cumsum(q_norm * self._delta_log_sigma)])
        cdf[-1] = 1.0
        self._cdf_values = cdf.clip(0.0, 1.0)
        self._last_refresh_status = "ok"


def build(args, total_steps: Optional[int]) -> InfoNoiseScheduler:
    """Build an InfoNoiseScheduler from args.

    Callers should already have checked args.infonoise_enabled == True; this
    doesn't re-check it (dispatch is centralized in
    timestep_samplers.__init__.build_timestep_sampler).
    """
    n_warm_cfg = int(getattr(args, "infonoise_N_warm", 0) or 0)
    if n_warm_cfg <= 0:
        n_warm_cfg = max(200, int((total_steps or 5000) * 0.2))
        logger.info(f"InfoNoise N_warm auto-set to {n_warm_cfg} steps (total_steps {total_steps} x 20%)")

    scheduler = InfoNoiseScheduler(
        K=int(getattr(args, "infonoise_K", 64) or 64),
        N_warm=n_warm_cfg,
        M=int(getattr(args, "infonoise_M", 100) or 100),
        B=int(getattr(args, "infonoise_B", 256) or 256),
        beta=float(getattr(args, "infonoise_beta", 0.9) or 0.9),
        N_min=int(getattr(args, "infonoise_N_min", 50) or 50),
        gate_pivot_c=float(getattr(args, "infonoise_gate_pivot_c", 0.15) or 0.0),
        baseline_shift=float(getattr(args, "timestep_shift", 3.0) or 3.0),
        baseline_mode=str(getattr(args, "timestep_sampling", "logit_normal") or "logit_normal"),
        baseline_dual_peak_config=config_from_args(args),
        baseline_mix_low_prob=float(getattr(args, "timestep_mix_low_prob", 0.0) or 0.0),
        baseline_timestep_schedule_shift=float(getattr(args, "timestep_schedule_shift", 1.0) or 1.0),
        # style_friendly also needs the user's configured window when used as
        # the warmup baseline (0.0 is a valid mean).
        baseline_style_snr_mean=(
            -6.0 if getattr(args, "style_snr_mean", None) is None else float(args.style_snr_mean)
        ),
        baseline_style_snr_sigma=(
            2.0 if getattr(args, "style_snr_sigma", None) is None else float(args.style_snr_sigma)
        ),
    )
    logger.info(
        f"InfoNoise enabled: K={scheduler.K}, N_warm={scheduler.N_warm}, "
        f"M={scheduler.M}, B={scheduler.B}, beta={scheduler.beta}, "
        f"gate_pivot_c={scheduler.gate_pivot_c}, "
        f"baseline={scheduler.baseline_mode}(shift={scheduler.baseline_shift}, "
        f"mix_low_prob={scheduler.baseline_mix_low_prob}, "
        f"timestep_schedule_shift={scheduler.baseline_timestep_schedule_shift})"
    )
    return scheduler
