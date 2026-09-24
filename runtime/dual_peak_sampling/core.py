"""Independent of AnimaTrainHub: t=0 is data, t=1 is noise.

Sample a mixture, NOT an average of samples (which would change its density).
The two peak locations describe individual components, not the final mixture.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch


@dataclass(frozen=True)
class DualPeakConfig:
    """Widths are standard deviations in log-SNR; weights are relative masses.

    Peak widths <= 2.8 keep each peak component unimodal in t, making the
    location parameter an actual mode, not a possible minimum between modes.
    The broad background is an unconstrained log-SNR normal component.
    """

    peak1_position: float = 0.525
    peak1_width: float = 0.55
    peak1_weight: float = 0.15
    peak2_position: float = 0.850
    peak2_width: float = 1.20
    peak2_weight: float = 0.35
    background_mean: float = -3.2
    background_width: float = 2.20
    background_weight: float = 0.45
    uniform_weight: float = 0.05

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        for name in ("peak1_position", "peak2_position"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be between 0 and 1 (exclusive)")
        for name in ("peak1_width", "peak2_width"):
            if not 0.0 < getattr(self, name) <= 2.8:
                raise ValueError(f"{name} must be in (0, 2.8]")
        if not 0.0 < self.background_width <= 6.0:
            raise ValueError("background_width must be in (0, 6]")
        if not -12.0 <= self.background_mean <= 6.0:
            raise ValueError("background_mean must be in [-12, 6]")
        for name in ("peak1_weight", "peak2_weight", "background_weight", "uniform_weight"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if sum(self.weights) <= 0.0:
            raise ValueError("At least one mixture weight must be positive")

    @property
    def weights(self) -> tuple[float, float, float, float]:
        return (self.peak1_weight, self.peak2_weight,
                self.background_weight, self.uniform_weight)

    @property
    def probabilities(self) -> tuple[float, ...]:
        total = sum(self.weights)
        return tuple(weight / total for weight in self.weights)

    @property
    def components(self) -> tuple[tuple[float, float], ...]:
        def logsnr_mean(position: float, width: float) -> float:
            # d/dt log p(t) = 0 at position, including the transform Jacobian.
            return -2.0 * (math.log(position) - math.log1p(-position)
                           - (width / 2.0) ** 2 * (2.0 * position - 1.0))

        return (
            (logsnr_mean(self.peak1_position, self.peak1_width), self.peak1_width),
            (logsnr_mean(self.peak2_position, self.peak2_width), self.peak2_width),
            (self.background_mean, self.background_width),
        )


DEFAULT_CONFIG = DualPeakConfig()


@torch.no_grad()
def sample_timesteps(
    batch_size: int,
    device: torch.device | str = "cpu",
    config: DualPeakConfig = DEFAULT_CONFIG,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return float32 [batch_size] on device, clamped to [1e-4, 1-1e-4].

    Uses the caller's/global PyTorch RNG; never seeds it or uses Python/NumPy
    randomness. For an explicit generator, its device must match `device`.
    Float32 is intentional even inside mixed-precision training: bf16 would
    round some high timesteps to exactly 1. No CPU/GPU synchronization needed.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 0:
        raise ValueError("batch_size must be a non-negative integer")
    p1, p2, p_background, _ = config.probabilities
    (m1, s1), (m2, s2), (mb, sb) = config.components
    options = dict(device=device, dtype=torch.float32, generator=generator)
    choose = torch.rand(batch_size, **options)
    z = torch.randn(batch_size, **options)
    mean = torch.where(choose < p1, m1, torch.where(choose < p1 + p2, m2, torch.full_like(choose, mb)))
    width = torch.where(choose < p1, s1, torch.where(choose < p1 + p2, s2, torch.full_like(choose, sb)))
    biased = torch.sigmoid(-0.5 * (mean + width * z))
    uniform = torch.rand(batch_size, **options)
    t = torch.where(choose < p1 + p2 + p_background, biased, uniform)
    return t.clamp(1e-4, 1.0 - 1e-4)


def density(t: torch.Tensor, config: DualPeakConfig = DEFAULT_CONFIG) -> torch.Tensor:
    """Analytic density BEFORE safety clamping, for plots and distribution tests.

    Accepts floating tensors strictly inside (0, 1). Not a loss weight.
    The safety clamp in sample_timesteps adds tiny endpoint point masses.
    """
    if not t.is_floating_point() or not bool(((t > 0) & (t < 1)).all()):
        raise ValueError("density expects floating t strictly inside (0, 1)")
    lam = 2.0 * (torch.log1p(-t) - torch.log(t))
    result = torch.full_like(t, config.probabilities[3])
    for weight, (mean, width) in zip(config.probabilities[:3], config.components):
        normal = torch.exp(-0.5 * ((lam - mean) / width).square())
        result = result + weight * normal / (width * math.sqrt(2.0 * math.pi)) * 2.0 / (t * (1.0 - t))
    return result
