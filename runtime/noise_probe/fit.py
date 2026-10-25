"""Fit a DualPeakConfig to target per-bin probabilities.

The mixture density is linear in the four weights, so for every candidate
(peak positions, widths, background) the best non-negative weights come from
NNLS on the components' bin masses; a coarse grid covers the rest.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Sequence

import torch

from dual_peak_sampling import DualPeakConfig, density

_FINE = 4000  # integration points per unit t; bins are >= 0.01 wide


def config_bin_masses(config: DualPeakConfig, edges: torch.Tensor) -> torch.Tensor:
    """Probability mass of each t bin under ``config`` (midpoint rule on a fine grid)."""
    n = int(_FINE * (float(edges[-1]) - float(edges[0])))
    t = torch.linspace(float(edges[0]), float(edges[-1]), n + 1, dtype=torch.float64)
    mid = ((t[:-1] + t[1:]) / 2).clamp(1e-6, 1 - 1e-6)
    mass = density(mid, config) * (t[1:] - t[:-1])
    idx = torch.bucketize(mid, edges[1:-1].to(torch.float64))
    out = torch.zeros(len(edges) - 1, dtype=torch.float64).index_add_(0, idx, mass)
    return out / out.sum()


_ONLY = dict(peak1_weight=0.0, peak2_weight=0.0, background_weight=0.0, uniform_weight=0.0)


def _peak1(pos, width, edges):
    return config_bin_masses(DualPeakConfig(**{**_ONLY, "peak1_weight": 1.0},
                                            peak1_position=pos, peak1_width=width), edges)


def _background(mean, width, edges):
    return config_bin_masses(DualPeakConfig(**{**_ONLY, "background_weight": 1.0},
                                            background_mean=mean, background_width=width), edges)


def fit_dual_peak(
    target: torch.Tensor,
    edges: torch.Tensor,
    *,
    peak1_positions: Sequence[float] = tuple(round(0.30 + 0.02 * i, 2) for i in range(23)),
    peak1_widths: Sequence[float] = (0.55, 0.8, 1.1),
    peak2_positions: Sequence[float] = tuple(round(0.66 + 0.02 * i, 2) for i in range(15)),
    peak2_widths: Sequence[float] = (0.8, 1.2, 1.6),
    background_means: Sequence[float] = (-4.0, -3.2, -2.8, -2.2, -1.5),
    background_widths: Sequence[float] = (1.8, 2.4, 3.0),
    uniform_weight: float = 0.05,
) -> tuple[DualPeakConfig, float]:
    """Return the grid point + NNLS weights closest to ``target`` and its L1 bin error.

    ``target`` is per-bin probability (sums to 1). Peak 1 is kept below peak 2
    so the two peaks keep their roles. The uniform share is fixed rather than
    fitted: free, it wins whenever the target has a shape the grid can't
    express, and the "fit" degenerates into plain uniform sampling.
    """
    from scipy.optimize import nnls

    target = target.to(torch.float64)
    target = target / target.sum()
    width = edges[1:] - edges[:-1]
    uniform = (width / width.sum()).numpy()
    # Fit the non-uniform remainder; uniform keeps its fixed share.
    b = (target.numpy() - uniform_weight * uniform).clip(min=0)

    p1 = {(p, w): _peak1(p, w, edges).numpy() for p, w in product(peak1_positions, peak1_widths)}
    # Peak 2 uses the same component math as peak 1; reuse it for the masses.
    p2 = {(p, w): _peak1(p, w, edges).numpy() for p, w in product(peak2_positions, peak2_widths)}
    bg = {(m, w): _background(m, w, edges).numpy() for m, w in product(background_means, background_widths)}

    import numpy as np

    best = None
    for (pos1, w1), m1 in p1.items():
        for (pos2, w2), m2 in p2.items():
            if pos2 <= pos1 + 0.1:
                continue
            for (bm, bw), mb in bg.items():
                a = np.stack([m1, m2, mb], axis=1)
                weights, _ = nnls(a, b)
                total = weights.sum()
                if total <= 0:
                    continue
                weights = np.append((1 - uniform_weight) * weights / total, uniform_weight)
                fitted = a @ weights[:3] + uniform_weight * uniform
                err = float(np.abs(fitted - target.numpy()).sum())
                if best is None or err < best[0]:
                    best = (err, pos1, w1, pos2, w2, bm, bw, weights)
    if best is None:
        raise ValueError("no grid point produced a valid fit")
    err, pos1, w1, pos2, w2, bm, bw, weights = best
    config = replace(
        DualPeakConfig(),
        peak1_position=pos1, peak1_width=w1, peak1_weight=round(float(weights[0]), 3),
        peak2_position=pos2, peak2_width=w2, peak2_weight=round(float(weights[1]), 3),
        background_mean=bm, background_width=bw, background_weight=round(float(weights[2]), 3),
        uniform_weight=round(float(weights[3]), 3),
    )
    return config, err


#: Ladder metric -> its value when the rung is identical to the original.
TRAIT_IDENTITY = {"palette": 0.0, "shapes": 1.0, "layout": 1.0}


def trait_target(
    t0s: Sequence[float],
    values: dict[str, Sequence[float]],
    edges: torch.Tensor,
    weights: dict[str, float],
) -> torch.Tensor:
    """Per-bin mass of where each trait gets decided, from a noise ladder.

    For a trait, ``change(t0) = |metric(t0) - identity| / |metric(t_max) - identity|``
    is how much of it the base model has replaced when it takes over at t0;
    made monotone, it is a CDF over t whose increments say in which band the
    trait is decided. Traits are mixed by ``weights`` (e.g. layout weighted
    down so training doesn't chase dataset poses).
    """
    t = [0.0, *map(float, t0s)]
    order = sorted(range(len(t)), key=lambda i: t[i])
    t = [t[i] for i in order]
    lo, hi = edges[:-1].tolist(), edges[1:].tolist()

    def cdf_at(x, cdf):
        if x >= t[-1]:
            return cdf[-1]
        for j in range(1, len(t)):
            if x <= t[j]:
                return cdf[j - 1] + (cdf[j] - cdf[j - 1]) * (x - t[j - 1]) / max(t[j] - t[j - 1], 1e-12)
        return cdf[-1]

    total = torch.zeros(len(lo), dtype=torch.float64)
    weight_sum = 0.0
    for name, weight in weights.items():
        if weight <= 0:
            continue
        identity = TRAIT_IDENTITY[name]
        change = [0.0, *(abs(float(v) - identity) for v in values[name])]
        change = [change[i] for i in order]
        running, cdf = 0.0, []
        for c in change:
            running = max(running, c)
            cdf.append(running)
        if cdf[-1] <= 0:
            continue
        cdf = [c / cdf[-1] for c in cdf]
        masses = torch.tensor([cdf_at(h, cdf) - cdf_at(l, cdf) for l, h in zip(lo, hi)], dtype=torch.float64)
        total += weight * masses / masses.sum()
        weight_sum += weight
    if weight_sum <= 0:
        raise ValueError("no trait changed along the ladder")
    return total / weight_sum
