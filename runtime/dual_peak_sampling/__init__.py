"""Portable, PyTorch-only dual-peak flow-matching timestep sampler."""

from .core import DualPeakConfig, density, sample_timesteps

__version__ = "0.1.0"
__all__ = ["DualPeakConfig", "density", "sample_timesteps"]
