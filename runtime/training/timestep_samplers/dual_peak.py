"""AnimaTrainHub config adapter; the reusable package knows nothing about Studio."""

from dataclasses import fields
import logging

from dual_peak_sampling import DualPeakConfig

logger = logging.getLogger(__name__)


def config_from_args(args) -> DualPeakConfig | None:
    if str(getattr(args, "timestep_sampling", "")).lower() != "dual_peak":
        return None
    values = {}
    for field in fields(DualPeakConfig):
        value = getattr(args, f"dual_peak_{field.name}", None)
        if value is not None:
            values[field.name] = float(value)
    config = DualPeakConfig(**values)
    if getattr(args, "infonoise_enabled", False):
        logger.warning("Dual Peak: InfoNoise uses this curve only until its adaptive distribution is ready.")
    if getattr(args, "leap_enabled", False):
        logger.warning("Dual Peak: the curve applies only to non-Leap steps; Leap samples its own timesteps.")
    if (getattr(args, "timestep_schedule_shift", 1.0) not in (None, 1.0)
            or getattr(args, "timestep_shift_resolution_aware", False)):
        logger.warning("Dual Peak: schedule/resolution shifts are enabled and will move the configured peaks.")
    return config
