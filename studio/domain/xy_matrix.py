"""XY matrix schema — axis enum + axis spec + matrix spec + value validation.

Design: loop over all images within a single task (amortizes the ~30s model-load
startup cost once). The frontend arranges the returned samples[].xy={xi,yi,xv,yv}
metadata into a grid by (yi, xi).

Axis value types are derived from the axis enum:
  lora_scale / cfg_scale → float
  steps                  → int
  lora_ckpt              → str (checkpoint file path)

History note: the lora_ckpt axis wasn't implemented in v1 because
AnimaLycorisAdapter lacked an unhook interface; it was added later once
detach() (utils/lycoris_adapter.py) plus the CACHE.apply_loras re-inject path
became available (runtime/anima_daemon.py:_run_xy).

Axis behavior:
  lora_scale: a global axis — iterates over all adapters and overrides every
              multiplier with the cell value (the old version only changed
              lora_configs[lora_index]; that's now deprecated).
  lora_ckpt:  mutates lora_configs[lora_index].path within the cell, then calls
              CACHE.apply_loras to re-inject (detach + reload state_dict).

Note: does NOT use `from __future__ import annotations` — under Pydantic v2 +
Python 3.12+'s deferred evaluation, that would turn typing._SpecialForm into a
schema key and raise AttributeError.
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

XYAxisType = Literal[
    "lora_scale",   # sets every LoRA's multiplier to the axis value (global)
    "steps",        # varies the sampling step count
    "cfg_scale",    # varies CFG
    "lora_ckpt",    # different step/epoch checkpoints from the same LoRA training run (for spotting overfitting)
]


class XYAxisSpec(BaseModel):
    """Single-axis definition: axis enum + values list + (for lora_ckpt) lora_index."""

    model_config = ConfigDict(extra="forbid")
    axis: XYAxisType = Field(..., description="The field this axis is bound to")
    values: list[Any] = Field(..., min_length=1, description="List of values this axis sweeps over")
    lora_index: Optional[int] = Field(
        None, ge=0,
        description="When axis=lora_ckpt, selects which lora_configs entry's path to modify",
    )


class XYMatrixSpec(BaseModel):
    """XY matrix: x axis is required, y is optional (None = single-axis N×1, degenerating to a single row)."""

    model_config = ConfigDict(extra="forbid")
    x: XYAxisSpec
    y: Optional[XYAxisSpec] = None


def _check_axis_values(axis: XYAxisSpec) -> None:
    """Validate the values type per the axis enum (float / int / string)."""
    int_axes = {"steps"}
    float_axes = {"lora_scale", "cfg_scale"}
    str_axes = {"lora_ckpt"}  # list of checkpoint paths
    needs_lora_index = {"lora_ckpt"}  # lora_scale became a global axis, so it no longer needs this

    if axis.axis in int_axes:
        for v in axis.values:
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"axis={axis.axis} values must be int, got {type(v).__name__}")
    elif axis.axis in float_axes:
        for v in axis.values:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"axis={axis.axis} values must be numbers, got {type(v).__name__}")
    elif axis.axis in str_axes:
        for v in axis.values:
            if not isinstance(v, str):
                raise ValueError(f"axis={axis.axis} values must be str, got {type(v).__name__}")

    if axis.axis in needs_lora_index and axis.lora_index is None:
        raise ValueError(f"axis={axis.axis} must set lora_index (which entry of lora_configs it binds to)")
    if axis.axis not in needs_lora_index and axis.lora_index is not None:
        raise ValueError(f"axis={axis.axis} must not set lora_index (only lora_ckpt may)")
